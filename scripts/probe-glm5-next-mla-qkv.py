#!/usr/bin/env python3
"""Generate a same-GGUF GLM-5.3 block-3 MLA Q/KV trunk oracle.

The quantized trunk contract intentionally uses F32 activation accumulation
through Q8_0 projections and weighted RMSNorm. It does not claim parity with
the original BF16 checkpoint; it is an implementation oracle for this GGUF.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import mmap
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_helpers():
    source = Path(__file__).with_name("probe-glm5-next-kda-payload.py")
    spec = importlib.util.spec_from_file_location("glm5_kda_payload", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import GGUF helpers from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_bytes(blob, data_start, tensors, name, shape, typ):
    dims, got_type, relative = tensors.get(name, (None, None, None))
    if dims != shape or got_type != typ:
        raise ValueError(
            f"{name}: expected {(shape, typ)}, got {(dims, got_type)}")
    count = int(np.prod(shape))
    if typ == 8:
        size = (shape[0] // 32) * 34 * int(np.prod(shape[1:]))
    elif typ == 0:
        size = count * 4
    else:
        raise ValueError(f"unsupported oracle tensor type {typ}")
    offset = data_start + relative
    return bytes(memoryview(blob)[offset:offset + size])


def q8_matrix(blob, data_start, tensors, name, shape):
    if shape[0] % 32:
        raise ValueError(f"{name}: Q8_0 input width is not block aligned")
    payload = tensor_bytes(blob, data_start, tensors, name, shape, 8)
    rows = int(np.prod(shape[1:]))
    blocks = shape[0] // 32
    raw = np.frombuffer(payload, dtype=np.uint8).reshape(rows, blocks, 34)
    scales = raw[:, :, :2].copy().view("<f2").reshape(rows, blocks).astype(np.float32)
    quants = raw[:, :, 2:].view(np.int8).astype(np.float32)
    matrix = (quants * scales[:, :, None]).reshape(rows, shape[0])
    return torch.from_numpy(matrix), payload


def f32_vector(blob, data_start, tensors, name, count):
    payload = tensor_bytes(blob, data_start, tensors, name, (count,), 0)
    values = torch.from_numpy(np.frombuffer(payload, dtype="<f4").copy())
    return values, payload


def deterministic_hidden(rows):
    axis = torch.arange(4096, dtype=torch.float32)
    values = []
    for row in range(rows):
        values.append(
            torch.sin((axis + 1 + 37 * row) * 0.0031) * 0.19
            + torch.cos((axis + 11 + 19 * row) * 0.0017) * 0.07
            + (row - (rows - 1) / 2) * 0.002)
    return torch.stack(values).to(torch.bfloat16).float()


def rms_norm_f32(values, weight, eps=1.0e-5):
    scale = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    return values * scale * weight


def dump_f32(value):
    return value.detach().float().contiguous().numpy().astype(
        "<f4", copy=False).tobytes()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dump-prefix", required=True, type=Path)
    args = parser.parse_args()
    if args.layer not in (*range(3, 44, 4), 45):
        parser.error("--layer must be a sparse-MLA layer")
    if args.rows < 2:
        parser.error("--rows must be at least 2")

    torch.set_grad_enabled(False)
    torch.set_num_threads(min(16, max(1, torch.get_num_threads())))
    helpers = load_helpers()
    data_start, tensors = helpers.load_directory(args.model)
    prefix = f"blk.{args.layer}"
    payload_hash = hashlib.sha256()
    with args.model.open("rb") as fp:
        blob = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        hidden = deterministic_hidden(args.rows)
        query_hidden = hidden[-1:]

        q_a_weight, payload = q8_matrix(
            blob, data_start, tensors, f"{prefix}.attn_q_a.weight",
            (4096, 1536))
        payload_hash.update(payload)
        q_a = F.linear(query_hidden, q_a_weight)
        del q_a_weight

        q_norm_weight, payload = f32_vector(
            blob, data_start, tensors, f"{prefix}.attn_q_a_norm.weight", 1536)
        payload_hash.update(payload)
        if not torch.equal(q_norm_weight, q_norm_weight.to(torch.bfloat16).float()):
            raise ValueError("q_a norm payload is not widened-BF16 exact")
        q_resid = rms_norm_f32(q_a, q_norm_weight)

        q_b_weight, payload = q8_matrix(
            blob, data_start, tensors, f"{prefix}.attn_q_b.weight",
            (1536, 16384))
        payload_hash.update(payload)
        query = F.linear(q_resid, q_b_weight).reshape(64, 256)
        del q_b_weight

        kv_weight, payload = q8_matrix(
            blob, data_start, tensors, f"{prefix}.attn_kv_a_mqa.weight",
            (4096, 512))
        payload_hash.update(payload)
        kv_raw = F.linear(hidden, kv_weight)
        del kv_weight

        kv_norm_weight, payload = f32_vector(
            blob, data_start, tensors, f"{prefix}.attn_kv_a_norm.weight", 512)
        payload_hash.update(payload)
        if not torch.equal(kv_norm_weight, kv_norm_weight.to(torch.bfloat16).float()):
            raise ValueError("KV norm payload is not widened-BF16 exact")
        kv_norm = rms_norm_f32(kv_raw, kv_norm_weight)

        k_b_weight, payload = q8_matrix(
            blob, data_start, tensors, f"{prefix}.attn_k_b.weight",
            (256, 512, 64))
        payload_hash.update(payload)
        k_b_weight = k_b_weight.reshape(64, 512, 256)
        qk_low = torch.einsum("hi,hji->hj", query, k_b_weight)
        del k_b_weight
        blob.close()

    arrays = {
        ".hidden.f32": dump_f32(hidden),
        ".q_a.f32": dump_f32(q_a),
        ".q_resid.f32": dump_f32(q_resid),
        ".query.f32": dump_f32(query),
        ".kv_raw.f32": dump_f32(kv_raw),
        ".kv_norm.f32": dump_f32(kv_norm),
        ".qk_low.f32": dump_f32(qk_low),
    }
    document = {
        "status": "same-GGUF Q8/F32 component oracle; not BF16 checkpoint parity",
        "model_size_bytes": args.model.stat().st_size,
        "layer": args.layer,
        "rows": args.rows,
        "query_row": args.rows - 1,
        "arithmetic_contract": "Q8_0 dequantized weights with F32 matmul and F32 weighted RMSNorm",
        "tensor_payload_sha256": payload_hash.hexdigest(),
        "dump_sha256": {
            suffix: hashlib.sha256(payload).hexdigest()
            for suffix, payload in arrays.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.dump_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    for suffix, payload in arrays.items():
        Path(str(args.dump_prefix) + suffix).write_bytes(payload)
    print("PASS GLM5-next same-GGUF MLA Q/KV trunk oracle")
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
