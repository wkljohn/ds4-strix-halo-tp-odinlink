#!/usr/bin/env python3
"""Generate a same-GGUF GLM-5.3 NoPE masked-score component oracle.

This pins the dtype-sensitive Transformers indexer path through real block-3
projections, true LayerNorm, learned pool-4 compression and masked pool scores.
It intentionally stops before top-k and sparse MLA consumption. The synthetic
q_resid and hidden query row are independent component inputs; this does not
yet prove their production q_a coupling.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import mmap
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


TRANSFORMERS_REFERENCE = (
    "https://github.com/huggingface/transformers/blob/"
    "155b89935a648278dd38c78184cbc40e6a65f14b/"
    "src/transformers/models/glm5_next/modeling_glm5_next.py"
)


def load_gguf_helpers():
    source = Path(__file__).with_name("probe-glm5-next-kda-payload.py")
    spec = importlib.util.spec_from_file_location("glm5_kda_payload", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import GGUF helpers from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_raw(blob, data_start, tensors, name, shape, typ):
    dims, got_type, relative = tensors.get(name, (None, None, None))
    if dims != shape or got_type != typ:
        raise ValueError(
            f"{name}: expected shape/type {(shape, typ)}, "
            f"got {(dims, got_type)}")
    offset = data_start + relative
    count = int(np.prod(shape))
    itemsize = 2 if typ == 30 else 4
    return memoryview(blob)[offset:offset + count * itemsize]


def bf16_matrix(blob, data_start, tensors, name, shape):
    raw = tensor_raw(blob, data_start, tensors, name, shape, 30)
    values = np.frombuffer(raw, dtype="<u2").copy()
    # GGUF dimensions are [input, output]; physical rows are output rows.
    return torch.from_numpy(values).view(torch.bfloat16).reshape(shape[1], shape[0])


def f32_vector(blob, data_start, tensors, name, count):
    raw = tensor_raw(blob, data_start, tensors, name, (count,), 0)
    return torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy())


def deterministic_inputs(rows):
    hidden_axis = torch.arange(4096, dtype=torch.float32)
    hidden = []
    for row in range(rows):
        value = (
            torch.sin((hidden_axis + 1 + 37 * row) * 0.0031) * 0.19
            + torch.cos((hidden_axis + 11 + 19 * row) * 0.0017) * 0.07
            + (row - 4.5) * 0.002
        )
        hidden.append(value)
    q_axis = torch.arange(1536, dtype=torch.float32)
    q_resid = (
        torch.sin((q_axis + 7) * 0.0043) * 0.23
        + torch.cos((q_axis + 17) * 0.0021) * 0.05
    ).unsqueeze(0)
    return torch.stack(hidden).to(torch.bfloat16), q_resid.to(torch.bfloat16)


def raw_pool(keys, gates, valid, ape, first_valid):
    rows, width = keys.shape
    pools = (rows + 3) // 4
    pooled = torch.zeros((pools, width), dtype=torch.bfloat16)
    indices = torch.full((pools, 4), -1, dtype=torch.int32)
    pool_valid = torch.zeros(pools, dtype=torch.int32)
    for pool in range(pools):
        start = first_valid + pool * 4
        members = []
        member_valid = []
        for member in range(4):
            row = start + member
            present = row < rows and bool(valid[row])
            members.append(min(row, rows - 1))
            member_valid.append(present)
            if present:
                indices[pool, member] = row
        pool_valid[pool] = int(all(member_valid))
        if not any(member_valid):
            continue
        grouped_keys = keys[members]
        grouped_gates = gates[members]
        logits = grouped_gates.float() + ape.float()
        mask = torch.tensor(member_valid, dtype=torch.bool).unsqueeze(1)
        logits = logits.masked_fill(~mask, float("-inf"))
        probabilities = torch.nan_to_num(
            logits.softmax(dim=0)).to(grouped_keys.dtype)
        pooled[pool] = (probabilities * grouped_keys).sum(dim=0)
    return pooled, indices, pool_valid


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def f32_bytes(value):
    return value.detach().float().contiguous().cpu().numpy().astype(
        "<f4", copy=False).tobytes()


def i32_bytes(value):
    return value.detach().to(torch.int32).contiguous().cpu().numpy().astype(
        "<i4", copy=False).tobytes()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--first-valid", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dump-prefix", required=True, type=Path)
    args = parser.parse_args()
    if args.layer < 3 or args.layer > 45:
        parser.error("--layer must be a sparse-MLA layer (3..45)")
    if args.rows < 5 or not 0 <= args.first_valid < args.rows:
        parser.error("invalid row/first-valid component case")

    torch.set_grad_enabled(False)
    torch.set_num_threads(min(16, max(1, torch.get_num_threads())))
    helpers = load_gguf_helpers()
    data_start, tensors = helpers.load_directory(args.model)
    prefix = f"blk.{args.layer}.indexer"
    with args.model.open("rb") as fp:
        blob = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        names = {
            "q": (f"{prefix}.attn_q_b.weight", (1536, 4096), 30),
            "k": (f"{prefix}.attn_k.weight", (4096, 128), 30),
            "weights": (f"{prefix}.proj.weight", (4096, 32), 30),
            "gate": (f"{prefix}.pool_gate.weight", (4096, 128), 30),
            "ape": (f"{prefix}.pool_ape.weight", (128, 4), 30),
            "norm_weight": (f"{prefix}.k_norm.weight", (128,), 0),
            "norm_bias": (f"{prefix}.k_norm.bias", (128,), 0),
        }
        payload_hash = hashlib.sha256()
        for name, shape, typ in names.values():
            payload_hash.update(tensor_raw(
                blob, data_start, tensors, name, shape, typ))

        wq = bf16_matrix(blob, data_start, tensors, names["q"][0], names["q"][1])
        wk = bf16_matrix(blob, data_start, tensors, names["k"][0], names["k"][1])
        ww = bf16_matrix(
            blob, data_start, tensors, names["weights"][0], names["weights"][1])
        wg = bf16_matrix(
            blob, data_start, tensors, names["gate"][0], names["gate"][1])
        ape = bf16_matrix(
            blob, data_start, tensors, names["ape"][0], names["ape"][1])
        norm_weight_f32 = f32_vector(
            blob, data_start, tensors, names["norm_weight"][0], 128)
        norm_bias_f32 = f32_vector(
            blob, data_start, tensors, names["norm_bias"][0], 128)
        for label, parameter in (
                ("k_norm.weight", norm_weight_f32),
                ("k_norm.bias", norm_bias_f32)):
            if not torch.equal(parameter, parameter.to(torch.bfloat16).float()):
                raise ValueError(
                    f"{label} F32 payload is not BF16-exact; this oracle "
                    "must be revised before accepting that checkpoint")
        norm_weight = norm_weight_f32.to(torch.bfloat16)
        norm_bias = norm_bias_f32.to(torch.bfloat16)

        hidden, q_resid = deterministic_inputs(args.rows)
        valid = torch.zeros(args.rows, dtype=torch.int32)
        valid[args.first_valid:] = 1
        q = F.linear(q_resid, wq).reshape(32, 128)
        k_raw = F.linear(hidden, wk)
        keys = F.layer_norm(
            k_raw, (128,), norm_weight, norm_bias, eps=1e-6)
        gates = F.linear(hidden, wg)
        weights_unscaled = F.linear(hidden[-1:], ww).reshape(32)
        weights = weights_unscaled.float() * (32.0 ** -0.5)
        pooled, pool_indices, pool_valid = raw_pool(
            keys, gates, valid, ape, args.first_valid)
        head_scores = torch.relu(
            q.float() @ pooled.float().T * (128.0 ** -0.5))
        scores = weights.unsqueeze(0) @ head_scores
        scores = scores.reshape(-1)
        scores = scores.masked_fill(
            ~pool_valid.bool(), torch.finfo(torch.float32).min)

        arrays = {
            ".hidden.f32": f32_bytes(hidden),
            ".qresid.f32": f32_bytes(q_resid),
            ".valid.u32": i32_bytes(valid),
            ".q.f32": f32_bytes(q),
            ".kraw.f32": f32_bytes(k_raw),
            ".key.f32": f32_bytes(keys),
            ".gate.f32": f32_bytes(gates),
            ".weights_unscaled.f32": f32_bytes(weights_unscaled),
            ".weights.f32": f32_bytes(weights),
            ".pooled.f32": f32_bytes(pooled),
            ".pool_indices.i32": i32_bytes(pool_indices),
            ".pool_valid.u32": i32_bytes(pool_valid),
            ".scores.f32": f32_bytes(scores),
        }
        document = {
            "status": "component oracle; not a promoted inference baseline",
            "formula_reference": TRANSFORMERS_REFERENCE,
            "model_size_bytes": args.model.stat().st_size,
            "tensor_payload_sha256": payload_hash.hexdigest(),
            "layer": args.layer,
            "rows": args.rows,
            "first_valid": args.first_valid,
            "query_row": args.rows - 1,
            "pools": (args.rows + 3) // 4,
            "dump_sha256": {
                suffix: digest(payload) for suffix, payload in arrays.items()
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.dump_prefix.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n")
        for suffix, payload in arrays.items():
            Path(str(args.dump_prefix) + suffix).write_bytes(payload)
        blob.close()

    print("PASS GLM5-next same-GGUF NoPE masked-score oracle")
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
