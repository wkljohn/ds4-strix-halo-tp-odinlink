#!/usr/bin/env python3
"""Generate a same-GGUF block-3 mHC-to-sparse-MLA decode oracle.

The ten-row case derives the attention input from real BF16 mHC pre-stage
weights, couples it to the Q/KV trunk, BF16 indexer, compact NoPE attention
and Q8_0 output projection, then carries the current-token branch through the
same mHC post-stage. It is an implementation oracle for the quantized GGUF,
not BF16-checkpoint parity.
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


def load_module(filename: str, name: str):
    source = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def f32_bytes(value: torch.Tensor) -> bytes:
    return value.detach().float().contiguous().numpy().astype(
        "<f4", copy=False).tobytes()


def i32_bytes(value: torch.Tensor) -> bytes:
    return value.detach().to(torch.int32).contiguous().numpy().astype(
        "<i4", copy=False).tobytes()


def bf16_boundary(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.bfloat16).float()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--first-valid", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dump-prefix", required=True, type=Path)
    args = parser.parse_args()
    if args.layer not in (*range(3, 44, 4), 45):
        parser.error("--layer must be a sparse-MLA layer")
    if args.rows != 10 or args.first_valid != 1:
        parser.error("the initial composition gate is pinned to rows=10, first-valid=1")

    torch.set_grad_enabled(False)
    torch.set_num_threads(min(16, max(1, torch.get_num_threads())))
    qkv = load_module("probe-glm5-next-mla-qkv.py", "glm5_mla_qkv")
    nope = load_module("probe-glm5-next-nope-score.py", "glm5_nope_score")
    mhc = load_module("probe-glm5-next-mhc-payload.py", "glm5_mhc")
    helpers = qkv.load_helpers()
    data_start, tensors = helpers.load_directory(args.model)
    prefix = f"blk.{args.layer}"
    indexer = f"{prefix}.indexer"
    payload_hash = hashlib.sha256()

    with args.model.open("rb") as fp:
        blob = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        hc_residual = mhc.deterministic_hidden(args.rows)
        hc_fn = mhc.tensor_view(
            blob, data_start, tensors, f"{prefix}.hc_attn_fn.weight",
            (16384, 24), 30).copy()
        hc_base = mhc.tensor_view(
            blob, data_start, tensors, f"{prefix}.hc_attn_base.weight",
            (24,), 0).copy()
        hc_scale = mhc.tensor_view(
            blob, data_start, tensors, f"{prefix}.hc_attn_scale.weight",
            (3,), 0).copy()
        for name, shape, typ in (
                (f"{prefix}.hc_attn_fn.weight", (16384, 24), 30),
                (f"{prefix}.hc_attn_base.weight", (24,), 0),
                (f"{prefix}.hc_attn_scale.weight", (3,), 0)):
            payload_hash.update(nope.tensor_raw(
                blob, data_start, tensors, name, shape, typ))
        hc_post, hc_comb, hc_collapsed = mhc.mhc_reference(
            hc_residual, hc_fn, hc_base, hc_scale)
        attn_norm_w, payload = qkv.f32_vector(
            blob, data_start, tensors, f"{prefix}.attn_norm.weight", 4096)
        payload_hash.update(payload)
        hidden = qkv.rms_norm_f32(
            torch.from_numpy(hc_collapsed.copy()), attn_norm_w)
        query_hidden = hidden[-1:]

        q_a_w, payload = qkv.q8_matrix(
            blob, data_start, tensors, f"{prefix}.attn_q_a.weight",
            (4096, 1536))
        payload_hash.update(payload)
        q_a = F.linear(query_hidden, q_a_w)
        del q_a_w
        q_norm, payload = qkv.f32_vector(
            blob, data_start, tensors, f"{prefix}.attn_q_a_norm.weight", 1536)
        payload_hash.update(payload)
        q_resid = qkv.rms_norm_f32(q_a, q_norm)

        q_b_w, payload = qkv.q8_matrix(
            blob, data_start, tensors, f"{prefix}.attn_q_b.weight",
            (1536, 16384))
        payload_hash.update(payload)
        query = F.linear(q_resid, q_b_w).reshape(64, 256)
        del q_b_w

        kv_w, payload = qkv.q8_matrix(
            blob, data_start, tensors, f"{prefix}.attn_kv_a_mqa.weight",
            (4096, 512))
        payload_hash.update(payload)
        kv_raw = F.linear(hidden, kv_w)
        del kv_w
        kv_norm_w, payload = qkv.f32_vector(
            blob, data_start, tensors, f"{prefix}.attn_kv_a_norm.weight", 512)
        payload_hash.update(payload)
        kv_norm = qkv.rms_norm_f32(kv_raw, kv_norm_w)

        k_b_w, payload = qkv.q8_matrix(
            blob, data_start, tensors, f"{prefix}.attn_k_b.weight",
            (256, 512, 64))
        payload_hash.update(payload)
        k_b_w = k_b_w.reshape(64, 512, 256)
        qk_low = torch.einsum("hi,hji->hj", query, k_b_w)
        del k_b_w

        bf16_specs = {
            "index_q": (f"{indexer}.attn_q_b.weight", (1536, 4096)),
            "index_k": (f"{indexer}.attn_k.weight", (4096, 128)),
            "index_weight": (f"{indexer}.proj.weight", (4096, 32)),
            "pool_gate": (f"{indexer}.pool_gate.weight", (4096, 128)),
            "pool_ape": (f"{indexer}.pool_ape.weight", (128, 4)),
        }
        bf16 = {}
        for label, (name, shape) in bf16_specs.items():
            raw = nope.tensor_raw(blob, data_start, tensors, name, shape, 30)
            payload_hash.update(raw)
            bf16[label] = nope.bf16_matrix(
                blob, data_start, tensors, name, shape).float()
            del raw

        norm_weight = nope.f32_vector(
            blob, data_start, tensors, f"{indexer}.k_norm.weight", 128)
        norm_bias = nope.f32_vector(
            blob, data_start, tensors, f"{indexer}.k_norm.bias", 128)
        payload_hash.update(nope.tensor_raw(
            blob, data_start, tensors, f"{indexer}.k_norm.weight", (128,), 0))
        payload_hash.update(nope.tensor_raw(
            blob, data_start, tensors, f"{indexer}.k_norm.bias", (128,), 0))
        for label, parameter in (("k_norm.weight", norm_weight),
                                 ("k_norm.bias", norm_bias)):
            if not torch.equal(parameter,
                               parameter.to(torch.bfloat16).float()):
                raise ValueError(
                    f"{label} F32 payload is not widened-BF16 exact")

        index_q = bf16_boundary(
            F.linear(q_resid.float(), bf16["index_q"])).reshape(32, 128)
        index_k_raw = bf16_boundary(
            F.linear(hidden.float(), bf16["index_k"]))
        index_key = F.layer_norm(
            index_k_raw.to(torch.bfloat16), (128,),
            norm_weight.to(torch.bfloat16), norm_bias.to(torch.bfloat16),
            eps=1.0e-6).float()
        pool_gate = bf16_boundary(
            F.linear(hidden.float(), bf16["pool_gate"]))
        # The ROCm BF16 projection keeps activations in F32, widens BF16
        # weights to F32, accumulates in F32, and rounds only its output at the
        # explicit following boundary. Do not round the mHC input here.
        head_weights_unscaled = bf16_boundary(F.linear(
            query_hidden.float(), bf16["index_weight"].float())).reshape(32)
        # round_bf16_inplace_tensor rounds its input first, then applies this
        # post-scale in F32; it does not round the scaled value again.
        head_weights = head_weights_unscaled * (32.0 ** -0.5)

        valid = torch.zeros(args.rows, dtype=torch.int32)
        valid[args.first_valid:] = 1
        pooled, pool_indices, pool_valid = nope.raw_pool(
            index_key.to(torch.bfloat16), pool_gate.to(torch.bfloat16), valid,
            bf16["pool_ape"].to(torch.bfloat16), args.first_valid)
        pooled = pooled.float()
        head_scores = torch.relu(
            index_q.float() @ pooled.T * (128.0 ** -0.5))
        pool_scores = (head_weights.unsqueeze(0) @ head_scores).reshape(-1)
        pool_scores = pool_scores.masked_fill(
            ~pool_valid.bool(), torch.finfo(torch.float32).min)

        complete = torch.nonzero(pool_valid, as_tuple=False).reshape(-1)
        select_count = min(2048 // 4, complete.numel())
        selected_pools = complete[torch.argsort(
            pool_scores[complete], descending=True, stable=True)[:select_count]]
        selected_tokens = []
        for pool in selected_pools.tolist():
            selected_tokens.extend(pool_indices[pool].tolist())
        visible = torch.nonzero(valid, as_tuple=False).reshape(-1)
        tail_count = visible.numel() & 3
        if tail_count:
            selected_tokens.extend(visible[-tail_count:].tolist())
        selected_tokens = torch.tensor(selected_tokens, dtype=torch.int32)
        if selected_tokens.numel() != 9 or torch.any(selected_tokens < 0):
            raise ValueError(f"unexpected compact selection {selected_tokens.tolist()}")

        chosen_kv = kv_norm[selected_tokens.long()]
        attention_scores = qk_low @ chosen_kv.T / math.sqrt(256.0)
        probabilities = torch.softmax(attention_scores, dim=-1)
        attention_lora = probabilities @ chosen_kv
        value_w, payload = qkv.q8_matrix(
            blob, data_start, tensors, f"{prefix}.attn_v_b.weight",
            (512, 256, 64))
        payload_hash.update(payload)
        value_w = value_w.reshape(64, 256, 512)
        heads = torch.einsum("hj,hdj->hd", attention_lora, value_w)
        attn_output_w, payload = qkv.q8_matrix(
            blob, data_start, tensors, f"{prefix}.attn_output.weight",
            (16384, 4096))
        payload_hash.update(payload)
        attn_output = F.linear(heads.reshape(1, 16384), attn_output_w)
        hc_carried = mhc.compose_carry(
            hc_residual[-1:], attn_output.detach().float().numpy(),
            hc_post[-1:], hc_comb[-1:])

        # Continue the exact block-3 state into the FFN pre-stage.  This is
        # deliberately part of the same oracle: a standalone MoE test fed by
        # synthetic hidden states cannot catch a stream-order or BF16-boundary
        # defect between attention mHC post and FFN mHC pre.
        hc_ffn_fn = mhc.tensor_view(
            blob, data_start, tensors, f"{prefix}.hc_ffn_fn.weight",
            (16384, 24), 30).copy()
        hc_ffn_base = mhc.tensor_view(
            blob, data_start, tensors, f"{prefix}.hc_ffn_base.weight",
            (24,), 0).copy()
        hc_ffn_scale = mhc.tensor_view(
            blob, data_start, tensors, f"{prefix}.hc_ffn_scale.weight",
            (3,), 0).copy()
        for name, shape, typ in (
                (f"{prefix}.hc_ffn_fn.weight", (16384, 24), 30),
                (f"{prefix}.hc_ffn_base.weight", (24,), 0),
                (f"{prefix}.hc_ffn_scale.weight", (3,), 0)):
            payload_hash.update(nope.tensor_raw(
                blob, data_start, tensors, name, shape, typ))
        ffn_post, ffn_comb, ffn_collapsed = mhc.mhc_reference(
            np.asarray(hc_carried, dtype=np.float32), hc_ffn_fn,
            hc_ffn_base, hc_ffn_scale)
        ffn_flat = np.asarray(hc_carried, dtype=np.float32).reshape(1, -1)
        ffn_denominator = np.sqrt(
            np.mean(ffn_flat * ffn_flat, axis=-1, keepdims=True) +
            np.float32(1.0e-5))
        ffn_mixed = np.asarray(
            (ffn_flat / ffn_denominator) @ hc_ffn_fn.astype(np.float32).T,
            dtype=np.float32)
        ffn_pre = (
            mhc.sigmoid(
                ffn_mixed[:, :4] * hc_ffn_scale[0] + hc_ffn_base[:4]) +
            np.float32(1.0e-6))
        ffn_split = np.concatenate(
            (ffn_pre, ffn_post, ffn_comb.reshape(1, -1)), axis=-1)
        ffn_norm_w, payload = qkv.f32_vector(
            blob, data_start, tensors, f"{prefix}.ffn_norm.weight", 4096)
        payload_hash.update(payload)
        ffn_hidden = qkv.rms_norm_f32(
            torch.from_numpy(ffn_collapsed.copy()), ffn_norm_w)

        router_w = mhc.tensor_view(
            blob, data_start, tensors, f"{prefix}.ffn_gate_inp.weight",
            (4096, 288), 0).copy().reshape(288, 4096)
        router_bias = mhc.tensor_view(
            blob, data_start, tensors, f"{prefix}.exp_probs_b.bias",
            (288,), 0).copy()
        for name, shape in (
                (f"{prefix}.ffn_gate_inp.weight", (4096, 288)),
                (f"{prefix}.exp_probs_b.bias", (288,))):
            payload_hash.update(nope.tensor_raw(
                blob, data_start, tensors, name, shape, 0))
        router_logits = F.linear(
            ffn_hidden, torch.from_numpy(router_w.copy()))
        router_probs = torch.sigmoid(router_logits)
        router_order = torch.argsort(
            router_probs + torch.from_numpy(router_bias.copy()), dim=-1,
            descending=True, stable=True)
        router_ids = router_order[:, :8].to(torch.int32)
        router_weights = torch.gather(router_probs, 1, router_ids.long())
        router_weights = (router_weights /
                          router_weights.sum(dim=-1, keepdim=True).clamp_min(
                              6.103515625e-5)) * 2.5

        shared_gate_w, payload = qkv.q8_matrix(
            blob, data_start, tensors, f"{prefix}.ffn_gate_shexp.weight",
            (4096, 2048))
        payload_hash.update(payload)
        shared_up_w, payload = qkv.q8_matrix(
            blob, data_start, tensors, f"{prefix}.ffn_up_shexp.weight",
            (4096, 2048))
        payload_hash.update(payload)
        shared_gate = F.linear(ffn_hidden, shared_gate_w).clamp(max=10.0)
        shared_up = F.linear(ffn_hidden, shared_up_w).clamp(-10.0, 10.0)
        shared_mid = F.silu(shared_gate) * shared_up
        del shared_gate_w, shared_up_w
        shared_down_w, payload = qkv.q8_matrix(
            blob, data_start, tensors, f"{prefix}.ffn_down_shexp.weight",
            (2048, 4096))
        payload_hash.update(payload)
        shared_output = F.linear(shared_mid, shared_down_w)
        del shared_down_w
        blob.close()

    arrays = {
        ".hc_residual.f32": np.asarray(hc_residual, dtype="<f4").tobytes(),
        ".hc_post.f32": np.asarray(hc_post, dtype="<f4").tobytes(),
        ".hc_comb.f32": np.asarray(hc_comb, dtype="<f4").tobytes(),
        ".hc_collapsed.f32": np.asarray(hc_collapsed, dtype="<f4").tobytes(),
        ".hc_carried.f32": np.asarray(hc_carried, dtype="<f4").tobytes(),
        ".ffn_post.f32": np.asarray(ffn_post, dtype="<f4").tobytes(),
        ".ffn_comb.f32": np.asarray(ffn_comb, dtype="<f4").tobytes(),
        ".ffn_split.f32": np.asarray(ffn_split, dtype="<f4").tobytes(),
        ".ffn_collapsed.f32": np.asarray(ffn_collapsed, dtype="<f4").tobytes(),
        ".ffn_hidden.f32": f32_bytes(ffn_hidden),
        ".router_logits.f32": f32_bytes(router_logits),
        ".router_probs.f32": f32_bytes(router_probs),
        ".router_ids.i32": i32_bytes(router_ids),
        ".router_weights.f32": f32_bytes(router_weights),
        ".shared_gate.f32": f32_bytes(shared_gate),
        ".shared_up.f32": f32_bytes(shared_up),
        ".shared_mid.f32": f32_bytes(shared_mid),
        ".shared_output.f32": f32_bytes(shared_output),
        ".hidden.f32": f32_bytes(hidden),
        ".q_a.f32": f32_bytes(q_a),
        ".q_resid.f32": f32_bytes(q_resid),
        ".query.f32": f32_bytes(query),
        ".kv_raw.f32": f32_bytes(kv_raw),
        ".kv_norm.f32": f32_bytes(kv_norm),
        ".qk_low.f32": f32_bytes(qk_low),
        ".valid.u32": i32_bytes(valid),
        ".index_q.f32": f32_bytes(index_q),
        ".index_k_raw.f32": f32_bytes(index_k_raw),
        ".index_key.f32": f32_bytes(index_key),
        ".pool_gate.f32": f32_bytes(pool_gate),
        ".head_weights.f32": f32_bytes(head_weights),
        ".pooled.f32": f32_bytes(pooled),
        ".pool_indices.i32": i32_bytes(pool_indices),
        ".pool_valid.u32": i32_bytes(pool_valid),
        ".pool_scores.f32": f32_bytes(pool_scores),
        ".selected_pools.u32": i32_bytes(selected_pools),
        ".selected_tokens.i32": i32_bytes(selected_tokens),
        ".heads.f32": f32_bytes(heads),
        ".attn_output.f32": f32_bytes(attn_output),
    }
    document = {
        "status": "same-GGUF mHC-to-sparse-MLA component oracle; not promoted inference",
        "model_size_bytes": args.model.stat().st_size,
        "layer": args.layer,
        "rows": args.rows,
        "first_valid": args.first_valid,
        "selected_count": selected_tokens.numel(),
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
    print("PASS GLM5-next sparse-MLA heads component oracle")
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
