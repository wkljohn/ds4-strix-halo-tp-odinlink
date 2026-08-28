#!/usr/bin/env python3
"""Independent real-GGUF composition oracle for GLM-5.3 dense block 0.

The implementation under test emits small FP32 stage traces. This script reads
the GGUF payload directly and transcribes the official Transformers mHC, KDA,
RMSNorm and dense-MLP equations with dequantized Q8_0 weights. It therefore
checks stage wiring and constants without sharing any DS4 GPU arithmetic.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import mmap
from pathlib import Path

import numpy as np


TRANSFORMERS_GLM5_NEXT = (
    "https://github.com/huggingface/transformers/blob/"
    "0b80c4ab9112ceecb7f5572719094679b6aba598/"
    "src/transformers/models/glm5_next/modeling_glm5_next.py"
)
# Formula pins in that file:
#   dense clamped SwiGLU: lines 80-97
#   mHC mapping/carry: lines 203-280 and 1136-1202
#   KDA normalization, 1/sqrt(head_dim), recurrence: lines 282-520


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_meta(tensors, name, shape, typ):
    dims, got_type, relative = tensors.get(name, (None, None, None))
    if dims != shape or got_type != typ:
        raise ValueError(f"{name}: expected {(shape, typ)}, got {(dims, got_type)}")
    return relative


def bf16_matrix(blob, data_start, tensors, name, shape):
    relative = tensor_meta(tensors, name, shape, 30)
    count = int(np.prod(shape))
    raw = np.frombuffer(blob, dtype="<u2", count=count,
                        offset=data_start + relative)
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(
        int(np.prod(shape[1:])), shape[0])


def bf16_row(blob, data_start, tensors, name, shape, row):
    relative = tensor_meta(tensors, name, shape, 30)
    if row < 0 or row >= int(np.prod(shape[1:])):
        raise ValueError(f"{name}: row {row} is out of range")
    raw = np.frombuffer(blob, dtype="<u2", count=shape[0],
                        offset=data_start + relative + row * shape[0] * 2)
    return (raw.astype(np.uint32) << 16).view(np.float32)


def f32_tensor(blob, data_start, tensors, name, shape):
    relative = tensor_meta(tensors, name, shape, 0)
    return np.frombuffer(blob, dtype="<f4", count=int(np.prod(shape)),
                         offset=data_start + relative).copy().reshape(shape)


def q8_matrix(blob, data_start, tensors, name, shape):
    relative = tensor_meta(tensors, name, shape, 8)
    rows = int(np.prod(shape[1:]))
    blocks = shape[0] // 32
    size = rows * blocks * 34
    raw = np.frombuffer(blob, dtype=np.uint8, count=size,
                        offset=data_start + relative).reshape(rows, blocks, 34)
    scales = raw[:, :, :2].copy().view("<f2").reshape(rows, blocks).astype(np.float32)
    quants = raw[:, :, 2:].view(np.int8).astype(np.float32)
    return np.asarray((quants * scales[:, :, None]).reshape(rows, shape[0]),
                      dtype=np.float32)


def project_bf16(x, blob, data_start, tensors, name, shape):
    weight = bf16_matrix(blob, data_start, tensors, name, shape)
    result = np.asarray(x @ weight.T, dtype=np.float32)
    del weight
    return result


def rms_weighted(x, weight, eps):
    scale = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(eps))
    return np.asarray(x / scale * weight, dtype=np.float32)


def official_mhc(hidden_streams, fn, base, scale, iterations=20,
                 hc_eps=1.0e-6, rms_eps=1.0e-5):
    """Direct transcription of Glm5NextTextHyperConnection.forward."""
    flat = hidden_streams.reshape(hidden_streams.shape[0], -1).astype(np.float32)
    flat /= np.sqrt(np.mean(flat * flat, axis=-1, keepdims=True) +
                    np.float32(rms_eps))
    mixed = np.asarray(flat @ fn.T, dtype=np.float32)
    pre_w, post_w, comb_w = np.split(mixed, [4, 8], axis=-1)
    pre = np.asarray(1.0 / (1.0 + np.exp(-(
        pre_w * scale[0] + base[:4]))) + np.float32(hc_eps), dtype=np.float32)
    post = np.asarray(2.0 / (1.0 + np.exp(-(
        post_w * scale[1] + base[4:8]))), dtype=np.float32)
    logits = comb_w.reshape(-1, 4, 4) * scale[2] + base[8:].reshape(1, 4, 4)
    logits -= np.max(logits, axis=-1, keepdims=True)
    comb = np.exp(logits)
    comb = np.asarray(comb / np.sum(comb, axis=-1, keepdims=True) +
                      np.float32(hc_eps), dtype=np.float32)
    comb /= np.sum(comb, axis=-2, keepdims=True) + np.float32(hc_eps)
    for _ in range(iterations - 1):
        comb /= np.sum(comb, axis=-1, keepdims=True) + np.float32(hc_eps)
        comb /= np.sum(comb, axis=-2, keepdims=True) + np.float32(hc_eps)
    collapsed = np.sum(pre[..., None] * hidden_streams, axis=1,
                       dtype=np.float32)
    return pre, post, np.asarray(comb, dtype=np.float32), collapsed


def kda_layer0(x, blob, data_start, tensors):
    # A one-token, zero-state block oracle covers the current-token KDA update
    # but not historical decay or the first three causal-convolution taps. The
    # multi-token KDA external-reference tests own those independent gates.
    prefix = "blk.0"
    norm = f32_tensor(blob, data_start, tensors,
                      f"{prefix}.attn_norm.weight", (4096,))
    hidden = rms_weighted(x, norm, 1.0e-5)
    projected = {}
    for label in ("q", "k", "v"):
        projected[label] = project_bf16(
            hidden, blob, data_start, tensors,
            f"{prefix}.kda_{label}.weight", (4096, 8192))
        conv = f32_tensor(blob, data_start, tensors,
                          f"{prefix}.kda_{label}_conv.weight", (4, 1, 8192))
        conv = conv.reshape(8192, 4)
        raw = np.asarray(projected[label] * conv[:, 3], dtype=np.float32)
        projected[label] = np.asarray(raw / (1.0 + np.exp(-raw)), dtype=np.float32)

    q = projected["q"].reshape(1, 64, 128)
    k = projected["k"].reshape(1, 64, 128)
    v = projected["v"].reshape(1, 64, 128)
    q = np.asarray(q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) +
                               np.float32(1.0e-6)) /
                   np.float32(math.sqrt(128.0)), dtype=np.float32)
    k = np.asarray(k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) +
                               np.float32(1.0e-6)), dtype=np.float32)

    f_low = project_bf16(hidden, blob, data_start, tensors,
                         f"{prefix}.kda_f_a.weight", (4096, 128))
    forget_projection = project_bf16(
        f_low, blob, data_start, tensors,
        f"{prefix}.kda_f_b.weight", (128, 8192)).reshape(1, 64, 128)
    g_low = project_bf16(hidden, blob, data_start, tensors,
                         f"{prefix}.kda_g_a.weight", (4096, 128))
    out_gate = project_bf16(
        g_low, blob, data_start, tensors,
        f"{prefix}.kda_g_b.weight", (128, 8192)).reshape(1, 64, 128)
    beta = project_bf16(hidden, blob, data_start, tensors,
                        f"{prefix}.kda_beta.weight", (4096, 64))
    beta = np.asarray(1.0 / (1.0 + np.exp(-beta)), dtype=np.float32)
    dt = f32_tensor(blob, data_start, tensors,
                    f"{prefix}.kda_dt_bias.weight", (8192,)).reshape(64, 128)
    a_log = f32_tensor(blob, data_start, tensors,
                       f"{prefix}.kda_a_log.weight", (64,))
    forget = np.asarray(
        -5.0 / (1.0 + np.exp(-np.exp(a_log)[None, :, None] *
                              (forget_projection + dt[None, :, :]))),
        dtype=np.float32)

    state = np.zeros((64, 128, 128), dtype=np.float32)
    state *= np.exp(forget[0])[:, :, None]
    prediction = np.einsum("hi,hij->hj", k[0], state, optimize=True)
    delta = (v[0] - prediction) * beta[0, :, None]
    state += k[0, :, :, None] * delta[:, None, :]
    core = np.einsum("hi,hij->hj", q[0], state, optimize=True)

    o_norm = f32_tensor(blob, data_start, tensors,
                        f"{prefix}.kda_o_norm.weight", (128,))
    gated = rms_weighted(core, o_norm, 1.0e-6)
    gated = np.asarray(gated * (1.0 / (1.0 + np.exp(-out_gate[0]))),
                       dtype=np.float32)
    return project_bf16(gated.reshape(1, 8192), blob, data_start, tensors,
                        f"{prefix}.kda_output.weight", (8192, 4096))


def read_trace(prefix: Path, name: str, shape):
    path = Path(f"{prefix}.{name}.f32")
    value = np.fromfile(path, dtype="<f4")
    if value.size != int(np.prod(shape)):
        raise ValueError(f"{path}: expected {shape}, got {value.size} values")
    return value.reshape(shape)


def compare(label, got, want, rtol=3.0e-4, atol=3.0e-5):
    delta = np.abs(got.astype(np.float64) - want.astype(np.float64))
    limit = atol + rtol * np.abs(want.astype(np.float64))
    max_abs = float(delta.max(initial=0.0))
    rel_l2 = float(np.linalg.norm(delta.ravel()) /
                   max(np.linalg.norm(want.astype(np.float64).ravel()), 1.0e-30))
    if not np.isfinite(got).all() or np.any(delta > limit):
        worst = int(np.argmax(delta - limit))
        raise AssertionError(
            f"{label}: mismatch max_abs={max_abs:.9g} rel_l2={rel_l2:.9g} "
            f"worst={worst} got={got.ravel()[worst]:.9g} "
            f"want={want.ravel()[worst]:.9g}")
    print(f"{label}: max_abs={max_abs:.9g} rel_l2={rel_l2:.9g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("trace_prefix", type=Path)
    args = parser.parse_args()
    root = Path(__file__).parents[1]
    payload = load_module(root / "scripts/probe-glm5-next-kda-payload.py",
                          "glm5_payload")
    data_start, tensors = payload.load_directory(args.model)
    with args.model.open("rb") as fp:
        blob = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        embedding = bf16_row(blob, data_start, tensors,
                             "token_embd.weight", (4096, 154880), 42)
        input_hc = np.repeat(embedding[None, None, :], 4, axis=1)
        compare("embedding_hc", read_trace(args.trace_prefix, "input_hc",
                                           (1, 4, 4096)), input_hc, 0.0, 0.0)

        def hc_weights(site):
            fn = bf16_matrix(blob, data_start, tensors,
                             f"blk.0.hc_{site}_fn.weight", (16384, 24))
            base = f32_tensor(blob, data_start, tensors,
                              f"blk.0.hc_{site}_base.weight", (24,))
            scale = f32_tensor(blob, data_start, tensors,
                               f"blk.0.hc_{site}_scale.weight", (3,))
            return fn, base, scale

        attn_pre, attn_post, attn_comb, collapsed = official_mhc(
            input_hc, *hc_weights("attn"))
        attn_split = np.concatenate(
            (attn_pre, attn_post, attn_comb.reshape(1, 16)), axis=-1)
        compare("attn_split", read_trace(args.trace_prefix, "attn_split",
                                         (1, 24)), attn_split)
        compare("attn_collapsed",
                read_trace(args.trace_prefix, "attn_collapsed", (1, 4096)),
                collapsed)
        attn = kda_layer0(collapsed, blob, data_start, tensors)
        compare("kda_output", read_trace(args.trace_prefix, "attn_output",
                                         (1, 4096)), attn)
        after_attn = (attn_post[..., None] * attn[:, None, :] +
                      np.matmul(attn_comb.transpose(0, 2, 1), input_hc))
        after_attn = np.asarray(after_attn, dtype=np.float32)
        compare("attention_carry", read_trace(args.trace_prefix, "after_attn",
                                              (1, 4, 4096)), after_attn)

        ffn_pre, ffn_post, ffn_comb, ffn_collapsed = official_mhc(
            after_attn, *hc_weights("ffn"))
        ffn_split = np.concatenate(
            (ffn_pre, ffn_post, ffn_comb.reshape(1, 16)), axis=-1)
        compare("ffn_split", read_trace(args.trace_prefix, "ffn_split",
                                        (1, 24)), ffn_split)
        ffn_norm = f32_tensor(blob, data_start, tensors,
                              "blk.0.ffn_norm.weight", (4096,))
        ffn_hidden = rms_weighted(ffn_collapsed, ffn_norm, 1.0e-5)
        compare("ffn_hidden", read_trace(args.trace_prefix, "ffn_hidden",
                                         (1, 4096)), ffn_hidden)
        gate = np.asarray(ffn_hidden @ q8_matrix(
            blob, data_start, tensors, "blk.0.ffn_gate.weight",
            (4096, 12288)).T, dtype=np.float32)
        up = np.asarray(ffn_hidden @ q8_matrix(
            blob, data_start, tensors, "blk.0.ffn_up.weight",
            (4096, 12288)).T, dtype=np.float32)
        gate = np.minimum(gate, np.float32(10.0))
        up = np.clip(up, np.float32(-10.0), np.float32(10.0))
        mid = np.asarray((gate / (1.0 + np.exp(-gate))) * up, dtype=np.float32)
        compare("ffn_mid", read_trace(args.trace_prefix, "ffn_mid",
                                      (1, 12288)), mid)
        down = np.asarray(mid @ q8_matrix(
            blob, data_start, tensors, "blk.0.ffn_down.weight",
            (12288, 4096)).T, dtype=np.float32)
        compare("ffn_down", read_trace(args.trace_prefix, "ffn_down",
                                       (1, 4096)), down)
        output_hc = (ffn_post[..., None] * down[:, None, :] +
                     np.matmul(ffn_comb.transpose(0, 2, 1), after_attn))
        output_hc = np.asarray(output_hc, dtype=np.float32)
        compare("output_hc", read_trace(args.trace_prefix, "output_hc",
                                        (1, 4, 4096)), output_hc)
        del embedding, input_hc
        blob.close()
    print("PASS independent same-GGUF GLM5 dense block-0 composition oracle")


if __name__ == "__main__":
    main()
