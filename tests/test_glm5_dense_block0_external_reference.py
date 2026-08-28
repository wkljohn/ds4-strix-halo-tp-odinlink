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
import struct
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


def official_mhc(hidden_streams, fn, base, scale, rms_eps, iterations=20,
                 hc_eps=1.0e-6):
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


def kda_layer(x, layer, blob, data_start, tensors, rms_norm_eps):
    """Official sequential KDA equations for one or more dense-layer rows."""
    prefix = f"blk.{layer}"
    norm = f32_tensor(blob, data_start, tensors,
                      f"{prefix}.attn_norm.weight", (4096,))
    hidden = rms_weighted(x, norm, rms_norm_eps)
    projected = {}
    for label in ("q", "k", "v"):
        projected[label] = project_bf16(
            hidden, blob, data_start, tensors,
            f"{prefix}.kda_{label}.weight", (4096, 8192))
        conv = f32_tensor(blob, data_start, tensors,
                          f"{prefix}.kda_{label}_conv.weight", (4, 1, 8192))
        conv = conv.reshape(8192, 4)
        history = np.zeros((8192, 3), dtype=np.float32)
        for token in range(x.shape[0]):
            current = projected[label][token].copy()
            raw = np.asarray(
                history[:, 0] * conv[:, 0] +
                history[:, 1] * conv[:, 1] +
                history[:, 2] * conv[:, 2] +
                current * conv[:, 3], dtype=np.float32)
            projected[label][token] = np.asarray(
                raw / (1.0 + np.exp(-raw)), dtype=np.float32)
            history[:, 0] = history[:, 1]
            history[:, 1] = history[:, 2]
            history[:, 2] = current

    rows = x.shape[0]
    q = projected["q"].reshape(rows, 64, 128)
    k = projected["k"].reshape(rows, 64, 128)
    v = projected["v"].reshape(rows, 64, 128)
    q = np.asarray(q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) +
                               np.float32(1.0e-6)) /
                   np.float32(math.sqrt(128.0)), dtype=np.float32)
    k = np.asarray(k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) +
                               np.float32(1.0e-6)), dtype=np.float32)

    f_low = project_bf16(hidden, blob, data_start, tensors,
                         f"{prefix}.kda_f_a.weight", (4096, 128))
    forget_projection = project_bf16(
        f_low, blob, data_start, tensors,
        f"{prefix}.kda_f_b.weight", (128, 8192)).reshape(rows, 64, 128)
    g_low = project_bf16(hidden, blob, data_start, tensors,
                         f"{prefix}.kda_g_a.weight", (4096, 128))
    out_gate = project_bf16(
        g_low, blob, data_start, tensors,
        f"{prefix}.kda_g_b.weight", (128, 8192)).reshape(rows, 64, 128)
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
    core = np.zeros((rows, 64, 128), dtype=np.float32)
    for token in range(rows):
        state *= np.exp(forget[token])[:, :, None]
        prediction = np.einsum("hi,hij->hj", k[token], state, optimize=True)
        delta = (v[token] - prediction) * beta[token, :, None]
        state += k[token, :, :, None] * delta[:, None, :]
        core[token] = np.einsum(
            "hi,hij->hj", q[token], state, optimize=True)

    o_norm = f32_tensor(blob, data_start, tensors,
                        f"{prefix}.kda_o_norm.weight", (128,))
    # Q/K L2 normalization above is defined with eps=1e-6.  The gated
    # output is a model RMSNorm and uses config.rms_norm_eps instead.
    gated = rms_weighted(core, o_norm, rms_norm_eps)
    gated = np.asarray(gated * (1.0 / (1.0 + np.exp(-out_gate))),
                       dtype=np.float32)
    return project_bf16(gated.reshape(rows, 8192), blob, data_start, tensors,
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
    parser.add_argument("--tokens", default="42")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--batch-output", type=Path)
    args = parser.parse_args()
    tokens = [int(value) for value in args.tokens.split(",") if value]
    if not tokens or len(tokens) > 33 or any(
            token < 0 or token >= 154880 for token in tokens):
        raise ValueError(f"invalid token sequence {args.tokens!r}")
    if args.layer < 0 or args.layer > 2:
        raise ValueError(f"dense trace layer {args.layer} is out of range")
    root = Path(__file__).parents[1]
    payload = load_module(root / "scripts/probe-glm5-next-kda-payload.py",
                          "glm5_payload")
    data_start, tensors, metadata = payload.load_directory(
        args.model, include_metadata=True)
    rms_norm_eps = metadata.get(
        "glm5-next.attention.layer_norm_rms_epsilon")
    hc_eps = metadata.get("glm5-next.hyper_connection.epsilon")
    expected_rms_eps = struct.unpack("<f", struct.pack("<f", 1.0e-5))[0]
    expected_hc_eps = struct.unpack("<f", struct.pack("<f", 1.0e-6))[0]
    if rms_norm_eps != expected_rms_eps:
        raise ValueError(
            f"unexpected GLM5-next RMS epsilon {rms_norm_eps!r}")
    if hc_eps != expected_hc_eps:
        raise ValueError(
            f"unexpected GLM5-next hyper-connection epsilon {hc_eps!r}")
    with args.model.open("rb") as fp:
        blob = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        embedding = np.stack([
            bf16_row(blob, data_start, tensors,
                     "token_embd.weight", (4096, 154880), token)
            for token in tokens])
        input_hc = np.repeat(embedding[:, None, :], 4, axis=1)
        last = slice(-1, None)

        def hc_weights(layer, site):
            fn = bf16_matrix(blob, data_start, tensors,
                             f"blk.{layer}.hc_{site}_fn.weight", (16384, 24))
            base = f32_tensor(blob, data_start, tensors,
                              f"blk.{layer}.hc_{site}_base.weight", (24,))
            scale = f32_tensor(blob, data_start, tensors,
                               f"blk.{layer}.hc_{site}_scale.weight", (3,))
            return fn, base, scale

        for layer in range(args.layer + 1):
            if layer == args.layer and not args.batch_output:
                compare("layer_input_hc", read_trace(
                    args.trace_prefix, "input_hc", (1, 4, 4096)),
                    input_hc[last], 0.0 if layer == 0 else 3.0e-4,
                    0.0 if layer == 0 else 3.0e-5)
            attn_weights = hc_weights(layer, "attn")
            attn_pre, attn_post, attn_comb, collapsed = official_mhc(
                input_hc, *attn_weights, rms_norm_eps, hc_eps=hc_eps)
            if layer == 0:
                wrong_pre, _, wrong_comb, _ = official_mhc(
                    input_hc, *attn_weights, rms_norm_eps, hc_eps=1.0e-5)
                wrong_delta = max(
                    float(np.max(np.abs(attn_pre - wrong_pre))),
                    float(np.max(np.abs(attn_comb - wrong_comb))))
                if wrong_delta <= 1.0e-7:
                    raise AssertionError(
                        "mHC oracle cannot distinguish metadata epsilon "
                        "from deliberate 1e-5 negative control")
            attn_split = np.concatenate(
                (attn_pre, attn_post, attn_comb.reshape(len(tokens), 16)),
                axis=-1)
            attn = kda_layer(collapsed, layer, blob, data_start, tensors,
                             rms_norm_eps)
            after_attn = (attn_post[..., None] * attn[:, None, :] +
                          np.matmul(attn_comb.transpose(0, 2, 1), input_hc))
            after_attn = np.asarray(after_attn, dtype=np.float32)
            ffn_pre, ffn_post, ffn_comb, ffn_collapsed = official_mhc(
                after_attn, *hc_weights(layer, "ffn"), rms_norm_eps,
                hc_eps=hc_eps)
            ffn_split = np.concatenate(
                (ffn_pre, ffn_post, ffn_comb.reshape(len(tokens), 16)),
                axis=-1)
            ffn_norm = f32_tensor(blob, data_start, tensors,
                                  f"blk.{layer}.ffn_norm.weight", (4096,))
            ffn_hidden = rms_weighted(ffn_collapsed, ffn_norm, rms_norm_eps)
            gate = np.asarray(ffn_hidden @ q8_matrix(
                blob, data_start, tensors, f"blk.{layer}.ffn_gate.weight",
                (4096, 12288)).T, dtype=np.float32)
            up = np.asarray(ffn_hidden @ q8_matrix(
                blob, data_start, tensors, f"blk.{layer}.ffn_up.weight",
                (4096, 12288)).T, dtype=np.float32)
            gate = np.minimum(gate, np.float32(10.0))
            up = np.clip(up, np.float32(-10.0), np.float32(10.0))
            mid = np.asarray((gate / (1.0 + np.exp(-gate))) * up,
                             dtype=np.float32)
            down = np.asarray(mid @ q8_matrix(
                blob, data_start, tensors, f"blk.{layer}.ffn_down.weight",
                (12288, 4096)).T, dtype=np.float32)
            output_hc = (ffn_post[..., None] * down[:, None, :] +
                         np.matmul(ffn_comb.transpose(0, 2, 1), after_attn))
            output_hc = np.asarray(output_hc, dtype=np.float32)
            if layer == args.layer and not args.batch_output:
                compare("attn_split", read_trace(
                    args.trace_prefix, "attn_split", (1, 24)),
                    attn_split[last])
                compare("attn_collapsed", read_trace(
                    args.trace_prefix, "attn_collapsed", (1, 4096)),
                    collapsed[last])
                compare("kda_output", read_trace(
                    args.trace_prefix, "attn_output", (1, 4096)),
                    attn[last])
                compare("attention_carry", read_trace(
                    args.trace_prefix, "after_attn", (1, 4, 4096)),
                    after_attn[last])
                compare("ffn_split", read_trace(
                    args.trace_prefix, "ffn_split", (1, 24)),
                    ffn_split[last])
                compare("ffn_hidden", read_trace(
                    args.trace_prefix, "ffn_hidden", (1, 4096)),
                    ffn_hidden[last])
                compare("ffn_mid", read_trace(
                    args.trace_prefix, "ffn_mid", (1, 12288)), mid[last])
                compare("ffn_down", read_trace(
                    args.trace_prefix, "ffn_down", (1, 4096)), down[last])
                compare("output_hc", read_trace(
                    args.trace_prefix, "output_hc", (1, 4, 4096)),
                    output_hc[last])
            input_hc = output_hc
        if args.batch_output:
            got = np.fromfile(args.batch_output, dtype="<f4")
            expected_shape = (len(tokens), 4, 4096)
            if got.size != int(np.prod(expected_shape)):
                raise ValueError(
                    f"{args.batch_output}: expected {expected_shape}, "
                    f"got {got.size} values")
            compare("batch_output_hc", got.reshape(expected_shape), input_hc,
                    rtol=3.0e-4, atol=3.0e-5)
        del embedding, input_hc
        blob.close()
    print("PASS independent same-GGUF GLM5 dense block-0 composition oracle")


if __name__ == "__main__":
    main()
