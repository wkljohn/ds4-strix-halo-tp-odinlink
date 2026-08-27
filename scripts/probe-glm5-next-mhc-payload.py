#!/usr/bin/env python3
"""Generate a deterministic same-GGUF GLM-5.3 mHC reference payload.

This is a component oracle, not a promoted inference baseline. It reads the
actual BF16/F32 mHC weights from one layer, applies the model's FP32 mapping
and Sinkhorn order to deterministic four-stream inputs, and optionally emits
raw FP32 arrays for an implementation-under-test to compare against.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import mmap
from pathlib import Path

import numpy as np


TRANSFORMERS_REFERENCE = (
    "https://github.com/huggingface/transformers/blob/"
    "155b89935a648278dd38c78184cbc40e6a65f14b/"
    "src/transformers/models/deepseek_v4/modular_deepseek_v4.py#L769-L845"
)


def load_gguf_helpers():
    source = Path(__file__).with_name("probe-glm5-next-kda-payload.py")
    spec = importlib.util.spec_from_file_location("glm5_kda_payload", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import GGUF helpers from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_view(blob, data_start, tensors, name, shape, typ):
    dims, got_type, rel = tensors.get(name, (None, None, None))
    if dims != shape or got_type != typ:
        raise ValueError(
            f"{name}: expected shape/type {(shape, typ)}, got {(dims, got_type)}")
    base = data_start + rel
    if typ == 30:
        raw = np.frombuffer(blob, dtype="<u2", count=int(np.prod(shape)), offset=base)
        # GGUF dimensions are [input, output]; physical rows are output rows.
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape[1], shape[0])
    if typ == 0:
        return np.frombuffer(blob, dtype="<f4", count=int(np.prod(shape)), offset=base)
    raise ValueError(f"unsupported tensor type {typ} for {name}")


def softmax_rows(values):
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def sigmoid(values):
    return np.asarray(1.0 / (1.0 + np.exp(-values)), dtype=np.float32)


def mhc_reference(hidden_streams, fn, base, scale, iterations=20,
                  hc_eps=1e-6, rms_eps=1e-5):
    """Independent NumPy transcription of the official FP32 mHC equations."""
    tokens, hc, width = hidden_streams.shape
    if (hc, fn.shape, base.shape, scale.shape) != (
            4, (24, hc * width), (24,), (3,)):
        raise ValueError("unexpected GLM-5.3 mHC dimensions")
    flat = hidden_streams.reshape(tokens, hc * width).astype(np.float32)
    denominator = np.sqrt(
        np.mean(flat * flat, axis=-1, keepdims=True) + np.float32(rms_eps))
    normalized = np.asarray(flat / denominator, dtype=np.float32)
    mixed = np.asarray(normalized @ fn.astype(np.float32).T, dtype=np.float32)

    pre_logits, post_logits, comb_logits = np.split(mixed, [hc, 2 * hc], axis=-1)
    pre = sigmoid(pre_logits * scale[0] + base[:hc]) + np.float32(hc_eps)
    post = np.float32(2.0) * sigmoid(
        post_logits * scale[1] + base[hc:2 * hc])
    comb_logits = (
        comb_logits.reshape(tokens, hc, hc) * scale[2] +
        base[2 * hc:].reshape(1, hc, hc))
    comb = softmax_rows(comb_logits).astype(np.float32) + np.float32(hc_eps)
    comb /= np.sum(comb, axis=-2, keepdims=True) + np.float32(hc_eps)
    for _ in range(iterations - 1):
        comb /= np.sum(comb, axis=-1, keepdims=True) + np.float32(hc_eps)
        comb /= np.sum(comb, axis=-2, keepdims=True) + np.float32(hc_eps)

    collapsed = np.sum(pre[..., None] * hidden_streams, axis=1, dtype=np.float32)
    return (np.asarray(post, dtype=np.float32),
            np.asarray(comb, dtype=np.float32),
            np.asarray(collapsed, dtype=np.float32))


def digest_array(value):
    return hashlib.sha256(np.asarray(value, dtype="<f4").tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--site", choices=("attn", "ffn"), default="attn")
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dump-prefix", type=Path)
    args = parser.parse_args()
    if not 0 <= args.layer < 45:
        parser.error("--layer must identify an mHC-bearing trunk layer (0..44)")
    if args.tokens < 1:
        parser.error("--tokens must be positive")

    helpers = load_gguf_helpers()
    data_start, tensors = helpers.load_directory(args.model)
    prefix = f"blk.{args.layer}.hc_{args.site}"
    with args.model.open("rb") as fp:
        blob = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        fn = tensor_view(blob, data_start, tensors, prefix + "_fn.weight",
                         (16384, 24), 30)
        base = tensor_view(blob, data_start, tensors, prefix + "_base.weight",
                           (24,), 0).copy()
        scale = tensor_view(blob, data_start, tensors, prefix + "_scale.weight",
                            (3,), 0).copy()
        tensor_payload = (
            np.asarray(fn, dtype="<f4").tobytes() +
            np.asarray(base, dtype="<f4").tobytes() +
            np.asarray(scale, dtype="<f4").tobytes())

        hidden = np.empty((args.tokens, 4, 4096), dtype=np.float32)
        for token in range(args.tokens):
            for stream in range(4):
                indices = np.arange(4096, dtype=np.float32)
                hidden[token, stream] = (
                    np.sin(np.float32(0.0031) *
                           (indices + np.float32(1 + 17 * token + 31 * stream))) *
                    np.float32(0.2) + np.float32(0.01 * (stream - 1.5)))
        post, comb, collapsed = mhc_reference(hidden, fn, base, scale)

    document = {
        "status": "component oracle; not a canonical or promoted baseline",
        "formula_reference": TRANSFORMERS_REFERENCE,
        "model_size_bytes": args.model.stat().st_size,
        "layer": args.layer,
        "site": args.site,
        "tokens": args.tokens,
        "tensor_payload_f32_sha256": hashlib.sha256(tensor_payload).hexdigest(),
        "input_f32_sha256": digest_array(hidden),
        "post_f32_sha256": digest_array(post),
        "comb_f32_sha256": digest_array(comb),
        "collapsed_f32_sha256": digest_array(collapsed),
        "collapsed_l2": float(np.linalg.norm(collapsed.astype(np.float64))),
        "comb_row_sum_max_error": float(
            np.max(np.abs(np.sum(comb, axis=-1) - np.float32(1.0)))),
        "comb_column_sum_max_error": float(
            np.max(np.abs(np.sum(comb, axis=-2) - np.float32(1.0)))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    if args.dump_prefix is not None:
        args.dump_prefix.parent.mkdir(parents=True, exist_ok=True)
        for suffix, value in {
                ".input.f32": hidden,
                ".post.f32": post,
                ".comb.f32": comb,
                ".collapsed.f32": collapsed}.items():
            Path(str(args.dump_prefix) + suffix).write_bytes(
                np.asarray(value, dtype="<f4").tobytes())
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
