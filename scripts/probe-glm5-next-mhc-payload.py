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


def deterministic_hidden(tokens):
    hidden = np.empty((tokens, 4, 4096), dtype=np.float32)
    indices = np.arange(4096, dtype=np.float32)
    for token in range(tokens):
        for stream in range(4):
            hidden[token, stream] = (
                np.sin(np.float32(0.0031) *
                       (indices + np.float32(1 + 17 * token + 31 * stream))) *
                np.float32(0.2) + np.float32(0.01 * (stream - 1.5)))
    return hidden


def carry_branch(collapsed, ordinal):
    """Deterministic nontrivial stand-in for the attention/FFN branch.

    The carry oracle isolates hyper-connection ordering. Real branch operators
    have their own same-GGUF gates and are composed only in the modal-block
    stage; this function prevents an identity/zero branch from hiding a post
    weight or stream-order defect.
    """
    width = collapsed.shape[-1]
    phase = np.sin(np.arange(width, dtype=np.float32) * np.float32(0.0017) +
                   np.float32(ordinal + 1) * np.float32(0.13))
    return np.asarray(
        collapsed * np.float32(0.375 + 0.125 * ordinal) +
        phase[None, :] * np.float32(0.01), dtype=np.float32)


def compose_carry(hidden, branch, post, comb):
    # comb is [token][source][destination], matching the official model and
    # ds4's hc_expand kernel. Keep the transpose explicit in the einsum.
    residual = np.einsum("tsd,tsw->tdw", comb, hidden, dtype=np.float32)
    return np.asarray(
        residual + post[:, :, None] * branch[:, None, :], dtype=np.float32)


def next_site(layer, site):
    return (layer, "ffn") if site == "attn" else (layer + 1, "attn")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--site", choices=("attn", "ffn"), default="attn")
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--carry-sites", type=int, default=0,
                        help="chain this many consecutive attn/ffn mHC sites")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dump-prefix", type=Path)
    args = parser.parse_args()
    if not 0 <= args.layer < 45:
        parser.error("--layer must identify an mHC-bearing trunk layer (0..44)")
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    if args.carry_sites < 0:
        parser.error("--carry-sites must be nonnegative")

    helpers = load_gguf_helpers()
    data_start, tensors = helpers.load_directory(args.model)
    with args.model.open("rb") as fp:
        blob = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        hidden = deterministic_hidden(args.tokens)
        initial_hidden = hidden.copy()
        site_count = args.carry_sites if args.carry_sites else 1
        layer, site = args.layer, args.site
        site_documents = []
        site_arrays = []
        tensor_payload = bytearray()
        for ordinal in range(site_count):
            if layer >= 45:
                raise ValueError(
                    "mHC carry crosses beyond the 45 trunk mHC blocks; "
                    "MTP mHC tensors use a separate namespace")
            prefix = f"blk.{layer}.hc_{site}"
            fn = tensor_view(blob, data_start, tensors, prefix + "_fn.weight",
                             (16384, 24), 30)
            base = tensor_view(blob, data_start, tensors,
                               prefix + "_base.weight", (24,), 0).copy()
            scale = tensor_view(blob, data_start, tensors,
                                prefix + "_scale.weight", (3,), 0).copy()
            tensor_payload.extend(np.asarray(fn, dtype="<f4").tobytes())
            tensor_payload.extend(np.asarray(base, dtype="<f4").tobytes())
            tensor_payload.extend(np.asarray(scale, dtype="<f4").tobytes())
            post, comb, collapsed = mhc_reference(hidden, fn, base, scale)
            branch = carry_branch(collapsed, ordinal)
            carried = compose_carry(hidden, branch, post, comb)
            site_documents.append({
                "ordinal": ordinal,
                "layer": layer,
                "site": site,
                "post_f32_sha256": digest_array(post),
                "comb_f32_sha256": digest_array(comb),
                "collapsed_f32_sha256": digest_array(collapsed),
                "branch_f32_sha256": digest_array(branch),
                "carried_f32_sha256": digest_array(carried),
                "carried_l2": float(np.linalg.norm(carried.astype(np.float64))),
                "comb_row_sum_max_error": float(
                    np.max(np.abs(np.sum(comb, axis=-1) - np.float32(1.0)))),
                "comb_column_sum_max_error": float(
                    np.max(np.abs(np.sum(comb, axis=-2) - np.float32(1.0)))),
            })
            site_arrays.append((post, comb, collapsed, branch, carried))
            hidden = carried
            layer, site = next_site(layer, site)

    document = {
        "status": "component oracle; not a canonical or promoted baseline",
        "formula_reference": TRANSFORMERS_REFERENCE,
        "model_size_bytes": args.model.stat().st_size,
        "layer": args.layer,
        "site": args.site,
        "tokens": args.tokens,
        "carry_sites": args.carry_sites,
        "tensor_payload_f32_sha256": hashlib.sha256(tensor_payload).hexdigest(),
        "input_f32_sha256": digest_array(initial_hidden),
        "post_f32_sha256": site_documents[0]["post_f32_sha256"],
        "comb_f32_sha256": site_documents[0]["comb_f32_sha256"],
        "collapsed_f32_sha256": site_documents[0]["collapsed_f32_sha256"],
        "collapsed_l2": float(
            np.linalg.norm(site_arrays[0][2].astype(np.float64))),
        "comb_row_sum_max_error": site_documents[0]["comb_row_sum_max_error"],
        "comb_column_sum_max_error":
            site_documents[0]["comb_column_sum_max_error"],
        "sites": site_documents,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    if args.dump_prefix is not None:
        args.dump_prefix.parent.mkdir(parents=True, exist_ok=True)
        dumps = {".input.f32": initial_hidden}
        if args.carry_sites:
            for ordinal, arrays in enumerate(site_arrays):
                post, comb, collapsed, branch, carried = arrays
                dumps.update({
                    f".site{ordinal}.post.f32": post,
                    f".site{ordinal}.comb.f32": comb,
                    f".site{ordinal}.collapsed.f32": collapsed,
                    f".site{ordinal}.branch.f32": branch,
                    f".site{ordinal}.carried.f32": carried,
                })
        else:
            post, comb, collapsed, _, _ = site_arrays[0]
            dumps.update({
                ".post.f32": post,
                ".comb.f32": comb,
                ".collapsed.f32": collapsed,
            })
        for suffix, value in dumps.items():
            Path(str(args.dump_prefix) + suffix).write_bytes(
                np.asarray(value, dtype="<f4").tobytes())
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
