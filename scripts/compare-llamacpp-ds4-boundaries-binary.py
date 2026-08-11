#!/usr/bin/env python3
"""Compare full binary llama.cpp and DS4 tensors at selected boundaries."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def read(path: Path, expected: int) -> np.ndarray:
    value = np.fromfile(path, dtype="<f4")
    if value.size != expected:
        raise ValueError(f"{path}: expected {expected} floats, got {value.size}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{path}: contains non-finite values")
    return value


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    ref = reference.astype(np.float64)
    got = actual.astype(np.float64)
    delta = got - ref
    rms_ref = math.sqrt(float(np.mean(ref * ref)))
    rms_got = math.sqrt(float(np.mean(got * got)))
    rms_error = math.sqrt(float(np.mean(delta * delta)))
    denom = math.sqrt(float(np.dot(ref, ref) * np.dot(got, got)))
    return {
        "rms_reference": rms_ref,
        "rms_actual": rms_got,
        "rms_error": rms_error,
        "relative_rms": rms_error / rms_ref if rms_ref else 0.0,
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "cosine": float(np.dot(ref, got) / denom) if denom else 1.0,
    }


def llama_row(binary_dir: Path, tensor: str, layer: int, position: int) -> np.ndarray:
    if tensor == "attn_low":
        path = binary_dir / f"attn_wo_a-{layer}__permuted___cont_-0.bin"
        return read(path, 20 * 8192).reshape(20, 8192)[position]
    if tensor == "l_last":
        path = binary_dir / f"l_last-{layer}-0.bin"
        return read(path, 20 * 4 * 4096).reshape(20, 4 * 4096)[position]
    path = binary_dir / f"{tensor}-{layer}-0.bin"
    return read(path, 20 * 4096).reshape(20, 4096)[position]


def ds4_row(capture: Path, tensor: str, layer: int, position: int) -> np.ndarray | None:
    peer = capture / "rank1"
    if tensor == "attn_low":
        first = read(capture / f"ds4_attn_low-{layer}_pos{position}.bin", 8192)
        second = read(peer / f"ds4_attn_low-{layer}_pos{position}.bin", 8192)
        if np.count_nonzero(first[4096:]) or np.count_nonzero(second[4096:]):
            raise ValueError("DS4 attention-low tensor is not compact at offset zero")
        return np.concatenate((first[:4096], second[:4096]))
    if tensor == "attn_out":
        first = read(capture / f"ds4_tp_attn_partial-{layer}_pos{position}.bin", 4096)
        second = read(peer / f"ds4_tp_attn_partial-{layer}_pos{position}.bin", 4096)
        return first + second
    names = {
        "attn_norm": "attn_norm",
        "ffn_norm": "ffn_norm",
        "ffn_moe_out": "ffn_moe_out",
        "ffn_out": "ffn_out",
        "l_last": "hc_ffn_post",
    }
    path = capture / f"ds4_{names[tensor]}-{layer}_pos{position}.bin"
    if not path.exists():
        return None
    return read(path, 4 * 4096 if tensor == "l_last" else 4096)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-binary-dir", type=Path, required=True)
    ap.add_argument("--ds4-capture", type=Path, required=True)
    ap.add_argument("--layers", default="0,1")
    ap.add_argument("--positions", default="17,18")
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()

    layers = [int(item) for item in args.layers.split(",") if item]
    positions = [int(item) for item in args.positions.split(",") if item]
    tensors = ("attn_norm", "attn_low", "attn_out", "ffn_norm",
               "ffn_moe_out", "ffn_out", "l_last")
    output: list[dict[str, object]] = []
    for position in positions:
        for layer in layers:
            for tensor in tensors:
                actual = ds4_row(args.ds4_capture, tensor, layer, position)
                if actual is None:
                    continue
                reference = llama_row(args.llama_binary_dir, tensor, layer, position)
                values: dict[str, object] = {
                    "position": position,
                    "layer": layer,
                    "tensor": tensor,
                    "elements": reference.size,
                }
                values.update(metrics(reference, actual))
                output.append(values)
                print(
                    f"pos={position} layer={layer} {tensor} n={reference.size} "
                    f"rel_rms={values['relative_rms']:.9g} "
                    f"max_abs={values['max_abs']:.9g} cosine={values['cosine']:.12g}"
                )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
