#!/usr/bin/env python3
"""Compare tokenwise llama.cpp and TP=2 DS4 attention boundaries.

llama.cpp names the tensor immediately before the grouped output-A projection
``attn_derope``.  DS4 names the same post-inverse-RoPE head buffer
``kqv_back``.  Each DS4 TP rank writes its 32 owned heads compactly at offset
zero, so the canonical 64-head value is rank 0's compact half followed by rank
1's compact half.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


N_HEAD = 64
HEAD_DIM = 512
N_GROUP = 8
OUT_RANK = 1024
N_EMBD = 4096


def read_f32(path: Path, expected: int) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    if values.size != expected:
        raise ValueError(f"{path}: expected {expected} floats, got {values.size}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: contains non-finite values")
    return values


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    ref = reference.astype(np.float64).ravel()
    got = actual.astype(np.float64).ravel()
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


def llama_tokenwise(
    dump: Path, tensor: str, layer: int, position: int, prefix_tokens: int
) -> np.ndarray:
    if position < prefix_tokens:
        raise ValueError(
            f"position {position} is inside the batched prefix; this comparator "
            "accepts only tokenwise tail positions"
        )
    invocation = position - prefix_tokens + 1
    elements = {
        "attn_derope": N_HEAD * HEAD_DIM,
        "attn_wo_a": N_GROUP * OUT_RANK,
        "attn_out": N_EMBD,
    }[tensor]
    return read_f32(dump / f"{tensor}-{layer}-{invocation}.bin", elements)


def ds4_tp_value(capture: Path, tensor: str, layer: int, position: int) -> np.ndarray:
    peer = capture / "rank1"
    if tensor == "attn_derope":
        rank_elements = N_HEAD * HEAD_DIM
        owned = rank_elements // 2
        rank0 = read_f32(capture / f"ds4_kqv_back-{layer}_pos{position}.bin", rank_elements)
        rank1 = read_f32(peer / f"ds4_kqv_back-{layer}_pos{position}.bin", rank_elements)
        return np.concatenate((rank0[:owned], rank1[:owned]))
    if tensor == "attn_wo_a":
        rank_elements = N_GROUP * OUT_RANK
        owned = rank_elements // 2
        rank0 = read_f32(capture / f"ds4_attn_low-{layer}_pos{position}.bin", rank_elements)
        rank1 = read_f32(peer / f"ds4_attn_low-{layer}_pos{position}.bin", rank_elements)
        if np.count_nonzero(rank0[owned:]) or np.count_nonzero(rank1[owned:]):
            raise ValueError("DS4 attn_low is not compact/zero outside its owned half")
        return np.concatenate((rank0[:owned], rank1[:owned]))
    if tensor == "attn_out":
        rank0 = read_f32(capture / f"ds4_tp_attn_partial-{layer}_pos{position}.bin", N_EMBD)
        rank1 = read_f32(peer / f"ds4_tp_attn_partial-{layer}_pos{position}.bin", N_EMBD)
        return rank0 + rank1
    raise ValueError(f"unsupported tensor: {tensor}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-binary-dir", type=Path, required=True)
    parser.add_argument("--ds4-capture", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--positions", default="17,18")
    parser.add_argument("--llama-prefix-tokens", type=int, default=17)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for position in (int(item) for item in args.positions.split(",") if item):
        for tensor in ("attn_derope", "attn_wo_a", "attn_out"):
            reference = llama_tokenwise(
                args.llama_binary_dir,
                tensor,
                args.layer,
                position,
                args.llama_prefix_tokens,
            )
            actual = ds4_tp_value(args.ds4_capture, tensor, args.layer, position)
            row: dict[str, object] = {
                "position": position,
                "layer": args.layer,
                "tensor": tensor,
                "elements": reference.size,
                "layout": "rank0-owned || rank1-owned" if tensor != "attn_out" else "rank0 + rank1",
            }
            row.update(metrics(reference, actual))
            rows.append(row)
            print(
                f"pos={position} layer={args.layer} {tensor} n={reference.size} "
                f"rel_rms={row['relative_rms']:.9g} max_abs={row['max_abs']:.9g} "
                f"cosine={row['cosine']:.12g}"
            )

            # A deliberate rank-order swap is a layout negative control for the
            # pre-A tensor.  It must be substantially worse than canonical order.
            if tensor == "attn_derope":
                reference_heads = reference.reshape(N_HEAD, HEAD_DIM)
                actual_heads = actual.reshape(N_HEAD, HEAD_DIM)
                for section, begin, end in (("nope", 0, 448), ("pe", 448, HEAD_DIM)):
                    section_row: dict[str, object] = {
                        "position": position,
                        "layer": args.layer,
                        "tensor": f"attn_derope_{section}",
                        "elements": N_HEAD * (end - begin),
                        "layout": f"per-head [{begin}:{end}]",
                    }
                    section_row.update(
                        metrics(reference_heads[:, begin:end], actual_heads[:, begin:end])
                    )
                    rows.append(section_row)
                    print(
                        f"  {section} rel_rms={section_row['relative_rms']:.9g} "
                        f"max_abs={section_row['max_abs']:.9g} "
                        f"cosine={section_row['cosine']:.12g}"
                    )

                owned = reference.size // 2
                swapped = np.concatenate((actual[owned:], actual[:owned]))
                negative = metrics(reference, swapped)
                if negative["relative_rms"] <= row["relative_rms"]:
                    raise ValueError(
                        "swapped-rank negative control did not worsen pre-A agreement"
                    )
                print(
                    f"  swapped-rank negative rel_rms={negative['relative_rms']:.9g} "
                    f"cosine={negative['cosine']:.12g}"
                )

    if not rows:
        raise ValueError("no comparison rows produced")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
