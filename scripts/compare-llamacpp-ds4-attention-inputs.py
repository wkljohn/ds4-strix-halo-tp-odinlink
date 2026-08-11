#!/usr/bin/env python3
"""Localize llama.cpp vs DS4 differences before DeepSeek4 attention output."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


N_HEAD = 64
HEAD_DIM = 512
OWNED_HEAD = N_HEAD // 2


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


def llama_tail(
    dump: Path, name: str, layer: int, position: int, prefix_tokens: int, elements: int
) -> np.ndarray:
    if position < prefix_tokens:
        raise ValueError(f"position {position} is not in the tokenwise tail")
    invocation = position - prefix_tokens + 1
    return read_f32(dump / f"{name}-{layer}-{invocation}.bin", elements)


def ds4_file(capture: Path, rank: int, name: str, layer: int, position: int, elements: int) -> np.ndarray:
    root = capture if rank == 0 else capture / "rank1"
    return read_f32(root / f"ds4_{name}-{layer}_pos{position}.bin", elements)


def compact_heads(capture: Path, name: str, layer: int, position: int) -> np.ndarray:
    full = N_HEAD * HEAD_DIM
    owned = OWNED_HEAD * HEAD_DIM
    rank0 = ds4_file(capture, 0, name, layer, position, full)
    rank1 = ds4_file(capture, 1, name, layer, position, full)
    return np.concatenate((rank0[:owned], rank1[:owned]))


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
    duplicated = (
        ("qr", "q_lora", 1024),
        ("qr_norm", "q_lora_norm", 1024),
        ("kv_norm", "KVnorm", HEAD_DIM),
        ("kv", "KVcur", HEAD_DIM),
    )
    compact = (
        ("q_norm", "Qnorm"),
        ("q", "Qcur"),
    )

    for position in (int(item) for item in args.positions.split(",") if item):
        for llama_name, ds4_name, elements in duplicated:
            reference = llama_tail(
                args.llama_binary_dir,
                llama_name,
                args.layer,
                position,
                args.llama_prefix_tokens,
                elements,
            )
            rank_values = [
                ds4_file(args.ds4_capture, rank, ds4_name, args.layer, position, elements)
                for rank in (0, 1)
            ]
            for rank, actual in enumerate(rank_values):
                row: dict[str, object] = {
                    "position": position,
                    "layer": args.layer,
                    "tensor": llama_name,
                    "ds4_tensor": ds4_name,
                    "rank": rank,
                    "elements": elements,
                }
                row.update(metrics(reference, actual))
                rows.append(row)
                print(
                    f"pos={position} {llama_name}/{ds4_name} rank={rank} "
                    f"rel_rms={row['relative_rms']:.9g} max_abs={row['max_abs']:.9g} "
                    f"cosine={row['cosine']:.12g}"
                )
            agreement = metrics(rank_values[0], rank_values[1])
            if agreement["max_abs"] != 0.0:
                print(
                    f"  rank disagreement rel_rms={agreement['relative_rms']:.9g} "
                    f"max_abs={agreement['max_abs']:.9g}"
                )

        for llama_name, ds4_name in compact:
            elements = N_HEAD * HEAD_DIM
            reference = llama_tail(
                args.llama_binary_dir,
                llama_name,
                args.layer,
                position,
                args.llama_prefix_tokens,
                elements,
            )
            actual = compact_heads(args.ds4_capture, ds4_name, args.layer, position)
            row = {
                "position": position,
                "layer": args.layer,
                "tensor": llama_name,
                "ds4_tensor": ds4_name,
                "rank": "assembled",
                "elements": elements,
            }
            row.update(metrics(reference, actual))
            rows.append(row)
            print(
                f"pos={position} {llama_name}/{ds4_name} assembled "
                f"rel_rms={row['relative_rms']:.9g} max_abs={row['max_abs']:.9g} "
                f"cosine={row['cosine']:.12g}"
            )

        # Inverse RoPE does not touch the first 448 values of each head, so the
        # DS4 kqv_back nope section is directly comparable with llama attn_raw.
        reference = llama_tail(
            args.llama_binary_dir,
            "attn_raw",
            args.layer,
            position,
            args.llama_prefix_tokens,
            N_HEAD * HEAD_DIM,
        ).reshape(N_HEAD, HEAD_DIM)[:, :448]
        actual = compact_heads(
            args.ds4_capture, "kqv_back", args.layer, position
        ).reshape(N_HEAD, HEAD_DIM)[:, :448]
        row = {
            "position": position,
            "layer": args.layer,
            "tensor": "attn_raw_nope",
            "ds4_tensor": "kqv_back_nope",
            "rank": "assembled",
            "elements": N_HEAD * 448,
        }
        row.update(metrics(reference, actual))
        rows.append(row)
        print(
            f"pos={position} attn_raw/kqv_back nope "
            f"rel_rms={row['relative_rms']:.9g} max_abs={row['max_abs']:.9g} "
            f"cosine={row['cosine']:.12g}"
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
