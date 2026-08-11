#!/usr/bin/env python3
"""Check TP attention-output K slices against saved real activations.

The decode path stores each rank's 4096-element attn_output_a result compactly
at offset zero, then applies one half of the 8192-wide Q8_0 attn_output_b
matrix.  This offline oracle reads those real rank dumps and emulates the
gfx1151 pack-4 kernel's float32 accumulation and wave reduction.  It therefore
tests the compact activation contract without launching either GPU or changing
the inference graph.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ErrorStats:
    rms_ref: float
    rms_error: float
    relative_rms: float
    max_abs: float
    cosine: float


def load_gguf_parser(repo: Path):
    parser_path = repo / "gguf-tools/mixed/splice_mixed_expert_layers_gguf.py"
    spec = importlib.util.spec_from_file_location("ds4_gguf_parser", parser_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load GGUF parser from {parser_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_f32(path: Path, expected: int) -> np.ndarray:
    data = np.fromfile(path, dtype="<f4")
    if data.size != expected:
        raise ValueError(f"{path}: expected {expected} float32 values, got {data.size}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path}: contains non-finite values")
    return data


def error_stats(reference: np.ndarray, actual: np.ndarray) -> ErrorStats:
    ref64 = reference.astype(np.float64)
    actual64 = actual.astype(np.float64)
    error = actual64 - ref64
    rms_ref = math.sqrt(float(np.mean(ref64 * ref64)))
    rms_error = math.sqrt(float(np.mean(error * error)))
    denom = math.sqrt(float(np.dot(ref64, ref64) * np.dot(actual64, actual64)))
    cosine = float(np.dot(ref64, actual64) / denom) if denom else 1.0
    return ErrorStats(
        rms_ref=rms_ref,
        rms_error=rms_error,
        relative_rms=rms_error / rms_ref if rms_ref else 0.0,
        max_abs=float(np.max(np.abs(error))),
        cosine=cosine,
    )


def emulate_pack4_slice(
    raw: np.memmap,
    x: np.ndarray,
    block_start: int,
    row_chunk: int,
) -> np.ndarray:
    """Emulate matmul_q8_0_f32_sharedx_warp_rows_w32_pack4_kernel.

    Each wave has four eight-lane groups.  A group consumes one Q8 block per
    b4 iteration, each lane consumes four adjacent values, and the final wave
    reduction uses offsets 16, 8, 4, 2, 1.  All arithmetic here is explicitly
    float32 to preserve the production accumulation tree closely.
    """

    if x.shape != (4096,):
        raise ValueError(f"expected a compact 4096-element activation, got {x.shape}")
    x_blocks = x.astype(np.float32, copy=False).reshape(128, 8, 4)
    out = np.empty(raw.shape[0], dtype=np.float32)

    for row0 in range(0, raw.shape[0], row_chunk):
        row1 = min(row0 + row_chunk, raw.shape[0])
        block_bytes = np.asarray(raw[row0:row1, block_start:block_start + 128, :])
        scales = block_bytes[:, :, :2].copy().view("<f2").reshape(row1 - row0, 128).astype(np.float32)
        quants = block_bytes[:, :, 2:].view(np.int8).reshape(row1 - row0, 128, 8, 4)
        lane_acc = np.zeros((row1 - row0, 32), dtype=np.float32)

        for b4 in range(0, 128, 4):
            for subgroup in range(4):
                block = b4 + subgroup
                q = quants[:, block, :, :].astype(np.float32)
                xv = x_blocks[block]
                term = q[:, :, 0] * xv[:, 0]
                term = term + q[:, :, 1] * xv[:, 1]
                term = term + q[:, :, 2] * xv[:, 2]
                term = term + q[:, :, 3] * xv[:, 3]
                term = term * scales[:, block, None]
                lanes = slice(subgroup * 8, (subgroup + 1) * 8)
                lane_acc[:, lanes] = lane_acc[:, lanes] + term

        for offset in (16, 8, 4, 2, 1):
            lane_acc[:, :offset] = lane_acc[:, :offset] + lane_acc[:, offset:2 * offset]
        out[row0:row1] = lane_acc[:, 0]

    return out


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--capture", type=Path, required=True,
                    help="capture directory containing rank-0 files and rank1/")
    ap.add_argument("--layers", default="0,1")
    ap.add_argument("--positions", default="17,18")
    ap.add_argument("--row-chunk", type=int, default=128)
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()

    layers = [int(value) for value in args.layers.split(",") if value]
    positions = [int(value) for value in args.positions.split(",") if value]
    if not layers or not positions or args.row_chunk <= 0:
        ap.error("layers, positions, and row-chunk must be non-empty/positive")

    gguf = load_gguf_parser(repo)
    info = gguf.parse_gguf(args.model)
    rows: list[dict[str, object]] = []

    for layer in layers:
        name = f"blk.{layer}.attn_output_b.weight"
        tensor = info.tensor_by_name.get(name)
        if tensor is None:
            raise ValueError(f"model has no tensor {name}")
        if tensor.ggml_type != 8 or tensor.dims != (8192, 4096):
            raise ValueError(
                f"{name}: expected Q8_0 (8192,4096), got type={tensor.ggml_type} dims={tensor.dims}"
            )
        raw = np.memmap(
            args.model,
            mode="r",
            dtype=np.uint8,
            offset=tensor.data_offset,
            shape=(4096, 256, 34),
        )

        for position in positions:
            rank0_low_path = args.capture / f"ds4_attn_low-{layer}_pos{position}.bin"
            rank1_low_path = args.capture / "rank1" / f"ds4_attn_low-{layer}_pos{position}.bin"
            rank0_actual_path = args.capture / f"ds4_tp_attn_partial-{layer}_pos{position}.bin"
            rank1_actual_path = args.capture / "rank1" / f"ds4_tp_attn_partial-{layer}_pos{position}.bin"

            rank0_storage = read_f32(rank0_low_path, 8192)
            rank1_storage = read_f32(rank1_low_path, 8192)
            if np.count_nonzero(rank0_storage[4096:]) or np.count_nonzero(rank1_storage[4096:]):
                raise ValueError(
                    f"layer {layer} position {position}: compact activation upper half is not zero"
                )
            rank0_low = rank0_storage[:4096]
            rank1_low = rank1_storage[:4096]
            actual0 = read_f32(rank0_actual_path, 4096)
            actual1 = read_f32(rank1_actual_path, 4096)

            reference0 = emulate_pack4_slice(raw, rank0_low, 0, args.row_chunk)
            reference1 = emulate_pack4_slice(raw, rank1_low, 128, args.row_chunk)
            comparisons = {
                "rank0_partial": (reference0, actual0),
                "rank1_partial": (reference1, actual1),
                "summed_partial": (reference0 + reference1, actual0 + actual1),
            }
            for comparison, (reference, actual) in comparisons.items():
                stats = error_stats(reference, actual)
                row = {
                    "layer": layer,
                    "position": position,
                    "comparison": comparison,
                    "rms_reference": stats.rms_ref,
                    "rms_error": stats.rms_error,
                    "relative_rms": stats.relative_rms,
                    "max_abs": stats.max_abs,
                    "cosine": stats.cosine,
                }
                rows.append(row)
                print(
                    f"layer={layer} pos={position} {comparison} "
                    f"rel_rms={stats.relative_rms:.9g} max_abs={stats.max_abs:.9g} "
                    f"cosine={stats.cosine:.12g}"
                )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    worst = max(float(row["relative_rms"]) for row in rows)
    print(f"worst_relative_rms={worst:.9g}")
    return 0 if worst < 1.0e-5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
