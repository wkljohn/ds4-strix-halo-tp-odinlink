#!/usr/bin/env python3
"""Compare llama-debug samples with non-perturbing DS4 TP boundary dumps.

The ordinary TP decode path folds the two attention-output rank partials
directly into the HC expansion and deliberately leaves ``attn_out``
unmaterialized.  This tool reconstructs that canonical attention vector by
summing the rank-0 and rank-1 ``tp_attn_partial`` files offline.  It also
checks that the downstream DS4 tensors are identical on both ranks.
"""

from __future__ import annotations

import argparse
import math
import re
import struct
from pathlib import Path


HEADER_RE = re.compile(
    r"common_debug_cb_eval:\s+(attn_out|ffn_norm|l_last)-(\d+)"
    r".*=\s*\{(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\}\s*$"
)
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?")


def rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def read_f32(path: Path, expected: int) -> list[float]:
    raw = path.read_bytes()
    if len(raw) != expected * 4:
        raise ValueError(f"{path}: expected {expected * 4} bytes, found {len(raw)}")
    return list(struct.unpack(f"<{expected}f", raw))


def parse_log(
    path: Path,
) -> dict[tuple[str, int], list[tuple[int, int, list[int], list[list[float]]]]]:
    lines = path.read_text(errors="replace").splitlines()
    result: dict[
        tuple[str, int], list[tuple[int, int, list[int], list[list[float]]]]
    ] = {}
    i = 0
    while i < len(lines):
        match = HEADER_RE.search(lines[i])
        if not match:
            i += 1
            continue
        name = match.group(1)
        layer = int(match.group(2))
        embd = int(match.group(3))
        second = int(match.group(4))
        third = int(match.group(5))
        n_hc = second if name == "l_last" else 1
        n_tokens = third if name == "l_last" else second
        visible = list(range(n_tokens)) if n_tokens <= 6 else [0, 1, 2, n_tokens - 3, n_tokens - 2, n_tokens - 1]
        rows: list[list[float]] = []
        i += 1
        while i < len(lines) and not lines[i].lstrip().startswith("sum ="):
            line = lines[i]
            if "...," in line and "[" in line and "]" in line:
                values = [float(value) for value in FLOAT_RE.findall(line)]
                if len(values) == 6:
                    rows.append(values)
            i += 1
        expected = len(visible) * n_hc
        if len(rows) != expected:
            raise ValueError(
                f"{name}-{layer}: expected {expected} sampled rows, found {len(rows)}"
            )
        result.setdefault((name, layer), []).append((n_tokens, n_hc, visible, rows))
        i += 1
    return result


def sample(
    evaluation: tuple[int, int, list[int], list[list[float]]], position: int
) -> list[float] | None:
    _, n_hc, visible, rows = evaluation
    if position not in visible:
        return None
    start = visible.index(position) * n_hc
    return [value for row in rows[start : start + n_hc] for value in row]


def llama_reference(
    evaluations: list[tuple[int, int, list[int], list[list[float]]]], position: int
) -> list[float]:
    candidates = [
        (evaluation[0], values)
        for evaluation in evaluations
        if (values := sample(evaluation, position)) is not None
    ]
    if not candidates:
        raise ValueError(f"llama log does not expose position {position}")
    return max(candidates, key=lambda item: item[0])[1]


def llama_self_drift(
    evaluations: list[tuple[int, int, list[int], list[list[float]]]], position: int
) -> float:
    candidates = [
        (evaluation[0], values)
        for evaluation in evaluations
        if (values := sample(evaluation, position)) is not None
    ]
    candidates.sort(key=lambda item: item[0])
    if len(candidates) < 2:
        return float("nan")
    return rms([a - b for a, b in zip(candidates[0][1], candidates[-1][1])])


def select_coords(values: list[float], embd: int, n_hc: int) -> list[float]:
    cols = (0, 1, 2, embd - 3, embd - 2, embd - 1)
    return [values[row * embd + col] for row in range(n_hc) for col in cols]


def ds4_values(
    root: Path, rank1: Path, tensor: str, layer: int, position: int
) -> tuple[list[float], list[float]]:
    if tensor == "attn_out":
        name = f"ds4_tp_attn_partial-{layer}_pos{position}.bin"
        first = read_f32(root / name, 4096)
        second = read_f32(rank1 / name, 4096)
        # Rank 0 then rank 1 is the canonical reduction order used by DS4.
        combined = [a + b for a, b in zip(first, second)]
        return select_coords(combined, 4096, 1), [0.0] * 6
    if tensor == "ffn_norm":
        name = f"ds4_ffn_norm-{layer}_pos{position}.bin"
        first = read_f32(root / name, 4096)
        second = read_f32(rank1 / name, 4096)
        return select_coords(first, 4096, 1), select_coords(second, 4096, 1)
    name = f"ds4_hc_ffn_post-{layer}_pos{position}.bin"
    first = read_f32(root / name, 4 * 4096)
    second = read_f32(rank1 / name, 4 * 4096)
    return select_coords(first, 4096, 4), select_coords(second, 4096, 4)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-log", required=True, type=Path)
    parser.add_argument("--ds4-root", required=True, type=Path)
    parser.add_argument("--rank1-root", required=True, type=Path)
    parser.add_argument("--positions", default="17,18")
    parser.add_argument("--layers", default="0,1")
    parser.add_argument("--self-drift-position", type=int, default=0)
    args = parser.parse_args()

    parsed = parse_log(args.llama_log)
    positions = [int(value) for value in args.positions.split(",")]
    layers = [int(value) for value in args.layers.split(",")]
    print(
        "position,layer,tensor,samples,max_abs,rms,mean_abs,ref_rms,rel_rms,"
        "llama_self_rms,diff_over_self,rank_parity_max,rank_parity_rms"
    )
    for position in positions:
        for layer in layers:
            for tensor in ("attn_out", "ffn_norm", "l_last"):
                evaluations = parsed.get((tensor, layer))
                if not evaluations:
                    continue
                reference = llama_reference(evaluations, position)
                ds4, peer = ds4_values(args.ds4_root, args.rank1_root, tensor, layer, position)
                if len(ds4) != len(reference):
                    raise ValueError(
                        f"{tensor}-{layer} pos{position}: DS4 has {len(ds4)} samples, "
                        f"llama has {len(reference)}"
                    )
                diffs = [abs(a - b) for a, b in zip(ds4, reference)]
                diff_rms = rms(diffs)
                ref_rms = rms(reference)
                self_rms = llama_self_drift(evaluations, args.self_drift_position)
                parity = ([abs(a - b) for a, b in zip(ds4, peer)]
                          if tensor != "attn_out" else [0.0])
                print(
                    f"{position},{layer},{tensor},{len(diffs)},{max(diffs):.9g},"
                    f"{diff_rms:.9g},{sum(diffs) / len(diffs):.9g},{ref_rms:.9g},"
                    f"{diff_rms / ref_rms if ref_rms else float('inf'):.9g},"
                    f"{self_rms:.9g},"
                    f"{diff_rms / self_rms if self_rms else float('inf'):.9g},"
                    f"{max(parity):.9g},{rms(parity):.9g}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
