#!/usr/bin/env python3
"""Compare DS4 HC layer dumps with llama-debug's sampled l_last rows.

llama-debug prints the first and last three embedding coordinates for the
first and last three token rows.  A 20-token reference therefore includes
positions 17, 18, and 19.  DS4's graph diagnostic stores the complete HC state
for one position; this script selects the identical 24 coordinates (four HC
rows times six embedding coordinates) and reports per-layer differences.

A debug log may contain multiple independent evaluations.  The comparison
uses the largest evaluation that exposes ``--position``.  The two smallest
evaluations that expose ``--self-drift-position`` provide llama.cpp's own
batch-shape numerical-drift envelope; this is only meaningful when both
evaluations started with a fresh cache, as the reference command used here did.
"""

from __future__ import annotations

import argparse
import math
import re
import struct
from pathlib import Path


HEADER_RE = re.compile(
    r"common_debug_cb_eval:\s+l_last-(\d+).*?=\s*\{(\d+),\s*(\d+),\s*(\d+),"
)
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_llama_evaluations(
    path: Path,
) -> tuple[int, int, dict[int, list[tuple[int, list[int], list[list[float]]]]]]:
    lines = path.read_text(errors="replace").splitlines()
    evaluations: dict[int, list[tuple[int, list[int], list[list[float]]]]] = {}
    embd = 0
    n_hc = 0

    i = 0
    while i < len(lines):
        match = HEADER_RE.search(lines[i])
        if not match:
            i += 1
            continue

        layer = int(match.group(1))
        this_embd = int(match.group(2))
        this_n_hc = int(match.group(3))
        n_tokens = int(match.group(4))
        visible_positions = list(range(min(3, n_tokens)))
        if n_tokens > 6:
            visible_positions.extend(range(n_tokens - 3, n_tokens))
        else:
            visible_positions = list(range(n_tokens))

        vector_rows: list[list[float]] = []
        i += 1
        while i < len(lines) and not lines[i].lstrip().startswith("sum ="):
            line = lines[i]
            if "...," in line and "[" in line and "]" in line:
                values = [float(x) for x in FLOAT_RE.findall(line)]
                if len(values) == 6:
                    vector_rows.append(values)
            i += 1

        expected_rows = len(visible_positions) * this_n_hc
        if len(vector_rows) != expected_rows:
            raise ValueError(
                f"layer {layer}: expected {expected_rows} sampled rows, "
                f"found {len(vector_rows)}"
            )
        evaluations.setdefault(layer, []).append(
            (n_tokens, visible_positions, vector_rows)
        )
        if embd and (embd != this_embd or n_hc != this_n_hc):
            raise ValueError("inconsistent l_last shapes across layers")
        embd, n_hc = this_embd, this_n_hc
        i += 1

    if not evaluations:
        raise ValueError(f"no l_last tensors found in {path}")
    return embd, n_hc, evaluations


def sampled_rows(
    evaluation: tuple[int, list[int], list[list[float]]], position: int, n_hc: int
) -> list[list[float]] | None:
    _, visible_positions, vector_rows = evaluation
    if position not in visible_positions:
        return None
    start = visible_positions.index(position) * n_hc
    return vector_rows[start : start + n_hc]


def flatten(rows: list[list[float]]) -> list[float]:
    return [value for row in rows for value in row]


def rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def correlation(a: list[float], b: list[float]) -> float:
    am = sum(a) / len(a)
    bm = sum(b) / len(b)
    ac = [value - am for value in a]
    bc = [value - bm for value in b]
    denom = math.sqrt(sum(value * value for value in ac) *
                      sum(value * value for value in bc))
    return sum(x * y for x, y in zip(ac, bc)) / denom if denom else float("nan")


def read_ds4_samples(path: Path, embd: int, n_hc: int) -> list[float]:
    raw = path.read_bytes()
    expected = embd * n_hc * 4
    if len(raw) != expected:
        raise ValueError(f"{path}: expected {expected} bytes, found {len(raw)}")
    values = struct.unpack(f"<{embd * n_hc}f", raw)
    cols = (0, 1, 2, embd - 3, embd - 2, embd - 1)
    return [values[hc * embd + col] for hc in range(n_hc) for col in cols]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-log", required=True, type=Path)
    parser.add_argument("--ds4-prefix", required=True, type=Path)
    parser.add_argument("--position", type=int, default=18)
    parser.add_argument("--self-drift-position", type=int, default=0)
    parser.add_argument("--correlation-output", type=Path)
    args = parser.parse_args()

    embd, n_hc, evaluations = parse_llama_evaluations(args.llama_log)
    print("layer,samples,max_abs,rms,mean_abs,ref_rms,rel_rms,"
          "llama_self_rms,diff_over_self")
    compared = 0
    correlation_rows: list[tuple[int, int, int, float]] = []
    for layer in sorted(evaluations):
        candidates = [
            (evaluation[0], rows)
            for evaluation in evaluations[layer]
            if (rows := sampled_rows(evaluation, args.position, n_hc)) is not None
        ]
        if not candidates:
            continue
        # Prefer the requested full reference rather than an incidental
        # smaller warm-up graph that happens to expose the same position.
        _, llama_rows = max(candidates, key=lambda item: item[0])
        llama = flatten(llama_rows)

        drift_candidates = [
            (evaluation[0], flatten(rows))
            for evaluation in evaluations[layer]
            if (rows := sampled_rows(
                evaluation, args.self_drift_position, n_hc
            )) is not None
        ]
        drift_candidates.sort(key=lambda item: item[0])
        self_rms = float("nan")
        if len(drift_candidates) >= 2:
            first = drift_candidates[0][1]
            second = drift_candidates[-1][1]
            self_rms = rms([a - b for a, b in zip(first, second)])

        dump = Path(f"{args.ds4_prefix}_hc_ffn_post-{layer}_pos{args.position}.bin")
        if not dump.exists():
            continue
        ds4 = read_ds4_samples(dump, embd, n_hc)
        diffs = [abs(a - b) for a, b in zip(ds4, llama)]
        diff_rms = rms(diffs)
        mean = sum(diffs) / len(diffs)
        ref_rms = rms(llama)
        rel_rms = diff_rms / ref_rms if ref_rms else float("inf")
        diff_over_self = diff_rms / self_rms if self_rms else float("inf")
        print(f"{layer},{len(diffs)},{max(diffs):.9g},{diff_rms:.9g},"
              f"{mean:.9g},{ref_rms:.9g},{rel_rms:.9g},{self_rms:.9g},"
              f"{diff_over_self:.9g}")
        if args.correlation_output:
            for ds4_row in range(n_hc):
                for llama_row in range(n_hc):
                    correlation_rows.append(
                        (layer, ds4_row, llama_row,
                         correlation(ds4[ds4_row * 6 : (ds4_row + 1) * 6],
                                     llama_rows[llama_row]))
                    )
        compared += 1

    if compared == 0:
        raise SystemExit("no matching DS4 layer dumps found")
    if args.correlation_output:
        with args.correlation_output.open("w") as output:
            output.write("layer,ds4_hc,llama_hc,correlation\n")
            for layer, ds4_row, llama_row, value in correlation_rows:
                output.write(f"{layer},{ds4_row},{llama_row},{value:.9g}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
