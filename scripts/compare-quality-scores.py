#!/usr/bin/env python3
"""Paired, fail-closed quality comparison for DS4 score_official TSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise ValueError(f"{path}: empty score table")
    required = {
        "id", "target_tokens", "nll", "avg_nll", "api_top1_count",
        "api_top1_match", "api_pair_total", "api_pair_agree",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path}: missing fields: {', '.join(sorted(missing))}")
    return rows


def manifest_for(path: Path) -> Path:
    return path.with_suffix(".manifest")


def number(row: dict[str, str], key: str, integer: bool = False) -> float | int:
    try:
        value = int(row[key]) if integer else float(row[key])
    except (KeyError, ValueError) as error:
        raise ValueError(f"case {row.get('id', '?')}: invalid {key}") from error
    if not integer and not math.isfinite(value):
        raise ValueError(f"case {row.get('id', '?')}: non-finite {key}")
    return value


def load_thresholds(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "baseline_id", "min_cases", "min_target_tokens",
        "max_mean_nll_delta", "max_ci95_high_nll_delta",
        "min_api_top1_rate_delta", "min_api_pair_rate_delta",
    }
    missing = required - value.keys()
    if missing:
        raise ValueError(f"{path}: missing thresholds: {', '.join(sorted(missing))}")
    if not isinstance(value["baseline_id"], str) or not value["baseline_id"]:
        raise ValueError(f"{path}: invalid baseline_id")
    for key in ("min_cases", "min_target_tokens"):
        if not isinstance(value[key], int) or value[key] <= 0:
            raise ValueError(f"{path}: {key} must be a positive integer")
    for key in required - {"baseline_id", "min_cases", "min_target_tokens"}:
        if not isinstance(value[key], (int, float)) or not math.isfinite(value[key]):
            raise ValueError(f"{path}: {key} must be finite")
    if "max_case_nll_delta" in value and (
            not isinstance(value["max_case_nll_delta"], (int, float)) or
            not math.isfinite(value["max_case_nll_delta"])):
        raise ValueError(f"{path}: max_case_nll_delta must be finite")
    return value


def ratio(rows: list[dict[str, str]], numerator: str,
          denominator: str) -> float | None:
    top = sum(number(row, numerator, True) for row in rows)
    bottom = sum(number(row, denominator, True) for row in rows)
    if bottom <= 0:
        return None
    return top / bottom


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        thresholds = load_thresholds(args.thresholds)
        reference = load_rows(args.reference)
        candidate = load_rows(args.candidate)
        if [row["id"] for row in reference] != [row["id"] for row in candidate]:
            raise ValueError("reference and candidate case ids/order differ")
        deltas: list[float] = []
        ref_nll = cand_nll = 0.0
        target_tokens = 0
        for ref_row, cand_row in zip(reference, candidate):
            ref_tokens = number(ref_row, "target_tokens", True)
            cand_tokens = number(cand_row, "target_tokens", True)
            if ref_tokens <= 0 or cand_tokens != ref_tokens:
                raise ValueError(f"case {ref_row['id']}: target-token counts differ")
            ref_value = number(ref_row, "nll")
            cand_value = number(cand_row, "nll")
            ref_avg = number(ref_row, "avg_nll")
            cand_avg = number(cand_row, "avg_nll")
            if not math.isclose(ref_value / ref_tokens, ref_avg, rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError(f"case {ref_row['id']}: reference avg_nll is inconsistent")
            if not math.isclose(cand_value / cand_tokens, cand_avg, rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError(f"case {ref_row['id']}: candidate avg_nll is inconsistent")
            ref_nll += ref_value
            cand_nll += cand_value
            target_tokens += ref_tokens
            deltas.append(cand_avg - ref_avg)
        if len(deltas) < 2:
            raise ValueError("at least two paired cases are required for an uncertainty bound")
        mean_delta = statistics.fmean(deltas)
        ci_half = 1.96 * statistics.stdev(deltas) / math.sqrt(len(deltas))
        weighted_delta = (cand_nll - ref_nll) / target_tokens
        max_case_delta = max(deltas)
        ref_top1 = ratio(reference, "api_top1_match", "api_top1_count")
        cand_top1 = ratio(candidate, "api_top1_match", "api_top1_count")
        ref_pair = ratio(reference, "api_pair_agree", "api_pair_total")
        cand_pair = ratio(candidate, "api_pair_agree", "api_pair_total")
        api_metrics_available = all(
            value is not None
            for value in (ref_top1, cand_top1, ref_pair, cand_pair))
        api_top1_delta = (
            cand_top1 - ref_top1 if cand_top1 is not None and
            ref_top1 is not None else None)
        api_pair_delta = (
            cand_pair - ref_pair if cand_pair is not None and
            ref_pair is not None else None)
        metrics = {
            "cases": len(deltas),
            "target_tokens": target_tokens,
            "reference_avg_nll": ref_nll / target_tokens,
            "candidate_avg_nll": cand_nll / target_tokens,
            "weighted_nll_delta": weighted_delta,
            "paired_mean_avg_nll_delta": mean_delta,
            "paired_ci95_low": mean_delta - ci_half,
            "paired_ci95_high": mean_delta + ci_half,
            "max_case_avg_nll_delta": max_case_delta,
            "api_metrics_available": api_metrics_available,
            "api_top1_rate_delta": api_top1_delta,
            "api_pair_rate_delta": api_pair_delta,
        }
        nll_screen_passed = bool(
            metrics["cases"] >= thresholds["min_cases"] and
            metrics["target_tokens"] >= thresholds["min_target_tokens"] and
            metrics["weighted_nll_delta"] <= thresholds["max_mean_nll_delta"] and
            metrics["paired_ci95_high"] <= thresholds["max_ci95_high_nll_delta"] and
            ("max_case_nll_delta" not in thresholds or
             max_case_delta <= thresholds["max_case_nll_delta"])
        )
        api_screen_passed = bool(
            api_metrics_available and
            api_top1_delta is not None and api_pair_delta is not None and
            api_top1_delta >= thresholds["min_api_top1_rate_delta"] and
            api_pair_delta >= thresholds["min_api_pair_rate_delta"])
        passed = nll_screen_passed and api_screen_passed
        blockers = []
        if not api_metrics_available:
            blockers.append("hosted API top-1/pair logprob metrics unavailable")
        result = {
            "schema_version": 1,
            "baseline_id": thresholds["baseline_id"],
            "sources": {
                "reference": str(args.reference.resolve()),
                "reference_sha256": sha256(args.reference),
                "reference_manifest": str(manifest_for(args.reference).resolve()),
                "reference_manifest_sha256": sha256(manifest_for(args.reference)),
                "candidate": str(args.candidate.resolve()),
                "candidate_sha256": sha256(args.candidate),
                "candidate_manifest": str(manifest_for(args.candidate).resolve()),
                "candidate_manifest_sha256": sha256(manifest_for(args.candidate)),
            },
            "thresholds_sha256": sha256(args.thresholds),
            "metrics": metrics,
            "nll_screen_passed": nll_screen_passed,
            "api_screen_passed": (
                api_screen_passed if api_metrics_available else None),
            "blockers": blockers,
            "passed": passed,
        }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"quality-scores: FAIL {error}", file=sys.stderr)
        return 1
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    if not passed:
        if not metrics["api_metrics_available"]:
            print("quality-scores: FAIL hosted API metrics unavailable; "
                  "paired NLL screen was still reported", file=sys.stderr)
        else:
            print("quality-scores: FAIL paired non-inferiority gate", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
