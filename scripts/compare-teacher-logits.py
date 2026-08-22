#!/usr/bin/env python3
"""Compare frozen-token decode logits without autoregressive cascade.

The tool is diagnostic unless an explicit versioned threshold JSON is supplied.
It never turns its built-in defaults into a lane-B acceptance policy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    logits = value.get("logits")
    vocab = value.get("vocab")
    if not isinstance(vocab, int) or not isinstance(logits, list) or len(logits) != vocab:
        raise ValueError(f"{path}: malformed logits/vocab")
    if any(not isinstance(item, (int, float)) or not math.isfinite(item)
           for item in logits):
        raise ValueError(f"{path}: logits contain null, NaN, Inf, or non-numbers")
    if value.get("source") != "ds4-bench-frozen-teacher":
        raise ValueError(f"{path}: not a frozen-teacher logit dump")
    for field in ("prefix_tokens", "decode_step", "position", "teacher_token",
                  "quant_bits", "argmax_id", "runner_up_id"):
        if not isinstance(value.get(field), int):
            raise ValueError(f"{path}: missing or invalid {field}")
    for field in ("quality", "dspark", "dspark_strict"):
        if type(value.get(field)) is not bool:
            raise ValueError(f"{path}: missing or invalid {field}")
    return value


def top_ids(values: list[float], count: int) -> list[int]:
    return sorted(range(len(values)), key=lambda idx: (-values[idx], idx))[:count]


def probability_metrics(reference: list[float], candidate: list[float]) -> tuple[float, float]:
    ref_max = max(reference)
    cand_max = max(candidate)
    ref_exp = [math.exp(item - ref_max) for item in reference]
    cand_exp = [math.exp(item - cand_max) for item in candidate]
    ref_sum = math.fsum(ref_exp)
    cand_sum = math.fsum(cand_exp)
    ref_log_z = ref_max + math.log(ref_sum)
    cand_log_z = cand_max + math.log(cand_sum)
    tvd = 0.0
    kl = 0.0
    for ref_logit, cand_logit, ref_e, cand_e in zip(
            reference, candidate, ref_exp, cand_exp):
        p = ref_e / ref_sum
        q = cand_e / cand_sum
        tvd += abs(p - q)
        if p:
            kl += p * ((ref_logit - ref_log_z) - (cand_logit - cand_log_z))
    return 0.5 * tvd, max(0.0, kl)


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = math.ceil(fraction * len(sorted_values)) - 1
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def load_thresholds(path: Path | None) -> dict | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    required = {
        "baseline_id", "e_bound", "max_abs", "p99_abs", "nmse", "tvd", "kl",
        "min_top5_overlap", "min_top20_overlap",
    }
    missing = required - value.keys()
    if missing:
        raise ValueError(f"{path}: missing thresholds: {', '.join(sorted(missing))}")
    if not isinstance(value["baseline_id"], str) or not value["baseline_id"]:
        raise ValueError(f"{path}: baseline_id must be a non-empty string")
    for key in ("e_bound", "max_abs", "p99_abs", "nmse", "tvd", "kl"):
        item = value[key]
        if not isinstance(item, (int, float)) or not math.isfinite(item) or item < 0:
            raise ValueError(f"{path}: {key} must be finite and nonnegative")
    for key, limit in (("min_top5_overlap", 5), ("min_top20_overlap", 20)):
        if not isinstance(value[key], int) or not 0 <= value[key] <= limit:
            raise ValueError(f"{path}: {key} must be between 0 and {limit}")
    return value


def compare_pair(reference: dict, candidate: dict, thresholds: dict | None) -> dict:
    for field in ("vocab", "prefix_tokens", "decode_step", "position",
                  "teacher_token", "quant_bits", "quality", "dspark",
                  "dspark_strict"):
        if reference.get(field) != candidate.get(field):
            raise ValueError(f"metadata {field}: {reference.get(field)!r} != {candidate.get(field)!r}")

    ref = reference["logits"]
    cand = candidate["logits"]
    differences = sorted(abs(a - b) for a, b in zip(ref, cand))
    sum_diff_sq = math.fsum((a - b) ** 2 for a, b in zip(ref, cand))
    sum_ref_sq = math.fsum(a * a for a in ref)
    ref_top20 = top_ids(ref, min(20, len(ref)))
    cand_top20 = top_ids(cand, min(20, len(cand)))
    ref_margin = ref[ref_top20[0]] - ref[ref_top20[1]] if len(ref_top20) > 1 else math.inf
    cand_margin = cand[cand_top20[0]] - cand[cand_top20[1]] if len(cand_top20) > 1 else math.inf
    tvd, kl = probability_metrics(ref, cand)
    result = {
        "decode_step": reference["decode_step"],
        "position": reference["position"],
        "teacher_token": reference["teacher_token"],
        "reference_argmax": ref_top20[0],
        "candidate_argmax": cand_top20[0],
        "argmax_equal": ref_top20[0] == cand_top20[0],
        "reference_margin": ref_margin,
        "candidate_margin": cand_margin,
        "max_abs": differences[-1],
        "p99_abs": percentile(differences, 0.99),
        "nmse": sum_diff_sq / sum_ref_sq if sum_ref_sq else (0.0 if not sum_diff_sq else math.inf),
        "tvd": tvd,
        "kl": kl,
        "top5_overlap": len(set(ref_top20[:5]) & set(cand_top20[:5])),
        "top20_overlap": len(set(ref_top20) & set(cand_top20)),
    }
    if thresholds is None:
        result["envelope_pass"] = None
        result["far_margin_inversion"] = None
    else:
        result["far_margin_inversion"] = (
            not result["argmax_equal"] and ref_margin > 2.0 * thresholds["e_bound"])
        result["envelope_pass"] = (
            not result["far_margin_inversion"] and
            result["max_abs"] <= thresholds["max_abs"] and
            result["p99_abs"] <= thresholds["p99_abs"] and
            result["nmse"] <= thresholds["nmse"] and
            result["tvd"] <= thresholds["tvd"] and
            result["kl"] <= thresholds["kl"] and
            result["top5_overlap"] >= thresholds["min_top5_overlap"] and
            result["top20_overlap"] >= thresholds["min_top20_overlap"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        thresholds = load_thresholds(args.thresholds)
        reference_files = sorted(args.reference_dir.glob("decode_*.logits.json"))
        candidate_files = sorted(args.candidate_dir.glob("decode_*.logits.json"))
        if not reference_files or [p.name for p in reference_files] != [p.name for p in candidate_files]:
            raise ValueError("reference and candidate decode-logit file sets must be non-empty and identical")
        steps = [compare_pair(load(ref), load(cand), thresholds)
                 for ref, cand in zip(reference_files, candidate_files)]
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"teacher-logits: FAIL {error}", file=sys.stderr)
        return 1

    mismatches = [item for item in steps if not item["argmax_equal"]]
    breaches = ([item for item in steps if not item["envelope_pass"]]
                if thresholds is not None else [])
    summary = {
        "baseline_id": thresholds["baseline_id"] if thresholds else None,
        "mode": "gate" if thresholds else "diagnostic",
        "steps": len(steps),
        "argmax_mismatches": len(mismatches),
        "first_argmax_mismatch": mismatches[0] if mismatches else None,
        "envelope_breaches": len(breaches) if thresholds else None,
        "first_envelope_breach": breaches[0] if breaches else None,
        "max_abs": max(item["max_abs"] for item in steps),
        "max_p99_abs": max(item["p99_abs"] for item in steps),
        "max_nmse": max(item["nmse"] for item in steps),
        "max_tvd": max(item["tvd"] for item in steps),
        "max_kl": max(item["kl"] for item in steps),
        "min_top5_overlap": min(item["top5_overlap"] for item in steps),
        "min_top20_overlap": min(item["top20_overlap"] for item in steps),
        "passed": not breaches if thresholds is not None else None,
    }
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    if thresholds is not None and breaches:
        print("teacher-logits: FAIL versioned lane-B envelope", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
