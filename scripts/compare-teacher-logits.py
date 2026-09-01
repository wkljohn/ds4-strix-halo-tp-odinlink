#!/usr/bin/env python3
"""Compare frozen-token decode logits without autoregressive cascade.

The tool is diagnostic unless an explicit versioned threshold JSON is supplied.
It never turns its built-in defaults into a lane-B acceptance policy.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import re
import shlex
import sys
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path, *, reference: bool = False) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    logits = value.get("logits")
    vocab = value.get("vocab")
    if not isinstance(vocab, int) or not isinstance(logits, list) or len(logits) != vocab:
        raise ValueError(f"{path}: malformed logits/vocab")
    if any(not isinstance(item, (int, float)) or not math.isfinite(item)
           for item in logits):
        raise ValueError(f"{path}: logits contain null, NaN, Inf, or non-numbers")
    allowed_sources = {
        "ds4-bench-frozen-teacher",
        "ds4-score-official-frozen-teacher",
    }
    if reference:
        allowed_sources.add("ds4-canonical-oracle")
    if value.get("source") not in allowed_sources:
        role = "reference" if reference else "candidate"
        raise ValueError(f"{path}: invalid {role} logit producer")
    for field in ("prefix_tokens", "decode_step", "position", "teacher_token",
                  "quant_bits", "argmax_id", "runner_up_id"):
        if not isinstance(value.get(field), int):
            raise ValueError(f"{path}: missing or invalid {field}")
    if value.get("source") == "ds4-score-official-frozen-teacher":
        if not isinstance(value.get("case_id"), str) or not value["case_id"]:
            raise ValueError(f"{path}: missing or invalid case_id")
        if not isinstance(value.get("case_step"), int):
            raise ValueError(f"{path}: missing or invalid case_step")
    for field in ("quality", "dspark", "dspark_strict"):
        if type(value.get(field)) is not bool:
            raise ValueError(f"{path}: missing or invalid {field}")
    return value


def top_ids(values: list[float], count: int) -> list[int]:
    selected = heapq.nlargest(
        count, enumerate(values), key=lambda item: (item[1], -item[0]))
    return [index for index, _ in selected]


def probability_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float, float]:
    ref_max = float(reference.max())
    cand_max = float(candidate.max())
    ref_exp = np.exp(reference - ref_max)
    cand_exp = np.exp(candidate - cand_max)
    ref_sum = float(ref_exp.sum(dtype=np.float64))
    cand_sum = float(cand_exp.sum(dtype=np.float64))
    ref_log_z = ref_max + math.log(ref_sum)
    cand_log_z = cand_max + math.log(cand_sum)
    p = ref_exp / ref_sum
    q = cand_exp / cand_sum
    tvd = 0.5 * float(np.abs(p - q).sum(dtype=np.float64))
    kl = float(np.sum(p * ((reference - ref_log_z) -
                           (candidate - cand_log_z)), dtype=np.float64))
    return tvd, max(0.0, kl), ref_log_z, cand_log_z


def distribution(values: list[float]) -> dict[str, float]:
    """Return a stable, explicit distribution summary for a non-empty sample."""
    sample = np.asarray(values, dtype=np.float64)
    if sample.size == 0 or not np.all(np.isfinite(sample)):
        raise ValueError("distribution input must be non-empty and finite")
    return {
        "min": float(sample.min()),
        "mean": float(sample.mean(dtype=np.float64)),
        "median": float(np.quantile(sample, 0.50, method="linear")),
        "p90": float(np.quantile(sample, 0.90, method="linear")),
        "p95": float(np.quantile(sample, 0.95, method="linear")),
        "p99": float(np.quantile(sample, 0.99, method="linear")),
        "max": float(sample.max()),
    }


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


def compare_pair(reference: dict, candidate: dict, thresholds: dict | None,
                 allow_quality_difference: bool = False) -> dict:
    for field in ("vocab", "prefix_tokens", "decode_step", "position",
                  "teacher_token", "quant_bits", "quality", "dspark",
                  "dspark_strict"):
        if field == "quality" and allow_quality_difference:
            continue
        if reference.get(field) != candidate.get(field):
            raise ValueError(f"metadata {field}: {reference.get(field)!r} != {candidate.get(field)!r}")
    if (reference.get("source") == "ds4-score-official-frozen-teacher" or
            candidate.get("source") == "ds4-score-official-frozen-teacher"):
        for field in ("case_id", "case_step"):
            if reference.get(field) != candidate.get(field):
                raise ValueError(
                    f"metadata {field}: {reference.get(field)!r} != {candidate.get(field)!r}")

    ref_values = reference["logits"]
    cand_values = candidate["logits"]
    ref = np.asarray(ref_values, dtype=np.float64)
    cand = np.asarray(cand_values, dtype=np.float64)
    differences = np.abs(ref - cand)
    sum_diff_sq = float(np.dot(differences, differences))
    sum_ref_sq = float(np.dot(ref, ref))
    ref_top20 = top_ids(ref_values, min(20, len(ref_values)))
    cand_top20 = top_ids(cand_values, min(20, len(cand_values)))
    ref_margin = ref[ref_top20[0]] - ref[ref_top20[1]] if len(ref_top20) > 1 else math.inf
    cand_margin = cand[cand_top20[0]] - cand[cand_top20[1]] if len(cand_top20) > 1 else math.inf
    teacher_token = reference["teacher_token"]
    if not 0 <= teacher_token < len(ref):
        raise ValueError(f"teacher_token {teacher_token} is outside vocab {len(ref)}")
    tvd, kl, ref_log_z, cand_log_z = probability_metrics(ref, cand)
    reference_teacher_nll = ref_log_z - float(ref[teacher_token])
    candidate_teacher_nll = cand_log_z - float(cand[teacher_token])
    result = {
        "decode_step": reference["decode_step"],
        "position": reference["position"],
        "teacher_token": reference["teacher_token"],
        "reference_argmax": ref_top20[0],
        "candidate_argmax": cand_top20[0],
        "argmax_equal": ref_top20[0] == cand_top20[0],
        "reference_margin": ref_margin,
        "candidate_margin": cand_margin,
        "max_abs": float(differences.max()),
        "p99_abs": float(np.quantile(differences, 0.99, method="higher")),
        "nmse": sum_diff_sq / sum_ref_sq if sum_ref_sq else (0.0 if not sum_diff_sq else math.inf),
        "tvd": tvd,
        "kl": kl,
        "top5_overlap": len(set(ref_top20[:5]) & set(cand_top20[:5])),
        "top20_overlap": len(set(ref_top20) & set(cand_top20)),
        "mean_signed_error": float(np.mean(cand - ref)),
        "positive_error_fraction": float(np.mean(cand > ref)),
        "reference_teacher_nll": reference_teacher_nll,
        "candidate_teacher_nll": candidate_teacher_nll,
        "teacher_nll_delta": candidate_teacher_nll - reference_teacher_nll,
    }
    if reference.get("source") == "ds4-score-official-frozen-teacher":
        result["case_id"] = reference["case_id"]
        result["case_step"] = reference["case_step"]
    if thresholds is None:
        result["envelope_pass"] = None
        result["far_margin_inversion"] = None
    else:
        result["far_margin_inversion"] = bool(
            not result["argmax_equal"] and ref_margin > 2.0 * thresholds["e_bound"])
        result["envelope_pass"] = bool(
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
    parser.add_argument("--steps-output", type=Path,
                        help="write every per-position metric as JSON Lines")
    parser.add_argument("--allow-quality-difference", action="store_true",
                        help="explicitly compare an unfused quality oracle with an optimized path")
    parser.add_argument(
        "--score-arm-mode", choices=(
            "kda-tp", "kda-kslice", "repeat", "full-split-order-null",
            "null-vs-kslice", "fallback-vs-kslice",
            "attn-scalar-vs-f32-gemm"),
        help="required arm relationship for score_official GLM5 diagnostics")
    args = parser.parse_args()
    try:
        thresholds = load_thresholds(args.thresholds)
        reference_files = sorted(args.reference_dir.glob("decode_*.logits.json"))
        candidate_files = sorted(args.candidate_dir.glob("decode_*.logits.json"))
        if not reference_files or [p.name for p in reference_files] != [p.name for p in candidate_files]:
            raise ValueError("reference and candidate decode-logit file sets must be non-empty and identical")
        first_pair = (load(reference_files[0], reference=True),
                      load(candidate_files[0]))
        score_official_source = any(
            value.get("source") == "ds4-score-official-frozen-teacher"
            for value in first_pair)
        if score_official_source:
            if args.score_arm_mode is None:
                raise ValueError("--score-arm-mode is required for score_official dumps")
            def read_manifest(path: Path) -> dict[str, str]:
                values: dict[str, str] = {}
                for line in path.read_text(encoding="utf-8").splitlines():
                    if "=" not in line:
                        raise ValueError(f"{path}: malformed manifest line")
                    key, value = line.split("=", 1)
                    if not key or key in values:
                        raise ValueError(f"{path}: duplicate or empty manifest key {key!r}")
                    values[key] = value
                return values

            reference_manifest = read_manifest(args.reference_dir / "manifest")
            candidate_manifest = read_manifest(args.candidate_dir / "manifest")
            identity_fields = (
                "producer", "source_commit", "source_dirty", "model",
                "model_size", "model_sample_sha256", "ds4_sha256",
                "scorer_sha256", "quality_input_sha256", "start_case",
                "cases", "teacher_positions", "rdma_profile",
            )
            missing = [field for field in identity_fields
                       if field not in reference_manifest or
                       field not in candidate_manifest]
            if missing:
                raise ValueError("teacher-logit manifests lack identity fields: " +
                                 ", ".join(missing))
            for field in identity_fields:
                if reference_manifest[field] != candidate_manifest[field]:
                    raise ValueError(
                        f"manifest {field}: {reference_manifest[field]!r} != "
                        f"{candidate_manifest[field]!r}")
            expected_arms = {
                "kda-tp": ("kda-off", "kda-tp"),
                "kda-kslice": ("kda-tp", "kda-kslice"),
                "repeat": ("kda-kslice", "kda-kslice"),
                "full-split-order-null": ("kda-tp", "kda-tp"),
                "null-vs-kslice": ("kda-tp", "kda-kslice"),
                "fallback-vs-kslice": ("kda-tp", "kda-kslice"),
                "attn-scalar-vs-f32-gemm": ("attn-scalar", "attn-gemm-f32"),
            }[args.score_arm_mode]
            actual_arms = (reference_manifest.get("teacher_arm"),
                           candidate_manifest.get("teacher_arm"))
            if actual_arms != expected_arms:
                raise ValueError(
                    f"score arm relationship {actual_arms!r} != {expected_arms!r}")

            def parse_env(encoded: str) -> dict[str, str]:
                values: dict[str, str] = {}
                for item in shlex.split(encoded):
                    if "=" not in item:
                        raise ValueError(f"malformed extra_env item {item!r}")
                    key, value = item.split("=", 1)
                    values[key] = value
                return values

            ref_env = parse_env(reference_manifest.get("extra_env", ""))
            cand_env = parse_env(candidate_manifest.get("extra_env", ""))
            selectors = {"DS4_GLM5_KDA_TP", "DS4_GLM5_KDA_OUTPUT_KSLICE"}
            if args.score_arm_mode == "full-split-order-null":
                null_key = "DS4_ROCM_BF16_FULL_SPLIT_ORDER"
                if null_key in ref_env or cand_env.get(null_key) != "1":
                    raise ValueError(
                        "full-split-order null requires the legal reorder only "
                        "in the candidate arm")
                selectors.add(null_key)
            elif args.score_arm_mode == "null-vs-kslice":
                null_key = "DS4_ROCM_BF16_FULL_SPLIT_ORDER"
                if ref_env.get(null_key) != "1" or null_key in cand_env:
                    raise ValueError(
                        "null-vs-kslice requires the legal reorder only in "
                        "the reference arm")
                selectors.add(null_key)
            elif args.score_arm_mode == "fallback-vs-kslice":
                fallback_key = "DS4_ROCM_DISABLE_BF16_DECODE_MLP64"
                if ref_env.get(fallback_key) != "1" or fallback_key in cand_env:
                    raise ValueError(
                        "fallback-vs-kslice requires the independent BF16 "
                        "fallback only in the reference arm")
                selectors.add(fallback_key)
            elif args.score_arm_mode == "attn-scalar-vs-f32-gemm":
                nope_key = "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE"
                f32_key = "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_F32"
                if (nope_key in ref_env or f32_key in ref_env or
                        cand_env.get(nope_key) != "1" or
                        cand_env.get(f32_key) != "1"):
                    raise ValueError(
                        "attention comparison requires FP32 NoPE GEMM only "
                        "in the candidate arm")
                selectors.update((nope_key, f32_key))
            if ({k: v for k, v in ref_env.items() if k not in selectors} !=
                    {k: v for k, v in cand_env.items() if k not in selectors}):
                raise ValueError("score arms differ outside the KDA selectors")
            selector_expectations = {
                "kda-off": ("0", "0"),
                "kda-tp": ("1", "0"),
                "kda-kslice": ("1", "1"),
                "attn-scalar": ("1", "0"),
                "attn-gemm-f32": ("1", "0"),
            }
            for manifest, env, arm in (
                    (reference_manifest, ref_env, actual_arms[0]),
                    (candidate_manifest, cand_env, actual_arms[1])):
                expected_tp, expected_slice = selector_expectations[arm]
                if (env.get("DS4_GLM5_KDA_TP"),
                        env.get("DS4_GLM5_KDA_OUTPUT_KSLICE")) != (
                            expected_tp, expected_slice):
                    raise ValueError(f"manifest arm {arm} has the wrong KDA selectors")
                expected_features = re.compile(
                    rf"GLM5 TP features: kda_tp={expected_tp} "
                    rf"kda_output_kslice={expected_slice}$")
                for field in ("coordinator_features", "worker_features"):
                    if not expected_features.search(manifest.get(field, "")):
                        raise ValueError(f"manifest arm {arm} has invalid {field}")
        elif args.score_arm_mode is not None:
            raise ValueError("--score-arm-mode applies only to score_official dumps")

        steps = [compare_pair(first_pair[0], first_pair[1], thresholds,
                              args.allow_quality_difference)]
        for ref_path, cand_path in zip(reference_files[1:], candidate_files[1:]):
            steps.append(compare_pair(load(ref_path, reference=True),
                                      load(cand_path), thresholds,
                                      args.allow_quality_difference))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"teacher-logits: FAIL {error}", file=sys.stderr)
        return 1

    mismatches = [item for item in steps if not item["argmax_equal"]]
    breaches = ([item for item in steps if not item["envelope_pass"]]
                if thresholds is not None else [])
    far_margin = ([item for item in steps if item["far_margin_inversion"]]
                  if thresholds is not None else [])
    mismatch_reference_margins = [
        float(item["reference_margin"]) for item in mismatches]
    distributions = {
        key: distribution([float(item[key]) for item in steps])
        for key in (
            "max_abs", "p99_abs", "nmse", "tvd", "kl",
            "reference_margin", "candidate_margin", "top5_overlap",
            "top20_overlap", "mean_signed_error", "positive_error_fraction",
            "reference_teacher_nll", "candidate_teacher_nll",
            "teacher_nll_delta",
        )
    }
    case_summaries = []
    if all("case_id" in item for item in steps):
        case_ids = list(dict.fromkeys(item["case_id"] for item in steps))
        for case_id in case_ids:
            case_steps = [item for item in steps if item["case_id"] == case_id]
            case_summaries.append({
                "case_id": case_id,
                "steps": len(case_steps),
                "argmax_mismatches": sum(
                    not item["argmax_equal"] for item in case_steps),
                "teacher_nll_reference_mean": distribution([
                    item["reference_teacher_nll"] for item in case_steps
                ])["mean"],
                "teacher_nll_candidate_mean": distribution([
                    item["candidate_teacher_nll"] for item in case_steps
                ])["mean"],
                "teacher_nll_delta_mean": distribution([
                    item["teacher_nll_delta"] for item in case_steps
                ])["mean"],
                "kl": distribution([item["kl"] for item in case_steps]),
                "tvd": distribution([item["tvd"] for item in case_steps]),
                "max_abs": distribution([
                    item["max_abs"] for item in case_steps]),
            })

    summary = {
        "baseline_id": thresholds["baseline_id"] if thresholds else None,
        "allow_quality_difference": args.allow_quality_difference,
        "mode": "gate" if thresholds else "diagnostic",
        "steps": len(steps),
        "argmax_mismatches": len(mismatches),
        "far_margin_inversions": len(far_margin) if thresholds else None,
        "first_argmax_mismatch": mismatches[0] if mismatches else None,
        "argmax_mismatch_reference_margin": (
            distribution(mismatch_reference_margins)
            if mismatch_reference_margins else None),
        "max_argmax_mismatch_reference_margin": (
            max(mismatch_reference_margins)
            if mismatch_reference_margins else None),
        "envelope_breaches": len(breaches) if thresholds else None,
        "first_envelope_breach": breaches[0] if breaches else None,
        "max_abs": max(item["max_abs"] for item in steps),
        "max_p99_abs": max(item["p99_abs"] for item in steps),
        "max_nmse": max(item["nmse"] for item in steps),
        "max_tvd": max(item["tvd"] for item in steps),
        "max_kl": max(item["kl"] for item in steps),
        "min_top5_overlap": min(item["top5_overlap"] for item in steps),
        "min_top20_overlap": min(item["top20_overlap"] for item in steps),
        "max_abs_mean_signed_error": max(
            abs(item["mean_signed_error"]) for item in steps),
        "positive_error_fraction_range": [
            min(item["positive_error_fraction"] for item in steps),
            max(item["positive_error_fraction"] for item in steps),
        ],
        "distributions": distributions,
        "case_summaries": case_summaries,
        "sources": {
            "reference_dir": str(args.reference_dir.resolve()),
            "candidate_dir": str(args.candidate_dir.resolve()),
            "reference_manifest": str((args.reference_dir / "manifest").resolve()),
            "reference_manifest_sha256": sha256(args.reference_dir / "manifest"),
            "candidate_manifest": str((args.candidate_dir / "manifest").resolve()),
            "candidate_manifest_sha256": sha256(args.candidate_dir / "manifest"),
            "pairs": [
                {
                    "name": reference.name,
                    "reference_sha256": sha256(reference),
                    "candidate_sha256": sha256(candidate),
                }
                for reference, candidate in zip(reference_files, candidate_files)
            ],
        },
        "thresholds_sha256": sha256(args.thresholds) if args.thresholds else None,
        "worst_steps": {
            key: max(steps, key=lambda item: item[key])
            for key in ("max_abs", "p99_abs", "nmse", "tvd", "kl")
        },
        "passed": not breaches and not far_margin if thresholds is not None else None,
    }
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    if args.steps_output:
        with args.steps_output.open("w", encoding="utf-8") as handle:
            for item in steps:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
    sys.stdout.write(encoded)
    if thresholds is not None and breaches:
        print("teacher-logits: FAIL versioned lane-B envelope", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
