#!/usr/bin/env python3
"""Deterministic non-inferiority gate for two ds4 frontier-logit dumps."""

import argparse
import json
import math
import sys
from pathlib import Path


def load_dump(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    logits = data.get("logits")
    vocab = data.get("vocab")
    if not isinstance(logits, list) or not isinstance(vocab, int):
        raise ValueError(f"{path}: missing logits/vocab")
    if len(logits) != vocab:
        raise ValueError(f"{path}: logits={len(logits)} vocab={vocab}")
    if not logits or any(not isinstance(v, (int, float)) or not math.isfinite(v)
                         for v in logits):
        raise ValueError(f"{path}: logits contain null, NaN, Inf, or non-numbers")
    return data


def ranking(logits: list[float], count: int) -> list[int]:
    return sorted(range(len(logits)), key=lambda idx: (-logits[idx], idx))[:count]


def distribution_metrics(reference: list[float], candidate: list[float]) -> tuple[float, float]:
    ref_max = max(reference)
    cand_max = max(candidate)
    ref_exp = [math.exp(value - ref_max) for value in reference]
    cand_exp = [math.exp(value - cand_max) for value in candidate]
    ref_sum = math.fsum(ref_exp)
    cand_sum = math.fsum(cand_exp)
    ref_log_z = ref_max + math.log(ref_sum)
    cand_log_z = cand_max + math.log(cand_sum)
    tvd_terms = []
    kl_terms = []
    for ref_logit, cand_logit, ref_e, cand_e in zip(
            reference, candidate, ref_exp, cand_exp):
        p = ref_e / ref_sum
        q = cand_e / cand_sum
        tvd_terms.append(abs(p - q))
        if p != 0.0:
            kl_terms.append(p * ((ref_logit - ref_log_z) -
                                 (cand_logit - cand_log_z)))
    return 0.5 * math.fsum(tvd_terms), max(0.0, math.fsum(kl_terms))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--max-tvd", type=float, default=1.0e-4)
    parser.add_argument("--max-kl", type=float, default=1.0e-4)
    args = parser.parse_args()
    if not (math.isfinite(args.max_tvd) and args.max_tvd >= 0.0 and
            math.isfinite(args.max_kl) and args.max_kl >= 0.0):
        parser.error("thresholds must be finite and nonnegative")

    try:
        reference = load_dump(args.reference)
        candidate = load_dump(args.candidate)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"frontier-logits: FAIL {error}", file=sys.stderr)
        return 1

    for field in ("vocab", "frontier_tokens", "prompt_tokens", "quant_bits"):
        if reference.get(field) != candidate.get(field):
            print(f"frontier-logits: FAIL metadata {field}: "
                  f"{reference.get(field)!r} != {candidate.get(field)!r}",
                  file=sys.stderr)
            return 1

    ref_logits = reference["logits"]
    cand_logits = candidate["logits"]
    ref_top20 = ranking(ref_logits, min(20, len(ref_logits)))
    cand_top20 = ranking(cand_logits, min(20, len(cand_logits)))
    ref_top5 = ref_top20[:min(5, len(ref_top20))]
    cand_top5 = cand_top20[:min(5, len(cand_top20))]
    argmax_ok = ref_top20[0] == cand_top20[0]
    top5_ok = ref_top5 == cand_top5
    top20_ok = set(ref_top20) == set(cand_top20)
    tvd, kl = distribution_metrics(ref_logits, cand_logits)
    passed = (argmax_ok and top5_ok and top20_ok and
              tvd <= args.max_tvd and kl <= args.max_kl)
    print(json.dumps({
        "argmax_equal": argmax_ok,
        "top5_rank_equal": top5_ok,
        "top20_set_equal": top20_ok,
        "tvd": tvd,
        "kl_reference_to_candidate": kl,
        "max_tvd": args.max_tvd,
        "max_kl": args.max_kl,
        "passed": passed,
    }, sort_keys=True))
    if not passed:
        print("frontier-logits: FAIL deterministic non-inferiority gate",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
