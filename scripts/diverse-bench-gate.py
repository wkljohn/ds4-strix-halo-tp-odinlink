#!/usr/bin/env python3
"""Create or verify the compact cross-disciplinary benchmark summary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


class Error(RuntimeError):
    pass


PROMPT_SHA256 = "24d19432acab4d4cd2971d938b3c013fcfad1010ed701218bc7bdc1b630ecfef"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_run(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise Error(f"diverse benchmark must contain exactly one result row: {path}")
    row = rows[0]
    try:
        result = {
            "ctx_tokens": int(row["ctx_tokens"]),
            "gen_tokens": int(row["gen_tokens"]),
            "prefill_tps": float(row["prefill_tps"]),
            "decode_tps": float(row["gen_tps"]),
            "steady_decode_tps": float(row["gen_steady_tps"]),
            "fingerprint": row["gen_token_fnv64"].lower(),
        }
    except (KeyError, ValueError) as error:
        raise Error(f"invalid benchmark CSV {path}: {error}") from error
    if result["ctx_tokens"] != 4096 or result["gen_tokens"] != 300:
        raise Error(f"unexpected diverse workload shape in {path}")
    if any(not math.isfinite(result[key]) or result[key] <= 0 for key in
           ("prefill_tps", "decode_tps", "steady_decode_tps")):
        raise Error(f"non-positive or non-finite metric in {path}")
    if not re.fullmatch(r"[0-9a-f]{16}", result["fingerprint"]):
        raise Error(f"invalid token fingerprint in {path}")
    return result


def manifest_for(path: Path) -> Path:
    if path.suffix != ".csv":
        raise Error(f"benchmark result must use a .csv suffix: {path}")
    return path.with_suffix(".manifest")


def read_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    expected = {
        "prompt_sha256": PROMPT_SHA256,
        "frontier": "4096",
        "generated_tokens": "300",
        "context": "4608",
        "prefill_chunk": "2048",
        "dspark": "0",
    }
    for key, wanted in expected.items():
        if values.get(key) != wanted:
            raise Error(f"unexpected {key} in {path}: {values.get(key)!r}")
    return values


def ref(path: Path) -> dict:
    path = path.resolve()
    manifest = manifest_for(path).resolve()
    read_manifest(manifest)
    return {
        "path": str(path),
        "sha256": digest(path),
        "manifest_path": str(manifest),
        "manifest_sha256": digest(manifest),
    }


def cv(values: list[float]) -> float:
    return statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else 0.0


def calculate(baseline_paths: list[Path], candidate_path: Path, lane: str) -> dict:
    if lane not in {"A", "B", "C"}:
        raise Error("lane must be A, B, or C")
    if len(baseline_paths) != 3:
        raise Error("exactly three frozen baseline runs are required")
    baseline = [read_run(path.resolve()) for path in baseline_paths]
    candidate = read_run(candidate_path.resolve())
    manifests = [read_manifest(manifest_for(path.resolve())) for path in baseline_paths]
    candidate_manifest = read_manifest(manifest_for(candidate_path.resolve()))
    if any(manifest.get("candidate") != "1" for manifest in [*manifests, candidate_manifest]):
        raise Error("all diverse baseline/candidate runs must use candidate-validation mode")
    identity_keys = (
        "bench_config_sha256", "model_size", "model_sample_sha256",
        "prompt_sha256", "frontier", "generated_tokens", "context",
        "prefill_chunk", "rdma_profile", "coordinator_rdma_device",
        "worker_rdma_device", "rdma_gid_index", "dspark",
    )
    for manifest in manifests:
        if any(manifest.get(key) != candidate_manifest.get(key) for key in identity_keys):
            raise Error("baseline/candidate diverse manifests do not describe a matched workload")
    if lane == "A":
        fingerprints = {run["fingerprint"] for run in baseline}
        if len(fingerprints) != 1 or candidate["fingerprint"] not in fingerprints:
            raise Error("lane A cross-disciplinary fingerprints do not match")

    prefill_values = [run["prefill_tps"] for run in baseline]
    decode_values = [run["decode_tps"] for run in baseline]
    prefill_median = statistics.median(prefill_values)
    decode_median = statistics.median(decode_values)
    prefill_cv = cv(prefill_values)
    decode_cv = cv(decode_values)
    allowed_prefill = max(5.0, 200.0 * prefill_cv)
    allowed_decode = max(3.0, 200.0 * decode_cv)
    prefill_change = 100.0 * (candidate["prefill_tps"] / prefill_median - 1.0)
    decode_change = 100.0 * (candidate["decode_tps"] / decode_median - 1.0)
    passed = prefill_change >= -allowed_prefill and decode_change >= -allowed_decode
    return {
        "schema_version": 1,
        "workload_id": "cross-discipline-long-v1",
        "disciplines": [
            "software-debugging",
            "quantitative-science",
            "policy-document-retrieval",
            "structured-data-analysis",
        ],
        "shape": {"frontier": 4096, "generated_tokens": 300,
                  "context": 4608, "prefill_chunk": 2048},
        "lane": lane,
        "baseline": [ref(path) for path in baseline_paths],
        "candidate": ref(candidate_path),
        "metrics": {
            "baseline_prefill_median": prefill_median,
            "baseline_decode_median": decode_median,
            "baseline_prefill_cv": prefill_cv,
            "baseline_decode_cv": decode_cv,
            "candidate_prefill_tps": candidate["prefill_tps"],
            "candidate_decode_tps": candidate["decode_tps"],
            "prefill_change_pct": prefill_change,
            "decode_change_pct": decode_change,
            "allowed_prefill_regression_pct": allowed_prefill,
            "allowed_decode_regression_pct": allowed_decode,
        },
        "passed": passed,
    }


def confined(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise Error(f"evidence escapes DS4_RESEARCH_ROOT: {resolved}")
    return resolved


def verify(summary_path: Path) -> dict:
    root = Path(os.environ["DS4_RESEARCH_ROOT"]).resolve()
    value = json.loads(summary_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("workload_id") != "cross-discipline-long-v1":
        raise Error("invalid diverse benchmark schema/workload")
    baseline = value.get("baseline", [])
    candidate = value.get("candidate", {})
    if len(baseline) != 3 or not isinstance(candidate, dict):
        raise Error("diverse benchmark requires three baselines and one candidate")
    paths = [confined(Path(item["path"]), root) for item in baseline]
    candidate_path = confined(Path(candidate["path"]), root)
    for item, path in zip([*baseline, candidate], [*paths, candidate_path]):
        if digest(path) != item.get("sha256"):
            raise Error(f"diverse evidence hash mismatch: {path}")
        manifest = confined(Path(item["manifest_path"]), root)
        if digest(manifest) != item.get("manifest_sha256"):
            raise Error(f"diverse manifest hash mismatch: {manifest}")
    recalculated = calculate(paths, candidate_path, str(value.get("lane", "")))
    for key in ("workload_id", "disciplines", "shape", "lane", "baseline", "candidate", "metrics", "passed"):
        if value.get(key) != recalculated.get(key):
            raise Error(f"diverse summary does not match source evidence: {key}")
    if value["passed"] is not True:
        raise Error("cross-disciplinary long benchmark regressed")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--lane", choices=("A", "B", "C"), required=True)
    create.add_argument("--baseline", action="append", type=Path, required=True)
    create.add_argument("--candidate", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("summary", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "create":
            value = calculate(args.baseline, args.candidate, args.lane)
            value["created_utc"] = datetime.now(timezone.utc).isoformat()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if not value["passed"]:
                raise Error("cross-disciplinary long benchmark regressed")
        else:
            value = verify(args.summary)
        metrics = value["metrics"]
        print(f"PASS diverse-long prefill={metrics['candidate_prefill_tps']:.2f} "
              f"decode={metrics['candidate_decode_tps']:.2f} "
              f"decode_change={metrics['decode_change_pct']:+.2f}%")
    except (Error, OSError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
