#!/usr/bin/env python3
import csv
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "scripts" / "compare-quality-scores.py"
FIELDS = [
    "id", "target_tokens", "nll", "avg_nll", "api_top1_count",
    "api_top1_match", "api_pair_total", "api_pair_agree",
]


def write_scores(path: Path, averages: list[float], top1: int = 9,
                 api_count: int = 10) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        for index, average in enumerate(averages):
            writer.writerow({
                "id": f"case_{index:03d}", "target_tokens": 10,
                "nll": average * 10, "avg_nll": average,
                "api_top1_count": api_count,
                "api_top1_match": top1 if api_count else 0,
                "api_pair_total": api_count,
                "api_pair_agree": 9 if api_count else 0,
            })
    path.with_suffix(".manifest").write_text(
        "model=/model.gguf\nmodel_size=1\nmodel_sample_sha256=" + "a" * 64 +
        "\nsource_commit=" + "0" * 40 + "\nsource_dirty=0\ndspark=0\n",
        encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        reference = root / "reference.tsv"
        candidate = root / "candidate.tsv"
        thresholds = root / "thresholds.json"
        write_scores(reference, [0.5, 0.6, 0.7])
        write_scores(candidate, [0.49, 0.59, 0.69])
        thresholds.write_text(json.dumps({
            "baseline_id": "sha256:" + "1" * 64,
            "min_cases": 3,
            "min_target_tokens": 30,
            "max_mean_nll_delta": 0.0,
            "max_ci95_high_nll_delta": 0.02,
            "min_api_top1_rate_delta": 0.0,
            "min_api_pair_rate_delta": 0.0,
        }), encoding="utf-8")
        passed = subprocess.run(
            [str(TOOL), str(reference), str(candidate), "--thresholds", str(thresholds)],
            text=True, capture_output=True, check=False)
        assert passed.returncode == 0, passed.stderr
        assert json.loads(passed.stdout)["passed"] is True

        write_scores(candidate, [0.48, 0.61, 0.68])
        noisy_better = subprocess.run(
            [str(TOOL), str(reference), str(candidate), "--thresholds", str(thresholds)],
            text=True, capture_output=True, check=False)
        assert noisy_better.returncode == 0, noisy_better.stderr
        assert json.loads(noisy_better.stdout)["metrics"]["paired_ci95_high"] > 0.0

        write_scores(candidate, [0.8, 0.9, 1.0])
        failed = subprocess.run(
            [str(TOOL), str(reference), str(candidate), "--thresholds", str(thresholds)],
            text=True, capture_output=True, check=False)
        assert failed.returncode != 0
        assert json.loads(failed.stdout)["passed"] is False

        write_scores(candidate, [0.49, 0.59, 0.69], top1=8)
        quality_drop = subprocess.run(
            [str(TOOL), str(reference), str(candidate), "--thresholds", str(thresholds)],
            text=True, capture_output=True, check=False)
        assert quality_drop.returncode != 0

        write_scores(reference, [0.5, 0.6, 0.7], api_count=0)
        write_scores(candidate, [0.49, 0.59, 0.69], api_count=0)
        no_api = subprocess.run(
            [str(TOOL), str(reference), str(candidate),
             "--thresholds", str(thresholds)],
            text=True, capture_output=True, check=False)
        assert no_api.returncode != 0
        no_api_result = json.loads(no_api.stdout)
        assert no_api_result["nll_screen_passed"] is True
        assert no_api_result["api_screen_passed"] is None
        assert no_api_result["passed"] is False
        assert no_api_result["blockers"]
        assert "paired NLL screen was still reported" in no_api.stderr
    print("test_compare_quality_scores: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
