#!/usr/bin/env python3
"""Small deterministic tests for the frozen-teacher trajectory comparator."""

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "scripts" / "compare-teacher-logits.py"


def dump(path: Path, logits: list[float], quality: bool = False,
         source: str = "ds4-bench-frozen-teacher") -> None:
    order = sorted(range(len(logits)), key=lambda idx: (-logits[idx], idx))
    value = {
        "source": source,
        "model": "/model.gguf",
        "backend": "rocm",
        "quality": quality,
        "dspark": False,
        "dspark_strict": False,
        "quant_bits": 4,
        "prefix_tokens": 2048,
        "decode_step": 0,
        "position": 2048,
        "vocab": len(logits),
        "teacher_token": 3,
        "teacher_logit": logits[3],
        "argmax_id": order[0],
        "argmax_logit": logits[order[0]],
        "runner_up_id": order[1],
        "runner_up_logit": logits[order[1]],
        "top1_margin": logits[order[0]] - logits[order[1]],
        "teacher_gap": logits[order[0]] - logits[3],
        "logits": logits,
    }
    path.write_text(json.dumps(value), encoding="utf-8")


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(TOOL), *args], check=False, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main() -> int:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        reference = root / "reference"
        candidate = root / "candidate"
        reference.mkdir()
        candidate.mkdir()
        (reference / "manifest").write_text("fixture=reference\n")
        (candidate / "manifest").write_text("fixture=candidate\n")
        dump(reference / "decode_000000.logits.json", [0.0, 1.0, 2.0, 3.0])
        dump(candidate / "decode_000000.logits.json", [0.0, 1.0, 2.0, 3.0])
        exact = run(str(reference), str(candidate))
        assert exact.returncode == 0, exact.stderr
        assert json.loads(exact.stdout)["argmax_mismatches"] == 0

        dump(reference / "decode_000000.logits.json", [0.0, 1.0, 2.0, 3.0],
             source="ds4-canonical-oracle")
        canonical = run(str(reference), str(candidate))
        assert canonical.returncode == 0, canonical.stderr
        dump(candidate / "decode_000000.logits.json", [0.0, 1.0, 2.0, 3.0],
             source="ds4-canonical-oracle")
        assert run(str(reference), str(candidate)).returncode == 1
        dump(candidate / "decode_000000.logits.json", [0.0, 1.0, 2.0, 3.0])

        dump(candidate / "decode_000000.logits.json",
             [0.0, 1.0, 2.0, 3.0], quality=True)
        assert run(str(reference), str(candidate)).returncode == 1
        oracle = run(str(reference), str(candidate), "--allow-quality-difference")
        assert oracle.returncode == 0, oracle.stderr
        assert json.loads(oracle.stdout)["allow_quality_difference"] is True

        dump(candidate / "decode_000000.logits.json", [0.0, 1.0, 4.0, 3.0])
        diagnostic = run(str(reference), str(candidate))
        assert diagnostic.returncode == 0, diagnostic.stderr
        assert json.loads(diagnostic.stdout)["argmax_mismatches"] == 1

        thresholds = root / "thresholds.json"
        thresholds.write_text(json.dumps({
            "baseline_id": "synthetic-v1",
            "e_bound": 0.1,
            "max_abs": 0.1,
            "p99_abs": 0.1,
            "nmse": 1.0e-4,
            "tvd": 1.0e-4,
            "kl": 1.0e-4,
            "min_top5_overlap": 4,
            "min_top20_overlap": 4,
        }), encoding="utf-8")
        gated = run(str(reference), str(candidate), "--thresholds", str(thresholds))
        assert gated.returncode == 1
        assert json.loads(gated.stdout)["passed"] is False
        dump(candidate / "decode_000000.logits.json",
             [0.0, 1.0, 4.0, 3.0], quality=True)
        oracle_gated = run(str(reference), str(candidate),
                           "--allow-quality-difference",
                           "--thresholds", str(thresholds))
        assert oracle_gated.returncode == 1
        assert json.loads(oracle_gated.stdout)["passed"] is False

        thresholds.write_text(json.dumps({
            "baseline_id": "synthetic-v1",
            "e_bound": 0.1,
            "max_abs": 0.1,
            "p99_abs": 0.1,
            "nmse": 0.01,
            "tvd": 0.1,
            "kl": 0.1,
            "min_top5_overlap": 4,
            "min_top20_overlap": 4,
        }), encoding="utf-8")
        dump(reference / "decode_000000.logits.json", [0.0, 1.0, 2.0, 2.05])
        dump(candidate / "decode_000000.logits.json", [0.0, 1.0, 2.06, 2.04])
        near_tie = run(str(reference), str(candidate),
                       "--thresholds", str(thresholds))
        assert near_tie.returncode == 0, near_tie.stderr
        near_summary = json.loads(near_tie.stdout)
        assert near_summary["argmax_mismatches"] == 1
        assert near_summary["far_margin_inversions"] == 0

        dump(reference / "decode_000000.logits.json", [0.0, 1.0, 2.0, 3.0])
        dump(candidate / "decode_000000.logits.json", [0.0, 1.0, 3.01, 2.99])
        far_flip = run(str(reference), str(candidate),
                       "--thresholds", str(thresholds))
        assert far_flip.returncode != 0
        assert json.loads(far_flip.stdout)["far_margin_inversions"] == 1
    print("test_compare_teacher_logits: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
