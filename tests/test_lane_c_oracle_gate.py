#!/usr/bin/env python3
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("candidate_gate", ROOT / "scripts" / "candidate-gate.py")
assert SPEC and SPEC.loader
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    with tempfile.TemporaryDirectory() as raw:
        archive = Path(raw)
        oracle_dir = archive / "oracle"
        candidate_dir = archive / "candidate"
        oracle_dir.mkdir()
        candidate_dir.mkdir()
        model = archive / "model.gguf"
        model.write_bytes(b"model")
        tokens = archive / "tokens.txt"
        tokens.write_text("\n".join(["2"] * 300) + "\n")
        token_sha = digest(tokens)
        generator = archive / "oracle-generator"
        generator.write_text("""#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--definition', required=True)
p.add_argument('--prefix', type=int, required=True)
p.add_argument('--token-file', required=True)
p.add_argument('--output-dir', required=True)
a = p.parse_args()
tokens = [int(item) for item in Path(a.token_file).read_text().split()]
out = Path(a.output_dir)
out.mkdir(parents=True, exist_ok=True)
for index, teacher in enumerate(tokens):
    value = {
        'source': 'ds4-canonical-oracle', 'model': a.model,
        'backend': 'rocm', 'quality': False, 'dspark': False,
        'dspark_strict': False, 'quant_bits': 4, 'prefix_tokens': a.prefix,
        'decode_step': index, 'position': a.prefix + index, 'vocab': 4,
        'teacher_token': teacher, 'teacher_logit': 3.0, 'argmax_id': 2,
        'argmax_logit': 3.0, 'runner_up_id': 3, 'runner_up_logit': 2.0,
        'top1_margin': 1.0, 'teacher_gap': 0.0,
        'logits': [0.0, 1.0, 3.0, 2.0],
    }
    (out / f'decode_{index:06d}.logits.json').write_text(json.dumps(value))
""")
        generator.chmod(0o755)
        generator_sha = digest(generator)
        runner = Path(sys.executable).resolve()
        runner_item = {"scope": "system", "path": str(runner),
                       "sha256": digest(runner)}
        generator_item = {"scope": "research", "path": generator.name,
                          "sha256": generator_sha}
        generator_closure = [generator_item, runner_item]
        closure_sha = GATE.canonical_sha256(generator_closure)
        generator_environment = {
            "kind": "python", "implementation": sys.implementation.name,
            "version": sys.version, "runner": runner_item, "packages": [],
        }
        environment_id = GATE.canonical_sha256(generator_environment)
        quality_ref = {"sha256": "1" * 64, "manifest_sha256": "2" * 64}
        numerical_thresholds = {
            "e_bound": 0.1, "max_abs": 0.1, "p99_abs": 0.1,
            "nmse": 0.0001, "tvd": 0.0001, "kl": 0.0001,
            "min_top5_overlap": 4, "min_top20_overlap": 4,
            "min_teacher_steps": 300, "allow_quality_difference": False,
        }
        quality_thresholds = {
            "min_cases": 100, "min_target_tokens": 2289,
            "max_mean_nll_delta": 0.0, "max_ci95_high_nll_delta": 0.02,
            "min_api_top1_rate_delta": 0.0, "min_api_pair_rate_delta": 0.0,
        }
        predecessor = {
            "schema_version": 1, "kind": "ds4-numerical-baseline",
            "key": {
                "model_sample_sha256": "a" * 64, "model_sha256": "b" * 64,
                "model_size": 5, "quantization": "Q4_K",
                "workload": {"frontier": "2048", "frozen_token_sha256": token_sha},
            },
            "reference": {
                "fnv64": "1234567890abcdef",
                "numerical": {"files": [{"name": f"decode_{i:06d}.logits.json",
                                            "sha256": "3" * 64} for i in range(300)],
                              "manifest_sha256": "4" * 64},
                "quality": quality_ref,
            },
            "thresholds": {"numerical": numerical_thresholds,
                           "quality": quality_thresholds,
                           "arithmetic_identity": {
                               "ignored_env_keys": [],
                               "ignored_runtime_feature_mask": 0,
                           }},
            "provenance": {"lane_origin": "bootstrap"},
        }
        predecessor_digest = GATE.canonical_sha256(predecessor)
        baseline_dir = archive / "baselines" / "sha256"
        baseline_dir.mkdir(parents=True)
        (baseline_dir / f"{predecessor_digest}.json").write_text(
            json.dumps(predecessor, indent=2, sort_keys=True) + "\n")
        reviews = []
        for kind in ("fable-review", "grok-review"):
            review = archive / f"{kind}.txt"
            review.write_text("approved governance-only oracle adoption\n")
            reviews.append({"kind": kind, "path": str(review), "sha256": digest(review)})
        amendment = {
            "schema_version": 1, "kind": "ds4-baseline-amendment",
            "amendment_id": "fixture-oracle-adoption",
            "baseline_id": f"sha256:{predecessor_digest}",
            "rationale": "Approve a canonical producer without changing the token reference.",
            "add_oracle_generator": {
                "id": "fixture-oracle", "entrypoint": generator_item,
                "runner": runner_item, "environment": generator_environment,
                "environment_id": environment_id,
                "closure": generator_closure, "closure_sha256": closure_sha,
            },
            "threshold_updates": {"oracle_numerical": numerical_thresholds},
            "evidence": reviews,
        }
        amendment_path = archive / "amendment.json"
        amendment_path.write_text(json.dumps(amendment, indent=2, sort_keys=True) + "\n")
        weak_first = json.loads(json.dumps(amendment))
        weak_first["amendment_id"] = "fixture-weak-first-oracle"
        weak_first["threshold_updates"]["oracle_numerical"]["max_abs"] = 1.0e6
        weak_first_path = archive / "weak-first.json"
        weak_first_path.write_text(json.dumps(weak_first, indent=2, sort_keys=True) + "\n")
        try:
            GATE.amend_baseline(ROOT, archive, weak_first_path)
        except GATE.GateError as error:
            assert "weakens max_abs" in str(error)
        else:
            raise AssertionError("first oracle envelope was looser than the numerical baseline")
        captured = io.StringIO()
        with redirect_stdout(captured):
            GATE.amend_baseline(ROOT, archive, amendment_path)
        successor_id = captured.getvalue().strip()
        _, successor = GATE.load_baseline(archive, successor_id)
        assert successor["reference"]["fnv64"] == predecessor["reference"]["fnv64"]
        assert successor["reference"]["quality_anchor"] == quality_ref
        assert successor["oracle_generators"][0]["closure_sha256"] == closure_sha
        assert successor["provenance"]["replaces"] == f"sha256:{predecessor_digest}"
        weakened = dict(numerical_thresholds)
        weakened["max_abs"] = 0.2
        weakening = {
            "schema_version": 1, "kind": "ds4-baseline-amendment",
            "amendment_id": "fixture-weaken-envelope", "baseline_id": successor_id,
            "rationale": "This fixture must be rejected.",
            "threshold_updates": {"oracle_numerical": weakened},
            "evidence": reviews,
        }
        weakening_path = archive / "weakening.json"
        weakening_path.write_text(json.dumps(weakening, indent=2, sort_keys=True) + "\n")
        try:
            GATE.amend_baseline(ROOT, archive, weakening_path)
        except GATE.GateError:
            pass
        else:
            raise AssertionError("baseline amendment weakened a numerical bound")
        model_sample = "a" * 64
        oracle_files = []
        predecessor_files = []
        for index in range(300):
            base = {
                "source": "ds4-canonical-oracle", "model": str(model),
                "backend": "rocm", "quality": False, "dspark": False,
                "dspark_strict": False, "quant_bits": 4, "prefix_tokens": 2048,
                "decode_step": index, "position": 2048 + index, "vocab": 4,
                "teacher_token": 2, "teacher_logit": 3.0, "argmax_id": 2,
                "argmax_logit": 3.0, "runner_up_id": 3, "runner_up_logit": 2.0,
                "top1_margin": 1.0, "teacher_gap": 0.0,
                "logits": [0.0, 1.0, 3.0, 2.0],
            }
            name = f"decode_{index:06d}.logits.json"
            oracle = oracle_dir / name
            candidate = candidate_dir / name
            encoded = json.dumps(base)
            oracle.write_text(encoded)
            candidate_value = dict(base)
            candidate_value["source"] = "ds4-bench-frozen-teacher"
            candidate.write_text(json.dumps(candidate_value))
            oracle_files.append({"name": name, "sha256": digest(oracle)})
            predecessor_files.append({"name": name, "sha256": "0" * 64})
        file_list_hash = hashlib.sha256("".join(
            f"{item['name']} {item['sha256']}\n" for item in oracle_files
        ).encode()).hexdigest()
        (oracle_dir / "manifest").write_text(
            f"oracle=1\ndefinition_id=corrected-v2\nmodel={model}\nmodel_size=5\n"
            f"model_sample_sha256={model_sample}\nsource_commit={'0' * 40}\n"
            f"source_dirty=0\ngenerator_id=fixture-oracle\n"
            f"generator_closure_sha256={closure_sha}\n"
            f"generator_environment_id={environment_id}\n"
            "toolchain_id=oracle-test\n"
            f"prefix_tokens=2048\nfile_count=300\nfiles_sha256={file_list_hash}\n"
            f"token_file={tokens}\ntoken_sha256={token_sha}\n")
        (candidate_dir / "manifest").write_text(
            f"model={model}\nmodel_size=5\nmodel_sample_sha256={model_sample}\n"
            f"source_commit={'0' * 40}\nsource_dirty=0\ntoolchain_id=test\n"
            f"run_id=candidate-numerical\nbench_config_sha256={'e' * 64}\n"
            f"ds4_sha256={'b' * 64}\npeer_ds4_sha256={'b' * 64}\n"
            f"ds4_bench_tp_sha256={'d' * 64}\ntp_runtime_features=0x00080001\n"
            "common_env=FIXTURE_COMMON=1\nworker_env=FIXTURE_WORKER=1\n"
            "coordinator_env=FIXTURE_COORDINATOR=1\nextra_env=FIXTURE_ARITHMETIC=1\n"
            "transport_library_path=\n"
            f"prefix_tokens=2048\nfrontier=2048\ncontext=4096\nprefill_chunk=2048\n"
            f"file_count=300\nfrozen_token_sha256={token_sha}\ndspark=0\n")
        baseline_id = "sha256:" + "1" * 64
        thresholds = {
            "baseline_id": baseline_id, "e_bound": 0.1, "max_abs": 0.1,
            "p99_abs": 0.1, "nmse": 0.0001, "tvd": 0.0001, "kl": 0.0001,
            "min_top5_overlap": 4, "min_top20_overlap": 4,
            "min_teacher_steps": 300, "allow_quality_difference": False,
        }
        threshold_path = archive / "thresholds.json"
        threshold_path.write_text(json.dumps(thresholds, indent=2, sort_keys=True) + "\n")
        summary = archive / "summary.json"
        result = subprocess.run([
            str(ROOT / "scripts" / "compare-teacher-logits.py"),
            str(oracle_dir), str(candidate_dir), "--thresholds", str(threshold_path),
            "--output", str(summary),
        ], text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr
        baseline = {
            "key": {"quantization": "Q4_K", "workload": {
                "frontier": "2048", "context": "4096",
                "prefill_chunk": "2048", "frozen_token_sha256": token_sha}},
            "reference": {
                "numerical": {"files": predecessor_files, "manifest_sha256": "0" * 64},
            },
            "thresholds": {
                "arithmetic_identity": {
                    "ignored_env_keys": [],
                    "ignored_runtime_feature_mask": 0,
                },
                "numerical": {key: value for key, value in thresholds.items()
                              if key != "baseline_id"},
                "oracle_numerical": {key: value for key, value in thresholds.items()
                                     if key != "baseline_id"},
            },
            "oracle_generators": [{
                "id": "fixture-oracle",
                "entrypoint": generator_item,
                "runner": runner_item,
                "environment": generator_environment,
                "environment_id": environment_id,
                "closure": generator_closure,
                "closure_sha256": closure_sha,
            }],
        }
        passed = GATE.verify_numerical_evidence(
            ROOT, archive, summary, baseline_id, baseline,
            {"path": str(model), "size": 5, "sample_sha256": model_sample}, "C",
            {"commit": "0" * 40}, {"id": "test"},
            {"changed": True, "id": "corrected-v2"})
        assert passed["passed"] is True
        baseline["reference"]["oracle_numerical"] = {
            "definition_id": "corrected-v2", "files": oracle_files,
            "manifest_sha256": digest(oracle_dir / "manifest"),
        }
        reused = GATE.verify_numerical_evidence(
            ROOT, archive, summary, baseline_id, baseline,
            {"path": str(model), "size": 5, "sample_sha256": model_sample}, "C",
            {"commit": "0" * 40}, {"id": "test"},
            {"changed": False, "id": "corrected-v2"})
        assert reused["passed"] is True
        try:
            GATE.verify_numerical_evidence(
                ROOT, archive, summary, baseline_id, baseline,
                {"path": str(model), "size": 5, "sample_sha256": model_sample}, "B",
                {"commit": "0" * 40}, {"id": "test"},
                {"changed": False, "id": ""})
        except GATE.GateError:
            pass
        else:
            raise AssertionError("lane B accepted a canonical oracle in place of its predecessor")
    print("test_lane_c_oracle_gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
