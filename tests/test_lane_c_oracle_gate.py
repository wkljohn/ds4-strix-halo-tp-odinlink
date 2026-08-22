#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import subprocess
import tempfile
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
        'source': 'ds4-bench-frozen-teacher', 'model': a.model,
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
        model_sample = "a" * 64
        oracle_files = []
        predecessor_files = []
        for index in range(300):
            base = {
                "source": "ds4-bench-frozen-teacher", "model": str(model),
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
            candidate.write_text(encoded)
            oracle_files.append({"name": name, "sha256": digest(oracle)})
            predecessor_files.append({"name": name, "sha256": "0" * 64})
        file_list_hash = hashlib.sha256("".join(
            f"{item['name']} {item['sha256']}\n" for item in oracle_files
        ).encode()).hexdigest()
        (oracle_dir / "manifest").write_text(
            f"oracle=1\ndefinition_id=corrected-v2\nmodel={model}\nmodel_size=5\n"
            f"model_sample_sha256={model_sample}\nsource_commit={'0' * 40}\n"
            f"source_dirty=0\ngenerator_id=fixture-oracle\n"
            f"generator_path={generator}\ngenerator_sha256={generator_sha}\n"
            "toolchain_id=oracle-test\n"
            f"prefix_tokens=2048\nfile_count=300\nfiles_sha256={file_list_hash}\n"
            f"token_file={tokens}\ntoken_sha256={token_sha}\n")
        (candidate_dir / "manifest").write_text(
            f"model={model}\nmodel_size=5\nmodel_sample_sha256={model_sample}\n"
            f"source_commit={'0' * 40}\nsource_dirty=0\ntoolchain_id=test\n"
            f"prefix_tokens=2048\nfile_count=300\nfrozen_token_sha256={token_sha}\ndspark=0\n")
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
                "frontier": "2048", "frozen_token_sha256": token_sha}},
            "reference": {
                "numerical": {"files": predecessor_files, "manifest_sha256": "0" * 64},
            },
            "thresholds": {"numerical": {key: value for key, value in thresholds.items()
                                            if key != "baseline_id"}},
            "oracle_generators": [{"id": "fixture-oracle", "sha256": generator_sha}],
        }
        passed = GATE.verify_numerical_evidence(
            ROOT, archive, summary, baseline_id, baseline,
            {"path": str(model), "size": 5, "sample_sha256": model_sample}, "C",
            {"commit": "0" * 40}, {"id": "test"},
            {"changed": True, "id": "corrected-v2"})
        assert passed["passed"] is True
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
