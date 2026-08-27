#!/usr/bin/env python3
"""Exercise append-only baseline genesis and its evidence bindings."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
GATE = REPO / "scripts" / "candidate-gate.py"
GLM_TP_LAYOUT = {
    "kind": "q4k-ffn-intermediate",
    "intermediate_size": 2048,
    "shards": [1024, 1024],
    "expert_count": 288,
    "experts_used": 8,
    "reduction": {
        "op": "sum", "scope": "all-ranks", "count": 42,
        "width": 4096, "dtype": "f32",
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sampled(path: Path) -> tuple[int, str]:
    size = path.stat().st_size
    result = hashlib.sha256(f"{size}\n".encode())
    with path.open("rb") as stream:
        for offset in (0, max(0, size // 2 - 4 * 1024 * 1024),
                       max(0, size - 8 * 1024 * 1024)):
            stream.seek(offset)
            result.update(stream.read(8 * 1024 * 1024))
    return size, result.hexdigest()


def write_manifest(path: Path, values: dict[str, object]) -> None:
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()))


def build_fixture(root: Path, structured: bool = False) -> Path:
    model = root / "model.gguf"
    model.write_bytes(b"synthetic model")
    model_size, model_sample = sampled(model)
    model_sha = digest(model)
    source_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    token_file = root / "frozen.tokens"
    token_file.write_text("1\n" * 300)
    token_sha = digest(token_file)
    prompt_sha = "a" * 64
    toolchain = "fixture-rocm-7.14"
    fnv = "1234567890abcdef"

    benchmarks: list[dict[str, str]] = []
    for provider in ("roce-v2", "odinlink"):
        for index in range(3):
            name = f"{provider}-{index + 1}"
            csv_path = root / f"{name}.csv"
            with csv_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "ctx_tokens", "prefill_tokens", "prefill_tps", "gen_tokens",
                    "gen_tps", "gen_first_ms", "gen_steady_tokens",
                    "gen_steady_tps", "kvcache_bytes", "gen_cycles",
                    "gen_token_fnv64",
                ])
                writer.writeheader()
                writer.writerow({
                    "ctx_tokens": 2048, "prefill_tokens": 2048,
                    "prefill_tps": 200 + index, "gen_tokens": 300,
                    "gen_tps": 19 + index / 100, "gen_first_ms": 60,
                    "gen_steady_tokens": 299, "gen_steady_tps": 19,
                    "kvcache_bytes": 0, "gen_cycles": 300,
                    "gen_token_fnv64": fnv,
                })
            manifest = csv_path.with_suffix(".manifest")
            manifest_values = {
                "tag": name, "run_id": f"{name}-run-id",
                "source_commit": source_commit, "source_dirty": 0,
                "model_size": model_size, "model_sample_sha256": model_sample,
                "toolchain_id": toolchain, "prompt_sha256": prompt_sha,
                "frontier": 2048, "generated_tokens": 300, "context": 4096,
                "prefill_chunk": 2048, "dspark": 0, "rdma_profile": provider,
                "ds4_sha256": "b" * 64, "peer_ds4_sha256": "b" * 64,
            }
            if structured:
                manifest_values.update({
                    "tp_weight_layout": "q4k-ffn-intermediate",
                    "tp_intermediate_size": "2048",
                    "tp_intermediate_shards": "1024/1024",
                    "tp_expert_count": "288", "tp_experts_used": "8",
                    "tp_reduce_op": "sum", "tp_reduce_scope": "all-ranks",
                    "tp_reduce_count": "42", "tp_reduce_width": "4096",
                    "tp_reduce_dtype": "f32",
                })
            write_manifest(manifest, manifest_values)
            benchmarks.append({
                "path": str(csv_path), "sha256": digest(csv_path),
                "manifest_sha256": digest(manifest),
            })

    numerical_dir = root / "numerical"
    numerical_dir.mkdir()
    numerical_files = []
    for index in range(300):
        path = numerical_dir / f"decode_{index:06d}.logits.json"
        value = {
            "source": "ds4-bench-frozen-teacher", "model": str(model),
            "backend": "rocm", "quality": False, "dspark": False,
            "dspark_strict": False, "quant_bits": 4,
            "prefix_tokens": 2048, "decode_step": index,
            "position": 2048 + index, "vocab": 4, "teacher_token": 3,
            "teacher_logit": 3.0, "argmax_id": 3, "argmax_logit": 3.0,
            "runner_up_id": 2, "runner_up_logit": 2.0,
            "top1_margin": 1.0, "teacher_gap": 0.0,
            "logits": [0.0, 1.0, 2.0, 3.0],
        }
        path.write_text(json.dumps(value))
        numerical_files.append({"name": path.name, "sha256": digest(path)})
    numerical_manifest = numerical_dir / "manifest"
    write_manifest(numerical_manifest, {
        "model_size": model_size, "model_sample_sha256": model_sample,
        "source_commit": source_commit, "source_dirty": 0,
        "toolchain_id": toolchain, "prefix_tokens": 2048,
        "file_count": 300, "frozen_token_sha256": token_sha, "dspark": 0,
    })

    quality_path = root / "quality.tsv"
    fields = ["id", "target_tokens", "nll", "avg_nll", "api_top1_count",
              "api_top1_match", "api_pair_total", "api_pair_agree"]
    with quality_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index in range(100):
            writer.writerow({
                "id": f"case-{index}", "target_tokens": 24, "nll": 12,
                "avg_nll": 0.5, "api_top1_count": 24,
                "api_top1_match": 22, "api_pair_total": 24,
                "api_pair_agree": 22,
            })
    quality_manifest = root / "quality.manifest"
    write_manifest(quality_manifest, {
        "model_size": model_size, "model_sample_sha256": model_sample,
        "source_commit": source_commit, "source_dirty": 0, "dspark": 0,
    })
    reviews = []
    for kind in ("fable-review", "grok-review"):
        path = root / f"{kind}.md"
        path.write_text("GO\n")
        reviews.append({"kind": kind, "path": str(path), "sha256": digest(path)})

    numerical_thresholds = {
        "e_bound": 0.1, "max_abs": 0.1, "p99_abs": 0.1,
        "nmse": 0.001, "tvd": 0.001, "kl": 0.001,
        "min_top5_overlap": 4, "min_top20_overlap": 4,
        "min_teacher_steps": 300, "allow_quality_difference": False,
    }
    quality_thresholds = {
        "min_cases": 100, "min_target_tokens": 2289,
        "max_mean_nll_delta": 0.0, "max_ci95_high_nll_delta": 0.02,
        "min_api_top1_rate_delta": 0.0, "min_api_pair_rate_delta": 0.0,
    }
    quality_ref = {"sha256": digest(quality_path),
                   "manifest_sha256": digest(quality_manifest)}
    record = {
        "schema_version": 2 if structured else 1,
        "kind": "ds4-numerical-baseline",
        "key": {
            "model_sample_sha256": model_sample, "model_sha256": model_sha,
            "model_size": model_size, "quantization": "Q4_K",
            "source_commit": source_commit, "toolchain_id": toolchain,
            "architecture": "gfx1151", "tp_degree": 2,
            "decode_mode": "ordinary-greedy",
            "workload_id": "ds4-bench-tp-2048x300",
            "workload": {
                "prompt_sha256": prompt_sha, "frontier": "2048",
                "generated_tokens": "300", "context": "4096",
                "prefill_chunk": "2048", "dspark": "0",
                "frozen_token_sha256": token_sha,
            },
            "rdma_providers": ["odinlink", "roce-v2"],
        },
        "reference": {
            "fnv64": fnv,
            "numerical": {"files": numerical_files,
                          "manifest_sha256": digest(numerical_manifest)},
            "quality": quality_ref, "quality_anchor": quality_ref,
        },
        "thresholds": {"numerical": numerical_thresholds,
                       "quality": quality_thresholds},
    }
    if structured:
        record["key"]["tp_layout"] = GLM_TP_LAYOUT
    else:
        record["key"]["expert_split"] = "128/128"
    genesis = {
        "schema_version": 1, "kind": "ds4-baseline-genesis",
        "genesis_id": "fixture-rocm714", "rationale": "test fixture",
        "record": record,
        "artifacts": {
            "model_path": str(model), "benchmarks": benchmarks,
            "frozen_token_file": str(token_file),
            "numerical_dir": str(numerical_dir),
            "quality_tsv": str(quality_path),
            "quality_manifest": str(quality_manifest),
        },
        "evidence": reviews,
    }
    genesis_path = root / "genesis.json"
    genesis_path.write_text(json.dumps(genesis, indent=2, sort_keys=True) + "\n")
    return genesis_path


def run_gate(root: Path, genesis: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["DS4_RESEARCH_ROOT"] = str(root)
    return subprocess.run(
        [sys.executable, str(GATE), "bootstrap-baseline", str(genesis)],
        cwd=REPO, env=environment, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def expect_failure(mutator, expected: str, structured: bool = False) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        genesis = build_fixture(root, structured=structured)
        mutator(root, genesis)
        result = run_gate(root, genesis)
        assert result.returncode != 0
        assert expected in result.stderr, result.stderr


def mutate_genesis(genesis: Path, callback) -> None:
    value = json.loads(genesis.read_text())
    callback(value)
    genesis.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def mutate_benchmark_manifest(genesis: Path, index: int, callback) -> None:
    value = json.loads(genesis.read_text())
    item = value["artifacts"]["benchmarks"][index]
    manifest = Path(item["path"]).with_suffix(".manifest")
    values = {}
    for line in manifest.read_text().splitlines():
        key, separator, content = line.partition("=")
        if separator:
            values[key] = content
    callback(values)
    write_manifest(manifest, values)
    item["manifest_sha256"] = digest(manifest)
    genesis.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        genesis = build_fixture(root)
        result = run_gate(root, genesis)
        if result.returncode != 0:
            raise AssertionError(result.stderr)
        baseline_id = result.stdout.strip()
        assert baseline_id.startswith("sha256:")
        baseline = root / "baselines" / "sha256" / f"{baseline_id[7:]}.json"
        assert baseline.is_file()
        value = json.loads(baseline.read_text())
        assert canonical(value) == baseline_id[7:]
        assert value["provenance"]["lane_origin"] == "bootstrap"
        duplicate = run_gate(root, genesis)
        assert duplicate.returncode != 0
        assert "refusing to overwrite" in duplicate.stderr

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        genesis = build_fixture(root, structured=True)
        result = run_gate(root, genesis)
        if result.returncode != 0:
            raise AssertionError(result.stderr)
        baseline_id = result.stdout.strip()
        baseline = root / "baselines" / "sha256" / f"{baseline_id[7:]}.json"
        value = json.loads(baseline.read_text())
        assert value["schema_version"] == 2
        assert value["key"]["tp_layout"] == GLM_TP_LAYOUT

    expect_failure(
        lambda _root, genesis: mutate_genesis(
            genesis, lambda value: value["record"]["reference"].__setitem__(
                "oracle_numerical", {
                    "definition_id": "planted", "files": [
                        {"name": "fake", "sha256": "f" * 64}
                    ], "manifest_sha256": "e" * 64,
                })),
        "cannot preapprove canonical oracles")
    expect_failure(
        lambda _root, genesis: mutate_genesis(
            genesis, lambda value: value["record"].__setitem__(
                "oracle_generators", [{"id": "planted"}])),
        "cannot preapprove canonical oracles")
    expect_failure(
        lambda _root, genesis: mutate_genesis(
            genesis, lambda value: value["record"]["thresholds"].__setitem__(
                "oracle_numerical", value["record"]["thresholds"]["numerical"])),
        "cannot preapprove canonical oracles")
    expect_failure(
        lambda _root, genesis: mutate_benchmark_manifest(
            genesis, 0,
            lambda values: (values.pop("ds4_sha256"),
                            values.pop("peer_ds4_sha256"))),
        "binaries differed across ranks")
    expect_failure(
        lambda _root, genesis: mutate_benchmark_manifest(
            genesis, 0,
            lambda values: values.__setitem__("peer_ds4_sha256", "c" * 64)),
        "binaries differed across ranks")
    expect_failure(
        lambda _root, genesis: mutate_benchmark_manifest(
            genesis, 0,
            lambda values: (values.__setitem__("ds4_sha256", "c" * 64),
                            values.__setitem__("peer_ds4_sha256", "c" * 64))),
        "benchmarks used different binaries")
    expect_failure(
        lambda _root, genesis: mutate_genesis(
            genesis, lambda value: (
                value["record"]["key"].pop("source_commit"),
                value["record"]["key"].pop("toolchain_id"))),
        "requires source and toolchain identity")
    expect_failure(
        lambda _root, genesis: mutate_genesis(
            genesis, lambda value: value["record"]["key"]["tp_layout"].__setitem__(
                "kind", "q4k-ffn-intermedate")),
        "unsupported kind", structured=True)
    expect_failure(
        lambda _root, genesis: mutate_genesis(
            genesis, lambda value: value["record"]["key"].__setitem__(
                "expert_split", "128/128")),
        "exactly one", structured=True)
    expect_failure(
        lambda _root, genesis: mutate_genesis(
            genesis, lambda value: value["record"]["key"]["tp_layout"].__setitem__(
                "shards", [1024, 1000])),
        "shards must be positive", structured=True)
    expect_failure(
        lambda _root, genesis: mutate_genesis(
            genesis, lambda value: value["record"]["key"]["tp_layout"].__setitem__(
                "experts_used", 289)),
        "invalid expert topology", structured=True)
    expect_failure(
        lambda _root, genesis: mutate_benchmark_manifest(
            genesis, 0, lambda values: values.pop("tp_reduce_width")),
        "differs in tp_reduce_width", structured=True)
    expect_failure(
        lambda _root, genesis: mutate_genesis(
            genesis, lambda value: value["record"].__setitem__(
                "schema_version", 1)),
        "schema v1 requires legacy", structured=True)

    def symlink_escape(root: Path, genesis: Path) -> None:
        value = json.loads(genesis.read_text())
        item = value["record"]["reference"]["numerical"]["files"][0]
        path = Path(value["artifacts"]["numerical_dir"]) / item["name"]
        outside = Path("/etc/hosts")
        path.unlink()
        path.symlink_to(outside)
        item["sha256"] = digest(outside)
        genesis.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    expect_failure(symlink_escape, "numerical file escapes canonical root")

    def tamper_quality(_root: Path, genesis: Path) -> None:
        value = json.loads(genesis.read_text())
        quality = Path(value["artifacts"]["quality_tsv"])
        quality.write_text(quality.read_text() + "tampered\n")
    expect_failure(tamper_quality, "quality binding mismatch")

    print("PASS baseline-genesis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
