#!/usr/bin/env python3
"""Create and validate DS4 performance-candidate promotion dossiers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
LANES = {"A", "B", "C"}
MIN_TEACHER_STEPS = 300
MIN_QUALITY_CASES = 100
MIN_QUALITY_TOKENS = 2289
COMMON_KINDS = {
    "ordinary-benchmark",
    "long-context",
    "transport-proof",
    "ordinary-regression",
    "cross-discipline-long",
    "fable-review",
    "grok-review",
}
LANE_KINDS = {
    "A": {"exact-fingerprint", "rollback-proof"},
    "B": {
        "full-logits",
        "teacher-forced",
        "semantic-retrieval",
        "numerical-envelope",
        "reference-score",
    },
    "C": {
        "full-logits",
        "teacher-forced",
        "semantic-retrieval",
        "numerical-envelope",
        "first-changed-boundary",
        "reference-score",
        "rebaseline",
    },
}
DSPARK_KINDS = {"same-stack-ordinary", "verifier-logits", "dspark-acceptance"}


class GateError(RuntimeError):
    pass


def run_git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolve_scoped_path(repo: Path, root: Path, item: dict) -> Path:
    scope = item.get("scope")
    relative = item.get("path")
    if scope not in {"repo", "research", "system"} or not isinstance(relative, str) or not relative:
        raise GateError("oracle closure entries require repo/research/system scope and a path")
    if scope == "system":
        path = Path(relative)
        if not path.is_absolute():
            raise GateError("system-scoped oracle closure paths must be absolute")
        return path.resolve()
    if Path(relative).is_absolute():
        raise GateError("repo/research oracle closure paths must be relative")
    base = repo if scope == "repo" else root
    path = (base / relative).resolve()
    if path == base or base not in path.parents:
        raise GateError("oracle closure path escapes its declared scope")
    return path


def verify_generator_closure(repo: Path, root: Path,
                             generator: dict) -> tuple[Path, Path]:
    closure = generator.get("closure")
    entrypoint = generator.get("entrypoint")
    runner = generator.get("runner")
    environment = generator.get("environment")
    if (not isinstance(closure, list) or not closure or
            not isinstance(entrypoint, dict) or not isinstance(runner, dict) or
            not isinstance(environment, dict) or not environment or
            not re.fullmatch(r"[0-9a-f]{64}",
                             str(generator.get("environment_id", ""))) or
            generator["environment_id"] != canonical_sha256(environment) or
            generator.get("closure_sha256") != canonical_sha256(closure)):
        raise GateError("approved oracle generator has an invalid dependency closure")
    resolved: dict[tuple[str, str], Path] = {}
    for item in closure:
        if (not isinstance(item, dict) or
                not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))):
            raise GateError("approved oracle closure has an invalid file hash")
        path = resolve_scoped_path(repo, root, item)
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise GateError(f"approved oracle dependency differs: {path}")
        key = (str(item["scope"]), str(item["path"]))
        if key in resolved:
            raise GateError("approved oracle closure contains duplicate paths")
        resolved[key] = path
    entry_key = (str(entrypoint.get("scope")), str(entrypoint.get("path")))
    runner_key = (str(runner.get("scope")), str(runner.get("path")))
    if entry_key not in resolved or entrypoint.get("sha256") != sha256(resolved[entry_key]):
        raise GateError("oracle generator entrypoint is not bound by its closure")
    if runner_key not in resolved or runner.get("sha256") != sha256(resolved[runner_key]):
        raise GateError("oracle generator runner is not bound by its closure")
    return resolved[entry_key], resolved[runner_key]


def validate_numerical_thresholds(thresholds: object, label: str) -> dict:
    if not isinstance(thresholds, dict):
        raise GateError(f"{label} thresholds are missing")
    required = {"e_bound", "max_abs", "p99_abs", "nmse", "tvd", "kl",
                "min_top5_overlap", "min_top20_overlap", "min_teacher_steps"}
    if required - thresholds.keys():
        raise GateError(f"{label} thresholds are incomplete")
    if set(thresholds) - (required | {"allow_quality_difference"}):
        raise GateError(f"{label} thresholds contain unknown keys")
    for key in ("e_bound", "max_abs", "p99_abs", "nmse", "tvd", "kl"):
        value = thresholds[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise GateError(f"{label}.{key} must be finite and nonnegative")
    for key, limit in (("min_top5_overlap", 5), ("min_top20_overlap", 20)):
        if not isinstance(thresholds[key], int) or not 0 <= thresholds[key] <= limit:
            raise GateError(f"{label}.{key} is outside its valid range")
    if (not isinstance(thresholds["min_teacher_steps"], int) or
            thresholds["min_teacher_steps"] < MIN_TEACHER_STEPS):
        raise GateError(f"{label} must require at least {MIN_TEACHER_STEPS} teacher steps")
    if ("allow_quality_difference" in thresholds and
            type(thresholds["allow_quality_difference"]) is not bool):
        raise GateError(f"{label}.allow_quality_difference must be boolean")
    return thresholds


def validate_quality_thresholds(thresholds: object, label: str) -> dict:
    if not isinstance(thresholds, dict):
        raise GateError(f"{label} thresholds are missing")
    required = {"min_cases", "min_target_tokens", "max_mean_nll_delta",
                "max_ci95_high_nll_delta", "min_api_top1_rate_delta",
                "min_api_pair_rate_delta"}
    if required - thresholds.keys():
        raise GateError(f"{label} thresholds are incomplete")
    if set(thresholds) != required:
        raise GateError(f"{label} thresholds contain unknown keys")
    if (not isinstance(thresholds["min_cases"], int) or
            thresholds["min_cases"] < MIN_QUALITY_CASES or
            not isinstance(thresholds["min_target_tokens"], int) or
            thresholds["min_target_tokens"] < MIN_QUALITY_TOKENS):
        raise GateError(f"{label} coverage is below production floors")
    for key in ("max_mean_nll_delta", "max_ci95_high_nll_delta",
                "min_api_top1_rate_delta", "min_api_pair_rate_delta"):
        value = thresholds[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise GateError(f"{label}.{key} must be finite")
    if thresholds["max_ci95_high_nll_delta"] <= 0:
        raise GateError(f"{label}.max_ci95_high_nll_delta must be positive")
    return thresholds


def validate_arithmetic_identity_contract(value: object) -> dict:
    if not isinstance(value, dict):
        raise GateError("arithmetic_identity thresholds are missing")
    if set(value) != {"ignored_env_keys", "ignored_runtime_feature_mask"}:
        raise GateError("arithmetic_identity thresholds contain unknown keys")
    keys = value["ignored_env_keys"]
    mask = value["ignored_runtime_feature_mask"]
    if (not isinstance(keys, list) or len(keys) != len(set(keys)) or
            any(not isinstance(key, str) or not re.fullmatch(
                r"[A-Z_][A-Z0-9_]*", key) for key in keys)):
        raise GateError("arithmetic_identity ignored_env_keys are invalid")
    if not isinstance(mask, int) or not 0 <= mask <= 0xffffffff:
        raise GateError("arithmetic_identity ignored_runtime_feature_mask is invalid")
    return value


def sampled_model_sha256(path: Path) -> tuple[int, str]:
    size = path.stat().st_size
    offsets = (0, max(0, size // 2 - 4 * 1024 * 1024),
               max(0, size - 8 * 1024 * 1024))
    digest = hashlib.sha256(f"{size}\n".encode())
    with path.open("rb") as stream:
        for offset in offsets:
            stream.seek(offset)
            digest.update(stream.read(8 * 1024 * 1024))
    return size, digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def roots() -> tuple[Path, Path]:
    repo = Path(run_git(Path.cwd(), "rev-parse", "--show-toplevel")).resolve()
    default_root = repo.parent / "research-results"
    root = Path(os.environ.get("DS4_RESEARCH_ROOT", default_root)).expanduser()
    if not root.is_absolute():
        raise GateError("DS4_RESEARCH_ROOT must be absolute")
    root = root.resolve()
    if root == repo or repo in root.parents:
        raise GateError("DS4_RESEARCH_ROOT must be outside the Git worktree")
    local = repo / "research-results"
    if local.exists() or local.is_symlink():
        raise GateError(f"forbidden worktree-local research path: {local}")
    if not root.is_dir():
        raise GateError(f"canonical research root is missing: {root}")
    return repo, root


def dossier_path(root: Path, candidate_id: str) -> Path:
    if not ID_RE.fullmatch(candidate_id):
        raise GateError("candidate id must use 1-96 letters, digits, '.', '_' or '-'")
    return root / "candidates" / candidate_id


def load_baseline(root: Path, baseline_id: str) -> tuple[Path, dict]:
    match = re.fullmatch(r"sha256:([0-9a-f]{64})", baseline_id)
    if not match:
        raise GateError("baseline_id must be sha256:<64 lowercase hex digits>")
    digest = match.group(1)
    path = root / "baselines" / "sha256" / f"{digest}.json"
    if not path.is_file():
        raise GateError(f"missing content-addressed baseline: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"invalid baseline JSON: {error}") from error
    if canonical_sha256(value) != digest:
        raise GateError(f"baseline content digest does not match its id: {path}")
    if value.get("schema_version") != 1 or value.get("kind") != "ds4-numerical-baseline":
        raise GateError(f"unsupported baseline schema: {path}")
    for section in ("key", "reference", "thresholds", "provenance"):
        if not isinstance(value.get(section), dict):
            raise GateError(f"baseline is missing {section}: {path}")
    fnv = str(value["reference"].get("fnv64", ""))
    if not re.fullmatch(r"[0-9a-f]{16}", fnv):
        raise GateError(f"baseline has invalid reference fingerprint: {path}")
    numerical = value["reference"].get("numerical")
    quality = value["reference"].get("quality")
    if (not isinstance(numerical, dict) or
            not isinstance(numerical.get("files"), list) or not numerical["files"]):
        raise GateError(f"baseline has no bound numerical reference: {path}")
    for item in numerical["files"]:
        if (not isinstance(item, dict) or not isinstance(item.get("name"), str) or
                not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))):
            raise GateError(f"baseline has invalid numerical reference hashes: {path}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(numerical.get("manifest_sha256", ""))):
        raise GateError(f"baseline has no bound numerical manifest: {path}")
    oracle = value["reference"].get("oracle_numerical")
    if oracle is not None:
        if (not isinstance(oracle, dict) or not isinstance(oracle.get("definition_id"), str) or
                not oracle["definition_id"] or not isinstance(oracle.get("files"), list) or
                not oracle["files"] or not re.fullmatch(
                    r"[0-9a-f]{64}", str(oracle.get("manifest_sha256", "")))):
            raise GateError(f"baseline has an invalid canonical oracle: {path}")
    generators = value.get("oracle_generators", [])
    if not isinstance(generators, list):
        raise GateError(f"baseline oracle_generators must be a list: {path}")
    for generator in generators:
        if (not isinstance(generator, dict) or not isinstance(generator.get("id"), str) or
                not generator["id"] or not isinstance(generator.get("entrypoint"), dict) or
                not isinstance(generator.get("runner"), dict) or
                not isinstance(generator.get("environment"), dict) or
                not re.fullmatch(r"[0-9a-f]{64}",
                                 str(generator.get("environment_id", ""))) or
                not isinstance(generator.get("closure"), list) or not generator["closure"] or
                not re.fullmatch(r"[0-9a-f]{64}",
                                 str(generator.get("closure_sha256", "")))):
            raise GateError(f"baseline has an invalid approved oracle generator: {path}")
        if generator.get("closure_sha256") != canonical_sha256(generator["closure"]):
            raise GateError(f"baseline oracle generator closure digest is invalid: {path}")
        if generator.get("environment_id") != canonical_sha256(generator["environment"]):
            raise GateError(f"baseline oracle generator environment digest is invalid: {path}")
    if (not isinstance(quality, dict) or
            not re.fullmatch(r"[0-9a-f]{64}", str(quality.get("sha256", ""))) or
            not re.fullmatch(r"[0-9a-f]{64}", str(quality.get("manifest_sha256", "")))):
        raise GateError(f"baseline has no bound quality reference: {path}")
    quality_anchor = value["reference"].get("quality_anchor", quality)
    if (not isinstance(quality_anchor, dict) or
            not re.fullmatch(r"[0-9a-f]{64}", str(quality_anchor.get("sha256", ""))) or
            not re.fullmatch(r"[0-9a-f]{64}",
                             str(quality_anchor.get("manifest_sha256", "")))):
        raise GateError(f"baseline has no bound quality anchor: {path}")
    numerical_thresholds = validate_numerical_thresholds(
        value["thresholds"].get("numerical"), "numerical")
    if len(numerical["files"]) < numerical_thresholds["min_teacher_steps"]:
        raise GateError(f"baseline numerical reference is shorter than its threshold: {path}")
    oracle_thresholds = value["thresholds"].get("oracle_numerical")
    if oracle_thresholds is not None:
        validate_numerical_thresholds(oracle_thresholds, "oracle_numerical")
    validate_quality_thresholds(value["thresholds"].get("quality"), "quality")
    validate_arithmetic_identity_contract(
        value["thresholds"].get("arithmetic_identity"))
    return path, value


def verify_review_evidence(root: Path, evidence: object,
                           label: str) -> list[dict]:
    if not isinstance(evidence, list):
        raise GateError(f"{label} requires review evidence")
    review_kinds = set()
    for item in evidence:
        if (not isinstance(item, dict) or item.get("kind") not in
                {"fable-review", "grok-review"} or
                not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))):
            raise GateError(f"{label} has invalid review evidence")
        path = Path(str(item.get("path", ""))).resolve()
        if path != root and root not in path.parents:
            raise GateError(f"{label} review escapes canonical research root")
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise GateError(f"{label} review hash mismatch")
        review_kinds.add(item["kind"])
    if review_kinds != {"fable-review", "grok-review"}:
        raise GateError(f"{label} requires both Fable and Grok reviews")
    return evidence


def bootstrap_baseline(repo: Path, root: Path, genesis_path: Path) -> None:
    """Create the first immutable numerical baseline from recomputable evidence."""
    genesis_path = genesis_path.resolve()
    if genesis_path != root and root not in genesis_path.parents:
        raise GateError("baseline genesis must live in the canonical research root")
    try:
        genesis = json.loads(genesis_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"invalid baseline genesis: {error}") from error
    if (genesis.get("schema_version") != 1 or
            genesis.get("kind") != "ds4-baseline-genesis" or
            not ID_RE.fullmatch(str(genesis.get("genesis_id", ""))) or
            not isinstance(genesis.get("rationale"), str) or
            not genesis["rationale"].strip()):
        raise GateError("invalid baseline genesis identity or rationale")
    reviews = verify_review_evidence(root, genesis.get("evidence"),
                                     "baseline genesis")
    record = genesis.get("record")
    artifacts = genesis.get("artifacts")
    if not isinstance(record, dict) or not isinstance(artifacts, dict):
        raise GateError("baseline genesis requires record and artifacts objects")
    record = json.loads(json.dumps(record))
    if (record.get("schema_version") != 1 or
            record.get("kind") != "ds4-numerical-baseline"):
        raise GateError("baseline genesis record has an unsupported schema")
    for section in ("key", "reference", "thresholds"):
        if not isinstance(record.get(section), dict):
            raise GateError(f"baseline genesis record is missing {section}")

    key = record["key"]
    reference = record["reference"]
    if ("oracle_numerical" in reference or
            "oracle_numerical" in record["thresholds"] or
            record.get("oracle_generators") not in (None, [])):
        raise GateError(
            "baseline genesis cannot preapprove canonical oracles or generators")
    workload = key.get("workload")
    arithmetic_contract = validate_arithmetic_identity_contract(
        record["thresholds"].get("arithmetic_identity"))
    if (not isinstance(workload, dict) or key.get("architecture") != "gfx1151" or
            key.get("tp_degree") != 2 or key.get("expert_split") != "128/128" or
            key.get("decode_mode") != "ordinary-greedy"):
        raise GateError("baseline genesis is not balanced ordinary gfx1151 TP=2")
    if (not re.fullmatch(r"[0-9a-f]{40}", str(key.get("source_commit", ""))) or
            not isinstance(key.get("toolchain_id"), str) or
            not key["toolchain_id"]):
        raise GateError("baseline genesis requires source and toolchain identity")
    fnv = str(reference.get("fnv64", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{16}", fnv):
        raise GateError("baseline genesis has an invalid reference fingerprint")
    providers = key.get("rdma_providers")
    if (not isinstance(providers, list) or
            set(providers) != {"odinlink", "roce-v2"}):
        raise GateError("baseline genesis requires OdinLink and RoCE v2")

    model_path = Path(str(artifacts.get("model_path", "")))
    if not model_path.is_absolute() or not model_path.is_file():
        raise GateError("baseline genesis model path is invalid")
    model_size, model_sample = sampled_model_sha256(model_path)
    if (model_size != key.get("model_size") or
            model_sample != key.get("model_sample_sha256") or
            sha256(model_path) != key.get("model_sha256")):
        raise GateError("baseline genesis model identity mismatch")

    benchmark_items = artifacts.get("benchmarks")
    if not isinstance(benchmark_items, list):
        raise GateError("baseline genesis requires benchmark artifacts")
    provider_counts = {provider: 0 for provider in providers}
    run_ids = set()
    binary_hashes = set()
    benchmark_arithmetic = set()
    for item in benchmark_items:
        if (not isinstance(item, dict) or
                not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))) or
                not re.fullmatch(r"[0-9a-f]{64}",
                                 str(item.get("manifest_sha256", "")))):
            raise GateError("baseline genesis has invalid benchmark binding")
        csv_path = Path(str(item.get("path", ""))).resolve()
        if csv_path != root and root not in csv_path.parents:
            raise GateError("baseline genesis benchmark escapes canonical root")
        if not csv_path.is_file() or sha256(csv_path) != item["sha256"]:
            raise GateError("baseline genesis benchmark hash mismatch")
        row, manifest, manifest_path = read_benchmark(csv_path)
        if sha256(manifest_path) != item["manifest_sha256"]:
            raise GateError("baseline genesis benchmark manifest hash mismatch")
        provider = manifest.get("rdma_profile")
        if provider not in provider_counts:
            raise GateError("baseline genesis benchmark used an undeclared provider")
        provider_counts[provider] += 1
        run_id = manifest.get("run_id")
        if not run_id or run_id in run_ids:
            raise GateError("baseline genesis benchmark run IDs are not distinct")
        run_ids.add(run_id)
        if (row["gen_token_fnv64"].lower() != fnv or
                manifest.get("source_dirty") != "0" or
                manifest.get("source_commit") != key.get("source_commit") or
                manifest.get("model_size") != str(model_size) or
                manifest.get("model_sample_sha256") != model_sample or
                manifest.get("toolchain_id") != key.get("toolchain_id") or
                manifest.get("dspark") != "0"):
            raise GateError("baseline genesis benchmark identity mismatch")
        for field in ("prompt_sha256", "frontier", "generated_tokens",
                      "context", "prefill_chunk", "dspark"):
            if manifest.get(field) != str(workload.get(field, "")):
                raise GateError(f"baseline genesis benchmark differs in {field}")
        local_binary = str(manifest.get("ds4_sha256", ""))
        peer_binary = str(manifest.get("peer_ds4_sha256", ""))
        if (not re.fullmatch(r"[0-9a-f]{64}", local_binary) or
                not re.fullmatch(r"[0-9a-f]{64}", peer_binary) or
                local_binary != peer_binary):
            raise GateError("baseline genesis binaries differed across ranks")
        binary_hashes.add(local_binary)
        benchmark_arithmetic.add(validate_arithmetic_manifest(
            manifest, "baseline genesis benchmark manifest", arithmetic_contract))
    if any(count < 3 for count in provider_counts.values()):
        raise GateError("baseline genesis requires three runs per RDMA provider")
    if len(binary_hashes) != 1:
        raise GateError("baseline genesis benchmarks used different binaries")
    if len(benchmark_arithmetic) != 1:
        raise GateError("baseline genesis benchmarks used different arithmetic configurations")

    token_path = Path(str(artifacts.get("frozen_token_file", ""))).resolve()
    if token_path != root and root not in token_path.parents:
        raise GateError("baseline genesis frozen-token file escapes canonical root")
    if (not token_path.is_file() or
            sha256(token_path) != workload.get("frozen_token_sha256")):
        raise GateError("baseline genesis frozen-token identity mismatch")

    numerical_dir = Path(str(artifacts.get("numerical_dir", ""))).resolve()
    if numerical_dir != root and root not in numerical_dir.parents:
        raise GateError("baseline genesis numerical directory escapes canonical root")
    numerical = reference.get("numerical")
    if not isinstance(numerical, dict) or not isinstance(numerical.get("files"), list):
        raise GateError("baseline genesis numerical reference is missing")
    numerical_manifest = numerical_dir / "manifest"
    if (not numerical_manifest.is_file() or
            sha256(numerical_manifest) != numerical.get("manifest_sha256")):
        raise GateError("baseline genesis numerical manifest mismatch")
    numerical_values = read_manifest(numerical_manifest)
    validate_numerical_geometry(
        numerical_values, workload, "baseline genesis numerical manifest")
    numerical_arithmetic = validate_arithmetic_manifest(
        numerical_values, "baseline genesis numerical manifest", arithmetic_contract)
    if (numerical_values.get("model_size") != str(model_size) or
            numerical_values.get("model_sample_sha256") != model_sample or
            numerical_values.get("source_commit") != key.get("source_commit") or
            numerical_values.get("source_dirty") != "0" or
            numerical_values.get("toolchain_id") != key.get("toolchain_id") or
            numerical_values.get("prefix_tokens") != str(workload.get("frontier")) or
            numerical_values.get("frozen_token_sha256") !=
            workload.get("frozen_token_sha256") or
            numerical_values.get("dspark") != "0"):
        raise GateError("baseline genesis numerical manifest identity mismatch")
    if benchmark_arithmetic != {numerical_arithmetic}:
        raise GateError("baseline genesis numerical and timed arithmetic differ")
    bound_names = set()
    for item in numerical["files"]:
        if (not isinstance(item, dict) or not isinstance(item.get("name"), str) or
                Path(item["name"]).name != item["name"] or
                not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))):
            raise GateError("baseline genesis has invalid numerical file binding")
        path = (numerical_dir / item["name"]).resolve()
        if (path != root and root not in path.parents) or not path.is_file():
            raise GateError("baseline genesis numerical file escapes canonical root")
        if sha256(path) != item["sha256"]:
            raise GateError("baseline genesis numerical file hash mismatch")
        bound_names.add(item["name"])
    if (len(bound_names) != len(numerical["files"]) or
            numerical_values.get("file_count") != str(len(bound_names))):
        raise GateError("baseline genesis numerical file count mismatch")

    quality_path = Path(str(artifacts.get("quality_tsv", ""))).resolve()
    quality_manifest = Path(str(artifacts.get("quality_manifest", ""))).resolve()
    for path in (quality_path, quality_manifest):
        if path != root and root not in path.parents:
            raise GateError("baseline genesis quality artifact escapes canonical root")
    quality = reference.get("quality")
    if (not isinstance(quality, dict) or not quality_path.is_file() or
            not quality_manifest.is_file() or sha256(quality_path) != quality.get("sha256") or
            sha256(quality_manifest) != quality.get("manifest_sha256")):
        raise GateError("baseline genesis quality binding mismatch")
    quality_values = read_manifest(quality_manifest)
    if (quality_values.get("model_size") != str(model_size) or
            quality_values.get("model_sample_sha256") != model_sample or
            quality_values.get("source_commit") != key.get("source_commit") or
            quality_values.get("source_dirty") != "0" or
            quality_values.get("dspark") != "0"):
        raise GateError("baseline genesis quality manifest identity mismatch")
    try:
        with quality_path.open(newline="", encoding="utf-8") as stream:
            quality_rows = list(csv.DictReader(stream, delimiter="\t"))
        target_tokens = sum(int(row["target_tokens"]) for row in quality_rows)
    except (OSError, KeyError, ValueError) as error:
        raise GateError(f"baseline genesis quality table is invalid: {error}") from error

    numerical_thresholds = validate_numerical_thresholds(
        record["thresholds"].get("numerical"), "numerical")
    quality_thresholds = validate_quality_thresholds(
        record["thresholds"].get("quality"), "quality")
    if len(bound_names) < numerical_thresholds["min_teacher_steps"]:
        raise GateError("baseline genesis numerical coverage is too small")
    if (len(quality_rows) < quality_thresholds["min_cases"] or
            target_tokens < quality_thresholds["min_target_tokens"]):
        raise GateError("baseline genesis quality coverage is too small")
    # Reopen the raw numerical and quality artifacts with the same independent
    # comparators used by candidate promotion.  A list of valid hashes is not
    # enough: malformed or incomplete self-authored payloads must not become
    # the immutable truth anchor merely because their bytes were bound.
    with tempfile.TemporaryDirectory(prefix="baseline-genesis-", dir=root) as temporary:
        temporary_path = Path(temporary)
        numerical_threshold_path = temporary_path / "numerical-thresholds.json"
        quality_threshold_path = temporary_path / "quality-thresholds.json"
        atomic_json(numerical_threshold_path,
                    {"baseline_id": "GENESIS", **numerical_thresholds})
        atomic_json(quality_threshold_path,
                    {"baseline_id": "GENESIS", **quality_thresholds})
        numerical_result = rerun_json_tool(
            [sys.executable, str(Path(__file__).with_name("compare-teacher-logits.py")),
             str(numerical_dir), str(numerical_dir),
             "--thresholds", str(numerical_threshold_path)],
            "baseline genesis numerical self-check")
        quality_result = rerun_json_tool(
            [sys.executable, str(Path(__file__).with_name("compare-quality-scores.py")),
             str(quality_path), str(quality_path),
             "--thresholds", str(quality_threshold_path)],
            "baseline genesis quality self-check")
    if (numerical_result.get("passed") is not True or
            numerical_result.get("steps") != len(bound_names)):
        raise GateError("baseline genesis numerical self-check did not pass")
    quality_metrics = quality_result.get("metrics", {})
    if (quality_result.get("passed") is not True or
            quality_metrics.get("cases") != len(quality_rows) or
            quality_metrics.get("target_tokens") != target_tokens):
        raise GateError("baseline genesis quality self-check did not pass")
    if "quality_anchor" not in reference:
        reference["quality_anchor"] = quality
    if reference["quality_anchor"] != quality:
        raise GateError("baseline genesis quality anchor must be its own reference")
    record["oracle_generators"] = []
    record["provenance"] = {
        "genesis_id": genesis["genesis_id"],
        "lane_origin": "bootstrap",
        "rationale": genesis["rationale"],
        "genesis_json_sha256": sha256(genesis_path),
        "evidence": reviews,
        "benchmarks": benchmark_items,
    }
    digest = canonical_sha256(record)
    baseline_path = root / "baselines" / "sha256" / f"{digest}.json"
    if baseline_path.exists():
        raise GateError(f"refusing to overwrite existing baseline: {baseline_path}")
    atomic_json(baseline_path, record)
    try:
        load_baseline(root, f"sha256:{digest}")
    except Exception:
        baseline_path.unlink(missing_ok=True)
        raise
    print(f"sha256:{digest}")


def init_candidate(repo: Path, root: Path, candidate_id: str, lane: str) -> None:
    lane = lane.upper()
    if lane not in LANES:
        raise GateError("lane must be A, B, or C")
    dossier = dossier_path(root, candidate_id)
    if dossier.exists():
        raise GateError(f"candidate dossier already exists: {dossier}")
    status = run_git(repo, "status", "--porcelain=v1", "-uall")
    value = {
        "schema_version": 2,
        "candidate_id": candidate_id,
        "lane": lane,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": str(repo),
            "branch": run_git(repo, "branch", "--show-current") or "DETACHED",
            "commit": run_git(repo, "rev-parse", "HEAD"),
            "dirty": bool(status),
        },
        "baseline_id": "",
        "model": {},
        "toolchain": {},
        "workload": {
            "architecture": "gfx1151",
            "tp_degree": 2,
            "expert_split": "128/128",
            "decode_mode": "ordinary-greedy",
            "workload_id": "ds4-bench-tp-2048x300",
        },
        "transport": {"providers": []},
        "target_definition": {"changed": False, "id": ""},
        "dspark": False,
        "evidence": [],
        "notes": "",
    }
    atomic_json(dossier / "candidate.json", value)
    print(dossier / "candidate.json")


def load_candidate(root: Path, candidate_id: str) -> tuple[Path, dict]:
    dossier = dossier_path(root, candidate_id)
    path = dossier / "candidate.json"
    if not path.is_file():
        raise GateError(f"missing candidate dossier: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"invalid candidate JSON: {error}") from error
    return dossier, value


def read_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise GateError(f"cannot read benchmark manifest {path}: {error}") from error
    for line in lines:
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


ARITHMETIC_MANIFEST_FIELDS = (
    "common_env", "worker_env", "coordinator_env", "extra_env",
    "tp_runtime_features",
)


def validate_arithmetic_manifest(values: dict[str, str], label: str,
                                 contract: dict) -> tuple[object, ...]:
    """Return a normalized identity for the arithmetic actually selected."""
    for field in ARITHMETIC_MANIFEST_FIELDS:
        if field not in values:
            raise GateError(f"{label} is missing arithmetic provenance: {field}")
    if "transport_library_path" not in values:
        raise GateError(f"{label} is missing transport library provenance")
    for field in ("bench_config_sha256", "ds4_sha256", "peer_ds4_sha256",
                  "ds4_bench_tp_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(values.get(field, ""))):
            raise GateError(f"{label} has invalid provenance hash: {field}")
    if values["ds4_sha256"] != values["peer_ds4_sha256"]:
        raise GateError(f"{label} used different engine binaries across ranks")
    if not re.fullmatch(r"0x[0-9a-f]{8}", values["tp_runtime_features"]):
        raise GateError(f"{label} has invalid negotiated TP runtime features")
    if not values.get("run_id"):
        raise GateError(f"{label} has no run identity")
    ignored = set(contract["ignored_env_keys"])
    normalized_env: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for field in ARITHMETIC_MANIFEST_FIELDS[:-1]:
        selected: dict[str, str] = {}
        try:
            words = shlex.split(values[field], posix=True)
        except ValueError as error:
            raise GateError(f"{label} has malformed {field}: {error}") from error
        for word in words:
            if not word:
                continue
            key, separator, setting = word.partition("=")
            if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise GateError(f"{label} has malformed {field} entry: {word}")
            if key in ignored:
                continue
            if key == "LD_LIBRARY_PATH" and values["transport_library_path"]:
                transport_components = set(
                    values["transport_library_path"].split(":"))
                setting = ":".join(
                    component for component in setting.split(":")
                    if component not in transport_components)
                if not setting:
                    continue
            selected[key] = setting
        normalized_env.append((field, tuple(sorted(selected.items()))))
    feature_mask = int(values["tp_runtime_features"], 16)
    feature_mask &= ~contract["ignored_runtime_feature_mask"] & 0xffffffff
    return (
        values["bench_config_sha256"], values["ds4_sha256"],
        values["peer_ds4_sha256"], values["ds4_bench_tp_sha256"],
        feature_mask, *normalized_env,
    )


def validate_numerical_geometry(values: dict[str, str], workload: dict,
                                label: str) -> None:
    expected = {
        "prefix_tokens": str(workload.get("frontier", "")),
        "frontier": str(workload.get("frontier", "")),
        "context": str(workload.get("context", "")),
        "prefill_chunk": str(workload.get("prefill_chunk", "")),
    }
    for field, wanted in expected.items():
        if not wanted or values.get(field) != wanted:
            raise GateError(f"{label} differs from baseline workload: {field}")


def read_benchmark(path: Path) -> tuple[dict[str, str], dict[str, str], Path]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as error:
        raise GateError(f"cannot read benchmark CSV {path}: {error}") from error
    if len(rows) != 1:
        raise GateError(f"candidate benchmark must contain one row: {path}")
    row = rows[0]
    fnv = str(row.get("gen_token_fnv64", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{16}", fnv):
        raise GateError(f"candidate benchmark has invalid fingerprint: {path}")
    try:
        if int(row.get("gen_tokens", "0")) < 300:
            raise GateError(f"candidate benchmark generated fewer than 300 tokens: {path}")
    except ValueError as error:
        raise GateError(f"candidate benchmark has invalid gen_tokens: {path}") from error
    manifest_path = path.with_suffix(".manifest")
    manifest = read_manifest(manifest_path)
    return row, manifest, manifest_path


def threshold_file(value: dict, baseline_id: str, section: str, directory: Path) -> Path:
    thresholds = value["thresholds"].get(section)
    if not isinstance(thresholds, dict):
        raise GateError(f"baseline has no {section} thresholds")
    payload = {"baseline_id": baseline_id, **thresholds}
    path = directory / f"{section}-thresholds.json"
    atomic_json(path, payload)
    return path


def rerun_json_tool(command: list[str], label: str) -> dict:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise GateError(f"{label} failed independent recomputation: {detail}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise GateError(f"{label} returned invalid JSON") from error


def verify_numerical_evidence(repo: Path, root: Path, summary_path: Path,
                              baseline_id: str, baseline: dict,
                              candidate_model: dict, lane: str,
                              source: dict, toolchain: dict,
                              target_definition: dict) -> dict:
    try:
        recorded = json.loads(summary_path.read_text(encoding="utf-8"))
        sources = recorded["sources"]
        reference_dir = Path(sources["reference_dir"]).resolve()
        candidate_dir = Path(sources["candidate_dir"]).resolve()
        reference_manifest = Path(sources["reference_manifest"]).resolve()
        candidate_manifest = Path(sources["candidate_manifest"]).resolve()
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise GateError(f"invalid numerical-envelope summary: {error}") from error
    for path in (reference_dir, candidate_dir, reference_manifest, candidate_manifest):
        if path != root and root not in path.parents:
            raise GateError(f"numerical source escapes canonical research root: {path}")
    if reference_dir == candidate_dir:
        raise GateError("numerical reference and candidate directories must differ")
    definition_changed = lane == "C" and target_definition.get("changed") is True
    threshold_section = "oracle_numerical" if lane == "C" else "numerical"
    with tempfile.TemporaryDirectory() as raw:
        threshold = threshold_file(baseline, baseline_id, threshold_section, Path(raw))
        command = [
            sys.executable, str(repo / "scripts" / "compare-teacher-logits.py"),
            str(reference_dir), str(candidate_dir), "--thresholds", str(threshold),
        ]
        allow_quality_difference = baseline["thresholds"][threshold_section].get(
            "allow_quality_difference", False)
        if recorded.get("allow_quality_difference") is not allow_quality_difference:
            raise GateError("numerical comparison mode differs from the baseline contract")
        if allow_quality_difference:
            command.append("--allow-quality-difference")
        recomputed = rerun_json_tool(command, "numerical envelope")
    ignored = {"thresholds_sha256"}
    if {key: value for key, value in recorded.items() if key not in ignored} != {
            key: value for key, value in recomputed.items() if key not in ignored}:
        raise GateError("numerical-envelope summary does not match its raw logits")
    reference_hashes = [
        {"name": item.get("name"), "sha256": item.get("reference_sha256")}
        for item in recorded.get("sources", {}).get("pairs", [])
    ]
    if definition_changed:
        oracle_identity = read_manifest(reference_manifest)
        approved_generators = {
            item["id"]: item for item in baseline.get("oracle_generators", [])
        }
        generator_id = oracle_identity.get("generator_id")
        approved = approved_generators.get(generator_id)
        if not isinstance(approved, dict):
            raise GateError("Lane C oracle generator is not approved by the predecessor")
        generator_path, runner_path = verify_generator_closure(repo, root, approved)
        file_list_hash = hashlib.sha256("".join(
            f"{item['name']} {item['sha256']}\n" for item in reference_hashes
        ).encode()).hexdigest()
        token_file = Path(oracle_identity.get("token_file", "")).resolve()
        if token_file != root and root not in token_file.parents:
            raise GateError("Lane C oracle token file escapes the research root")
        try:
            token_sha256 = sha256(token_file)
        except OSError as error:
            raise GateError(f"cannot hash Lane C oracle token file: {error}") from error
        if (oracle_identity.get("oracle") != "1" or
                oracle_identity.get("definition_id") != target_definition.get("id") or
                oracle_identity.get("model") != candidate_model["path"] or
                oracle_identity.get("model_size") != str(candidate_model["size"]) or
                oracle_identity.get("model_sample_sha256") != candidate_model["sample_sha256"] or
                oracle_identity.get("generator_closure_sha256") !=
                    approved["closure_sha256"] or
                oracle_identity.get("generator_environment_id") !=
                    approved["environment_id"] or
                not oracle_identity.get("toolchain_id") or
                oracle_identity.get("prefix_tokens") != str(
                    baseline["key"]["workload"]["frontier"]) or
                oracle_identity.get("file_count") != str(len(reference_hashes)) or
                oracle_identity.get("files_sha256") != file_list_hash):
            raise GateError("new Lane C oracle manifest is not a bound canonical producer")
        if (token_sha256 != oracle_identity.get("token_sha256") or
                token_sha256 != baseline["key"]["workload"].get("frozen_token_sha256")):
            raise GateError("Lane C oracle token sequence differs from the frozen workload")
        with tempfile.TemporaryDirectory() as generated_raw:
            generated_dir = Path(generated_raw)
            command = ([str(generator_path)] if runner_path == generator_path else
                       [str(runner_path), str(generator_path)])
            command.extend([
                "--model", candidate_model["path"], "--definition", target_definition["id"],
                "--prefix", str(baseline["key"]["workload"]["frontier"]),
                "--token-file", str(token_file), "--output-dir", str(generated_dir),
            ])
            try:
                result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
            except OSError as error:
                raise GateError(f"cannot execute approved Lane C oracle: {error}") from error
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip()
                raise GateError(f"approved Lane C oracle generator failed: {detail}")
            regenerated = [
                {"name": path.name, "sha256": sha256(path)}
                for path in sorted(generated_dir.glob("decode_*.logits.json"))
            ]
        if regenerated != reference_hashes:
            raise GateError("approved Lane C generator did not reproduce the oracle dumps")
    else:
        reference_section = (baseline["reference"].get("oracle_numerical")
                             if lane == "C" else baseline["reference"].get("numerical"))
        if not isinstance(reference_section, dict) or not isinstance(reference_section.get("files"), list):
            raise GateError(f"lane {lane} baseline has no bound numerical oracle")
        if lane == "C" and target_definition.get("id") != reference_section.get("definition_id"):
            raise GateError("Lane C target definition differs without declaring a new oracle")
        if reference_hashes != reference_section["files"]:
            label = "canonical oracle" if lane == "C" else "predecessor baseline"
            raise GateError(f"numerical reference is not the bound {label} artifact")
        if recorded["sources"].get("reference_manifest_sha256") != reference_section.get("manifest_sha256"):
            raise GateError("numerical reference manifest is not bound by the baseline")
    expected_bits = {"Q4_K": 4, "Q2_K": 2}.get(baseline["key"].get("quantization"))
    try:
        expected_prefix = int(baseline["key"]["workload"]["frontier"])
    except (KeyError, TypeError, ValueError) as error:
        raise GateError("baseline has an invalid numerical frontier") from error
    for item in recorded["sources"]["pairs"]:
        reference_dump = json.loads((reference_dir / item["name"]).read_text(encoding="utf-8"))
        candidate_dump = json.loads((candidate_dir / item["name"]).read_text(encoding="utf-8"))
        expected_reference_source = (
            "ds4-canonical-oracle" if lane == "C"
            else "ds4-bench-frozen-teacher")
        if reference_dump.get("source") != expected_reference_source:
            raise GateError("reference logit dump has an unexpected producer schema")
        if candidate_dump.get("source") != "ds4-bench-frozen-teacher":
            raise GateError("candidate logit dump has an unexpected producer schema")
        for dump in (reference_dump, candidate_dump):
            if expected_bits is not None and dump.get("quant_bits") != expected_bits:
                raise GateError("teacher-logit quantization differs from the baseline model")
            if dump.get("prefix_tokens") != expected_prefix:
                raise GateError("teacher-logit prefix differs from the baseline workload")
            if dump.get("dspark") is not False:
                raise GateError("ordinary numerical promotion cannot use DSpark logits")
        if reference_dump.get("model") != candidate_model["path"]:
            raise GateError("reference teacher logits came from a different model path")
        if candidate_dump.get("model") != candidate_model["path"]:
            raise GateError("candidate teacher logits came from a different model path")
    candidate_identity = read_manifest(candidate_manifest)
    validate_numerical_geometry(
        candidate_identity, baseline["key"]["workload"],
        "numerical candidate manifest")
    candidate_arithmetic = validate_arithmetic_manifest(
        candidate_identity, "numerical candidate manifest",
        baseline["thresholds"]["arithmetic_identity"])
    if (candidate_identity.get("model") != candidate_model["path"] or
            candidate_identity.get("model_size") != str(candidate_model["size"]) or
            candidate_identity.get("model_sample_sha256") != candidate_model["sample_sha256"] or
            candidate_identity.get("source_commit") != source["commit"] or
            candidate_identity.get("source_dirty") != "0" or
            candidate_identity.get("toolchain_id") != toolchain["id"] or
            candidate_identity.get("prefix_tokens") != str(expected_prefix) or
            candidate_identity.get("file_count") != str(len(recorded["sources"]["pairs"])) or
            candidate_identity.get("frozen_token_sha256") !=
                baseline["key"]["workload"].get("frozen_token_sha256") or
            candidate_identity.get("dspark") != "0"):
        raise GateError("numerical candidate manifest does not match the ordinary candidate")
    minimum = baseline["thresholds"][threshold_section]["min_teacher_steps"]
    if (recorded.get("baseline_id") != baseline_id or
            recorded.get("passed") is not True or
            recorded.get("far_margin_inversions") != 0 or
            not isinstance(recorded.get("steps"), int) or
            recorded["steps"] < minimum):
        raise GateError("numerical envelope did not pass the versioned baseline")
    recorded["_candidate_arithmetic_identity"] = candidate_arithmetic
    return recorded


def verify_quality_evidence(repo: Path, root: Path, summary_path: Path,
                            baseline_id: str, baseline: dict,
                            candidate_model: dict, source: dict,
                            reference_section: dict | None = None,
                            label: str = "predecessor") -> dict:
    try:
        recorded = json.loads(summary_path.read_text(encoding="utf-8"))
        sources = recorded["sources"]
        reference = Path(sources["reference"]).resolve()
        candidate = Path(sources["candidate"]).resolve()
        reference_manifest = Path(sources["reference_manifest"]).resolve()
        candidate_manifest = Path(sources["candidate_manifest"]).resolve()
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise GateError(f"invalid reference-score summary: {error}") from error
    for path in (reference, candidate, reference_manifest, candidate_manifest):
        if path != root and root not in path.parents:
            raise GateError(f"quality source escapes canonical research root: {path}")
    if reference == candidate:
        raise GateError("quality reference and candidate TSVs must differ")
    with tempfile.TemporaryDirectory() as raw:
        threshold = threshold_file(baseline, baseline_id, "quality", Path(raw))
        recomputed = rerun_json_tool([
            sys.executable, str(repo / "scripts" / "compare-quality-scores.py"),
            str(reference), str(candidate), "--thresholds", str(threshold),
        ], "quality score")
    ignored = {"thresholds_sha256"}
    if {key: value for key, value in recorded.items() if key not in ignored} != {
            key: value for key, value in recomputed.items() if key not in ignored}:
        raise GateError("reference-score summary does not match its source TSVs")
    if reference_section is None:
        reference_section = baseline["reference"]["quality"]
    if recorded.get("sources", {}).get("reference_sha256") != reference_section["sha256"]:
        raise GateError(f"quality reference is not the {label} artifact")
    if (recorded.get("sources", {}).get("reference_manifest_sha256") !=
            reference_section["manifest_sha256"]):
        raise GateError(f"quality reference manifest is not the {label} artifact")
    candidate_identity = read_manifest(candidate_manifest)
    if (candidate_identity.get("model") != candidate_model["path"] or
            candidate_identity.get("model_size") != str(candidate_model["size"]) or
            candidate_identity.get("model_sample_sha256") != candidate_model["sample_sha256"] or
            candidate_identity.get("source_commit") != source["commit"] or
            candidate_identity.get("source_dirty") != "0" or
            candidate_identity.get("dspark") != "0"):
        raise GateError("quality candidate manifest does not match the ordinary candidate")
    if recorded.get("baseline_id") != baseline_id or recorded.get("passed") is not True:
        raise GateError("reference quality comparison did not pass")
    return recorded


def check_candidate(repo: Path, root: Path, candidate_id: str) -> tuple[Path, dict, dict]:
    dossier, value = load_candidate(root, candidate_id)
    if value.get("schema_version") != 2 or value.get("candidate_id") != candidate_id:
        raise GateError("candidate schema/id mismatch")
    lane = str(value.get("lane", "")).upper()
    if lane not in LANES:
        raise GateError("invalid candidate lane")
    source = value.get("source")
    if not isinstance(source, dict) or not re.fullmatch(r"[0-9a-f]{40}", str(source.get("commit", ""))):
        raise GateError("source.commit must be a full Git object id")
    if source.get("dirty") is not False:
        raise GateError("performance promotion requires a clean source worktree snapshot")
    if run_git(repo, "rev-parse", "HEAD") != source["commit"]:
        raise GateError("gate must run from the candidate source commit")
    if run_git(repo, "status", "--porcelain=v1", "-uall"):
        raise GateError("gate must run from a clean candidate worktree")
    baseline = None
    if lane in {"B", "C"}:
        _, baseline = load_baseline(root, str(value.get("baseline_id", "")))

    evidence = value.get("evidence")
    if not isinstance(evidence, list):
        raise GateError("evidence must be a list")
    kinds: list[str] = []
    diverse_summaries: list[Path] = []
    paths_by_kind: dict[str, list[Path]] = {}
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise GateError(f"evidence[{index}] must be an object")
        kind = str(item.get("kind", ""))
        path_text = str(item.get("path", ""))
        expected = str(item.get("sha256", ""))
        if not kind or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise GateError(f"evidence[{index}] has invalid kind or SHA-256")
        allowed_kinds = (COMMON_KINDS | set().union(*LANE_KINDS.values()) |
                         DSPARK_KINDS | {"candidate-benchmark", "quality-anchor-score"})
        if kind not in allowed_kinds:
            raise GateError(f"evidence[{index}] uses unknown kind: {kind}")
        path = Path(path_text)
        if not path.is_absolute():
            path = dossier / path
        path = path.resolve()
        if path != root and root not in path.parents:
            raise GateError(f"evidence escapes canonical root: {path}")
        if not path.is_file():
            raise GateError(f"missing evidence file: {path}")
        actual = sha256(path)
        if actual != expected:
            raise GateError(f"evidence hash mismatch: {path}")
        if kind == "candidate-benchmark":
            manifest_expected = str(item.get("manifest_sha256", ""))
            manifest = path.with_suffix(".manifest")
            if not re.fullmatch(r"[0-9a-f]{64}", manifest_expected):
                raise GateError(f"candidate benchmark has no manifest hash: {path}")
            if not manifest.is_file() or sha256(manifest) != manifest_expected:
                raise GateError(f"candidate benchmark manifest hash mismatch: {manifest}")
        kinds.append(kind)
        paths_by_kind.setdefault(kind, []).append(path)
        if kind == "cross-discipline-long":
            diverse_summaries.append(path)

    required_kinds = COMMON_KINDS | LANE_KINDS[lane]
    if value.get("dspark") is True:
        required_kinds |= DSPARK_KINDS
    missing_kinds = sorted(required_kinds - set(kinds))
    if missing_kinds:
        raise GateError("missing evidence kinds: " + ", ".join(missing_kinds))
    if kinds.count("candidate-benchmark") != 3:
        raise GateError("exactly three candidate-benchmark artifacts are required")
    if len(diverse_summaries) != 1:
        raise GateError("exactly one cross-disciplinary long summary is required")
    diverse_tool = Path(__file__).with_name("diverse-bench-gate.py")
    diverse_env = os.environ.copy()
    diverse_env["DS4_RESEARCH_ROOT"] = str(root)
    result = subprocess.run(
        [sys.executable, str(diverse_tool), "verify", str(diverse_summaries[0])],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=diverse_env,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise GateError(f"cross-disciplinary long evidence failed: {detail}")
    summary = json.loads(diverse_summaries[0].read_text(encoding="utf-8"))
    if str(summary.get("lane", "")).upper() != lane:
        raise GateError("cross-disciplinary long evidence uses a different lane")

    benchmark_rows = [read_benchmark(path) for path in paths_by_kind["candidate-benchmark"]]
    fingerprints = {row[0]["gen_token_fnv64"].lower() for row in benchmark_rows}
    if len(fingerprints) != 1:
        raise GateError("candidate benchmarks do not have one deterministic fingerprint")
    candidate_fingerprint = next(iter(fingerprints))
    for _, manifest, _ in benchmark_rows:
        if manifest.get("candidate") != "1" or manifest.get("candidate_lane", "A") != lane:
            raise GateError("candidate benchmark manifest has the wrong candidate lane")
        if manifest.get("source_dirty") != "0":
            raise GateError("performance promotion requires a clean source worktree")
        if manifest.get("source_commit") != source["commit"]:
            raise GateError("candidate benchmark source commit differs from dossier")
        if lane in {"B", "C"} and manifest.get("baseline_id") != value["baseline_id"]:
            raise GateError("candidate benchmark uses a different numerical baseline")
    run_ids = {manifest.get("run_id") for _, manifest, _ in benchmark_rows}
    run_tags = {manifest.get("tag") for _, manifest, _ in benchmark_rows}
    if (None in run_ids or "" in run_ids or len(run_ids) != 3 or
            None in run_tags or "" in run_tags or len(run_tags) != 3):
        raise GateError("candidate benchmarks must be three distinct recorded runs")
    if lane == "A":
        expected = {manifest.get("expected_fnv64", "").lower()
                    for _, manifest, _ in benchmark_rows}
        if expected != {candidate_fingerprint}:
            raise GateError("lane A candidate did not reproduce its expected fingerprint")
    else:
        model = value.get("model", {})
        if not isinstance(model, dict):
            raise GateError("candidate model metadata is missing")
        if (not re.fullmatch(r"[0-9a-f]{64}", str(model.get("sample_sha256", ""))) or
                not re.fullmatch(r"[0-9a-f]{64}", str(model.get("sha256", ""))) or
                not isinstance(model.get("size"), int) or model["size"] <= 0 or
                not isinstance(model.get("quantization"), str) or not model["quantization"] or
                not isinstance(model.get("path"), str) or not Path(model["path"]).is_absolute()):
            raise GateError("candidate model identity is incomplete")
        try:
            actual_size, actual_sample = sampled_model_sha256(Path(model["path"]))
        except OSError as error:
            raise GateError(f"cannot verify candidate model artifact: {error}") from error
        if actual_size != model["size"] or actual_sample != model["sample_sha256"]:
            raise GateError("candidate model artifact does not match the dossier identity")
        if sha256(Path(model["path"])) != model["sha256"]:
            raise GateError("candidate model full SHA-256 differs from the dossier identity")
        toolchain = value.get("toolchain", {})
        workload = value.get("workload", {})
        transport = value.get("transport", {})
        if not isinstance(toolchain, dict) or not isinstance(toolchain.get("id"), str) or not toolchain["id"]:
            raise GateError("candidate toolchain.id is required")
        target_definition = value.get("target_definition", {})
        if not isinstance(target_definition, dict):
            raise GateError("candidate target_definition is missing")
        if lane == "C":
            if (type(target_definition.get("changed")) is not bool or
                    not isinstance(target_definition.get("id"), str) or
                    not target_definition["id"]):
                raise GateError("Lane C requires an explicit target definition id")
        if (not isinstance(workload, dict) or workload.get("architecture") != "gfx1151" or
                workload.get("tp_degree") != 2 or workload.get("expert_split") != "128/128" or
                workload.get("decode_mode") != "ordinary-greedy" or
                not isinstance(workload.get("workload_id"), str) or not workload["workload_id"]):
            raise GateError("candidate workload is not the ordinary balanced gfx1151 TP=2 contract")
        providers = transport.get("providers") if isinstance(transport, dict) else None
        if (not isinstance(providers, list) or not providers or
                any(provider not in {"odinlink", "roce-v2"} for provider in providers)):
            raise GateError("candidate transport.providers must name validated RDMA providers")
        if lane == "C" and set(providers) != {"odinlink", "roce-v2"}:
            raise GateError("Lane C requires both OdinLink and RoCE v2 validation")
        for _, manifest, _ in benchmark_rows:
            if (manifest.get("model_sample_sha256") != model.get("sample_sha256") or
                    manifest.get("model_size") != str(model.get("size", "")) or
                    manifest.get("model") != model.get("path")):
                raise GateError("candidate benchmark model differs from dossier")
        baseline_key = baseline["key"]
        if (baseline_key.get("model_sample_sha256") != model.get("sample_sha256") or
                baseline_key.get("model_sha256") != model.get("sha256") or
                baseline_key.get("model_size") != model.get("size") or
                baseline_key.get("quantization") != model.get("quantization")):
            raise GateError("candidate model does not match the predecessor baseline")
        baseline_workload = baseline["key"].get("workload")
        if not isinstance(baseline_workload, dict):
            raise GateError("predecessor baseline has no benchmark workload contract")
        for _, manifest, _ in benchmark_rows:
            for key in ("prompt_sha256", "frontier", "generated_tokens", "context",
                        "prefill_chunk", "dspark"):
                if manifest.get(key) != str(baseline_workload.get(key, "")):
                    raise GateError(f"candidate benchmark differs from baseline workload: {key}")
            if manifest.get("rdma_profile") not in providers:
                raise GateError("candidate benchmark used an undeclared RDMA provider")
        numerical_paths = paths_by_kind.get("numerical-envelope", [])
        quality_paths = paths_by_kind.get("reference-score", [])
        if len(numerical_paths) != 1 or len(quality_paths) != 1:
            raise GateError("lane B/C requires one numerical and one quality summary")
        numerical_result = verify_numerical_evidence(
            repo, root, numerical_paths[0], value["baseline_id"], baseline,
            model, lane, source, toolchain, target_definition)
        timed_arithmetic = {
            validate_arithmetic_manifest(
                manifest, "candidate benchmark manifest",
                baseline["thresholds"]["arithmetic_identity"])
            for _, manifest, _ in benchmark_rows
        }
        numerical_arithmetic = tuple(
            numerical_result.pop("_candidate_arithmetic_identity"))
        if timed_arithmetic != {numerical_arithmetic}:
            raise GateError("candidate numerical and timed arithmetic configurations differ")
        quality_result = verify_quality_evidence(
            repo, root, quality_paths[0], value["baseline_id"], baseline, model, source)
        quality_anchor = baseline["reference"].get(
            "quality_anchor", baseline["reference"]["quality"])
        if quality_anchor == baseline["reference"]["quality"]:
            quality_anchor_result = quality_result
        else:
            anchor_paths = paths_by_kind.get("quality-anchor-score", [])
            if len(anchor_paths) != 1:
                raise GateError("lane B/C requires one cumulative quality-anchor summary")
            quality_anchor_result = verify_quality_evidence(
                repo, root, anchor_paths[0], value["baseline_id"], baseline,
                model, source, quality_anchor, "immutable anchor")

    print(f"PASS candidate={candidate_id} lane={lane} evidence={len(evidence)}")
    return dossier, value, {
        "fingerprint": candidate_fingerprint,
        "baseline": baseline,
        "benchmark_manifests": [manifest for _, manifest, _ in benchmark_rows],
        "numerical_result": numerical_result if lane in {"B", "C"} else None,
        "quality_result": quality_result if lane in {"B", "C"} else None,
        "quality_anchor_result": quality_anchor_result if lane in {"B", "C"} else None,
        "target_definition": value.get("target_definition"),
    }


def promote_candidate(repo: Path, root: Path, candidate_id: str) -> None:
    dossier, value, derived = check_candidate(repo, root, candidate_id)
    if (dossier / "PROMOTED.json").exists():
        raise GateError("candidate was already promoted; promotion is append-only")
    candidate_file = dossier / "candidate.json"
    new_baseline_id = None
    if value["lane"] in {"B", "C"}:
        model = value["model"]
        workload = value["workload"]
        toolchain = value["toolchain"]
        record = {
            "schema_version": 1,
            "kind": "ds4-numerical-baseline",
            "key": {
                "model_sample_sha256": model["sample_sha256"],
                "model_sha256": model["sha256"],
                "model_size": model["size"],
                "quantization": model["quantization"],
                "source_commit": value["source"]["commit"],
                "toolchain_id": toolchain["id"],
                "architecture": workload["architecture"],
                "tp_degree": workload["tp_degree"],
                "expert_split": workload["expert_split"],
                "decode_mode": workload["decode_mode"],
                "workload_id": workload["workload_id"],
                "workload": {
                    key: derived["benchmark_manifests"][0][key]
                    for key in ("prompt_sha256", "frontier", "generated_tokens",
                                "context", "prefill_chunk", "dspark")
                },
                "rdma_providers": sorted(value["transport"]["providers"]),
            },
            "reference": {
                "fnv64": derived["fingerprint"],
                "numerical": {
                    "files": [
                        {"name": item["name"], "sha256": item["candidate_sha256"]}
                        for item in derived["numerical_result"]["sources"]["pairs"]
                    ],
                    "manifest_sha256": derived["numerical_result"]["sources"]["candidate_manifest_sha256"],
                },
                "quality": {
                    "sha256": derived["quality_result"]["sources"]["candidate_sha256"],
                    "manifest_sha256": derived["quality_result"]["sources"]["candidate_manifest_sha256"],
                },
                "quality_anchor": derived["baseline"]["reference"].get(
                    "quality_anchor", derived["baseline"]["reference"]["quality"]),
            },
            "thresholds": derived["baseline"]["thresholds"],
            "provenance": {
                "candidate_id": candidate_id,
                "lane_origin": value["lane"],
                "replaces": value["baseline_id"],
                "candidate_json_sha256": sha256(candidate_file),
                "evidence": value["evidence"],
            },
            "oracle_generators": derived["baseline"].get("oracle_generators", []),
        }
        record["key"]["workload"]["frozen_token_sha256"] = \
            derived["baseline"]["key"]["workload"]["frozen_token_sha256"]
        if value["lane"] == "C" and derived["target_definition"].get("changed") is True:
            record["reference"]["oracle_numerical"] = {
                "definition_id": derived["target_definition"]["id"],
                "files": [
                    {"name": item["name"], "sha256": item["reference_sha256"]}
                    for item in derived["numerical_result"]["sources"]["pairs"]
                ],
                "manifest_sha256": derived["numerical_result"]["sources"]["reference_manifest_sha256"],
            }
        elif derived["baseline"]["reference"].get("oracle_numerical") is not None:
            record["reference"]["oracle_numerical"] = derived["baseline"]["reference"]["oracle_numerical"]
        digest = canonical_sha256(record)
        baseline_path = root / "baselines" / "sha256" / f"{digest}.json"
        if baseline_path.exists():
            raise GateError(f"refusing to overwrite existing baseline: {baseline_path}")
        atomic_json(baseline_path, record)
        new_baseline_id = f"sha256:{digest}"
    promoted = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "lane": value["lane"],
        "source_commit": value["source"]["commit"],
        "candidate_json_sha256": sha256(candidate_file),
        "new_baseline_id": new_baseline_id,
        "promoted_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(dossier / "PROMOTED.json", promoted)
    print(dossier / "PROMOTED.json")


def amend_baseline(repo: Path, root: Path, amendment_path: Path) -> None:
    amendment_path = amendment_path.resolve()
    if amendment_path != root and root not in amendment_path.parents:
        raise GateError("baseline amendment must live in the canonical research root")
    try:
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"invalid baseline amendment: {error}") from error
    if (amendment.get("schema_version") != 1 or
            amendment.get("kind") != "ds4-baseline-amendment" or
            not ID_RE.fullmatch(str(amendment.get("amendment_id", ""))) or
            not isinstance(amendment.get("rationale"), str) or
            not amendment["rationale"].strip()):
        raise GateError("invalid baseline amendment identity or rationale")
    _, predecessor = load_baseline(root, str(amendment.get("baseline_id", "")))
    reviews = verify_review_evidence(root, amendment.get("evidence"),
                                     "baseline amendment")

    record = json.loads(json.dumps(predecessor))
    generator = amendment.get("add_oracle_generator")
    if generator is not None:
        if (not isinstance(generator, dict) or
                not ID_RE.fullmatch(str(generator.get("id", "")))):
            raise GateError("baseline amendment has an invalid oracle generator")
        if any(item.get("id") == generator["id"]
               for item in record.get("oracle_generators", [])):
            raise GateError("oracle generator id already exists in the baseline lineage")
        verify_generator_closure(repo, root, generator)
        record.setdefault("oracle_generators", []).append(generator)

    updates = amendment.get("threshold_updates", {})
    if not isinstance(updates, dict) or any(
            key not in {"numerical", "oracle_numerical", "quality",
                        "arithmetic_identity"} for key in updates):
        raise GateError("baseline amendment has unsupported threshold updates")
    for section, thresholds in updates.items():
        if not isinstance(thresholds, dict):
            raise GateError(f"invalid {section} threshold update")
        if section in {"numerical", "oracle_numerical"}:
            validate_numerical_thresholds(thresholds, section)
            previous = record["thresholds"].get(
                section, record["thresholds"]["numerical"])
            for key in ("e_bound", "max_abs", "p99_abs", "nmse", "tvd", "kl"):
                if thresholds[key] > previous[key]:
                    raise GateError(f"{section} amendment weakens {key}")
            for key in ("min_top5_overlap", "min_top20_overlap", "min_teacher_steps"):
                if thresholds[key] < previous[key]:
                    raise GateError(f"{section} amendment weakens {key}")
            if (previous.get("allow_quality_difference", False) is False and
                    thresholds.get("allow_quality_difference", False) is True):
                raise GateError(f"{section} amendment enables quality differences")
        elif section == "quality":
            validate_quality_thresholds(thresholds, "quality")
            previous = record["thresholds"]["quality"]
            for key in ("min_cases", "min_target_tokens", "min_api_top1_rate_delta",
                        "min_api_pair_rate_delta"):
                if key in previous and thresholds.get(key, previous[key]) < previous[key]:
                    raise GateError(f"quality amendment weakens {key}")
            for key in ("max_mean_nll_delta", "max_ci95_high_nll_delta"):
                if key in previous and thresholds.get(key, previous[key]) > previous[key]:
                    raise GateError(f"quality amendment weakens {key}")
        else:
            validate_arithmetic_identity_contract(thresholds)
            previous = record["thresholds"]["arithmetic_identity"]
            if not set(thresholds["ignored_env_keys"]).issubset(
                    previous["ignored_env_keys"]):
                raise GateError("arithmetic_identity amendment adds env exemptions")
            if (thresholds["ignored_runtime_feature_mask"] |
                    previous["ignored_runtime_feature_mask"]) != \
                    previous["ignored_runtime_feature_mask"]:
                raise GateError("arithmetic_identity amendment adds feature exemptions")
        record.setdefault("thresholds", {})[section] = thresholds
    if generator is None and not updates:
        raise GateError("baseline amendment contains no governed change")
    if ("oracle_numerical" in updates and
            "oracle_numerical" not in predecessor["thresholds"] and generator is None):
        raise GateError("first canonical-oracle envelope must adopt its generator")
    if generator is not None and "oracle_numerical" not in record["thresholds"]:
        raise GateError("an adopted oracle generator requires a canonical-oracle envelope")
    if (len(record["reference"]["numerical"]["files"]) <
            record["thresholds"]["numerical"]["min_teacher_steps"]):
        raise GateError("numerical threshold update exceeds the bound reference length")
    if ("oracle_numerical" in record["thresholds"] and
            len(record["reference"]["numerical"]["files"]) <
            record["thresholds"]["oracle_numerical"]["min_teacher_steps"]):
        raise GateError("oracle threshold update exceeds the frozen reference length")

    record["reference"].setdefault("quality_anchor", record["reference"]["quality"])
    record["provenance"] = {
        "amendment_id": amendment["amendment_id"],
        "amendment_kind": "governance-only",
        "replaces": amendment["baseline_id"],
        "amendment_json_sha256": sha256(amendment_path),
        "evidence": reviews,
    }
    digest = canonical_sha256(record)
    baseline_path = root / "baselines" / "sha256" / f"{digest}.json"
    if baseline_path.exists():
        raise GateError(f"refusing to overwrite existing baseline: {baseline_path}")
    atomic_json(baseline_path, record)
    print(f"sha256:{digest}")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("candidate_id")
    init.add_argument("lane")
    for command in ("check", "promote"):
        child = sub.add_parser(command)
        child.add_argument("candidate_id")
    amend = sub.add_parser("amend-baseline")
    amend.add_argument("amendment", type=Path)
    genesis = sub.add_parser("bootstrap-baseline")
    genesis.add_argument("genesis", type=Path)
    args = parser.parse_args()
    try:
        repo, root = roots()
        if args.command == "init":
            init_candidate(repo, root, args.candidate_id, args.lane)
        elif args.command == "check":
            check_candidate(repo, root, args.candidate_id)
        elif args.command == "promote":
            promote_candidate(repo, root, args.candidate_id)
        elif args.command == "amend-baseline":
            amend_baseline(repo, root, args.amendment)
        else:
            bootstrap_baseline(repo, root, args.genesis)
    except (GateError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
