#!/usr/bin/env python3
"""Create and validate DS4 performance-candidate promotion dossiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
LANES = {"A", "B", "C"}
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
COMMON_CLAIMS = {
    "mandatory_rdma",
    "no_rdma_fallback",
    "no_persistent_weight_cache",
    "ordinary_no_regression",
    "long_context_pass",
    "cross_discipline_long_pass",
}
LANE_CLAIMS = {
    "A": {"exact_fingerprint", "rollback_reproduces_baseline"},
    "B": {
        "self_deterministic_3_of_3",
        "numerical_envelope_pass",
        "far_margin_flips_zero",
        "semantic_retrieval_pass",
    },
    "C": {
        "self_deterministic_3_of_3",
        "numerical_envelope_pass",
        "far_margin_flips_zero",
        "semantic_retrieval_pass",
        "teacher_control_pass",
        "reference_quality_neutral_or_better",
        "rebaseline_approved",
    },
}
DSPARK_KINDS = {"same-stack-ordinary", "verifier-logits", "dspark-acceptance"}
DSPARK_CLAIMS = {
    "accepted_target_quality_preserved",
    "acceptance_profile_recorded",
    "same_stack_target_used",
}


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


def init_candidate(repo: Path, root: Path, candidate_id: str, lane: str) -> None:
    lane = lane.upper()
    if lane not in LANES:
        raise GateError("lane must be A, B, or C")
    dossier = dossier_path(root, candidate_id)
    if dossier.exists():
        raise GateError(f"candidate dossier already exists: {dossier}")
    status = run_git(repo, "status", "--porcelain=v1", "-uall")
    value = {
        "schema_version": 1,
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
        "transport": {"providers": []},
        "dspark": False,
        "claims": {key: False for key in sorted(COMMON_CLAIMS | LANE_CLAIMS[lane])},
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


def check_candidate(root: Path, candidate_id: str) -> tuple[Path, dict]:
    dossier, value = load_candidate(root, candidate_id)
    if value.get("schema_version") != 1 or value.get("candidate_id") != candidate_id:
        raise GateError("candidate schema/id mismatch")
    lane = str(value.get("lane", "")).upper()
    if lane not in LANES:
        raise GateError("invalid candidate lane")
    source = value.get("source")
    if not isinstance(source, dict) or not re.fullmatch(r"[0-9a-f]{40}", str(source.get("commit", ""))):
        raise GateError("source.commit must be a full Git object id")
    if lane in {"B", "C"} and not value.get("baseline_id"):
        raise GateError(f"lane {lane} requires a versioned baseline_id")

    evidence = value.get("evidence")
    if not isinstance(evidence, list):
        raise GateError("evidence must be a list")
    kinds: list[str] = []
    diverse_summaries: list[Path] = []
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise GateError(f"evidence[{index}] must be an object")
        kind = str(item.get("kind", ""))
        path_text = str(item.get("path", ""))
        expected = str(item.get("sha256", ""))
        if not kind or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise GateError(f"evidence[{index}] has invalid kind or SHA-256")
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
        kinds.append(kind)
        if kind == "cross-discipline-long":
            diverse_summaries.append(path)

    required_kinds = COMMON_KINDS | LANE_KINDS[lane]
    if value.get("dspark") is True:
        required_kinds |= DSPARK_KINDS
    missing_kinds = sorted(required_kinds - set(kinds))
    if missing_kinds:
        raise GateError("missing evidence kinds: " + ", ".join(missing_kinds))
    if kinds.count("candidate-benchmark") < 3:
        raise GateError("at least three candidate-benchmark artifacts are required")
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
    expected_diverse_mode = "dspark" if value.get("dspark") is True else "ordinary"
    if summary.get("mode") != expected_diverse_mode:
        raise GateError(
            f"cross-disciplinary evidence mode must be {expected_diverse_mode}"
        )

    claims = value.get("claims")
    if not isinstance(claims, dict):
        raise GateError("claims must be an object")
    required_claims = COMMON_CLAIMS | LANE_CLAIMS[lane]
    if value.get("dspark") is True:
        required_claims |= DSPARK_CLAIMS
    false_claims = sorted(key for key in required_claims if claims.get(key) is not True)
    if false_claims:
        raise GateError("unpassed claims: " + ", ".join(false_claims))

    print(f"PASS candidate={candidate_id} lane={lane} evidence={len(evidence)}")
    return dossier, value


def promote_candidate(root: Path, candidate_id: str) -> None:
    dossier, value = check_candidate(root, candidate_id)
    candidate_file = dossier / "candidate.json"
    promoted = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "lane": value["lane"],
        "source_commit": value["source"]["commit"],
        "candidate_json_sha256": sha256(candidate_file),
        "promoted_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(dossier / "PROMOTED.json", promoted)
    print(dossier / "PROMOTED.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("candidate_id")
    init.add_argument("lane")
    for command in ("check", "promote"):
        child = sub.add_parser(command)
        child.add_argument("candidate_id")
    args = parser.parse_args()
    try:
        repo, root = roots()
        if args.command == "init":
            init_candidate(repo, root, args.candidate_id, args.lane)
        elif args.command == "check":
            check_candidate(root, args.candidate_id)
        else:
            promote_candidate(root, args.candidate_id)
    except (GateError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
