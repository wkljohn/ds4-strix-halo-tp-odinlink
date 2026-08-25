#!/usr/bin/env python3
"""Prove diagnostic/provider drift is exempt but arithmetic drift is not."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "candidate_gate", ROOT / "scripts" / "candidate-gate.py")
assert SPEC and SPEC.loader
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def manifest(common: str, worker: str, coordinator: str,
             features: str, transport_path: str = "") -> dict[str, str]:
    return {
        "run_id": "fixture-run",
        "bench_config_sha256": "a" * 64,
        "ds4_sha256": "b" * 64,
        "peer_ds4_sha256": "b" * 64,
        "ds4_bench_tp_sha256": "c" * 64,
        "common_env": common,
        "worker_env": worker,
        "coordinator_env": coordinator,
        "extra_env": "''",
        "tp_runtime_features": features,
        "transport_library_path": transport_path,
    }


def main() -> int:
    arithmetic = "DS4_ROCM_Q4K_WMMA_FUSE_MID=1 DS4_TP_HOST_CALLBACK=1"
    frozen_common = f"{arithmetic} DS4_TP_GREEDY_TOP2=0"
    timed_roce_common = f"{arithmetic} DS4_TP_GREEDY_TOP2=1"
    timed_odin_common = (
        f"{timed_roce_common} DS4_TP_ODINLINK_BATCH_ASYNC=1 "
        "DS4_TP_VERBS_LIB=/provider.so LD_LIBRARY_PATH=/rocm/lib:/provider"
    )
    contract = {
        "ignored_env_keys": [
            "DS4_BENCH_EXPECT_GREEDY_TOP2", "DS4_TP_GREEDY_TOP2",
            "DS4_TP_ODINLINK_BATCH_ASYNC", "DS4_TP_VERBS_LIB",
        ],
        # ODINLINK_BATCH_ASYNC (bit 2) and GREEDY_TOP2 (bit 5).
        "ignored_runtime_feature_mask": 0x24,
    }
    frozen_common += " DS4_TP_GREEDY_TOP2=1 DS4_TP_GREEDY_TOP2=0 LD_LIBRARY_PATH=/rocm/lib"
    timed_roce_common += " LD_LIBRARY_PATH=/rocm/lib"
    frozen = manifest(frozen_common, frozen_common, frozen_common, "0x00080001")
    timed_roce = manifest(
        timed_roce_common, timed_roce_common,
        f"{timed_roce_common} DS4_BENCH_EXPECT_GREEDY_TOP2=1",
        "0x00080021")
    timed_odin = manifest(
        timed_odin_common, timed_odin_common,
        f"{timed_odin_common} DS4_BENCH_EXPECT_GREEDY_TOP2=1",
        "0x00080025", "/provider")

    expected = GATE.validate_arithmetic_manifest(frozen, "frozen", contract)
    assert GATE.validate_arithmetic_manifest(
        timed_roce, "timed RoCE", contract) == expected
    assert GATE.validate_arithmetic_manifest(
        timed_odin, "timed OdinLink", contract) == expected

    changed = dict(timed_roce)
    changed["common_env"] = changed["common_env"].replace(
        "DS4_ROCM_Q4K_WMMA_FUSE_MID=1",
        "DS4_ROCM_Q4K_WMMA_FUSE_MID=0")
    assert GATE.validate_arithmetic_manifest(
        changed, "changed arithmetic", contract) != expected

    changed_hash = dict(timed_roce)
    changed_hash["ds4_bench_tp_sha256"] = "d" * 64
    assert GATE.validate_arithmetic_manifest(
        changed_hash, "changed producer", contract) != expected

    changed_feature = dict(timed_roce)
    changed_feature["tp_runtime_features"] = "0x00090021"
    assert GATE.validate_arithmetic_manifest(
        changed_feature, "changed non-exempt feature", contract) != expected

    changed_runtime = dict(timed_odin)
    changed_runtime["common_env"] = changed_runtime["common_env"].replace(
        "/rocm/lib:/provider", "/rocm-7.15/lib:/provider")
    assert GATE.validate_arithmetic_manifest(
        changed_runtime, "changed ROCm runtime", contract) != expected

    geometry = {
        "prefix_tokens": "2048", "frontier": "2048",
        "context": "4096", "prefill_chunk": "2048",
    }
    workload = {"frontier": "2048", "context": "4096",
                "prefill_chunk": "2048"}
    GATE.validate_numerical_geometry(geometry, workload, "geometry")
    changed_geometry = dict(geometry)
    changed_geometry["prefill_chunk"] = "4096"
    try:
        GATE.validate_numerical_geometry(
            changed_geometry, workload, "changed geometry")
    except GATE.GateError:
        pass
    else:
        raise AssertionError("changed frozen prefill geometry passed identity")
    print("PASS arithmetic-identity-manifests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
