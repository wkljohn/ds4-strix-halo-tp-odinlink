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
    if source == "ds4-score-official-frozen-teacher":
        value["case_id"] = "case_000"
        value["case_step"] = 0
    path.write_text(json.dumps(value), encoding="utf-8")


def score_manifest(path: Path, *, arm: str,
                   model_hash: str = "model-hash",
                   full_split_order: bool = False,
                   fallback_mlp64: bool = False,
                   attention_f32_gemm: bool = False,
                   attention_repair: str | None = None,
                   attention_repairs: tuple[str, ...] = ()) -> None:
    selectors = {
        "kda-off": ("0", "0"),
        "kda-tp": ("1", "0"),
        "kda-kslice": ("1", "1"),
        "attn-scalar": ("1", "0"),
        "attn-gemm-f32": ("1", "0"),
    }
    kda_tp, kslice = selectors[arm]
    fields = {
        "producer": "gguf-tools/quality-testing/score_official.c",
        "source_commit": "source-commit",
        "source_dirty": "0",
        "model": "/model.gguf",
        "model_size": "123",
        "model_sample_sha256": model_hash,
        "ds4_sha256": "ds4-hash",
        "scorer_sha256": "scorer-hash",
        "quality_input_sha256": "quality-hash",
        "start_case": "0",
        "cases": "3",
        "teacher_positions": "381",
        "rdma_profile": "roce-v2",
        "teacher_arm": arm,
        "extra_env": (f"DS4_GLM5_KDA_TP={kda_tp} "
                      f"DS4_GLM5_KDA_OUTPUT_KSLICE={kslice} "
                      "DS4_GLM5_NEXT_PREFILL_BATCH=512 " +
                      ("DS4_ROCM_BF16_FULL_SPLIT_ORDER=1 "
                       if full_split_order else "") +
                      ("DS4_ROCM_DISABLE_BF16_DECODE_MLP64=1 "
                       if fallback_mlp64 else "") +
                      ("DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE=1 "
                       "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_F32=1 "
                       if attention_f32_gemm else "") +
                      (f"{attention_repair}=1 " if attention_repair else "") +
                      "".join(f"{repair}=1 " for repair in attention_repairs)),
        "coordinator_features": (f"GLM5 TP features: kda_tp={kda_tp} "
                                 f"kda_output_kslice={kslice}"),
        "worker_features": (f"GLM5 TP features: kda_tp={kda_tp} "
                            f"kda_output_kslice={kslice}"),
    }
    path.write_text("".join(f"{key}={value}\n" for key, value in fields.items()),
                    encoding="utf-8")


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
        exact_summary = json.loads(exact.stdout)
        assert exact_summary["argmax_mismatches"] == 0
        assert exact_summary["distributions"]["max_abs"]["max"] == 0.0
        assert exact_summary["distributions"]["teacher_nll_delta"]["mean"] == 0.0
        assert exact_summary["case_summaries"] == []
        assert exact_summary["argmax_mismatch_reference_margin"] is None
        assert exact_summary["max_argmax_mismatch_reference_margin"] is None

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
        diagnostic_summary = json.loads(diagnostic.stdout)
        assert diagnostic_summary["max_argmax_mismatch_reference_margin"] == 1.0
        assert diagnostic_summary["argmax_mismatch_reference_margin"]["max"] == 1.0

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

        dump(reference / "decode_000000.logits.json", [0.0, 1.0, 2.0, 3.0],
             source="ds4-score-official-frozen-teacher")
        dump(candidate / "decode_000000.logits.json", [0.0, 1.0, 2.0, 3.0],
             source="ds4-score-official-frozen-teacher")
        score_manifest(reference / "manifest", arm="kda-tp")
        score_manifest(candidate / "manifest", arm="kda-kslice")
        score_exact = run(str(reference), str(candidate),
                          "--score-arm-mode", "kda-kslice")
        assert score_exact.returncode == 0, score_exact.stderr
        score_summary = json.loads(score_exact.stdout)
        assert len(score_summary["case_summaries"]) == 1
        assert score_summary["case_summaries"][0]["case_id"] == "case_000"
        assert score_summary["case_summaries"][0]["steps"] == 1
        score_manifest(candidate / "manifest", arm="kda-kslice",
                       model_hash="wrong-model")
        score_mismatch = run(str(reference), str(candidate),
                             "--score-arm-mode", "kda-kslice")
        assert score_mismatch.returncode == 1
        assert "manifest model_sample_sha256" in score_mismatch.stderr

        score_manifest(reference / "manifest", arm="kda-tp")
        score_manifest(candidate / "manifest", arm="kda-tp",
                       full_split_order=True)
        reorder_null = run(str(reference), str(candidate),
                           "--score-arm-mode", "full-split-order-null")
        assert reorder_null.returncode == 0, reorder_null.stderr
        score_manifest(reference / "manifest", arm="kda-tp",
                       full_split_order=True)
        invalid_null = run(str(reference), str(candidate),
                           "--score-arm-mode", "full-split-order-null")
        assert invalid_null.returncode == 1
        assert "legal reorder only in the candidate" in invalid_null.stderr

        score_manifest(reference / "manifest", arm="kda-tp",
                       full_split_order=True)
        score_manifest(candidate / "manifest", arm="kda-kslice")
        null_vs_kslice = run(str(reference), str(candidate),
                             "--score-arm-mode", "null-vs-kslice")
        assert null_vs_kslice.returncode == 0, null_vs_kslice.stderr
        score_manifest(candidate / "manifest", arm="kda-kslice",
                       full_split_order=True)
        invalid_null_vs_kslice = run(
            str(reference), str(candidate),
            "--score-arm-mode", "null-vs-kslice")
        assert invalid_null_vs_kslice.returncode == 1
        assert "legal reorder only in the reference" in \
            invalid_null_vs_kslice.stderr

        score_manifest(reference / "manifest", arm="kda-tp",
                       fallback_mlp64=True)
        score_manifest(candidate / "manifest", arm="kda-kslice")
        fallback_vs_kslice = run(
            str(reference), str(candidate),
            "--score-arm-mode", "fallback-vs-kslice")
        assert fallback_vs_kslice.returncode == 0, \
            fallback_vs_kslice.stderr
        score_manifest(candidate / "manifest", arm="kda-kslice",
                       fallback_mlp64=True)
        invalid_fallback_vs_kslice = run(
            str(reference), str(candidate),
            "--score-arm-mode", "fallback-vs-kslice")
        assert invalid_fallback_vs_kslice.returncode == 1
        assert "independent BF16 fallback only in the reference" in \
            invalid_fallback_vs_kslice.stderr

        score_manifest(reference / "manifest", arm="attn-scalar")
        score_manifest(candidate / "manifest", arm="attn-gemm-f32",
                       attention_f32_gemm=True)
        attention = run(
            str(reference), str(candidate),
            "--score-arm-mode", "attn-scalar-vs-f32-gemm")
        assert attention.returncode == 0, attention.stderr
        score_manifest(reference / "manifest", arm="attn-scalar",
                       attention_f32_gemm=True)
        invalid_attention = run(
            str(reference), str(candidate),
            "--score-arm-mode", "attn-scalar-vs-f32-gemm")
        assert invalid_attention.returncode == 1
        assert "FP32 NoPE GEMM only in the candidate" in \
            invalid_attention.stderr

        for suffix, repair in (
                ("sync", "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_SYNC"),
                ("postdiv", "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_POSTDIV"),
                ("default-math",
                 "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_DEFAULT_MATH")):
            score_manifest(reference / "manifest", arm="attn-scalar")
            score_manifest(candidate / "manifest", arm="attn-gemm-f32",
                           attention_f32_gemm=True,
                           attention_repair=repair)
            repaired = run(
                str(reference), str(candidate),
                "--score-arm-mode", f"attn-scalar-vs-f32-gemm-{suffix}")
            assert repaired.returncode == 0, repaired.stderr

        score_manifest(reference / "manifest", arm="attn-scalar")
        score_manifest(
            candidate / "manifest", arm="attn-gemm-f32",
            attention_f32_gemm=True,
            attention_repairs=(
                "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_POSTDIV",
                "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_DEFAULT_MATH"))
        combined = run(
            str(reference), str(candidate), "--score-arm-mode",
            "attn-scalar-vs-f32-gemm-postdiv-default-math")
        assert combined.returncode == 0, combined.stderr

        score_manifest(reference / "manifest", arm="attn-scalar")
        score_manifest(
            candidate / "manifest", arm="attn-gemm-f32",
            attention_f32_gemm=True,
            attention_repairs=(
                "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_POSTDIV",
                "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_PV_SCALAR"))
        pv_scalar = run(
            str(reference), str(candidate), "--score-arm-mode",
            "attn-scalar-vs-f32-gemm-postdiv-pv-scalar")
        assert pv_scalar.returncode == 0, pv_scalar.stderr

        score_manifest(reference / "manifest", arm="attn-scalar")
        score_manifest(
            candidate / "manifest", arm="attn-gemm-f32",
            attention_f32_gemm=True,
            attention_repairs=(
                "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_POSTDIV",
                "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_PV_SCALAR",
                "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_DEFAULT_MATH"))
        pv_scalar_default = run(
            str(reference), str(candidate), "--score-arm-mode",
            "attn-scalar-vs-f32-gemm-postdiv-pv-scalar-default-math")
        assert pv_scalar_default.returncode == 0, pv_scalar_default.stderr

        score_manifest(reference / "manifest", arm="attn-scalar")
        score_manifest(
            candidate / "manifest", arm="attn-gemm-f32",
            attention_f32_gemm=True,
            attention_repairs=(
                "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_POSTDIV",
                "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_PV_SCALAR",
                "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_SCORE_SCALAR"))
        score_pv_scalar = run(
            str(reference), str(candidate), "--score-arm-mode",
            "attn-scalar-vs-f32-gemm-postdiv-score-pv-scalar")
        assert score_pv_scalar.returncode == 0, score_pv_scalar.stderr

        score_manifest(reference / "manifest", arm="attn-scalar")
        score_manifest(candidate / "manifest", arm="attn-scalar")
        attention_repeat = run(
            str(reference), str(candidate),
            "--score-arm-mode", "attn-repeat")
        assert attention_repeat.returncode == 0, attention_repeat.stderr
    print("test_compare_teacher_logits: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
