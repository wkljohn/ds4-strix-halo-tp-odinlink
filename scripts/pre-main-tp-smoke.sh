#!/bin/bash
# Pre-main correctness gate: both supported quantizations must pass the same
# isolated semantic suite, exact 2048+300 trajectory, and mandatory-RDMA checks.
set -euo pipefail

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
Q4_MODEL=${1:?usage: pre-main-tp-smoke.sh Q4_MODEL Q2_MODEL}
Q2_MODEL=${2:?usage: pre-main-tp-smoke.sh Q4_MODEL Q2_MODEL}
Q4_FNV64=${DS4_PREMAIN_Q4_FNV64:-5f8a983422299d76}
Q2_FNV64=${DS4_PREMAIN_Q2_FNV64:-f9cb3a8a17e95c71}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

# The hello regression test proves independently launched ranks reject an
# IQ2 integer-WMMA mismatch before entering a different MoE arithmetic path.
make -C "$REPO" test-tp-hello

# Keep the production metadata classifier in the gate: both paths must select
# the now-correct skip-aware kernels without relying on a filename or override.
q4_config=$(DS4_BENCH_CANDIDATE=1 DS4_BENCH_EXPECT_FNV64="$Q4_FNV64" \
  DS4_BENCH_VALIDATE_CONFIG_ONLY=1 \
  "$REPO/run-tp-ds4-bench.sh" "premain-q4-config-$STAMP" "$Q4_MODEL")
q2_config=$(DS4_BENCH_CANDIDATE=1 DS4_BENCH_EXPECT_FNV64="$Q2_FNV64" \
  DS4_BENCH_VALIDATE_CONFIG_ONLY=1 \
  "$REPO/run-tp-ds4-bench.sh" "premain-q2-config-$STAMP" "$Q2_MODEL")
grep -q 'routed_expert_family=Q4_K .*tp_prefill_skip_unowned=1' <<<"$q4_config"
grep -q 'routed_expert_family=HYBRID_Q2 .*tp_prefill_skip_unowned=0 .*q2_zero_weight_tile_skip=1' <<<"$q2_config"

DS4_BENCH_CANDIDATE=1 DS4_BENCH_EXPECT_FNV64="$Q4_FNV64" \
  "$REPO/run-tp-ds4-bench.sh" "premain-q4-$STAMP" "$Q4_MODEL"
DS4_BENCH_CANDIDATE=1 DS4_BENCH_EXPECT_FNV64="$Q2_FNV64" \
  "$REPO/run-tp-ds4-bench.sh" "premain-q2-$STAMP" "$Q2_MODEL"

grep -q 'negotiated=0x.* iq2_i8=1 ' \
  "$REPO/research-results/quant-comparison-2026-08-10/coordinator-premain-q2-$STAMP.log"
grep -q 'negotiated=0x.* iq2_i8=1 ' \
  "$REPO/research-results/quant-comparison-2026-08-10/worker-premain-q2-$STAMP.log"
grep -q 'IQ2_XXS fused integer WMMA active tokens=16 hot_experts=' \
  "$REPO/research-results/quant-comparison-2026-08-10/coordinator-premain-q2-$STAMP.log"

echo "PRE_MAIN_TP_SMOKE_PASSED q4=$Q4_FNV64 q2=$Q2_FNV64"
