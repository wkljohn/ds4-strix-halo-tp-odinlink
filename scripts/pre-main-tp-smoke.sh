#!/bin/bash
# Pre-main correctness gate: both supported quantizations must pass the same
# isolated semantic suite, exact 2048+300 trajectory, and mandatory-RDMA checks.
set -euo pipefail

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$REPO/scripts/ds4-research-root.sh"
Q4_MODEL=${1:?usage: pre-main-tp-smoke.sh Q4_MODEL Q2_MODEL}
Q2_MODEL=${2:?usage: pre-main-tp-smoke.sh Q4_MODEL Q2_MODEL}
Q4_FNV64=${DS4_PREMAIN_Q4_FNV64:-5f8a983422299d76}
Q2_FNV64=${DS4_PREMAIN_Q2_FNV64:-f9cb3a8a17e95c71}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BENCH_CONFIG=${DS4_BENCH_CONFIG:-$REPO/bench.env.local}
if [[ -r $BENCH_CONFIG ]]; then
  # Resolve the same durable output directory as run-tp-ds4-bench.sh. The two
  # scripts must not disagree when bench.env.local moves research artifacts.
  # shellcheck disable=SC1090
  source "$BENCH_CONFIG"
fi
ds4_resolve_research_roots "$REPO"
PREMAIN_OUT=${DS4_BENCH_OUT:-$DS4_RESEARCH_ROOT/bench-runs}

# The hello regression test proves independently launched ranks reject an
# IQ2 integer-WMMA mismatch before entering a different MoE arithmetic path.
"$REPO/tests/test_bench_env_precedence.sh"
"$REPO/tests/test_research_root_contract.sh"
"$REPO/tests/test_candidate_gate.sh"
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

require_feature_bit() {
  local log=$1 bit=$2 hex value
  hex=$(sed -n 's/.* negotiated=0x\([0-9a-fA-F][0-9a-fA-F]*\).*/\1/p' "$log" | head -n 1)
  [[ -n $hex ]] || { echo "error: no negotiated feature mask in $log" >&2; return 1; }
  value=$((16#$hex))
  (( (value & bit) != 0 )) || {
    printf 'error: required TP feature bit 0x%x absent from %s (mask=0x%s)\n' \
      "$bit" "$log" "$hex" >&2
    return 1
  }
}

# These exact kernels are production defaults. Check both independently
# launched ranks rather than relying only on the model-level fingerprint.
for model in q4 q2; do
  require_feature_bit "$PREMAIN_OUT/coordinator-premain-$model-$STAMP.log" $((1 << 17))
  require_feature_bit "$PREMAIN_OUT/worker-premain-$model-$STAMP.log" $((1 << 17))
  require_feature_bit "$PREMAIN_OUT/coordinator-premain-$model-$STAMP.log" $((1 << 18))
  require_feature_bit "$PREMAIN_OUT/worker-premain-$model-$STAMP.log" $((1 << 18))
done

grep -q 'negotiated=0x.* iq2_i8=1 ' \
  "$PREMAIN_OUT/coordinator-premain-q2-$STAMP.log"
grep -q 'negotiated=0x.* iq2_i8=1 ' \
  "$PREMAIN_OUT/worker-premain-q2-$STAMP.log"
grep -q 'IQ2_XXS fused integer WMMA active tokens=16 hot_experts=' \
  "$PREMAIN_OUT/coordinator-premain-q2-$STAMP.log"

echo "PRE_MAIN_TP_SMOKE_PASSED q4=$Q4_FNV64 q2=$Q2_FNV64"
