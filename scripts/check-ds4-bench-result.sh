#!/bin/bash
# Reject incomplete, transport-failed, or trajectory-divergent TP benchmarks.
set -euo pipefail

CSV=${1:?usage: check-ds4-bench-result.sh CSV COORD_LOG WORKER_LOG EXPECTED_FNV64 TOKENS}
COORD_LOG=${2:?missing coordinator log}
WORKER_LOG=${3:?missing worker log}
EXPECTED_FNV64=${4:-}
EXPECTED_TOKENS=${5:?missing expected generated-token count}
REQUIRE_SEMANTIC=${6:-0}

for path in "$CSV" "$COORD_LOG" "$WORKER_LOG"; do
  [[ -r $path ]] || { echo "error: missing benchmark evidence: $path" >&2; exit 1; }
done
[[ $EXPECTED_TOKENS =~ ^[1-9][0-9]*$ ]] || {
  echo "error: invalid expected token count: $EXPECTED_TOKENS" >&2; exit 2;
}
if [[ -n $EXPECTED_FNV64 && ! $EXPECTED_FNV64 =~ ^[0-9a-fA-F]{16}$ ]]; then
  echo "error: expected FNV64 must be 16 hexadecimal digits" >&2
  exit 2
fi
[[ $REQUIRE_SEMANTIC == 0 || $REQUIRE_SEMANTIC == 1 ]] || {
  echo "error: semantic requirement must be 0 or 1" >&2; exit 2;
}

read_csv_field() {
  local name=$1
  awk -F, -v name="$name" '
    NR == 1 { for (i = 1; i <= NF; i++) if ($i == name) col = i; next }
    NF && col { value = $col }
    END { if (value != "") print value; else exit 1 }
  ' "$CSV"
}

ACTUAL_TOKENS=$(read_csv_field gen_tokens) || {
  echo "error: benchmark CSV has no gen_tokens result" >&2; exit 1;
}
ACTUAL_FNV64=$(read_csv_field gen_token_fnv64) || {
  echo "error: benchmark CSV has no gen_token_fnv64 result" >&2; exit 1;
}
[[ $ACTUAL_TOKENS == "$EXPECTED_TOKENS" ]] || {
  echo "error: benchmark generated $ACTUAL_TOKENS/$EXPECTED_TOKENS tokens; rejecting result" >&2
  exit 1
}

grep -q 'worker connected, transport=rdma' "$COORD_LOG" || {
  echo "error: benchmark did not use coordinator RDMA; rejecting result" >&2; exit 1;
}
grep -q 'leader connected, transport=rdma' "$WORKER_LOG" || {
  echo "error: benchmark did not use worker RDMA; rejecting result" >&2; exit 1;
}
grep -q '"fallback_calls":0' "$COORD_LOG" || {
  echo "error: coordinator provider reported fallback traffic; rejecting result" >&2; exit 1;
}
grep -q '"fallback_calls":0' "$WORKER_LOG" || {
  echo "error: worker provider reported fallback traffic; rejecting result" >&2; exit 1;
}
if grep -Eqi 'timeout waiting|transport failed|decode .* failed|kernel (launch )?failed|nan detected' \
     "$COORD_LOG" "$WORKER_LOG"; then
  echo "error: benchmark logs contain a transport, decode, or kernel failure; rejecting result" >&2
  exit 1
fi
if [[ $REQUIRE_SEMANTIC == 1 ]]; then
  grep -q 'ds4-bench: semantic smoke passed ' "$COORD_LOG" || {
    echo "error: candidate did not pass the thinking semantic smoke; rejecting result" >&2
    exit 1
  }
  if grep -q 'ds4-bench: semantic smoke FAILED ' "$COORD_LOG"; then
    echo "error: candidate logged a failed thinking semantic smoke; rejecting result" >&2
    exit 1
  fi
fi

if [[ -n $EXPECTED_FNV64 && ${ACTUAL_FNV64,,} != ${EXPECTED_FNV64,,} ]]; then
  echo "error: token fingerprint mismatch: expected ${EXPECTED_FNV64,,}, got ${ACTUAL_FNV64,,}; rejecting candidate" >&2
  exit 1
fi

echo "validated_token_fingerprint=${ACTUAL_FNV64,,}"
