#!/bin/bash
set -euo pipefail

REPO=$(cd -- "$(dirname -- "$0")/.." && pwd)
CHECK="$REPO/scripts/check-glm5-prefill-proof.sh"
TMP=$(mktemp -d)
trap 'rm -rf -- "$TMP"' EXIT

write_pair() {
  local batch=$1 frontier=$2 tiles=$3 rows=$4 scalar=$5 min=$6 max=$7
  printf 'ds4: GLM5 prefill execution rank=0 start=0 prompt_tokens=%s requested_batch=%s batched_tiles=%s batched_rows=%s scalar_rows=%s min_tile=%s max_tile=%s\n' \
    "$frontier" "$batch" "$tiles" "$rows" "$scalar" "$min" "$max" >"$TMP/coordinator.log"
  printf 'ds4: GLM5 prefill execution rank=1 start=0 prompt_tokens=%s requested_batch=%s batched_tiles=%s batched_rows=%s scalar_rows=%s min_tile=%s max_tile=%s\n' \
    "$frontier" "$batch" "$tiles" "$rows" "$scalar" "$min" "$max" >"$TMP/worker.log"
}

write_pair 512 2048 4 2048 0 512 512
"$CHECK" "$TMP/coordinator.log" "$TMP/worker.log" 512 2048 >/dev/null

write_pair 512 600 2 600 0 88 512
"$CHECK" "$TMP/coordinator.log" "$TMP/worker.log" 512 600 >/dev/null

write_pair 1024 4096 2 2048 2048 1024 1024
"$CHECK" "$TMP/coordinator.log" "$TMP/worker.log" 1024 4096 >/dev/null

write_pair 512 2048 4 2048 0 512 512
sed -i 's/rank=1/rank=1/; s/batched_tiles=4/batched_tiles=3/' "$TMP/worker.log"
if "$CHECK" "$TMP/coordinator.log" "$TMP/worker.log" 512 2048 >/dev/null 2>&1; then
  echo "FAIL: mismatched worker proof was accepted" >&2
  exit 1
fi

echo "test_glm5_prefill_proof: PASS"
