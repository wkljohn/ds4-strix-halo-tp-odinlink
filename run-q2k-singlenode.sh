#!/bin/bash
# Single-node Q2_K serve test on node 1 only. No TP, no OdinLink, no peer.
#
# Q2_K is 90.9 GiB against a ~96 GiB GPU pool, so the memory margin is the main
# risk: there is a documented pre-existing OOM at roughly a 4.8 GiB margin on
# large prefill with the Q4_K model. Start at a modest context and walk up.
#
# Usage: ./run-q2k-singlenode.sh <tag> <ctx> [extra env...]
set -uo pipefail

TAG="${1:?usage: run-q2k-singlenode.sh <tag> <ctx> [ENV=1 ...]}"
CTX="${2:?need ctx}"
shift 2
EXTRA=("$@")

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$REPO/scripts/ds4-research-root.sh"
ds4_resolve_research_roots "$REPO"
MODEL=${DS4_BENCH_MODEL:-}
OUT=${DS4_BENCH_OUT:-$DS4_RESEARCH_ROOT/q2k-singlenode}
[[ -r $MODEL ]] || {
  echo "error: set DS4_BENCH_MODEL to the local Q2_K GGUF" >&2
  exit 2
}
mkdir -p "$OUT"
LOG="$OUT/q2k-$TAG-ctx$CTX.log"
DRIVER_DIR=$(mktemp -d)
DRIVER=$DRIVER_DIR/q2k-driver.txt
trap 'rmdir "$DRIVER_DIR" 2>/dev/null || true' EXIT

# No DSpark drafter here: it costs 10.15 GiB resident and the margin is already
# thin. Establish plain target-only decode first, then decide.
ENVV=(
  DS4_ROCM_Q8_SMALL_BATCH_TILE=1
  DS4_ROCM_Q8_SMALL_BATCH_DP4A=1
  "${EXTRA[@]}"
)

echo "=== q2k single-node: tag=$TAG ctx=$CTX ==="
echo "env: ${EXTRA[*]:-<none>}"
if pgrep -x ds4 >/dev/null; then
  echo "error: another ds4 process is running; refusing to stop it" >&2
  exit 1
fi

{
  for i in 1 2 3; do
    echo "/reset"
    echo "Write a Python function that returns the factorial of a non-negative integer."
  done
  echo "/quit"
} > "$DRIVER"

env "${ENVV[@]}" ./ds4 --rocm -m "$MODEL" \
    -c "$CTX" --temp 0 --seed 42 --nothink -n 60 \
    < "$DRIVER" > "$LOG" 2>&1

rm -f "$DRIVER"
rmdir "$DRIVER_DIR"
trap - EXIT

echo "=== complete ==="
grep -E "prefill: |generation: |q2k|Q2_K|OOM|out of memory|error" "$LOG" | tail -12
echo "RUN_DONE"
