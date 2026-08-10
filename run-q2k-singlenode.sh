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

REPO=/home/wkljohn/Desktop/cc/ds4-strix-halo-tp
MODELS=/home/wkljohn/Desktop/cc/models/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF
MODEL="$MODELS/DeepSeek-V4-Flash-Q2_K-0731.gguf"
OUT="$REPO/research-results/q2k-singlenode-2026-08-07"
mkdir -p "$OUT"
LOG="$OUT/q2k-$TAG-ctx$CTX.log"

# No DSpark drafter here: it costs 10.15 GiB resident and the margin is already
# thin. Establish plain target-only decode first, then decide.
ENVV=(
  DS4_ROCM_Q8_SMALL_BATCH_TILE=1
  DS4_ROCM_Q8_SMALL_BATCH_DP4A=1
  "${EXTRA[@]}"
)

echo "=== q2k single-node: tag=$TAG ctx=$CTX ==="
echo "env: ${EXTRA[*]:-<none>}"
pkill -x ds4 2>/dev/null
sleep 2

{
  for i in 1 2 3; do
    echo "/reset"
    echo "Write a Python function that returns the factorial of a non-negative integer."
  done
  echo "/quit"
} > /tmp/claude-1000/q2k-driver-$TAG.txt

env "${ENVV[@]}" ./ds4 --rocm -m "$MODEL" \
    -c "$CTX" --temp 0 --seed 42 --nothink -n 60 \
    < /tmp/claude-1000/q2k-driver-$TAG.txt > "$LOG" 2>&1

echo "=== complete ==="
grep -E "prefill: |generation: |q2k|Q2_K|OOM|out of memory|error" "$LOG" | tail -12
echo "RUN_DONE"
