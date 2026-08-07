#!/bin/bash
# One DSpark TP=2 benchmark run: launch peer worker + local coordinator together,
# drive three identical /reset repetitions, capture logs, then tear down.
#
# Usage: ./run-dspark-bench.sh <tag> [extra env assignments...]
#   e.g. ./run-dspark-bench.sh baseline
#        ./run-dspark-bench.sh chain DS4_DSPARK_CHAIN_CYCLES=1
set -uo pipefail

TAG="${1:?usage: run-dspark-bench.sh <tag> [EXTRA_ENV=1 ...]}"
shift
EXTRA_ENV=("$@")

REPO=/home/wkljohn/Desktop/cc/ds4-strix-halo-tp
PEER_REPO=/home/wkljohn/Desktop/cc/ds4-strix-halo-tp
MODELS=/home/wkljohn/Desktop/cc/models/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF
MODEL="$MODELS/DeepSeek-V4-Flash-Q4_K-0731.gguf"
MTP="$MODELS/dspark-abliterated/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"
OUT="$REPO/research-results/dspark-17tps-2026-08-07"
mkdir -p "$OUT"

COORD_LOG="$OUT/coordinator-$TAG.log"
WORKER_LOG="$OUT/worker-$TAG.log"

# Standard DSpark test configuration from the handover note.
# NOTE: DS4_DSPARK_RESIDENT_Q8=1 is COORDINATOR-ONLY. Only rank 0 runs the
# drafter (that is why the expert split is asymmetric 118/138), and rank 1
# warms 86.43 GiB, leaving ~9.09 GiB free -- below the 10.15 GiB the resident
# Q8 drafter needs. Setting it on the worker makes the worker abort at startup
# and the coordinator then hangs forever waiting for a peer that never arrives.
COMMON_ENV=(
  DS4_TP_EXPERT_SPLIT=118
  DS4_ROCM_TP_SKIP_UNOWNED=1
  DS4_DSPARK_SCHEDULER=0
  DS4_DSPARK_SUPPORT_TOPK=4
  DS4_DSPARK_MAX_DRAFT_TOKENS=5
  DS4_DSPARK_STATS=1
  DS4_ROCM_Q8_SMALL_BATCH_TILE=1
  DS4_ROCM_Q8_SMALL_BATCH_DP4A=1
  DS4_TP_VERBS_LIB=/home/wkljohn/Desktop/cc/OdinLink-Five/build/verbs/libodl_tb5_verbs.so.0.1.0
  LD_LIBRARY_PATH=/home/wkljohn/Desktop/cc/OdinLink-Five/build/lib:/home/wkljohn/Desktop/cc/OdinLink-Five/build/verbs
)
WORKER_ENV=("${COMMON_ENV[@]}" "${EXTRA_ENV[@]}")
COORD_ENV=("${COMMON_ENV[@]}" DS4_DSPARK_RESIDENT_Q8=1 "${EXTRA_ENV[@]}")

echo "=== run $TAG ==="
echo "env: ${EXTRA_ENV[*]:-<none>}"

# Terminate only our exact commands, never broad system processes.
cleanup() {
    pkill -f "ds4 --role coordinator --tensor-parallel" 2>/dev/null
    ssh peer "pkill -f 'ds4 --role worker --tensor-parallel'" 2>/dev/null
}
cleanup
sleep 2

# The peer filesystem is NOT shared: its log directory must exist before the
# worker redirect, or the launch silently fails and the coordinator hangs.
ssh peer "mkdir -p $OUT"

# Worker and coordinator start concurrently; never wait for the coordinator port.
ssh peer "cd $PEER_REPO && setsid -f env ${WORKER_ENV[*]} ./ds4 --role worker --tensor-parallel \
    --coordinator 10.4.0.1 9000 --rocm -m '$MODEL' --mtp '$MTP' --dspark -c 128 \
    > $WORKER_LOG 2>&1" &

# Three identical repetitions with /reset between them.
{
  for i in 1 2 3; do
    echo "/reset"
    echo "Write a Python function that returns the factorial of a non-negative integer."
  done
  echo "/quit"
} > /tmp/claude-1000/dspark-driver-$TAG.txt

env "${COORD_ENV[@]}" ./ds4 --role coordinator --tensor-parallel \
    --listen 0.0.0.0 9000 --rocm -m "$MODEL" --mtp "$MTP" --dspark \
    -c 128 --temp 0 --seed 42 --nothink -n 60 \
    < /tmp/claude-1000/dspark-driver-$TAG.txt > "$COORD_LOG" 2>&1

echo "=== run $TAG complete ==="
cleanup
grep -E "generation: |DSpark stats" "$COORD_LOG" | tail -10
echo "RUN_DONE"
