#!/bin/bash
# One long-prompt DSpark TP=2 benchmark run: launch peer worker + local
# coordinator together, drive identical /reset repetitions, capture logs, then
# tear down. The interactive CLI treats every input line as a new request, so
# flatten the prose fixture to one line before feeding it.
#
# Usage: ./run-dspark-bench.sh <tag> [extra env assignments...]
#   e.g. ./run-dspark-bench.sh baseline
#        ./run-dspark-bench.sh chain DS4_DSPARK_CHAIN_CYCLES=1
set -euo pipefail

TAG="${1:?usage: run-dspark-bench.sh <tag> [EXTRA_ENV=1 ...]}"
shift
EXTRA_ENV=("$@")
REPEATS=${DS4_BENCH_REPEATS:-3}
[[ $REPEATS =~ ^[1-9][0-9]*$ ]] || { echo "error: DS4_BENCH_REPEATS must be positive" >&2; exit 2; }

REPO=/home/wkljohn/Desktop/cc/ds4-strix-halo-tp
PEER_REPO=/home/wkljohn/Desktop/cc/ds4-strix-halo-tp
PEER_MGMT=${DS4_PEER_MGMT:-wkljohn@10.10.0.216}
COORDINATOR_ADDR=${DS4_COORDINATOR_ADDR:-10.10.0.181}
PEER_SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o HostKeyAlias=10.4.0.2 "$PEER_MGMT")
MODELS=/home/wkljohn/Desktop/cc/models/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF
MODEL="$MODELS/DeepSeek-V4-Flash-Q4_K-0731.gguf"
MTP="$MODELS/dspark-abliterated/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"
OUT="$REPO/research-results/dspark-17tps-2026-08-07"
mkdir -p "$OUT"

COORD_LOG="$OUT/coordinator-long-$TAG.log"
WORKER_LOG="$OUT/worker-long-$TAG.log"

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
  # 6 is the support model's TRAINED default, not a tuned value. The handover's
  # standard config specified 4, which silently overrode the drafter BELOW what it
  # was trained for and cost ~10pp of acceptance. Measured 2026-08-07:
  #   top-k 4 -> accept 85.71%, avg_accept 4.00, 12 cycles/60 tok, 15.11 t/s
  #   top-k 6 -> accept 96.08%, avg_accept 4.455, 11 cycles/60 tok, 16.17 t/s
  # Every run's own banner had been printing "trained/default top-k=6" the whole
  # time. Only 4 (24 runs) and 2 (1 run) had ever been tested.
  # metal_graph_dspark_support_topk() returns DS4_N_EXPERT_USED (=6) when unset.
  DS4_DSPARK_SUPPORT_TOPK=6
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

if [[ ! -r /dev/odl_tb5_0 ]]; then
    echo "error: local OdinLink device /dev/odl_tb5_0 is unavailable" >&2
    exit 1
fi
if ! "${PEER_SSH[@]}" 'test -r /dev/odl_tb5_0'; then
    echo "error: peer OdinLink device /dev/odl_tb5_0 is unavailable" >&2
    exit 1
fi

# Terminate only our exact commands, never broad system processes.
cleanup() {
    pkill -f "ds4 --role coordinator --tensor-parallel" 2>/dev/null || true
    "${PEER_SSH[@]}" "pkill -f 'ds4 --role worker --tensor-parallel'" 2>/dev/null || true
}
trap cleanup EXIT
cleanup
sleep 2

# The peer filesystem is NOT shared: its log directory must exist before the
# worker redirect, or the launch silently fails and the coordinator hangs.
"${PEER_SSH[@]}" "mkdir -p $OUT"

# Worker and coordinator start concurrently; never wait for the coordinator port.
"${PEER_SSH[@]}" "cd $PEER_REPO && setsid -f env ${WORKER_ENV[*]} ./ds4 --role worker --tensor-parallel \
    --coordinator $COORDINATOR_ADDR 9000 --transport rdma --rocm -m '$MODEL' --mtp '$MTP' --dspark -c 4096 \
    > $WORKER_LOG 2>&1" &

# Three identical repetitions with /reset between them.
{
  for ((i = 1; i <= REPEATS; i++)); do
    echo "/reset"
    tr '\n' ' ' < "$REPO/bench-prompt-long.txt"
    echo
  done
  echo "/quit"
} > "$OUT/driver-$TAG.txt"

env "${COORD_ENV[@]}" ./ds4 --role coordinator --tensor-parallel \
    --listen 0.0.0.0 9000 --transport rdma --rocm -m "$MODEL" --mtp "$MTP" --dspark \
    -c 4096 --temp 0 --seed 42 --nothink -n 256 \
    < "$OUT/driver-$TAG.txt" > "$COORD_LOG" 2>&1

echo "=== run $TAG complete ==="
cleanup
trap - EXIT
if ! grep -q 'worker connected, transport=rdma' "$COORD_LOG"; then
    echo "error: benchmark did not use RDMA; rejecting result" >&2
    exit 1
fi
grep -E "generation: |DSpark stats" "$COORD_LOG" | tail -10
echo "RUN_DONE"
