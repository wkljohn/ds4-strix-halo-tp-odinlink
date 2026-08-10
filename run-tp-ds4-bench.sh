#!/bin/bash
# Fixed-frontier, fixed-token TP=2 benchmark over mandatory OdinLink RDMA.
#
# Usage: ./run-tp-ds4-bench.sh <tag> <model.gguf> [EXTRA_ENV=1 ...]
set -euo pipefail

TAG="${1:?usage: run-tp-ds4-bench.sh <tag> <model.gguf> [EXTRA_ENV=1 ...]}"
MODEL="${2:?usage: run-tp-ds4-bench.sh <tag> <model.gguf> [EXTRA_ENV=1 ...]}"
shift 2
EXTRA_ENV=("$@")

REPO=/home/wkljohn/Desktop/cc/ds4-strix-halo-tp
PEER_REPO=/home/wkljohn/Desktop/cc/ds4-strix-halo-tp
PEER_MGMT=${DS4_PEER_MGMT:-wkljohn@10.10.0.216}
COORDINATOR_ADDR=${DS4_COORDINATOR_ADDR:-10.10.0.181}
PROMPT_FILE=${DS4_BENCH_PROMPT_FILE:-$REPO/research-results/2026-08-06/prompts/codex-attn-rowsplit-implement-brief.md}
FRONTIER=${DS4_BENCH_FRONTIER:-2048}
TOKENS=${DS4_BENCH_TOKENS:-300}
CONTEXT=${DS4_BENCH_CONTEXT:-4096}
PREFILL_CHUNK=${DS4_BENCH_PREFILL_CHUNK:-4096}
DSPARK=${DS4_BENCH_DSPARK:-0}
MTP=${DS4_BENCH_MTP:-/home/wkljohn/Desktop/cc/models/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF/dspark-abliterated/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf}
OUT="$REPO/research-results/quant-comparison-2026-08-10"
PEER_SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o HostKeyAlias=10.4.0.2 "$PEER_MGMT")

[[ $FRONTIER =~ ^[1-9][0-9]*$ ]] || { echo "error: invalid frontier" >&2; exit 2; }
[[ $TOKENS =~ ^[1-9][0-9]*$ ]] || { echo "error: invalid generated-token count" >&2; exit 2; }
(( CONTEXT > FRONTIER + TOKENS )) || { echo "error: context must exceed frontier + tokens" >&2; exit 2; }

COMMON_ENV=(
  DS4_ROCM_TP_SKIP_UNOWNED=1
  DS4_TP_ODINLINK_BATCH_ASYNC=1
  DS4_TP_VERBS_LIB=/home/wkljohn/Desktop/cc/OdinLink-Five/build/verbs/libodl_tb5_verbs.so.0.1.0
  LD_LIBRARY_PATH=/home/wkljohn/Desktop/cc/OdinLink-Five/build/lib:/home/wkljohn/Desktop/cc/OdinLink-Five/build/verbs
  "${EXTRA_ENV[@]}"
)
WORKER_ENV=("${COMMON_ENV[@]}")
COORD_ENV=("${COMMON_ENV[@]}")
WORKER_ARGS=()
COORD_ARGS=()
if [[ $DSPARK == 1 ]]; then
  DSPARK_ENV=(
    DS4_TP_EXPERT_SPLIT=118
    DS4_DSPARK_SCHEDULER=0
    DS4_DSPARK_SUPPORT_TOPK=6
    DS4_DSPARK_MAX_DRAFT_TOKENS=5
    DS4_DSPARK_STATS=1
    DS4_ROCM_Q8_SMALL_BATCH_TILE=1
    DS4_ROCM_Q8_SMALL_BATCH_DP4A=1
  )
  WORKER_ENV+=("${DSPARK_ENV[@]}")
  COORD_ENV+=("${DSPARK_ENV[@]}" DS4_DSPARK_RESIDENT_Q8=1)
  WORKER_ARGS=(--mtp "$MTP" --dspark)
  COORD_ARGS=(--mtp "$MTP" --dspark)
elif [[ $DSPARK != 0 ]]; then
  echo "error: DS4_BENCH_DSPARK must be 0 or 1" >&2
  exit 2
fi

COORD_LOG="$OUT/coordinator-$TAG.log"
WORKER_LOG="$OUT/worker-$TAG.log"
CSV="$OUT/$TAG.csv"
mkdir -p "$OUT"

[[ -r $MODEL && -r $PROMPT_FILE ]] || { echo "error: missing local model or prompt" >&2; exit 1; }
if [[ $DSPARK == 1 && ! -r $MTP ]]; then
  echo "error: missing local DSpark model: $MTP" >&2
  exit 1
fi
[[ -r /dev/odl_tb5_0 ]] || { echo "error: local OdinLink device unavailable" >&2; exit 1; }
"${PEER_SSH[@]}" "test -r '$MODEL' -a -r /dev/odl_tb5_0" || {
  echo "error: peer model or OdinLink device unavailable" >&2; exit 1;
}
if [[ $DSPARK == 1 ]]; then
  "${PEER_SSH[@]}" "test -r '$MTP'" || { echo "error: peer DSpark model missing" >&2; exit 1; }
  LOCAL_MTP_SIZE=$(stat -c %s "$MTP")
  PEER_MTP_SIZE=$("${PEER_SSH[@]}" "stat -c %s '$MTP'")
  [[ $LOCAL_MTP_SIZE == "$PEER_MTP_SIZE" ]] || {
    echo "error: DSpark model sizes differ" >&2; exit 1;
  }
fi
LOCAL_MODEL_SIZE=$(stat -c %s "$MODEL")
PEER_MODEL_SIZE=$("${PEER_SSH[@]}" "stat -c %s '$MODEL'")
[[ $LOCAL_MODEL_SIZE == "$PEER_MODEL_SIZE" ]] || {
  echo "error: model sizes differ: local=$LOCAL_MODEL_SIZE peer=$PEER_MODEL_SIZE" >&2; exit 1;
}
LOCAL_DS4_HASH=$(sha256sum "$REPO/ds4" | awk '{print $1}')
PEER_DS4_HASH=$("${PEER_SSH[@]}" "sha256sum '$PEER_REPO/ds4'" | awk '{print $1}')
[[ $LOCAL_DS4_HASH == "$PEER_DS4_HASH" ]] || {
  echo "error: worker binary hashes differ" >&2; exit 1;
}

cleanup() {
  pkill -f '[d]s4-bench.*--role coordinator.*--tensor-parallel' 2>/dev/null || true
  "${PEER_SSH[@]}" "pkill -f '[d]s4 --role worker --tensor-parallel'" 2>/dev/null || true
}
trap cleanup EXIT
cleanup
sleep 2
"${PEER_SSH[@]}" "mkdir -p '$OUT'"

echo "=== ds4-bench $TAG ==="
echo "model: $MODEL"
echo "workload: frontier=$FRONTIER generated_tokens=$TOKENS context=$CONTEXT prefill_chunk=$PREFILL_CHUNK"
if [[ $DSPARK == 1 ]]; then echo "dspark: 1 mtp=$MTP"; else echo "dspark: 0"; fi
echo "ds4_sha256: $LOCAL_DS4_HASH"

"${PEER_SSH[@]}" "cd '$PEER_REPO' && setsid -f env ${WORKER_ENV[*]} ./ds4 \
  --role worker --tensor-parallel --coordinator '$COORDINATOR_ADDR' 9000 \
  --transport rdma --rocm -m '$MODEL' -c '$CONTEXT' \
  --prefill-chunk '$PREFILL_CHUNK' \
  ${WORKER_ARGS[*]} \
  > '$WORKER_LOG' 2>&1" &

env "${COORD_ENV[@]}" "$REPO/ds4-bench-tp" \
  --role coordinator --tensor-parallel --listen 0.0.0.0 9000 \
  --transport rdma --rocm -m "$MODEL" --prompt-file "$PROMPT_FILE" \
  --ctx-start "$FRONTIER" --ctx-max "$FRONTIER" --ctx-alloc "$CONTEXT" \
  --prefill-chunk "$PREFILL_CHUNK" --gen-tokens "$TOKENS" --csv "$CSV" \
  "${COORD_ARGS[@]}" \
  > "$COORD_LOG" 2>&1

cleanup
trap - EXIT
grep -q 'worker connected, transport=rdma' "$COORD_LOG" || {
  echo "error: benchmark did not use RDMA; rejecting result" >&2; exit 1;
}
grep -q '"fallback_calls":0' "$COORD_LOG" || {
  echo "error: RDMA provider reported fallback traffic; rejecting result" >&2; exit 1;
}
cat "$CSV"
echo RUN_DONE
