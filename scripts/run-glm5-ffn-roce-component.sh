#!/usr/bin/env bash
# Real block-3 FFN router/Q4_K bounded-window gate over mandatory RoCE v2.
set -euo pipefail

[[ $# == 4 ]] || {
  echo "usage: $0 TAG LOCAL_MODEL PEER_SSH PEER_MODEL" >&2
  exit 2
}
TAG=$1 MODEL=$2 PEER=$3 PEER_MODEL=$4
[[ $TAG =~ ^[A-Za-z0-9._-]+$ ]] || { echo "error: invalid tag" >&2; exit 2; }
for value in "$MODEL" "$PEER_MODEL"; do
  [[ $value != *"'"* && $value != *$'\n'* ]] || {
    echo "error: model paths may not contain quotes or newlines" >&2
    exit 2
  }
done

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$REPO/scripts/ds4-research-root.sh"
ds4_resolve_research_roots "$REPO"

LOCAL_DEVICE=${DS4_LOCAL_RDMA_DEVICE:-mlx5_0}
PEER_DEVICE=${DS4_PEER_RDMA_DEVICE:-mlx5_1}
HOST=${DS4_COORDINATOR_ADDR:-192.168.99.1}
PORT=${DS4_GLM5_FFN_TP_PORT:-15920}
TIMEOUT=${DS4_GLM5_TP_CONNECT_TIMEOUT_SEC:-120}
PEER_DIR=${DS4_GLM5_PEER_TEST_DIR:-/home/wkljohn/Desktop/cc/glm5-node2-test/ffn-compose}
OUT=$DS4_RESEARCH_ROOT/glm5-next-tp2/$TAG
BINARY=$REPO/tests/test_rocm_glm5_q4k_shard_compose
PEER_BINARY=$PEER_DIR/test_rocm_glm5_q4k_shard_compose
ORACLE=$OUT/raw/mla-ffn-layer3
PEER_INPUT=$PEER_DIR/mla-ffn-layer3.ffn_hidden.f32
PEER_SPLIT=$PEER_DIR/mla-ffn-layer3.ffn_split.f32
PEER_RESIDUAL=$PEER_DIR/mla-ffn-layer3.hc_carried.f32
PEER_ROUTER_IDS=$PEER_DIR/mla-ffn-layer3.router_ids.i32
PEER_ROUTER_WEIGHTS=$PEER_DIR/mla-ffn-layer3.router_weights.f32
PEER_SHARED_OUTPUT=$PEER_DIR/mla-ffn-layer3.shared_output.f32
LOCAL_HOME=${HOME:?HOME is required}
PEER_HOME=$(ssh -o BatchMode=yes "$PEER" 'printf %s "$HOME"')

for value in "$HOST" "$LOCAL_DEVICE" "$PEER_DEVICE" "$PEER_DIR" \
             "$LOCAL_HOME" "$PEER_HOME"; do
  [[ $value != *"'"* && $value != *$'\n'* ]] || {
    echo "error: remote values may not contain quotes or newlines" >&2
    exit 2
  }
done
[[ $PORT =~ ^[0-9]+$ ]] && (( PORT >= 1024 && PORT <= 65535 )) || {
  echo "error: invalid port" >&2; exit 2;
}
[[ -f $MODEL ]] || { echo "error: missing local model" >&2; exit 2; }
[[ ! -e $OUT ]] || { echo "error: refusing to overwrite $OUT" >&2; exit 1; }
mkdir -p "$OUT/raw"

make -C "$REPO" -j"$(nproc)" tests/test_rocm_glm5_q4k_shard_compose
python3 "$REPO/scripts/probe-glm5-next-mla-compose.py" \
  --layer 3 --rows 10 --first-valid 1 \
  --output "$OUT/oracle.json" --dump-prefix "$ORACLE" "$MODEL" \
  >"$OUT/oracle.log"

CONTROL_ENV=(
  DS4_GLM5_MODEL="$MODEL"
  DS4_GLM5_ROUTER_MOE_DYNAMIC=1
  DS4_GLM5_FFN_INPUT_F32="$ORACLE.ffn_hidden.f32"
  DS4_GLM5_FFN_SPLIT_F32="$ORACLE.ffn_split.f32"
  DS4_GLM5_FFN_RESIDUAL_F32="$ORACLE.hc_carried.f32"
  DS4_GLM5_FFN_ROUTER_IDS_I32="$ORACLE.router_ids.i32"
  DS4_GLM5_FFN_ROUTER_WEIGHTS_F32="$ORACLE.router_weights.f32"
  DS4_GLM5_FFN_SHARED_OUTPUT_F32="$ORACLE.shared_output.f32"
)
env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  HOME="$LOCAL_HOME" "${CONTROL_ENV[@]}" "$BINARY" \
  >"$OUT/control.log" 2>&1
grep -q 'PASS same-GGUF GLM5 Q4_K .*tp_roce=0' "$OUT/control.log" || {
  echo "error: local real-FFN Q4_K control failed" >&2; exit 1;
}
ROUTED_EXPECTED=$(sed -n 's/.* token0_shard_fnv=\([0-9A-Fa-f]\{16\}\).*/\1/p' \
  "$OUT/control.log" | tail -n1)
EXPECTED=$(sed -n 's/.* ffn_combined_fnv=\([0-9A-Fa-f]\{16\}\).*/\1/p' \
  "$OUT/control.log" | tail -n1)
BLOCK_EXPECTED=$(sed -n 's/.* block_carried_fnv=\([0-9A-Fa-f]\{16\}\).*/\1/p' \
  "$OUT/control.log" | tail -n1)
[[ $ROUTED_EXPECTED =~ ^[0-9A-Fa-f]{16}$ &&
   $EXPECTED =~ ^[0-9A-Fa-f]{16}$ &&
   $BLOCK_EXPECTED =~ ^[0-9A-Fa-f]{16}$ ]] || {
  echo "error: missing one-token composed FNV" >&2; exit 1;
}

ssh -o BatchMode=yes "$PEER" "mkdir -p -- '$PEER_DIR'; test -f '$PEER_MODEL'"
scp -q -o BatchMode=yes "$BINARY" "$PEER:$PEER_BINARY"
scp -q -o BatchMode=yes "$ORACLE.ffn_hidden.f32" "$PEER:$PEER_INPUT"
scp -q -o BatchMode=yes "$ORACLE.ffn_split.f32" "$PEER:$PEER_SPLIT"
scp -q -o BatchMode=yes "$ORACLE.hc_carried.f32" "$PEER:$PEER_RESIDUAL"
scp -q -o BatchMode=yes "$ORACLE.router_ids.i32" "$PEER:$PEER_ROUTER_IDS"
scp -q -o BatchMode=yes "$ORACLE.router_weights.f32" "$PEER:$PEER_ROUTER_WEIGHTS"
scp -q -o BatchMode=yes "$ORACLE.shared_output.f32" "$PEER:$PEER_SHARED_OUTPUT"
LOCAL_SHA=$(sha256sum "$BINARY" | awk '{print $1}')
PEER_SHA=$(ssh -o BatchMode=yes "$PEER" "sha256sum '$PEER_BINARY'" | awk '{print $1}')
[[ $LOCAL_SHA == "$PEER_SHA" ]] || { echo "error: binary mismatch" >&2; exit 1; }
LOCAL_SIZE=$(stat -c %s "$MODEL")
PEER_SIZE=$(ssh -o BatchMode=yes "$PEER" "stat -c %s '$PEER_MODEL'")
[[ $LOCAL_SIZE == "$PEER_SIZE" ]] || { echo "error: model size mismatch" >&2; exit 1; }

cat >"$OUT/run.env" <<EOF
tag=$TAG
source_head=$(git -C "$REPO" rev-parse HEAD)
stage_source_sha256=$(sha256sum \
  "$REPO/scripts/probe-glm5-next-mla-compose.py" \
  "$REPO/tests/test_rocm_glm5_ffn_shared_compose.cu" \
  "$REPO/tests/test_rocm_glm5_q4k_shard_compose.cu" \
  "$REPO/scripts/run-glm5-ffn-roce-component.sh" \
  "$REPO/Makefile" | sha256sum | awk '{print $1}')
binary_sha256=$LOCAL_SHA
model_size=$LOCAL_SIZE
expected_composed_fnv=$EXPECTED
expected_routed_fnv=$ROUTED_EXPECTED
expected_block_fnv=$BLOCK_EXPECTED
host=$HOST
port=$PORT
local_device=$LOCAL_DEVICE
peer_device=$PEER_DEVICE
EOF

env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  HOME="$LOCAL_HOME" "${CONTROL_ENV[@]}" \
  DS4_GLM5_TP_EXCLUSIVE_RANK_LOCAL=1 DS4_GLM5_TP_WINDOW_CACHE=1 \
  DS4_GLM5_TP_EXPECT_COMPOSED_FNV="$EXPECTED" \
  DS4_GLM5_TP_EXPECT_BLOCK_FNV="$BLOCK_EXPECTED" \
  DS4_GLM5_TP_CONNECT_TIMEOUT_SEC="$TIMEOUT" \
  DS4_GLM5_TP_ROLE=leader DS4_GLM5_TP_HOST="$HOST" \
  DS4_GLM5_TP_PORT="$PORT" DS4_GLM5_TP_RDMA_DEVICE="$LOCAL_DEVICE" \
  "$BINARY" >"$OUT/leader.log" 2>&1 &
leader_pid=$!
ssh -o BatchMode=yes "$PEER" \
  "env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME='$PEER_HOME' DS4_GLM5_MODEL='$PEER_MODEL' DS4_GLM5_ROUTER_MOE_DYNAMIC=1 DS4_GLM5_FFN_INPUT_F32='$PEER_INPUT' DS4_GLM5_FFN_SPLIT_F32='$PEER_SPLIT' DS4_GLM5_FFN_RESIDUAL_F32='$PEER_RESIDUAL' DS4_GLM5_FFN_ROUTER_IDS_I32='$PEER_ROUTER_IDS' DS4_GLM5_FFN_ROUTER_WEIGHTS_F32='$PEER_ROUTER_WEIGHTS' DS4_GLM5_FFN_SHARED_OUTPUT_F32='$PEER_SHARED_OUTPUT' DS4_GLM5_TP_EXCLUSIVE_RANK_LOCAL=1 DS4_GLM5_TP_WINDOW_CACHE=1 DS4_GLM5_TP_EXPECT_COMPOSED_FNV='$EXPECTED' DS4_GLM5_TP_EXPECT_BLOCK_FNV='$BLOCK_EXPECTED' DS4_GLM5_TP_CONNECT_TIMEOUT_SEC='$TIMEOUT' DS4_GLM5_TP_ROLE=worker DS4_GLM5_TP_HOST='$HOST' DS4_GLM5_TP_PORT='$PORT' DS4_GLM5_TP_RDMA_DEVICE='$PEER_DEVICE' '$PEER_BINARY'" \
  >"$OUT/worker.log" 2>&1 &
worker_pid=$!
set +e
wait "$leader_pid"; leader_rc=$?
wait "$worker_pid"; worker_rc=$?
set -e
if (( leader_rc != 0 || worker_rc != 0 )); then
  echo "error: FFN RoCE gate failed (leader=$leader_rc worker=$worker_rc)" >&2
  tail -n 40 "$OUT/leader.log" "$OUT/worker.log" >&2
  exit 1
fi

for log in "$OUT/leader.log" "$OUT/worker.log"; do
  grep -q 'transport=rdma' "$log"
  grep -q 'rdma GID index 3 (RoCE v2)' "$log"
  grep -q 'mlx5 queue pair uses RC' "$log"
  grep -q 'registered host slab as 3 MRs' "$log"
  grep -q 'bytes=16384 .*exclusive_rank_local=1 compare=fnv' "$log"
  grep -q 'PASS same-GGUF GLM5 Q4_K window_cache=1 .*cache_bytes=56623104 .*packed_table_bytes=0 shared_fold=1 .*tp_roce=1' "$log"
  grep -q "composed_fnv=$EXPECTED" "$log"
  grep -q "block_carried_fnv=$BLOCK_EXPECTED" "$log"
done
grep -q 'role=leader local_half=0' "$OUT/leader.log"
grep -q 'role=worker local_half=1' "$OUT/worker.log"
printf 'PASS GLM5 block-3 real-FFN RoCE component tag=%s binary_sha256=%s composed_fnv=%s block_fnv=%s\n' \
  "$TAG" "$LOCAL_SHA" "$EXPECTED" "$BLOCK_EXPECTED"
