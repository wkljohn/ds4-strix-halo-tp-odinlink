#!/usr/bin/env bash
# Reproduce the bounded two-process GLM-5.3 Q4_K transport-fidelity gate.
set -euo pipefail

usage() {
  echo "usage: $0 TAG LOCAL_MODEL PEER_SSH PEER_MODEL" >&2
  exit 2
}

[[ $# == 4 ]] || usage
TAG=$1
MODEL=$2
PEER=$3
PEER_MODEL=$4
[[ $TAG =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "error: TAG may contain only letters, digits, '.', '_' and '-'" >&2
  exit 2
}
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
COORDINATOR_ADDR=${DS4_COORDINATOR_ADDR:-192.168.99.1}
PORT_BASE=${DS4_GLM5_TP_PORT_BASE:-15720}
SEEDS=${DS4_GLM5_ROUTER_SEEDS:-"2 3 6"}
PEER_DIR=${DS4_GLM5_PEER_TEST_DIR:-/home/wkljohn/Desktop/cc/glm5-node2-test}
OUT=$DS4_RESEARCH_ROOT/glm5-next-tp2/$TAG
BINARY=$REPO/tests/test_rocm_glm5_q4k_shard_compose
PEER_BINARY=$PEER_DIR/test_rocm_glm5_q4k_shard_compose

sample_fingerprint() {
  python3 - "$1" <<'PY'
import hashlib, os, sys
p = sys.argv[1]
n = os.path.getsize(p)
h = hashlib.sha256()
with open(p, "rb", buffering=0) as f:
    for off in (0, max(0, n // 2 - 524288), max(0, n - 1048576)):
        f.seek(off)
        h.update(f.read(1048576))
print(h.hexdigest())
PY
}

require_log() {
  local log=$1 pattern=$2 label=$3
  grep -q -- "$pattern" "$log" || {
    echo "error: missing $label in $log" >&2
    return 1
  }
}

[[ -f $MODEL ]] || { echo "error: local model not found: $MODEL" >&2; exit 2; }
[[ $PORT_BASE =~ ^[1-9][0-9]*$ ]] && (( PORT_BASE >= 1024 && PORT_BASE <= 65530 )) || {
  echo "error: invalid DS4_GLM5_TP_PORT_BASE" >&2
  exit 2
}

make -C "$REPO" -j"$(nproc)" tests/test_rocm_glm5_q4k_shard_compose
mkdir -p "$OUT"
ssh -o BatchMode=yes "$PEER" "mkdir -p -- '$PEER_DIR'; test -f '$PEER_MODEL'"
scp -q -o BatchMode=yes "$BINARY" "$PEER:$PEER_BINARY"

LOCAL_SHA=$(sha256sum "$BINARY" | awk '{print $1}')
PEER_SHA=$(ssh -o BatchMode=yes "$PEER" "sha256sum '$PEER_BINARY'" | awk '{print $1}')
[[ $LOCAL_SHA == "$PEER_SHA" ]] || { echo "error: binary checksum mismatch" >&2; exit 1; }
LOCAL_SIZE=$(stat -c %s "$MODEL")
PEER_SIZE=$(ssh -o BatchMode=yes "$PEER" "stat -c %s '$PEER_MODEL'")
[[ $LOCAL_SIZE == "$PEER_SIZE" ]] || { echo "error: model size mismatch" >&2; exit 1; }
LOCAL_SAMPLE=$(sample_fingerprint "$MODEL")
PEER_SAMPLE=$(ssh -o BatchMode=yes "$PEER" "python3 - '$PEER_MODEL'" <<'PY'
import hashlib, os, sys
p = sys.argv[1]
n = os.path.getsize(p)
h = hashlib.sha256()
with open(p, "rb", buffering=0) as f:
    for off in (0, max(0, n // 2 - 524288), max(0, n - 1048576)):
        f.seek(off)
        h.update(f.read(1048576))
print(h.hexdigest())
PY
)
[[ $LOCAL_SAMPLE == "$PEER_SAMPLE" ]] || {
  echo "error: sampled model fingerprint mismatch" >&2
  exit 1
}

index=0
for seed in $SEEDS; do
  [[ $seed =~ ^[0-9]+$ ]] || { echo "error: invalid seed: $seed" >&2; exit 2; }
  port=$((PORT_BASE + index))
  run=$OUT/seed-$seed
  mkdir -p "$run"
  timeout 150s env \
    DS4_GLM5_MODEL="$MODEL" \
    DS4_GLM5_ROUTER_MOE_DYNAMIC=1 \
    DS4_GLM5_ROUTER_JITTER_SEED="$seed" \
    DS4_GLM5_TP_ROLE=leader \
    DS4_GLM5_TP_HOST="$COORDINATOR_ADDR" \
    DS4_GLM5_TP_PORT="$port" \
    DS4_GLM5_TP_RDMA_DEVICE="$LOCAL_DEVICE" \
    "$BINARY" >"$run/leader.log" 2>&1 &
  leader_pid=$!
  timeout 150s ssh -o BatchMode=yes "$PEER" \
    "DS4_GLM5_MODEL='$PEER_MODEL' DS4_GLM5_ROUTER_MOE_DYNAMIC=1 DS4_GLM5_ROUTER_JITTER_SEED='$seed' DS4_GLM5_TP_ROLE=worker DS4_GLM5_TP_HOST='$COORDINATOR_ADDR' DS4_GLM5_TP_PORT='$port' DS4_GLM5_TP_RDMA_DEVICE='$PEER_DEVICE' '$PEER_BINARY'" \
    >"$run/worker.log" 2>&1 &
  worker_pid=$!
  set +e
  wait "$leader_pid"; leader_rc=$?
  wait "$worker_pid"; worker_rc=$?
  set -e
  if (( leader_rc != 0 || worker_rc != 0 )); then
    echo "error: seed $seed failed (leader=$leader_rc worker=$worker_rc)" >&2
    tail -30 "$run/leader.log" "$run/worker.log" >&2
    exit 1
  fi
  for log in "$run/leader.log" "$run/worker.log"; do
    require_log "$log" 'transport=rdma' 'mandatory RDMA proof'
    require_log "$log" 'rdma GID index 3 (RoCE v2)' 'RoCE v2 proof'
    require_log "$log" 'mlx5 queue pair uses RC' 'RC proof'
    require_log "$log" 'registered host slab as 3 MRs' 'three-MR proof'
    require_log "$log" 'GLM5 bounded TP RoCE .*direct=1 .*bad=0' 'direct clean composition'
    require_log "$log" 'tp_roce=1' 'TP RoCE PASS marker'
  done
  echo "PASS seed=$seed"
  index=$((index + 1))
done

printf 'PASS GLM5 bounded RoCE component tag=%s binary_sha256=%s\n' \
  "$TAG" "$LOCAL_SHA"
