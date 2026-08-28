#!/usr/bin/env bash
# Reproduce the two-node GLM-5.3 mHC-to-sparse-MLA RoCE component gate.
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
  echo "error: invalid tag" >&2
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
HOST=${DS4_COORDINATOR_ADDR:-192.168.99.1}
PORT=${DS4_GLM5_MLA_TP_PORT:-15880}
TIMEOUT=${DS4_GLM5_TP_CONNECT_TIMEOUT_SEC:-120}
BLOCK_SESSION_PROBE=${DS4_GLM5_BLOCK_SESSION_PROBE:-0}
FFN_PREROUTER=${DS4_GLM5_BLOCK_FFN_PREROUTER:-0}
PEER_DIR=${DS4_GLM5_PEER_TEST_DIR:-/home/wkljohn/Desktop/cc/glm5-node2-test/mla-compose}
OUT=$DS4_RESEARCH_ROOT/glm5-next-tp2/$TAG
BINARY=$REPO/tests/test_rocm_glm5_mla_compose
PEER_BINARY=$PEER_DIR/test_rocm_glm5_mla_compose
ORACLE=$DS4_RESEARCH_ROOT/glm5-next-tp2/raw/mla-mhc-compose-layer3-v4
PEER_ORACLE=$PEER_DIR/mla-mhc-compose-layer3-v4
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
  echo "error: invalid port" >&2
  exit 2
}
[[ $TIMEOUT =~ ^[1-9][0-9]*$ ]] || {
  echo "error: invalid timeout" >&2
  exit 2
}
[[ $BLOCK_SESSION_PROBE == 0 || $BLOCK_SESSION_PROBE == 1 ]] || {
  echo "error: DS4_GLM5_BLOCK_SESSION_PROBE must be 0 or 1" >&2
  exit 2
}
[[ $FFN_PREROUTER == 0 || $FFN_PREROUTER == 1 ]] || {
  echo "error: DS4_GLM5_BLOCK_FFN_PREROUTER must be 0 or 1" >&2
  exit 2
}
[[ -f $MODEL ]] || { echo "error: missing local model" >&2; exit 2; }

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

field() {
  sed -n "s/.*[[:space:]]$2=\([0-9A-Fa-f]\{16\}\).*/\1/p" "$1" | tail -n 1
}

line_field() {
  grep -F -- "$2" "$1" |
    sed -n "s/.*[[:space:]]$3=\([0-9A-Fa-f]\{16\}\).*/\1/p" |
    tail -n 1
}

mkdir -p "$(dirname -- "$OUT")"
[[ ! -e $OUT ]] || {
  echo "error: refusing to overwrite existing evidence directory $OUT" >&2
  exit 1
}
mkdir "$OUT"
make -C "$REPO" -j"$(nproc)" tests/test_rocm_glm5_mla_compose
python3 "$REPO/scripts/probe-glm5-next-mla-compose.py" \
  --layer 3 --rows 10 --first-valid 1 \
  --output "$DS4_RESEARCH_ROOT/glm5-next-tp2/mla-mhc-compose-layer3-v4-oracle.json" \
  --dump-prefix "$ORACLE" "$MODEL" >"$OUT/oracle.log"

ssh -o BatchMode=yes "$PEER" "mkdir -p -- '$PEER_DIR'; test -f '$PEER_MODEL'"
scp -q -o BatchMode=yes "$BINARY" "$PEER:$PEER_BINARY"
for artifact in "$ORACLE".*; do
  scp -q -o BatchMode=yes "$artifact" "$PEER:$PEER_DIR/"
done

LOCAL_SHA=$(sha256sum "$BINARY" | awk '{print $1}')
PEER_SHA=$(ssh -o BatchMode=yes "$PEER" "sha256sum '$PEER_BINARY'" | awk '{print $1}')
[[ $LOCAL_SHA == "$PEER_SHA" ]] || {
  echo "error: binary checksum mismatch" >&2
  exit 1
}
LOCAL_SIZE=$(stat -c %s "$MODEL")
PEER_SIZE=$(ssh -o BatchMode=yes "$PEER" "stat -c %s '$PEER_MODEL'")
[[ $LOCAL_SIZE == "$PEER_SIZE" ]] || {
  echo "error: model size mismatch" >&2
  exit 1
}
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

cat >"$OUT/run.env" <<EOF
tag=$TAG
source_head=$(git -C "$REPO" rev-parse HEAD)
test_source_sha256=$(sha256sum "$REPO/tests/test_rocm_glm5_mla_compose.cu" | awk '{print $1}')
launcher_sha256=$(sha256sum "$REPO/scripts/run-glm5-mla-roce-component.sh" | awk '{print $1}')
binary_sha256=$LOCAL_SHA
model_size=$LOCAL_SIZE
model_sample_sha256=$LOCAL_SAMPLE
host=$HOST
port=$PORT
local_device=$LOCAL_DEVICE
peer_device=$PEER_DEVICE
timeout_sec=$TIMEOUT
block_session_probe=$BLOCK_SESSION_PROBE
ffn_prerouter=$FFN_PREROUTER
EOF

env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  HOME="$LOCAL_HOME" \
  DS4_GLM5_MODEL="$MODEL" \
  DS4_GLM5_MLA_COMPOSE_ORACLE_PREFIX="$ORACLE" \
  DS4_GLM5_TP_ROLE=leader \
  DS4_GLM5_TP_HOST="$HOST" \
  DS4_GLM5_TP_PORT="$PORT" \
  DS4_GLM5_TP_RDMA_DEVICE="$LOCAL_DEVICE" \
  DS4_GLM5_TP_CONNECT_TIMEOUT_SEC="$TIMEOUT" \
  DS4_GLM5_BLOCK_SESSION_PROBE="$BLOCK_SESSION_PROBE" \
  DS4_GLM5_BLOCK_FFN_PREROUTER="$FFN_PREROUTER" \
  "$BINARY" >"$OUT/leader.log" 2>&1 &
leader_pid=$!
ssh -o BatchMode=yes "$PEER" \
  "env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME='$PEER_HOME' DS4_GLM5_MODEL='$PEER_MODEL' DS4_GLM5_MLA_COMPOSE_ORACLE_PREFIX='$PEER_ORACLE' DS4_GLM5_TP_ROLE=worker DS4_GLM5_TP_HOST='$HOST' DS4_GLM5_TP_PORT='$PORT' DS4_GLM5_TP_RDMA_DEVICE='$PEER_DEVICE' DS4_GLM5_TP_CONNECT_TIMEOUT_SEC='$TIMEOUT' DS4_GLM5_BLOCK_SESSION_PROBE='$BLOCK_SESSION_PROBE' DS4_GLM5_BLOCK_FFN_PREROUTER='$FFN_PREROUTER' '$PEER_BINARY'" \
  >"$OUT/worker.log" 2>&1 &
worker_pid=$!
set +e
wait "$leader_pid"; leader_rc=$?
wait "$worker_pid"; worker_rc=$?
set -e
if (( leader_rc != 0 || worker_rc != 0 )); then
  echo "error: MLA RoCE gate failed (leader=$leader_rc worker=$worker_rc)" >&2
  tail -n 40 "$OUT/leader.log" "$OUT/worker.log" >&2
  exit 1
fi

for log in "$OUT/leader.log" "$OUT/worker.log"; do
  require_log "$log" 'transport=rdma' 'mandatory RDMA proof'
  require_log "$log" 'rdma GID index 3 (RoCE v2)' 'RoCE v2 proof'
  require_log "$log" 'mlx5 queue pair uses RC' 'RC proof'
  require_log "$log" 'registered host slab as 3 MRs' 'three-MR proof'
  require_log "$log" 'GLM5 MLA TP RoCE .*bytes=16384 direct=1' \
    'direct attention-output exchange'
  require_log "$log" 'GLM5 MLA RoCE upload .*sum_fnv=.*peer_fnv=' \
    'poisoned GPU upload and peer-consumption proof'
  require_log "$log" 'PASS same-GGUF GLM5 block-3 mHC-to-sparse-MLA gate' \
    'composition PASS marker'
done
if [[ $BLOCK_SESSION_PROBE == 1 ]]; then
  for log in "$OUT/leader.log" "$OUT/worker.log"; do
    require_log "$log" 'GLM5 block-session RoCE .*layer=3 seq=2 .*direct=1' \
      'second direct block-stage exchange'
  done
fi
if [[ $FFN_PREROUTER == 1 ]]; then
  for log in "$OUT/leader.log" "$OUT/worker.log"; do
    require_log "$log" 'GLM5 block FFN prerouter source=serial' \
      'serial FFN prerouter control'
    require_log "$log" 'GLM5 block FFN prerouter source=roce' \
      'RoCE-carried FFN prerouter path'
  done
  LEADER_SERIAL_ROUTE=$(line_field "$OUT/leader.log" \
    'GLM5 block FFN prerouter source=serial' route_fnv)
  LEADER_ROCE_ROUTE=$(line_field "$OUT/leader.log" \
    'GLM5 block FFN prerouter source=roce' route_fnv)
  WORKER_SERIAL_ROUTE=$(line_field "$OUT/worker.log" \
    'GLM5 block FFN prerouter source=serial' route_fnv)
  WORKER_ROCE_ROUTE=$(line_field "$OUT/worker.log" \
    'GLM5 block FFN prerouter source=roce' route_fnv)
  [[ $LEADER_SERIAL_ROUTE == "$LEADER_ROCE_ROUTE" &&
     $WORKER_SERIAL_ROUTE == "$WORKER_ROCE_ROUTE" &&
     $LEADER_ROCE_ROUTE == "$WORKER_ROCE_ROUTE" ]] || {
    echo "error: serial/RoCE FFN prerouter hash chain did not close" >&2
    exit 1
  }
fi
require_log "$OUT/leader.log" "role=leader device=$LOCAL_DEVICE" \
  'leader ownership'
require_log "$OUT/worker.log" "role=worker device=$PEER_DEVICE" \
  'worker ownership'

LEADER_LOCAL=$(line_field "$OUT/leader.log" 'GLM5 MLA TP RoCE' local_fnv)
LEADER_PEER=$(line_field "$OUT/leader.log" 'GLM5 MLA TP RoCE' peer_fnv)
LEADER_SUM=$(line_field "$OUT/leader.log" 'GLM5 MLA TP RoCE' composed_fnv)
LEADER_UPLOAD=$(field "$OUT/leader.log" sum_fnv)
WORKER_LOCAL=$(line_field "$OUT/worker.log" 'GLM5 MLA TP RoCE' local_fnv)
WORKER_PEER=$(line_field "$OUT/worker.log" 'GLM5 MLA TP RoCE' peer_fnv)
WORKER_SUM=$(line_field "$OUT/worker.log" 'GLM5 MLA TP RoCE' composed_fnv)
WORKER_UPLOAD=$(field "$OUT/worker.log" sum_fnv)
[[ $LEADER_LOCAL == "$WORKER_PEER" &&
   $WORKER_LOCAL == "$LEADER_PEER" &&
   $LEADER_LOCAL != "$WORKER_LOCAL" &&
   $LEADER_SUM == "$WORKER_SUM" &&
   $LEADER_UPLOAD == "$LEADER_SUM" &&
   $WORKER_UPLOAD == "$WORKER_SUM" ]] || {
  echo "error: MLA RoCE payload hash chain did not close" >&2
  exit 1
}
if [[ $BLOCK_SESSION_PROBE == 1 ]]; then
  LEADER_BLOCK_LOCAL=$(line_field "$OUT/leader.log" \
    'GLM5 block-session RoCE' local_fnv)
  LEADER_BLOCK_PEER=$(line_field "$OUT/leader.log" \
    'GLM5 block-session RoCE' peer_fnv)
  LEADER_BLOCK_SUM=$(line_field "$OUT/leader.log" \
    'GLM5 block-session RoCE' composed_fnv)
  WORKER_BLOCK_LOCAL=$(line_field "$OUT/worker.log" \
    'GLM5 block-session RoCE' local_fnv)
  WORKER_BLOCK_PEER=$(line_field "$OUT/worker.log" \
    'GLM5 block-session RoCE' peer_fnv)
  WORKER_BLOCK_SUM=$(line_field "$OUT/worker.log" \
    'GLM5 block-session RoCE' composed_fnv)
  [[ $LEADER_BLOCK_LOCAL == "$WORKER_BLOCK_PEER" &&
     $WORKER_BLOCK_LOCAL == "$LEADER_BLOCK_PEER" &&
     $LEADER_BLOCK_LOCAL != "$WORKER_BLOCK_LOCAL" &&
     $LEADER_BLOCK_SUM == "$WORKER_BLOCK_SUM" ]] || {
    echo "error: second block-stage RoCE hash chain did not close" >&2
    exit 1
  }
fi

printf 'PASS GLM5 MLA RoCE component tag=%s binary_sha256=%s composed_fnv=%s\n' \
  "$TAG" "$LOCAL_SHA" "$LEADER_SUM"
