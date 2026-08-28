#!/usr/bin/env bash
# Two-host production-executor gate: token 42 through dense layers 0-2 and
# the first GLM-5.3 MLA+routed layer over mandatory RoCE v2.
set -euo pipefail

if [[ $# != 4 ]]; then
  echo "usage: $0 TAG LOCAL_MODEL PEER_SSH PEER_MODEL" >&2
  exit 2
fi
TAG=$1
MODEL=$2
PEER=$3
PEER_MODEL=$4
[[ $TAG =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "error: invalid tag" >&2
  exit 2
}
for value in "$MODEL" "$PEER" "$PEER_MODEL"; do
  [[ $value != *"'"* && $value != *$'\n'* ]] || {
    echo "error: arguments may not contain quotes or newlines" >&2
    exit 2
  }
done
[[ -f $MODEL ]] || { echo "error: missing local model" >&2; exit 2; }

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$REPO/scripts/ds4-research-root.sh"
ds4_resolve_research_roots "$REPO"

HOST=${DS4_COORDINATOR_ADDR:-192.168.99.1}
PORT=${DS4_GLM5_PREFIX_TP_PORT:-15883}
LOCAL_DEVICE=${DS4_LOCAL_RDMA_DEVICE:-mlx5_0}
PEER_DEVICE=${DS4_PEER_RDMA_DEVICE:-mlx5_1}
TIMEOUT=${DS4_GLM5_TP_CONNECT_TIMEOUT_SEC:-180}
FULL_TRUNK=${DS4_GLM5_FULL_TRUNK:-0}
PEER_DIR=${DS4_GLM5_PEER_TEST_DIR:-/home/wkljohn/Desktop/cc/glm5-node2-test/prefix-layer3}
BINARY=$REPO/tests/test_rocm_glm5_prefix_layer3_tp
PEER_BINARY=$PEER_DIR/test_rocm_glm5_prefix_layer3_tp
OUT=$DS4_RESEARCH_ROOT/glm5-next-tp2/$TAG
LOCAL_HOME=${HOME:?HOME is required}
PEER_HOME=$(ssh -o BatchMode=yes "$PEER" 'printf %s "$HOME"')

for value in "$HOST" "$LOCAL_DEVICE" "$PEER_DEVICE" "$PEER_DIR" \
             "$LOCAL_HOME" "$PEER_HOME"; do
  [[ $value != *"'"* && $value != *$'\n'* ]] || {
    echo "error: environment values may not contain quotes or newlines" >&2
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
[[ $FULL_TRUNK == 0 || $FULL_TRUNK == 1 ]] || {
  echo "error: DS4_GLM5_FULL_TRUNK must be 0 or 1" >&2
  exit 2
}

read_local_memory_layout() {
  local mem_kib gtt_bytes carveout
  mem_kib=$(awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo)
  gtt_bytes=$(cat /sys/class/drm/card0/device/mem_info_gtt_total)
  carveout=$(cat /sys/class/drm/card0/device/uma/carveout)
  [[ $mem_kib =~ ^[0-9]+$ && $gtt_bytes =~ ^[0-9]+$ &&
     $carveout =~ ^[0-9]+$ ]] || return 1
  printf '%s %s %s\n' "$mem_kib" "$gtt_bytes" "$carveout"
}

read_peer_memory_layout() {
  ssh -o BatchMode=yes "$PEER" \
    "mem_kib=\$(awk '/^MemTotal:/ { print \$2; exit }' /proc/meminfo); gtt_bytes=\$(cat /sys/class/drm/card0/device/mem_info_gtt_total); carveout=\$(cat /sys/class/drm/card0/device/uma/carveout); test -n \"\$mem_kib\" -a -n \"\$gtt_bytes\" -a -n \"\$carveout\" || exit 1; case \"\$mem_kib \$gtt_bytes \$carveout\" in *[!0-9\ ]*) exit 1;; esac; printf '%s %s %s\\n' \"\$mem_kib\" \"\$gtt_bytes\" \"\$carveout\""
}

LOCAL_MEM_KIB=0 LOCAL_GTT_BYTES=0 LOCAL_CARVEOUT=-1
PEER_MEM_KIB=0 PEER_GTT_BYTES=0 PEER_CARVEOUT=-1
if [[ $FULL_TRUNK == 1 ]]; then
  read -r LOCAL_MEM_KIB LOCAL_GTT_BYTES LOCAL_CARVEOUT \
    < <(read_local_memory_layout)
  read -r PEER_MEM_KIB PEER_GTT_BYTES PEER_CARVEOUT \
    < <(read_peer_memory_layout)
  min_mem_kib=$((112 * 1024 * 1024))
  min_gtt_bytes=$((112 * 1024 * 1024 * 1024))
  (( LOCAL_MEM_KIB >= min_mem_kib && PEER_MEM_KIB >= min_mem_kib &&
     LOCAL_GTT_BYTES >= min_gtt_bytes && PEER_GTT_BYTES >= min_gtt_bytes )) || {
    echo "error: full GLM5 trunk requires >=112 GiB real RAM and GTT on both hosts" >&2
    printf 'local mem_kib=%s gtt_bytes=%s carveout=%s; peer mem_kib=%s gtt_bytes=%s carveout=%s\n' \
      "$LOCAL_MEM_KIB" "$LOCAL_GTT_BYTES" "$LOCAL_CARVEOUT" \
      "$PEER_MEM_KIB" "$PEER_GTT_BYTES" "$PEER_CARVEOUT" >&2
    exit 1
  }
  (( LOCAL_CARVEOUT == 0 && PEER_CARVEOUT == 0 )) || {
    echo "error: full GLM5 trunk requires firmware UMA carveout option 0 (512 MiB) on both hosts" >&2
    printf 'local carveout=%s; peer carveout=%s\n' \
      "$LOCAL_CARVEOUT" "$PEER_CARVEOUT" >&2
    exit 1
  }
fi

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

mkdir -p "$(dirname -- "$OUT")"
[[ ! -e $OUT ]] || {
  echo "error: refusing to overwrite evidence directory $OUT" >&2
  exit 1
}
mkdir "$OUT"

make -C "$REPO" -j"$(nproc)" tests/test_rocm_glm5_prefix_layer3_tp
ssh -o BatchMode=yes "$PEER" "mkdir -p -- '$PEER_DIR'; test -f '$PEER_MODEL'"
scp -q -o BatchMode=yes "$BINARY" "$PEER:$PEER_BINARY"

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

printf '%s\n' \
  "tag=$TAG" \
  "source_head=$(git -C "$REPO" rev-parse HEAD)" \
  "test_source_sha256=$(sha256sum "$REPO/tests/test_rocm_glm5_prefix_layer3_tp.cu" | awk '{print $1}')" \
  "launcher_sha256=$(sha256sum "$REPO/scripts/run-glm5-prefix-layer3-roce.sh" | awk '{print $1}')" \
  "binary_sha256=$LOCAL_SHA" \
  "model_size=$LOCAL_SIZE" \
  "model_sample_sha256=$LOCAL_SAMPLE" \
  "host=$HOST" \
  "port=$PORT" \
  "local_device=$LOCAL_DEVICE" \
  "peer_device=$PEER_DEVICE" \
  "timeout_sec=$TIMEOUT" >"$OUT/run.env"
printf 'full_trunk=%s\n' "$FULL_TRUNK" >>"$OUT/run.env"
if [[ $FULL_TRUNK == 1 ]]; then
  printf '%s\n' \
    "local_mem_kib=$LOCAL_MEM_KIB" \
    "local_gtt_bytes=$LOCAL_GTT_BYTES" \
    "local_carveout_index=$LOCAL_CARVEOUT" \
    "peer_mem_kib=$PEER_MEM_KIB" \
    "peer_gtt_bytes=$PEER_GTT_BYTES" \
    "peer_carveout_index=$PEER_CARVEOUT" >>"$OUT/run.env"
fi

common_path=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
env -i PATH="$common_path" HOME="$LOCAL_HOME" \
  DS4_GLM5_MODEL="$MODEL" \
  DS4_GLM5_TP_ROLE=leader \
  DS4_GLM5_TP_HOST="$HOST" \
  DS4_GLM5_TP_PORT="$PORT" \
  DS4_GLM5_TP_RDMA_DEVICE="$LOCAL_DEVICE" \
  DS4_GLM5_TP_CONNECT_TIMEOUT_SEC="$TIMEOUT" \
  DS4_GLM5_FULL_TRUNK="$FULL_TRUNK" \
  "$BINARY" >"$OUT/leader.log" 2>&1 &
leader_pid=$!
ssh -o BatchMode=yes "$PEER" \
  "env -i PATH='$common_path' HOME='$PEER_HOME' DS4_GLM5_MODEL='$PEER_MODEL' DS4_GLM5_TP_ROLE=worker DS4_GLM5_TP_HOST='$HOST' DS4_GLM5_TP_PORT='$PORT' DS4_GLM5_TP_RDMA_DEVICE='$PEER_DEVICE' DS4_GLM5_TP_CONNECT_TIMEOUT_SEC='$TIMEOUT' DS4_GLM5_FULL_TRUNK='$FULL_TRUNK' '$PEER_BINARY'" \
  >"$OUT/worker.log" 2>&1 &
worker_pid=$!

set +e
wait "$leader_pid"; leader_rc=$?
wait "$worker_pid"; worker_rc=$?
set -e
if (( leader_rc != 0 || worker_rc != 0 )); then
  echo "error: prefix-layer3 gate failed (leader=$leader_rc worker=$worker_rc)" >&2
  tail -n 60 "$OUT/leader.log" "$OUT/worker.log" >&2
  exit 1
fi

for log in "$OUT/leader.log" "$OUT/worker.log"; do
  grep -q 'transport=rdma' "$log"
  grep -q 'rdma GID index 3 (RoCE v2)' "$log"
  grep -q 'mlx5 queue pair uses RC' "$log"
  grep -q 'registered host slab as 3 MRs' "$log"
  grep -q 'PASS GLM5 prefix->layer3' "$log"
  if [[ $FULL_TRUNK == 1 ]]; then
    grep -q 'packed_q4_bytes=85614133248 rdma=1' "$log"
    grep -q 'GLM5 full-trunk post-install' "$log"
  else
    grep -q 'window_cache_bytes=0 rdma=1' "$log"
  fi
done
LEADER_OUTPUT=$(sed -n 's/.* token0 role=.* output=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/leader.log" | tail -1)
WORKER_OUTPUT=$(sed -n 's/.* token0 role=.* output=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/worker.log" | tail -1)
[[ -n $LEADER_OUTPUT && $LEADER_OUTPUT == "$WORKER_OUTPUT" ]] || {
  echo "error: all-rank layer3 output hashes differ" >&2
  exit 1
}
if [[ $FULL_TRUNK == 1 ]]; then
  LEADER_TRUNK=$(sed -n 's/.* trunk_output=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/leader.log" | tail -1)
  WORKER_TRUNK=$(sed -n 's/.* trunk_output=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/worker.log" | tail -1)
  LEADER_LOGITS=$(sed -n 's/.* logits=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/leader.log" | tail -1)
  WORKER_LOGITS=$(sed -n 's/.* logits=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/worker.log" | tail -1)
  [[ -n $LEADER_TRUNK && $LEADER_TRUNK == "$WORKER_TRUNK" &&
     -n $LEADER_LOGITS && $LEADER_LOGITS == "$WORKER_LOGITS" ]] || {
    echo "error: all-rank full-trunk or logit hashes differ" >&2
    exit 1
  }
fi

printf 'PASS GLM5 prefix-layer3 RoCE tag=%s binary_sha256=%s output_fnv=%s\n' \
  "$TAG" "$LOCAL_SHA" "$LEADER_OUTPUT"
