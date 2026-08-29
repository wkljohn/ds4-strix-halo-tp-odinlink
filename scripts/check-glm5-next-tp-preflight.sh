#!/usr/bin/env bash
# Fail-closed preflight for the GLM-5.3 Q2 TP=2 RoCE gate.
set -euo pipefail

usage() {
    echo "usage: $0 LOCAL_MODEL PEER_SSH PEER_MODEL" >&2
    exit 2
}
[[ $# == 3 ]] || usage
LOCAL_MODEL=$1
PEER=$2
PEER_MODEL=$3
[[ -f $LOCAL_MODEL ]] || { echo "error: local model not found: $LOCAL_MODEL" >&2; exit 1; }
for v in "$LOCAL_MODEL" "$PEER_MODEL"; do
    [[ $v != *"'"* && $v != *$'\n'* ]] || { echo "error: unsafe model path" >&2; exit 2; }
done

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_SIZE=$(stat -c %s "$LOCAL_MODEL")
if ! PEER_INFO=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER" \
    "test -f '$PEER_MODEL' && stat -c '%s' '$PEER_MODEL' && df -Pk '$PEER_MODEL' | awk 'NR==2 {print \$4}'"); then
    echo "error: peer model is missing or cannot be inspected" >&2
    exit 1
fi
PEER_SIZE=$(printf '%s\n' "$PEER_INFO" | sed -n '1p')
PEER_FREE_KIB=$(printf '%s\n' "$PEER_INFO" | sed -n '2p' | tr -d ' ')
[[ $MODEL_SIZE == "$PEER_SIZE" ]] || {
    echo "error: model size mismatch between ranks" >&2
    exit 1
}
# Hash both independent filesystems concurrently; sequential hashing adds
# several unnecessary minutes on 90-GiB artifacts.
HASH_DIR=$(mktemp -d)
trap 'rm -f "$HASH_DIR/local" "$HASH_DIR/peer"; rmdir "$HASH_DIR"' EXIT
sha256sum "$LOCAL_MODEL" >"$HASH_DIR/local" &
LOCAL_HASH_PID=$!
ssh -o BatchMode=yes "$PEER" "sha256sum '$PEER_MODEL'" >"$HASH_DIR/peer" &
PEER_HASH_PID=$!
wait "$LOCAL_HASH_PID"
wait "$PEER_HASH_PID"
MODEL_SHA=$(awk '{print $1}' "$HASH_DIR/local")
PEER_SHA=$(awk '{print $1}' "$HASH_DIR/peer")
[[ $MODEL_SIZE == "$PEER_SIZE" && $MODEL_SHA == "$PEER_SHA" ]] || {
    echo "error: model size/hash mismatch between ranks" >&2
    printf 'local size=%s sha=%s\npeer  size=%s sha=%s\n' \
        "$MODEL_SIZE" "$MODEL_SHA" "$PEER_SIZE" "$PEER_SHA" >&2
    exit 1
}

GFX=$(rocminfo 2>/dev/null | awk '/Name:[[:space:]]+gfx/{print $2; exit}')
[[ $GFX == gfx1151 ]] || { echo "error: local ROCm target is '$GFX', expected gfx1151" >&2; exit 1; }
PEER_GFX=$(ssh -o BatchMode=yes "$PEER" \
    "rocminfo 2>/dev/null | awk '/Name:[[:space:]]+gfx/{print \$2; exit}'")
[[ $PEER_GFX == gfx1151 ]] || { echo "error: peer ROCm target is '$PEER_GFX', expected gfx1151" >&2; exit 1; }

GID_INDEX=${DS4_RDMA_GID_INDEX:-3}
LOCAL_DEV=${DS4_LOCAL_RDMA_DEVICE:-mlx5_0}
PEER_DEV=${DS4_PEER_RDMA_DEVICE:-mlx5_1}
LOCAL_GID="/sys/class/infiniband/$LOCAL_DEV/ports/1/gid_attrs/types/$GID_INDEX"
[[ -r $LOCAL_GID && $(cat "$LOCAL_GID") == "RoCE v2" ]] || {
    echo "error: local $LOCAL_DEV GID $GID_INDEX is not RoCE v2" >&2; exit 1;
}
ssh -o BatchMode=yes "$PEER" \
    "test \"\$(cat /sys/class/infiniband/$PEER_DEV/ports/1/gid_attrs/types/$GID_INDEX)\" = 'RoCE v2'" || {
    echo "error: peer $PEER_DEV GID $GID_INDEX is not RoCE v2" >&2; exit 1;
}

LOCAL_FREE_KIB=$(df -Pk "$LOCAL_MODEL" | awk 'NR==2 {print $4}')
[[ $LOCAL_FREE_KIB =~ ^[0-9]+$ && $PEER_FREE_KIB =~ ^[0-9]+$ ]] || {
    echo "error: could not read free-space figures" >&2; exit 1;
}
printf 'PASS GLM5 TP preflight\nmodel_bytes=%s\nmodel_sha256=%s\ngfx=%s/%s\n' \
    "$MODEL_SIZE" "$MODEL_SHA" "$GFX" "$PEER_GFX"
printf 'free_space_kib=%s/%s rdma=%s/%s gid=%s\n' \
    "$LOCAL_FREE_KIB" "$PEER_FREE_KIB" "$LOCAL_DEV" "$PEER_DEV" "$GID_INDEX"
printf 'next=run the staged Q2 TP harness only after both local model paths are identical\n'
