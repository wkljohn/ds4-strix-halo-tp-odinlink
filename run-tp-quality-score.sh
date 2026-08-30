#!/bin/bash
# Score tracked official continuations through a real TP=2 RDMA path.
# Usage: ./run-tp-quality-score.sh TAG MODEL.gguf [NAME=VALUE ...]
set -euo pipefail

TAG=${1:?usage: run-tp-quality-score.sh TAG MODEL.gguf [NAME=VALUE ...]}
MODEL=${2:?usage: run-tp-quality-score.sh TAG MODEL.gguf [NAME=VALUE ...]}
shift 2
EXTRA_ENV=("$@")
[[ $TAG =~ ^[A-Za-z0-9._-]+$ ]] || { echo "error: invalid tag" >&2; exit 2; }

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$REPO/scripts/ds4-research-root.sh"
BENCH_CONFIG=${DS4_BENCH_CONFIG:-$REPO/bench.env.local}
[[ ! -r $BENCH_CONFIG ]] || source "$BENCH_CONFIG"
ds4_resolve_research_roots "$REPO"
PEER_REPO=${DS4_PEER_REPO:-$REPO}
PEER_MGMT=${DS4_PEER_MGMT:-}
PEER_HOST_KEY_ALIAS=${DS4_PEER_HOST_KEY_ALIAS:-${PEER_MGMT#*@}}
COORDINATOR_ADDR=${DS4_COORDINATOR_ADDR:-}
PORT=${DS4_QUALITY_PORT:-9002}
CONTEXT=${DS4_QUALITY_CONTEXT:-4096}
MAX_CASES=${DS4_QUALITY_MAX_CASES:-100}
MANIFEST=${DS4_QUALITY_MANIFEST:-$REPO/gguf-tools/quality-testing/data/flash/manifest.tsv}
OUT=${DS4_QUALITY_OUT:-$DS4_RESEARCH_ROOT/accuracy-acceleration-2026-08-14}
PEER_OUT=${DS4_PEER_QUALITY_OUT:-$DS4_PEER_RESEARCH_ROOT/accuracy-acceleration-2026-08-14}
SCORER=$REPO/gguf-tools/quality-testing/score_official
COORD_LOG=$OUT/coordinator-$TAG.log
WORKER_LOG=$OUT/worker-$TAG.log
REMOTE_WORKER_LOG=$PEER_OUT/worker-$TAG.log
SCORES=$OUT/$TAG.tsv
SCORES_MANIFEST=$OUT/$TAG.manifest
WORKER_PIDFILE=$PEER_OUT/worker-$TAG.pid
PEER_SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes
  -o "HostKeyAlias=$PEER_HOST_KEY_ALIAS" "$PEER_MGMT")
PEER_SCP=(scp -o BatchMode=yes -o StrictHostKeyChecking=yes
  -o "HostKeyAlias=$PEER_HOST_KEY_ALIAS")
RDMA_PROFILE=${DS4_QUALITY_RDMA_PROFILE:-${DS4_BENCH_RDMA_PROFILE:-odinlink}}
case $RDMA_PROFILE in
  odinlink)
    LOCAL_RDMA_DEVICE=${DS4_LOCAL_RDMA_DEVICE:-odl_tb5_0}
    PEER_RDMA_DEVICE=${DS4_PEER_RDMA_DEVICE:-odl_tb5_0}
    ODINLINK_ROOT=${DS4_ODINLINK_ROOT:-}
    [[ -n $ODINLINK_ROOT ]] || {
      echo "error: OdinLink quality scoring requires DS4_ODINLINK_ROOT" >&2; exit 2;
    }
    VERBS_LIB=$ODINLINK_ROOT/build/verbs/libodl_tb5_verbs.so.0.1.0
    ODL_LD_PATH=$ODINLINK_ROOT/build/lib:$ODINLINK_ROOT/build/verbs
    ;;
  roce-v2)
    LOCAL_RDMA_DEVICE=${DS4_LOCAL_RDMA_DEVICE:-mlx5_0}
    PEER_RDMA_DEVICE=${DS4_PEER_RDMA_DEVICE:-mlx5_1}
    RDMA_GID_INDEX=${DS4_RDMA_GID_INDEX:-3}
    ;;
  *)
    echo "error: DS4_QUALITY_RDMA_PROFILE must be odinlink or roce-v2" >&2
    exit 2
    ;;
esac

for required in PEER_MGMT COORDINATOR_ADDR; do
  [[ -n ${!required} ]] || { echo "error: set $required in bench.env.local or the environment" >&2; exit 2; }
done

[[ $PORT =~ ^[1-9][0-9]*$ && $CONTEXT =~ ^[1-9][0-9]*$ && $MAX_CASES =~ ^[1-9][0-9]*$ ]] || {
  echo "error: invalid port, context, or max-case count" >&2; exit 2;
}
for kv in "${EXTRA_ENV[@]}"; do
  [[ $kv =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || { echo "error: expected NAME=VALUE: $kv" >&2; exit 2; }
  case $kv in
    DS4_ROCM_ENABLE_Q8_F16_CACHE=*|DS4_ROCM_STREAM_Q8_F16_CACHE_GB=*)
      echo "error: persistent Q8-to-FP16 caches are excluded from accuracy tests" >&2; exit 2 ;;
    DS4_TP_GREEDY_TOP2=*)
      echo "error: quality scoring requires full-vocabulary RDMA logits" >&2; exit 2 ;;
    DS4_TP_EXPERT_SPLIT=*)
      echo "error: the accuracy baseline requires the balanced 128/128 split" >&2; exit 2 ;;
  esac
done
[[ -r $MODEL && -r $MANIFEST && -x $SCORER && -x $REPO/ds4 ]] || {
  echo "error: missing model, manifest, scorer, or ds4 binary" >&2; exit 1;
}
if [[ $RDMA_PROFILE == odinlink ]]; then
  [[ -r /dev/odl_tb5_0 && -r $VERBS_LIB ]] || {
    echo "error: local OdinLink device or provider unavailable" >&2; exit 1;
  }
  "${PEER_SSH[@]}" "test -r '$MODEL' -a -r /dev/odl_tb5_0 -a -r '$VERBS_LIB' -a -x '$PEER_REPO/ds4'" || {
    echo "error: peer model, binary, OdinLink device, or provider unavailable" >&2; exit 1;
  }
else
  [[ $RDMA_GID_INDEX =~ ^[0-9]+$ && $RDMA_GID_INDEX -le 255 ]] || {
    echo "error: DS4_RDMA_GID_INDEX must be in 0..255" >&2; exit 2;
  }
  grep -qx 'RoCE v2' "/sys/class/infiniband/$LOCAL_RDMA_DEVICE/ports/1/gid_attrs/types/$RDMA_GID_INDEX" || {
    echo "error: local RoCE v2 GID unavailable" >&2; exit 1;
  }
  "${PEER_SSH[@]}" "test -r '$MODEL' -a -x '$PEER_REPO/ds4' && test \"\$(cat '/sys/class/infiniband/$PEER_RDMA_DEVICE/ports/1/gid_attrs/types/$RDMA_GID_INDEX')\" = 'RoCE v2'" || {
    echo "error: peer model, binary, or RoCE v2 GID unavailable" >&2; exit 1;
  }
fi

LOCAL_SIZE=$(stat -c %s "$MODEL")
PEER_SIZE=$("${PEER_SSH[@]}" "stat -c %s '$MODEL'")
[[ $LOCAL_SIZE == "$PEER_SIZE" ]] || { echo "error: model sizes differ" >&2; exit 1; }
sample_fingerprint() {
  local path=$1 size half tail
  size=$(stat -c %s "$path")
  half=$((size / 2 > 4194304 ? size / 2 - 4194304 : 0))
  tail=$((size > 8388608 ? size - 8388608 : 0))
  {
    printf '%s\n' "$size"
    dd if="$path" iflag=skip_bytes,count_bytes skip=0 count=8388608 status=none
    dd if="$path" iflag=skip_bytes,count_bytes skip="$half" count=8388608 status=none
    dd if="$path" iflag=skip_bytes,count_bytes skip="$tail" count=8388608 status=none
  } | sha256sum | awk '{print $1}'
}
LOCAL_MODEL_FINGERPRINT=$(sample_fingerprint "$MODEL")
printf -v MODEL_Q '%q' "$MODEL"
PEER_MODEL_FINGERPRINT=$("${PEER_SSH[@]}" "p=$MODEL_Q; s=\$(stat -c %s \"\$p\"); h=\$((s / 2 > 4194304 ? s / 2 - 4194304 : 0)); t=\$((s > 8388608 ? s - 8388608 : 0)); { printf '%s\\n' \"\$s\"; dd if=\"\$p\" iflag=skip_bytes,count_bytes skip=0 count=8388608 status=none; dd if=\"\$p\" iflag=skip_bytes,count_bytes skip=\"\$h\" count=8388608 status=none; dd if=\"\$p\" iflag=skip_bytes,count_bytes skip=\"\$t\" count=8388608 status=none; } | sha256sum | awk '{print \$1}'")
[[ $LOCAL_MODEL_FINGERPRINT == "$PEER_MODEL_FINGERPRINT" ]] || {
  echo "error: sampled model fingerprints differ" >&2; exit 1;
}
LOCAL_HASH=$(sha256sum "$REPO/ds4" | awk '{print $1}')
PEER_HASH=$("${PEER_SSH[@]}" "sha256sum '$PEER_REPO/ds4'" | awk '{print $1}')
[[ $LOCAL_HASH == "$PEER_HASH" ]] || { echo "error: rank binary hashes differ" >&2; exit 1; }
SOURCE_COMMIT=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)
if git -C "$REPO" diff --quiet --ignore-submodules -- 2>/dev/null &&
   git -C "$REPO" diff --cached --quiet --ignore-submodules -- 2>/dev/null &&
   [[ -z $(git -C "$REPO" ls-files --others --exclude-standard) ]]; then
  SOURCE_DIRTY=0
else
  SOURCE_DIRTY=1
fi

if pgrep -af '[s]core_official.*--role coordinator' >/dev/null; then
  echo "error: a quality coordinator is already running" >&2; exit 1
fi
if "${PEER_SSH[@]}" "pgrep -af '[d]s4 .*--role worker.*--tensor-parallel'" >/dev/null; then
  echo "error: a TP worker is already running on the peer" >&2; exit 1
fi

mkdir -p "$OUT"
printf -v PEER_OUT_Q '%q' "$PEER_OUT"
"${PEER_SSH[@]}" "mkdir -p $PEER_OUT_Q"
COMMON_ENV=(
  env -i PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8
  DS4_TP_TIMEOUT_SEC=60
  DS4_TP_RDMA_LOGITS=1
  DS4_ROCM_Q4K_DECODE_STAGE_XQ=1
  DS4_ROCM_Q8_DECODE_PAIR_DP4A=0
  DS4_ROCM_Q4K_DECODE_SPLIT_GATE_UP=1
  DS4_ROCM_TP_SKIP_UNOWNED=1
  DS4_ROCM_TP_PREFILL_SKIP_UNOWNED=0
  DS4_ROCM_TP_ZERO_WEIGHT_TILE_SKIP=1
  DS4_ROCM_SHARED_GU_SWIGLU_FUSE=1
  "${EXTRA_ENV[@]}"
)
LOCAL_RDMA_ARGS=(--rdma-device "$LOCAL_RDMA_DEVICE")
PEER_RDMA_ARGS=(--rdma-device "$PEER_RDMA_DEVICE")
if [[ $RDMA_PROFILE == odinlink ]]; then
  COMMON_ENV+=(
    DS4_TP_ODINLINK_BATCH_ASYNC=1
    DS4_TP_VERBS_LIB="$VERBS_LIB"
    LD_LIBRARY_PATH="$ODL_LD_PATH"
  )
else
  LOCAL_RDMA_ARGS+=(--rdma-gid-index "$RDMA_GID_INDEX")
  PEER_RDMA_ARGS+=(--rdma-gid-index "$RDMA_GID_INDEX")
fi
MODEL_ARCH=$(python3 "$REPO/scripts/gguf_tensor_types.py" --architecture "$MODEL") || {
  echo "error: unable to inspect model architecture" >&2; exit 1;
}
PREFILL_ARGS=(--prefill-chunk 4096)
if [[ $MODEL_ARCH == glm5-next ]]; then
  PREFILL_ARGS=()
  COMMON_ENV+=(
    DS4_GLM5_NEXT_ENABLE_ORDINARY=1
    DS4_TP_BIG_DIRECT=1
    DS4_TP_GREEDY_TOP2=0
    DS4_ROCM_TEMPORAL_COMPRESSOR=0
  )
fi

echo "quality_tag=$TAG"
echo "ds4_sha256=$LOCAL_HASH"
echo "scorer_sha256=$(sha256sum "$SCORER" | awk '{print $1}')"
echo "model_sample_sha256=$LOCAL_MODEL_FINGERPRINT"
echo "manifest_sha256=$(sha256sum "$MANIFEST" | awk '{print $1}')"

WORKER_STARTED=0
worker_running() {
  "${PEER_SSH[@]}" "test -r '$WORKER_PIDFILE' || exit 1; p=\$(cat '$WORKER_PIDFILE'); case \"\$p\" in ''|*[!0-9]*) exit 1;; esac; test -r /proc/\$p/cmdline || exit 1; tr '\\0' ' ' < /proc/\$p/cmdline | grep -q -- './ds4 --role worker --tensor-parallel'"
}
cleanup() {
  local rc=$?
  if (( WORKER_STARTED == 1 )) && worker_running; then
    "${PEER_SSH[@]}" "p=\$(cat '$WORKER_PIDFILE'); tr '\\0' ' ' < /proc/\$p/cmdline | grep -q -- './ds4 --role worker --tensor-parallel' && kill -TERM \"\$p\"" || true
  fi
  return "$rc"
}
trap cleanup EXIT

WORKER_APP=(./ds4 --role worker --tensor-parallel --coordinator "$COORDINATOR_ADDR" "$PORT"
  --transport rdma --rocm -m "$MODEL" -c "$CONTEXT"
  "${PREFILL_ARGS[@]}" "${PEER_RDMA_ARGS[@]}")
printf -v WORKER_CMD_Q '%q ' "${COMMON_ENV[@]}" "${WORKER_APP[@]}"
printf -v PEER_REPO_Q '%q' "$PEER_REPO"
printf -v WORKER_LOG_Q '%q' "$REMOTE_WORKER_LOG"
printf -v WORKER_PIDFILE_Q '%q' "$WORKER_PIDFILE"
"${PEER_SSH[@]}" "cd $PEER_REPO_Q || exit 1; nohup setsid $WORKER_CMD_Q > $WORKER_LOG_Q 2>&1 < /dev/null & p=\$!; echo \$p > $WORKER_PIDFILE_Q"
WORKER_STARTED=1

"${COMMON_ENV[@]}" "$SCORER" "$MODEL" "$MANIFEST" "$SCORES" "$CONTEXT" \
  --max-cases "$MAX_CASES" --role coordinator --tensor-parallel \
  --listen 0.0.0.0 "$PORT" --transport rdma --rocm \
  "${LOCAL_RDMA_ARGS[@]}" >"$COORD_LOG" 2>&1

for _ in $(seq 1 180); do worker_running || break; sleep 1; done
worker_running && { echo "error: worker did not exit after STOP" >&2; exit 1; }
WORKER_STARTED=0
trap - EXIT
"${PEER_SCP[@]}" "$PEER_MGMT:$REMOTE_WORKER_LOG" "$WORKER_LOG"

[[ $(awk 'NR > 1 {n++} END {print n+0}' "$SCORES") == "$MAX_CASES" ]] || {
  echo "error: incomplete score table" >&2; exit 1;
}
for log in "$COORD_LOG" "$WORKER_LOG"; do
  grep -q 'transport=rdma' "$log" || { echo "error: RDMA not confirmed in $log" >&2; exit 1; }
  ! grep -Eq 'fallback_calls=[1-9]|transport fallback|Connection timed out|Protocol error' "$log" || {
    echo "error: fallback or transport failure in $log" >&2; exit 1;
  }
done
TARGET_TOKENS=$(awk -F'\t' 'NR > 1 {sum += $3} END {print sum+0}' "$SCORES")
printf -v EXTRA_ENV_Q '%q ' "${EXTRA_ENV[@]}"
{
  printf 'tag=%s\n' "$TAG"
  printf 'source_commit=%s\n' "$SOURCE_COMMIT"
  printf 'source_dirty=%s\n' "$SOURCE_DIRTY"
  printf 'model=%s\n' "$MODEL"
  printf 'model_size=%s\n' "$LOCAL_SIZE"
  printf 'model_sample_sha256=%s\n' "$LOCAL_MODEL_FINGERPRINT"
  printf 'ds4_sha256=%s\n' "$LOCAL_HASH"
  printf 'scorer_sha256=%s\n' "$(sha256sum "$SCORER" | awk '{print $1}')"
  printf 'quality_input_sha256=%s\n' "$(sha256sum "$MANIFEST" | awk '{print $1}')"
  printf 'cases=%s\n' "$MAX_CASES"
  printf 'target_tokens=%s\n' "$TARGET_TOKENS"
  printf 'context=%s\n' "$CONTEXT"
  printf 'rdma_profile=%s\n' "$RDMA_PROFILE"
  printf 'local_rdma_device=%s\n' "$LOCAL_RDMA_DEVICE"
  printf 'peer_rdma_device=%s\n' "$PEER_RDMA_DEVICE"
  printf 'rdma_gid_index=%s\n' "${RDMA_GID_INDEX:-n/a}"
  printf 'dspark=0\n'
  printf 'extra_env=%s\n' "$EXTRA_ENV_Q"
} > "$SCORES_MANIFEST"
echo "scores=$SCORES"
echo "scores_manifest=$SCORES_MANIFEST"
grep -E 'cases=|api_ref_tokens=' "$COORD_LOG" | tail -2
echo QUALITY_RUN_DONE
