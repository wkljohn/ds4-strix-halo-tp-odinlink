#!/bin/bash
# Manage one 256K TP=2 DS4 API backend behind a local Caddy reverse proxy.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)
# shellcheck disable=SC1091
source "$REPO/scripts/ds4-research-root.sh"
CONFIG=${DS4_DEPLOY_CONFIG:-$SCRIPT_DIR/config.env.local}
[[ -r $CONFIG ]] || {
  echo "error: copy $SCRIPT_DIR/config.env.example to $CONFIG and edit it" >&2
  exit 2
}
# shellcheck disable=SC1090
source "$CONFIG"
ds4_resolve_research_roots "$REPO"

: "${MODEL:?MODEL is required}"
: "${PEER_MGMT:?PEER_MGMT is required}"
: "${PEER_REPO:?PEER_REPO is required}"
: "${COORDINATOR_RDMA_ADDR:?COORDINATOR_RDMA_ADDR is required}"

RDMA_PROFILE=${RDMA_PROFILE:-odinlink}
case $RDMA_PROFILE in
  odinlink)
    : "${ODINLINK_ROOT:?ODINLINK_ROOT is required for OdinLink}"
    LOCAL_RDMA_DEVICE=${LOCAL_RDMA_DEVICE:-odl_tb5_0}
    PEER_RDMA_DEVICE=${PEER_RDMA_DEVICE:-odl_tb5_0}
    ;;
  roce-v2)
    LOCAL_RDMA_DEVICE=${LOCAL_RDMA_DEVICE:-mlx5_0}
    PEER_RDMA_DEVICE=${PEER_RDMA_DEVICE:-mlx5_1}
    RDMA_GID_INDEX=${RDMA_GID_INDEX:-3}
    ;;
  *)
    echo "error: RDMA_PROFILE must be odinlink or roce-v2" >&2
    exit 2
    ;;
esac
CONTEXT=${CONTEXT:-262144}
if [[ -z ${PREFILL_CHUNK:-} ]]; then
  if [[ $RDMA_PROFILE == roce-v2 ]]; then PREFILL_CHUNK=2048; else PREFILL_CHUNK=4096; fi
fi
TP_TIMEOUT_SEC=${TP_TIMEOUT_SEC:-60}
DEFAULT_TEMPERATURE=${DEFAULT_TEMPERATURE:-0}
DSPARK=${DSPARK:-0}
GLM5_ENABLE_ORDINARY=${GLM5_ENABLE_ORDINARY:-0}
GLM5_FULL_LOGITS=${GLM5_FULL_LOGITS:-0}
PREFILL_FFN_WAVEFRONT=${PREFILL_FFN_WAVEFRONT:-1}
Q8_M256_K128=${Q8_M256_K128:-1}
HC_STAGE_EXACT_COOP=${HC_STAGE_EXACT_COOP:-1}
INDEXER_TOPK_RADIX_TREE=${INDEXER_TOPK_RADIX_TREE:-1}
Q4K_KSHARD_RESEARCH=${Q4K_KSHARD_RESEARCH:-1}
ATTN_DECODE_SEQTILE_RESEARCH=${ATTN_DECODE_SEQTILE_RESEARCH:-8}
if [[ -z ${EXPERT_SPLIT:-} ]]; then
  if [[ $DSPARK == 1 ]]; then EXPERT_SPLIT=118; else EXPERT_SPLIT=128; fi
fi
TP_PORT=${TP_PORT:-9000}
API_HOST=${API_HOST:-127.0.0.1}
API_PORT=${API_PORT:-8090}
PEER_HOST_KEY_ALIAS=${PEER_HOST_KEY_ALIAS:-$PEER_MGMT}
RUNTIME=$SCRIPT_DIR/runtime
LOCAL_PIDFILE=$RUNTIME/coordinator.pid
COORD_UNIT=${COORD_UNIT:-ds4-tp-coordinator}
WORKER_PIDFILE=$DS4_PEER_RESEARCH_ROOT/deployment/worker.pid
LOCAL_LOG=$DS4_RESEARCH_ROOT/deployment/coordinator.log
WORKER_LOG=$DS4_PEER_RESEARCH_ROOT/deployment/worker.log
if [[ $RDMA_PROFILE == odinlink ]]; then
  VERBS_LIB=$ODINLINK_ROOT/build/verbs/libodl_tb5_verbs.so.0.1.0
  ODL_LD_PATH=$ODINLINK_ROOT/build/lib:$ODINLINK_ROOT/build/verbs
fi
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes \
     -o "HostKeyAlias=$PEER_HOST_KEY_ALIAS" "$PEER_MGMT")

is_uint() { [[ $1 =~ ^[1-9][0-9]*$ ]]; }
is_uint "$CONTEXT" && is_uint "$PREFILL_CHUNK" && is_uint "$EXPERT_SPLIT" &&
  is_uint "$TP_TIMEOUT_SEC" && is_uint "$TP_PORT" &&
  is_uint "$API_PORT" || {
    echo "error: context, chunk, expert split, TP timeout, and ports must be positive integers" >&2
    exit 2
  }
(( EXPERT_SPLIT < 256 )) || { echo "error: expert split must be in 1..255" >&2; exit 2; }
[[ $DSPARK == 0 || $DSPARK == 1 ]] || { echo "error: DSPARK must be 0 or 1" >&2; exit 2; }
[[ $GLM5_ENABLE_ORDINARY == 0 || $GLM5_ENABLE_ORDINARY == 1 ]] || {
  echo "error: GLM5_ENABLE_ORDINARY must be 0 or 1" >&2; exit 2;
}
[[ $GLM5_FULL_LOGITS == 0 || $GLM5_FULL_LOGITS == 1 ]] || {
  echo "error: GLM5_FULL_LOGITS must be 0 or 1" >&2; exit 2;
}
[[ $PREFILL_FFN_WAVEFRONT == 0 || $PREFILL_FFN_WAVEFRONT == 1 ]] || {
  echo "error: PREFILL_FFN_WAVEFRONT must be 0 or 1" >&2; exit 2;
}
[[ $Q8_M256_K128 == 0 || $Q8_M256_K128 == 1 ]] || {
  echo "error: Q8_M256_K128 must be 0 or 1" >&2; exit 2;
}
[[ $HC_STAGE_EXACT_COOP == 0 || $HC_STAGE_EXACT_COOP == 1 ]] || {
  echo "error: HC_STAGE_EXACT_COOP must be 0 or 1" >&2; exit 2;
}
[[ $INDEXER_TOPK_RADIX_TREE == 0 || $INDEXER_TOPK_RADIX_TREE == 1 ]] || {
  echo "error: INDEXER_TOPK_RADIX_TREE must be 0 or 1" >&2; exit 2;
}
[[ $Q4K_KSHARD_RESEARCH == 0 || $Q4K_KSHARD_RESEARCH == 1 ]] || {
  echo "error: Q4K_KSHARD_RESEARCH must be 0 or 1" >&2; exit 2;
}
[[ $ATTN_DECODE_SEQTILE_RESEARCH =~ ^[0-9]+$ ]] || {
  echo "error: ATTN_DECODE_SEQTILE_RESEARCH must be a non-negative integer" >&2; exit 2;
}
if [[ $DSPARK == 1 ]]; then : "${MTP:?MTP is required when DSPARK=1}"; fi
(( CONTEXT == 262144 )) || echo "warning: deployment context is $CONTEXT, not 262144" >&2

sample_fingerprint() {
  local path=$1 size middle tail
  size=$(stat -c %s "$path")
  middle=$((size > 8388608 ? size / 2 - 4194304 : 0))
  tail=$((size > 8388608 ? size - 8388608 : 0))
  {
    printf '%s\n' "$size"
    dd if="$path" iflag=skip_bytes,count_bytes count=8388608 status=none
    dd if="$path" iflag=skip_bytes,count_bytes skip="$middle" count=8388608 status=none
    dd if="$path" iflag=skip_bytes,count_bytes skip="$tail" count=8388608 status=none
  } | sha256sum | awk '{print $1}'
}

remote_fingerprint() {
  local path=$1 quoted
  printf -v quoted '%q' "$path"
  "${SSH[@]}" "p=$quoted; s=\$(stat -c %s \"\$p\"); m=\$((s > 8388608 ? s / 2 - 4194304 : 0)); t=\$((s > 8388608 ? s - 8388608 : 0)); { printf '%s\\n' \"\$s\"; dd if=\"\$p\" iflag=skip_bytes,count_bytes count=8388608 status=none; dd if=\"\$p\" iflag=skip_bytes,count_bytes skip=\"\$m\" count=8388608 status=none; dd if=\"\$p\" iflag=skip_bytes,count_bytes skip=\"\$t\" count=8388608 status=none; } | sha256sum | awk '{print \$1}'"
}

pid_matches() {
  local pidfile=$1 needle=$2 pid
  [[ -r $pidfile ]] || return 1
  pid=$(<"$pidfile")
  [[ $pid =~ ^[1-9][0-9]*$ && -r /proc/$pid/cmdline ]] || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq -- "$needle"
}

coord_is_active() {
  if [[ $RDMA_PROFILE == roce-v2 ]]; then
    sudo -n systemctl is-active --quiet "$COORD_UNIT.service"
  else
    systemctl --user is-active --quiet "$COORD_UNIT.service"
  fi
}

coord_main_pid() {
  if [[ $RDMA_PROFILE == roce-v2 ]]; then
    sudo -n systemctl show -p MainPID --value "$COORD_UNIT.service"
  else
    systemctl --user show -p MainPID --value "$COORD_UNIT.service"
  fi
}

coord_stop_service() {
  if [[ $RDMA_PROFILE == roce-v2 ]]; then
    sudo -n systemctl stop "$COORD_UNIT.service"
  else
    systemctl --user stop "$COORD_UNIT.service"
  fi
}

preflight() {
  [[ -x $REPO/ds4 && -x $REPO/ds4-server ]] || {
    echo "error: run 'make strix-halo' before deployment" >&2; exit 1;
  }
  [[ ${DS4_SERVER_SHA256:-} =~ ^[0-9a-fA-F]{64}$ ]] || {
    echo "error: set DS4_SERVER_SHA256 to 'sha256sum ./ds4-server' after the final build" >&2
    exit 1
  }
  local local_server_hash
  local_server_hash=$(sha256sum "$REPO/ds4-server" | awk '{print $1}')
  [[ $local_server_hash == "${DS4_SERVER_SHA256,,}" ]] || {
    echo "error: local ds4-server does not match DS4_SERVER_SHA256; rebuild and update the pin" >&2
    exit 1
  }
  [[ -r $MODEL ]] || { echo "error: local model is missing" >&2; exit 1; }
  if [[ $RDMA_PROFILE == odinlink ]]; then
    [[ -r /dev/odl_tb5_0 && -r $VERBS_LIB ]] || {
      echo "error: local OdinLink device or provider is missing" >&2; exit 1;
    }
    "${SSH[@]}" "test -x '$PEER_REPO/ds4' -a -r '$MODEL' -a -r /dev/odl_tb5_0 -a -r '$VERBS_LIB'" || {
      echo "error: peer binary, model, OdinLink device, or provider is missing" >&2; exit 1;
    }
  else
    sudo -n true || {
      echo "error: passwordless sudo is required to give the RoCE service unlimited memlock" >&2
      exit 1
    }
    grep -qx 'RoCE v2' \
      "/sys/class/infiniband/$LOCAL_RDMA_DEVICE/ports/1/gid_attrs/types/$RDMA_GID_INDEX" || {
      echo "error: local mlx5 GID is not available as RoCE v2" >&2; exit 1;
    }
    "${SSH[@]}" "test -x '$PEER_REPO/ds4' -a -r '$MODEL' && test \"\$(cat '/sys/class/infiniband/$PEER_RDMA_DEVICE/ports/1/gid_attrs/types/$RDMA_GID_INDEX')\" = 'RoCE v2'" || {
      echo "error: peer binary/model or RoCE v2 GID is unavailable" >&2; exit 1;
    }
    local peer_memlock
    peer_memlock=$("${SSH[@]}" 'ulimit -l')
    if [[ $peer_memlock != unlimited ]] &&
       { [[ ! $peer_memlock =~ ^[0-9]+$ ]] || (( peer_memlock < 131072 )); }; then
      echo "error: peer locked-memory limit is ${peer_memlock} KiB; RoCE deployment requires at least 128 MiB" >&2
      exit 1
    fi
  fi
  local local_bin peer_bin
  local_bin=$(sha256sum "$REPO/ds4" | awk '{print $1}')
  peer_bin=$("${SSH[@]}" "sha256sum '$PEER_REPO/ds4'" | awk '{print $1}')
  [[ $local_bin == "$peer_bin" ]] || { echo "error: node binaries differ" >&2; exit 1; }
  [[ $(sample_fingerprint "$MODEL") == "$(remote_fingerprint "$MODEL")" ]] || {
    echo "error: target-model fingerprints differ" >&2; exit 1;
  }
  if [[ $DSPARK == 1 ]]; then
    [[ -r $MTP ]] && "${SSH[@]}" "test -r '$MTP'" || {
      echo "error: local or peer drafter is missing" >&2; exit 1;
    }
    [[ $(sample_fingerprint "$MTP") == "$(remote_fingerprint "$MTP")" ]] || {
      echo "error: drafter fingerprints differ" >&2; exit 1;
    }
  fi
  systemctl is-active --quiet caddy || { echo "error: Caddy is not active" >&2; exit 1; }
  caddy validate --config /etc/caddy/Caddyfile >/dev/null
  mkdir -p "$RUNTIME" "$(dirname -- "$LOCAL_LOG")"
  printf -v peer_research_deploy_q '%q' "$DS4_PEER_RESEARCH_ROOT/deployment"
  "${SSH[@]}" "mkdir -p $peer_research_deploy_q"
}

start() {
  preflight
  pid_matches "$LOCAL_PIDFILE" "ds4-server" && {
    echo "error: owned coordinator is already running" >&2; exit 1;
  }
  coord_is_active && {
    echo "error: coordinator service $COORD_UNIT is already running" >&2; exit 1;
  }
  ss -ltnH "sport = :$API_PORT" | grep -q . && {
    echo "error: API port $API_PORT is already listening" >&2; exit 1;
  }
  "${SSH[@]}" "if test -r '$WORKER_PIDFILE'; then p=\$(cat '$WORKER_PIDFILE'); case \"\$p\" in ''|*[!0-9]*) exit 0;; esac; test -r /proc/\$p/cmdline && tr '\\0' ' ' < /proc/\$p/cmdline | grep -Fq -- 'ds4 --role worker' && exit 7; fi; exit 0" || {
    rc=$?; [[ $rc == 7 ]] && echo "error: owned worker is already running" >&2; exit "$rc";
  }

  local -a common worker coordinator decode_env support_args prefill_args
  local -a worker_rdma_args coordinator_rdma_args
  local routed_family tp_prefill_skip_unowned model_is_glm5=0
  if python3 "$REPO/scripts/check-glm5-next-gguf.py" "$MODEL" >/dev/null 2>&1; then
    model_is_glm5=1
  fi
  if (( model_is_glm5 == 0 )); then
    prefill_args=(--prefill-chunk "$PREFILL_CHUNK")
  fi
  if [[ $RDMA_PROFILE == odinlink ]]; then
    common=(env
      DS4_TP_ODINLINK_BATCH_ASYNC=1
      DS4_TP_VERBS_LIB="$VERBS_LIB"
      LD_LIBRARY_PATH="$ODL_LD_PATH")
    worker_rdma_args=(--rdma-device "$PEER_RDMA_DEVICE")
    coordinator_rdma_args=(--rdma-device "$LOCAL_RDMA_DEVICE")
  else
    common=(env -u DS4_TP_VERBS_LIB -u ODL_VERBS_WC_STREAM_COPY
                -u DS4_TP_ODINLINK_BATCH_ASYNC)
    worker_rdma_args=(--rdma-device "$PEER_RDMA_DEVICE"
                      --rdma-gid-index "$RDMA_GID_INDEX")
    coordinator_rdma_args=(--rdma-device "$LOCAL_RDMA_DEVICE"
                           --rdma-gid-index "$RDMA_GID_INDEX")
  fi
  common+=(
    DS4_TP_EXPERT_SPLIT="$EXPERT_SPLIT"
    DS4_TP_TIMEOUT_SEC="$TP_TIMEOUT_SEC"
    DS4_ROCM_ENABLE_Q8_F16_CACHE=0
    DS4_ROCM_STREAM_Q8_F16_CACHE_GB=0
    DS4_ROCM_Q8_DECODE_PAIR_DP4A=0
    DS4_ROCM_Q4K_DECODE_STAGE_XQ=1)
  if [[ $GLM5_ENABLE_ORDINARY == 1 ]]; then
    echo "warning: GLM-5.3 ordinary session integration is explicitly enabled; this is an experimental deployment" >&2
    common+=(DS4_GLM5_NEXT_ENABLE_ORDINARY=1)
  fi
  if [[ $GLM5_FULL_LOGITS == 1 ]]; then
    echo "warning: GLM full-logits TP mode is explicitly enabled; transport and memory cost may increase" >&2
    common+=(DS4_TP_RANK0_FULL_LOGITS=1 DS4_TP_GREEDY_TOP2=0)
  fi
  if [[ $DSPARK == 1 ]]; then
    echo "warning: DSpark is experimental and is not target-fingerprint exact" >&2
    decode_env=(
      DS4_DSPARK_SUPPORT_TOPK=6
      DS4_DSPARK_MAX_DRAFT_TOKENS=5
      DS4_ROCM_Q8_SMALL_BATCH_TILE=1
      DS4_ROCM_Q8_SMALL_BATCH_DP4A=1)
    support_args=(--mtp "$MTP" --dspark)
  else
    routed_family=$(python3 "$REPO/scripts/gguf_tensor_types.py" --routed-family "$MODEL") || {
      echo "error: unable to inspect routed-expert quantization in $MODEL" >&2
      exit 1
    }
    case $routed_family in
      Q4_K)
        tp_prefill_skip_unowned=1
        ;;
      HYBRID_Q2)
        tp_prefill_skip_unowned=0
        common+=(DS4_ROCM_TP_ZERO_WEIGHT_TILE_SKIP=1)
        ;;
      *)
        echo "error: unsupported routed-expert quantization: $routed_family" >&2
        exit 1
        ;;
    esac
    decode_env=(
      DS4_TP_GREEDY_TOP2=1
      DS4_TP_HOST_CALLBACK=1
      DS4_TP_PREFILL_FFN_WAVEFRONT="$PREFILL_FFN_WAVEFRONT"
      DS4_ROCM_TEMPORAL_COMPRESSOR=1
      DS4_ROCM_Q4K_DECODE_SPLIT_GATE_UP=1
      DS4_ROCM_Q8_BATCH_WMMA_M256_K128="$Q8_M256_K128"
      DS4_ROCM_Q4K_WMMA_PAIR_GATE_UP=1
      DS4_ROCM_Q4K_WMMA_FUSE_MID=1
      DS4_ROCM_TP_SKIP_UNOWNED=1
      DS4_ROCM_TP_PREFILL_SKIP_UNOWNED="$tp_prefill_skip_unowned"
      DS4_ROCM_SHARED_GU_SWIGLU_FUSE=1
      DS4_ROCM_HC_STAGE_EXACT_COOP="$HC_STAGE_EXACT_COOP"
      DS4_ROCM_INDEXER_TOPK_RADIX_TREE="$INDEXER_TOPK_RADIX_TREE"
      DS4_ROCM_Q4K_KSHARD_RESEARCH="$Q4K_KSHARD_RESEARCH"
      DS4_ROCM_ATTN_DECODE_SEQTILE_RESEARCH="$ATTN_DECODE_SEQTILE_RESEARCH")
    support_args=()
  fi
  common+=("${decode_env[@]}")
  worker=("${common[@]}" ./ds4 --role worker --tensor-parallel
    --coordinator "$COORDINATOR_RDMA_ADDR" "$TP_PORT" --transport rdma --rocm
    -m "$MODEL" -c "$CONTEXT" "${prefill_args[@]}"
    "${worker_rdma_args[@]}"
    "${support_args[@]}")
  coordinator=("${common[@]}")
  if [[ $DSPARK == 1 ]]; then coordinator+=(DS4_DSPARK_RESIDENT_Q8=1); fi
  coordinator+=(./ds4-server
    --role coordinator --tensor-parallel --listen 0.0.0.0 "$TP_PORT"
    --transport rdma --rocm -m "$MODEL" -c "$CONTEXT"
    "${coordinator_rdma_args[@]}"
    "${prefill_args[@]}" "${support_args[@]}"
    --default-temperature "$DEFAULT_TEMPERATURE"
    --host "$API_HOST" --port "$API_PORT")

  local worker_q repo_q log_q pid_q
  printf -v worker_q '%q ' "${worker[@]}"
  printf -v repo_q '%q' "$PEER_REPO"
  printf -v log_q '%q' "$WORKER_LOG"
  printf -v pid_q '%q' "$WORKER_PIDFILE"
  # Keep the background operator scoped to ds4 itself. With `cd && cmd &`,
  # POSIX shells background the whole AND-list and `$!` names a wrapper shell.
  "${SSH[@]}" "cd $repo_q || exit 1; nohup setsid $worker_q >$log_q 2>&1 </dev/null & p=\$!; echo \$p >$pid_q"

  # RoCE registration needs more than the user manager's inherited 8 MiB hard
  # memlock limit. A system transient unit can genuinely raise that limit;
  # `systemctl --user show` only reports the requested, not effective, value.
  if [[ $RDMA_PROFILE == roce-v2 ]]; then
    sudo -n systemd-run --unit="$COORD_UNIT" --collect --service-type=exec \
      --uid="$(id -u)" --gid="$(id -g)" \
      --property="WorkingDirectory=$REPO" \
      --property="LimitMEMLOCK=infinity" \
      --property="StandardOutput=append:$LOCAL_LOG" \
      --property="StandardError=append:$LOCAL_LOG" \
      "${coordinator[@]}" >/dev/null
  else
    systemd-run --user --unit="$COORD_UNIT" --collect --service-type=exec \
      --property="WorkingDirectory=$REPO" \
      --property="StandardOutput=append:$LOCAL_LOG" \
      --property="StandardError=append:$LOCAL_LOG" \
      "${coordinator[@]}" >/dev/null
  fi
  local coord_pid
  coord_pid=$(coord_main_pid)
  [[ $coord_pid =~ ^[1-9][0-9]*$ ]] || {
    echo "error: coordinator service failed to start" >&2
    return 1
  }
  if [[ $RDMA_PROFILE == roce-v2 ]]; then
    local effective_memlock
    effective_memlock=$(awk '$1 == "Max" && $2 == "locked" && $3 == "memory" { print $4 }' "/proc/$coord_pid/limits")
    [[ $effective_memlock == unlimited ]] || {
      echo "error: coordinator effective memlock is $effective_memlock, expected unlimited" >&2
      coord_stop_service
      return 1
    }
  fi
  printf '%s\n' "$coord_pid" >"$LOCAL_PIDFILE"
  echo "started both ranks over $RDMA_PROFILE; model loading can take several minutes"
  echo "follow: $0 logs"
}

stop() {
  if coord_is_active; then
    coord_stop_service
    echo "stopped coordinator service $COORD_UNIT"
  elif pid_matches "$LOCAL_PIDFILE" "ds4-server"; then
    pid=$(<"$LOCAL_PIDFILE"); kill -TERM "$pid"; echo "stopped coordinator $pid"
  fi
  "${SSH[@]}" "if test -r '$WORKER_PIDFILE'; then p=\$(cat '$WORKER_PIDFILE'); case \"\$p\" in ''|*[!0-9]*) exit 0;; esac; if test -r /proc/\$p/cmdline && tr '\\0' ' ' < /proc/\$p/cmdline | grep -Fq -- 'ds4 --role worker'; then kill -TERM \"\$p\"; echo stopped-worker-\$p; fi; fi"
}

status() {
  local coord_ok=0 worker_ok=0 api_ok=0 worker_state
  if coord_is_active; then
    echo "coordinator: running pid $(coord_main_pid)"
    coord_ok=1
  elif pid_matches "$LOCAL_PIDFILE" "ds4-server"; then
    echo "coordinator: running pid $(<"$LOCAL_PIDFILE")"
    coord_ok=1
  else
    echo "coordinator: stopped"
  fi
  worker_state=$("${SSH[@]}" "if test -r '$WORKER_PIDFILE'; then p=\$(cat '$WORKER_PIDFILE'); if test -r /proc/\$p/cmdline && tr '\\0' ' ' < /proc/\$p/cmdline | grep -Fq -- 'ds4 --role worker'; then echo worker-running-pid-\$p; else echo worker-stopped; fi; else echo worker-stopped; fi" 2>/dev/null) || worker_state=worker-unreachable
  echo "$worker_state"
  [[ $worker_state == worker-running-pid-* ]] && worker_ok=1
  if curl --silent --show-error --fail --max-time 3 "http://$API_HOST:$API_PORT/health" >/dev/null; then
    api_ok=1
    if ((coord_ok && worker_ok)); then
      echo "api: ready on $API_HOST:$API_PORT"
    else
      echo "api: listener healthy but TP pair incomplete"
    fi
  else
    echo "api: loading or unavailable"
  fi
  ((coord_ok && worker_ok && api_ok))
}

logs() {
  echo "== coordinator =="
  tail -n 40 "$LOCAL_LOG" 2>/dev/null || true
  echo "== worker =="
  "${SSH[@]}" "tail -n 40 '$WORKER_LOG' 2>/dev/null || true"
}

case ${1:-status} in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  logs) logs ;;
  *) echo "usage: $0 {start|stop|restart|status|logs}" >&2; exit 2 ;;
esac
