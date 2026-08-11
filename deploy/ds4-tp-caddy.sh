#!/bin/bash
# Manage one 256K TP=2 DS4 API backend behind a local Caddy reverse proxy.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)
CONFIG=${DS4_DEPLOY_CONFIG:-$SCRIPT_DIR/config.env.local}
[[ -r $CONFIG ]] || {
  echo "error: copy $SCRIPT_DIR/config.env.example to $CONFIG and edit it" >&2
  exit 2
}
# shellcheck disable=SC1090
source "$CONFIG"

: "${MODEL:?MODEL is required}"
: "${PEER_MGMT:?PEER_MGMT is required}"
: "${PEER_REPO:?PEER_REPO is required}"
: "${ODINLINK_ROOT:?ODINLINK_ROOT is required}"
: "${COORDINATOR_RDMA_ADDR:?COORDINATOR_RDMA_ADDR is required}"

CONTEXT=${CONTEXT:-262144}
PREFILL_CHUNK=${PREFILL_CHUNK:-4096}
EXPERT_SPLIT=${EXPERT_SPLIT:-118}
TP_TIMEOUT_SEC=${TP_TIMEOUT_SEC:-60}
DEFAULT_TEMPERATURE=${DEFAULT_TEMPERATURE:-0}
DSPARK=${DSPARK:-0}
TP_PORT=${TP_PORT:-9000}
API_HOST=${API_HOST:-127.0.0.1}
API_PORT=${API_PORT:-8090}
PEER_HOST_KEY_ALIAS=${PEER_HOST_KEY_ALIAS:-$PEER_MGMT}
RUNTIME=$SCRIPT_DIR/runtime
LOCAL_PIDFILE=$RUNTIME/coordinator.pid
WORKER_PIDFILE=$PEER_REPO/research-results/deployment/worker.pid
LOCAL_LOG=$REPO/research-results/deployment/coordinator.log
WORKER_LOG=$PEER_REPO/research-results/deployment/worker.log
VERBS_LIB=$ODINLINK_ROOT/build/verbs/libodl_tb5_verbs.so.0.1.0
ODL_LD_PATH=$ODINLINK_ROOT/build/lib:$ODINLINK_ROOT/build/verbs
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

preflight() {
  [[ -x $REPO/ds4 && -x $REPO/ds4-server ]] || {
    echo "error: run 'make strix-halo' before deployment" >&2; exit 1;
  }
  [[ -r $MODEL && -r /dev/odl_tb5_0 && -r $VERBS_LIB ]] || {
    echo "error: local model, OdinLink device, or provider is missing" >&2; exit 1;
  }
  "${SSH[@]}" "test -x '$PEER_REPO/ds4' -a -r '$MODEL' -a -r /dev/odl_tb5_0 -a -r '$VERBS_LIB'" || {
    echo "error: peer binary, model, OdinLink device, or provider is missing" >&2; exit 1;
  }
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
  "${SSH[@]}" "mkdir -p '$PEER_REPO/research-results/deployment'"
}

start() {
  preflight
  pid_matches "$LOCAL_PIDFILE" "ds4-server" && {
    echo "error: owned coordinator is already running" >&2; exit 1;
  }
  ss -ltnH "sport = :$API_PORT" | grep -q . && {
    echo "error: API port $API_PORT is already listening" >&2; exit 1;
  }
  "${SSH[@]}" "if test -r '$WORKER_PIDFILE'; then p=\$(cat '$WORKER_PIDFILE'); case \"\$p\" in ''|*[!0-9]*) exit 0;; esac; test -r /proc/\$p/cmdline && tr '\\0' ' ' < /proc/\$p/cmdline | grep -Fq -- 'ds4 --role worker' && exit 7; fi; exit 0" || {
    rc=$?; [[ $rc == 7 ]] && echo "error: owned worker is already running" >&2; exit "$rc";
  }

  local -a common worker coordinator decode_env support_args
  common=(env
    DS4_TP_EXPERT_SPLIT="$EXPERT_SPLIT"
    DS4_TP_TIMEOUT_SEC="$TP_TIMEOUT_SEC"
    DS4_TP_ODINLINK_BATCH_ASYNC=1
    DS4_TP_VERBS_LIB="$VERBS_LIB"
    LD_LIBRARY_PATH="$ODL_LD_PATH"
    DS4_ROCM_Q8_DECODE_PAIR_DP4A=0
    DS4_ROCM_Q4K_DECODE_STAGE_XQ=1)
  if [[ $DSPARK == 1 ]]; then
    echo "warning: DSpark is experimental and is not target-fingerprint exact" >&2
    decode_env=(
      DS4_DSPARK_SUPPORT_TOPK=6
      DS4_DSPARK_MAX_DRAFT_TOKENS=5
      DS4_ROCM_Q8_SMALL_BATCH_TILE=1
      DS4_ROCM_Q8_SMALL_BATCH_DP4A=1)
    support_args=(--mtp "$MTP" --dspark)
  else
    decode_env=(DS4_TP_RANK0_FULL_LOGITS=1)
    support_args=()
  fi
  common+=("${decode_env[@]}")
  worker=("${common[@]}" ./ds4 --role worker --tensor-parallel
    --coordinator "$COORDINATOR_RDMA_ADDR" "$TP_PORT" --transport rdma --rocm
    -m "$MODEL" -c "$CONTEXT" --prefill-chunk "$PREFILL_CHUNK"
    "${support_args[@]}")
  coordinator=("${common[@]}")
  if [[ $DSPARK == 1 ]]; then coordinator+=(DS4_DSPARK_RESIDENT_Q8=1); fi
  coordinator+=(./ds4-server
    --role coordinator --tensor-parallel --listen 0.0.0.0 "$TP_PORT"
    --transport rdma --rocm -m "$MODEL" -c "$CONTEXT"
    --prefill-chunk "$PREFILL_CHUNK" "${support_args[@]}"
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

  (cd "$REPO"; nohup setsid "${coordinator[@]}" >"$LOCAL_LOG" 2>&1 </dev/null & echo $! >"$LOCAL_PIDFILE")
  echo "started both ranks; model loading can take several minutes"
  echo "follow: $0 logs"
}

stop() {
  if pid_matches "$LOCAL_PIDFILE" "ds4-server"; then
    pid=$(<"$LOCAL_PIDFILE"); kill -TERM "$pid"; echo "stopped coordinator $pid"
  fi
  "${SSH[@]}" "if test -r '$WORKER_PIDFILE'; then p=\$(cat '$WORKER_PIDFILE'); case \"\$p\" in ''|*[!0-9]*) exit 0;; esac; if test -r /proc/\$p/cmdline && tr '\\0' ' ' < /proc/\$p/cmdline | grep -Fq -- 'ds4 --role worker'; then kill -TERM \"\$p\"; echo stopped-worker-\$p; fi; fi"
}

status() {
  if pid_matches "$LOCAL_PIDFILE" "ds4-server"; then echo "coordinator: running pid $(<"$LOCAL_PIDFILE")"; else echo "coordinator: stopped"; fi
  "${SSH[@]}" "if test -r '$WORKER_PIDFILE'; then p=\$(cat '$WORKER_PIDFILE'); if test -r /proc/\$p/cmdline && tr '\\0' ' ' < /proc/\$p/cmdline | grep -Fq -- 'ds4 --role worker'; then echo worker-running-pid-\$p; else echo worker-stopped; fi; else echo worker-stopped; fi"
  if curl --silent --show-error --fail --max-time 3 "http://$API_HOST:$API_PORT/health" >/dev/null; then
    echo "api: ready on $API_HOST:$API_PORT"
  else
    echo "api: loading or unavailable"
  fi
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
