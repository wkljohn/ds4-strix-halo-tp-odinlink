#!/bin/bash
# Fixed-frontier, fixed-token TP=2 benchmark over mandatory OdinLink RDMA.
#
# Usage: ./run-tp-ds4-bench.sh <tag> <model.gguf> [EXTRA_ENV=1 ...]
set -euo pipefail

TAG="${1:?usage: run-tp-ds4-bench.sh <tag> <model.gguf> [EXTRA_ENV=1 ...]}"
MODEL="${2:?usage: run-tp-ds4-bench.sh <tag> <model.gguf> [EXTRA_ENV=1 ...]}"
shift 2
EXTRA_ENV=("$@")
[[ $TAG =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "error: tag may contain only letters, digits, '.', '_' and '-'" >&2
  exit 2
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${DS4_BENCH_REPO:-$SCRIPT_DIR}
PEER_REPO=${DS4_PEER_REPO:-$REPO}
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
ROCPROF=${DS4_BENCH_ROCPROF:-0}
ROCPROF_RUNTIME=${DS4_BENCH_ROCPROF_RUNTIME:-1}
ROCPROF_REGION=${DS4_BENCH_ROCPROF_REGION:-decode}
ROCPROF_RANK=${DS4_BENCH_ROCPROF_RANK:-coordinator}
SHOW_OUTPUT=${DS4_BENCH_SHOW_OUTPUT:-0}
CANDIDATE=${DS4_BENCH_CANDIDATE:-0}
EXPECTED_FNV64=${DS4_BENCH_EXPECT_FNV64:-}
TP_TIMEOUT_SEC=${DS4_BENCH_TP_TIMEOUT_SEC:-60}
TP_TIMEOUT_EXPLICIT=${DS4_BENCH_TP_TIMEOUT_SEC+x}
DECODE_SELF_CHECK=${DS4_BENCH_DECODE_SELF_CHECK:-0}
TEACHER_FORCE_CONTROL=${DS4_BENCH_TEACHER_FORCE_CONTROL:-0}
QUALITY=${DS4_BENCH_QUALITY:-0}
ALLOW_NONSTANDARD_SPLIT=${DS4_BENCH_ALLOW_NONSTANDARD_SPLIT:-0}
EXPECT_GREEDY_TOP2=0
CANDIDATE_ARGS=()
CLEAN_ENV=(env -i PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8)
PEER_SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o HostKeyAlias=10.4.0.2 "$PEER_MGMT")
PEER_SCP=(scp -o BatchMode=yes -o StrictHostKeyChecking=yes -o HostKeyAlias=10.4.0.2)

[[ $FRONTIER =~ ^[1-9][0-9]*$ ]] || { echo "error: invalid frontier" >&2; exit 2; }
[[ $TOKENS =~ ^[1-9][0-9]*$ ]] || { echo "error: invalid generated-token count" >&2; exit 2; }
(( CONTEXT > FRONTIER + TOKENS )) || { echo "error: context must exceed frontier + tokens" >&2; exit 2; }
[[ $TP_TIMEOUT_SEC =~ ^[1-9][0-9]*$ ]] || { echo "error: invalid TP timeout" >&2; exit 2; }
if [[ $ROCPROF == 1 && -z $TP_TIMEOUT_EXPLICIT ]]; then
  # Coordinator-only tracing can delay one rank substantially. This affects
  # diagnostics only; production retains the 60-second fail-closed timeout.
  TP_TIMEOUT_SEC=300
fi
if [[ $ROCPROF_REGION != decode && $ROCPROF_REGION != prefill ]]; then
  echo "error: DS4_BENCH_ROCPROF_REGION must be decode or prefill" >&2
  exit 2
fi
if [[ $ROCPROF_RANK != coordinator && $ROCPROF_RANK != worker ]]; then
  echo "error: DS4_BENCH_ROCPROF_RANK must be coordinator or worker" >&2
  exit 2
fi
if [[ $ROCPROF == 1 && $ROCPROF_RUNTIME != 1 ]]; then
  echo "error: kernel-only rocprof is unsafe for asymmetric TP; use DS4_BENCH_ROCPROF_RUNTIME=1" >&2
  exit 2
fi
if [[ -n ${DS4_TP_EXPERT_SPLIT+x} ]]; then
  echo "error: ambient DS4_TP_EXPERT_SPLIT is not accepted; pass inference settings as trailing NAME=VALUE arguments" >&2
  exit 2
fi
if [[ $CANDIDATE == 1 ]]; then
  [[ $EXPECTED_FNV64 =~ ^[0-9a-fA-F]{16}$ ]] || {
    echo "error: candidate runs require DS4_BENCH_EXPECT_FNV64" >&2; exit 2;
  }
  SHOW_OUTPUT=1
  CANDIDATE_ARGS=(--semantic-smoke)
  if [[ $ROCPROF != 0 ]]; then
    echo "error: candidate timing cannot run under rocprof" >&2
    exit 2
  fi
elif [[ $CANDIDATE != 0 ]]; then
  echo "error: DS4_BENCH_CANDIDATE must be 0 or 1" >&2
  exit 2
fi
for env_kv in "${EXTRA_ENV[@]}"; do
  [[ $env_kv =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || {
    echo "error: experiment settings must be NAME=VALUE pairs: $env_kv" >&2
    exit 2
  }
  case $env_kv in
    DS4_ROCM_ENABLE_Q8_F16_CACHE=*|DS4_ROCM_STREAM_Q8_F16_CACHE_GB=*)
      echo "error: ds4-bench-tp results must not use the memory-heavy Q8-to-FP16 cache" >&2
      exit 2
      ;;
  esac
  if [[ $env_kv == DS4_TP_EXPERT_SPLIT=* && $DSPARK == 0 ]]; then
    if [[ $CANDIDATE == 1 || $ALLOW_NONSTANDARD_SPLIT != 1 ]]; then
      echo "error: non-DSpark production/candidate runs require the balanced 128/128 expert split" >&2
      echo "error: reserve DS4_TP_EXPERT_SPLIT=118 (46/54) for DSpark; set DS4_BENCH_ALLOW_NONSTANDARD_SPLIT=1 only for diagnostics" >&2
      exit 2
    fi
    echo "warning: running a nonstandard non-DSpark expert split; result is diagnostic only" >&2
  fi
  case $env_kv in
    DS4_TP_GREEDY_TOP2=|DS4_TP_GREEDY_TOP2=0) ;;
    DS4_TP_GREEDY_TOP2=*) EXPECT_GREEDY_TOP2=1 ;;
  esac
  if [[ $CANDIDATE == 1 ]]; then
    case $env_kv in
      DS4_*GRAPH_DUMP*=*|DS4_*PROFILE*=*|DS4_*TP_REFERENCE*=*|DS4_ORACLE_*=*)
        echo "error: candidate timing cannot enable graph dumps, profilers, or reference oracles: ${env_kv%%=*}" >&2
        exit 2
        ;;
    esac
  fi
done
if [[ $DECODE_SELF_CHECK == 1 ]]; then
  CANDIDATE_ARGS+=(--decode-self-check)
elif [[ $DECODE_SELF_CHECK != 0 ]]; then
  echo "error: DS4_BENCH_DECODE_SELF_CHECK must be 0 or 1" >&2
  exit 2
fi
if [[ $TEACHER_FORCE_CONTROL == 1 ]]; then
  CANDIDATE_ARGS+=(--teacher-force-control)
elif [[ $TEACHER_FORCE_CONTROL != 0 ]]; then
  echo "error: DS4_BENCH_TEACHER_FORCE_CONTROL must be 0 or 1" >&2
  exit 2
fi
if [[ $DECODE_SELF_CHECK == 1 && $TEACHER_FORCE_CONTROL == 1 ]]; then
  echo "error: choose only one TP decode diagnostic per run" >&2
  exit 2
fi

CURRENT_OPT_ENV=(
  DS4_ROCM_Q4K_DECODE_STAGE_XQ=1
)
if [[ $DSPARK == 1 ]]; then
  # Paired one-token DP4A changes the committed target trajectory unless the
  # five-row verifier uses identical arithmetic. Keep the exact production
  # DSpark path; the runtime independently enforces this safety invariant.
  CURRENT_OPT_ENV+=(DS4_ROCM_Q8_DECODE_PAIR_DP4A=0)
else
  CURRENT_OPT_ENV+=(
    DS4_TP_GREEDY_TOP2=1
    DS4_ROCM_Q8_DECODE_PAIR_DP4A=0
    DS4_ROCM_Q4K_DECODE_SPLIT_GATE_UP=1
    DS4_ROCM_TP_SKIP_UNOWNED=1
    DS4_ROCM_TP_PREFILL_SKIP_UNOWNED=1
    DS4_ROCM_SHARED_GU_SWIGLU_FUSE=1
  )
  EXPECT_GREEDY_TOP2=1
fi

COMMON_ENV=(
  DS4_TP_TIMEOUT_SEC="$TP_TIMEOUT_SEC"
  DS4_TP_ODINLINK_BATCH_ASYNC=1
  DS4_TP_VERBS_LIB=/home/wkljohn/Desktop/cc/OdinLink-Five/build/verbs/libodl_tb5_verbs.so.0.1.0
  LD_LIBRARY_PATH=/home/wkljohn/Desktop/cc/OdinLink-Five/build/lib:/home/wkljohn/Desktop/cc/OdinLink-Five/build/verbs
  "${CURRENT_OPT_ENV[@]}"
)
WORKER_ENV=("${COMMON_ENV[@]}")
COORD_ENV=("${COMMON_ENV[@]}")
if [[ $CANDIDATE == 1 && $EXPECT_GREEDY_TOP2 == 1 ]]; then
  COORD_ENV+=(DS4_BENCH_EXPECT_GREEDY_TOP2=1)
fi
WORKER_ARGS=()
COORD_ARGS=()
if [[ $QUALITY == 1 ]]; then
  WORKER_ARGS+=(--quality)
  COORD_ARGS+=(--quality)
elif [[ $QUALITY != 0 ]]; then
  echo "error: DS4_BENCH_QUALITY must be 0 or 1" >&2
  exit 2
fi
if [[ $SHOW_OUTPUT == 1 ]]; then
  COORD_ARGS+=(--show-output)
elif [[ $SHOW_OUTPUT != 0 ]]; then
  echo "error: DS4_BENCH_SHOW_OUTPUT must be 0 or 1" >&2
  exit 2
fi
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
  COORD_ARGS+=(--mtp "$MTP" --dspark)
elif [[ $DSPARK != 0 ]]; then
  echo "error: DS4_BENCH_DSPARK must be 0 or 1" >&2
  exit 2
fi

# Caller-supplied experiment switches intentionally come last so a bounded
# diagnostic can override a benchmark default (for example draft width 1)
# without editing the production configuration above.
WORKER_ENV+=("${EXTRA_ENV[@]}")
COORD_ENV+=("${EXTRA_ENV[@]}")

COORD_LOG="$OUT/coordinator-$TAG.log"
WORKER_LOG="$OUT/worker-$TAG.log"
CSV="$OUT/$TAG.csv"
WORKER_PIDFILE="$OUT/worker-$TAG.pid"
mkdir -p "$OUT"

sample_fingerprint() {
  local path=$1 size half tail
  size=$(stat -c %s "$path")
  half=$(( size / 2 > 4194304 ? size / 2 - 4194304 : 0 ))
  tail=$(( size > 8388608 ? size - 8388608 : 0 ))
  {
    printf '%s\n' "$size"
    dd if="$path" iflag=skip_bytes,count_bytes skip=0 count=8388608 status=none
    dd if="$path" iflag=skip_bytes,count_bytes skip="$half" count=8388608 status=none
    dd if="$path" iflag=skip_bytes,count_bytes skip="$tail" count=8388608 status=none
  } | sha256sum | awk '{print $1}'
}

remote_sample_fingerprint() {
  local path=$1 quoted
  printf -v quoted '%q' "$path"
  "${PEER_SSH[@]}" "p=$quoted; s=\$(stat -c %s \"\$p\"); h=\$((s / 2 > 4194304 ? s / 2 - 4194304 : 0)); t=\$((s > 8388608 ? s - 8388608 : 0)); { printf '%s\\n' \"\$s\"; dd if=\"\$p\" iflag=skip_bytes,count_bytes skip=0 count=8388608 status=none; dd if=\"\$p\" iflag=skip_bytes,count_bytes skip=\"\$h\" count=8388608 status=none; dd if=\"\$p\" iflag=skip_bytes,count_bytes skip=\"\$t\" count=8388608 status=none; } | sha256sum | awk '{print \$1}'"
}

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
  LOCAL_MTP_FINGERPRINT=$(sample_fingerprint "$MTP")
  PEER_MTP_FINGERPRINT=$(remote_sample_fingerprint "$MTP")
  [[ $LOCAL_MTP_FINGERPRINT == "$PEER_MTP_FINGERPRINT" ]] || {
    echo "error: sampled DSpark model fingerprints differ" >&2; exit 1;
  }
fi
LOCAL_MODEL_SIZE=$(stat -c %s "$MODEL")
PEER_MODEL_SIZE=$("${PEER_SSH[@]}" "stat -c %s '$MODEL'")
[[ $LOCAL_MODEL_SIZE == "$PEER_MODEL_SIZE" ]] || {
  echo "error: model sizes differ: local=$LOCAL_MODEL_SIZE peer=$PEER_MODEL_SIZE" >&2; exit 1;
}
LOCAL_MODEL_FINGERPRINT=$(sample_fingerprint "$MODEL")
PEER_MODEL_FINGERPRINT=$(remote_sample_fingerprint "$MODEL")
[[ $LOCAL_MODEL_FINGERPRINT == "$PEER_MODEL_FINGERPRINT" ]] || {
  echo "error: sampled model fingerprints differ" >&2; exit 1;
}
LOCAL_DS4_HASH=$(sha256sum "$REPO/ds4" | awk '{print $1}')
PEER_DS4_HASH=$("${PEER_SSH[@]}" "sha256sum '$PEER_REPO/ds4'" | awk '{print $1}')
[[ $LOCAL_DS4_HASH == "$PEER_DS4_HASH" ]] || {
  echo "error: worker binary hashes differ" >&2; exit 1;
}
LOCAL_BENCH_HASH=$(sha256sum "$REPO/ds4-bench-tp" | awk '{print $1}')

if pgrep -af '[d]s4-bench-tp.*--role coordinator.*--tensor-parallel' >/dev/null; then
  echo "error: a TP benchmark coordinator is already running; refusing to kill it" >&2
  exit 1
fi
if "${PEER_SSH[@]}" "pgrep -af '[d]s4 .*--role worker.*--tensor-parallel'" >/dev/null; then
  echo "error: a TP worker is already running on the peer; refusing to kill it" >&2
  exit 1
fi

WORKER_STARTED=0
worker_is_running() {
  "${PEER_SSH[@]}" "test -r '$WORKER_PIDFILE' || exit 1; p=\$(cat '$WORKER_PIDFILE'); case \"\$p\" in ''|*[!0-9]*) exit 1;; esac; test -r /proc/\$p/cmdline || exit 1; tr '\\0' ' ' < /proc/\$p/cmdline | grep -q -- './ds4 --role worker --tensor-parallel'"
}

wait_worker() {
  local limit=${1:-180} i
  for ((i = 0; i < limit; i++)); do
    worker_is_running || return 0
    sleep 1
  done
  return 1
}

terminate_owned_worker() {
  (( WORKER_STARTED == 1 )) || return 0
  worker_is_running || return 0
  echo "warning: worker did not exit after coordinator disconnect; sending TERM to its verified PID" >&2
  "${PEER_SSH[@]}" "p=\$(cat '$WORKER_PIDFILE'); case \"\$p\" in ''|*[!0-9]*) exit 1;; esac; test -r /proc/\$p/cmdline || exit 0; tr '\\0' ' ' < /proc/\$p/cmdline | grep -q -- './ds4 --role worker --tensor-parallel' || exit 1; kill -TERM \"\$p\""
  wait_worker 60 || {
    echo "error: owned worker ignored TERM; leaving it intact to avoid unsafe GPU/RDMA teardown" >&2
    return 1
  }
}

cleanup() {
  local rc=$?
  if (( WORKER_STARTED == 1 )); then
    wait_worker 180 || terminate_owned_worker || true
  fi
  return "$rc"
}
trap cleanup EXIT
"${PEER_SSH[@]}" "mkdir -p '$OUT'"

# The two TP nodes do not share a filesystem. Diagnostic graph-dump prefixes
# are passed to both ranks, so create their parent directory on both machines
# before either process starts. Keep automatic creation inside the durable
# research tree; reject a surprising path instead of mutating an arbitrary
# peer directory.
for env_kv in "${EXTRA_ENV[@]}"; do
  case $env_kv in
    DS4_ROCM_GRAPH_DUMP_PREFIX=*|DS4_METAL_GRAPH_DUMP_PREFIX=*)
      dump_prefix=${env_kv#*=}
      dump_dir=$(dirname -- "$dump_prefix")
      case $dump_dir/ in
        "$REPO/research-results/"*) ;;
        *)
          echo "error: graph dump directory must be under $REPO/research-results: $dump_dir" >&2
          exit 2
          ;;
      esac
      mkdir -p "$dump_dir"
      printf -v dump_dir_q '%q' "$dump_dir"
      "${PEER_SSH[@]}" "mkdir -p $dump_dir_q"
      ;;
  esac
done

echo "=== ds4-bench $TAG ==="
echo "model: $MODEL"
echo "workload: frontier=$FRONTIER generated_tokens=$TOKENS context=$CONTEXT prefill_chunk=$PREFILL_CHUNK"
if [[ $DSPARK == 1 ]]; then echo "dspark: 1 mtp=$MTP"; else echo "dspark: 0"; fi
if [[ $ROCPROF == 1 ]]; then echo "rocprof: rank=$ROCPROF_RANK kernel trace (diagnostic; timing is not benchmark evidence)"; fi
echo "ds4_sha256: $LOCAL_DS4_HASH"
echo "ds4_bench_tp_sha256: $LOCAL_BENCH_HASH"
echo "model_sample_sha256: $LOCAL_MODEL_FINGERPRINT"
if [[ $DSPARK == 1 ]]; then echo "mtp_sample_sha256: $LOCAL_MTP_FINGERPRINT resident_q8=1"; fi

WORKER_APP=(./ds4
  --role worker --tensor-parallel --coordinator "$COORDINATOR_ADDR" 9000
  --transport rdma --rocm -m "$MODEL" -c "$CONTEXT"
  --prefill-chunk "$PREFILL_CHUNK"
  "${WORKER_ARGS[@]}")
WORKER_CMD=("${CLEAN_ENV[@]}" "${WORKER_ENV[@]}" "${WORKER_APP[@]}")
if [[ $ROCPROF == 1 && $ROCPROF_RANK == worker ]]; then
  ROCPROF_OUT="$OUT/rocprof-$TAG-worker"
  printf -v ROCPROF_OUT_Q '%q' "$ROCPROF_OUT"
  "${PEER_SSH[@]}" "mkdir -p $ROCPROF_OUT_Q"
  # The worker has no benchmark-level ROCTx boundary. Trace its complete
  # runtime and separate prefill by kernel family/count; model residency is
  # dominated by page warming and copies rather than these compute kernels.
  WORKER_CMD=("${CLEAN_ENV[@]}" "${WORKER_ENV[@]}"
              rocprofv3 --runtime-trace --stats --summary
              --summary-units usec --output-directory "$ROCPROF_OUT" --
              "${WORKER_APP[@]}")
fi
printf -v WORKER_CMD_Q '%q ' "${WORKER_CMD[@]}"
printf -v PEER_REPO_Q '%q' "$PEER_REPO"
printf -v WORKER_LOG_Q '%q' "$WORKER_LOG"
printf -v WORKER_PIDFILE_Q '%q' "$WORKER_PIDFILE"
"${PEER_SSH[@]}" "cd $PEER_REPO_Q || exit 1; nohup setsid $WORKER_CMD_Q > $WORKER_LOG_Q 2>&1 < /dev/null & p=\$!; echo \$p > $WORKER_PIDFILE_Q"
WORKER_STARTED=1

COORD_CMD=("$REPO/ds4-bench-tp")
if [[ $ROCPROF == 1 && $ROCPROF_RANK == coordinator ]]; then
  ROCPROF_OUT="$OUT/rocprof-$TAG"
  mkdir -p "$ROCPROF_OUT"
  ROCPROF_TRACE=(--runtime-trace --selected-regions)
  COORD_ENV+=(DS4_BENCH_ROCPROF_SELECTED_REGIONS=1
             DS4_BENCH_ROCPROF_REGION="$ROCPROF_REGION")
  COORD_CMD=(rocprofv3 "${ROCPROF_TRACE[@]}"
             --stats --summary --summary-units usec
             --output-directory "$ROCPROF_OUT" -- "$REPO/ds4-bench-tp")
elif [[ $ROCPROF != 0 && $ROCPROF != 1 ]]; then
  echo "error: DS4_BENCH_ROCPROF must be 0 or 1" >&2
  exit 2
fi

"${CLEAN_ENV[@]}" "${COORD_ENV[@]}" "${COORD_CMD[@]}" \
  --role coordinator --tensor-parallel --listen 0.0.0.0 9000 \
  --transport rdma --rocm -m "$MODEL" --prompt-file "$PROMPT_FILE" \
  --ctx-start "$FRONTIER" --ctx-max "$FRONTIER" --ctx-alloc "$CONTEXT" \
  --prefill-chunk "$PREFILL_CHUNK" --gen-tokens "$TOKENS" --csv "$CSV" \
  "${CANDIDATE_ARGS[@]}" \
  "${COORD_ARGS[@]}" \
  > "$COORD_LOG" 2>&1

wait_worker 180 || {
  echo "error: worker did not exit gracefully after STOP" >&2
  terminate_owned_worker || true
  exit 1
}
WORKER_STARTED=0
trap - EXIT
"${PEER_SCP[@]}" "$PEER_MGMT:$WORKER_LOG" "$WORKER_LOG"
if [[ $DECODE_SELF_CHECK == 1 || $TEACHER_FORCE_CONTROL == 1 ]]; then
  echo "TP_DECODE_DIAGNOSTIC_DONE"
  exit 0
fi
"$REPO/scripts/check-ds4-bench-result.sh" \
  "$CSV" "$COORD_LOG" "$WORKER_LOG" "$EXPECTED_FNV64" "$TOKENS" "$CANDIDATE"
cat "$CSV"
echo RUN_DONE
