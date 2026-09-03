#!/bin/bash
# Fixed-frontier, fixed-token TP=2 benchmark over mandatory RDMA.
#
# Usage: ./run-tp-ds4-bench.sh <tag> <model.gguf> [EXTRA_ENV=1 ...]
# Copy bench.env.example to bench.env.local before the first real run.
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
BENCH_CONFIG=${DS4_BENCH_CONFIG:-$SCRIPT_DIR/bench.env.local}

# bench.env.local supplies machine-local defaults. Preserve every explicitly
# exported DS4_* value from the invoking command so a one-run provider/address
# override cannot be silently clobbered by a plain assignment in that file.
# Use indirect expansion and printf -v rather than eval so arbitrary values are
# restored byte-for-byte. Process substitution plus `|| true` is intentional:
# compgen returns nonzero when a clean environment has no matching variables.
declare -A DS4_INVOKING_ENV=()
while IFS= read -r ds4_env_name; do
  DS4_INVOKING_ENV["$ds4_env_name"]=${!ds4_env_name}
done < <(compgen -e DS4_ || true)
if [[ -r $BENCH_CONFIG ]]; then
  # shellcheck disable=SC1090
  source "$BENCH_CONFIG"
fi
DS4_CONFIG_RDMA_PROFILE=${DS4_BENCH_RDMA_PROFILE-}
if (( ${#DS4_INVOKING_ENV[@]} )); then
  for ds4_env_name in "${!DS4_INVOKING_ENV[@]}"; do
    printf -v "$ds4_env_name" '%s' "${DS4_INVOKING_ENV[$ds4_env_name]}"
    export "$ds4_env_name"
  done
fi
unset ds4_env_name
if [[ ${DS4_INVOKING_ENV[DS4_BENCH_RDMA_PROFILE]+set} == set &&
      ${DS4_INVOKING_ENV[DS4_BENCH_RDMA_PROFILE]} != "$DS4_CONFIG_RDMA_PROFILE" ]]; then
  # Provider-specific values from a different config profile are unsafe. Keep
  # only values explicitly supplied with this invocation; otherwise use the
  # selected provider's device defaults and require its address explicitly.
  [[ ${DS4_INVOKING_ENV[DS4_LOCAL_RDMA_DEVICE]+set} == set ]] ||
    unset DS4_LOCAL_RDMA_DEVICE
  [[ ${DS4_INVOKING_ENV[DS4_PEER_RDMA_DEVICE]+set} == set ]] ||
    unset DS4_PEER_RDMA_DEVICE
  [[ ${DS4_INVOKING_ENV[DS4_RDMA_GID_INDEX]+set} == set ]] ||
    unset DS4_RDMA_GID_INDEX
  [[ ${DS4_INVOKING_ENV[DS4_COORDINATOR_ADDR]+set} == set ]] ||
    unset DS4_COORDINATOR_ADDR
fi
if [[ -r $BENCH_CONFIG ]]; then
  BENCH_CONFIG_SHA256=$(sha256sum "$BENCH_CONFIG" | awk '{print $1}')
else
  BENCH_CONFIG_SHA256=none
fi
REPO=${DS4_BENCH_REPO:-$SCRIPT_DIR}
PEER_REPO=${DS4_PEER_REPO:-$REPO}
# shellcheck disable=SC1091
source "$REPO/scripts/ds4-research-root.sh"
ds4_resolve_research_roots "$REPO"
PEER_MGMT=${DS4_PEER_MGMT:-}
PEER_HOST_KEY_ALIAS=${DS4_PEER_HOST_KEY_ALIAS:-${PEER_MGMT#*@}}
RDMA_PROFILE=${DS4_BENCH_RDMA_PROFILE:-odinlink}
PREFILL_CHUNK_EXPLICIT=${DS4_BENCH_PREFILL_CHUNK+x}
if [[ -n $PREFILL_CHUNK_EXPLICIT && -z ${DS4_BENCH_PREFILL_CHUNK-} ]]; then
  echo "error: DS4_BENCH_PREFILL_CHUNK cannot be empty" >&2
  exit 2
fi
case $RDMA_PROFILE in
  odinlink)
    COORDINATOR_ADDR=${DS4_COORDINATOR_ADDR:-}
    LOCAL_RDMA_DEVICE=${DS4_LOCAL_RDMA_DEVICE:-odl_tb5_0}
    PEER_RDMA_DEVICE=${DS4_PEER_RDMA_DEVICE:-odl_tb5_0}
    # A RoCE-oriented bench.env.local may leave mlx5 names exported.  Never
    # let those stale values reach an OdinLink run: the provider would be
    # loaded explicitly while DS4 probes the wrong device and fails with an
    # opaque "no active port" error.  Require the provider namespace here so
    # profile switches are fail-closed and explicit.
    [[ $LOCAL_RDMA_DEVICE == odl_tb5_* && $PEER_RDMA_DEVICE == odl_tb5_* ]] || {
      echo "error: odinlink profile requires odl_tb5_* devices (got $LOCAL_RDMA_DEVICE / $PEER_RDMA_DEVICE)" >&2
      echo "error: set DS4_LOCAL_RDMA_DEVICE=odl_tb5_0 and DS4_PEER_RDMA_DEVICE=odl_tb5_0" >&2
      exit 2
    }
    ;;
  roce-v2)
    COORDINATOR_ADDR=${DS4_COORDINATOR_ADDR:-}
    LOCAL_RDMA_DEVICE=${DS4_LOCAL_RDMA_DEVICE:-mlx5_0}
    PEER_RDMA_DEVICE=${DS4_PEER_RDMA_DEVICE:-mlx5_1}
    RDMA_GID_INDEX=${DS4_RDMA_GID_INDEX:-3}
    [[ $LOCAL_RDMA_DEVICE == mlx5_* && $PEER_RDMA_DEVICE == mlx5_* ]] || {
      echo "error: roce-v2 profile requires mlx5_* devices (got $LOCAL_RDMA_DEVICE / $PEER_RDMA_DEVICE)" >&2
      echo "error: set DS4_LOCAL_RDMA_DEVICE and DS4_PEER_RDMA_DEVICE to the ConnectX devices" >&2
      exit 2
    }
    ;;
  *)
    echo "error: DS4_BENCH_RDMA_PROFILE must be odinlink or roce-v2" >&2
    exit 2
    ;;
esac
PROMPT_FILE=${DS4_BENCH_PROMPT_FILE:-$REPO/bench-prompts/codex-attn-rowsplit-implement-brief.md}
FRONTIER=${DS4_BENCH_FRONTIER:-2048}
FRONTIER_MAX=${DS4_BENCH_FRONTIER_MAX:-$FRONTIER}
STEP_INCR=${DS4_BENCH_STEP_INCR:-2048}
STEP_MUL=${DS4_BENCH_STEP_MUL:-1}
TOKENS=${DS4_BENCH_TOKENS:-300}
CONTEXT=${DS4_BENCH_CONTEXT:-4096}
PREFILL_CHUNK=${DS4_BENCH_PREFILL_CHUNK:-4096}
if [[ $RDMA_PROFILE == roce-v2 && -z $PREFILL_CHUNK_EXPLICIT ]]; then
  # ConnectX-4 Lx registers the 2048-row mapped slab but exhausts its
  # pin/translation resources with the 4096-row direct layout.
  PREFILL_CHUNK=2048
fi
DSPARK=${DS4_BENCH_DSPARK:-0}
MTP=${DS4_BENCH_MTP:-}
if [[ ${DS4_BENCH_OUT+x} ]]; then
  OUT=$DS4_BENCH_OUT
  PEER_OUT=${DS4_PEER_BENCH_OUT:-$OUT}
else
  OUT=$DS4_RESEARCH_ROOT/bench-runs
  PEER_OUT=$DS4_PEER_RESEARCH_ROOT/bench-runs
fi
ROCPROF=${DS4_BENCH_ROCPROF:-0}
ROCPROF_BIN=${DS4_BENCH_ROCPROF_BIN:-rocprofv3}
# 1 captures the complete runtime domain, including asynchronous memory-copy
# records.  2 captures HIP runtime calls plus kernels and markers, avoiding a
# rocprofiler-sdk 1.3.x shutdown hang when an HSA copy completion callback is
# lost.  3 captures only kernels and markers when the SDK's HIP selected-region
# pause itself fails to drain. All modes preserve the selected-region boundary.
ROCPROF_RUNTIME=${DS4_BENCH_ROCPROF_RUNTIME:-1}
ROCPROF_REGION=${DS4_BENCH_ROCPROF_REGION:-decode}
ROCPROF_RANK=${DS4_BENCH_ROCPROF_RANK:-coordinator}
SHOW_OUTPUT=${DS4_BENCH_SHOW_OUTPUT:-0}
CANDIDATE=${DS4_BENCH_CANDIDATE:-0}
CANDIDATE_LANE=${DS4_BENCH_LANE:-A}
BASELINE_ID=${DS4_BENCH_BASELINE_ID:-}
EXPECTED_FNV64=${DS4_BENCH_EXPECT_FNV64:-}
TP_TIMEOUT_SEC=${DS4_BENCH_TP_TIMEOUT_SEC:-60}
TP_TIMEOUT_EXPLICIT=${DS4_BENCH_TP_TIMEOUT_SEC+x}
DECODE_SELF_CHECK=${DS4_BENCH_DECODE_SELF_CHECK:-0}
TEACHER_FORCE_CONTROL=${DS4_BENCH_TEACHER_FORCE_CONTROL:-0}
EXPECTED_TEACHER_FNV64=${DS4_BENCH_EXPECT_TEACHER_FNV64:-}
RECORD_TEACHER_BASELINE=${DS4_BENCH_RECORD_TEACHER_BASELINE:-0}
QUALITY=${DS4_BENCH_QUALITY:-0}
ALLOW_NONSTANDARD_SPLIT=${DS4_BENCH_ALLOW_NONSTANDARD_SPLIT:-0}
ALLOW_GLM_BATCH1=${DS4_BENCH_ALLOW_GLM_BATCH1:-0}
VALIDATE_CONFIG_ONLY=${DS4_BENCH_VALIDATE_CONFIG_ONLY:-0}
DUMP_FRONTIER_LOGITS_DIR=${DS4_BENCH_DUMP_FRONTIER_LOGITS_DIR:-}
DUMP_GENERATED_TOKEN_FILE=${DS4_BENCH_DUMP_GENERATED_TOKEN_FILE:-}
FROZEN_TOKEN_FILE=${DS4_BENCH_FROZEN_TOKEN_FILE:-}
FROZEN_LOGITS_DIR=${DS4_BENCH_FROZEN_LOGITS_DIR:-}
DECLARED_TOOLCHAIN_ID=${DS4_BENCH_TOOLCHAIN_ID:-}
EXPECTED_TOOLCHAIN_SHA256=${DS4_BENCH_EXPECT_TOOLCHAIN_SHA256:-}
EXPECT_GREEDY_TOP2=0
GLM5_PREFILL_BATCH=
GLM5_PREFILL_BATCH_SEEN=0
CANDIDATE_ARGS=()
CLEAN_ENV=(env -i PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8)
PEER_SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes
          -o "HostKeyAlias=$PEER_HOST_KEY_ALIAS" "$PEER_MGMT")
PEER_SCP=(scp -o BatchMode=yes -o StrictHostKeyChecking=yes
          -o "HostKeyAlias=$PEER_HOST_KEY_ALIAS")

[[ $FRONTIER =~ ^[1-9][0-9]*$ ]] || { echo "error: invalid frontier" >&2; exit 2; }
[[ $FRONTIER_MAX =~ ^[1-9][0-9]*$ ]] || { echo "error: invalid frontier max" >&2; exit 2; }
(( FRONTIER_MAX >= FRONTIER )) || { echo "error: frontier max must be >= frontier" >&2; exit 2; }
[[ $STEP_INCR =~ ^[1-9][0-9]*$ ]] || { echo "error: invalid frontier step increment" >&2; exit 2; }
[[ $STEP_MUL =~ ^[1-9][0-9]*(\.[0-9]+)?$ ]] || { echo "error: invalid frontier step multiplier" >&2; exit 2; }
[[ $TOKENS =~ ^[1-9][0-9]*$ ]] || { echo "error: invalid generated-token count" >&2; exit 2; }
(( CONTEXT > FRONTIER_MAX + TOKENS )) || { echo "error: context must exceed frontier max + tokens" >&2; exit 2; }
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
if [[ $ROCPROF == 1 && $ROCPROF_RUNTIME != 1 && $ROCPROF_RUNTIME != 2 &&
      $ROCPROF_RUNTIME != 3 ]]; then
  echo "error: DS4_BENCH_ROCPROF_RUNTIME must be 1 (full runtime), 2 (HIP API + kernel), or 3 (kernel only)" >&2
  exit 2
fi
if [[ $ROCPROF == 1 ]]; then
  if [[ $ROCPROF_BIN == */* ]]; then
    [[ -x $ROCPROF_BIN ]] || {
      echo "error: profiler is not executable: $ROCPROF_BIN" >&2
      exit 2
    }
    ROCPROF_RESOLVED=$ROCPROF_BIN
  else
    ROCPROF_RESOLVED=$(command -v "$ROCPROF_BIN") || {
      echo "error: profiler is not on PATH: $ROCPROF_BIN" >&2
      exit 2
    }
  fi
  ROCPROF_SHA256=$(sha256sum "$ROCPROF_RESOLVED" | awk '{print $1}')
else
  ROCPROF_RESOLVED=
  ROCPROF_SHA256=
fi
if [[ $ROCPROF == 1 && $ROCPROF_RUNTIME == 3 && $TP_TIMEOUT_SEC -lt 600 ]]; then
  echo "error: asymmetric kernel tracing requires DS4_BENCH_TP_TIMEOUT_SEC >= 600" >&2
  exit 2
fi
if [[ -n ${DS4_TP_EXPERT_SPLIT+x} ]]; then
  echo "error: ambient DS4_TP_EXPERT_SPLIT is not accepted; pass inference settings as trailing NAME=VALUE arguments" >&2
  exit 2
fi
if [[ $CANDIDATE == 1 ]]; then
  [[ $FRONTIER == "$FRONTIER_MAX" ]] || {
    echo "error: candidate timing requires one fixed frontier; multi-frontier sweeps are diagnostic only" >&2
    exit 2
  }
  [[ $CANDIDATE_LANE == A || $CANDIDATE_LANE == B || $CANDIDATE_LANE == C ]] || {
    echo "error: DS4_BENCH_LANE must be A, B, or C" >&2; exit 2;
  }
  if [[ $CANDIDATE_LANE == A ]]; then
    [[ $EXPECTED_FNV64 =~ ^[0-9a-fA-F]{16}$ ]] || {
      echo "error: lane A candidates require DS4_BENCH_EXPECT_FNV64" >&2; exit 2;
    }
    [[ -z $BASELINE_ID ]] || {
      echo "error: lane A uses the exact expected fingerprint, not DS4_BENCH_BASELINE_ID" >&2; exit 2;
    }
  else
    [[ $BASELINE_ID =~ ^sha256:[0-9a-f]{64}$ ]] || {
      echo "error: lane $CANDIDATE_LANE candidates require a content-addressed DS4_BENCH_BASELINE_ID" >&2; exit 2;
    }
    [[ -z $EXPECTED_FNV64 ]] || {
      echo "error: lane $CANDIDATE_LANE records a proposed new fingerprint; do not pin it to the predecessor FNV" >&2; exit 2;
    }
  fi
  SHOW_OUTPUT=1
  CANDIDATE_ARGS=(--semantic-smoke)
  if [[ $ROCPROF != 0 ]]; then
    echo "error: candidate timing cannot run under rocprof" >&2
    exit 2
  fi
  if [[ -n $DUMP_FRONTIER_LOGITS_DIR ]]; then
    echo "error: candidate timing cannot dump frontier logits" >&2
    exit 2
  fi
elif [[ $CANDIDATE != 0 ]]; then
  echo "error: DS4_BENCH_CANDIDATE must be 0 or 1" >&2
  exit 2
fi
[[ $VALIDATE_CONFIG_ONLY == 0 || $VALIDATE_CONFIG_ONLY == 1 ]] || {
  echo "error: DS4_BENCH_VALIDATE_CONFIG_ONLY must be 0 or 1" >&2
  exit 2
}
ATTN_STATIC_DIRECT_REQUESTED=0
ATTN_STATIC_DIRECT_T2_REQUESTED=0
GLM5_SPARSE_BATCH_BRIDGE=0
GLM5_SPARSE_BATCH_BRIDGE_SEEN=0
GLM5_SPARSE_BATCH_OUTPUT=0
GLM5_SPARSE_BATCH_OUTPUT_SEEN=0
for env_kv in "${EXTRA_ENV[@]}"; do
  [[ $env_kv =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || {
    echo "error: experiment settings must be NAME=VALUE pairs: $env_kv" >&2
    exit 2
  }
  case $env_kv in
    DS4_GLM5_NEXT_PREFILL_BATCH=*)
      (( GLM5_PREFILL_BATCH_SEEN == 0 )) || {
        echo "error: DS4_GLM5_NEXT_PREFILL_BATCH was supplied more than once" >&2
        exit 2
      }
      GLM5_PREFILL_BATCH=${env_kv#*=}
      GLM5_PREFILL_BATCH_SEEN=1
      [[ $GLM5_PREFILL_BATCH =~ ^[1-9][0-9]*$ ]] || {
        echo "error: DS4_GLM5_NEXT_PREFILL_BATCH must be an integer from 1 to 1024" >&2
        exit 2
      }
      (( GLM5_PREFILL_BATCH <= 1024 )) || {
        echo "error: DS4_GLM5_NEXT_PREFILL_BATCH must be at most 1024" >&2
        exit 2
      }
      ;;
    DS4_GLM5_SPARSE_BATCH_BRIDGE=*)
      (( GLM5_SPARSE_BATCH_BRIDGE_SEEN == 0 )) || {
        echo "error: DS4_GLM5_SPARSE_BATCH_BRIDGE was supplied more than once" >&2
        exit 2
      }
      GLM5_SPARSE_BATCH_BRIDGE=${env_kv#*=}
      GLM5_SPARSE_BATCH_BRIDGE_SEEN=1
      [[ $GLM5_SPARSE_BATCH_BRIDGE == 0 ||
         $GLM5_SPARSE_BATCH_BRIDGE == 1 ]] || {
        echo "error: DS4_GLM5_SPARSE_BATCH_BRIDGE must be 0 or 1" >&2
        exit 2
      }
      ;;
    DS4_GLM5_SPARSE_BATCH_OUTPUT=*)
      (( GLM5_SPARSE_BATCH_OUTPUT_SEEN == 0 )) || {
        echo "error: DS4_GLM5_SPARSE_BATCH_OUTPUT was supplied more than once" >&2
        exit 2
      }
      GLM5_SPARSE_BATCH_OUTPUT=${env_kv#*=}
      GLM5_SPARSE_BATCH_OUTPUT_SEEN=1
      [[ $GLM5_SPARSE_BATCH_OUTPUT == 0 ||
         $GLM5_SPARSE_BATCH_OUTPUT == 1 ]] || {
        echo "error: DS4_GLM5_SPARSE_BATCH_OUTPUT must be 0 or 1" >&2
        exit 2
      }
      ;;
    DS4_ROCM_ENABLE_Q8_F16_CACHE=*|DS4_ROCM_STREAM_Q8_F16_CACHE_GB=*)
      echo "error: ds4-bench-tp results must not use the memory-heavy Q8-to-FP16 cache" >&2
      exit 2
      ;;
  esac
  case $env_kv in
    DS4_ROCM_ATTENTION_PREFILL_STATIC_FLASH_DIRECT=1)
      ATTN_STATIC_DIRECT_REQUESTED=1
      ;;
    DS4_ROCM_ATTENTION_PREFILL_STATIC_FLASH_DIRECT_T2=1)
      ATTN_STATIC_DIRECT_T2_REQUESTED=1
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
      DS4_*GRAPH_DUMP*=*|DS4_*PROFILE*=*|DS4_*TP_REFERENCE*=*|DS4_ORACLE_*=*|DS4_ROCM_GLM_CAUSAL_ATTN_COMPARE=*)
        echo "error: candidate timing cannot enable graph dumps, profilers, or reference oracles: ${env_kv%%=*}" >&2
        exit 2
        ;;
    esac
  fi
done
if [[ $ATTN_STATIC_DIRECT_T2_REQUESTED == 1 &&
      $ATTN_STATIC_DIRECT_REQUESTED != 1 ]]; then
  echo "error: paired-query static attention requires DS4_ROCM_ATTENTION_PREFILL_STATIC_FLASH_DIRECT=1" >&2
  exit 2
fi
if [[ $DECODE_SELF_CHECK == 1 ]]; then
  CANDIDATE_ARGS+=(--decode-self-check)
elif [[ $DECODE_SELF_CHECK != 0 ]]; then
  echo "error: DS4_BENCH_DECODE_SELF_CHECK must be 0 or 1" >&2
  exit 2
fi
if [[ $TEACHER_FORCE_CONTROL == 1 ]]; then
  [[ $RECORD_TEACHER_BASELINE == 0 || $RECORD_TEACHER_BASELINE == 1 ]] || {
    echo "error: DS4_BENCH_RECORD_TEACHER_BASELINE must be 0 or 1" >&2
    exit 2
  }
  if [[ $RECORD_TEACHER_BASELINE == 1 ]]; then
    [[ $CANDIDATE == 0 ]] || {
      echo "error: a teacher baseline recording is diagnostic, not a candidate gate" >&2
      exit 2
    }
    [[ -z $EXPECTED_TEACHER_FNV64 ]] || {
      echo "error: do not supply an expected teacher signature while recording" >&2
      exit 2
    }
    CANDIDATE_ARGS+=(--teacher-force-control)
  else
    [[ $EXPECTED_TEACHER_FNV64 =~ ^[0-9a-fA-F]{16}$ ]] || {
      echo "error: teacher-force gate requires DS4_BENCH_EXPECT_TEACHER_FNV64" >&2
      exit 2
    }
    CANDIDATE_ARGS+=(--teacher-force-control
                     --teacher-force-expect-fnv64 "$EXPECTED_TEACHER_FNV64")
  fi
elif [[ $TEACHER_FORCE_CONTROL != 0 ]]; then
  echo "error: DS4_BENCH_TEACHER_FORCE_CONTROL must be 0 or 1" >&2
  exit 2
fi
if [[ $DECODE_SELF_CHECK == 1 && $TEACHER_FORCE_CONTROL == 1 ]]; then
  echo "error: choose only one TP decode diagnostic per run" >&2
  exit 2
fi
if [[ -n $FROZEN_TOKEN_FILE || -n $FROZEN_LOGITS_DIR ]]; then
  [[ -n $FROZEN_TOKEN_FILE && -n $FROZEN_LOGITS_DIR ]] || {
    echo "error: DS4_BENCH_FROZEN_TOKEN_FILE and DS4_BENCH_FROZEN_LOGITS_DIR must be used together" >&2
    exit 2
  }
  [[ -n $DECLARED_TOOLCHAIN_ID ]] || {
    echo "error: frozen-token evidence requires DS4_BENCH_TOOLCHAIN_ID" >&2
    exit 2
  }
  [[ $CANDIDATE == 0 && $ROCPROF == 0 ]] || {
    echo "error: frozen-token full-logit comparison is diagnostic, not benchmark evidence" >&2
    exit 2
  }
  [[ $DECODE_SELF_CHECK == 0 && $TEACHER_FORCE_CONTROL == 0 ]] || {
    echo "error: choose only one TP decode diagnostic per run" >&2
    exit 2
  }
fi

[[ -r $MODEL && -r $PROMPT_FILE ]] || { echo "error: missing local model or prompt" >&2; exit 1; }
MODEL_ARCH=$(python3 "$REPO/scripts/gguf_tensor_types.py" --architecture "$MODEL") || {
  echo "error: unable to inspect general.architecture in $MODEL" >&2
  exit 1
}
if [[ $MODEL_ARCH == glm5-next ]]; then
  (( GLM5_PREFILL_BATCH_SEEN == 1 )) || {
    echo "error: GLM5 benchmark runs must explicitly set DS4_GLM5_NEXT_PREFILL_BATCH" >&2
    echo "error: batch=1 is only a scalar regression guard; use a fixed batch >1 for prefill candidates" >&2
    exit 2
  }
  if (( GLM5_PREFILL_BATCH == 1 )); then
    [[ $CANDIDATE == 0 && $ALLOW_GLM_BATCH1 == 1 ]] || {
      echo "error: GLM5 batch=1 is not batched-prefill evidence" >&2
      echo "error: set DS4_BENCH_ALLOW_GLM_BATCH1=1 only for a non-candidate scalar regression guard" >&2
      exit 2
    }
    echo "warning: GLM5 batch=1 run is a scalar regression guard, not prefill performance evidence" >&2
  fi
  if (( GLM5_SPARSE_BATCH_OUTPUT == 1 &&
        GLM5_SPARSE_BATCH_BRIDGE != 1 )); then
    echo "error: DS4_GLM5_SPARSE_BATCH_OUTPUT=1 requires DS4_GLM5_SPARSE_BATCH_BRIDGE=1" >&2
    exit 2
  fi
elif (( GLM5_SPARSE_BATCH_BRIDGE_SEEN == 1 )); then
  echo "error: DS4_GLM5_SPARSE_BATCH_BRIDGE applies only to GLM5 benchmarks" >&2
  exit 2
elif (( GLM5_SPARSE_BATCH_OUTPUT_SEEN == 1 )); then
  echo "error: DS4_GLM5_SPARSE_BATCH_OUTPUT applies only to GLM5 benchmarks" >&2
  exit 2
elif (( GLM5_PREFILL_BATCH_SEEN == 1 )); then
  echo "error: DS4_GLM5_NEXT_PREFILL_BATCH applies only to GLM5 benchmarks" >&2
  exit 2
fi
CURRENT_OPT_ENV=()
if [[ $DSPARK == 1 ]]; then
  [[ $MODEL_ARCH != glm5-next ]] || {
    echo "error: GLM5.3 ordinary TP benchmark does not support DSpark/MTP" >&2
    exit 2
  }
  # Paired one-token DP4A changes the committed target trajectory unless the
  # five-row verifier uses identical arithmetic. Keep the exact production
  # DSpark path; the runtime independently enforces this safety invariant.
  CURRENT_OPT_ENV+=(DS4_ROCM_Q8_DECODE_PAIR_DP4A=0)
else
  ROUTED_FAMILY=$(python3 "$REPO/scripts/gguf_tensor_types.py" --routed-family "$MODEL") || {
    echo "error: unable to inspect routed-expert quantization in $MODEL" >&2
    exit 1
  }
  case $ROUTED_FAMILY in
    Q4_K) TP_PREFILL_SKIP_UNOWNED=1 ;;
    HYBRID_Q2) TP_PREFILL_SKIP_UNOWNED=0 ;;
    *)
      echo "error: unsupported routed-expert quantization: $ROUTED_FAMILY" >&2
      exit 1
      ;;
  esac
  if [[ $MODEL_ARCH == glm5-next ]]; then
    # GLM5.3 owns its KDA/MLA and routed-expert dispatch. Do not leak the
    # DeepSeek compressor, greedy-vocabulary split, or skip/fusion switches
    # into this independently quality-gated arithmetic path.
    CURRENT_OPT_ENV+=(
      DS4_GLM5_NEXT_ENABLE_ORDINARY=1
      DS4_GLM5_PREFILL_PROOF=1
      DS4_TP_BIG_DIRECT=1
      DS4_TP_GREEDY_TOP2=0
      DS4_ROCM_TEMPORAL_COMPRESSOR=0
    )
    EXPECT_GREEDY_TOP2=0
  elif [[ $MODEL_ARCH != deepseek4 ]]; then
    echo "error: ds4-bench-tp has no validated TP profile for architecture $MODEL_ARCH" >&2
    exit 1
  elif [[ $ROUTED_FAMILY == HYBRID_Q2 ]]; then
    CURRENT_OPT_ENV+=(DS4_ROCM_TP_ZERO_WEIGHT_TILE_SKIP=1)
    if [[ $CANDIDATE == 1 ]]; then
      for env_kv in "${EXTRA_ENV[@]}"; do
        if [[ $env_kv == DS4_ROCM_TP_PREFILL_SKIP_UNOWNED=* &&
              $env_kv != DS4_ROCM_TP_PREFILL_SKIP_UNOWNED=0 ]]; then
          echo "error: hybrid Q2 candidate cannot enable trajectory-changing peer-route omission" >&2
          exit 2
        fi
      done
    fi
  fi
  if [[ $MODEL_ARCH == deepseek4 ]]; then
    CURRENT_OPT_ENV+=(
      DS4_ROCM_Q4K_DECODE_STAGE_XQ=1
      DS4_TP_GREEDY_TOP2=1
      DS4_TP_HOST_CALLBACK=1
      DS4_TP_PREFILL_FFN_WAVEFRONT=1
      DS4_ROCM_TEMPORAL_COMPRESSOR=1
      DS4_ROCM_Q8_DECODE_PAIR_DP4A=0
      DS4_ROCM_Q8_BATCH_WMMA_M256_K128=1
      DS4_ROCM_Q4K_DECODE_SPLIT_GATE_UP=1
      DS4_ROCM_Q4K_WMMA_PAIR_GATE_UP=1
      DS4_ROCM_Q4K_WMMA_FUSE_MID=1
      DS4_ROCM_TP_SKIP_UNOWNED=1
      DS4_ROCM_TP_PREFILL_SKIP_UNOWNED="$TP_PREFILL_SKIP_UNOWNED"
      DS4_ROCM_SHARED_GU_SWIGLU_FUSE=1
    )
    EXPECT_GREEDY_TOP2=1
  fi
fi

# Distributional quality diagnostics require complete vocabulary logits.
# Production greedy-top2 deliberately exposes only two global candidates and
# therefore cannot serve as a full-logit oracle. This override is symmetric
# across ranks and applies only to the pre-timing diagnostic modes.
if [[ $DECODE_SELF_CHECK == 1 || $TEACHER_FORCE_CONTROL == 1 ||
      -n $FROZEN_TOKEN_FILE ]]; then
  CURRENT_OPT_ENV+=(DS4_TP_GREEDY_TOP2=0)
  EXPECT_GREEDY_TOP2=0
fi

[[ -n $COORDINATOR_ADDR ]] || {
  echo "error: set DS4_COORDINATOR_ADDR in bench.env.local or the invoking environment" >&2; exit 2;
}

if [[ $VALIDATE_CONFIG_ONLY == 1 ]]; then
  echo "validated_config model_arch=$MODEL_ARCH rdma_profile=$RDMA_PROFILE coordinator_addr=$COORDINATOR_ADDR coordinator_rdma_device=$LOCAL_RDMA_DEVICE worker_rdma_device=$PEER_RDMA_DEVICE rdma_gid_index=${RDMA_GID_INDEX:-n/a} prefill_chunk=$PREFILL_CHUNK prefill_batch=${GLM5_PREFILL_BATCH:-n/a} prefill_arg=$([[ $MODEL_ARCH == glm5-next ]] && echo omitted || echo passed) routed_expert_family=${ROUTED_FAMILY:-DSPARK} candidate=$CANDIDATE candidate_lane=$CANDIDATE_LANE baseline_id=${BASELINE_ID:-n/a} tp_prefill_skip_unowned=${TP_PREFILL_SKIP_UNOWNED:-n/a} q2_zero_weight_tile_skip=$([[ $MODEL_ARCH == deepseek4 && ${ROUTED_FAMILY:-} == HYBRID_Q2 ]] && echo 1 || echo 0)"
  exit 0
fi

[[ -n $PEER_MGMT ]] || {
  echo "error: set DS4_PEER_MGMT in bench.env.local" >&2; exit 2;
}
COMMON_ENV=(DS4_TP_TIMEOUT_SEC="$TP_TIMEOUT_SEC" "${CURRENT_OPT_ENV[@]}")
RDMA_ARGS=(--rdma-device "$LOCAL_RDMA_DEVICE")
WORKER_RDMA_ARGS=(--rdma-device "$PEER_RDMA_DEVICE")
if [[ $RDMA_PROFILE == odinlink ]]; then
  ODINLINK_ROOT=${DS4_ODINLINK_ROOT:-}
  [[ -n $ODINLINK_ROOT ]] || {
    echo "error: set DS4_ODINLINK_ROOT in bench.env.local" >&2; exit 2;
  }
  VERBS_LIB=$ODINLINK_ROOT/build/verbs/libodl_tb5_verbs.so.0.1.0
  ODL_LD_PATH=$ODINLINK_ROOT/build/lib:$ODINLINK_ROOT/build/verbs
  COMMON_ENV+=(
    DS4_TP_ODINLINK_BATCH_ASYNC=1
    DS4_TP_VERBS_LIB="$VERBS_LIB"
    LD_LIBRARY_PATH="$ODL_LD_PATH"
  )
else
  RDMA_ARGS+=(--rdma-gid-index "$RDMA_GID_INDEX")
  WORKER_RDMA_ARGS+=(--rdma-gid-index "$RDMA_GID_INDEX")
fi
WORKER_ENV=("${COMMON_ENV[@]}")
COORD_ENV=("${COMMON_ENV[@]}")

# GLM-5.3 uses the GLM-DSA graph scheduler, which rejects the legacy
# --prefill-chunk option. DeepSeek and all other validated paths retain the
# explicit chunk argument and its provider-specific defaults.
PREFILL_ARGS=(--prefill-chunk "$PREFILL_CHUNK")
if [[ $MODEL_ARCH == glm5-next ]]; then
  PREFILL_ARGS=()
fi
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
if [[ -n $DUMP_FRONTIER_LOGITS_DIR ]]; then
  case $DUMP_FRONTIER_LOGITS_DIR/ in
    "$DS4_RESEARCH_ROOT/"*) ;;
    *)
      echo "error: frontier logits directory must be under $DS4_RESEARCH_ROOT" >&2
      exit 2
      ;;
  esac
  mkdir -p "$DUMP_FRONTIER_LOGITS_DIR"
  COORD_ARGS+=(--dump-frontier-logits-dir "$DUMP_FRONTIER_LOGITS_DIR")
fi
if [[ -n $DUMP_GENERATED_TOKEN_FILE ]]; then
  case $DUMP_GENERATED_TOKEN_FILE in
    "$DS4_RESEARCH_ROOT"/*) ;;
    *) echo "error: generated-token file must be under $DS4_RESEARCH_ROOT" >&2; exit 2 ;;
  esac
  [[ $FRONTIER == "$FRONTIER_MAX" && $TOKENS -gt 0 ]] || {
    echo "error: generated-token export requires one fixed frontier and positive generation" >&2
    exit 2
  }
  [[ $CANDIDATE == 0 && ! -e $DUMP_GENERATED_TOKEN_FILE ]] || {
    echo "error: generated-token export is diagnostic-only and must not overwrite evidence" >&2
    exit 2
  }
  mkdir -p "$(dirname -- "$DUMP_GENERATED_TOKEN_FILE")"
  COORD_ARGS+=(--dump-generated-token-file "$DUMP_GENERATED_TOKEN_FILE")
fi
if [[ -n $FROZEN_TOKEN_FILE ]]; then
  [[ -r $FROZEN_TOKEN_FILE ]] || {
    echo "error: missing frozen token file: $FROZEN_TOKEN_FILE" >&2
    exit 1
  }
  case $FROZEN_TOKEN_FILE in
    "$DS4_RESEARCH_ROOT"/*) ;;
    *) echo "error: frozen token file must be under $DS4_RESEARCH_ROOT" >&2; exit 2 ;;
  esac
  case $FROZEN_LOGITS_DIR/ in
    "$DS4_RESEARCH_ROOT/"*) ;;
    *) echo "error: frozen logits directory must be under $DS4_RESEARCH_ROOT" >&2; exit 2 ;;
  esac
  mkdir -p "$FROZEN_LOGITS_DIR"
  if find "$FROZEN_LOGITS_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "error: frozen logits directory must be empty: $FROZEN_LOGITS_DIR" >&2
    exit 2
  fi
  COORD_ARGS+=(--teacher-force-token-file "$FROZEN_TOKEN_FILE"
               --teacher-force-logits-dir "$FROZEN_LOGITS_DIR")
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
REMOTE_WORKER_LOG="$PEER_OUT/worker-$TAG.log"
CSV="$OUT/$TAG.csv"
MANIFEST="$OUT/$TAG.manifest"
WORKER_PIDFILE="$PEER_OUT/worker-$TAG.pid"
WORKER_STATUSFILE="$PEER_OUT/worker-$TAG.status"
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

if [[ $DSPARK == 1 && ( -z $MTP || ! -r $MTP ) ]]; then
  echo "error: missing local DSpark model: $MTP" >&2
  exit 1
fi
if [[ $RDMA_PROFILE == odinlink ]]; then
  [[ -r /dev/odl_tb5_0 && -r $VERBS_LIB ]] || {
    echo "error: local OdinLink device or provider unavailable" >&2; exit 1;
  }
  "${PEER_SSH[@]}" "test -r '$MODEL' -a -r /dev/odl_tb5_0 -a -r '$VERBS_LIB'" || {
    echo "error: peer model, OdinLink device, or provider unavailable" >&2; exit 1;
  }
else
  [[ -r /sys/class/infiniband/$LOCAL_RDMA_DEVICE/ports/1/gid_attrs/types/$RDMA_GID_INDEX ]] || {
    echo "error: local mlx5 RoCE device unavailable: $LOCAL_RDMA_DEVICE" >&2; exit 1;
  }
  grep -qx 'RoCE v2' \
    "/sys/class/infiniband/$LOCAL_RDMA_DEVICE/ports/1/gid_attrs/types/$RDMA_GID_INDEX" || {
    echo "error: local GID index $RDMA_GID_INDEX is not RoCE v2" >&2; exit 1;
  }
  "${PEER_SSH[@]}" "test -r '$MODEL' && test \"\$(cat '/sys/class/infiniband/$PEER_RDMA_DEVICE/ports/1/gid_attrs/types/$RDMA_GID_INDEX')\" = 'RoCE v2'" || {
    echo "error: peer model or RoCE v2 GID unavailable" >&2; exit 1;
  }
fi
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
command -v readelf >/dev/null 2>&1 || {
  echo "error: readelf is required to bind ROCm compiler provenance" >&2; exit 1;
}
BINARY_TOOLCHAIN_COMMENT=$(LC_ALL=C readelf -p .comment "$REPO/ds4" 2>/dev/null |
  sed -n 's/^  \[[^]]*\]  //p' |
  grep -E 'AMD clang version|Linker: AMD LLD' |
  paste -sd ';' -)
[[ -n $BINARY_TOOLCHAIN_COMMENT ]] || {
  echo "error: ds4 binary has no AMD compiler/linker provenance" >&2; exit 1;
}
BINARY_TOOLCHAIN_SHA256=$(printf '%s\n' "$BINARY_TOOLCHAIN_COMMENT" |
  sha256sum | awk '{print $1}')
[[ -z $EXPECTED_TOOLCHAIN_SHA256 ||
   $EXPECTED_TOOLCHAIN_SHA256 == "$BINARY_TOOLCHAIN_SHA256" ]] || {
  echo "error: binary toolchain fingerprint mismatch: expected=$EXPECTED_TOOLCHAIN_SHA256 actual=$BINARY_TOOLCHAIN_SHA256" >&2
  exit 1
}
BINARY_RUNPATH=$(LC_ALL=C readelf -d "$REPO/ds4" 2>/dev/null |
  sed -n 's/.*\(RPATH\|RUNPATH\).*\[\(.*\)\]/\2/p' |
  paste -sd ';' -)
[[ -n $BINARY_RUNPATH ]] || BINARY_RUNPATH=none
TOOLCHAIN_ID=${DECLARED_TOOLCHAIN_ID:-elf-comment-sha256:$BINARY_TOOLCHAIN_SHA256}
LOCAL_BENCH_HASH=$(sha256sum "$REPO/ds4-bench-tp" | awk '{print $1}')
PROMPT_HASH=$(sha256sum "$PROMPT_FILE" | awk '{print $1}')
PROMPT_SIZE=$(stat -c %s "$PROMPT_FILE")
if [[ -n $FROZEN_TOKEN_FILE ]]; then
  FROZEN_TOKEN_HASH=$(sha256sum "$FROZEN_TOKEN_FILE" | awk '{print $1}')
else
  FROZEN_TOKEN_HASH=
fi

# Preserve the complete non-secret control configuration beside every run.
# Performance numbers without this record are not eligible for a baseline or
# candidate comparison because a single environment drift can change both the
# token trajectory and the selected kernel path.
SOURCE_COMMIT=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)
RUN_ID="$(date -u +%Y%m%dT%H%M%S.%NZ)-$$-${RANDOM}"
if git -C "$REPO" diff --quiet --ignore-submodules -- 2>/dev/null &&
   git -C "$REPO" diff --cached --quiet --ignore-submodules -- 2>/dev/null; then
  SOURCE_DIRTY=0
else
  SOURCE_DIRTY=1
fi
if [[ -n $FROZEN_TOKEN_FILE && -n $(git -C "$REPO" ls-files --others --exclude-standard) ]]; then
  SOURCE_DIRTY=1
fi
printf -v COMMON_ENV_Q '%q ' "${COMMON_ENV[@]}"
printf -v WORKER_ENV_Q '%q ' "${WORKER_ENV[@]}"
printf -v COORD_ENV_Q '%q ' "${COORD_ENV[@]}"
printf -v EXTRA_ENV_Q '%q ' "${EXTRA_ENV[@]}"
{
  printf 'tag=%s\n' "$TAG"
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'source_commit=%s\n' "$SOURCE_COMMIT"
  printf 'source_dirty=%s\n' "$SOURCE_DIRTY"
  printf 'bench_config=%s\n' "$BENCH_CONFIG"
  printf 'bench_config_sha256=%s\n' "$BENCH_CONFIG_SHA256"
  printf 'ds4_sha256=%s\n' "$LOCAL_DS4_HASH"
  printf 'peer_ds4_sha256=%s\n' "$PEER_DS4_HASH"
  printf 'ds4_bench_tp_sha256=%s\n' "$LOCAL_BENCH_HASH"
  printf 'model=%s\n' "$MODEL"
  printf 'model_arch=%s\n' "$MODEL_ARCH"
  printf 'model_size=%s\n' "$LOCAL_MODEL_SIZE"
  printf 'model_sample_sha256=%s\n' "$LOCAL_MODEL_FINGERPRINT"
  printf 'prompt=%s\n' "$PROMPT_FILE"
  printf 'prompt_size=%s\n' "$PROMPT_SIZE"
  printf 'prompt_sha256=%s\n' "$PROMPT_HASH"
  printf 'dump_generated_token_file=%s\n' "$DUMP_GENERATED_TOKEN_FILE"
  printf 'frozen_token_file=%s\n' "$FROZEN_TOKEN_FILE"
  printf 'frozen_token_sha256=%s\n' "$FROZEN_TOKEN_HASH"
  printf 'frozen_logits_dir=%s\n' "$FROZEN_LOGITS_DIR"
  printf 'toolchain_id=%s\n' "$TOOLCHAIN_ID"
  printf 'binary_toolchain_sha256=%s\n' "$BINARY_TOOLCHAIN_SHA256"
  printf 'binary_toolchain_comment=%s\n' "$BINARY_TOOLCHAIN_COMMENT"
  printf 'binary_runpath=%s\n' "$BINARY_RUNPATH"
  printf 'expected_binary_toolchain_sha256=%s\n' "$EXPECTED_TOOLCHAIN_SHA256"
  printf 'rocprof_binary=%s\n' "$ROCPROF_RESOLVED"
  printf 'rocprof_sha256=%s\n' "$ROCPROF_SHA256"
  printf 'frontier=%s\n' "$FRONTIER"
  printf 'frontier_max=%s\n' "$FRONTIER_MAX"
  printf 'step_incr=%s\n' "$STEP_INCR"
  printf 'step_mul=%s\n' "$STEP_MUL"
  printf 'generated_tokens=%s\n' "$TOKENS"
  printf 'context=%s\n' "$CONTEXT"
  printf 'prefill_chunk=%s\n' "$PREFILL_CHUNK"
  printf 'prefill_batch=%s\n' "${GLM5_PREFILL_BATCH:-n/a}"
  printf 'rdma_profile=%s\n' "$RDMA_PROFILE"
  printf 'coordinator_addr=%s\n' "$COORDINATOR_ADDR"
  printf 'coordinator_rdma_device=%s\n' "$LOCAL_RDMA_DEVICE"
  printf 'worker_rdma_device=%s\n' "$PEER_RDMA_DEVICE"
  printf 'rdma_gid_index=%s\n' "${RDMA_GID_INDEX:-n/a}"
  printf 'candidate=%s\n' "$CANDIDATE"
  printf 'candidate_lane=%s\n' "$CANDIDATE_LANE"
  printf 'baseline_id=%s\n' "$BASELINE_ID"
  printf 'expected_fnv64=%s\n' "${EXPECTED_FNV64,,}"
  printf 'dspark=%s\n' "$DSPARK"
  printf 'common_env=%s\n' "$COMMON_ENV_Q"
  printf 'worker_env=%s\n' "$WORKER_ENV_Q"
  printf 'coordinator_env=%s\n' "$COORD_ENV_Q"
  printf 'extra_env=%s\n' "$EXTRA_ENV_Q"
} > "$MANIFEST"

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
  "${PEER_SSH[@]}" "test -r '$WORKER_PIDFILE' || exit 1; p=\$(cat '$WORKER_PIDFILE'); case \"\$p\" in ''|*[!0-9]*) exit 1;; esac; test -r /proc/\$p/cmdline || exit 1; tr '\\0' ' ' < /proc/\$p/cmdline | grep -q -- 'tp-worker-supervisor'"
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
  "${PEER_SSH[@]}" "p=\$(cat '$WORKER_PIDFILE'); case \"\$p\" in ''|*[!0-9]*) exit 1;; esac; test -r /proc/\$p/cmdline || exit 0; tr '\\0' ' ' < /proc/\$p/cmdline | grep -q -- 'tp-worker-supervisor' || exit 1; kill -TERM \"\$p\""
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
printf -v PEER_OUT_Q '%q' "$PEER_OUT"
"${PEER_SSH[@]}" "mkdir -p $PEER_OUT_Q"

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
        "$DS4_RESEARCH_ROOT/"*) ;;
        *)
          echo "error: graph dump directory must be under $DS4_RESEARCH_ROOT: $dump_dir" >&2
          exit 2
          ;;
      esac
      if [[ $DS4_RESEARCH_ROOT != "$DS4_PEER_RESEARCH_ROOT" ]]; then
        echo "error: graph dumps currently require matching canonical paths on both independent filesystems" >&2
        exit 2
      fi
      mkdir -p "$dump_dir"
      printf -v dump_dir_q '%q' "$dump_dir"
      "${PEER_SSH[@]}" "mkdir -p $dump_dir_q"
      ;;
  esac
done

echo "=== ds4-bench $TAG ==="
echo "model: $MODEL"
echo "workload: frontier=$FRONTIER frontier_max=$FRONTIER_MAX step_incr=$STEP_INCR step_mul=$STEP_MUL generated_tokens=$TOKENS context=$CONTEXT prefill_chunk=$PREFILL_CHUNK prefill_batch=${GLM5_PREFILL_BATCH:-n/a}"
echo "rdma_profile: $RDMA_PROFILE coordinator_device=$LOCAL_RDMA_DEVICE worker_device=$PEER_RDMA_DEVICE"
if [[ $DSPARK == 1 ]]; then echo "dspark: 1 mtp=$MTP"; else echo "dspark: 0"; fi
if [[ $DSPARK == 0 ]]; then echo "routed_expert_family: $ROUTED_FAMILY"; fi
if [[ $ROCPROF == 1 ]]; then echo "rocprof: binary=$ROCPROF_RESOLVED rank=$ROCPROF_RANK kernel trace (diagnostic; timing is not benchmark evidence)"; fi
echo "ds4_sha256: $LOCAL_DS4_HASH"
echo "ds4_bench_tp_sha256: $LOCAL_BENCH_HASH"
echo "model_sample_sha256: $LOCAL_MODEL_FINGERPRINT"
if [[ $DSPARK == 1 ]]; then echo "mtp_sample_sha256: $LOCAL_MTP_FINGERPRINT resident_q8=1"; fi

WORKER_APP=(./ds4
  --role worker --tensor-parallel --coordinator "$COORDINATOR_ADDR" 9000
  --transport rdma --rocm -m "$MODEL" -c "$CONTEXT"
  "${PREFILL_ARGS[@]}"
  "${WORKER_RDMA_ARGS[@]}"
  "${WORKER_ARGS[@]}")
WORKER_CMD=("${CLEAN_ENV[@]}" "${WORKER_ENV[@]}" "${WORKER_APP[@]}")
if [[ $ROCPROF == 1 && $ROCPROF_RANK == worker ]]; then
  ROCPROF_OUT="$PEER_OUT/rocprof-$TAG-worker"
  printf -v ROCPROF_OUT_Q '%q' "$ROCPROF_OUT"
  "${PEER_SSH[@]}" "mkdir -p $ROCPROF_OUT_Q"
  # The worker has no benchmark-level ROCTx boundary. Trace its complete
  # runtime and separate prefill by kernel family/count; model residency is
  # dominated by page warming and copies rather than these compute kernels.
  if [[ $ROCPROF_RUNTIME == 1 ]]; then
    WORKER_TRACE=(--runtime-trace)
  elif [[ $ROCPROF_RUNTIME == 2 ]]; then
    WORKER_TRACE=(--hip-runtime-trace --kernel-trace --marker-trace)
  else
    WORKER_TRACE=(--kernel-trace --marker-trace)
  fi
  WORKER_CMD=("${CLEAN_ENV[@]}" "${WORKER_ENV[@]}"
              "$ROCPROF_RESOLVED" "${WORKER_TRACE[@]}" --stats --summary
              --summary-units usec --output-directory "$ROCPROF_OUT" --
              "${WORKER_APP[@]}")
fi
printf -v WORKER_CMD_Q '%q ' "${WORKER_CMD[@]}"
printf -v PEER_REPO_Q '%q' "$PEER_REPO"
printf -v WORKER_LOG_Q '%q' "$REMOTE_WORKER_LOG"
printf -v WORKER_PIDFILE_Q '%q' "$WORKER_PIDFILE"
printf -v WORKER_STATUSFILE_Q '%q' "$WORKER_STATUSFILE"
SUPERVISOR="$REPO/scripts/tp-worker-supervisor.sh"
SUPERVISOR_HASH=$(sha256sum "$SUPERVISOR" | awk '{print $1}')
PEER_SUPERVISOR_HASH=$("${PEER_SSH[@]}" "sha256sum '$PEER_REPO/scripts/tp-worker-supervisor.sh' 2>/dev/null | awk '{print \$1}'")
if [[ $SUPERVISOR_HASH != "$PEER_SUPERVISOR_HASH" ]]; then
  "${PEER_SCP[@]}" "$SUPERVISOR" "$PEER_MGMT:$PEER_REPO/scripts/tp-worker-supervisor.sh"
  "${PEER_SSH[@]}" "chmod 755 '$PEER_REPO/scripts/tp-worker-supervisor.sh'"
fi
"${PEER_SSH[@]}" ": > $WORKER_STATUSFILE_Q; cd $PEER_REPO_Q || exit 1; nohup setsid '$PEER_REPO/scripts/tp-worker-supervisor.sh' $WORKER_STATUSFILE_Q $WORKER_CMD_Q > $WORKER_LOG_Q 2>&1 < /dev/null & p=\$!; echo \$p > $WORKER_PIDFILE_Q"
WORKER_STARTED=1

COORD_CMD=("$REPO/ds4-bench-tp")
if [[ $ROCPROF == 1 && $ROCPROF_RANK == coordinator ]]; then
  ROCPROF_OUT="$OUT/rocprof-$TAG"
  mkdir -p "$ROCPROF_OUT"
  if [[ $ROCPROF_RUNTIME == 1 ]]; then
    ROCPROF_TRACE=(--runtime-trace --selected-regions)
  elif [[ $ROCPROF_RUNTIME == 2 ]]; then
    ROCPROF_TRACE=(--hip-runtime-trace --kernel-trace --marker-trace
                   --selected-regions)
  else
    ROCPROF_TRACE=(--kernel-trace --marker-trace --selected-regions)
  fi
  COORD_ENV+=(DS4_BENCH_ROCPROF_SELECTED_REGIONS=1
             DS4_BENCH_ROCPROF_REGION="$ROCPROF_REGION")
  COORD_CMD=("$ROCPROF_RESOLVED" "${ROCPROF_TRACE[@]}"
             --stats --summary --summary-units usec
             --output-directory "$ROCPROF_OUT" -- "$REPO/ds4-bench-tp")
elif [[ $ROCPROF != 0 && $ROCPROF != 1 ]]; then
  echo "error: DS4_BENCH_ROCPROF must be 0 or 1" >&2
  exit 2
fi

"${CLEAN_ENV[@]}" "${COORD_ENV[@]}" "${COORD_CMD[@]}" \
  --role coordinator --tensor-parallel --listen 0.0.0.0 9000 \
  --transport rdma --rocm -m "$MODEL" --prompt-file "$PROMPT_FILE" \
  "${RDMA_ARGS[@]}" \
  --ctx-start "$FRONTIER" --ctx-max "$FRONTIER_MAX" \
  --step-incr "$STEP_INCR" --step-mul "$STEP_MUL" \
  --ctx-alloc "$CONTEXT" \
  "${PREFILL_ARGS[@]}" --gen-tokens "$TOKENS" --csv "$CSV" \
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
"${PEER_SCP[@]}" "$PEER_MGMT:$REMOTE_WORKER_LOG" "$WORKER_LOG"
if ! "${PEER_SSH[@]}" "test -r '$WORKER_STATUSFILE'"; then
  echo "error: worker exit-status record is missing" >&2
  exit 1
fi
"${PEER_SSH[@]}" "cat '$WORKER_STATUSFILE'" > "$OUT/worker-$TAG.status"
if [[ $MODEL_ARCH == glm5-next ]]; then
  "$REPO/scripts/check-glm5-prefill-proof.sh" \
    "$COORD_LOG" "$WORKER_LOG" "$GLM5_PREFILL_BATCH" "$FRONTIER" \
    "$GLM5_SPARSE_BATCH_BRIDGE"
fi
if [[ $DECODE_SELF_CHECK == 1 ]]; then
  grep -q 'ds4-bench: decode self-check complete .*argmax_mismatches=0 ' \
    "$COORD_LOG" || {
      echo "error: decode self-check did not prove zero argmax mismatches" >&2
      exit 1
    }
  "$REPO/scripts/check-tp-rdma-logs.sh" \
    "$COORD_LOG" "$WORKER_LOG" "$RDMA_PROFILE"
  echo "TP_DECODE_SELF_CHECK_PASSED"
  exit 0
fi
if [[ $TEACHER_FORCE_CONTROL == 1 ]]; then
  "$REPO/scripts/check-tp-rdma-logs.sh" \
    "$COORD_LOG" "$WORKER_LOG" "$RDMA_PROFILE"
  if [[ $RECORD_TEACHER_BASELINE == 1 ]]; then
    grep -E 'ds4-bench: teacher-force control complete .*mismatch_fnv64=[0-9a-f]{16} .*enforced=0' \
      "$COORD_LOG" || {
        echo "error: teacher-force baseline signature was not recorded" >&2
        exit 1
      }
    echo "TP_TEACHER_FORCE_BASELINE_RECORDED_NOT_VALIDATED"
    exit 0
  fi
  grep -qi "ds4-bench: teacher-force control complete .*mismatch_fnv64=${EXPECTED_TEACHER_FNV64} .*expected_fnv64=${EXPECTED_TEACHER_FNV64} enforced=1" \
    "$COORD_LOG" || {
      echo "error: teacher-force control did not match its enforced signature" >&2
      exit 1
    }
  echo "TP_TEACHER_FORCE_CONTROL_PASSED"
  exit 0
fi
if [[ -n $FROZEN_TOKEN_FILE ]]; then
  "$REPO/scripts/check-tp-rdma-logs.sh" \
    "$COORD_LOG" "$WORKER_LOG" "$RDMA_PROFILE"
  completion=$(grep -E 'ds4-bench: frozen teacher logits complete prefix=[0-9]+ tokens=[0-9]+ ' \
    "$COORD_LOG" | tail -1 || true)
  [[ -n $completion ]] || {
    echo "error: frozen teacher full-logit diagnostic did not complete" >&2
    exit 1
  }
  FROZEN_COUNT=$(sed -n 's/.* tokens=\([0-9][0-9]*\) .*/\1/p' <<<"$completion")
  FROZEN_PREFIX=$(sed -n 's/.* prefix=\([0-9][0-9]*\) .*/\1/p' <<<"$completion")
  ACTUAL_DUMPS=$(find "$FROZEN_LOGITS_DIR" -maxdepth 1 -type f \
    -name 'decode_*.logits.json' | wc -l)
  [[ $FROZEN_COUNT =~ ^[1-9][0-9]*$ && $ACTUAL_DUMPS == "$FROZEN_COUNT" ]] || {
    echo "error: frozen teacher dump count mismatch: expected=$FROZEN_COUNT actual=$ACTUAL_DUMPS" >&2
    exit 1
  }
  [[ $SOURCE_DIRTY == 0 ]] || {
    echo "error: frozen numerical evidence requires a clean source tree" >&2
    exit 1
  }
  {
    printf 'tag=%s\n' "$TAG"
    printf 'source_commit=%s\n' "$SOURCE_COMMIT"
    printf 'source_dirty=%s\n' "$SOURCE_DIRTY"
    printf 'toolchain_id=%s\n' "$TOOLCHAIN_ID"
    printf 'model=%s\n' "$MODEL"
    printf 'model_size=%s\n' "$LOCAL_MODEL_SIZE"
    printf 'model_sample_sha256=%s\n' "$LOCAL_MODEL_FINGERPRINT"
    printf 'ds4_sha256=%s\n' "$LOCAL_DS4_HASH"
    printf 'ds4_bench_tp_sha256=%s\n' "$LOCAL_BENCH_HASH"
    printf 'prompt_sha256=%s\n' "$PROMPT_HASH"
    printf 'prefix_tokens=%s\n' "$FROZEN_PREFIX"
    printf 'frozen_token_sha256=%s\n' "$FROZEN_TOKEN_HASH"
    printf 'file_count=%s\n' "$FROZEN_COUNT"
    printf 'rdma_profile=%s\n' "$RDMA_PROFILE"
    printf 'dspark=0\n'
  } > "$FROZEN_LOGITS_DIR/manifest"
  echo "frozen_logits_manifest=$FROZEN_LOGITS_DIR/manifest"
  echo "TP_FROZEN_TEACHER_LOGITS_RECORDED_NOT_BENCHMARKED"
  exit 0
fi
if [[ -n $DUMP_GENERATED_TOKEN_FILE ]]; then
  DUMP_GENERATED_TOKEN_COUNT=$(wc -l < "$DUMP_GENERATED_TOKEN_FILE")
  [[ $DUMP_GENERATED_TOKEN_COUNT == "$TOKENS" ]] || {
    echo "error: generated-token file count mismatch: expected=$TOKENS actual=$DUMP_GENERATED_TOKEN_COUNT" >&2
    exit 1
  }
  awk 'BEGIN { ok=1 } !/^[0-9]+$/ { ok=0 } END { exit !ok }' \
    "$DUMP_GENERATED_TOKEN_FILE" || {
      echo "error: generated-token file contains a malformed token ID" >&2
      exit 1
    }
  DUMP_GENERATED_TOKEN_SHA256=$(sha256sum "$DUMP_GENERATED_TOKEN_FILE" |
    awk '{print $1}')
  printf 'dump_generated_token_sha256=%s\n' \
    "$DUMP_GENERATED_TOKEN_SHA256" >> "$MANIFEST"
fi
"$REPO/scripts/check-ds4-bench-result.sh" \
  "$CSV" "$COORD_LOG" "$WORKER_LOG" "$EXPECTED_FNV64" "$TOKENS" "$CANDIDATE" \
  "$RDMA_PROFILE"
cat "$CSV"
echo RUN_DONE
