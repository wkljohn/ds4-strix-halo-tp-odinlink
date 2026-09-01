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

RDMA_PROFILE=${DS4_GLM5_RDMA_PROFILE:-roce-v2}
ODL_WC_STREAM_COPY=${DS4_GLM5_ODINLINK_WC_STREAM_COPY:-}
[[ -z $ODL_WC_STREAM_COPY || $ODL_WC_STREAM_COPY == 0 ||
   $ODL_WC_STREAM_COPY == 1 ]] || {
  echo "error: DS4_GLM5_ODINLINK_WC_STREAM_COPY must be 0 or 1" >&2
  exit 2
}
case $RDMA_PROFILE in
  roce-v2)
    [[ -z $ODL_WC_STREAM_COPY ]] || {
      echo "error: OdinLink WC-copy control is invalid for RoCE v2" >&2
      exit 2
    }
    HOST=${DS4_COORDINATOR_ADDR:-192.168.99.1}
    LOCAL_DEVICE=${DS4_LOCAL_RDMA_DEVICE:-mlx5_0}
    PEER_DEVICE=${DS4_PEER_RDMA_DEVICE:-mlx5_1}
    GID_INDEX=${DS4_RDMA_GID_INDEX:-3}
    LOCAL_VERBS_LIB=
    PEER_VERBS_LIB=
    ;;
  odinlink)
    HOST=${DS4_COORDINATOR_ADDR:-10.4.0.1}
    LOCAL_DEVICE=${DS4_LOCAL_RDMA_DEVICE:-odl_tb5_0}
    PEER_DEVICE=${DS4_PEER_RDMA_DEVICE:-odl_tb5_0}
    GID_INDEX=${DS4_RDMA_GID_INDEX:-}
    LOCAL_VERBS_LIB=${DS4_TP_VERBS_LIB:-}
    PEER_VERBS_LIB=${DS4_GLM5_PEER_VERBS_LIB:-$LOCAL_VERBS_LIB}
    [[ -n $LOCAL_VERBS_LIB && -f $LOCAL_VERBS_LIB ]] || {
      echo "error: OdinLink requires DS4_TP_VERBS_LIB naming its local standalone provider" >&2
      exit 2
    }
    [[ $LOCAL_VERBS_LIB == *odl_tb5_verbs* ]] || {
      echo "error: OdinLink provider path must name odl_tb5_verbs" >&2
      exit 2
    }
    [[ -n $PEER_VERBS_LIB ]] || {
      echo "error: OdinLink requires DS4_GLM5_PEER_VERBS_LIB or an identical provider path" >&2
      exit 2
    }
    LOCAL_ODL_LIB_DIR=${DS4_GLM5_ODINLINK_LIB_DIR:-$(dirname -- "$(dirname -- "$LOCAL_VERBS_LIB")")/lib}
    PEER_ODL_LIB_DIR=${DS4_GLM5_PEER_ODINLINK_LIB_DIR:-$(dirname -- "$(dirname -- "$PEER_VERBS_LIB")")/lib}
    for value in "$LOCAL_VERBS_LIB" "$PEER_VERBS_LIB" \
                 "$LOCAL_ODL_LIB_DIR" "$PEER_ODL_LIB_DIR"; do
      [[ $value != *"'"* && $value != *$'\n'* ]] || {
        echo "error: OdinLink library paths may not contain quotes or newlines" >&2
        exit 2
      }
    done
    [[ -f $LOCAL_ODL_LIB_DIR/libodl_tb5.so.0 ]] || {
      echo "error: local OdinLink base library is absent from $LOCAL_ODL_LIB_DIR" >&2
      exit 2
    }
    ;;
  *)
    echo "error: DS4_GLM5_RDMA_PROFILE must be roce-v2 or odinlink" >&2
    exit 2
    ;;
esac
PORT=${DS4_GLM5_PREFIX_TP_PORT:-15883}
TIMEOUT=${DS4_GLM5_TP_CONNECT_TIMEOUT_SEC:-180}
FULL_TRUNK=${DS4_GLM5_FULL_TRUNK:-0}
FULL_TOKENS=${DS4_GLM5_FULL_TOKENS:-1}
TEXT_PROMPT=${DS4_GLM5_TEXT_PROMPT:-}
TEXT_GENERATE=${DS4_GLM5_TEXT_GENERATE:-4}
TEXT_TEACHER_IDS=${DS4_GLM5_TEXT_TEACHER_IDS:-}
PERF_MODE=${DS4_GLM5_PERF_MODE:-0}
LOGIT_DUMP_STEP=${DS4_GLM5_TEXT_LOGIT_DUMP_STEP:-}
LOGIT_DUMP_ALL=${DS4_GLM5_TEXT_LOGIT_DUMP_ALL:-0}
LAYER_TIMING=${DS4_GLM5_LAYER_TIMING:-0}
BATCH_PREFILL_TEST=${DS4_GLM5_BATCH_PREFILL_TEST:-0}
BATCH_PREFILL_COMPARE=${DS4_GLM5_BATCH_PREFILL_COMPARE:-0}
KDA_ROUTED_BATCH_TEST=${DS4_GLM5_KDA_ROUTED_BATCH_TEST:-0}
KDA_ATTENTION_ONLY_TEST=${DS4_GLM5_KDA_ATTENTION_ONLY_TEST:-0}
KDA_ROUTED_BATCH_ROWS=${DS4_GLM5_KDA_ROUTED_BATCH_ROWS:-3}
KDA_ROUTED_PROFILE_REPEATS=${DS4_GLM5_KDA_ROUTED_PROFILE_REPEATS:-0}
KDA_ROUTED_CONTINUATION_ROWS=${DS4_GLM5_KDA_ROUTED_CONTINUATION_ROWS:-1}
MLA_ROUTED_BATCH_TEST=${DS4_GLM5_MLA_ROUTED_BATCH_TEST:-0}
MLA_ATTENTION_ONLY_TEST=${DS4_GLM5_MLA_ATTENTION_ONLY_TEST:-0}
MLA_ATTENTION_MODE=${DS4_GLM5_MLA_ATTENTION_MODE:-}
MLA_OUTPUT_DUMP=${DS4_GLM5_MLA_OUTPUT_DUMP:-}
MLA_SPARSE_BOUNDARY_TEST=${DS4_GLM5_MLA_SPARSE_BOUNDARY_TEST:-0}
MLA_LONGKEY_ATTENTION_TEST=${DS4_GLM5_MLA_LONGKEY_ATTENTION_TEST:-0}
MLA_ROUTED_BATCH_ROWS=${DS4_GLM5_MLA_ROUTED_BATCH_ROWS:-3}
MLA_ROUTED_PREFIX_ROWS=${DS4_GLM5_MLA_ROUTED_PREFIX_ROWS:-0}
MLA_ROUTED_CONTINUATION_ROWS=${DS4_GLM5_MLA_ROUTED_CONTINUATION_ROWS:-1}
ROCPROF_RANK=${DS4_GLM5_ROCPROF_RANK:-}
BF16_TOKTILE_DISABLE=${DS4_ROCM_DISABLE_BF16_BATCH_TOKTILE:-}
BF16_LOWRANK128_TOKTILE_DISABLE=${DS4_ROCM_DISABLE_BF16_LOWRANK128_TOKTILE:-}
BF16_TAIL25_DISABLE=${DS4_ROCM_DISABLE_BF16_TAIL25_FUSION:-}
BF16_DECODE_MLP64_DISABLE=${DS4_ROCM_DISABLE_BF16_DECODE_MLP64:-}
BF16_TOKTILE_VERBOSE=${DS4_ROCM_BF16_BATCH_TOKTILE_VERBOSE:-}
Q4K_WMMA_MIN_COUNT=${DS4_ROCM_Q4K_WMMA_MIN_COUNT:-}
Q4K_WMMA_DISABLE=${DS4_ROCM_DISABLE_Q4K_WMMA:-}
BIGGATE_PROFILE=${DS4_TP_BIGGATE_PROFILE:-}
SMALL_GATE_DISABLE=${DS4_GLM5_DISABLE_SMALL_GATE:-}
KDA_TP=${DS4_GLM5_KDA_TP:-}
KDA_OUTPUT_KSLICE=${DS4_GLM5_KDA_OUTPUT_KSLICE:-}
GPU_ROW_GATE=${DS4_GLM5_GPU_ROW_GATE:-}
SMALL_GATE=${DS4_GLM5_SMALL_GATE:-}
RESIDENT_EXPERTS=${DS4_GLM5_NEXT_RESIDENT_EXPERTS:-}
WARM_RESIDENT=${DS4_GLM5_NEXT_WARM_RESIDENT:-}
TP_SKIP_UNOWNED=${DS4_ROCM_TP_PREFILL_SKIP_UNOWNED:-}
Q2_DOWN_FORCE_SCALAR=${DS4_ROCM_Q2_DOWN_FORCE_SCALAR:-}
EXPERT_TILE_M=${DS4_ROCM_EXPERT_TILE_M:-}
SHARED_SERIAL=${DS4_ROCM_GLM5_BATCH_SHARED_SERIAL:-}
SHARED_DOWN_SERIAL=${DS4_ROCM_GLM5_BATCH_SHARED_DOWN_SERIAL:-}
SHARED_DOWN_F32_ENABLE=${DS4_ROCM_GLM5_ENABLE_SHARED_DOWN_F32:-}
SHARED_DOWN_F32_DISABLE=${DS4_ROCM_GLM5_DISABLE_SHARED_DOWN_F32:-}
STRIDED_F32_ROWS_DISABLE=${DS4_ROCM_GLM5_DISABLE_STRIDED_F32_ROWS:-}
ZERO_WORKSPACE=${DS4_GLM5_ZERO_WORKSPACE:-}
ZERO_MOE_EACH_LAYER=${DS4_GLM5_ZERO_MOE_EACH_LAYER:-}
ZERO_PERSISTENT_MLA=${DS4_GLM5_ZERO_PERSISTENT_MLA:-}
DISABLE_Q4_BATCH_MLA_OUTPUT=${DS4_GLM5_DISABLE_Q4_BATCH_MLA_OUTPUT:-}
CAUSAL_ATTN_GEMM_NOPE=${DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE:-}
CAUSAL_ATTN_GEMM_NOPE_F32=${DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_F32:-}
KDA_BATCH_KSLICE_OUTPUT=${DS4_ROCM_GLM5_BATCH_KSLICE_OUTPUT:-}
KDA_KSLICE_ROUTE_TOKTILE=${DS4_ROCM_KSLICE_ROUTE_TOKTILE:-}
KDA_ATTENTION_MODE=${DS4_GLM5_KDA_ATTENTION_MODE:-}
KDA_OUTPUT_DUMP=${DS4_GLM5_KDA_OUTPUT_DUMP:-}
MOE_SERIAL=${DS4_ROCM_GLM5_BATCH_MOE_SERIAL:-}
IQ2_SORTED_DISABLE=${DS4_ROCM_DISABLE_RESIDENT_IQ2_SORTED:-}
Q8_MID_DOWN=${DS4_ROCM_GLM5_BATCH_Q8_MID_DOWN:-}
BATCH_LAYER_TRACE=${DS4_GLM5_BATCH_LAYER_TRACE:-}
TRACE_PREFIX=${DS4_GLM5_NEXT_TRACE_PREFIX:-}
TRACE_LAYER=${DS4_GLM5_NEXT_TRACE_LAYER:-}
TRACE_TOKEN=${DS4_GLM5_NEXT_TRACE_TOKEN:-}
TRACE_FFN_SAME_INPUT=${DS4_GLM5_TRACE_FFN_SAME_INPUT:-}
TRACE_POST_ONLY=${DS4_GLM5_TRACE_POST_ONLY:-}
ROUTE_HASH_TRACE=${DS4_GLM5_ROUTE_HASH_TRACE:-}
HC_HASH_TRACE=${DS4_GLM5_HC_HASH_TRACE:-}
DEVICE_LAYER_CAPTURE=${DS4_GLM5_DEVICE_LAYER_CAPTURE:-}
DEVICE_MLA_STAGE_CAPTURE=${DS4_GLM5_DEVICE_MLA_STAGE_CAPTURE:-}
LAYER_COMPLETION_DIAGNOSTIC=${DS4_GLM5_LAYER_COMPLETION_DIAGNOSTIC:-}
RDMA_HOST_COHERENT=${DS4_ROCM_RDMA_HOST_COHERENT:-}
RDMA_CACHE_FENCE=${DS4_ROCM_RDMA_CACHE_FENCE:-}
BF16_FULL_SPLIT_ORDER=${DS4_ROCM_BF16_FULL_SPLIT_ORDER:-}
EXPECTED_GENERATED_FNV=${DS4_GLM5_EXPECT_GENERATED_FNV:-}
PEER_DIR=${DS4_GLM5_PEER_TEST_DIR:-/home/wkljohn/Desktop/cc/glm5-node2-test/prefix-layer3}
PEER_RESEARCH_ROOT=${DS4_GLM5_PEER_RESEARCH_ROOT:-$DS4_RESEARCH_ROOT}
BINARY=$REPO/tests/test_rocm_glm5_prefix_layer3_tp
PEER_BINARY=$PEER_DIR/test_rocm_glm5_prefix_layer3_tp
OUT=$DS4_RESEARCH_ROOT/glm5-next-tp2/$TAG
EXPECTED_MODEL_SHA=${DS4_GLM5_MODEL_SHA256:-}
LOCAL_HOME=${HOME:?HOME is required}
PEER_HOME=$(ssh -o BatchMode=yes "$PEER" 'printf %s "$HOME"')

for value in "$HOST" "$LOCAL_DEVICE" "$PEER_DEVICE" "$PEER_DIR" \
             "$PEER_RESEARCH_ROOT" \
             "$LOCAL_HOME" "$PEER_HOME" "$TEXT_PROMPT" \
             "$LOCAL_VERBS_LIB" "$PEER_VERBS_LIB" "$TRACE_PREFIX" \
             "$TRACE_LAYER" "$TRACE_TOKEN" "$TRACE_FFN_SAME_INPUT" \
             "$MLA_ATTENTION_MODE" "$MLA_OUTPUT_DUMP"; do
  [[ $value != *"'"* && $value != *$'\n'* ]] || {
    echo "error: environment values may not contain quotes or newlines" >&2
    exit 2
  }
done
[[ -z $MLA_ATTENTION_MODE || $MLA_ATTENTION_MODE == sequential ||
   $MLA_ATTENTION_MODE == batch ]] || {
  echo "error: DS4_GLM5_MLA_ATTENTION_MODE must be sequential or batch" >&2
  exit 2
}
for value in "$CAUSAL_ATTN_GEMM_NOPE" "$CAUSAL_ATTN_GEMM_NOPE_F32"; do
  [[ -z $value || $value == 0 || $value == 1 ]] || {
    echo "error: GLM causal-attention GEMM selectors must be 0 or 1" >&2
    exit 2
  }
done
[[ $CAUSAL_ATTN_GEMM_NOPE_F32 != 1 ||
   $CAUSAL_ATTN_GEMM_NOPE == 1 ]] || {
  echo "error: FP32 NoPE GEMM requires DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE=1" >&2
  exit 2
}
[[ $TEXT_TEACHER_IDS != *"'"* &&
   $TEXT_TEACHER_IDS != *$'\n'* ]] || {
  echo "error: teacher IDs may not contain quotes or newlines" >&2
  exit 2
}
[[ $PORT =~ ^[0-9]+$ ]] && (( PORT >= 1024 && PORT <= 65535 )) || {
  echo "error: invalid port" >&2
  exit 2
}
[[ $TIMEOUT =~ ^[1-9][0-9]*$ ]] || {
  echo "error: invalid timeout" >&2
  exit 2
}
[[ -z $GID_INDEX || ($GID_INDEX =~ ^[0-9]+$ && GID_INDEX -le 255) ]] || {
  echo "error: DS4_RDMA_GID_INDEX must be empty or in 0..255" >&2
  exit 2
}
[[ $FULL_TRUNK == 0 || $FULL_TRUNK == 1 ]] || {
  echo "error: DS4_GLM5_FULL_TRUNK must be 0 or 1" >&2
  exit 2
}
[[ $FULL_TOKENS == 1 || $FULL_TOKENS == 2 ]] || {
  echo "error: DS4_GLM5_FULL_TOKENS must be 1 or 2" >&2
  exit 2
}
[[ $FULL_TRUNK == 1 || $FULL_TOKENS == 1 ]] || {
  echo "error: DS4_GLM5_FULL_TOKENS=2 requires DS4_GLM5_FULL_TRUNK=1" >&2
  exit 2
}
[[ $TEXT_GENERATE =~ ^[1-9][0-9]*$ ]] && (( TEXT_GENERATE <= 512 )) || {
  echo "error: DS4_GLM5_TEXT_GENERATE must be 1..512" >&2
  exit 2
}
[[ $PERF_MODE == 0 || $PERF_MODE == 1 ]] || {
  echo "error: DS4_GLM5_PERF_MODE must be 0 or 1" >&2
  exit 2
}
[[ -z $LOGIT_DUMP_STEP ||
   ($LOGIT_DUMP_STEP =~ ^[0-9]+$ &&
    $LOGIT_DUMP_STEP -lt $TEXT_GENERATE &&
    $PERF_MODE == 0 && -n $TEXT_TEACHER_IDS) ]] || {
  echo "error: DS4_GLM5_TEXT_LOGIT_DUMP_STEP requires a bounded full-logit teacher step" >&2
  exit 2
}
[[ $LOGIT_DUMP_ALL == 0 || $LOGIT_DUMP_ALL == 1 ]] || {
  echo "error: DS4_GLM5_TEXT_LOGIT_DUMP_ALL must be 0 or 1" >&2
  exit 2
}
[[ $LOGIT_DUMP_ALL == 0 ||
   (-z $LOGIT_DUMP_STEP && $PERF_MODE == 0 && -n $TEXT_TEACHER_IDS) ]] || {
  echo "error: all-position logit dump requires teacher-forced full-logit mode and no single dump step" >&2
  exit 2
}
[[ $LAYER_TIMING == 0 || $LAYER_TIMING == 1 ]] || {
  echo "error: DS4_GLM5_LAYER_TIMING must be 0 or 1" >&2
  exit 2
}
[[ $BATCH_PREFILL_TEST == 0 || $BATCH_PREFILL_TEST == 1 ]] || {
  echo "error: DS4_GLM5_BATCH_PREFILL_TEST must be 0 or 1" >&2
  exit 2
}
[[ $BATCH_PREFILL_TEST == 0 || -n $TEXT_PROMPT ]] || {
  echo "error: batch prefill test requires DS4_GLM5_TEXT_PROMPT" >&2
  exit 2
}
[[ $BATCH_PREFILL_COMPARE == 0 || $BATCH_PREFILL_COMPARE == 1 ]] || {
  echo "error: DS4_GLM5_BATCH_PREFILL_COMPARE must be 0 or 1" >&2
  exit 2
}
[[ $BATCH_PREFILL_COMPARE == 0 || $BATCH_PREFILL_TEST == 1 ]] || {
  echo "error: batch prefill comparison requires batch prefill test" >&2
  exit 2
}
[[ $KDA_ROUTED_BATCH_TEST == 0 || $KDA_ROUTED_BATCH_TEST == 1 ]] || {
  echo "error: DS4_GLM5_KDA_ROUTED_BATCH_TEST must be 0 or 1" >&2
  exit 2
}
[[ $KDA_ATTENTION_ONLY_TEST == 0 || $KDA_ATTENTION_ONLY_TEST == 1 ]] || {
  echo "error: DS4_GLM5_KDA_ATTENTION_ONLY_TEST must be 0 or 1" >&2
  exit 2
}
[[ $KDA_ATTENTION_ONLY_TEST == 0 || $KDA_ROUTED_BATCH_TEST == 1 ]] || {
  echo "error: KDA attention-only mode requires the KDA batch fixture" >&2
  exit 2
}
[[ $KDA_ROUTED_BATCH_TEST == 0 ||
   ($FULL_TRUNK == 1 && -z $TEXT_PROMPT) ]] || {
  echo "error: KDA routed batch test requires full trunk and no text prompt" >&2
  exit 2
}
[[ $KDA_ROUTED_BATCH_ROWS == 1 || $KDA_ROUTED_BATCH_ROWS == 3 ||
   $KDA_ROUTED_BATCH_ROWS == 33 ]] || {
  echo "error: DS4_GLM5_KDA_ROUTED_BATCH_ROWS must be 1, 3, or 33" >&2
  exit 2
}
[[ $KDA_ROUTED_PROFILE_REPEATS =~ ^[0-9]+$ &&
   $KDA_ROUTED_PROFILE_REPEATS -le 16 ]] || {
  echo "error: DS4_GLM5_KDA_ROUTED_PROFILE_REPEATS must be 0..16" >&2
  exit 2
}
[[ $KDA_ROUTED_PROFILE_REPEATS == 0 ||
   ($KDA_ROUTED_BATCH_TEST == 1 && $KDA_ROUTED_BATCH_ROWS == 33) ]] || {
  echo "error: KDA routed profile repeats require the 33-row batch test" >&2
  exit 2
}
[[ -z $KDA_KSLICE_ROUTE_TOKTILE ||
   $KDA_KSLICE_ROUTE_TOKTILE == 0 ||
   $KDA_KSLICE_ROUTE_TOKTILE == 1 ]] || {
  echo "error: DS4_ROCM_KSLICE_ROUTE_TOKTILE must be 0, 1, or unset" >&2
  exit 2
}
[[ $KDA_ROUTED_CONTINUATION_ROWS == 1 ||
   $KDA_ROUTED_CONTINUATION_ROWS == 16 ]] || {
  echo "error: DS4_GLM5_KDA_ROUTED_CONTINUATION_ROWS must be 1 or 16" >&2
  exit 2
}
[[ $MLA_ROUTED_BATCH_TEST == 0 || $MLA_ROUTED_BATCH_TEST == 1 ]] || {
  echo "error: DS4_GLM5_MLA_ROUTED_BATCH_TEST must be 0 or 1" >&2
  exit 2
}
[[ $MLA_ATTENTION_ONLY_TEST == 0 || $MLA_ATTENTION_ONLY_TEST == 1 ]] || {
  echo "error: DS4_GLM5_MLA_ATTENTION_ONLY_TEST must be 0 or 1" >&2
  exit 2
}
[[ $MLA_ATTENTION_ONLY_TEST == 0 || $MLA_ROUTED_BATCH_TEST == 1 ]] || {
  echo "error: MLA attention-only mode requires the MLA batch fixture" >&2
  exit 2
}
[[ $MLA_SPARSE_BOUNDARY_TEST == 0 || $MLA_SPARSE_BOUNDARY_TEST == 1 ]] || {
  echo "error: DS4_GLM5_MLA_SPARSE_BOUNDARY_TEST must be 0 or 1" >&2
  exit 2
}
[[ $MLA_SPARSE_BOUNDARY_TEST == 0 ||
   ($FULL_TRUNK == 0 && -z $TEXT_PROMPT &&
    $KDA_ROUTED_BATCH_TEST == 0 && $MLA_ROUTED_BATCH_TEST == 0) ]] || {
  echo "error: MLA sparse boundary must be the isolated layer-3 gate" >&2
  exit 2
}
[[ $MLA_LONGKEY_ATTENTION_TEST == 0 ||
   $MLA_LONGKEY_ATTENTION_TEST == 1 ]] || {
  echo "error: DS4_GLM5_MLA_LONGKEY_ATTENTION_TEST must be 0 or 1" >&2
  exit 2
}
[[ $MLA_LONGKEY_ATTENTION_TEST == 0 ||
   ($FULL_TRUNK == 0 && -z $TEXT_PROMPT &&
    $KDA_ROUTED_BATCH_TEST == 0 && $MLA_ROUTED_BATCH_TEST == 0 &&
    $MLA_SPARSE_BOUNDARY_TEST == 0) ]] || {
  echo "error: MLA long-key attention must be the isolated component gate" >&2
  exit 2
}
[[ $MLA_ROUTED_BATCH_TEST == 0 ||
   ($FULL_TRUNK == 1 && -z $TEXT_PROMPT) ]] || {
  echo "error: MLA routed batch test requires full trunk and no text prompt" >&2
  exit 2
}
[[ $MLA_ROUTED_BATCH_ROWS == 3 || $MLA_ROUTED_BATCH_ROWS == 5 ||
   $MLA_ROUTED_BATCH_ROWS == 33 || $MLA_ROUTED_BATCH_ROWS == 121 ||
   $MLA_ROUTED_BATCH_ROWS == 256 ]] || {
  echo "error: DS4_GLM5_MLA_ROUTED_BATCH_ROWS must be 3, 5, 33, 121, or 256" >&2
  exit 2
}
[[ $MLA_ROUTED_PREFIX_ROWS =~ ^[0-3]$ ]] || {
  echo "error: DS4_GLM5_MLA_ROUTED_PREFIX_ROWS must be 0, 1, 2, or 3" >&2
  exit 2
}
[[ $MLA_ROUTED_CONTINUATION_ROWS == 1 ||
   $MLA_ROUTED_CONTINUATION_ROWS == 16 ]] || {
  echo "error: DS4_GLM5_MLA_ROUTED_CONTINUATION_ROWS must be 1 or 16" >&2
  exit 2
}
[[ $MLA_ROUTED_CONTINUATION_ROWS == 1 ||
   ($MLA_ROUTED_BATCH_TEST == 1 && $MLA_ROUTED_BATCH_ROWS == 33) ]] || {
  echo "error: 16 MLA continuation rows require the 33-row batch test" >&2
  exit 2
}
[[ $KDA_ROUTED_BATCH_TEST == 0 || $MLA_ROUTED_BATCH_TEST == 0 ]] || {
  echo "error: select only one isolated routed batch test" >&2
  exit 2
}
[[ $KDA_ROUTED_CONTINUATION_ROWS == 1 ||
   ($KDA_ROUTED_BATCH_TEST == 1 && $KDA_ROUTED_BATCH_ROWS == 33) ]] || {
  echo "error: 16 continuation rows require the 33-row batch test" >&2
  exit 2
}
[[ -z $ROCPROF_RANK || $ROCPROF_RANK == leader ]] || {
  echo "error: DS4_GLM5_ROCPROF_RANK currently supports only leader" >&2
  exit 2
}
[[ -z $BF16_TOKTILE_DISABLE || $BF16_TOKTILE_DISABLE == 1 ]] || {
  echo "error: DS4_ROCM_DISABLE_BF16_BATCH_TOKTILE must be empty or 1" >&2
  exit 2
}
[[ -z $BF16_LOWRANK128_TOKTILE_DISABLE ||
   $BF16_LOWRANK128_TOKTILE_DISABLE == 1 ]] || {
  echo "error: DS4_ROCM_DISABLE_BF16_LOWRANK128_TOKTILE must be empty or 1" >&2
  exit 2
}
[[ -z $BF16_TAIL25_DISABLE || $BF16_TAIL25_DISABLE == 1 ]] || {
  echo "error: DS4_ROCM_DISABLE_BF16_TAIL25_FUSION must be empty or 1" >&2
  exit 2
}
[[ -z $BF16_TOKTILE_VERBOSE || $BF16_TOKTILE_VERBOSE == 1 ]] || {
  echo "error: DS4_ROCM_BF16_BATCH_TOKTILE_VERBOSE must be empty or 1" >&2
  exit 2
}
[[ -z $Q4K_WMMA_MIN_COUNT ||
   ($Q4K_WMMA_MIN_COUNT =~ ^[0-9]+$ &&
    $Q4K_WMMA_MIN_COUNT -ge 1 && $Q4K_WMMA_MIN_COUNT -le 16) ]] || {
  echo "error: DS4_ROCM_Q4K_WMMA_MIN_COUNT must be empty or 1..16" >&2
  exit 2
}
[[ -z $BIGGATE_PROFILE || $BIGGATE_PROFILE == 1 ]] || {
  echo "error: DS4_TP_BIGGATE_PROFILE must be empty or 1" >&2
  exit 2
}
[[ -z $SMALL_GATE_DISABLE || $SMALL_GATE_DISABLE == 1 ]] || {
  echo "error: DS4_GLM5_DISABLE_SMALL_GATE must be empty or 1" >&2
  exit 2
}
[[ -z $KDA_TP || $KDA_TP == 1 ]] || {
  echo "error: DS4_GLM5_KDA_TP must be empty or 1" >&2
  exit 2
}
[[ -z $KDA_OUTPUT_KSLICE || $KDA_OUTPUT_KSLICE == 1 ]] || {
  echo "error: DS4_GLM5_KDA_OUTPUT_KSLICE must be empty or 1" >&2
  exit 2
}
[[ -z $GPU_ROW_GATE || $GPU_ROW_GATE == 1 ]] || {
  echo "error: DS4_GLM5_GPU_ROW_GATE must be empty or 1" >&2
  exit 2
}
[[ -z $SMALL_GATE || $SMALL_GATE == 1 ]] || {
  echo "error: DS4_GLM5_SMALL_GATE must be empty or 1" >&2
  exit 2
}
[[ -z $KDA_OUTPUT_KSLICE || -n $KDA_TP ]] || {
  echo "error: DS4_GLM5_KDA_OUTPUT_KSLICE requires DS4_GLM5_KDA_TP=1" >&2
  exit 2
}
[[ -z $EXPECTED_GENERATED_FNV ||
   $EXPECTED_GENERATED_FNV =~ ^[0-9a-f]{16}$ ]] || {
  echo "error: DS4_GLM5_EXPECT_GENERATED_FNV must be a 16-digit lowercase FNV64" >&2
  exit 2
}
[[ $PERF_MODE == 0 ||
   (-n $TEXT_PROMPT && -z $TEXT_TEACHER_IDS &&
    -n $EXPECTED_GENERATED_FNV) ]] || {
  echo "error: performance mode requires greedy text and an expected generated FNV" >&2
  exit 2
}
[[ -z $EXPECTED_MODEL_SHA || $EXPECTED_MODEL_SHA =~ ^[0-9a-f]{64}$ ]] || {
  echo "error: DS4_GLM5_MODEL_SHA256 must be a lowercase SHA-256" >&2
  exit 2
}
[[ -z $TEXT_PROMPT || $FULL_TRUNK == 1 ]] || {
  echo "error: DS4_GLM5_TEXT_PROMPT requires DS4_GLM5_FULL_TRUNK=1" >&2
  exit 2
}
[[ -z $TEXT_TEACHER_IDS ||
   $TEXT_TEACHER_IDS =~ ^[0-9]+(,[0-9]+)*$ ]] || {
  echo "error: DS4_GLM5_TEXT_TEACHER_IDS must be comma-separated integers" >&2
  exit 2
}
[[ -z $TEXT_TEACHER_IDS || -n $TEXT_PROMPT ]] || {
  echo "error: DS4_GLM5_TEXT_TEACHER_IDS requires DS4_GLM5_TEXT_PROMPT" >&2
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

SOURCE_FILES=(
  Makefile .gitignore
  ds4.c ds4.h ds4_gpu.h ds4_tp.c ds4_tp.h
  ds4_glm5_kda.c ds4_glm5_kda.h
  ds4_glm5_next_runtime.c ds4_glm5_next_runtime.h
  ds4_glm5_next_state.c ds4_glm5_next_exec.c ds4_glm5_next_exec.h
  ds4_rocm_compat.cu rocm/ds4_rocm_common.cuh
  rocm/ds4_rocm_bf16_toktile.cuh
  rocm/ds4_rocm_matmul.cuh rocm/ds4_rocm_glm.cuh
  rocm/ds4_rocm_glm5_kda.cuh
  rocm/ds4_rocm_moe.cuh rocm/ds4_rocm_moe_launch.cuh
  rocm/ds4_rocm_runtime.cuh
  scripts/run-glm5-prefix-layer3-roce.sh
  tests/glm5_gguf_test.hpp tests/glm5_next_real_offsets.hpp
  tests/test_rocm_bf16_batch_gemm.cu
  tests/test_glm5_next_runtime_offsets.c
  tests/test_rocm_glm5_kda_bf16_rowslice.cu
  tests/test_rocm_glm5_kda_layer.cu
  tests/test_rocm_glm5_prefix_layer3_tp.cu
  tests/test_tp_glm5_phase_transition.cu tests/test_tp_hello.c
)
(cd "$REPO" && sha256sum "${SOURCE_FILES[@]}") >"$OUT/source-files.sha256"
(cd "$REPO" && git status --short) >"$OUT/source.status"
(cd "$REPO" && git diff --binary -- "${SOURCE_FILES[@]}") >"$OUT/source.diff"
for source_file in "${SOURCE_FILES[@]}"; do
  if ! (cd "$REPO" &&
        git ls-files --error-unmatch -- "$source_file" >/dev/null 2>&1); then
    set +e
    (cd "$REPO" &&
      git diff --no-index --binary -- /dev/null "$source_file") \
      >>"$OUT/source.diff"
    source_diff_rc=$?
    set -e
    [[ $source_diff_rc == 1 ]] || {
      echo "error: failed to archive untracked source $source_file" >&2
      exit 1
    }
  fi
done
SOURCE_DIFF_SHA=$(sha256sum "$OUT/source.diff" | awk '{print $1}')

make -C "$REPO" -j"$(nproc)" tests/test_rocm_glm5_prefix_layer3_tp
ssh -o BatchMode=yes "$PEER" \
  "mkdir -p -- '$PEER_DIR' '$PEER_RESEARCH_ROOT/glm5-next-tp2/$TAG'; test -f '$PEER_MODEL'"
if [[ $RDMA_PROFILE == odinlink ]]; then
  [[ -e /dev/odl_tb5_0 ]] || {
    echo "error: local OdinLink device is absent" >&2
    exit 1
  }
  ssh -o BatchMode=yes "$PEER" \
    "test -e /dev/odl_tb5_0; test -f '$PEER_VERBS_LIB'; test -f '$PEER_ODL_LIB_DIR/libodl_tb5.so.0'" || {
      echo "error: peer OdinLink device, provider, or base library is absent" >&2
      exit 1
    }
fi
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
MODEL_FULL_SHA=
if [[ -n $EXPECTED_MODEL_SHA ]]; then
  sha256sum "$MODEL" >"$OUT/model.local.sha256" &
  local_model_hash_pid=$!
  ssh -o BatchMode=yes "$PEER" "sha256sum '$PEER_MODEL'" \
    >"$OUT/model.peer.sha256" &
  peer_model_hash_pid=$!
  set +e
  wait "$local_model_hash_pid"; local_model_hash_rc=$?
  wait "$peer_model_hash_pid"; peer_model_hash_rc=$?
  set -e
  (( local_model_hash_rc == 0 && peer_model_hash_rc == 0 )) || {
    echo "error: full model SHA-256 failed on one or both nodes" >&2
    exit 1
  }
  LOCAL_FULL_SHA=$(awk '{print $1}' "$OUT/model.local.sha256")
  PEER_FULL_SHA=$(awk '{print $1}' "$OUT/model.peer.sha256")
  [[ $LOCAL_FULL_SHA == "$EXPECTED_MODEL_SHA" &&
     $PEER_FULL_SHA == "$EXPECTED_MODEL_SHA" ]] || {
    echo "error: full model SHA-256 does not match the required artifact" >&2
    exit 1
  }
  MODEL_FULL_SHA=$LOCAL_FULL_SHA
fi

printf '%s\n' \
  "tag=$TAG" \
  "source_head=$(git -C "$REPO" rev-parse HEAD)" \
  "test_source_sha256=$(sha256sum "$REPO/tests/test_rocm_glm5_prefix_layer3_tp.cu" | awk '{print $1}')" \
  "launcher_sha256=$(sha256sum "$REPO/scripts/run-glm5-prefix-layer3-roce.sh" | awk '{print $1}')" \
  "binary_sha256=$LOCAL_SHA" \
  "model_size=$LOCAL_SIZE" \
  "model_sample_sha256=$LOCAL_SAMPLE" \
  "model_full_sha256=$MODEL_FULL_SHA" \
  "source_diff_sha256=$SOURCE_DIFF_SHA" \
  "rdma_profile=$RDMA_PROFILE" \
  "host=$HOST" \
  "port=$PORT" \
  "local_device=$LOCAL_DEVICE" \
  "peer_device=$PEER_DEVICE" \
  "rdma_gid_index=$GID_INDEX" \
  "timeout_sec=$TIMEOUT" >"$OUT/run.env"
if [[ $RDMA_PROFILE == odinlink ]]; then
  LOCAL_VERBS_SHA=$(sha256sum "$LOCAL_VERBS_LIB" | awk '{print $1}')
  PEER_VERBS_SHA=$(ssh -o BatchMode=yes "$PEER" \
    "sha256sum '$PEER_VERBS_LIB'" | awk '{print $1}')
  [[ $LOCAL_VERBS_SHA == "$PEER_VERBS_SHA" ]] || {
    echo "error: OdinLink provider checksum mismatch" >&2
    exit 1
  }
  LOCAL_ODL_BASE_SHA=$(sha256sum "$LOCAL_ODL_LIB_DIR/libodl_tb5.so.0" |
    awk '{print $1}')
  PEER_ODL_BASE_SHA=$(ssh -o BatchMode=yes "$PEER" \
    "sha256sum '$PEER_ODL_LIB_DIR/libodl_tb5.so.0'" | awk '{print $1}')
  [[ $LOCAL_ODL_BASE_SHA == "$PEER_ODL_BASE_SHA" ]] || {
    echo "error: OdinLink base-library checksum mismatch" >&2
    exit 1
  }
  printf '%s\n' \
    "local_verbs_lib=$LOCAL_VERBS_LIB" \
    "local_verbs_sha256=$LOCAL_VERBS_SHA" \
    "peer_verbs_lib=$PEER_VERBS_LIB" \
    "peer_verbs_sha256=$PEER_VERBS_SHA" \
    "local_odinlink_lib_dir=$LOCAL_ODL_LIB_DIR" \
    "peer_odinlink_lib_dir=$PEER_ODL_LIB_DIR" \
    "odinlink_base_sha256=$LOCAL_ODL_BASE_SHA" \
    >>"$OUT/run.env"
fi
printf 'full_trunk=%s\n' "$FULL_TRUNK" >>"$OUT/run.env"
printf 'full_tokens=%s\n' "$FULL_TOKENS" >>"$OUT/run.env"
printf 'text_mode=%s\n' "$([[ -n $TEXT_PROMPT ]] && printf 1 || printf 0)" \
  >>"$OUT/run.env"
printf 'text_generate=%s\n' "$TEXT_GENERATE" >>"$OUT/run.env"
printf 'perf_mode=%s\n' "$PERF_MODE" >>"$OUT/run.env"
printf 'text_logit_dump_step=%s\n' "$LOGIT_DUMP_STEP" >>"$OUT/run.env"
printf 'text_logit_dump_all=%s\n' "$LOGIT_DUMP_ALL" >>"$OUT/run.env"
printf 'layer_timing=%s\n' "$LAYER_TIMING" >>"$OUT/run.env"
printf 'batch_prefill_test=%s\n' "$BATCH_PREFILL_TEST" >>"$OUT/run.env"
printf 'batch_prefill_compare=%s\n' "$BATCH_PREFILL_COMPARE" >>"$OUT/run.env"
printf 'kda_routed_batch_test=%s\n' "$KDA_ROUTED_BATCH_TEST" >>"$OUT/run.env"
printf 'kda_attention_only_test=%s\n' "$KDA_ATTENTION_ONLY_TEST" >>"$OUT/run.env"
printf 'kda_routed_batch_rows=%s\n' "$KDA_ROUTED_BATCH_ROWS" >>"$OUT/run.env"
printf 'kda_routed_profile_repeats=%s\n' "$KDA_ROUTED_PROFILE_REPEATS" >>"$OUT/run.env"
printf 'kda_routed_continuation_rows=%s\n' "$KDA_ROUTED_CONTINUATION_ROWS" >>"$OUT/run.env"
printf 'mla_routed_batch_test=%s\n' "$MLA_ROUTED_BATCH_TEST" >>"$OUT/run.env"
printf 'mla_attention_only_test=%s\n' "$MLA_ATTENTION_ONLY_TEST" >>"$OUT/run.env"
printf 'mla_sparse_boundary_test=%s\n' "$MLA_SPARSE_BOUNDARY_TEST" >>"$OUT/run.env"
printf 'mla_longkey_attention_test=%s\n' "$MLA_LONGKEY_ATTENTION_TEST" >>"$OUT/run.env"
printf 'mla_routed_batch_rows=%s\n' "$MLA_ROUTED_BATCH_ROWS" >>"$OUT/run.env"
printf 'mla_routed_prefix_rows=%s\n' "$MLA_ROUTED_PREFIX_ROWS" >>"$OUT/run.env"
printf 'mla_routed_continuation_rows=%s\n' "$MLA_ROUTED_CONTINUATION_ROWS" >>"$OUT/run.env"
printf 'rocprof_rank=%s\n' "$ROCPROF_RANK" >>"$OUT/run.env"
printf 'bf16_toktile_disabled=%s\n' "$([[ -n $BF16_TOKTILE_DISABLE ]] && printf 1 || printf 0)" >>"$OUT/run.env"
printf 'bf16_lowrank128_toktile_disabled=%s\n' \
  "$([[ -n $BF16_LOWRANK128_TOKTILE_DISABLE ]] && printf 1 || printf 0)" \
  >>"$OUT/run.env"
printf 'bf16_tail25_disabled=%s\n' \
  "$([[ -n $BF16_TAIL25_DISABLE ]] && printf 1 || printf 0)" \
  >>"$OUT/run.env"
printf 'bf16_decode_mlp64_disabled=%s\n' \
  "$([[ -n $BF16_DECODE_MLP64_DISABLE ]] && printf 1 || printf 0)" \
  >>"$OUT/run.env"
printf 'bf16_toktile_verbose=%s\n' "$([[ -n $BF16_TOKTILE_VERBOSE ]] && printf 1 || printf 0)" >>"$OUT/run.env"
printf 'q4k_wmma_min_count=%s\n' "${Q4K_WMMA_MIN_COUNT:-default}" >>"$OUT/run.env"
printf 'biggate_profile=%s\n' "$([[ -n $BIGGATE_PROFILE ]] && printf 1 || printf 0)" >>"$OUT/run.env"
printf 'small_gate_disabled=%s\n' \
  "$([[ -n $SMALL_GATE_DISABLE ]] && printf 1 || printf 0)" \
  >>"$OUT/run.env"
printf 'kda_tp=%s\n' "$([[ -n $KDA_TP ]] && printf 1 || printf 0)" \
  >>"$OUT/run.env"
printf 'kda_output_kslice=%s\n' \
  "$([[ -n $KDA_OUTPUT_KSLICE ]] && printf 1 || printf 0)" \
  >>"$OUT/run.env"
printf 'resident_experts=%s\n' \
  "$([[ -n $RESIDENT_EXPERTS ]] && printf 1 || printf 0)" >>"$OUT/run.env"
printf 'warm_resident=%s\n' \
  "$([[ -n $WARM_RESIDENT ]] && printf 1 || printf 0)" >>"$OUT/run.env"
printf 'tp_skip_unowned=%s\n' \
  "$([[ -n $TP_SKIP_UNOWNED ]] && printf 1 || printf 0)" >>"$OUT/run.env"
printf 'q8_mid_down=%s\n' \
  "$([[ -n $Q8_MID_DOWN ]] && printf 1 || printf 0)" >>"$OUT/run.env"
printf 'odinlink_wc_stream_copy=%s\n' \
  "${ODL_WC_STREAM_COPY:-default}" >>"$OUT/run.env"
printf 'shared_down_f32_disabled=%s\n' \
  "$([[ -n $SHARED_DOWN_F32_DISABLE ]] && printf 1 || printf 0)" \
  >>"$OUT/run.env"
printf 'shared_down_f32_enabled=%s\n' \
  "$([[ -n $SHARED_DOWN_F32_ENABLE ]] && printf 1 || printf 0)" \
  >>"$OUT/run.env"
printf 'strided_f32_rows_disabled=%s\n' \
  "$([[ -n $STRIDED_F32_ROWS_DISABLE ]] && printf 1 || printf 0)" \
  >>"$OUT/run.env"
printf 'zero_workspace=%s\nq4_batch_mla_output_disabled=%s\n' \
  "$([[ -n $ZERO_WORKSPACE ]] && printf 1 || printf 0)" \
  "$([[ -n $DISABLE_Q4_BATCH_MLA_OUTPUT ]] && printf 1 || printf 0)" \
  >>"$OUT/run.env"
printf 'batch_layer_trace=%s\ntrace_layer=%s\ntrace_token=%s\n' \
  "$([[ -n $BATCH_LAYER_TRACE ]] && printf 1 || printf 0)" \
  "$TRACE_LAYER" "$TRACE_TOKEN" >>"$OUT/run.env"
printf 'trace_ffn_same_input=%s\n' \
  "$([[ -n $TRACE_FFN_SAME_INPUT ]] && printf 1 || printf 0)" \
  >>"$OUT/run.env"
printf 'trace_post_only=%s\n' \
  "$([[ -n $TRACE_POST_ONLY ]] && printf 1 || printf 0)" \
  >>"$OUT/run.env"
printf 'route_hash_trace=%s\n' \
  "$([[ -n $ROUTE_HASH_TRACE ]] && printf 1 || printf 0)" \
  >>"$OUT/run.env"
printf 'expected_generated_fnv=%s\n' "$EXPECTED_GENERATED_FNV" >>"$OUT/run.env"
if [[ -n $TEXT_PROMPT ]]; then
  printf 'text_prompt=%s\n' "$TEXT_PROMPT" >>"$OUT/run.env"
  printf 'text_prompt_sha256=%s\n' \
    "$(printf %s "$TEXT_PROMPT" | sha256sum | awk '{print $1}')" \
    >>"$OUT/run.env"
fi
if [[ -n $TEXT_TEACHER_IDS ]]; then
  printf 'text_teacher_ids=%s\n' "$TEXT_TEACHER_IDS" >>"$OUT/run.env"
  printf 'text_teacher_ids_sha256=%s\n' \
    "$(printf %s "$TEXT_TEACHER_IDS" | sha256sum | awk '{print $1}')" \
    >>"$OUT/run.env"
fi
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
text_env=()
remote_text_env=
local_rdma_env=()
remote_rdma_env=
local_candidate_env=()
remote_candidate_env=
if [[ -n $GID_INDEX ]]; then
  local_rdma_env+=(DS4_GLM5_TP_RDMA_GID_INDEX="$GID_INDEX")
  remote_rdma_env+=" DS4_GLM5_TP_RDMA_GID_INDEX='$GID_INDEX'"
fi
if [[ $RDMA_PROFILE == odinlink ]]; then
  local_rdma_env+=(DS4_TP_VERBS_LIB="$LOCAL_VERBS_LIB")
  local_rdma_env+=(LD_LIBRARY_PATH="$LOCAL_ODL_LIB_DIR")
  remote_rdma_env+=" DS4_TP_VERBS_LIB='$PEER_VERBS_LIB' LD_LIBRARY_PATH='$PEER_ODL_LIB_DIR'"
  if [[ -n $ODL_WC_STREAM_COPY ]]; then
    local_rdma_env+=(ODL_VERBS_WC_STREAM_COPY="$ODL_WC_STREAM_COPY")
    remote_rdma_env+=" ODL_VERBS_WC_STREAM_COPY='$ODL_WC_STREAM_COPY'"
  fi
fi
if [[ -n $BF16_TOKTILE_DISABLE ]]; then
  local_candidate_env+=(DS4_ROCM_DISABLE_BF16_BATCH_TOKTILE=1)
  remote_candidate_env+=' DS4_ROCM_DISABLE_BF16_BATCH_TOKTILE=1'
fi
if [[ -n $BF16_LOWRANK128_TOKTILE_DISABLE ]]; then
  local_candidate_env+=(DS4_ROCM_DISABLE_BF16_LOWRANK128_TOKTILE=1)
  remote_candidate_env+=' DS4_ROCM_DISABLE_BF16_LOWRANK128_TOKTILE=1'
fi
if [[ -n $BF16_TAIL25_DISABLE ]]; then
  local_candidate_env+=(DS4_ROCM_DISABLE_BF16_TAIL25_FUSION=1)
  remote_candidate_env+=' DS4_ROCM_DISABLE_BF16_TAIL25_FUSION=1'
fi
if [[ -n $BF16_DECODE_MLP64_DISABLE ]]; then
  local_candidate_env+=(DS4_ROCM_DISABLE_BF16_DECODE_MLP64=1)
  remote_candidate_env+=' DS4_ROCM_DISABLE_BF16_DECODE_MLP64=1'
fi
if [[ -n $BF16_TOKTILE_VERBOSE ]]; then
  local_candidate_env+=(DS4_ROCM_BF16_BATCH_TOKTILE_VERBOSE=1)
  remote_candidate_env+=' DS4_ROCM_BF16_BATCH_TOKTILE_VERBOSE=1'
fi
if [[ -n $Q4K_WMMA_MIN_COUNT ]]; then
  local_candidate_env+=(
    DS4_ROCM_Q4K_WMMA_MIN_COUNT="$Q4K_WMMA_MIN_COUNT")
  remote_candidate_env+=" DS4_ROCM_Q4K_WMMA_MIN_COUNT='$Q4K_WMMA_MIN_COUNT'"
fi
if [[ -n $Q4K_WMMA_DISABLE ]]; then
  local_candidate_env+=(DS4_ROCM_DISABLE_Q4K_WMMA="$Q4K_WMMA_DISABLE")
  remote_candidate_env+=" DS4_ROCM_DISABLE_Q4K_WMMA='$Q4K_WMMA_DISABLE'"
fi
if [[ -n $BIGGATE_PROFILE ]]; then
  local_candidate_env+=(DS4_TP_BIGGATE_PROFILE=1)
  remote_candidate_env+=' DS4_TP_BIGGATE_PROFILE=1'
fi
if [[ -n $SMALL_GATE_DISABLE ]]; then
  local_candidate_env+=(DS4_GLM5_DISABLE_SMALL_GATE=1)
  remote_candidate_env+=' DS4_GLM5_DISABLE_SMALL_GATE=1'
fi
if [[ -n $KDA_TP ]]; then
  local_candidate_env+=(DS4_GLM5_KDA_TP=1)
  remote_candidate_env+=' DS4_GLM5_KDA_TP=1'
fi
if [[ -n $KDA_OUTPUT_KSLICE ]]; then
  local_candidate_env+=(DS4_GLM5_KDA_OUTPUT_KSLICE=1)
  remote_candidate_env+=' DS4_GLM5_KDA_OUTPUT_KSLICE=1'
fi
if [[ -n $GPU_ROW_GATE ]]; then
  local_candidate_env+=(DS4_GLM5_GPU_ROW_GATE=1)
  remote_candidate_env+=' DS4_GLM5_GPU_ROW_GATE=1'
fi
if [[ -n $SMALL_GATE ]]; then
  local_candidate_env+=(DS4_GLM5_SMALL_GATE=1)
  remote_candidate_env+=' DS4_GLM5_SMALL_GATE=1'
fi
if [[ ${DS4_GLM5_ALLOW_PREFIX_MISMATCH:-} == 1 ]]; then
  local_candidate_env+=(DS4_GLM5_ALLOW_PREFIX_MISMATCH=1)
  remote_candidate_env+=' DS4_GLM5_ALLOW_PREFIX_MISMATCH=1'
fi
if [[ -n $RESIDENT_EXPERTS ]]; then
  local_candidate_env+=(DS4_GLM5_NEXT_RESIDENT_EXPERTS="$RESIDENT_EXPERTS")
  remote_candidate_env+=" DS4_GLM5_NEXT_RESIDENT_EXPERTS='$RESIDENT_EXPERTS'"
fi
if [[ -n $TP_SKIP_UNOWNED ]]; then
  local_candidate_env+=(DS4_ROCM_TP_PREFILL_SKIP_UNOWNED="$TP_SKIP_UNOWNED")
  remote_candidate_env+=" DS4_ROCM_TP_PREFILL_SKIP_UNOWNED='$TP_SKIP_UNOWNED'"
fi
if [[ -n $Q2_DOWN_FORCE_SCALAR ]]; then
  local_candidate_env+=(DS4_ROCM_Q2_DOWN_FORCE_SCALAR="$Q2_DOWN_FORCE_SCALAR")
  remote_candidate_env+=" DS4_ROCM_Q2_DOWN_FORCE_SCALAR='$Q2_DOWN_FORCE_SCALAR'"
fi
if [[ -n $EXPERT_TILE_M ]]; then
  local_candidate_env+=(DS4_ROCM_EXPERT_TILE_M="$EXPERT_TILE_M")
  remote_candidate_env+=" DS4_ROCM_EXPERT_TILE_M='$EXPERT_TILE_M'"
fi
if [[ -n $SHARED_SERIAL ]]; then
  local_candidate_env+=(DS4_ROCM_GLM5_BATCH_SHARED_SERIAL="$SHARED_SERIAL")
  remote_candidate_env+=" DS4_ROCM_GLM5_BATCH_SHARED_SERIAL='$SHARED_SERIAL'"
fi
if [[ -n $KDA_BATCH_KSLICE_OUTPUT ]]; then
  local_candidate_env+=(DS4_ROCM_GLM5_BATCH_KSLICE_OUTPUT="$KDA_BATCH_KSLICE_OUTPUT")
  remote_candidate_env+=" DS4_ROCM_GLM5_BATCH_KSLICE_OUTPUT='$KDA_BATCH_KSLICE_OUTPUT'"
fi
if [[ $KDA_KSLICE_ROUTE_TOKTILE == 1 ]]; then
  local_candidate_env+=(DS4_ROCM_KSLICE_ROUTE_TOKTILE=1)
  remote_candidate_env+=' DS4_ROCM_KSLICE_ROUTE_TOKTILE=1'
fi
if [[ -n $KDA_ATTENTION_MODE ]]; then
  local_candidate_env+=(DS4_GLM5_KDA_ATTENTION_MODE="$KDA_ATTENTION_MODE")
  remote_candidate_env+=" DS4_GLM5_KDA_ATTENTION_MODE='$KDA_ATTENTION_MODE'"
fi
if [[ -n $KDA_OUTPUT_DUMP ]]; then
  local_candidate_env+=(DS4_GLM5_KDA_OUTPUT_DUMP="$KDA_OUTPUT_DUMP")
  remote_candidate_env+=" DS4_GLM5_KDA_OUTPUT_DUMP='$KDA_OUTPUT_DUMP'"
fi
if [[ -n $MOE_SERIAL ]]; then
  local_candidate_env+=(DS4_ROCM_GLM5_BATCH_MOE_SERIAL="$MOE_SERIAL")
  remote_candidate_env+=" DS4_ROCM_GLM5_BATCH_MOE_SERIAL='$MOE_SERIAL'"
fi
if [[ -n $IQ2_SORTED_DISABLE ]]; then
  local_candidate_env+=(DS4_ROCM_DISABLE_RESIDENT_IQ2_SORTED="$IQ2_SORTED_DISABLE")
  remote_candidate_env+=" DS4_ROCM_DISABLE_RESIDENT_IQ2_SORTED='$IQ2_SORTED_DISABLE'"
fi
if [[ -n $Q8_MID_DOWN ]]; then
  local_candidate_env+=(DS4_ROCM_GLM5_BATCH_Q8_MID_DOWN="$Q8_MID_DOWN")
  remote_candidate_env+=" DS4_ROCM_GLM5_BATCH_Q8_MID_DOWN='$Q8_MID_DOWN'"
fi
if [[ -n $SHARED_DOWN_SERIAL ]]; then
  local_candidate_env+=(DS4_ROCM_GLM5_BATCH_SHARED_DOWN_SERIAL="$SHARED_DOWN_SERIAL")
  remote_candidate_env+=" DS4_ROCM_GLM5_BATCH_SHARED_DOWN_SERIAL='$SHARED_DOWN_SERIAL'"
fi
if [[ -n $SHARED_DOWN_F32_DISABLE ]]; then
  local_candidate_env+=(DS4_ROCM_GLM5_DISABLE_SHARED_DOWN_F32="$SHARED_DOWN_F32_DISABLE")
  remote_candidate_env+=" DS4_ROCM_GLM5_DISABLE_SHARED_DOWN_F32='$SHARED_DOWN_F32_DISABLE'"
fi
if [[ -n $SHARED_DOWN_F32_ENABLE ]]; then
  local_candidate_env+=(DS4_ROCM_GLM5_ENABLE_SHARED_DOWN_F32="$SHARED_DOWN_F32_ENABLE")
  remote_candidate_env+=" DS4_ROCM_GLM5_ENABLE_SHARED_DOWN_F32='$SHARED_DOWN_F32_ENABLE'"
fi
if [[ -n $STRIDED_F32_ROWS_DISABLE ]]; then
  local_candidate_env+=(DS4_ROCM_GLM5_DISABLE_STRIDED_F32_ROWS="$STRIDED_F32_ROWS_DISABLE")
  remote_candidate_env+=" DS4_ROCM_GLM5_DISABLE_STRIDED_F32_ROWS='$STRIDED_F32_ROWS_DISABLE'"
fi
if [[ -n $ZERO_WORKSPACE ]]; then
  local_candidate_env+=(DS4_GLM5_ZERO_WORKSPACE="$ZERO_WORKSPACE")
  remote_candidate_env+=" DS4_GLM5_ZERO_WORKSPACE='$ZERO_WORKSPACE'"
fi
if [[ -n $ZERO_MOE_EACH_LAYER ]]; then
  local_candidate_env+=(DS4_GLM5_ZERO_MOE_EACH_LAYER="$ZERO_MOE_EACH_LAYER")
  remote_candidate_env+=" DS4_GLM5_ZERO_MOE_EACH_LAYER='$ZERO_MOE_EACH_LAYER'"
fi
if [[ -n $ZERO_PERSISTENT_MLA ]]; then
  local_candidate_env+=(DS4_GLM5_ZERO_PERSISTENT_MLA="$ZERO_PERSISTENT_MLA")
  remote_candidate_env+=" DS4_GLM5_ZERO_PERSISTENT_MLA='$ZERO_PERSISTENT_MLA'"
fi
if [[ -n $DISABLE_Q4_BATCH_MLA_OUTPUT ]]; then
  local_candidate_env+=(DS4_GLM5_DISABLE_Q4_BATCH_MLA_OUTPUT="$DISABLE_Q4_BATCH_MLA_OUTPUT")
  remote_candidate_env+=" DS4_GLM5_DISABLE_Q4_BATCH_MLA_OUTPUT='$DISABLE_Q4_BATCH_MLA_OUTPUT'"
fi
if [[ -n $CAUSAL_ATTN_GEMM_NOPE ]]; then
  local_candidate_env+=(DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE="$CAUSAL_ATTN_GEMM_NOPE")
  remote_candidate_env+=" DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE='$CAUSAL_ATTN_GEMM_NOPE'"
fi
if [[ -n $CAUSAL_ATTN_GEMM_NOPE_F32 ]]; then
  local_candidate_env+=(DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_F32="$CAUSAL_ATTN_GEMM_NOPE_F32")
  remote_candidate_env+=" DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_F32='$CAUSAL_ATTN_GEMM_NOPE_F32'"
fi
if [[ -n $MLA_ATTENTION_MODE ]]; then
  local_candidate_env+=(DS4_GLM5_MLA_ATTENTION_MODE="$MLA_ATTENTION_MODE")
  remote_candidate_env+=" DS4_GLM5_MLA_ATTENTION_MODE='$MLA_ATTENTION_MODE'"
fi
if [[ -n $MLA_OUTPUT_DUMP ]]; then
  local_candidate_env+=(DS4_GLM5_MLA_OUTPUT_DUMP="$MLA_OUTPUT_DUMP")
  remote_candidate_env+=" DS4_GLM5_MLA_OUTPUT_DUMP='$MLA_OUTPUT_DUMP'"
fi
if [[ -n $BATCH_LAYER_TRACE ]]; then
  local_candidate_env+=(DS4_GLM5_BATCH_LAYER_TRACE="$BATCH_LAYER_TRACE")
  remote_candidate_env+=" DS4_GLM5_BATCH_LAYER_TRACE='$BATCH_LAYER_TRACE'"
fi
if [[ -n $TRACE_PREFIX ]]; then
  local_candidate_env+=(DS4_GLM5_NEXT_TRACE_PREFIX="$OUT/trace")
  remote_candidate_env+=" DS4_GLM5_NEXT_TRACE_PREFIX='$PEER_RESEARCH_ROOT/glm5-next-tp2/$TAG/trace'"
  local_candidate_env+=(DS4_GLM5_NEXT_TRACE_LAYER="$TRACE_LAYER" DS4_GLM5_NEXT_TRACE_TOKEN="$TRACE_TOKEN")
  remote_candidate_env+=" DS4_GLM5_NEXT_TRACE_LAYER='$TRACE_LAYER' DS4_GLM5_NEXT_TRACE_TOKEN='$TRACE_TOKEN'"
  if [[ -n $TRACE_FFN_SAME_INPUT ]]; then
    local_candidate_env+=(DS4_GLM5_TRACE_FFN_SAME_INPUT="$TRACE_FFN_SAME_INPUT")
    remote_candidate_env+=" DS4_GLM5_TRACE_FFN_SAME_INPUT='$TRACE_FFN_SAME_INPUT'"
  fi
  if [[ -n $TRACE_POST_ONLY ]]; then
    local_candidate_env+=(DS4_GLM5_TRACE_POST_ONLY=1)
    remote_candidate_env+=' DS4_GLM5_TRACE_POST_ONLY=1'
  fi
fi
if [[ -n $ROUTE_HASH_TRACE ]]; then
  local_candidate_env+=(DS4_GLM5_ROUTE_HASH_TRACE=1)
  remote_candidate_env+=' DS4_GLM5_ROUTE_HASH_TRACE=1'
fi
if [[ -n $HC_HASH_TRACE ]]; then
  local_candidate_env+=(DS4_GLM5_HC_HASH_TRACE=1)
  remote_candidate_env+=' DS4_GLM5_HC_HASH_TRACE=1'
fi
if [[ -n $DEVICE_LAYER_CAPTURE ]]; then
  local_candidate_env+=(DS4_GLM5_DEVICE_LAYER_CAPTURE="$DEVICE_LAYER_CAPTURE")
  remote_candidate_env+=" DS4_GLM5_DEVICE_LAYER_CAPTURE='$DEVICE_LAYER_CAPTURE'"
fi
if [[ -n $DEVICE_MLA_STAGE_CAPTURE ]]; then
  local_candidate_env+=(
    DS4_GLM5_DEVICE_MLA_STAGE_CAPTURE="$DEVICE_MLA_STAGE_CAPTURE")
  remote_candidate_env+=" DS4_GLM5_DEVICE_MLA_STAGE_CAPTURE='$DEVICE_MLA_STAGE_CAPTURE'"
fi
if [[ -n $LAYER_COMPLETION_DIAGNOSTIC ]]; then
  local_candidate_env+=(
    DS4_GLM5_LAYER_COMPLETION_DIAGNOSTIC="$LAYER_COMPLETION_DIAGNOSTIC")
  remote_candidate_env+=" DS4_GLM5_LAYER_COMPLETION_DIAGNOSTIC='$LAYER_COMPLETION_DIAGNOSTIC'"
fi
if [[ -n $RDMA_HOST_COHERENT ]]; then
  local_candidate_env+=(DS4_ROCM_RDMA_HOST_COHERENT="$RDMA_HOST_COHERENT")
  remote_candidate_env+=" DS4_ROCM_RDMA_HOST_COHERENT='$RDMA_HOST_COHERENT'"
fi
if [[ -n $RDMA_CACHE_FENCE ]]; then
  local_candidate_env+=(DS4_ROCM_RDMA_CACHE_FENCE="$RDMA_CACHE_FENCE")
  remote_candidate_env+=" DS4_ROCM_RDMA_CACHE_FENCE='$RDMA_CACHE_FENCE'"
fi
if [[ -n $BF16_FULL_SPLIT_ORDER ]]; then
  local_candidate_env+=(DS4_ROCM_BF16_FULL_SPLIT_ORDER="$BF16_FULL_SPLIT_ORDER")
  remote_candidate_env+=" DS4_ROCM_BF16_FULL_SPLIT_ORDER='$BF16_FULL_SPLIT_ORDER'"
fi
if [[ -n $WARM_RESIDENT ]]; then
  local_candidate_env+=(DS4_GLM5_NEXT_WARM_RESIDENT="$WARM_RESIDENT")
  remote_candidate_env+=" DS4_GLM5_NEXT_WARM_RESIDENT='$WARM_RESIDENT'"
fi
# Persist the exact runtime candidate/diagnostic selectors after assembling
# them.  These switches intentionally do not change the binary hash, so a
# run archive that omits them cannot distinguish two otherwise matched arms.
# Keep the original array order: it is the same order passed to env(1), and
# duplicate keys (if ever introduced) remain auditable with last-one-wins
# semantics rather than being hidden by sorting or deduplication.
for candidate_entry in "${local_candidate_env[@]}"; do
  printf 'candidate_env=%s\n' "$candidate_entry" >>"$OUT/run.env"
done
if [[ -n $TEXT_PROMPT ]]; then
  text_env+=(DS4_GLM5_TEXT_PROMPT="$TEXT_PROMPT")
  text_env+=(DS4_GLM5_TEXT_GENERATE="$TEXT_GENERATE")
  text_env+=(DS4_GLM5_PERF_MODE="$PERF_MODE")
  remote_text_env=" DS4_GLM5_TEXT_PROMPT='$TEXT_PROMPT' DS4_GLM5_TEXT_GENERATE='$TEXT_GENERATE' DS4_GLM5_PERF_MODE='$PERF_MODE'"
  if [[ -n $TEXT_TEACHER_IDS ]]; then
    text_env+=(DS4_GLM5_TEXT_TEACHER_IDS="$TEXT_TEACHER_IDS")
    remote_text_env+=" DS4_GLM5_TEXT_TEACHER_IDS='$TEXT_TEACHER_IDS'"
  fi
  if [[ -n $LOGIT_DUMP_STEP ]]; then
    printf -v logit_dump_name 'decode_%06d.logits.f32' "$LOGIT_DUMP_STEP"
    local_logit_dump="$OUT/$logit_dump_name"
    peer_logit_dump="$PEER_RESEARCH_ROOT/glm5-next-tp2/$TAG/$logit_dump_name"
    text_env+=(DS4_GLM5_TEXT_LOGIT_DUMP_STEP="$LOGIT_DUMP_STEP")
    text_env+=(DS4_GLM5_TEXT_LOGIT_DUMP_PATH="$local_logit_dump")
    remote_text_env+=" DS4_GLM5_TEXT_LOGIT_DUMP_STEP='$LOGIT_DUMP_STEP' DS4_GLM5_TEXT_LOGIT_DUMP_PATH='$peer_logit_dump'"
  fi
  if [[ $LOGIT_DUMP_ALL == 1 ]]; then
    local_logit_dir="$OUT/logits"
    peer_logit_dir="$PEER_RESEARCH_ROOT/glm5-next-tp2/$TAG/logits"
    mkdir "$local_logit_dir"
    ssh -o BatchMode=yes "$PEER" "mkdir -p -- '$peer_logit_dir'"
    text_env+=(DS4_GLM5_TEXT_LOGIT_DUMP_ALL=1)
    text_env+=(DS4_GLM5_TEXT_LOGIT_DUMP_DIR="$local_logit_dir")
    remote_text_env+=" DS4_GLM5_TEXT_LOGIT_DUMP_ALL=1 DS4_GLM5_TEXT_LOGIT_DUMP_DIR='$peer_logit_dir'"
  fi
fi
local_command=("$BINARY")
if [[ $ROCPROF_RANK == leader ]]; then
  ROCPROF=$REPO/../toolchains/rocm-7.14.0-gfx1151/install/bin/rocprofv3
  [[ -x $ROCPROF ]] || {
    echo "error: missing source-pinned ROCm 7.14 rocprofv3: $ROCPROF" >&2
    exit 1
  }
  mkdir "$OUT/rocprof-leader"
  local_command=(
    "$ROCPROF" --runtime-trace --stats --summary-per-domain
    --summary-units msec --output-format csv
    --output-directory "$OUT/rocprof-leader" -- "$BINARY"
  )
fi
env -i PATH="$common_path" HOME="$LOCAL_HOME" \
  DS4_GLM5_MODEL="$MODEL" \
  DS4_GLM5_TP_ROLE=leader \
  DS4_GLM5_TP_HOST="$HOST" \
  DS4_GLM5_TP_PORT="$PORT" \
  DS4_GLM5_TP_RDMA_DEVICE="$LOCAL_DEVICE" \
  DS4_GLM5_TP_CONNECT_TIMEOUT_SEC="$TIMEOUT" \
  DS4_GLM5_FULL_TRUNK="$FULL_TRUNK" \
  DS4_GLM5_FULL_TOKENS="$FULL_TOKENS" \
  DS4_GLM5_KDA_ROUTED_BATCH_TEST="$KDA_ROUTED_BATCH_TEST" \
  DS4_GLM5_KDA_ATTENTION_ONLY_TEST="$KDA_ATTENTION_ONLY_TEST" \
  DS4_GLM5_KDA_ROUTED_BATCH_ROWS="$KDA_ROUTED_BATCH_ROWS" \
  DS4_GLM5_KDA_ROUTED_PROFILE_REPEATS="$KDA_ROUTED_PROFILE_REPEATS" \
  DS4_GLM5_KDA_ROUTED_CONTINUATION_ROWS="$KDA_ROUTED_CONTINUATION_ROWS" \
  DS4_GLM5_MLA_ROUTED_BATCH_TEST="$MLA_ROUTED_BATCH_TEST" \
  DS4_GLM5_MLA_ATTENTION_ONLY_TEST="$MLA_ATTENTION_ONLY_TEST" \
  DS4_GLM5_MLA_SPARSE_BOUNDARY_TEST="$MLA_SPARSE_BOUNDARY_TEST" \
  DS4_GLM5_MLA_LONGKEY_ATTENTION_TEST="$MLA_LONGKEY_ATTENTION_TEST" \
  DS4_GLM5_MLA_ROUTED_BATCH_ROWS="$MLA_ROUTED_BATCH_ROWS" \
  DS4_GLM5_MLA_ROUTED_PREFIX_ROWS="$MLA_ROUTED_PREFIX_ROWS" \
  DS4_GLM5_MLA_ROUTED_CONTINUATION_ROWS="$MLA_ROUTED_CONTINUATION_ROWS" \
  DS4_GLM5_BATCH_PREFILL_TEST="$BATCH_PREFILL_TEST" \
  DS4_GLM5_BATCH_PREFILL_COMPARE="$BATCH_PREFILL_COMPARE" \
  DS4_GLM5_LAYER_TIMING="$LAYER_TIMING" \
  "${local_rdma_env[@]}" \
  "${local_candidate_env[@]}" \
  "${text_env[@]}" \
  "${local_command[@]}" >"$OUT/leader.log" 2>&1 &
leader_pid=$!
ssh -o BatchMode=yes "$PEER" \
  "env -i PATH='$common_path' HOME='$PEER_HOME' DS4_GLM5_MODEL='$PEER_MODEL' DS4_GLM5_TP_ROLE=worker DS4_GLM5_TP_HOST='$HOST' DS4_GLM5_TP_PORT='$PORT' DS4_GLM5_TP_RDMA_DEVICE='$PEER_DEVICE' DS4_GLM5_TP_CONNECT_TIMEOUT_SEC='$TIMEOUT' DS4_GLM5_FULL_TRUNK='$FULL_TRUNK' DS4_GLM5_FULL_TOKENS='$FULL_TOKENS' DS4_GLM5_KDA_ROUTED_BATCH_TEST='$KDA_ROUTED_BATCH_TEST' DS4_GLM5_KDA_ATTENTION_ONLY_TEST='$KDA_ATTENTION_ONLY_TEST' DS4_GLM5_KDA_ROUTED_BATCH_ROWS='$KDA_ROUTED_BATCH_ROWS' DS4_GLM5_KDA_ROUTED_PROFILE_REPEATS='$KDA_ROUTED_PROFILE_REPEATS' DS4_GLM5_KDA_ROUTED_CONTINUATION_ROWS='$KDA_ROUTED_CONTINUATION_ROWS' DS4_GLM5_MLA_ROUTED_BATCH_TEST='$MLA_ROUTED_BATCH_TEST' DS4_GLM5_MLA_ATTENTION_ONLY_TEST='$MLA_ATTENTION_ONLY_TEST' DS4_GLM5_MLA_SPARSE_BOUNDARY_TEST='$MLA_SPARSE_BOUNDARY_TEST' DS4_GLM5_MLA_LONGKEY_ATTENTION_TEST='$MLA_LONGKEY_ATTENTION_TEST' DS4_GLM5_MLA_ROUTED_BATCH_ROWS='$MLA_ROUTED_BATCH_ROWS' DS4_GLM5_MLA_ROUTED_PREFIX_ROWS='$MLA_ROUTED_PREFIX_ROWS' DS4_GLM5_MLA_ROUTED_CONTINUATION_ROWS='$MLA_ROUTED_CONTINUATION_ROWS' DS4_GLM5_BATCH_PREFILL_TEST='$BATCH_PREFILL_TEST' DS4_GLM5_BATCH_PREFILL_COMPARE='$BATCH_PREFILL_COMPARE' DS4_GLM5_LAYER_TIMING='$LAYER_TIMING' DS4_ROCM_GLM5_BATCH_KSLICE_OUTPUT='$KDA_BATCH_KSLICE_OUTPUT'$remote_rdma_env$remote_candidate_env$remote_text_env '$PEER_BINARY'" \
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
  if [[ $RDMA_PROFILE == roce-v2 ]]; then
    grep -q "rdma GID index $GID_INDEX (RoCE v2)" "$log"
    grep -q 'mlx5 queue pair uses RC' "$log"
    grep -q 'registered host slab as 3 MRs' "$log"
    ! grep -q 'rdma device odl_tb5_' "$log"
  else
    grep -q 'rdma device odl_tb5_0' "$log"
    grep -q 'provider rejected UC, using RC' "$log"
    if [[ $ODL_WC_STREAM_COPY == 0 ]]; then
      ! grep -q '"enabled":true' "$log"
      ! grep -Eq '"stream_calls":[1-9][0-9]*' "$log"
    else
      grep -q '"enabled":true' "$log"
      grep -q '"fallback_calls":0' "$log"
    fi
    ! grep -q 'rdma device mlx5_' "$log"
  fi
  if [[ $KDA_ATTENTION_ONLY_TEST == 1 ]]; then
    grep -q 'PASS GLM5 KDA-attention batch role=' "$log"
  elif [[ $KDA_ROUTED_BATCH_TEST == 1 ]]; then
    grep -q 'PASS GLM5 KDA+routed batch role=' "$log"
  elif [[ $MLA_ATTENTION_ONLY_TEST == 1 ]]; then
    grep -q 'PASS GLM5 MLA-attention batch role=' "$log"
  elif [[ $MLA_SPARSE_BOUNDARY_TEST == 1 ]]; then
    grep -q 'PASS GLM5 sparse-MLA crossover role=' "$log"
  elif [[ $MLA_LONGKEY_ATTENTION_TEST == 1 ]]; then
    grep -q 'PASS GLM5 MLA long-key attention role=' "$log"
  elif [[ $MLA_ROUTED_BATCH_TEST == 1 ]]; then
    grep -q 'PASS GLM5 MLA+routed batch role=' "$log"
  elif [[ -n $TEXT_PROMPT ]]; then
    grep -q 'GLM5 text prompt role=' "$log"
    grep -q 'PASS GLM5 real-text role=' "$log"
    if [[ $BATCH_PREFILL_TEST == 1 ]]; then
      grep -q 'batch_prefill=1' "$log"
    fi
    if [[ $BATCH_PREFILL_COMPARE == 1 ]]; then
      grep -q 'GLM5 complete batch prompt comparison role=' "$log"
      grep -q 'PASS GLM5 complete batch state comparison role=' "$log"
    fi
    if [[ $PERF_MODE == 1 ]]; then
      grep -q 'GLM5 staged timing role=' "$log"
      grep -q 'full_logit_validation=0' "$log"
    fi
    if [[ -n $TEXT_TEACHER_IDS ]]; then
      grep -q 'PASS GLM5 teacher-forced role=' "$log"
    fi
  else
    grep -q 'PASS GLM5 prefix->layer3' "$log"
  fi
  if [[ $FULL_TRUNK == 1 ]]; then
    if grep -q 'GLM5 mixed-Q2 typed residency oracle' "$log"; then
      grep -q 'packed_q4_bytes=0' "$log"
      ! grep -Eq 'packed_q4_bytes=[1-9][0-9]*' "$log"
    else
      grep -q 'packed_q4_bytes=85614133248 rdma=1' "$log"
    fi
    grep -q 'GLM5 full-trunk post-install' "$log"
  else
    grep -q 'window_cache_bytes=0 rdma=1' "$log"
  fi
done
if [[ -n $LOGIT_DUMP_STEP ]]; then
  expected_logit_bytes=$((154880 * 4))
  [[ $(stat -c %s "$local_logit_dump") == "$expected_logit_bytes" ]] || {
    echo "error: local bounded logit dump has the wrong size" >&2
    exit 1
  }
  peer_logit_size=$(ssh -o BatchMode=yes "$PEER" "stat -c %s '$peer_logit_dump'")
  [[ $peer_logit_size == "$expected_logit_bytes" ]] || {
    echo "error: peer bounded logit dump has the wrong size" >&2
    exit 1
  }
  local_logit_sha=$(sha256sum "$local_logit_dump" | awk '{print $1}')
  peer_logit_sha=$(ssh -o BatchMode=yes "$PEER" \
    "sha256sum '$peer_logit_dump'" | awk '{print $1}')
  [[ $local_logit_sha == "$peer_logit_sha" ]] || {
    echo "error: all-rank bounded logit dumps differ" >&2
    exit 1
  }
  printf 'text_logit_dump_sha256=%s\n' "$local_logit_sha" >>"$OUT/run.env"
fi
if [[ $LOGIT_DUMP_ALL == 1 ]]; then
  expected_logit_bytes=$((154880 * 4))
  local_logit_count=$(find "$local_logit_dir" -maxdepth 1 -type f \
    -name 'decode_*.logits.f32' -size "${expected_logit_bytes}c" | wc -l)
  peer_logit_count=$(ssh -o BatchMode=yes "$PEER" \
    "find '$peer_logit_dir' -maxdepth 1 -type f -name 'decode_*.logits.f32' -size '${expected_logit_bytes}c' | wc -l")
  [[ $local_logit_count == "$TEXT_GENERATE" &&
     $peer_logit_count == "$TEXT_GENERATE" ]] || {
    echo "error: all-position logit dump count or size mismatch" >&2
    exit 1
  }
  (cd "$local_logit_dir" && sha256sum decode_*.logits.f32) \
    >"$OUT/logits.sha256"
  ssh -o BatchMode=yes "$PEER" \
    "cd '$peer_logit_dir' && sha256sum decode_*.logits.f32" \
    >"$OUT/logits.peer.sha256"
  cmp "$OUT/logits.sha256" "$OUT/logits.peer.sha256" || {
    echo "error: all-rank all-position logit manifests differ" >&2
    exit 1
  }
  printf 'text_logit_manifest_sha256=%s\n' \
    "$(sha256sum "$OUT/logits.sha256" | awk '{print $1}')" \
    >>"$OUT/run.env"
fi
if [[ $KDA_ATTENTION_ONLY_TEST == 1 ]]; then
  LEADER_OUTPUT=$(sed -n '/PASS GLM5 KDA-attention batch role=/s/.* output=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/leader.log" | tail -1)
  WORKER_OUTPUT=$(sed -n '/PASS GLM5 KDA-attention batch role=/s/.* output=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/worker.log" | tail -1)
  [[ -n $LEADER_OUTPUT && $LEADER_OUTPUT == "$WORKER_OUTPUT" ]] || {
    echo "error: all-rank KDA-attention hashes differ" >&2
    exit 1
  }
elif [[ $KDA_ROUTED_BATCH_TEST == 1 ]]; then
  LEADER_OUTPUT=$(sed -n '/PASS GLM5 KDA+routed batch role=/s/.* output=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/leader.log" | tail -1)
  WORKER_OUTPUT=$(sed -n '/PASS GLM5 KDA+routed batch role=/s/.* output=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/worker.log" | tail -1)
  LEADER_CONT=$(sed -n '/PASS GLM5 KDA+routed batch role=/s/.* continuation=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/leader.log" | tail -1)
  WORKER_CONT=$(sed -n '/PASS GLM5 KDA+routed batch role=/s/.* continuation=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/worker.log" | tail -1)
  [[ -n $LEADER_OUTPUT && $LEADER_OUTPUT == "$WORKER_OUTPUT" &&
     -n $LEADER_CONT && $LEADER_CONT == "$WORKER_CONT" ]] || {
    echo "error: all-rank KDA+routed batch or continuation hashes differ" >&2
    exit 1
  }
elif [[ $MLA_ATTENTION_ONLY_TEST == 1 ]]; then
  LEADER_OUTPUT=$(sed -n '/PASS GLM5 MLA-attention batch role=/s/.* output=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/leader.log" | tail -1)
  WORKER_OUTPUT=$(sed -n '/PASS GLM5 MLA-attention batch role=/s/.* output=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/worker.log" | tail -1)
  [[ -n $LEADER_OUTPUT && $LEADER_OUTPUT == "$WORKER_OUTPUT" ]] || {
    echo "error: all-rank MLA-attention hashes differ" >&2
    exit 1
  }
elif [[ $MLA_SPARSE_BOUNDARY_TEST == 1 ]]; then
  LEADER_OUTPUT=$(sed -n '/PASS GLM5 sparse-MLA crossover role=/s/.* sparse=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/leader.log" | tail -1)
  WORKER_OUTPUT=$(sed -n '/PASS GLM5 sparse-MLA crossover role=/s/.* sparse=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/worker.log" | tail -1)
  [[ -n $LEADER_OUTPUT && $LEADER_OUTPUT == "$WORKER_OUTPUT" ]] || {
    echo "error: all-rank sparse-MLA crossover hashes differ" >&2
    exit 1
  }
elif [[ $MLA_LONGKEY_ATTENTION_TEST == 1 ]]; then
  LEADER_OUTPUT=$(sed -n '/GLM5 MLA long-key attention role=/s/.* gemm=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/leader.log" | tail -1)
  WORKER_OUTPUT=$(sed -n '/GLM5 MLA long-key attention role=/s/.* gemm=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/worker.log" | tail -1)
  [[ -n $LEADER_OUTPUT && $LEADER_OUTPUT == "$WORKER_OUTPUT" ]] || {
    echo "error: all-rank MLA long-key GEMM hashes differ" >&2
    exit 1
  }
elif [[ $MLA_ROUTED_BATCH_TEST == 1 ]]; then
  LEADER_OUTPUT=$(sed -n '/PASS GLM5 MLA+routed batch role=/s/.* output=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/leader.log" | tail -1)
  WORKER_OUTPUT=$(sed -n '/PASS GLM5 MLA+routed batch role=/s/.* output=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/worker.log" | tail -1)
  LEADER_CONT=$(sed -n '/PASS GLM5 MLA+routed batch role=/s/.* continuation=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/leader.log" | tail -1)
  WORKER_CONT=$(sed -n '/PASS GLM5 MLA+routed batch role=/s/.* continuation=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/worker.log" | tail -1)
  [[ -n $LEADER_OUTPUT && $LEADER_OUTPUT == "$WORKER_OUTPUT" &&
     -n $LEADER_CONT && $LEADER_CONT == "$WORKER_CONT" ]] || {
    echo "error: all-rank MLA+routed batch or continuation hashes differ" >&2
    exit 1
  }
elif [[ -n $TEXT_PROMPT ]]; then
  LEADER_OUTPUT=$(sed -n 's/.* generated_fnv=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/leader.log" | tail -1)
  WORKER_OUTPUT=$(sed -n 's/.* generated_fnv=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/worker.log" | tail -1)
  LEADER_IDS=$(sed -n '/GLM5 text prompt /s/.* token_fnv=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/leader.log" | tail -1)
  WORKER_IDS=$(sed -n '/GLM5 text prompt /s/.* token_fnv=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/worker.log" | tail -1)
  [[ -n $LEADER_OUTPUT && $LEADER_OUTPUT == "$WORKER_OUTPUT" &&
     -n $LEADER_IDS && $LEADER_IDS == "$WORKER_IDS" ]] || {
    echo "error: all-rank prompt/generated hashes differ" >&2
    exit 1
  }
  [[ -z $EXPECTED_GENERATED_FNV ||
     $LEADER_OUTPUT == "$EXPECTED_GENERATED_FNV" ]] || {
    echo "error: generated fingerprint does not match the required reference" >&2
    exit 1
  }
else
  if [[ $FULL_TRUNK == 1 ]]; then
    LEADER_OUTPUT=$(sed -n 's/.* trunk_output=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/leader.log" | tail -1)
    WORKER_OUTPUT=$(sed -n 's/.* trunk_output=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/worker.log" | tail -1)
  else
    LEADER_OUTPUT=$(sed -n 's/.* token0 role=.* output=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/leader.log" | tail -1)
    WORKER_OUTPUT=$(sed -n 's/.* token0 role=.* output=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/worker.log" | tail -1)
  fi
  [[ -n $LEADER_OUTPUT && $LEADER_OUTPUT == "$WORKER_OUTPUT" ]] || {
    echo "error: all-rank layer3 output hashes differ" >&2
    exit 1
  }
fi
if [[ $FULL_TRUNK == 1 && -z $TEXT_PROMPT &&
      $KDA_ROUTED_BATCH_TEST == 0 && $MLA_ROUTED_BATCH_TEST == 0 ]]; then
  LEADER_TRUNK=$(sed -n 's/.* trunk_output=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/leader.log" | tail -1)
  WORKER_TRUNK=$(sed -n 's/.* trunk_output=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/worker.log" | tail -1)
  LEADER_LOGITS=$(sed -n 's/.* logits=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/leader.log" | tail -1)
  WORKER_LOGITS=$(sed -n 's/.* logits=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/worker.log" | tail -1)
  [[ -n $LEADER_TRUNK && $LEADER_TRUNK == "$WORKER_TRUNK" &&
     -n $LEADER_LOGITS && $LEADER_LOGITS == "$WORKER_LOGITS" ]] || {
    echo "error: all-rank full-trunk or logit hashes differ" >&2
    exit 1
  }
  if [[ $FULL_TOKENS == 2 ]]; then
    LEADER_TRUNK2=$(sed -n 's/.* greedy2_trunk=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/leader.log" | tail -1)
    WORKER_TRUNK2=$(sed -n 's/.* greedy2_trunk=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/worker.log" | tail -1)
    LEADER_LOGITS2=$(sed -n 's/.* greedy2_logits=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/leader.log" | tail -1)
    WORKER_LOGITS2=$(sed -n 's/.* greedy2_logits=\([0-9a-f]\{16\}\).*/\1/p' "$OUT/worker.log" | tail -1)
    [[ -n $LEADER_TRUNK2 && $LEADER_TRUNK2 == "$WORKER_TRUNK2" &&
       -n $LEADER_LOGITS2 && $LEADER_LOGITS2 == "$WORKER_LOGITS2" ]] || {
      echo "error: all-rank greedy token-2 trunk or logit hashes differ" >&2
      exit 1
    }
  fi
fi

printf 'PASS GLM5 prefix-layer3 %s tag=%s binary_sha256=%s output_fnv=%s\n' \
  "$RDMA_PROFILE" "$TAG" "$LOCAL_SHA" "$LEADER_OUTPUT"
printf '%s\n' \
  "leader_rc=$leader_rc" \
  "worker_rc=$worker_rc" \
  "transport=$RDMA_PROFILE" \
  "output_fnv=$LEADER_OUTPUT" >"$OUT/run.status"
(cd "$OUT" &&
  find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%P\0' |
    sort -z | xargs -0 sha256sum) >"$OUT/SHA256SUMS"
