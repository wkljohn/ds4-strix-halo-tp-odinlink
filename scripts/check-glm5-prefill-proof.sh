#!/bin/bash
# Prove that both TP ranks executed the requested GLM batched-prefill shape.
set -euo pipefail

COORD_LOG=${1:?usage: check-glm5-prefill-proof.sh COORD_LOG WORKER_LOG BATCH FRONTIER [SPARSE_BATCH_BRIDGE] [SPARSE_BATCH_VALUE] [SPARSE_ATTN_HEAD_SHARED] [SPARSE_ATTN_F16_GEMM]}
WORKER_LOG=${2:?missing worker log}
BATCH=${3:?missing requested batch}
FRONTIER=${4:?missing frontier}
SPARSE_BATCH_BRIDGE=${5:-0}
SPARSE_BATCH_VALUE=${6:-0}
SPARSE_ATTN_HEAD_SHARED=${7:-0}
SPARSE_ATTN_F16_GEMM=${8:-0}

[[ $BATCH =~ ^[1-9][0-9]*$ && $BATCH -le 1024 ]] || {
  echo "error: invalid GLM prefill batch: $BATCH" >&2
  exit 2
}
[[ $FRONTIER =~ ^[1-9][0-9]*$ ]] || {
  echo "error: invalid GLM frontier: $FRONTIER" >&2
  exit 2
}
[[ $SPARSE_BATCH_BRIDGE == 0 || $SPARSE_BATCH_BRIDGE == 1 ]] || {
  echo "error: sparse batch bridge proof selector must be 0 or 1" >&2
  exit 2
}
[[ $SPARSE_BATCH_VALUE == 0 || $SPARSE_BATCH_VALUE == 1 ]] || {
  echo "error: sparse batch value proof selector must be 0 or 1" >&2
  exit 2
}
if (( SPARSE_BATCH_VALUE == 1 && SPARSE_BATCH_BRIDGE != 1 )); then
  echo "error: sparse batch value proof requires sparse batch bridge" >&2
  exit 2
fi
[[ $SPARSE_ATTN_HEAD_SHARED == 0 || $SPARSE_ATTN_HEAD_SHARED == 1 ]] || {
  echo "error: sparse attention head-shared proof selector must be 0 or 1" >&2
  exit 2
}
if (( SPARSE_ATTN_HEAD_SHARED == 1 && SPARSE_BATCH_VALUE != 1 )); then
  echo "error: sparse attention head-shared proof requires sparse batch value" >&2
  exit 2
fi
[[ $SPARSE_ATTN_F16_GEMM == 0 || $SPARSE_ATTN_F16_GEMM == 1 ]] || {
  echo "error: sparse attention F16 GEMM proof selector must be 0 or 1" >&2
  exit 2
}
if (( SPARSE_ATTN_F16_GEMM == 1 && SPARSE_ATTN_HEAD_SHARED != 1 )); then
  echo "error: sparse attention F16 GEMM proof requires head-shared attention" >&2
  exit 2
fi

if (( SPARSE_BATCH_BRIDGE == 1 )); then
  planned_rows=$FRONTIER
else
  planned_rows=$(( FRONTIER < 2048 ? FRONTIER : 2048 ))
fi
remaining=$planned_rows
processed=0
batched_tiles=0
batched_rows=0
scalar_rows=0
min_tile=0
max_tile=0
while (( remaining > 0 )); do
  chunk=$(( remaining < BATCH ? remaining : BATCH ))
  # The runtime planner must never let one tile straddle the dense-to-sparse
  # boundary, even when the requested batch does not divide 2048.
  if (( processed < 2048 && chunk > 2048 - processed )); then
    chunk=$((2048 - processed))
  fi
  if (( BATCH < 2 || chunk < 2 )); then
    chunk=1
    scalar_rows=$((scalar_rows + 1))
  else
    batched_tiles=$((batched_tiles + 1))
    batched_rows=$((batched_rows + chunk))
    if (( min_tile == 0 || chunk < min_tile )); then min_tile=$chunk; fi
    if (( chunk > max_tile )); then max_tile=$chunk; fi
  fi
  processed=$((processed + chunk))
  remaining=$((remaining - chunk))
done
if (( FRONTIER > planned_rows )); then
  scalar_rows=$((scalar_rows + FRONTIER - planned_rows))
fi

for spec in "0:$COORD_LOG" "1:$WORKER_LOG"; do
  rank=${spec%%:*}
  log=${spec#*:}
  [[ -r $log ]] || {
    echo "error: missing GLM prefill proof log: $log" >&2
    exit 1
  }
  expected="ds4: GLM5 prefill execution rank=$rank start=0 prompt_tokens=$FRONTIER requested_batch=$BATCH batched_tiles=$batched_tiles batched_rows=$batched_rows scalar_rows=$scalar_rows min_tile=$min_tile max_tile=$max_tile"
  grep -Fqx "$expected" "$log" || {
    echo "error: rank $rank did not prove the expected GLM prefill execution" >&2
    echo "error: expected: $expected" >&2
    grep -F 'ds4: GLM5 prefill execution ' "$log" >&2 || true
    exit 1
  }
  if (( SPARSE_BATCH_VALUE == 1 && FRONTIER > 2048 )); then
    expected_attention=scalar-rows
    (( SPARSE_ATTN_HEAD_SHARED == 1 )) && expected_attention=head-shared-r16
    (( SPARSE_ATTN_F16_GEMM == 1 )) && expected_attention=f16-gemm-r16
    grep -Eq "^ds4: GLM5 sparse deferred attention and batch value projection engaged rank=$rank layer=[0-9]+ pos0=[0-9]+ rows=[0-9]+ selected_stride=2051 selected_live_min=20(48|49|50|51) selected_live_max=20(48|49|50|51) attention=$expected_attention$" "$log" || {
      echo "error: rank $rank did not prove deferred attention and batch value execution" >&2
      exit 1
    }
    if (( SPARSE_ATTN_F16_GEMM == 1 )); then
      grep -Fq 'GLM5 sparse NoPE F16 GEMM scratch reserved before prefill' "$log" || {
        echo "error: rank $rank did not prove prefill-time F16 GEMM scratch reservation" >&2
        exit 1
      }
    fi
  fi
done

echo "validated_glm5_prefill_execution=batch:$BATCH,frontier:$FRONTIER,bridge:$SPARSE_BATCH_BRIDGE,sparse_value:$SPARSE_BATCH_VALUE,sparse_head_shared:$SPARSE_ATTN_HEAD_SHARED,sparse_f16_gemm:$SPARSE_ATTN_F16_GEMM,tiles:$batched_tiles,batched_rows:$batched_rows,scalar_rows:$scalar_rows"
