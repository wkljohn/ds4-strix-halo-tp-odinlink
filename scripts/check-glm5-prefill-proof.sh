#!/bin/bash
# Prove that both TP ranks executed the requested GLM batched-prefill shape.
set -euo pipefail

COORD_LOG=${1:?usage: check-glm5-prefill-proof.sh COORD_LOG WORKER_LOG BATCH FRONTIER}
WORKER_LOG=${2:?missing worker log}
BATCH=${3:?missing requested batch}
FRONTIER=${4:?missing frontier}

[[ $BATCH =~ ^[1-9][0-9]*$ && $BATCH -le 1024 ]] || {
  echo "error: invalid GLM prefill batch: $BATCH" >&2
  exit 2
}
[[ $FRONTIER =~ ^[1-9][0-9]*$ ]] || {
  echo "error: invalid GLM frontier: $FRONTIER" >&2
  exit 2
}

dense_rows=$(( FRONTIER < 2048 ? FRONTIER : 2048 ))
remaining=$dense_rows
batched_tiles=0
batched_rows=0
scalar_rows=0
min_tile=0
max_tile=0
while (( remaining > 0 )); do
  chunk=$(( remaining < BATCH ? remaining : BATCH ))
  if (( BATCH < 2 || chunk < 2 )); then
    chunk=1
    scalar_rows=$((scalar_rows + 1))
  else
    batched_tiles=$((batched_tiles + 1))
    batched_rows=$((batched_rows + chunk))
    if (( min_tile == 0 || chunk < min_tile )); then min_tile=$chunk; fi
    if (( chunk > max_tile )); then max_tile=$chunk; fi
  fi
  remaining=$((remaining - chunk))
done
if (( FRONTIER > dense_rows )); then
  scalar_rows=$((scalar_rows + FRONTIER - dense_rows))
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
done

echo "validated_glm5_prefill_execution=batch:$BATCH,frontier:$FRONTIER,tiles:$batched_tiles,batched_rows:$batched_rows,scalar_rows:$scalar_rows"
