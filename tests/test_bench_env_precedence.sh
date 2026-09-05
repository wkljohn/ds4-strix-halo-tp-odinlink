#!/bin/bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
test_dir=$(mktemp -d)
trap 'rm -rf -- "$test_dir"' EXIT
config=$test_dir/bench.env
model=$test_dir/model.gguf
glm_model=$test_dir/glm-model.gguf

# Config-only validation still inspects model metadata so provider tests cannot
# bypass architecture-specific safety checks. Generate the smallest valid GGUF
# header with a non-GLM architecture instead of relying on /dev/null.
python3 - "$model" "$glm_model" <<'PY'
import struct
import sys

key = b"general.architecture"
for path, value in ((sys.argv[1], b"deepseek4"),
                    (sys.argv[2], b"glm5-next")):
    with open(path, "wb") as out:
        out.write(b"GGUF")
        out.write(struct.pack("<IQQ", 3, 0, 1))
        out.write(struct.pack("<Q", len(key)))
        out.write(key)
        out.write(struct.pack("<I", 8))
        out.write(struct.pack("<Q", len(value)))
        out.write(value)
PY

printf '%s\n' \
  'DS4_BENCH_RDMA_PROFILE=roce-v2' \
  'DS4_COORDINATOR_ADDR=192.168.99.1' \
  > "$config"

# Minimal GGUF whose only metadata is general.architecture; the harness
# inspects the model architecture before its validate-config exit, so
# /dev/null is unusable as the model argument.
model=$test_dir/model.gguf
python3 - "$model" <<'PY'
import struct, sys
def s(x):
    b = x.encode()
    return struct.pack('<Q', len(b)) + b
with open(sys.argv[1], 'wb') as f:
    f.write(b'GGUF')
    f.write(struct.pack('<I', 3))
    f.write(struct.pack('<Q', 0))
    f.write(struct.pack('<Q', 1))
    f.write(s('general.architecture'))
    f.write(struct.pack('<I', 8))
    f.write(s('deepseek4'))
PY

run_validate() {
  env -i PATH="$PATH" LANG=C.UTF-8 \
    DS4_BENCH_CONFIG="$config" \
    DS4_BENCH_VALIDATE_CONFIG_ONLY=1 \
    DS4_BENCH_DSPARK=1 \
    DS4_BENCH_PROMPT_FILE=/dev/null \
    "$@" \
    "$repo/run-tp-ds4-bench.sh" env-precedence "$model"
}

expect_contains() {
  local name=$1 needle=$2
  shift 2
  local output
  output=$(run_validate "$@")
  if [[ $output != *"$needle"* ]]; then
    printf 'FAIL %s: missing %q in %q\n' "$name" "$needle" "$output" >&2
    return 1
  fi
  printf 'PASS %s\n' "$name"
}

expect_contains config-default \
  'rdma_profile=roce-v2 coordinator_addr=192.168.99.1' \
  DS4_BENCH_PREFILL_CHUNK=2048

expect_contains invocation-provider-and-address-win \
  'rdma_profile=odinlink coordinator_addr=10.4.0.1 coordinator_rdma_device=odl_tb5_0 worker_rdma_device=odl_tb5_0 rdma_gid_index=n/a prefill_chunk=2048' \
  DS4_BENCH_RDMA_PROFILE=odinlink \
  DS4_COORDINATOR_ADDR=10.4.0.1

expect_contains odinlink-attention-direct-default \
  'tp_prefill_split_min_attn=2048' \
  DS4_BENCH_RDMA_PROFILE=odinlink \
  DS4_COORDINATOR_ADDR=10.4.0.1

expect_contains roce-attention-split-default \
  'tp_prefill_split_min_attn=2048' \
  DS4_BENCH_RDMA_PROFILE=roce-v2

if run_validate DS4_BENCH_RDMA_PROFILE=odinlink \
    >"$test_dir/mixed-profile.out" 2>&1; then
  echo 'FAIL changed-provider-requires-address: unexpectedly succeeded' >&2
  exit 1
fi
grep -q 'set DS4_COORDINATOR_ADDR' "$test_dir/mixed-profile.out"
echo 'PASS changed-provider-requires-address'

expect_contains roce-safe-prefill-default \
  'rdma_gid_index=3 prefill_chunk=2048' \
  DS4_BENCH_RDMA_PROFILE=roce-v2

expect_contains explicit-prefill-wins \
  'prefill_chunk=1024' \
  DS4_BENCH_PREFILL_CHUNK=1024

if run_validate DS4_BENCH_PREFILL_CHUNK= >"$test_dir/empty.out" 2>&1; then
  echo 'FAIL empty-prefill-rejected: unexpectedly succeeded' >&2
  exit 1
fi
grep -q 'DS4_BENCH_PREFILL_CHUNK cannot be empty' "$test_dir/empty.out"
echo 'PASS empty-prefill-rejected'

if run_validate DS4_BENCH_RDMA_PROFILE=bogus >"$test_dir/bogus.out" 2>&1; then
  echo 'FAIL invalid-provider-still-rejected: unexpectedly succeeded' >&2
  exit 1
fi
grep -q 'must be odinlink, roce-v2, or ib-mlx4' "$test_dir/bogus.out"
echo 'PASS invalid-provider-still-rejected'

expect_contains ib-mlx4-profile-defaults \
  'rdma_profile=ib-mlx4 coordinator_addr=192.168.100.1 coordinator_rdma_device=ibp195s0 worker_rdma_device=ibp195s0 rdma_gid_index=0 prefill_chunk=2048' \
  DS4_BENCH_RDMA_PROFILE=ib-mlx4 \
  DS4_COORDINATOR_ADDR=192.168.100.1

if env -i PATH="$PATH" LANG=C.UTF-8 \
    DS4_BENCH_CONFIG="$config" \
    DS4_BENCH_VALIDATE_CONFIG_ONLY=1 \
    DS4_BENCH_DSPARK=1 \
    DS4_BENCH_PROMPT_FILE=/dev/null \
    DS4_TP_EXPERT_SPLIT=118 \
    "$repo/run-tp-ds4-bench.sh" env-precedence "$model" \
    >"$test_dir/split.out" 2>&1; then
  echo 'FAIL ambient-expert-split-still-rejected: unexpectedly succeeded' >&2
  exit 1
fi
grep -q 'ambient DS4_TP_EXPERT_SPLIT is not accepted' "$test_dir/split.out"
echo 'PASS ambient-expert-split-still-rejected'

if env -i PATH="$PATH" LANG=C.UTF-8 \
    DS4_BENCH_CONFIG="$config" \
    DS4_BENCH_VALIDATE_CONFIG_ONLY=1 \
    DS4_BENCH_PROMPT_FILE=/dev/null \
    "$repo/run-tp-ds4-bench.sh" sparse-output-parent "$glm_model" \
    DS4_GLM5_NEXT_PREFILL_BATCH=256 \
    DS4_GLM5_SPARSE_BATCH_OUTPUT=1 \
    >"$test_dir/sparse-output-parent.out" 2>&1; then
  echo 'FAIL sparse-output-requires-bridge: unexpectedly succeeded' >&2
  exit 1
fi
grep -q 'DS4_GLM5_SPARSE_BATCH_OUTPUT=1 requires DS4_GLM5_SPARSE_BATCH_BRIDGE=1' \
  "$test_dir/sparse-output-parent.out"
echo 'PASS sparse-output-requires-bridge'

if env -i PATH="$PATH" LANG=C.UTF-8 \
    DS4_BENCH_CONFIG="$config" \
    DS4_BENCH_VALIDATE_CONFIG_ONLY=1 \
    DS4_BENCH_PROMPT_FILE=/dev/null \
    "$repo/run-tp-ds4-bench.sh" sparse-output-valid "$glm_model" \
    DS4_GLM5_NEXT_PREFILL_BATCH=256 \
    DS4_GLM5_SPARSE_BATCH_BRIDGE=1 \
    DS4_GLM5_SPARSE_BATCH_OUTPUT=1 \
    >"$test_dir/sparse-output-valid.out" 2>&1; then
  echo 'FAIL sparse-output-with-bridge: incomplete fake model unexpectedly succeeded' >&2
  exit 1
fi
! grep -q 'requires DS4_GLM5_SPARSE_BATCH_BRIDGE=1' \
  "$test_dir/sparse-output-valid.out"
grep -q 'unsupported routed-expert quantization' \
  "$test_dir/sparse-output-valid.out"
echo 'PASS sparse-output-with-bridge'

if env -i PATH="$PATH" LANG=C.UTF-8 \
    DS4_BENCH_CONFIG="$config" \
    DS4_BENCH_VALIDATE_CONFIG_ONLY=1 \
    DS4_BENCH_PROMPT_FILE=/dev/null \
    "$repo/run-tp-ds4-bench.sh" sparse-prelude-parent "$glm_model" \
    DS4_GLM5_NEXT_PREFILL_BATCH=256 \
    DS4_GLM5_SPARSE_BATCH_BRIDGE=1 \
    DS4_GLM5_SPARSE_BATCH_PRELUDE=1 \
    >"$test_dir/sparse-prelude-parent.out" 2>&1; then
  echo 'FAIL sparse-prelude-requires-output: unexpectedly succeeded' >&2
  exit 1
fi
grep -q 'DS4_GLM5_SPARSE_BATCH_PRELUDE=1 requires DS4_GLM5_SPARSE_BATCH_OUTPUT=1' \
  "$test_dir/sparse-prelude-parent.out"
echo 'PASS sparse-prelude-requires-output'

if env -i PATH="$PATH" LANG=C.UTF-8 \
    DS4_BENCH_CONFIG="$config" \
    DS4_BENCH_VALIDATE_CONFIG_ONLY=1 \
    DS4_BENCH_PROMPT_FILE=/dev/null \
    "$repo/run-tp-ds4-bench.sh" sparse-attn-parent "$glm_model" \
    DS4_GLM5_NEXT_PREFILL_BATCH=256 \
    DS4_GLM5_SPARSE_BATCH_BRIDGE=1 \
    DS4_GLM5_SPARSE_BATCH_OUTPUT=1 \
    DS4_GLM5_SPARSE_BATCH_VALUE=1 \
    >"$test_dir/sparse-attn-parent.out" 2>&1; then
  echo 'FAIL sparse-attn-requires-prelude: unexpectedly succeeded' >&2
  exit 1
fi
grep -q 'DS4_GLM5_SPARSE_BATCH_VALUE=1 requires DS4_GLM5_SPARSE_BATCH_PRELUDE=1' \
  "$test_dir/sparse-attn-parent.out"
echo 'PASS sparse-attn-requires-prelude'

if env -i PATH="$PATH" LANG=C.UTF-8 \
    DS4_BENCH_CONFIG="$config" \
    DS4_BENCH_VALIDATE_CONFIG_ONLY=1 \
    DS4_BENCH_PROMPT_FILE=/dev/null \
    "$repo/run-tp-ds4-bench.sh" sparse-head-shared-parent "$glm_model" \
    DS4_GLM5_NEXT_PREFILL_BATCH=256 \
    DS4_GLM5_SPARSE_BATCH_BRIDGE=1 \
    DS4_GLM5_SPARSE_BATCH_OUTPUT=1 \
    DS4_GLM5_SPARSE_BATCH_PRELUDE=1 \
    DS4_ROCM_GLM5_SPARSE_ATTN_HEAD_SHARED=1 \
    >"$test_dir/sparse-head-shared-parent.out" 2>&1; then
  echo 'FAIL sparse-head-shared-requires-value: unexpectedly succeeded' >&2
  exit 1
fi
grep -q 'DS4_ROCM_GLM5_SPARSE_ATTN_HEAD_SHARED=1 requires DS4_GLM5_SPARSE_BATCH_VALUE=1' \
  "$test_dir/sparse-head-shared-parent.out"
echo 'PASS sparse-head-shared-requires-value'

if env -i PATH="$PATH" LANG=C.UTF-8 \
    DS4_BENCH_CONFIG="$config" \
    DS4_BENCH_VALIDATE_CONFIG_ONLY=1 \
    DS4_BENCH_PROMPT_FILE=/dev/null \
    "$repo/run-tp-ds4-bench.sh" sparse-f16-parent "$glm_model" \
    DS4_GLM5_NEXT_PREFILL_BATCH=256 \
    DS4_GLM5_SPARSE_BATCH_BRIDGE=1 \
    DS4_GLM5_SPARSE_BATCH_OUTPUT=1 \
    DS4_GLM5_SPARSE_BATCH_PRELUDE=1 \
    DS4_ROCM_GLM5_NOPE_ATTN_SHARED_PV=1 \
    DS4_GLM5_SPARSE_BATCH_VALUE=1 \
    DS4_ROCM_GLM5_SPARSE_ATTN_F16_GEMM=1 \
    >"$test_dir/sparse-f16-parent.out" 2>&1; then
  echo 'FAIL sparse-f16-requires-head-shared: unexpectedly succeeded' >&2
  exit 1
fi
grep -q 'DS4_ROCM_GLM5_SPARSE_ATTN_F16_GEMM=1 requires DS4_ROCM_GLM5_SPARSE_ATTN_HEAD_SHARED=1' \
  "$test_dir/sparse-f16-parent.out"
echo 'PASS sparse-f16-requires-head-shared'

if env -i PATH="$PATH" LANG=C.UTF-8 \
    DS4_BENCH_CONFIG="$config" \
    DS4_BENCH_VALIDATE_CONFIG_ONLY=1 \
    DS4_BENCH_PROMPT_FILE=/dev/null \
    "$repo/run-tp-ds4-bench.sh" sparse-f16-selector-parent "$glm_model" \
    DS4_GLM5_NEXT_PREFILL_BATCH=256 \
    DS4_ROCM_GLM5_SPARSE_ATTN_COMPARE_F16_LAYER=43 \
    >"$test_dir/sparse-f16-selector-parent.out" 2>&1; then
  echo 'FAIL sparse-f16-selector-requires-compare: unexpectedly succeeded' >&2
  exit 1
fi
grep -q 'comparison selectors require DS4_ROCM_GLM5_SPARSE_ATTN_COMPARE_F16=1' \
  "$test_dir/sparse-f16-selector-parent.out"
echo 'PASS sparse-f16-selector-requires-compare'

if env -i PATH="$PATH" LANG=C.UTF-8 \
    DS4_BENCH_CONFIG="$config" \
    DS4_BENCH_VALIDATE_CONFIG_ONLY=1 \
    DS4_BENCH_PROMPT_FILE=/dev/null \
    "$repo/run-tp-ds4-bench.sh" sparse-attn-valid "$glm_model" \
    DS4_GLM5_NEXT_PREFILL_BATCH=256 \
    DS4_GLM5_SPARSE_BATCH_BRIDGE=1 \
    DS4_GLM5_SPARSE_BATCH_OUTPUT=1 \
    DS4_GLM5_SPARSE_BATCH_PRELUDE=1 \
    DS4_ROCM_GLM5_NOPE_ATTN_SHARED_PV=1 \
    DS4_GLM5_SPARSE_BATCH_VALUE=1 \
    DS4_ROCM_GLM5_SPARSE_ATTN_HEAD_SHARED=1 \
    DS4_ROCM_GLM5_SPARSE_ATTN_F16_GEMM=1 \
    >"$test_dir/sparse-attn-valid.out" 2>&1; then
  echo 'FAIL sparse-attn-valid-chain: incomplete fake model unexpectedly succeeded' >&2
  exit 1
fi
! grep -q 'requires DS4_GLM5_SPARSE_BATCH_PRELUDE=1' \
  "$test_dir/sparse-attn-valid.out"
! grep -q 'requires DS4_ROCM_GLM5_SPARSE_ATTN_HEAD_SHARED=1' \
  "$test_dir/sparse-attn-valid.out"
grep -q 'unsupported routed-expert quantization' \
  "$test_dir/sparse-attn-valid.out"
echo 'PASS sparse-attn-valid-chain'
