#!/usr/bin/env bash
# Full build: CPU changes must never inherit stale objects from another commit.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo/scripts/ds4-research-root.sh"
ds4_resolve_research_roots "$repo"
revision=$(git -C "$repo" rev-parse HEAD)
profile=${DS4_RESEARCH_PROFILE:-0}
case $profile in
  0) flavor=full ;;
  1) flavor=profile ;;
  *) printf 'error: DS4_RESEARCH_PROFILE must be 0 or 1\n' >&2; exit 2 ;;
esac
artifact=$DS4_RESEARCH_ROOT/builds/glm53-$flavor-${revision:0:7}
hip=${DS4_ROCM_HOME:-/home/wkljohn/Desktop/cc/toolchains/rocm-10.0.0-gfx1151/install}
python3 "$repo/scripts/check-rocm-toolchain.py" --home "$hip" --hipcc "$hip/bin/hipcc"
git -C "$repo" diff --quiet
git -C "$repo" diff --cached --quiet
[[ ! -e $artifact && -x $hip/bin/hipcc ]]
git -C "$repo" worktree add --detach "$artifact" "$revision"
cd "$artifact"
git rev-parse HEAD > BUILD-SOURCE
"$hip/bin/hipcc" --version > compiler.txt
core='ds4.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o ds4_layer_pack.o ds4_glm5_kda.o ds4_glm5_next_runtime.o ds4_glm5_next_state.o ds4_glm5_next_exec.o'
cflags='-O3 -ffast-math -g -march=native -Wall -Wextra -std=c99 -D_GNU_SOURCE -fno-finite-math-only -DDS4_ROCM_BUILD -DDS4_ROCM_TP_READY=1'
# Command-line CFLAGS override the Makefile's +=, so explicitly mirror PROFILE
# here while the Makefile applies it to the HIP compilation flags.
if [[ $profile == 1 ]]; then cflags+=' -DDS4_ENABLE_PROFILING=1'; fi
args=(-j3 ds4 ds4-bench-tp tests/test_rocm_glm5_indexer_score_one
      gguf-tools/quality-testing/score_official
      tests/test_rocm_glm5_indexer_select
      tests/test_rocm_glm5_expert_pairs tests/test_glm5_expert_pairs
      tests/test_rocm_glm5_layer_verify
      tests/test_rocm_glm5_mla_prelude_small_m
      tests/test_rocm_glm5_small_m
      tests/test_rocm_glm5_six_prefill_exact
      tests/test_rocm_glm5_q4k_batch_partition
      tests/test_tp_native_cycle tests/test_glm5_native_session
      tests/test_glm5_target_commit tests/test_glm5_shared_route_order
      "HIPCC=$hip/bin/hipcc" "ROCM_HOME=$hip" "CORE_OBJS=$core"
      "PROFILE=$profile" "CFLAGS=$cflags"
      "DS4_LINK=$hip/bin/hipcc -O3 -ffast-math -g -fno-finite-math-only -pthread -D__HIP_PLATFORM_AMD__ --offload-arch=gfx1151 -mno-wavefrontsize64 -DDS4_GFX1151_WAVE32=1"
      "DS4_LINK_LIBS=-L$hip/lib -Wl,-rpath,$hip/lib -lm -pthread -lhipblas -lhipblaslt")
printf '%q ' make "${args[@]}" > BUILD-COMMAND
printf '\n' >> BUILD-COMMAND
make "${args[@]}" > build.log 2>&1
sha256sum ds4 ds4-bench-tp *.o tests/test_rocm_glm5_indexer_score_one \
  gguf-tools/quality-testing/score_official{,.o,.c} run-tp-quality-score.sh \
  tests/test_rocm_glm5_indexer_select tests/test_rocm_glm5_expert_pairs \
  tests/test_rocm_glm5_layer_verify \
  tests/test_rocm_glm5_mla_prelude_small_m \
  tests/test_rocm_glm5_small_m \
  tests/test_rocm_glm5_six_prefill_exact \
  tests/test_rocm_glm5_q4k_batch_partition \
  tests/test_glm5_expert_pairs tests/test_tp_native_cycle tests/test_glm5_native_session \
  tests/test_glm5_target_commit tests/test_glm5_shared_route_order tests/*.o rocm/*.cuh \
  run-tp-ds4-bench.sh scripts/tp-worker-supervisor.sh scripts/build-glm53-research.sh \
  scripts/run-glm53-q4-pair-tp.sh \
  > BUILD-SHA256SUMS
printf 'artifact=%s\n' "$artifact"
