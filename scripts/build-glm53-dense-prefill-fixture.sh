#!/usr/bin/env bash
# New test-hook object plus an independent frozen production control.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo/scripts/ds4-research-root.sh"
ds4_resolve_research_roots "$repo"
revision=$(git -C "$repo" rev-parse HEAD)
artifact=$DS4_RESEARCH_ROOT/builds/glm53-dense-prefill-fixture-${revision:0:7}
donor=$DS4_RESEARCH_ROOT/builds/glm53-full-6eba8bd
sdk=${DS4_ROCM_HOME:-/home/wkljohn/Desktop/cc/toolchains/rocm-10.0.0-gfx1151/install}
python3 "$repo/scripts/check-rocm-toolchain.py" --home "$sdk" --hipcc "$sdk/bin/hipcc"
[[ -z $(git -C "$repo" status --porcelain) && ! -e $artifact ]]
[[ $(git -C "$donor" rev-parse HEAD) == 6eba8bdb134b8153cbc1dc1cf4dd5b4de2f52957 ]]
[[ -z $(git -C "$donor" status --porcelain) ]]
(cd "$donor" && sha256sum --check --quiet BUILD-SHA256SUMS)
for header in ds4_gpu.h ds4_gpu_mgpu.h ds4_tp.h tests/glm5_gguf_test.hpp; do
    cmp "$repo/$header" "$donor/$header"
done
mkdir "$artifact"
git -C "$repo" worktree add --detach "$artifact/source" "$revision"
cd "$artifact"
printf 'fixture_and_hook_source=%s\ncontrol_inference_source=%s\ncontrol_artifact=%s\n' \
    "$revision" 6eba8bdb134b8153cbc1dc1cf4dd5b4de2f52957 "$donor" > BUILD-SOURCE
"$sdk/bin/hipcc" --version > compiler.txt
run() {
    printf '%q ' "$@" >> BUILD-COMMAND
    printf '\n' >> BUILD-COMMAND
    "$@" >> build.log 2>&1
}
run make -C source -j2 tests/test_rocm_glm5_dense_q8_prefill \
    "HIPCC=$sdk/bin/hipcc" "ROCM_HOME=$sdk" PROFILE=0
flags=(-O3 -g -pthread -D__HIP_PLATFORM_AMD__ --offload-arch=gfx1151
       -mno-wavefrontsize64 -DDS4_GFX1151_WAVE32=1 -std=c++17
       -isystem "$sdk/include" -fno-fast-math -ffp-contract=off)
run "$sdk/bin/hipcc" "${flags[@]}" -DDENSE_PREFILL_CONTROL_ONLY=1 \
    -I"$artifact/source" -c source/tests/test_rocm_glm5_dense_q8_prefill.cu -o control-fixture.o
objects=("$donor/ds4_rocm.o" "$donor/ds4_rocm_compat.o" "$donor/ds4_rocm_unavailable.o")
run "$sdk/bin/hipcc" "${flags[@]}" -Wl,--gc-sections control-fixture.o \
    "${objects[@]}" -L"$sdk/lib" -Wl,-rpath,"$sdk/lib" \
    -lhipblas -lhipblaslt -o test-dense-control
git -C source ls-files -z -- '*.h' '*.cuh' '*.cu' '*.inc' Makefile \
    scripts/build-glm53-dense-prefill-fixture.sh scripts/check-rocm-toolchain.py \
    scripts/ds4-research-root.sh | (cd source && xargs -0 sha256sum) > SOURCE-SHA256SUMS
sha256sum source/tests/test_rocm_glm5_dense_q8_prefill \
    source/tests/test_rocm_glm5_dense_q8_prefill.o source/ds4_rocm_test_hooks.o \
    source/ds4_rocm_compat.o source/ds4_rocm_unavailable.o test-dense-control \
    control-fixture.o BUILD-SOURCE BUILD-COMMAND compiler.txt SOURCE-SHA256SUMS \
    "$donor/BUILD-SHA256SUMS" "$donor/BUILD-COMMAND" "${objects[@]}" > BUILD-SHA256SUMS
printf 'artifact=%s\n' "$artifact"
