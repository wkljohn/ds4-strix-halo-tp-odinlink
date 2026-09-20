#!/usr/bin/env bash
# Component probe with an independently compiled, immutable parent kernel.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo/scripts/ds4-research-root.sh"
ds4_resolve_research_roots "$repo"
revision=$(git -C "$repo" rev-parse HEAD)
parent=7dacc2318f8e82517210a3ee6a55e83b62d3c00b
artifact=$DS4_RESEARCH_ROOT/builds/glm53-six-native-${revision:0:7}
hip=${DS4_ROCM_HOME:-/home/wkljohn/Desktop/cc/toolchains/rocm-10.0.0-gfx1151/install}
python3 "$repo/scripts/check-rocm-toolchain.py" --home "$hip" --hipcc "$hip/bin/hipcc"
[[ -z $(git -C "$repo" status --porcelain) && ! -e $artifact ]]
mkdir "$artifact"
git -C "$repo" worktree add --detach "$artifact/source" "$revision"
git -C "$repo" worktree add --detach "$artifact/control" "$parent"
cd "$artifact"
printf 'candidate=%s\nparent=%s\n' "$revision" "$parent" > BUILD-SOURCE
"$hip/bin/hipcc" --version > compiler.txt
flags=(-O3 -g -pthread -D__HIP_PLATFORM_AMD__ --offload-arch=gfx1151
       -mno-wavefrontsize64 -DDS4_GFX1151_WAVE32=1 -std=c++17
       -isystem "$hip/include")
run() {
    printf '%q ' "$@" >> BUILD-COMMAND
    printf '\n' >> BUILD-COMMAND
    "$@" >> build.log 2>&1
}
run "$hip/bin/hipcc" "${flags[@]}" -ffast-math -fno-finite-math-only \
    -Dglm5_six_geometry=glm5_six_incumbent \
    -c control/tests/glm5_bf16_six_geometry_kernels.cu -o control.o
run "$hip/bin/hipcc" "${flags[@]}" -ffast-math -fno-finite-math-only \
    -c source/tests/glm5_bf16_six_native_kernels.cu -o candidate.o
run "$hip/bin/hipcc" "${flags[@]}" -fno-fast-math -ffp-contract=off \
    -c source/tests/test_rocm_glm5_six_native.cu -o fixture.o
run "$hip/bin/hipcc" "${flags[@]}" -fno-fast-math -ffp-contract=off \
    fixture.o candidate.o control.o -L"$hip/lib" -Wl,-rpath,"$hip/lib" \
    -o test-six-native
sha256sum test-six-native ./*.o BUILD-SOURCE BUILD-COMMAND compiler.txt \
    source/rocm/ds4_rocm_bf16_toktile.cuh control/rocm/ds4_rocm_bf16_toktile.cuh \
    source/tests/{glm5_bf16_six_native_kernels.cu,test_rocm_glm5_six_native.cu,glm5_gguf_test.hpp} \
    control/tests/glm5_bf16_six_geometry_kernels.cu \
    source/scripts/build-glm53-six-native.sh > BUILD-SHA256SUMS
printf 'artifact=%s\n' "$artifact"
