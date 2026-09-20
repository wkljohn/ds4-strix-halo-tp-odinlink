#!/usr/bin/env bash
# New precise fixture linked to unchanged, frozen production HIP objects.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo/scripts/ds4-research-root.sh"
ds4_resolve_research_roots "$repo"
revision=$(git -C "$repo" rev-parse HEAD)
artifact=$DS4_RESEARCH_ROOT/builds/glm53-mla-output-fixture-${revision:0:7}
donor=$DS4_RESEARCH_ROOT/builds/glm53-full-6eba8bd
sdk=${DS4_ROCM_HOME:-/home/wkljohn/Desktop/cc/toolchains/rocm-10.0.0-gfx1151/install}
python3 "$repo/scripts/check-rocm-toolchain.py" --home "$sdk" --hipcc "$sdk/bin/hipcc"
[[ -z $(git -C "$repo" status --porcelain) && ! -e $artifact ]]
[[ $(git -C "$donor" rev-parse HEAD) == 6eba8bdb134b8153cbc1dc1cf4dd5b4de2f52957 ]]
[[ -z $(git -C "$donor" status --porcelain) ]]
(cd "$donor" && sha256sum --check --quiet BUILD-SHA256SUMS)
for header in ds4_gpu.h ds4_gpu_mgpu.h tests/glm5_gguf_test.hpp; do
    cmp "$repo/$header" "$donor/$header"
done
mkdir "$artifact"
git -C "$repo" worktree add --detach "$artifact/source" "$revision"
cd "$artifact"
printf 'fixture_source=%s\ninference_object_source=%s\ninference_artifact=%s\n' \
    "$revision" 6eba8bdb134b8153cbc1dc1cf4dd5b4de2f52957 "$donor" > BUILD-SOURCE
"$sdk/bin/hipcc" --version > compiler.txt
flags=(-O3 -g -pthread -D__HIP_PLATFORM_AMD__ --offload-arch=gfx1151
       -mno-wavefrontsize64 -DDS4_GFX1151_WAVE32=1 -std=c++17
       -isystem "$sdk/include")
run() {
    printf '%q ' "$@" >> BUILD-COMMAND
    printf '\n' >> BUILD-COMMAND
    "$@" >> build.log 2>&1
}
run "$sdk/bin/hipcc" "${flags[@]}" -fno-fast-math -ffp-contract=off \
    -I"$artifact/source" -c source/tests/test_rocm_glm5_mla_output_wmma.cu -o fixture.o
objects=("$donor/tests/ds4_tp_hello_test.o" "$donor/ds4_rocm.o"
         "$donor/ds4_rocm_compat.o" "$donor/ds4_rocm_unavailable.o")
run "$sdk/bin/hipcc" "${flags[@]}" -fno-fast-math -ffp-contract=off \
    -Wl,--gc-sections fixture.o "${objects[@]}" -L"$sdk/lib" \
    -Wl,-rpath,"$sdk/lib" -lhipblas -lhipblaslt -o test-mla-output
sha256sum test-mla-output fixture.o BUILD-SOURCE BUILD-COMMAND compiler.txt \
    source/tests/test_rocm_glm5_mla_output_wmma.cu source/tests/glm5_gguf_test.hpp \
    source/ds4_gpu.h source/ds4_gpu_mgpu.h source/scripts/build-glm53-mla-output-fixture.sh \
    "$donor/BUILD-SHA256SUMS" "$donor/BUILD-COMMAND" "${objects[@]}" \
    > BUILD-SHA256SUMS
printf 'artifact=%s\n' "$artifact"
