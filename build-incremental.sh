#!/bin/bash
# Incremental ROCm relink: rebuild only what changed, then link ds4.
# The `strix-halo` Makefile target passes -B (force full rebuild); this runs the
# same inner make WITHOUT -B so only stale objects recompile.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

if [[ -n "${DS4_ROCM_HOME:-}" ]]; then
  ROCM_HOME=$DS4_ROCM_HOME
elif [[ -x "$SCRIPT_DIR/../toolchains/rocm-7.14.0-gfx1151/install/bin/hipcc" ]]; then
  ROCM_HOME=$SCRIPT_DIR/../toolchains/rocm-7.14.0-gfx1151/install
elif [[ -x /opt/rocm-7.14.0/bin/hipcc ]]; then
  ROCM_HOME=/opt/rocm-7.14.0
else
  echo "error: ROCm 7.14 hipcc not found; set DS4_ROCM_HOME" >&2
  exit 1
fi
HIPCC=$ROCM_HOME/bin/hipcc

make -j4 ds4 \
  CORE_OBJS="ds4.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o ds4_layer_pack.o ds4_glm5_kda.o ds4_glm5_next_runtime.o" \
  CFLAGS="-O3 -ffast-math -g -march=native -Wall -Wextra -std=c99 -D_GNU_SOURCE -fno-finite-math-only -DDS4_ROCM_BUILD -DDS4_ROCM_TP_READY=1" \
  DS4_LINK="$HIPCC -O3 -ffast-math -g -fno-finite-math-only -pthread -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument --offload-arch=gfx1151 -mno-wavefrontsize64 -DDS4_GFX1151_WAVE32=1" \
  DS4_LINK_LIBS="-lm -pthread -L$ROCM_HOME/lib -Wl,-rpath,$ROCM_HOME/lib -lhipblas -lhipblaslt"

echo "BUILD_OK"
sha256sum ds4
