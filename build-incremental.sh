#!/bin/bash
# Incremental ROCm relink: rebuild only what changed, then link ds4.
# The `strix-halo` Makefile target passes -B (force full rebuild); this runs the
# same inner make WITHOUT -B so only stale objects recompile.
set -euo pipefail
cd /home/wkljohn/Desktop/cc/ds4-strix-halo-tp

HIPCC=$(command -v hipcc 2>/dev/null || echo /opt/rocm/bin/hipcc)

make -j4 ds4 \
  CORE_OBJS="ds4.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o ds4_layer_pack.o" \
  CFLAGS="-O3 -ffast-math -march=native -Wall -Wextra -std=c99 -D_GNU_SOURCE -fno-finite-math-only -DDS4_ROCM_BUILD -DDS4_ROCM_TP_READY=1" \
  DS4_LINK="$HIPCC -O3 -ffast-math -g -fno-finite-math-only -pthread -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument --offload-arch=gfx1151" \
  DS4_LINK_LIBS="-lm -pthread -lhipblas -lhipblaslt"

echo "BUILD_OK"
sha256sum ds4
