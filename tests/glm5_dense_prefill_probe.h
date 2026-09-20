#ifndef GLM5_DENSE_PREFILL_PROBE_H
#define GLM5_DENSE_PREFILL_PROBE_H
#include "ds4_gpu.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Test builds only. Original Q8_0 K4096/N12288, clamp10, M256/M1024.
 * mode0 exposes the incumbent F32 tile; mode1 exposes paired WMMA128;
 * mode2 always stores two explicit WMMA128 projections plus standalone SwiGLU.
 * All tensor ranges must be disjoint; store_gate_up is exactly0 or1. */
int ds4_gpu_test_glm5_dense_prefill(
    ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
    const void *model_map, uint64_t model_size, uint64_t gate_offset,
    uint64_t up_offset, const ds4_gpu_tensor *x, uint32_t rows,
    unsigned mode, unsigned store_gate_up);
#ifdef __cplusplus
}
#endif
#endif
