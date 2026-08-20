/* Production-path regression gate for the optional Q4_K hot-tile fused mid.
 *
 * Run this executable in separate control/candidate processes because the
 * production environment switches are intentionally snapshotted once.  The
 * Makefile target compares both the routed mid buffer and final MoE output
 * bit-for-bit.  Its synthetic 4096x2048x4096 layer exercises the real
 * shape-gated production kernels, including mixed 5/6/7/16/17-count buckets.
 */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg) do {                                                \
    if (!(cond)) {                                                           \
        fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);           \
        return 1;                                                            \
    }                                                                        \
} while (0)

typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

enum {
    Q4_K_TYPE = 12,
    Q4_K_BLOCK_BYTES = 144,
    N_EXPERT = 6,
    N_USED = 6,
    N_TOKENS = 32,
    IN_DIM = 4096,
    MID_DIM = 2048,
    OUT_DIM = 4096,
};

static void pack_q4k_block(unsigned char *dst, uint32_t seed) {
    dst[0] = 0x00; dst[1] = 0x28; /* d = 1/32 */
    dst[2] = 0x00; dst[3] = 0x00; /* dmin = 0 */
    for (uint32_t i = 0; i < 4; i++) dst[4 + i] = 1;
    for (uint32_t i = 4; i < 8; i++) dst[4 + i] = 0;
    for (uint32_t i = 8; i < 12; i++) dst[4 + i] = 1;
    for (uint32_t i = 0; i < 128; i++) {
        const uint8_t lo = (uint8_t)((seed + 3u * i + 1u) & 15u);
        const uint8_t hi = (uint8_t)((seed + 5u * i + 7u) & 15u);
        dst[16 + i] = (uint8_t)(lo | (hi << 4));
    }
}

static void pack_table(unsigned char *dst, uint32_t rows,
                       uint32_t blocks_per_row, uint32_t salt) {
    for (uint32_t e = 0; e < N_EXPERT; e++) {
        for (uint32_t row = 0; row < rows; row++) {
            for (uint32_t block = 0; block < blocks_per_row; block++) {
                const uint64_t i = ((uint64_t)e * rows + row) *
                                   blocks_per_row + block;
                pack_q4k_block(dst + i * Q4_K_BLOCK_BYTES,
                               salt + 17u * e + 13u * row + 7u * block);
            }
        }
    }
}

static int alloc_tensor(ds4_gpu_tensor *t, uint64_t bytes) {
    memset(t, 0, sizeof(*t));
    return ds4_gpu_tensor_alloc_on(t, 0, bytes) == 0;
}

static int upload(ds4_gpu_tensor *t, const void *src, uint64_t bytes) {
    return ds4_gpu_tensor_write(t, 0, src, bytes) != 0;
}

static uint64_t fnv1a64(const void *data, size_t bytes, uint64_t hash) {
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < bytes; i++) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

int main(int argc, char **argv) {
    CHECK(argc == 2, "usage: test_rocm_q4k_fused_mid OUTPUT");
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm backend");

    const uint64_t gate_row_bytes = 16u * Q4_K_BLOCK_BYTES;
    const uint64_t down_row_bytes = 8u * Q4_K_BLOCK_BYTES;
    const uint64_t gate_expert_bytes = (uint64_t)MID_DIM * gate_row_bytes;
    const uint64_t down_expert_bytes = (uint64_t)OUT_DIM * down_row_bytes;
    const uint64_t gate_table_bytes = N_EXPERT * gate_expert_bytes;
    const uint64_t down_table_bytes = N_EXPERT * down_expert_bytes;
    const uint64_t gate_off = 0;
    const uint64_t up_off = gate_table_bytes;
    const uint64_t down_off = 2u * gate_table_bytes;
    const uint64_t model_bytes = down_off + down_table_bytes;
    unsigned char *model = (unsigned char *)malloc((size_t)model_bytes);
    CHECK(model, "allocate synthetic Q4_K model");
    pack_table(model + gate_off, MID_DIM, 16, 11);
    pack_table(model + up_off, MID_DIM, 16, 37);
    pack_table(model + down_off, OUT_DIM, 8, 73);
    CHECK(ds4_gpu_set_model_map(model, model_bytes), "install synthetic model");

    const uint32_t pairs = N_TOKENS * N_USED;
    const uint64_t mid_bytes = (uint64_t)pairs * MID_DIM * sizeof(float);
    const uint64_t out_bytes = (uint64_t)N_TOKENS * OUT_DIM * sizeof(float);
    const uint64_t down_scratch_bytes = (uint64_t)pairs * OUT_DIM * sizeof(float);
    ds4_gpu_tensor out = {}, gate = {}, up = {}, mid = {}, down = {};
    ds4_gpu_tensor selected = {}, weights = {}, x = {};
    CHECK(alloc_tensor(&out, out_bytes), "allocate output");
    CHECK(alloc_tensor(&gate, mid_bytes), "allocate gate scratch");
    CHECK(alloc_tensor(&up, mid_bytes), "allocate up scratch");
    CHECK(alloc_tensor(&mid, mid_bytes), "allocate mid scratch");
    CHECK(alloc_tensor(&down, down_scratch_bytes), "allocate down scratch");
    CHECK(alloc_tensor(&selected, pairs * sizeof(int32_t)), "allocate routes");
    CHECK(alloc_tensor(&weights, pairs * sizeof(float)), "allocate weights");
    CHECK(alloc_tensor(&x, (uint64_t)N_TOKENS * IN_DIM * sizeof(float)),
          "allocate inputs");

    int32_t h_selected[pairs];
    float h_weights[pairs];
    float *h_x = (float *)malloc((size_t)N_TOKENS * IN_DIM * sizeof(float));
    CHECK(h_x, "allocate host inputs");
    const uint32_t counts[N_EXPERT] = {5, 6, 7, 16, 17, 141};
    uint32_t p = 0;
    for (uint32_t e = 0; e < N_EXPERT; e++) {
        for (uint32_t i = 0; i < counts[e]; i++, p++) h_selected[p] = (int32_t)e;
    }
    CHECK(p == pairs, "route histogram covers all pairs");
    for (uint32_t i = 0; i < pairs; i++)
        h_weights[i] = i == 0 ? 0.0f : 0.125f + (float)(i % 11u) * 0.03125f;
    for (uint64_t i = 0; i < (uint64_t)N_TOKENS * IN_DIM; i++)
        h_x[i] = (float)((int)(i % 61u) - 30) * 0.015625f;
    CHECK(upload(&selected, h_selected, sizeof(h_selected)), "upload routes");
    CHECK(upload(&weights, h_weights, sizeof(h_weights)), "upload weights");
    CHECK(upload(&x, h_x, (uint64_t)N_TOKENS * IN_DIM * sizeof(float)),
          "upload inputs");

    bool mid_is_f16 = true;
    CHECK(ds4_gpu_routed_moe_batch_tensor(
              &out, &gate, &up, &mid, &down, model, model_bytes,
              gate_off, up_off, down_off, Q4_K_TYPE, Q4_K_TYPE,
              gate_expert_bytes, gate_row_bytes,
              down_expert_bytes, down_row_bytes,
              IN_DIM, MID_DIM, OUT_DIM, &selected, &weights,
              N_EXPERT, N_USED, 3.0f, &x, 0, N_TOKENS,
              &mid_is_f16, true),
          "run production Q4_K batch path");
    CHECK(!mid_is_f16, "Q4_K path keeps FP32 mid");

    float *h_mid = (float *)malloc((size_t)mid_bytes);
    float *h_out = (float *)malloc((size_t)out_bytes);
    CHECK(h_mid && h_out, "allocate readback");
    CHECK(ds4_gpu_tensor_read(&mid, 0, h_mid, mid_bytes), "read mid");
    CHECK(ds4_gpu_tensor_read(&out, 0, h_out, out_bytes), "read output");
    FILE *f = fopen(argv[1], "wb");
    CHECK(f, "open output artifact");
    CHECK(fwrite(h_mid, 1, (size_t)mid_bytes, f) == mid_bytes,
          "write mid artifact");
    CHECK(fwrite(h_out, 1, (size_t)out_bytes, f) == out_bytes,
          "write output artifact");
    CHECK(fclose(f) == 0, "close output artifact");
    uint64_t hash = fnv1a64(h_mid, (size_t)mid_bytes,
                            UINT64_C(1469598103934665603));
    hash = fnv1a64(h_out, (size_t)out_bytes, hash);
    printf("test_rocm_q4k_fused_mid: PASS fnv64=%016llx fused=%s\n",
           (unsigned long long)hash,
           getenv("DS4_ROCM_Q4K_WMMA_FUSE_MID") ?: "unset");

    free(h_out); free(h_mid); free(h_x); free(model);
    ds4_gpu_tensor_free_in_place(&x);
    ds4_gpu_tensor_free_in_place(&weights);
    ds4_gpu_tensor_free_in_place(&selected);
    ds4_gpu_tensor_free_in_place(&down);
    ds4_gpu_tensor_free_in_place(&mid);
    ds4_gpu_tensor_free_in_place(&up);
    ds4_gpu_tensor_free_in_place(&gate);
    ds4_gpu_tensor_free_in_place(&out);
    ds4_gpu_cleanup();
    return 0;
}
