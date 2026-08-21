/* Slot-balance oracle: rank0 slots 0-2 + rank1 slots 3-5, then add,
 * vs shipped full one_tensor. Drives ds4_gpu_routed_moe_one_tensor.
 */
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <hip/hip_runtime.h>
#include <math.h>
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
    QK_K = 256,
    Q4_K_BLOCK_BYTES = 144,
    N_TOTAL = 8,
    N_USED = 6,
    IN_DIM = 4096,
    MID_DIM = 2048,
    OUT_DIM = 4096,
};

static void pack_q4k_block(unsigned char *dst, uint32_t seed) {
    dst[0] = 0x00; dst[1] = 0x28; dst[2] = 0x00; dst[3] = 0x00;
    for (uint32_t i = 0; i < 4; i++) dst[4 + i] = 1;
    for (uint32_t i = 4; i < 8; i++) dst[4 + i] = 0;
    for (uint32_t i = 8; i < 12; i++) dst[4 + i] = 1;
    for (uint32_t i = 0; i < 128; i++) {
        const uint8_t lo = (uint8_t)((seed + 3u * i + 1u) & 15u);
        const uint8_t hi = (uint8_t)((seed + 5u * i + 7u) & 15u);
        dst[16 + i] = (uint8_t)(lo | (hi << 4));
    }
}

static void pack_q4k_table(unsigned char *dst, uint32_t experts,
                           uint32_t rows, uint32_t blocks_per_row,
                           uint32_t salt) {
    for (uint32_t e = 0; e < experts; e++) {
        for (uint32_t row = 0; row < rows; row++) {
            for (uint32_t b = 0; b < blocks_per_row; b++) {
                const uint64_t i = ((uint64_t)e * rows + row) *
                                   blocks_per_row + b;
                pack_q4k_block(dst + i * Q4_K_BLOCK_BYTES,
                               salt + 17u * e + 13u * row + 7u * b);
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

static int run_moe(ds4_gpu_tensor *out, ds4_gpu_tensor *gate,
                   ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
                   ds4_gpu_tensor *down, ds4_gpu_tensor *selected,
                   ds4_gpu_tensor *weights, ds4_gpu_tensor *x,
                   const void *model, uint64_t model_bytes,
                   uint64_t gate_off, uint64_t up_off, uint64_t down_off,
                   uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
                   uint64_t down_expert_bytes, uint64_t down_row_bytes) {
    return ds4_gpu_routed_moe_one_tensor(
        out, gate, up, mid, down, model, model_bytes,
        gate_off, up_off, down_off, Q4_K_TYPE, Q4_K_TYPE,
        gate_expert_bytes, gate_row_bytes,
        down_expert_bytes, down_row_bytes,
        IN_DIM, MID_DIM, OUT_DIM, selected, weights,
        N_TOTAL, N_USED, 0.0f, x, NULL, 0, true);
}

int main(void) {
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0, "ROCm");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "init");
    CHECK(setenv("DS4_ROCM_Q4K_DECODE_STAGE_XQ", "1", 1) == 0, "xq");
    CHECK(unsetenv("DS4_ROCM_TP_SLOT_BALANCE") == 0, "no compact");
    CHECK(setenv("DS4_ROCM_TP_SKIP_UNOWNED", "1", 1) == 0, "skip");

    const uint64_t in_blocks = IN_DIM / QK_K;
    const uint64_t mid_blocks = MID_DIM / QK_K;
    const uint64_t gate_row_bytes = in_blocks * Q4_K_BLOCK_BYTES;
    const uint64_t down_row_bytes = mid_blocks * Q4_K_BLOCK_BYTES;
    const uint64_t gate_expert_bytes = MID_DIM * gate_row_bytes;
    const uint64_t down_expert_bytes = OUT_DIM * down_row_bytes;
    const uint64_t gate_off = 0;
    const uint64_t up_off = N_TOTAL * gate_expert_bytes;
    const uint64_t down_off = up_off + N_TOTAL * gate_expert_bytes;
    const uint64_t model_bytes = down_off + N_TOTAL * down_expert_bytes;
    unsigned char *model = (unsigned char *)malloc((size_t)model_bytes);
    CHECK(model, "model");
    pack_q4k_table(model + gate_off, N_TOTAL, MID_DIM, (uint32_t)in_blocks, 11);
    pack_q4k_table(model + up_off, N_TOTAL, MID_DIM, (uint32_t)in_blocks, 37);
    pack_q4k_table(model + down_off, N_TOTAL, OUT_DIM, (uint32_t)mid_blocks, 73);
    CHECK(ds4_gpu_set_model_map(model, model_bytes), "map");

    ds4_gpu_tensor out = {}, gate = {}, up = {}, mid = {}, down = {};
    ds4_gpu_tensor selected = {}, wts = {}, x = {};
    CHECK(alloc_tensor(&out, OUT_DIM * sizeof(float)), "out");
    CHECK(alloc_tensor(&gate, (uint64_t)N_USED * MID_DIM * sizeof(float)), "gate");
    CHECK(alloc_tensor(&up, (uint64_t)N_USED * MID_DIM * sizeof(float)), "up");
    CHECK(alloc_tensor(&mid, (uint64_t)N_USED * MID_DIM * sizeof(float)), "mid");
    CHECK(alloc_tensor(&down, (uint64_t)N_USED * OUT_DIM * sizeof(float)), "down");
    CHECK(alloc_tensor(&selected, N_USED * sizeof(int32_t)), "sel");
    CHECK(alloc_tensor(&wts, N_USED * sizeof(float)), "w");
    CHECK(alloc_tensor(&x, IN_DIM * sizeof(float)), "x");

    const int32_t route[N_USED] = {0, 1, 2, 3, 4, 5};
    const float w_all[N_USED] = {0.31f, 0.23f, 0.17f, 0.13f, 0.09f, 0.07f};
    const float w0[N_USED] = {0.31f, 0.23f, 0.17f, 0.0f, 0.0f, 0.0f};
    const float w1[N_USED] = {0.0f, 0.0f, 0.0f, 0.13f, 0.09f, 0.07f};
    float hx[IN_DIM];
    for (uint32_t i = 0; i < IN_DIM; i++)
        hx[i] = (float)((int)(i % 31u) - 15) * 0.03125f;
    CHECK(upload(&x, hx, sizeof(hx)), "hx");
    CHECK(upload(&selected, route, sizeof(route)), "route");

    CHECK(upload(&wts, w_all, sizeof(w_all)), "wall");
    CHECK(run_moe(&out, &gate, &up, &mid, &down, &selected, &wts, &x,
                  model, model_bytes, gate_off, up_off, down_off,
                  gate_expert_bytes, gate_row_bytes,
                  down_expert_bytes, down_row_bytes), "gold");
    CHECK(hipDeviceSynchronize() == hipSuccess, "sync gold");
    float gold[OUT_DIM];
    CHECK(ds4_gpu_tensor_read(&out, 0, gold, sizeof(gold)), "read gold");

    CHECK(upload(&wts, w0, sizeof(w0)), "w0");
    CHECK(run_moe(&out, &gate, &up, &mid, &down, &selected, &wts, &x,
                  model, model_bytes, gate_off, up_off, down_off,
                  gate_expert_bytes, gate_row_bytes,
                  down_expert_bytes, down_row_bytes), "rank0");
    CHECK(hipDeviceSynchronize() == hipSuccess, "sync r0");
    float r0[OUT_DIM];
    CHECK(ds4_gpu_tensor_read(&out, 0, r0, sizeof(r0)), "read r0");

    CHECK(upload(&wts, w1, sizeof(w1)), "w1");
    CHECK(run_moe(&out, &gate, &up, &mid, &down, &selected, &wts, &x,
                  model, model_bytes, gate_off, up_off, down_off,
                  gate_expert_bytes, gate_row_bytes,
                  down_expert_bytes, down_row_bytes), "rank1");
    CHECK(hipDeviceSynchronize() == hipSuccess, "sync r1");
    float r1[OUT_DIM];
    CHECK(ds4_gpu_tensor_read(&out, 0, r1, sizeof(r1)), "read r1");

    ds4_gpu_tensor a = {}, b = {}, sum = {};
    CHECK(alloc_tensor(&a, sizeof(r0)), "a");
    CHECK(alloc_tensor(&b, sizeof(r1)), "b");
    CHECK(alloc_tensor(&sum, sizeof(gold)), "sum");
    CHECK(upload(&a, r0, sizeof(r0)), "ua");
    CHECK(upload(&b, r1, sizeof(r1)), "ub");
    CHECK(ds4_gpu_add_tensor(&sum, &a, &b, OUT_DIM) != 0, "add");
    float got[OUT_DIM];
    CHECK(ds4_gpu_tensor_read(&sum, 0, got, sizeof(got)), "read sum");
    float max_abs = 0.0f, max_rel = 0.0f;
    for (uint32_t i = 0; i < OUT_DIM; i++) {
        const float e = fabsf(got[i] - gold[i]);
        const float d = fmaxf(1.0f, fabsf(gold[i]));
        if (e > max_abs) max_abs = e;
        if (e / d > max_rel) max_rel = e / d;
    }
    /* Two 3-slot sums added are not the same association as one 6-slot
     * sum6 tree. Token fingerprint is the production gate. */
    printf("test_rocm_q4k_slot_balance_oracle: max_abs=%.6e max_rel=%.6e\n",
           max_abs, max_rel);
    CHECK(max_rel < 1e-5f, "slot-balance add documented-ULP vs full");
    printf("test_rocm_q4k_slot_balance_oracle: PASS\n");

    ds4_gpu_tensor_free_in_place(&sum);
    ds4_gpu_tensor_free_in_place(&b);
    ds4_gpu_tensor_free_in_place(&a);
    ds4_gpu_tensor_free_in_place(&x);
    ds4_gpu_tensor_free_in_place(&wts);
    ds4_gpu_tensor_free_in_place(&selected);
    ds4_gpu_tensor_free_in_place(&down);
    ds4_gpu_tensor_free_in_place(&mid);
    ds4_gpu_tensor_free_in_place(&up);
    ds4_gpu_tensor_free_in_place(&gate);
    ds4_gpu_tensor_free_in_place(&out);
    free(model);
    ds4_gpu_cleanup();
    return 0;
}
