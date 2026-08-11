/* Regression test for one-token Q4_K TP expert skipping.
 *
 * Unowned experts retain selected == 0 and routing weight == 0.  With the
 * opt-in skip enabled, decode kernels must overwrite their intermediate rows
 * with zero and the direct six-slot down kernel must omit them. Reuse poisoned
 * scratch across changing masks:
 * a test which starts from zero would hide the historical stale-row failure.
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
    QK_K = 256,
    Q4_K_BLOCK_BYTES = 144,
    N_EXPERT = 8,
    N_USED = 6,
    IN_DIM = 256,
    MID_DIM = 256,
    OUT_DIM = 256,
};

static void pack_q4k_block(unsigned char *dst, uint32_t seed) {
    /* d=1/32, dmin=0. All eight 32-value groups have scale 1, min 0. */
    dst[0] = 0x00;
    dst[1] = 0x28;
    dst[2] = 0x00;
    dst[3] = 0x00;
    for (uint32_t i = 0; i < 4; i++) dst[4 + i] = 1;
    for (uint32_t i = 4; i < 8; i++) dst[4 + i] = 0;
    for (uint32_t i = 8; i < 12; i++) dst[4 + i] = 1;
    for (uint32_t i = 0; i < 128; i++) {
        const uint8_t lo = (uint8_t)((seed + 3u * i + 1u) & 15u);
        const uint8_t hi = (uint8_t)((seed + 5u * i + 7u) & 15u);
        dst[16 + i] = (uint8_t)(lo | (hi << 4));
    }
}

static void pack_expert_table(unsigned char *dst, uint32_t salt) {
    for (uint32_t e = 0; e < N_EXPERT; e++) {
        for (uint32_t row = 0; row < MID_DIM; row++) {
            pack_q4k_block(dst + ((uint64_t)e * MID_DIM + row) *
                                    Q4_K_BLOCK_BYTES,
                           salt + 17u * e + 13u * row);
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

static int poison(ds4_gpu_tensor *t, uint64_t bytes) {
    void *p = malloc((size_t)bytes);
    if (!p) return 0;
    memset(p, 0x7f, (size_t)bytes);
    const int ok = upload(t, p, bytes);
    free(p);
    return ok;
}

static int run_one(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down,
        ds4_gpu_tensor *selected,
        ds4_gpu_tensor *weights,
        ds4_gpu_tensor *x,
        const ds4_gpu_tensor *add_in,
        const void *model,
        uint64_t model_bytes,
        uint64_t gate_off,
        uint64_t up_off,
        uint64_t down_off) {
    const uint64_t row_bytes = Q4_K_BLOCK_BYTES;
    const uint64_t expert_bytes = (uint64_t)MID_DIM * row_bytes;
    return ds4_gpu_routed_moe_one_tensor(
        out, gate, up, mid, down, model, model_bytes,
        gate_off, up_off, down_off, Q4_K_TYPE, Q4_K_TYPE,
        expert_bytes, row_bytes, expert_bytes, row_bytes,
        IN_DIM, MID_DIM, OUT_DIM, selected, weights,
        N_EXPERT, N_USED, 0.0f, x, add_in, 0, true);
}

static int all_zero_bits(const float *v, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        uint32_t bits = 0;
        memcpy(&bits, v + i, sizeof(bits));
        if ((bits & UINT32_C(0x7fffffff)) != 0) return 0;
    }
    return 1;
}

int main(void) {
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm backend");

    const uint64_t expert_bytes =
        (uint64_t)MID_DIM * Q4_K_BLOCK_BYTES;
    const uint64_t table_bytes = (uint64_t)N_EXPERT * expert_bytes;
    const uint64_t gate_off = 0;
    const uint64_t up_off = table_bytes;
    const uint64_t down_off = 2u * table_bytes;
    const uint64_t model_bytes = 3u * table_bytes;
    unsigned char *model = (unsigned char *)malloc((size_t)model_bytes);
    CHECK(model, "allocate synthetic Q4_K model");
    pack_expert_table(model + gate_off, 11);
    pack_expert_table(model + up_off, 37);
    pack_expert_table(model + down_off, 73);
    CHECK(ds4_gpu_set_model_map(model, model_bytes), "install synthetic model");

    const uint64_t pair_values = (uint64_t)N_USED * MID_DIM;
    const uint64_t pair_bytes = pair_values * sizeof(float);
    const uint64_t out_bytes = (uint64_t)OUT_DIM * sizeof(float);
    ds4_gpu_tensor out = {}, gate = {}, up = {}, mid = {}, down = {};
    ds4_gpu_tensor selected = {}, weights = {}, x = {}, add_in = {};
    CHECK(alloc_tensor(&out, out_bytes), "allocate out");
    CHECK(alloc_tensor(&gate, pair_bytes), "allocate gate/midq scratch");
    CHECK(alloc_tensor(&up, pair_bytes), "allocate up scratch");
    CHECK(alloc_tensor(&mid, pair_bytes), "allocate mid scratch");
    CHECK(alloc_tensor(&down, pair_bytes), "allocate down/xq scratch");
    CHECK(alloc_tensor(&selected, N_USED * sizeof(int32_t)), "allocate selection");
    CHECK(alloc_tensor(&weights, N_USED * sizeof(float)), "allocate weights");
    CHECK(alloc_tensor(&x, IN_DIM * sizeof(float)), "allocate input");
    CHECK(alloc_tensor(&add_in, out_bytes), "allocate addend");

    float hx[IN_DIM];
    for (uint32_t i = 0; i < IN_DIM; i++)
        hx[i] = (float)((int)(i % 31u) - 15) * 0.03125f;
    const float active_weights[N_USED] = {0.31f, 0.23f, 0.17f, 0.13f, 0.09f, 0.07f};
    CHECK(upload(&x, hx, sizeof(hx)), "upload input");

    /* An all-skipped first call must overwrite poisoned mid and output rows. */
    const int32_t all_skipped[N_USED] = {0, 0, 0, 0, 0, 0};
    const float zero_weights[N_USED] = {0, 0, 0, 0, 0, 0};
    CHECK(poison(&out, out_bytes) && poison(&gate, pair_bytes) &&
          poison(&up, pair_bytes) && poison(&mid, pair_bytes) &&
          poison(&down, pair_bytes), "poison scratch");
    CHECK(upload(&selected, all_skipped, sizeof(all_skipped)) &&
          upload(&weights, zero_weights, sizeof(zero_weights)),
          "upload all-skipped route");
    CHECK(setenv("DS4_ROCM_TP_SKIP_UNOWNED", "1", 1) == 0,
          "enable zero-weight skip");
    CHECK(run_one(&out, &gate, &up, &mid, &down, &selected, &weights, &x, NULL,
                  model, model_bytes, gate_off, up_off, down_off),
          "run all-skipped route");
    float *host_mid = (float *)malloc((size_t)pair_bytes);
    float host_out[OUT_DIM];
    CHECK(host_mid, "allocate readback");
    CHECK(ds4_gpu_tensor_read(&mid, 0, host_mid, pair_bytes) &&
          ds4_gpu_tensor_read(&out, 0, host_out, out_bytes),
          "read all-skipped result");
    CHECK(all_zero_bits(host_mid, pair_values), "skipped mid rows are fresh zero");
    CHECK(all_zero_bits(host_out, OUT_DIM), "all-skipped output is fresh zero");

    /* Call A writes nonzero rows. Call B skips some of those rows without any
     * scratch reset. Compare B with the exact sentinel-0 masking scheme. */
    const int32_t route_a[N_USED] = {0, 1, 2, 3, 7, 7};
    const int32_t route_b[N_USED] = {0, 0, 7, 0, 3, 4};
    const float route_b_weights[N_USED] =
        {0.0f, 0.0f, 0.31f, 0.23f, 0.17f, 0.13f};
    CHECK(upload(&selected, route_a, sizeof(route_a)) &&
          upload(&weights, active_weights, sizeof(active_weights)) &&
          unsetenv("DS4_ROCM_TP_SKIP_UNOWNED") == 0 &&
          run_one(&out, &gate, &up, &mid, &down, &selected, &weights, &x, NULL,
                  model, model_bytes, gate_off, up_off, down_off),
          "run active route A");
    CHECK(upload(&selected, route_b, sizeof(route_b)) &&
          upload(&weights, route_b_weights, sizeof(route_b_weights)) &&
          setenv("DS4_ROCM_TP_SKIP_UNOWNED", "1", 1) == 0 &&
          run_one(&out, &gate, &up, &mid, &down, &selected, &weights, &x, NULL,
                  model, model_bytes, gate_off, up_off, down_off),
          "run skip route B");
    float skip_out[OUT_DIM];
    CHECK(ds4_gpu_tensor_read(&out, 0, skip_out, sizeof(skip_out)),
          "read skip output");
    CHECK(ds4_gpu_tensor_read(&mid, 0, host_mid, pair_bytes), "read skip mid");
    CHECK(all_zero_bits(host_mid, 2u * MID_DIM),
          "newly skipped rows overwrite route A values");

    CHECK(poison(&out, out_bytes) && poison(&gate, pair_bytes) &&
          poison(&up, pair_bytes) && poison(&mid, pair_bytes) &&
          poison(&down, pair_bytes), "repoison masking control");
    CHECK(upload(&selected, route_b, sizeof(route_b)) &&
          upload(&weights, route_b_weights, sizeof(route_b_weights)) &&
          unsetenv("DS4_ROCM_TP_SKIP_UNOWNED") == 0 &&
          run_one(&out, &gate, &up, &mid, &down, &selected, &weights, &x, NULL,
                  model, model_bytes, gate_off, up_off, down_off),
          "run sentinel-zero masking control");
    float mask_out[OUT_DIM];
    CHECK(ds4_gpu_tensor_read(&out, 0, mask_out, sizeof(mask_out)),
          "read masking output");
    CHECK(memcmp(skip_out, mask_out, sizeof(skip_out)) == 0,
          "skip and zero-weight masking outputs are bit-exact");

    /* Folding the shared-expert addend into the direct down kernel must match
     * the established post-kernel ds4_gpu_add_tensor path bit for bit. */
    float host_add[OUT_DIM];
    for (uint32_t i = 0; i < OUT_DIM; i++)
        host_add[i] = (float)((int)(i % 19u) - 9) * 0.00390625f;
    CHECK(upload(&add_in, host_add, sizeof(host_add)), "upload addend");
    CHECK(unsetenv("DS4_ROCM_Q4K_DECODE_FUSE_ADDEND") == 0 &&
          setenv("DS4_ROCM_TP_SKIP_UNOWNED", "1", 1) == 0 &&
          run_one(&out, &gate, &up, &mid, &down, &selected, &weights, &x,
                  &add_in, model, model_bytes, gate_off, up_off, down_off),
          "run post-add control");
    float post_add_out[OUT_DIM];
    CHECK(ds4_gpu_tensor_read(&out, 0, post_add_out, sizeof(post_add_out)),
          "read post-add control");
    CHECK(setenv("DS4_ROCM_Q4K_DECODE_FUSE_ADDEND", "1", 1) == 0 &&
          run_one(&out, &gate, &up, &mid, &down, &selected, &weights, &x,
                  &add_in, model, model_bytes, gate_off, up_off, down_off),
          "run fused addend");
    float fused_add_out[OUT_DIM];
    CHECK(ds4_gpu_tensor_read(&out, 0, fused_add_out, sizeof(fused_add_out)),
          "read fused addend");
    CHECK(memcmp(post_add_out, fused_add_out, sizeof(post_add_out)) == 0,
          "fused and post-kernel addend outputs are bit-exact");

    printf("test_rocm_q4k_skip_unowned: PASS staged_xq=%s\n",
           getenv("DS4_ROCM_Q4K_DECODE_STAGE_XQ") ?: "unset");
    free(host_mid);
    free(model);
    ds4_gpu_tensor_free_in_place(&add_in);
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
