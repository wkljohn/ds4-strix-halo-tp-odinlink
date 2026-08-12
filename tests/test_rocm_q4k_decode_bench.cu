/* Production-shape, cache-free Q4_K one-token MoE microbenchmark.
 *
 * This is intentionally a whole routed-MoE call rather than a toy dot test:
 * it covers activation quantization, six gate/up experts, SwiGLU, middle
 * quantization, the direct six-expert down sum, and the optional fused addend.
 * Run in separate processes with DS4_ROCM_Q4K_DECODE_STAGE_XQ=0/1 because the
 * production dispatcher resolves that switch once per process.
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
    N_TOTAL_EXPERT = 8,
    N_USED = 6,
    IN_DIM = 4096,
    MID_DIM = 2048,
    OUT_DIM = 4096,
    WARMUP = 8,
    ITERS = 80,
};

static void pack_q4k_block(unsigned char *dst, uint32_t seed) {
    /* d=1/32, dmin=0; all sub-block scales are one and mins are zero. */
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

static uint64_t fnv1a64(const void *data, size_t bytes) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < bytes; i++) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static int run_one(ds4_gpu_tensor *out, ds4_gpu_tensor *gate,
                   ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
                   ds4_gpu_tensor *down, ds4_gpu_tensor *selected,
                   ds4_gpu_tensor *weights, ds4_gpu_tensor *x,
                   ds4_gpu_tensor *add_in, const void *model,
                   uint64_t model_bytes, uint64_t gate_off,
                   uint64_t up_off, uint64_t down_off,
                   uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
                   uint64_t down_expert_bytes, uint64_t down_row_bytes) {
    return ds4_gpu_routed_moe_one_tensor(
        out, gate, up, mid, down, model, model_bytes,
        gate_off, up_off, down_off, Q4_K_TYPE, Q4_K_TYPE,
        gate_expert_bytes, gate_row_bytes,
        down_expert_bytes, down_row_bytes,
        IN_DIM, MID_DIM, OUT_DIM, selected, weights,
        N_TOTAL_EXPERT, N_USED, 0.0f, x, add_in, 0, true);
}

int main(void) {
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm backend");

    const uint64_t in_blocks = IN_DIM / QK_K;
    const uint64_t mid_blocks = MID_DIM / QK_K;
    const uint64_t gate_row_bytes = in_blocks * Q4_K_BLOCK_BYTES;
    const uint64_t down_row_bytes = mid_blocks * Q4_K_BLOCK_BYTES;
    const uint64_t gate_expert_bytes = MID_DIM * gate_row_bytes;
    const uint64_t down_expert_bytes = OUT_DIM * down_row_bytes;
    const uint64_t gate_off = 0;
    const uint64_t up_off = N_TOTAL_EXPERT * gate_expert_bytes;
    const uint64_t down_off = up_off + N_TOTAL_EXPERT * gate_expert_bytes;
    const uint64_t model_bytes = down_off +
                                 N_TOTAL_EXPERT * down_expert_bytes;

    unsigned char *model = (unsigned char *)malloc((size_t)model_bytes);
    CHECK(model, "allocate production-shape Q4_K model");
    pack_q4k_table(model + gate_off, N_TOTAL_EXPERT, MID_DIM,
                   (uint32_t)in_blocks, 11);
    pack_q4k_table(model + up_off, N_TOTAL_EXPERT, MID_DIM,
                   (uint32_t)in_blocks, 37);
    pack_q4k_table(model + down_off, N_TOTAL_EXPERT, OUT_DIM,
                   (uint32_t)mid_blocks, 73);
    CHECK(ds4_gpu_set_model_map(model, model_bytes), "install synthetic model");

    const uint64_t pair_values = (uint64_t)N_USED * MID_DIM;
    ds4_gpu_tensor out = {}, gate = {}, up = {}, mid = {}, down = {};
    ds4_gpu_tensor selected = {}, weights = {}, x = {}, add_in = {};
    CHECK(alloc_tensor(&out, OUT_DIM * sizeof(float)), "allocate out");
    CHECK(alloc_tensor(&gate, pair_values * sizeof(float)), "allocate gate");
    CHECK(alloc_tensor(&up, pair_values * sizeof(float)), "allocate up");
    CHECK(alloc_tensor(&mid, pair_values * sizeof(float)), "allocate mid");
    CHECK(alloc_tensor(&down, (uint64_t)N_USED * OUT_DIM * sizeof(float)),
          "allocate down scratch");
    CHECK(alloc_tensor(&selected, N_USED * sizeof(int32_t)), "allocate selection");
    CHECK(alloc_tensor(&weights, N_USED * sizeof(float)), "allocate weights");
    CHECK(alloc_tensor(&x, IN_DIM * sizeof(float)), "allocate input");
    CHECK(alloc_tensor(&add_in, OUT_DIM * sizeof(float)), "allocate addend");

    const int32_t route[N_USED] = {0, 1, 2, 3, 4, 5};
    const float route_weights[N_USED] = {0.31f, 0.23f, 0.17f, 0.13f, 0.09f, 0.07f};
    float hx[IN_DIM], hadd[OUT_DIM];
    for (uint32_t i = 0; i < IN_DIM; i++)
        hx[i] = (float)((int)(i % 31u) - 15) * 0.03125f;
    for (uint32_t i = 0; i < OUT_DIM; i++)
        hadd[i] = (float)((int)(i % 19u) - 9) * 0.00390625f;
    CHECK(upload(&selected, route, sizeof(route)), "upload route");
    CHECK(upload(&weights, route_weights, sizeof(route_weights)), "upload weights");
    CHECK(upload(&x, hx, sizeof(hx)), "upload input");
    CHECK(upload(&add_in, hadd, sizeof(hadd)), "upload addend");
    CHECK(setenv("DS4_ROCM_TP_SKIP_UNOWNED", "1", 1) == 0,
          "enable exact unowned skip");
    CHECK(setenv("DS4_ROCM_Q4K_DECODE_FUSE_ADDEND", "1", 1) == 0,
          "enable production fused addend");

    for (uint32_t i = 0; i < WARMUP; i++) {
        CHECK(run_one(&out, &gate, &up, &mid, &down, &selected, &weights,
                      &x, &add_in, model, model_bytes, gate_off, up_off,
                      down_off, gate_expert_bytes, gate_row_bytes,
                      down_expert_bytes, down_row_bytes), "warmup routed MoE");
    }
    CHECK(hipDeviceSynchronize() == hipSuccess, "synchronize warmup");

    hipEvent_t start = nullptr, stop = nullptr;
    CHECK(hipEventCreate(&start) == hipSuccess &&
          hipEventCreate(&stop) == hipSuccess, "create timing events");
    CHECK(hipEventRecord(start) == hipSuccess, "record timing start");
    for (uint32_t i = 0; i < ITERS; i++) {
        CHECK(run_one(&out, &gate, &up, &mid, &down, &selected, &weights,
                      &x, &add_in, model, model_bytes, gate_off, up_off,
                      down_off, gate_expert_bytes, gate_row_bytes,
                      down_expert_bytes, down_row_bytes), "timed routed MoE");
    }
    CHECK(hipEventRecord(stop) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess, "finish timing");
    float elapsed_ms = 0.0f;
    CHECK(hipEventElapsedTime(&elapsed_ms, start, stop) == hipSuccess,
          "read timing");

    float hout[OUT_DIM];
    CHECK(ds4_gpu_tensor_read(&out, 0, hout, sizeof(hout)), "read output");
    const char *dump_path = getenv("DS4_TEST_OUTPUT_FILE");
    if (dump_path && dump_path[0]) {
        FILE *dump = fopen(dump_path, "wb");
        CHECK(dump, "open output dump");
        CHECK(fwrite(hout, 1, sizeof(hout), dump) == sizeof(hout),
              "write output dump");
        CHECK(fclose(dump) == 0, "close output dump");
    }
    printf("test_rocm_q4k_decode_bench: stage_xq=%s avg_ms=%.6f "
           "output_fnv64=%016llx model_mib=%.2f\n",
           getenv("DS4_ROCM_Q4K_DECODE_STAGE_XQ") ?: "unset",
           elapsed_ms / (float)ITERS,
           (unsigned long long)fnv1a64(hout, sizeof(hout)),
           (double)model_bytes / (1024.0 * 1024.0));

    CHECK(hipEventDestroy(stop) == hipSuccess, "destroy stop event");
    CHECK(hipEventDestroy(start) == hipSuccess, "destroy start event");
    ds4_gpu_tensor_free_in_place(&add_in);
    ds4_gpu_tensor_free_in_place(&x);
    ds4_gpu_tensor_free_in_place(&weights);
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
