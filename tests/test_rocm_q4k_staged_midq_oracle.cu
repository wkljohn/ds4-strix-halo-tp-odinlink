/* Production-shape gfx1151 oracle for staged-MIDQ one-token Q4_K down.
 *
 * Drives shipped ds4_gpu_routed_moe_one_tensor. Run in separate processes
 * with DS4_ROCM_Q4K_DECODE_STAGE_MIDQ=0/1 because the dispatcher resolves
 * that switch once per process. Makefile dumps three cases (unowned skip,
 * negative expert, fused addend) and cmp's them bitwise.
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

static int read_out(ds4_gpu_tensor *out, float *hout) {
    CHECK(hipDeviceSynchronize() == hipSuccess, "sync before read");
    CHECK(ds4_gpu_tensor_read(out, 0, hout, OUT_DIM * sizeof(float)),
          "read output");
    return 0;
}

int main(void) {
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm backend");
    CHECK(setenv("DS4_ROCM_Q4K_DECODE_STAGE_XQ", "1", 1) == 0,
          "stage XQ like production decode");

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

    const int32_t route_skip[N_USED] = {0, 1, 2, 0, 0, 0};
    const float weights_skip[N_USED] = {0.31f, 0.23f, 0.17f, 0.0f, 0.0f, 0.0f};
    const int32_t route_neg[N_USED] = {0, 1, 2, 3, 4, -1};
    const float weights_neg[N_USED] = {0.31f, 0.23f, 0.17f, 0.13f, 0.09f, 0.07f};
    float hx[IN_DIM], hadd[OUT_DIM];
    for (uint32_t i = 0; i < IN_DIM; i++)
        hx[i] = (float)((int)(i % 31u) - 15) * 0.03125f;
    for (uint32_t i = 0; i < OUT_DIM; i++)
        hadd[i] = (float)((int)(i % 19u) - 9) * 0.00390625f;
    CHECK(upload(&x, hx, sizeof(hx)), "upload input");
    CHECK(upload(&add_in, hadd, sizeof(hadd)), "upload addend");

    float hout_skip[OUT_DIM], hout_neg[OUT_DIM], hout_fuse[OUT_DIM];

    CHECK(setenv("DS4_ROCM_TP_SKIP_UNOWNED", "1", 1) == 0, "skip unowned");
    CHECK(unsetenv("DS4_ROCM_Q4K_DECODE_FUSE_ADDEND") == 0, "clear fuse");
    CHECK(upload(&selected, route_skip, sizeof(route_skip)), "upload skip route");
    CHECK(upload(&weights, weights_skip, sizeof(weights_skip)), "upload skip weights");
    for (uint32_t i = 0; i < WARMUP; i++) {
        CHECK(run_one(&out, &gate, &up, &mid, &down, &selected, &weights,
                      &x, &add_in, model, model_bytes, gate_off, up_off,
                      down_off, gate_expert_bytes, gate_row_bytes,
                      down_expert_bytes, down_row_bytes), "warmup skip");
    }
    CHECK(hipDeviceSynchronize() == hipSuccess, "sync warmup");

    hipEvent_t start = nullptr, stop = nullptr;
    CHECK(hipEventCreate(&start) == hipSuccess &&
          hipEventCreate(&stop) == hipSuccess, "create timing events");
    CHECK(hipEventRecord(start) == hipSuccess, "record timing start");
    for (uint32_t i = 0; i < ITERS; i++) {
        CHECK(run_one(&out, &gate, &up, &mid, &down, &selected, &weights,
                      &x, &add_in, model, model_bytes, gate_off, up_off,
                      down_off, gate_expert_bytes, gate_row_bytes,
                      down_expert_bytes, down_row_bytes), "timed skip");
    }
    CHECK(hipEventRecord(stop) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess, "finish timing");
    float elapsed_ms = 0.0f;
    CHECK(hipEventElapsedTime(&elapsed_ms, start, stop) == hipSuccess,
          "read timing");
    CHECK(read_out(&out, hout_skip) == 0, "read skip");

    CHECK(unsetenv("DS4_ROCM_TP_SKIP_UNOWNED") == 0, "clear skip");
    CHECK(upload(&selected, route_neg, sizeof(route_neg)), "upload neg route");
    CHECK(upload(&weights, weights_neg, sizeof(weights_neg)), "upload neg weights");
    CHECK(run_one(&out, &gate, &up, &mid, &down, &selected, &weights,
                  &x, &add_in, model, model_bytes, gate_off, up_off,
                  down_off, gate_expert_bytes, gate_row_bytes,
                  down_expert_bytes, down_row_bytes), "neg expert");
    CHECK(read_out(&out, hout_neg) == 0, "read neg");

    CHECK(setenv("DS4_ROCM_TP_SKIP_UNOWNED", "1", 1) == 0, "restore skip");
    CHECK(setenv("DS4_ROCM_Q4K_DECODE_FUSE_ADDEND", "1", 1) == 0, "fuse addend");
    CHECK(upload(&selected, route_skip, sizeof(route_skip)), "upload fuse route");
    CHECK(upload(&weights, weights_skip, sizeof(weights_skip)), "upload fuse weights");
    CHECK(run_one(&out, &gate, &up, &mid, &down, &selected, &weights,
                  &x, &add_in, model, model_bytes, gate_off, up_off,
                  down_off, gate_expert_bytes, gate_row_bytes,
                  down_expert_bytes, down_row_bytes), "fused addend");
    CHECK(read_out(&out, hout_fuse) == 0, "read fuse");

    const char *dump_path = getenv("DS4_TEST_OUTPUT_FILE");
    if (dump_path && dump_path[0]) {
        FILE *dump = fopen(dump_path, "wb");
        CHECK(dump, "open output dump");
        CHECK(fwrite(hout_skip, 1, sizeof(hout_skip), dump) == sizeof(hout_skip),
              "write skip dump");
        CHECK(fwrite(hout_neg, 1, sizeof(hout_neg), dump) == sizeof(hout_neg),
              "write neg dump");
        CHECK(fwrite(hout_fuse, 1, sizeof(hout_fuse), dump) == sizeof(hout_fuse),
              "write fuse dump");
        CHECK(fclose(dump) == 0, "close output dump");
    }

    printf("test_rocm_q4k_staged_midq_oracle: stage_midq=%s avg_ms=%.6f "
           "skip_fnv64=%016llx neg_fnv64=%016llx fuse_fnv64=%016llx "
           "model_mib=%.2f\n",
           getenv("DS4_ROCM_Q4K_DECODE_STAGE_MIDQ") ?: "unset",
           elapsed_ms / (float)ITERS,
           (unsigned long long)fnv1a64(hout_skip, sizeof(hout_skip)),
           (unsigned long long)fnv1a64(hout_neg, sizeof(hout_neg)),
           (unsigned long long)fnv1a64(hout_fuse, sizeof(hout_fuse)),
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
