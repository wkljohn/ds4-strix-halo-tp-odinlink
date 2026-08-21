/* Exact width-2 through width-5 oracle for the TP Q8_0 weight-outer path. */
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
    N_GROUP = 8,
    OWNED_GROUP = 4,
    GROUP_DIM = 4096,
    RANK = 1024,
    OUT_DIM = 4096,
    QK = 32,
    Q8_BLOCK_BYTES = 34,
    WARMUP = 3,
    ITERS = 16,
};

static void pack_q8(unsigned char *dst, uint64_t blocks, uint32_t salt) {
    for (uint64_t b = 0; b < blocks; b++) {
        unsigned char *p = dst + b * Q8_BLOCK_BYTES;
        /* Non-power-of-two IEEE binary16 scales exercise the scale rounding
         * point instead of letting an exact exponent shift hide a changed
         * accumulator DAG. */
        static const uint16_t scales[8] = {
            0x3155u, 0x3555u, 0x3966u, 0x3a66u,
            0x3d33u, 0x3eabu, 0x40cdu, 0x41b3u,
        };
        const uint16_t d = scales[(b + salt) & 7u];
        p[0] = (unsigned char)(d & 0xffu);
        p[1] = (unsigned char)(d >> 8);
        for (uint32_t i = 0; i < QK; i++) {
            p[2u + i] = (unsigned char)(int8_t)
                ((int)((b * 13u + i * 7u + salt) % 31u) - 15);
        }
    }
}

static int alloc_tensor(ds4_gpu_tensor *t, uint64_t bytes) {
    memset(t, 0, sizeof(*t));
    return ds4_gpu_tensor_alloc_on(t, 0, bytes) == 0;
}

static int run_projection(ds4_gpu_tensor *out, ds4_gpu_tensor *low,
                          const void *model, uint64_t model_bytes,
                          uint64_t out_a_offset, uint64_t out_b_offset,
                          const ds4_gpu_tensor *heads, uint32_t group0,
                          uint32_t n_tokens) {
    return ds4_gpu_attention_output_q8_tp_batch_tensor(
        out, low, model, model_bytes, out_a_offset, out_b_offset,
        GROUP_DIM, RANK, N_GROUP, group0, OWNED_GROUP, OUT_DIM,
        heads, n_tokens);
}

static int compare_exact(const char *name, const float *ref,
                         const float *got, uint64_t n) {
    uint64_t mismatch = 0;
    double max_abs = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        if (memcmp(ref + i, got + i, sizeof(float)) != 0) mismatch++;
        const double d = fabs((double)ref[i] - (double)got[i]);
        if (d > max_abs) max_abs = d;
    }
    fprintf(stderr, "%s mismatch=%llu/%llu max_abs=%.9g\n", name,
            (unsigned long long)mismatch, (unsigned long long)n, max_abs);
    return mismatch == 0;
}

static void report_token_mismatches(const char *name, const float *ref,
                                    const float *got, uint64_t row_elems,
                                    uint32_t n_tokens) {
    for (uint32_t t = 0; t < n_tokens; t++) {
        uint64_t mismatch = 0;
        uint64_t parity_mismatch[2] = {0, 0};
        double max_abs = 0.0;
        for (uint64_t i = 0; i < row_elems; i++) {
            const uint64_t j = (uint64_t)t * row_elems + i;
            if (memcmp(ref + j, got + j, sizeof(float)) != 0) {
                mismatch++;
                parity_mismatch[i & 1u]++;
            }
            const double d = fabs((double)ref[j] - (double)got[j]);
            if (d > max_abs) max_abs = d;
        }
        fprintf(stderr, "%s token=%u mismatch=%llu/%llu even=%llu odd=%llu "
                "max_abs=%.9g\n",
                name, t, (unsigned long long)mismatch,
                (unsigned long long)row_elems,
                (unsigned long long)parity_mismatch[0],
                (unsigned long long)parity_mismatch[1], max_abs);
    }
}

int main(int argc, char **argv) {
    const uint32_t n_tokens = argc > 1 ? (uint32_t)atoi(argv[1]) : 2u;
    CHECK(n_tokens >= 2u && n_tokens <= 5u,
          "usage: test_rocm_q8_attn_out_weight_outer [2-5]");
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm");

    const uint64_t blocks_a = GROUP_DIM / QK;
    const uint64_t blocks_b = (uint64_t)N_GROUP * RANK / QK;
    const uint64_t out_a_blocks = (uint64_t)N_GROUP * RANK * blocks_a;
    const uint64_t out_b_blocks = (uint64_t)OUT_DIM * blocks_b;
    const uint64_t out_a_bytes = out_a_blocks * Q8_BLOCK_BYTES;
    const uint64_t out_b_bytes = out_b_blocks * Q8_BLOCK_BYTES;
    const uint64_t out_a_offset = 0;
    const uint64_t out_b_offset = out_a_bytes;
    const uint64_t model_bytes = out_a_bytes + out_b_bytes;
    unsigned char *model = (unsigned char *)malloc((size_t)model_bytes);
    CHECK(model, "allocate synthetic Q8 model");
    pack_q8(model + out_a_offset, out_a_blocks, 11u);
    pack_q8(model + out_b_offset, out_b_blocks, 29u);
    CHECK(ds4_gpu_set_model_map(model, model_bytes), "install Q8 model map");

    const uint64_t heads_count =
        (uint64_t)n_tokens * N_GROUP * GROUP_DIM;
    const uint64_t low_count =
        (uint64_t)n_tokens * OWNED_GROUP * RANK;
    const uint64_t out_count = (uint64_t)n_tokens * OUT_DIM;
    float *heads_host = (float *)malloc(heads_count * sizeof(float));
    float *low_ref = (float *)malloc(low_count * sizeof(float));
    float *low_got = (float *)malloc(low_count * sizeof(float));
    float *out_ref = (float *)malloc(out_count * sizeof(float));
    float *out_got = (float *)malloc(out_count * sizeof(float));
    CHECK(heads_host && low_ref && low_got && out_ref && out_got,
          "allocate host oracle buffers");
    for (uint64_t i = 0; i < heads_count; i++) {
        const int32_t a = (int32_t)((i * 17u + (i / GROUP_DIM) * 23u) % 257u);
        heads_host[i] = (float)(a - 128) / 257.0f +
                        (float)((i * 19u) % 37u) / 100003.0f;
    }

    ds4_gpu_tensor heads = {}, serial_low = {}, candidate_low = {};
    ds4_gpu_tensor serial_out = {}, candidate_out = {};
    CHECK(alloc_tensor(&heads, heads_count * sizeof(float)), "heads tensor");
    CHECK(alloc_tensor(&serial_low, low_count * sizeof(float)), "serial low");
    CHECK(alloc_tensor(&candidate_low, low_count * sizeof(float)),
          "candidate low");
    CHECK(alloc_tensor(&serial_out, out_count * sizeof(float)), "serial out");
    CHECK(alloc_tensor(&candidate_out, out_count * sizeof(float)),
          "candidate out");
    CHECK(ds4_gpu_tensor_write(&heads, 0, heads_host,
                               heads_count * sizeof(float)), "upload heads");

    for (uint32_t group0 = 0; group0 <= 4u; group0 += 4u) {
        CHECK(unsetenv("DS4_ROCM_Q8_ATTN_OUT_WEIGHT_OUTER") == 0,
              "select serial oracle");
        CHECK(run_projection(&serial_out, &serial_low, model, model_bytes,
                             out_a_offset, out_b_offset, &heads, group0,
                             n_tokens),
              "run serial projection");
        CHECK(ds4_gpu_tensor_read(&serial_low, 0, low_ref,
                                  low_count * sizeof(float)), "read serial low");
        CHECK(ds4_gpu_tensor_read(&serial_out, 0, out_ref,
                                  out_count * sizeof(float)), "read serial out");

        CHECK(setenv("DS4_ROCM_Q8_ATTN_OUT_WEIGHT_OUTER", "1", 1) == 0,
              "select weight-outer candidate");
        CHECK(run_projection(&candidate_out, &candidate_low, model, model_bytes,
                             out_a_offset, out_b_offset, &heads, group0,
                             n_tokens),
              "run weight-outer projection");
        CHECK(ds4_gpu_tensor_read(&candidate_low, 0, low_got,
                                  low_count * sizeof(float)),
              "read candidate low");
        CHECK(ds4_gpu_tensor_read(&candidate_out, 0, out_got,
                                  out_count * sizeof(float)),
              "read candidate out");
        char label[64];
        snprintf(label, sizeof(label), "rank%u low", group0 / 4u);
        report_token_mismatches(label, low_ref, low_got,
                                (uint64_t)OWNED_GROUP * RANK, n_tokens);
        CHECK(compare_exact(label, low_ref, low_got, low_count),
              "low projection must be bit-exact");
        snprintf(label, sizeof(label), "rank%u out", group0 / 4u);
        report_token_mismatches(label, out_ref, out_got, OUT_DIM, n_tokens);
        CHECK(compare_exact(label, out_ref, out_got, out_count),
              "output projection must be bit-exact");
    }

    hipEvent_t start = nullptr, stop = nullptr;
    CHECK(hipEventCreate(&start) == hipSuccess &&
          hipEventCreate(&stop) == hipSuccess, "create timing events");
    CHECK(unsetenv("DS4_ROCM_Q8_ATTN_OUT_WEIGHT_OUTER") == 0,
          "select serial timing");
    for (uint32_t i = 0; i < WARMUP; i++)
        CHECK(run_projection(&serial_out, &serial_low, model, model_bytes,
                             out_a_offset, out_b_offset, &heads, 0u,
                             n_tokens),
              "warm serial");
    CHECK(hipDeviceSynchronize() == hipSuccess, "finish serial warmup");
    CHECK(hipEventRecord(start) == hipSuccess, "serial timer start");
    for (uint32_t i = 0; i < ITERS; i++)
        CHECK(run_projection(&serial_out, &serial_low, model, model_bytes,
                             out_a_offset, out_b_offset, &heads, 0u,
                             n_tokens),
              "time serial");
    CHECK(hipEventRecord(stop) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess, "serial timer stop");
    float serial_ms = 0.0f;
    CHECK(hipEventElapsedTime(&serial_ms, start, stop) == hipSuccess,
          "read serial timer");

    CHECK(setenv("DS4_ROCM_Q8_ATTN_OUT_WEIGHT_OUTER", "1", 1) == 0,
          "select candidate timing");
    for (uint32_t i = 0; i < WARMUP; i++)
        CHECK(run_projection(&candidate_out, &candidate_low, model, model_bytes,
                             out_a_offset, out_b_offset, &heads, 0u,
                             n_tokens),
              "warm candidate");
    CHECK(hipDeviceSynchronize() == hipSuccess, "finish candidate warmup");
    CHECK(hipEventRecord(start) == hipSuccess, "candidate timer start");
    for (uint32_t i = 0; i < ITERS; i++)
        CHECK(run_projection(&candidate_out, &candidate_low, model, model_bytes,
                             out_a_offset, out_b_offset, &heads, 0u,
                             n_tokens),
              "time candidate");
    CHECK(hipEventRecord(stop) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess, "candidate timer stop");
    float candidate_ms = 0.0f;
    CHECK(hipEventElapsedTime(&candidate_ms, start, stop) == hipSuccess,
          "read candidate timer");
    fprintf(stderr,
            "width=%u serial_ms=%.3f candidate_ms=%.3f speedup=%.3fx\n",
            n_tokens, serial_ms / ITERS, candidate_ms / ITERS,
            serial_ms / candidate_ms);

    CHECK(hipEventDestroy(stop) == hipSuccess, "destroy stop event");
    CHECK(hipEventDestroy(start) == hipSuccess, "destroy start event");
    ds4_gpu_tensor_free_in_place(&candidate_out);
    ds4_gpu_tensor_free_in_place(&serial_out);
    ds4_gpu_tensor_free_in_place(&candidate_low);
    ds4_gpu_tensor_free_in_place(&serial_low);
    ds4_gpu_tensor_free_in_place(&heads);
    free(out_got);
    free(out_ref);
    free(low_got);
    free(low_ref);
    free(heads_host);
    free(model);
    ds4_gpu_cleanup();
    fprintf(stderr, "test_rocm_q8_attn_out_weight_outer: PASS\n");
    return 0;
}
