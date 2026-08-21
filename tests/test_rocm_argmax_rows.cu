/* Exactness and production-shape speed gate for DSpark verifier row argmax. */
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

enum { N_VOCAB = 129280, MAX_ROWS = 4, WARMUP = 4, ITERS = 20 };

static int alloc_tensor(ds4_gpu_tensor *t, uint64_t bytes) {
    memset(t, 0, sizeof(*t));
    return ds4_gpu_tensor_alloc_on(t, 0, bytes) == 0;
}

static float value_at(uint64_t i, uint32_t row) {
    const uint32_t x = (uint32_t)((i * 1664525ull + 1013904223ull +
                                   (uint64_t)row * 2246822519ull) & 0x00ffffffu);
    return (float)((int32_t)x - 0x007fffff) / 8388607.0f;
}

static int run_width(ds4_gpu_tensor *reference,
                     ds4_gpu_tensor *candidate,
                     ds4_gpu_tensor *logits,
                     uint32_t rows) {
    uint32_t ref[MAX_ROWS] = {}, got[MAX_ROWS] = {};
    CHECK(ds4_gpu_indexer_topk_tensor(reference, logits, N_VOCAB, rows, 1u),
          "reference top-1");
    CHECK(ds4_gpu_argmax_rows_tensor(candidate, logits, N_VOCAB, rows),
          "batched argmax");
    CHECK(ds4_gpu_tensor_read(reference, 0, ref,
                              (uint64_t)rows * sizeof(ref[0])),
          "read reference");
    CHECK(ds4_gpu_tensor_read(candidate, 0, got,
                              (uint64_t)rows * sizeof(got[0])),
          "read candidate");
    CHECK(memcmp(ref, got, (uint64_t)rows * sizeof(ref[0])) == 0,
          "top-1 indices must match bitwise");

    for (uint32_t i = 0; i < WARMUP; i++) {
        CHECK(ds4_gpu_indexer_topk_tensor(reference, logits,
                                           N_VOCAB, rows, 1u),
              "warm reference");
        CHECK(ds4_gpu_argmax_rows_tensor(candidate, logits, N_VOCAB, rows),
              "warm candidate");
    }
    CHECK(hipDeviceSynchronize() == hipSuccess, "finish warmup");
    hipEvent_t start = nullptr, stop = nullptr;
    CHECK(hipEventCreate(&start) == hipSuccess &&
          hipEventCreate(&stop) == hipSuccess, "create timing events");
    CHECK(hipEventRecord(start) == hipSuccess, "reference start");
    for (uint32_t i = 0; i < ITERS; i++) {
        CHECK(ds4_gpu_indexer_topk_tensor(reference, logits,
                                           N_VOCAB, rows, 1u),
              "time reference");
    }
    CHECK(hipEventRecord(stop) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess, "reference stop");
    float reference_ms = 0.0f;
    CHECK(hipEventElapsedTime(&reference_ms, start, stop) == hipSuccess,
          "reference elapsed");
    CHECK(hipEventRecord(start) == hipSuccess, "candidate start");
    for (uint32_t i = 0; i < ITERS; i++) {
        CHECK(ds4_gpu_argmax_rows_tensor(candidate, logits, N_VOCAB, rows),
              "time candidate");
    }
    CHECK(hipEventRecord(stop) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess, "candidate stop");
    float candidate_ms = 0.0f;
    CHECK(hipEventElapsedTime(&candidate_ms, start, stop) == hipSuccess,
          "candidate elapsed");
    fprintf(stderr,
            "rows=%u serial_ms=%.6f batched_ms=%.6f speedup=%.3fx indices=",
            rows, reference_ms / ITERS, candidate_ms / ITERS,
            reference_ms / candidate_ms);
    for (uint32_t row = 0; row < rows; row++) {
        fprintf(stderr, "%s%u", row ? "," : "", got[row]);
    }
    fputc('\n', stderr);
    CHECK(reference_ms / candidate_ms >= 10.0f,
          "production-shape speedup must be at least 10x");
    CHECK(hipEventDestroy(stop) == hipSuccess &&
          hipEventDestroy(start) == hipSuccess, "destroy timing events");
    return 0;
}

int main(void) {
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm");

    const uint64_t count = (uint64_t)MAX_ROWS * N_VOCAB;
    float *host = (float *)malloc(count * sizeof(float));
    CHECK(host != nullptr, "allocate logits");
    for (uint32_t row = 0; row < MAX_ROWS; row++) {
        for (uint32_t col = 0; col < N_VOCAB; col++) {
            host[(uint64_t)row * N_VOCAB + col] = value_at(col, row);
        }
    }
    /* Exercise the strict-greater/lower-index tie contract and all-negative
     * input without introducing NaNs, which are rejected by model gates. */
    host[17] = 4.25f;
    host[10000] = 4.25f;
    host[(uint64_t)1 * N_VOCAB] = 5.0f;
    for (uint32_t col = 0; col < N_VOCAB; col++) {
        host[(uint64_t)2 * N_VOCAB + col] = -INFINITY;
    }
    host[(uint64_t)3 * N_VOCAB + 777] = 6.5f;
    host[(uint64_t)3 * N_VOCAB + 778] = 6.5f;

    ds4_gpu_tensor logits = {}, reference = {}, candidate = {};
    CHECK(alloc_tensor(&logits, count * sizeof(float)), "logits tensor");
    CHECK(alloc_tensor(&reference, MAX_ROWS * sizeof(uint32_t)),
          "reference indices");
    CHECK(alloc_tensor(&candidate, MAX_ROWS * sizeof(uint32_t)),
          "candidate indices");
    CHECK(ds4_gpu_tensor_write(&logits, 0, host, count * sizeof(float)),
          "upload logits");
    for (uint32_t rows = 1; rows <= MAX_ROWS; rows++) {
        CHECK(run_width(&reference, &candidate, &logits, rows) == 0,
              "width gate");
    }

    ds4_gpu_tensor_free_in_place(&candidate);
    ds4_gpu_tensor_free_in_place(&reference);
    ds4_gpu_tensor_free_in_place(&logits);
    free(host);
    ds4_gpu_cleanup();
    fprintf(stderr, "test_rocm_argmax_rows: PASS\n");
    return 0;
}
