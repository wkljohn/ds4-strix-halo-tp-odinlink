/* Exactness and timing oracle for DSpark F16 indexer and HC projections. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg) do {                                                \
    if (!(cond)) {                                                           \
        fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);           \
        exit(1);                                                             \
    }                                                                        \
} while (0)

typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

static ds4_gpu_tensor tensor_row(ds4_gpu_tensor *base, uint32_t row,
                                 uint64_t elems_per_row) {
    ds4_gpu_tensor result = *base;
    result.ptr = (char *)base->ptr +
        (uint64_t)row * elems_per_row * sizeof(float);
    result.bytes = elems_per_row * sizeof(float);
    result.owner = 0;
    return result;
}

static int run_shape(uint32_t width, uint64_t in_dim, uint64_t out_dim,
                     uint32_t weight_sets) {
    const uint64_t weight_elems = in_dim * out_dim;
    const uint64_t one_weight_bytes = weight_elems * sizeof(uint16_t);
    const uint64_t model_bytes = one_weight_bytes * weight_sets;
    const uint64_t x_count = (uint64_t)width * in_dim;
    const uint64_t y_count = (uint64_t)width * out_dim;
    uint16_t *model = (uint16_t *)malloc((size_t)model_bytes);
    float *host_x = (float *)malloc(x_count * sizeof(float));
    float *host_ref = (float *)malloc(y_count * sizeof(float));
    float *host_got = (float *)malloc(y_count * sizeof(float));
    CHECK(model && host_x && host_ref && host_got, "allocate host buffers");
    for (uint64_t i = 0; i < weight_elems * weight_sets; i++) {
        const int v = (int)((i * 37u + (i >> 7u) * 13u + 19u) % 2047u) - 1023;
        model[i] = __half_as_ushort(__float2half((float)v * 0x1.0p-13f));
    }
    for (uint64_t i = 0; i < x_count; i++) {
        uint32_t bits = UINT32_C(0x3e800000) |
            ((uint32_t)(i * UINT64_C(2654435761) + 0x13579u) &
             UINT32_C(0x007fffff));
        if ((i * 17u + 5u) & 1u) bits |= UINT32_C(0x80000000);
        memcpy(host_x + i, &bits, sizeof(bits));
    }
    CHECK(ds4_gpu_set_model_map(model, model_bytes), "install model map");

    ds4_gpu_tensor x = {}, ref = {}, got = {};
    CHECK(ds4_gpu_tensor_alloc_on(&x, 0, x_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&ref, 0, y_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&got, 0, y_count * sizeof(float)) == 0,
          "allocate tensors");
    CHECK(ds4_gpu_tensor_write(&x, 0, host_x, x_count * sizeof(float)),
          "upload activations");
    for (uint32_t row = 0u; row < width; row++) {
        ds4_gpu_tensor xr = tensor_row(&x, row, in_dim);
        ds4_gpu_tensor yr = tensor_row(&ref, row, out_dim);
        CHECK(ds4_gpu_matmul_f16_tensor(&yr, model, model_bytes, 0u,
                                        in_dim, out_dim, &xr, 1u),
              "serial reference");
    }
    CHECK(ds4_gpu_matmul_f16_ordered_rows_exact_tensor(
              &got, model, model_bytes, 0u, in_dim, out_dim, &x, width),
          "ordered-row candidate");
    CHECK(ds4_gpu_tensor_read(&ref, 0, host_ref, y_count * sizeof(float)) &&
          ds4_gpu_tensor_read(&got, 0, host_got, y_count * sizeof(float)),
          "read outputs");
    uint64_t mismatches = 0u, first = 0u;
    for (uint64_t i = 0u; i < y_count; i++) {
        if (memcmp(host_ref + i, host_got + i, sizeof(float)) != 0) {
            if (mismatches == 0u) first = i;
            mismatches++;
        }
    }
    if (mismatches) {
        fprintf(stderr, "mismatch=%llu first=%llu ref=%a got=%a\n",
                (unsigned long long)mismatches,
                (unsigned long long)first, host_ref[first], host_got[first]);
    }
    CHECK(mismatches == 0u, "candidate must be bit-exact");

    hipEvent_t start = nullptr, stop = nullptr;
    CHECK(hipEventCreate(&start) == hipSuccess &&
          hipEventCreate(&stop) == hipSuccess, "create events");
    const uint32_t warmup = 2u, iterations = 32u;
    float serial_ms = 0.0f, batched_ms = 0.0f, candidate_ms = 0.0f;
    for (uint32_t phase = 0u; phase < 3u; phase++) {
        /* Phase 1 measures the production non-quality tiny-batch selector.
         * The exactness oracle and ordered serial reference remain in quality
         * mode so their FP32 reduction contract is unchanged. */
        ds4_gpu_set_quality(phase != 1u);
        for (uint32_t i = 0u; i < warmup + iterations; i++) {
            if (i == warmup) CHECK(hipEventRecord(start) == hipSuccess,
                                   "record start");
            const uint64_t weight_offset =
                (uint64_t)(i % weight_sets) * one_weight_bytes;
            if (phase == 0u) {
                for (uint32_t row = 0u; row < width; row++) {
                    ds4_gpu_tensor xr = tensor_row(&x, row, in_dim);
                    ds4_gpu_tensor yr = tensor_row(&ref, row, out_dim);
                    CHECK(ds4_gpu_matmul_f16_tensor(
                              &yr, model, model_bytes, weight_offset,
                              in_dim, out_dim, &xr, 1u), "timed reference");
                }
            } else if (phase == 1u) {
                CHECK(ds4_gpu_matmul_f16_tensor(
                          &got, model, model_bytes, weight_offset,
                          in_dim, out_dim, &x, width),
                      "timed production batch");
            } else {
                CHECK(ds4_gpu_matmul_f16_ordered_rows_exact_tensor(
                          &got, model, model_bytes, weight_offset,
                          in_dim, out_dim,
                          &x, width), "timed candidate");
            }
        }
        CHECK(hipEventRecord(stop) == hipSuccess &&
              hipEventSynchronize(stop) == hipSuccess, "finish timing");
        float elapsed = 0.0f;
        CHECK(hipEventElapsedTime(&elapsed, start, stop) == hipSuccess,
              "read timing");
        if (phase == 0u) serial_ms = elapsed / iterations;
        else if (phase == 1u) batched_ms = elapsed / iterations;
        else candidate_ms = elapsed / iterations;
    }
    ds4_gpu_set_quality(true);
    fprintf(stderr,
            "width=%u in=%llu out=%llu sets=%u serial_ms=%.6f "
            "batched_ms=%.6f candidate_ms=%.6f serial_speedup=%.3fx "
            "batch_speedup=%.3fx exact=yes\n",
            width, (unsigned long long)in_dim, (unsigned long long)out_dim,
            weight_sets,
            serial_ms, batched_ms, candidate_ms, serial_ms / candidate_ms,
            batched_ms / candidate_ms);

    (void)hipEventDestroy(stop);
    (void)hipEventDestroy(start);
    ds4_gpu_tensor_free_in_place(&got);
    ds4_gpu_tensor_free_in_place(&ref);
    ds4_gpu_tensor_free_in_place(&x);
    free(host_got);
    free(host_ref);
    free(host_x);
    free(model);
    return 1;
}

int main(int argc, char **argv) {
    CHECK(argc == 6, "usage: test token WIDTH IN_DIM OUT_DIM WEIGHT_SETS");
    CHECK(strcmp(argv[1], "token") == 0, "mode must be token");
    const uint32_t width = (uint32_t)strtoul(argv[2], NULL, 10);
    const uint64_t in_dim = strtoull(argv[3], NULL, 10);
    const uint64_t out_dim = strtoull(argv[4], NULL, 10);
    const uint32_t weight_sets = (uint32_t)strtoul(argv[5], NULL, 10);
    CHECK(width >= 2u && width <= 5u && in_dim != 0u && out_dim != 0u &&
          weight_sets != 0u, "valid shape arguments");
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm backend");
    /* The live exact verifier selects the ordered one-row fallback by
     * disabling arithmetic-changing shared-X shortcuts. */
    ds4_gpu_set_quality(true);
    CHECK(run_shape(width, in_dim, out_dim, weight_sets), "shape oracle");
    ds4_gpu_cleanup();
    return 0;
}
