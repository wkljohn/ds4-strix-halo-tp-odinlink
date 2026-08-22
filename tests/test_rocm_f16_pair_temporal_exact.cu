/* Exactness and timing oracle for DSpark verifier F16 compressor pairs. */

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
        return 1;                                                            \
    }                                                                        \
} while (0)

typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

enum { IN_DIM = 4096, MAX_ROWS = 5 };

static ds4_gpu_tensor tensor_rows(ds4_gpu_tensor *base, uint32_t row,
                                  uint32_t rows, uint64_t elems_per_row) {
    ds4_gpu_tensor result = *base;
    result.ptr = (char *)base->ptr +
        (uint64_t)row * elems_per_row * sizeof(float);
    result.bytes = (uint64_t)rows * elems_per_row * sizeof(float);
    result.owner = 0;
    return result;
}

static uint64_t mismatch_count(const float *a, const float *b,
                               uint64_t count, uint64_t *first) {
    uint64_t mismatches = 0;
    for (uint64_t i = 0; i < count; i++) {
        if (memcmp(a + i, b + i, sizeof(float)) != 0) {
            if (mismatches == 0) *first = i;
            mismatches++;
        }
    }
    return mismatches;
}

static int run_serial(ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
                      const void *model, uint64_t model_size,
                      uint64_t off0, uint64_t off1,
                      ds4_gpu_tensor *x, uint32_t width,
                      uint64_t out_dim) {
    for (uint32_t row = 0; row < width; row++) {
        ds4_gpu_tensor xr = tensor_rows(x, row, 1, IN_DIM);
        ds4_gpu_tensor y0 = tensor_rows(out0, row, 1, out_dim);
        ds4_gpu_tensor y1 = tensor_rows(out1, row, 1, out_dim);
        if (!ds4_gpu_matmul_f16_pair_tensor(
                &y0, &y1, model, model_size, off0, off1,
                IN_DIM, out_dim, &xr, 1)) return 0;
    }
    return 1;
}

static int run_temporal(ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
                        const void *model, uint64_t model_size,
                        uint64_t off0, uint64_t off1,
                        ds4_gpu_tensor *x, uint32_t width,
                        uint64_t out_dim) {
    uint32_t row = 0;
    while (width - row >= 2) {
        uint32_t rows = width - row;
        if (rows > 4) rows = 4;
        ds4_gpu_tensor xr = tensor_rows(x, row, rows, IN_DIM);
        ds4_gpu_tensor y0 = tensor_rows(out0, row, rows, out_dim);
        ds4_gpu_tensor y1 = tensor_rows(out1, row, rows, out_dim);
        if (!ds4_gpu_matmul_f16_pair_temporal_tensor(
                &y0, &y1, model, model_size, off0, off1,
                IN_DIM, out_dim, &xr, rows)) return 0;
        row += rows;
    }
    if (row < width) {
        ds4_gpu_tensor xr = tensor_rows(x, row, 1, IN_DIM);
        ds4_gpu_tensor y0 = tensor_rows(out0, row, 1, out_dim);
        ds4_gpu_tensor y1 = tensor_rows(out1, row, 1, out_dim);
        if (!ds4_gpu_matmul_f16_pair_tensor(
                &y0, &y1, model, model_size, off0, off1,
                IN_DIM, out_dim, &xr, 1)) return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    const uint32_t width = argc > 1 ?
        (uint32_t)strtoul(argv[1], NULL, 10) : MAX_ROWS;
    const uint64_t out_dim = argc > 2 ?
        (uint64_t)strtoull(argv[2], NULL, 10) : 1024u;
    CHECK(width >= 2 && width <= MAX_ROWS, "width must be 2..5");
    CHECK(out_dim == 128u || out_dim == 256u ||
          out_dim == 512u || out_dim == 1024u,
          "out_dim must be 128, 256, 512, or 1024");
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm backend");

    const uint64_t weight_elems = (uint64_t)IN_DIM * out_dim;
    const uint64_t weight_bytes = weight_elems * sizeof(uint16_t);
    const uint64_t model_size = 2 * weight_bytes;
    uint16_t *model = (uint16_t *)malloc((size_t)model_size);
    CHECK(model, "allocate synthetic model");
    for (uint64_t i = 0; i < 2 * weight_elems; i++) {
        const int v = (int)((i * 37u + (i >> 7u) * 13u + 19u) % 2047u) - 1023;
        const float f = (float)v * 0x1.0p-13f;
        model[i] = __half_as_ushort(__float2half(f));
    }
    CHECK(ds4_gpu_set_model_map(model, model_size), "install synthetic model");

    const uint64_t x_count = (uint64_t)width * IN_DIM;
    const uint64_t y_count = (uint64_t)width * out_dim;
    float *host_x = (float *)malloc(x_count * sizeof(float));
    float *host_ref = (float *)malloc(2 * y_count * sizeof(float));
    float *host_got = (float *)malloc(2 * y_count * sizeof(float));
    CHECK(host_x && host_ref && host_got, "allocate host buffers");
    for (uint64_t i = 0; i < x_count; i++) {
        uint32_t bits = UINT32_C(0x3e800000) |
            ((uint32_t)(i * UINT64_C(2654435761) + 0x13579u) &
             UINT32_C(0x007fffff));
        if ((i * 17u + 5u) & 1u) bits |= UINT32_C(0x80000000);
        memcpy(host_x + i, &bits, sizeof(bits));
    }

    ds4_gpu_tensor x = {}, ref0 = {}, ref1 = {}, got0 = {}, got1 = {};
    CHECK(ds4_gpu_tensor_alloc_on(&x, 0, x_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&ref0, 0, y_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&ref1, 0, y_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&got0, 0, y_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&got1, 0, y_count * sizeof(float)) == 0,
          "allocate tensors");
    CHECK(ds4_gpu_tensor_write(&x, 0, host_x, x_count * sizeof(float)),
          "upload activations");
    CHECK(run_serial(&ref0, &ref1, model, model_size, 0, weight_bytes,
                     &x, width, out_dim), "serial reference");
    CHECK(run_temporal(&got0, &got1, model, model_size, 0, weight_bytes,
                       &x, width, out_dim), "temporal candidate");
    CHECK(ds4_gpu_tensor_read(&ref0, 0, host_ref,
                              y_count * sizeof(float)) &&
          ds4_gpu_tensor_read(&ref1, 0, host_ref + y_count,
                              y_count * sizeof(float)) &&
          ds4_gpu_tensor_read(&got0, 0, host_got,
                              y_count * sizeof(float)) &&
          ds4_gpu_tensor_read(&got1, 0, host_got + y_count,
                              y_count * sizeof(float)), "read outputs");
    uint64_t first = 0;
    const uint64_t mismatches =
        mismatch_count(host_ref, host_got, 2 * y_count, &first);
    if (mismatches) {
        fprintf(stderr, "mismatch=%llu first=%llu ref=%a got=%a\n",
                (unsigned long long)mismatches,
                (unsigned long long)first,
                host_ref[first], host_got[first]);
    }
    CHECK(mismatches == 0, "temporal outputs must be bit-exact");

    hipEvent_t start = nullptr, stop = nullptr;
    CHECK(hipEventCreate(&start) == hipSuccess &&
          hipEventCreate(&stop) == hipSuccess, "create events");
    const uint32_t warmup = 5, iterations = 40;
    float serial_ms = 0.0f, temporal_ms = 0.0f;
    for (uint32_t phase = 0; phase < 2; phase++) {
        for (uint32_t i = 0; i < warmup + iterations; i++) {
            if (i == warmup) CHECK(hipEventRecord(start) == hipSuccess,
                                   "record start");
            CHECK(phase == 0 ?
                  run_serial(&ref0, &ref1, model, model_size, 0, weight_bytes,
                             &x, width, out_dim) :
                  run_temporal(&got0, &got1, model, model_size, 0, weight_bytes,
                               &x, width, out_dim), "timed projection");
        }
        CHECK(hipEventRecord(stop) == hipSuccess &&
              hipEventSynchronize(stop) == hipSuccess, "finish timing");
        float elapsed = 0.0f;
        CHECK(hipEventElapsedTime(&elapsed, start, stop) == hipSuccess,
              "read timing");
        if (phase == 0) serial_ms = elapsed / iterations;
        else temporal_ms = elapsed / iterations;
    }
    fprintf(stderr,
            "width=%u out_dim=%llu serial_ms=%.6f temporal_ms=%.6f "
            "speedup=%.3fx exact=yes\n",
            width, (unsigned long long)out_dim,
            serial_ms, temporal_ms, serial_ms / temporal_ms);

    (void)hipEventDestroy(stop);
    (void)hipEventDestroy(start);
    ds4_gpu_tensor_free_in_place(&got1);
    ds4_gpu_tensor_free_in_place(&got0);
    ds4_gpu_tensor_free_in_place(&ref1);
    ds4_gpu_tensor_free_in_place(&ref0);
    ds4_gpu_tensor_free_in_place(&x);
    ds4_gpu_cleanup();
    free(host_got);
    free(host_ref);
    free(host_x);
    free(model);
    return 0;
}
