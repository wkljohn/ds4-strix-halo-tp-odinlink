/* Numerical and timing oracle for the compact-Q8 asymmetric pair projection. */

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

static void pack_q8(unsigned char *weights, uint64_t in_dim, uint64_t out_dim,
                    uint32_t salt) {
    const uint64_t blocks = in_dim / 32u;
    for (uint64_t row = 0; row < out_dim; row++) {
        for (uint64_t block = 0; block < blocks; block++) {
            unsigned char *dst = weights + (row * blocks + block) * 34u;
            dst[0] = 0x00u;
            dst[1] = 0x34u; /* fp16 0.25 */
            for (uint64_t lane = 0; lane < 32u; lane++) {
                const int value = (int)((row * 17u + block * 13u +
                    lane * 5u + salt) % 31u) - 15;
                dst[2u + lane] = (unsigned char)(int8_t)value;
            }
        }
    }
}

int main(void) {
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm backend");

    const uint64_t in_dim = 4096u, out0_dim = 1024u, out1_dim = 512u;
    const uint64_t row_bytes = (in_dim / 32u) * 34u;
    const uint64_t w0_bytes = out0_dim * row_bytes;
    const uint64_t w1_bytes = out1_dim * row_bytes;
    const uint64_t model_size = w0_bytes + w1_bytes;
    unsigned char *model = (unsigned char *)malloc((size_t)model_size);
    float *host_x = (float *)malloc((size_t)in_dim * sizeof(float));
    float *ref0 = (float *)malloc((size_t)out0_dim * sizeof(float));
    float *ref1 = (float *)malloc((size_t)out1_dim * sizeof(float));
    float *got0 = (float *)malloc((size_t)out0_dim * sizeof(float));
    float *got1 = (float *)malloc((size_t)out1_dim * sizeof(float));
    CHECK(model && host_x && ref0 && ref1 && got0 && got1,
          "allocate host buffers");
    pack_q8(model, in_dim, out0_dim, 11u);
    pack_q8(model + w0_bytes, in_dim, out1_dim, 23u);
    for (uint64_t i = 0; i < in_dim; i++)
        host_x[i] = (float)((int)(i % 53u) - 26) * 0.015625f;
    CHECK(ds4_gpu_set_model_map(model, model_size), "install synthetic model");

    ds4_gpu_tensor x = {}, out0 = {}, out1 = {};
    CHECK(ds4_gpu_tensor_alloc_on(&x, 0, in_dim * sizeof(float)) == 0, "x");
    CHECK(ds4_gpu_tensor_alloc_on(&out0, 0, out0_dim * sizeof(float)) == 0,
          "out0");
    CHECK(ds4_gpu_tensor_alloc_on(&out1, 0, out1_dim * sizeof(float)) == 0,
          "out1");
    CHECK(ds4_gpu_tensor_write(&x, 0, host_x, in_dim * sizeof(float)),
          "upload x");

    CHECK(setenv("DS4_ROCM_Q8_PAIR_F32_PACK4", "0", 1) == 0,
          "select reference");
    CHECK(ds4_gpu_matmul_q8_0_pair_tensor(&out0, &out1, model, model_size,
              0, w0_bytes, in_dim, out0_dim, out1_dim, &x, 1) &&
          ds4_gpu_tensor_read(&out0, 0, ref0, out0_dim * sizeof(float)) &&
          ds4_gpu_tensor_read(&out1, 0, ref1, out1_dim * sizeof(float)),
          "run reference");

    hipEvent_t start = nullptr, stop = nullptr;
    CHECK(hipEventCreate(&start) == hipSuccess &&
          hipEventCreate(&stop) == hipSuccess, "create timing events");
    const uint32_t warmup = 10u, iterations = 200u;
    for (uint32_t i = 0; i < warmup; i++)
        CHECK(ds4_gpu_matmul_q8_0_pair_tensor(&out0, &out1, model, model_size,
                  0, w0_bytes, in_dim, out0_dim, out1_dim, &x, 1),
              "warm reference");
    CHECK(hipDeviceSynchronize() == hipSuccess, "finish reference warmup");
    CHECK(hipEventRecord(start) == hipSuccess, "record reference start");
    for (uint32_t i = 0; i < iterations; i++)
        CHECK(ds4_gpu_matmul_q8_0_pair_tensor(&out0, &out1, model, model_size,
                  0, w0_bytes, in_dim, out0_dim, out1_dim, &x, 1),
              "time reference");
    CHECK(hipEventRecord(stop) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess, "finish reference timing");
    float reference_ms = 0.0f;
    CHECK(hipEventElapsedTime(&reference_ms, start, stop) == hipSuccess,
          "read reference timing");

    CHECK(setenv("DS4_ROCM_Q8_PAIR_F32_PACK4", "1", 1) == 0,
          "select candidate");
    CHECK(ds4_gpu_matmul_q8_0_pair_tensor(&out0, &out1, model, model_size,
              0, w0_bytes, in_dim, out0_dim, out1_dim, &x, 1) &&
          ds4_gpu_tensor_read(&out0, 0, got0, out0_dim * sizeof(float)) &&
          ds4_gpu_tensor_read(&out1, 0, got1, out1_dim * sizeof(float)),
          "run candidate");

    uint64_t different = 0;
    double max_abs = 0.0, sum_sq = 0.0, ref_sq = 0.0;
    for (uint64_t set = 0; set < 2; set++) {
        const uint64_t count = set ? out1_dim : out0_dim;
        const float *ref = set ? ref1 : ref0;
        const float *got = set ? got1 : got0;
        for (uint64_t i = 0; i < count; i++) {
            CHECK(isfinite(got[i]), "candidate output must be finite");
            const double delta = (double)got[i] - (double)ref[i];
            const double abs_delta = fabs(delta);
            if (memcmp(&got[i], &ref[i], sizeof(float)) != 0) different++;
            if (abs_delta > max_abs) max_abs = abs_delta;
            sum_sq += delta * delta;
            ref_sq += (double)ref[i] * (double)ref[i];
        }
    }
    const double nrmse = sqrt(sum_sq / ref_sq);
    fprintf(stderr, "test_rocm_q8_pair_pack4: different=%llu max_abs=%g "
            "nrmse=%g\n", (unsigned long long)different, max_abs, nrmse);
    CHECK(max_abs <= 0.001 && nrmse <= 1.0e-6,
          "candidate must remain numerically close");

    for (uint32_t i = 0; i < warmup; i++)
        CHECK(ds4_gpu_matmul_q8_0_pair_tensor(&out0, &out1, model, model_size,
                  0, w0_bytes, in_dim, out0_dim, out1_dim, &x, 1), "warm");
    CHECK(hipDeviceSynchronize() == hipSuccess, "finish warmup");
    CHECK(hipEventRecord(start) == hipSuccess, "record start");
    for (uint32_t i = 0; i < iterations; i++)
        CHECK(ds4_gpu_matmul_q8_0_pair_tensor(&out0, &out1, model, model_size,
                  0, w0_bytes, in_dim, out0_dim, out1_dim, &x, 1), "time");
    CHECK(hipEventRecord(stop) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess, "finish timing");
    float pack4_ms = 0.0f;
    CHECK(hipEventElapsedTime(&pack4_ms, start, stop) == hipSuccess,
          "read timing");
    fprintf(stderr, "test_rocm_q8_pair_pack4: reference=%.3f us "
            "candidate=%.3f us change=%+.1f%%\n",
            (double)reference_ms * 1000.0 / iterations,
            (double)pack4_ms * 1000.0 / iterations,
            100.0 * ((double)pack4_ms / (double)reference_ms - 1.0));

    ds4_gpu_tensor_free_in_place(&x);
    ds4_gpu_tensor_free_in_place(&out0);
    ds4_gpu_tensor_free_in_place(&out1);
    free(model); free(host_x); free(ref0); free(ref1); free(got0); free(got1);
    ds4_gpu_cleanup();
    return 0;
}
