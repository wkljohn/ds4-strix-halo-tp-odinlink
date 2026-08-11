/* Bit-exact oracle for compact-Q8 shared gate/up plus fused SwiGLU. */

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

    const uint64_t in_dim = 4096u;
    const uint64_t out_dim = 1024u;
    const uint64_t one_weight_bytes = out_dim * (in_dim / 32u) * 34u;
    const uint64_t model_size = 2u * one_weight_bytes;
    unsigned char *model = (unsigned char *)malloc((size_t)model_size);
    float *host_x = (float *)malloc((size_t)in_dim * sizeof(float));
    float *host_ref = (float *)malloc((size_t)out_dim * sizeof(float));
    float *host_got = (float *)malloc((size_t)out_dim * sizeof(float));
    CHECK(model && host_x && host_ref && host_got, "allocate host buffers");
    pack_q8(model, in_dim, out_dim, 11u);
    pack_q8(model + one_weight_bytes, in_dim, out_dim, 23u);
    for (uint64_t i = 0; i < in_dim; i++) {
        host_x[i] = (float)((int)(i % 53u) - 26) * 0.015625f;
    }
    CHECK(ds4_gpu_set_model_map(model, model_size), "install synthetic model");

    ds4_gpu_tensor x = {}, gate = {}, up = {}, ref = {}, got = {};
    CHECK(ds4_gpu_tensor_alloc_on(&x, 0, in_dim * sizeof(float)) == 0, "x");
    CHECK(ds4_gpu_tensor_alloc_on(&gate, 0, out_dim * sizeof(float)) == 0, "gate");
    CHECK(ds4_gpu_tensor_alloc_on(&up, 0, out_dim * sizeof(float)) == 0, "up");
    CHECK(ds4_gpu_tensor_alloc_on(&ref, 0, out_dim * sizeof(float)) == 0, "ref");
    CHECK(ds4_gpu_tensor_alloc_on(&got, 0, out_dim * sizeof(float)) == 0, "got");
    CHECK(ds4_gpu_tensor_write(&x, 0, host_x, in_dim * sizeof(float)), "upload x");

    const float clamp = 7.0f;
    CHECK(ds4_gpu_matmul_q8_0_pair_tensor(&gate, &up, model, model_size,
              0, one_weight_bytes, in_dim, out_dim, out_dim, &x, 1) &&
          ds4_gpu_swiglu_tensor(&ref, &gate, &up, (uint32_t)out_dim, clamp, 1.0f),
          "run reference path");
    CHECK(setenv("DS4_ROCM_SHARED_GU_SWIGLU_FUSE", "1", 1) == 0,
          "enable candidate");
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(&gate, &up, &got,
              model, model_size, 0, one_weight_bytes, in_dim, out_dim, &x,
              clamp) &&
          ds4_gpu_tensor_read(&ref, 0, host_ref, out_dim * sizeof(float)) &&
          ds4_gpu_tensor_read(&got, 0, host_got, out_dim * sizeof(float)),
          "run and read candidate");

    uint64_t mismatches = 0, first = 0;
    for (uint64_t i = 0; i < out_dim; i++) {
        if (memcmp(&host_ref[i], &host_got[i], sizeof(float)) != 0) {
            if (mismatches == 0) first = i;
            mismatches++;
        }
    }
    if (mismatches) {
        fprintf(stderr, "mismatches=%llu first=%llu ref=%a got=%a\n",
                (unsigned long long)mismatches, (unsigned long long)first,
                host_ref[first], host_got[first]);
    }
    CHECK(mismatches == 0, "fused mid output must be bit-exact");
    fprintf(stderr, "test_rocm_shared_gu_swiglu_fused: PASS (%llu floats)\n",
            (unsigned long long)out_dim);

    hipEvent_t start = nullptr, stop = nullptr;
    CHECK(hipEventCreate(&start) == hipSuccess &&
          hipEventCreate(&stop) == hipSuccess, "create timing events");
    const uint32_t warmup = 10u;
    const uint32_t iterations = 200u;
    for (uint32_t i = 0; i < warmup; i++) {
        CHECK(ds4_gpu_matmul_q8_0_pair_tensor(&gate, &up, model, model_size,
                  0, one_weight_bytes, in_dim, out_dim, out_dim, &x, 1) &&
              ds4_gpu_swiglu_tensor(&ref, &gate, &up, (uint32_t)out_dim,
                                    clamp, 1.0f), "warm reference");
        CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(&gate, &up, &got,
                  model, model_size, 0, one_weight_bytes, in_dim, out_dim, &x,
                  clamp), "warm candidate");
    }
    CHECK(hipDeviceSynchronize() == hipSuccess, "finish warmup");
    CHECK(hipEventRecord(start) == hipSuccess, "record reference start");
    for (uint32_t i = 0; i < iterations; i++) {
        CHECK(ds4_gpu_matmul_q8_0_pair_tensor(&gate, &up, model, model_size,
                  0, one_weight_bytes, in_dim, out_dim, out_dim, &x, 1) &&
              ds4_gpu_swiglu_tensor(&ref, &gate, &up, (uint32_t)out_dim,
                                    clamp, 1.0f), "time reference");
    }
    CHECK(hipEventRecord(stop) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess, "finish reference timing");
    float ref_ms = 0.0f;
    CHECK(hipEventElapsedTime(&ref_ms, start, stop) == hipSuccess,
          "read reference timing");
    CHECK(hipEventRecord(start) == hipSuccess, "record candidate start");
    for (uint32_t i = 0; i < iterations; i++) {
        CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(&gate, &up, &got,
                  model, model_size, 0, one_weight_bytes, in_dim, out_dim, &x,
                  clamp), "time candidate");
    }
    CHECK(hipEventRecord(stop) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess, "finish candidate timing");
    float fused_ms = 0.0f;
    CHECK(hipEventElapsedTime(&fused_ms, start, stop) == hipSuccess,
          "read candidate timing");
    fprintf(stderr,
            "test_rocm_shared_gu_swiglu_fused: reference=%.3f us "
            "fused=%.3f us change=%+.1f%%\n",
            (double)ref_ms * 1000.0 / iterations,
            (double)fused_ms * 1000.0 / iterations,
            100.0 * ((double)fused_ms / (double)ref_ms - 1.0));
    CHECK(hipEventDestroy(start) == hipSuccess &&
          hipEventDestroy(stop) == hipSuccess, "destroy timing events");

    ds4_gpu_tensor_free_in_place(&x);
    ds4_gpu_tensor_free_in_place(&gate);
    ds4_gpu_tensor_free_in_place(&up);
    ds4_gpu_tensor_free_in_place(&ref);
    ds4_gpu_tensor_free_in_place(&got);
    free(model);
    free(host_x);
    free(host_ref);
    free(host_got);
    ds4_gpu_cleanup();
    return 0;
}
