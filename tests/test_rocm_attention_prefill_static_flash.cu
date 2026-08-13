/* Focused regression test for the opt-in single-pass static mixed-attention
 * prefill kernel.  The legacy two-pass kernel is the numerical oracle here;
 * the flash form may reorder FP32 arithmetic but must remain close, finite,
 * and exactly deterministic across identical launches. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define CHECK(cond, msg) do {                                                \
    if (!(cond)) {                                                           \
        fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);          \
        return 1;                                                            \
    }                                                                        \
} while (0)

typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

static double rel_rms(const std::vector<float> &a,
                      const std::vector<float> &b) {
    double diff2 = 0.0, ref2 = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = (double)a[i] - b[i];
        diff2 += d * d;
        ref2 += (double)b[i] * b[i];
    }
    return sqrt(diff2 / fmax(ref2, 1.0e-30));
}

int main(void) {
    constexpr uint32_t n_tokens = 32, n_comp = 8, window = 16;
    constexpr uint32_t ratio = 4, n_head = 16, head_dim = 512;
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");

    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm backend");

    std::vector<float> sinks(n_head);
    std::vector<float> q((size_t)n_tokens * n_head * head_dim);
    std::vector<float> raw((size_t)n_tokens * head_dim);
    std::vector<float> comp((size_t)n_comp * head_dim);
    for (uint32_t h = 0; h < n_head; h++) sinks[h] = -0.4f + 0.03f * h;
    for (size_t i = 0; i < q.size(); i++)
        q[i] = 0.09f * sinf((float)(i * 13u + 7u) * 0.0017f);
    for (size_t i = 0; i < raw.size(); i++)
        raw[i] = 0.23f * cosf((float)(i * 11u + 5u) * 0.0023f);
    for (size_t i = 0; i < comp.size(); i++)
        comp[i] = 0.19f * sinf((float)(i * 17u + 3u) * 0.0031f);

    CHECK(ds4_gpu_set_model_map(sinks.data(), sinks.size() * sizeof(float)),
          "install attention sinks");
    ds4_gpu_tensor q_dev = {}, raw_dev = {}, comp_dev = {}, heads_dev = {};
    CHECK(ds4_gpu_tensor_alloc_on(&q_dev, 0, q.size() * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&raw_dev, 0, raw.size() * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&comp_dev, 0, comp.size() * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&heads_dev, 0, q.size() * sizeof(float)) == 0,
          "allocate tensors");
    CHECK(ds4_gpu_tensor_write(&q_dev, 0, q.data(), q.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(&raw_dev, 0, raw.data(), raw.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(&comp_dev, 0, comp.data(), comp.size() * sizeof(float)),
          "upload tensors");

    std::vector<float> legacy(q.size()), flash1(q.size()), flash2(q.size());
    setenv("DS4_ROCM_ATTENTION_PREFILL_STATIC_FLASH", "0", 1);
    CHECK(ds4_gpu_attention_prefill_static_mixed_heads_tensor(
              &heads_dev, sinks.data(), sinks.size() * sizeof(float), 0,
              &q_dev, &raw_dev, &comp_dev, 0, n_tokens, n_comp, window,
              ratio, n_head, head_dim) &&
          ds4_gpu_tensor_read(&heads_dev, 0, legacy.data(), legacy.size() * sizeof(float)),
          "run legacy two-pass attention");

    setenv("DS4_ROCM_ATTENTION_PREFILL_STATIC_FLASH", "1", 1);
    CHECK(ds4_gpu_attention_prefill_static_mixed_heads_tensor(
              &heads_dev, sinks.data(), sinks.size() * sizeof(float), 0,
              &q_dev, &raw_dev, &comp_dev, 0, n_tokens, n_comp, window,
              ratio, n_head, head_dim) &&
          ds4_gpu_tensor_read(&heads_dev, 0, flash1.data(), flash1.size() * sizeof(float)),
          "run flash attention first time");
    CHECK(ds4_gpu_attention_prefill_static_mixed_heads_tensor(
              &heads_dev, sinks.data(), sinks.size() * sizeof(float), 0,
              &q_dev, &raw_dev, &comp_dev, 0, n_tokens, n_comp, window,
              ratio, n_head, head_dim) &&
          ds4_gpu_tensor_read(&heads_dev, 0, flash2.data(), flash2.size() * sizeof(float)),
          "run flash attention second time");

    CHECK(memcmp(flash1.data(), flash2.data(), flash1.size() * sizeof(float)) == 0,
          "flash attention must be bit-deterministic");
    for (float v : flash1) CHECK(isfinite(v), "flash output must be finite");
    const double error = rel_rms(flash1, legacy);
    fprintf(stderr, "test_rocm_attention_prefill_static_flash: rel_rms=%g\n", error);
    CHECK(error <= 2.0e-5, "flash attention must remain close to two-pass oracle");

    ds4_gpu_tensor_free_in_place(&q_dev);
    ds4_gpu_tensor_free_in_place(&raw_dev);
    ds4_gpu_tensor_free_in_place(&comp_dev);
    ds4_gpu_tensor_free_in_place(&heads_dev);
    ds4_gpu_cleanup();
    fprintf(stderr, "test_rocm_attention_prefill_static_flash: PASS\n");
    return 0;
}
