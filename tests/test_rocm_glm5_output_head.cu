#include "ds4_glm5_next_exec.h"
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "tests/glm5_gguf_test.hpp"
#include "tests/glm5_next_real_offsets.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define CHECK(expr, message) do {                                           \
    if (!(expr)) {                                                          \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__);    \
        return false;                                                       \
    }                                                                       \
} while (0)

static float bf16_to_f32(uint16_t bits) {
    uint32_t word = (uint32_t)bits << 16u;
    float value = 0.0f;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

static uint64_t fnv64(const void *data, size_t bytes) {
    const unsigned char *p = static_cast<const unsigned char *>(data);
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static bool run_test(void) {
    constexpr uint32_t width = 4096u, hc = 4u, vocab = 154880u;
    const char *model = std::getenv("DS4_GLM5_MODEL");
    CHECK(model && model[0], "model environment");

    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM5 GGUF");
    ds4_glm5_next_model_offsets offsets = {};
    CHECK(glm5_next_bind_real_offsets(gguf, offsets),
          "bind validated real GLM5 offsets");
    uint32_t root_tensors = 0u;
    for (const auto &entry : gguf.tensors) {
        if (entry.first.rfind("blk.", 0u) == 0u) continue;
        CHECK(entry.first == "token_embd.weight" ||
              entry.first == "output_norm.weight" ||
              entry.first == "output.weight",
              "unknown root tensor cannot silently change the output rule");
        ++root_tensors;
    }
    CHECK(root_tensors == 3u,
          "output rule covers the complete root-tensor set");
    const uint64_t output_bytes = (uint64_t)vocab * width * sizeof(uint16_t);
    CHECK(offsets.output <= gguf.size &&
          output_bytes <= gguf.size - offsets.output,
          "complete BF16 output tensor lies inside the GGUF mapping");

    ds4_gpu_config config = {};
    config.n_gpus = 1u;
    config.device_indices[0] = 0u;
    CHECK(ds4_gpu_init_multi(&config) &&
          ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map) &&
          ds4_gpu_set_model_map(gguf.map, gguf.size),
          "initialize gfx1151 model mapping");

    ds4_glm5_next_workspace *workspace =
        ds4_glm5_next_workspace_create();
    ds4_gpu_tensor *hidden = ds4_gpu_tensor_alloc(
        (uint64_t)hc * width * sizeof(float));
    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(
        (uint64_t)vocab * sizeof(float));
    ds4_gpu_tensor *short_hidden = ds4_gpu_tensor_alloc(
        (uint64_t)hc * width * sizeof(float) - sizeof(float));
    ds4_gpu_tensor *short_logits = ds4_gpu_tensor_alloc(
        (uint64_t)vocab * sizeof(float) - sizeof(float));
    CHECK(workspace && hidden && logits && short_hidden && short_logits,
          "allocate output-head workspace");

    std::vector<float> host_hidden((size_t)hc * width);
    for (uint32_t stream = 0u; stream < hc; ++stream) {
        for (uint32_t col = 0u; col < width; ++col) {
            const int value = (int)((col * 37u + stream * 101u +
                                     (col >> 4u) * 13u) % 1021u) - 510;
            host_hidden[(uint64_t)stream * width + col] =
                (float)value / (2048.0f + 17.0f * stream);
        }
    }
    ds4_glm5_next_exec_ctx exec = {};
    exec.model_map = gguf.map;
    exec.model_size = gguf.size;
    exec.model = &offsets;
    CHECK(!ds4_glm5_next_output_logits(
              &exec, workspace, short_hidden, logits) &&
          !ds4_glm5_next_output_logits(
              &exec, workspace, hidden, short_logits),
          "undersized hidden and vocabulary buffers fail closed");
    CHECK(ds4_gpu_tensor_write(hidden, 0u, host_hidden.data(),
                               host_hidden.size() * sizeof(float)) &&
          ds4_glm5_next_output_logits(
              &exec, workspace, hidden, logits) &&
          ds4_gpu_synchronize(),
          "execute replicated BF16 output head");

    std::vector<float> got(vocab);
    CHECK(ds4_gpu_tensor_read(logits, 0u, got.data(),
                              got.size() * sizeof(float)),
          "read complete vocabulary logits");
    CHECK(ds4_gpu_tensor_fill_f32(
              logits, std::numeric_limits<float>::quiet_NaN(), vocab) &&
          ds4_glm5_next_output_logits(
              &exec, workspace, hidden, logits) &&
          ds4_gpu_synchronize(),
          "repeat complete vocabulary projection");
    std::vector<float> repeated(vocab);
    CHECK(ds4_gpu_tensor_read(logits, 0u, repeated.data(),
                              repeated.size() * sizeof(float)) &&
          std::memcmp(got.data(), repeated.data(),
                      got.size() * sizeof(float)) == 0,
          "output logits are bit-deterministic on repeat");

    const float *norm = reinterpret_cast<const float *>(
        reinterpret_cast<const char *>(gguf.map) + offsets.output_norm);
    const uint16_t *weights = reinterpret_cast<const uint16_t *>(
        reinterpret_cast<const char *>(gguf.map) + offsets.output);
    std::vector<float> normalized(width);
    double sum_sq = 0.0;
    for (uint32_t col = 0u; col < width; ++col) {
        float mean = 0.0f;
        for (uint32_t stream = 0u; stream < hc; ++stream)
            mean += host_hidden[(uint64_t)stream * width + col] * 0.25f;
        normalized[col] = mean;
        sum_sq += (double)mean * mean;
    }
    const float inv_rms = 1.0f /
        std::sqrt((float)(sum_sq / width) + 1.0e-5f);
    for (uint32_t col = 0u; col < width; ++col)
        normalized[col] *= inv_rms * norm[col];

    uint32_t cpu_argmax = 0u, gpu_argmax = 0u;
    float cpu_best = -std::numeric_limits<float>::infinity();
    float gpu_best = -std::numeric_limits<float>::infinity();
    double max_abs = 0.0, sum_abs = 0.0;
    uint64_t bad = 0u;
    for (uint32_t row = 0u; row < vocab; ++row) {
        float want = 0.0f;
        const uint16_t *weight_row = weights + (uint64_t)row * width;
        for (uint32_t col = 0u; col < width; ++col)
            want += bf16_to_f32(weight_row[col]) * normalized[col];
        const float have = got[row];
        const double delta = std::fabs((double)have - want);
        const double limit = 1.0e-4 + 2.0e-5 * std::fabs((double)want);
        max_abs = std::max(max_abs, delta);
        sum_abs += delta;
        if (!std::isfinite(have) || delta > limit) ++bad;
        if (want > cpu_best) { cpu_best = want; cpu_argmax = row; }
        if (have > gpu_best) { gpu_best = have; gpu_argmax = row; }
    }
    CHECK(bad == 0u, "full vocabulary matches independent BF16 CPU oracle");
    CHECK(cpu_argmax == gpu_argmax, "CPU/GPU argmax agreement");

    std::fprintf(stderr,
        "PASS GLM5 output head logits=%u argmax=%u hash=%016llx "
        "max_abs=%.9g mean_abs=%.9g\n",
        vocab, gpu_argmax,
        (unsigned long long)fnv64(got.data(), got.size() * sizeof(float)),
        max_abs, sum_abs / vocab);
    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(hidden);
    ds4_gpu_tensor_free(short_logits);
    ds4_gpu_tensor_free(short_hidden);
    ds4_glm5_next_workspace_destroy(workspace);
    ds4_gpu_cleanup();
    return true;
}

int main(void) { return run_test() ? 0 : 1; }
