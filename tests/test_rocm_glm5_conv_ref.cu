#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <vector>

#include "ds4_gpu.h"

typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

#define CHECK(expr, message) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "FAIL %s\n", message); \
        return false; \
    } \
} while (0)

static void host_conv4(std::vector<float> &out,
                       std::vector<float> &history,
                       const std::vector<float> &input,
                       const std::vector<float> &weight,
                       uint32_t pos,
                       uint32_t n_tokens,
                       uint32_t channels) {
    for (uint32_t token = 0; token < n_tokens; ++token) {
        for (uint32_t channel = 0; channel < channels; ++channel) {
            const uint64_t h = (uint64_t)channel * 3u;
            const uint64_t w = (uint64_t)channel * 4u;
            const float current = input[(uint64_t)(pos + token) * channels + channel];
            const float raw = history[h] * weight[w] +
                              history[h + 1] * weight[w + 1] +
                              history[h + 2] * weight[w + 2] +
                              current * weight[w + 3];
            out[(uint64_t)(pos + token) * channels + channel] =
                raw / (1.0f + std::exp(-raw));
            history[h] = history[h + 1];
            history[h + 1] = history[h + 2];
            history[h + 2] = current;
        }
    }
}

static float max_error(const std::vector<float> &a,
                       const std::vector<float> &b) {
    if (a.size() != b.size()) return INFINITY;
    float error = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        error = std::max(error, std::fabs(a[i] - b[i]));
    return error;
}

static bool run_chunking_case(std::initializer_list<uint32_t> chunks,
                              bool nonzero_history) {
    constexpr uint32_t channels = 8192;
    uint32_t tokens = 0;
    for (uint32_t count : chunks) tokens += count;
    std::vector<float> input((size_t)tokens * channels);
    std::vector<float> weight((size_t)channels * 4u);
    std::vector<float> initial((size_t)channels * 3u);
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = 0.03f * float(int(i % 19u) - 9);
    for (size_t i = 0; i < weight.size(); ++i)
        weight[i] = 0.002f * float(int(i % 13u) - 6);
    if (nonzero_history) {
        for (size_t i = 0; i < initial.size(); ++i)
            initial[i] = 0.01f * float(int(i % 11u) - 5);
    }
    std::vector<float> expected(input.size());
    std::vector<float> expected_history = initial;
    host_conv4(expected, expected_history, input, weight, 0, tokens, channels);

    ds4_gpu_tensor *input_gpu = ds4_gpu_tensor_alloc(input.size() * sizeof(float));
    ds4_gpu_tensor *weight_gpu = ds4_gpu_tensor_alloc(weight.size() * sizeof(float));
    ds4_gpu_tensor *history_gpu = ds4_gpu_tensor_alloc(initial.size() * sizeof(float));
    ds4_gpu_tensor *output_gpu = ds4_gpu_tensor_alloc(expected.size() * sizeof(float));
    CHECK(input_gpu && weight_gpu && history_gpu && output_gpu,
          "allocate full-width conv tensors");
    CHECK(ds4_gpu_tensor_write(input_gpu, 0, input.data(), input.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(weight_gpu, 0, weight.data(), weight.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(history_gpu, 0, initial.data(), initial.size() * sizeof(float)),
          "upload conv inputs");

    uint32_t pos = 0;
    for (uint32_t count : chunks) {
        const uint64_t offset = (uint64_t)pos * channels * sizeof(float);
        const uint64_t bytes = (uint64_t)count * channels * sizeof(float);
        ds4_gpu_tensor *input_view = ds4_gpu_tensor_view(input_gpu, offset, bytes);
        ds4_gpu_tensor *output_view = ds4_gpu_tensor_view(output_gpu, offset, bytes);
        CHECK(input_view && output_view, "create conv chunk views");
        const int encoded = ds4_gpu_glm5_causal_conv4_tensor(
            output_view, history_gpu, input_view, weight_gpu, count, channels);
        ds4_gpu_tensor_free(output_view);
        ds4_gpu_tensor_free(input_view);
        CHECK(encoded, "encode conv chunk");
        pos += count;
    }
    CHECK(ds4_gpu_synchronize(), "synchronize conv chunks");
    std::vector<float> got(expected.size());
    std::vector<float> got_history(initial.size());
    CHECK(ds4_gpu_tensor_read(output_gpu, 0, got.data(), got.size() * sizeof(float)) &&
          ds4_gpu_tensor_read(history_gpu, 0, got_history.data(),
                              got_history.size() * sizeof(float)),
          "read conv outputs");
    const float output_error = max_error(expected, got);
    const float history_error = max_error(expected_history, got_history);
    ds4_gpu_tensor_free(output_gpu);
    ds4_gpu_tensor_free(history_gpu);
    ds4_gpu_tensor_free(weight_gpu);
    ds4_gpu_tensor_free(input_gpu);
    CHECK(output_error <= 2.0e-6f && history_error == 0.0f,
          "conv output/history match host oracle");
    std::fprintf(stderr,
                 "PASS GLM5 conv chunks=%zu nonzero=%d output_err=%.9g history_err=%.9g\n",
                 chunks.size(), nonzero_history ? 1 : 0,
                 output_error, history_error);
    return true;
}

static bool rejected_calls_preserve_history(void) {
    constexpr uint32_t channels = 8192;
    std::vector<float> history((size_t)channels * 3u);
    for (size_t i = 0; i < history.size(); ++i)
        history[i] = 0.01f * float(int(i % 7u) - 3);
    std::vector<float> input(channels, 0.25f);
    std::vector<float> weight((size_t)channels * 4u, 0.125f);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)channels * sizeof(float));
    ds4_gpu_tensor *in = ds4_gpu_tensor_alloc((uint64_t)channels * sizeof(float));
    ds4_gpu_tensor *w = ds4_gpu_tensor_alloc((uint64_t)channels * 4u * sizeof(float));
    ds4_gpu_tensor *h = ds4_gpu_tensor_alloc((uint64_t)channels * 3u * sizeof(float));
    ds4_gpu_tensor *short_out = ds4_gpu_tensor_alloc(
        (uint64_t)(channels - 1u) * sizeof(float));
    ds4_gpu_tensor *short_in = ds4_gpu_tensor_alloc(
        (uint64_t)(channels - 1u) * sizeof(float));
    ds4_gpu_tensor *short_w = ds4_gpu_tensor_alloc(
        ((uint64_t)channels * 4u - 1u) * sizeof(float));
    ds4_gpu_tensor *short_h = ds4_gpu_tensor_alloc(
        ((uint64_t)channels * 3u - 1u) * sizeof(float));
    CHECK(out && in && w && h && short_out && short_in && short_w && short_h,
          "allocate rejection tensors");
    CHECK(ds4_gpu_tensor_write(in, 0, input.data(), input.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(w, 0, weight.data(), weight.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(h, 0, history.data(), history.size() * sizeof(float)),
          "upload rejection tensors");
    CHECK(!ds4_gpu_glm5_causal_conv4_tensor(NULL, h, in, w, 1, channels),
          "reject null output");
    CHECK(!ds4_gpu_glm5_causal_conv4_tensor(out, h, in, w, 0, channels),
          "reject zero tokens");
    CHECK(!ds4_gpu_glm5_causal_conv4_tensor(out, h, in, w, 1, channels - 1u),
          "reject non-GLM width");
    CHECK(!ds4_gpu_glm5_causal_conv4_tensor(short_out, h, in, w, 1, channels),
          "reject short output");
    CHECK(!ds4_gpu_glm5_causal_conv4_tensor(out, h, short_in, w, 1, channels),
          "reject short input");
    CHECK(!ds4_gpu_glm5_causal_conv4_tensor(out, h, in, short_w, 1, channels),
          "reject short weight");
    CHECK(!ds4_gpu_glm5_causal_conv4_tensor(out, short_h, in, w, 1, channels),
          "reject short history");
    CHECK(!ds4_gpu_glm5_causal_conv4_tensor(h, h, in, w, 1, channels),
          "reject output/history alias");
    CHECK(ds4_gpu_synchronize(), "synchronize rejected conv calls");
    std::vector<float> after(history.size());
    CHECK(ds4_gpu_tensor_read(h, 0, after.data(), after.size() * sizeof(float)),
          "read rejection history");
    const bool preserved = max_error(history, after) == 0.0f;
    ds4_gpu_tensor_free(short_h);
    ds4_gpu_tensor_free(short_w);
    ds4_gpu_tensor_free(short_in);
    ds4_gpu_tensor_free(short_out);
    ds4_gpu_tensor_free(h);
    ds4_gpu_tensor_free(w);
    ds4_gpu_tensor_free(in);
    ds4_gpu_tensor_free(out);
    CHECK(preserved, "rejected calls preserve history");
    std::fprintf(stderr, "PASS GLM5 conv rejection preserves history\n");
    return true;
}

int main(void) {
    if (!ds4_gpu_init()) {
        std::fprintf(stderr, "FAIL initialize ROCm backend\n");
        return 1;
    }
    bool ok = true;
    for (uint32_t length : {1u, 2u, 3u, 127u, 128u, 129u}) {
        ok &= run_chunking_case({length}, false);
        ok &= run_chunking_case({length}, true);
    }
    ok &= run_chunking_case({1u, 128u}, true);
    ok &= run_chunking_case({2u, 127u}, true);
    ok &= run_chunking_case({3u, 126u}, true);
    ok &= run_chunking_case({127u, 1u, 1u}, true);
    ok &= rejected_calls_preserve_history();
    ds4_gpu_cleanup();
    return ok ? 0 : 1;
}
