#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "tests/glm5_gguf_test.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#define CHECK(expr, message) do {                                           \
    if (!(expr)) {                                                          \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__);    \
        return false;                                                       \
    }                                                                       \
} while (0)

typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

namespace {

constexpr uint32_t kLayer = 3;
constexpr uint32_t kInput = 4096;
constexpr uint32_t kExperts = 288;
constexpr uint32_t kUsed = 8;
constexpr uint32_t kTokens = 32;
constexpr float kScale = 2.5f;
static_assert(kExperts % 32u == 0u, "router expert count must tile wave32");
static_assert(kUsed < kExperts, "top-k boundary requires an unselected expert");

struct RuntimeGuard {
    bool active = false;
    ~RuntimeGuard() {
        if (active) ds4_gpu_cleanup();
    }
};

struct TensorSet {
    ds4_gpu_tensor input = {}, logits = {}, probs = {};
    ds4_gpu_tensor selected = {}, weights = {};
    ~TensorSet() {
        ds4_gpu_tensor_free_in_place(&weights);
        ds4_gpu_tensor_free_in_place(&selected);
        ds4_gpu_tensor_free_in_place(&probs);
        ds4_gpu_tensor_free_in_place(&logits);
        ds4_gpu_tensor_free_in_place(&input);
    }
};

bool alloc(ds4_gpu_tensor &tensor, uint64_t bytes) {
    std::memset(&tensor, 0, sizeof(tensor));
    return ds4_gpu_tensor_alloc_on(&tensor, 0, bytes) == 0;
}

bool upload(ds4_gpu_tensor &tensor, const void *source, uint64_t bytes) {
    return alloc(tensor, bytes) &&
           ds4_gpu_tensor_write(&tensor, 0, source, bytes) != 0;
}

uint64_t fnv1a64(const void *data, uint64_t bytes) {
    const auto *p = static_cast<const uint8_t *>(data);
    uint64_t hash = UINT64_C(1469598103934665603);
    for (uint64_t i = 0; i < bytes; ++i) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

float stable_sigmoid(float value) {
    if (value >= 0.0f) {
        const float exponential = std::exp(-value);
        return 1.0f / (1.0f + exponential);
    }
    const float exponential = std::exp(value);
    return exponential / (1.0f + exponential);
}

void cpu_router(const float *logits, const float *bias,
                int32_t *selected, float *weights, float *probs) {
    std::vector<uint32_t> order(kExperts);
    std::iota(order.begin(), order.end(), 0u);
    for (uint32_t expert = 0; expert < kExperts; ++expert)
        probs[expert] = stable_sigmoid(logits[expert]);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        const float av = probs[a] + bias[a];
        const float bv = probs[b] + bias[b];
        return av > bv || (av == bv && a < b);
    });
    float sum = 0.0f;
    for (uint32_t slot = 0; slot < kUsed; ++slot) {
        selected[slot] = (int32_t)order[slot];
        weights[slot] = probs[order[slot]];
        sum += weights[slot];
    }
    sum = std::max(sum, 6.103515625e-5f);
    for (uint32_t slot = 0; slot < kUsed; ++slot)
        weights[slot] = weights[slot] / sum * kScale;
}

struct ErrorStats {
    uint64_t count = 0, bad = 0, nonfinite = 0;
    double max_abs = 0.0, max_rel = 0.0, max_gate_ratio = 0.0;
    void add(float actual, float expected, double absolute, double relative) {
        ++count;
        if (!std::isfinite(actual) || !std::isfinite(expected)) {
            ++bad;
            ++nonfinite;
            return;
        }
        const double error = std::fabs((double)actual - expected);
        const double tolerance = absolute + relative * std::fabs((double)expected);
        max_abs = std::max(max_abs, error);
        max_rel = std::max(max_rel,
                           error / std::max(1.0e-12, std::fabs((double)expected)));
        max_gate_ratio = std::max(max_gate_ratio, error / tolerance);
        if (error > tolerance) ++bad;
    }
    bool pass() const { return count != 0 && bad == 0 && nonfinite == 0; }
};

bool run_test() {
    const char *model_path = std::getenv("DS4_GLM5_MODEL");
    CHECK(model_path && model_path[0], "DS4_GLM5_MODEL");
    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model_path), "open GLM5 GGUF");
    std::string architecture;
    float metadata_scale = 0.0f;
    bool metadata_norm = false;
    CHECK(gguf.metadata("general.architecture", architecture) &&
          architecture == "glm5-next" &&
          gguf.metadata("glm5-next.expert_weights_scale", metadata_scale) &&
          metadata_scale == kScale &&
          gguf.metadata("glm5-next.expert_weights_norm", metadata_norm) &&
          metadata_norm,
          "bind GLM5 top-8 normalization metadata");

    uint64_t router_offset = 0, bias_offset = 0;
    char router_name[64], bias_name[64];
    std::snprintf(router_name, sizeof(router_name),
                  "blk.%u.ffn_gate_inp.weight", kLayer);
    std::snprintf(bias_name, sizeof(bias_name),
                  "blk.%u.exp_probs_b.bias", kLayer);
    CHECK(gguf.tensor(router_name,
                      {kInput, kExperts}, 0u, router_offset) &&
          gguf.tensor(bias_name,
                      {kExperts}, 0u, bias_offset),
          "bind layer-3 F32 router and correction bias");
    const uint64_t router_bytes =
        (uint64_t)kExperts * kInput * sizeof(float);
    const uint64_t bias_bytes = (uint64_t)kExperts * sizeof(float);
    CHECK(router_offset <= gguf.size && router_bytes <= gguf.size - router_offset &&
          bias_offset <= gguf.size && bias_bytes <= gguf.size - bias_offset,
          "router payload ranges");
    const float *router = reinterpret_cast<const float *>(gguf.map + router_offset);
    const float *bias = reinterpret_cast<const float *>(gguf.map + bias_offset);

    uint32_t jitter_seed = 0;
    CHECK(glm5_test_router_seed(jitter_seed),
          "valid DS4_GLM5_ROUTER_JITTER_SEED");
    std::vector<float> input((size_t)kTokens * kInput);
    for (uint32_t token = 0; token < kTokens; ++token) {
        for (uint32_t column = 0; column < kInput; ++column) {
            input[(size_t)token * kInput + column] =
                glm5_test_router_input(token, column, jitter_seed);
        }
    }

    std::vector<float> cpu_logits((size_t)kTokens * kExperts);
    for (uint32_t token = 0; token < kTokens; ++token) {
        const float *x = input.data() + (size_t)token * kInput;
        for (uint32_t expert = 0; expert < kExperts; ++expert) {
            const float *row = router + (uint64_t)expert * kInput;
            double sum = 0.0;
            for (uint32_t column = 0; column < kInput; ++column)
                sum += (double)row[column] * x[column];
            cpu_logits[(size_t)token * kExperts + expert] = (float)sum;
        }
    }

    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "gfx1151 device available");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    RuntimeGuard runtime;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm backend");
    runtime.active = true;

    TensorSet tensors;
    const uint64_t input_bytes = (uint64_t)input.size() * sizeof(float);
    const uint64_t logits_bytes =
        (uint64_t)kTokens * kExperts * sizeof(float);
    const uint64_t selected_bytes =
        (uint64_t)kTokens * kUsed * sizeof(int32_t);
    const uint64_t weights_bytes =
        (uint64_t)kTokens * kUsed * sizeof(float);
    CHECK(upload(tensors.input, input.data(), input_bytes) &&
          alloc(tensors.logits, logits_bytes) &&
          alloc(tensors.probs, logits_bytes) &&
          alloc(tensors.selected, selected_bytes) &&
          alloc(tensors.weights, weights_bytes),
          "allocate router component tensors");
    CHECK(ds4_gpu_matmul_f32_tensor(
              &tensors.logits, gguf.map, gguf.size, router_offset,
              kInput, kExperts, &tensors.input, kTokens),
          "execute real-weight router F32 GEMM");
    CHECK(ds4_gpu_synchronize(), "synchronize real-weight router F32 GEMM");
    CHECK(ds4_gpu_glm_router_select_batch_tensor(
              &tensors.selected, &tensors.weights, &tensors.probs,
              gguf.map, gguf.size, bias_offset, &tensors.logits,
              kExperts, kUsed, kScale, kTokens),
          "execute GLM sigmoid/bias top-8 selection");
    CHECK(ds4_gpu_synchronize(), "synchronize GLM top-8 selection");

    std::vector<float> gpu_logits((size_t)kTokens * kExperts);
    std::vector<float> gpu_probs((size_t)kTokens * kExperts);
    std::vector<int32_t> gpu_selected((size_t)kTokens * kUsed);
    std::vector<float> gpu_weights((size_t)kTokens * kUsed);
    CHECK(ds4_gpu_tensor_read(&tensors.logits, 0, gpu_logits.data(), logits_bytes) &&
          ds4_gpu_tensor_read(&tensors.probs, 0, gpu_probs.data(), logits_bytes) &&
          ds4_gpu_tensor_read(&tensors.selected, 0, gpu_selected.data(), selected_bytes) &&
          ds4_gpu_tensor_read(&tensors.weights, 0, gpu_weights.data(), weights_bytes),
          "read router outputs");

    std::vector<float> repeat_logits(gpu_logits.size());
    std::vector<int32_t> repeat_selected(gpu_selected.size());
    std::vector<float> repeat_weights(gpu_weights.size());
    CHECK(ds4_gpu_matmul_f32_tensor(
              &tensors.logits, gguf.map, gguf.size, router_offset,
              kInput, kExperts, &tensors.input, kTokens) &&
          ds4_gpu_glm_router_select_batch_tensor(
              &tensors.selected, &tensors.weights, &tensors.probs,
              gguf.map, gguf.size, bias_offset, &tensors.logits,
              kExperts, kUsed, kScale, kTokens) &&
          ds4_gpu_synchronize() &&
          ds4_gpu_tensor_read(&tensors.logits, 0, repeat_logits.data(),
                              logits_bytes) &&
          ds4_gpu_tensor_read(&tensors.selected, 0, repeat_selected.data(),
                              selected_bytes) &&
          ds4_gpu_tensor_read(&tensors.weights, 0, repeat_weights.data(),
                              weights_bytes),
          "repeat router GEMM and selection");
    CHECK(std::memcmp(gpu_logits.data(), repeat_logits.data(),
                      logits_bytes) == 0 &&
          std::memcmp(gpu_selected.data(), repeat_selected.data(),
                      selected_bytes) == 0 &&
          std::memcmp(gpu_weights.data(), repeat_weights.data(),
                      weights_bytes) == 0,
          "bitwise deterministic repeated router GEMM and selection");

    std::vector<int32_t> cpu_selected(gpu_selected.size());
    std::vector<float> cpu_weights(gpu_weights.size());
    std::vector<float> cpu_probs(gpu_probs.size());
    ErrorStats logit_stats, probability_stats, weight_stats;
    uint64_t id_mismatch = 0, gpu_logit_route_mismatch = 0;
    uint64_t weight_sum_bad = 0;
    double max_weight_sum_error = 0.0;
    double min_choice_margin = INFINITY;
    for (uint32_t token = 0; token < kTokens; ++token) {
        cpu_router(cpu_logits.data() + (size_t)token * kExperts, bias,
                   cpu_selected.data() + (size_t)token * kUsed,
                   cpu_weights.data() + (size_t)token * kUsed,
                   cpu_probs.data() + (size_t)token * kExperts);
        std::vector<int32_t> gpu_logit_selected(kUsed);
        std::vector<float> gpu_logit_weights(kUsed), gpu_logit_probs(kExperts);
        cpu_router(gpu_logits.data() + (size_t)token * kExperts, bias,
                   gpu_logit_selected.data(), gpu_logit_weights.data(),
                   gpu_logit_probs.data());
        for (uint32_t slot = 0; slot < kUsed; ++slot) {
            id_mismatch += gpu_selected[(size_t)token * kUsed + slot] !=
                           cpu_selected[(size_t)token * kUsed + slot];
            gpu_logit_route_mismatch +=
                gpu_selected[(size_t)token * kUsed + slot] !=
                gpu_logit_selected[slot];
            weight_stats.add(gpu_weights[(size_t)token * kUsed + slot],
                             cpu_weights[(size_t)token * kUsed + slot],
                             2.0e-6, 2.0e-5);
        }
        std::vector<float> choices(kExperts);
        for (uint32_t expert = 0; expert < kExperts; ++expert) {
            choices[expert] = cpu_probs[(size_t)token * kExperts + expert] +
                              bias[expert];
            logit_stats.add(gpu_logits[(size_t)token * kExperts + expert],
                            cpu_logits[(size_t)token * kExperts + expert],
                            2.0e-5, 2.0e-5);
            probability_stats.add(gpu_probs[(size_t)token * kExperts + expert],
                                  cpu_probs[(size_t)token * kExperts + expert],
                                  2.0e-6, 2.0e-5);
        }
        std::sort(choices.begin(), choices.end(), std::greater<float>());
        min_choice_margin = std::min(
            min_choice_margin, (double)choices[kUsed - 1u] - choices[kUsed]);
        double weight_sum = 0.0;
        for (uint32_t slot = 0; slot < kUsed; ++slot)
            weight_sum += gpu_weights[(size_t)token * kUsed + slot];
        const double weight_sum_error = std::fabs(weight_sum - kScale);
        max_weight_sum_error = std::max(max_weight_sum_error,
                                        weight_sum_error);
        weight_sum_bad += weight_sum_error > 2.0e-6;
    }

    CHECK(ds4_gpu_glm_router_select_batch_tensor(
              &tensors.selected, &tensors.weights, &tensors.probs,
              gguf.map, gguf.size, bias_offset, &tensors.logits,
              287u, kUsed, kScale, kTokens) == 0 &&
          ds4_gpu_glm_router_select_batch_tensor(
              &tensors.selected, &tensors.weights, &tensors.probs,
              gguf.map, gguf.size, bias_offset, &tensors.logits,
              kExperts, 9u, kScale, kTokens) == 0 &&
          ds4_gpu_glm_router_select_batch_tensor(
              &tensors.selected, &tensors.weights, &tensors.probs,
              gguf.map, gguf.size, gguf.size - sizeof(float), &tensors.logits,
              kExperts, kUsed, kScale, kTokens) == 0,
          "router contract fails closed for wrong expert geometry and bias range");
    CHECK(ds4_gpu_glm_router_select_batch_tensor(
              &tensors.selected, &tensors.weights, &tensors.probs,
              gguf.map, gguf.size, bias_offset, &tensors.logits,
              384u, kUsed, kScale, kTokens) == 0 &&
          ds4_gpu_glm_router_select_batch_tensor(
              &tensors.selected, &tensors.weights, &tensors.probs,
              gguf.map, gguf.size, bias_offset, &tensors.logits,
              kExperts, kUsed, std::numeric_limits<float>::quiet_NaN(),
              kTokens) == 0,
          "router rejects whitelisted geometry with undersized tensors and NaN scale");

    std::fprintf(stderr,
        "GLM5 same-GGUF router layer=%u tokens=%u experts=%u topk=%u seed=%u "
        "id_mismatch=%llu gpu_logit_route_mismatch=%llu "
        "min_choice_margin=%.9g "
        "logits_bad=%llu max_abs=%.9g max_gate_ratio=%.9g "
        "probs_bad=%llu max_abs=%.9g max_gate_ratio=%.9g "
        "weights_bad=%llu max_abs=%.9g max_gate_ratio=%.9g "
        "weight_sum_bad=%llu max_weight_sum_error=%.9g "
        "router_fnv=%016llx bias_fnv=%016llx token0_ids_fnv=%016llx "
        "selected_fnv=%016llx "
        "weights_fnv=%016llx\n",
        kLayer, kTokens, kExperts, kUsed, jitter_seed,
        (unsigned long long)id_mismatch,
        (unsigned long long)gpu_logit_route_mismatch,
        min_choice_margin,
        (unsigned long long)logit_stats.bad,
        logit_stats.max_abs, logit_stats.max_gate_ratio,
        (unsigned long long)probability_stats.bad,
        probability_stats.max_abs, probability_stats.max_gate_ratio,
        (unsigned long long)weight_stats.bad,
        weight_stats.max_abs, weight_stats.max_gate_ratio,
        (unsigned long long)weight_sum_bad, max_weight_sum_error,
        (unsigned long long)fnv1a64(router, router_bytes),
        (unsigned long long)fnv1a64(bias, bias_bytes),
        (unsigned long long)fnv1a64(gpu_selected.data(),
                                    kUsed * sizeof(int32_t)),
        (unsigned long long)fnv1a64(gpu_selected.data(), selected_bytes),
        (unsigned long long)fnv1a64(gpu_weights.data(), weights_bytes));
    CHECK(logit_stats.pass() && probability_stats.pass() && weight_stats.pass(),
          "bounded independent router numerical reference");
    CHECK(id_mismatch == 0 && gpu_logit_route_mismatch == 0,
          "exact deterministic top-8 router IDs");
    CHECK(weight_sum_bad == 0,
          "top-8 expert weights sum to GGUF-configured scale");
    CHECK(min_choice_margin > std::max(
              1.0e-5, 8.0 * (probability_stats.max_abs + 2.0e-6)),
          "router top-8 boundary has useful numerical margin");
    std::fprintf(stderr,
                 "PASS same-GGUF GLM5 F32 router and deterministic top-8\n");
    return true;
}

}  // namespace

int main() { return run_test() ? 0 : 1; }
