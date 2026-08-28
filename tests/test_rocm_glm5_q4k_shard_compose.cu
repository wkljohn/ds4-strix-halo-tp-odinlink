#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "tests/glm5_gguf_test.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(expr, message) do {                                           \
    if (!(expr)) {                                                          \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__);    \
        return false;                                                       \
    }                                                                       \
} while (0)

typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}
extern "C" int ds4_gpu_routed_moe_batch_q4k_direct_control(
        ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *,
        ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, const void *,
        const void *, uint64_t, uint64_t, uint64_t, uint64_t,
        const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t,
        float, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t);
extern "C" int ds4_gpu_q8k_quantize_research_control(
        ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t);

namespace {

constexpr uint32_t kInput = 4096;
constexpr uint32_t kFullMid = 2048;
constexpr uint32_t kHalfMid = 1024;
constexpr uint32_t kOutput = 4096;
constexpr uint32_t kTotalExperts = 288;
constexpr uint32_t kUsed = 8;
// Shipping Q4_K prefill selects the sorted WMMA path at 32 tokens.
constexpr uint32_t kTokens = 32;
constexpr uint32_t kQ4Block = 144;
constexpr uint32_t kQk = 256;
constexpr float kClamp = 10.0f;
constexpr uint32_t kExpertIds[kUsed] = {0, 1, 17, 63, 127, 191, 255, 287};
static_assert((uint64_t)kTokens * kOutput <= UINT32_MAX,
              "ds4_gpu_add_tensor element count must fit uint32_t");

struct Q4KBlock {
    uint16_t d, dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct Q8KBlock {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};

struct Q81Comparison {
    uint64_t changed_values = 0;
    uint64_t changed_scales = 0;
    uint64_t changed_sums = 0;
    float max_scale_delta = 0.0f;
};

static_assert(sizeof(Q4KBlock) == kQ4Block, "Q4_K layout drift");
static_assert(sizeof(Q8KBlock) == 292, "Q8_K layout drift");

struct RuntimeGuard {
    bool active = false;
    ~RuntimeGuard() {
        if (active) ds4_gpu_cleanup();
    }
};

struct DeviceWeights {
    void *gate_full = nullptr;
    void *up_full = nullptr;
    void *down_full = nullptr;
    void *gate_half[2] = {};
    void *up_half[2] = {};
    void *down_half[2] = {};

    ~DeviceWeights() {
        auto release = [](void *ptr, const char *name) {
            if (!ptr) return;
            const hipError_t rc = hipFree(ptr);
            if (rc != hipSuccess)
                std::fprintf(stderr, "WARN hipFree %s: %s\n", name,
                             hipGetErrorString(rc));
        };
        for (uint32_t half = 0; half < 2; ++half) {
            release(down_half[half], "down_half");
            release(up_half[half], "up_half");
            release(gate_half[half], "gate_half");
        }
        release(down_full, "down_full");
        release(up_full, "up_full");
        release(gate_full, "gate_full");
    }
};

struct ComponentTensors {
    ds4_gpu_tensor selected = {}, weights = {}, input = {}, input_q8 = {};
    ds4_gpu_tensor gate = {}, up = {}, mid = {}, down = {};
    ds4_gpu_tensor out_full = {}, out_half0 = {}, out_half1 = {}, out_sum = {};

    ~ComponentTensors() {
        ds4_gpu_tensor_free_in_place(&out_sum);
        ds4_gpu_tensor_free_in_place(&out_half1);
        ds4_gpu_tensor_free_in_place(&out_half0);
        ds4_gpu_tensor_free_in_place(&out_full);
        ds4_gpu_tensor_free_in_place(&down);
        ds4_gpu_tensor_free_in_place(&mid);
        ds4_gpu_tensor_free_in_place(&up);
        ds4_gpu_tensor_free_in_place(&gate);
        ds4_gpu_tensor_free_in_place(&input_q8);
        ds4_gpu_tensor_free_in_place(&input);
        ds4_gpu_tensor_free_in_place(&weights);
        ds4_gpu_tensor_free_in_place(&selected);
    }
};

bool alloc_tensor(ds4_gpu_tensor &tensor, uint64_t bytes) {
    std::memset(&tensor, 0, sizeof(tensor));
    return ds4_gpu_tensor_alloc_on(&tensor, 0, bytes) == 0;
}

bool upload(ds4_gpu_tensor &tensor, const void *src, uint64_t bytes) {
    return alloc_tensor(tensor, bytes) &&
           ds4_gpu_tensor_write(&tensor, 0, src, bytes) != 0;
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

float half_round(float value) {
    return __half2float(__float2half_rn(value));
}

float half_bits(uint16_t bits) {
    half value;
    std::memcpy(&value, &bits, sizeof(value));
    return __half2float(value);
}

Q81Comparison compare_q81_effective(const std::vector<Q8KBlock> &expected,
                                    const std::vector<Q8KBlock> &actual) {
    Q81Comparison result;
    if (expected.size() != actual.size()) {
        result.changed_values = UINT64_MAX;
        return result;
    }
    for (size_t block = 0; block < expected.size(); ++block) {
        result.max_scale_delta = std::max(
            result.max_scale_delta,
            std::fabs(expected[block].d - actual[block].d));
        result.changed_scales +=
            half_round(expected[block].d) != half_round(actual[block].d);
        for (uint32_t group = 0; group < 8u; ++group) {
            const int expected_sum = expected[block].bsums[2u * group] +
                                     expected[block].bsums[2u * group + 1u];
            const int actual_sum = actual[block].bsums[2u * group] +
                                   actual[block].bsums[2u * group + 1u];
            result.changed_sums +=
                half_round(expected[block].d * (float)expected_sum) !=
                half_round(actual[block].d * (float)actual_sum);
        }
        for (uint32_t i = 0; i < kQk; ++i)
            result.changed_values +=
                expected[block].qs[i] != actual[block].qs[i];
    }
    return result;
}

bool valid_q8k_capture(const std::vector<Q8KBlock> &blocks) {
    bool any_nonzero = false;
    for (const Q8KBlock &block : blocks) {
        if (!std::isfinite(block.d)) return false;
        any_nonzero |= block.d != 0.0f;
        for (uint32_t group = 0; group < 16u; ++group) {
            int sum = 0;
            for (uint32_t i = 0; i < 16u; ++i)
                sum += block.qs[group * 16u + i];
            if (sum != block.bsums[group]) return false;
        }
    }
    return any_nonzero;
}

void q4k_scale_min(uint32_t group, const uint8_t *packed,
                   uint8_t &scale, uint8_t &minimum) {
    // Keep this local scalar derivation deliberately separate from both the
    // production packed 32-bit unpacker and other test helpers.
    if (group < 4u) {
        scale = packed[group] & 63u;
        minimum = packed[group + 4u] & 63u;
    } else {
        scale = (packed[group + 4u] & 15u) |
                ((packed[group - 4u] >> 6u) << 4u);
        minimum = (packed[group + 4u] >> 4u) |
                  ((packed[group] >> 6u) << 4u);
    }
}

// Host reconstruction of the shipping 256-value Q8_K quantizer. The tree
// reduction matters for equal-magnitude signed extrema, so mirror its strict
// greater-than reduction instead of using a scalar max-abs shortcut. Because
// the production kernel owns quantization, compiler changes to reciprocal
// lowering can move inputs across an int8 rounding boundary; the reported
// gate ratio and reference/actual hashes distinguish that from silent drift.
void quantize_q8k(const float *source, Q8KBlock &out) {
    float magnitude[256], signed_value[256];
    for (uint32_t i = 0; i < 256u; ++i) {
        magnitude[i] = std::fabs(source[i]);
        signed_value[i] = source[i];
    }
    for (uint32_t stride = 128u; stride != 0u; stride >>= 1u) {
        for (uint32_t i = 0; i < stride; ++i) {
            if (magnitude[i + stride] > magnitude[i]) {
                magnitude[i] = magnitude[i + stride];
                signed_value[i] = signed_value[i + stride];
            }
        }
    }
    if (magnitude[0] == 0.0f) {
        std::memset(&out, 0, sizeof(out));
        return;
    }
    const float inverse = -127.0f / signed_value[0];
    for (uint32_t i = 0; i < 256u; ++i) {
        int quantized = (int)std::lrintf(inverse * source[i]);
        quantized = std::max(-128, std::min(127, quantized));
        out.qs[i] = (int8_t)quantized;
    }
    for (uint32_t group = 0; group < 16u; ++group) {
        int sum = 0;
        for (uint32_t i = 0; i < 16u; ++i)
            sum += out.qs[group * 16u + i];
        out.bsums[group] = (int16_t)sum;
    }
    out.d = 1.0f / inverse;
}

// Independent scalar reconstruction of the Q8_K -> Q8_1 integer-WMMA
// arithmetic. It deliberately performs scalar nibble extraction and dot
// products rather than reusing any production GPU loader or matrix intrinsic.
float q4k_q81_reference(const Q4KBlock *weights,
                        const Q8KBlock *activations,
                        uint32_t blocks) {
    float result = 0.0f;
    for (uint32_t block = 0; block < blocks; ++block) {
        const Q4KBlock &weight = weights[block];
        const Q8KBlock &activation = activations[block];
        const float wd = half_bits(weight.d);
        const float wm = half_bits(weight.dmin);
        for (uint32_t group = 0; group < 8u; ++group) {
            uint8_t scale, minimum;
            q4k_scale_min(group, weight.scales, scale, minimum);
            const uint32_t byte_offset = (group >> 1u) * 32u;
            const uint32_t shift = (group & 1u) ? 4u : 0u;
            int dot = 0;
            for (uint32_t i = 0; i < 32u; ++i) {
                const int q4 = (weight.qs[byte_offset + i] >> shift) & 15u;
                dot += q4 * (int)activation.qs[group * 32u + i];
            }
            const int qsum = activation.bsums[group * 2u] +
                             activation.bsums[group * 2u + 1u];
            const float weight_scale = half_round(wd * (float)scale);
            const float weight_min = half_round(-wm * (float)minimum);
            const float act_scale = half_round(activation.d);
            const float act_sum = half_round(activation.d * (float)qsum);
            result += weight_scale * act_scale * (float)dot +
                      weight_min * act_sum;
        }
    }
    return result;
}

float router_sigmoid(float value) {
    if (value >= 0.0f) {
        const float exponential = std::exp(-value);
        return 1.0f / (1.0f + exponential);
    }
    const float exponential = std::exp(value);
    return exponential / (1.0f + exponential);
}

float router_reference(const float *router, const float *bias,
                       const float *input, uint32_t expert_ids[kUsed],
                       float expert_weights[kUsed], float scale) {
    std::vector<float> probs(kTotalExperts);
    std::vector<uint32_t> order(kTotalExperts);
    for (uint32_t expert = 0; expert < kTotalExperts; ++expert) {
        const float *row = router + (uint64_t)expert * kInput;
        double logit = 0.0;
        for (uint32_t column = 0; column < kInput; ++column)
            logit += (double)row[column] * input[column];
        probs[expert] = router_sigmoid((float)logit);
        order[expert] = expert;
    }
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        const float av = probs[a] + bias[a];
        const float bv = probs[b] + bias[b];
        return av > bv || (av == bv && a < b);
    });
    float sum = 0.0f;
    for (uint32_t slot = 0; slot < kUsed; ++slot) {
        expert_ids[slot] = order[slot];
        expert_weights[slot] = probs[order[slot]];
        sum += expert_weights[slot];
    }
    sum = std::max(sum, 6.103515625e-5f);
    for (uint32_t slot = 0; slot < kUsed; ++slot)
        expert_weights[slot] = expert_weights[slot] / sum * scale;
    return sum;
}

struct OracleStats {
    uint64_t count = 0, bad = 0, nonfinite = 0;
    double max_abs = 0.0, max_rel = 0.0, max_gate_ratio = 0.0;
    float max_abs_actual = 0.0f, max_abs_expected = 0.0f;

    void add(float actual, float expected, double absolute, double relative) {
        ++count;
        if (!std::isfinite(actual) || !std::isfinite(expected)) {
            ++nonfinite;
            ++bad;
            return;
        }
        const double error = std::fabs((double)actual - expected);
        if (error > max_abs) {
            max_abs = error;
            max_abs_actual = actual;
            max_abs_expected = expected;
        }
        max_rel = std::max(max_rel,
                           error / std::max(1.0e-12, std::fabs((double)expected)));
        const double tolerance = absolute + relative * std::fabs((double)expected);
        max_gate_ratio = std::max(max_gate_ratio, error / tolerance);
        if (error > tolerance) ++bad;
    }

    bool pass() const { return count != 0 && bad == 0 && nonfinite == 0; }
};

bool tensor_range(const Glm5TestGGUF &gguf, uint64_t offset, uint64_t bytes) {
    return offset <= gguf.size && bytes <= gguf.size - offset;
}

bool gather_experts(const Glm5TestGGUF &gguf, uint64_t offset,
                    const uint32_t expert_ids[kUsed],
                    uint64_t expert_bytes, std::vector<uint8_t> &compact) {
    compact.resize((size_t)kUsed * expert_bytes);
    for (uint32_t slot = 0; slot < kUsed; ++slot) {
        const uint64_t source =
            offset + (uint64_t)expert_ids[slot] * expert_bytes;
        CHECK(tensor_range(gguf, source, expert_bytes),
              "real GGUF expert range");
        std::memcpy(compact.data() + (uint64_t)slot * expert_bytes,
                    gguf.map + source, (size_t)expert_bytes);
    }
    return true;
}

void pack_gate_half(const std::vector<uint8_t> &full,
                    uint64_t full_expert_bytes, uint64_t half_expert_bytes,
                    uint32_t half, std::vector<uint8_t> &packed) {
    packed.resize((size_t)kUsed * half_expert_bytes);
    for (uint32_t expert = 0; expert < kUsed; ++expert) {
        std::memcpy(packed.data() + (uint64_t)expert * half_expert_bytes,
                    full.data() + (uint64_t)expert * full_expert_bytes +
                        (uint64_t)half * half_expert_bytes,
                    (size_t)half_expert_bytes);
    }
}

void pack_down_half(const std::vector<uint8_t> &full,
                    uint64_t full_expert_bytes, uint64_t full_row_bytes,
                    uint64_t half_expert_bytes, uint64_t half_row_bytes,
                    uint32_t half, std::vector<uint8_t> &packed) {
    packed.resize((size_t)kUsed * half_expert_bytes);
    for (uint32_t expert = 0; expert < kUsed; ++expert) {
        for (uint32_t row = 0; row < kOutput; ++row) {
            const uint64_t source = (uint64_t)expert * full_expert_bytes +
                                    (uint64_t)row * full_row_bytes +
                                    (uint64_t)half * half_row_bytes;
            const uint64_t destination =
                (uint64_t)expert * half_expert_bytes +
                (uint64_t)row * half_row_bytes;
            std::memcpy(packed.data() + destination, full.data() + source,
                        (size_t)half_row_bytes);
        }
    }
}

bool launch(ds4_gpu_tensor &out, ds4_gpu_tensor &gate,
            ds4_gpu_tensor &up, ds4_gpu_tensor &mid,
            ds4_gpu_tensor &down, const void *gate_weight,
            const void *up_weight, const void *down_weight,
            uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
            uint64_t down_expert_bytes, uint64_t down_row_bytes,
            const ds4_gpu_tensor &selected, const ds4_gpu_tensor &weights,
            const ds4_gpu_tensor &input, uint32_t mid_dim) {
    const int launched = ds4_gpu_routed_moe_batch_q4k_direct_control(
        &out, &gate, &up, &mid, &down,
        gate_weight, up_weight, down_weight,
        gate_expert_bytes, gate_row_bytes,
        down_expert_bytes, down_row_bytes,
        &selected, &weights, kUsed, kUsed, kClamp, &input,
        3u, kTokens, mid_dim);
    if (!launched || !ds4_gpu_synchronize()) return false;
    return hipGetLastError() == hipSuccess;
}

bool run_test() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    CHECK(model && model[0], "DS4_GLM5_MODEL");
    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM5 GGUF file");

    uint64_t gate_offset = 0, up_offset = 0, down_offset = 0;
    CHECK(gguf.tensor("blk.3.ffn_gate_exps.weight",
                      {kInput, kFullMid, kTotalExperts}, 12u, gate_offset) &&
          gguf.tensor("blk.3.ffn_up_exps.weight",
                      {kInput, kFullMid, kTotalExperts}, 12u, up_offset) &&
          gguf.tensor("blk.3.ffn_down_exps.weight",
                      {kFullMid, kOutput, kTotalExperts}, 12u, down_offset),
          "bind same-GGUF layer-3 routed Q4_K tensors");

    const uint64_t gate_row_bytes = (kInput / kQk) * kQ4Block;
    const uint64_t full_gate_expert = (uint64_t)kFullMid * gate_row_bytes;
    const uint64_t half_gate_expert = (uint64_t)kHalfMid * gate_row_bytes;
    const uint64_t full_down_row = (kFullMid / kQk) * kQ4Block;
    const uint64_t half_down_row = (kHalfMid / kQk) * kQ4Block;
    const uint64_t full_down_expert = (uint64_t)kOutput * full_down_row;
    const uint64_t half_down_expert = (uint64_t)kOutput * half_down_row;
    CHECK(full_gate_expert == full_down_expert &&
          half_gate_expert == half_down_expert,
          "GLM5 2048/1024 expert byte symmetry");

    // The split is exactly four Q4_K/Q8_K blocks. Production dynamically
    // quantizes each 256-value intermediate block independently, so both
    // halves preserve the full path's quantization scales. A future change to
    // row-wide activation scaling invalidates this split-consistency premise.

    const char *bridge_value = std::getenv("DS4_GLM5_ROUTER_MOE_BRIDGE");
    CHECK(!bridge_value || (bridge_value[0] == '1' && bridge_value[1] == '\0'),
          "DS4_GLM5_ROUTER_MOE_BRIDGE must be exactly 1 when set");
    const bool router_bridge = bridge_value && std::strcmp(bridge_value, "1") == 0;
    uint32_t jitter_seed = 0;
    CHECK(glm5_test_router_seed(jitter_seed),
          "valid DS4_GLM5_ROUTER_JITTER_SEED");
    uint32_t expert_ids[kUsed];
    std::copy(kExpertIds, kExpertIds + kUsed, expert_ids);
    float bridge_weights[kUsed] = {};
    const float *bridge_router = nullptr;
    float bridge_scale = 0.0f;
    float bridge_prob_sum = 0.0f;

    std::vector<int32_t> selected((size_t)kTokens * kUsed);
    std::vector<float> route_weights((size_t)kTokens * kUsed);
    std::vector<float> input((size_t)kTokens * kInput);
    for (uint32_t token = 0; token < kTokens; ++token) {
        for (uint32_t i = 0; i < kInput; ++i) {
            if (router_bridge) {
                input[(size_t)token * kInput + i] =
                    glm5_test_router_input(token, i, jitter_seed);
            } else {
                const int value = (int)((i * 17u + token * 13u) % 127u) - 63;
                input[(size_t)token * kInput + i] = (float)value / 256.0f;
            }
        }
    }
    if (router_bridge) {
        bool normalize = false;
        uint64_t router_offset = 0, bias_offset = 0;
        CHECK(gguf.metadata("glm5-next.expert_weights_scale", bridge_scale) &&
              bridge_scale == 2.5f &&
              gguf.metadata("glm5-next.expert_weights_norm", normalize) &&
              normalize &&
              gguf.tensor("blk.3.ffn_gate_inp.weight",
                          {kInput, kTotalExperts}, 0u, router_offset) &&
              gguf.tensor("blk.3.exp_probs_b.bias",
                          {kTotalExperts}, 0u, bias_offset),
              "bind same-GGUF normalized top-8 router bridge");
        const uint64_t router_bytes =
            (uint64_t)kTotalExperts * kInput * sizeof(float);
        const uint64_t bias_bytes = (uint64_t)kTotalExperts * sizeof(float);
        CHECK(tensor_range(gguf, router_offset, router_bytes) &&
              tensor_range(gguf, bias_offset, bias_bytes),
              "router bridge payload ranges");
        bridge_router =
            reinterpret_cast<const float *>(gguf.map + router_offset);
        // This bridge intentionally validates one real router-selected expert
        // set, not dynamic per-token routing. Token 0 supplies the eight real
        // IDs and weights; all 32 arithmetic rows reuse that set under a
        // coprime compact-slot permutation to exercise every WMMA expert tile.
        bridge_prob_sum = router_reference(
            bridge_router,
            reinterpret_cast<const float *>(gguf.map + bias_offset),
            input.data(), expert_ids, bridge_weights, bridge_scale);
    }
    for (uint32_t token = 0; token < kTokens; ++token) {
        float sum = 0.0f;
        for (uint32_t slot = 0; slot < kUsed; ++slot) {
            // A fixed coprime permutation makes sorted-pair preparation
            // non-trivial while every token still exercises all eight slots.
            const uint32_t compact_slot = (slot * 5u + token * 3u) & 7u;
            selected[(size_t)token * kUsed + slot] = (int32_t)compact_slot;
            const float value = router_bridge
                ? bridge_weights[compact_slot]
                : (float)(kUsed - slot + (token & 1u)) * 0.03125f;
            route_weights[(size_t)token * kUsed + slot] = value;
            sum += value;
        }
        if (!router_bridge) {
            for (uint32_t slot = 0; slot < kUsed; ++slot)
                route_weights[(size_t)token * kUsed + slot] /= sum;
        }
    }
    OracleStats association_oracle;
    if (router_bridge) {
        for (uint32_t token = 0; token < kTokens; ++token) {
            for (uint32_t slot = 0; slot < kUsed; ++slot) {
                const uint64_t pair = (uint64_t)token * kUsed + slot;
                const uint32_t compact_slot = (uint32_t)selected[pair];
                const uint32_t real_expert = expert_ids[compact_slot];
                const float *row = bridge_router + (uint64_t)real_expert * kInput;
                double logit = 0.0;
                for (uint32_t column = 0; column < kInput; ++column)
                    logit += (double)row[column] * input[column];
                const float expected =
                    router_sigmoid((float)logit) / bridge_prob_sum * bridge_scale;
                association_oracle.add(route_weights[pair], expected,
                                       1.0e-7, 1.0e-7);
            }
        }
        CHECK(association_oracle.pass(),
              "router expert-to-compact-slot weight association");
    }

    std::vector<uint8_t> gate_full, up_full, down_full;
    std::vector<uint8_t> gate_half[2], up_half[2], down_half[2];
    CHECK(gather_experts(gguf, gate_offset, expert_ids,
                         full_gate_expert, gate_full) &&
          gather_experts(gguf, up_offset, expert_ids,
                         full_gate_expert, up_full) &&
          gather_experts(gguf, down_offset, expert_ids,
                         full_down_expert, down_full),
          "gather eight real GLM5 experts");
    for (uint32_t half = 0; half < 2; ++half) {
        pack_gate_half(gate_full, full_gate_expert, half_gate_expert,
                       half, gate_half[half]);
        pack_gate_half(up_full, full_gate_expert, half_gate_expert,
                       half, up_half[half]);
        pack_down_half(down_full, full_down_expert, full_down_row,
                       half_down_expert, half_down_row,
                       half, down_half[half]);
    }

    CHECK(setenv("DS4_ROCM_Q4K_KSHARD_RESEARCH", "1", 1) == 0 &&
          setenv("DS4_ROCM_Q4K_WMMA_PAIR_GATE_UP", "1", 1) == 0 &&
          setenv("DS4_ROCM_Q4K_WMMA_FUSE_MID", "1", 1) == 0 &&
          unsetenv("DS4_ROCM_Q4K_WMMA_LAYER_LOG") == 0 &&
          unsetenv("DS4_ROCM_TP_PREFILL_SKIP_UNOWNED") == 0 &&
          unsetenv("DS4_ROCM_TP_SKIP_UNOWNED") == 0,
          "enable isolated packed arithmetic control");
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "gfx1151 device available");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    RuntimeGuard runtime;
    CHECK(ds4_gpu_init_multi(&config), "initialize gfx1151");
    runtime.active = true;

    DeviceWeights device;
    const uint64_t full_table_bytes = (uint64_t)kUsed * full_gate_expert;
    const uint64_t half_table_bytes = (uint64_t)kUsed * half_gate_expert;
    CHECK(hipMalloc(&device.gate_full, full_table_bytes) == hipSuccess &&
          hipMalloc(&device.up_full, full_table_bytes) == hipSuccess &&
          hipMalloc(&device.down_full, full_table_bytes) == hipSuccess,
          "allocate compact full expert tables");
    CHECK(hipMemcpy(device.gate_full, gate_full.data(), full_table_bytes,
                    hipMemcpyHostToDevice) == hipSuccess &&
          hipMemcpy(device.up_full, up_full.data(), full_table_bytes,
                    hipMemcpyHostToDevice) == hipSuccess &&
          hipMemcpy(device.down_full, down_full.data(), full_table_bytes,
                    hipMemcpyHostToDevice) == hipSuccess,
          "upload compact full expert tables");
    for (uint32_t half = 0; half < 2; ++half) {
        CHECK(hipMalloc(&device.gate_half[half], half_table_bytes) == hipSuccess &&
              hipMalloc(&device.up_half[half], half_table_bytes) == hipSuccess &&
              hipMalloc(&device.down_half[half], half_table_bytes) == hipSuccess,
              "allocate compact half expert tables");
        CHECK(hipMemcpy(device.gate_half[half], gate_half[half].data(),
                        half_table_bytes, hipMemcpyHostToDevice) == hipSuccess &&
              hipMemcpy(device.up_half[half], up_half[half].data(),
                        half_table_bytes, hipMemcpyHostToDevice) == hipSuccess &&
              hipMemcpy(device.down_half[half], down_half[half].data(),
                        half_table_bytes, hipMemcpyHostToDevice) == hipSuccess,
              "upload compact half expert tables");
    }

    ComponentTensors tensors;
    ds4_gpu_tensor &selected_gpu = tensors.selected;
    ds4_gpu_tensor &weights_gpu = tensors.weights;
    ds4_gpu_tensor &input_gpu = tensors.input;
    ds4_gpu_tensor &input_q8_gpu = tensors.input_q8;
    ds4_gpu_tensor &gate = tensors.gate;
    ds4_gpu_tensor &up = tensors.up;
    ds4_gpu_tensor &mid = tensors.mid;
    ds4_gpu_tensor &down = tensors.down;
    ds4_gpu_tensor &out_full = tensors.out_full;
    ds4_gpu_tensor &out_half0 = tensors.out_half0;
    ds4_gpu_tensor &out_half1 = tensors.out_half1;
    ds4_gpu_tensor &out_sum = tensors.out_sum;
    const uint64_t pair_count = (uint64_t)kTokens * kUsed;
    const uint64_t mid_bytes = pair_count * kFullMid * sizeof(float);
    const uint64_t down_bytes = pair_count * kOutput * sizeof(float);
    const uint64_t out_bytes = (uint64_t)kTokens * kOutput * sizeof(float);
    CHECK(upload(selected_gpu, selected.data(),
                 pair_count * sizeof(int32_t)) &&
          upload(weights_gpu, route_weights.data(),
                 pair_count * sizeof(float)) &&
          upload(input_gpu, input.data(),
                 (uint64_t)input.size() * sizeof(float)) &&
          alloc_tensor(input_q8_gpu,
                       (uint64_t)kTokens * (kInput / kQk) * sizeof(Q8KBlock)) &&
          alloc_tensor(gate, mid_bytes) && alloc_tensor(up, mid_bytes) &&
          alloc_tensor(mid, mid_bytes) && alloc_tensor(down, down_bytes) &&
          alloc_tensor(out_full, out_bytes) &&
          alloc_tensor(out_half0, out_bytes) &&
          alloc_tensor(out_half1, out_bytes) &&
          alloc_tensor(out_sum, out_bytes),
          "allocate GLM5 MoE component tensors");

    CHECK(ds4_gpu_q8k_quantize_research_control(
              &input_q8_gpu, &input_gpu, kInput, kTokens) &&
          ds4_gpu_synchronize(), "capture production Q8_K activation blocks");
    std::vector<Q8KBlock> production_input_q8(
        (size_t)kTokens * (kInput / kQk));
    CHECK(ds4_gpu_tensor_read(
              &input_q8_gpu, 0, production_input_q8.data(),
              (uint64_t)production_input_q8.size() * sizeof(Q8KBlock)),
          "read production Q8_K activation blocks");

    CHECK(launch(out_full, gate, up, mid, down,
                 device.gate_full, device.up_full, device.down_full,
                 full_gate_expert, gate_row_bytes,
                 full_down_expert, full_down_row,
                 selected_gpu, weights_gpu, input_gpu, kFullMid),
          "execute full 2048 top-8 Q4_K control");
    std::vector<float> full_mid((size_t)pair_count * kFullMid);
    CHECK(ds4_gpu_tensor_read(&mid, 0, full_mid.data(), mid_bytes),
          "read full-path pair-major intermediate for host oracle");
    std::vector<Q8KBlock> production_mid_q8(
        (size_t)pair_count * (kFullMid / kQk));
    CHECK(ds4_gpu_tensor_read(
              &gate, 0, production_mid_q8.data(),
              (uint64_t)production_mid_q8.size() * sizeof(Q8KBlock)),
          "read production intermediate Q8_K blocks before shard reuse");
    CHECK(valid_q8k_capture(production_mid_q8),
          "production intermediate Q8_K scratch alias and block invariants");
    CHECK(launch(out_half0, gate, up, mid, down,
                 device.gate_half[0], device.up_half[0], device.down_half[0],
                 half_gate_expert, gate_row_bytes,
                 half_down_expert, half_down_row,
                 selected_gpu, weights_gpu, input_gpu, kHalfMid) &&
          launch(out_half1, gate, up, mid, down,
                 device.gate_half[1], device.up_half[1], device.down_half[1],
                 half_gate_expert, gate_row_bytes,
                 half_down_expert, half_down_row,
                 selected_gpu, weights_gpu, input_gpu, kHalfMid),
          "execute both 1024 top-8 Q4_K shards");
    CHECK(ds4_gpu_add_tensor(&out_sum, &out_half0, &out_half1,
                             kTokens * kOutput) &&
          ds4_gpu_synchronize(), "compose both rank outputs");

    std::vector<float> reference((size_t)kTokens * kOutput);
    std::vector<float> candidate((size_t)kTokens * kOutput);
    std::vector<float> half0((size_t)kTokens * kOutput);
    std::vector<float> half1((size_t)kTokens * kOutput);
    CHECK(ds4_gpu_tensor_read(&out_full, 0, reference.data(), out_bytes) &&
          ds4_gpu_tensor_read(&out_sum, 0, candidate.data(), out_bytes) &&
          ds4_gpu_tensor_read(&out_half0, 0, half0.data(), out_bytes) &&
          ds4_gpu_tensor_read(&out_half1, 0, half1.data(), out_bytes),
          "read full, individual-half, and composed outputs");

    std::vector<Q8KBlock> input_q8((size_t)kTokens * (kInput / kQk));
    for (uint32_t token = 0; token < kTokens; ++token) {
        for (uint32_t block = 0; block < kInput / kQk; ++block) {
            quantize_q8k(input.data() + (size_t)token * kInput +
                             (size_t)block * kQk,
                         input_q8[(size_t)token * (kInput / kQk) + block]);
        }
    }
    uint64_t q8_changed_blocks = 0;
    for (size_t block = 0; block < input_q8.size(); ++block) {
        const Q8KBlock &expected = input_q8[block];
        const Q8KBlock &actual = production_input_q8[block];
        q8_changed_blocks += std::memcmp(&expected, &actual,
                                         sizeof(Q8KBlock)) != 0;
    }
    const Q81Comparison input_q81 =
        compare_q81_effective(input_q8, production_input_q8);
    std::fprintf(stderr,
        "GLM5 Q8_K host-production comparison blocks=%zu changed_blocks=%llu "
        "q81_changed_values=%llu q81_changed_scales=%llu "
        "q81_changed_sums=%llu max_scale_delta=%.9g host_fnv=%016llx "
        "production_fnv=%016llx\n",
        input_q8.size(), (unsigned long long)q8_changed_blocks,
        (unsigned long long)input_q81.changed_values,
        (unsigned long long)input_q81.changed_scales,
        (unsigned long long)input_q81.changed_sums,
        input_q81.max_scale_delta,
        (unsigned long long)fnv1a64(input_q8.data(),
                                    input_q8.size() * sizeof(Q8KBlock)),
        (unsigned long long)fnv1a64(production_input_q8.data(),
                                    production_input_q8.size() * sizeof(Q8KBlock)));
    CHECK(input_q81.changed_values == 0 && input_q81.changed_scales == 0 &&
          input_q81.changed_sums == 0,
          "host Q8_K oracle matches production-effective Q8_1 input");
    std::vector<Q8KBlock> mid_q8((size_t)pair_count * (kFullMid / kQk));
    for (uint64_t pair = 0; pair < pair_count; ++pair) {
        for (uint32_t block = 0; block < kFullMid / kQk; ++block) {
            quantize_q8k(full_mid.data() + pair * kFullMid +
                             (uint64_t)block * kQk,
                         mid_q8[pair * (kFullMid / kQk) + block]);
        }
    }
    const Q81Comparison mid_q81 =
        compare_q81_effective(mid_q8, production_mid_q8);
    std::fprintf(stderr,
        "GLM5 intermediate Q8_K host-production comparison blocks=%zu "
        "q81_changed_values=%llu q81_changed_scales=%llu "
        "q81_changed_sums=%llu max_scale_delta=%.9g host_fnv=%016llx "
        "production_fnv=%016llx\n",
        mid_q8.size(), (unsigned long long)mid_q81.changed_values,
        (unsigned long long)mid_q81.changed_scales,
        (unsigned long long)mid_q81.changed_sums,
        mid_q81.max_scale_delta,
        (unsigned long long)fnv1a64(mid_q8.data(),
                                    mid_q8.size() * sizeof(Q8KBlock)),
        (unsigned long long)fnv1a64(production_mid_q8.data(),
                                    production_mid_q8.size() * sizeof(Q8KBlock)));
    CHECK(!router_bridge ||
          (mid_q81.changed_values == 0 && mid_q81.changed_scales == 0 &&
           mid_q81.changed_sums == 0),
          "host Q8_K oracle matches production-effective Q8_1 intermediate");

    // Sample separated tokens and rows but cover every route slot. The mid
    // leg is an independent same-GGUF scalar arithmetic reference. The final
    // output leg deliberately starts from the production intermediate and
    // therefore validates down projection plus ordered top-8 accumulation,
    // not gate/up a second time.
    constexpr uint32_t oracle_tokens[] = {0u, 7u, 16u, 31u};
    constexpr uint32_t oracle_mid_rows[] =
        {0u, 1u, 255u, 511u, 1023u, 1024u, 1535u, 2047u};
    constexpr uint32_t oracle_out_rows[] =
        {0u, 1u, 127u, 1023u, 2047u, 3071u, 4095u};
    OracleStats mid_oracle, output_oracle;
    std::vector<float> oracle_values;
    std::vector<float> oracle_actual_values;
    for (uint32_t token : oracle_tokens) {
        const Q8KBlock *token_q8 =
            input_q8.data() + (size_t)token * (kInput / kQk);
        for (uint32_t slot = 0; slot < kUsed; ++slot) {
            const uint64_t pair = (uint64_t)token * kUsed + slot;
            const uint32_t expert = (uint32_t)selected[pair];
            for (uint32_t row : oracle_mid_rows) {
                const auto *gate_row = reinterpret_cast<const Q4KBlock *>(
                    gate_full.data() + (uint64_t)expert * full_gate_expert +
                    (uint64_t)row * gate_row_bytes);
                const auto *up_row = reinterpret_cast<const Q4KBlock *>(
                    up_full.data() + (uint64_t)expert * full_gate_expert +
                    (uint64_t)row * gate_row_bytes);
                float gate_ref = q4k_q81_reference(
                    gate_row, token_q8, kInput / kQk);
                float up_ref = q4k_q81_reference(
                    up_row, token_q8, kInput / kQk);
                gate_ref = std::min(gate_ref, kClamp);
                up_ref = std::max(-kClamp, std::min(up_ref, kClamp));
                const float expected =
                    (gate_ref / (1.0f + std::exp(-gate_ref))) * up_ref *
                    route_weights[pair];
                mid_oracle.add(full_mid[pair * kFullMid + row], expected,
                               1.0e-7, 1.0e-5);
                oracle_values.push_back(expected);
                oracle_actual_values.push_back(
                    full_mid[pair * kFullMid + row]);
            }
        }
        for (uint32_t row : oracle_out_rows) {
            float expected = 0.0f;
            for (uint32_t slot = 0; slot < kUsed; ++slot) {
                const uint64_t pair = (uint64_t)token * kUsed + slot;
                const uint32_t expert = (uint32_t)selected[pair];
                const auto *down_row = reinterpret_cast<const Q4KBlock *>(
                    down_full.data() + (uint64_t)expert * full_down_expert +
                    (uint64_t)row * full_down_row);
                expected += q4k_q81_reference(
                    down_row,
                    mid_q8.data() + pair * (kFullMid / kQk),
                    kFullMid / kQk);
            }
            output_oracle.add(reference[(size_t)token * kOutput + row],
                              expected, 1.0e-6, 1.0e-5);
            oracle_values.push_back(expected);
            oracle_actual_values.push_back(
                reference[(size_t)token * kOutput + row]);
        }
    }
    std::fprintf(stderr,
        "GLM5 Q4_K independent scalar oracle mid=%llu bad=%llu "
        "max_abs=%.9g max_abs_actual=%.9g max_abs_expected=%.9g "
        "max_rel=%.9g max_gate_ratio=%.9g "
        "output=%llu bad=%llu max_abs=%.9g max_rel=%.9g "
        "max_gate_ratio=%.9g reference_fnv=%016llx actual_fnv=%016llx\n",
        (unsigned long long)mid_oracle.count,
        (unsigned long long)mid_oracle.bad,
        mid_oracle.max_abs, mid_oracle.max_abs_actual,
        mid_oracle.max_abs_expected, mid_oracle.max_rel,
        mid_oracle.max_gate_ratio,
        (unsigned long long)output_oracle.count,
        (unsigned long long)output_oracle.bad,
        output_oracle.max_abs, output_oracle.max_rel,
        output_oracle.max_gate_ratio,
        (unsigned long long)fnv1a64(oracle_values.data(),
                                    oracle_values.size() * sizeof(float)),
        (unsigned long long)fnv1a64(oracle_actual_values.data(),
                                    oracle_actual_values.size() * sizeof(float)));
    const bool oracle_pass = mid_oracle.pass() && output_oracle.pass();
    long double error_sq = 0.0L, reference_sq = 0.0L;
    long double half0_sq = 0.0L, half1_sq = 0.0L;
    long double half_delta_sq = 0.0L;
    long double half0_full_delta_sq = 0.0L, half1_full_delta_sq = 0.0L;
    float max_abs = 0.0f, max_rel = 0.0f;
    uint64_t changed = 0, nonfinite = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const float ref = reference[i], got = candidate[i];
        const float h0 = half0[i], h1 = half1[i];
        if (!std::isfinite(ref) || !std::isfinite(got) ||
            !std::isfinite(h0) || !std::isfinite(h1)) ++nonfinite;
        const float error = std::fabs(got - ref);
        max_abs = std::max(max_abs, error);
        max_rel = std::max(max_rel, error / std::max(1.0f, std::fabs(ref)));
        error_sq += (long double)error * error;
        reference_sq += (long double)ref * ref;
        half0_sq += (long double)h0 * h0;
        half1_sq += (long double)h1 * h1;
        const long double half_delta = (long double)h0 - h1;
        const long double half0_full_delta = (long double)h0 - ref;
        const long double half1_full_delta = (long double)h1 - ref;
        half_delta_sq += half_delta * half_delta;
        half0_full_delta_sq += half0_full_delta * half0_full_delta;
        half1_full_delta_sq += half1_full_delta * half1_full_delta;
        changed += std::memcmp(&ref, &got, sizeof(float)) != 0;
    }
    const double nmse = reference_sq == 0.0L
        ? (double)error_sq : (double)(error_sq / reference_sq);
    const bool independent_halves =
        half0_sq > 1.0e-12L && half1_sq > 1.0e-12L &&
        half_delta_sq > 1.0e-12L &&
        half0_full_delta_sq > 1.0e-12L &&
        half1_full_delta_sq > 1.0e-12L;
    const bool pass = nonfinite == 0 && reference_sq > 1.0e-12L &&
                      independent_halves &&
                      max_abs <= 3.0e-7f && max_rel <= 3.0e-7f &&
                      nmse <= 1.0e-11;
    const uint64_t gate_weight_fnv = fnv1a64(gate_full.data(),
                                              gate_full.size());
    const uint64_t up_weight_fnv = fnv1a64(up_full.data(), up_full.size());
    const uint64_t down_weight_fnv = fnv1a64(down_full.data(),
                                              down_full.size());
    const uint64_t expert_ids_fnv = fnv1a64(expert_ids, sizeof(expert_ids));
    const uint64_t route_weights_fnv =
        fnv1a64(route_weights.data(), route_weights.size() * sizeof(float));
    std::fprintf(stderr,
        "GLM5 Q4_K expert set router_bridge=%d seed_active=%d seed=%u "
        "ids=%u,%u,%u,%u,%u,%u,%u,%u ids_fnv=%016llx "
        "route_source=%s distinct_route_sets=%u association_pairs=%llu "
        "route_weights_fnv=%016llx association_bad=%llu "
        "association_max_abs=%.9g\n",
        router_bridge, router_bridge, jitter_seed,
        expert_ids[0], expert_ids[1], expert_ids[2],
        expert_ids[3], expert_ids[4], expert_ids[5], expert_ids[6],
        expert_ids[7], (unsigned long long)expert_ids_fnv,
        router_bridge ? "token0" : "synthetic",
        router_bridge ? 1u : 0u,
        (unsigned long long)association_oracle.count,
        (unsigned long long)route_weights_fnv,
        (unsigned long long)association_oracle.bad,
        association_oracle.max_abs);
    std::fprintf(stderr,
        "GLM5 Q4_K shard split-consistency layer=3 descriptor_experts=288 "
        "compact_table=8 router_bridge=%d "
        "tokens=%u changed_telemetry=%llu nonfinite=%llu "
        "independent_halves=%d max_abs=%.9g max_rel=%.9g nmse=%.9g "
        "gate_weight_fnv=%016llx up_weight_fnv=%016llx "
        "down_weight_fnv=%016llx full_fnv=%016llx shard_fnv=%016llx\n",
        router_bridge, kTokens, (unsigned long long)changed,
        (unsigned long long)nonfinite,
        independent_halves, max_abs, max_rel, nmse,
        (unsigned long long)gate_weight_fnv,
        (unsigned long long)up_weight_fnv,
        (unsigned long long)down_weight_fnv,
        (unsigned long long)fnv1a64(reference.data(), out_bytes),
        (unsigned long long)fnv1a64(candidate.data(), out_bytes));
    CHECK(pass, "same-GGUF 1024/1024 top-8 numerical envelope");
    CHECK(oracle_pass,
          "sampled same-GGUF scalar mid and conditioned-down arithmetic oracle");
    std::fprintf(stderr,
                 "PASS same-GGUF GLM5 Q4_K router_bridge=%d "
                 "route_source=%s sampled "
                 "scalar-mid, conditioned-down, and top-8 shard "
                 "split-consistency\n",
                 router_bridge, router_bridge ? "token0" : "synthetic");
    return true;
}

}  // namespace

int main() { return run_test() ? 0 : 1; }
