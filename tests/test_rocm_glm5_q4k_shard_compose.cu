#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "tests/glm5_gguf_test.hpp"
extern "C" {
#include "ds4_tp.h"
#include "ds4_glm5_next_runtime.h"
}

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CHECK(expr, message) do {                                           \
    if (!(expr)) {                                                          \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__);    \
        return false;                                                       \
    }                                                                       \
} while (0)

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
constexpr uint32_t kHc = 4;
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
constexpr uint32_t kWmmaMinCount = 6;
// The production routed-Q4_K down path currently fixes this to one, so every
// selected expert uses the integer-WMMA down projection.
constexpr uint32_t kDownWmmaMinCount = 1;
constexpr uint64_t kDynamicMidSumRoundingBudget = 2;
static_assert(kDownWmmaMinCount == 1u,
              "production down-path threshold assumption");
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

struct TpGuard {
    ds4_tp *tp = nullptr;
    void *slab = nullptr;

    ~TpGuard() {
        if (tp) ds4_tp_free(tp);
        if (slab) {
            const hipError_t rc = hipHostFree(slab);
            if (rc != hipSuccess)
                std::fprintf(stderr, "WARN hipHostFree TP slab: %s\n",
                             hipGetErrorString(rc));
        }
    }
};

struct WindowCacheGuard {
    ds4_gpu_q4k_window_cache *cache = nullptr;
    ~WindowCacheGuard() { reset(); }
    void reset() {
        if (cache) {
            ds4_gpu_q4k_window_cache_destroy(cache);
            cache = nullptr;
        }
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

struct RouterTensors {
    ds4_gpu_tensor input = {}, logits = {}, probs = {};
    ds4_gpu_tensor selected = {}, weights = {};

    ~RouterTensors() {
        ds4_gpu_tensor_free_in_place(&weights);
        ds4_gpu_tensor_free_in_place(&selected);
        ds4_gpu_tensor_free_in_place(&probs);
        ds4_gpu_tensor_free_in_place(&logits);
        ds4_gpu_tensor_free_in_place(&input);
    }
};

struct SharedTensors {
    ds4_gpu_tensor input = {}, gate = {}, up = {}, mid = {}, output = {};
    ~SharedTensors() {
        ds4_gpu_tensor_free_in_place(&output);
        ds4_gpu_tensor_free_in_place(&mid);
        ds4_gpu_tensor_free_in_place(&up);
        ds4_gpu_tensor_free_in_place(&gate);
        ds4_gpu_tensor_free_in_place(&input);
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

bool read_exact_f32(const char *path, size_t count, std::vector<float> &out) {
    if (!path || !path[0]) return false;
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return false;
    const std::streamoff size = input.tellg();
    if (size < 0 || (uint64_t)size != (uint64_t)count * sizeof(float))
        return false;
    input.seekg(0);
    out.resize(count);
    return (bool)input.read((char *)out.data(), size);
}

bool read_exact_i32(const char *path, size_t count, std::vector<int32_t> &out) {
    if (!path || !path[0]) return false;
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return false;
    const std::streamoff size = input.tellg();
    if (size < 0 || (uint64_t)size != (uint64_t)count * sizeof(int32_t))
        return false;
    input.seekg(0);
    out.resize(count);
    return (bool)input.read((char *)out.data(), size);
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

// Independent reconstruction of the cold-expert DP4A path. Unlike integer
// WMMA, this path retains the F32 Q8_K scale and reduces K blocks through the
// same eight quarter-wave lanes used by moe_gate_up_q4K_cold_tile16_kernel.
float q4k_q8k_cold_reference(const Q4KBlock *weights,
                             const Q8KBlock *activations,
                             uint32_t blocks) {
    float lane_sum[8] = {};
    for (uint32_t lane = 0; lane < 8u; ++lane) {
        for (uint32_t block = lane; block < blocks; block += 8u) {
            const Q4KBlock &weight = weights[block];
            const Q8KBlock &activation = activations[block];
            const float wd = half_bits(weight.d);
            const float wm = half_bits(weight.dmin);
            int isum = 0, summs = 0;
            for (uint32_t group = 0; group < 8u; ++group) {
                uint8_t scale, minimum;
                q4k_scale_min(group, weight.scales, scale, minimum);
                const uint32_t byte_offset = (group >> 1u) * 32u;
                const uint32_t shift = (group & 1u) ? 4u : 0u;
                int dot = 0;
                for (uint32_t i = 0; i < 32u; ++i) {
                    const int q4 =
                        (weight.qs[byte_offset + i] >> shift) & 15u;
                    dot += q4 * (int)activation.qs[group * 32u + i];
                }
                isum += (int)scale * dot;
                summs += (int)minimum *
                    (activation.bsums[group * 2u] +
                     activation.bsums[group * 2u + 1u]);
            }
            lane_sum[lane] += activation.d * wd * (float)isum -
                              activation.d * wm * (float)summs;
        }
    }
    for (uint32_t offset = 4u; offset != 0u; offset >>= 1u) {
        for (uint32_t lane = 0; lane < offset; ++lane)
            lane_sum[lane] += lane_sum[lane + offset];
    }
    return lane_sum[0];
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
                    const std::vector<uint32_t> &expert_ids,
                    uint64_t expert_bytes, std::vector<uint8_t> &compact) {
    if (expert_ids.empty()) return false;
    compact.resize(expert_ids.size() * expert_bytes);
    for (size_t slot = 0; slot < expert_ids.size(); ++slot) {
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
    const size_t expert_count = full.size() / full_expert_bytes;
    packed.resize(expert_count * half_expert_bytes);
    for (size_t expert = 0; expert < expert_count; ++expert) {
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
    const size_t expert_count = full.size() / full_expert_bytes;
    packed.resize(expert_count * half_expert_bytes);
    for (size_t expert = 0; expert < expert_count; ++expert) {
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

bool gather_gate_half_direct(const Glm5TestGGUF &gguf, uint64_t offset,
                             const std::vector<uint32_t> &expert_ids,
                             uint64_t full_expert_bytes,
                             uint64_t half_expert_bytes, uint32_t half,
                             std::vector<uint8_t> &packed) {
    packed.resize(expert_ids.size() * half_expert_bytes);
    for (size_t compact = 0; compact < expert_ids.size(); ++compact) {
        const uint64_t source = offset +
            (uint64_t)expert_ids[compact] * full_expert_bytes +
            (uint64_t)half * half_expert_bytes;
        CHECK(tensor_range(gguf, source, half_expert_bytes),
              "rank-local gate/up expert range");
        std::memcpy(packed.data() + compact * half_expert_bytes,
                    gguf.map + source, (size_t)half_expert_bytes);
    }
    return true;
}

bool gather_down_half_direct(const Glm5TestGGUF &gguf, uint64_t offset,
                             const std::vector<uint32_t> &expert_ids,
                             uint64_t full_expert_bytes,
                             uint64_t full_row_bytes,
                             uint64_t half_expert_bytes,
                             uint64_t half_row_bytes, uint32_t half,
                             std::vector<uint8_t> &packed) {
    packed.resize(expert_ids.size() * half_expert_bytes);
    for (size_t compact = 0; compact < expert_ids.size(); ++compact) {
        for (uint32_t row = 0; row < kOutput; ++row) {
            const uint64_t source = offset +
                (uint64_t)expert_ids[compact] * full_expert_bytes +
                (uint64_t)row * full_row_bytes +
                (uint64_t)half * half_row_bytes;
            CHECK(tensor_range(gguf, source, half_row_bytes),
                  "rank-local down expert range");
            std::memcpy(packed.data() + compact * half_expert_bytes +
                            (uint64_t)row * half_row_bytes,
                        gguf.map + source, (size_t)half_row_bytes);
        }
    }
    return true;
}

bool declare_window_half(const Glm5TestGGUF &gguf, uint64_t gate_offset,
                         uint64_t up_offset, uint64_t down_offset,
                         uint32_t half) {
    const uint32_t row_base = half * kHalfMid;
    const uint64_t column_base = (uint64_t)half *
        ((kHalfMid / kQk) * kQ4Block);
    const uint64_t gate_row_bytes = (kInput / kQk) * kQ4Block;
    const uint64_t down_row_bytes = (kFullMid / kQk) * kQ4Block;
    const uint64_t half_down_row_bytes = (kHalfMid / kQk) * kQ4Block;
    return ds4_gpu_q4k_packed_slice_declare(
               gguf.map, gguf.size, gate_offset, kTotalExperts, kFullMid,
               gate_row_bytes, row_base, kHalfMid, 0u, gate_row_bytes,
               DS4_GPU_Q4K_PACKED_ROW_RANGE) &&
           ds4_gpu_q4k_packed_slice_declare(
               gguf.map, gguf.size, up_offset, kTotalExperts, kFullMid,
               gate_row_bytes, row_base, kHalfMid, 0u, gate_row_bytes,
               DS4_GPU_Q4K_PACKED_ROW_RANGE) &&
           ds4_gpu_q4k_packed_slice_declare(
               gguf.map, gguf.size, down_offset, kTotalExperts, kOutput,
               down_row_bytes, 0u, kOutput, column_base,
               half_down_row_bytes, DS4_GPU_Q4K_PACKED_K_RANGE);
}

ds4_gpu_q4k_window_cache_config window_cache_config(
        const Glm5TestGGUF &gguf, uint64_t gate_offset,
        uint64_t up_offset, uint64_t down_offset, uint32_t half) {
    ds4_gpu_q4k_window_cache_config config = {};
    config.model_map = gguf.map;
    config.gate_offset = gate_offset;
    config.up_offset = up_offset;
    config.down_offset = down_offset;
    config.n_expert = kTotalExperts;
    config.gate_row_base = half * kHalfMid;
    config.gate_row_count = kHalfMid;
    config.gate_column_byte_count = (kInput / kQk) * kQ4Block;
    config.down_row_count = kOutput;
    config.down_column_byte_base =
        (uint64_t)half * ((kHalfMid / kQk) * kQ4Block);
    config.down_column_byte_count = (kHalfMid / kQk) * kQ4Block;
    config.slots = kUsed;
    return config;
}

bool parse_fnv64(const char *value, uint64_t &hash) {
    if (!value || std::strlen(value) != 16) return false;
    hash = 0;
    for (size_t i = 0; i < 16; ++i) {
        const unsigned char c = (unsigned char)value[i];
        uint64_t digit = 0;
        if (c >= '0' && c <= '9') digit = c - '0';
        else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10u;
        else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10u;
        else return false;
        hash = (hash << 4u) | digit;
    }
    return true;
}

bool launch(ds4_gpu_tensor &out, ds4_gpu_tensor &gate,
            ds4_gpu_tensor &up, ds4_gpu_tensor &mid,
            ds4_gpu_tensor &down, const void *gate_weight,
            const void *up_weight, const void *down_weight,
            uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
            uint64_t down_expert_bytes, uint64_t down_row_bytes,
            const ds4_gpu_tensor &selected, const ds4_gpu_tensor &weights,
            const ds4_gpu_tensor &input, uint32_t total_experts,
            uint32_t mid_dim, uint32_t token_count = kTokens) {
    const int launched = ds4_gpu_routed_moe_batch_q4k_direct_control(
        &out, &gate, &up, &mid, &down,
        gate_weight, up_weight, down_weight,
        gate_expert_bytes, gate_row_bytes,
        down_expert_bytes, down_row_bytes,
        &selected, &weights, total_experts, kUsed, kClamp, &input,
        3u, token_count, mid_dim);
    if (!launched || !ds4_gpu_synchronize()) return false;
    return hipGetLastError() == hipSuccess;
}

bool run_shared_half(const Glm5TestGGUF &gguf, uint64_t gate_offset,
                     uint64_t up_offset, uint64_t down_offset,
                     const std::vector<float> &input, uint32_t half,
                     std::vector<float> &output) {
    CHECK(input.size() == kInput && half < 2u,
          "shared half input geometry");
    constexpr uint64_t q8_block_bytes = 34u;
    constexpr uint64_t q8_qk = 32u;
    const uint64_t gate_row_bytes = (kInput / q8_qk) * q8_block_bytes;
    const uint64_t row_offset = (uint64_t)half * kHalfMid * gate_row_bytes;
    SharedTensors tensors;
    CHECK(upload(tensors.input, input.data(),
                 (uint64_t)input.size() * sizeof(float)) &&
          alloc_tensor(tensors.gate,
                       (uint64_t)kHalfMid * sizeof(float)) &&
          alloc_tensor(tensors.up,
                       (uint64_t)kHalfMid * sizeof(float)) &&
          alloc_tensor(tensors.mid,
                       (uint64_t)kHalfMid * sizeof(float)) &&
          alloc_tensor(tensors.output,
                       (uint64_t)kOutput * sizeof(float)),
          "allocate shared-expert half tensors");
    CHECK(ds4_gpu_matmul_q8_0_tensor(
              &tensors.gate, gguf.map, gguf.size, gate_offset + row_offset,
              kInput, kHalfMid, &tensors.input, 1u) &&
          ds4_gpu_matmul_q8_0_tensor(
              &tensors.up, gguf.map, gguf.size, up_offset + row_offset,
              kInput, kHalfMid, &tensors.input, 1u) &&
          ds4_gpu_swiglu_tensor(&tensors.mid, &tensors.gate, &tensors.up,
                                kHalfMid, kClamp, 1.0f) &&
          ds4_gpu_matmul_q8_0_kslice_tensor(
              &tensors.output, gguf.map, gguf.size, down_offset, kFullMid,
              (uint64_t)half * kHalfMid, kHalfMid, kOutput, &tensors.mid,
              0u) &&
          ds4_gpu_synchronize(),
          "execute shared-expert rank half");
    output.resize(kOutput);
    CHECK(ds4_gpu_tensor_read(&tensors.output, 0u, output.data(),
                              (uint64_t)kOutput * sizeof(float)),
          "read shared-expert rank half");
    return true;
}

bool run_hc_block_carry(const std::vector<float> &residual,
                        const std::vector<float> &split,
                        const std::vector<float> &branch,
                        std::vector<float> &carried) {
    CHECK(residual.size() == (size_t)kHc * kInput && split.size() == 24u &&
          branch.size() == kOutput,
          "FFN final mHC carry geometry");
    ds4_gpu_tensor residual_t = {}, split_t = {}, branch_t = {}, carried_t = {};
    const auto release = [&]() {
        ds4_gpu_tensor_free_in_place(&carried_t);
        ds4_gpu_tensor_free_in_place(&branch_t);
        ds4_gpu_tensor_free_in_place(&split_t);
        ds4_gpu_tensor_free_in_place(&residual_t);
    };
    bool ok = upload(residual_t, residual.data(),
                     (uint64_t)residual.size() * sizeof(float)) &&
              upload(split_t, split.data(),
                     (uint64_t)split.size() * sizeof(float)) &&
              upload(branch_t, branch.data(),
                     (uint64_t)branch.size() * sizeof(float)) &&
              alloc_tensor(carried_t,
                           (uint64_t)residual.size() * sizeof(float));
    if (ok) {
        ok = ds4_gpu_hc_expand_split_tensor(
                 &carried_t, &branch_t, &residual_t, &split_t,
                 kInput, kHc) &&
             ds4_gpu_synchronize();
    }
    if (ok) {
        carried.resize(residual.size());
        ok = ds4_gpu_tensor_read(
            &carried_t, 0u, carried.data(),
            (uint64_t)carried.size() * sizeof(float));
    }
    release();
    CHECK(ok, "execute/read final FFN mHC carry");
    return true;
}

bool run_add2(const std::vector<float> &a, const std::vector<float> &b,
              std::vector<float> &sum) {
    CHECK(a.size() == kOutput && b.size() == kOutput,
          "FFN two-vector sum geometry");
    ds4_gpu_tensor at = {}, bt = {}, out = {};
    const auto release = [&]() {
        ds4_gpu_tensor_free_in_place(&out);
        ds4_gpu_tensor_free_in_place(&bt);
        ds4_gpu_tensor_free_in_place(&at);
    };
    bool ok = upload(at, a.data(), (uint64_t)a.size() * sizeof(float)) &&
              upload(bt, b.data(), (uint64_t)b.size() * sizeof(float)) &&
              alloc_tensor(out, (uint64_t)kOutput * sizeof(float));
    if (ok) {
        ok = ds4_gpu_add_tensor(&out, &at, &bt, kOutput) &&
             ds4_gpu_synchronize();
    }
    if (ok) {
        sum.resize(kOutput);
        ok = ds4_gpu_tensor_read(&out, 0u, sum.data(),
                                 (uint64_t)kOutput * sizeof(float));
    }
    release();
    CHECK(ok, "execute/read FFN two-vector sum");
    return true;
}

bool run_roce_composition(const Glm5TestGGUF &gguf,
                          const std::vector<float> &reference,
                          const std::vector<float> &half0,
                          const std::vector<float> &half1,
                          uint64_t route_contract_hash,
                          bool exclusive_rank_local = false,
                          uint64_t expected_composed_fnv = 0,
                          uint32_t token_count = kTokens,
                          std::vector<float> *composed_output = nullptr) {
    const char *role_value = std::getenv("DS4_GLM5_TP_ROLE");
    if (!role_value) return true;
    CHECK(std::strcmp(role_value, "leader") == 0 ||
          std::strcmp(role_value, "worker") == 0,
          "DS4_GLM5_TP_ROLE must be exactly leader or worker");
    const bool leader = std::strcmp(role_value, "leader") == 0;
    const char *host = std::getenv("DS4_GLM5_TP_HOST");
    const char *device = std::getenv("DS4_GLM5_TP_RDMA_DEVICE");
    const char *port_value = std::getenv("DS4_GLM5_TP_PORT");
    const char *connect_timeout =
        std::getenv("DS4_GLM5_TP_CONNECT_TIMEOUT_SEC");
    if (!connect_timeout || !connect_timeout[0]) connect_timeout = "120";
    CHECK(host && host[0] && device && device[0] && port_value && port_value[0],
          "bounded TP host, RDMA device, and port are required");
    char *port_end = nullptr;
    const long port = std::strtol(port_value, &port_end, 10);
    CHECK(port_end && *port_end == '\0' && port >= 1024 && port <= 65535,
          "bounded TP port range");
    CHECK(token_count != 0u && token_count <= kTokens,
          "bounded TP token count");
    const std::string direct_rows = std::to_string(token_count);
    CHECK(setenv("DS4_TP_BIG_DIRECT", "1", 1) == 0 &&
          setenv("DS4_TP_BIG_DIRECT_MAX_ROWS", direct_rows.c_str(), 1) == 0 &&
          setenv("DS4_TP_CONNECT_TIMEOUT_SEC", connect_timeout, 1) == 0 &&
          unsetenv("DS4_TP_VERBS_LIB") == 0,
          "select bounded system-verbs RoCE slab");
    std::fprintf(stderr,
        "GLM5 bounded TP setup system_verbs=1 big_direct=1 rows=%u "
        "connect_timeout=%ss route_contract=%016llx\n",
        token_count, connect_timeout,
        (unsigned long long)route_contract_hash);

    ds4_tp_options options = {};
    options.role = leader ? DS4_TP_LEADER : DS4_TP_WORKER;
    options.requested = true;
    options.listen_host = leader ? host : nullptr;
    options.listen_port = leader ? (int)port : 0;
    options.leader_host = leader ? nullptr : host;
    options.leader_port = leader ? 0 : (int)port;
    options.transport = DS4_TP_TRANSPORT_RDMA;
    options.rdma_device = device;
    options.rdma_gid_index = 3;
    options.rdma_gid_index_set = true;

    ds4_tp_identity identity = {};
    identity.gguf_bytes = gguf.size;
    identity.model_id = 3u;  // DS4_VARIANT_GLM53
    // Keep the real graph depth in the hello/slab geometry. The current bulk
    // capability predicate requires the normal >=4 MiB batch region even for
    // a direct big-region payload; this test does not use that staging region.
    identity.n_layer = 46;
    identity.n_embd = kOutput;
    identity.n_vocab = 154880u;
    identity.quant_bits = 4;
    identity.ctx_size = token_count;
    identity.runtime_features =
        DS4_TP_FEATURE_Q4K_WMMA | DS4_TP_FEATURE_Q4K_KSHARD;
    identity.gate_slot_start = 3u * DS4_TP_GATES_PER_LAYER;
    identity.gate_slot_step = 1u;
    CHECK(ds4_glm5_next_build_tp_gate_mask(identity.gate_slot_mask,
                                            &identity.gates_per_token),
          "GLM5.3 hybrid TP gate schedule");

    TpGuard transport;
    char error[256] = {};
    CHECK(ds4_tp_create(&transport.tp, &options, &identity,
                        error, sizeof(error)), error);
    CHECK(ds4_tp_is_rdma(transport.tp),
          "explicit bounded TP transport is RDMA");
    CHECK(ds4_tp_requires_host_slab(transport.tp),
          "mlx5 selects mapped host slab");
    const uint64_t slab_bytes = ds4_tp_alloc_slab_bytes(transport.tp);
    CHECK(slab_bytes != 0 &&
          hipHostMalloc(&transport.slab, slab_bytes,
                        hipHostMallocMapped) == hipSuccess,
          "allocate bounded mapped RoCE slab");
    CHECK(ds4_tp_attach_slab(transport.tp, transport.slab,
                             error, sizeof(error)), error);
    CHECK(ds4_tp_big_gate_is_rdma_capable(transport.tp),
          "bounded TP bulk gate is RDMA capable after slab registration");
    CHECK(ds4_tp_hash_check(transport.tp, UINT64_C(0x474c4d5200000001),
                            route_contract_hash,
                            error, sizeof(error)) == 1,
          error);

    const std::vector<float> &local =
        exclusive_rank_local ? half0 : (leader ? half0 : half1);
    const uint64_t bytes = (uint64_t)local.size() * sizeof(float);
    const uint64_t capacity_bytes =
        (uint64_t)ds4_tp_big_capacity_rows(transport.tp) *
        kOutput * sizeof(float);
    CHECK(ds4_tp_big_capacity_rows(transport.tp) >= token_count &&
          bytes <= capacity_bytes,
          "bounded TP payload fits each registered big region");
    auto *base = static_cast<uint8_t *>(transport.slab);
    float *out = reinterpret_cast<float *>(
        base + ds4_tp_slab_big_out_offset(transport.tp));
    float *in = reinterpret_cast<float *>(
        base + ds4_tp_slab_big_in_offset(transport.tp));
    CHECK(local.size() == (size_t)token_count * kOutput &&
          (exclusive_rank_local ||
           (local.size() == reference.size() && half0.size() == half1.size())),
          "bounded TP output shapes");
    std::memcpy(out, local.data(), (size_t)bytes);
    std::memset(in, 0, (size_t)bytes);
    CHECK(ds4_tp_big_gate_is_direct(transport.tp, out, in, bytes),
          "bounded TP payload uses the registered direct regions");
    CHECK(ds4_tp_big_gate_exchange(transport.tp, 3u, 1u,
                                   out, in, bytes),
          "exchange bounded GLM5 half over RoCE");

    OracleStats composition;
    std::vector<float> composed(local.size());
    for (size_t i = 0; i < local.size(); ++i) {
        // The serial control uses the GPU tensor-add path. Exact equality here
        // also gates host-add/GPU-add parity for these bounded oracle vectors.
        composed[i] = out[i] + in[i];
        if (!exclusive_rank_local)
            composition.add(composed[i], reference[i], 3.0e-7, 3.0e-7);
    }
    const uint64_t composed_fnv =
        fnv1a64(composed.data(), bytes);
    if (composed_output) *composed_output = composed;
    CHECK(ds4_tp_hash_check(transport.tp, UINT64_C(0x474c4d5200000002),
                            composed_fnv,
                            error, sizeof(error)) == 1,
          error);
    if (exclusive_rank_local) {
        std::fprintf(stderr,
            "GLM5 bounded TP RoCE role=%s device=%s bytes=%llu "
            "direct=1 exclusive_rank_local=1 compare=fnv "
            "local_fnv=%016llx peer_fnv=%016llx composed_fnv=%016llx "
            "blessed_fnv=%016llx\n",
            role_value, device, (unsigned long long)bytes,
            (unsigned long long)fnv1a64(out, bytes),
            (unsigned long long)fnv1a64(in, bytes),
            (unsigned long long)composed_fnv,
            (unsigned long long)expected_composed_fnv);
    } else {
        std::fprintf(stderr,
            "GLM5 bounded TP RoCE role=%s device=%s bytes=%llu "
            "direct=1 exclusive_rank_local=0 "
            "bad=%llu max_abs=%.9g max_rel=%.9g "
            "local_fnv=%016llx peer_fnv=%016llx composed_fnv=%016llx "
            "reference_fnv=%016llx\n",
            role_value, device, (unsigned long long)bytes,
            (unsigned long long)composition.bad,
            composition.max_abs, composition.max_rel,
            (unsigned long long)fnv1a64(out, bytes),
            (unsigned long long)fnv1a64(in, bytes),
            (unsigned long long)composed_fnv,
            (unsigned long long)fnv1a64(reference.data(), bytes));
    }
    CHECK(exclusive_rank_local ? composed_fnv == expected_composed_fnv :
                                composition.pass(),
          exclusive_rank_local ?
              "exclusive rank-local RoCE output matches blessed composition" :
              "bounded two-process RoCE composition matches full Q4_K oracle");
    return true;
}

bool run_test() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    CHECK(model && model[0], "DS4_GLM5_MODEL");
    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM5 GGUF file");

    uint64_t gate_offset = 0, up_offset = 0, down_offset = 0;
    uint64_t shared_gate_offset = 0, shared_up_offset = 0;
    uint64_t shared_down_offset = 0;
    CHECK(gguf.tensor("blk.3.ffn_gate_exps.weight",
                      {kInput, kFullMid, kTotalExperts}, 12u, gate_offset) &&
          gguf.tensor("blk.3.ffn_up_exps.weight",
                      {kInput, kFullMid, kTotalExperts}, 12u, up_offset) &&
          gguf.tensor("blk.3.ffn_down_exps.weight",
                      {kFullMid, kOutput, kTotalExperts}, 12u, down_offset) &&
          gguf.tensor("blk.3.ffn_gate_shexp.weight",
                      {kInput, kFullMid}, 8u, shared_gate_offset) &&
          gguf.tensor("blk.3.ffn_up_shexp.weight",
                      {kInput, kFullMid}, 8u, shared_up_offset) &&
          gguf.tensor("blk.3.ffn_down_shexp.weight",
                      {kFullMid, kOutput}, 8u, shared_down_offset),
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
    const char *dynamic_value = std::getenv("DS4_GLM5_ROUTER_MOE_DYNAMIC");
    CHECK(!bridge_value || (bridge_value[0] == '1' && bridge_value[1] == '\0'),
          "DS4_GLM5_ROUTER_MOE_BRIDGE must be exactly 1 when set");
    CHECK(!dynamic_value ||
              (dynamic_value[0] == '1' && dynamic_value[1] == '\0'),
          "DS4_GLM5_ROUTER_MOE_DYNAMIC must be exactly 1 when set");
    const bool router_bridge = bridge_value != nullptr;
    const bool router_dynamic = dynamic_value != nullptr;
    CHECK(!(router_bridge && router_dynamic),
          "router bridge and dynamic modes are mutually exclusive");
    const bool router_mode = router_bridge || router_dynamic;
    uint32_t jitter_seed = 0;
    CHECK(glm5_test_router_seed(jitter_seed),
          "valid DS4_GLM5_ROUTER_JITTER_SEED");
    std::vector<uint32_t> expert_ids(kExpertIds, kExpertIds + kUsed);
    float bridge_weights[kUsed] = {};
    const float *bridge_router = nullptr;
    const float *bridge_bias = nullptr;
    float bridge_scale = 0.0f;
    std::vector<float> route_sums(kTokens, 0.0f);

    std::vector<int32_t> selected((size_t)kTokens * kUsed);
    std::vector<float> route_weights((size_t)kTokens * kUsed);
    std::vector<uint32_t> real_route_ids((size_t)kTokens * kUsed);
    std::vector<float> input((size_t)kTokens * kInput);
    const char *ffn_input_path = std::getenv("DS4_GLM5_FFN_INPUT_F32");
    const char *ffn_split_path = std::getenv("DS4_GLM5_FFN_SPLIT_F32");
    const char *ffn_residual_path =
        std::getenv("DS4_GLM5_FFN_RESIDUAL_F32");
    const char *ffn_router_ids_path =
        std::getenv("DS4_GLM5_FFN_ROUTER_IDS_I32");
    const char *ffn_router_weights_path =
        std::getenv("DS4_GLM5_FFN_ROUTER_WEIGHTS_F32");
    const char *ffn_shared_output_path =
        std::getenv("DS4_GLM5_FFN_SHARED_OUTPUT_F32");
    const bool real_ffn_input = ffn_input_path != nullptr;
    CHECK((ffn_input_path == nullptr) == (ffn_split_path == nullptr) &&
          (ffn_input_path == nullptr) == (ffn_residual_path == nullptr) &&
          (ffn_input_path == nullptr) == (ffn_router_ids_path == nullptr) &&
          (ffn_input_path == nullptr) ==
              (ffn_router_weights_path == nullptr) &&
          (ffn_input_path == nullptr) ==
              (ffn_shared_output_path == nullptr),
          "real FFN input, mHC split, and router oracle travel together");
    std::vector<float> ffn_input, ffn_split, ffn_residual;
    std::vector<int32_t> ffn_router_ids;
    std::vector<float> ffn_router_weights;
    std::vector<float> ffn_shared_output;
    uint64_t ffn_split_hash = 0u;
    if (real_ffn_input) {
        CHECK(router_dynamic,
              "real FFN input requires production dynamic router mode");
        CHECK(read_exact_f32(ffn_input_path, kInput, ffn_input) &&
              read_exact_f32(ffn_split_path, 24u, ffn_split) &&
              read_exact_f32(ffn_residual_path, (size_t)kHc * kInput,
                             ffn_residual) &&
              read_exact_i32(ffn_router_ids_path, kUsed, ffn_router_ids) &&
              read_exact_f32(ffn_router_weights_path, kUsed,
                             ffn_router_weights) &&
              read_exact_f32(ffn_shared_output_path, kOutput,
                             ffn_shared_output),
              "read block-3 FFN state and independent router oracle");
        ffn_split_hash = fnv1a64(ffn_split.data(),
                                 ffn_split.size() * sizeof(float));
    }
    for (uint32_t token = 0; token < kTokens; ++token) {
        for (uint32_t i = 0; i < kInput; ++i) {
            if (real_ffn_input) {
                // The bounded decode gate needs one real route. Replicating
                // it across the batch keeps the existing hot-expert window
                // and scalar arithmetic controls while token zero remains the
                // exact attention-carried block-3 FFN state.
                input[(size_t)token * kInput + i] = ffn_input[i];
            } else if (router_mode) {
                input[(size_t)token * kInput + i] =
                    glm5_test_router_input(token, i, jitter_seed);
            } else {
                const int value = (int)((i * 17u + token * 13u) % 127u) - 63;
                input[(size_t)token * kInput + i] = (float)value / 256.0f;
            }
        }
    }
    uint64_t router_offset = 0, bias_offset = 0;
    if (router_mode) {
        bool normalize = false;
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
        bridge_bias =
            reinterpret_cast<const float *>(gguf.map + bias_offset);
    }

    const std::string wmma_min_count = std::to_string(kWmmaMinCount);
    CHECK(setenv("DS4_ROCM_Q4K_KSHARD_RESEARCH", "1", 1) == 0 &&
          setenv("DS4_ROCM_Q4K_WMMA_MIN_COUNT",
                 wmma_min_count.c_str(), 1) == 0 &&
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

    uint64_t gpu_router_id_mismatch = 0;
    OracleStats gpu_router_weight_oracle;

    if (router_bridge) {
        // The bounded bridge keeps one real route set. Token 0 supplies the
        // IDs and weights; all rows reuse it under a compact-slot permutation.
        uint32_t token_ids[kUsed];
        route_sums[0] = router_reference(
            bridge_router, bridge_bias, input.data(), token_ids,
            bridge_weights, bridge_scale);
        expert_ids.assign(token_ids, token_ids + kUsed);
        for (uint32_t token = 0; token < kTokens; ++token) {
            route_sums[token] = route_sums[0];
            for (uint32_t slot = 0; slot < kUsed; ++slot) {
                const uint32_t compact_slot = (slot * 5u + token * 3u) & 7u;
                const uint64_t pair = (uint64_t)token * kUsed + slot;
                selected[pair] = (int32_t)compact_slot;
                route_weights[pair] = bridge_weights[compact_slot];
                real_route_ids[pair] = expert_ids[compact_slot];
            }
        }
    } else if (router_dynamic) {
        // Use the production GPU router's actual IDs and weights. The CPU
        // implementation remains only an independent oracle for exact IDs and
        // bounded weights. The compact table uses deterministic first-
        // occurrence order and preserves production top-8 slot order.
        std::vector<uint32_t> cpu_route_ids((size_t)kTokens * kUsed);
        std::vector<float> cpu_route_weights((size_t)kTokens * kUsed);
        for (uint32_t token = 0; token < kTokens; ++token) {
            route_sums[token] = router_reference(
                bridge_router, bridge_bias,
                input.data() + (size_t)token * kInput,
                cpu_route_ids.data() + (size_t)token * kUsed,
                cpu_route_weights.data() + (size_t)token * kUsed,
                bridge_scale);
        }

        const uint64_t input_bytes =
            (uint64_t)input.size() * sizeof(float);
        const uint64_t logits_bytes =
            (uint64_t)kTokens * kTotalExperts * sizeof(float);
        const uint64_t selected_bytes =
            (uint64_t)kTokens * kUsed * sizeof(int32_t);
        const uint64_t weights_bytes =
            (uint64_t)kTokens * kUsed * sizeof(float);
        std::vector<int32_t> gpu_route_ids((size_t)kTokens * kUsed);
        std::vector<float> gpu_route_weights((size_t)kTokens * kUsed);
        {
            RouterTensors router;
            CHECK(upload(router.input, input.data(), input_bytes) &&
                  alloc_tensor(router.logits, logits_bytes) &&
                  alloc_tensor(router.probs, logits_bytes) &&
                  alloc_tensor(router.selected, selected_bytes) &&
                  alloc_tensor(router.weights, weights_bytes),
                  "allocate production GPU router bridge tensors");
            CHECK(ds4_gpu_matmul_f32_tensor(
                      &router.logits, gguf.map, gguf.size, router_offset,
                      kInput, kTotalExperts, &router.input, kTokens),
                  "execute production GLM5 router GEMM for MoE bridge");
            CHECK(ds4_gpu_glm_router_select_batch_tensor(
                      &router.selected, &router.weights, &router.probs,
                      gguf.map, gguf.size, bias_offset, &router.logits,
                      kTotalExperts, kUsed, bridge_scale, kTokens),
                  "execute production GLM5 top-8 selector for MoE bridge");
            CHECK(ds4_gpu_synchronize() &&
                  ds4_gpu_tensor_read(&router.selected, 0,
                                      gpu_route_ids.data(), selected_bytes) &&
                  ds4_gpu_tensor_read(&router.weights, 0,
                                      gpu_route_weights.data(), weights_bytes),
                  "read production GPU router IDs and weights");
        }

        for (uint64_t pair = 0; pair < (uint64_t)kTokens * kUsed; ++pair) {
            CHECK(gpu_route_ids[pair] >= 0 &&
                  (uint32_t)gpu_route_ids[pair] < kTotalExperts,
                  "production GPU route ID range");
            gpu_router_id_mismatch +=
                (uint32_t)gpu_route_ids[pair] != cpu_route_ids[pair];
            gpu_router_weight_oracle.add(
                gpu_route_weights[pair], cpu_route_weights[pair],
                2.0e-6, 2.0e-5);
        }
        CHECK(gpu_router_id_mismatch == 0,
              "production GPU and independent CPU router IDs agree");
        CHECK(gpu_router_weight_oracle.pass(),
              "production GPU router weights satisfy independent oracle");
        if (real_ffn_input) {
            OracleStats python_weight_oracle;
            for (uint32_t slot = 0; slot < kUsed; ++slot) {
                CHECK(gpu_route_ids[slot] == ffn_router_ids[slot],
                      "GPU route IDs match independent Python oracle");
                python_weight_oracle.add(
                    gpu_route_weights[slot], ffn_router_weights[slot],
                    2.0e-6, 2.0e-5);
            }
            CHECK(python_weight_oracle.pass(),
                  "GPU route weights match independent Python oracle");
            std::fprintf(stderr,
                "GLM5 real-state Python router oracle ids_exact=1 "
                "weights_bad=%llu max_abs=%.9g\n",
                (unsigned long long)python_weight_oracle.bad,
                python_weight_oracle.max_abs);
        }

        expert_ids.clear();
        for (uint32_t token = 0; token < kTokens; ++token) {
            for (uint32_t slot = 0; slot < kUsed; ++slot) {
                const uint64_t pair = (uint64_t)token * kUsed + slot;
                const uint32_t real_id = (uint32_t)gpu_route_ids[pair];
                auto found = std::find(expert_ids.begin(), expert_ids.end(),
                                       real_id);
                if (found == expert_ids.end()) {
                    expert_ids.push_back(real_id);
                    found = expert_ids.end() - 1;
                }
                selected[pair] = (int32_t)(found - expert_ids.begin());
                route_weights[pair] = gpu_route_weights[pair];
                real_route_ids[pair] = real_id;
            }
        }
        CHECK(expert_ids.size() >= kUsed &&
              expert_ids.size() <= kTotalExperts,
              "dynamic top-8 compact expert union size");
    } else {
        for (uint32_t token = 0; token < kTokens; ++token) {
            float sum = 0.0f;
            for (uint32_t slot = 0; slot < kUsed; ++slot) {
                const uint32_t compact_slot = (slot * 5u + token * 3u) & 7u;
                const uint64_t pair = (uint64_t)token * kUsed + slot;
                selected[pair] = (int32_t)compact_slot;
                const float value =
                    (float)(kUsed - slot + (token & 1u)) * 0.03125f;
                route_weights[pair] = value;
                sum += value;
            }
            for (uint32_t slot = 0; slot < kUsed; ++slot)
                route_weights[(size_t)token * kUsed + slot] /= sum;
        }
    }
    std::vector<uint32_t> compact_counts(expert_ids.size(), 0u);
    for (int32_t compact : selected) {
        CHECK(compact >= 0 && (size_t)compact < expert_ids.size(),
              "selected compact expert range");
        ++compact_counts[(uint32_t)compact];
    }
    OracleStats association_oracle;
    if (router_mode) {
        for (uint32_t token = 0; token < kTokens; ++token) {
            for (uint32_t slot = 0; slot < kUsed; ++slot) {
                const uint64_t pair = (uint64_t)token * kUsed + slot;
                const uint32_t compact_slot = (uint32_t)selected[pair];
                const uint32_t real_expert = expert_ids[compact_slot];
                const float *row = bridge_router + (uint64_t)real_expert * kInput;
                double logit = 0.0;
                const float *route_input = input.data() +
                    (router_dynamic ? (size_t)token * kInput : 0u);
                for (uint32_t column = 0; column < kInput; ++column)
                    logit += (double)row[column] * route_input[column];
                const float expected =
                    router_sigmoid((float)logit) / route_sums[token] *
                    bridge_scale;
                association_oracle.add(
                    route_weights[pair], expected,
                    router_dynamic ? 2.0e-6 : 1.0e-7,
                    router_dynamic ? 2.0e-5 : 1.0e-7);
            }
        }
        CHECK(association_oracle.pass(),
              "router expert-to-compact-slot weight association");
    }

    const char *exclusive_value =
        std::getenv("DS4_GLM5_TP_EXCLUSIVE_RANK_LOCAL");
    const char *window_value =
        std::getenv("DS4_GLM5_TP_WINDOW_CACHE");
    CHECK(!exclusive_value ||
              (exclusive_value[0] == '1' && exclusive_value[1] == '\0'),
          "DS4_GLM5_TP_EXCLUSIVE_RANK_LOCAL must be exactly 1 when set");
    CHECK(!window_value ||
              (window_value[0] == '1' && window_value[1] == '\0'),
          "DS4_GLM5_TP_WINDOW_CACHE must be exactly 1 when set");
    CHECK(!window_value || exclusive_value,
          "window-cache TP mode requires exclusive rank-local mode");
    if (exclusive_value) {
        const char *role = std::getenv("DS4_GLM5_TP_ROLE");
        CHECK(router_dynamic && role &&
              (std::strcmp(role, "leader") == 0 ||
               std::strcmp(role, "worker") == 0),
              "exclusive rank-local mode requires dynamic leader/worker");
        const uint32_t local_half =
            std::strcmp(role, "leader") == 0 ? 0u : 1u;
        const char *expected_value =
            std::getenv("DS4_GLM5_TP_EXPECT_COMPOSED_FNV");
        const char *expected_block_value =
            std::getenv("DS4_GLM5_TP_EXPECT_BLOCK_FNV");
        uint64_t expected_composed = 0;
        uint64_t expected_block = 0;
        CHECK(parse_fnv64(expected_value, expected_composed),
              "exclusive rank-local mode requires a 16-digit control FNV");
        CHECK(!real_ffn_input ||
                  parse_fnv64(expected_block_value, expected_block),
              "real FFN mode requires a 16-digit final block FNV");

        if (window_value) {
            CHECK(ds4_gpu_set_model_map(gguf.map, gguf.size) &&
                  ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map),
                  "bind same-GGUF map for bounded window loading");
            CHECK(declare_window_half(gguf, gate_offset, up_offset,
                                      down_offset, local_half),
                  "declare only the local rank-half expert windows");
            CHECK(ds4_gpu_q4k_packed_slice_bytes() == 0u,
                  "window declaration creates no packed table residency");
            ds4_gpu_q4k_window_cache_config cache_cfg = window_cache_config(
                gguf, gate_offset, up_offset, down_offset, local_half);
            WindowCacheGuard cache_guard;
            cache_guard.cache = ds4_gpu_q4k_window_cache_create(&cache_cfg);
            CHECK(cache_guard.cache,
                  "create bounded local rank-half window cache");

            std::vector<int32_t> token_ids(kUsed);
            std::vector<float> token_weights(kUsed);
            for (uint32_t slot = 0; slot < kUsed; ++slot) {
                token_ids[slot] = (int32_t)real_route_ids[slot];
                token_weights[slot] = route_weights[slot];
            }
            const uint64_t routed_ids_fnv = fnv1a64(
                token_ids.data(), token_ids.size() * sizeof(int32_t));
            const uint64_t route_weights_fnv = fnv1a64(
                token_weights.data(), token_weights.size() * sizeof(float));
            const uint64_t route_contract_hash = routed_ids_fnv ^
                ((route_weights_fnv << 1u) | (route_weights_fnv >> 63u)) ^
                ((ffn_split_hash << 7u) | (ffn_split_hash >> 57u));
            ComponentTensors tensors;
            const uint64_t pair_count = kUsed;
            const uint64_t mid_bytes = pair_count * kHalfMid * sizeof(float);
            const uint64_t down_bytes = pair_count * kOutput * sizeof(float);
            const uint64_t out_bytes = (uint64_t)kOutput * sizeof(float);
            CHECK(upload(tensors.selected, token_ids.data(),
                         pair_count * sizeof(int32_t)) &&
                  upload(tensors.weights, token_weights.data(),
                         pair_count * sizeof(float)) &&
                  upload(tensors.input, input.data(),
                         (uint64_t)kInput * sizeof(float)) &&
                  alloc_tensor(tensors.gate, mid_bytes) &&
                  alloc_tensor(tensors.up, mid_bytes) &&
                  alloc_tensor(tensors.mid, mid_bytes) &&
                  alloc_tensor(tensors.down, down_bytes) &&
                  alloc_tensor(tensors.out_full, out_bytes),
                  "allocate one-token cache-backed MoE tensors");
            CHECK(ds4_gpu_routed_moe_one_packed_q4k_window_tensor(
                      &tensors.out_full, &tensors.gate, &tensors.up,
                      &tensors.mid, &tensors.down, cache_guard.cache,
                      &tensors.selected, &tensors.weights, kUsed, kClamp,
                      &tensors.input, nullptr, 3u) &&
                  ds4_gpu_synchronize(),
                  "execute one-token bounded cache-backed rank half");
            std::vector<float> local_output(kOutput);
            CHECK(ds4_gpu_tensor_read(&tensors.out_full, 0u,
                                      local_output.data(), out_bytes),
                  "read one-token cache-backed rank output");
            std::vector<float> shared_local, local_combined;
            CHECK(!real_ffn_input ||
                      (run_shared_half(
                           gguf, shared_gate_offset, shared_up_offset,
                           shared_down_offset, ffn_input, local_half,
                           shared_local) &&
                       run_add2(local_output, shared_local,
                                local_combined)),
                  "fold rank-local shared expert into routed output");
            const std::vector<float> &exchange_output =
                real_ffn_input ? local_combined : local_output;
            ds4_gpu_q4k_window_cache_stats stats = {};
            CHECK(ds4_gpu_q4k_window_cache_get_stats(cache_guard.cache, &stats) &&
                  stats.slot_count == kUsed &&
                  stats.resident_count == kUsed &&
                  stats.prepares == 1u && stats.hits == 0u &&
                  stats.misses == kUsed && stats.fills == kUsed &&
                  stats.evictions == 0u &&
                  stats.capacity_bytes == 56623104u &&
                  ds4_gpu_q4k_packed_slice_bytes() == 0u,
                  "bounded cache accounting and zero full-table residency");
            const std::vector<float> empty;
            std::vector<float> roce_composed;
            CHECK(run_roce_composition(
                      gguf, empty, exchange_output, empty,
                      route_contract_hash, true, expected_composed, 1u,
                      &roce_composed),
                  "cache-backed one-token RoCE composition");
            uint64_t block_fnv = 0u;
            if (real_ffn_input) {
                std::vector<float> carried;
                CHECK(run_hc_block_carry(ffn_residual, ffn_split,
                                         roce_composed, carried),
                      "carry combined RoCE FFN output through final mHC");
                block_fnv = fnv1a64(
                    carried.data(),
                    (uint64_t)carried.size() * sizeof(float));
                CHECK(block_fnv == expected_block,
                      "RoCE-composed final block matches local control");
            }
            std::fprintf(stderr,
                "PASS same-GGUF GLM5 Q4_K window_cache=1 "
                "exclusive_rank_local=1 role=%s local_half=%u tokens=1 "
                "cache_bytes=%llu fills=%llu evictions=%llu "
                "packed_table_bytes=%llu shared_fold=%d "
                "block_carried_fnv=%016llx tp_roce=1\n",
                role, local_half,
                (unsigned long long)stats.capacity_bytes,
                (unsigned long long)stats.fills,
                (unsigned long long)stats.evictions,
                (unsigned long long)ds4_gpu_q4k_packed_slice_bytes(),
                real_ffn_input,
                (unsigned long long)block_fnv);
            cache_guard.reset();
            CHECK(ds4_gpu_q4k_packed_slice_bytes() == 0u,
                  "window destroy releases every packed slice byte");
            return true;
        }

        uint32_t distinct_route_sets = 0;
        for (uint32_t token = 0; token < kTokens; ++token) {
            bool seen = false;
            for (uint32_t prior = 0; prior < token && !seen; ++prior) {
                seen = std::memcmp(
                    real_route_ids.data() + (size_t)token * kUsed,
                    real_route_ids.data() + (size_t)prior * kUsed,
                    kUsed * sizeof(uint32_t)) == 0;
            }
            distinct_route_sets += !seen;
        }
        uint32_t hot_experts = 0, cold_experts = 0;
        for (uint32_t count : compact_counts) {
            hot_experts += count >= kWmmaMinCount;
            cold_experts += count != 0 && count < kWmmaMinCount;
        }
        CHECK((real_ffn_input ||
               (distinct_route_sets == kTokens && expert_ids.size() > kUsed &&
                hot_experts != 0 && cold_experts != 0)),
              "exclusive rank-local route diversity and hot/cold coverage");

        std::vector<uint8_t> local_gate, local_up, local_down;
        CHECK(gather_gate_half_direct(
                  gguf, gate_offset, expert_ids, full_gate_expert,
                  half_gate_expert, local_half, local_gate) &&
              gather_gate_half_direct(
                  gguf, up_offset, expert_ids, full_gate_expert,
                  half_gate_expert, local_half, local_up) &&
              gather_down_half_direct(
                  gguf, down_offset, expert_ids,
                  full_down_expert, full_down_row,
                  half_down_expert, half_down_row,
                  local_half, local_down),
              "gather only the rank-local 1024-column expert half");
        const uint32_t compact_expert_count = (uint32_t)expert_ids.size();
        const uint64_t table_bytes =
            (uint64_t)compact_expert_count * half_gate_expert;
        CHECK(local_gate.size() == table_bytes &&
              local_up.size() == table_bytes &&
              local_down.size() == table_bytes,
              "rank-local packed table byte contract");

        DeviceWeights device;
        CHECK(hipMalloc(&device.gate_half[local_half], table_bytes) == hipSuccess &&
              hipMalloc(&device.up_half[local_half], table_bytes) == hipSuccess &&
              hipMalloc(&device.down_half[local_half], table_bytes) == hipSuccess,
              "allocate only rank-local expert tables");
        CHECK(hipMemcpy(device.gate_half[local_half], local_gate.data(),
                        table_bytes, hipMemcpyHostToDevice) == hipSuccess &&
              hipMemcpy(device.up_half[local_half], local_up.data(),
                        table_bytes, hipMemcpyHostToDevice) == hipSuccess &&
              hipMemcpy(device.down_half[local_half], local_down.data(),
                        table_bytes, hipMemcpyHostToDevice) == hipSuccess,
              "upload only rank-local expert tables");

        ComponentTensors tensors;
        const uint64_t pair_count = (uint64_t)kTokens * kUsed;
        const uint64_t mid_bytes = pair_count * kHalfMid * sizeof(float);
        const uint64_t down_bytes = pair_count * kOutput * sizeof(float);
        const uint64_t out_bytes = (uint64_t)kTokens * kOutput * sizeof(float);
        CHECK(upload(tensors.selected, selected.data(),
                     pair_count * sizeof(int32_t)) &&
              upload(tensors.weights, route_weights.data(),
                     pair_count * sizeof(float)) &&
              upload(tensors.input, input.data(),
                     (uint64_t)input.size() * sizeof(float)) &&
              alloc_tensor(tensors.gate, mid_bytes) &&
              alloc_tensor(tensors.up, mid_bytes) &&
              alloc_tensor(tensors.mid, mid_bytes) &&
              alloc_tensor(tensors.down, down_bytes) &&
              alloc_tensor(tensors.out_full, out_bytes),
              "allocate exclusive rank-local component tensors");
        CHECK(launch(
                  tensors.out_full, tensors.gate, tensors.up, tensors.mid,
                  tensors.down, device.gate_half[local_half],
                  device.up_half[local_half], device.down_half[local_half],
                  half_gate_expert, gate_row_bytes,
                  half_down_expert, half_down_row,
                  tensors.selected, tensors.weights, tensors.input,
                  compact_expert_count, kHalfMid),
              "execute only the rank-local 1024-column Q4_K shard");
        std::vector<float> local_output((size_t)kTokens * kOutput);
        CHECK(ds4_gpu_tensor_read(&tensors.out_full, 0,
                                  local_output.data(), out_bytes),
              "read exclusive rank-local Q4_K output");
        const uint64_t routed_ids_fnv = fnv1a64(
            real_route_ids.data(), real_route_ids.size() * sizeof(uint32_t));
        const uint64_t route_weights_fnv = fnv1a64(
            route_weights.data(), route_weights.size() * sizeof(float));
        const uint64_t route_contract_hash =
            routed_ids_fnv ^
            ((route_weights_fnv << 1u) | (route_weights_fnv >> 63u)) ^
            ((ffn_split_hash << 7u) | (ffn_split_hash >> 57u));
        const std::vector<float> empty;
        CHECK(run_roce_composition(
                  gguf, empty, local_output, empty, route_contract_hash,
                  true, expected_composed),
              "exclusive rank-local RoCE composition");
        std::fprintf(stderr,
            "PASS same-GGUF GLM5 Q4_K exclusive_rank_local=1 role=%s "
            "local_half=%u compact_table=%u distinct_routes=%u "
            "hot_experts=%u cold_experts=%u "
            "gate_fnv=%016llx up_fnv=%016llx down_fnv=%016llx "
            "tp_roce=1\n",
            role, local_half, compact_expert_count, distinct_route_sets,
            hot_experts, cold_experts,
            (unsigned long long)fnv1a64(local_gate.data(), table_bytes),
            (unsigned long long)fnv1a64(local_up.data(), table_bytes),
            (unsigned long long)fnv1a64(local_down.data(), table_bytes));
        return true;
    }

    std::vector<uint8_t> gate_full, up_full, down_full;
    std::vector<uint8_t> gate_half[2], up_half[2], down_half[2];
    CHECK(gather_experts(gguf, gate_offset, expert_ids,
                         full_gate_expert, gate_full) &&
          gather_experts(gguf, up_offset, expert_ids,
                         full_gate_expert, up_full) &&
          gather_experts(gguf, down_offset, expert_ids,
                         full_down_expert, down_full),
          "gather compact real GLM5 expert union");
    for (uint32_t half = 0; half < 2; ++half) {
        pack_gate_half(gate_full, full_gate_expert, half_gate_expert,
                       half, gate_half[half]);
        pack_gate_half(up_full, full_gate_expert, half_gate_expert,
                       half, up_half[half]);
        pack_down_half(down_full, full_down_expert, full_down_row,
                       half_down_expert, half_down_row,
                       half, down_half[half]);
    }

    DeviceWeights device;
    CHECK(expert_ids.size() <= UINT32_MAX,
          "compact expert union fits runtime descriptor");
    const uint32_t compact_expert_count = (uint32_t)expert_ids.size();
    const uint64_t full_table_bytes =
        (uint64_t)compact_expert_count * full_gate_expert;
    const uint64_t half_table_bytes =
        (uint64_t)compact_expert_count * half_gate_expert;
    CHECK(gate_full.size() == full_table_bytes &&
          up_full.size() == full_table_bytes &&
          down_full.size() == full_table_bytes,
          "compact full table byte contract");
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
                 selected_gpu, weights_gpu, input_gpu,
                 compact_expert_count, kFullMid),
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
                 selected_gpu, weights_gpu, input_gpu,
                 compact_expert_count, kHalfMid) &&
          launch(out_half1, gate, up, mid, down,
                 device.gate_half[1], device.up_half[1], device.down_half[1],
                 half_gate_expert, gate_row_bytes,
                 half_down_expert, half_down_row,
                 selected_gpu, weights_gpu, input_gpu,
                 compact_expert_count, kHalfMid),
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

    // Decode uses a one-token route. Re-run both direct rank halves with an
    // actual one-token launch: merely hashing row zero of the 32-token result
    // is not equivalent because expert hot/cold dispatch counts are batch
    // scoped. This direct-table composition is the independent oracle for the
    // bounded cache-backed RoCE decode gate.
    CHECK(launch(out_half0, gate, up, mid, down,
                 device.gate_half[0], device.up_half[0], device.down_half[0],
                 half_gate_expert, gate_row_bytes,
                 half_down_expert, half_down_row,
                 selected_gpu, weights_gpu, input_gpu,
                 compact_expert_count, kHalfMid, 1u) &&
          launch(out_half1, gate, up, mid, down,
                 device.gate_half[1], device.up_half[1], device.down_half[1],
                 half_gate_expert, gate_row_bytes,
                 half_down_expert, half_down_row,
                 selected_gpu, weights_gpu, input_gpu,
                 compact_expert_count, kHalfMid, 1u) &&
          launch(out_full, gate, up, mid, down,
                 device.gate_full, device.up_full, device.down_full,
                 full_gate_expert, gate_row_bytes,
                 full_down_expert, full_down_row,
                 selected_gpu, weights_gpu, input_gpu,
                 compact_expert_count, kFullMid, 1u) &&
          ds4_gpu_add_tensor(&out_sum, &out_half0, &out_half1, kOutput) &&
          ds4_gpu_synchronize(),
          "execute independent one-token direct-table shard oracle");
    std::vector<float> one_token_full(kOutput);
    std::vector<float> one_token_candidate(kOutput);
    std::vector<float> one_token_half0(kOutput), one_token_half1(kOutput);
    std::vector<float> one_token_mid((size_t)kUsed * kFullMid);
    std::vector<Q8KBlock> one_token_mid_q8(
        (size_t)kUsed * (kFullMid / kQk));
    CHECK(ds4_gpu_tensor_read(&out_full, 0, one_token_full.data(),
                              (uint64_t)kOutput * sizeof(float)) &&
          ds4_gpu_tensor_read(&out_sum, 0, one_token_candidate.data(),
                              (uint64_t)kOutput * sizeof(float)) &&
          ds4_gpu_tensor_read(&out_half0, 0, one_token_half0.data(),
                              (uint64_t)kOutput * sizeof(float)) &&
          ds4_gpu_tensor_read(&out_half1, 0, one_token_half1.data(),
                              (uint64_t)kOutput * sizeof(float)) &&
          ds4_gpu_tensor_read(&mid, 0, one_token_mid.data(),
                              (uint64_t)one_token_mid.size() * sizeof(float)) &&
          ds4_gpu_tensor_read(
              &gate, 0, one_token_mid_q8.data(),
              (uint64_t)one_token_mid_q8.size() * sizeof(Q8KBlock)),
          "read full and split one-token direct-table oracles");
    CHECK(valid_q8k_capture(one_token_mid_q8),
          "one-token intermediate Q8_K scratch invariants");
    OracleStats one_token_composition;
    long double one_token_error_sq = 0.0L;
    long double one_token_reference_sq = 0.0L;
    double one_token_scaled_rel = 0.0;
    for (uint32_t row = 0; row < kOutput; ++row) {
        one_token_composition.add(one_token_candidate[row],
                                  one_token_full[row], 3.0e-7, 3.0e-7);
        const long double error =
            (long double)one_token_candidate[row] - one_token_full[row];
        one_token_scaled_rel = std::max(
            one_token_scaled_rel,
            (double)std::fabs(error) /
                std::max(1.0, std::fabs((double)one_token_full[row])));
        one_token_error_sq += error * error;
        one_token_reference_sq +=
            (long double)one_token_full[row] * one_token_full[row];
    }
    const double one_token_nmse = one_token_reference_sq == 0.0L
        ? (double)one_token_error_sq
        : (double)(one_token_error_sq / one_token_reference_sq);
    std::fprintf(stderr,
        "GLM5 one-token Q4_K full/split oracle bad=%llu nonfinite=%llu "
        "max_abs=%.9g max_rel=%.9g scaled_rel=%.9g nmse=%.9g "
        "full_fnv=%016llx "
        "split_fnv=%016llx\n",
        (unsigned long long)one_token_composition.bad,
        (unsigned long long)one_token_composition.nonfinite,
        one_token_composition.max_abs, one_token_composition.max_rel,
        one_token_scaled_rel, one_token_nmse,
        (unsigned long long)fnv1a64(
            one_token_full.data(), (uint64_t)kOutput * sizeof(float)),
        (unsigned long long)fnv1a64(
            one_token_candidate.data(), (uint64_t)kOutput * sizeof(float)));
    CHECK(one_token_composition.pass() &&
          one_token_reference_sq > 1.0e-12L &&
          one_token_composition.max_abs <= 3.0e-7 &&
          one_token_scaled_rel <= 3.0e-7 &&
          one_token_nmse <= 1.0e-11,
          "one-token 1024+1024 shards match full-2048 direct-table oracle");
    if (real_ffn_input) {
        constexpr uint32_t mid_rows[] =
            {0u, 1u, 255u, 511u, 1023u, 1024u, 1535u, 2047u};
        constexpr uint32_t out_rows[] =
            {0u, 1u, 127u, 1023u, 2047u, 3071u, 4095u};
        const Q8KBlock *token_q8 = production_input_q8.data();
        OracleStats cold_mid_oracle, cold_out_oracle;
        for (uint32_t slot = 0; slot < kUsed; ++slot) {
            const uint32_t compact = (uint32_t)selected[slot];
            const uint32_t real_expert = expert_ids[compact];
            for (uint32_t row : mid_rows) {
                const auto *gate_row = reinterpret_cast<const Q4KBlock *>(
                    gguf.map + gate_offset +
                    (uint64_t)real_expert * full_gate_expert +
                    (uint64_t)row * gate_row_bytes);
                const auto *up_row = reinterpret_cast<const Q4KBlock *>(
                    gguf.map + up_offset +
                    (uint64_t)real_expert * full_gate_expert +
                    (uint64_t)row * gate_row_bytes);
                float gate_ref = q4k_q8k_cold_reference(
                    gate_row, token_q8, kInput / kQk);
                float up_ref = q4k_q8k_cold_reference(
                    up_row, token_q8, kInput / kQk);
                gate_ref = std::min(gate_ref, kClamp);
                up_ref = std::max(-kClamp, std::min(up_ref, kClamp));
                const float expected =
                    (gate_ref / (1.0f + std::exp(-gate_ref))) * up_ref *
                    route_weights[slot];
                cold_mid_oracle.add(
                    one_token_mid[(size_t)slot * kFullMid + row], expected,
                    2.0e-6, 2.0e-5);
            }
        }
        for (uint32_t row : out_rows) {
            float expected = 0.0f;
            for (uint32_t slot = 0; slot < kUsed; ++slot) {
                const uint32_t compact = (uint32_t)selected[slot];
                const uint32_t real_expert = expert_ids[compact];
                const auto *down_row = reinterpret_cast<const Q4KBlock *>(
                    gguf.map + down_offset +
                    (uint64_t)real_expert * full_down_expert +
                    (uint64_t)row * full_down_row);
                expected += q4k_q81_reference(
                    down_row,
                    one_token_mid_q8.data() +
                        (size_t)slot * (kFullMid / kQk),
                    kFullMid / kQk);
            }
            // The one-token down leg uses a different integer-WMMA reduction
            // order than this scalar reconstruction. Its deliberately wider
            // envelope is a real-magnitude expert/row/sign/layout guard, not
            // a close-parity claim. Tight down arithmetic parity remains in
            // the diverse synthetic control below (1e-6 / 1e-5).
            cold_out_oracle.add(one_token_full[row], expected,
                                5.0e-4, 2.0e-3);
        }
        std::fprintf(stderr,
            "GLM5 real-state one-token cold scalar oracle mid=%llu "
            "mid_bad=%llu mid_max_abs=%.9g output=%llu output_bad=%llu "
            "output_max_abs=%.9g output_max_rel=%.9g "
            "output_gate_ratio=%.9g output_max_expected=%.9g\n",
            (unsigned long long)cold_mid_oracle.count,
            (unsigned long long)cold_mid_oracle.bad,
            cold_mid_oracle.max_abs,
            (unsigned long long)cold_out_oracle.count,
            (unsigned long long)cold_out_oracle.bad,
            cold_out_oracle.max_abs, cold_out_oracle.max_rel,
            cold_out_oracle.max_gate_ratio,
            cold_out_oracle.max_abs_expected);
        CHECK(cold_mid_oracle.pass(),
              "real-state one-token cold Q4_K scalar mid parity");
        CHECK(cold_out_oracle.pass(),
              "real-state one-token down layout/sign envelope");
    }
    uint64_t token0_ffn_combined_fnv = 0u;
    uint64_t token0_block_fnv = 0u;
    if (real_ffn_input) {
        std::vector<float> shared_half0, shared_half1, shared_sum;
        std::vector<float> partial0, partial1, combined, carried;
        CHECK(run_shared_half(gguf, shared_gate_offset, shared_up_offset,
                              shared_down_offset, ffn_input, 0u,
                              shared_half0) &&
              run_shared_half(gguf, shared_gate_offset, shared_up_offset,
                              shared_down_offset, ffn_input, 1u,
                              shared_half1) &&
              run_add2(shared_half0, shared_half1, shared_sum) &&
              run_add2(one_token_half0, shared_half0, partial0) &&
              run_add2(one_token_half1, shared_half1, partial1) &&
              run_add2(partial0, partial1, combined) &&
              run_hc_block_carry(ffn_residual, ffn_split, combined, carried),
              "compose real shared+routed FFN and final mHC carry");
        OracleStats shared_python_oracle;
        for (uint32_t row = 0; row < kOutput; ++row) {
            shared_python_oracle.add(shared_sum[row], ffn_shared_output[row],
                                     2.0e-5, 2.0e-5);
        }
        CHECK(shared_python_oracle.pass(),
              "shared rank halves match independent Python full output");
        token0_ffn_combined_fnv = fnv1a64(
            combined.data(), (uint64_t)combined.size() * sizeof(float));
        token0_block_fnv = fnv1a64(
            carried.data(), (uint64_t)carried.size() * sizeof(float));
        std::fprintf(stderr,
            "GLM5 real-state block-3 local control "
            "ffn_combined_fnv=%016llx block_carried_fnv=%016llx "
            "shared0_fnv=%016llx shared1_fnv=%016llx "
            "shared_sum_fnv=%016llx shared_python_bad=%llu "
            "shared_python_max_abs=%.9g\n",
            (unsigned long long)token0_ffn_combined_fnv,
            (unsigned long long)token0_block_fnv,
            (unsigned long long)fnv1a64(
                shared_half0.data(),
                (uint64_t)shared_half0.size() * sizeof(float)),
            (unsigned long long)fnv1a64(
                shared_half1.data(),
                (uint64_t)shared_half1.size() * sizeof(float)),
            (unsigned long long)fnv1a64(
                shared_sum.data(),
                (uint64_t)shared_sum.size() * sizeof(float)),
            (unsigned long long)shared_python_oracle.bad,
            shared_python_oracle.max_abs);
    }

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
    // Dynamic production-router weights can place an intermediate exactly on
    // a host/GPU F32 quantizer rounding boundary. Keep the compact values and
    // effective half scales exact, bound that diagnostic to at most two of
    // 16,384 effective sums, and use the captured production blocks below.
    CHECK(!router_mode ||
          (mid_q81.changed_values == 0 && mid_q81.changed_scales == 0 &&
           mid_q81.changed_sums <=
               (router_dynamic ? kDynamicMidSumRoundingBudget : 0u)),
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
    uint64_t mid_oracle_hot = 0, mid_oracle_cold = 0;
    std::vector<float> oracle_values;
    std::vector<float> oracle_actual_values;
    for (uint32_t token : oracle_tokens) {
        // Use the independently inspected production activation capture for
        // arithmetic validation. The host quantizer comparison above remains
        // the separate quantization oracle; the cold path consumes its F32
        // scale, so an effective-F16-only host substitute is insufficient.
        const Q8KBlock *token_q8 =
            production_input_q8.data() + (size_t)token * (kInput / kQk);
        for (uint32_t slot = 0; slot < kUsed; ++slot) {
            const uint64_t pair = (uint64_t)token * kUsed + slot;
            const uint32_t expert = (uint32_t)selected[pair];
            const bool hot_wmma =
                compact_counts[expert] >= kWmmaMinCount;
            for (uint32_t row : oracle_mid_rows) {
                const uint32_t real_expert = expert_ids[expert];
                const auto *gate_row = reinterpret_cast<const Q4KBlock *>(
                    gguf.map + gate_offset +
                    (uint64_t)real_expert * full_gate_expert +
                    (uint64_t)row * gate_row_bytes);
                const auto *up_row = reinterpret_cast<const Q4KBlock *>(
                    gguf.map + up_offset +
                    (uint64_t)real_expert * full_gate_expert +
                    (uint64_t)row * gate_row_bytes);
                const auto reference_dot = hot_wmma
                    ? q4k_q81_reference : q4k_q8k_cold_reference;
                float gate_ref = reference_dot(
                    gate_row, token_q8, kInput / kQk);
                float up_ref = reference_dot(
                    up_row, token_q8, kInput / kQk);
                mid_oracle_hot += hot_wmma;
                mid_oracle_cold += !hot_wmma;
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
                const uint32_t real_expert = expert_ids[expert];
                const auto *down_row = reinterpret_cast<const Q4KBlock *>(
                    gguf.map + down_offset +
                    (uint64_t)real_expert * full_down_expert +
                    (uint64_t)row * full_down_row);
                expected += q4k_q81_reference(
                    down_row,
                    production_mid_q8.data() + pair * (kFullMid / kQk),
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
        "hot=%llu cold=%llu "
        "max_abs=%.9g max_abs_actual=%.9g max_abs_expected=%.9g "
        "max_rel=%.9g max_gate_ratio=%.9g "
        "output=%llu bad=%llu max_abs=%.9g max_rel=%.9g "
        "max_gate_ratio=%.9g reference_fnv=%016llx actual_fnv=%016llx\n",
        (unsigned long long)mid_oracle.count,
        (unsigned long long)mid_oracle.bad,
        (unsigned long long)mid_oracle_hot,
        (unsigned long long)mid_oracle_cold,
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
    const uint64_t expert_ids_fnv =
        fnv1a64(expert_ids.data(), expert_ids.size() * sizeof(uint32_t));
    const uint64_t token0_ids_fnv = router_mode
        ? fnv1a64(expert_ids.data(), kUsed * sizeof(uint32_t)) : 0u;
    const uint64_t routed_ids_fnv = router_mode
        ? fnv1a64(real_route_ids.data(),
                  real_route_ids.size() * sizeof(uint32_t)) : 0u;
    const uint64_t route_weights_fnv =
        fnv1a64(route_weights.data(), route_weights.size() * sizeof(float));
    const uint64_t token0_shard_fnv = fnv1a64(
        one_token_candidate.data(), (uint64_t)kOutput * sizeof(float));
    uint32_t distinct_route_sets = router_bridge ? 1u : 0u;
    if (router_dynamic) {
        for (uint32_t token = 0; token < kTokens; ++token) {
            bool seen = false;
            for (uint32_t prior = 0; prior < token && !seen; ++prior) {
                seen = std::memcmp(
                    real_route_ids.data() + (size_t)token * kUsed,
                    real_route_ids.data() + (size_t)prior * kUsed,
                    kUsed * sizeof(uint32_t)) == 0;
            }
            distinct_route_sets += !seen;
        }
    }
    const char *route_source = router_dynamic ? "production-gpu-per-token" :
                               router_bridge ? "token0" : "synthetic";
    std::fprintf(stderr,
        "GLM5 Q4_K expert set router_bridge=%d router_dynamic=%d "
        "seed_active=%d seed=%u compact_count=%u "
        "token0_ids=%u,%u,%u,%u,%u,%u,%u,%u "
        "compact_ids_fnv=%016llx token0_ids_fnv=%016llx "
        "routed_ids_fnv=%016llx "
        "route_source=%s distinct_route_sets=%u gpu_router_id_mismatch=%llu "
        "gpu_router_weight_bad=%llu gpu_router_weight_max_abs=%.9g "
        "association_pairs=%llu "
        "route_weights_fnv=%016llx association_bad=%llu "
        "association_max_abs=%.9g\n",
        router_bridge, router_dynamic, router_mode, jitter_seed,
        compact_expert_count,
        expert_ids[0], expert_ids[1], expert_ids[2],
        expert_ids[3], expert_ids[4], expert_ids[5], expert_ids[6],
        expert_ids[7], (unsigned long long)expert_ids_fnv,
        (unsigned long long)token0_ids_fnv,
        (unsigned long long)routed_ids_fnv,
        route_source, distinct_route_sets,
        (unsigned long long)gpu_router_id_mismatch,
        (unsigned long long)gpu_router_weight_oracle.bad,
        gpu_router_weight_oracle.max_abs,
        (unsigned long long)association_oracle.count,
        (unsigned long long)route_weights_fnv,
        (unsigned long long)association_oracle.bad,
        association_oracle.max_abs);
    std::fprintf(stderr,
        "GLM5 Q4_K shard split-consistency layer=3 descriptor_experts=288 "
        "compact_table=%u router_bridge=%d router_dynamic=%d "
        "tokens=%u changed_telemetry=%llu nonfinite=%llu "
        "independent_halves=%d max_abs=%.9g max_rel=%.9g nmse=%.9g "
        "gate_weight_fnv=%016llx up_weight_fnv=%016llx "
        "down_weight_fnv=%016llx full_fnv=%016llx shard_fnv=%016llx "
        "token0_full_fnv=%016llx token0_shard_fnv=%016llx "
        "token0_max_abs=%.9g token0_max_rel=%.9g token0_nmse=%.9g\n",
        compact_expert_count, router_bridge, router_dynamic, kTokens,
        (unsigned long long)changed,
        (unsigned long long)nonfinite,
        independent_halves, max_abs, max_rel, nmse,
        (unsigned long long)gate_weight_fnv,
        (unsigned long long)up_weight_fnv,
        (unsigned long long)down_weight_fnv,
        (unsigned long long)fnv1a64(reference.data(), out_bytes),
        (unsigned long long)fnv1a64(candidate.data(), out_bytes),
        (unsigned long long)fnv1a64(
            one_token_full.data(), (uint64_t)kOutput * sizeof(float)),
        (unsigned long long)token0_shard_fnv,
        one_token_composition.max_abs, one_token_scaled_rel,
        one_token_nmse);
    CHECK(!router_dynamic || real_ffn_input || distinct_route_sets == kTokens,
          "dynamic mode requires a distinct real route set per token");
    CHECK(!router_dynamic || real_ffn_input || compact_expert_count > kUsed,
          "dynamic mode requires a real multi-token expert union");
    CHECK(!router_dynamic || real_ffn_input ||
              (mid_oracle_hot != 0 && mid_oracle_cold != 0),
          "dynamic mode samples both hot-WMMA and cold-DP4A experts");
    CHECK(pass, "same-GGUF 1024/1024 top-8 numerical envelope");
    CHECK(oracle_pass,
          "sampled same-GGUF scalar mid and conditioned-down arithmetic oracle");
    CHECK(!std::getenv("DS4_GLM5_TP_ROLE") || router_dynamic,
          "bounded TP RoCE composition requires dynamic GPU routing");
    const uint64_t route_contract_hash =
        routed_ids_fnv ^
        ((route_weights_fnv << 1u) | (route_weights_fnv >> 63u)) ^
        ((ffn_split_hash << 7u) | (ffn_split_hash >> 57u));
    CHECK(run_roce_composition(gguf, reference, half0, half1,
                               route_contract_hash),
          "bounded TP RoCE composition");
    std::fprintf(stderr,
                 "PASS same-GGUF GLM5 Q4_K router_bridge=%d "
                 "router_dynamic=%d "
                 "route_source=%s tp_roce=%d sampled "
                 "scalar-mid, conditioned-down, and top-8 shard "
                 "split-consistency\n",
                 router_bridge, router_dynamic, route_source,
                 std::getenv("DS4_GLM5_TP_ROLE") != nullptr);
    return true;
}

}  // namespace

int main() { return run_test() ? 0 : 1; }
