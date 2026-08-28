#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
extern "C" {
#include "ds4_tp.h"
}

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#define CHECK(expr, message) do {                                           \
    if (!(expr)) {                                                          \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__);    \
        return false;                                                       \
    }                                                                       \
} while (0)

namespace {

uint32_t float_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float bits_float(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint16_t reference_bf16_bits(float value) {
    const uint32_t bits = float_bits(value);
    const uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u)
        return (uint16_t)((bits >> 16u) | 0x0040u);
    return (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

uint32_t reference_output_bits(float value, float scale) {
    const float widened = bits_float((uint32_t)reference_bf16_bits(value) << 16u);
    return float_bits(widened * scale);
}

bool run_once(const std::vector<uint32_t> &input_bits, float scale,
              std::vector<uint32_t> &output_bits) {
    std::vector<float> input(input_bits.size());
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = bits_float(input_bits[i]);
    ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(input.size() * sizeof(float));
    CHECK(tensor &&
          ds4_gpu_tensor_write(tensor, 0u, input.data(),
                               input.size() * sizeof(float)) &&
          ds4_gpu_round_bf16_inplace_tensor(tensor, input.size(), scale) &&
          ds4_gpu_synchronize(), "execute BF16 RNE boundary");
    std::vector<float> output(input.size());
    CHECK(ds4_gpu_tensor_read(tensor, 0u, output.data(),
                              output.size() * sizeof(float)),
          "read BF16 RNE boundary");
    output_bits.resize(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        output_bits[i] = float_bits(output[i]);
        const uint32_t expected = reference_output_bits(input[i], scale);
        if (std::isnan(bits_float(expected)) && scale != 1.0f) {
            CHECK(std::isnan(output[i]) &&
                  (output_bits[i] >> 31u) == (expected >> 31u),
                  "scaled BF16 NaN remains signed NaN");
        } else {
            if (output_bits[i] != expected)
                std::fprintf(stderr,
                    "BF16 mismatch i=%zu input=%08x scale=%g got=%08x expected=%08x\n",
                    i, input_bits[i], scale, output_bits[i], expected);
            CHECK(output_bits[i] == expected, "bit-exact BF16 RNE output");
        }
    }
    CHECK(ds4_gpu_round_bf16_inplace_tensor(tensor, 0u, scale) &&
          !ds4_gpu_round_bf16_inplace_tensor(
              tensor, input.size() + 1u, scale) &&
          !ds4_gpu_round_bf16_inplace_tensor(
              tensor, input.size(), std::numeric_limits<float>::infinity()) &&
          !ds4_gpu_round_bf16_inplace_tensor(nullptr, 1u, 1.0f),
          "BF16 RNE wrapper fails closed");
    ds4_gpu_tensor_free(tensor);
    return true;
}

bool run_test() {
    const std::vector<uint32_t> values = {
        0x00000000u, 0x80000000u,  // +0, -0
        0x3f800000u, 0xbf800000u,  // exact +1, -1
        0x3f808000u, 0x3f818000u,  // ties: even down, odd up
        0xbf808000u, 0xbf818000u,
        0x3f807fffu, 0x3f808001u,  // immediately around a tie
        0x00000001u, 0x00008000u, 0x00008001u,
        0x00010000u, 0x007f0000u, 0x007fffffu,  // BF16 subnormal boundaries
        0x80008001u,  // negative minimum BF16 subnormal after rounding
        0x00800000u, 0x7f7fffffu,  // minimum normal, max finite -> inf
        0x7f800000u, 0xff800000u,  // infinities
        0x7f800001u, 0x7fc12345u, 0xff812345u,  // NaNs/payloads
    };
    ds4_gpu_config config = {};
    config.n_gpus = 1u;
    config.device_indices[0] = 0u;
    CHECK(ds4_gpu_init_multi(&config), "initialize gfx1151");
    ds4_tp_test_reset_exchange_calls();
    std::vector<uint32_t> first, second, scaled;
    CHECK(run_once(values, 1.0f, first) &&
          run_once(values, 1.0f, second) && first == second,
          "BF16 RNE is deterministic");
    /* The test links without crtfastmath host FTZ so the CPU oracle also pins
     * scaled F32 subnormal results produced from BF16 subnormals. */
    CHECK(run_once(values, 0.1767766952966369f, scaled),
          "BF16 RNE applies post-round F32 scale");
    CHECK(ds4_tp_test_get_exchange_calls() == 0u,
          "BF16 RNE invokes no TP exchange API");
    ds4_gpu_cleanup();
    std::fprintf(stderr,
                 "PASS bit-exact GLM5 BF16 RNE boundary gate (%zu values)\n",
                 values.size());
    return true;
}

}  // namespace

int main() { return run_test() ? 0 : 1; }
