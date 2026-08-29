#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "tests/glm5_gguf_test.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(expr, message) do {                                         \
    if (!(expr)) {                                                        \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__); \
        return false;                                                     \
    }                                                                     \
} while (0)

static float f16_to_f32(uint16_t value) {
    const uint32_t sign = (uint32_t)(value & 0x8000u) << 16u;
    uint32_t exponent = (value >> 10u) & 0x1fu;
    uint32_t fraction = value & 0x03ffu;
    uint32_t bits = 0u;
    if (exponent == 0u) {
        if (fraction == 0u) bits = sign;
        else {
            int shift = 0;
            while ((fraction & 0x0400u) == 0u) { fraction <<= 1u; ++shift; }
            fraction &= 0x03ffu;
            bits = sign | (uint32_t)(113 - shift) << 23u | fraction << 13u;
        }
    } else if (exponent == 31u) {
        bits = sign | 0x7f800000u | fraction << 13u;
    } else {
        bits = sign | (exponent + 112u) << 23u | fraction << 13u;
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static void scale_min(uint32_t group, const uint8_t *scales,
                      uint8_t &scale, uint8_t &minimum) {
    if (group < 4u) {
        scale = scales[group] & 63u;
        minimum = scales[group + 4u] & 63u;
    } else {
        scale = (scales[group + 4u] & 0x0fu) |
                ((scales[group - 4u] >> 6u) << 4u);
        minimum = (scales[group + 4u] >> 4u) |
                  ((scales[group] >> 6u) << 4u);
    }
}

static float q4k_dot(const uint8_t *row, const float *x, uint32_t width) {
    float sum = 0.0f;
    for (uint32_t block = 0u; block < width / 256u; ++block) {
        const uint8_t *w = row + (uint64_t)block * 144u;
        uint16_t d_bits = 0u, dmin_bits = 0u;
        std::memcpy(&d_bits, w, 2u);
        std::memcpy(&dmin_bits, w + 2u, 2u);
        const float d = f16_to_f32(d_bits);
        const float dmin = f16_to_f32(dmin_bits);
        const uint8_t *scales = w + 4u;
        const uint8_t *quant = w + 16u;
        for (uint32_t group = 0u; group < 8u; ++group) {
            uint8_t sc = 0u, mn = 0u;
            scale_min(group, scales, sc, mn);
            const uint32_t qbase = (group >> 1u) * 32u;
            const uint32_t shift = (group & 1u) ? 4u : 0u;
            for (uint32_t lane = 0u; lane < 32u; ++lane) {
                const uint32_t q = (quant[qbase + lane] >> shift) & 0x0fu;
                const float weight = d * (float)sc * (float)q -
                                     dmin * (float)mn;
                sum += weight * x[(uint64_t)block * 256u + group * 32u + lane];
            }
        }
    }
    return sum;
}

static bool run_test(void) {
    constexpr uint32_t width = 4096u, rows = 64u;
    const char *model = std::getenv("DS4_GLM5_MODEL");
    CHECK(model && model[0], "model environment");
    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM5 GGUF");
    uint64_t base = 0u;
    uint32_t full_rows = 0u;
    if (gguf.tensor("blk.0.kda_q.weight", {width, 8192u}, 12u, base)) {
        full_rows = 8192u;
    } else {
        /* Q4 control files keep KDA in BF16; exercise the same compact Q4_K
         * loader against the first routed expert instead.  The first expert
         * is a contiguous 4096x2048 row slice of the 3-D GGUF tensor. */
        CHECK(gguf.tensor("blk.3.ffn_gate_exps.weight",
                          {width, 2048u, 288u}, 12u, base),
              "bind Q4_K routed expert control");
        full_rows = 2048u;
    }
    ds4_gpu_config config = {};
    config.n_gpus = 1u;
    config.device_indices[0] = 0u;
    CHECK(ds4_gpu_init_multi(&config) &&
          ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map) &&
          ds4_gpu_set_model_map(gguf.map, gguf.size),
          "initialize gfx1151 model mapping");
    std::vector<float> x(width);
    for (uint32_t i = 0u; i < width; ++i)
        x[i] = ((int)((i * 73u + 19u) % 509u) - 254) / 1024.0f;
    ds4_gpu_tensor *x_gpu = ds4_gpu_tensor_alloc((uint64_t)width * sizeof(float));
    ds4_gpu_tensor *out_gpu = ds4_gpu_tensor_alloc((uint64_t)rows * sizeof(float));
    CHECK(x_gpu && out_gpu &&
          ds4_gpu_tensor_write(x_gpu, 0u, x.data(),
                               (uint64_t)width * sizeof(float)),
          "upload deterministic activation");
    const uint64_t row_bytes = (width / 256u) * 144u;
    for (uint32_t half = 0u; half < 2u; ++half) {
        const uint32_t row0 = half * (full_rows / 2u);
        CHECK(ds4_gpu_matmul_q4_k_tensor(
                  out_gpu, gguf.map, gguf.size,
                  base + (uint64_t)row0 * row_bytes,
                  width, rows, x_gpu, 1u) && ds4_gpu_synchronize(),
              "execute compact Q4_K row slice");
        std::vector<float> got(rows);
        CHECK(ds4_gpu_tensor_read(out_gpu, 0u, got.data(),
                                  (uint64_t)rows * sizeof(float)),
              "read projection result");
        double se = 0.0, ref2 = 0.0, dot = 0.0, got2 = 0.0;
        for (uint32_t row = 0u; row < rows; ++row) {
            const uint8_t *weight = gguf.map + base +
                (uint64_t)(row0 + row) * row_bytes;
            const float want = q4k_dot(weight, x.data(), width);
            CHECK(std::isfinite(got[row]), "finite Q4_K projection");
            const double error = (double)got[row] - want;
            se += error * error;
            ref2 += (double)want * want;
            dot += (double)want * got[row];
            got2 += (double)got[row] * got[row];
        }
        const double nrmse = std::sqrt(se / ref2);
        const double cosine = dot / std::sqrt(ref2 * got2);
        std::fprintf(stderr, "half=%u nrmse=%.9g cosine=%.12g\n",
                     half, nrmse, cosine);
        CHECK(nrmse < 2.0e-6 && cosine > 0.999999999,
              "Q4_K projection matches CPU oracle");
    }
    ds4_gpu_tensor_free(out_gpu);
    ds4_gpu_tensor_free(x_gpu);
    ds4_gpu_cleanup();
    std::fprintf(stderr, "PASS same-GGUF GLM5 compact Q4_K KDA projection\n");
    return true;
}

int main(void) { return run_test() ? 0 : 1; }
