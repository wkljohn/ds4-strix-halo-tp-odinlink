/* Regression test for the production DeepSeek-V4 mixed-KV decode attention.
 *
 * The non-RoPE 448 values are rounded through block-scaled E4M3 and the
 * RoPE tail is rounded to F16 before attention sees the raw cache.  Generate
 * that cache through the real GPU writers, verify its layout/precision, then
 * compare the default old-HIP attention kernel with an independent host
 * softmax oracle.  A wrapped ring and a masked compressed row make the test
 * sensitive to row order, sinks, and mask handling. */

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
        fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);           \
        return 1;                                                            \
    }                                                                        \
} while (0)

typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

static float e4m3_value(int i) {
    const int exp = (i >> 3) & 15;
    const int mant = i & 7;
    if (exp == 0) return (float)mant * 0.001953125f;
    return (1.0f + (float)mant * 0.125f) * exp2f((float)exp - 7.0f);
}

static float e4m3_round(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (e4m3_value(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        const float bd = fabsf(ax - e4m3_value(best));
        const float nd = fabsf(ax - e4m3_value(best + 1));
        if (nd < bd || (nd == bd && ((best + 1) & 1) == 0 && (best & 1))) {
            best++;
        }
    }
    return sign * e4m3_value(best);
}

/* Match f32_to_f16_bits_hip_round used by store_raw_kv_batch_kernel. */
static uint16_t f16_bits_store_round(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    const uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = u & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

static float f16_from_bits(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 31u;
    uint32_t mant = h & 1023u;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) u = sign;
        else {
            int shift = 0;
            while ((mant & 0x400u) == 0u) { mant <<= 1; shift++; }
            mant &= 0x3ffu;
            u = sign | ((uint32_t)(127 - 14 - shift) << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        u = sign | 0x7f800000u | (mant << 13);
    } else {
        u = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static void mixed_cache_round(float *row, uint32_t head_dim, uint32_t n_rot) {
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t off = 0; off < n_nope; off += 64u) {
        float amax = 1.0e-4f;
        for (uint32_t i = 0; i < 64u; i++) amax = fmaxf(amax, fabsf(row[off + i]));
        const float scale = exp2f(ceilf(log2f(amax / 448.0f)));
        for (uint32_t i = 0; i < 64u; i++) {
            row[off + i] = e4m3_round(fminf(448.0f,
                                            fmaxf(-448.0f, row[off + i] / scale))) * scale;
        }
    }
    for (uint32_t d = 0; d < head_dim; d++) {
        row[d] = f16_from_bits(f16_bits_store_round(row[d]));
    }
}

static void attention_reference(std::vector<float> &out,
                                const std::vector<float> &q,
                                const std::vector<float> &raw,
                                const std::vector<float> &comp,
                                const std::vector<float> &mask,
                                const std::vector<float> &sinks,
                                uint32_t n_raw, uint32_t raw_cap,
                                uint32_t raw_start, uint32_t n_comp,
                                uint32_t n_head, uint32_t head_dim) {
    const double scale = 1.0 / sqrt((double)head_dim);
    std::vector<double> scores((size_t)n_raw + n_comp);
    for (uint32_t h = 0; h < n_head; h++) {
        double max_score = sinks[h];
        for (uint32_t r = 0; r < n_raw; r++) {
            const uint32_t row = (raw_start + r) % raw_cap;
            double dot = 0.0;
            for (uint32_t d = 0; d < head_dim; d++) {
                dot += (double)q[(size_t)h * head_dim + d] *
                       raw[(size_t)row * head_dim + d];
            }
            scores[r] = dot * scale;
            max_score = fmax(max_score, scores[r]);
        }
        for (uint32_t c = 0; c < n_comp; c++) {
            double score = -3.4e38;
            if (mask[c] > -5.0e29f) {
                double dot = 0.0;
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot += (double)q[(size_t)h * head_dim + d] *
                           comp[(size_t)c * head_dim + d];
                }
                score = dot * scale + mask[c];
            }
            scores[n_raw + c] = score;
            max_score = fmax(max_score, score);
        }
        double denom = exp((double)sinks[h] - max_score);
        for (double &score : scores) {
            score = exp(score - max_score);
            denom += score;
        }
        for (uint32_t d = 0; d < head_dim; d++) {
            double acc = 0.0;
            for (uint32_t r = 0; r < n_raw; r++) {
                const uint32_t row = (raw_start + r) % raw_cap;
                acc += scores[r] * (double)raw[(size_t)row * head_dim + d];
            }
            for (uint32_t c = 0; c < n_comp; c++) {
                acc += scores[n_raw + c] * (double)comp[(size_t)c * head_dim + d];
            }
            out[(size_t)h * head_dim + d] = (float)(acc / denom);
        }
    }
}

static double rel_rms(const std::vector<float> &a, const std::vector<float> &b) {
    double diff2 = 0.0, ref2 = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = (double)a[i] - b[i];
        diff2 += d * d;
        ref2 += (double)b[i] * b[i];
    }
    return sqrt(diff2 / fmax(ref2, 1.0e-30));
}

static double max_scaled_error(const std::vector<float> &a,
                               const std::vector<float> &b) {
    double max_diff = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        max_diff = fmax(max_diff, fabs((double)a[i] - b[i]));
        max_ref = fmax(max_ref, fabs((double)b[i]));
    }
    return max_diff / fmax(max_ref, 1.0e-30);
}

int main(void) {
    constexpr uint32_t head_dim = 512, n_rot = 64, n_head = 4;
    constexpr uint32_t raw_cap = 7, n_raw = 5, raw_start = 5, n_comp = 3;
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0, "ROCm device required");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm backend");

    std::vector<float> sinks(n_head), q((size_t)n_head * head_dim);
    std::vector<float> kv_in((size_t)n_raw * head_dim);
    std::vector<float> raw((size_t)raw_cap * head_dim, -77.0f);
    std::vector<float> raw_expected = raw;
    std::vector<float> raw_unquantized = raw;
    std::vector<float> comp((size_t)n_comp * head_dim);
    std::vector<float> mask = {0.0f, -1.0e30f, -0.35f};
    for (uint32_t h = 0; h < n_head; h++) {
        sinks[h] = -0.3f + 0.11f * (float)h;
        for (uint32_t d = 0; d < head_dim; d++) {
            q[(size_t)h * head_dim + d] =
                0.055f * sinf((float)(1u + h * 17u + d * 3u) * 0.071f) +
                0.013f * cosf((float)(d + h * 5u) * 0.019f);
        }
    }
    for (uint32_t r = 0; r < n_raw; r++) {
        float *row = kv_in.data() + (size_t)r * head_dim;
        for (uint32_t d = 0; d < head_dim; d++) {
            row[d] = 0.37f * sinf((float)(d * 5u + r * 101u + 3u) * 0.013f) +
                     0.021f * (float)((int)r - 2);
        }
        std::vector<float> expected(row, row + head_dim);
        mixed_cache_round(expected.data(), head_dim, n_rot);
        const uint32_t physical = (raw_start + r) % raw_cap;
        memcpy(raw_expected.data() + (size_t)physical * head_dim,
               expected.data(), head_dim * sizeof(float));
        std::vector<float> plain(row, row + head_dim);
        for (float &v : plain) v = f16_from_bits(f16_bits_store_round(v));
        memcpy(raw_unquantized.data() + (size_t)physical * head_dim,
               plain.data(), head_dim * sizeof(float));
    }
    for (uint32_t c = 0; c < n_comp; c++) {
        float *row = comp.data() + (size_t)c * head_dim;
        for (uint32_t d = 0; d < head_dim; d++) {
            row[d] = 0.29f * cosf((float)(d * 7u + c * 137u + 11u) * 0.017f) -
                     0.018f * (float)c;
        }
        mixed_cache_round(row, head_dim, n_rot);
    }

    CHECK(ds4_gpu_set_model_map(sinks.data(), sinks.size() * sizeof(float)),
          "install attention sinks");
    ds4_gpu_tensor q_dev = {}, kv_dev = {}, raw_dev = {}, comp_dev = {};
    ds4_gpu_tensor mask_dev = {}, heads_dev = {};
    CHECK(ds4_gpu_tensor_alloc_on(&q_dev, 0, q.size() * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&kv_dev, 0, kv_in.size() * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&raw_dev, 0, raw.size() * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&comp_dev, 0, comp.size() * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&mask_dev, 0, mask.size() * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&heads_dev, 0, q.size() * sizeof(float)) == 0,
          "allocate tensors");
    CHECK(ds4_gpu_tensor_write(&q_dev, 0, q.data(), q.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(&kv_dev, 0, kv_in.data(), kv_in.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(&raw_dev, 0, raw.data(), raw.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(&comp_dev, 0, comp.data(), comp.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(&mask_dev, 0, mask.data(), mask.size() * sizeof(float)),
          "upload tensors");
    CHECK(ds4_gpu_dsv4_fp8_kv_quantize_tensor(&kv_dev, n_raw, head_dim, n_rot) &&
          ds4_gpu_store_raw_kv_batch_tensor(&raw_dev, &kv_dev, raw_cap,
                                             raw_start, n_raw, head_dim) &&
          ds4_gpu_tensor_read(&raw_dev, 0, raw.data(), raw.size() * sizeof(float)),
          "generate and read mixed raw cache");
    double cache_max = 0.0;
    for (size_t i = 0; i < raw.size(); i++) {
        cache_max = fmax(cache_max, fabs((double)raw[i] - raw_expected[i]));
    }
    CHECK(cache_max == 0.0, "GPU mixed cache must match E4M3/F16 host contract exactly");

    ds4_gpu_set_quality(false);
    CHECK(ds4_gpu_attention_decode_heads_tensor(
              &heads_dev, sinks.data(), sinks.size() * sizeof(float), 0,
              &q_dev, &raw_dev, n_raw, raw_cap, raw_start,
              &comp_dev, 0, n_comp, &mask_dev, 1, n_head, head_dim),
          "run production mixed attention");
    std::vector<float> got(q.size()), quality_got(q.size()), ref(q.size());
    std::vector<float> wrong(q.size()), unquantized_ref(q.size());
    CHECK(ds4_gpu_tensor_read(&heads_dev, 0, got.data(), got.size() * sizeof(float)),
          "read attention output");
    attention_reference(ref, q, raw, comp, mask, sinks, n_raw, raw_cap,
                        raw_start, n_comp, n_head, head_dim);
    attention_reference(wrong, q, raw, comp, mask, sinks, n_raw, raw_cap,
                        (raw_start + 1u) % raw_cap, n_comp, n_head, head_dim);
    attention_reference(unquantized_ref, q, raw_unquantized, comp, mask, sinks,
                        n_raw, raw_cap, raw_start, n_comp, n_head, head_dim);
    const double error = rel_rms(got, ref);
    const double max_error = max_scaled_error(got, ref);
    const double wrong_error = rel_rms(got, wrong);
    const double unquantized_error = rel_rms(got, unquantized_ref);

    ds4_gpu_set_quality(true);
    CHECK(ds4_gpu_attention_decode_heads_tensor(
              &heads_dev, sinks.data(), sinks.size() * sizeof(float), 0,
              &q_dev, &raw_dev, n_raw, raw_cap, raw_start,
              &comp_dev, 0, n_comp, &mask_dev, 1, n_head, head_dim) &&
          ds4_gpu_tensor_read(&heads_dev, 0, quality_got.data(),
                              quality_got.size() * sizeof(float)),
          "run and read quality attention");
    const double quality_error = rel_rms(quality_got, ref);
    fprintf(stderr,
            "test_rocm_attention_decode_mixed: cache_max=%g fast_rel_rms=%g "
            "fast_max_scaled=%g quality_rel_rms=%g wrong_ring_rel_rms=%g "
            "unquantized_rel_rms=%g\n", cache_max, error, max_error,
            quality_error, wrong_error, unquantized_error);
    CHECK(error <= 1.0e-5 && max_error <= 1.0e-4,
          "production fast attention must match mixed-cache oracle");
    CHECK(quality_error <= 1.0e-5,
          "quality attention must match the same mixed-cache oracle");
    CHECK(wrong_error >= 1.0e-2 && wrong_error >= error * 100.0,
          "ring-order negative control must fail decisively");
    CHECK(unquantized_error >= 1.0e-3 && unquantized_error >= error * 100.0,
          "unquantized-cache negative control must fail decisively");

    ds4_gpu_tensor_free_in_place(&q_dev);
    ds4_gpu_tensor_free_in_place(&kv_dev);
    ds4_gpu_tensor_free_in_place(&raw_dev);
    ds4_gpu_tensor_free_in_place(&comp_dev);
    ds4_gpu_tensor_free_in_place(&mask_dev);
    ds4_gpu_tensor_free_in_place(&heads_dev);
    ds4_gpu_cleanup();
    fprintf(stderr, "test_rocm_attention_decode_mixed: PASS\n");
    return 0;
}
