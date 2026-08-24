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

static int time_heads8_decode_path(float *usec_per_call,
                                   ds4_gpu_tensor *heads,
                                   const std::vector<float> &sinks,
                                   const ds4_gpu_tensor *q,
                                   const std::vector<ds4_gpu_tensor> &raw_ring,
                                   const std::vector<ds4_gpu_tensor> &comp_ring,
                                   uint32_t raw_start) {
    constexpr int warmup = 20;
    constexpr int iterations = 400;
    constexpr uint32_t n_raw = 128u, raw_cap = 160u, n_comp = 523u;
    constexpr uint32_t n_head = 32u, head_dim = 512u;
    hipEvent_t start = NULL, stop = NULL;
    CHECK(!raw_ring.empty() && raw_ring.size() == comp_ring.size(),
          "valid cold-KV timing ring");
    ds4_gpu_set_quality(false);
    for (int i = 0; i < warmup; i++) {
        const size_t slot = (size_t)i % raw_ring.size();
        CHECK(ds4_gpu_attention_decode_heads_tensor(
                  heads, sinks.data(), sinks.size() * sizeof(float), 0,
                  q, &raw_ring[slot], n_raw, raw_cap, raw_start,
                  &comp_ring[slot], 0, n_comp, NULL, 0, n_head, head_dim),
              "warm up heads8 timing path");
    }
    CHECK(hipDeviceSynchronize() == hipSuccess,
          "synchronize heads8 timing warmup");
    CHECK(hipEventCreate(&start) == hipSuccess &&
          hipEventCreate(&stop) == hipSuccess,
          "create heads8 timing events");
    CHECK(hipEventRecord(start, NULL) == hipSuccess,
          "record heads8 timing start");
    for (int i = 0; i < iterations; i++) {
        const size_t slot = (size_t)i % raw_ring.size();
        CHECK(ds4_gpu_attention_decode_heads_tensor(
                  heads, sinks.data(), sinks.size() * sizeof(float), 0,
                  q, &raw_ring[slot], n_raw, raw_cap, raw_start,
                  &comp_ring[slot], 0, n_comp, NULL, 0, n_head, head_dim),
              "run heads8 timing path");
    }
    CHECK(hipEventRecord(stop, NULL) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess,
          "finish heads8 timing events");
    float elapsed_ms = 0.0f;
    CHECK(hipEventElapsedTime(&elapsed_ms, start, stop) == hipSuccess,
          "read heads8 timing events");
    CHECK(hipEventDestroy(start) == hipSuccess &&
          hipEventDestroy(stop) == hipSuccess,
          "destroy heads8 timing events");
    *usec_per_call = elapsed_ms * 1000.0f / (float)iterations;
    return 0;
}

static int run_heads8_decode_shape_oracle(void) {
    constexpr uint32_t head_dim = 512u, n_head = 32u;
    constexpr uint32_t raw_cap = 160u, n_raw = 128u, comp_cap = 1024u;
    const uint32_t raw_starts[] = {0u, 1u, 80u, 159u};
    const uint32_t comp_counts[] = {256u, 511u, 512u, 523u, 587u, 1024u};

    std::vector<float> sinks(n_head);
    std::vector<float> q((size_t)n_head * head_dim);
    std::vector<float> raw((size_t)raw_cap * head_dim);
    std::vector<float> comp((size_t)comp_cap * head_dim);
    std::vector<float> mask(comp_cap, 0.0f);
    for (uint32_t h = 0; h < n_head; h++) {
        sinks[h] = -0.41f + 0.027f * (float)h;
        for (uint32_t d = 0; d < head_dim; d++) {
            q[(size_t)h * head_dim + d] =
                0.031f * sinf((float)(d * 11u + h * 47u + 5u) * 0.0091f) +
                0.017f * cosf((float)(d * 3u + h * 13u + 7u) * 0.021f);
        }
    }
    for (uint32_t r = 0; r < raw_cap; r++) {
        float *row = raw.data() + (size_t)r * head_dim;
        for (uint32_t d = 0; d < head_dim; d++) {
            row[d] = 0.23f * sinf((float)(d * 5u + r * 97u + 3u) * 0.007f) +
                     0.014f * cosf((float)(d + r * 19u) * 0.031f);
        }
        mixed_cache_round(row, head_dim, 64u);
    }
    for (uint32_t c = 0; c < comp_cap; c++) {
        float *row = comp.data() + (size_t)c * head_dim;
        for (uint32_t d = 0; d < head_dim; d++) {
            row[d] = 0.19f * cosf((float)(d * 7u + c * 71u + 11u) * 0.0083f) -
                     0.011f * sinf((float)(d * 2u + c * 29u) * 0.019f);
        }
        mixed_cache_round(row, head_dim, 64u);
    }

    CHECK(ds4_gpu_set_model_map(sinks.data(), sinks.size() * sizeof(float)),
          "install heads8 attention sinks");
    ds4_gpu_tensor q_dev = {}, raw_dev = {}, comp_dev = {}, heads_dev = {};
    CHECK(ds4_gpu_tensor_alloc_on(&q_dev, 0, q.size() * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&raw_dev, 0, raw.size() * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&comp_dev, 0, comp.size() * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&heads_dev, 0, q.size() * sizeof(float)) == 0,
          "allocate heads8 oracle tensors");
    CHECK(ds4_gpu_tensor_write(&q_dev, 0, q.data(), q.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(&raw_dev, 0, raw.data(), raw.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(&comp_dev, 0, comp.data(), comp.size() * sizeof(float)),
          "upload heads8 oracle tensors");

    std::vector<float> candidate(q.size()), dense(q.size()), ref(q.size());
    for (uint32_t raw_start : raw_starts) {
        for (uint32_t n_comp : comp_counts) {
            ds4_gpu_set_quality(false);
            CHECK(ds4_gpu_attention_decode_heads_tensor(
                      &heads_dev, sinks.data(), sinks.size() * sizeof(float), 0,
                      &q_dev, &raw_dev, n_raw, raw_cap, raw_start,
                      &comp_dev, 0, n_comp, NULL, 0, n_head, head_dim) &&
                  ds4_gpu_tensor_read(&heads_dev, 0, candidate.data(),
                                      candidate.size() * sizeof(float)),
                  "run and read heads8 candidate");
            ds4_gpu_set_quality(true);
            CHECK(ds4_gpu_attention_decode_heads_tensor(
                      &heads_dev, sinks.data(), sinks.size() * sizeof(float), 0,
                      &q_dev, &raw_dev, n_raw, raw_cap, raw_start,
                      &comp_dev, 0, n_comp, NULL, 0, n_head, head_dim) &&
                  ds4_gpu_tensor_read(&heads_dev, 0, dense.data(),
                                      dense.size() * sizeof(float)),
                  "run and read dense decode reference kernel");
            attention_reference(ref, q, raw, comp, mask, sinks,
                                n_raw, raw_cap, raw_start, n_comp,
                                n_head, head_dim);
            const double candidate_rel = rel_rms(candidate, ref);
            const double candidate_max = max_scaled_error(candidate, ref);
            const double dense_rel = rel_rms(dense, ref);
            const double candidate_dense = rel_rms(candidate, dense);
            fprintf(stderr,
                    "heads8_oracle raw_start=%u n_comp=%u candidate_rel=%g "
                    "candidate_max=%g dense_rel=%g candidate_dense=%g\n",
                    raw_start, n_comp, candidate_rel, candidate_max,
                    dense_rel, candidate_dense);
            CHECK(candidate_rel <= 2.0e-5 && candidate_max <= 3.0e-4,
                  "heads8 candidate must stay inside Lane-B oracle envelope");
            CHECK(dense_rel <= 1.0e-5,
                  "dense decode kernel must match large-shape oracle");
            CHECK(candidate_dense <= 2.0e-5,
                  "heads8 and dense decode kernels must remain numerically close");
        }
    }

    /* The original low-amplitude sinusoids cannot distinguish reduction
     * orders that diverge on production-scale normalized activations.  Keep
     * the same layout but raise Q/K RMS enough to make score association
     * errors visible, and compare directly against the incumbent device
     * kernel rather than only the FP64 host oracle. */
    std::vector<float> q_stress = q;
    std::vector<float> raw_stress = raw;
    std::vector<float> comp_stress = comp;
    for (float &v : q_stress) v *= 24.0f;
    for (float &v : raw_stress) v *= 4.0f;
    for (float &v : comp_stress) v *= 5.0f;
    CHECK(ds4_gpu_tensor_write(&q_dev, 0, q_stress.data(),
                               q_stress.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(&raw_dev, 0, raw_stress.data(),
                               raw_stress.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(&comp_dev, 0, comp_stress.data(),
                               comp_stress.size() * sizeof(float)),
          "upload production-scale heads8 stress tensors");
    ds4_gpu_set_quality(false);
    CHECK(ds4_gpu_attention_decode_heads_tensor(
              &heads_dev, sinks.data(), sinks.size() * sizeof(float), 0,
              &q_dev, &raw_dev, n_raw, raw_cap, 159u,
              &comp_dev, 0, 523u, NULL, 0, n_head, head_dim) &&
          ds4_gpu_tensor_read(&heads_dev, 0, candidate.data(),
                              candidate.size() * sizeof(float)),
          "run production-scale heads8 candidate");
    ds4_gpu_set_quality(true);
    CHECK(ds4_gpu_attention_decode_heads_tensor(
              &heads_dev, sinks.data(), sinks.size() * sizeof(float), 0,
              &q_dev, &raw_dev, n_raw, raw_cap, 159u,
              &comp_dev, 0, 523u, NULL, 0, n_head, head_dim) &&
          ds4_gpu_tensor_read(&heads_dev, 0, dense.data(),
                              dense.size() * sizeof(float)),
          "run production-scale incumbent device kernel");
    const double stress_rel = rel_rms(candidate, dense);
    const double stress_max = max_scaled_error(candidate, dense);
    fprintf(stderr,
            "heads8_stress_device_ab candidate_dense=%g max_scaled=%g\n",
            stress_rel, stress_max);
    CHECK(stress_rel <= 2.0e-5 && stress_max <= 3.0e-4,
          "production-scale candidate must match incumbent device kernel");
    const char *heads_dump = getenv("DS4_ATTN_TEST_HEADS_DUMP");
    if (heads_dump && heads_dump[0]) {
        FILE *stream = fopen(heads_dump, "wb");
        CHECK(stream != NULL, "open production-scale heads dump");
        CHECK(fwrite(candidate.data(), sizeof(float), candidate.size(), stream) ==
                  candidate.size(),
              "write production-scale heads dump");
        CHECK(fclose(stream) == 0, "close production-scale heads dump");
    }

    CHECK(ds4_gpu_tensor_write(&q_dev, 0, q.data(), q.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(&raw_dev, 0, raw.data(), raw.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(&comp_dev, 0, comp.data(), comp.size() * sizeof(float)),
          "restore heads8 timing tensors");

    constexpr size_t timing_ring_size = 21u;
    std::vector<ds4_gpu_tensor> raw_ring(timing_ring_size);
    std::vector<ds4_gpu_tensor> comp_ring(timing_ring_size);
    for (size_t i = 0; i < timing_ring_size; i++) {
        CHECK(ds4_gpu_tensor_alloc_on(&raw_ring[i], 0,
                                      raw.size() * sizeof(float)) == 0 &&
              ds4_gpu_tensor_alloc_on(&comp_ring[i], 0,
                                      comp.size() * sizeof(float)) == 0,
              "allocate cold-KV timing ring");
        CHECK(ds4_gpu_tensor_write(&raw_ring[i], 0, raw.data(),
                                   raw.size() * sizeof(float)) &&
              ds4_gpu_tensor_write(&comp_ring[i], 0, comp.data(),
                                   comp.size() * sizeof(float)),
              "upload cold-KV timing ring");
    }

    float active_a = 0.0f, active_b = 0.0f;
    CHECK(time_heads8_decode_path(&active_a, &heads_dev, sinks,
                                  &q_dev, raw_ring, comp_ring, 159u) == 0 &&
          time_heads8_decode_path(&active_b, &heads_dev, sinks,
                                  &q_dev, raw_ring, comp_ring, 159u) == 0,
          "time active contiguous decode path");
    const float active_mid = 0.5f * (active_a + active_b);
    const char *mode = getenv("DS4_ROCM_ATTN_DECODE_SEQTILE_RESEARCH")
        ? "candidate" : "incumbent";
    fprintf(stderr,
            "contiguous_seqtile_timing mode=%s n_raw=128 n_comp=523 "
            "heads=32 raw_start=159 active_a_us=%.3f active_b_us=%.3f "
            "active_mid_us=%.3f\n",
            mode, active_a, active_b, active_mid);

    for (size_t i = 0; i < timing_ring_size; i++) {
        ds4_gpu_tensor_free_in_place(&raw_ring[i]);
        ds4_gpu_tensor_free_in_place(&comp_ring[i]);
    }

    ds4_gpu_set_quality(false);
    ds4_gpu_tensor_free_in_place(&q_dev);
    ds4_gpu_tensor_free_in_place(&raw_dev);
    ds4_gpu_tensor_free_in_place(&comp_dev);
    ds4_gpu_tensor_free_in_place(&heads_dev);
    return 0;
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
    CHECK(run_heads8_decode_shape_oracle() == 0,
          "large decode-shape heads8 oracle");
    ds4_gpu_cleanup();
    fprintf(stderr, "test_rocm_attention_decode_mixed: PASS\n");
    return 0;
}
