#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "tests/glm5_gguf_test.hpp"

#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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

namespace {

constexpr uint32_t kHeads = 64u;
constexpr uint32_t kQkNope = 256u;
constexpr uint32_t kKvLora = 512u;
constexpr uint32_t kValue = 256u;
constexpr uint32_t kRows = 10u;
constexpr uint32_t kSelected = 7u;

float half_to_float(uint16_t value) {
    const uint32_t sign = (uint32_t)(value & 0x8000u) << 16u;
    uint32_t exponent = (value >> 10u) & 0x1fu;
    uint32_t fraction = value & 0x03ffu;
    uint32_t bits = 0u;
    if (exponent == 0u) {
        if (fraction == 0u) {
            bits = sign;
        } else {
            int shift = 0;
            while ((fraction & 0x0400u) == 0u) {
                fraction <<= 1u;
                ++shift;
            }
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

float q8_dot(const uint8_t *row, const float *x) {
    float total = 0.0f;
    for (uint32_t block = 0u; block < kKvLora / 32u; ++block) {
        uint16_t scale_bits = 0u;
        std::memcpy(&scale_bits, row + (uint64_t)block * 34u,
                    sizeof(scale_bits));
        const float scale = half_to_float(scale_bits);
        const int8_t *quant =
            (const int8_t *)(row + (uint64_t)block * 34u + 2u);
        float subtotal = 0.0f;
        for (uint32_t i = 0u; i < 32u; ++i) {
            subtotal += (float)quant[i] * x[block * 32u + i];
        }
        total += scale * subtotal;
    }
    return total;
}

uint64_t fnv1a64_bytes(const void *data, size_t size) {
    const uint8_t *bytes = static_cast<const uint8_t *>(data);
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

bool run_test() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    CHECK(model && model[0], "GLM5 model environment");

    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM5 GGUF directory");
    uint64_t value_offset = 0u;
    CHECK(gguf.tensor("blk.3.attn_v_b.weight", {512u, 256u, 64u},
                      8u, value_offset),
          "bind real block-3 Q8 value projection");

    ds4_gpu_config config = {};
    config.n_gpus = 1u;
    config.device_indices[0] = 0u;
    CHECK(ds4_gpu_init_multi(&config) &&
          ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map) &&
          ds4_gpu_set_model_map(gguf.map, gguf.size),
          "initialize gfx1151 and register model map");

    constexpr uint32_t kStoreTokens = 3u;
    constexpr uint32_t kStoreCap = 5u;
    std::vector<float> store_source((uint64_t)kStoreTokens * kKvLora);
    for (size_t i = 0; i < store_source.size(); ++i) {
        store_source[i] =
            (float)((int)((i * 17u) % 47u) - 23) * 0.002731f +
            (float)((int)(i % 5u) - 2) * 0.000013f;
    }
    std::vector<float> store_sentinel((uint64_t)kStoreCap * kKvLora,
                                      123.25f);
    ds4_gpu_tensor *d_store_source = ds4_gpu_tensor_alloc(
        (uint64_t)store_source.size() * sizeof(float));
    ds4_gpu_tensor *d_store_f32 = ds4_gpu_tensor_alloc(
        (uint64_t)store_sentinel.size() * sizeof(float));
    CHECK(d_store_source && d_store_f32 &&
          ds4_gpu_tensor_write(d_store_source, 0u, store_source.data(),
                               (uint64_t)store_source.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(d_store_f32, 0u, store_sentinel.data(),
                               (uint64_t)store_sentinel.size() * sizeof(float)) &&
          ds4_gpu_glm_store_compact_kv_tensor(
              d_store_f32, nullptr, d_store_source, d_store_source,
              1u, kStoreTokens, kStoreCap, kKvLora, kKvLora, 0u, false) &&
          ds4_gpu_synchronize(),
          "store F32 zero-RoPE compact KV");
    std::vector<float> stored_f32(store_sentinel.size());
    CHECK(ds4_gpu_tensor_read(d_store_f32, 0u, stored_f32.data(),
                              (uint64_t)stored_f32.size() * sizeof(float)),
          "read F32 zero-RoPE compact KV");
    for (uint32_t row = 0u; row < kStoreCap; ++row) {
        for (uint32_t j = 0u; j < kKvLora; ++j) {
            const float expected = row >= 1u && row < 1u + kStoreTokens
                ? store_source[(uint64_t)(row - 1u) * kKvLora + j]
                : 123.25f;
            CHECK(std::memcmp(&stored_f32[(uint64_t)row * kKvLora + j],
                              &expected, sizeof(float)) == 0,
                  "F32 store preserves payload and outer sentinels");
        }
    }

    std::vector<uint16_t> store_sentinel_f16(
        (uint64_t)kStoreCap * kKvLora);
    const __half sentinel_half = __float2half_rn(-17.5f);
    uint16_t sentinel_half_bits = 0u;
    std::memcpy(&sentinel_half_bits, &sentinel_half, sizeof(uint16_t));
    std::fill(store_sentinel_f16.begin(), store_sentinel_f16.end(),
              sentinel_half_bits);
    ds4_gpu_tensor *d_store_f16 = ds4_gpu_tensor_alloc(
        (uint64_t)store_sentinel_f16.size() * sizeof(uint16_t));
    CHECK(d_store_f16 &&
          ds4_gpu_tensor_write(d_store_f16, 0u, store_sentinel_f16.data(),
                               (uint64_t)store_sentinel_f16.size() * sizeof(uint16_t)) &&
          ds4_gpu_glm_store_compact_kv_tensor(
              d_store_f16, nullptr, d_store_source, d_store_source,
              1u, kStoreTokens, kStoreCap, kKvLora, kKvLora, 0u, true) &&
          ds4_gpu_synchronize(),
          "store F16 zero-RoPE compact KV");
    std::vector<uint16_t> stored_f16(store_sentinel_f16.size());
    CHECK(ds4_gpu_tensor_read(d_store_f16, 0u, stored_f16.data(),
                              (uint64_t)stored_f16.size() * sizeof(uint16_t)),
          "read F16 zero-RoPE compact KV");
    for (uint32_t row = 0u; row < kStoreCap; ++row) {
        for (uint32_t j = 0u; j < kKvLora; ++j) {
            uint16_t expected = sentinel_half_bits;
            if (row >= 1u && row < 1u + kStoreTokens) {
                const __half rounded = __float2half_rn(
                    store_source[(uint64_t)(row - 1u) * kKvLora + j]);
                std::memcpy(&expected, &rounded, sizeof(uint16_t));
            }
            CHECK(stored_f16[(uint64_t)row * kKvLora + j] == expected,
                  "F16 store preserves rounded payload and outer sentinels");
        }
    }
    CHECK(!ds4_gpu_glm_store_compact_kv_tensor(
              d_store_f32, nullptr, d_store_source, d_store_source,
              1u, 2u, kStoreCap, kKvLora + 64u, kKvLora,
              64u, false),
          "nonzero-RoPE store rejects a null cache");
    CHECK(!ds4_gpu_glm_store_compact_kv_tensor(
              d_store_f32, d_store_f32, d_store_source, d_store_source,
              1u, kStoreTokens, kStoreCap, kKvLora, kKvLora, 0u, false),
          "zero-RoPE store rejects a non-null cache");
    CHECK(!ds4_gpu_glm_store_compact_kv_tensor(
              d_store_f32, nullptr, d_store_source, d_store_source,
              1u, 2u, kStoreCap, kKvLora - 1u, kKvLora, 0u, false),
          "compact store rejects lora width beyond raw row");
    CHECK(!ds4_gpu_glm_store_compact_kv_tensor(
              d_store_f32, d_store_f32, d_store_source, d_store_source,
              1u, 2u, kStoreCap, kKvLora + 63u, kKvLora, 64u, false),
          "compact store rejects rope tail beyond raw row");
    CHECK(!ds4_gpu_glm_store_compact_kv_tensor(
              d_store_f32, nullptr, d_store_source, d_store_source,
              4u, kStoreTokens, kStoreCap, kKvLora, kKvLora, 0u, false),
          "zero-RoPE store rejects an overflowing position span");
    std::fprintf(stderr,
                 "GLM5 NoPE compact KV store F32/F16 payloads and sentinels exact\n");

    std::vector<float> query((uint64_t)kHeads * kQkNope);
    std::vector<float> qk_low((uint64_t)kHeads * kKvLora);
    std::vector<float> cache((uint64_t)kRows * kKvLora);
    std::vector<int32_t> selected = {8, 1, -1, 6, 3, 0, 5};
    for (size_t i = 0; i < query.size(); ++i) {
        query[i] = (float)((int)((i * 11u) % 41u) - 20) * 0.001953125f;
    }
    for (size_t i = 0; i < qk_low.size(); ++i) {
        qk_low[i] = (float)((int)((i * 7u) % 37u) - 18) * 0.00390625f;
    }
    for (size_t i = 0; i < cache.size(); ++i) {
        cache[i] =
            (float)((int)((i * 13u) % 43u) - 21) * 0.003713f +
            (float)((int)(i % 7u) - 3) * 0.000011f;
    }

    const uint64_t value_row_bytes = (kKvLora / 32u) * 34u;
    const uint8_t *value_weight =
        (const uint8_t *)gguf.map + value_offset;
    const auto reference = [&](const std::vector<float> &cache_values,
                               uint32_t qk_dim) {
        std::vector<float> out((uint64_t)kHeads * kValue, 0.0f);
        std::vector<float> scores(kSelected);
        std::vector<float> lora(kKvLora);
        const float score_scale = 1.0f / std::sqrt((float)qk_dim);
        for (uint32_t head = 0u; head < kHeads; ++head) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (uint32_t s = 0u; s < kSelected; ++s) {
                const int32_t row = selected[s];
                float score = -std::numeric_limits<float>::infinity();
                if (row >= 0 && (uint32_t)row < kRows) {
                    const float *low =
                        qk_low.data() + (uint64_t)head * kKvLora;
                    const float *kv =
                        cache_values.data() + (uint64_t)row * kKvLora;
                    score = 0.0f;
                    for (uint32_t j = 0u; j < kKvLora; ++j) {
                        score += low[j] * kv[j];
                    }
                    score *= score_scale;
                }
                scores[s] = score;
                maximum = std::max(maximum, score);
            }
            float denominator = 0.0f;
            for (uint32_t s = 0u; s < kSelected; ++s) {
                scores[s] = std::exp(scores[s] - maximum);
                denominator += scores[s];
            }
            for (uint32_t j = 0u; j < kKvLora; ++j) {
                float total = 0.0f;
                for (uint32_t s = 0u; s < kSelected; ++s) {
                    const int32_t row = selected[s];
                    if (row >= 0 && (uint32_t)row < kRows) {
                        total += scores[s] *
                            cache_values[(uint64_t)row * kKvLora + j];
                    }
                }
                lora[j] = total / denominator;
            }
            for (uint32_t value = 0u; value < kValue; ++value) {
                const uint8_t *row = value_weight +
                    ((uint64_t)head * kValue + value) * value_row_bytes;
                out[(uint64_t)head * kValue + value] =
                    q8_dot(row, lora.data());
            }
        }
        return out;
    };
    const std::vector<float> expected = reference(cache, kQkNope);

    ds4_gpu_tensor *d_query = ds4_gpu_tensor_alloc(
        (uint64_t)query.size() * sizeof(float));
    ds4_gpu_tensor *d_low = ds4_gpu_tensor_alloc(
        (uint64_t)qk_low.size() * sizeof(float));
    ds4_gpu_tensor *d_cache = ds4_gpu_tensor_alloc(
        (uint64_t)cache.size() * sizeof(float));
    ds4_gpu_tensor *d_selected = ds4_gpu_tensor_alloc(
        (uint64_t)selected.size() * sizeof(int32_t));
    ds4_gpu_tensor *d_heads = ds4_gpu_tensor_alloc(
        (uint64_t)expected.size() * sizeof(float));
    CHECK(d_query && d_low && d_cache && d_selected && d_heads,
          "allocate bounded NoPE attention tensors");
    CHECK(ds4_gpu_tensor_write(d_query, 0u, query.data(),
                               (uint64_t)query.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(d_low, 0u, qk_low.data(),
                               (uint64_t)qk_low.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(d_cache, 0u, cache.data(),
                               (uint64_t)cache.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(d_selected, 0u, selected.data(),
                               (uint64_t)selected.size() * sizeof(int32_t)),
          "upload NoPE attention inputs");

    CHECK(ds4_gpu_glm_attention_indexed_decode_typed_tensor(
              d_heads, d_query, d_low, d_cache, nullptr,
              gguf.map, gguf.size, value_offset, 8u, d_selected,
              kSelected, kRows, false, kHeads, kKvLora, kQkNope, 0u,
              kValue, 0u, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f) &&
          ds4_gpu_synchronize(),
          "execute zero-RoPE indexed decode attention");
    std::vector<float> got(expected.size());
    CHECK(ds4_gpu_tensor_read(d_heads, 0u, got.data(),
                              (uint64_t)got.size() * sizeof(float)),
          "read NoPE attention output");

    double maximum = 0.0;
    double error2 = 0.0;
    double reference2 = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        CHECK(std::isfinite(got[i]) && std::isfinite(expected[i]),
              "finite NoPE attention output");
        const double error = (double)got[i] - expected[i];
        maximum = std::max(maximum, std::fabs(error));
        error2 += error * error;
        reference2 += (double)expected[i] * expected[i];
    }
    const double nmse = error2 / std::max(reference2, 1.0e-30);
    std::fprintf(stderr,
                 "GLM5 NoPE indexed attention count=%zu max_abs=%.9g "
                 "nmse=%.9g\n", got.size(), maximum, nmse);
    CHECK(maximum <= 2.0e-5 && nmse <= 1.0e-10,
          "NoPE attention numerical envelope");

    /* Production-sized Lane-A gate.  The selected rows retain the indexer's
     * four-row pool geometry while permuting pool order and retaining invalid
     * sentinels.  Compare the lora output directly so the following Q8 value
     * projection cannot hide an arithmetic difference. */
    constexpr uint32_t kProductionRows = 2051u;
    constexpr uint32_t kProductionSelected = 2048u;
    std::vector<float> production_query((uint64_t)kHeads * kQkNope);
    std::vector<float> production_low((uint64_t)kHeads * kKvLora);
    std::vector<float> production_cache((uint64_t)kProductionRows * kKvLora);
    std::vector<int32_t> production_selected(kProductionSelected);
    for (size_t i = 0; i < production_query.size(); ++i) {
        production_query[i] =
            (float)((int)((i * 19u) % 67u) - 33) * 0.001271f;
    }
    for (size_t i = 0; i < production_low.size(); ++i) {
        production_low[i] =
            (float)((int)((i * 23u) % 71u) - 35) * 0.001819f;
    }
    for (size_t i = 0; i < production_cache.size(); ++i) {
        production_cache[i] =
            (float)((int)((i * 29u) % 79u) - 39) * 0.001117f +
            (float)((int)(i % 11u) - 5) * 0.000007f;
    }
    for (uint32_t i = 0u; i < kProductionSelected; ++i) {
        const uint32_t pool = i >> 2u;
        const uint32_t lane = i & 3u;
        production_selected[i] =
            (int32_t)((((pool * 157u + 17u) & 511u) << 2u) + lane);
    }
    production_selected[127u] = -1;
    production_selected[1023u] = -1;
    production_selected[2047u] = -1;

    ds4_gpu_tensor *d_production_query = ds4_gpu_tensor_alloc(
        (uint64_t)production_query.size() * sizeof(float));
    ds4_gpu_tensor *d_production_low = ds4_gpu_tensor_alloc(
        (uint64_t)production_low.size() * sizeof(float));
    ds4_gpu_tensor *d_production_cache = ds4_gpu_tensor_alloc(
        (uint64_t)production_cache.size() * sizeof(float));
    ds4_gpu_tensor *d_production_selected = ds4_gpu_tensor_alloc(
        (uint64_t)production_selected.size() * sizeof(int32_t));
    ds4_gpu_tensor *d_production_incumbent = ds4_gpu_tensor_alloc(
        (uint64_t)kHeads * kKvLora * sizeof(float));
    ds4_gpu_tensor *d_production_candidate = ds4_gpu_tensor_alloc(
        (uint64_t)kHeads * kKvLora * sizeof(float));
    CHECK(d_production_query && d_production_low && d_production_cache &&
          d_production_selected && d_production_incumbent &&
          d_production_candidate,
          "allocate production-sized NoPE lora tensors");
    CHECK(ds4_gpu_tensor_write(
              d_production_query, 0u, production_query.data(),
              (uint64_t)production_query.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(
              d_production_low, 0u, production_low.data(),
              (uint64_t)production_low.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(
              d_production_cache, 0u, production_cache.data(),
              (uint64_t)production_cache.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(
              d_production_selected, 0u, production_selected.data(),
              (uint64_t)production_selected.size() * sizeof(int32_t)),
          "upload production-sized NoPE lora inputs");

    unsetenv("DS4_ROCM_GLM5_NOPE_ATTN_EXACT");
    CHECK(ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
              d_production_incumbent, d_production_query, d_production_low,
              d_production_cache, nullptr, d_production_selected,
              1u, kProductionSelected, kProductionRows, false,
              kHeads, kKvLora, kQkNope, 0u, 1u,
              1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f) &&
          ds4_gpu_synchronize(),
          "execute production-sized incumbent NoPE lora attention");
    CHECK(setenv("DS4_ROCM_GLM5_NOPE_ATTN_EXACT", "1", 1) == 0 &&
          ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
              d_production_candidate, d_production_query, d_production_low,
              d_production_cache, nullptr, d_production_selected,
              1u, kProductionSelected, kProductionRows, false,
              kHeads, kKvLora, kQkNope, 0u, 1u,
              1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f) &&
          ds4_gpu_synchronize(),
          "execute production-sized candidate NoPE lora attention");
    std::vector<float> production_incumbent((uint64_t)kHeads * kKvLora);
    std::vector<float> production_candidate((uint64_t)kHeads * kKvLora);
    CHECK(ds4_gpu_tensor_read(
              d_production_incumbent, 0u, production_incumbent.data(),
              (uint64_t)production_incumbent.size() * sizeof(float)) &&
          ds4_gpu_tensor_read(
              d_production_candidate, 0u, production_candidate.data(),
              (uint64_t)production_candidate.size() * sizeof(float)),
          "read production-sized NoPE lora outputs");
    CHECK(std::memcmp(production_incumbent.data(), production_candidate.data(),
                      production_incumbent.size() * sizeof(float)) == 0,
          "production-sized NoPE lora output is bit-identical");
    const uint64_t production_fnv = fnv1a64_bytes(
        production_candidate.data(), production_candidate.size() * sizeof(float));
    std::fprintf(stderr,
                 "GLM5 NoPE production lora rows=%u selected=%u "
                 "bit_identical=1 fnv64=%016llx\n",
                 kProductionRows, kProductionSelected,
                 (unsigned long long)production_fnv);

    ds4_gpu_tensor_free(d_production_candidate);
    ds4_gpu_tensor_free(d_production_incumbent);
    ds4_gpu_tensor_free(d_production_selected);
    ds4_gpu_tensor_free(d_production_cache);
    ds4_gpu_tensor_free(d_production_low);
    ds4_gpu_tensor_free(d_production_query);

    std::vector<uint16_t> cache_f16(cache.size());
    std::vector<float> cache_f16_reference(cache.size());
    for (size_t i = 0; i < cache.size(); ++i) {
        const __half value = __float2half_rn(cache[i]);
        std::memcpy(&cache_f16[i], &value, sizeof(uint16_t));
        cache_f16_reference[i] = half_to_float(cache_f16[i]);
    }
    ds4_gpu_tensor *d_cache_f16 = ds4_gpu_tensor_alloc(
        (uint64_t)cache_f16.size() * sizeof(uint16_t));
    CHECK(d_cache_f16 &&
          ds4_gpu_tensor_write(d_cache_f16, 0u, cache_f16.data(),
                               (uint64_t)cache_f16.size() * sizeof(uint16_t)) &&
          ds4_gpu_glm_attention_indexed_decode_typed_tensor(
              d_heads, d_query, d_low, d_cache_f16, nullptr,
              gguf.map, gguf.size, value_offset, 8u, d_selected,
              kSelected, kRows, true, kHeads, kKvLora, kQkNope, 0u,
              kValue, 0u, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f) &&
          ds4_gpu_synchronize() &&
          ds4_gpu_tensor_read(d_heads, 0u, got.data(),
                              (uint64_t)got.size() * sizeof(float)),
          "execute and read F16-cache NoPE attention");
    const std::vector<float> expected_f16 =
        reference(cache_f16_reference, kQkNope);
    maximum = 0.0;
    error2 = 0.0;
    reference2 = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double error = (double)got[i] - expected_f16[i];
        maximum = std::max(maximum, std::fabs(error));
        error2 += error * error;
        reference2 += (double)expected_f16[i] * expected_f16[i];
    }
    const double f16_nmse = error2 / std::max(reference2, 1.0e-30);
    std::fprintf(stderr,
                 "GLM5 NoPE F16-cache attention count=%zu max_abs=%.9g "
                 "nmse=%.9g\n", got.size(), maximum, f16_nmse);
    CHECK(maximum <= 2.0e-5 && f16_nmse <= 1.0e-10,
          "F16-cache NoPE attention numerical envelope");

    CHECK(!ds4_gpu_glm_attention_indexed_decode_typed_tensor(
              d_heads, d_query, d_low, d_cache, nullptr,
              gguf.map, gguf.size, value_offset, 8u, d_selected,
              kSelected, kRows, false, kHeads, kKvLora, kQkNope, 64u,
              kValue, 0u, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f),
          "nonzero RoPE rejects a null cache");

    constexpr uint32_t kRope = 64u;
    std::vector<float> query_rope((uint64_t)kHeads * (kQkNope + kRope), 0.0f);
    std::vector<float> rope_cache((uint64_t)kRows * kRope, 0.0f);
    for (uint32_t head = 0u; head < kHeads; ++head) {
        std::memcpy(query_rope.data() + (uint64_t)head * (kQkNope + kRope),
                    query.data() + (uint64_t)head * kQkNope,
                    (uint64_t)kQkNope * sizeof(float));
    }
    ds4_gpu_tensor *d_query_rope = ds4_gpu_tensor_alloc(
        (uint64_t)query_rope.size() * sizeof(float));
    ds4_gpu_tensor *d_rope = ds4_gpu_tensor_alloc(
        (uint64_t)rope_cache.size() * sizeof(float));
    CHECK(d_query_rope && d_rope &&
          ds4_gpu_tensor_write(d_query_rope, 0u, query_rope.data(),
                               (uint64_t)query_rope.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(d_rope, 0u, rope_cache.data(),
                               (uint64_t)rope_cache.size() * sizeof(float)) &&
          ds4_gpu_glm_attention_indexed_decode_typed_tensor(
              d_heads, d_query_rope, d_low, d_cache, d_rope,
              gguf.map, gguf.size, value_offset, 8u, d_selected,
              kSelected, kRows, false, kHeads, kKvLora, kQkNope, kRope,
              kValue, 0u, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f) &&
          ds4_gpu_synchronize(),
          "unchanged nonzero-RoPE control executes");
    CHECK(ds4_gpu_tensor_read(d_heads, 0u, got.data(),
                              (uint64_t)got.size() * sizeof(float)),
          "read nonzero-RoPE control output");
    const std::vector<float> expected_rope =
        reference(cache, kQkNope + kRope);
    maximum = 0.0;
    error2 = 0.0;
    reference2 = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double error = (double)got[i] - expected_rope[i];
        maximum = std::max(maximum, std::fabs(error));
        error2 += error * error;
        reference2 += (double)expected_rope[i] * expected_rope[i];
    }
    const double rope_nmse = error2 / std::max(reference2, 1.0e-30);
    std::fprintf(stderr,
                 "GLM RoPE regression control count=%zu max_abs=%.9g "
                 "nmse=%.9g\n", got.size(), maximum, rope_nmse);
    CHECK(maximum <= 2.0e-5 && rope_nmse <= 1.0e-10,
          "nonzero-RoPE control numerical envelope");

    ds4_gpu_tensor dummy_rope = {};
    CHECK(!ds4_gpu_glm_attention_indexed_decode_typed_tensor(
              d_heads, d_query, d_low, d_cache, &dummy_rope,
              gguf.map, gguf.size, value_offset, 8u, d_selected,
              kSelected, kRows, false, kHeads, kKvLora, kQkNope, 1u,
              kValue, 0u, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f),
          "odd RoPE width remains rejected");

    ds4_gpu_tensor_free(d_rope);
    ds4_gpu_tensor_free(d_query_rope);
    ds4_gpu_tensor_free(d_cache_f16);
    ds4_gpu_tensor_free(d_heads);
    ds4_gpu_tensor_free(d_selected);
    ds4_gpu_tensor_free(d_cache);
    ds4_gpu_tensor_free(d_low);
    ds4_gpu_tensor_free(d_query);
    ds4_gpu_tensor_free(d_store_f16);
    ds4_gpu_tensor_free(d_store_f32);
    ds4_gpu_tensor_free(d_store_source);
    ds4_gpu_cleanup();
    std::fprintf(stderr,
                 "PASS real-Q8 GLM5 zero-RoPE indexed attention gate\n");
    return true;
}

}  // namespace

int main() { return run_test() ? 0 : 1; }
