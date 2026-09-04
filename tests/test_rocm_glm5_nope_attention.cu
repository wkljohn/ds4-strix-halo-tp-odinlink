#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "tests/glm5_gguf_test.hpp"

#include <hip/hip_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

extern "C" int ds4_rocm_glm5_sparse_attention_exact_rows(
        ds4_gpu_tensor *lora_out,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *selected,
        uint32_t n_tokens,
        uint32_t selected_stride,
        uint32_t cache_cap);

extern "C" int ds4_rocm_glm5_sparse_attention_f16_gemm_rows(
        ds4_gpu_tensor *lora_out,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *selected,
        uint32_t n_tokens,
        uint32_t selected_stride,
        uint32_t cache_cap);

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
    constexpr uint32_t kProductionSelected = 2051u;
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
    for (uint32_t i = 0u; i < 2048u; ++i) {
        const uint32_t pool = i >> 2u;
        const uint32_t lane = i & 3u;
        production_selected[i] =
            (int32_t)((((pool * 157u + 17u) & 511u) << 2u) + lane);
    }
    production_selected[2048u] = 2048;
    production_selected[2049u] = 2049;
    production_selected[2050u] = 2050;
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
    ds4_gpu_tensor *d_production_shared = ds4_gpu_tensor_alloc(
        (uint64_t)kHeads * kKvLora * sizeof(float));
    CHECK(d_production_query && d_production_low && d_production_cache &&
          d_production_selected && d_production_incumbent &&
          d_production_candidate && d_production_shared,
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
    unsetenv("DS4_ROCM_GLM5_NOPE_ATTN_EXACT");
    CHECK(setenv("DS4_ROCM_GLM5_NOPE_ATTN_SHARED_PV", "1", 1) == 0 &&
          ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
              d_production_shared, d_production_query, d_production_low,
              d_production_cache, nullptr, d_production_selected,
              1u, kProductionSelected, kProductionRows, false,
              kHeads, kKvLora, kQkNope, 0u, 1u,
              1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f) &&
          ds4_gpu_synchronize(),
          "execute production-sized shared-PV NoPE lora attention");
    std::vector<float> production_incumbent((uint64_t)kHeads * kKvLora);
    std::vector<float> production_candidate((uint64_t)kHeads * kKvLora);
    std::vector<float> production_shared((uint64_t)kHeads * kKvLora);
    CHECK(ds4_gpu_tensor_read(
              d_production_incumbent, 0u, production_incumbent.data(),
              (uint64_t)production_incumbent.size() * sizeof(float)) &&
          ds4_gpu_tensor_read(
              d_production_candidate, 0u, production_candidate.data(),
              (uint64_t)production_candidate.size() * sizeof(float)) &&
          ds4_gpu_tensor_read(
              d_production_shared, 0u, production_shared.data(),
              (uint64_t)production_shared.size() * sizeof(float)),
          "read production-sized NoPE lora outputs");
    CHECK(std::memcmp(production_incumbent.data(), production_candidate.data(),
                      production_incumbent.size() * sizeof(float)) == 0,
          "production-sized NoPE lora output is bit-identical");
    CHECK(std::memcmp(production_incumbent.data(), production_shared.data(),
                      production_incumbent.size() * sizeof(float)) == 0,
          "production-sized shared-PV NoPE lora output is bit-identical");
    const uint64_t production_fnv = fnv1a64_bytes(
        production_candidate.data(), production_candidate.size() * sizeof(float));
    std::fprintf(stderr,
                 "GLM5 NoPE production lora rows=%u selected=%u "
                 "exact_bit_identical=1 shared_pv_bit_identical=1 "
                 "fnv64=%016llx\n",
                 kProductionRows, kProductionSelected,
                 (unsigned long long)production_fnv);

    /* The sparse-prefill executor stores one fixed-width, -1-padded selected
     * list per query.  Compare the multi-row launch against the established
     * exact shared-PV one-row path to catch row-stride or batch-indexing drift
     * before exercising the full TP graph. */
    constexpr uint32_t kBatchRows = 16u;
    std::vector<float> batch_query(
        (uint64_t)kBatchRows * production_query.size());
    std::vector<float> batch_low(
        (uint64_t)kBatchRows * production_low.size());
    std::vector<int32_t> batch_selected(
        (uint64_t)kBatchRows * kProductionSelected);
    uint32_t batch_live[kBatchRows] = {};
    for (uint32_t row = 0u; row < kBatchRows; ++row) {
        batch_live[row] = 2048u + ((row + 1u) & 3u);
    }
    for (uint32_t row = 0u; row < kBatchRows; ++row) {
        std::memcpy(batch_query.data() +
                        (uint64_t)row * production_query.size(),
                    production_query.data(),
                    production_query.size() * sizeof(float));
        std::memcpy(batch_low.data() +
                        (uint64_t)row * production_low.size(),
                    production_low.data(),
                    production_low.size() * sizeof(float));
        std::memcpy(batch_selected.data() +
                        (uint64_t)row * kProductionSelected,
                    production_selected.data(),
                    production_selected.size() * sizeof(int32_t));
        for (size_t i = 0; i < production_low.size(); ++i) {
            batch_low[(uint64_t)row * production_low.size() + i] +=
                (float)row * (float)((int)(i % 7u) - 3) * 0.00000037f;
        }
        const uint32_t live = batch_live[row];
        for (uint32_t s = live; s < kProductionSelected; ++s) {
            batch_selected[(uint64_t)row * kProductionSelected + s] = -1;
        }
    }
    const uint64_t batch_lora_floats =
        (uint64_t)kBatchRows * kHeads * kKvLora;
    ds4_gpu_tensor *d_batch_query = ds4_gpu_tensor_alloc(
        (uint64_t)batch_query.size() * sizeof(float));
    ds4_gpu_tensor *d_batch_low = ds4_gpu_tensor_alloc(
        (uint64_t)batch_low.size() * sizeof(float));
    ds4_gpu_tensor *d_batch_selected = ds4_gpu_tensor_alloc(
        (uint64_t)batch_selected.size() * sizeof(int32_t));
    ds4_gpu_tensor *d_batch_lora = ds4_gpu_tensor_alloc(
        batch_lora_floats * sizeof(float));
    ds4_gpu_tensor *d_head_shared_lora = ds4_gpu_tensor_alloc(
        batch_lora_floats * sizeof(float));
    ds4_gpu_tensor *d_f16_gemm_lora = ds4_gpu_tensor_alloc(
        batch_lora_floats * sizeof(float));
    ds4_gpu_tensor *d_scalar_lora = ds4_gpu_tensor_alloc(
        batch_lora_floats * sizeof(float));
    CHECK(d_batch_query && d_batch_low && d_batch_selected &&
          d_batch_lora && d_head_shared_lora && d_f16_gemm_lora &&
          d_scalar_lora &&
          ds4_gpu_tensor_write(
              d_batch_query, 0u, batch_query.data(),
              (uint64_t)batch_query.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(
              d_batch_low, 0u, batch_low.data(),
              (uint64_t)batch_low.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(
              d_batch_selected, 0u, batch_selected.data(),
              (uint64_t)batch_selected.size() * sizeof(int32_t)),
          "allocate and upload fixed-width NoPE batch oracle");
    unsetenv("DS4_ROCM_GLM5_NOPE_ATTN_SHARED_PV");
    CHECK(ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
              d_batch_lora, d_batch_query, d_batch_low,
              d_production_cache, nullptr, d_batch_selected,
              kBatchRows, kProductionSelected, kProductionRows, false,
              kHeads, kKvLora, kQkNope, 0u, 1u,
              1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f),
          "execute fixed-width NoPE batch oracle");
    CHECK(ds4_rocm_glm5_sparse_attention_exact_rows(
              d_head_shared_lora, d_batch_low, d_production_cache,
              d_batch_selected, kBatchRows, kProductionSelected,
              kProductionRows),
          "execute fixed-stride head-shared NoPE batch oracle");
    CHECK(ds4_rocm_glm5_sparse_attention_f16_gemm_rows(
              d_f16_gemm_lora, d_batch_low, d_production_cache,
              d_batch_selected, kBatchRows, kProductionSelected,
              kProductionRows),
          "execute fixed-stride F16 GEMM NoPE batch oracle");
    CHECK(setenv("DS4_ROCM_GLM5_NOPE_ATTN_SHARED_PV", "1", 1) == 0,
          "enable exact shared-PV scalar oracle");
    const uint64_t query_row_bytes =
        (uint64_t)kHeads * kQkNope * sizeof(float);
    const uint64_t low_row_bytes =
        (uint64_t)kHeads * kKvLora * sizeof(float);
    const uint64_t selected_row_bytes =
        (uint64_t)kProductionSelected * sizeof(int32_t);
    for (uint32_t row = 0u; row < kBatchRows; ++row) {
        ds4_gpu_tensor *q_view = ds4_gpu_tensor_view(
            d_batch_query, (uint64_t)row * query_row_bytes,
            query_row_bytes);
        ds4_gpu_tensor *low_view = ds4_gpu_tensor_view(
            d_batch_low, (uint64_t)row * low_row_bytes,
            low_row_bytes);
        ds4_gpu_tensor *selected_view = ds4_gpu_tensor_view(
            d_batch_selected, (uint64_t)row * selected_row_bytes,
            selected_row_bytes);
        ds4_gpu_tensor *out_view = ds4_gpu_tensor_view(
            d_scalar_lora, (uint64_t)row * low_row_bytes,
            low_row_bytes);
        const int row_ok = q_view && low_view && selected_view && out_view &&
            ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
                out_view, q_view, low_view, d_production_cache, nullptr,
                selected_view, 1u, batch_live[row], kProductionRows,
                false, kHeads, kKvLora, kQkNope, 0u, 1u,
                1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ds4_gpu_tensor_free(out_view);
        ds4_gpu_tensor_free(selected_view);
        ds4_gpu_tensor_free(low_view);
        ds4_gpu_tensor_free(q_view);
        CHECK(row_ok, "execute exact shared-PV scalar batch row");
    }
    std::vector<float> batch_lora(batch_lora_floats);
    std::vector<float> head_shared_lora(batch_lora_floats);
    std::vector<float> f16_gemm_lora(batch_lora_floats);
    std::vector<float> scalar_lora(batch_lora_floats);
    CHECK(ds4_gpu_synchronize() &&
          ds4_gpu_tensor_read(d_batch_lora, 0u, batch_lora.data(),
                              batch_lora_floats * sizeof(float)) &&
          ds4_gpu_tensor_read(d_head_shared_lora, 0u,
                              head_shared_lora.data(),
                              batch_lora_floats * sizeof(float)) &&
          ds4_gpu_tensor_read(d_f16_gemm_lora, 0u,
                              f16_gemm_lora.data(),
                              batch_lora_floats * sizeof(float)) &&
          ds4_gpu_tensor_read(d_scalar_lora, 0u, scalar_lora.data(),
                              batch_lora_floats * sizeof(float)),
          "read fixed-width NoPE batch oracle outputs");
    CHECK(std::memcmp(batch_lora.data(), scalar_lora.data(),
                      batch_lora_floats * sizeof(float)) == 0,
          "boundary NoPE batch is bit-identical to live-count scalar path");
    CHECK(std::memcmp(head_shared_lora.data(), scalar_lora.data(),
                      batch_lora_floats * sizeof(float)) == 0,
          "boundary head-shared NoPE batch is bit-identical to scalar path");
    long double f16_error2 = 0.0L;
    long double f16_reference2 = 0.0L;
    long double f16_candidate2 = 0.0L;
    long double f16_dot = 0.0L;
    double f16_max_abs = 0.0;
    uint64_t f16_nonfinite = 0u;
    for (uint64_t i = 0u; i < batch_lora_floats; ++i) {
        const double reference = scalar_lora[i];
        const double candidate = f16_gemm_lora[i];
        if (!std::isfinite(candidate)) {
            ++f16_nonfinite;
            continue;
        }
        const double difference = candidate - reference;
        f16_error2 += (long double)difference * difference;
        f16_reference2 += (long double)reference * reference;
        f16_candidate2 += (long double)candidate * candidate;
        f16_dot += (long double)reference * candidate;
        f16_max_abs = std::max(f16_max_abs, std::fabs(difference));
    }
    const double f16_gemm_nmse = (double)(
        f16_error2 / std::max(f16_reference2, 1.0e-30L));
    const double f16_gemm_cosine =
        f16_reference2 > 0.0L && f16_candidate2 > 0.0L ?
        (double)(f16_dot /
            std::sqrt(f16_reference2 * f16_candidate2)) : 0.0;
    const double f16_reference_rms = std::sqrt((double)(
        f16_reference2 / std::max<uint64_t>(batch_lora_floats, 1u)));
    std::fprintf(stderr,
                 "GLM5 NoPE F16 GEMM boundary rows=%u selected=%u "
                 "live_pattern=2049,2050,2051,2048 max_abs=%.9g nmse=%.9g "
                 "cosine=%.12g "
                 "nonfinite=%llu\n",
                 kBatchRows, kProductionSelected, f16_max_abs, f16_gemm_nmse,
                 f16_gemm_cosine, (unsigned long long)f16_nonfinite);
    CHECK(f16_nonfinite == 0u && f16_gemm_nmse <= 2.0e-4 &&
          f16_max_abs <= 0.05 * f16_reference_rms &&
          f16_gemm_cosine >= 0.9999,
          "boundary F16 GEMM NoPE batch remains numerically coherent");
    constexpr uint32_t kWarmIterations = 3u;
    constexpr uint32_t kTimedIterations = 21u;
    for (uint32_t i = 0u; i < kWarmIterations; ++i) {
        CHECK(ds4_rocm_glm5_sparse_attention_exact_rows(
                  d_head_shared_lora, d_batch_low, d_production_cache,
                  d_batch_selected, kBatchRows, kProductionSelected,
                  kProductionRows) &&
              ds4_rocm_glm5_sparse_attention_f16_gemm_rows(
                  d_f16_gemm_lora, d_batch_low, d_production_cache,
                  d_batch_selected, kBatchRows, kProductionSelected,
                  kProductionRows) && ds4_gpu_synchronize(),
              "warm sparse NoPE boundary implementations");
    }
    std::vector<double> exact_wall_us;
    std::vector<double> f16_wall_us;
    exact_wall_us.reserve(kTimedIterations);
    f16_wall_us.reserve(kTimedIterations);
    auto time_call = [&](bool f16) {
        const auto begin = std::chrono::steady_clock::now();
        const int launched = f16 ?
            ds4_rocm_glm5_sparse_attention_f16_gemm_rows(
                d_f16_gemm_lora, d_batch_low, d_production_cache,
                d_batch_selected, kBatchRows, kProductionSelected,
                kProductionRows) :
            ds4_rocm_glm5_sparse_attention_exact_rows(
                d_head_shared_lora, d_batch_low, d_production_cache,
                d_batch_selected, kBatchRows, kProductionSelected,
                kProductionRows);
        if (!launched || !ds4_gpu_synchronize()) return -1.0;
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::micro>(end - begin).count();
    };
    for (uint32_t i = 0u; i < kTimedIterations; ++i) {
        const bool f16_first = (i & 1u) != 0u;
        const double first = time_call(f16_first);
        const double second = time_call(!f16_first);
        CHECK(first > 0.0 && second > 0.0,
              "time sparse NoPE boundary implementations");
        (f16_first ? f16_wall_us : exact_wall_us).push_back(first);
        (f16_first ? exact_wall_us : f16_wall_us).push_back(second);
    }
    std::sort(exact_wall_us.begin(), exact_wall_us.end());
    std::sort(f16_wall_us.begin(), f16_wall_us.end());
    const double exact_median_us = exact_wall_us[kTimedIterations / 2u];
    const double f16_median_us = f16_wall_us[kTimedIterations / 2u];
    std::fprintf(stderr,
                 "GLM5 NoPE boundary warm wall rows=%u exact_us=%.3f "
                 "f16_gemm_us=%.3f change=%.1f%% samples=%u\n",
                 kBatchRows, exact_median_us, f16_median_us,
                 100.0 * (f16_median_us / exact_median_us - 1.0),
                 kTimedIterations);
    std::fprintf(stderr,
                 "GLM5 NoPE fixed-width batch rows=%u selected=%u "
                 "live_pattern=2049,2050,2051,2048 scalar_bit_identical=1 "
                 "head_shared_bit_identical=1 fnv64=%016llx\n",
                 kBatchRows, kProductionSelected,
                 (unsigned long long)fnv1a64_bytes(
                     batch_lora.data(),
                     batch_lora_floats * sizeof(float)));
    const uint32_t tail_rows[] = {1u, 3u, 15u};
    for (uint32_t rows : tail_rows) {
        CHECK(ds4_rocm_glm5_sparse_attention_exact_rows(
                  d_head_shared_lora, d_batch_low, d_production_cache,
                  d_batch_selected, rows, kProductionSelected,
                  kProductionRows) > 0 &&
              ds4_rocm_glm5_sparse_attention_f16_gemm_rows(
                  d_f16_gemm_lora, d_batch_low, d_production_cache,
                  d_batch_selected, rows, kProductionSelected,
                  kProductionRows) > 0 && ds4_gpu_synchronize(),
              "execute sparse NoPE F16 tail geometry");
        const uint64_t count = (uint64_t)rows * kHeads * kKvLora;
        CHECK(ds4_gpu_tensor_read(
                  d_head_shared_lora, 0u, head_shared_lora.data(),
                  count * sizeof(float)) &&
              ds4_gpu_tensor_read(
                  d_f16_gemm_lora, 0u, f16_gemm_lora.data(),
                  count * sizeof(float)),
              "read sparse NoPE F16 tail geometry");
        long double error2 = 0.0L, reference2 = 0.0L;
        double maximum = 0.0;
        uint64_t nonfinite = 0u;
        for (uint64_t i = 0u; i < count; ++i) {
            if (!std::isfinite(f16_gemm_lora[i])) {
                ++nonfinite;
                continue;
            }
            const double error =
                (double)f16_gemm_lora[i] - head_shared_lora[i];
            error2 += (long double)error * error;
            reference2 += (long double)head_shared_lora[i] *
                          head_shared_lora[i];
            maximum = std::max(maximum, std::fabs(error));
        }
        const double tail_nmse = (double)(
            error2 / std::max(reference2, 1.0e-30L));
        const double tail_rms = std::sqrt((double)(
            reference2 / std::max<uint64_t>(count, 1u)));
        std::fprintf(stderr,
                     "GLM5 NoPE F16 GEMM tail rows=%u max_abs=%.9g "
                     "nmse=%.9g ref_rms=%.9g nonfinite=%llu\n",
                     rows, maximum, tail_nmse, tail_rms,
                     (unsigned long long)nonfinite);
        CHECK(nonfinite == 0u && tail_nmse <= 2.0e-4 &&
              maximum <= 0.05 * tail_rms,
              "sparse NoPE F16 tail remains inside Lane-B envelope");
    }
    CHECK(ds4_rocm_glm5_sparse_attention_f16_gemm_rows(
              d_f16_gemm_lora, d_batch_low, d_production_cache,
              d_batch_selected, 17u, kProductionSelected,
              kProductionRows) == -1,
          "sparse NoPE F16 rejects rows above the bounded tile");
    CHECK(ds4_rocm_glm5_sparse_attention_f16_gemm_rows(
              d_f16_gemm_lora, d_batch_low, d_production_cache,
              d_batch_selected, 16u, kProductionSelected - 1u,
              kProductionRows) == -1,
          "sparse NoPE F16 rejects a noncanonical selected stride");
    unsetenv("DS4_ROCM_GLM5_NOPE_ATTN_SHARED_PV");
    ds4_gpu_tensor_free(d_scalar_lora);
    ds4_gpu_tensor_free(d_f16_gemm_lora);
    ds4_gpu_tensor_free(d_head_shared_lora);
    ds4_gpu_tensor_free(d_batch_lora);
    ds4_gpu_tensor_free(d_batch_selected);
    ds4_gpu_tensor_free(d_batch_low);
    ds4_gpu_tensor_free(d_batch_query);

    const uint32_t partial_counts[] = {
        1u, 7u, 17u, 25u, 2047u, 2048u, 2049u, 2050u
    };
    for (uint32_t partial_count : partial_counts) {
        std::vector<int32_t> padded_selected = production_selected;
        for (uint32_t i = partial_count; i < kProductionSelected; ++i)
            padded_selected[i] = -1;
        CHECK(ds4_gpu_tensor_write(
                  d_production_selected, 0u, padded_selected.data(),
                  (uint64_t)padded_selected.size() * sizeof(int32_t)),
              "upload padded partial selected row");
        unsetenv("DS4_ROCM_GLM5_NOPE_ATTN_SHARED_PV");
        CHECK(ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
                  d_production_incumbent, d_production_query, d_production_low,
                  d_production_cache, nullptr, d_production_selected,
                  1u, partial_count, kProductionRows, false,
                  kHeads, kKvLora, kQkNope, 0u, 1u,
                  1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f) &&
              ds4_gpu_synchronize(),
              "execute partial-tile incumbent NoPE lora attention");
        CHECK(setenv("DS4_ROCM_GLM5_NOPE_ATTN_SHARED_PV", "1", 1) == 0 &&
              ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
                  d_production_shared, d_production_query, d_production_low,
                  d_production_cache, nullptr, d_production_selected,
                  1u, partial_count, kProductionRows, false,
                  kHeads, kKvLora, kQkNope, 0u, 1u,
                  1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f) &&
              ds4_gpu_synchronize() &&
              ds4_gpu_tensor_read(
                  d_production_incumbent, 0u, production_incumbent.data(),
                  (uint64_t)production_incumbent.size() * sizeof(float)) &&
              ds4_gpu_tensor_read(
                  d_production_shared, 0u, production_shared.data(),
                  (uint64_t)production_shared.size() * sizeof(float)),
              "execute and read partial-tile shared-PV attention");
        CHECK(std::memcmp(production_incumbent.data(), production_shared.data(),
                          production_incumbent.size() * sizeof(float)) == 0,
              "partial-tile shared-PV output is bit-identical");
        CHECK(ds4_rocm_glm5_sparse_attention_exact_rows(
                  d_production_shared, d_production_low, d_production_cache,
                  d_production_selected, 1u, kProductionSelected,
                  kProductionRows) > 0 &&
              ds4_rocm_glm5_sparse_attention_f16_gemm_rows(
                  d_production_candidate, d_production_low,
                  d_production_cache, d_production_selected, 1u,
                  kProductionSelected, kProductionRows) > 0 &&
              ds4_gpu_synchronize() &&
              ds4_gpu_tensor_read(
                  d_production_shared, 0u, production_shared.data(),
                  (uint64_t)production_shared.size() * sizeof(float)) &&
              ds4_gpu_tensor_read(
                  d_production_candidate, 0u, production_candidate.data(),
                  (uint64_t)production_candidate.size() * sizeof(float)),
              "execute and read padded partial F16 attention");
        long double partial_error2 = 0.0L, partial_reference2 = 0.0L;
        double partial_maximum = 0.0;
        uint64_t partial_nonfinite = 0u;
        for (size_t i = 0u; i < production_shared.size(); ++i) {
            if (!std::isfinite(production_candidate[i])) {
                ++partial_nonfinite;
                continue;
            }
            const double error =
                (double)production_candidate[i] - production_shared[i];
            partial_error2 += (long double)error * error;
            partial_reference2 += (long double)production_shared[i] *
                                  production_shared[i];
            partial_maximum = std::max(partial_maximum, std::fabs(error));
        }
        const double partial_nmse = (double)(partial_error2 /
            std::max(partial_reference2, 1.0e-30L));
        const double partial_rms = std::sqrt((double)(partial_reference2 /
            std::max<size_t>(production_shared.size(), 1u)));
        CHECK(partial_nonfinite == 0u && partial_nmse <= 2.0e-4 &&
              partial_maximum <= 0.05 * partial_rms,
              "padded partial F16 attention remains inside Lane-B envelope");
        std::fprintf(stderr,
                     "GLM5 NoPE partial selected=%u bit_identical=1 "
                     "f16_max_abs=%.9g f16_nmse=%.9g fnv64=%016llx\n",
                     partial_count, partial_maximum, partial_nmse,
                     (unsigned long long)fnv1a64_bytes(
                         production_shared.data(),
                         production_shared.size() * sizeof(float)));
    }

    std::vector<int32_t> all_invalid(kProductionSelected, -1);
    CHECK(ds4_gpu_tensor_write(d_production_selected, 0u, all_invalid.data(),
                               all_invalid.size() * sizeof(int32_t)),
          "upload all-invalid selected rows");
    unsetenv("DS4_ROCM_GLM5_NOPE_ATTN_SHARED_PV");
    CHECK(ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
              d_production_incumbent, d_production_query, d_production_low,
              d_production_cache, nullptr, d_production_selected,
              1u, 25u, kProductionRows, false,
              kHeads, kKvLora, kQkNope, 0u, 1u,
              1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f) &&
          ds4_gpu_synchronize() &&
          setenv("DS4_ROCM_GLM5_NOPE_ATTN_SHARED_PV", "1", 1) == 0 &&
          ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
              d_production_shared, d_production_query, d_production_low,
              d_production_cache, nullptr, d_production_selected,
              1u, 25u, kProductionRows, false,
              kHeads, kKvLora, kQkNope, 0u, 1u,
              1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f) &&
          ds4_gpu_synchronize() &&
          ds4_gpu_tensor_read(
              d_production_incumbent, 0u, production_incumbent.data(),
              (uint64_t)production_incumbent.size() * sizeof(float)) &&
          ds4_gpu_tensor_read(
              d_production_shared, 0u, production_shared.data(),
              (uint64_t)production_shared.size() * sizeof(float)),
          "execute all-invalid incumbent and shared-PV attention");
    CHECK(std::memcmp(production_incumbent.data(), production_shared.data(),
                      production_incumbent.size() * sizeof(float)) == 0,
          "all-invalid shared-PV output is bit-identical");
    for (float value : production_shared) {
        CHECK(value == 0.0f && !std::signbit(value),
              "all-invalid shared-PV output is positive zero");
    }
    CHECK(ds4_rocm_glm5_sparse_attention_f16_gemm_rows(
              d_production_candidate, d_production_low, d_production_cache,
              d_production_selected, 1u, kProductionSelected,
              kProductionRows) > 0 && ds4_gpu_synchronize() &&
          ds4_gpu_tensor_read(
              d_production_candidate, 0u, production_candidate.data(),
              (uint64_t)production_candidate.size() * sizeof(float)),
          "execute and read all-invalid F16 attention");
    for (float value : production_candidate) {
        CHECK(value == 0.0f && !std::signbit(value),
              "all-invalid F16 output is positive zero");
    }
    std::fprintf(stderr,
                 "GLM5 NoPE all-invalid selected=25 bit_identical=1 "
                 "positive_zero=1\n");
    unsetenv("DS4_ROCM_GLM5_NOPE_ATTN_SHARED_PV");

    ds4_gpu_tensor_free(d_production_shared);
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
