#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
extern "C" {
#include "ds4_tp.h"
}
#include "tests/glm5_gguf_test.hpp"

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

namespace {

constexpr uint32_t kDim = 128u;
constexpr uint32_t kPool = 4u;

float bf16_to_f32(uint16_t value) {
    union { uint32_t u; float f; } bits = {(uint32_t)value << 16};
    return bits.f;
}

float round_bf16(float value) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t magnitude = bits & 0x7fffffffu;
    uint16_t rounded;
    if (magnitude > 0x7f800000u) {
        rounded = (uint16_t)((bits >> 16u) | 0x0040u);
    } else {
        rounded = (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
    }
    return bf16_to_f32(rounded);
}

uint32_t float_bits(float value) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float deterministic_key(uint32_t row, uint32_t channel, uint32_t pattern) {
    if (pattern != 0u) {
        const uint32_t mixed = row * 2246822519u + channel * 3266489917u +
                               pattern * 668265263u;
        return (float)((int)(mixed % 4093u) - 2046) / 1024.0f;
    }
    return std::sin((float)(row * 131u + channel * 17u + 3u) * 0.00031f) *
        0.3f + std::cos((float)(row * 7u + channel * 29u + 11u) * 0.00017f) *
        0.05f;
}

float deterministic_gate(uint32_t row, uint32_t channel, uint32_t pattern) {
    if (pattern != 0u) {
        const uint32_t mixed = row * 2654435761u + channel * 2246822519u +
                               pattern * 3266489917u;
        return (float)((int)(mixed % 2047u) - 1023) / 128.0f;
    }
    return std::sin((float)(row * 19u + channel * 23u + 5u) * 0.00043f) *
        0.7f + (float)((row + 3u * channel) % 11u) * 0.013f;
}

bool run_case(const Glm5TestGGUF &gguf, uint64_t ape_offset,
              uint32_t n_rows, uint32_t first_valid,
              bool invalidate_middle, uint32_t pattern = 0u) {
    const uint32_t n_pools = (n_rows + 3u) / 4u;
    std::vector<float> keys((size_t)n_rows * kDim);
    std::vector<float> gates((size_t)n_rows * kDim);
    std::vector<uint32_t> valid(n_rows, 1u);
    for (uint32_t row = 0; row < n_rows; ++row) {
        if (row < first_valid) valid[row] = 0u;
        for (uint32_t d = 0; d < kDim; ++d) {
            keys[(uint64_t)row * kDim + d] =
                deterministic_key(row, d, pattern);
            gates[(uint64_t)row * kDim + d] =
                deterministic_gate(row, d, pattern);
        }
    }
    if (invalidate_middle && n_rows >= first_valid + 8u)
        valid[first_valid + 5u] = 0u;

    const uint16_t *ape = (const uint16_t *)(gguf.map + ape_offset);
    std::vector<float> expected((size_t)n_pools * kDim, 0.0f);
    std::vector<int32_t> expected_indices((size_t)n_pools * kPool, -1);
    std::vector<uint32_t> expected_valid(n_pools, 0u);
    for (uint32_t pool = 0; pool < n_pools; ++pool) {
        const uint32_t start = first_valid + pool * kPool;
        bool complete = true;
        bool any_valid = false;
        bool member_valid[kPool];
        for (uint32_t j = 0; j < kPool; ++j) {
            member_valid[j] = start < n_rows && j < n_rows - start &&
                              valid[start + j] != 0u;
            complete = complete && member_valid[j];
            any_valid = any_valid || member_valid[j];
        }
        expected_valid[pool] = complete ? 1u : 0u;
        for (uint32_t j = 0; j < kPool; ++j)
            expected_indices[(uint64_t)pool * kPool + j] =
                member_valid[j] ? (int32_t)(start + j) : -1;
        if (!any_valid) continue;
        for (uint32_t d = 0; d < kDim; ++d) {
            float logits[kPool], maximum = -INFINITY, denominator = 0.0f;
            for (uint32_t j = 0; j < kPool; ++j) {
                logits[j] = member_valid[j]
                    ? round_bf16(gates[(uint64_t)(start + j) * kDim + d]) +
                        bf16_to_f32(ape[(uint64_t)j * kDim + d])
                    : -INFINITY;
                maximum = std::max(maximum, logits[j]);
            }
            for (uint32_t j = 0; j < kPool; ++j) {
                logits[j] = std::exp(logits[j] - maximum);
                denominator += logits[j];
            }
            for (uint32_t j = 0; j < kPool; ++j) {
                if (member_valid[j]) {
                    const float probability =
                        round_bf16(logits[j] / denominator);
                    const float key = round_bf16(
                        keys[(uint64_t)(start + j) * kDim + d]);
                    expected[(uint64_t)pool * kDim + d] +=
                        round_bf16(probability * key);
                }
            }
            expected[(uint64_t)pool * kDim + d] =
                round_bf16(expected[(uint64_t)pool * kDim + d]);
        }
    }

    const uint64_t rows_bytes = (uint64_t)n_rows * kDim * sizeof(float);
    const uint64_t pooled_bytes = (uint64_t)n_pools * kDim * sizeof(float);
    ds4_gpu_tensor *d_keys = ds4_gpu_tensor_alloc(rows_bytes);
    ds4_gpu_tensor *d_gates = ds4_gpu_tensor_alloc(rows_bytes);
    ds4_gpu_tensor *d_valid = ds4_gpu_tensor_alloc((uint64_t)n_rows * sizeof(uint32_t));
    ds4_gpu_tensor *d_pooled = ds4_gpu_tensor_alloc(pooled_bytes);
    ds4_gpu_tensor *d_indices = ds4_gpu_tensor_alloc((uint64_t)n_pools * kPool * sizeof(int32_t));
    ds4_gpu_tensor *d_pool_valid = ds4_gpu_tensor_alloc((uint64_t)n_pools * sizeof(uint32_t));
    CHECK(d_keys && d_gates && d_valid && d_pooled && d_indices &&
          d_pool_valid &&
          ds4_gpu_tensor_write(d_keys, 0u, keys.data(), rows_bytes) &&
          ds4_gpu_tensor_write(d_gates, 0u, gates.data(), rows_bytes) &&
          ds4_gpu_tensor_write(d_valid, 0u, valid.data(),
                               (uint64_t)n_rows * sizeof(uint32_t)) &&
          ds4_gpu_glm5_kpool_tensor(
              d_pooled, d_indices, d_pool_valid, d_keys, d_gates, d_valid,
              gguf.map, gguf.size, ape_offset, n_rows, kDim, kPool,
              first_valid) && ds4_gpu_synchronize(),
          "execute GLM5 learned pool-4 compression");

    std::vector<float> got(expected.size());
    std::vector<int32_t> got_indices(expected_indices.size());
    std::vector<uint32_t> got_valid(expected_valid.size());
    CHECK(ds4_gpu_tensor_read(d_pooled, 0u, got.data(), pooled_bytes) &&
          ds4_gpu_tensor_read(d_indices, 0u, got_indices.data(),
                              got_indices.size() * sizeof(int32_t)) &&
          ds4_gpu_tensor_read(d_pool_valid, 0u, got_valid.data(),
                              got_valid.size() * sizeof(uint32_t)),
          "read GLM5 learned pool-4 outputs");
    double max_abs = 0.0;
    uint64_t mismatches = 0u;
    for (size_t i = 0; i < got.size(); ++i) {
        CHECK(std::isfinite(got[i]), "finite GLM5 pooled key");
        CHECK((float_bits(got[i]) & 0xffffu) == 0u,
              "GLM5 pooled key preserves upstream BF16 boundary");
        mismatches += float_bits(got[i]) != float_bits(expected[i]);
        max_abs = std::max(max_abs,
                           std::fabs((double)got[i] - expected[i]));
    }
    CHECK(got_indices == expected_indices && got_valid == expected_valid &&
          max_abs <= 1.0e-2 && mismatches <= got.size() / 10000u,
          "GLM5 pool indices, validity and numerical envelope");
    std::fprintf(stderr,
        "GLM5 kpool rows=%u first=%u middle_invalid=%d pattern=%u pools=%u "
        "mismatches=%llu max_abs=%.9g\n",
        n_rows, first_valid, invalidate_middle ? 1 : 0, pattern, n_pools,
        (unsigned long long)mismatches, max_abs);

    ds4_gpu_tensor_free(d_pool_valid);
    ds4_gpu_tensor_free(d_indices);
    ds4_gpu_tensor_free(d_pooled);
    ds4_gpu_tensor_free(d_valid);
    ds4_gpu_tensor_free(d_gates);
    ds4_gpu_tensor_free(d_keys);
    return true;
}

bool run_batch_publication_equivalence(const Glm5TestGGUF &gguf,
                                       uint64_t ape_offset) {
    constexpr uint32_t n_rows = 256u;
    constexpr uint32_t n_pools = n_rows / kPool;
    std::vector<float> keys((size_t)n_rows * kDim);
    std::vector<float> gates((size_t)n_rows * kDim);
    std::vector<uint32_t> valid(n_rows, 1u);
    for (uint32_t row = 0u; row < n_rows; ++row) {
        for (uint32_t d = 0u; d < kDim; ++d) {
            keys[(uint64_t)row * kDim + d] =
                deterministic_key(row, d, 7u);
            gates[(uint64_t)row * kDim + d] =
                deterministic_gate(row, d, 7u);
        }
    }
    const uint64_t rows_bytes = (uint64_t)n_rows * kDim * sizeof(float);
    const uint64_t pooled_bytes = (uint64_t)n_pools * kDim * sizeof(float);
    const uint64_t indices_bytes =
        (uint64_t)n_pools * kPool * sizeof(int32_t);
    const uint64_t valid_bytes = (uint64_t)n_pools * sizeof(uint32_t);
    ds4_gpu_tensor *d_keys = ds4_gpu_tensor_alloc(rows_bytes);
    ds4_gpu_tensor *d_gates = ds4_gpu_tensor_alloc(rows_bytes);
    ds4_gpu_tensor *d_input_valid =
        ds4_gpu_tensor_alloc((uint64_t)n_rows * sizeof(uint32_t));
    ds4_gpu_tensor *d_reference = ds4_gpu_tensor_alloc(pooled_bytes);
    ds4_gpu_tensor *d_candidate = ds4_gpu_tensor_alloc(pooled_bytes);
    ds4_gpu_tensor *d_ref_indices = ds4_gpu_tensor_alloc(indices_bytes);
    ds4_gpu_tensor *d_got_indices = ds4_gpu_tensor_alloc(indices_bytes);
    ds4_gpu_tensor *d_ref_valid = ds4_gpu_tensor_alloc(valid_bytes);
    ds4_gpu_tensor *d_got_valid = ds4_gpu_tensor_alloc(valid_bytes);
    CHECK(d_keys && d_gates && d_input_valid && d_reference && d_candidate &&
          d_ref_indices && d_got_indices && d_ref_valid && d_got_valid &&
          ds4_gpu_tensor_write(d_keys, 0u, keys.data(), rows_bytes) &&
          ds4_gpu_tensor_write(d_gates, 0u, gates.data(), rows_bytes) &&
          ds4_gpu_tensor_write(d_input_valid, 0u, valid.data(),
                               (uint64_t)n_rows * sizeof(uint32_t)) &&
          ds4_gpu_glm5_kpool_tensor(
              d_reference, d_ref_indices, d_ref_valid, d_keys, d_gates,
              d_input_valid, gguf.map, gguf.size, ape_offset, n_rows,
              kDim, kPool, 0u) &&
          ds4_gpu_glm5_publish_pools_batch_tensor(
              d_candidate, d_got_indices, d_got_valid, d_keys, d_gates,
              gguf.map, gguf.size, ape_offset, 0u, n_rows, kDim) &&
          ds4_gpu_synchronize(),
          "execute aligned GLM5 batch pool publication");
    std::vector<float> reference((size_t)n_pools * kDim);
    std::vector<float> candidate(reference.size());
    std::vector<int32_t> ref_indices((size_t)n_pools * kPool);
    std::vector<int32_t> got_indices(ref_indices.size());
    std::vector<uint32_t> ref_valid(n_pools), got_valid(n_pools);
    CHECK(ds4_gpu_tensor_read(d_reference, 0u, reference.data(),
                              pooled_bytes) &&
          ds4_gpu_tensor_read(d_candidate, 0u, candidate.data(),
                              pooled_bytes) &&
          ds4_gpu_tensor_read(d_ref_indices, 0u, ref_indices.data(),
                              indices_bytes) &&
          ds4_gpu_tensor_read(d_got_indices, 0u, got_indices.data(),
                              indices_bytes) &&
          ds4_gpu_tensor_read(d_ref_valid, 0u, ref_valid.data(),
                              valid_bytes) &&
          ds4_gpu_tensor_read(d_got_valid, 0u, got_valid.data(),
                              valid_bytes),
          "read aligned GLM5 batch pool publication");
    CHECK(std::memcmp(reference.data(), candidate.data(), pooled_bytes) == 0 &&
          ref_indices == got_indices && ref_valid == got_valid,
          "aligned batch pool publication is bit-identical");
    std::fprintf(stderr,
                 "GLM5 batch pool publication rows=%u pools=%u bit_exact=1\n",
                 n_rows, n_pools);
    ds4_gpu_tensor_free(d_got_valid);
    ds4_gpu_tensor_free(d_ref_valid);
    ds4_gpu_tensor_free(d_got_indices);
    ds4_gpu_tensor_free(d_ref_indices);
    ds4_gpu_tensor_free(d_candidate);
    ds4_gpu_tensor_free(d_reference);
    ds4_gpu_tensor_free(d_input_valid);
    ds4_gpu_tensor_free(d_gates);
    ds4_gpu_tensor_free(d_keys);
    return true;
}

bool run_test() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    CHECK(model && model[0], "DS4_GLM5_MODEL environment");
    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM5 GGUF directory");
    uint64_t ape_offset = 0u;
    CHECK(gguf.tensor("blk.3.indexer.pool_ape.weight", {128u, 4u}, 30u,
                      ape_offset),
          "bind real block-3 BF16 pool APE");
    ds4_gpu_config config = {};
    config.n_gpus = 1u;
    config.device_indices[0] = 0u;
    CHECK(ds4_gpu_init_multi(&config) &&
          ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map) &&
          ds4_gpu_set_model_map(gguf.map, gguf.size),
          "initialize gfx1151 and register GLM5 model map");
    ds4_tp_test_reset_exchange_calls();

    const uint32_t lengths[] = {1u, 3u, 4u, 5u, 2047u, 2048u, 2049u, 8192u};
    for (uint32_t length : lengths)
        CHECK(run_case(gguf, ape_offset, length, 0u, false),
              "GLM5 kpool length sweep");
    CHECK(run_case(gguf, ape_offset, 19u, 2u, false),
          "GLM5 kpool left-padding alignment");
    CHECK(run_case(gguf, ape_offset, 9u, 1u, false) &&
          run_case(gguf, ape_offset, 17u, 1u, false),
          "GLM5 kpool upstream raw-axis count with left padding");
    CHECK(run_case(gguf, ape_offset, 20u, 0u, true),
          "GLM5 kpool invalid middle pool");
    for (uint32_t pattern = 1u; pattern <= 8u; ++pattern)
        CHECK(run_case(gguf, ape_offset, 8192u, pattern & 3u, false,
                       pattern),
              "GLM5 kpool wide-logit numerical stress");
    CHECK(run_batch_publication_equivalence(gguf, ape_offset),
          "GLM5 aligned batch pool publication equivalence");
    CHECK(ds4_tp_test_get_exchange_calls() == 0u,
          "GLM5 kpool invokes no TP exchange API");

    /* Fail closed rather than reinterpret a GLM-5.2/other layout. */
    ds4_gpu_tensor *tiny = ds4_gpu_tensor_alloc(16u);
    CHECK(tiny && !ds4_gpu_glm5_kpool_tensor(
              tiny, tiny, tiny, tiny, tiny, tiny, gguf.map, gguf.size,
              ape_offset, 1u, 64u, 4u, 0u) &&
          !ds4_gpu_glm5_kpool_tensor(
              tiny, tiny, tiny, tiny, tiny, tiny, gguf.map, gguf.size,
              ape_offset, 1u, 128u, 8u, 0u),
          "GLM5 kpool rejects wrong head and pool dimensions");
    ds4_gpu_tensor_free(tiny);
    ds4_gpu_cleanup();
    std::fprintf(stderr,
                 "PASS same-GGUF GLM5 learned pool-4 boundary sweep\n");
    return true;
}

}  // namespace

int main() { return run_test() ? 0 : 1; }
