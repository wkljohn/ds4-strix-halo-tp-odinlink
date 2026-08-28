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
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#define CHECK(expr, message) do {                                           \
    if (!(expr)) {                                                          \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__);    \
        return false;                                                       \
    }                                                                       \
} while (0)

namespace {

constexpr uint32_t kRows = 10u;
constexpr uint32_t kFirstValid = 1u;
constexpr uint32_t kHidden = 4096u;
constexpr uint32_t kQRank = 1536u;
constexpr uint32_t kHeads = 32u;
constexpr uint32_t kHeadDim = 128u;
constexpr uint32_t kPools = 3u;

template <typename T>
bool read_array(const std::string &path, size_t count, std::vector<T> &out) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return false;
    const std::streamoff size = input.tellg();
    if (size < 0 || (uint64_t)size != (uint64_t)count * sizeof(T)) return false;
    input.seekg(0);
    out.resize(count);
    return (bool)input.read((char *)out.data(), size);
}

uint32_t float_bits(float value) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

uint32_t bf16_order(uint16_t bits) {
    return (bits & 0x8000u)
        ? 0x8000u - (bits & 0x7fffu)
        : 0x8000u + bits;
}

struct ErrorMetric {
    double max_abs = 0.0;
    double nmse = 0.0;
    uint64_t bit_mismatches = 0u;
};

bool compare_values(const char *name, const std::vector<float> &got,
                    const std::vector<float> &expected,
                    double max_abs_limit, double nmse_limit,
                    bool require_bf16, ErrorMetric *metric_out = nullptr) {
    CHECK(got.size() == expected.size(), "comparison shape");
    ErrorMetric metric;
    double error2 = 0.0;
    double reference2 = 0.0;
    double reference_max = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        CHECK(std::isfinite(got[i]) && std::isfinite(expected[i]),
              "finite component output");
        if (require_bf16) {
            CHECK((float_bits(got[i]) & 0xffffu) == 0u,
                  "device output preserves BF16 boundary");
            CHECK((float_bits(expected[i]) & 0xffffu) == 0u,
                  "oracle output preserves BF16 boundary");
            const uint32_t got_order =
                bf16_order((uint16_t)(float_bits(got[i]) >> 16u));
            const uint32_t expected_order =
                bf16_order((uint16_t)(float_bits(expected[i]) >> 16u));
            const uint32_t ulp_distance = got_order > expected_order
                ? got_order - expected_order : expected_order - got_order;
            CHECK(ulp_distance <= 1u,
                  "device output stays within one BF16 ULP");
        }
        const double difference = (double)got[i] - expected[i];
        metric.max_abs = std::max(metric.max_abs, std::fabs(difference));
        error2 += difference * difference;
        reference2 += (double)expected[i] * expected[i];
        reference_max = std::max(reference_max, std::fabs((double)expected[i]));
        metric.bit_mismatches += float_bits(got[i]) != float_bits(expected[i]);
    }
    metric.nmse = error2 / std::max(reference2, 1.0e-30);
    std::fprintf(stderr,
        "GLM5 NoPE %-16s count=%zu mismatch=%llu max_abs=%.9g nmse=%.9g\n",
        name, got.size(), (unsigned long long)metric.bit_mismatches,
        metric.max_abs, metric.nmse);
    CHECK(metric.max_abs <= max_abs_limit && metric.nmse <= nmse_limit,
          "component numerical envelope");
    CHECK(reference_max >= 1.0e-6,
          "component reference is non-degenerate");
    if (metric_out) *metric_out = metric;
    return true;
}

bool read_tensor(ds4_gpu_tensor *tensor, size_t count,
                 std::vector<float> &out) {
    out.resize(count);
    return ds4_gpu_tensor_read(tensor, 0u, out.data(),
                               (uint64_t)count * sizeof(float));
}

bool run_test() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    const char *oracle_prefix = std::getenv("DS4_GLM5_NOPE_ORACLE_PREFIX");
    CHECK(model && model[0] && oracle_prefix && oracle_prefix[0],
          "model and NoPE oracle environment");

    std::vector<float> hidden, q_resid, expected_q, expected_kraw,
        expected_key, expected_gate, expected_weights_unscaled,
        expected_weights, expected_pooled, expected_scores;
    std::vector<uint32_t> valid, expected_pool_valid;
    std::vector<int32_t> expected_pool_indices;
    const std::string base(oracle_prefix);
    CHECK(read_array(base + ".hidden.f32", kRows * kHidden, hidden) &&
          read_array(base + ".qresid.f32", kQRank, q_resid) &&
          read_array(base + ".valid.u32", kRows, valid) &&
          read_array(base + ".q.f32", kHeads * kHeadDim, expected_q) &&
          read_array(base + ".kraw.f32", kRows * kHeadDim, expected_kraw) &&
          read_array(base + ".key.f32", kRows * kHeadDim, expected_key) &&
          read_array(base + ".gate.f32", kRows * kHeadDim, expected_gate) &&
          read_array(base + ".weights_unscaled.f32", kHeads,
                     expected_weights_unscaled) &&
          read_array(base + ".weights.f32", kHeads, expected_weights) &&
          read_array(base + ".pooled.f32", kPools * kHeadDim,
                     expected_pooled) &&
          read_array(base + ".pool_indices.i32", kPools * 4u,
                     expected_pool_indices) &&
          read_array(base + ".pool_valid.u32", kPools,
                     expected_pool_valid) &&
          read_array(base + ".scores.f32", kPools, expected_scores),
          "read same-GGUF NoPE oracle dumps");

    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM5 GGUF directory");
    uint64_t q_offset = 0u, k_offset = 0u, weight_offset = 0u,
             gate_offset = 0u, ape_offset = 0u,
             norm_weight_offset = 0u, norm_bias_offset = 0u;
    CHECK(gguf.tensor("blk.3.indexer.attn_q_b.weight", {1536u, 4096u},
                      30u, q_offset) &&
          gguf.tensor("blk.3.indexer.attn_k.weight", {4096u, 128u},
                      30u, k_offset) &&
          gguf.tensor("blk.3.indexer.proj.weight", {4096u, 32u},
                      30u, weight_offset) &&
          gguf.tensor("blk.3.indexer.pool_gate.weight", {4096u, 128u},
                      30u, gate_offset) &&
          gguf.tensor("blk.3.indexer.pool_ape.weight", {128u, 4u},
                      30u, ape_offset) &&
          gguf.tensor("blk.3.indexer.k_norm.weight", {128u},
                      0u, norm_weight_offset) &&
          gguf.tensor("blk.3.indexer.k_norm.bias", {128u},
                      0u, norm_bias_offset),
          "bind real block-3 indexer tensors");

    ds4_gpu_config config = {};
    config.n_gpus = 1u;
    config.device_indices[0] = 0u;
    CHECK(ds4_gpu_init_multi(&config) &&
          ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map) &&
          ds4_gpu_set_model_map(gguf.map, gguf.size),
          "initialize gfx1151 and register model map");
    ds4_tp_test_reset_exchange_calls();

    const auto alloc_f32 = [](uint64_t count) {
        return ds4_gpu_tensor_alloc(count * sizeof(float));
    };
    ds4_gpu_tensor *d_hidden = alloc_f32((uint64_t)kRows * kHidden);
    ds4_gpu_tensor *d_q_resid = alloc_f32(kQRank);
    ds4_gpu_tensor *d_q = alloc_f32(kHeads * kHeadDim);
    ds4_gpu_tensor *d_kraw = alloc_f32((uint64_t)kRows * kHeadDim);
    ds4_gpu_tensor *d_key = alloc_f32((uint64_t)kRows * kHeadDim);
    ds4_gpu_tensor *d_gate = alloc_f32((uint64_t)kRows * kHeadDim);
    ds4_gpu_tensor *d_weights_unscaled = alloc_f32(kHeads);
    ds4_gpu_tensor *d_weights = alloc_f32(kHeads);
    ds4_gpu_tensor *d_valid = ds4_gpu_tensor_alloc(
        (uint64_t)kRows * sizeof(uint32_t));
    ds4_gpu_tensor *d_pooled = alloc_f32(kPools * kHeadDim);
    ds4_gpu_tensor *d_pool_indices = ds4_gpu_tensor_alloc(
        (uint64_t)kPools * 4u * sizeof(int32_t));
    ds4_gpu_tensor *d_pool_valid = ds4_gpu_tensor_alloc(
        (uint64_t)kPools * sizeof(uint32_t));
    ds4_gpu_tensor *d_scores = alloc_f32(kPools);
    CHECK(d_hidden && d_q_resid && d_q && d_kraw && d_key && d_gate &&
          d_weights_unscaled && d_weights && d_valid && d_pooled &&
          d_pool_indices && d_pool_valid && d_scores,
          "allocate bounded NoPE component tensors");
    ds4_gpu_tensor *d_query_hidden = ds4_gpu_tensor_view(
        d_hidden, (uint64_t)(kRows - 1u) * kHidden * sizeof(float),
        (uint64_t)kHidden * sizeof(float));
    CHECK(d_query_hidden &&
          ds4_gpu_tensor_write(d_hidden, 0u, hidden.data(),
                               (uint64_t)hidden.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(d_q_resid, 0u, q_resid.data(),
                               (uint64_t)q_resid.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(d_valid, 0u, valid.data(),
                               (uint64_t)valid.size() * sizeof(uint32_t)),
          "upload NoPE component inputs");

    CHECK(ds4_gpu_matmul_bf16_tensor(
              d_q, gguf.map, gguf.size, q_offset, kQRank,
              kHeads * kHeadDim, d_q_resid, 1u) &&
          ds4_gpu_round_bf16_inplace_tensor(
              d_q, kHeads * kHeadDim, 1.0f) &&
          ds4_gpu_matmul_bf16_tensor(
              d_kraw, gguf.map, gguf.size, k_offset, kHidden,
              kHeadDim, d_hidden, kRows) &&
          ds4_gpu_round_bf16_inplace_tensor(
              d_kraw, (uint64_t)kRows * kHeadDim, 1.0f) &&
          ds4_gpu_glm_store_indexer_k_tensor(
              d_key, d_kraw, gguf.map, gguf.size,
              norm_weight_offset, norm_bias_offset, 0u, kRows, kRows,
              kHeadDim, 0u, 1u, 1.0e-6f, 1.0f, 1.0f,
              0.0f, 1.0f, 0.0f, 0.0f, false) &&
          ds4_gpu_round_bf16_inplace_tensor(
              d_key, (uint64_t)kRows * kHeadDim, 1.0f) &&
          ds4_gpu_matmul_bf16_tensor(
              d_gate, gguf.map, gguf.size, gate_offset, kHidden,
              kHeadDim, d_hidden, kRows) &&
          ds4_gpu_round_bf16_inplace_tensor(
              d_gate, (uint64_t)kRows * kHeadDim, 1.0f) &&
          ds4_gpu_matmul_bf16_tensor(
              d_weights_unscaled, gguf.map, gguf.size, weight_offset,
              kHidden, kHeads, d_query_hidden, 1u) &&
          ds4_gpu_round_bf16_inplace_tensor(
              d_weights_unscaled, kHeads, 1.0f) &&
          ds4_gpu_tensor_copy(d_weights, 0u, d_weights_unscaled, 0u,
                              (uint64_t)kHeads * sizeof(float)) &&
          ds4_gpu_round_bf16_inplace_tensor(
              d_weights, kHeads, 0.1767766952966369f) &&
          ds4_gpu_glm5_kpool_tensor(
              d_pooled, d_pool_indices, d_pool_valid,
              d_key, d_gate, d_valid, gguf.map, gguf.size, ape_offset,
              kRows, kHeadDim, 4u, kFirstValid) &&
          ds4_gpu_glm_indexer_score_one_tensor(
              d_scores, d_q, d_weights, d_pooled, kPools,
              kHeads, kHeadDim, 0.08838834764831845f, false) &&
          ds4_gpu_glm5_mask_pool_scores_tensor(
              d_scores, d_pool_valid, kPools) &&
          ds4_gpu_synchronize(),
          "execute real block-3 NoPE masked-score path");

    std::vector<float> got_q, got_kraw, got_key, got_gate,
        got_weights_unscaled, got_weights, got_pooled, got_scores;
    std::vector<int32_t> got_pool_indices(kPools * 4u);
    std::vector<uint32_t> got_pool_valid(kPools);
    CHECK(read_tensor(d_q, kHeads * kHeadDim, got_q) &&
          read_tensor(d_kraw, kRows * kHeadDim, got_kraw) &&
          read_tensor(d_key, kRows * kHeadDim, got_key) &&
          read_tensor(d_gate, kRows * kHeadDim, got_gate) &&
          read_tensor(d_weights_unscaled, kHeads, got_weights_unscaled) &&
          read_tensor(d_weights, kHeads, got_weights) &&
          read_tensor(d_pooled, kPools * kHeadDim, got_pooled) &&
          read_tensor(d_scores, kPools, got_scores) &&
          ds4_gpu_tensor_read(d_pool_indices, 0u, got_pool_indices.data(),
                              (uint64_t)got_pool_indices.size() * sizeof(int32_t)) &&
          ds4_gpu_tensor_read(d_pool_valid, 0u, got_pool_valid.data(),
                              (uint64_t)got_pool_valid.size() * sizeof(uint32_t)),
          "read NoPE component outputs");

    CHECK(compare_values("q projection", got_q, expected_q,
                         0.03125, 1.0e-8, true) &&
          compare_values("k projection", got_kraw, expected_kraw,
                         0.001953125, 1.0e-8, true) &&
          compare_values("k layernorm", got_key, expected_key,
                         0.000244140625, 1.0e-8, true) &&
          compare_values("pool gate", got_gate, expected_gate,
                         0.0000152587890625, 1.0e-8, true) &&
          compare_values("head weights", got_weights_unscaled,
                         expected_weights_unscaled, 0.03125, 1.0e-8, true) &&
          compare_values("scaled weights", got_weights, expected_weights,
                         1.0e-5, 1.0e-10, false) &&
          compare_values("pooled keys", got_pooled, expected_pooled,
                         0.03125, 1.0e-8, true),
          "NoPE intermediate numerical gates");
    CHECK(got_pool_indices == expected_pool_indices &&
          got_pool_valid == expected_pool_valid,
          "NoPE pool structure matches oracle");
    CHECK(expected_pool_valid == std::vector<uint32_t>({1u, 1u, 0u}) &&
          float_bits(got_scores[2]) == float_bits(-std::numeric_limits<float>::max()) &&
          float_bits(expected_scores[2]) == float_bits(-std::numeric_limits<float>::max()),
          "invalid tail pool uses exact finite minimum mask");
    got_scores.resize(2u);
    expected_scores.resize(2u);
    CHECK(compare_values("masked scores", got_scores, expected_scores,
                         1.0e-4, 1.0e-8, false),
          "NoPE final valid-score numerical gate");
    CHECK(ds4_tp_test_get_exchange_calls() == 0u,
          "rank-local NoPE score invokes no TP exchange");

    ds4_gpu_tensor_free(d_query_hidden);
    ds4_gpu_tensor_free(d_scores);
    ds4_gpu_tensor_free(d_pool_valid);
    ds4_gpu_tensor_free(d_pool_indices);
    ds4_gpu_tensor_free(d_pooled);
    ds4_gpu_tensor_free(d_valid);
    ds4_gpu_tensor_free(d_weights);
    ds4_gpu_tensor_free(d_weights_unscaled);
    ds4_gpu_tensor_free(d_gate);
    ds4_gpu_tensor_free(d_key);
    ds4_gpu_tensor_free(d_kraw);
    ds4_gpu_tensor_free(d_q);
    ds4_gpu_tensor_free(d_q_resid);
    ds4_gpu_tensor_free(d_hidden);
    ds4_gpu_cleanup();
    std::fprintf(stderr,
                 "PASS same-GGUF GLM5 block-3 NoPE masked-score gate\n");
    return true;
}

}  // namespace

int main() { return run_test() ? 0 : 1; }
