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
constexpr uint32_t kHeads = 64u;
constexpr uint32_t kHeadDim = 256u;
constexpr uint32_t kKvLora = 512u;
constexpr uint32_t kIndexHeads = 32u;
constexpr uint32_t kIndexDim = 128u;
constexpr uint32_t kPools = 3u;
constexpr uint32_t kSelectedPools = 2u;
constexpr uint32_t kSelectedTokens = 9u;
constexpr uint32_t kTokenBudget = 2048u;
constexpr uint32_t kExpandedWidth = kTokenBudget + 3u;

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

template <typename T>
bool read_tensor(ds4_gpu_tensor *tensor, size_t count, std::vector<T> &out) {
    out.resize(count);
    return ds4_gpu_tensor_read(tensor, 0u, out.data(),
                               (uint64_t)count * sizeof(T));
}

bool compare_values(const char *name, const std::vector<float> &got,
                    const std::vector<float> &expected,
                    double max_abs_limit, double nmse_limit) {
    CHECK(got.size() == expected.size(), "MLA composition comparison shape");
    double maximum = 0.0, error2 = 0.0, reference2 = 0.0;
    double reference_max = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        CHECK(std::isfinite(got[i]) && std::isfinite(expected[i]),
              "finite MLA composition output");
        const double error = (double)got[i] - expected[i];
        maximum = std::max(maximum, std::fabs(error));
        error2 += error * error;
        reference2 += (double)expected[i] * expected[i];
        reference_max = std::max(reference_max,
                                 std::fabs((double)expected[i]));
    }
    const double nmse = error2 / std::max(reference2, 1.0e-30);
    std::fprintf(stderr,
                 "GLM5 MLA compose %-12s count=%zu max_abs=%.9g "
                 "reference_max=%.9g nmse=%.9g\n",
                 name, got.size(), maximum, reference_max, nmse);
    CHECK(reference_max >= 1.0e-6, "non-degenerate MLA composition reference");
    CHECK(maximum <= max_abs_limit && nmse <= nmse_limit,
          "MLA composition numerical envelope");
    return true;
}

bool run_test() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    const char *oracle_prefix =
        std::getenv("DS4_GLM5_MLA_COMPOSE_ORACLE_PREFIX");
    CHECK(model && model[0] && oracle_prefix && oracle_prefix[0],
          "model and MLA composition oracle environment");
    const std::string base(oracle_prefix);

    std::vector<float> hidden, expected_q_resid, expected_query,
        expected_kv_norm, expected_qk_low, expected_index_q,
        expected_index_key, expected_pool_gate, expected_head_weights,
        expected_pooled, expected_pool_scores, expected_heads;
    std::vector<uint32_t> valid, expected_pool_valid, expected_selected_pools;
    std::vector<int32_t> expected_pool_indices, expected_selected_tokens;
    CHECK(read_array(base + ".hidden.f32", kRows * kHidden, hidden) &&
          read_array(base + ".q_resid.f32", kQRank, expected_q_resid) &&
          read_array(base + ".query.f32", kHeads * kHeadDim,
                     expected_query) &&
          read_array(base + ".kv_norm.f32", kRows * kKvLora,
                     expected_kv_norm) &&
          read_array(base + ".qk_low.f32", kHeads * kKvLora,
                     expected_qk_low) &&
          read_array(base + ".valid.u32", kRows, valid) &&
          read_array(base + ".index_q.f32", kIndexHeads * kIndexDim,
                     expected_index_q) &&
          read_array(base + ".index_key.f32", kRows * kIndexDim,
                     expected_index_key) &&
          read_array(base + ".pool_gate.f32", kRows * kIndexDim,
                     expected_pool_gate) &&
          read_array(base + ".head_weights.f32", kIndexHeads,
                     expected_head_weights) &&
          read_array(base + ".pooled.f32", kPools * kIndexDim,
                     expected_pooled) &&
          read_array(base + ".pool_indices.i32", kPools * 4u,
                     expected_pool_indices) &&
          read_array(base + ".pool_valid.u32", kPools,
                     expected_pool_valid) &&
          read_array(base + ".pool_scores.f32", kPools,
                     expected_pool_scores) &&
          read_array(base + ".selected_pools.u32", kSelectedPools,
                     expected_selected_pools) &&
          read_array(base + ".selected_tokens.i32", kSelectedTokens,
                     expected_selected_tokens) &&
          read_array(base + ".heads.f32", kHeads * kHeadDim,
                     expected_heads),
          "read sparse-MLA heads composition oracle dumps");

    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM5 GGUF directory");
    uint64_t q_a = 0u, q_norm = 0u, q_b = 0u, kv_a = 0u,
             kv_norm_w = 0u, k_b = 0u, v_b = 0u, index_q_w = 0u,
             index_k_w = 0u, index_weight_w = 0u, pool_gate_w = 0u,
             pool_ape = 0u, index_norm_w = 0u, index_norm_b = 0u;
    CHECK(gguf.tensor("blk.3.attn_q_a.weight", {4096u, 1536u}, 8u, q_a) &&
          gguf.tensor("blk.3.attn_q_a_norm.weight", {1536u}, 0u, q_norm) &&
          gguf.tensor("blk.3.attn_q_b.weight", {1536u, 16384u}, 8u, q_b) &&
          gguf.tensor("blk.3.attn_kv_a_mqa.weight", {4096u, 512u}, 8u, kv_a) &&
          gguf.tensor("blk.3.attn_kv_a_norm.weight", {512u}, 0u, kv_norm_w) &&
          gguf.tensor("blk.3.attn_k_b.weight", {256u, 512u, 64u}, 8u, k_b) &&
          gguf.tensor("blk.3.attn_v_b.weight", {512u, 256u, 64u}, 8u, v_b) &&
          gguf.tensor("blk.3.indexer.attn_q_b.weight", {1536u, 4096u}, 30u, index_q_w) &&
          gguf.tensor("blk.3.indexer.attn_k.weight", {4096u, 128u}, 30u, index_k_w) &&
          gguf.tensor("blk.3.indexer.proj.weight", {4096u, 32u}, 30u, index_weight_w) &&
          gguf.tensor("blk.3.indexer.pool_gate.weight", {4096u, 128u}, 30u, pool_gate_w) &&
          gguf.tensor("blk.3.indexer.pool_ape.weight", {128u, 4u}, 30u, pool_ape) &&
          gguf.tensor("blk.3.indexer.k_norm.weight", {128u}, 0u, index_norm_w) &&
          gguf.tensor("blk.3.indexer.k_norm.bias", {128u}, 0u, index_norm_b),
          "bind every real block-3 sparse-MLA tensor");

    ds4_gpu_config config = {};
    config.n_gpus = 1u;
    config.device_indices[0] = 0u;
    CHECK(ds4_gpu_init_multi(&config) &&
          ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map) &&
          ds4_gpu_set_model_map(gguf.map, gguf.size),
          "initialize gfx1151 and register model map");
    ds4_tp_test_reset_exchange_calls();
    const auto f32 = [](uint64_t n) {
        return ds4_gpu_tensor_alloc(n * sizeof(float));
    };
    ds4_gpu_tensor *d_hidden = f32((uint64_t)kRows * kHidden);
    ds4_gpu_tensor *d_q_a = f32(kQRank);
    ds4_gpu_tensor *d_q_resid = f32(kQRank);
    ds4_gpu_tensor *d_query = f32((uint64_t)kHeads * kHeadDim);
    ds4_gpu_tensor *d_kv_raw = f32((uint64_t)kRows * kKvLora);
    ds4_gpu_tensor *d_kv_norm = f32((uint64_t)kRows * kKvLora);
    ds4_gpu_tensor *d_qk_low = f32((uint64_t)kHeads * kKvLora);
    ds4_gpu_tensor *d_cache = f32((uint64_t)kRows * kKvLora);
    ds4_gpu_tensor *d_index_q = f32((uint64_t)kIndexHeads * kIndexDim);
    ds4_gpu_tensor *d_index_k_raw = f32((uint64_t)kRows * kIndexDim);
    ds4_gpu_tensor *d_index_key = f32((uint64_t)kRows * kIndexDim);
    ds4_gpu_tensor *d_pool_gate = f32((uint64_t)kRows * kIndexDim);
    ds4_gpu_tensor *d_head_weights = f32(kIndexHeads);
    ds4_gpu_tensor *d_valid = ds4_gpu_tensor_alloc(kRows * sizeof(uint32_t));
    ds4_gpu_tensor *d_pooled = f32((uint64_t)kPools * kIndexDim);
    ds4_gpu_tensor *d_pool_indices = ds4_gpu_tensor_alloc(
        (uint64_t)kPools * 4u * sizeof(int32_t));
    ds4_gpu_tensor *d_pool_valid = ds4_gpu_tensor_alloc(
        kPools * sizeof(uint32_t));
    ds4_gpu_tensor *d_pool_scores = f32(kPools);
    ds4_gpu_tensor *d_selected_pools = ds4_gpu_tensor_alloc(
        kSelectedPools * sizeof(uint32_t));
    ds4_gpu_tensor *d_selected_tokens = ds4_gpu_tensor_alloc(
        (uint64_t)kExpandedWidth * sizeof(int32_t));
    ds4_gpu_tensor *d_heads = f32((uint64_t)kHeads * kHeadDim);
    CHECK(d_hidden && d_q_a && d_q_resid && d_query && d_kv_raw &&
          d_kv_norm && d_qk_low && d_cache && d_index_q &&
          d_index_k_raw && d_index_key && d_pool_gate && d_head_weights &&
          d_valid && d_pooled && d_pool_indices && d_pool_valid &&
          d_pool_scores && d_selected_pools && d_selected_tokens && d_heads,
          "allocate bounded sparse-MLA heads composition tensors");
    ds4_gpu_tensor *d_last_hidden = ds4_gpu_tensor_view(
        d_hidden, (uint64_t)(kRows - 1u) * kHidden * sizeof(float),
        (uint64_t)kHidden * sizeof(float));
    CHECK(d_last_hidden &&
          ds4_gpu_tensor_write(d_hidden, 0u, hidden.data(),
                               (uint64_t)hidden.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(d_valid, 0u, valid.data(),
                               (uint64_t)valid.size() * sizeof(uint32_t)),
          "upload sparse-MLA heads composition inputs");

    CHECK(ds4_gpu_matmul_q8_0_tensor(d_q_a, gguf.map, gguf.size, q_a,
                                     kHidden, kQRank, d_last_hidden, 1u) &&
          ds4_gpu_rms_norm_weight_tensor(d_q_resid, d_q_a, gguf.map,
                                         gguf.size, q_norm, kQRank, 1.0e-5f) &&
          ds4_gpu_matmul_q8_0_tensor(d_query, gguf.map, gguf.size, q_b,
                                     kQRank, kHeads * kHeadDim,
                                     d_q_resid, 1u) &&
          ds4_gpu_matmul_q8_0_tensor(d_kv_raw, gguf.map, gguf.size, kv_a,
                                     kHidden, kKvLora, d_hidden, kRows) &&
          ds4_gpu_glm_kv_lora_rms_norm_tensor(
              d_kv_norm, d_kv_raw, gguf.map, gguf.size, kv_norm_w,
              kRows, kKvLora, kKvLora, 1.0e-5f) &&
          ds4_gpu_glm_store_compact_kv_tensor(
              d_cache, nullptr, d_kv_norm, d_kv_raw, 0u, kRows, kRows,
              kKvLora, kKvLora, 0u, false) &&
          ds4_gpu_glm_qk_lowrank_typed_tensor(
              d_qk_low, d_query, gguf.map, gguf.size, k_b, 8u, kHeads,
              kKvLora, kHeadDim, kHeadDim),
          "execute real Q/KV trunk and compact NoPE store");

    CHECK(ds4_gpu_matmul_bf16_tensor(
              d_index_q, gguf.map, gguf.size, index_q_w, kQRank,
              kIndexHeads * kIndexDim, d_q_resid, 1u) &&
          ds4_gpu_round_bf16_inplace_tensor(
              d_index_q, kIndexHeads * kIndexDim, 1.0f) &&
          ds4_gpu_matmul_bf16_tensor(
              d_index_k_raw, gguf.map, gguf.size, index_k_w, kHidden,
              kIndexDim, d_hidden, kRows) &&
          ds4_gpu_round_bf16_inplace_tensor(
              d_index_k_raw, (uint64_t)kRows * kIndexDim, 1.0f) &&
          ds4_gpu_glm_store_indexer_k_tensor(
              d_index_key, d_index_k_raw, gguf.map, gguf.size,
              index_norm_w, index_norm_b, 0u, kRows, kRows, kIndexDim,
              0u, 1u, 1.0e-6f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
              false) &&
          ds4_gpu_round_bf16_inplace_tensor(
              d_index_key, (uint64_t)kRows * kIndexDim, 1.0f) &&
          ds4_gpu_matmul_bf16_tensor(
              d_pool_gate, gguf.map, gguf.size, pool_gate_w, kHidden,
              kIndexDim, d_hidden, kRows) &&
          ds4_gpu_round_bf16_inplace_tensor(
              d_pool_gate, (uint64_t)kRows * kIndexDim, 1.0f) &&
          ds4_gpu_matmul_bf16_tensor(
              d_head_weights, gguf.map, gguf.size, index_weight_w,
              kHidden, kIndexHeads, d_last_hidden, 1u) &&
          ds4_gpu_round_bf16_inplace_tensor(
              d_head_weights, kIndexHeads, 1.0f) &&
          ds4_gpu_round_bf16_inplace_tensor(
              d_head_weights, kIndexHeads, 0.1767766952966369f) &&
          ds4_gpu_glm5_kpool_tensor(
              d_pooled, d_pool_indices, d_pool_valid, d_index_key,
              d_pool_gate, d_valid, gguf.map, gguf.size, pool_ape,
              kRows, kIndexDim, 4u, kFirstValid) &&
          ds4_gpu_glm_indexer_score_one_tensor(
              d_pool_scores, d_index_q, d_head_weights, d_pooled, kPools,
              kIndexHeads, kIndexDim, 0.08838834764831845f, false) &&
          ds4_gpu_glm5_mask_pool_scores_tensor(
              d_pool_scores, d_pool_valid, kPools) &&
          ds4_gpu_indexer_topk_tensor(
              d_selected_pools, d_pool_scores, kPools, 1u,
              kSelectedPools) &&
          ds4_gpu_glm5_expand_pool_selection_tensor(
              d_selected_tokens, d_selected_pools, d_pool_indices,
              d_pool_valid, d_valid, kPools, kSelectedPools, kRows,
              kFirstValid, kRows - kFirstValid, kTokenBudget, 4u),
          "execute coupled BF16 indexer and compact selection");

    CHECK(ds4_gpu_glm_attention_indexed_decode_typed_tensor(
              d_heads, d_query, d_qk_low, d_cache, nullptr,
              gguf.map, gguf.size, v_b, 8u, d_selected_tokens,
              kSelectedTokens, kRows, false, kHeads, kKvLora, kHeadDim,
              0u, kHeadDim, 0u, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f) &&
          ds4_gpu_synchronize(),
          "execute selected zero-RoPE attention and real value projection");

    std::vector<float> got_q_resid, got_query, got_kv_norm, got_qk_low,
        got_index_q, got_index_key, got_pool_gate, got_head_weights,
        got_pooled, got_pool_scores, got_heads;
    std::vector<uint32_t> got_pool_valid, got_selected_pools;
    std::vector<int32_t> got_pool_indices, got_expanded_tokens;
    CHECK(read_tensor(d_q_resid, kQRank, got_q_resid) &&
          read_tensor(d_query, kHeads * kHeadDim, got_query) &&
          read_tensor(d_kv_norm, kRows * kKvLora, got_kv_norm) &&
          read_tensor(d_qk_low, kHeads * kKvLora, got_qk_low) &&
          read_tensor(d_index_q, kIndexHeads * kIndexDim, got_index_q) &&
          read_tensor(d_index_key, kRows * kIndexDim, got_index_key) &&
          read_tensor(d_pool_gate, kRows * kIndexDim, got_pool_gate) &&
          read_tensor(d_head_weights, kIndexHeads, got_head_weights) &&
          read_tensor(d_pooled, kPools * kIndexDim, got_pooled) &&
          read_tensor(d_pool_scores, kPools, got_pool_scores) &&
          read_tensor(d_pool_indices, kPools * 4u, got_pool_indices) &&
          read_tensor(d_pool_valid, kPools, got_pool_valid) &&
          read_tensor(d_selected_pools, kSelectedPools, got_selected_pools) &&
          read_tensor(d_selected_tokens, kExpandedWidth, got_expanded_tokens) &&
          read_tensor(d_heads, kHeads * kHeadDim, got_heads),
          "read every sparse-MLA heads composition boundary");

    CHECK(compare_values("q_resid", got_q_resid, expected_q_resid,
                         5.0e-6, 2.0e-12) &&
          compare_values("query", got_query, expected_query,
                         1.0e-5, 2.0e-12) &&
          compare_values("kv_norm", got_kv_norm, expected_kv_norm,
                         4.0e-6, 2.0e-12) &&
          compare_values("qk_low", got_qk_low, expected_qk_low,
                         8.0e-6, 2.0e-12) &&
          compare_values("index_q", got_index_q, expected_index_q,
                         0.015625, 5.0e-10) &&
          compare_values("index_key", got_index_key, expected_index_key,
                         0.000244140625, 1.0e-8) &&
          compare_values("pool_gate", got_pool_gate, expected_pool_gate,
                         0.0000152587890625, 1.0e-8) &&
          compare_values("head_weights", got_head_weights,
                         expected_head_weights, 1.0e-5, 1.0e-10) &&
          compare_values("pooled", got_pooled, expected_pooled,
                         0.03125, 1.0e-8),
          "sparse-MLA heads intermediate numerical gates");
    CHECK(got_pool_indices == expected_pool_indices &&
          got_pool_valid == expected_pool_valid &&
          got_selected_pools == expected_selected_pools,
          "exact pool structure and top-k order");
    CHECK(std::equal(expected_selected_tokens.begin(),
                     expected_selected_tokens.end(),
                     got_expanded_tokens.begin()),
          "exact compact selected-row prefix");
    for (uint32_t i = kSelectedTokens; i < kExpandedWidth; ++i) {
        CHECK(got_expanded_tokens[i] == -1,
              "fixed-width selection suffix is fail-closed padding");
    }
    CHECK(std::isfinite(got_pool_scores[0]) &&
          std::isfinite(got_pool_scores[1]) &&
          got_pool_scores[2] == -std::numeric_limits<float>::max(),
          "invalid tail pool remains finite-min masked");
    got_pool_scores.resize(2u);
    expected_pool_scores.resize(2u);
    CHECK(compare_values("pool_scores", got_pool_scores,
                         expected_pool_scores, 3.0e-5, 1.0e-12) &&
          compare_values("heads", got_heads, expected_heads,
                         1.0e-6, 5.0e-13),
          "selected score and final attention output gates");
    CHECK(ds4_tp_test_get_exchange_calls() == 0u,
          "rank-local sparse-MLA heads invokes no TP exchange");

    ds4_gpu_tensor_free(d_last_hidden);
    ds4_gpu_tensor_free(d_heads);
    ds4_gpu_tensor_free(d_selected_tokens);
    ds4_gpu_tensor_free(d_selected_pools);
    ds4_gpu_tensor_free(d_pool_scores);
    ds4_gpu_tensor_free(d_pool_valid);
    ds4_gpu_tensor_free(d_pool_indices);
    ds4_gpu_tensor_free(d_pooled);
    ds4_gpu_tensor_free(d_valid);
    ds4_gpu_tensor_free(d_head_weights);
    ds4_gpu_tensor_free(d_pool_gate);
    ds4_gpu_tensor_free(d_index_key);
    ds4_gpu_tensor_free(d_index_k_raw);
    ds4_gpu_tensor_free(d_index_q);
    ds4_gpu_tensor_free(d_cache);
    ds4_gpu_tensor_free(d_qk_low);
    ds4_gpu_tensor_free(d_kv_norm);
    ds4_gpu_tensor_free(d_kv_raw);
    ds4_gpu_tensor_free(d_query);
    ds4_gpu_tensor_free(d_q_resid);
    ds4_gpu_tensor_free(d_q_a);
    ds4_gpu_tensor_free(d_hidden);
    ds4_gpu_cleanup();
    std::fprintf(stderr,
                 "PASS same-GGUF GLM5 block-3 sparse-MLA heads gate\n");
    return true;
}

}  // namespace

int main() { return run_test() ? 0 : 1; }
