#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
extern "C" {
#include "ds4_tp.h"
}

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#define CHECK(expr, message) do {                                           \
    if (!(expr)) {                                                          \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__);    \
        return false;                                                       \
    }                                                                       \
} while (0)

namespace {

constexpr uint32_t kPool = 4u;
constexpr uint32_t kTokenBudget = 2048u;
constexpr uint32_t kOutputWidth = kTokenBudget + kPool - 1u;

struct Case {
    uint32_t n_rows;
    uint32_t first_valid;
    uint32_t invalid_pool;
};

float score_for_pool(uint32_t pool) {
    const uint32_t mixed = pool * 2654435761u + 2246822519u;
    return (float)(mixed & 0x00ffffffu) / 16777216.0f;
}

bool run_case(const Case &test_case) {
    const uint32_t n_rows = test_case.n_rows;
    const uint32_t first = test_case.first_valid;
    const uint32_t visible = n_rows - first;
    const uint32_t n_pools = (n_rows + kPool - 1u) / kPool;
    std::vector<uint32_t> valid_keys(n_rows, 1u);
    for (uint32_t row = 0; row < first; ++row) valid_keys[row] = 0u;
    std::vector<uint32_t> pool_valid(n_pools, 0u);
    std::vector<int32_t> pool_indices((size_t)n_pools * kPool, -1);
    uint32_t complete_count = 0u;
    for (uint32_t pool = 0; pool < n_pools; ++pool) {
        const uint32_t start = first + pool * kPool;
        bool complete = start <= n_rows && n_rows - start >= kPool;
        if (pool == test_case.invalid_pool) complete = false;
        pool_valid[pool] = complete ? 1u : 0u;
        if (complete) ++complete_count;
        for (uint32_t member = 0; member < kPool; ++member) {
            const uint32_t row = start + member;
            pool_indices[(uint64_t)pool * kPool + member] =
                row < n_rows ? (int32_t)row : -1;
        }
    }
    const uint32_t select_count = std::min(kTokenBudget / kPool,
                                            complete_count);
    std::vector<float> scores(n_pools);
    for (uint32_t pool = 0; pool < n_pools; ++pool)
        scores[pool] = score_for_pool(pool);
    if (test_case.invalid_pool < n_pools)
        scores[test_case.invalid_pool] = 1000.0f;

    std::vector<uint32_t> expected_pools;
    expected_pools.reserve(complete_count);
    for (uint32_t pool = 0; pool < n_pools; ++pool)
        if (pool_valid[pool]) expected_pools.push_back(pool);
    std::sort(expected_pools.begin(), expected_pools.end(),
              [&](uint32_t a, uint32_t b) {
                  if (scores[a] != scores[b]) return scores[a] > scores[b];
                  return a < b;
              });
    expected_pools.resize(select_count);
    std::sort(expected_pools.begin(), expected_pools.end());

    ds4_gpu_tensor *d_scores = ds4_gpu_tensor_alloc((uint64_t)n_pools * sizeof(float));
    ds4_gpu_tensor *d_pool_valid = ds4_gpu_tensor_alloc((uint64_t)n_pools * sizeof(uint32_t));
    ds4_gpu_tensor *d_pool_indices = ds4_gpu_tensor_alloc((uint64_t)n_pools * kPool * sizeof(int32_t));
    ds4_gpu_tensor *d_valid_keys = ds4_gpu_tensor_alloc((uint64_t)n_rows * sizeof(uint32_t));
    ds4_gpu_tensor *d_selected_pools = ds4_gpu_tensor_alloc(
        (uint64_t)std::max(select_count, 1u) * sizeof(uint32_t));
    ds4_gpu_tensor *d_selected_tokens = ds4_gpu_tensor_alloc((uint64_t)kOutputWidth * sizeof(int32_t));
    CHECK(d_scores && d_pool_valid && d_pool_indices && d_valid_keys &&
          d_selected_pools && d_selected_tokens &&
          ds4_gpu_tensor_write(d_scores, 0u, scores.data(),
                               (uint64_t)n_pools * sizeof(float)) &&
          ds4_gpu_tensor_write(d_pool_valid, 0u, pool_valid.data(),
                               (uint64_t)n_pools * sizeof(uint32_t)) &&
          ds4_gpu_tensor_write(d_pool_indices, 0u, pool_indices.data(),
                               (uint64_t)n_pools * kPool * sizeof(int32_t)) &&
          ds4_gpu_tensor_write(d_valid_keys, 0u, valid_keys.data(),
                               (uint64_t)n_rows * sizeof(uint32_t)) &&
          ds4_gpu_glm5_mask_pool_scores_tensor(
              d_scores, d_pool_valid, n_pools) &&
          (select_count == 0u || ds4_gpu_indexer_topk_tensor(
              d_selected_pools, d_scores, n_pools, 1u, select_count)) &&
          ds4_gpu_glm5_expand_pool_selection_tensor(
              d_selected_tokens, d_selected_pools, d_pool_indices,
              d_pool_valid, d_valid_keys, n_pools, select_count, n_rows,
              first, visible, kTokenBudget, kPool) &&
          ds4_gpu_synchronize(),
          "mask, top-k and expand GLM5 pool selection");

    std::vector<float> masked_scores(n_pools);
    std::vector<uint32_t> got_pools(select_count);
    std::vector<int32_t> got_tokens(kOutputWidth);
    CHECK(ds4_gpu_tensor_read(d_scores, 0u, masked_scores.data(),
                              (uint64_t)n_pools * sizeof(float)) &&
          (select_count == 0u || ds4_gpu_tensor_read(
              d_selected_pools, 0u, got_pools.data(),
              (uint64_t)select_count * sizeof(uint32_t))) &&
          ds4_gpu_tensor_read(d_selected_tokens, 0u, got_tokens.data(),
                              (uint64_t)kOutputWidth * sizeof(int32_t)),
          "read GLM5 raw token selection");
    for (uint32_t pool = 0; pool < n_pools; ++pool) {
        if (pool_valid[pool]) CHECK(masked_scores[pool] == scores[pool],
                                    "valid GLM5 pool score preserved");
        else CHECK(std::isfinite(masked_scores[pool]) &&
                   masked_scores[pool] == -FLT_MAX,
                   "invalid GLM5 pool score uses finite-min sentinel");
    }
    std::vector<uint32_t> sorted_got_pools = got_pools;
    std::sort(sorted_got_pools.begin(), sorted_got_pools.end());
    CHECK(sorted_got_pools == expected_pools,
          "exact GLM5 top-k pool set");
    for (uint32_t slot = 0; slot < select_count; ++slot) {
        const uint32_t pool = got_pools[slot];
        CHECK(pool < n_pools && pool_valid[pool],
              "expanded GLM5 pool is valid");
        for (uint32_t member = 0; member < kPool; ++member)
            CHECK(got_tokens[(uint64_t)slot * kPool + member] ==
                  pool_indices[(uint64_t)pool * kPool + member],
                  "selected GLM5 pool expands in raw-token order");
    }
    const uint32_t expanded = select_count * kPool;
    std::vector<int32_t> independently_visible;
    for (uint32_t row = 0; row < n_rows; ++row)
        if (valid_keys[row]) independently_visible.push_back((int32_t)row);
    const uint32_t complete_coverage =
        (uint32_t)(independently_visible.size() / kPool) * kPool;
    const uint32_t tail_count =
        (uint32_t)independently_visible.size() - complete_coverage;
    for (uint32_t member = 0; member < 3u; ++member) {
        const int32_t expected = member < tail_count
            ? independently_visible[complete_coverage + member] : -1;
        CHECK(got_tokens[expanded + member] == expected,
              "GLM5 visible tail appended after selected pools");
    }
    for (uint32_t i = expanded + 3u; i < kOutputWidth; ++i)
        CHECK(got_tokens[i] == -1, "GLM5 selection padding is fail-closed");
    std::fprintf(stderr,
        "GLM5 select rows=%u first=%u pools=%u complete=%u selected=%u tail=%u\n",
        n_rows, first, n_pools, complete_count, select_count, tail_count);

    ds4_gpu_tensor_free(d_selected_tokens);
    ds4_gpu_tensor_free(d_selected_pools);
    ds4_gpu_tensor_free(d_valid_keys);
    ds4_gpu_tensor_free(d_pool_indices);
    ds4_gpu_tensor_free(d_pool_valid);
    ds4_gpu_tensor_free(d_scores);
    return true;
}

bool run_test() {
    ds4_gpu_config config = {};
    config.n_gpus = 1u;
    config.device_indices[0] = 0u;
    CHECK(ds4_gpu_init_multi(&config), "initialize gfx1151");
    ds4_tp_test_reset_exchange_calls();
    CHECK(run_case({1u, 0u, UINT32_MAX}) &&
          run_case({3u, 0u, UINT32_MAX}),
          "all-tail GLM5 selection without a complete pool");
    CHECK(run_case({3u, 3u, UINT32_MAX}),
          "all-padding GLM5 selection remains fail-closed");
    CHECK(run_case({5u, 0u, UINT32_MAX}),
          "short GLM5 selection with one-token tail");
    CHECK(run_case({9u, 1u, UINT32_MAX}) &&
          run_case({19u, 2u, UINT32_MAX}),
          "left-padded GLM5 pool and tail alignment");
    CHECK(run_case({8195u, 0u, UINT32_MAX}),
          "long GLM5 selection with three-token tail");
    CHECK(run_case({8192u, 0u, 5u}),
          "GLM5 selection masks high-scoring invalid pool");
    CHECK(run_case({32771u, 0u, 7u}),
          "GLM5 selection masks an invalid pool on the large top-k path");
    CHECK(ds4_tp_test_get_exchange_calls() == 0u,
          "GLM5 indexer selection invokes no TP exchange API");
    ds4_gpu_cleanup();
    std::fprintf(stderr,
                 "PASS GLM5 pool score/top-k/raw-tail selection gate\n");
    return true;
}

}  // namespace

int main() { return run_test() ? 0 : 1; }
