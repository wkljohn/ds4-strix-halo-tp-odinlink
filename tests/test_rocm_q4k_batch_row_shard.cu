/* Exact production-kernel oracle for cache-free Q4_K routed-expert row
 * sharding.  The full-row and two-half paths both launch DS4's shipping
 * sorted-pair WMMA kernels and cold DP4A complements.  Only indexing, compact
 * strides, and the accepted TP ownership-group fold differ. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

enum {
    QK_K = 256,
    Q4_K_BLOCK_BYTES = 144,
    Q8_K_BLOCK_BYTES = 292,
    N_TOTAL_EXPERT = 12,
    N_USED = 6,
    EXPERT_SPLIT = 6,
    IN_DIM = 4096,
    MID_DIM = 2048,
    OUT_DIM = 4096,
};

#define FAIL(...) do { fprintf(stderr, "FAIL: " __VA_ARGS__); return 0; } while (0)
#define CHECK(cond, msg) do {                                                \
    if (!(cond)) {                                                           \
        fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);           \
        return 1;                                                            \
    }                                                                        \
} while (0)

static void pack_q4k_block(unsigned char *dst, uint32_t seed) {
    dst[0] = 0x00; dst[1] = 0x28; /* d = 1/32 */
    dst[2] = 0x00; dst[3] = 0x00; /* dmin = 0 */
    for (uint32_t i = 0; i < 4; i++) dst[4 + i] = 1;
    for (uint32_t i = 4; i < 8; i++) dst[4 + i] = 0;
    for (uint32_t i = 8; i < 12; i++) dst[4 + i] = 1;
    for (uint32_t i = 0; i < 128; i++) {
        const uint8_t lo = (uint8_t)((seed + 3u * i + 1u) & 15u);
        const uint8_t hi = (uint8_t)((seed + 5u * i + 7u) & 15u);
        dst[16 + i] = (uint8_t)(lo | (hi << 4));
    }
}

static void pack_q4k_table(unsigned char *dst, uint32_t experts,
                           uint32_t rows, uint32_t blocks_per_row,
                           uint32_t salt) {
    for (uint32_t expert = 0; expert < experts; expert++) {
        for (uint32_t row = 0; row < rows; row++) {
            for (uint32_t block = 0; block < blocks_per_row; block++) {
                const uint64_t i = ((uint64_t)expert * rows + row) *
                                   blocks_per_row + block;
                pack_q4k_block(dst + i * Q4_K_BLOCK_BYTES,
                               salt + 17u * expert + 13u * row + 7u * block);
            }
        }
    }
}

static int alloc_tensor(ds4_gpu_tensor *t, uint64_t bytes) {
    memset(t, 0, sizeof(*t));
    return ds4_gpu_tensor_alloc_on(t, 0, bytes) == 0;
}

static int upload(ds4_gpu_tensor *t, const void *src, uint64_t bytes) {
    return ds4_gpu_tensor_write(t, 0, src, bytes) != 0;
}

static uint64_t fnv1a64(const void *data, size_t bytes) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < bytes; i++) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static float add_exact_host(float a, float b) {
    volatile float av = a;
    volatile float bv = b;
    volatile float sum = av + bv;
    return sum;
}

static void fill_routes(int32_t *selected, float *weights,
                        uint32_t n_tokens) {
    for (uint32_t token = 0; token < n_tokens; token++) {
        int32_t *route = selected + (uint64_t)token * N_USED;
        float *weight = weights + (uint64_t)token * N_USED;
        if (token == 0u) {
            /* All six on rank 0 proves the empty-rank1 +0.0 fold. */
            const int32_t first[N_USED] = {0, 1, 2, 3, 4, 5};
            memcpy(route, first, sizeof(first));
        } else if (token == 1u) {
            /* The opposite ownership extreme, without consuming expert 10. */
            const int32_t second[N_USED] = {6, 7, 8, 9, 6, 7};
            memcpy(route, second, sizeof(second));
        } else {
            route[0] = 0;
            route[1] = 1 + (int32_t)(token % 4u);
            route[2] = 6 + (int32_t)(token % 3u);
            route[3] = 8 + (int32_t)(token % 2u);
            route[4] = 0;
            route[5] = token == 2u ? 10 : 2 + (int32_t)(token % 3u);
        }
        for (uint32_t slot = 0; slot < N_USED; slot++) {
            weight[slot] = (float)(7u + ((token * 11u + slot * 5u) % 23u)) /
                           128.0f;
        }
        if (token == 0u) weight[5] = 0.0f;
    }
}

static int compare_compact_half(const char *stage, uint32_t n_tokens,
                                const float *full, const float *compact,
                                uint32_t pairs, uint32_t full_rows,
                                uint32_t row_base, uint32_t row_count) {
    for (uint32_t pair = 0; pair < pairs; pair++) {
        const float *expected = full + (uint64_t)pair * full_rows + row_base;
        const float *got = compact + (uint64_t)pair * row_count;
        if (memcmp(expected, got, (size_t)row_count * sizeof(float)) != 0) {
            uint32_t row = 0;
            while (row < row_count &&
                   memcmp(expected + row, got + row, sizeof(float)) == 0) row++;
            fprintf(stderr,
                    "FAIL: batch row shard stage=%s tokens=%u pair=%u row=%u "
                    "ref=%a got=%a\n", stage, n_tokens, pair,
                    row_base + row, expected[row], got[row]);
            return 0;
        }
    }
    return 1;
}

static int run_batch_case(
        uint32_t n_tokens, const void *model, uint64_t model_bytes,
        uint64_t gate_off, uint64_t up_off, uint64_t down_off,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes) {
    const uint32_t pairs = n_tokens * N_USED;
    const uint32_t mid_half = MID_DIM / 2u;
    const uint32_t out_half = OUT_DIM / 2u;
    const uint64_t route_bytes = (uint64_t)pairs * sizeof(int32_t);
    const uint64_t weight_bytes = (uint64_t)pairs * sizeof(float);
    const uint64_t x_bytes = (uint64_t)n_tokens * IN_DIM * sizeof(float);
    const uint64_t mid_full_bytes = (uint64_t)pairs * MID_DIM * sizeof(float);
    const uint64_t mid_half_bytes = (uint64_t)pairs * mid_half * sizeof(float);
    const uint64_t down_full_bytes = (uint64_t)pairs * OUT_DIM * sizeof(float);
    const uint64_t down_half_bytes = (uint64_t)pairs * out_half * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tokens * OUT_DIM * sizeof(float);
    const uint64_t xq_bytes = (uint64_t)n_tokens * (IN_DIM / QK_K) *
                              Q8_K_BLOCK_BYTES;
    const uint64_t midq_bytes = (uint64_t)pairs * (MID_DIM / QK_K) *
                                Q8_K_BLOCK_BYTES;

    int32_t *hselected = (int32_t *)malloc((size_t)route_bytes);
    float *hweights = (float *)malloc((size_t)weight_bytes);
    float *hx = (float *)malloc((size_t)x_bytes);
    float *hadd0 = (float *)malloc((size_t)out_bytes);
    float *hadd1 = (float *)malloc((size_t)out_bytes);
    float *href_mid = (float *)malloc((size_t)mid_full_bytes);
    float *hhalf = (float *)malloc((size_t)(down_half_bytes > mid_half_bytes ?
                                            down_half_bytes : mid_half_bytes));
    float *href_down = (float *)malloc((size_t)down_full_bytes);
    float *hgot = (float *)malloc((size_t)out_bytes);
    float *hexpected = (float *)malloc((size_t)out_bytes);
    if (!hselected || !hweights || !hx || !hadd0 || !hadd1 || !href_mid ||
        !hhalf || !href_down || !hgot || !hexpected) {
        FAIL("host allocation for %u-token oracle\n", n_tokens);
    }
    fill_routes(hselected, hweights, n_tokens);
    for (uint64_t i = 0; i < (uint64_t)n_tokens * IN_DIM; i++) {
        const uint32_t token = (uint32_t)(i / IN_DIM);
        const uint32_t col = (uint32_t)(i - (uint64_t)token * IN_DIM);
        hx[i] = (float)((int)((col * 7u + token * 13u) % 61u) - 30) /
                512.0f;
    }
    for (uint64_t i = 0; i < (uint64_t)n_tokens * OUT_DIM; i++) {
        const uint32_t token = (uint32_t)(i / OUT_DIM);
        const uint32_t row = (uint32_t)(i - (uint64_t)token * OUT_DIM);
        /* Accepted prefill splits shared-expert work by token rows: exactly
         * one rank folds a shared value into its routed partial before the
         * canonical rank0+rank1 all-reduce. */
        const int rank0_owns_token = token < (n_tokens + 1u) / 2u;
        hadd0[i] = rank0_owns_token ?
            (float)((int)((row + 3u * token) % 29u) - 14) / 2048.0f : 0.0f;
        hadd1[i] = rank0_owns_token ? 0.0f :
            (float)((int)((5u * row + token) % 31u) - 15) / 4096.0f;
    }

    ds4_gpu_tensor selected = {}, weights = {}, x = {}, xq = {};
    ds4_gpu_tensor gate_full = {}, up_full = {}, mid_full = {};
    ds4_gpu_tensor gate_half = {}, up_half = {}, mid_compact = {};
    ds4_gpu_tensor down_full = {}, down_compact = {}, midq = {};
    ds4_gpu_tensor out = {}, add0 = {}, add1 = {};
    if (!alloc_tensor(&selected, route_bytes) ||
        !alloc_tensor(&weights, weight_bytes) ||
        !alloc_tensor(&x, x_bytes) || !alloc_tensor(&xq, xq_bytes) ||
        !alloc_tensor(&gate_full, mid_full_bytes) ||
        !alloc_tensor(&up_full, mid_full_bytes) ||
        !alloc_tensor(&mid_full, mid_full_bytes) ||
        !alloc_tensor(&gate_half, mid_half_bytes) ||
        !alloc_tensor(&up_half, mid_half_bytes) ||
        !alloc_tensor(&mid_compact, mid_half_bytes) ||
        !alloc_tensor(&down_full, down_full_bytes) ||
        !alloc_tensor(&down_compact, down_half_bytes) ||
        !alloc_tensor(&midq, midq_bytes) || !alloc_tensor(&out, out_bytes) ||
        !alloc_tensor(&add0, out_bytes) || !alloc_tensor(&add1, out_bytes)) {
        FAIL("device allocation for %u-token oracle\n", n_tokens);
    }
    if (!upload(&selected, hselected, route_bytes) ||
        !upload(&weights, hweights, weight_bytes) || !upload(&x, hx, x_bytes) ||
        !upload(&add0, hadd0, out_bytes) || !upload(&add1, hadd1, out_bytes)) {
        FAIL("input upload for %u-token oracle\n", n_tokens);
    }

    if (!ds4_gpu_rocm_q4k_batch_row_shard_gate_up_tensor(
            &gate_full, &up_full, &mid_full, &xq, model, model_bytes,
            gate_off, up_off, gate_expert_bytes, gate_row_bytes,
            IN_DIM, MID_DIM, &selected, &weights, N_TOTAL_EXPERT, N_USED,
            n_tokens, 0u, MID_DIM, 0.0f, &x) ||
        !ds4_gpu_tensor_read(&mid_full, 0, href_mid, mid_full_bytes)) {
        FAIL("full gate/up reference for %u tokens\n", n_tokens);
    }
    for (uint32_t half = 0; half < 2u; half++) {
        const uint32_t row_base = half * mid_half;
        if (!ds4_gpu_rocm_q4k_batch_row_shard_gate_up_tensor(
                &gate_half, &up_half, &mid_compact, &xq, model, model_bytes,
                gate_off, up_off, gate_expert_bytes, gate_row_bytes,
                IN_DIM, MID_DIM, &selected, &weights, N_TOTAL_EXPERT, N_USED,
                n_tokens, row_base, mid_half, 0.0f, &x) ||
            !ds4_gpu_tensor_read(&mid_compact, 0, hhalf, mid_half_bytes) ||
            !compare_compact_half(half == 0u ? "gate-low" : "gate-high",
                                  n_tokens, href_mid, hhalf, pairs, MID_DIM,
                                  row_base, mid_half)) {
            FAIL("compact gate/up half %u for %u tokens\n", half, n_tokens);
        }
    }

    if (!ds4_gpu_rocm_q4k_batch_row_shard_down_tensor(
            &down_full, &midq, &mid_full, model, model_bytes, down_off,
            down_expert_bytes, down_row_bytes, MID_DIM, OUT_DIM, &selected,
            N_TOTAL_EXPERT, N_USED, n_tokens, 0u, OUT_DIM) ||
        !ds4_gpu_tensor_read(&down_full, 0, href_down, down_full_bytes)) {
        FAIL("full down reference for %u tokens\n", n_tokens);
    }
    for (uint32_t half = 0; half < 2u; half++) {
        const uint32_t row_base = half * out_half;
        if (!ds4_gpu_rocm_q4k_batch_row_shard_down_tensor(
                &down_compact, &midq, &mid_full, model, model_bytes, down_off,
                down_expert_bytes, down_row_bytes, MID_DIM, OUT_DIM, &selected,
                N_TOTAL_EXPERT, N_USED, n_tokens, row_base, out_half) ||
            !ds4_gpu_tensor_read(&down_compact, 0, hhalf, down_half_bytes) ||
            !compare_compact_half(half == 0u ? "down-low" : "down-high",
                                  n_tokens, href_down, hhalf, pairs, OUT_DIM,
                                  row_base, out_half) ||
            !ds4_gpu_rocm_q4k_batch_row_shard_reduce_tensor(
                &out, &down_compact, &add0, &add1, &selected, OUT_DIM,
                N_USED, n_tokens, EXPERT_SPLIT, row_base, out_half)) {
            FAIL("compact down/reduce half %u for %u tokens\n", half, n_tokens);
        }
    }
    if (hipDeviceSynchronize() != hipSuccess ||
        !ds4_gpu_tensor_read(&out, 0, hgot, out_bytes)) {
        FAIL("output read for %u tokens\n", n_tokens);
    }
    for (uint32_t token = 0; token < n_tokens; token++) {
        for (uint32_t row = 0; row < OUT_DIM; row++) {
            float group0 = 0.0f, group1 = 0.0f;
            for (uint32_t slot = 0; slot < N_USED; slot++) {
                const uint32_t pair = token * N_USED + slot;
                const float value = href_down[(uint64_t)pair * OUT_DIM + row];
                if ((uint32_t)hselected[pair] < EXPERT_SPLIT)
                    group0 = add_exact_host(group0, value);
                else
                    group1 = add_exact_host(group1, value);
            }
            const uint64_t off = (uint64_t)token * OUT_DIM + row;
            group0 = add_exact_host(group0, hadd0[off]);
            group1 = add_exact_host(group1, hadd1[off]);
            hexpected[off] = add_exact_host(group0, group1);
        }
    }
    if (memcmp(hexpected, hgot, (size_t)out_bytes) != 0) {
        uint64_t first = 0;
        const uint64_t values = (uint64_t)n_tokens * OUT_DIM;
        while (first < values &&
               memcmp(hexpected + first, hgot + first, sizeof(float)) == 0)
            first++;
        FAIL("batch reduce tokens=%u token=%llu row=%llu ref=%a got=%a\n",
             n_tokens, (unsigned long long)(first / OUT_DIM),
             (unsigned long long)(first % OUT_DIM),
             first < values ? hexpected[first] : 0.0f,
             first < values ? hgot[first] : 0.0f);
    }

    printf("batch_row_shard_oracle tokens=%u split=%u fnv64=%016llx "
           "exact=1 gate_wmma=1 down_wmma=1 cold=1\n",
           n_tokens, EXPERT_SPLIT,
           (unsigned long long)fnv1a64(hgot, (size_t)out_bytes));

    ds4_gpu_tensor_free_in_place(&add1);
    ds4_gpu_tensor_free_in_place(&add0);
    ds4_gpu_tensor_free_in_place(&out);
    ds4_gpu_tensor_free_in_place(&midq);
    ds4_gpu_tensor_free_in_place(&down_compact);
    ds4_gpu_tensor_free_in_place(&down_full);
    ds4_gpu_tensor_free_in_place(&mid_compact);
    ds4_gpu_tensor_free_in_place(&up_half);
    ds4_gpu_tensor_free_in_place(&gate_half);
    ds4_gpu_tensor_free_in_place(&mid_full);
    ds4_gpu_tensor_free_in_place(&up_full);
    ds4_gpu_tensor_free_in_place(&gate_full);
    ds4_gpu_tensor_free_in_place(&xq);
    ds4_gpu_tensor_free_in_place(&x);
    ds4_gpu_tensor_free_in_place(&weights);
    ds4_gpu_tensor_free_in_place(&selected);
    free(hexpected); free(hgot); free(href_down); free(hhalf); free(href_mid);
    free(hadd1); free(hadd0); free(hx); free(hweights); free(hselected);
    return 1;
}

int main(void) {
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(setenv("DS4_ROCM_Q4K_WMMA_MIN_COUNT", "6", 1) == 0,
          "pin production gate/up WMMA crossover");
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm backend");

    const uint64_t in_blocks = IN_DIM / QK_K;
    const uint64_t mid_blocks = MID_DIM / QK_K;
    const uint64_t gate_row_bytes = in_blocks * Q4_K_BLOCK_BYTES;
    const uint64_t down_row_bytes = mid_blocks * Q4_K_BLOCK_BYTES;
    const uint64_t gate_expert_bytes = MID_DIM * gate_row_bytes;
    const uint64_t down_expert_bytes = OUT_DIM * down_row_bytes;
    const uint64_t gate_off = 0;
    const uint64_t up_off = N_TOTAL_EXPERT * gate_expert_bytes;
    const uint64_t down_off = up_off + N_TOTAL_EXPERT * gate_expert_bytes;
    const uint64_t model_bytes = down_off + N_TOTAL_EXPERT * down_expert_bytes;
    unsigned char *model = (unsigned char *)malloc((size_t)model_bytes);
    CHECK(model, "allocate synthetic batch model");
    pack_q4k_table(model + gate_off, N_TOTAL_EXPERT, MID_DIM,
                   (uint32_t)in_blocks, 11u);
    pack_q4k_table(model + up_off, N_TOTAL_EXPERT, MID_DIM,
                   (uint32_t)in_blocks, 37u);
    pack_q4k_table(model + down_off, N_TOTAL_EXPERT, OUT_DIM,
                   (uint32_t)mid_blocks, 73u);
    CHECK(ds4_gpu_set_model_map(model, model_bytes), "install batch model map");
    CHECK(EXPERT_SPLIT * 2u == N_TOTAL_EXPERT,
          "ordinary TP split must be balanced");

    const uint32_t token_cases[] = {6u, 129u, 2048u};
    for (uint32_t i = 0; i < sizeof(token_cases) / sizeof(token_cases[0]); i++) {
        CHECK(run_batch_case(token_cases[i], model, model_bytes,
                             gate_off, up_off, down_off,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes),
              "exact production-kernel batch row-shard oracle");
    }
    free(model);
    ds4_gpu_cleanup();
    return 0;
}
