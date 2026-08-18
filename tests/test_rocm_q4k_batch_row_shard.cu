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

static void pack_q4k_row_span(unsigned char *dst, const unsigned char *src,
                              uint32_t experts, uint32_t full_rows,
                              uint32_t row_base, uint32_t row_count,
                              uint64_t row_bytes) {
    const uint64_t full_expert_bytes = (uint64_t)full_rows * row_bytes;
    const uint64_t packed_expert_bytes = (uint64_t)row_count * row_bytes;
    for (uint32_t expert = 0; expert < experts; expert++) {
        memcpy(dst + (uint64_t)expert * packed_expert_bytes,
               src + (uint64_t)expert * full_expert_bytes +
                   (uint64_t)row_base * row_bytes,
               (size_t)packed_expert_bytes);
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

#if 0 /* Dedicated packed-row registry superseded by packed-slice tests. */
static int validate_packed_registry(
        const unsigned char *model, uint64_t model_bytes,
        uint64_t gate_off, uint64_t up_off, uint64_t down_off,
        uint64_t gate_row_bytes, uint64_t down_row_bytes) {
    const uint64_t offsets[3] = {gate_off, up_off, down_off};
    const uint32_t full_rows[3] = {MID_DIM, MID_DIM, OUT_DIM};
    const uint64_t row_bytes[3] = {
        gate_row_bytes, gate_row_bytes, down_row_bytes};
    const ds4_gpu_q4k_packed_kind kinds[3] = {
        DS4_GPU_Q4K_PACKED_GATE_UP,
        DS4_GPU_Q4K_PACKED_GATE_UP,
        DS4_GPU_Q4K_PACKED_DOWN};

    /* Capture the exact full-row one-token reference before declarations
     * deliberately make the original routed tensors unavailable to generic
     * linear resolution. */
    const uint64_t mid_full_bytes = (uint64_t)N_USED * MID_DIM * sizeof(float);
    const uint64_t out_full_bytes = (uint64_t)OUT_DIM * sizeof(float);
    const uint64_t xq_bytes = (uint64_t)(IN_DIM / QK_K) * Q8_K_BLOCK_BYTES;
    const uint64_t midq_full_bytes =
        (uint64_t)N_USED * (MID_DIM / QK_K) * Q8_K_BLOCK_BYTES;
    const uint64_t batch_down_full_bytes =
        (uint64_t)N_USED * OUT_DIM * sizeof(float);
    int32_t selected_host[N_USED] = {0, 7, 2, 9, 4, 11};
    float weights_host[N_USED] = {0.125f, 0.09375f, 0.15625f,
                                  0.0625f, 0.109375f, 0.140625f};
    float *x_host = (float *)malloc((size_t)IN_DIM * sizeof(float));
    float *mid_expected = (float *)malloc((size_t)mid_full_bytes);
    float *out_expected = (float *)malloc((size_t)out_full_bytes);
    float *batch_mid_expected = (float *)malloc((size_t)mid_full_bytes);
    float *batch_out_expected = (float *)malloc((size_t)out_full_bytes);
    const uint64_t compact_host_bytes =
        mid_full_bytes / 2u > out_full_bytes / 2u
            ? mid_full_bytes / 2u : out_full_bytes / 2u;
    float *compact_host = (float *)malloc((size_t)compact_host_bytes);
    if (!x_host || !mid_expected || !out_expected ||
        !batch_mid_expected || !batch_out_expected || !compact_host) {
        FAIL("packed execution host allocation\n");
    }
    for (uint32_t i = 0; i < IN_DIM; i++) {
        x_host[i] = (float)((int)((i * 7u) % 61u) - 30) / 512.0f;
    }
    ds4_gpu_tensor selected = {}, weights = {}, x = {}, xq = {};
    ds4_gpu_tensor gate_full = {}, up_full = {}, mid_full = {}, midq_full = {};
    ds4_gpu_tensor batch_down_full = {};
    ds4_gpu_tensor out_full = {}, group0_full = {}, group1_full = {};
    if (!alloc_tensor(&selected, sizeof(selected_host)) ||
        !alloc_tensor(&weights, sizeof(weights_host)) ||
        !alloc_tensor(&x, (uint64_t)IN_DIM * sizeof(float)) ||
        !alloc_tensor(&xq, xq_bytes) ||
        !alloc_tensor(&gate_full, mid_full_bytes) ||
        !alloc_tensor(&up_full, mid_full_bytes) ||
        !alloc_tensor(&mid_full, mid_full_bytes) ||
        !alloc_tensor(&midq_full, midq_full_bytes) ||
        !alloc_tensor(&batch_down_full, batch_down_full_bytes) ||
        !alloc_tensor(&out_full, out_full_bytes) ||
        !alloc_tensor(&group0_full, out_full_bytes) ||
        !alloc_tensor(&group1_full, out_full_bytes) ||
        !upload(&selected, selected_host, sizeof(selected_host)) ||
        !upload(&weights, weights_host, sizeof(weights_host)) ||
        !upload(&x, x_host, (uint64_t)IN_DIM * sizeof(float)) ||
        !ds4_gpu_rocm_q4k_row_shard_gate_up_tensor(
            &mid_full, &gate_full, &xq, model, model_bytes,
            gate_off, up_off, (uint64_t)MID_DIM * gate_row_bytes,
            gate_row_bytes, IN_DIM, MID_DIM, &selected, &weights,
            N_TOTAL_EXPERT, N_USED, 0u, MID_DIM, 0.0f, &x) ||
        !ds4_gpu_rocm_q8k_quantize_rows_tensor(
            &midq_full, &mid_full, MID_DIM, N_USED) ||
        !ds4_gpu_rocm_q4k_row_shard_down_tensor(
            &out_full, &group0_full, &group1_full, NULL, NULL,
            &midq_full, model, model_bytes, down_off,
            (uint64_t)OUT_DIM * down_row_bytes, down_row_bytes,
            MID_DIM, OUT_DIM, &selected, &weights, N_TOTAL_EXPERT,
            N_USED, EXPERT_SPLIT, 0u, OUT_DIM) ||
        !ds4_gpu_tensor_read(&mid_full, 0, mid_expected, mid_full_bytes) ||
        !ds4_gpu_tensor_read(&out_full, 0, out_expected, out_full_bytes)) {
        FAIL("packed execution full-row reference\n");
    }
    /* Batch kernels intentionally use a different WMMA/cold reduction tree
     * than the one-token DP4A path.  Preserve a separate exact oracle rather
     * than accepting a tolerance or comparing two different arithmetic
     * contracts. */
    if (!ds4_gpu_rocm_q4k_batch_row_shard_gate_up_tensor(
            &gate_full, &up_full, &mid_full, &xq, model, model_bytes,
            gate_off, up_off, (uint64_t)MID_DIM * gate_row_bytes,
            gate_row_bytes, IN_DIM, MID_DIM, &selected, &weights,
            N_TOTAL_EXPERT, N_USED, 1u, 0u, MID_DIM, 0.0f, &x) ||
        !ds4_gpu_tensor_read(&mid_full, 0, batch_mid_expected,
                             mid_full_bytes) ||
        !ds4_gpu_rocm_q4k_batch_row_shard_down_tensor(
            &batch_down_full, &midq_full, &mid_full, model, model_bytes,
            down_off, (uint64_t)OUT_DIM * down_row_bytes, down_row_bytes,
            MID_DIM, OUT_DIM, &selected, N_TOTAL_EXPERT, N_USED,
            1u, 0u, OUT_DIM) ||
        !ds4_gpu_rocm_q4k_batch_row_shard_reduce_tensor(
            &out_full, &batch_down_full, NULL, NULL, &selected,
            OUT_DIM, N_USED, 1u, EXPERT_SPLIT, 0u, OUT_DIM) ||
        !ds4_gpu_tensor_read(&out_full, 0, batch_out_expected,
                             out_full_bytes)) {
        FAIL("packed execution full-row batch reference\n");
    }
    uint64_t total_packed = 0;
    uint64_t combined_hash = 0;
    for (uint32_t tensor = 0; tensor < 3u; tensor++) {
        const uint32_t rows = full_rows[tensor] / 2u;
        const uint64_t bytes =
            (uint64_t)N_TOTAL_EXPERT * rows * row_bytes[tensor];
        float *expected = (float *)malloc((size_t)bytes);
        float *got = (float *)malloc((size_t)bytes);
        if (!expected || !got) FAIL("packed registry host buffers\n");
        for (uint32_t half = 0; half < 2u; half++) {
            const uint32_t row_base = half * rows;
            if (!ds4_gpu_q4k_packed_rows_declare(
                    model, model_bytes, offsets[tensor], N_TOTAL_EXPERT,
                    full_rows[tensor], row_bytes[tensor], row_base, rows,
                    kinds[tensor]) ||
                !ds4_gpu_q4k_packed_rows_load(
                    model, offsets[tensor], row_base, rows)) {
                FAIL("packed registry declare/load tensor=%u half=%u\n",
                     tensor, half);
            }
            pack_q4k_row_span((unsigned char *)expected,
                              model + offsets[tensor], N_TOTAL_EXPERT,
                              full_rows[tensor], row_base, rows,
                              row_bytes[tensor]);
            if (!ds4_gpu_q4k_packed_rows_readback(
                    model, offsets[tensor], row_base, rows, got, bytes) ||
                memcmp(expected, got, (size_t)bytes) != 0) {
                FAIL("packed registry bytes tensor=%u half=%u\n",
                     tensor, half);
            }
            combined_hash ^= fnv1a64(got, (size_t)bytes) +
                             UINT64_C(0x9e3779b97f4a7c15) *
                                 (1u + tensor * 2u + half);
            total_packed += bytes;
        }
        free(got);
        free(expected);
    }
    if (ds4_gpu_q4k_packed_rows_bytes() != total_packed) {
        FAIL("packed registry byte accounting ref=%llu got=%llu\n",
             (unsigned long long)total_packed,
             (unsigned long long)ds4_gpu_q4k_packed_rows_bytes());
    }

    /* Exercise the actual descriptor-backed compact decode functions.  No
     * appended synthetic packed tensor offsets are passed here: the original
     * GGUF offsets must resolve exclusively through the registry. */
    const uint32_t mid_half = MID_DIM / 2u;
    const uint32_t out_half = OUT_DIM / 2u;
    const uint64_t mid_half_bytes =
        (uint64_t)N_USED * mid_half * sizeof(float);
    const uint64_t midq_half_bytes =
        (uint64_t)N_USED * (mid_half / QK_K) * Q8_K_BLOCK_BYTES;
    const uint64_t out_half_bytes = (uint64_t)out_half * sizeof(float);
    ds4_gpu_tensor gate_half = {}, up_half = {};
    ds4_gpu_tensor mid_half_t[2] = {}, midq_half[2] = {};
    ds4_gpu_tensor out_half_t = {}, group0_half = {}, group1_half = {};
    if (!alloc_tensor(&gate_half, mid_half_bytes) ||
        !alloc_tensor(&up_half, mid_half_bytes) ||
        !alloc_tensor(&mid_half_t[0], mid_half_bytes) ||
        !alloc_tensor(&mid_half_t[1], mid_half_bytes) ||
        !alloc_tensor(&midq_half[0], midq_half_bytes) ||
        !alloc_tensor(&midq_half[1], midq_half_bytes) ||
        !alloc_tensor(&out_half_t, out_half_bytes) ||
        !alloc_tensor(&group0_half, out_half_bytes) ||
        !alloc_tensor(&group1_half, out_half_bytes)) {
        FAIL("packed execution compact allocation\n");
    }
    for (uint32_t half = 0; half < 2u; half++) {
        const uint32_t row_base = half * mid_half;
        if (!ds4_gpu_rocm_q4k_packed_row_gate_up_tensor(
                &mid_half_t[half], &gate_half, &xq, model,
                gate_off, up_off, gate_row_bytes, IN_DIM, MID_DIM,
                &selected, &weights, N_TOTAL_EXPERT, N_USED,
                row_base, mid_half, 0.0f, &x) ||
            !ds4_gpu_rocm_q8k_quantize_rows_tensor(
                &midq_half[half], &mid_half_t[half], mid_half, N_USED) ||
            !ds4_gpu_tensor_read(&mid_half_t[half], 0, compact_host,
                                 mid_half_bytes) ||
            !compare_compact_half(half ? "packed-gate-high" :
                                          "packed-gate-low",
                                  1u, mid_expected, compact_host,
                                  N_USED, MID_DIM, row_base, mid_half)) {
            FAIL("packed execution gate/up half %u\n", half);
        }
    }
    for (uint32_t half = 0; half < 2u; half++) {
        const uint32_t row_base = half * out_half;
        if (!ds4_gpu_rocm_q4k_packed_row_down_tensor(
                &out_half_t, &group0_half, &group1_half, NULL, NULL,
                &midq_half[0], &midq_half[1], model, down_off,
                down_row_bytes, MID_DIM, OUT_DIM, &selected, &weights,
                N_TOTAL_EXPERT, N_USED, EXPERT_SPLIT,
                row_base, out_half) ||
            !ds4_gpu_tensor_read(&out_half_t, 0, compact_host,
                                 out_half_bytes) ||
            memcmp(compact_host, out_expected + row_base,
                   (size_t)out_half_bytes) != 0) {
            FAIL("packed execution down half %u\n", half);
        }
    }
    /* The batch entry point must resolve both descriptors and retain its own
     * exact WMMA/cold compact-output arithmetic.  Re-quantize both compact
     * halves only after the decode oracle above has consumed its DP4A data. */
    for (uint32_t half = 0; half < 2u; half++) {
        const uint32_t row_base = half * mid_half;
        if (!ds4_gpu_rocm_q4k_batch_row_shard_gate_up_tensor(
                &gate_half, &up_half, &mid_half_t[half], &xq,
                model, model_bytes, gate_off, up_off,
                (uint64_t)MID_DIM * gate_row_bytes, gate_row_bytes,
                IN_DIM, MID_DIM, &selected, &weights,
                N_TOTAL_EXPERT, N_USED, 1u, row_base, mid_half, 0.0f, &x) ||
            !ds4_gpu_tensor_read(&mid_half_t[half], 0, compact_host,
                                 mid_half_bytes) ||
            !compare_compact_half(half ? "packed-batch-gate-high" :
                                          "packed-batch-gate-low",
                                  1u, batch_mid_expected, compact_host,
                                  N_USED, MID_DIM, row_base, mid_half) ||
            !ds4_gpu_rocm_q8k_quantize_rows_tensor(
                &midq_half[half], &mid_half_t[half], mid_half, N_USED)) {
            FAIL("packed execution batch gate/up half %u\n", half);
        }
    }
    const uint64_t down_pairs_half_bytes =
        (uint64_t)N_USED * out_half * sizeof(float);
    ds4_gpu_tensor down_pairs_half = {}, stitched_midq = {};
    if (!alloc_tensor(&down_pairs_half, down_pairs_half_bytes) ||
        !alloc_tensor(&stitched_midq, midq_full_bytes) ||
        !ds4_gpu_rocm_q4k_batch_packed_row_down_split_midq_tensor(
            &down_pairs_half, &stitched_midq,
            &midq_half[0], &midq_half[1], model, down_off,
            down_row_bytes, MID_DIM, OUT_DIM, &selected,
            N_TOTAL_EXPERT, N_USED, 1u, 0u, out_half) ||
        !ds4_gpu_rocm_q4k_batch_row_shard_reduce_compact_add_tensor(
            &out_half_t, &down_pairs_half, NULL, NULL, &selected,
            N_USED, 1u, EXPERT_SPLIT, out_half) ||
        !ds4_gpu_tensor_read(&out_half_t, 0, compact_host,
                             out_half_bytes) ||
        memcmp(compact_host, batch_out_expected,
               (size_t)out_half_bytes) != 0) {
        FAIL("packed execution batch split-mid down/reduce\n");
    }
    printf("packed_decode_primitives fnv64=%016llx exact=1 "
           "batch_split_mid=1 linear_source_blocked=1\n",
           (unsigned long long)fnv1a64(out_expected,
                                       (size_t)out_full_bytes));

    /* A support/MTP map may replace the short-lived generic range-cache
     * identity. Packed target slabs deliberately follow model-image lifetime
     * and must remain readable across that transition. */
    unsigned char alternate_map[64] = {};
    const uint64_t first_half_bytes =
        (uint64_t)N_TOTAL_EXPERT * (MID_DIM / 2u) * gate_row_bytes;
    unsigned char *lifetime = (unsigned char *)malloc((size_t)first_half_bytes);
    unsigned char *lifetime_expected =
        (unsigned char *)malloc((size_t)first_half_bytes);
    if (!lifetime || !lifetime_expected ||
        !ds4_gpu_set_model_map(alternate_map, sizeof(alternate_map)) ||
        !ds4_gpu_q4k_packed_rows_readback(
            model, gate_off, 0u, MID_DIM / 2u,
            lifetime, first_half_bytes)) {
        FAIL("packed registry image lifetime transition\n");
    }
    pack_q4k_row_span(lifetime_expected, model + gate_off, N_TOTAL_EXPERT,
                      MID_DIM, 0u, MID_DIM / 2u, gate_row_bytes);
    if (memcmp(lifetime_expected, lifetime, (size_t)first_half_bytes) != 0 ||
        !ds4_gpu_set_model_map(model, model_bytes)) {
        FAIL("packed registry image lifetime bytes\n");
    }
    free(lifetime_expected);
    free(lifetime);

    /* Declarations must block every legacy path even if a linear image was
     * already cached earlier by this synthetic arithmetic oracle. */
    const uint64_t gate_bytes =
        (uint64_t)N_TOTAL_EXPERT * MID_DIM * gate_row_bytes;
    if (ds4_gpu_cache_model_range(model, model_bytes, gate_off, gate_bytes,
                                  "packed-negative-linear")) {
        FAIL("packed registry allowed legacy linear resolution\n");
    }
    const uint64_t span_off = gate_off;
    const uint64_t span_size = gate_bytes;
    if (ds4_gpu_set_model_map_spans(model, model_bytes,
                                    &span_off, &span_size, 1u,
                                    span_size)) {
        FAIL("packed registry allowed intersecting model span\n");
    }
    printf("packed_row_registry bytes=%llu fnv64=%016llx "
           "linear_fail_closed=1 span_fail_closed=1 image_lifetime=1\n",
           (unsigned long long)total_packed,
           (unsigned long long)combined_hash);

    ds4_gpu_tensor_free_in_place(&group1_half);
    ds4_gpu_tensor_free_in_place(&group0_half);
    ds4_gpu_tensor_free_in_place(&out_half_t);
    ds4_gpu_tensor_free_in_place(&midq_half[1]);
    ds4_gpu_tensor_free_in_place(&midq_half[0]);
    ds4_gpu_tensor_free_in_place(&mid_half_t[1]);
    ds4_gpu_tensor_free_in_place(&mid_half_t[0]);
    ds4_gpu_tensor_free_in_place(&stitched_midq);
    ds4_gpu_tensor_free_in_place(&down_pairs_half);
    ds4_gpu_tensor_free_in_place(&up_half);
    ds4_gpu_tensor_free_in_place(&gate_half);
    ds4_gpu_tensor_free_in_place(&group1_full);
    ds4_gpu_tensor_free_in_place(&group0_full);
    ds4_gpu_tensor_free_in_place(&out_full);
    ds4_gpu_tensor_free_in_place(&batch_down_full);
    ds4_gpu_tensor_free_in_place(&midq_full);
    ds4_gpu_tensor_free_in_place(&mid_full);
    ds4_gpu_tensor_free_in_place(&up_full);
    ds4_gpu_tensor_free_in_place(&gate_full);
    ds4_gpu_tensor_free_in_place(&xq);
    ds4_gpu_tensor_free_in_place(&x);
    ds4_gpu_tensor_free_in_place(&weights);
    ds4_gpu_tensor_free_in_place(&selected);
    free(compact_host);
    free(batch_out_expected);
    free(batch_mid_expected);
    free(out_expected);
    free(mid_expected);
    free(x_host);
    return 1;
}

#endif
static int run_batch_case(
        uint32_t n_tokens, const void *model, uint64_t model_bytes,
        uint64_t gate_off, uint64_t up_off, uint64_t down_off,
        const uint64_t packed_gate_off[2],
        const uint64_t packed_up_off[2],
        const uint64_t packed_down_off[2],
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
                packed_gate_off[half], packed_up_off[half],
                (uint64_t)mid_half * gate_row_bytes, gate_row_bytes,
                IN_DIM, MID_DIM, &selected, &weights, N_TOTAL_EXPERT, N_USED,
                n_tokens, 0u, mid_half, 0.0f, &x) ||
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
                &down_compact, &midq, &mid_full, model, model_bytes,
                packed_down_off[half],
                (uint64_t)out_half * down_row_bytes, down_row_bytes,
                MID_DIM, OUT_DIM, &selected, N_TOTAL_EXPERT, N_USED,
                n_tokens, 0u, out_half) ||
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
    const uint64_t full_model_bytes = down_off +
                                      N_TOTAL_EXPERT * down_expert_bytes;
    const uint64_t packed_gate_bytes =
        (uint64_t)N_TOTAL_EXPERT * (MID_DIM / 2u) * gate_row_bytes;
    const uint64_t packed_down_bytes =
        (uint64_t)N_TOTAL_EXPERT * (OUT_DIM / 2u) * down_row_bytes;
    uint64_t packed_gate_off[2], packed_up_off[2], packed_down_off[2];
    uint64_t model_bytes = full_model_bytes;
    for (uint32_t half = 0; half < 2u; half++) {
        packed_gate_off[half] = model_bytes;
        model_bytes += packed_gate_bytes;
        packed_up_off[half] = model_bytes;
        model_bytes += packed_gate_bytes;
        packed_down_off[half] = model_bytes;
        model_bytes += packed_down_bytes;
    }
    unsigned char *model = (unsigned char *)malloc((size_t)model_bytes);
    CHECK(model, "allocate synthetic batch model");
    pack_q4k_table(model + gate_off, N_TOTAL_EXPERT, MID_DIM,
                   (uint32_t)in_blocks, 11u);
    pack_q4k_table(model + up_off, N_TOTAL_EXPERT, MID_DIM,
                   (uint32_t)in_blocks, 37u);
    pack_q4k_table(model + down_off, N_TOTAL_EXPERT, OUT_DIM,
                   (uint32_t)mid_blocks, 73u);
    for (uint32_t half = 0; half < 2u; half++) {
        pack_q4k_row_span(model + packed_gate_off[half], model + gate_off,
                          N_TOTAL_EXPERT, MID_DIM,
                          half * (MID_DIM / 2u), MID_DIM / 2u,
                          gate_row_bytes);
        pack_q4k_row_span(model + packed_up_off[half], model + up_off,
                          N_TOTAL_EXPERT, MID_DIM,
                          half * (MID_DIM / 2u), MID_DIM / 2u,
                          gate_row_bytes);
        pack_q4k_row_span(model + packed_down_off[half], model + down_off,
                          N_TOTAL_EXPERT, OUT_DIM,
                          half * (OUT_DIM / 2u), OUT_DIM / 2u,
                          down_row_bytes);
    }
    CHECK(ds4_gpu_set_model_map(model, model_bytes), "install batch model map");
    CHECK(EXPERT_SPLIT * 2u == N_TOTAL_EXPERT,
          "ordinary TP split must be balanced");

    const uint32_t token_cases[] = {6u, 129u, 2048u};
    for (uint32_t i = 0; i < sizeof(token_cases) / sizeof(token_cases[0]); i++) {
        CHECK(run_batch_case(token_cases[i], model, model_bytes,
                             gate_off, up_off, down_off,
                             packed_gate_off, packed_up_off, packed_down_off,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes),
              "exact production-kernel batch row-shard oracle");
    }
    free(model);
    ds4_gpu_cleanup();
    return 0;
}
