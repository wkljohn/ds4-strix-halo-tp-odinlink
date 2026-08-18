/* Production-shape, cache-free Q4_K one-token MoE microbenchmark.
 *
 * This is intentionally a whole routed-MoE call rather than a toy dot test:
 * it covers activation quantization, six gate/up experts, SwiGLU, middle
 * quantization, the direct six-expert down sum, and the optional fused addend.
 * Run in separate processes with DS4_ROCM_Q4K_DECODE_STAGE_XQ=0/1 because the
 * production dispatcher resolves that switch once per process.
 */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg) do {                                                \
    if (!(cond)) {                                                           \
        fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);           \
        return 1;                                                            \
    }                                                                        \
} while (0)

typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

enum {
    Q4_K_TYPE = 12,
    QK_K = 256,
    Q4_K_BLOCK_BYTES = 144,
    Q8_K_BLOCK_BYTES = 292,
    N_TOTAL_EXPERT = 12,
    N_USED = 6,
    IN_DIM = 4096,
    MID_DIM = 2048,
    OUT_DIM = 4096,
    WARMUP = 8,
    ITERS = 80,
};

static void pack_q4k_block(unsigned char *dst, uint32_t seed) {
    /* d=1/32, dmin=0; all sub-block scales are one and mins are zero. */
    dst[0] = 0x00;
    dst[1] = 0x28;
    dst[2] = 0x00;
    dst[3] = 0x00;
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
    for (uint32_t e = 0; e < experts; e++) {
        for (uint32_t row = 0; row < rows; row++) {
            for (uint32_t b = 0; b < blocks_per_row; b++) {
                const uint64_t i = ((uint64_t)e * rows + row) *
                                   blocks_per_row + b;
                pack_q4k_block(dst + i * Q4_K_BLOCK_BYTES,
                               salt + 17u * e + 13u * row + 7u * b);
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

static int run_one(ds4_gpu_tensor *out, ds4_gpu_tensor *gate,
                   ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
                   ds4_gpu_tensor *down, ds4_gpu_tensor *selected,
                   ds4_gpu_tensor *weights, ds4_gpu_tensor *x,
                   ds4_gpu_tensor *add_in, const void *model,
                   uint64_t model_bytes, uint64_t gate_off,
                   uint64_t up_off, uint64_t down_off,
                   uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
                   uint64_t down_expert_bytes, uint64_t down_row_bytes) {
    return ds4_gpu_routed_moe_one_tensor(
        out, gate, up, mid, down, model, model_bytes,
        gate_off, up_off, down_off, Q4_K_TYPE, Q4_K_TYPE,
        gate_expert_bytes, gate_row_bytes,
        down_expert_bytes, down_row_bytes,
        IN_DIM, MID_DIM, OUT_DIM, selected, weights,
        N_TOTAL_EXPERT, N_USED, 0.0f, x, add_in, 0, true);
}

static int run_reference_rank(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, ds4_gpu_tensor *down,
        ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
        ds4_gpu_tensor *x, ds4_gpu_tensor *add_in,
        const void *model, uint64_t model_bytes,
        uint64_t gate_off, uint64_t up_off, uint64_t down_off,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes,
        const int32_t route[N_USED], const float route_weights[N_USED],
        uint32_t expert_split, uint32_t rank) {
    int32_t local_route[N_USED];
    float local_weights[N_USED];
    const uint32_t lo = rank == 0u ? 0u : expert_split;
    const uint32_t hi = rank == 0u ? expert_split : N_TOTAL_EXPERT;
    for (uint32_t slot = 0; slot < N_USED; slot++) {
        const int32_t expert = route[slot];
        const int owned = expert >= (int32_t)lo && expert < (int32_t)hi;
        local_route[slot] = owned ? expert - (int32_t)lo : 0;
        local_weights[slot] = owned ? route_weights[slot] : 0.0f;
    }
    if (!upload(selected, local_route, sizeof(local_route)) ||
        !upload(weights, local_weights, sizeof(local_weights))) return 0;
    return ds4_gpu_routed_moe_one_tensor(
        out, gate, up, mid, down, model, model_bytes,
        gate_off + (uint64_t)lo * gate_expert_bytes,
        up_off + (uint64_t)lo * gate_expert_bytes,
        down_off + (uint64_t)lo * down_expert_bytes,
        Q4_K_TYPE, Q4_K_TYPE,
        gate_expert_bytes, gate_row_bytes,
        down_expert_bytes, down_row_bytes,
        IN_DIM, MID_DIM, OUT_DIM, selected, weights,
        hi - lo, N_USED, 0.0f, x, add_in, 0, true);
}

typedef struct {
    const char *name;
    int32_t route[N_USED];
    float weights[N_USED];
    uint32_t split;
} row_shard_case;

static int validate_row_shard_case(
        const row_shard_case *tc,
        ds4_gpu_tensor *reference0, ds4_gpu_tensor *reference1,
        ds4_gpu_tensor *candidate,
        ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down, ds4_gpu_tensor *candidate_mid,
        ds4_gpu_tensor *candidate_xq, ds4_gpu_tensor *candidate_midq,
        ds4_gpu_tensor *candidate_rank0, ds4_gpu_tensor *candidate_rank1,
        ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
        ds4_gpu_tensor *x, ds4_gpu_tensor *rank0_add,
        ds4_gpu_tensor *rank1_add,
        const void *model, uint64_t model_bytes,
        uint64_t gate_off, uint64_t up_off, uint64_t down_off,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes) {
    float hmid0[N_USED * MID_DIM], hmid1[N_USED * MID_DIM];
    if (!run_reference_rank(reference0, gate, up, mid, down, selected,
                            weights, x, rank0_add, model, model_bytes,
                            gate_off, up_off, down_off, gate_expert_bytes,
                            gate_row_bytes, down_expert_bytes, down_row_bytes,
                            tc->route, tc->weights, tc->split, 0u) ||
        !ds4_gpu_tensor_read(mid, 0, hmid0, sizeof(hmid0)) ||
        !run_reference_rank(reference1, gate, up, mid, down, selected,
                            weights, x, rank1_add, model, model_bytes,
                            gate_off, up_off, down_off, gate_expert_bytes,
                            gate_row_bytes, down_expert_bytes, down_row_bytes,
                            tc->route, tc->weights, tc->split, 1u) ||
        !ds4_gpu_tensor_read(mid, 0, hmid1, sizeof(hmid1))) {
        fprintf(stderr, "FAIL: %s production reference\n", tc->name);
        return 0;
    }
    if (!upload(selected, tc->route, sizeof(tc->route)) ||
        !upload(weights, tc->weights, sizeof(tc->weights))) {
        fprintf(stderr, "FAIL: %s restore global route\n", tc->name);
        return 0;
    }
    if (!ds4_gpu_rocm_q4k_row_shard_gate_up_tensor(
            candidate_mid, gate, candidate_xq, model, model_bytes,
            gate_off, up_off, gate_expert_bytes, gate_row_bytes,
            IN_DIM, MID_DIM, selected, weights, N_TOTAL_EXPERT, N_USED,
            0u, MID_DIM / 2u, 0.0f, x) ||
        !ds4_gpu_rocm_q4k_row_shard_gate_up_tensor(
            candidate_mid, gate, candidate_xq, model, model_bytes,
            gate_off, up_off, gate_expert_bytes, gate_row_bytes,
            IN_DIM, MID_DIM, selected, weights, N_TOTAL_EXPERT, N_USED,
            MID_DIM / 2u, MID_DIM / 2u, 0.0f, x) ||
        hipDeviceSynchronize() != hipSuccess) {
        fprintf(stderr, "FAIL: %s row-shard gate/up candidate\n", tc->name);
        return 0;
    }
    float hmid_candidate[N_USED * MID_DIM];
    if (!ds4_gpu_tensor_read(candidate_mid, 0, hmid_candidate,
                             sizeof(hmid_candidate))) return 0;
    for (uint32_t slot = 0; slot < N_USED; slot++) {
        const float *expected = (uint32_t)tc->route[slot] < tc->split ?
            hmid0 : hmid1;
        const uint64_t base = (uint64_t)slot * MID_DIM;
        if (memcmp(expected + base, hmid_candidate + base,
                   MID_DIM * sizeof(float)) != 0) {
            uint32_t row = 0;
            while (row < MID_DIM &&
                   memcmp(expected + base + row, hmid_candidate + base + row,
                          sizeof(float)) == 0) row++;
            fprintf(stderr,
                    "FAIL: row-shard mid case=%s slot=%u row=%u ref=%a got=%a\n",
                    tc->name, slot, row,
                    row < MID_DIM ? expected[base + row] : 0.0f,
                    row < MID_DIM ? hmid_candidate[base + row] : 0.0f);
            return 0;
        }
    }
    if (!ds4_gpu_rocm_q8k_quantize_row_shard_tensor(
            candidate_midq, candidate_mid, MID_DIM, N_USED,
            0u, MID_DIM / 2u) ||
        !ds4_gpu_rocm_q8k_quantize_row_shard_tensor(
            candidate_midq, candidate_mid, MID_DIM, N_USED,
            MID_DIM / 2u, MID_DIM / 2u) ||
        !ds4_gpu_rocm_q4k_row_shard_down_tensor(
            candidate, candidate_rank0, candidate_rank1,
            rank0_add, rank1_add, candidate_midq,
            model, model_bytes, down_off, down_expert_bytes,
            down_row_bytes, MID_DIM, OUT_DIM, selected, weights,
            N_TOTAL_EXPERT, N_USED, tc->split, 0u, OUT_DIM / 2u) ||
        !ds4_gpu_rocm_q4k_row_shard_down_tensor(
            candidate, candidate_rank0, candidate_rank1,
            rank0_add, rank1_add, candidate_midq,
            model, model_bytes, down_off, down_expert_bytes,
            down_row_bytes, MID_DIM, OUT_DIM, selected, weights,
            N_TOTAL_EXPERT, N_USED, tc->split,
            OUT_DIM / 2u, OUT_DIM / 2u) ||
        hipDeviceSynchronize() != hipSuccess) {
        fprintf(stderr, "FAIL: %s row-shard candidate\n", tc->name);
        return 0;
    }

    float h0[OUT_DIM], h1[OUT_DIM], hc[OUT_DIM], href[OUT_DIM];
    if (!ds4_gpu_tensor_read(reference0, 0, h0, sizeof(h0)) ||
        !ds4_gpu_tensor_read(reference1, 0, h1, sizeof(h1)) ||
        !ds4_gpu_tensor_read(candidate, 0, hc, sizeof(hc))) {
        fprintf(stderr, "FAIL: %s output read\n", tc->name);
        return 0;
    }
    for (uint32_t i = 0; i < OUT_DIM; i++) href[i] = h0[i] + h1[i];
    if (memcmp(href, hc, sizeof(href)) != 0) {
        float masked[N_USED], hgroup[OUT_DIM];
        for (uint32_t group = 0; group < 2u; group++) {
            for (uint32_t slot = 0; slot < N_USED; slot++) {
                const uint32_t expert = (uint32_t)tc->route[slot];
                const int own = group == 0u ? expert < tc->split :
                                               expert >= tc->split;
                masked[slot] = own ? tc->weights[slot] : 0.0f;
            }
            if (!upload(weights, masked, sizeof(masked)) ||
                !ds4_gpu_rocm_q4k_row_shard_down_tensor(
                    candidate, candidate_rank0, candidate_rank1,
                    group == 0u ? rank0_add : NULL,
                    group == 1u ? rank1_add : NULL, candidate_midq,
                    model, model_bytes, down_off, down_expert_bytes,
                    down_row_bytes, MID_DIM, OUT_DIM, selected, weights,
                    N_TOTAL_EXPERT, N_USED, tc->split, 0u, OUT_DIM / 2u) ||
                !ds4_gpu_rocm_q4k_row_shard_down_tensor(
                    candidate, candidate_rank0, candidate_rank1,
                    group == 0u ? rank0_add : NULL,
                    group == 1u ? rank1_add : NULL, candidate_midq,
                    model, model_bytes, down_off, down_expert_bytes,
                    down_row_bytes, MID_DIM, OUT_DIM, selected, weights,
                    N_TOTAL_EXPERT, N_USED, tc->split,
                    OUT_DIM / 2u, OUT_DIM / 2u) ||
                !ds4_gpu_tensor_read(candidate, 0, hgroup, sizeof(hgroup))) {
                fprintf(stderr, "FAIL: %s grouped diagnostic\n", tc->name);
                return 0;
            }
            const float *group_ref = group == 0u ? h0 : h1;
            if (memcmp(group_ref, hgroup, sizeof(hgroup)) != 0) {
                uint32_t group_first = 0;
                while (group_first < OUT_DIM &&
                       memcmp(group_ref + group_first, hgroup + group_first,
                              sizeof(float)) == 0) group_first++;
                fprintf(stderr,
                        "FAIL: row-shard group=%u first=%u ref=%a got=%a\n",
                        group, group_first,
                        group_first < OUT_DIM ? group_ref[group_first] : 0.0f,
                        group_first < OUT_DIM ? hgroup[group_first] : 0.0f);
            }
        }
        uint32_t first = 0;
        while (first < OUT_DIM &&
               memcmp(&href[first], &hc[first], sizeof(float)) == 0) first++;
        fprintf(stderr,
                "FAIL: row-shard case=%s first=%u ref=%a got=%a "
                "ref_fnv=%016llx got_fnv=%016llx\n",
                tc->name, first,
                first < OUT_DIM ? href[first] : 0.0f,
                first < OUT_DIM ? hc[first] : 0.0f,
                (unsigned long long)fnv1a64(href, sizeof(href)),
                (unsigned long long)fnv1a64(hc, sizeof(hc)));
        return 0;
    }
    printf("row_shard_oracle case=%s split=%u fnv64=%016llx exact=1\n",
           tc->name, tc->split,
           (unsigned long long)fnv1a64(hc, sizeof(hc)));
    return 1;
}

int main(void) {
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
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
    const uint64_t model_bytes = down_off +
                                 N_TOTAL_EXPERT * down_expert_bytes;

    unsigned char *model = (unsigned char *)malloc((size_t)model_bytes);
    CHECK(model, "allocate production-shape Q4_K model");
    pack_q4k_table(model + gate_off, N_TOTAL_EXPERT, MID_DIM,
                   (uint32_t)in_blocks, 11);
    pack_q4k_table(model + up_off, N_TOTAL_EXPERT, MID_DIM,
                   (uint32_t)in_blocks, 37);
    pack_q4k_table(model + down_off, N_TOTAL_EXPERT, OUT_DIM,
                   (uint32_t)mid_blocks, 73);
    CHECK(ds4_gpu_set_model_map(model, model_bytes), "install synthetic model");

    const uint64_t pair_values = (uint64_t)N_USED * MID_DIM;
    ds4_gpu_tensor out = {}, gate = {}, up = {}, mid = {}, down = {};
    ds4_gpu_tensor selected = {}, weights = {}, x = {}, add_in = {};
    ds4_gpu_tensor reference0 = {}, reference1 = {}, candidate = {};
    ds4_gpu_tensor candidate_mid = {}, candidate_xq = {}, candidate_midq = {};
    ds4_gpu_tensor candidate_rank0 = {}, candidate_rank1 = {};
    ds4_gpu_tensor rank0_add = {}, rank1_add = {};
    CHECK(alloc_tensor(&out, OUT_DIM * sizeof(float)), "allocate out");
    CHECK(alloc_tensor(&gate, pair_values * sizeof(float)), "allocate gate");
    CHECK(alloc_tensor(&up, pair_values * sizeof(float)), "allocate up");
    CHECK(alloc_tensor(&mid, pair_values * sizeof(float)), "allocate mid");
    CHECK(alloc_tensor(&down, (uint64_t)N_USED * OUT_DIM * sizeof(float)),
          "allocate down scratch");
    CHECK(alloc_tensor(&selected, N_USED * sizeof(int32_t)), "allocate selection");
    CHECK(alloc_tensor(&weights, N_USED * sizeof(float)), "allocate weights");
    CHECK(alloc_tensor(&x, IN_DIM * sizeof(float)), "allocate input");
    CHECK(alloc_tensor(&add_in, OUT_DIM * sizeof(float)), "allocate addend");
    CHECK(alloc_tensor(&reference0, OUT_DIM * sizeof(float)),
          "allocate rank0 reference");
    CHECK(alloc_tensor(&reference1, OUT_DIM * sizeof(float)),
          "allocate rank1 reference");
    CHECK(alloc_tensor(&candidate, OUT_DIM * sizeof(float)),
          "allocate row-shard candidate");
    CHECK(alloc_tensor(&candidate_rank0, OUT_DIM * sizeof(float)),
          "allocate row-shard rank0 routed scratch");
    CHECK(alloc_tensor(&candidate_rank1, OUT_DIM * sizeof(float)),
          "allocate row-shard rank1 routed scratch");
    CHECK(alloc_tensor(&candidate_mid, pair_values * sizeof(float)),
          "allocate row-shard mid");
    CHECK(alloc_tensor(&candidate_xq,
                       (IN_DIM / QK_K) * Q8_K_BLOCK_BYTES),
          "allocate row-shard xq");
    CHECK(alloc_tensor(&candidate_midq,
                       (uint64_t)N_USED * (MID_DIM / QK_K) *
                           Q8_K_BLOCK_BYTES),
          "allocate row-shard midq");
    CHECK(alloc_tensor(&rank0_add, OUT_DIM * sizeof(float)),
          "allocate rank0 shared addend");
    CHECK(alloc_tensor(&rank1_add, OUT_DIM * sizeof(float)),
          "allocate rank1 shared addend");

    /* A representative TP rank owns about half of the six selected experts.
     * Remapping uses expert 0 plus a zero route weight for unowned slots. */
    const int32_t route[N_USED] = {0, 1, 2, 0, 0, 0};
    const float route_weights[N_USED] = {0.31f, 0.23f, 0.17f, 0.0f, 0.0f, 0.0f};
    float hx[IN_DIM], hadd[OUT_DIM], h_rank0_add[OUT_DIM], h_rank1_add[OUT_DIM];
    for (uint32_t i = 0; i < IN_DIM; i++)
        hx[i] = (float)((int)(i % 31u) - 15) * 0.03125f;
    for (uint32_t i = 0; i < OUT_DIM; i++)
        hadd[i] = (float)((int)(i % 19u) - 9) * 0.00390625f;
    for (uint32_t i = 0; i < OUT_DIM; i++) {
        h_rank0_add[i] = (float)((int)(i % 23u) - 11) * 0.001953125f;
        h_rank1_add[i] = (float)((int)(i % 29u) - 14) * 0.0009765625f;
    }
    CHECK(upload(&selected, route, sizeof(route)), "upload route");
    CHECK(upload(&weights, route_weights, sizeof(route_weights)), "upload weights");
    CHECK(upload(&x, hx, sizeof(hx)), "upload input");
    CHECK(upload(&add_in, hadd, sizeof(hadd)), "upload addend");
    CHECK(upload(&rank0_add, h_rank0_add, sizeof(h_rank0_add)),
          "upload rank0 shared addend");
    CHECK(upload(&rank1_add, h_rank1_add, sizeof(h_rank1_add)),
          "upload rank1 shared addend");
    CHECK(setenv("DS4_ROCM_TP_SKIP_UNOWNED", "1", 1) == 0,
          "enable exact unowned skip");
    for (uint32_t i = 0; i < WARMUP; i++) {
        CHECK(run_one(&out, &gate, &up, &mid, &down, &selected, &weights,
                      &x, &add_in, model, model_bytes, gate_off, up_off,
                      down_off, gate_expert_bytes, gate_row_bytes,
                      down_expert_bytes, down_row_bytes), "warmup routed MoE");
    }
    CHECK(hipDeviceSynchronize() == hipSuccess, "synchronize warmup");

    hipEvent_t start = nullptr, stop = nullptr;
    CHECK(hipEventCreate(&start) == hipSuccess &&
          hipEventCreate(&stop) == hipSuccess, "create timing events");
    CHECK(hipEventRecord(start) == hipSuccess, "record timing start");
    for (uint32_t i = 0; i < ITERS; i++) {
        CHECK(run_one(&out, &gate, &up, &mid, &down, &selected, &weights,
                      &x, &add_in, model, model_bytes, gate_off, up_off,
                      down_off, gate_expert_bytes, gate_row_bytes,
                      down_expert_bytes, down_row_bytes), "timed routed MoE");
    }
    CHECK(hipEventRecord(stop) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess, "finish timing");
    float elapsed_ms = 0.0f;
    CHECK(hipEventElapsedTime(&elapsed_ms, start, stop) == hipSuccess,
          "read timing");

    float hout[OUT_DIM];
    CHECK(ds4_gpu_tensor_read(&out, 0, hout, sizeof(hout)), "read output");
    const char *dump_path = getenv("DS4_TEST_OUTPUT_FILE");
    if (dump_path && dump_path[0]) {
        FILE *dump = fopen(dump_path, "wb");
        CHECK(dump, "open output dump");
        CHECK(fwrite(hout, 1, sizeof(hout), dump) == sizeof(hout),
              "write output dump");
        CHECK(fclose(dump) == 0, "close output dump");
    }
    printf("test_rocm_q4k_decode_bench: stage_xq=%s avg_ms=%.6f "
           "output_fnv64=%016llx model_mib=%.2f\n",
           getenv("DS4_ROCM_Q4K_DECODE_STAGE_XQ") ?: "unset",
           elapsed_ms / (float)ITERS,
           (unsigned long long)fnv1a64(hout, sizeof(hout)),
           (double)model_bytes / (1024.0 * 1024.0));

    const row_shard_case row_cases[] = {
        {"cross-split", {0, 1, 2, 6, 7, 8},
         {0.31f, 0.23f, 0.17f, 0.13f, 0.09f, 0.07f}, 6u},
        {"empty-rank1-group", {0, 1, 2, 3, 4, 5},
         {0.31f, 0.23f, 0.17f, 0.13f, 0.09f, 0.07f}, 6u},
        {"zero-weight-slot", {0, 1, 2, 6, 7, 8},
         {0.31f, 0.23f, 0.0f, 0.13f, 0.09f, 0.07f}, 6u},
        {"interleaved-groups", {0, 6, 1, 7, 2, 8},
         {0.31f, 0.23f, 0.17f, 0.13f, 0.09f, 0.07f}, 6u},
        {"odd-split", {0, 5, 1, 6, 2, 7},
         {0.31f, 0.23f, 0.17f, 0.13f, 0.09f, 0.07f}, 5u},
    };
    for (uint32_t i = 0; i < sizeof(row_cases) / sizeof(row_cases[0]); i++) {
        CHECK(validate_row_shard_case(
                  &row_cases[i], &reference0, &reference1, &candidate,
                  &gate, &up, &mid, &down, &candidate_mid,
                  &candidate_xq, &candidate_midq,
                  &candidate_rank0, &candidate_rank1,
                  &selected, &weights,
                  &x, &rank0_add, &rank1_add, model, model_bytes,
                  gate_off, up_off, down_off, gate_expert_bytes,
                  gate_row_bytes, down_expert_bytes, down_row_bytes),
              "bitwise Q4_K row-shard oracle");
    }

    /* Compare one rank's balanced row shard against the current three-full-
     * expert control.  The peer Q8_K half is prepared before timing, matching
     * a completed mid exchange; transport is deliberately measured only by
     * ds4-bench-tp, not hidden in this arithmetic microbenchmark. */
    const row_shard_case *perf = &row_cases[0];
    float reference_rank_ms[2] = {0.0f, 0.0f};
    float candidate_rank_ms[2] = {0.0f, 0.0f};
    for (uint32_t rank = 0; rank < 2u; rank++) {
        CHECK(run_reference_rank(
                  rank == 0u ? &reference0 : &reference1,
                  &gate, &up, &mid, &down, &selected, &weights, &x,
                  rank == 0u ? &rank0_add : &rank1_add,
                  model, model_bytes, gate_off, up_off, down_off,
                  gate_expert_bytes, gate_row_bytes,
                  down_expert_bytes, down_row_bytes,
                  perf->route, perf->weights, perf->split, rank),
              "prepare reference rank timing route");
        const uint32_t lo = rank == 0u ? 0u : perf->split;
        const uint32_t hi = rank == 0u ? perf->split : N_TOTAL_EXPERT;
        ds4_gpu_tensor *rank_out = rank == 0u ? &reference0 : &reference1;
        ds4_gpu_tensor *rank_add = rank == 0u ? &rank0_add : &rank1_add;
        CHECK(hipEventRecord(start) == hipSuccess,
              "record reference-rank timing start");
        for (uint32_t i = 0; i < ITERS; i++) {
            CHECK(ds4_gpu_routed_moe_one_tensor(
                      rank_out, &gate, &up, &mid, &down,
                      model, model_bytes,
                      gate_off + (uint64_t)lo * gate_expert_bytes,
                      up_off + (uint64_t)lo * gate_expert_bytes,
                      down_off + (uint64_t)lo * down_expert_bytes,
                      Q4_K_TYPE, Q4_K_TYPE,
                      gate_expert_bytes, gate_row_bytes,
                      down_expert_bytes, down_row_bytes,
                      IN_DIM, MID_DIM, OUT_DIM, &selected, &weights,
                      hi - lo, N_USED, 0.0f, &x, rank_add, 0, true),
                  "timed reference rank");
        }
        CHECK(hipEventRecord(stop) == hipSuccess &&
              hipEventSynchronize(stop) == hipSuccess &&
              hipEventElapsedTime(&reference_rank_ms[rank], start, stop) ==
                  hipSuccess,
              "finish reference-rank timing");
        reference_rank_ms[rank] /= (float)ITERS;
    }

    CHECK(upload(&selected, perf->route, sizeof(perf->route)) &&
          upload(&weights, perf->weights, sizeof(perf->weights)),
          "restore row-shard timing route");
    CHECK(ds4_gpu_rocm_q4k_row_shard_gate_up_tensor(
              &candidate_mid, &gate, &candidate_xq, model, model_bytes,
              gate_off, up_off, gate_expert_bytes, gate_row_bytes,
              IN_DIM, MID_DIM, &selected, &weights,
              N_TOTAL_EXPERT, N_USED, 0u, MID_DIM / 2u, 0.0f, &x) &&
          ds4_gpu_rocm_q8k_quantize_row_shard_tensor(
              &candidate_midq, &candidate_mid, MID_DIM, N_USED,
              0u, MID_DIM / 2u) &&
          ds4_gpu_rocm_q4k_row_shard_gate_up_tensor(
              &candidate_mid, &gate, &candidate_xq, model, model_bytes,
              gate_off, up_off, gate_expert_bytes, gate_row_bytes,
              IN_DIM, MID_DIM, &selected, &weights,
              N_TOTAL_EXPERT, N_USED, MID_DIM / 2u, MID_DIM / 2u,
              0.0f, &x) &&
          ds4_gpu_rocm_q8k_quantize_row_shard_tensor(
              &candidate_midq, &candidate_mid, MID_DIM, N_USED,
              MID_DIM / 2u, MID_DIM / 2u),
          "prepare both row-shard mid halves");
    for (uint32_t rank = 0; rank < 2u; rank++) {
        const uint32_t mid_base = rank * (MID_DIM / 2u);
        const uint32_t out_base = rank * (OUT_DIM / 2u);
        auto run_candidate_rank = [&]() -> int {
            return ds4_gpu_rocm_q4k_row_shard_gate_up_tensor(
                       &candidate_mid, &gate, &candidate_xq,
                       model, model_bytes,
                       gate_off, up_off, gate_expert_bytes, gate_row_bytes,
                       IN_DIM, MID_DIM, &selected, &weights,
                       N_TOTAL_EXPERT, N_USED, mid_base, MID_DIM / 2u,
                       0.0f, &x) &&
                   ds4_gpu_rocm_q8k_quantize_row_shard_tensor(
                       &candidate_midq, &candidate_mid, MID_DIM, N_USED,
                       mid_base, MID_DIM / 2u) &&
                   ds4_gpu_rocm_q4k_row_shard_down_tensor(
                       &candidate, &candidate_rank0, &candidate_rank1,
                       &rank0_add, &rank1_add, &candidate_midq,
                       model, model_bytes, down_off, down_expert_bytes,
                       down_row_bytes, MID_DIM, OUT_DIM, &selected, &weights,
                       N_TOTAL_EXPERT, N_USED, perf->split,
                       out_base, OUT_DIM / 2u);
        };
        for (uint32_t i = 0; i < WARMUP; i++) {
            CHECK(run_candidate_rank(), "warm row-shard rank timing");
        }
        CHECK(hipDeviceSynchronize() == hipSuccess,
              "synchronize row-shard rank warmup");
        CHECK(hipEventRecord(start) == hipSuccess,
              "record row-shard rank timing start");
        for (uint32_t i = 0; i < ITERS; i++) {
            CHECK(run_candidate_rank(), "timed row-shard rank");
        }
        CHECK(hipEventRecord(stop) == hipSuccess &&
              hipEventSynchronize(stop) == hipSuccess &&
              hipEventElapsedTime(&candidate_rank_ms[rank], start, stop) ==
                  hipSuccess,
              "finish row-shard rank timing");
        candidate_rank_ms[rank] /= (float)ITERS;
    }
    const float reference_critical =
        reference_rank_ms[0] > reference_rank_ms[1] ?
            reference_rank_ms[0] : reference_rank_ms[1];
    const float candidate_critical =
        candidate_rank_ms[0] > candidate_rank_ms[1] ?
            candidate_rank_ms[0] : candidate_rank_ms[1];
    printf("row_shard_economics ref_rank_ms=%.6f/%.6f "
           "candidate_rank_ms=%.6f/%.6f critical_change=%+.1f%%\n",
           reference_rank_ms[0], reference_rank_ms[1],
           candidate_rank_ms[0], candidate_rank_ms[1],
           100.0f * (candidate_critical / reference_critical - 1.0f));

    CHECK(hipEventDestroy(stop) == hipSuccess, "destroy stop event");
    CHECK(hipEventDestroy(start) == hipSuccess, "destroy start event");
    ds4_gpu_tensor_free_in_place(&add_in);
    ds4_gpu_tensor_free_in_place(&rank1_add);
    ds4_gpu_tensor_free_in_place(&rank0_add);
    ds4_gpu_tensor_free_in_place(&candidate_midq);
    ds4_gpu_tensor_free_in_place(&candidate_xq);
    ds4_gpu_tensor_free_in_place(&candidate_mid);
    ds4_gpu_tensor_free_in_place(&candidate_rank1);
    ds4_gpu_tensor_free_in_place(&candidate_rank0);
    ds4_gpu_tensor_free_in_place(&candidate);
    ds4_gpu_tensor_free_in_place(&reference1);
    ds4_gpu_tensor_free_in_place(&reference0);
    ds4_gpu_tensor_free_in_place(&x);
    ds4_gpu_tensor_free_in_place(&weights);
    ds4_gpu_tensor_free_in_place(&selected);
    ds4_gpu_tensor_free_in_place(&down);
    ds4_gpu_tensor_free_in_place(&mid);
    ds4_gpu_tensor_free_in_place(&up);
    ds4_gpu_tensor_free_in_place(&gate);
    ds4_gpu_tensor_free_in_place(&out);
    free(model);
    ds4_gpu_cleanup();
    return 0;
}
