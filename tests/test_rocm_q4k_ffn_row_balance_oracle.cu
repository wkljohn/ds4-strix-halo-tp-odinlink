/* FFN row-balance and gate-free K-shard numerical oracle.
 *
 * Exact Mechanism I: shipped ds4_gpu_routed_moe_one_tensor (full mid 2048,
 * out 4096) versus independently computed 1024-row mid halves followed by
 * full-K down on each 2048-row output half and concatenation.
 *
 * Lane-B precheck: incumbent expert-id TP ownership with full-K down versus
 * all six routes on both ranks, each using one 1024-row gate/up and matching
 * four-block down-K half. The existing final rank add is retained, but the
 * FP32 reduction association intentionally changes and is measured rather
 * than claimed bit-identical.
 */
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <hip/hip_runtime.h>
#include <math.h>
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
    N_USED = 6,
    IN_DIM = 4096,
    MID_DIM = 2048,
    MID_HALF = 1024,
    OUT_DIM = 4096,
    OUT_HALF = 2048,
    N_TOTAL_MAX = 12,
    TIMING_WARMUP = 8,
    TIMING_SAMPLES = 33,
};

static void pack_q4k_block(unsigned char *dst, uint32_t seed) {
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

static void pack_row_span(unsigned char *dst, const unsigned char *src,
                          uint32_t experts, uint32_t full_rows,
                          uint32_t row0, uint32_t n_rows, uint32_t row_bytes) {
    for (uint32_t e = 0; e < experts; e++) {
        for (uint32_t r = 0; r < n_rows; r++) {
            memcpy(dst + ((uint64_t)e * n_rows + r) * row_bytes,
                   src + ((uint64_t)e * full_rows + row0 + r) * row_bytes,
                   row_bytes);
        }
    }
}

static void pack_k_span(unsigned char *dst, const unsigned char *src,
                        uint32_t experts, uint32_t rows,
                        uint32_t full_blocks_per_row,
                        uint32_t block0, uint32_t n_blocks) {
    const uint32_t full_row_bytes = full_blocks_per_row * Q4_K_BLOCK_BYTES;
    const uint32_t packed_row_bytes = n_blocks * Q4_K_BLOCK_BYTES;
    for (uint32_t e = 0; e < experts; e++) {
        for (uint32_t row = 0; row < rows; row++) {
            memcpy(dst + ((uint64_t)e * rows + row) * packed_row_bytes,
                   src + ((uint64_t)e * rows + row) * full_row_bytes +
                       (uint64_t)block0 * Q4_K_BLOCK_BYTES,
                   packed_row_bytes);
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

static int compare_float_ascending(const void *a, const void *b) {
    const float av = *(const float *)a;
    const float bv = *(const float *)b;
    return av < bv ? -1 : av > bv ? 1 : 0;
}

static int add_outputs(float *host_out, const float *a, const float *b) {
    ds4_gpu_tensor da = {}, db = {}, sum = {};
    int ok = alloc_tensor(&da, OUT_DIM * sizeof(float)) &&
             alloc_tensor(&db, OUT_DIM * sizeof(float)) &&
             alloc_tensor(&sum, OUT_DIM * sizeof(float)) &&
             upload(&da, a, OUT_DIM * sizeof(float)) &&
             upload(&db, b, OUT_DIM * sizeof(float)) &&
             ds4_gpu_add_tensor(&sum, &da, &db, OUT_DIM) != 0 &&
             hipDeviceSynchronize() == hipSuccess &&
             ds4_gpu_tensor_read(&sum, 0, host_out,
                                 OUT_DIM * sizeof(float)) != 0;
    ds4_gpu_tensor_free_in_place(&sum);
    ds4_gpu_tensor_free_in_place(&db);
    ds4_gpu_tensor_free_in_place(&da);
    return ok;
}

static int run_moe(ds4_gpu_tensor *out, ds4_gpu_tensor *gate,
                   ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
                   ds4_gpu_tensor *down, ds4_gpu_tensor *selected,
                   ds4_gpu_tensor *weights, ds4_gpu_tensor *x,
                   ds4_gpu_tensor *add_in, const void *model,
                   uint64_t model_bytes, uint64_t gate_off,
                   uint64_t up_off, uint64_t down_off,
                   uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
                   uint64_t down_expert_bytes, uint64_t down_row_bytes,
                   uint32_t mid_dim, uint32_t out_dim, uint32_t n_total) {
    return ds4_gpu_routed_moe_one_tensor(
        out, gate, up, mid, down, model, model_bytes,
        gate_off, up_off, down_off, Q4_K_TYPE, Q4_K_TYPE,
        gate_expert_bytes, gate_row_bytes,
        down_expert_bytes, down_row_bytes,
        IN_DIM, mid_dim, out_dim, selected, weights,
        n_total, N_USED, 0.0f, x, add_in, 0, true);
}

struct timing_arm {
    const char *name;
    const void *model_map;
    uint64_t model_bytes;
    uint64_t gate_off;
    uint64_t up_off;
    uint64_t down_off;
    uint64_t gate_expert_bytes;
    uint64_t gate_row_bytes;
    uint64_t down_expert_bytes;
    uint64_t down_row_bytes;
    uint32_t mid_dim;
    const ds4_gpu_tensor *weights;
};

static int time_arm_once(
        const struct timing_arm *arm,
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down, ds4_gpu_tensor *selected,
        ds4_gpu_tensor *x, hipEvent_t start, hipEvent_t stop,
        float *elapsed_ms) {
    CHECK(ds4_gpu_set_model_map(arm->model_map, arm->model_bytes),
          "install resident timing map");
    CHECK(hipEventRecord(start) == hipSuccess, "timing start");
    CHECK(run_moe(out, gate, up, mid, down, selected,
                  (ds4_gpu_tensor *)arm->weights, x, NULL,
                  arm->model_map, arm->model_bytes,
                  arm->gate_off, arm->up_off, arm->down_off,
                  arm->gate_expert_bytes, arm->gate_row_bytes,
                  arm->down_expert_bytes, arm->down_row_bytes,
                  arm->mid_dim, OUT_DIM, N_TOTAL_MAX),
          arm->name);
    CHECK(hipEventRecord(stop) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess, "timing stop");
    CHECK(hipEventElapsedTime(elapsed_ms, start, stop) == hipSuccess,
          "timing elapsed");
    return 0;
}

static int run_kshard_shape_cost_gate(
        const unsigned char *full_model,
        uint64_t full_gate_off, uint64_t full_up_off,
        uint64_t full_down_off, uint64_t full_model_bytes,
        uint64_t gate_row_bytes, uint64_t down_row_bytes,
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down, ds4_gpu_tensor *selected,
        ds4_gpu_tensor *x) {
    const uint64_t full_gate_expert = MID_DIM * gate_row_bytes;
    const uint64_t full_down_expert = OUT_DIM * down_row_bytes;
    const uint64_t half_gate_expert = MID_HALF * gate_row_bytes;
    const uint32_t half_blocks = MID_HALF / QK_K;
    const uint64_t half_down_row_bytes =
        (uint64_t)half_blocks * Q4_K_BLOCK_BYTES;
    const uint64_t half_down_expert = OUT_DIM * half_down_row_bytes;
    const uint64_t packed_gu_bytes = N_TOTAL_MAX * half_gate_expert;
    const uint64_t packed_down_bytes = N_TOTAL_MAX * half_down_expert;
    const uint64_t packed_map_bytes = packed_gu_bytes * 2u + packed_down_bytes;

    unsigned char *gu0 = (unsigned char *)malloc((size_t)packed_gu_bytes);
    unsigned char *gu1 = (unsigned char *)malloc((size_t)packed_gu_bytes);
    unsigned char *up0 = (unsigned char *)malloc((size_t)packed_gu_bytes);
    unsigned char *up1 = (unsigned char *)malloc((size_t)packed_gu_bytes);
    unsigned char *map0 = (unsigned char *)malloc((size_t)packed_map_bytes);
    unsigned char *map1 = (unsigned char *)malloc((size_t)packed_map_bytes);
    CHECK(gu0 && gu1 && up0 && up1 && map0 && map1,
          "timing host maps");

    pack_row_span(gu0, full_model + full_gate_off, N_TOTAL_MAX, MID_DIM,
                  0u, MID_HALF, (uint32_t)gate_row_bytes);
    pack_row_span(gu1, full_model + full_gate_off, N_TOTAL_MAX, MID_DIM,
                  MID_HALF, MID_HALF, (uint32_t)gate_row_bytes);
    pack_row_span(up0, full_model + full_up_off, N_TOTAL_MAX, MID_DIM,
                  0u, MID_HALF, (uint32_t)gate_row_bytes);
    pack_row_span(up1, full_model + full_up_off, N_TOTAL_MAX, MID_DIM,
                  MID_HALF, MID_HALF, (uint32_t)gate_row_bytes);
    memcpy(map0, gu0, (size_t)packed_gu_bytes);
    memcpy(map0 + packed_gu_bytes, up0, (size_t)packed_gu_bytes);
    memcpy(map1, gu1, (size_t)packed_gu_bytes);
    memcpy(map1 + packed_gu_bytes, up1, (size_t)packed_gu_bytes);
    pack_k_span(map0 + packed_gu_bytes * 2u,
                full_model + full_down_off, N_TOTAL_MAX, OUT_DIM,
                MID_DIM / QK_K, 0u, half_blocks);
    pack_k_span(map1 + packed_gu_bytes * 2u,
                full_model + full_down_off, N_TOTAL_MAX, OUT_DIM,
                MID_DIM / QK_K, half_blocks, half_blocks);

    const uint64_t full_base = 0u;
    const uint64_t packed0_base = full_model_bytes;
    const uint64_t packed1_base = packed0_base + packed_map_bytes;
    const uint64_t resident_model_bytes = packed1_base + packed_map_bytes;
    unsigned char *resident_model =
        (unsigned char *)malloc((size_t)resident_model_bytes);
    CHECK(resident_model, "timing resident source image");
    memcpy(resident_model + full_base, full_model, (size_t)full_model_bytes);
    memcpy(resident_model + packed0_base, map0, (size_t)packed_map_bytes);
    memcpy(resident_model + packed1_base, map1, (size_t)packed_map_bytes);
    FILE *resident_file = tmpfile();
    CHECK(resident_file, "timing resident backing file");
    CHECK(fwrite(resident_model, 1, (size_t)resident_model_bytes,
                 resident_file) == resident_model_bytes,
          "write timing resident backing file");
    CHECK(fflush(resident_file) == 0, "flush timing resident backing file");
    CHECK(ds4_gpu_set_model_map(resident_model, resident_model_bytes),
          "install timing source map");
    CHECK(ds4_gpu_set_model_fd(fileno(resident_file)),
          "install timing source fd");
    const uint64_t resident_offsets[3] = {
        full_base, packed0_base, packed1_base,
    };
    const uint64_t resident_sizes[3] = {
        full_model_bytes, packed_map_bytes, packed_map_bytes,
    };
    CHECK(ds4_gpu_set_model_map_spans(
              resident_model, resident_model_bytes,
              resident_offsets, resident_sizes, 3u, full_model_bytes),
          "install device-resident timing spans");

    ds4_gpu_tensor weights_full = {}, weights_packed = {};
    CHECK(alloc_tensor(&weights_full, N_USED * sizeof(float)),
          "timing full weights");
    CHECK(alloc_tensor(&weights_packed, N_USED * sizeof(float)),
          "timing packed weights");
    const int32_t routes[N_USED] = {0, 1, 2, 6, 7, 8};
    const float packed_weights[N_USED] =
        {0.31f, 0.23f, 0.17f, 0.13f, 0.09f, 0.07f};
    const float full_weights[N_USED] =
        {0.31f, 0.23f, 0.17f, 0.0f, 0.0f, 0.0f};
    CHECK(upload(selected, routes, sizeof(routes)), "timing routes");
    CHECK(upload(&weights_full, full_weights, sizeof(full_weights)),
          "timing full weights upload");
    CHECK(upload(&weights_packed, packed_weights, sizeof(packed_weights)),
          "timing packed weights upload");

    struct timing_arm arms[3] = {};
    arms[0].name = "full3";
    arms[0].model_map = resident_model;
    arms[0].model_bytes = resident_model_bytes;
    arms[0].gate_off = full_base + full_gate_off;
    arms[0].up_off = full_base + full_up_off;
    arms[0].down_off = full_base + full_down_off;
    arms[0].gate_expert_bytes = full_gate_expert;
    arms[0].gate_row_bytes = gate_row_bytes;
    arms[0].down_expert_bytes = full_down_expert;
    arms[0].down_row_bytes = down_row_bytes;
    arms[0].mid_dim = MID_DIM;
    arms[0].weights = &weights_full;
    arms[1] = arms[0];
    arms[1].name = "packed6-half0";
    arms[1].gate_off = packed0_base;
    arms[1].up_off = packed0_base + packed_gu_bytes;
    arms[1].down_off = packed0_base + packed_gu_bytes * 2u;
    arms[1].gate_expert_bytes = half_gate_expert;
    arms[1].down_expert_bytes = half_down_expert;
    arms[1].down_row_bytes = half_down_row_bytes;
    arms[1].mid_dim = MID_HALF;
    arms[1].weights = &weights_packed;
    arms[2] = arms[1];
    arms[2].name = "packed6-half1";
    arms[2].gate_off = packed1_base;
    arms[2].up_off = packed1_base + packed_gu_bytes;
    arms[2].down_off = packed1_base + packed_gu_bytes * 2u;

    CHECK(setenv("DS4_ROCM_TP_SKIP_UNOWNED", "1", 1) == 0,
          "timing skip unowned");
    CHECK(unsetenv("DS4_ROCM_Q4K_DECODE_STAGE_MIDQ") == 0,
          "timing midq off");
    hipEvent_t start = nullptr, stop = nullptr;
    CHECK(hipEventCreate(&start) == hipSuccess &&
          hipEventCreate(&stop) == hipSuccess, "timing events");
    float ignored = 0.0f;
    for (uint32_t i = 0; i < TIMING_WARMUP; i++) {
        const uint32_t order[3] = {
            (i & 1u) ? 2u : 0u,
            1u,
            (i & 1u) ? 0u : 2u,
        };
        for (uint32_t j = 0; j < 3u; j++) {
            CHECK(time_arm_once(&arms[order[j]], out, gate, up, mid, down,
                                selected, x, start, stop, &ignored) == 0,
                  "timing warmup arm");
        }
    }

    float samples[3][TIMING_SAMPLES];
    for (uint32_t i = 0; i < TIMING_SAMPLES; i++) {
        const uint32_t order[3] = {
            (i & 1u) ? 2u : 0u,
            1u,
            (i & 1u) ? 0u : 2u,
        };
        for (uint32_t j = 0; j < 3u; j++) {
            const uint32_t arm = order[j];
            CHECK(time_arm_once(&arms[arm], out, gate, up, mid, down,
                                selected, x, start, stop,
                                &samples[arm][i]) == 0,
                  "timing sample arm");
        }
    }
    float med[3];
    for (uint32_t arm = 0; arm < 3u; arm++) {
        qsort(samples[arm], TIMING_SAMPLES, sizeof(float),
              compare_float_ascending);
        med[arm] = samples[arm][TIMING_SAMPLES / 2u];
    }

    const float packed_worst = fmaxf(med[1], med[2]);
    const float ratio = packed_worst / med[0];
    const float graph_full3_ms = 0.249f;
    const float critical_full3_ms = 0.358f;
    const float wait33_ms = 0.044f;
    const uint32_t layers = 43u;
    const float graph_delta = fmaxf(0.0f, graph_full3_ms - med[0]);
    const float graph_packed_ms = packed_worst + graph_delta;
    const float critical_packed_ms = graph_packed_ms + wait33_ms;
    const float save_ms =
        (float)layers * (critical_full3_ms - critical_packed_ms);
    const char *decision =
        (ratio <= 1.10f && save_ms >= 2.0f) ? "PASS" :
        (ratio <= 1.10f && save_ms >= 1.8f) ? "CONDITIONAL" :
        (ratio >= 1.25f || save_ms < 1.5f || save_ms <= 0.0f) ? "STOP" :
        "DIAGNOSE";
    const double unique_mib =
        3.0 * (double)(full_gate_expert * 2u + full_down_expert) /
        1048576.0;
    printf("test_rocm_q4k_kshard_shape_cost: full3_ms=%.6f "
           "half0_ms=%.6f half1_ms=%.6f packed_worst_ms=%.6f "
           "ratio=%.6f graph_delta_ms=%.6f graph_packed_ms=%.6f "
           "critical_packed_ms=%.6f layers=%u modeled_save_ms=%.6f "
           "decision=%s samples=%u warmup=%u unique_mib=%.2f "
           "full_mid=%u packed_mid=%u full_down_blocks=%u "
           "packed_down_blocks=%u gate_grid_full=16x6 "
           "gate_grid_packed=8x6 down_grid=128x1\n",
           med[0], med[1], med[2], packed_worst, ratio, graph_delta,
           graph_packed_ms, critical_packed_ms, layers, save_ms, decision,
           TIMING_SAMPLES, TIMING_WARMUP, unique_mib, MID_DIM, MID_HALF,
           MID_DIM / QK_K, MID_HALF / QK_K);

    CHECK(hipEventDestroy(stop) == hipSuccess, "destroy timing stop");
    CHECK(hipEventDestroy(start) == hipSuccess, "destroy timing start");
    ds4_gpu_tensor_free_in_place(&weights_packed);
    ds4_gpu_tensor_free_in_place(&weights_full);
    CHECK(ds4_gpu_set_model_fd(-1), "clear timing source fd");
    CHECK(fclose(resident_file) == 0, "close timing resident backing file");
    free(resident_model);
    free(map1); free(map0); free(up1); free(up0); free(gu1); free(gu0);
    CHECK(ds4_gpu_set_model_map(full_model, full_model_bytes),
          "restore oracle model map");
    return 0;
}

struct oracle_case {
    const char *name;
    uint32_t n_total;
    int32_t route[N_USED];
    float weights[N_USED];
    int skip_unowned;
    int fuse_addend;
};

static int run_case(const struct oracle_case *c,
                    const unsigned char *full_model,
                    uint64_t full_gate_off, uint64_t full_up_off,
                    uint64_t full_down_off, uint64_t full_model_bytes,
                    uint64_t gate_row_bytes, uint64_t down_row_bytes,
                    ds4_gpu_tensor *out, ds4_gpu_tensor *gate,
                    ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
                    ds4_gpu_tensor *down, ds4_gpu_tensor *selected,
                    ds4_gpu_tensor *weights, ds4_gpu_tensor *x,
                    ds4_gpu_tensor *add_in,
                    ds4_gpu_tensor *shared0,
                    ds4_gpu_tensor *shared1) {
    const uint32_t n_total = c->n_total;
    const uint64_t full_gate_expert = MID_DIM * gate_row_bytes;
    const uint64_t full_down_expert = OUT_DIM * down_row_bytes;
    const uint64_t half_gate_expert = MID_HALF * gate_row_bytes;
    const uint64_t half_down_expert = OUT_HALF * down_row_bytes;

    CHECK(upload(selected, c->route, sizeof(c->route)), "route");
    CHECK(upload(weights, c->weights, sizeof(c->weights)), "weights");
    if (c->skip_unowned) {
        CHECK(setenv("DS4_ROCM_TP_SKIP_UNOWNED", "1", 1) == 0, "skip on");
    } else {
        CHECK(unsetenv("DS4_ROCM_TP_SKIP_UNOWNED") == 0, "skip off");
    }
    if (c->fuse_addend) {
        CHECK(setenv("DS4_ROCM_Q4K_DECODE_FUSE_ADDEND", "1", 1) == 0, "fuse on");
    } else {
        CHECK(unsetenv("DS4_ROCM_Q4K_DECODE_FUSE_ADDEND") == 0, "fuse off");
    }

    CHECK(ds4_gpu_set_model_map(full_model, full_model_bytes), "map full");
    CHECK(run_moe(out, gate, up, mid, down, selected, weights, x,
                  c->fuse_addend ? add_in : NULL, full_model, full_model_bytes,
                  full_gate_off, full_up_off, full_down_off,
                  full_gate_expert, gate_row_bytes,
                  full_down_expert, down_row_bytes,
                  MID_DIM, OUT_DIM, n_total), "gold");
    CHECK(hipDeviceSynchronize() == hipSuccess, "sync gold");
    float gold_out[OUT_DIM];
    float gold_mid[N_USED * MID_DIM];
    CHECK(ds4_gpu_tensor_read(out, 0, gold_out, sizeof(gold_out)), "read gold out");
    CHECK(ds4_gpu_tensor_read(mid, 0, gold_mid, sizeof(gold_mid)), "read gold mid");

    const uint64_t packed_gu_bytes = n_total * half_gate_expert;
    const uint64_t packed_dn_bytes = n_total * half_down_expert;
    unsigned char *gu0 = (unsigned char *)malloc((size_t)packed_gu_bytes);
    unsigned char *gu1 = (unsigned char *)malloc((size_t)packed_gu_bytes);
    unsigned char *up0 = (unsigned char *)malloc((size_t)packed_gu_bytes);
    unsigned char *up1 = (unsigned char *)malloc((size_t)packed_gu_bytes);
    unsigned char *dn0 = (unsigned char *)malloc((size_t)packed_dn_bytes);
    unsigned char *dn1 = (unsigned char *)malloc((size_t)packed_dn_bytes);
    CHECK(gu0 && gu1 && up0 && up1 && dn0 && dn1, "pack buffers");
    pack_row_span(gu0, full_model + full_gate_off, n_total, MID_DIM, 0,
                  MID_HALF, (uint32_t)gate_row_bytes);
    pack_row_span(gu1, full_model + full_gate_off, n_total, MID_DIM, MID_HALF,
                  MID_HALF, (uint32_t)gate_row_bytes);
    pack_row_span(up0, full_model + full_up_off, n_total, MID_DIM, 0,
                  MID_HALF, (uint32_t)gate_row_bytes);
    pack_row_span(up1, full_model + full_up_off, n_total, MID_DIM, MID_HALF,
                  MID_HALF, (uint32_t)gate_row_bytes);
    pack_row_span(dn0, full_model + full_down_off, n_total, OUT_DIM, 0,
                  OUT_HALF, (uint32_t)down_row_bytes);
    pack_row_span(dn1, full_model + full_down_off, n_total, OUT_DIM, OUT_HALF,
                  OUT_HALF, (uint32_t)down_row_bytes);

    const uint64_t mid_map_bytes = packed_gu_bytes * 2u + packed_dn_bytes;
    unsigned char *map0 = (unsigned char *)malloc((size_t)mid_map_bytes);
    unsigned char *map1 = (unsigned char *)malloc((size_t)mid_map_bytes);
    CHECK(map0 && map1, "half maps");
    memcpy(map0, gu0, (size_t)packed_gu_bytes);
    memcpy(map0 + packed_gu_bytes, up0, (size_t)packed_gu_bytes);
    memcpy(map0 + packed_gu_bytes * 2u, dn0, (size_t)packed_dn_bytes);
    memcpy(map1, gu1, (size_t)packed_gu_bytes);
    memcpy(map1 + packed_gu_bytes, up1, (size_t)packed_gu_bytes);
    memcpy(map1 + packed_gu_bytes * 2u, dn1, (size_t)packed_dn_bytes);

    float mid0[N_USED * MID_HALF], mid1[N_USED * MID_HALF];
    CHECK(ds4_gpu_set_model_map(map0, mid_map_bytes), "map mid0");
    CHECK(run_moe(out, gate, up, mid, down, selected, weights, x, NULL,
                  map0, mid_map_bytes, 0, packed_gu_bytes, packed_gu_bytes * 2u,
                  half_gate_expert, gate_row_bytes,
                  half_down_expert, down_row_bytes,
                  MID_HALF, OUT_HALF, n_total), "mid0");
    CHECK(hipDeviceSynchronize() == hipSuccess, "sync mid0");
    CHECK(ds4_gpu_tensor_read(mid, 0, mid0, sizeof(mid0)), "read mid0");

    CHECK(ds4_gpu_set_model_map(map1, mid_map_bytes), "map mid1");
    CHECK(run_moe(out, gate, up, mid, down, selected, weights, x, NULL,
                  map1, mid_map_bytes, 0, packed_gu_bytes, packed_gu_bytes * 2u,
                  half_gate_expert, gate_row_bytes,
                  half_down_expert, down_row_bytes,
                  MID_HALF, OUT_HALF, n_total), "mid1");
    CHECK(hipDeviceSynchronize() == hipSuccess, "sync mid1");
    CHECK(ds4_gpu_tensor_read(mid, 0, mid1, sizeof(mid1)), "read mid1");

    float concat_mid[N_USED * MID_DIM];
    for (uint32_t s = 0; s < N_USED; s++) {
        memcpy(concat_mid + s * MID_DIM, mid0 + s * MID_HALF,
               MID_HALF * sizeof(float));
        memcpy(concat_mid + s * MID_DIM + MID_HALF, mid1 + s * MID_HALF,
               MID_HALF * sizeof(float));
    }
    CHECK(memcmp(concat_mid, gold_mid, sizeof(gold_mid)) == 0,
          "concat mid halves vs gold mid");

    /* Full-K down over output-row halves. GU tables stay full-width so the
     * 8-block mid is reconstructed (same bits as concat_mid); down is packed. */
    const uint64_t down_map_bytes =
        n_total * full_gate_expert * 2u + packed_dn_bytes;
    unsigned char *dmap0 = (unsigned char *)malloc((size_t)down_map_bytes);
    unsigned char *dmap1 = (unsigned char *)malloc((size_t)down_map_bytes);
    CHECK(dmap0 && dmap1, "down maps");
    memcpy(dmap0, full_model + full_gate_off, (size_t)(n_total * full_gate_expert));
    memcpy(dmap0 + n_total * full_gate_expert,
           full_model + full_up_off, (size_t)(n_total * full_gate_expert));
    memcpy(dmap0 + n_total * full_gate_expert * 2u, dn0, (size_t)packed_dn_bytes);
    memcpy(dmap1, full_model + full_gate_off, (size_t)(n_total * full_gate_expert));
    memcpy(dmap1 + n_total * full_gate_expert,
           full_model + full_up_off, (size_t)(n_total * full_gate_expert));
    memcpy(dmap1 + n_total * full_gate_expert * 2u, dn1, (size_t)packed_dn_bytes);
    const uint64_t d_up = n_total * full_gate_expert;
    const uint64_t d_dn = d_up * 2u;

    float out0[OUT_HALF], out1[OUT_HALF];
    CHECK(ds4_gpu_set_model_map(dmap0, down_map_bytes), "map down0");
    CHECK(run_moe(out, gate, up, mid, down, selected, weights, x,
                  c->fuse_addend ? add_in : NULL, dmap0, down_map_bytes,
                  0, d_up, d_dn, full_gate_expert, gate_row_bytes,
                  half_down_expert, down_row_bytes,
                  MID_DIM, OUT_HALF, n_total), "down0");
    CHECK(hipDeviceSynchronize() == hipSuccess, "sync down0");
    CHECK(ds4_gpu_tensor_read(out, 0, out0, sizeof(out0)), "read out0");

    ds4_gpu_tensor add_hi = {};
    if (c->fuse_addend) {
        add_hi = *add_in;
        add_hi.ptr = (char *)add_in->ptr + (uint64_t)OUT_HALF * sizeof(float);
        add_hi.bytes = (uint64_t)OUT_HALF * sizeof(float);
        add_hi.owner = 0;
    }
    CHECK(ds4_gpu_set_model_map(dmap1, down_map_bytes), "map down1");
    CHECK(run_moe(out, gate, up, mid, down, selected, weights, x,
                  c->fuse_addend ? &add_hi : NULL, dmap1, down_map_bytes,
                  0, d_up, d_dn, full_gate_expert, gate_row_bytes,
                  half_down_expert, down_row_bytes,
                  MID_DIM, OUT_HALF, n_total), "down1");
    CHECK(hipDeviceSynchronize() == hipSuccess, "sync down1");
    CHECK(ds4_gpu_tensor_read(out, 0, out1, sizeof(out1)), "read out1");

    float concat_out[OUT_DIM];
    memcpy(concat_out, out0, sizeof(out0));
    memcpy(concat_out + OUT_HALF, out1, sizeof(out1));
    CHECK(memcmp(concat_out, gold_out, sizeof(gold_out)) == 0,
          "concat out halves vs gold");

    printf("test_rocm_q4k_ffn_row_balance_oracle: %s PASS\n", c->name);

    /* Lane-B precheck: keep the exact row-independent gate/up halves but
     * partition the down K dimension instead of output rows. This removes
     * the extra MOE_MID exchange required by the exact output-row design,
     * at the cost of a different legal FP32 reduction association. Compare
     * against the actual TP ownership association, including per-rank shared
     * addends, before any production dispatcher or residency change. */
    const uint32_t expert_split = n_total / 2u;
    float rank0_weights[N_USED], rank1_weights[N_USED];
    for (uint32_t slot = 0; slot < N_USED; slot++) {
        const int32_t expert = c->route[slot];
        const bool rank0 = expert >= 0 && (uint32_t)expert < expert_split;
        rank0_weights[slot] = rank0 ? c->weights[slot] : 0.0f;
        rank1_weights[slot] = rank0 ? 0.0f : c->weights[slot];
    }

    float current0[OUT_DIM], current1[OUT_DIM], current_tp[OUT_DIM];
    CHECK(ds4_gpu_set_model_map(full_model, full_model_bytes),
          "map K-shard current");
    CHECK(upload(selected, c->route, sizeof(c->route)),
          "K-shard current route");
    CHECK(upload(weights, rank0_weights, sizeof(rank0_weights)),
          "K-shard current rank0 weights");
    CHECK(run_moe(out, gate, up, mid, down, selected, weights, x, shared0,
                  full_model, full_model_bytes, full_gate_off, full_up_off,
                  full_down_off, full_gate_expert, gate_row_bytes,
                  full_down_expert, down_row_bytes, MID_DIM, OUT_DIM,
                  n_total), "K-shard current rank0");
    CHECK(hipDeviceSynchronize() == hipSuccess &&
          ds4_gpu_tensor_read(out, 0, current0, sizeof(current0)),
          "read K-shard current rank0");
    CHECK(upload(weights, rank1_weights, sizeof(rank1_weights)),
          "K-shard current rank1 weights");
    CHECK(run_moe(out, gate, up, mid, down, selected, weights, x, shared1,
                  full_model, full_model_bytes, full_gate_off, full_up_off,
                  full_down_off, full_gate_expert, gate_row_bytes,
                  full_down_expert, down_row_bytes, MID_DIM, OUT_DIM,
                  n_total), "K-shard current rank1");
    CHECK(hipDeviceSynchronize() == hipSuccess &&
          ds4_gpu_tensor_read(out, 0, current1, sizeof(current1)),
          "read K-shard current rank1");
    CHECK(add_outputs(current_tp, current0, current1),
          "combine K-shard current TP");

    const uint32_t half_blocks = MID_HALF / QK_K;
    const uint64_t k_down_row_bytes =
        (uint64_t)half_blocks * Q4_K_BLOCK_BYTES;
    const uint64_t k_down_expert_bytes = OUT_DIM * k_down_row_bytes;
    const uint64_t k_map_bytes =
        packed_gu_bytes * 2u + n_total * k_down_expert_bytes;
    unsigned char *kmap0 = (unsigned char *)malloc((size_t)k_map_bytes);
    unsigned char *kmap1 = (unsigned char *)malloc((size_t)k_map_bytes);
    CHECK(kmap0 && kmap1, "K-shard maps");
    memcpy(kmap0, gu0, (size_t)packed_gu_bytes);
    memcpy(kmap0 + packed_gu_bytes, up0, (size_t)packed_gu_bytes);
    memcpy(kmap1, gu1, (size_t)packed_gu_bytes);
    memcpy(kmap1 + packed_gu_bytes, up1, (size_t)packed_gu_bytes);
    pack_k_span(kmap0 + packed_gu_bytes * 2u,
                full_model + full_down_off, n_total, OUT_DIM,
                MID_DIM / QK_K, 0u, half_blocks);
    pack_k_span(kmap1 + packed_gu_bytes * 2u,
                full_model + full_down_off, n_total, OUT_DIM,
                MID_DIM / QK_K, half_blocks, half_blocks);

    float candidate0[OUT_DIM], candidate1[OUT_DIM], candidate_tp[OUT_DIM];
    CHECK(upload(weights, c->weights, sizeof(c->weights)),
          "K-shard candidate weights");
    CHECK(ds4_gpu_set_model_map(kmap0, k_map_bytes),
          "map K-shard candidate rank0");
    CHECK(run_moe(out, gate, up, mid, down, selected, weights, x, shared0,
                  kmap0, k_map_bytes, 0u, packed_gu_bytes,
                  packed_gu_bytes * 2u, half_gate_expert, gate_row_bytes,
                  k_down_expert_bytes, k_down_row_bytes, MID_HALF, OUT_DIM,
                  n_total), "K-shard candidate rank0");
    CHECK(hipDeviceSynchronize() == hipSuccess &&
          ds4_gpu_tensor_read(out, 0, candidate0, sizeof(candidate0)),
          "read K-shard candidate rank0");
    CHECK(ds4_gpu_set_model_map(kmap1, k_map_bytes),
          "map K-shard candidate rank1");
    CHECK(run_moe(out, gate, up, mid, down, selected, weights, x, shared1,
                  kmap1, k_map_bytes, 0u, packed_gu_bytes,
                  packed_gu_bytes * 2u, half_gate_expert, gate_row_bytes,
                  k_down_expert_bytes, k_down_row_bytes, MID_HALF, OUT_DIM,
                  n_total), "K-shard candidate rank1");
    CHECK(hipDeviceSynchronize() == hipSuccess &&
          ds4_gpu_tensor_read(out, 0, candidate1, sizeof(candidate1)),
          "read K-shard candidate rank1");
    CHECK(add_outputs(candidate_tp, candidate0, candidate1),
          "combine K-shard candidate TP");

    float abs_errors[OUT_DIM];
    double diff_sq = 0.0, ref_sq = 0.0;
    float max_abs = 0.0f, max_rel = 0.0f;
    uint32_t changed = 0u;
    for (uint32_t row = 0; row < OUT_DIM; row++) {
        const float error = fabsf(candidate_tp[row] - current_tp[row]);
        const float rel = error / fmaxf(1.0f, fabsf(current_tp[row]));
        abs_errors[row] = error;
        if (error > max_abs) max_abs = error;
        if (rel > max_rel) max_rel = rel;
        if (memcmp(&candidate_tp[row], &current_tp[row], sizeof(float)) != 0)
            changed++;
        diff_sq += (double)error * error;
        ref_sq += (double)current_tp[row] * current_tp[row];
    }
    qsort(abs_errors, OUT_DIM, sizeof(abs_errors[0]),
          compare_float_ascending);
    const float p99_abs = abs_errors[(OUT_DIM * 99u) / 100u];
    const double nmse = diff_sq / fmax(1e-30, ref_sq);
    printf("test_rocm_q4k_ffn_kshard_oracle: case=%s "
           "changed=%u/%u max_abs=%.6e p99_abs=%.6e max_rel=%.6e "
           "nmse=%.6e\n",
           c->name, changed, OUT_DIM, max_abs, p99_abs, max_rel, nmse);
    CHECK(max_rel <= 2e-5f && nmse <= 1e-10,
          "K-shard numerical precheck envelope");

    free(kmap1);
    free(kmap0);
    free(gu0); free(gu1); free(up0); free(up1); free(dn0); free(dn1);
    free(map0); free(map1); free(dmap0); free(dmap1);
    return 0;
}

int main(void) {
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm");
    CHECK(setenv("DS4_ROCM_Q4K_DECODE_STAGE_XQ", "1", 1) == 0, "stage XQ");
    CHECK(unsetenv("DS4_ROCM_Q4K_DECODE_STAGE_MIDQ") == 0, "midq off");

    const uint64_t in_blocks = IN_DIM / QK_K;
    const uint64_t mid_blocks = MID_DIM / QK_K;
    const uint64_t gate_row_bytes = in_blocks * Q4_K_BLOCK_BYTES;
    const uint64_t down_row_bytes = mid_blocks * Q4_K_BLOCK_BYTES;
    const uint64_t full_gate_expert = MID_DIM * gate_row_bytes;
    const uint64_t full_down_expert = OUT_DIM * down_row_bytes;
    const uint64_t full_gate_off = 0;
    const uint64_t full_up_off = N_TOTAL_MAX * full_gate_expert;
    const uint64_t full_down_off = full_up_off + N_TOTAL_MAX * full_gate_expert;
    const uint64_t full_model_bytes =
        full_down_off + N_TOTAL_MAX * full_down_expert;

    unsigned char *full_model = (unsigned char *)malloc((size_t)full_model_bytes);
    CHECK(full_model, "full model");
    pack_q4k_table(full_model + full_gate_off, N_TOTAL_MAX, MID_DIM,
                   (uint32_t)in_blocks, 11);
    pack_q4k_table(full_model + full_up_off, N_TOTAL_MAX, MID_DIM,
                   (uint32_t)in_blocks, 37);
    pack_q4k_table(full_model + full_down_off, N_TOTAL_MAX, OUT_DIM,
                   (uint32_t)mid_blocks, 73);

    ds4_gpu_tensor out = {}, gate = {}, up = {}, mid = {}, down = {};
    ds4_gpu_tensor selected = {}, wts = {}, x = {}, add_in = {};
    ds4_gpu_tensor shared0 = {}, shared1 = {};
    CHECK(alloc_tensor(&out, OUT_DIM * sizeof(float)), "out");
    CHECK(alloc_tensor(&gate, (uint64_t)N_USED * MID_DIM * sizeof(float)), "gate");
    CHECK(alloc_tensor(&up, (uint64_t)N_USED * MID_DIM * sizeof(float)), "up");
    CHECK(alloc_tensor(&mid, (uint64_t)N_USED * MID_DIM * sizeof(float)), "mid");
    CHECK(alloc_tensor(&down, (uint64_t)N_USED * OUT_DIM * sizeof(float)), "down");
    CHECK(alloc_tensor(&selected, N_USED * sizeof(int32_t)), "selected");
    CHECK(alloc_tensor(&wts, N_USED * sizeof(float)), "weights");
    CHECK(alloc_tensor(&x, IN_DIM * sizeof(float)), "x");
    CHECK(alloc_tensor(&add_in, OUT_DIM * sizeof(float)), "add");
    CHECK(alloc_tensor(&shared0, OUT_DIM * sizeof(float)), "shared0");
    CHECK(alloc_tensor(&shared1, OUT_DIM * sizeof(float)), "shared1");

    float hx[IN_DIM], hadd[OUT_DIM], hshared0[OUT_DIM], hshared1[OUT_DIM];
    for (uint32_t i = 0; i < IN_DIM; i++)
        hx[i] = (float)((int)(i % 31u) - 15) * 0.03125f;
    for (uint32_t i = 0; i < OUT_DIM; i++)
        hadd[i] = (float)((int)(i % 19u) - 9) * 0.00390625f;
    for (uint32_t i = 0; i < OUT_DIM; i++) {
        hshared0[i] = (float)((int)(i % 23u) - 11) * 0.0029296875f;
        hshared1[i] = (float)((int)(i % 29u) - 14) * 0.001953125f;
    }
    CHECK(upload(&x, hx, sizeof(hx)), "upload x");
    CHECK(upload(&add_in, hadd, sizeof(hadd)), "upload add");
    CHECK(upload(&shared0, hshared0, sizeof(hshared0)), "upload shared0");
    CHECK(upload(&shared1, hshared1, sizeof(hshared1)), "upload shared1");

    if (run_kshard_shape_cost_gate(
            full_model, full_gate_off, full_up_off, full_down_off,
            full_model_bytes, gate_row_bytes, down_row_bytes,
            &out, &gate, &up, &mid, &down, &selected, &x) != 0) {
        return 1;
    }

    const struct oracle_case cases[] = {
        {"6/6", 8,
         {0, 1, 2, 3, 4, 5},
         {0.31f, 0.23f, 0.17f, 0.13f, 0.09f, 0.07f}, 0, 0},
        {"unowned-zero-weight", 8,
         {0, 1, 2, 0, 0, 0},
         {0.31f, 0.23f, 0.17f, 0.0f, 0.0f, 0.0f}, 1, 0},
        {"fused-addend", 8,
         {0, 1, 2, 3, 4, 5},
         {0.31f, 0.23f, 0.17f, 0.13f, 0.09f, 0.07f}, 0, 1},
        {"5/7", 12,
         {0, 1, 2, 5, 8, 11},
         {0.29f, 0.21f, 0.16f, 0.14f, 0.11f, 0.09f}, 0, 0},
        {"6/0-ownership", 12,
         {0, 1, 2, 3, 4, 5},
         {0.27f, 0.22f, 0.18f, 0.14f, 0.11f, 0.08f}, 0, 0},
        {"1/5-ownership", 12,
         {0, 6, 7, 8, 9, 10},
         {0.28f, 0.22f, 0.17f, 0.14f, 0.11f, 0.08f}, 0, 0},
    };
    for (uint32_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        if (run_case(&cases[i], full_model, full_gate_off, full_up_off,
                     full_down_off, full_model_bytes, gate_row_bytes,
                     down_row_bytes, &out, &gate, &up, &mid, &down,
                     &selected, &wts, &x, &add_in,
                     &shared0, &shared1) != 0) {
            return 1;
        }
    }

    ds4_gpu_tensor_free_in_place(&shared1);
    ds4_gpu_tensor_free_in_place(&shared0);
    ds4_gpu_tensor_free_in_place(&add_in);
    ds4_gpu_tensor_free_in_place(&x);
    ds4_gpu_tensor_free_in_place(&wts);
    ds4_gpu_tensor_free_in_place(&selected);
    ds4_gpu_tensor_free_in_place(&down);
    ds4_gpu_tensor_free_in_place(&mid);
    ds4_gpu_tensor_free_in_place(&up);
    ds4_gpu_tensor_free_in_place(&gate);
    ds4_gpu_tensor_free_in_place(&out);
    free(full_model);
    ds4_gpu_cleanup();
    return 0;
}
