/* Mechanism I: FFN output-row balance oracle (Codex gpt-5.6-sol GATE-7).
 *
 * Gold: shipped ds4_gpu_routed_moe_one_tensor (full mid 2048, out 4096).
 * Candidate: independently compute 1024-row mid halves, concatenate, then
 * full-K down on each 2048-row output half and concatenate. No K-shard.
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
    N_USED = 6,
    IN_DIM = 4096,
    MID_DIM = 2048,
    MID_HALF = 1024,
    OUT_DIM = 4096,
    OUT_HALF = 2048,
    N_TOTAL_MAX = 12,
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

static int alloc_tensor(ds4_gpu_tensor *t, uint64_t bytes) {
    memset(t, 0, sizeof(*t));
    return ds4_gpu_tensor_alloc_on(t, 0, bytes) == 0;
}

static int upload(ds4_gpu_tensor *t, const void *src, uint64_t bytes) {
    return ds4_gpu_tensor_write(t, 0, src, bytes) != 0;
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
                    ds4_gpu_tensor *add_in) {
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
    CHECK(alloc_tensor(&out, OUT_DIM * sizeof(float)), "out");
    CHECK(alloc_tensor(&gate, (uint64_t)N_USED * MID_DIM * sizeof(float)), "gate");
    CHECK(alloc_tensor(&up, (uint64_t)N_USED * MID_DIM * sizeof(float)), "up");
    CHECK(alloc_tensor(&mid, (uint64_t)N_USED * MID_DIM * sizeof(float)), "mid");
    CHECK(alloc_tensor(&down, (uint64_t)N_USED * OUT_DIM * sizeof(float)), "down");
    CHECK(alloc_tensor(&selected, N_USED * sizeof(int32_t)), "selected");
    CHECK(alloc_tensor(&wts, N_USED * sizeof(float)), "weights");
    CHECK(alloc_tensor(&x, IN_DIM * sizeof(float)), "x");
    CHECK(alloc_tensor(&add_in, OUT_DIM * sizeof(float)), "add");

    float hx[IN_DIM], hadd[OUT_DIM];
    for (uint32_t i = 0; i < IN_DIM; i++)
        hx[i] = (float)((int)(i % 31u) - 15) * 0.03125f;
    for (uint32_t i = 0; i < OUT_DIM; i++)
        hadd[i] = (float)((int)(i % 19u) - 9) * 0.00390625f;
    CHECK(upload(&x, hx, sizeof(hx)), "upload x");
    CHECK(upload(&add_in, hadd, sizeof(hadd)), "upload add");

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
    };
    for (uint32_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        if (run_case(&cases[i], full_model, full_gate_off, full_up_off,
                     full_down_off, full_model_bytes, gate_row_bytes,
                     down_row_bytes, &out, &gate, &up, &mid, &down,
                     &selected, &wts, &x, &add_in) != 0) {
            return 1;
        }
    }

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
