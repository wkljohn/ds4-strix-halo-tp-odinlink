/* DSpark-width Q4_K routed-MoE oracle.
 *
 * Compares one width-2..5 batch invocation with the corresponding calls to
 * the shipped one-token path.  "first-owner" enables the existing expert-
 * grouped tile kernel together with the exact direct-sum6 down fold.  Width 2
 * is the first integration gate; widths 3..5 extend the same arithmetic
 * oracle only after it passes.  This isolated test does not alter production
 * dispatch.
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
    N_TOTAL = 16,
    N_USED = 6,
    N_TOK_MAX = 5,
    IN_DIM = 4096,
    MID_DIM = 2048,
    OUT_DIM = 4096,
    WARMUP = 3,
    ITERS = 12,
};

static void pack_q4k_block(unsigned char *dst, uint32_t seed) {
    dst[0] = 0x00; dst[1] = 0x28; dst[2] = 0x00; dst[3] = 0x00;
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

static ds4_gpu_tensor tensor_view(const ds4_gpu_tensor *src,
                                  uint64_t off, uint64_t bytes) {
    ds4_gpu_tensor v = *src;
    v.ptr = (char *)src->ptr + off;
    v.bytes = bytes;
    v.owner = 0;
    return v;
}

struct buffers {
    ds4_gpu_tensor out, gate, up, mid, down, selected, weights, x;
};

static int run_one(struct buffers *b, const void *model, uint64_t model_bytes,
                   uint64_t gate_off, uint64_t up_off, uint64_t down_off,
                   uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
                   uint64_t down_expert_bytes, uint64_t down_row_bytes,
                   uint32_t tok) {
    ds4_gpu_tensor sel = tensor_view(&b->selected,
        (uint64_t)tok * N_USED * sizeof(int32_t), N_USED * sizeof(int32_t));
    ds4_gpu_tensor w = tensor_view(&b->weights,
        (uint64_t)tok * N_USED * sizeof(float), N_USED * sizeof(float));
    ds4_gpu_tensor x = tensor_view(&b->x,
        (uint64_t)tok * IN_DIM * sizeof(float), IN_DIM * sizeof(float));
    return ds4_gpu_routed_moe_one_tensor(
        &b->out, &b->gate, &b->up, &b->mid, &b->down,
        model, model_bytes, gate_off, up_off, down_off,
        Q4_K_TYPE, Q4_K_TYPE, gate_expert_bytes, gate_row_bytes,
        down_expert_bytes, down_row_bytes, IN_DIM, MID_DIM, OUT_DIM,
        &sel, &w, N_TOTAL, N_USED, 0.0f, &x, NULL, 0, true);
}

static int run_batch(struct buffers *b, const void *model,
                     uint64_t model_bytes, uint64_t gate_off,
                     uint64_t up_off, uint64_t down_off,
                     uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
                     uint64_t down_expert_bytes, uint64_t down_row_bytes,
                     uint32_t n_tok) {
    bool mid_is_f16 = false;
    const int ok = ds4_gpu_routed_moe_batch_tensor(
        &b->out, &b->gate, &b->up, &b->mid, &b->down,
        model, model_bytes, gate_off, up_off, down_off,
        Q4_K_TYPE, Q4_K_TYPE, gate_expert_bytes, gate_row_bytes,
        down_expert_bytes, down_row_bytes, IN_DIM, MID_DIM, OUT_DIM,
        &b->selected, &b->weights, N_TOTAL, N_USED, 0.0f, &b->x,
        0, n_tok, &mid_is_f16, true);
    return ok && !mid_is_f16;
}

static void compare_f32(const char *name, const float *a, const float *b,
                        uint64_t n, uint64_t *mismatch_out,
                        float *max_abs_out) {
    uint64_t mismatch = 0;
    float max_abs = 0.0f;
    for (uint64_t i = 0; i < n; i++) {
        if (memcmp(a + i, b + i, sizeof(float)) != 0) mismatch++;
        const float d = fabsf(a[i] - b[i]);
        if (d > max_abs) max_abs = d;
    }
    printf("%s mismatch=%llu/%llu max_abs=%.9g\n", name,
           (unsigned long long)mismatch, (unsigned long long)n, max_abs);
    *mismatch_out = mismatch;
    *max_abs_out = max_abs;
}

int main(int argc, char **argv) {
    const uint32_t n_tok = argc > 1 ? (uint32_t)atoi(argv[1]) : 2u;
    const char *tile = argc > 2 ? argv[2] : "8";
    const bool first_owner = argc <= 3 || strcmp(argv[3], "first-owner") == 0;
    CHECK(n_tok >= 2u && n_tok <= N_TOK_MAX,
          "usage: test_rocm_q4k_verify_batch_oracle [2-5] [4|8] [serial|first-owner]");
    CHECK(strcmp(tile, "4") == 0 || strcmp(tile, "8") == 0,
          "tile width must be 4 or 8");
    CHECK(argc <= 3 || first_owner || strcmp(argv[3], "serial") == 0,
          "mode must be serial or first-owner");
    CHECK(setenv("DS4_ROCM_Q4K_SORTED_MIN_TOKENS", first_owner ? "2" : "32", 1) == 0,
          "sorted width gate");
    CHECK(setenv("DS4_ROCM_Q4K_VERIFY_FIRST_OWNER",
                 first_owner ? "1" : "0", 1) == 0,
          "first-owner gate");
    CHECK(setenv("DS4_ROCM_DISABLE_Q4K_WMMA", "1", 1) == 0,
          "DP4A isolation");
    CHECK(setenv("DS4_ROCM_Q4K_VERIFY_DIRECT_SUM6", "1", 1) == 0,
          "exact verifier down");
    CHECK(setenv("DS4_ROCM_Q4K_VERIFY_WRITE_AUX", "1", 1) == 0,
          "gate/up localization");
    CHECK(setenv("DS4_ROCM_EXPERT_TILE_M", tile, 1) == 0, "tile width");
    CHECK(unsetenv("DS4_ROCM_TP_SKIP_UNOWNED") == 0, "full local routes");

    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm");
    ds4_gpu_set_q4k_verify_batch_mode(true);

    const uint64_t in_blocks = IN_DIM / QK_K;
    const uint64_t mid_blocks = MID_DIM / QK_K;
    const uint64_t gate_row_bytes = in_blocks * Q4_K_BLOCK_BYTES;
    const uint64_t down_row_bytes = mid_blocks * Q4_K_BLOCK_BYTES;
    const uint64_t gate_expert_bytes = MID_DIM * gate_row_bytes;
    const uint64_t down_expert_bytes = OUT_DIM * down_row_bytes;
    const uint64_t gate_off = 0;
    const uint64_t up_off = N_TOTAL * gate_expert_bytes;
    const uint64_t down_off = up_off + N_TOTAL * gate_expert_bytes;
    const bool swap_gate_up = getenv("DS4_TEST_SWAP_GATE_UP") != NULL;
    const uint64_t test_gate_off = swap_gate_up ? up_off : gate_off;
    const uint64_t test_up_off = swap_gate_up ? gate_off : up_off;
    const uint64_t model_bytes = down_off + N_TOTAL * down_expert_bytes;

    unsigned char *model = (unsigned char *)malloc((size_t)model_bytes);
    CHECK(model, "model allocation");
    pack_q4k_table(model + gate_off, N_TOTAL, MID_DIM,
                   (uint32_t)in_blocks, 11);
    pack_q4k_table(model + up_off, N_TOTAL, MID_DIM,
                   (uint32_t)in_blocks, 37);
    pack_q4k_table(model + down_off, N_TOTAL, OUT_DIM,
                   (uint32_t)mid_blocks, 73);
    CHECK(ds4_gpu_set_model_map(model, model_bytes), "map model");

    struct buffers b = {};
    const uint64_t pairs = (uint64_t)n_tok * N_USED;
    const uint64_t max_pairs = (uint64_t)N_TOK_MAX * N_USED;
    CHECK(alloc_tensor(&b.out, (uint64_t)N_TOK_MAX * OUT_DIM * sizeof(float)), "out");
    CHECK(alloc_tensor(&b.gate, max_pairs * MID_DIM * sizeof(float)), "gate");
    CHECK(alloc_tensor(&b.up, max_pairs * MID_DIM * sizeof(float)), "up");
    CHECK(alloc_tensor(&b.mid, max_pairs * MID_DIM * sizeof(float)), "mid");
    CHECK(alloc_tensor(&b.down, max_pairs * OUT_DIM * sizeof(float)), "down");
    CHECK(alloc_tensor(&b.selected, max_pairs * sizeof(int32_t)), "selected");
    CHECK(alloc_tensor(&b.weights, max_pairs * sizeof(float)), "weights");
    CHECK(alloc_tensor(&b.x, (uint64_t)N_TOK_MAX * IN_DIM * sizeof(float)), "x");

    int32_t routes[N_TOK_MAX][N_USED] = {
        {0, 1, 2, 3, 4, 5},
        {0, 1, 6, 7, 8, 9},
        {2, 3, 6, 7, 10, 11},
        {4, 5, 8, 9, 10, 11},
        {0, 2, 4, 6, 8, 10},
    };
    float hw[N_TOK_MAX][N_USED];
    float *hx = (float *)malloc((size_t)N_TOK_MAX * IN_DIM * sizeof(float));
    CHECK(hx, "activation allocation");
    for (uint32_t t = 0; t < N_TOK_MAX; t++) {
        for (uint32_t s = 0; s < N_USED; s++)
            hw[t][s] = (float)(31u - 3u * s + t) / 127.0f;
        for (uint32_t i = 0; i < IN_DIM; i++)
            hx[(uint64_t)t * IN_DIM + i] =
                (float)((int)((i * 7u + t * 11u) % 61u) - 30) * 0.015625f;
    }
    /* Model the TP remap: peer routes collapse to local expert 0/weight 0,
     * alongside a real live local expert 0. This catches first-owner aliasing
     * and stale-output bugs that clean top-k routes cannot expose. */
    routes[1][N_USED - 1u] = 0;
    hw[1][N_USED - 1u] = 0.0f;
    CHECK(upload(&b.selected, routes, sizeof(routes)), "upload routes");
    CHECK(upload(&b.weights, hw, sizeof(hw)), "upload weights");
    CHECK(upload(&b.x, hx, (uint64_t)N_TOK_MAX * IN_DIM * sizeof(float)),
          "upload activations");

    float *serial_out = (float *)malloc((size_t)n_tok * OUT_DIM * sizeof(float));
    float *serial_mid = (float *)malloc((size_t)pairs * MID_DIM * sizeof(float));
    float *serial_up = (float *)malloc((size_t)pairs * MID_DIM * sizeof(float));
    float *batch_out = (float *)malloc((size_t)n_tok * OUT_DIM * sizeof(float));
    float *batch_mid = (float *)malloc((size_t)pairs * MID_DIM * sizeof(float));
    float *batch_up = (float *)malloc((size_t)pairs * MID_DIM * sizeof(float));
    CHECK(serial_out && serial_mid && serial_up &&
          batch_out && batch_mid && batch_up, "host outputs");

    for (uint32_t t = 0; t < n_tok; t++) {
        CHECK(run_one(&b, model, model_bytes, test_gate_off, test_up_off, down_off,
                      gate_expert_bytes, gate_row_bytes,
                      down_expert_bytes, down_row_bytes, t), "serial oracle");
        CHECK(hipDeviceSynchronize() == hipSuccess, "sync serial oracle");
        CHECK(ds4_gpu_tensor_read(&b.out, 0,
              serial_out + (uint64_t)t * OUT_DIM,
              (uint64_t)OUT_DIM * sizeof(float)), "read serial out");
        CHECK(ds4_gpu_tensor_read(&b.mid, 0,
              serial_mid + (uint64_t)t * N_USED * MID_DIM,
              (uint64_t)N_USED * MID_DIM * sizeof(float)), "read serial mid");
        CHECK(ds4_gpu_tensor_read(&b.up, 0,
              serial_up + (uint64_t)t * N_USED * MID_DIM,
              (uint64_t)N_USED * MID_DIM * sizeof(float)), "read serial up");
    }

    CHECK(run_batch(&b, model, model_bytes, test_gate_off, test_up_off, down_off,
                    gate_expert_bytes, gate_row_bytes,
                    down_expert_bytes, down_row_bytes, n_tok), "batch candidate");
    CHECK(hipDeviceSynchronize() == hipSuccess, "sync batch candidate");
    CHECK(ds4_gpu_tensor_read(&b.out, 0, batch_out,
          (uint64_t)n_tok * OUT_DIM * sizeof(float)), "read batch out");
    CHECK(ds4_gpu_tensor_read(&b.mid, 0, batch_mid,
          pairs * MID_DIM * sizeof(float)), "read batch mid");
    CHECK(ds4_gpu_tensor_read(&b.up, 0, batch_up,
          pairs * MID_DIM * sizeof(float)), "read batch up");

    /* Zero-weight TP-remapped routes are intentionally not projected by the
     * candidate. Their auxiliaries are dead because direct-sum6 skips the
     * slot, but they must be freshly overwritten with canonical +0 rather
     * than stale data. Compare arithmetic only for live routes. */
    uint64_t zero_mismatch = 0;
    const float positive_zero = 0.0f;
    for (uint32_t t = 0; t < n_tok; ++t) {
        for (uint32_t s = 0; s < N_USED; ++s) {
            if (hw[t][s] != 0.0f) continue;
            const uint64_t base = ((uint64_t)t * N_USED + s) * MID_DIM;
            for (uint32_t row = 0; row < MID_DIM; ++row) {
                if (memcmp(batch_mid + base + row, &positive_zero,
                           sizeof(float)) != 0) {
                    ++zero_mismatch;
                }
                serial_mid[base + row] = batch_mid[base + row];
                serial_up[base + row] = batch_up[base + row];
            }
        }
    }
    printf("zero-route noncanonical=%llu\n",
           (unsigned long long)zero_mismatch);

    uint64_t up_mismatch = 0;
    uint64_t mid_mismatch = 0, out_mismatch = 0;
    float up_max = 0.0f;
    float mid_max = 0.0f, out_max = 0.0f;
    compare_f32("up", serial_up, batch_up, pairs * MID_DIM,
                &up_mismatch, &up_max);
    compare_f32("mid", serial_mid, batch_mid, pairs * MID_DIM,
                &mid_mismatch, &mid_max);
    compare_f32("out", serial_out, batch_out,
                (uint64_t)n_tok * OUT_DIM, &out_mismatch, &out_max);

    hipEvent_t start, stop;
    CHECK(hipEventCreate(&start) == hipSuccess &&
          hipEventCreate(&stop) == hipSuccess, "events");
    for (uint32_t i = 0; i < WARMUP; i++) {
        for (uint32_t t = 0; t < n_tok; t++)
            CHECK(run_one(&b, model, model_bytes, test_gate_off, test_up_off, down_off,
                          gate_expert_bytes, gate_row_bytes,
                          down_expert_bytes, down_row_bytes, t), "warm serial");
    }
    CHECK(hipDeviceSynchronize() == hipSuccess, "warm serial sync");
    CHECK(hipEventRecord(start) == hipSuccess, "serial start");
    for (uint32_t i = 0; i < ITERS; i++) {
        for (uint32_t t = 0; t < n_tok; t++)
            CHECK(run_one(&b, model, model_bytes, test_gate_off, test_up_off, down_off,
                          gate_expert_bytes, gate_row_bytes,
                          down_expert_bytes, down_row_bytes, t), "timed serial");
    }
    CHECK(hipEventRecord(stop) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess, "serial stop");
    float serial_ms = 0.0f;
    CHECK(hipEventElapsedTime(&serial_ms, start, stop) == hipSuccess,
          "serial elapsed");

    for (uint32_t i = 0; i < WARMUP; i++)
        CHECK(run_batch(&b, model, model_bytes, test_gate_off, test_up_off, down_off,
                        gate_expert_bytes, gate_row_bytes,
                        down_expert_bytes, down_row_bytes, n_tok), "warm batch");
    CHECK(hipDeviceSynchronize() == hipSuccess, "warm batch sync");
    CHECK(hipEventRecord(start) == hipSuccess, "batch start");
    for (uint32_t i = 0; i < ITERS; i++)
        CHECK(run_batch(&b, model, model_bytes, test_gate_off, test_up_off, down_off,
                        gate_expert_bytes, gate_row_bytes,
                        down_expert_bytes, down_row_bytes, n_tok), "timed batch");
    CHECK(hipEventRecord(stop) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess, "batch stop");
    float batch_ms = 0.0f;
    CHECK(hipEventElapsedTime(&batch_ms, start, stop) == hipSuccess,
          "batch elapsed");
    printf("mode=%s width=%u tile=%s serial_ms=%.3f batch_ms=%.3f speedup=%.3fx\n",
           first_owner ? "first-owner" : "serial", n_tok, tile,
           serial_ms / ITERS, batch_ms / ITERS,
           serial_ms / batch_ms);

    CHECK(hipEventDestroy(stop) == hipSuccess, "destroy stop event");
    CHECK(hipEventDestroy(start) == hipSuccess, "destroy start event");
    free(batch_up); free(batch_mid); free(batch_out);
    free(serial_up); free(serial_mid); free(serial_out);
    free(hx); free(model);
    ds4_gpu_tensor_free_in_place(&b.x);
    ds4_gpu_tensor_free_in_place(&b.weights);
    ds4_gpu_tensor_free_in_place(&b.selected);
    ds4_gpu_tensor_free_in_place(&b.down);
    ds4_gpu_tensor_free_in_place(&b.mid);
    ds4_gpu_tensor_free_in_place(&b.up);
    ds4_gpu_tensor_free_in_place(&b.gate);
    ds4_gpu_tensor_free_in_place(&b.out);
    ds4_gpu_cleanup();

    CHECK(up_mismatch == 0, "batch up must be bit-exact to serial rows");
    CHECK(zero_mismatch == 0, "zero-weight routes must be canonical +0");
    CHECK(mid_mismatch == 0, "batch mid must be bit-exact to serial rows");
    CHECK(out_mismatch == 0, "batch out must be bit-exact to serial rows");
    printf("test_rocm_q4k_verify_batch_oracle: PASS\n");
    return 0;
}
