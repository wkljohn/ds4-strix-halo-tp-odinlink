/* Bitwise and isolated speed gate for exact H=2 DSpark indexed attention. */
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
    N_HEAD = 32,
    HEAD_DIM = 512,
    RAW_CAP = 256,
    MAX_COMP = 576,
    TOP_K = 512,
    MAX_W = 5,
    WARMUP = 4,
    ITERS = 30,
};

static int alloc_tensor(ds4_gpu_tensor *t, uint64_t bytes) {
    memset(t, 0, sizeof(*t));
    return ds4_gpu_tensor_alloc_on(t, 0, bytes) == 0;
}

static float value_at(uint64_t i, uint32_t salt) {
    const int32_t a = (int32_t)((i * 48271u + salt * 131u) % 4001u) - 2000;
    const int32_t b = (int32_t)((i * 73u + salt * 29u) % 211u) - 105;
    return (float)a / 1999.0f + (float)b / 100003.0f;
}

static int run_serial(ds4_gpu_tensor *heads,
                      const ds4_gpu_tensor *q,
                      const ds4_gpu_tensor *raw,
                      const ds4_gpu_tensor *comp,
                      const ds4_gpu_tensor *topk,
                      const float *sinks,
                      const uint32_t *n_raw,
                      const uint32_t *raw_start,
                      const uint32_t *n_comp,
                      const uint32_t *visible,
                      uint32_t raw_cap,
                      uint32_t width) {
    const uint64_t head_row_bytes = (uint64_t)N_HEAD * HEAD_DIM * sizeof(float);
    const uint64_t topk_row_bytes = (uint64_t)TOP_K * sizeof(int32_t);
    for (uint32_t t = 0; t < width; t++) {
        ds4_gpu_tensor *o = ds4_gpu_tensor_view(
            heads, (uint64_t)t * head_row_bytes, head_row_bytes);
        ds4_gpu_tensor *qt = ds4_gpu_tensor_view(
            q, (uint64_t)t * head_row_bytes, head_row_bytes);
        ds4_gpu_tensor *kt = ds4_gpu_tensor_view(
            topk, (uint64_t)t * topk_row_bytes, topk_row_bytes);
        const uint32_t pos = visible[t] * 4u - 1u;
        const int ok = o && qt && kt &&
            ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                o, sinks, (uint64_t)N_HEAD * sizeof(float), 0u, qt, raw,
                comp, 0u, kt, 1u, pos, n_raw[t], raw_cap, raw_start[t],
                n_comp[t], TOP_K, 0u, 4u, N_HEAD, HEAD_DIM);
        ds4_gpu_tensor_free(kt);
        ds4_gpu_tensor_free(qt);
        ds4_gpu_tensor_free(o);
        if (!ok) return 0;
    }
    return 1;
}

static int compare_outputs(const char *label,
                           const float *ref,
                           const float *got,
                           uint32_t width) {
    const uint64_t count = (uint64_t)width * N_HEAD * HEAD_DIM;
    uint64_t mismatch = 0u;
    double max_abs = 0.0;
    for (uint64_t i = 0; i < count; i++) {
        if (memcmp(ref + i, got + i, sizeof(float)) != 0) mismatch++;
        const double d = fabs((double)ref[i] - (double)got[i]);
        if (d > max_abs) max_abs = d;
    }
    fprintf(stderr, "%s width=%u mismatch=%llu/%llu max_abs=%.9g\n",
            label, width, (unsigned long long)mismatch,
            (unsigned long long)count, max_abs);
    return mismatch == 0u;
}

int main(int argc, char **argv) {
    const uint32_t width = argc > 1 ? (uint32_t)atoi(argv[1]) : 5u;
    CHECK(width >= 2u && width <= MAX_W,
          "usage: test_rocm_attention_exact_head2 [2-5]");
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config cfg = {};
    cfg.n_gpus = 1;
    cfg.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&cfg), "initialize ROCm");
    ds4_gpu_set_quality(false);

    const uint64_t q_count = (uint64_t)MAX_W * N_HEAD * HEAD_DIM;
    const uint64_t raw_count = (uint64_t)RAW_CAP * HEAD_DIM;
    const uint64_t comp_count = (uint64_t)MAX_COMP * HEAD_DIM;
    const uint64_t topk_count = (uint64_t)MAX_W * TOP_K;
    const uint64_t out_count = q_count;
    float *q_h = (float *)malloc(q_count * sizeof(float));
    float *raw_h = (float *)malloc(raw_count * sizeof(float));
    float *comp_h = (float *)malloc(comp_count * sizeof(float));
    int32_t *topk_h = (int32_t *)malloc(topk_count * sizeof(int32_t));
    float *ref_h = (float *)malloc(out_count * sizeof(float));
    float *got_h = (float *)malloc(out_count * sizeof(float));
    CHECK(q_h && raw_h && comp_h && topk_h && ref_h && got_h,
          "allocate host buffers");
    float sinks[N_HEAD];
    for (uint32_t h = 0; h < N_HEAD; h++)
        sinks[h] = value_at(h, 3u) * 0.2f;
    for (uint64_t i = 0; i < q_count; i++) q_h[i] = value_at(i, 11u) * 0.08f;
    for (uint64_t i = 0; i < raw_count; i++) raw_h[i] = value_at(i, 23u) * 0.12f;
    for (uint64_t i = 0; i < comp_count; i++) comp_h[i] = value_at(i, 37u) * 0.10f;

    uint32_t n_raw[MAX_W] = {126u, 127u, 128u, 128u, 128u};
    uint32_t raw_start[MAX_W] = {251u, 252u, 253u, 254u, 255u};
    uint32_t n_comp[MAX_W] = {557u, 557u, 558u, 558u, 559u};
    uint32_t visible[MAX_W] = {553u, 554u, 555u, 556u, 557u};
    for (uint32_t t = 0; t < MAX_W; t++) {
        for (uint32_t i = 0; i < TOP_K; i++) {
            int32_t c = (int32_t)((i * 509u + t * 31u + 7u) % visible[t]);
            if ((i % 97u) == 0u) c = -1;
            else if ((i % 89u) == 0u) c = (int32_t)(visible[t] + 3u);
            topk_h[(uint64_t)t * TOP_K + i] = c;
        }
    }

    ds4_gpu_tensor q = {}, raw = {}, comp = {}, topk = {};
    ds4_gpu_tensor serial = {}, candidate = {};
    CHECK(alloc_tensor(&q, q_count * sizeof(float)), "q tensor");
    CHECK(alloc_tensor(&raw, raw_count * sizeof(float)), "raw tensor");
    CHECK(alloc_tensor(&comp, comp_count * sizeof(float)), "comp tensor");
    CHECK(alloc_tensor(&topk, topk_count * sizeof(int32_t)), "topk tensor");
    CHECK(alloc_tensor(&serial, out_count * sizeof(float)), "serial output");
    CHECK(alloc_tensor(&candidate, out_count * sizeof(float)), "candidate output");
    CHECK(ds4_gpu_tensor_write(&q, 0, q_h, q_count * sizeof(float)) &&
          ds4_gpu_tensor_write(&raw, 0, raw_h, raw_count * sizeof(float)) &&
          ds4_gpu_tensor_write(&comp, 0, comp_h, comp_count * sizeof(float)) &&
          ds4_gpu_tensor_write(&topk, 0, topk_h, topk_count * sizeof(int32_t)),
          "upload inputs");
    CHECK(ds4_gpu_set_model_map(sinks, sizeof(sinks)), "install sinks");

    CHECK(run_serial(&serial, &q, &raw, &comp, &topk, sinks, n_raw,
                     raw_start, n_comp, visible, RAW_CAP, width),
          "run serial oracle");
    CHECK(ds4_gpu_attention_indexed_mixed_exact_head2_batch_tensor(
              &candidate, sinks, sizeof(sinks), 0u, &q, &raw, &comp, 0u, &topk,
              width, RAW_CAP, TOP_K, N_HEAD, HEAD_DIM, n_raw, raw_start,
              n_comp, visible), "run H=2 candidate");
    const uint64_t bytes = (uint64_t)width * N_HEAD * HEAD_DIM * sizeof(float);
    CHECK(ds4_gpu_tensor_read(&serial, 0, ref_h, bytes) &&
          ds4_gpu_tensor_read(&candidate, 0, got_h, bytes), "read outputs");
    CHECK(compare_outputs("head2", ref_h, got_h, width),
          "H=2 output must be bit-exact");

    /* Force the wrapper's raw-overwrite safety fallback. */
    uint32_t sat_n_raw[MAX_W] = {126u, 126u, 127u, 127u, 128u};
    uint32_t sat_start[MAX_W] = {123u, 124u, 125u, 126u, 127u};
    CHECK(run_serial(&serial, &q, &raw, &comp, &topk, sinks, sat_n_raw,
                     sat_start, n_comp, visible, 128u, width),
          "run saturated serial oracle");
    CHECK(ds4_gpu_attention_indexed_mixed_exact_head2_batch_tensor(
              &candidate, sinks, sizeof(sinks), 0u, &q, &raw, &comp, 0u, &topk,
              width, 128u, TOP_K, N_HEAD, HEAD_DIM, sat_n_raw, sat_start,
              n_comp, visible), "run saturated fallback");
    CHECK(ds4_gpu_tensor_read(&serial, 0, ref_h, bytes) &&
          ds4_gpu_tensor_read(&candidate, 0, got_h, bytes),
          "read saturated outputs");
    CHECK(compare_outputs("saturated-fallback", ref_h, got_h, width),
          "saturated fallback must be bit-exact");

    for (uint32_t i = 0; i < WARMUP; i++) {
        CHECK(run_serial(&serial, &q, &raw, &comp, &topk, sinks, n_raw,
                         raw_start, n_comp, visible, RAW_CAP, width),
              "warm serial");
        CHECK(ds4_gpu_attention_indexed_mixed_exact_head2_batch_tensor(
                  &candidate, sinks, sizeof(sinks), 0u, &q, &raw, &comp,
                  0u, &topk, width, RAW_CAP, TOP_K, N_HEAD, HEAD_DIM, n_raw,
                  raw_start, n_comp, visible), "warm candidate");
    }
    CHECK(hipDeviceSynchronize() == hipSuccess, "finish warmup");
    hipEvent_t start = nullptr, stop = nullptr;
    CHECK(hipEventCreate(&start) == hipSuccess && hipEventCreate(&stop) == hipSuccess,
          "create timing events");
    CHECK(hipEventRecord(start) == hipSuccess, "serial start");
    for (uint32_t i = 0; i < ITERS; i++)
        CHECK(run_serial(&serial, &q, &raw, &comp, &topk, sinks, n_raw,
                         raw_start, n_comp, visible, RAW_CAP, width),
              "time serial");
    CHECK(hipEventRecord(stop) == hipSuccess && hipEventSynchronize(stop) == hipSuccess,
          "serial stop");
    float serial_ms = 0.0f;
    CHECK(hipEventElapsedTime(&serial_ms, start, stop) == hipSuccess,
          "serial elapsed");
    CHECK(hipEventRecord(start) == hipSuccess, "candidate start");
    for (uint32_t i = 0; i < ITERS; i++)
        CHECK(ds4_gpu_attention_indexed_mixed_exact_head2_batch_tensor(
                  &candidate, sinks, sizeof(sinks), 0u, &q, &raw, &comp,
                  0u, &topk, width, RAW_CAP, TOP_K, N_HEAD, HEAD_DIM, n_raw,
                  raw_start, n_comp, visible), "time candidate");
    CHECK(hipEventRecord(stop) == hipSuccess && hipEventSynchronize(stop) == hipSuccess,
          "candidate stop");
    float candidate_ms = 0.0f;
    CHECK(hipEventElapsedTime(&candidate_ms, start, stop) == hipSuccess,
          "candidate elapsed");
    fprintf(stderr, "width=%u serial_ms=%.6f head2_ms=%.6f speedup=%.3fx\n",
            width, serial_ms / ITERS, candidate_ms / ITERS,
            serial_ms / candidate_ms);
    CHECK(serial_ms / candidate_ms >= 1.0f,
          "unstaged H=2 prototype must not regress any width");

    CHECK(hipEventDestroy(stop) == hipSuccess, "destroy stop");
    CHECK(hipEventDestroy(start) == hipSuccess, "destroy start");
    ds4_gpu_tensor_free_in_place(&candidate);
    ds4_gpu_tensor_free_in_place(&serial);
    ds4_gpu_tensor_free_in_place(&topk);
    ds4_gpu_tensor_free_in_place(&comp);
    ds4_gpu_tensor_free_in_place(&raw);
    ds4_gpu_tensor_free_in_place(&q);
    free(got_h); free(ref_h); free(topk_h); free(comp_h); free(raw_h); free(q_h);
    ds4_gpu_cleanup();
    fprintf(stderr, "test_rocm_attention_exact_head2: PASS\n");
    return 0;
}
