/* Exactness and production-shape speed gate for the DSpark indexer token loop. */
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
    N_HEAD = 64,
    HEAD_DIM = 128,
    TOP_K = 512,
    MAX_W = 5,
    MAX_COMP = 576,
    WARMUP = 8,
    ITERS = 80,
};

static int alloc_tensor(ds4_gpu_tensor *t, uint64_t bytes) {
    memset(t, 0, sizeof(*t));
    return ds4_gpu_tensor_alloc_on(t, 0, bytes) == 0;
}

static float nonbinary_value(uint64_t i, uint32_t salt) {
    const int32_t a = (int32_t)((i * 1103515245ull + 12345u + salt * 97u) % 2003u);
    const int32_t b = (int32_t)((i * 37u + salt * 19u) % 101u);
    return (float)(a - 1001) / 997.0f + (float)(b - 50) / 100003.0f;
}

static int compare_prefix(const char *label, const float *ref, const float *got,
                          const uint32_t *counts, uint32_t width,
                          uint32_t stride) {
    uint64_t mismatch = 0;
    double max_abs = 0.0;
    for (uint32_t t = 0; t < width; t++) {
        for (uint32_t c = 0; c < counts[t]; c++) {
            const uint64_t i = (uint64_t)t * stride + c;
            if (memcmp(ref + i, got + i, sizeof(float)) != 0) mismatch++;
            const double d = fabs((double)ref[i] - (double)got[i]);
            if (d > max_abs) max_abs = d;
        }
        for (uint32_t c = counts[t]; c < stride; c++) {
            const uint64_t i = (uint64_t)t * stride + c;
            if (!isinf(got[i]) || got[i] >= 0.0f) mismatch++;
        }
    }
    fprintf(stderr, "%s width=%u mismatch=%llu max_abs=%.9g\n", label, width,
            (unsigned long long)mismatch, max_abs);
    return mismatch == 0;
}

static int run_reference(ds4_gpu_tensor *scores,
                         const ds4_gpu_tensor *q,
                         const ds4_gpu_tensor *weights,
                         const ds4_gpu_tensor *index_comp,
                         const uint32_t *counts,
                         uint32_t width,
                         uint32_t stride,
                         float scale) {
    for (uint32_t t = 0; t < width; t++) {
        ds4_gpu_tensor *score_row = ds4_gpu_tensor_view(
            scores, (uint64_t)t * stride * sizeof(float),
            (uint64_t)counts[t] * sizeof(float));
        ds4_gpu_tensor *q_row = ds4_gpu_tensor_view(
            q, (uint64_t)t * N_HEAD * HEAD_DIM * sizeof(float),
            (uint64_t)N_HEAD * HEAD_DIM * sizeof(float));
        ds4_gpu_tensor *weight_row = ds4_gpu_tensor_view(
            weights, (uint64_t)t * N_HEAD * sizeof(float),
            (uint64_t)N_HEAD * sizeof(float));
        const int ok = score_row && q_row && weight_row &&
            ds4_gpu_indexer_score_one_tensor(score_row, q_row, weight_row,
                                             index_comp, counts[t], N_HEAD,
                                             HEAD_DIM, scale);
        ds4_gpu_tensor_free(weight_row);
        ds4_gpu_tensor_free(q_row);
        ds4_gpu_tensor_free(score_row);
        if (!ok) return 0;
    }
    return 1;
}

static int compare_topk_and_mask(ds4_gpu_tensor *ref_scores,
                                 ds4_gpu_tensor *got_scores,
                                 const uint32_t *counts,
                                 uint32_t width,
                                 uint32_t stride) {
    ds4_gpu_tensor ref_selected = {}, got_selected = {};
    ds4_gpu_tensor ref_mask = {}, got_mask = {};
    const uint64_t selected_bytes = (uint64_t)width * TOP_K * sizeof(uint32_t);
    const uint64_t mask_bytes = (uint64_t)width * stride * sizeof(float);
    if (!alloc_tensor(&ref_selected, selected_bytes) ||
        !alloc_tensor(&got_selected, selected_bytes) ||
        !alloc_tensor(&ref_mask, mask_bytes) ||
        !alloc_tensor(&got_mask, mask_bytes)) return 0;
    int ok = 1;
    for (uint32_t t = 0; ok && t < width; t++) {
        if (counts[t] <= TOP_K) continue;
        ds4_gpu_tensor *rs = ds4_gpu_tensor_view(
            ref_scores, (uint64_t)t * stride * sizeof(float),
            (uint64_t)counts[t] * sizeof(float));
        ds4_gpu_tensor *gs = ds4_gpu_tensor_view(
            got_scores, (uint64_t)t * stride * sizeof(float),
            (uint64_t)counts[t] * sizeof(float));
        ds4_gpu_tensor *ri = ds4_gpu_tensor_view(
            &ref_selected, (uint64_t)t * TOP_K * sizeof(uint32_t),
            (uint64_t)TOP_K * sizeof(uint32_t));
        ds4_gpu_tensor *gi = ds4_gpu_tensor_view(
            &got_selected, (uint64_t)t * TOP_K * sizeof(uint32_t),
            (uint64_t)TOP_K * sizeof(uint32_t));
        ds4_gpu_tensor *rm = ds4_gpu_tensor_view(
            &ref_mask, (uint64_t)t * stride * sizeof(float),
            (uint64_t)counts[t] * sizeof(float));
        ds4_gpu_tensor *gm = ds4_gpu_tensor_view(
            &got_mask, (uint64_t)t * stride * sizeof(float),
            (uint64_t)counts[t] * sizeof(float));
        ok = rs && gs && ri && gi && rm && gm &&
             ds4_gpu_indexer_topk_tensor(ri, rs, counts[t], 1u, TOP_K) &&
             ds4_gpu_indexer_topk_tensor(gi, gs, counts[t], 1u, TOP_K) &&
             ds4_gpu_dsv4_topk_mask_tensor(rm, ri, counts[t], 1u, TOP_K) &&
             ds4_gpu_dsv4_topk_mask_tensor(gm, gi, counts[t], 1u, TOP_K);
        ds4_gpu_tensor_free(gm);
        ds4_gpu_tensor_free(rm);
        ds4_gpu_tensor_free(gi);
        ds4_gpu_tensor_free(ri);
        ds4_gpu_tensor_free(gs);
        ds4_gpu_tensor_free(rs);
    }
    if (ok) {
        uint32_t *a = (uint32_t *)malloc(selected_bytes);
        uint32_t *b = (uint32_t *)malloc(selected_bytes);
        float *ma = (float *)malloc(mask_bytes);
        float *mb = (float *)malloc(mask_bytes);
        ok = a && b && ma && mb &&
             ds4_gpu_tensor_read(&ref_selected, 0, a, selected_bytes) &&
             ds4_gpu_tensor_read(&got_selected, 0, b, selected_bytes) &&
             ds4_gpu_tensor_read(&ref_mask, 0, ma, mask_bytes) &&
             ds4_gpu_tensor_read(&got_mask, 0, mb, mask_bytes);
        for (uint32_t t = 0; ok && t < width; t++) {
            if (counts[t] <= TOP_K) continue;
            ok = memcmp(a + (uint64_t)t * TOP_K,
                        b + (uint64_t)t * TOP_K,
                        (uint64_t)TOP_K * sizeof(uint32_t)) == 0 &&
                 memcmp(ma + (uint64_t)t * stride,
                        mb + (uint64_t)t * stride,
                        (uint64_t)counts[t] * sizeof(float)) == 0;
        }
        free(mb); free(ma); free(b); free(a);
    }
    ds4_gpu_tensor_free_in_place(&got_mask);
    ds4_gpu_tensor_free_in_place(&ref_mask);
    ds4_gpu_tensor_free_in_place(&got_selected);
    ds4_gpu_tensor_free_in_place(&ref_selected);
    return ok;
}

static int run_case(ds4_gpu_tensor *ref_scores,
                    ds4_gpu_tensor *got_scores,
                    const ds4_gpu_tensor *q,
                    const ds4_gpu_tensor *weights,
                    const ds4_gpu_tensor *index_comp,
                    const uint32_t *counts,
                    uint32_t width,
                    uint32_t stride,
                    float scale,
                    const char *label) {
    const uint64_t bytes = (uint64_t)width * stride * sizeof(float);
    float *ref = (float *)malloc(bytes);
    float *got = (float *)malloc(bytes);
    if (!ref || !got) return 0;
    int ok = run_reference(ref_scores, q, weights, index_comp, counts,
                           width, stride, scale) &&
             ds4_gpu_indexer_scores_exact_token_loop_tensor(
                 got_scores, q, weights, index_comp, stride, width, counts,
                 N_HEAD, HEAD_DIM, scale) &&
             ds4_gpu_tensor_read(ref_scores, 0, ref, bytes) &&
             ds4_gpu_tensor_read(got_scores, 0, got, bytes) &&
             compare_prefix(label, ref, got, counts, width, stride) &&
             compare_topk_and_mask(ref_scores, got_scores, counts, width,
                                   stride);
    free(got); free(ref);
    return ok;
}

int main(int argc, char **argv) {
    const uint32_t width = argc > 1 ? (uint32_t)atoi(argv[1]) : 5u;
    CHECK(width >= 1u && width <= MAX_W,
          "usage: test_rocm_indexer_exact_token_loop [1-5]");
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm");

    const uint64_t q_count = (uint64_t)MAX_W * N_HEAD * HEAD_DIM;
    const uint64_t weight_count = (uint64_t)MAX_W * N_HEAD;
    const uint64_t comp_count = (uint64_t)MAX_COMP * HEAD_DIM;
    float *q_host = (float *)malloc(q_count * sizeof(float));
    float *weight_host = (float *)malloc(weight_count * sizeof(float));
    float *comp_host = (float *)malloc(comp_count * sizeof(float));
    CHECK(q_host && weight_host && comp_host, "allocate oracle inputs");
    for (uint64_t i = 0; i < q_count; i++) q_host[i] = nonbinary_value(i, 11u);
    for (uint64_t i = 0; i < weight_count; i++)
        weight_host[i] = nonbinary_value(i, 29u) * 0.125f;
    for (uint64_t i = 0; i < comp_count; i++) {
        float v = nonbinary_value(i, 47u);
        /* The first two rows force many tiny signed dots around the ReLU
         * boundary; later rows retain full-mantissa production-scale values. */
        if (i < 2u * HEAD_DIM) v *= 1.0e-7f;
        comp_host[i] = v;
    }

    ds4_gpu_tensor q = {}, weights = {}, comp = {}, ref_scores = {}, got_scores = {};
    CHECK(alloc_tensor(&q, q_count * sizeof(float)), "q tensor");
    CHECK(alloc_tensor(&weights, weight_count * sizeof(float)), "weights tensor");
    CHECK(alloc_tensor(&comp, comp_count * sizeof(float)), "compressed K tensor");
    CHECK(alloc_tensor(&ref_scores, (uint64_t)MAX_W * MAX_COMP * sizeof(float)),
          "reference scores tensor");
    CHECK(alloc_tensor(&got_scores, (uint64_t)MAX_W * MAX_COMP * sizeof(float)),
          "candidate scores tensor");
    CHECK(ds4_gpu_tensor_write(&q, 0, q_host, q_count * sizeof(float)), "upload q");
    CHECK(ds4_gpu_tensor_write(&weights, 0, weight_host,
                               weight_count * sizeof(float)), "upload weights");
    CHECK(ds4_gpu_tensor_write(&comp, 0, comp_host, comp_count * sizeof(float)),
          "upload compressed K");
    const float scale = 1.0f / sqrtf((float)(N_HEAD * HEAD_DIM));

    uint32_t mixed[MAX_W] = {557u, 557u, 558u, 558u, 559u};
    uint32_t flat[MAX_W] = {576u, 576u, 576u, 576u, 576u};
    uint32_t boundary[MAX_W] = {510u, 511u, 512u, 513u, 514u};
    CHECK(run_case(&ref_scores, &got_scores, &q, &weights, &comp,
                   mixed, width, MAX_COMP, scale, "mixed-count"),
          "mixed visibility exactness/top-k");
    CHECK(run_case(&ref_scores, &got_scores, &q, &weights, &comp,
                   flat, width, MAX_COMP, scale, "cap-flat"),
          "cap-flat exactness/top-k");
    CHECK(run_case(&ref_scores, &got_scores, &q, &weights, &comp,
                   boundary, width, MAX_COMP, scale, "topk-boundary"),
          "TOP_K boundary exactness/top-k");

    for (uint32_t i = 0; i < WARMUP; i++) {
        CHECK(run_reference(&ref_scores, &q, &weights, &comp, mixed,
                            width, MAX_COMP, scale), "warm serial");
        CHECK(ds4_gpu_indexer_scores_exact_token_loop_tensor(
                  &got_scores, &q, &weights, &comp, MAX_COMP, width, mixed,
                  N_HEAD, HEAD_DIM, scale), "warm candidate");
    }
    CHECK(hipDeviceSynchronize() == hipSuccess, "finish warmup");
    hipEvent_t start = nullptr, stop = nullptr;
    CHECK(hipEventCreate(&start) == hipSuccess && hipEventCreate(&stop) == hipSuccess,
          "create events");
    CHECK(hipEventRecord(start) == hipSuccess, "serial start");
    for (uint32_t i = 0; i < ITERS; i++)
        CHECK(run_reference(&ref_scores, &q, &weights, &comp, mixed,
                            width, MAX_COMP, scale), "time serial");
    CHECK(hipEventRecord(stop) == hipSuccess && hipEventSynchronize(stop) == hipSuccess,
          "serial stop");
    float serial_ms = 0.0f;
    CHECK(hipEventElapsedTime(&serial_ms, start, stop) == hipSuccess,
          "serial elapsed");
    CHECK(hipEventRecord(start) == hipSuccess, "candidate start");
    for (uint32_t i = 0; i < ITERS; i++)
        CHECK(ds4_gpu_indexer_scores_exact_token_loop_tensor(
                  &got_scores, &q, &weights, &comp, MAX_COMP, width, mixed,
                  N_HEAD, HEAD_DIM, scale), "time candidate");
    CHECK(hipEventRecord(stop) == hipSuccess && hipEventSynchronize(stop) == hipSuccess,
          "candidate stop");
    float candidate_ms = 0.0f;
    CHECK(hipEventElapsedTime(&candidate_ms, start, stop) == hipSuccess,
          "candidate elapsed");
    fprintf(stderr, "width=%u serial_ms=%.6f candidate_ms=%.6f speedup=%.3fx\n",
            width, serial_ms / ITERS, candidate_ms / ITERS,
            serial_ms / candidate_ms);
    CHECK(width == 1u || serial_ms / candidate_ms >= 1.25f,
          "W2-W5 isolated speedup must be at least 1.25x");

    CHECK(hipEventDestroy(stop) == hipSuccess, "destroy stop");
    CHECK(hipEventDestroy(start) == hipSuccess, "destroy start");
    ds4_gpu_tensor_free_in_place(&got_scores);
    ds4_gpu_tensor_free_in_place(&ref_scores);
    ds4_gpu_tensor_free_in_place(&comp);
    ds4_gpu_tensor_free_in_place(&weights);
    ds4_gpu_tensor_free_in_place(&q);
    free(comp_host); free(weight_host); free(q_host);
    ds4_gpu_cleanup();
    fprintf(stderr, "test_rocm_indexer_exact_token_loop: PASS\n");
    return 0;
}
