/* Production-shape oracle and rotating-input timing gate for the optional
 * gfx1151 indexed sequence-tiled one-token attention research path. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define CHECK(cond, msg) do {                                                \
    if (!(cond)) {                                                           \
        fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);           \
        return 1;                                                            \
    }                                                                        \
} while (0)

typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

enum {
    HEAD_DIM = 512,
    N_HEAD = 32,
    TOP_K = 512,
    RAW_CAP_MAX = 160,
    COMP_CAP = 1024,
};

struct indexed_case {
    const char *name;
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_comp;
    uint32_t pos0;
    uint32_t ratio;
    uint32_t top_k;
    uint32_t topk_pattern;
};

static double rel_rms(const std::vector<float> &a,
                      const std::vector<float> &b) {
    double diff2 = 0.0, ref2 = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = (double)a[i] - b[i];
        diff2 += d * d;
        ref2 += (double)b[i] * b[i];
    }
    return sqrt(diff2 / fmax(ref2, 1.0e-30));
}

static double max_scaled_error(const std::vector<float> &a,
                               const std::vector<float> &b) {
    double max_diff = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        max_diff = fmax(max_diff, fabs((double)a[i] - b[i]));
        max_ref = fmax(max_ref, fabs((double)b[i]));
    }
    return max_diff / fmax(max_ref, 1.0e-30);
}

static void make_topk(std::vector<int32_t> &topk, uint32_t pattern) {
    for (uint32_t i = 0; i < TOP_K; i++) topk[i] = (int32_t)i;
    if (pattern == 1u) {
        /* Duplicate the same selected row on opposite sides of the first
         * compressed tile boundary (logical rows 159/160 with n_raw=128). */
        topk[31] = 17;
        topk[32] = 17;
        topk[100] = -1;
        topk[101] = 511; /* visible_comp - 1 in the production case */
        topk[102] = 512; /* exactly visible_comp: must be filtered */
        topk[103] = 900; /* in n_comp, beyond visibility */
    } else if (pattern == 2u) {
        for (uint32_t i = 0; i < TOP_K; i++) {
            if (i < 100u) topk[i] = (int32_t)i;
            else if (i & 1u) topk[i] = -1;
            else topk[i] = 100 + (int32_t)i; /* all outside n_comp */
        }
    } else if (pattern == 3u) {
        for (uint32_t i = 0; i < TOP_K; i++) {
            topk[i] = (i & 1u) ? -1 : (int32_t)(COMP_CAP + i);
        }
    }
}

static void indexed_reference(std::vector<float> &out,
                              const std::vector<float> &q,
                              const std::vector<float> &raw,
                              const std::vector<float> &comp,
                              const std::vector<int32_t> &topk,
                              const std::vector<float> &sinks,
                              const indexed_case &tc) {
    uint32_t visible_comp = tc.n_comp;
    if (tc.ratio != 0u) {
        visible_comp = (tc.pos0 + 1u) / tc.ratio;
        if (visible_comp > tc.n_comp) visible_comp = tc.n_comp;
    }
    std::vector<uint32_t> selected;
    selected.reserve(tc.top_k);
    for (uint32_t i = 0; i < tc.top_k; i++) {
        const int32_t ci = topk[i];
        if (ci < 0) continue;
        const uint32_t c = (uint32_t)ci;
        if (c < tc.n_comp && c < visible_comp) selected.push_back(c);
    }

    const double scale = 1.0 / sqrt((double)HEAD_DIM);
    std::vector<double> scores((size_t)tc.n_raw + selected.size());
    for (uint32_t h = 0; h < N_HEAD; h++) {
        double max_score = sinks[h];
        for (uint32_t r = 0; r < tc.n_raw; r++) {
            const uint32_t row = (tc.raw_start + r) % tc.raw_cap;
            double dot = 0.0;
            for (uint32_t d = 0; d < HEAD_DIM; d++) {
                dot += (double)q[(size_t)h * HEAD_DIM + d] *
                       raw[(size_t)row * HEAD_DIM + d];
            }
            scores[r] = dot * scale;
            max_score = fmax(max_score, scores[r]);
        }
        for (size_t s = 0; s < selected.size(); s++) {
            const uint32_t row = selected[s];
            double dot = 0.0;
            for (uint32_t d = 0; d < HEAD_DIM; d++) {
                dot += (double)q[(size_t)h * HEAD_DIM + d] *
                       comp[(size_t)row * HEAD_DIM + d];
            }
            scores[(size_t)tc.n_raw + s] = dot * scale;
            max_score = fmax(max_score, scores[(size_t)tc.n_raw + s]);
        }
        double denom = exp((double)sinks[h] - max_score);
        for (double &score : scores) {
            score = exp(score - max_score);
            denom += score;
        }
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            double acc = 0.0;
            for (uint32_t r = 0; r < tc.n_raw; r++) {
                const uint32_t row = (tc.raw_start + r) % tc.raw_cap;
                acc += scores[r] * (double)raw[(size_t)row * HEAD_DIM + d];
            }
            for (size_t s = 0; s < selected.size(); s++) {
                acc += scores[(size_t)tc.n_raw + s] *
                       (double)comp[(size_t)selected[s] * HEAD_DIM + d];
            }
            out[(size_t)h * HEAD_DIM + d] = (float)(acc / denom);
        }
    }
}

static int run_indexed(ds4_gpu_tensor *heads,
                       const std::vector<float> &sinks,
                       const ds4_gpu_tensor *q,
                       const ds4_gpu_tensor *raw,
                       const ds4_gpu_tensor *comp,
                       const ds4_gpu_tensor *topk,
                       const indexed_case &tc) {
    return ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        heads, sinks.data(), sinks.size() * sizeof(float), 0,
        q, raw, comp, 0, topk, 1, tc.pos0,
        tc.n_raw, tc.raw_cap, tc.raw_start, tc.n_comp, tc.top_k,
        128u, tc.ratio, N_HEAD, HEAD_DIM);
}

static int time_active_path(float *usec_per_call,
                            ds4_gpu_tensor *heads,
                            const std::vector<float> &sinks,
                            const ds4_gpu_tensor *q,
                            const std::vector<ds4_gpu_tensor> &raw_ring,
                            const std::vector<ds4_gpu_tensor> &comp_ring,
                            const std::vector<ds4_gpu_tensor> &topk_ring,
                            const indexed_case &tc) {
    constexpr int warmup = 20;
    constexpr int iterations = 400;
    hipEvent_t start = NULL, stop = NULL;
    CHECK(raw_ring.size() == 21u && comp_ring.size() == raw_ring.size() &&
          topk_ring.size() == raw_ring.size(), "complete indexed timing ring");
    for (int i = 0; i < warmup; i++) {
        const size_t slot = (size_t)i % raw_ring.size();
        CHECK(run_indexed(heads, sinks, q, &raw_ring[slot], &comp_ring[slot],
                          &topk_ring[slot], tc), "warm up indexed timing path");
    }
    CHECK(hipDeviceSynchronize() == hipSuccess, "sync indexed warmup");
    CHECK(hipEventCreate(&start) == hipSuccess &&
          hipEventCreate(&stop) == hipSuccess, "create indexed timing events");
    CHECK(hipEventRecord(start, NULL) == hipSuccess, "record indexed start");
    for (int i = 0; i < iterations; i++) {
        const size_t slot = (size_t)i % raw_ring.size();
        CHECK(run_indexed(heads, sinks, q, &raw_ring[slot], &comp_ring[slot],
                          &topk_ring[slot], tc), "run indexed timing path");
    }
    CHECK(hipEventRecord(stop, NULL) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess, "finish indexed timing");
    float elapsed_ms = 0.0f;
    CHECK(hipEventElapsedTime(&elapsed_ms, start, stop) == hipSuccess,
          "read indexed timing");
    CHECK(hipEventDestroy(start) == hipSuccess &&
          hipEventDestroy(stop) == hipSuccess, "destroy indexed timing events");
    *usec_per_call = elapsed_ms * 1000.0f / (float)iterations;
    return 0;
}

int main(void) {
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm backend");
    ds4_gpu_set_quality(false);

    const bool candidate =
        getenv("DS4_ROCM_ATTN_DECODE_INDEXED_SEQTILE_RESEARCH") != NULL;
    std::vector<float> sinks(N_HEAD);
    std::vector<float> q((size_t)N_HEAD * HEAD_DIM);
    std::vector<float> raw((size_t)RAW_CAP_MAX * HEAD_DIM);
    std::vector<float> comp((size_t)COMP_CAP * HEAD_DIM);
    for (uint32_t h = 0; h < N_HEAD; h++) {
        sinks[h] = -0.37f + 0.013f * (float)h;
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            q[(size_t)h * HEAD_DIM + d] =
                0.031f * sinf((float)(h * 47u + d * 11u + 5u) * 0.0091f) +
                0.017f * cosf((float)(h * 13u + d * 3u + 7u) * 0.021f);
        }
    }
    for (uint32_t r = 0; r < RAW_CAP_MAX; r++) {
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            raw[(size_t)r * HEAD_DIM + d] =
                0.23f * sinf((float)(r * 97u + d * 5u + 3u) * 0.007f) +
                0.014f * cosf((float)(r * 19u + d) * 0.031f);
        }
    }
    for (uint32_t c = 0; c < COMP_CAP; c++) {
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            comp[(size_t)c * HEAD_DIM + d] =
                0.19f * cosf((float)(c * 71u + d * 7u + 11u) * 0.0083f) -
                0.011f * sinf((float)(c * 29u + d * 2u) * 0.019f);
        }
    }

    CHECK(ds4_gpu_set_model_map(sinks.data(), sinks.size() * sizeof(float)),
          "install indexed attention sinks");
    ds4_gpu_tensor q_dev = {}, raw_dev = {}, comp_dev = {};
    ds4_gpu_tensor topk_dev = {}, heads_dev = {};
    CHECK(ds4_gpu_tensor_alloc_on(&q_dev, 0, q.size() * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&raw_dev, 0, raw.size() * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&comp_dev, 0, comp.size() * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&topk_dev, 0, TOP_K * sizeof(int32_t)) == 0 &&
          ds4_gpu_tensor_alloc_on(&heads_dev, 0, q.size() * sizeof(float)) == 0,
          "allocate indexed oracle tensors");
    CHECK(ds4_gpu_tensor_write(&q_dev, 0, q.data(), q.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(&raw_dev, 0, raw.data(), raw.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(&comp_dev, 0, comp.data(), comp.size() * sizeof(float)),
          "upload indexed oracle tensors");

    const indexed_case cases[] = {
        {"production-duplicates-visibility", 128u, 160u, 159u, 1024u,
         2047u, 4u, 512u, 1u},
        {"short-comp-invalid-duplicates", 128u, 160u, 151u, 100u,
         511u, 4u, 512u, 2u},
        {"empty-compressed-empty-tiles", 3u, 160u, 159u, 523u,
         0u, 4u, 512u, 3u},
        {"growth-boundary-523", 128u, 160u, 159u, 523u,
         2047u, 4u, 512u, 1u},
        {"partial-topk-511", 128u, 160u, 159u, 523u,
         2047u, 4u, 511u, 0u},
    };
    std::vector<float> got(q.size()), ref(q.size());
    std::vector<int32_t> topk(TOP_K);
    for (const indexed_case &tc : cases) {
        make_topk(topk, tc.topk_pattern);
        CHECK(ds4_gpu_tensor_write(&topk_dev, 0, topk.data(),
                                   topk.size() * sizeof(int32_t)),
              "upload indexed topk case");
        CHECK(run_indexed(&heads_dev, sinks, &q_dev, &raw_dev, &comp_dev,
                          &topk_dev, tc) &&
              ds4_gpu_tensor_read(&heads_dev, 0, got.data(),
                                   got.size() * sizeof(float)),
              "run and read indexed oracle case");
        indexed_reference(ref, q, raw, comp, topk, sinks, tc);
        const double error = rel_rms(got, ref);
        const double max_error = max_scaled_error(got, ref);
        fprintf(stderr,
                "indexed_seqtile_oracle mode=%s case=%s rel=%g max_scaled=%g\n",
                candidate ? "candidate" : "incumbent", tc.name,
                error, max_error);
        CHECK(error <= 2.0e-5 && max_error <= 3.0e-4,
              "indexed attention must stay inside Lane-B oracle envelope");
    }

    const indexed_case timing_case = {
        "production-timing", 128u, 160u, 159u, 523u, 2047u, 4u, 512u, 0u
    };
    make_topk(topk, timing_case.topk_pattern);
    std::vector<ds4_gpu_tensor> raw_ring(21), comp_ring(21), topk_ring(21);
    for (size_t i = 0; i < raw_ring.size(); i++) {
        CHECK(ds4_gpu_tensor_alloc_on(&raw_ring[i], 0,
                                      raw.size() * sizeof(float)) == 0 &&
              ds4_gpu_tensor_alloc_on(&comp_ring[i], 0,
                                      comp.size() * sizeof(float)) == 0 &&
              ds4_gpu_tensor_alloc_on(&topk_ring[i], 0,
                                      topk.size() * sizeof(int32_t)) == 0,
              "allocate indexed rotating input ring");
        CHECK(ds4_gpu_tensor_write(&raw_ring[i], 0, raw.data(),
                                   raw.size() * sizeof(float)) &&
              ds4_gpu_tensor_write(&comp_ring[i], 0, comp.data(),
                                   comp.size() * sizeof(float)) &&
              ds4_gpu_tensor_write(&topk_ring[i], 0, topk.data(),
                                   topk.size() * sizeof(int32_t)),
              "upload indexed rotating input ring");
    }
    float timing_us = 0.0f;
    CHECK(time_active_path(&timing_us, &heads_dev, sinks, &q_dev,
                           raw_ring, comp_ring, topk_ring, timing_case) == 0,
          "time indexed active path");
    fprintf(stderr,
            "indexed_seqtile_timing mode=%s raw=128 selected=512 heads=32 "
            "usec=%.3f\n",
            candidate ? "candidate" : "incumbent", timing_us);

    for (size_t i = 0; i < raw_ring.size(); i++) {
        ds4_gpu_tensor_free_in_place(&raw_ring[i]);
        ds4_gpu_tensor_free_in_place(&comp_ring[i]);
        ds4_gpu_tensor_free_in_place(&topk_ring[i]);
    }
    ds4_gpu_tensor_free_in_place(&q_dev);
    ds4_gpu_tensor_free_in_place(&raw_dev);
    ds4_gpu_tensor_free_in_place(&comp_dev);
    ds4_gpu_tensor_free_in_place(&topk_dev);
    ds4_gpu_tensor_free_in_place(&heads_dev);
    ds4_gpu_cleanup();
    fprintf(stderr, "test_rocm_attention_decode_indexed_seqtile: PASS\n");
    return 0;
}
