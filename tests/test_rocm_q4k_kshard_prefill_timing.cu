/* H8 packed Q4_K prefill-shape timing oracle.
 *
 * A profiling-only production run freezes all 43 routed layers' top-6 expert
 * IDs and weights.  The incumbent arms preserve six slots and apply the exact
 * TP remap for rank 0/rank 1: owned IDs are rebased onto 128 experts and peer
 * routes become negative sentinels for sorted-prefill omission.  The H8 arm
 * keeps the same six global routes over 256 resident half-K experts.
 *
 * hipEvent windows include the complete 43-layer dispatch sequence. Model
 * copies, layout construction, route uploads, and allocation are outside the
 * window. A separate one-token control uses production's expert-0/zero-weight
 * tail representation; a fictitious 2049-token launch is never measured.
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
extern "C" int ds4_gpu_routed_moe_batch_q4k_direct_control(
        ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *,
        ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, const void *,
        const void *, uint64_t, uint64_t, uint64_t, uint64_t,
        const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t,
        float, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t);

enum {
    IN_DIM = 4096,
    MID_FULL = 2048,
    MID_HALF = 1024,
    OUT_DIM = 4096,
    N_TOTAL_FULL = 128,
    N_TOTAL_HALF = 256,
    N_USED = 6,
    N_LAYERS = 43,
    PREFILL_TOKENS = 2048,
    QK_K = 256,
    Q4_K_BYTES = 144,
    WARMUP = 8,
    SAMPLES = 31,
};

static constexpr double BASELINE_TPS = 272.62;
static constexpr double PASS_TPS = BASELINE_TPS * 0.99;
static constexpr double DIAGNOSE_TPS = BASELINE_TPS * 0.98;
static constexpr double MAX_IQR_FRAC = 0.03;

struct route_set {
    ds4_gpu_tensor selected[N_LAYERS];
    ds4_gpu_tensor weights[N_LAYERS];
    ds4_gpu_tensor tail_selected;
    ds4_gpu_tensor tail_weights;
};

struct arm {
    const char *name;
    const void *gate_w;
    const void *up_w;
    const void *down_w;
    uint64_t gate_expert_bytes;
    uint64_t gate_row_bytes;
    uint64_t down_expert_bytes;
    uint64_t down_row_bytes;
    uint32_t n_total;
    uint32_t mid_dim;
    const route_set *routes;
};

struct workspace {
    ds4_gpu_tensor out;
    ds4_gpu_tensor gate;
    ds4_gpu_tensor up;
    ds4_gpu_tensor mid;
    ds4_gpu_tensor down;
    ds4_gpu_tensor x;
    uint32_t n_tokens;
};

struct stats {
    float min;
    float q1;
    float median;
    float q3;
    float max;
    float iqr_frac;
};

static int alloc_tensor(ds4_gpu_tensor *tensor, uint64_t bytes) {
    memset(tensor, 0, sizeof(*tensor));
    return ds4_gpu_tensor_alloc_on(tensor, 0, bytes) == 0;
}

static int alloc_upload(ds4_gpu_tensor *tensor, const void *src,
                        uint64_t bytes) {
    return alloc_tensor(tensor, bytes) &&
           ds4_gpu_tensor_write(tensor, 0, src, bytes) != 0;
}

static int compare_float(const void *a, const void *b) {
    const float av = *(const float *)a;
    const float bv = *(const float *)b;
    return av < bv ? -1 : av > bv ? 1 : 0;
}

static struct stats summarize(float values[SAMPLES]) {
    qsort(values, SAMPLES, sizeof(values[0]), compare_float);
    struct stats s = {};
    s.min = values[0];
    s.q1 = values[SAMPLES / 4];
    s.median = values[SAMPLES / 2];
    s.q3 = values[(3 * SAMPLES) / 4];
    s.max = values[SAMPLES - 1];
    s.iqr_frac = s.median > 0.0f ? (s.q3 - s.q1) / s.median : INFINITY;
    return s;
}

static int read_exact(const char *path, void *dst, size_t bytes) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;
    const size_t got = fread(dst, 1, bytes, fp);
    const int extra = fgetc(fp);
    const int close_rc = fclose(fp);
    return got == bytes && extra == EOF && close_rc == 0;
}

static int routes_load(const char *prefix,
                       route_set *rank0, route_set *rank1,
                       route_set *half) {
    const size_t pairs = (size_t)PREFILL_TOKENS * N_USED;
    int32_t *global_ids = (int32_t *)malloc(pairs * sizeof(global_ids[0]));
    float *global_weights = (float *)malloc(pairs * sizeof(global_weights[0]));
    int32_t *ids0 = (int32_t *)malloc(pairs * sizeof(ids0[0]));
    int32_t *ids1 = (int32_t *)malloc(pairs * sizeof(ids1[0]));
    float *weights0 = (float *)malloc(pairs * sizeof(weights0[0]));
    float *weights1 = (float *)malloc(pairs * sizeof(weights1[0]));
    CHECK(global_ids && global_weights && ids0 && ids1 && weights0 && weights1,
          "route host buffers");

    uint64_t owned_hist[7] = {};
    uint64_t duplicate_rows = 0;
    for (uint32_t il = 0; il < N_LAYERS; ++il) {
        char ids_path[1024];
        char weights_path[1024];
        snprintf(ids_path, sizeof(ids_path),
                 "%s_layer%u_pos0_topk.i32", prefix, il);
        snprintf(weights_path, sizeof(weights_path),
                 "%s_layer%u_pos0_weights.f32", prefix, il);
        CHECK(read_exact(ids_path, global_ids,
                         pairs * sizeof(global_ids[0])), "read frozen route IDs");
        CHECK(read_exact(weights_path, global_weights,
                         pairs * sizeof(global_weights[0])), "read frozen route weights");

        for (uint32_t t = 0; t < PREFILL_TOKENS; ++t) {
            uint32_t owned0 = 0;
            for (uint32_t j = 0; j < N_USED; ++j) {
                const size_t p = (size_t)t * N_USED + j;
                const int32_t expert = global_ids[p];
                CHECK(expert >= 0 && expert < N_TOTAL_HALF,
                      "frozen expert ID range");
                CHECK(isfinite(global_weights[p]), "finite frozen route weight");
                for (uint32_t k = 0; k < j; ++k) {
                    if (global_ids[(size_t)t * N_USED + k] == expert) {
                        duplicate_rows++;
                    }
                }
                const bool own0 = expert < N_TOTAL_FULL;
                const bool own1 = !own0;
                ids0[p] = own0 ? expert : -1;
                ids1[p] = own1 ? expert - N_TOTAL_FULL : -1;
                weights0[p] = own0 ? global_weights[p] : 0.0f;
                weights1[p] = own1 ? global_weights[p] : 0.0f;
                owned0 += own0 ? 1u : 0u;
            }
            owned_hist[owned0]++;
        }
        CHECK(duplicate_rows == 0, "no duplicate experts per frozen token");
        CHECK(alloc_upload(&rank0->selected[il], ids0,
                           pairs * sizeof(ids0[0])) &&
              alloc_upload(&rank0->weights[il], weights0,
                           pairs * sizeof(weights0[0])) &&
              alloc_upload(&rank1->selected[il], ids1,
                           pairs * sizeof(ids1[0])) &&
              alloc_upload(&rank1->weights[il], weights1,
                           pairs * sizeof(weights1[0])) &&
              alloc_upload(&half->selected[il], global_ids,
                           pairs * sizeof(global_ids[0])) &&
              alloc_upload(&half->weights[il], global_weights,
                           pairs * sizeof(global_weights[0])),
              "upload frozen routes");

        if (il == 0u) {
            int32_t tail0[N_USED], tail1[N_USED], tailh[N_USED];
            float tailw0[N_USED], tailw1[N_USED], tailwh[N_USED];
            for (uint32_t j = 0; j < N_USED; ++j) {
                const int32_t expert = global_ids[j];
                const bool own0 = expert < N_TOTAL_FULL;
                tail0[j] = own0 ? expert : 0;
                tail1[j] = own0 ? 0 : expert - N_TOTAL_FULL;
                tailh[j] = expert;
                tailw0[j] = own0 ? global_weights[j] : 0.0f;
                tailw1[j] = own0 ? 0.0f : global_weights[j];
                tailwh[j] = global_weights[j];
            }
            CHECK(alloc_upload(&rank0->tail_selected, tail0, sizeof(tail0)) &&
                  alloc_upload(&rank0->tail_weights, tailw0, sizeof(tailw0)) &&
                  alloc_upload(&rank1->tail_selected, tail1, sizeof(tail1)) &&
                  alloc_upload(&rank1->tail_weights, tailw1, sizeof(tailw1)) &&
                  alloc_upload(&half->tail_selected, tailh, sizeof(tailh)) &&
                  alloc_upload(&half->tail_weights, tailwh, sizeof(tailwh)),
                  "upload one-token tail routes");
        }
    }

    printf("test_rocm_q4k_kshard_prefill_routes: layers=%u tokens=%u "
           "owned_hist=[%llu,%llu,%llu,%llu,%llu,%llu,%llu]\n",
           N_LAYERS, PREFILL_TOKENS,
           (unsigned long long)owned_hist[0],
           (unsigned long long)owned_hist[1],
           (unsigned long long)owned_hist[2],
           (unsigned long long)owned_hist[3],
           (unsigned long long)owned_hist[4],
           (unsigned long long)owned_hist[5],
           (unsigned long long)owned_hist[6]);
    free(weights1);
    free(weights0);
    free(ids1);
    free(ids0);
    free(global_weights);
    free(global_ids);
    return 0;
}

static void routes_free(route_set *routes) {
    ds4_gpu_tensor_free_in_place(&routes->tail_weights);
    ds4_gpu_tensor_free_in_place(&routes->tail_selected);
    for (uint32_t il = N_LAYERS; il-- > 0;) {
        ds4_gpu_tensor_free_in_place(&routes->weights[il]);
        ds4_gpu_tensor_free_in_place(&routes->selected[il]);
    }
}

__global__ static void fill_input(float *x, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) x[i] = (float)((int)(i % 37u) - 18) * 0.015625f;
}

static int workspace_init(workspace *w, uint32_t n_tokens) {
    memset(w, 0, sizeof(*w));
    w->n_tokens = n_tokens;
    const uint64_t pairs = (uint64_t)n_tokens * N_USED;
    CHECK(alloc_tensor(&w->out, (uint64_t)n_tokens * OUT_DIM * sizeof(float)),
          "timing out");
    CHECK(alloc_tensor(&w->gate, pairs * MID_FULL * sizeof(float)),
          "timing gate");
    CHECK(alloc_tensor(&w->up, pairs * MID_FULL * sizeof(float)),
          "timing up");
    CHECK(alloc_tensor(&w->mid, pairs * MID_FULL * sizeof(float)),
          "timing mid");
    CHECK(alloc_tensor(&w->down, pairs * OUT_DIM * sizeof(float)),
          "timing down");
    CHECK(alloc_tensor(&w->x, (uint64_t)n_tokens * IN_DIM * sizeof(float)),
          "timing x");
    const uint64_t x_count = (uint64_t)n_tokens * IN_DIM;
    fill_input<<<(x_count + 255u) / 256u, 256>>>((float *)w->x.ptr, x_count);
    CHECK(hipGetLastError() == hipSuccess && hipDeviceSynchronize() == hipSuccess,
          "timing input fill");
    return 0;
}

static void workspace_free(workspace *w) {
    ds4_gpu_tensor_free_in_place(&w->x);
    ds4_gpu_tensor_free_in_place(&w->down);
    ds4_gpu_tensor_free_in_place(&w->mid);
    ds4_gpu_tensor_free_in_place(&w->up);
    ds4_gpu_tensor_free_in_place(&w->gate);
    ds4_gpu_tensor_free_in_place(&w->out);
}

static int launch_one(const arm *a, workspace *w, uint32_t il,
                      bool tail, float clamp) {
    const ds4_gpu_tensor *selected = tail ? &a->routes->tail_selected
                                          : &a->routes->selected[il];
    const ds4_gpu_tensor *weights = tail ? &a->routes->tail_weights
                                         : &a->routes->weights[il];
    return ds4_gpu_routed_moe_batch_q4k_direct_control(
        &w->out, &w->gate, &w->up, &w->mid, &w->down,
        a->gate_w, a->up_w, a->down_w,
        a->gate_expert_bytes, a->gate_row_bytes,
        a->down_expert_bytes, a->down_row_bytes,
        selected, weights, a->n_total, N_USED, clamp,
        &w->x, il, w->n_tokens, a->mid_dim);
}

static int time_arm(const arm *a, workspace *w, bool tail, float clamp,
                    hipEvent_t start, hipEvent_t stop, float *elapsed_ms) {
    CHECK(hipEventRecord(start) == hipSuccess, "timing start");
    const uint32_t launches = tail ? 1u : N_LAYERS;
    for (uint32_t il = 0; il < launches; ++il) {
        CHECK(launch_one(a, w, il, tail, clamp), a->name);
    }
    CHECK(hipEventRecord(stop) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess, "timing stop");
    CHECK(hipEventElapsedTime(elapsed_ms, start, stop) == hipSuccess,
          "timing elapsed");
    return 0;
}

static int measure(const arm arms[3], uint32_t n_tokens, bool tail,
                   float clamp, stats out_stats[3]) {
    workspace w = {};
    CHECK(workspace_init(&w, n_tokens) == 0, "timing workspace");
    hipEvent_t start = nullptr, stop = nullptr;
    CHECK(hipEventCreate(&start) == hipSuccess &&
          hipEventCreate(&stop) == hipSuccess, "timing events");
    float ignored = 0.0f;
    for (uint32_t i = 0; i < WARMUP; ++i) {
        for (uint32_t j = 0; j < 3u; ++j) {
            const uint32_t arm_i = (i + j) % 3u;
            CHECK(time_arm(&arms[arm_i], &w, tail, clamp,
                           start, stop, &ignored) == 0, "timing warmup");
        }
    }
    float samples[3][SAMPLES];
    for (uint32_t i = 0; i < SAMPLES; ++i) {
        for (uint32_t j = 0; j < 3u; ++j) {
            const uint32_t arm_i = (i + j) % 3u;
            CHECK(time_arm(&arms[arm_i], &w, tail, clamp,
                           start, stop, &samples[arm_i][i]) == 0,
                  "timing sample");
        }
    }
    for (uint32_t i = 0; i < 3u; ++i) out_stats[i] = summarize(samples[i]);
    CHECK(hipEventDestroy(stop) == hipSuccess, "destroy stop event");
    CHECK(hipEventDestroy(start) == hipSuccess, "destroy start event");
    workspace_free(&w);
    return 0;
}

static void print_stats(const char *shape, const arm *a, const stats *s) {
    printf("test_rocm_q4k_kshard_prefill_timing: shape=%s arm=%s "
           "min_ms=%.6f q1_ms=%.6f median_ms=%.6f q3_ms=%.6f max_ms=%.6f "
           "iqr_frac=%.6f warmup=%u samples=%u\n",
           shape, a->name, s->min, s->q1, s->median, s->q3, s->max,
           s->iqr_frac, WARMUP, SAMPLES);
}

int main(void) {
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    const char *route_prefix = getenv("DS4_TEST_Q4K_ROUTE_CAPTURE_PREFIX");
    CHECK(route_prefix && route_prefix[0], "frozen route prefix required");
    CHECK(setenv("DS4_ROCM_Q4K_KSHARD_RESEARCH", "1", 1) == 0 &&
          setenv("DS4_ROCM_Q4K_WMMA_PAIR_GATE_UP", "1", 1) == 0 &&
          setenv("DS4_ROCM_Q4K_WMMA_FUSE_MID", "1", 1) == 0 &&
          setenv("DS4_ROCM_TP_PREFILL_SKIP_UNOWNED", "1", 1) == 0,
          "enable production timing controls");
    CHECK(unsetenv("DS4_ROCM_Q4K_WMMA_LAYER_LOG") == 0,
          "disable intrusive layer logging");
    const char *clamp_env = getenv("DS4_TEST_Q4K_CLAMP");
    const float clamp = clamp_env ? strtof(clamp_env, NULL) : 10.0f;

    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm");

    route_set rank0_routes = {}, rank1_routes = {}, half_routes = {};
    CHECK(routes_load(route_prefix, &rank0_routes, &rank1_routes,
                      &half_routes) == 0, "load frozen production routes");

    const uint64_t gate_row_bytes = 16u * Q4_K_BYTES;
    const uint64_t full_gate_expert = MID_FULL * gate_row_bytes;
    const uint64_t half_gate_expert = MID_HALF * gate_row_bytes;
    const uint64_t full_down_row_bytes = 8u * Q4_K_BYTES;
    const uint64_t half_down_row_bytes = 4u * Q4_K_BYTES;
    const uint64_t full_down_expert = OUT_DIM * full_down_row_bytes;
    const uint64_t half_down_expert = OUT_DIM * half_down_row_bytes;
    const uint64_t full_gate_bytes = N_TOTAL_FULL * full_gate_expert;
    const uint64_t full_down_bytes = N_TOTAL_FULL * full_down_expert;
    const uint64_t half_gate_bytes = N_TOTAL_HALF * half_gate_expert;
    const uint64_t half_down_bytes = N_TOTAL_HALF * half_down_expert;

    void *full_gate_w = NULL, *full_up_w = NULL, *full_down_w = NULL;
    void *half_gate_w = NULL, *half_up_w = NULL, *half_down_w = NULL;
    CHECK(hipMalloc(&full_gate_w, full_gate_bytes) == hipSuccess &&
          hipMalloc(&full_up_w, full_gate_bytes) == hipSuccess &&
          hipMalloc(&full_down_w, full_down_bytes) == hipSuccess &&
          hipMalloc(&half_gate_w, half_gate_bytes) == hipSuccess &&
          hipMalloc(&half_up_w, half_gate_bytes) == hipSuccess &&
          hipMalloc(&half_down_w, half_down_bytes) == hipSuccess,
          "resident timing weights");
    CHECK(hipMemset(full_gate_w, 0x11, full_gate_bytes) == hipSuccess &&
          hipMemset(full_up_w, 0x13, full_gate_bytes) == hipSuccess &&
          hipMemset(full_down_w, 0x17, full_down_bytes) == hipSuccess &&
          hipMemset(half_gate_w, 0x11, half_gate_bytes) == hipSuccess &&
          hipMemset(half_up_w, 0x13, half_gate_bytes) == hipSuccess &&
          hipMemset(half_down_w, 0x17, half_down_bytes) == hipSuccess &&
          hipDeviceSynchronize() == hipSuccess,
          "initialize resident timing weights");

    arm arms[3] = {
        {"full128-rank0-sixslot", full_gate_w, full_up_w, full_down_w,
         full_gate_expert, gate_row_bytes, full_down_expert,
         full_down_row_bytes, N_TOTAL_FULL, MID_FULL, &rank0_routes},
        {"full128-rank1-sixslot", full_gate_w, full_up_w, full_down_w,
         full_gate_expert, gate_row_bytes, full_down_expert,
         full_down_row_bytes, N_TOTAL_FULL, MID_FULL, &rank1_routes},
        {"half256-sixslot", half_gate_w, half_up_w, half_down_w,
         half_gate_expert, gate_row_bytes, half_down_expert,
         half_down_row_bytes, N_TOTAL_HALF, MID_HALF, &half_routes},
    };

    stats prefill_stats[3] = {};
    CHECK(measure(arms, PREFILL_TOKENS, false, clamp, prefill_stats) == 0,
          "measure 2048-token production sequence");
    for (uint32_t i = 0; i < 3u; ++i) print_stats("2048x43", &arms[i], &prefill_stats[i]);

    const double incumbent_ms = fmax(prefill_stats[0].median,
                                     prefill_stats[1].median);
    const double candidate_ms = prefill_stats[2].median;
    const double delta_ms = candidate_ms - incumbent_ms;
    const double baseline_ms = 1000.0 * PREFILL_TOKENS / BASELINE_TPS;
    const double modeled_ms = baseline_ms + delta_ms;
    const double modeled_tps = 1000.0 * PREFILL_TOKENS / modeled_ms;
    const bool noisy = prefill_stats[0].iqr_frac > MAX_IQR_FRAC ||
                       prefill_stats[1].iqr_frac > MAX_IQR_FRAC ||
                       prefill_stats[2].iqr_frac > MAX_IQR_FRAC;
    const char *decision = noisy ? "INVALID_NOISE" :
        modeled_tps >= PASS_TPS ? "PASS" :
        modeled_tps >= DIAGNOSE_TPS ? "DIAGNOSE" : "STOP";
    printf("test_rocm_q4k_kshard_prefill_gate: baseline_tps=%.6f "
           "baseline_ms=%.6f incumbent_critical_ms=%.6f candidate_ms=%.6f "
           "delta_ms=%.6f modeled_tps=%.6f pass_tps=%.6f "
           "diagnose_tps=%.6f max_iqr_frac=%.6f decision=%s\n",
           BASELINE_TPS, baseline_ms, incumbent_ms, candidate_ms, delta_ms,
           modeled_tps, PASS_TPS, DIAGNOSE_TPS, MAX_IQR_FRAC, decision);

    stats tail_stats[3] = {};
    CHECK(measure(arms, 1u, true, clamp, tail_stats) == 0,
          "measure separate one-token tail");
    for (uint32_t i = 0; i < 3u; ++i) print_stats("tail1", &arms[i], &tail_stats[i]);

    CHECK(hipFree(half_down_w) == hipSuccess &&
          hipFree(half_up_w) == hipSuccess &&
          hipFree(half_gate_w) == hipSuccess &&
          hipFree(full_down_w) == hipSuccess &&
          hipFree(full_up_w) == hipSuccess &&
          hipFree(full_gate_w) == hipSuccess, "free resident timing weights");
    routes_free(&half_routes);
    routes_free(&rank1_routes);
    routes_free(&rank0_routes);
    ds4_gpu_cleanup();
    return noisy || modeled_tps < DIAGNOSE_TPS ? 1 : 0;
}
