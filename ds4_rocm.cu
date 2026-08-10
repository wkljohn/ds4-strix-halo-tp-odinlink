#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifdef __HIP_PLATFORM_AMD__
#include "ds4_rocm.h"
#include <hipblaslt/hipblaslt.h>

#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFULL
#define MASK_T uint64_t
#define DS4_GPU_BACKEND_NAME "ROCm"
#define DS4_GPU_LOG_PREFIX "ds4: ROCm "
#define DS4_GPU_BLAS_NAME "hipBLAS"
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cub/block/block_radix_sort.cuh>

#define FULL_WARP_MASK 0xFFFFFFFFu
#define MASK_T uint32_t
#define DS4_GPU_BACKEND_NAME "CUDA"
#define DS4_GPU_LOG_PREFIX "ds4: CUDA "
#define DS4_GPU_BLAS_NAME "cuBLAS"
#endif

#include <stdint.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <unordered_map>
#include <vector>

#include "ds4_gpu.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_QK_K 256
#define DS4_ROCM_UNUSED __attribute__((unused))

enum {
    /* attention_decode_mixed_kernel stores raw-window scores plus visible
     * compressed scores in shared memory.  The host routes larger unmasked
     * decode calls to the online attention kernel so this fixed buffer never
     * becomes an out-of-bounds write at long context. */
    DS4_ROCM_ATTENTION_SCORE_CAP = 8192u,
    DS4_ROCM_ATTENTION_RAW_SCORE_CAP = 256u
};

struct ds4_gpu_tensor {
    void *ptr;
    uint64_t bytes;
    int owner;
};

typedef struct {
    uint8_t scales[CUDA_QK_K / 16];
    uint8_t qs[CUDA_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} cuda_block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[CUDA_QK_K / 2];
} cuda_block_q4_K;

typedef struct {
    float d;
    int8_t qs[CUDA_QK_K];
    int16_t bsums[CUDA_QK_K / 16];
} cuda_block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[CUDA_QK_K / 8];
} cuda_block_iq2_xxs;

#include "ds4_iq2_tables_cuda.inc"

#include "rocm/ds4_rocm_runtime.cuh"

#include "rocm/ds4_rocm_common.cuh"

#include "rocm/ds4_rocm_q8.cuh"

#include "rocm/ds4_rocm_norm_rope.cuh"

#include "rocm/ds4_rocm_fp8_kv.cuh"

#include "rocm/ds4_rocm_attention.cuh"

#include "rocm/ds4_rocm_hc.cuh"

#include "rocm/ds4_rocm_output.cuh"

#include "rocm/ds4_rocm_indexer.cuh"

#include "rocm/ds4_rocm_embedding_launch.cuh"

#include "rocm/ds4_rocm_matmul.cuh"

extern "C" void ds4_gpu_rocm_mark_speculative_decode(void) {
    g_q8_decode_pair_dp4a_speculative = 1;
    const char *pair = getenv("DS4_ROCM_Q8_DECODE_PAIR_DP4A");
    const char *allow =
        getenv("DS4_ROCM_Q8_DECODE_PAIR_DP4A_SPECULATIVE");
    if (pair && pair[0] == '1' && pair[1] == '\0' &&
        (!allow || allow[0] != '1' || allow[1] != '\0')) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "paired Q8 DP4A disabled for speculative target consistency; "
                "ordinary decode remains accelerated\n");
    }
}

#include "rocm/ds4_rocm_fp8_kv_launch.cuh"

#include "rocm/ds4_rocm_compressor.cuh"

#include "rocm/ds4_rocm_attention_launch.cuh"

#include "rocm/ds4_rocm_shared_expert.cuh"

#include "rocm/ds4_rocm_misc_launch.cuh"
#include "rocm/ds4_rocm_router.cuh"

#include "rocm/ds4_rocm_moe.cuh"

#include "rocm/ds4_rocm_moe_launch.cuh"

#include "rocm/ds4_rocm_glm.cuh"

#include "rocm/ds4_rocm_hc_output_launch.cuh"

#include "rocm/ds4_rocm_current_api_compat.cuh"

#define DS4_ROCM_TP_READY 1
/* ------------------------------------------------------------------------
 * DS4-TP-gfx1151 (patch 4): ROCm tensor-parallel gate runtime.
 *
 * Mirrors the Metal design at ds4_metal.m:8272 - kernels ahead of the gate
 * leave a partial in a slab slot, the GPU signals arrival, a service thread
 * runs the transport exchange, the CPU signals release, and the pre-encoded
 * combine kernel waits on it.
 *
 * PRIMITIVE MAPPING
 *   MTLSharedEvent GPU-signal  ->  hipStreamWriteValue64(stream, addr, seq)
 *   MTLSharedEvent CPU-signal  ->  host store
 *   encoder wait on event      ->  hipStreamWaitValue64(..., WaitValueGte)
 *
 * WHY THE NULL STREAM. Every compute kernel launches <<<grid,block,shmem>>>
 * with no stream argument (rocm/ds4_rocm_matmul.cuh:15), so nothing else
 * orders against them.
 *
 * MEASURED, NOT ASSUMED (T0 probe, gfx1151, ROCm 7.2): host stores into
 * hipMalloc memory succeed, and a GPU kernel write becomes host-visible with
 * no explicit sync (~1.6k spins). The slab is ds4_gpu_tensor_alloc ->
 * cudaMalloc -> hipMalloc (rocm/ds4_rocm_runtime.cuh:5894), i.e. device
 * memory - on this UMA APU that is still host-reachable. On a discrete GPU it
 * would NOT be, and this runtime would need a hipHostMalloc slab instead.
 *
 * PAYLOAD VISIBILITY - SETTLED BY MEASUREMENT (T2 probe,
 * scripts/t2_payload_visibility_probe.cpp): 20,000 iterations of
 * [producing kernel -> hipStreamWriteValue64 -> host spin -> validate 4096
 * floats] gave ZERO stale reads. So on gfx1151/ROCm 7.2 flag arrival DOES
 * imply the producing kernel's stores are host-visible.
 *
 * This is the failure Metal measured and designed around ("a flag write
 * carries no memory-visibility guarantee for the payload buffer ... measured:
 * stale rows in the first sub-kick", ds4_metal.m:8776-8783). It is real on
 * Apple's hardware; HIP's stream-ordered barrier packet behaves differently
 * here. Re-run the probe on any new ROCm or silicon rather than assuming it
 * carries over - and note it exercised ONE producing kernel on ONE stream,
 * which is the gate's shape but not the whole engine's.
 *
 * TWO INDEPENDENT CHANNELS, mirroring Metal. Metal keeps g_tp_seq for row
 * gates (ds4_metal.m:8377) and g_tp_batch_seq for batch/big (8312), with two
 * release events (8543-8544). A single counter is WRONG: ds4_tp.c:876-881
 * maps seq->slot as the identity for DS4, so one batch gate would shift every
 * later row gate's slot permanently, tripping the "gate order broke" check at
 * ds4_tp.c:955-960.
 *
 * DEADLOCK SAFETY. The release is signalled even when the exchange FAILS - a
 * GPU blocked in WaitValue cannot be interrupted. The failure latches and
 * surfaces via ds4_gpu_tp_failed(). Note the counter is advanced BEFORE the
 * failure check so a transient error cannot desynchronise the two ranks.
 * ------------------------------------------------------------------------ */

#include <pthread.h>
#include <sched.h>

/* Metal uses 1024 (ds4_metal.m:8302). DS4-Flash is 43 layers -> 86 gates per
 * token (ds4.c:56463), so 64 would wrap within a single token. */
#define DS4_TP_RING 1024

enum ds4_tp_kind { DS4_TP_ROW = 0, DS4_TP_BATCH = 1, DS4_TP_BIG = 2 };

struct ds4_tp_req {
    uint32_t kind, layer, gate, rows;
    const void           *out_ptr;
    void                 *in_ptr;
    uint64_t              bytes;
};

/* channel 0 = row gates, channel 1 = batch/big gates */
struct ds4_tp_chan {
    volatile uint64_t *gpu_flag;   /* GPU writes arrival seq */
    volatile uint64_t *cpu_flag;   /* service thread writes release seq */
    uint64_t           seq;        /* encoder-side, single-threaded */
    struct ds4_tp_req  ring[DS4_TP_RING];
};

static struct ds4_tp_chan       g_tp_chan[2];
static ds4_gpu_tp_exchange_fn   g_tp_fn        = NULL;
static void                    *g_tp_ud        = NULL;
static ds4_gpu_tp_batch_exchange_fn g_tp_batch_fn = NULL;
static ds4_gpu_tp_big_exchange_fn   g_tp_big_fn   = NULL;
static pthread_t                g_tp_thread;
static volatile int             g_tp_thread_live = 0;
static volatile int             g_tp_run         = 0;
static int                      g_tp_failed      = 0;   /* atomics only */
static int                      g_tp_session_batch = 0;
static int                      g_tp_expert_shard_suspended = 0;
static uint32_t                 g_tp_expert_split = 0;
static int                      g_tp_keepalive_paused = 0;
static int                      g_tp_attn_head_split = 0;
static uint32_t                 g_tp_split_rank = 0;
static uint32_t                 g_tp_split_world = 1;
static void                    *g_tp_sig_alloc = NULL;

/* Diagnostic-only range reduction for the F32 partial sum immediately before
 * an FFN TP gate.  Each invocation writes one compact host-mapped record; the
 * service thread consumes it only after the gate arrival word is visible. */
struct ds4_tp_ffn_range_record {
    uint64_t seq;
    uint64_t elements;
    uint64_t nan_count, pos_inf_count, neg_inf_count;
    uint64_t over_f16_count, over_f16_margin_count;
    uint64_t below_f16_normal_count, below_f16_subnormal_count;
    float finite_min, finite_max, finite_max_abs;
};

struct ds4_tp_ffn_range_total {
    uint64_t invocations, elements;
    uint64_t nan_count, pos_inf_count, neg_inf_count;
    uint64_t over_f16_count, over_f16_margin_count;
    uint64_t below_f16_normal_count, below_f16_subnormal_count;
    float finite_min, finite_max, finite_max_abs;
};

static int g_tp_ffn_range_profile = -1;
static ds4_tp_ffn_range_record *g_tp_ffn_range_host[2] = {};
static ds4_tp_ffn_range_record *g_tp_ffn_range_device[2] = {};
static ds4_tp_ffn_range_total g_tp_ffn_range_total[2] = {};

static inline int ds4_tp_ffn_range_enabled(void) {
#if defined(DS4_ENABLE_PROFILING) && DS4_ENABLE_PROFILING
    if (g_tp_ffn_range_profile < 0) {
        const char *s = getenv("DS4_TP_FFN_RANGE_PROFILE");
        g_tp_ffn_range_profile = s && strcmp(s, "1") == 0 ? 1 : 0;
    }
    return g_tp_ffn_range_profile;
#else
    return 0;
#endif
}

__global__ static void ds4_tp_ffn_range_kernel(
        const float *values, uint64_t elements, uint64_t seq,
        ds4_tp_ffn_range_record *out) {
    __shared__ ds4_tp_ffn_range_record partial[256];
    ds4_tp_ffn_range_record p = {};
    p.finite_min = INFINITY;
    p.finite_max = -INFINITY;
    for (uint64_t i = threadIdx.x; i < elements; i += blockDim.x) {
        float x = values[i];
        if (isnan(x)) { p.nan_count++; continue; }
        if (isinf(x)) {
            if (x > 0.0f) p.pos_inf_count++; else p.neg_inf_count++;
            continue;
        }
        float ax = fabsf(x);
        p.finite_min = fminf(p.finite_min, x);
        p.finite_max = fmaxf(p.finite_max, x);
        p.finite_max_abs = fmaxf(p.finite_max_abs, ax);
        if (ax > 65504.0f) p.over_f16_count++;
        if (ax > 32752.0f) p.over_f16_margin_count++;
        if (ax != 0.0f && ax < 0x1p-14f) p.below_f16_normal_count++;
        if (ax != 0.0f && ax < 0x1p-24f) p.below_f16_subnormal_count++;
    }
    partial[threadIdx.x] = p;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            ds4_tp_ffn_range_record &a = partial[threadIdx.x];
            const ds4_tp_ffn_range_record &b = partial[threadIdx.x + stride];
            a.nan_count += b.nan_count;
            a.pos_inf_count += b.pos_inf_count;
            a.neg_inf_count += b.neg_inf_count;
            a.over_f16_count += b.over_f16_count;
            a.over_f16_margin_count += b.over_f16_margin_count;
            a.below_f16_normal_count += b.below_f16_normal_count;
            a.below_f16_subnormal_count += b.below_f16_subnormal_count;
            a.finite_min = fminf(a.finite_min, b.finite_min);
            a.finite_max = fmaxf(a.finite_max, b.finite_max);
            a.finite_max_abs = fmaxf(a.finite_max_abs, b.finite_max_abs);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[0].seq = seq;
        partial[0].elements = elements;
        *out = partial[0];
    }
}
/* DS4-TP-gfx1151 (patch 13): the gate MUST NOT use the null stream.
 * MEASURED (scripts/t4_null_stream_gate_probe.cpp, gfx1151/ROCm 7.2):
 *   created stream, no kernel : ARRIVAL SEEN 0.00 s
 *   NULL stream,    no kernel : NEVER SEEN (12 s budget)  <-- same code
 * hipStreamWriteValue64 simply never lands on stream 0 here, with nothing
 * queued ahead of it. T3 missed this because it used hipStreamCreate.
 * T5 then verified a dedicated stream still ORDERS correctly against the
 * null-stream compute kernels (legacy null-stream implicit sync), so the
 * gate's blocking semantics are preserved. */
static hipStream_t              g_tp_stream = NULL;
/* ds4_tp.c is plain C; declare the hook setter with C linkage rather than
 * including ds4_tp.h into this .cu. Must match ds4_tp.h exactly. */
extern "C" {
typedef int (*ds4_tp_devcopy_fn)(void *dst, const void *src, uint64_t bytes);
void ds4_tp_set_devcopy(ds4_tp_devcopy_fn fn);
}

static hipStream_t              g_tp_copy_stream = NULL;

/* Device DMA instead of a CPU read of write-combining device memory.
 * Both ends are hipMalloc'd (the slab and the graph tensor), so this is a
 * device-to-device copy the DMA engine can do while the GPU is parked in
 * WaitValue. Synchronous from the caller's view: ds4_tp.c expects the bytes to
 * be in place on return. */
static int ds4_tp_devcopy_hip(void *dst, const void *src, uint64_t bytes) {
    if (!g_tp_copy_stream || !dst || !src || bytes == 0) return 0;
    if (hipMemcpyAsync(dst, src, (size_t)bytes, hipMemcpyDeviceToDevice,
                       g_tp_copy_stream) != hipSuccess) {
        (void)hipGetLastError();
        return 0;
    }
    if (hipStreamSynchronize(g_tp_copy_stream) != hipSuccess) {
        (void)hipGetLastError();
        return 0;
    }
    return 1;
}
static int                      g_tp_sig_is_host = 0;

static int ds4_tp_fail_get(void) { return __atomic_load_n(&g_tp_failed, __ATOMIC_ACQUIRE); }
static void ds4_tp_fail_set(void) { __atomic_store_n(&g_tp_failed, 1, __ATOMIC_RELEASE); }

static void ds4_tp_ffn_range_print(const char *gate_type, int layer,
                                   const ds4_tp_ffn_range_total *t) {
    const uint64_t finite = t->elements - t->nan_count -
                            t->pos_inf_count - t->neg_inf_count;
    fprintf(stderr,
            "{\"ds4_tp_ffn_range\":true,\"rank\":%u,\"gate_type\":\"%s\","
            "\"layer\":%d,\"invocation_count\":%llu,\"element_count\":%llu,"
            "\"finite_min\":",
            g_tp_split_rank, gate_type, layer,
            (unsigned long long)t->invocations,
            (unsigned long long)t->elements);
    if (finite) fprintf(stderr, "%.9g", t->finite_min); else fputs("null", stderr);
    fputs(",\"finite_max\":", stderr);
    if (finite) fprintf(stderr, "%.9g", t->finite_max); else fputs("null", stderr);
    fputs(",\"finite_max_abs\":", stderr);
    if (finite) fprintf(stderr, "%.9g", t->finite_max_abs); else fputs("null", stderr);
    fprintf(stderr,
            ",\"abs_gt_65504_count\":%llu,\"abs_gt_65504_fraction\":%.17g,"
            "\"abs_gt_32752_count\":%llu,\"abs_gt_32752_fraction\":%.17g,"
            "\"nonzero_below_f16_normal_count\":%llu,"
            "\"nonzero_below_f16_subnormal_count\":%llu,"
            "\"nan_count\":%llu,\"pos_inf_count\":%llu,\"neg_inf_count\":%llu}\n",
            (unsigned long long)t->over_f16_count,
            t->elements ? (double)t->over_f16_count / (double)t->elements : 0.0,
            (unsigned long long)t->over_f16_margin_count,
            t->elements ? (double)t->over_f16_margin_count / (double)t->elements : 0.0,
            (unsigned long long)t->below_f16_normal_count,
            (unsigned long long)t->below_f16_subnormal_count,
            (unsigned long long)t->nan_count,
            (unsigned long long)t->pos_inf_count,
            (unsigned long long)t->neg_inf_count);
}

static void ds4_tp_ffn_range_consume(int ch, uint64_t seq,
                                     const ds4_tp_req *req) {
    if (!g_tp_ffn_range_host[ch]) return;
    const ds4_tp_ffn_range_record *r =
        &g_tp_ffn_range_host[ch][seq % DS4_TP_RING];
    if (r->seq != seq) return; /* This gate type was not instrumented. */
    ds4_tp_ffn_range_total one = {};
    one.invocations = 1;
    one.elements = r->elements;
    one.nan_count = r->nan_count;
    one.pos_inf_count = r->pos_inf_count;
    one.neg_inf_count = r->neg_inf_count;
    one.over_f16_count = r->over_f16_count;
    one.over_f16_margin_count = r->over_f16_margin_count;
    one.below_f16_normal_count = r->below_f16_normal_count;
    one.below_f16_subnormal_count = r->below_f16_subnormal_count;
    one.finite_min = r->finite_min;
    one.finite_max = r->finite_max;
    one.finite_max_abs = r->finite_max_abs;
    const int type = ch == 0 ? 0 : 1;
    const char *name = type == 0 ? "decode-row" : "prefill-big";
    ds4_tp_ffn_range_print(name, (int)req->layer, &one);

    ds4_tp_ffn_range_total *t = &g_tp_ffn_range_total[type];
    const uint64_t old_finite = t->elements - t->nan_count -
                                t->pos_inf_count - t->neg_inf_count;
    const uint64_t new_finite = r->elements - r->nan_count -
                                r->pos_inf_count - r->neg_inf_count;
    if (new_finite) {
        if (!old_finite || r->finite_min < t->finite_min) t->finite_min = r->finite_min;
        if (!old_finite || r->finite_max > t->finite_max) t->finite_max = r->finite_max;
        if (!old_finite || r->finite_max_abs > t->finite_max_abs) t->finite_max_abs = r->finite_max_abs;
    }
    t->invocations++;
    t->elements += r->elements;
    t->nan_count += r->nan_count;
    t->pos_inf_count += r->pos_inf_count;
    t->neg_inf_count += r->neg_inf_count;
    t->over_f16_count += r->over_f16_count;
    t->over_f16_margin_count += r->over_f16_margin_count;
    t->below_f16_normal_count += r->below_f16_normal_count;
    t->below_f16_subnormal_count += r->below_f16_subnormal_count;
}

/* Drive one channel forward if the GPU has reached `next`. Returns 1 if it
 * processed a gate. */
static int g_tp_trace = -1;
static inline int tp_trace(void) {
#if defined(DS4_ENABLE_PROFILING) && DS4_ENABLE_PROFILING
    if (g_tp_trace < 0) g_tp_trace = getenv("DS4_TP_TRACE") ? 1 : 0;
    return g_tp_trace;
#else
    return 0;
#endif
}

struct ds4_tp_interval_stat {
    uint64_t count;
    uint64_t sum_ns;
    uint64_t min_ns;
    uint64_t max_ns;
};

/* Row gates alternate ATTN (gate=0) / FFN (gate=1) within a layer (matches
 * DS4_TP_GATE_TRACE's "g=0"/"g=1"). Bucketing release_to_arrival by which
 * gate is ARRIVING splits the aggregate per-gate compute time in two without
 * needing the full stage profiler (which stalls, see
 * DECODE-PROFILER-STALL.md): the interval ending at an FFN-gate arrival is
 * router+routed_moe+shared_ffn compute; the interval ending at an ATTN-gate
 * arrival is the next layer's attention compute. Sized 4 defensively - this
 * model always uses exactly 2, but a future model's row-gate count per
 * layer must not index out of bounds. */
#define DS4_TP_GATE_BUCKETS 4
#define DS4_TP_KIND_BUCKETS 3

struct ds4_tp_interval_profile {
    struct ds4_tp_interval_stat detect[2];
    struct ds4_tp_interval_stat detect_by_kind[DS4_TP_KIND_BUCKETS];
    struct ds4_tp_interval_stat callback[2];
    struct ds4_tp_interval_stat callback_by_kind[DS4_TP_KIND_BUCKETS];
    /* Host-side upper bound from noticing an arrival through publishing its
     * release.  Unlike a compute-stream event, this cannot absorb unrelated
     * default-stream work. */
    struct ds4_tp_interval_stat detect_to_release_by_kind[DS4_TP_KIND_BUCKETS];
    /* callback_by_gate[ch][gate] is the DIRECT, trustworthy measurement of a
     * gate's real cross-rank exchange cost - measured purely via
     * clock_gettime() on the service thread around the g_tp_fn() call, with
     * no GPU stream involved at all. Use this, not a GPU-event split around
     * ds4_gpu_tp_gate_encode(), to separate real compute from gate-wait:
     * g_tp_stream is created via plain hipStreamCreate (a *blocking* stream,
     * ds4_rocm.cu:622), so under legacy default-stream semantics any stream-0
     * event recorded after a gate encode call absorbs the ENTIRE wait for
     * g_tp_stream's pending WaitValue to clear, contaminating whichever GPU-
     * event interval happens to end there - this bit us once, see
     * DECODE-ACCELERATION-PLAN.md's "Stage 0d" correction. */
    struct ds4_tp_interval_stat callback_by_gate[2][DS4_TP_GATE_BUCKETS];
    struct ds4_tp_interval_stat release_to_arrival[2];
    struct ds4_tp_interval_stat release_to_arrival_by_kind[DS4_TP_KIND_BUCKETS];
    struct ds4_tp_interval_stat release_to_arrival_by_gate[2][DS4_TP_GATE_BUCKETS];
    uint64_t last_miss_ns[2];
    uint64_t last_release_ns[2];
    uint64_t next_report;
};

static inline uint64_t ds4_tp_monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static inline void ds4_tp_interval_add(struct ds4_tp_interval_stat *s,
                                       uint64_t ns) {
    s->count++;
    s->sum_ns += ns;
    if (s->count == 1 || ns < s->min_ns) s->min_ns = ns;
    if (ns > s->max_ns) s->max_ns = ns;
}

static void ds4_tp_interval_print_one(int ch, const char *name,
                                      const struct ds4_tp_interval_stat *s) {
    if (!s->count) return;
    fprintf(stderr, DS4_GPU_LOG_PREFIX
            "TP service intervals rank=%u ch=%d %s count=%llu "
            "mean_us=%.3f min_us=%.3f max_us=%.3f\n",
            g_tp_split_rank, ch, name, (unsigned long long)s->count,
            (double)s->sum_ns / (double)s->count / 1000.0,
            (double)s->min_ns / 1000.0, (double)s->max_ns / 1000.0);
}

static void ds4_tp_interval_print(const struct ds4_tp_interval_profile *p) {
    static const char *gate_names[DS4_TP_GATE_BUCKETS] = {
        "release_to_arrival_gate=0_attn", "release_to_arrival_gate=1_ffn",
        "release_to_arrival_gate=2", "release_to_arrival_gate=3+"
    };
    static const char *cb_gate_names[DS4_TP_GATE_BUCKETS] = {
        "callback_gate=0_attn", "callback_gate=1_ffn",
        "callback_gate=2", "callback_gate=3+"
    };
    static const char *kind_names[DS4_TP_KIND_BUCKETS] = {
        "row", "batch", "big"
    };
    for (int ch = 0; ch < 2; ch++) {
        ds4_tp_interval_print_one(ch, "detect_upper_bound", &p->detect[ch]);
        ds4_tp_interval_print_one(ch, "callback", &p->callback[ch]);
        for (int g = 0; g < DS4_TP_GATE_BUCKETS; g++) {
            ds4_tp_interval_print_one(ch, cb_gate_names[g],
                                      &p->callback_by_gate[ch][g]);
        }
        ds4_tp_interval_print_one(ch, "release_to_arrival",
                                  &p->release_to_arrival[ch]);
        for (int g = 0; g < DS4_TP_GATE_BUCKETS; g++) {
            ds4_tp_interval_print_one(ch, gate_names[g],
                                      &p->release_to_arrival_by_gate[ch][g]);
        }
    }
    for (int kind = 0; kind < DS4_TP_KIND_BUCKETS; kind++) {
        char name[64];
        snprintf(name, sizeof(name), "detect_upper_bound_kind=%s", kind_names[kind]);
        ds4_tp_interval_print_one(kind == DS4_TP_ROW ? 0 : 1, name,
                                  &p->detect_by_kind[kind]);
        snprintf(name, sizeof(name), "callback_kind=%s", kind_names[kind]);
        ds4_tp_interval_print_one(kind == DS4_TP_ROW ? 0 : 1, name,
                                  &p->callback_by_kind[kind]);
        snprintf(name, sizeof(name), "detect_to_release_kind=%s", kind_names[kind]);
        ds4_tp_interval_print_one(kind == DS4_TP_ROW ? 0 : 1, name,
                                  &p->detect_to_release_by_kind[kind]);
        snprintf(name, sizeof(name), "release_to_arrival_kind=%s", kind_names[kind]);
        ds4_tp_interval_print_one(kind == DS4_TP_ROW ? 0 : 1, name,
                                  &p->release_to_arrival_by_kind[kind]);
    }
}

static int ds4_tp_pump(int ch, uint64_t *next) {
    struct ds4_tp_chan *c = &g_tp_chan[ch];
    uint64_t arrived = __atomic_load_n(c->gpu_flag, __ATOMIC_ACQUIRE);
    if (arrived < *next) return 0;
    if (tp_trace()) fprintf(stderr, "[tp] pump ch=%d arrived=%llu next=%llu kind=%d\n",
                            ch, (unsigned long long)arrived,
                            (unsigned long long)*next,
                            (int)c->ring[*next % DS4_TP_RING].kind);

    const struct ds4_tp_req *r = &c->ring[*next % DS4_TP_RING];
    int ok = 0;
    switch (r->kind) {
    case DS4_TP_ROW:
        ok = g_tp_fn ? g_tp_fn(g_tp_ud, r->layer, r->gate, *next) : 0;
        break;
    case DS4_TP_BATCH:
        /* Metal dispatches on rows/bytes (ds4_metal.m:8522-8541). Falling back
         * to the row callback here would exchange the wrong buffer. */
        ok = g_tp_batch_fn ? g_tp_batch_fn(g_tp_ud, r->layer, r->rows, *next) : 0;
        break;
    case DS4_TP_BIG:
        /* ds4_gpu.h:246-248 - the callback takes RAW POINTERS to the
         * CPU-visible bounce buffers, not tensors, and seq comes third:
         *   (ud, layer, seq, const void *out, void *in, bytes)
         * The T0 probe established hipMalloc memory is host-reachable on this
         * UMA APU, so the snapshotted device pointer is directly usable. */
        ok = (g_tp_big_fn && r->out_ptr && r->in_ptr)
             ? g_tp_big_fn(g_tp_ud, r->layer, *next,
                           r->out_ptr, r->in_ptr, r->bytes)
             : 0;
        break;
    default: ok = 0; break;
    }
    if (!ok) {
        ds4_tp_fail_set();
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "TP exchange failed (kind %u layer %u seq %llu); releasing "
                "anyway so the GPU does not hang\n",
                r->kind, r->layer, (unsigned long long)*next);
    }
    __atomic_store_n(c->cpu_flag, *next, __ATOMIC_RELEASE);
    (*next)++;
    return 1;
}

/* Range-profiled pump is separate so the default-off service hot path has no
 * added profiling branch or memory access. */
static int ds4_tp_pump_ffn_range(int ch, uint64_t *next) {
    struct ds4_tp_chan *c = &g_tp_chan[ch];
    uint64_t arrived = __atomic_load_n(c->gpu_flag, __ATOMIC_ACQUIRE);
    if (arrived < *next) return 0;
    const struct ds4_tp_req *r = &c->ring[*next % DS4_TP_RING];
    ds4_tp_ffn_range_consume(ch, *next, r);
    int ok = 0;
    switch (r->kind) {
    case DS4_TP_ROW:
        ok = g_tp_fn ? g_tp_fn(g_tp_ud, r->layer, r->gate, *next) : 0;
        break;
    case DS4_TP_BATCH:
        ok = g_tp_batch_fn ? g_tp_batch_fn(g_tp_ud, r->layer, r->rows, *next) : 0;
        break;
    case DS4_TP_BIG:
        ok = (g_tp_big_fn && r->out_ptr && r->in_ptr)
             ? g_tp_big_fn(g_tp_ud, r->layer, *next,
                           r->out_ptr, r->in_ptr, r->bytes)
             : 0;
        break;
    default: ok = 0; break;
    }
    if (!ok) {
        ds4_tp_fail_set();
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "TP exchange failed (kind %u layer %u seq %llu); releasing "
                "anyway so the GPU does not hang\n",
                r->kind, r->layer, (unsigned long long)*next);
    }
    __atomic_store_n(c->cpu_flag, *next, __ATOMIC_RELEASE);
    (*next)++;
    return 1;
}

/* Profiled copy of the pump keeps the default-off hot path above completely
 * free of clock reads and profiling branches.  Detection is necessarily an
 * upper bound: the GPU write happened between the last miss and this hit. */
static int ds4_tp_pump_profile(int ch, uint64_t *next,
                               struct ds4_tp_interval_profile *p) {
    struct ds4_tp_chan *c = &g_tp_chan[ch];
    uint64_t noticed_ns = ds4_tp_monotonic_ns();
    uint64_t arrived = __atomic_load_n(c->gpu_flag, __ATOMIC_ACQUIRE);
    if (arrived < *next) {
        p->last_miss_ns[ch] = noticed_ns;
        return 0;
    }
    const struct ds4_tp_req *r = &c->ring[*next % DS4_TP_RING];
    if (p->last_miss_ns[ch]) {
        /* Consume it: back-to-back hits with no intervening miss (a burst of
         * already-arrived gates) must not keep reusing this stale timestamp,
         * which would otherwise record a growing, meaningless interval for
         * every hit in the burst instead of just the first. */
        const uint64_t detect_ns = noticed_ns - p->last_miss_ns[ch];
        ds4_tp_interval_add(&p->detect[ch], detect_ns);
        if (r->kind < DS4_TP_KIND_BUCKETS) {
            ds4_tp_interval_add(&p->detect_by_kind[r->kind], detect_ns);
        }
        p->last_miss_ns[ch] = 0;
    }
    ds4_tp_ffn_range_consume(ch, *next, r);
    if (p->last_release_ns[ch]) {
        uint64_t dt = noticed_ns - p->last_release_ns[ch];
        ds4_tp_interval_add(&p->release_to_arrival[ch], dt);
        if (r->kind < DS4_TP_KIND_BUCKETS) {
            ds4_tp_interval_add(&p->release_to_arrival_by_kind[r->kind], dt);
        }
        if (r->kind == DS4_TP_ROW) {
            uint32_t gb = r->gate < DS4_TP_GATE_BUCKETS ?
                          r->gate : DS4_TP_GATE_BUCKETS - 1;
            ds4_tp_interval_add(&p->release_to_arrival_by_gate[ch][gb], dt);
        }
    }
    if (tp_trace()) fprintf(stderr, "[tp] pump ch=%d arrived=%llu next=%llu kind=%d\n",
                            ch, (unsigned long long)arrived,
                            (unsigned long long)*next,
                            (int)r->kind);

    int ok = 0;
    uint64_t callback_start_ns = ds4_tp_monotonic_ns();
    switch (r->kind) {
    case DS4_TP_ROW:
        ok = g_tp_fn ? g_tp_fn(g_tp_ud, r->layer, r->gate, *next) : 0;
        break;
    case DS4_TP_BATCH:
        ok = g_tp_batch_fn ? g_tp_batch_fn(g_tp_ud, r->layer, r->rows, *next) : 0;
        break;
    case DS4_TP_BIG:
        ok = (g_tp_big_fn && r->out_ptr && r->in_ptr)
             ? g_tp_big_fn(g_tp_ud, r->layer, *next,
                           r->out_ptr, r->in_ptr, r->bytes)
             : 0;
        break;
    default: ok = 0; break;
    }
    uint64_t callback_end_ns = ds4_tp_monotonic_ns();
    uint64_t callback_ns = callback_end_ns - callback_start_ns;
    ds4_tp_interval_add(&p->callback[ch], callback_ns);
    if (r->kind < DS4_TP_KIND_BUCKETS) {
        ds4_tp_interval_add(&p->callback_by_kind[r->kind], callback_ns);
    }
    if (r->kind == DS4_TP_ROW) {
        uint32_t cb_gb = r->gate < DS4_TP_GATE_BUCKETS ?
                         r->gate : DS4_TP_GATE_BUCKETS - 1;
        ds4_tp_interval_add(&p->callback_by_gate[ch][cb_gb], callback_ns);
    }
    if (!ok) {
        ds4_tp_fail_set();
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "TP exchange failed (kind %u layer %u seq %llu); releasing "
                "anyway so the GPU does not hang\n",
                r->kind, r->layer, (unsigned long long)*next);
    }
    __atomic_store_n(c->cpu_flag, *next, __ATOMIC_RELEASE);
    const uint64_t released_ns = ds4_tp_monotonic_ns();
    if (r->kind < DS4_TP_KIND_BUCKETS) {
        ds4_tp_interval_add(&p->detect_to_release_by_kind[r->kind],
                            released_ns - noticed_ns);
    }
    p->last_release_ns[ch] = released_ns;
    (*next)++;
    return 1;
}

static void *ds4_tp_service_thread(void *arg) {
    (void)arg;
    uint64_t next[2] = { 1, 1 };
    uint32_t idle_streak = 0;
#if defined(DS4_ENABLE_PROFILING) && DS4_ENABLE_PROFILING
    const char *profile_env = getenv("DS4_TP_SERVICE_INTERVAL_PROFILE");
    const char *verify_stage_env = getenv("DS4_DSPARK_VERIFY_STAGE_EVENTS");
    const int profile_enabled =
        (profile_env && strcmp(profile_env, "1") == 0) ||
        (verify_stage_env && verify_stage_env[0] &&
         strcmp(verify_stage_env, "0") != 0);
#else
    const int profile_enabled = 0;
#endif
    const int ffn_range_enabled = ds4_tp_ffn_range_enabled() &&
                                  g_tp_ffn_range_host[0] &&
                                  g_tp_ffn_range_host[1];
    struct ds4_tp_interval_profile profile = {};
    profile.next_report = 500;
    while (g_tp_run) {
        int did;
        if (__builtin_expect(profile_enabled, 0)) {
            did = ds4_tp_pump_profile(0, &next[0], &profile) |
                  ds4_tp_pump_profile(1, &next[1], &profile);
            if (profile.callback[0].count >= profile.next_report) {
                ds4_tp_interval_print(&profile);
                profile.next_report += 500;
            }
        } else if (__builtin_expect(ffn_range_enabled, 0)) {
            did = ds4_tp_pump_ffn_range(0, &next[0]) |
                  ds4_tp_pump_ffn_range(1, &next[1]);
        } else {
            did = ds4_tp_pump(0, &next[0]) | ds4_tp_pump(1, &next[1]);
        }
        /* Metal tight-spins before yielding: "yielding here measurably delays
         * the release wake-up" (ds4_metal.m:8492-8496).
         *
         * DS4-TP-gfx1151 (patch 17), two defects in the original:
         *
         * 1. The back-off was 1<<14 = 16384 PAUSE. On Zen, PAUSE is ~40-65
         *    cycles, so that is ~200-300 us of BLIND time per miss - during
         *    which an arrived gate sits unnoticed. With up to 86 gates/token on
         *    both ranks that is a large share of a ~120 ms token, and it is pure
         *    detection latency, not work. 1<<8 keeps the spin cheap (~3-5 us)
         *    while still not hammering the flag line.
         *
         * 2. The yield guard tested `gpu_flag == 0`, but the gpu flags are
         *    MONOTONIC sequence counters - after the very first gate they are
         *    never zero again, so sched_yield() was unreachable dead code and
         *    this thread burned a core forever. Yield on a miss STREAK instead,
         *    which is what the guard was trying to express. */
        if (!did) {
            for (int i = 0; i < (1 << 8); i++) __builtin_ia32_pause();
            if (++idle_streak >= 64u) { idle_streak = 0; sched_yield(); }
        } else {
            idle_streak = 0;
        }
    }
    /* DRAIN: release anything already enqueued on the stream, or the GPU waits
     * forever and shutdown deadlocks (Metal drains at ds4_metal.m:8478-8481). */
    for (int ch = 0; ch < 2; ch++) {
        uint64_t arrived = __atomic_load_n(g_tp_chan[ch].gpu_flag, __ATOMIC_ACQUIRE);
        while (next[ch] <= arrived) {
            __atomic_store_n(g_tp_chan[ch].cpu_flag, next[ch], __ATOMIC_RELEASE);
            next[ch]++;
        }
        /* Also release the encoder's high-water mark: gates may be queued on
         * the stream that have not yet written their arrival flag. */
        __atomic_store_n(g_tp_chan[ch].cpu_flag, g_tp_chan[ch].seq + 8, __ATOMIC_RELEASE);
    }
    if (profile_enabled) ds4_tp_interval_print(&profile);
    return NULL;
}

extern "C" int ds4_gpu_tp_init(uint32_t rank,
                               ds4_gpu_tensor *slab, uint64_t gpu_flags_off,
                               ds4_gpu_tp_exchange_fn fn, void *ud) {
    if (!slab || !slab->ptr || !fn) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tp_init: bad arguments\n");
        return 0;
    }
    int can_wait = 0, dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        hipDeviceGetAttribute(&can_wait, hipDeviceAttributeCanUseStreamWaitValue, dev)
            != hipSuccess || !can_wait) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "tp_init: device %d lacks hipStreamWaitValue support (attr=%d)\n",
                dev, can_wait);
        return 0;
    }

    g_tp_split_rank = rank;
    g_tp_split_world = 2;   /* ds4 TP is always a two-way split */
    memset(g_tp_ffn_range_total, 0, sizeof(g_tp_ffn_range_total));

    unsigned char *base = (unsigned char *)slab->ptr;
    /* Two arrival words. ds4_tp.c sizes this region as slots*4; we use the
     * first 16 bytes for two 64-bit counters. */
    g_tp_chan[0].gpu_flag = (volatile uint64_t *)(base + gpu_flags_off);
    g_tp_chan[1].gpu_flag = (volatile uint64_t *)(base + gpu_flags_off + 8);

    /* hip_runtime_api.h:3119 documents hipStreamWaitValue* as requiring memory
     * from hipExtMallocWithFlags(hipMallocSignalMemory). hipHostMalloc happens
     * to work on this platform but is not the documented contract, so prefer
     * signal memory and fall back only if unavailable. */
    void *sig = NULL;
    g_tp_sig_is_host = 0;
    if (hipExtMallocWithFlags(&sig, 64, hipMallocSignalMemory) != hipSuccess || !sig) {
        /* DS4-TP-gfx1151 (patch 18): CLEAR the latched error.
         *
         * hipMallocSignalMemory is unavailable on this build, so this call
         * fails every run and leaves hipErrorInvalidValue latched. cudaGetLastError()
         * returns the last error from ANYWHERE, so the next unrelated
         * `cuda_ok(cudaGetLastError(), ...)` in the engine reports a spurious
         * "invalid argument" and its caller fails for no reason.
         *
         * This is not hypothetical: it made ds4_gpu_tensor_fill_f32 look broken,
         * and it made the batch prefill path fail with "raw KV batch store
         * failed" for any prompt large enough to use it - i.e. every prompt over
         * ~32 tokens - while the store itself had launched fine. */
        (void)hipGetLastError();
        if (hipHostMalloc(&sig, 64, hipHostMallocMapped) != hipSuccess || !sig) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "tp_init: release-word alloc failed\n");
            return 0;
        }
        g_tp_sig_is_host = 1;
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "tp_init: hipMallocSignalMemory unavailable, using host memory\n");
    }
    g_tp_sig_alloc = sig;
    g_tp_chan[0].cpu_flag = (volatile uint64_t *)sig;
    g_tp_chan[1].cpu_flag = (volatile uint64_t *)((char *)sig + 8);

    for (int ch = 0; ch < 2; ch++) {
        g_tp_chan[ch].seq = 0;
        memset(g_tp_chan[ch].ring, 0, sizeof(g_tp_chan[ch].ring));
        __atomic_store_n(g_tp_chan[ch].cpu_flag, 0, __ATOMIC_RELEASE);
        __atomic_store_n(g_tp_chan[ch].gpu_flag, 0, __ATOMIC_RELEASE);
    }
    g_tp_fn = fn; g_tp_ud = ud;
    __atomic_store_n(&g_tp_failed, 0, __ATOMIC_RELEASE);

    /* DS4-TP-gfx1151 (patch 21): device-side staging copy for the big gate.
     *
     * MUST be a NON-BLOCKING stream. A copy on the null stream would, under
     * legacy implicit-sync semantics, wait for g_tp_stream - which is parked on
     * the very gate this copy is servicing. That deadlocks. */
    /* OPT-IN ONLY (DS4_TP_DEVCOPY=1). Enabling this by default BREAKS the
     * transfer: "timeout waiting for bulk RDMA round (33/64 recvs)". The GPU's
     * DMA writes into the ibv-registered slab are not reliably visible to the
     * OdinLink NIC - a coherence problem between the GPU copy engine and the
     * device's view of registered memory, which hipStreamSynchronize does not
     * address (it orders the copy, it does not make it externally visible).
     *
     * The measurement that motivated this stands: staging is 64% of big-gate
     * time at 200 MB/s. Fixing it needs either a system-scope release fence
     * after the copy, fine-grained/host-coherent slab memory, or the
     * slab-direct approach (make `direct` true so no staging copy happens at
     * all), which is the cleaner fix and avoids the coherence question. */
    if (getenv("DS4_TP_DEVCOPY")) {
        if (!g_tp_copy_stream &&
            hipStreamCreateWithFlags(&g_tp_copy_stream, hipStreamNonBlocking) != hipSuccess) {
            g_tp_copy_stream = NULL;
        }
        if (g_tp_copy_stream) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "tp: DS4_TP_DEVCOPY enabled - EXPERIMENTAL, known to fail bulk RDMA rounds\n");
            ds4_tp_set_devcopy(ds4_tp_devcopy_hip);
        }
    }

    if (hipStreamCreate(&g_tp_stream) != hipSuccess || !g_tp_stream) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tp_init: gate stream create failed\n");
        if (g_tp_sig_is_host) hipHostFree(sig); else hipFree(sig);
        g_tp_sig_alloc = NULL;
        return 0;
    }
    if (ds4_tp_ffn_range_enabled()) {
        for (int ch = 0; ch < 2; ch++) {
            void *host = NULL;
            void *device = NULL;
            const size_t bytes = DS4_TP_RING * sizeof(ds4_tp_ffn_range_record);
            if (hipHostMalloc(&host, bytes, hipHostMallocMapped) != hipSuccess ||
                !host || hipHostGetDevicePointer(&device, host, 0) != hipSuccess ||
                !device) {
                (void)hipGetLastError();
                if (host) (void)hipHostFree(host);
                fprintf(stderr, DS4_GPU_LOG_PREFIX
                        "TP FFN range profiler disabled: mapped record allocation failed\n");
                for (int j = 0; j < ch; j++) {
                    (void)hipHostFree(g_tp_ffn_range_host[j]);
                    g_tp_ffn_range_host[j] = NULL;
                    g_tp_ffn_range_device[j] = NULL;
                }
                break;
            }
            memset(host, 0, bytes);
            g_tp_ffn_range_host[ch] = (ds4_tp_ffn_range_record *)host;
            g_tp_ffn_range_device[ch] = (ds4_tp_ffn_range_record *)device;
        }
    }
    g_tp_run = 1;
    if (pthread_create(&g_tp_thread, NULL, ds4_tp_service_thread, NULL) != 0) {
        g_tp_run = 0;
        if (g_tp_sig_is_host) hipHostFree(sig); else hipFree(sig);
        g_tp_sig_alloc = NULL;
        for (int ch = 0; ch < 2; ch++) {
            if (g_tp_ffn_range_host[ch])
                (void)hipHostFree(g_tp_ffn_range_host[ch]);
            g_tp_ffn_range_host[ch] = NULL;
            g_tp_ffn_range_device[ch] = NULL;
        }
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tp_init: service thread failed\n");
        return 0;
    }
    g_tp_thread_live = 1;
    {
        const char *pin_cpu_str = getenv("DS4_TP_SERVICE_THREAD_PIN_CPU");
        if (pin_cpu_str && *pin_cpu_str) {
            int pin_cpu = atoi(pin_cpu_str);
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(pin_cpu, &cpuset);
            int rc = pthread_setaffinity_np(g_tp_thread, sizeof(cpuset), &cpuset);
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "tp: service thread pin cpu=%d rc=%d\n", pin_cpu, rc);
        }
    }
    fprintf(stderr, DS4_GPU_LOG_PREFIX
            "ROCm TP rank %u ready (2 channels, HIP stream wait-value)\n", rank);
    return 1;
}

extern "C" void ds4_gpu_tp_shutdown(void) {
    if (!g_tp_thread_live) return;
    g_tp_run = 0;
    pthread_join(g_tp_thread, NULL);        /* drains before returning */
    hipDeviceSynchronize();                 /* no pending WaitValue on the word */
    if (ds4_tp_ffn_range_enabled()) {
        ds4_tp_ffn_range_print("decode-row", -1, &g_tp_ffn_range_total[0]);
        ds4_tp_ffn_range_print("prefill-big", -1, &g_tp_ffn_range_total[1]);
    }
    g_tp_thread_live = 0;
    ds4_tp_set_devcopy(NULL);
    if (g_tp_copy_stream) { (void)hipStreamDestroy(g_tp_copy_stream); g_tp_copy_stream = NULL; }
    if (g_tp_stream) { (void)hipStreamDestroy(g_tp_stream); g_tp_stream = NULL; }
    if (g_tp_sig_alloc) {
        if (g_tp_sig_is_host) hipHostFree(g_tp_sig_alloc); else hipFree(g_tp_sig_alloc);
        g_tp_sig_alloc = NULL;
    }
    for (int ch = 0; ch < 2; ch++) {
        if (g_tp_ffn_range_host[ch])
            (void)hipHostFree(g_tp_ffn_range_host[ch]);
        g_tp_ffn_range_host[ch] = NULL;
        g_tp_ffn_range_device[ch] = NULL;
    }
    g_tp_fn = NULL; g_tp_ud = NULL;
}

/* DIAGNOSTIC ONLY (DECODE-PROFILER-STALL.md): opt-in probe for the stage-
 * profiler stall. Hypothesis (Codex root-cause pass, 2026-08-05): after many
 * back-to-back cudaDeviceSynchronize() calls fully drain the device,
 * hipStreamWriteValue64() on the otherwise-idle g_tp_stream returns
 * hipSuccess but its packet is never submitted/kicked to the GPU queue, so
 * the arrival word is never observed. A trivial kernel dispatched on the
 * same stream between the write and the wait should force a normal queue
 * submission/kick if that's the real mechanism. Not a production fix - gate
 * behind an explicit env var, never on by default. */
__global__ static void ds4_tp_gate_kick_kernel(void) {}

static int g_tp_gate_kick = -1;
static inline int ds4_tp_gate_kick_enabled(void) {
    if (g_tp_gate_kick < 0) g_tp_gate_kick = getenv("DS4_TP_GATE_KICK") ? 1 : 0;
    return g_tp_gate_kick;
}

/* Publish arrival on `ch`, then block the null stream until release. */
static int ds4_tp_encode(int ch, const struct ds4_tp_req *req) {
    if (!g_tp_thread_live) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "TP gate encoded before tp_init\n");
        return 0;
    }
    struct ds4_tp_chan *c = &g_tp_chan[ch];
    /* Advance BEFORE the failure check: an early return here would leave this
     * rank's counter behind its peer's permanently. */
    uint64_t seq = ++c->seq;
    if (seq - __atomic_load_n(c->cpu_flag, __ATOMIC_ACQUIRE) >= DS4_TP_RING) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "TP gate queue overflow on channel %d (seq %llu)\n",
                ch, (unsigned long long)seq);
        ds4_tp_fail_set();
        return 0;
    }
    c->ring[seq % DS4_TP_RING] = *req;
    if (ds4_tp_fail_get()) return 0;

    if (tp_trace()) fprintf(stderr, "[tp] encode ch=%d seq=%llu kind=%d layer=%u -> enqueue\n",
                            ch, (unsigned long long)seq, (int)req->kind, req->layer);
    if (hipStreamWriteValue64(g_tp_stream, (void *)c->gpu_flag, (int64_t)seq, 0) != hipSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "TP gate: stream write failed\n");
        ds4_tp_fail_set();
        return 0;
    }
    if (ds4_tp_gate_kick_enabled()) {
        ds4_tp_gate_kick_kernel<<<1, 1, 0, g_tp_stream>>>();
        hipError_t kick_err = hipGetLastError();
        if (kick_err != hipSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "TP gate: kick kernel launch failed: %s\n",
                    hipGetErrorString(kick_err));
            ds4_tp_fail_set();
            return 0;
        }
    }
    if (hipStreamWaitValue64(g_tp_stream, (void *)c->cpu_flag, (int64_t)seq,
                             hipStreamWaitValueGte, ~0ULL) != hipSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "TP gate: stream wait failed\n");
        ds4_tp_fail_set();
        return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_tp_gate_encode(uint32_t layer, uint32_t gate) {
    struct ds4_tp_req r = {}; r.kind = DS4_TP_ROW; r.layer = layer; r.gate = gate;
    return ds4_tp_encode(0, &r);
}

extern "C" void ds4_gpu_tp_ffn_range_profile(const ds4_gpu_tensor *tensor,
                                                uint64_t elements,
                                                uint32_t layer,
                                                int prefill_big) {
    (void)layer; /* Layer is carried by the immediately following gate request. */
    if (__builtin_expect(!ds4_tp_ffn_range_enabled(), 1)) return;
    const int ch = prefill_big ? 1 : 0;
    if (!tensor || !tensor->ptr || !elements || !g_tp_ffn_range_device[ch]) return;
    const uint64_t seq = g_tp_chan[ch].seq + 1;
    ds4_tp_ffn_range_record *record =
        &g_tp_ffn_range_device[ch][seq % DS4_TP_RING];
    ds4_tp_ffn_range_kernel<<<1, 256>>>((const float *)tensor->ptr,
                                        elements, seq, record);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "TP FFN range kernel launch failed (layer %u): %s\n",
                layer, hipGetErrorString(err));
    }
}

extern "C" int ds4_gpu_tp_batch_gate_encode(uint32_t layer, uint32_t rows) {
    struct ds4_tp_req r = {}; r.kind = DS4_TP_BATCH; r.layer = layer; r.rows = rows;
    return ds4_tp_encode(1, &r);
}

extern "C" int ds4_gpu_tp_big_gate_encode(uint32_t layer, uint32_t rows,
                                          const ds4_gpu_tensor *out_t,
                                          ds4_gpu_tensor *in_t,
                                          uint64_t bytes) {
    /* Metal refuses a big gate without CPU-visible bounce pointers
     * (ds4_metal.m:8789-8794); the payload description is the whole point. */
    if (!out_t || !in_t || bytes == 0) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "TP big gate: missing bounce buffers\n");
        return 0;
    }
    struct ds4_tp_req r = {};
    r.kind = DS4_TP_BIG; r.layer = layer; r.rows = rows;
    /* The request is consumed asynchronously by the TP service thread.  Most
     * callers pass non-owning tensor views and free the small host-side view
     * descriptors immediately after this function returns.  Snapshot the
     * device addresses here instead of queueing descriptor pointers that the
     * service thread would later dereference after free.  The graph-owned
     * backing allocations remain live across the gate. */
    r.out_ptr = out_t->ptr; r.in_ptr = in_t->ptr; r.bytes = bytes;
    return ds4_tp_encode(1, &r);
}

extern "C" void ds4_gpu_tp_set_batch_exchange(ds4_gpu_tp_batch_exchange_fn fn) { g_tp_batch_fn = fn; }
extern "C" void ds4_gpu_tp_set_big_exchange(ds4_gpu_tp_big_exchange_fn fn)     { g_tp_big_fn = fn; }
extern "C" void ds4_gpu_tp_set_session_batch_mode(int enabled)                 { g_tp_session_batch = enabled; }
extern "C" void ds4_gpu_tp_set_expert_split(uint32_t first_rank1)              { g_tp_expert_split = first_rank1; }
extern "C" void ds4_gpu_tp_suspend_expert_sharding(int suspend)                { g_tp_expert_shard_suspended = suspend; }
extern "C" void ds4_gpu_tp_keepalive_pause(int paused)                         { g_tp_keepalive_paused = paused; }
extern "C" void ds4_gpu_tp_set_attn_head_split(int enabled)                    { g_tp_attn_head_split = enabled; }

/* Latched failure, polled by the engine at command-buffer boundaries
 * (ds4.c:59903). */
extern "C" int ds4_gpu_tp_failed(void) { return ds4_tp_fail_get(); }


/* ------------------------------------------------------------------------
 * Expert sharding, correctness-first.
 *
 * THE BUG THIS FIXES. Metal shards experts inside routed_moe_one via
 * ds4_gpu_tp_expert_range (ds4_metal.m:8327-8342) and offsets the gate/up/down
 * weight bases by first_expert. ROCm has NO such concept - `grep tp_rank|
 * tp_world rocm/` returns nothing, and 33 kernel sites index selected[] as a
 * GLOBAL expert id. Under TP=2 both ranks therefore compute the FULL expert
 * set and the gate exchange adds the peer's identical partial: the FFN
 * contribution DOUBLES, silently, with no error anywhere.
 *
 * WHY MASK INSTEAD OF REBASE. Metal's base-offset scheme also halves the
 * weights mapped and the work done - it is an optimisation as well as a
 * correctness mechanism. Reproducing it means threading a tp_expert_base
 * parameter through 33 kernels in 5 files. Zeroing the ROUTING WEIGHT of
 * experts this rank does not own achieves correctness alone with one small
 * kernel, because the weight is a per-(tok,slot) multiplier applied before
 * accumulation (rocm/ds4_rocm_moe.cuh:587 and 5 siblings).
 *
 * IT IS EXACT, NOT AN APPROXIMATION.
 *     full  = SUM over the 6 selected e of  w_e * f_e(x)
 *     rank0 = SUM over selected e <  half of w_e * f_e(x)
 *     rank1 = SUM over selected e >= half of w_e * f_e(x)
 *     rank0 + rank1 = full
 * DS4 sets norm_topk_prob, but the normalisation survives because the two
 * partial sums recombine; neither rank renormalises.
 *
 * COST: no memory saving and no compute saving - both ranks still map all
 * experts and evaluate all six. Phase C's ~73 GB/node footprint does NOT
 * follow from this, and decode does not get faster. This buys a CORRECT TP=2
 * run, which is the prerequisite for judging whether the optimisation is worth
 * building.
 * ------------------------------------------------------------------------ */
__global__ void ds4_tp_shard_remap_kernel(int32_t *sel_dst, float *w_dst,
                                          const int32_t *sel_src, const float *w_src,
                                          uint32_t n, int32_t lo, int32_t hi,
                                          int32_t unowned_sentinel) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int32_t e = sel_src[i];
    const bool own = (e >= lo && e < hi);
    /* Rebase onto the shard: the launcher is handed gate/up/down offsets that
     * already start at expert `lo`, so an owned expert must be addressed as
     * e-lo. Normally unowned pairs get index 0 and weight 0. The experimental
     * skip path uses -1 instead, allowing Q4_K kernels to avoid the known-zero
     * dot products while preserving the same zero contribution. */
    sel_dst[i] = own ? (e - lo) : unowned_sentinel;
    w_dst[i]   = own ? w_src[i] : 0.0f;
}

/* Owned contiguous expert range.  A zero/out-of-range boundary retains the
 * historical half split; DSpark's standard two-node mode sets 118/138. */
static void ds4_tp_expert_range(uint32_t n_total, int32_t *lo, int32_t *hi) {
    *lo = 0; *hi = (int32_t)n_total;
    if (g_tp_split_world != 2 || g_tp_expert_shard_suspended) return;
    const uint32_t low = g_tp_expert_split > 0u &&
                         g_tp_expert_split < n_total ?
                         g_tp_expert_split : n_total / 2u;
    if (g_tp_split_rank == 1) { *lo = (int32_t)low; }
    else                      { *hi = (int32_t)low; }
}

/* Rebase the routing selection onto this rank's expert shard.
 *
 * REPLACES the earlier weight-masking approach, which was incompatible with TP
 * residency: masking left the launcher asking for the WHOLE layer's expert
 * tensor (1152 MiB for this model), but TP maps only this rank's half, so every
 * layer missed the resident image and paged the unowned half back in from disk
 * until the device ran out. See docs/EXPERT-SHARD-DESIGN-FLAW.md.
 *
 * Here the caller also shifts gate/up/down by lo*expert_bytes and passes
 * n_total_expert = hi-lo, so the launcher only ever addresses resident memory.
 *
 * `scratch` is ONE caller allocation of n_pairs*(sizeof(int32_t)+sizeof(float))
 * split in two - cuda_tmp_alloc hands back a single shared global buffer, so
 * two separate calls would silently alias.
 *
 * Returns 1 when remapped (out_base/out_count set), 0 when TP is inactive or
 * the shard is suspended (Metal suspends around the DSpark support model,
 * ds4_metal.m:8646-8649), in which case the caller uses its inputs unchanged. */
extern "C" int ds4_gpu_tp_expert_shard_remap(
        const int32_t *selected, const float *weights, void *scratch,
        uint32_t n_pairs, uint32_t n_total_expert,
        const int32_t **out_selected, const float **out_weights,
        uint32_t *out_base, uint32_t *out_count) {
    if (g_tp_split_world != 2 || g_tp_expert_shard_suspended) return 0;
    if (!selected || !weights || !scratch || n_pairs == 0) return 0;
    int32_t lo = 0, hi = 0;
    ds4_tp_expert_range(n_total_expert, &lo, &hi);
    if (lo == 0 && hi == (int32_t)n_total_expert) return 0;

    int32_t *sel_dst = (int32_t *)scratch;
    float   *w_dst   = (float *)(sel_dst + n_pairs);
    const uint32_t threads = 256u;
    const uint32_t blocks = (n_pairs + threads - 1u) / threads;
    const char *skip_env = getenv("DS4_ROCM_TP_SKIP_UNOWNED");
    const bool skip_unowned = skip_env && skip_env[0] == '1' && skip_env[1] == '\0';
    static bool logged_skip_unowned = false;
    if (skip_unowned && !logged_skip_unowned) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "experimental TP unowned Q4_K expert skipping enabled\n");
        logged_skip_unowned = true;
    }
    hipLaunchKernelGGL(ds4_tp_shard_remap_kernel, dim3(blocks), dim3(threads), 0, 0,
                       sel_dst, w_dst, selected, weights, n_pairs, lo, hi,
                       skip_unowned ? -1 : 0);
    if (out_selected) *out_selected = sel_dst;
    if (out_weights)  *out_weights  = w_dst;
    if (out_base)     *out_base     = (uint32_t)lo;
    if (out_count)    *out_count    = (uint32_t)(hi - lo);
    return 1;
}

/* Exposed so callers can size the scratch buffer and know whether to bother. */
extern "C" int ds4_gpu_tp_expert_shard_active(void) {
    return (g_tp_split_world == 2 && !g_tp_expert_shard_suspended) ? 1 : 0;
}


extern "C" void ds4_gpu_model_residency_skip(int skip) {
    (void)skip;
}

/* DS4-TP-gfx1151 (patch 4): these three now live in the TP runtime above. */

/* DS4-TP-gfx1151 (patch 10): implemented in rocm/ds4_rocm_matmul.cuh */

/* DS4-TP-gfx1151 (patch 11): implemented in rocm/ds4_rocm_hc_output_launch.cuh */

/* DS4-TP-gfx1151 (patch 11): implemented in rocm/ds4_rocm_hc_output_launch.cuh */
