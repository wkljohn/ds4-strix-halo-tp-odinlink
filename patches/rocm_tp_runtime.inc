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
 * STILL OPEN: whether a PAYLOAD written by preceding kernels is visible when
 * the flag arrives. Metal measured stale rows here and switched big-gate
 * arrival to a shared event for exactly that reason (ds4_metal.m:8776-8783).
 * The T2 loop test must settle this before any correctness claim.
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
    const ds4_gpu_tensor *out_t;
    ds4_gpu_tensor       *in_t;
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
static int                      g_tp_keepalive_paused = 0;
static int                      g_tp_attn_head_split = 0;
static void                    *g_tp_sig_alloc = NULL;
static int                      g_tp_sig_is_host = 0;

static int ds4_tp_fail_get(void) { return __atomic_load_n(&g_tp_failed, __ATOMIC_ACQUIRE); }
static void ds4_tp_fail_set(void) { __atomic_store_n(&g_tp_failed, 1, __ATOMIC_RELEASE); }

/* Drive one channel forward if the GPU has reached `next`. Returns 1 if it
 * processed a gate. */
static int ds4_tp_pump(int ch, uint64_t *next) {
    struct ds4_tp_chan *c = &g_tp_chan[ch];
    uint64_t arrived = __atomic_load_n(c->gpu_flag, __ATOMIC_ACQUIRE);
    if (arrived < *next) return 0;

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
         * UMA APU, so ->ptr is directly usable. */
        ok = (g_tp_big_fn && r->out_t && r->in_t)
             ? g_tp_big_fn(g_tp_ud, r->layer, *next,
                           (const void *)r->out_t->ptr,
                           (void *)r->in_t->ptr, r->bytes)
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

static void *ds4_tp_service_thread(void *arg) {
    (void)arg;
    uint64_t next[2] = { 1, 1 };
    while (g_tp_run) {
        int did = ds4_tp_pump(0, &next[0]) | ds4_tp_pump(1, &next[1]);
        /* Metal tight-spins before yielding: "yielding here measurably delays
         * the release wake-up" (ds4_metal.m:8492-8496). */
        if (!did) {
            for (int i = 0; i < (1 << 14); i++) __builtin_ia32_pause();
            if (!__atomic_load_n(g_tp_chan[0].gpu_flag, __ATOMIC_ACQUIRE) &&
                !__atomic_load_n(g_tp_chan[1].gpu_flag, __ATOMIC_ACQUIRE))
                sched_yield();
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
    return NULL;
}

extern "C" int ds4_gpu_tp_init(uint32_t rank,
                               ds4_gpu_tensor *slab, uint64_t gpu_flags_off,
                               ds4_gpu_tp_exchange_fn fn, void *ud) {
    (void)rank;
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

    g_tp_run = 1;
    if (pthread_create(&g_tp_thread, NULL, ds4_tp_service_thread, NULL) != 0) {
        g_tp_run = 0;
        if (g_tp_sig_is_host) hipHostFree(sig); else hipFree(sig);
        g_tp_sig_alloc = NULL;
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tp_init: service thread failed\n");
        return 0;
    }
    g_tp_thread_live = 1;
    fprintf(stderr, DS4_GPU_LOG_PREFIX
            "ROCm TP rank %u ready (2 channels, HIP stream wait-value)\n", rank);
    return 1;
}

extern "C" void ds4_gpu_tp_shutdown(void) {
    if (!g_tp_thread_live) return;
    g_tp_run = 0;
    pthread_join(g_tp_thread, NULL);        /* drains before returning */
    hipDeviceSynchronize();                 /* no pending WaitValue on the word */
    g_tp_thread_live = 0;
    if (g_tp_sig_alloc) {
        if (g_tp_sig_is_host) hipHostFree(g_tp_sig_alloc); else hipFree(g_tp_sig_alloc);
        g_tp_sig_alloc = NULL;
    }
    g_tp_fn = NULL; g_tp_ud = NULL;
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

    if (hipStreamWriteValue64(0, (void *)c->gpu_flag, (int64_t)seq, 0) != hipSuccess ||
        hipStreamWaitValue64(0, (void *)c->cpu_flag, (int64_t)seq,
                             hipStreamWaitValueGte, ~0ULL) != hipSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "TP gate: stream write/wait failed\n");
        ds4_tp_fail_set();
        return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_tp_gate_encode(uint32_t layer, uint32_t gate) {
    struct ds4_tp_req r = {}; r.kind = DS4_TP_ROW; r.layer = layer; r.gate = gate;
    return ds4_tp_encode(0, &r);
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
    r.out_t = out_t; r.in_t = in_t; r.bytes = bytes;
    return ds4_tp_encode(1, &r);
}

extern "C" void ds4_gpu_tp_set_batch_exchange(ds4_gpu_tp_batch_exchange_fn fn) { g_tp_batch_fn = fn; }
extern "C" void ds4_gpu_tp_set_big_exchange(ds4_gpu_tp_big_exchange_fn fn)     { g_tp_big_fn = fn; }
extern "C" void ds4_gpu_tp_set_session_batch_mode(int enabled)                 { g_tp_session_batch = enabled; }
extern "C" void ds4_gpu_tp_suspend_expert_sharding(int suspend)                { g_tp_expert_shard_suspended = suspend; }
extern "C" void ds4_gpu_tp_keepalive_pause(int paused)                         { g_tp_keepalive_paused = paused; }
extern "C" void ds4_gpu_tp_set_attn_head_split(int enabled)                    { g_tp_attn_head_split = enabled; }

/* Latched failure, polled by the engine at command-buffer boundaries
 * (ds4.c:59903). */
extern "C" int ds4_gpu_tp_failed(void) { return ds4_tp_fail_get(); }
