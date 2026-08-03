/* ------------------------------------------------------------------------
 * DS4-TP-gfx1151 (patch 4): ROCm tensor-parallel gate runtime.
 *
 * Replaces the "tensor parallelism is Metal-only" stubs with a real HIP
 * implementation, mirroring the Metal design documented at ds4_metal.m:8272:
 *
 *   kernels ahead of the gate leave a partial block output in a slab slot,
 *   the GPU signals arrival, a service thread runs the transport exchange,
 *   the CPU signals release, and the pre-encoded combine kernel waits on it.
 *
 * HOW THE METAL PRIMITIVES MAP TO HIP
 *   MTLSharedEvent GPU-signal  ->  hipStreamWriteValue64(stream, addr, seq)
 *   MTLSharedEvent CPU-signal  ->  plain host store (UMA: GPU sees it)
 *   encoder wait on event      ->  hipStreamWaitValue64(..., WaitValueGte)
 *   "close the current encoder" ->  implicit: HIP stream ordering already
 *                                   serialises the write/wait against the
 *                                   compute kernels queued before/after.
 *
 * WHY THE DEFAULT STREAM. Every compute kernel in this backend launches as
 * <<<grid, block, shmem>>> with no stream argument, i.e. the null stream (see
 * rocm/ds4_rocm_matmul.cuh:15). Fencing anywhere else would not order against
 * them.
 *
 * WHY 64-BIT. Sequence numbers are monotonic for the life of the process. At
 * ~87 gates/token and ~10 tok/s a 32-bit counter wraps in about 57 days of
 * continuous decoding; a wrap would make WaitValueGte compare against a
 * smaller number and hang. 64-bit removes the failure mode rather than
 * documenting it.
 *
 * WHY THE RELEASE WORD IS NOT IN THE SLAB. The slab's GPU-flag region
 * (ds4_tp_slab_gpu_flags_offset) is defined as GPU-WRITTEN arrival flags and
 * is part of the wire layout both ranks agree on. The CPU->GPU release is
 * local to this rank and must not be confused with peer-visible state, so it
 * gets its own hipHostMalloc'd word.
 *
 * DEADLOCK SAFETY, taken from Metal verbatim in spirit: the release is
 * signalled even when the exchange FAILS. A GPU blocked in WaitValue cannot
 * be interrupted, so failing without releasing would hang the process. The
 * failure latches in g_tp_failed and the eval aborts at the next boundary.
 * ------------------------------------------------------------------------ */

#include <pthread.h>

static uint32_t                 g_tp_rank        = 0;
static unsigned char           *g_tp_slab_base   = NULL;
static volatile uint64_t       *g_tp_gpu_flag    = NULL;  /* GPU writes arrival seq */
static volatile uint64_t       *g_tp_cpu_flag    = NULL;  /* service thread writes release seq */
static ds4_gpu_tp_exchange_fn   g_tp_fn          = NULL;
static void                    *g_tp_ud          = NULL;
static pthread_t                g_tp_thread;
static volatile int             g_tp_thread_live = 0;
static volatile int             g_tp_run         = 0;
static volatile int             g_tp_failed      = 0;
static uint64_t                 g_tp_seq         = 0;      /* encoder-side, single-threaded */
static int                      g_tp_session_batch = 0;
static int                      g_tp_expert_shard_suspended = 0;
static int                      g_tp_keepalive_paused = 0;
static int                      g_tp_attn_head_split = 0;
static ds4_gpu_tp_batch_exchange_fn g_tp_batch_fn = NULL;
static ds4_gpu_tp_big_exchange_fn   g_tp_big_fn   = NULL;

/* Per-gate identity, recorded by the encoder so the service thread can pass
 * layer/gate back through the exchange callback. Indexed by seq % depth; the
 * stream executes gates in order, so depth 2 would suffice - 64 is slack for
 * session-batch mode without being a real cost. */
#define DS4_TP_RING 64
static struct { uint32_t layer, gate; } g_tp_ring[DS4_TP_RING];

static void *ds4_tp_service_thread(void *arg) {
    (void)arg;
    uint64_t next = 1;
    while (g_tp_run) {
        /* UMA: the slab is host-visible, so this is an ordinary load of memory
         * the GPU wrote. __atomic_load_n gives us the acquire ordering that
         * makes the payload written before the flag visible to this thread. */
        uint64_t arrived = __atomic_load_n(g_tp_gpu_flag, __ATOMIC_ACQUIRE);
        if (arrived < next) {
            sched_yield();
            continue;
        }
        uint32_t layer = g_tp_ring[next % DS4_TP_RING].layer;
        uint32_t gate  = g_tp_ring[next % DS4_TP_RING].gate;
        int ok = g_tp_fn ? g_tp_fn(g_tp_ud, layer, gate, next) : 0;
        if (!ok) {
            g_tp_failed = 1;
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "TP exchange failed at layer %u gate %u seq %llu; "
                    "releasing anyway so the GPU does not hang\n",
                    layer, gate, (unsigned long long)next);
        }
        /* Release even on failure - see the deadlock note above. */
        __atomic_store_n(g_tp_cpu_flag, next, __ATOMIC_RELEASE);
        next++;
    }
    return NULL;
}

extern "C" int ds4_gpu_tp_init(uint32_t rank,
                               ds4_gpu_tensor *slab, uint64_t gpu_flags_off,
                               ds4_gpu_tp_exchange_fn fn, void *ud) {
    if (!slab || !slab->ptr || !fn) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tp_init: bad arguments\n");
        return 0;
    }
    /* The whole design rests on the device supporting stream wait-on-value.
     * Check it rather than discovering a hang at the first gate. */
    int can_wait = 0, dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        hipDeviceGetAttribute(&can_wait, hipDeviceAttributeCanUseStreamWaitValue, dev)
            != hipSuccess || !can_wait) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "tp_init: device %d does not support hipStreamWaitValue "
                "(attr=%d); TP gates cannot fence\n", dev, can_wait);
        return 0;
    }
    g_tp_rank      = rank;
    g_tp_slab_base = (unsigned char *)slab->ptr;
    g_tp_gpu_flag  = (volatile uint64_t *)(g_tp_slab_base + gpu_flags_off);
    g_tp_fn        = fn;
    g_tp_ud        = ud;
    g_tp_seq       = 0;
    g_tp_failed    = 0;

    void *rel = NULL;
    if (hipHostMalloc(&rel, sizeof(uint64_t), hipHostMallocMapped) != hipSuccess || !rel) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tp_init: release word alloc failed\n");
        return 0;
    }
    g_tp_cpu_flag = (volatile uint64_t *)rel;
    __atomic_store_n(g_tp_cpu_flag, 0, __ATOMIC_RELEASE);
    __atomic_store_n(g_tp_gpu_flag, 0, __ATOMIC_RELEASE);
    memset(g_tp_ring, 0, sizeof(g_tp_ring));

    g_tp_run = 1;
    if (pthread_create(&g_tp_thread, NULL, ds4_tp_service_thread, NULL) != 0) {
        g_tp_run = 0;
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tp_init: service thread failed\n");
        return 0;
    }
    g_tp_thread_live = 1;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "TP rank %u ready (HIP stream wait-value)\n", rank);
    return 1;
}

extern "C" void ds4_gpu_tp_shutdown(void) {
    if (!g_tp_thread_live) return;
    g_tp_run = 0;
    pthread_join(g_tp_thread, NULL);
    g_tp_thread_live = 0;
    if (g_tp_cpu_flag) { hipHostFree((void *)g_tp_cpu_flag); g_tp_cpu_flag = NULL; }
    g_tp_fn = NULL; g_tp_ud = NULL; g_tp_slab_base = NULL; g_tp_gpu_flag = NULL;
}

/* Encode one gate on the null stream: publish arrival, then block until the
 * service thread releases. Ordering against the surrounding compute kernels is
 * the stream's job. */
static int ds4_tp_encode_common(uint32_t layer, uint32_t gate) {
    if (!g_tp_thread_live) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "TP gate encoded before tp_init\n");
        return 0;
    }
    if (g_tp_failed) return 0;
    uint64_t seq = ++g_tp_seq;
    g_tp_ring[seq % DS4_TP_RING].layer = layer;
    g_tp_ring[seq % DS4_TP_RING].gate  = gate;

    if (hipStreamWriteValue64(0, (void *)g_tp_gpu_flag, (int64_t)seq, 0) != hipSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "TP gate: stream write failed\n");
        g_tp_failed = 1;
        return 0;
    }
    if (hipStreamWaitValue64(0, (void *)g_tp_cpu_flag, (int64_t)seq,
                             hipStreamWaitValueGte, ~0ULL) != hipSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "TP gate: stream wait failed\n");
        g_tp_failed = 1;
        return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_tp_gate_encode(uint32_t layer, uint32_t gate) {
    return ds4_tp_encode_common(layer, gate);
}

extern "C" int ds4_gpu_tp_batch_gate_encode(uint32_t layer, uint32_t rows) {
    (void)rows;
    return ds4_tp_encode_common(layer, 0u);
}

extern "C" int ds4_gpu_tp_big_gate_encode(uint32_t layer, uint32_t rows,
                                          const ds4_gpu_tensor *out_t,
                                          ds4_gpu_tensor *in_t,
                                          uint64_t bytes) {
    (void)rows; (void)out_t; (void)in_t; (void)bytes;
    return ds4_tp_encode_common(layer, 0u);
}

extern "C" void ds4_gpu_tp_set_batch_exchange(ds4_gpu_tp_batch_exchange_fn fn) { g_tp_batch_fn = fn; }
extern "C" void ds4_gpu_tp_set_big_exchange(ds4_gpu_tp_big_exchange_fn fn)     { g_tp_big_fn = fn; }
extern "C" void ds4_gpu_tp_set_session_batch_mode(int enabled)                 { g_tp_session_batch = enabled; }
extern "C" void ds4_gpu_tp_suspend_expert_sharding(int suspend)                { g_tp_expert_shard_suspended = suspend; }
extern "C" void ds4_gpu_tp_keepalive_pause(int paused)                         { g_tp_keepalive_paused = paused; }
extern "C" void ds4_gpu_tp_set_attn_head_split(int enabled)                    { g_tp_attn_head_split = enabled; }

/* Latched failure, polled by the engine at command-buffer boundaries
 * (ds4.c:59903). The service thread sets it when an exchange fails; it still
 * releases the GPU in that case, so this is how the failure actually surfaces
 * rather than as a hang. */
extern "C" int ds4_gpu_tp_failed(void) { return g_tp_failed; }
