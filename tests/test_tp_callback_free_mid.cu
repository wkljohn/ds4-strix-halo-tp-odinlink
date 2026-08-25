/* Compare the rejected row-shard MOE_MID host callback with a persistent
 * producer-event transport worker over DS4's real RDMA decode slab. */
#include <hip/hip_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include <thread>
extern "C" {
#include "ds4_tp.h"
}

static constexpr uint32_t kEmbd = 4096u;
static constexpr uint32_t kThreads = 256u;

static uint64_t now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static void hip_check(hipError_t rc, const char *what) {
    if (rc == hipSuccess) return;
    std::fprintf(stderr, "FAIL %s: %s\n", what, hipGetErrorString(rc));
    std::exit(1);
}

__global__ static void fill_payload(float *out, uint32_t rank, uint64_t seq) {
    const float base = (float)(rank * 100000u + (uint32_t)seq);
    for (uint32_t i = threadIdx.x; i < kEmbd; i += blockDim.x)
        out[i] = base + (float)(i & 63u);
}
__global__ static void verify_payload(const float *in, uint32_t peer,
                                      uint64_t seq, uint32_t *error) {
    const float base = (float)(peer * 100000u + (uint32_t)seq);
    for (uint32_t i = threadIdx.x; i < kEmbd; i += blockDim.x)
        if (in[i] != base + (float)(i & 63u)) atomicCAS(error, 0u, 2u);
}
__global__ static void wait_and_verify_payload(
        const float *in, const volatile uint64_t *completion,
        uint32_t peer, uint64_t seq, uint32_t *error) {
    __shared__ uint32_t ready;
    if (threadIdx.x == 0) {
        uint64_t seen;
        do {
            seen = *completion;
            __builtin_amdgcn_s_sleep(1);
        } while (seen < seq);
        if (seen == UINT64_MAX) atomicCAS(error, 0u, 1u);
        __threadfence_system();
        ready = *error == 0u;
    }
    __syncthreads();
    if (!ready) return;
    const float base = (float)(peer * 100000u + (uint32_t)seq);
    for (uint32_t i = threadIdx.x; i < kEmbd; i += blockDim.x)
        if (in[i] != base + (float)(i & 63u)) atomicCAS(error, 0u, 2u);
}

struct callback_job {
    ds4_tp *tp;
    uint32_t gate;
    uint64_t seq;
    std::atomic<int> ok{0};
    uint64_t elapsed_ns = 0;
};
static void exchange_callback(void *opaque) {
    auto *job = static_cast<callback_job *>(opaque);
    const uint64_t begin = now_ns();
    const int ok = ds4_tp_gate_exchange(job->tp, 0u, job->gate, job->seq);
    job->elapsed_ns = now_ns() - begin;
    job->ok.store(ok, std::memory_order_release);
}

struct async_mid_worker {
    ds4_tp *tp = nullptr;
    hipEvent_t producer = nullptr;
    volatile uint64_t *completion = nullptr;
    std::thread thread;
    /* 0=startup, 1=pending, 2=done/idle, 3=stop.  Production already owns a
     * polling transport service thread; this mailbox measures that topology
     * instead of adding a condition-variable wake to every layer. */
    std::atomic<uint32_t> state{0};
    bool wait_event = true;
    uint64_t seq = 0, queued_ns = 0;
    uint64_t event_wait_ns = 0, exchange_ns = 0, release_ns = 0;
    int ok = 0;
};
static void async_worker_main(async_mid_worker *w) {
    w->state.store(2u, std::memory_order_release);
    uint32_t idle = 0;
    for (;;) {
        uint32_t state = w->state.load(std::memory_order_acquire);
        if (state == 3u) return;
        if (state != 1u) {
            for (int i = 0; i < 64; ++i) __builtin_ia32_pause();
            if (++idle == 1024u) { idle = 0u; sched_yield(); }
            continue;
        }
        idle = 0u;
        const uint64_t seq = w->seq, queued_ns = w->queued_ns;
        const uint64_t wait_begin = now_ns();
        hipError_t event_rc = hipSuccess;
        if (w->wait_event) {
            event_rc = hipErrorNotReady;
            while (event_rc == hipErrorNotReady) {
                event_rc = hipEventQuery(w->producer);
                if (event_rc == hipErrorNotReady)
                    for (int i = 0; i < 64; ++i) __builtin_ia32_pause();
                if (now_ns() - wait_begin > UINT64_C(5000000000)) {
                    event_rc = hipErrorUnknown;
                    break;
                }
            }
        }
        const uint64_t event_done = now_ns();
        int ok = 0;
        uint64_t exchange_done = event_done;
        if (event_rc == hipSuccess) {
            ok = ds4_tp_gate_exchange(w->tp, 0u, 0u, seq);
            exchange_done = now_ns();
        }
        if (ok) {
            std::atomic_thread_fence(std::memory_order_release);
            __atomic_store_n(w->completion, seq, __ATOMIC_RELEASE);
        } else {
            __atomic_store_n(w->completion, UINT64_MAX, __ATOMIC_RELEASE);
        }
        w->event_wait_ns = event_done - queued_ns;
        w->exchange_ns = exchange_done - event_done;
        w->release_ns = exchange_done - queued_ns;
        w->ok = ok;
        w->state.store(2u, std::memory_order_release);
    }
}
static void async_submit(async_mid_worker *w, uint64_t seq, bool wait_event) {
    if (w->state.load(std::memory_order_acquire) != 2u) {
        std::fprintf(stderr, "FAIL async worker reused before completion\n");
        std::exit(1);
    }
    w->seq = seq;
    w->wait_event = wait_event;
    w->queued_ns = now_ns();
    w->ok = 0;
    w->state.store(1u, std::memory_order_release);
}

struct handoff_job {
    async_mid_worker *worker;
    uint64_t seq;
};
static void async_handoff_callback(void *opaque) {
    auto *job = static_cast<handoff_job *>(opaque);
    async_submit(job->worker, job->seq, false);
}
static int async_wait(async_mid_worker *w, uint64_t *event_wait_ns,
                      uint64_t *exchange_ns, uint64_t *release_ns) {
    const uint64_t deadline = now_ns() + UINT64_C(6000000000);
    uint32_t idle = 0;
    while (w->state.load(std::memory_order_acquire) != 2u) {
        if (now_ns() >= deadline) return 0;
        for (int i = 0; i < 64; ++i) __builtin_ia32_pause();
        if (++idle == 1024u) { idle = 0u; sched_yield(); }
    }
    *event_wait_ns = w->event_wait_ns;
    *exchange_ns = w->exchange_ns;
    *release_ns = w->release_ns;
    return w->ok;
}

struct arm_stats {
    uint64_t elapsed_ns = 0, mid_ns = 0, mid_event_wait_ns = 0, final_ns = 0;
    uint64_t errors = 0;
};
static void clear_error(uint32_t *p) {
    hip_check(hipMemset(p, 0, sizeof(*p)), "clear error");
}
static void slab_views(ds4_tp *tp, void *slab, uint32_t gate,
                       float **out, const float **in) {
    *out = (float *)((char *)slab + ds4_tp_slab_out_offset(tp, 0u, gate));
    *in = (const float *)((char *)slab + ds4_tp_slab_in_offset(tp, 0u, gate));
}

static arm_stats run_callback_arm(ds4_tp *tp, void *slab, uint32_t rank,
                                  uint64_t iterations, uint64_t &next_seq,
                                  uint32_t *error) {
    arm_stats s;
    const uint32_t peer = rank ^ 1u;
    const uint64_t begin = now_ns();
    for (uint64_t i = 0; i < iterations; ++i) {
        const uint64_t mid_seq = next_seq++, final_seq = next_seq++;
        float *mid_out, *final_out;
        const float *mid_in, *final_in;
        slab_views(tp, slab, 0u, &mid_out, &mid_in);
        slab_views(tp, slab, 1u, &final_out, &final_in);
        clear_error(error);
        fill_payload<<<1, kThreads>>>(mid_out, rank, mid_seq);
        callback_job mid{tp, 0u, mid_seq};
        hip_check(hipLaunchHostFunc(nullptr, exchange_callback, &mid),
                  "enqueue MID callback");
        verify_payload<<<1, kThreads>>>(mid_in, peer, mid_seq, error);
        fill_payload<<<1, kThreads>>>(final_out, rank, final_seq);
        callback_job final{tp, 1u, final_seq};
        hip_check(hipLaunchHostFunc(nullptr, exchange_callback, &final),
                  "enqueue FFN callback");
        verify_payload<<<1, kThreads>>>(final_in, peer, final_seq, error);
        hip_check(hipDeviceSynchronize(), "callback arm sync");
        if (!mid.ok.load(std::memory_order_acquire) ||
            !final.ok.load(std::memory_order_acquire)) s.errors++;
        s.mid_ns += mid.elapsed_ns;
        s.final_ns += final.elapsed_ns;
        uint32_t code = 0;
        hip_check(hipMemcpy(&code, error, sizeof(code), hipMemcpyDeviceToHost),
                  "callback error read");
        if (code) s.errors++;
        if (s.errors) break;
    }
    s.elapsed_ns = now_ns() - begin;
    return s;
}

static arm_stats run_async_arm(ds4_tp *tp, void *slab, uint32_t rank,
                               uint64_t iterations, uint64_t &next_seq,
                               uint32_t *error, async_mid_worker *worker,
                               const volatile uint64_t *device_completion,
                               bool callback_handoff) {
    arm_stats s;
    const uint32_t peer = rank ^ 1u;
    const uint64_t begin = now_ns();
    for (uint64_t i = 0; i < iterations; ++i) {
        const uint64_t mid_seq = next_seq++, final_seq = next_seq++;
        float *mid_out, *final_out;
        const float *mid_in, *final_in;
        slab_views(tp, slab, 0u, &mid_out, &mid_in);
        slab_views(tp, slab, 1u, &final_out, &final_in);
        clear_error(error);
        fill_payload<<<1, kThreads>>>(mid_out, rank, mid_seq);
        handoff_job handoff{worker, mid_seq};
        if (callback_handoff) {
            hip_check(hipLaunchHostFunc(nullptr, async_handoff_callback,
                                       &handoff),
                      "enqueue MID handoff callback");
        } else {
            hip_check(hipEventRecord(worker->producer, nullptr),
                      "record MID producer event");
            async_submit(worker, mid_seq, true);
        }
        wait_and_verify_payload<<<1, kThreads>>>(
            mid_in, device_completion, peer, mid_seq, error);
        fill_payload<<<1, kThreads>>>(final_out, rank, final_seq);
        callback_job final{tp, 1u, final_seq};
        hip_check(hipLaunchHostFunc(nullptr, exchange_callback, &final),
                  "enqueue async FFN callback");
        verify_payload<<<1, kThreads>>>(final_in, peer, final_seq, error);
        hip_check(hipDeviceSynchronize(), "async arm sync");
        uint64_t event_ns = 0, exchange_ns = 0, release_ns = 0;
        if (!async_wait(worker, &event_ns, &exchange_ns, &release_ns) ||
            !final.ok.load(std::memory_order_acquire)) s.errors++;
        s.mid_event_wait_ns += event_ns;
        s.mid_ns += exchange_ns;
        s.final_ns += final.elapsed_ns;
        uint32_t code = 0;
        hip_check(hipMemcpy(&code, error, sizeof(code), hipMemcpyDeviceToHost),
                  "async error read");
        if (code) s.errors++;
        if (s.errors) break;
    }
    s.elapsed_ns = now_ns() - begin;
    return s;
}

static void usage(const char *p) {
    std::fprintf(stderr, "usage: %s leader|worker ADDRESS PORT RDMA_DEVICE "
                         "GID_INDEX_OR_-1 [ITERATIONS=4300]\n", p);
}

int main(int argc, char **argv) {
    if (argc < 6 || argc > 7 ||
        (std::strcmp(argv[1], "leader") && std::strcmp(argv[1], "worker"))) {
        usage(argv[0]);
        return 2;
    }
    const bool leader = std::strcmp(argv[1], "leader") == 0;
    const uint32_t rank = leader ? 0u : 1u;
    const char *address = argv[2];
    const int port = std::atoi(argv[3]);
    const char *device = argv[4];
    const int gid = std::atoi(argv[5]);
    const uint64_t iterations = argc == 7 ? std::strtoull(argv[6], nullptr, 10)
                                          : UINT64_C(4300);
    if (port <= 0 || port > 65535 || !device[0] || !iterations) {
        usage(argv[0]);
        return 2;
    }
    (void)setenv("DS4_TP_BIG_DIRECT_MAX_ROWS", "1", 0);
    ds4_tp_options opt{};
    opt.role = leader ? DS4_TP_LEADER : DS4_TP_WORKER;
    opt.requested = true;
    opt.transport = DS4_TP_TRANSPORT_RDMA;
    opt.rdma_device = device;
    opt.rdma_gid_index = gid;
    opt.rdma_gid_index_set = gid >= 0;
    if (leader) { opt.listen_host = address; opt.listen_port = port; }
    else { opt.leader_host = address; opt.leader_port = port; }
    ds4_tp_identity id{};
    id.gguf_bytes = 1u;
    id.model_id = 0x4d494444u;
    id.n_layer = 1u;
    id.n_embd = kEmbd;
    id.n_vocab = 1u;
    id.quant_bits = 4u;
    id.ctx_size = 4096u;
    id.gate_slot_start = 0u;
    id.gate_slot_step = 1u;
    id.gates_per_token = 2u;
    char err[512]{};
    ds4_tp *tp = nullptr;
    if (!ds4_tp_create(&tp, &opt, &id, err, sizeof(err))) {
        std::fprintf(stderr, "FAIL create: %s\n", err);
        return 1;
    }
    if (!ds4_tp_is_rdma(tp)) {
        std::fprintf(stderr, "FAIL explicit RDMA unavailable\n");
        return 1;
    }
    const uint64_t slab_bytes = ds4_tp_alloc_slab_bytes(tp);
    const bool mapped = ds4_tp_requires_host_slab(tp);
    void *host_slab = nullptr, *device_slab = nullptr;
    if (mapped) {
        hip_check(hipHostMalloc(&host_slab, (size_t)slab_bytes,
                                hipHostMallocMapped), "host slab alloc");
        hip_check(hipHostGetDevicePointer(&device_slab, host_slab, 0),
                  "host slab device pointer");
    } else {
        hip_check(hipMalloc(&device_slab, (size_t)slab_bytes), "device slab alloc");
        host_slab = device_slab;
    }
    hip_check(hipMemset(device_slab, 0, (size_t)slab_bytes), "clear slab");
    hip_check(hipDeviceSynchronize(), "clear slab sync");
    if (!ds4_tp_attach_slab(tp, host_slab, err, sizeof(err))) {
        std::fprintf(stderr, "FAIL attach slab: %s\n", err);
        return 1;
    }
    void *completion_host = nullptr, *completion_device = nullptr;
    hip_check(hipHostMalloc(&completion_host, 64u, hipHostMallocMapped),
              "completion alloc");
    hip_check(hipHostGetDevicePointer(&completion_device, completion_host, 0),
              "completion device pointer");
    std::memset(completion_host, 0, 64u);
    uint32_t *device_error = nullptr;
    hip_check(hipMalloc(&device_error, sizeof(*device_error)), "error alloc");
    async_mid_worker worker;
    worker.tp = tp;
    worker.completion = (volatile uint64_t *)completion_host;
    hip_check(hipEventCreateWithFlags(&worker.producer, hipEventDisableTiming),
              "producer event create");
    worker.thread = std::thread(async_worker_main, &worker);

    uint64_t next_seq = 1u;
    const arm_stats callback = run_callback_arm(
        tp, device_slab, rank, iterations, next_seq, device_error);
    const arm_stats async = run_async_arm(
        tp, device_slab, rank, iterations, next_seq, device_error, &worker,
        (const volatile uint64_t *)completion_device, false);
    const arm_stats handoff = run_async_arm(
        tp, device_slab, rank, iterations, next_seq, device_error, &worker,
        (const volatile uint64_t *)completion_device, true);
    worker.state.store(3u, std::memory_order_release);
    worker.thread.join();
    hip_check(hipEventDestroy(worker.producer), "producer event destroy");

    const double callback_us = (double)callback.elapsed_ns / iterations / 1000.0;
    const double async_us = (double)async.elapsed_ns / iterations / 1000.0;
    const double saved_us = callback_us - async_us;
    const double handoff_us = (double)handoff.elapsed_ns / iterations / 1000.0;
    const double handoff_saved_us = callback_us - handoff_us;
    const bool pass = callback.errors == 0u && async.errors == 0u &&
                      handoff.errors == 0u && handoff_saved_us >= 10.0;
    std::printf(
        "TP_CALLBACK_FREE_MID rank=%u provider=%s allocator=%s iterations=%llu "
        "callback_us_per_layer=%.3f async_us_per_layer=%.3f saved_us=%.3f "
        "handoff_us_per_layer=%.3f handoff_saved_us=%.3f "
        "callback_mid_exchange_us=%.3f async_mid_exchange_us=%.3f "
        "async_event_to_ready_us=%.3f callback_final_exchange_us=%.3f "
        "async_final_exchange_us=%.3f handoff_mid_exchange_us=%.3f "
        "handoff_submit_to_ready_us=%.3f handoff_final_exchange_us=%.3f "
        "callback_errors=%llu async_errors=%llu handoff_errors=%llu "
        "verdict=%s\n",
        rank, device, mapped ? "hipHostMallocMapped" : "hipMalloc",
        (unsigned long long)iterations, callback_us, async_us, saved_us,
        handoff_us, handoff_saved_us,
        (double)callback.mid_ns / iterations / 1000.0,
        (double)async.mid_ns / iterations / 1000.0,
        (double)async.mid_event_wait_ns / iterations / 1000.0,
        (double)callback.final_ns / iterations / 1000.0,
        (double)async.final_ns / iterations / 1000.0,
        (double)handoff.mid_ns / iterations / 1000.0,
        (double)handoff.mid_event_wait_ns / iterations / 1000.0,
        (double)handoff.final_ns / iterations / 1000.0,
        (unsigned long long)callback.errors,
        (unsigned long long)async.errors,
        (unsigned long long)handoff.errors, pass ? "GO" : "STOP");

    (void)hipFree(device_error);
    (void)hipHostFree(completion_host);
    ds4_tp_free(tp);
    if (mapped) (void)hipHostFree(host_slab); else (void)hipFree(device_slab);
    return pass ? 0 : 1;
}
