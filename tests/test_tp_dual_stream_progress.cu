/* Watchdog-bounded two-node reproducer for the rejected decode overlap.
 *
 * The legacy topology was:
 *   - routed work on the legacy default stream;
 *   - shared work on a nonblocking stream, joined by a host stream wait;
 *   - arrival/release words on a blocking gate stream;
 *   - one real RDMA row exchange per gate.
 *
 * Sustained inference lost progress at gate sequences 1306 and 1482.  This
 * oracle recreates that ordering without loading a model.  If an arrival word
 * does not appear by the per-gate deadline, the service thread still performs
 * the symmetric exchange and publishes the release word.  The test therefore
 * records the failure without leaving a GPU wait packet or peer QP stranded.
 * It is diagnostic-only and is not part of the ordinary test suite.
 */

#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

extern "C" {
#include "ds4_tp.h"
}

static constexpr uint32_t kEmbd = 4096u;
static constexpr uint64_t kSharedValues = 2u * 1024u * 1024u;
static constexpr uint64_t kRoutedValues = 6u * 1024u * 1024u;
static constexpr uint32_t kThreads = 256u;

static void usage(const char *argv0) {
    std::fprintf(stderr,
        "usage: %s leader|worker ADDRESS PORT RDMA_DEVICE GID_INDEX_OR_-1 "
        "[ITERATIONS=4000] [ARRIVAL_TIMEOUT_MS=250] [device|mapped]\n",
        argv0);
}

static void hip_check(hipError_t rc, const char *what) {
    if (rc == hipSuccess) return;
    std::fprintf(stderr, "FAIL %s: %s\n", what, hipGetErrorString(rc));
    std::exit(1);
}

__global__ static void init_values(float *values, uint64_t n, uint32_t salt) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) values[i] = (float)((i + salt) & 255u) * 0x1p-8f;
}

__global__ static void bandwidth_work(float *dst, const float *src,
                                      uint64_t n, float bias) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i] * 1.00000011920928955078125f + bias;
}

__global__ static void fill_payload(float *out, uint32_t rank, uint64_t seq) {
    const uint32_t i = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kEmbd) out[i] = (float)(rank * 100000u + (uint32_t)seq) +
                             (float)(i & 63u);
}

struct service_state {
    ds4_tp *tp = nullptr;
    volatile uint64_t *arrival = nullptr;
    volatile uint64_t *release = nullptr;
    uint64_t iterations = 0;
    uint64_t timeout_ns = 0;
    std::atomic<uint64_t> arrival_timeouts{0};
    std::atomic<uint64_t> first_timeout{0};
    std::atomic<uint64_t> transport_errors{0};
    std::atomic<uint64_t> completed{0};
};

static uint64_t monotonic_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void exchange_service(service_state *state) {
    for (uint64_t seq = 1; seq <= state->iterations; ++seq) {
        const uint64_t deadline = monotonic_ns() + state->timeout_ns;
        bool arrived = false;
        while (!(arrived = __atomic_load_n(state->arrival,
                                            __ATOMIC_ACQUIRE) >= seq)) {
            if (monotonic_ns() >= deadline) break;
            for (int i = 0; i < 64; ++i) __builtin_ia32_pause();
            std::this_thread::yield();
        }
        if (!arrived) {
            state->arrival_timeouts.fetch_add(1, std::memory_order_relaxed);
            uint64_t expected = 0;
            (void)state->first_timeout.compare_exchange_strong(
                expected, seq, std::memory_order_relaxed);
        }

        const uint32_t gate = (uint32_t)((seq - 1u) & 1u);
        if (!ds4_tp_gate_exchange(state->tp, 0u, gate, seq)) {
            state->transport_errors.fetch_add(1, std::memory_order_relaxed);
        }
        std::atomic_thread_fence(std::memory_order_release);
        __atomic_store_n(state->release, seq, __ATOMIC_RELEASE);
        state->completed.store(seq, std::memory_order_release);
        if (state->transport_errors.load(std::memory_order_relaxed)) return;
    }
}

int main(int argc, char **argv) {
    if (argc < 6 || argc > 9 ||
        (std::strcmp(argv[1], "leader") != 0 &&
         std::strcmp(argv[1], "worker") != 0)) {
        usage(argv[0]);
        return 2;
    }
    const bool leader = std::strcmp(argv[1], "leader") == 0;
    const char *address = argv[2];
    const int port = std::atoi(argv[3]);
    const char *device = argv[4];
    const int gid = std::atoi(argv[5]);
    const uint64_t iterations = argc >= 7 ? std::strtoull(argv[6], nullptr, 10)
                                           : UINT64_C(4000);
    const uint64_t timeout_ms = argc >= 8 ? std::strtoull(argv[7], nullptr, 10)
                                          : UINT64_C(250);
    const char *flag_allocator = argc >= 9 ? argv[8] : "device";
    const bool mapped_flag = std::strcmp(flag_allocator, "mapped") == 0;
    if (port <= 0 || port > 65535 || iterations < 1307u || timeout_ms == 0u ||
        (!mapped_flag && std::strcmp(flag_allocator, "device") != 0)) {
        usage(argv[0]);
        return 2;
    }

    ds4_tp_options opt{};
    opt.role = leader ? DS4_TP_LEADER : DS4_TP_WORKER;
    opt.requested = true;
    opt.transport = DS4_TP_TRANSPORT_RDMA;
    opt.rdma_device = device;
    opt.rdma_gid_index = gid;
    opt.rdma_gid_index_set = gid >= 0;
    if (leader) {
        opt.listen_host = address;
        opt.listen_port = port;
    } else {
        opt.leader_host = address;
        opt.leader_port = port;
    }

    ds4_tp_identity identity{};
    identity.gguf_bytes = UINT64_C(1);
    identity.model_id = UINT32_C(0x42505247);
    identity.n_layer = 1u;
    identity.n_embd = kEmbd;
    identity.n_vocab = 1u;
    identity.quant_bits = 4u;
    identity.ctx_size = 4096u;
    identity.gate_slot_start = 0u;
    identity.gate_slot_step = 1u;
    identity.gates_per_token = 2u;

    char error_text[256] = {};
    ds4_tp *tp = nullptr;
    if (!ds4_tp_create(&tp, &opt, &identity,
                       error_text, sizeof(error_text))) {
        std::fprintf(stderr, "FAIL create: %s\n", error_text);
        return 1;
    }
    if (!ds4_tp_is_rdma(tp)) {
        std::fprintf(stderr, "FAIL explicit RDMA unavailable\n");
        ds4_tp_free(tp);
        return 1;
    }

    const uint64_t slab_bytes = ds4_tp_alloc_slab_bytes(tp);
    void *slab_host = nullptr;
    void *slab_device = nullptr;
    const bool host_slab = ds4_tp_requires_host_slab(tp);
    if (host_slab) {
        hip_check(hipHostMalloc(&slab_host, (size_t)slab_bytes,
                                hipHostMallocMapped), "slab host alloc");
        hip_check(hipHostGetDevicePointer(&slab_device, slab_host, 0),
                  "slab device pointer");
    } else {
        hip_check(hipMalloc(&slab_device, (size_t)slab_bytes),
                  "slab device alloc");
        slab_host = slab_device;
    }
    hip_check(hipMemset(slab_device, 0, (size_t)slab_bytes), "clear slab");
    hip_check(hipDeviceSynchronize(), "clear slab sync");
    if (!ds4_tp_attach_slab(tp, slab_host, error_text, sizeof(error_text))) {
        std::fprintf(stderr, "FAIL attach slab: %s\n", error_text);
        return 1;
    }

    void *arrival_host = nullptr;
    void *arrival_device = nullptr;
    if (mapped_flag) {
        hip_check(hipHostMalloc(&arrival_host, 64u, hipHostMallocMapped),
                  "arrival mapped alloc");
        hip_check(hipHostGetDevicePointer(&arrival_device, arrival_host, 0),
                  "arrival mapped device pointer");
        std::memset(arrival_host, 0, 64u);
    } else {
        hip_check(hipMalloc(&arrival_device, 64u), "arrival device alloc");
        arrival_host = arrival_device; /* gfx1151 UMA CPU-visible device VA */
        hip_check(hipMemset(arrival_device, 0, 64u), "clear arrival");
    }
    void *release_host = nullptr;
    void *release_device = nullptr;
    hip_check(hipHostMalloc(&release_host, 64u, hipHostMallocMapped),
              "release mapped alloc");
    hip_check(hipHostGetDevicePointer(&release_device, release_host, 0),
              "release mapped device pointer");
    std::memset(release_host, 0, 64u);

    float *shared_src = nullptr, *shared_dst = nullptr;
    float *routed_src = nullptr, *routed_dst = nullptr;
    hip_check(hipMalloc(&shared_src, kSharedValues * sizeof(float)),
              "shared source alloc");
    hip_check(hipMalloc(&shared_dst, kSharedValues * sizeof(float)),
              "shared destination alloc");
    hip_check(hipMalloc(&routed_src, kRoutedValues * sizeof(float)),
              "routed source alloc");
    hip_check(hipMalloc(&routed_dst, kRoutedValues * sizeof(float)),
              "routed destination alloc");
    init_values<<<(uint32_t)((kSharedValues + kThreads - 1u) / kThreads),
                  kThreads>>>(shared_src, kSharedValues, 17u);
    init_values<<<(uint32_t)((kRoutedValues + kThreads - 1u) / kThreads),
                  kThreads>>>(routed_src, kRoutedValues, 31u);
    hip_check(hipDeviceSynchronize(), "initialize work buffers");

    hipStream_t gate_stream = nullptr, shared_stream = nullptr;
    hipEvent_t input_ready = nullptr;
    hip_check(hipStreamCreate(&gate_stream), "blocking gate stream");
    hip_check(hipStreamCreateWithFlags(&shared_stream, hipStreamNonBlocking),
              "nonblocking shared stream");
    hip_check(hipEventCreateWithFlags(&input_ready, hipEventDisableTiming),
              "input event");

    service_state state;
    state.tp = tp;
    state.arrival = (volatile uint64_t *)arrival_host;
    state.release = (volatile uint64_t *)release_host;
    state.iterations = iterations;
    state.timeout_ns = timeout_ms * UINT64_C(1000000);
    std::thread service(exchange_service, &state);

    const uint32_t rank = leader ? 0u : 1u;
    const auto begin = std::chrono::steady_clock::now();
    bool enqueue_ok = true;
    for (uint64_t seq = 1; seq <= iterations && enqueue_ok; ++seq) {
        const uint32_t gate = (uint32_t)((seq - 1u) & 1u);
        float *out = (float *)((char *)slab_device +
            ds4_tp_slab_out_offset(tp, 0u, gate));
        fill_payload<<<(kEmbd + kThreads - 1u) / kThreads, kThreads>>>(
            out, rank, seq);
        hipError_t err = hipGetLastError();
        if (err == hipSuccess) err = hipEventRecord(input_ready, nullptr);
        if (err == hipSuccess)
            err = hipStreamWaitEvent(shared_stream, input_ready, 0);
        if (err == hipSuccess) {
            bandwidth_work<<<
                (uint32_t)((kSharedValues + kThreads - 1u) / kThreads),
                kThreads, 0, shared_stream>>>(
                    shared_dst, shared_src, kSharedValues, (float)seq * 0x1p-24f);
            err = hipGetLastError();
        }
        if (err == hipSuccess) {
            bandwidth_work<<<
                (uint32_t)((kRoutedValues + kThreads - 1u) / kThreads),
                kThreads>>>(
                    routed_dst, routed_src, kRoutedValues, (float)seq * 0x1p-25f);
            err = hipGetLastError();
        }
        /* This host join is the exact old async-shared call shape. */
        if (err == hipSuccess) err = hipStreamSynchronize(shared_stream);
        if (err == hipSuccess)
            err = hipStreamWriteValue64(gate_stream, arrival_device,
                                        (int64_t)seq, 0);
        if (err == hipSuccess)
            err = hipStreamWaitValue64(gate_stream, release_device,
                                       (int64_t)seq,
                                       hipStreamWaitValueGte, ~UINT64_C(0));
        if (err != hipSuccess) {
            std::fprintf(stderr, "FAIL enqueue seq=%llu: %s\n",
                         (unsigned long long)seq, hipGetErrorString(err));
            enqueue_ok = false;
        }
    }
    hipError_t finish_err = hipDeviceSynchronize();
    service.join();
    const double elapsed_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();

    const uint64_t arrival_timeouts =
        state.arrival_timeouts.load(std::memory_order_relaxed);
    const uint64_t first_timeout =
        state.first_timeout.load(std::memory_order_relaxed);
    const uint64_t transport_errors =
        state.transport_errors.load(std::memory_order_relaxed);
    const uint64_t completed = state.completed.load(std::memory_order_relaxed);
    std::printf(
        "TP_DUAL_STREAM_PROGRESS rank=%u provider=%s flag_allocator=%s "
        "iterations=%llu completed=%llu arrival_timeouts=%llu "
        "first_timeout_seq=%llu transport_errors=%llu finish=%s "
        "elapsed_s=%.3f verdict=%s\n",
        rank, device, flag_allocator,
        (unsigned long long)iterations, (unsigned long long)completed,
        (unsigned long long)arrival_timeouts,
        (unsigned long long)first_timeout,
        (unsigned long long)transport_errors,
        hipGetErrorString(finish_err), elapsed_s,
        enqueue_ok && finish_err == hipSuccess && !transport_errors &&
                completed == iterations && arrival_timeouts == 0u
            ? "PROGRESS_PASS" :
        enqueue_ok && finish_err == hipSuccess && !transport_errors &&
                completed == iterations && arrival_timeouts != 0u
            ? "LEGACY_FAILURE_REPRODUCED" : "HARNESS_FAIL");

    (void)hipEventDestroy(input_ready);
    (void)hipStreamDestroy(shared_stream);
    (void)hipStreamDestroy(gate_stream);
    (void)hipFree(routed_dst);
    (void)hipFree(routed_src);
    (void)hipFree(shared_dst);
    (void)hipFree(shared_src);
    (void)hipHostFree(release_host);
    if (mapped_flag) (void)hipHostFree(arrival_host);
    else (void)hipFree(arrival_device);
    ds4_tp_free(tp);
    if (host_slab) (void)hipHostFree(slab_host);
    else (void)hipFree(slab_device);

    /* Both clean progress and a safely released legacy failure are useful
     * diagnostic outcomes. Transport errors or incomplete cleanup are not. */
    return enqueue_ok && finish_err == hipSuccess && !transport_errors &&
                   completed == iterations ? 0 : 1;
}
