/* Sustained two-node TP completion-word ordering oracle.
 *
 * It uses DS4's real control handshake, provider-selected QP, registered slab,
 * and decode SEND/RECV exchange.  After the receive CQE proves that the peer
 * payload landed, a transport thread publishes a monotonically increasing
 * CPU completion word in GPU-visible memory.  A GPU consumer already spinning
 * on that word verifies every float in the received payload.  This exercises
 * the exact direction needed by a future callback-free consumer gate while
 * avoiding the unreliable GPU-to-host stream-signal direction.
 *
 * Usage on both nodes (start leader first):
 *   test_tp_completion_ordering leader 0.0.0.0 5598 mlx5_0 3 25800
 *   test_tp_completion_ordering worker 192.168.99.1 5598 mlx5_1 3 25800
 * Use GID -1 for OdinLink and select its provider with DS4_TP_VERBS_LIB.
 */

#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

extern "C" {
#include "ds4_tp.h"
}

static constexpr uint32_t kEmbd = 4096u;
static constexpr uint32_t kThreads = 256u;

__global__ static void fill_payload(float *out, uint32_t rank, uint64_t seq) {
    const float base = (float)(rank * 100000u + (uint32_t)seq);
    for (uint32_t i = threadIdx.x; i < kEmbd; i += blockDim.x) {
        out[i] = base + (float)(i & 63u);
    }
}

__global__ static void wait_and_verify_payload(
        const float *in,
        const volatile uint64_t *completion,
        uint32_t peer_rank,
        uint64_t seq,
        uint32_t *error) {
    __shared__ uint32_t ready;
    if (threadIdx.x == 0) {
        uint64_t seen = 0;
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

    const float base = (float)(peer_rank * 100000u + (uint32_t)seq);
    for (uint32_t i = threadIdx.x; i < kEmbd; i += blockDim.x) {
        const float want = base + (float)(i & 63u);
        if (in[i] != want) atomicCAS(error, 0u, 2u);
    }
}

static void usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage: %s leader|worker ADDRESS PORT RDMA_DEVICE "
                 "GID_INDEX_OR_-1 [ITERATIONS=25800]\n",
                 argv0);
}

static void hip_check(hipError_t rc, const char *what) {
    if (rc != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(rc));
        std::exit(1);
    }
}

int main(int argc, char **argv) {
    if (argc < 6 || argc > 7 ||
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
    const uint64_t iterations = argc == 7 ? std::strtoull(argv[6], nullptr, 10)
                                          : UINT64_C(25800);
    if (port <= 0 || port > 65535 || !device[0] || iterations == 0) {
        usage(argv[0]);
        return 2;
    }

    /* Keep the oracle slab small; it needs only the ordinary one-row gates. */
    (void)setenv("DS4_TP_BIG_DIRECT_MAX_ROWS", "1", 0);
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
    identity.model_id = 4u;
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
        std::fprintf(stderr, "create failed: %s\n", error_text);
        return 1;
    }

    const uint64_t slab_bytes = ds4_tp_alloc_slab_bytes(tp);
    void *host_slab = nullptr;
    void *device_slab = nullptr;
    const bool mapped_host = ds4_tp_requires_host_slab(tp);
    if (mapped_host) {
        hip_check(hipHostMalloc(&host_slab, (size_t)slab_bytes,
                                hipHostMallocMapped), "hipHostMalloc slab");
        hip_check(hipHostGetDevicePointer(&device_slab, host_slab, 0),
                  "hipHostGetDevicePointer slab");
    } else {
        hip_check(hipMalloc(&device_slab, (size_t)slab_bytes), "hipMalloc slab");
        host_slab = device_slab;
    }
    hip_check(hipMemset(device_slab, 0, (size_t)slab_bytes), "clear slab");
    hip_check(hipDeviceSynchronize(), "clear slab sync");
    if (!ds4_tp_attach_slab(tp, host_slab, error_text, sizeof(error_text))) {
        std::fprintf(stderr, "attach slab failed: %s\n", error_text);
        ds4_tp_free(tp);
        if (mapped_host) (void)hipHostFree(host_slab);
        else (void)hipFree(device_slab);
        return 1;
    }

    /* Completion words deliberately use a tiny host-mapped allocation even
     * when OdinLink's payload slab is hipMalloc. CPU stores into a hipMalloc
     * word are not a prompt/reliable GPU notification mechanism on gfx1151;
     * the payload remains in its production allocator and is ordered before
     * this word by the receive CQE plus the host release store. */
    void *completion_host_raw = nullptr;
    void *completion_device_raw = nullptr;
    hip_check(hipHostMalloc(&completion_host_raw, 64u, hipHostMallocMapped),
              "completion alloc");
    hip_check(hipHostGetDevicePointer(&completion_device_raw,
                                      completion_host_raw, 0),
              "completion device pointer");
    std::memset(completion_host_raw, 0, 64u);
    volatile uint64_t *host_completion =
        (volatile uint64_t *)completion_host_raw;
    const volatile uint64_t *device_completion =
        (const volatile uint64_t *)completion_device_raw;
    uint32_t *device_error = nullptr;
    hip_check(hipMalloc(&device_error, sizeof(*device_error)), "error alloc");

    uint64_t timeouts = 0;
    uint64_t stale = 0;
    uint64_t transport_errors = 0;
    const uint32_t rank = leader ? 0u : 1u;
    const uint32_t peer_rank = rank ^ 1u;
    const auto begin = std::chrono::steady_clock::now();
    for (uint64_t seq = 1; seq <= iterations; ++seq) {
        const uint32_t gate = (uint32_t)((seq - 1u) & 1u);
        float *out = (float *)((char *)device_slab +
                ds4_tp_slab_out_offset(tp, 0u, gate));
        const float *in = (const float *)((const char *)device_slab +
                ds4_tp_slab_in_offset(tp, 0u, gate));
        fill_payload<<<1, kThreads>>>(out, rank, seq);
        hip_check(hipGetLastError(), "fill launch");
        hip_check(hipDeviceSynchronize(), "fill sync");
        hip_check(hipMemset(device_error, 0, sizeof(*device_error)),
                  "error clear");
        wait_and_verify_payload<<<1, kThreads>>>(
                in, device_completion, peer_rank, seq, device_error);
        hip_check(hipGetLastError(), "consumer launch");

        std::atomic<int> exchange_ok{0};
        std::mutex watchdog_mutex;
        std::condition_variable watchdog_cv;
        bool consumer_done = false;
        std::thread watchdog([&] {
            std::unique_lock<std::mutex> lock(watchdog_mutex);
            if (!watchdog_cv.wait_for(lock, std::chrono::seconds(5),
                                      [&] { return consumer_done; })) {
                __atomic_store_n(host_completion, UINT64_MAX,
                                 __ATOMIC_RELEASE);
            }
        });
        std::thread transport([&] {
            const int ok = ds4_tp_gate_exchange(tp, 0u, gate, seq);
            exchange_ok.store(ok, std::memory_order_release);
            if (ok) {
                std::atomic_thread_fence(std::memory_order_release);
                __atomic_store_n(host_completion, seq, __ATOMIC_RELEASE);
            }
        });
        hip_check(hipDeviceSynchronize(), "consumer sync");
        {
            std::lock_guard<std::mutex> lock(watchdog_mutex);
            consumer_done = true;
        }
        watchdog_cv.notify_one();
        watchdog.join();
        transport.join();

        uint32_t code = 0;
        hip_check(hipMemcpy(&code, device_error, sizeof(code),
                            hipMemcpyDeviceToHost), "error read");
        if (!exchange_ok.load(std::memory_order_acquire)) transport_errors++;
        if (code == 1u) timeouts++;
        if (code == 2u) stale++;
        if (transport_errors || timeouts) break;
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
    std::printf("provider_device=%s rank=%u allocator=%s iterations=%llu "
                "transport_errors=%llu completion_timeouts=%llu "
                "stale_payloads=%llu elapsed_s=%.3f verdict=%s\n",
                device, rank, mapped_host ? "hipHostMallocMapped" : "hipMalloc",
                (unsigned long long)iterations,
                (unsigned long long)transport_errors,
                (unsigned long long)timeouts,
                (unsigned long long)stale,
                seconds,
                (!transport_errors && !timeouts && !stale) ? "PASS" : "FAIL");

    (void)hipFree(device_error);
    (void)hipHostFree(completion_host_raw);
    ds4_tp_free(tp);
    if (mapped_host) (void)hipHostFree(host_slab);
    else (void)hipFree(device_slab);
    return (!transport_errors && !timeouts && !stale) ? 0 : 1;
}
