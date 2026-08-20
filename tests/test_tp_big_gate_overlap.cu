/* Two-node, model-free proof for pipelining an already-complete TP prefill
 * tensor over the real selected RDMA provider.  It uses DS4's production
 * hello, slab registration, RC QP, and big-gate exchange; only the row-local
 * GPU work is synthetic.  Both ranks must run the same chunk/work settings.
 *
 * This is a research gate, not an inference benchmark.  A candidate may move
 * into the model only when this test is bit-exact and hides at least half of
 * the measured wire time without transport fallback. */

#include <hip/hip_runtime.h>

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

extern "C" {
#include "ds4_tp.h"
}

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

static constexpr uint32_t kRows = 2048u;
static constexpr uint32_t kWidth = 4096u;
static constexpr uint64_t kValues = (uint64_t)kRows * kWidth;
static constexpr uint64_t kBytes = kValues * sizeof(float);

static void fail_hip(hipError_t rc, const char *where) {
    if (rc == hipSuccess) return;
    std::fprintf(stderr, "FAIL %s: %s\n", where, hipGetErrorString(rc));
    std::exit(1);
}

__global__ static void init_rank_kernel(float *out, uint64_t n,
                                        uint32_t rank) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int32_t centered = (int32_t)(i % 1024u) - 512;
    out[i] = (float)centered * (1.0f / 1024.0f) + (float)rank * 0.25f;
}

/* Stand-in for the next layer's row-local norm/projection prefix.  The loop
 * is deliberately dependent so it cannot be folded away.  Serial and
 * pipelined arms execute the identical kernel and canonical rank ordering. */
__global__ static void combine_row_work_kernel(
        float *dst, const float *rank0, const float *rank1,
        uint64_t n, uint32_t work_iters) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = rank0[i] + rank1[i];
    const float anchor = v;
    for (uint32_t j = 0; j < work_iters; ++j) {
        const float bias = (j & 1u) ? 0x1p-20f : -0x1p-20f;
        v = fmaf(v, 1.00000011920928955078125f, bias);
    }
    dst[i] = v + anchor * 0x1p-12f;
}

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
            clock::now().time_since_epoch()).count();
}

struct exchange_pipeline {
    ds4_tp *tp = nullptr;
    const uint8_t *send = nullptr;
    uint8_t *recv = nullptr;
    uint32_t chunks = 0;
    uint64_t chunk_bytes = 0;
    uint64_t seq_base = 0;
    std::mutex mutex;
    std::condition_variable ready;
    uint32_t completed = 0;
    bool failed = false;
};

static void exchange_worker(exchange_pipeline *p) {
    for (uint32_t chunk = 0; chunk < p->chunks; ++chunk) {
        const uint64_t off = (uint64_t)chunk * p->chunk_bytes;
        const int ok = ds4_tp_big_gate_exchange(
                p->tp, chunk, p->seq_base + chunk,
                p->send + off, p->recv + off, p->chunk_bytes);
        {
            std::lock_guard<std::mutex> lock(p->mutex);
            if (!ok) p->failed = true;
            p->completed = chunk + 1u;
        }
        p->ready.notify_all();
        if (!ok) return;
    }
}

static bool wait_chunk(exchange_pipeline *p, uint32_t want) {
    std::unique_lock<std::mutex> lock(p->mutex);
    p->ready.wait(lock, [&] { return p->failed || p->completed >= want; });
    return !p->failed;
}

static int gpu_row_exchange(void *, uint32_t, uint32_t, uint64_t) {
    return 1;
}

static int gpu_big_exchange(void *opaque, uint32_t layer, uint64_t seq,
                            const void *out, void *in, uint64_t bytes) {
    return ds4_tp_big_gate_exchange(static_cast<ds4_tp *>(opaque), layer,
                                    seq, out, in, bytes);
}

struct gpu_wave_bridge {
    ds4_gpu_tp_big_wave_ready_fn ready;
    void *ready_ud;
};

static void gpu_wave_ready_bridge(void *opaque, uint32_t wave) {
    auto *bridge = static_cast<gpu_wave_bridge *>(opaque);
    bridge->ready(bridge->ready_ud, wave);
}

static int gpu_big_wave_exchange(
        void *opaque, uint32_t layer, uint64_t seq,
        const void *out, void *in, uint64_t bytes,
        uint64_t wave_bytes, uint32_t waves,
        ds4_gpu_tp_big_wave_ready_fn ready, void *ready_ud) {
    gpu_wave_bridge bridge = {ready, ready_ud};
    return ds4_tp_big_gate_exchange_waves(
        static_cast<ds4_tp *>(opaque), layer, seq, out, in, bytes,
        wave_bytes, waves, gpu_wave_ready_bridge, &bridge);
}

static void launch_chunk(float *dst, const float *local, const float *peer,
                         uint64_t value_off, uint64_t values,
                         uint32_t rank, uint32_t work_iters) {
    const float *rank0 = rank == 0u ? local + value_off : peer + value_off;
    const float *rank1 = rank == 0u ? peer + value_off : local + value_off;
    const uint32_t threads = 256u;
    const uint32_t blocks = (uint32_t)((values + threads - 1u) / threads);
    combine_row_work_kernel<<<blocks, threads>>>(
            dst + value_off, rank0, rank1, values, work_iters);
    fail_hip(hipGetLastError(), "combine_row_work launch");
}

static double wire_only(ds4_tp *tp, const uint8_t *send, uint8_t *recv,
                        uint32_t chunks, uint64_t chunk_bytes,
                        uint64_t seq_base) {
    const double begin = now_ms();
    for (uint32_t i = 0; i < chunks; ++i) {
        const uint64_t off = (uint64_t)i * chunk_bytes;
        if (!ds4_tp_big_gate_exchange(tp, i, seq_base + i,
                                      send + off, recv + off, chunk_bytes)) {
            std::fprintf(stderr, "FAIL wire-only exchange chunk=%u\n", i);
            std::exit(1);
        }
    }
    return now_ms() - begin;
}

static double compute_only(float *dst, const float *local, const float *peer,
                           uint32_t chunks, uint64_t chunk_values,
                           uint32_t rank, uint32_t work_iters) {
    const double begin = now_ms();
    for (uint32_t i = 0; i < chunks; ++i) {
        launch_chunk(dst, local, peer, (uint64_t)i * chunk_values,
                     chunk_values, rank, work_iters);
    }
    fail_hip(hipDeviceSynchronize(), "compute-only synchronize");
    return now_ms() - begin;
}

static double serial(ds4_tp *tp, float *dst,
                     const uint8_t *send_host, uint8_t *recv_host,
                     const float *send_device, const float *recv_device,
                     uint32_t chunks, uint64_t chunk_bytes,
                     uint32_t rank, uint32_t work_iters, uint64_t seq_base) {
    const uint64_t chunk_values = chunk_bytes / sizeof(float);
    const double begin = now_ms();
    for (uint32_t i = 0; i < chunks; ++i) {
        const uint64_t byte_off = (uint64_t)i * chunk_bytes;
        if (!ds4_tp_big_gate_exchange(tp, i, seq_base + i,
                                      send_host + byte_off,
                                      recv_host + byte_off, chunk_bytes)) {
            std::fprintf(stderr, "FAIL serial exchange chunk=%u\n", i);
            std::exit(1);
        }
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    for (uint32_t i = 0; i < chunks; ++i) {
        launch_chunk(dst, send_device, recv_device,
                     (uint64_t)i * chunk_values, chunk_values,
                     rank, work_iters);
    }
    fail_hip(hipDeviceSynchronize(), "serial synchronize");
    return now_ms() - begin;
}

static double pipelined(ds4_tp *tp, float *dst,
                        const uint8_t *send_host, uint8_t *recv_host,
                        const float *send_device, const float *recv_device,
                        uint32_t chunks, uint64_t chunk_bytes,
                        uint32_t rank, uint32_t work_iters,
                        uint64_t seq_base) {
    exchange_pipeline p;
    p.tp = tp;
    p.send = send_host;
    p.recv = recv_host;
    p.chunks = chunks;
    p.chunk_bytes = chunk_bytes;
    p.seq_base = seq_base;
    const uint64_t chunk_values = chunk_bytes / sizeof(float);
    const double begin = now_ms();
    std::thread network(exchange_worker, &p);
    for (uint32_t i = 0; i < chunks; ++i) {
        if (!wait_chunk(&p, i + 1u)) {
            network.join();
            std::fprintf(stderr, "FAIL pipelined exchange chunk=%u\n", i);
            std::exit(1);
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        launch_chunk(dst, send_device, recv_device,
                     (uint64_t)i * chunk_values, chunk_values,
                     rank, work_iters);
    }
    fail_hip(hipDeviceSynchronize(), "pipeline synchronize");
    network.join();
    return now_ms() - begin;
}

static double production_single(
        ds4_gpu_tensor *send, ds4_gpu_tensor *recv, float *dst,
        const float *send_device, const float *recv_device,
        uint32_t chunks, uint64_t chunk_bytes,
        uint32_t rank, uint32_t work_iters) {
    const uint64_t chunk_values = chunk_bytes / sizeof(float);
    const double begin = now_ms();
    if (!ds4_gpu_tp_big_gate_encode(0u, kRows, send, recv, kBytes)) {
        std::fprintf(stderr, "FAIL production single big gate\n");
        std::exit(1);
    }
    for (uint32_t i = 0; i < chunks; ++i) {
        launch_chunk(dst, send_device, recv_device,
                     (uint64_t)i * chunk_values, chunk_values,
                     rank, work_iters);
    }
    fail_hip(hipDeviceSynchronize(), "production single synchronize");
    return now_ms() - begin;
}

static double production_wave_pipeline(
        ds4_tp *tp, ds4_gpu_tensor *slab, float *dst,
        const float *send_device, const float *recv_device,
        uint32_t chunks, uint64_t chunk_bytes,
        uint32_t rank, uint32_t work_iters) {
    const uint64_t chunk_values = chunk_bytes / sizeof(float);
    const uint64_t send_base = ds4_tp_slab_big_out_offset(tp);
    const uint64_t recv_base = ds4_tp_slab_big_in_offset(tp);
    ds4_gpu_tensor *send = ds4_gpu_tensor_view(slab, send_base, kBytes);
    ds4_gpu_tensor *recv = ds4_gpu_tensor_view(slab, recv_base, kBytes);
    if (!send || !recv) {
        std::fprintf(stderr, "FAIL production wave full views\n");
        std::exit(1);
    }
    const double begin = now_ms();
    const uint64_t first_seq = ds4_gpu_tp_big_gate_waves_kick(
        0u, kRows, send, recv, kBytes, chunks);
    if (!first_seq) {
        std::fprintf(stderr, "FAIL production wave kick\n");
        std::exit(1);
    }
    for (uint32_t i = 0; i < chunks; ++i) {
        if (!ds4_gpu_tp_big_gate_wait(first_seq + i)) {
            std::fprintf(stderr,
                         "FAIL production wave wait chunk=%u\n", i);
            std::exit(1);
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        launch_chunk(dst, send_device, recv_device,
                     (uint64_t)i * chunk_values, chunk_values,
                     rank, work_iters);
    }
    fail_hip(hipDeviceSynchronize(), "production wave synchronize");
    ds4_gpu_tensor_free(recv);
    ds4_gpu_tensor_free(send);
    return now_ms() - begin;
}

static void usage(const char *argv0) {
    std::fprintf(stderr,
        "usage: %s <leader|worker> <leader-host> <port> <rdma-device> "
        "<gid-index> [chunks=8] [work-iters=4096]\n", argv0);
}

int main(int argc, char **argv) {
    if (argc < 6 || argc > 8) {
        usage(argv[0]);
        return 2;
    }
    const bool leader = std::strcmp(argv[1], "leader") == 0;
    const bool worker = std::strcmp(argv[1], "worker") == 0;
    if (!leader && !worker) {
        usage(argv[0]);
        return 2;
    }
    const char *leader_host = argv[2];
    const int port = std::atoi(argv[3]);
    const char *device = argv[4];
    const int gid = std::atoi(argv[5]);
    const uint32_t chunks = argc > 6 ? (uint32_t)std::strtoul(argv[6], nullptr, 10) : 8u;
    const uint32_t work_iters = argc > 7 ? (uint32_t)std::strtoul(argv[7], nullptr, 10) : 4096u;
    if (port <= 0 || gid < 0 || chunks < 2u || chunks > 32u ||
        kRows % chunks != 0u || work_iters == 0u) {
        usage(argv[0]);
        return 2;
    }

    (void)setenv("DS4_TP_BIG_DIRECT", "1", 1);
    (void)setenv("DS4_TP_BIG_DIRECT_MAX_ROWS", "2048", 1);

    ds4_tp_options opt = {};
    opt.requested = true;
    opt.role = leader ? DS4_TP_LEADER : DS4_TP_WORKER;
    opt.listen_host = leader ? leader_host : nullptr;
    opt.listen_port = leader ? port : 0;
    opt.leader_host = worker ? leader_host : nullptr;
    opt.leader_port = worker ? port : 0;
    opt.transport = DS4_TP_TRANSPORT_RDMA;
    opt.rdma_device = device;
    opt.rdma_gid_index = gid;
    opt.rdma_gid_index_set = true;

    ds4_tp_identity id = {};
    id.gguf_bytes = 1u;
    id.model_id = 0x54455354u;
    /* Ensure the registered staging region covers a full 4 MiB RDMA round.
     * The former one-layer identity silently sent big-gate payloads over the
     * TCP fallback despite having negotiated an RDMA QP. */
    id.n_layer = 64u;
    id.n_embd = kWidth;
    id.n_vocab = 1u;
    id.quant_bits = 4u;
    id.ctx_size = kRows;

    char err[512] = {};
    ds4_tp *tp = nullptr;
    if (!ds4_tp_create(&tp, &opt, &id, err, sizeof(err))) {
        std::fprintf(stderr, "FAIL ds4_tp_create: %s\n", err);
        return 1;
    }
    if (!ds4_tp_is_rdma(tp) || ds4_tp_big_capacity_rows(tp) < kRows) {
        std::fprintf(stderr, "FAIL explicit RDMA/direct slab unavailable\n");
        ds4_tp_free(tp);
        return 1;
    }

    const uint64_t slab_bytes = ds4_tp_alloc_slab_bytes(tp);
    void *slab_host = nullptr;
    void *slab_device = nullptr;
    const bool host_slab = ds4_tp_requires_host_slab(tp);
    if (host_slab) {
        fail_hip(hipHostMalloc(&slab_host, (size_t)slab_bytes,
                               hipHostMallocMapped), "hipHostMalloc slab");
        fail_hip(hipHostGetDevicePointer(&slab_device, slab_host, 0),
                 "hipHostGetDevicePointer slab");
    } else {
        fail_hip(hipMalloc(&slab_host, (size_t)slab_bytes), "hipMalloc slab");
        slab_device = slab_host;
    }
    if (!ds4_tp_attach_slab(tp, slab_host, err, sizeof(err))) {
        std::fprintf(stderr, "FAIL ds4_tp_attach_slab: %s\n", err);
        if (host_slab) (void)hipHostFree(slab_host);
        else (void)hipFree(slab_host);
        ds4_tp_free(tp);
        return 1;
    }
    if (!ds4_tp_big_gate_is_rdma_capable(tp)) {
        std::fprintf(stderr,
                     "FAIL big-gate payload would fall back from RDMA\n");
        ds4_tp_free(tp);
        if (host_slab) (void)hipHostFree(slab_host);
        else (void)hipFree(slab_host);
        return 1;
    }

    const uint64_t out_off = ds4_tp_slab_big_out_offset(tp);
    const uint64_t in_off = ds4_tp_slab_big_in_offset(tp);
    auto *send_host = (uint8_t *)slab_host + out_off;
    auto *recv_host = (uint8_t *)slab_host + in_off;
    auto *send_device = (float *)((uint8_t *)slab_device + out_off);
    auto *recv_device = (float *)((uint8_t *)slab_device + in_off);
    float *serial_out = nullptr;
    float *pipeline_out = nullptr;
    float *production_single_out = nullptr;
    float *production_out = nullptr;
    fail_hip(hipMalloc(&serial_out, (size_t)kBytes), "serial output allocation");
    fail_hip(hipMalloc(&pipeline_out, (size_t)kBytes), "pipeline output allocation");
    fail_hip(hipMalloc(&production_single_out, (size_t)kBytes),
             "production single output allocation");
    fail_hip(hipMalloc(&production_out, (size_t)kBytes),
             "production split output allocation");
    init_rank_kernel<<<(uint32_t)((kValues + 255u) / 256u), 256u>>>(
            send_device, kValues, leader ? 0u : 1u);
    fail_hip(hipGetLastError(), "init rank launch");
    fail_hip(hipDeviceSynchronize(), "init rank synchronize");

    const uint64_t chunk_bytes = kBytes / chunks;
    const uint64_t chunk_values = kValues / chunks;
    const uint32_t rank = leader ? 0u : 1u;

    /* First use includes QP/cache/clock warmup and is not evidence.  Consume
     * one identical untimed pass before the measured arms. */
    (void)wire_only(tp, send_host, recv_host, chunks, chunk_bytes, 500u);
    const double wire_ms = wire_only(tp, send_host, recv_host, chunks,
                                     chunk_bytes, 1000u);
    std::atomic_thread_fence(std::memory_order_acquire);
    const double compute_ms = compute_only(serial_out, send_device, recv_device,
                                           chunks, chunk_values, rank,
                                           work_iters);
    const double serial_ms = serial(tp, serial_out, send_host, recv_host,
                                    send_device, recv_device, chunks,
                                    chunk_bytes, rank, work_iters, 2000u);
    const double pipeline_ms = pipelined(tp, pipeline_out, send_host, recv_host,
                                         send_device, recv_device, chunks,
                                         chunk_bytes, rank, work_iters, 3000u);

    ds4_gpu_tensor slab_tensor = {};
    slab_tensor.ptr = slab_device;
    slab_tensor.bytes = slab_bytes;
    slab_tensor.owner = 0;
    if (!ds4_gpu_tp_init(rank, &slab_tensor,
                         ds4_tp_slab_gpu_flags_offset(tp),
                         gpu_row_exchange, tp)) {
        std::fprintf(stderr,
                     "FAIL initialize production ROCm split gate\n");
        return 1;
    }
    ds4_gpu_tp_set_big_exchange(gpu_big_exchange);
    ds4_gpu_tp_set_big_wave_exchange(gpu_big_wave_exchange);
    ds4_gpu_tensor *full_send = ds4_gpu_tensor_view(
        &slab_tensor, out_off, kBytes);
    ds4_gpu_tensor *full_recv = ds4_gpu_tensor_view(
        &slab_tensor, in_off, kBytes);
    if (!full_send || !full_recv) {
        std::fprintf(stderr, "FAIL production single full views\n");
        return 1;
    }
    const double production_single_ms = production_single(
        full_send, full_recv, production_single_out, send_device, recv_device,
        chunks, chunk_bytes, rank, work_iters);
    ds4_gpu_tensor_free(full_recv);
    ds4_gpu_tensor_free(full_send);
    const double production_ms = production_wave_pipeline(
        tp, &slab_tensor, production_out, send_device, recv_device,
        chunks, chunk_bytes, rank, work_iters);

    std::vector<float> serial_host(kValues), pipeline_host(kValues);
    std::vector<float> production_single_host(kValues);
    std::vector<float> production_host(kValues);
    fail_hip(hipMemcpy(serial_host.data(), serial_out, (size_t)kBytes,
                       hipMemcpyDeviceToHost), "copy serial result");
    fail_hip(hipMemcpy(pipeline_host.data(), pipeline_out, (size_t)kBytes,
                       hipMemcpyDeviceToHost), "copy pipeline result");
    fail_hip(hipMemcpy(production_single_host.data(), production_single_out,
                       (size_t)kBytes, hipMemcpyDeviceToHost),
             "copy production single result");
    fail_hip(hipMemcpy(production_host.data(), production_out, (size_t)kBytes,
                       hipMemcpyDeviceToHost), "copy production split result");
    uint64_t mismatches = 0;
    uint64_t production_single_mismatches = 0;
    uint64_t production_mismatches = 0;
    for (uint64_t i = 0; i < kValues; ++i) {
        if (std::memcmp(&serial_host[i], &pipeline_host[i], sizeof(float)))
            ++mismatches;
        if (std::memcmp(&serial_host[i], &production_single_host[i],
                        sizeof(float)))
            ++production_single_mismatches;
        if (std::memcmp(&serial_host[i], &production_host[i], sizeof(float)))
            ++production_mismatches;
    }

    const double hidden_ms = serial_ms - pipeline_ms;
    const double hidden_wire_fraction = wire_ms > 0.0 ? hidden_ms / wire_ms : 0.0;
    std::printf(
        "TP_BIG_GATE_OVERLAP rank=%u provider=%s bytes=%llu chunks=%u "
        "work_iters=%u wire_ms=%.3f compute_ms=%.3f serial_ms=%.3f "
        "pipeline_ms=%.3f production_single_ms=%.3f production_ms=%.3f hidden_ms=%.3f "
        "hidden_wire_fraction=%.3f mismatches=%llu "
        "production_single_mismatches=%llu production_mismatches=%llu\n",
        rank, device, (unsigned long long)kBytes, chunks, work_iters,
        wire_ms, compute_ms, serial_ms, pipeline_ms, production_single_ms,
        production_ms,
        hidden_ms, hidden_wire_fraction, (unsigned long long)mismatches,
        (unsigned long long)production_single_mismatches,
        (unsigned long long)production_mismatches);

    const bool balanced = compute_ms >= wire_ms * 0.75 &&
                          compute_ms <= wire_ms * 1.50;
    const bool production_faster = production_ms < production_single_ms;
    const bool pass = mismatches == 0u &&
                      production_single_mismatches == 0u &&
                      production_mismatches == 0u &&
                      balanced && hidden_wire_fraction >= 0.50 &&
                      pipeline_ms < serial_ms && production_faster;
    std::printf("%s exact=%d production_single_exact=%d production_exact=%d balanced=%d "
                "hide_ge_50pct=%d production_faster=%d\n",
                pass ? "PASS" : "FAIL",
                mismatches == 0u, production_single_mismatches == 0u,
                production_mismatches == 0u, balanced,
                hidden_wire_fraction >= 0.50, production_faster);

    ds4_gpu_tp_shutdown();
    (void)hipFree(production_out);
    (void)hipFree(production_single_out);
    (void)hipFree(pipeline_out);
    (void)hipFree(serial_out);
    ds4_tp_free(tp);
    if (host_slab) (void)hipHostFree(slab_host);
    else (void)hipFree(slab_host);
    return pass ? 0 : 1;
}
