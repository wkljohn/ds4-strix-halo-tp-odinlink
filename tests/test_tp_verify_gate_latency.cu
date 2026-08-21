/* Model-free latency and integrity gate for the DSpark verifier's TP payload.
 * It deliberately calls DS4's production big-gate API with an 80 KiB buffer
 * outside the registered slab, matching the five-row 4096-wide FP32 verifier
 * exchange.  Every iteration changes and validates the complete payload, so
 * stale receives, slot reuse, ordering errors, and silent fallback fail. */

#include <hip/hip_runtime.h>

extern "C" {
#include "ds4_tp.h"
}

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

static constexpr uint32_t kWidth = 4096u;
static constexpr uint32_t kLayers = 64u;

static double now_us() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::micro>(
               clock::now().time_since_epoch()).count();
}

static void hip_check(hipError_t rc, const char *where) {
    if (rc == hipSuccess) return;
    std::fprintf(stderr, "FAIL %s: %s\n", where, hipGetErrorString(rc));
    std::exit(1);
}

static uint8_t payload_byte(uint32_t rank, uint32_t iteration, uint64_t i) {
    uint64_t x = i + UINT64_C(0x9e3779b97f4a7c15) * (iteration + 1u);
    x ^= UINT64_C(0xd1b54a32d192ed03) * (rank + 1u);
    x ^= x >> 29;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 32;
    return (uint8_t)x;
}

static double percentile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    const size_t i = (size_t)(q * (double)(values.size() - 1u));
    return values[i];
}

static void usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage: %s <leader|worker> <leader-host> <port> "
                 "<rdma-device> <gid-index> [iterations=200] [rows=5] "
                 "[big|batch|rows]\n",
                 argv0);
}

int main(int argc, char **argv) {
    if (argc < 6 || argc > 9) {
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
    const uint32_t iterations = argc >= 7
        ? (uint32_t)std::strtoul(argv[6], nullptr, 10) : 200u;
    const uint32_t rows = argc >= 8
        ? (uint32_t)std::strtoul(argv[7], nullptr, 10) : 5u;
    const char *mode = argc == 9 ? argv[8] : "big";
    const bool mode_big = std::strcmp(mode, "big") == 0;
    const bool mode_batch = std::strcmp(mode, "batch") == 0;
    const bool mode_rows = std::strcmp(mode, "rows") == 0;
    const uint64_t bytes = (uint64_t)rows * kWidth * sizeof(float);
    if (port <= 0 || gid < 0 || iterations < 10u || rows < 1u || rows > 8u ||
        (!mode_big && !mode_batch && !mode_rows)) {
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
    id.model_id = UINT32_C(0x56524659); /* VRFY */
    id.n_layer = kLayers;
    id.n_embd = kWidth;
    id.n_vocab = 1u;
    id.quant_bits = 4u;
    id.ctx_size = 2048u;
    if (std::getenv("DS4_TP_VERIFY_ROW_BATCH"))
        id.runtime_features |= DS4_TP_FEATURE_VERIFY_ROW_BATCH;

    char err[512] = {};
    ds4_tp *tp = nullptr;
    if (!ds4_tp_create(&tp, &opt, &id, err, sizeof(err))) {
        std::fprintf(stderr, "FAIL ds4_tp_create: %s\n", err);
        return 1;
    }
    if (!ds4_tp_is_rdma(tp)) {
        std::fprintf(stderr, "FAIL explicit RDMA unavailable\n");
        ds4_tp_free(tp);
        return 1;
    }

    const uint64_t slab_bytes = ds4_tp_alloc_slab_bytes(tp);
    const bool host_slab = ds4_tp_requires_host_slab(tp);
    void *slab_host = nullptr;
    if (host_slab) {
        hip_check(hipHostMalloc(&slab_host, (size_t)slab_bytes,
                                hipHostMallocMapped), "hipHostMalloc slab");
    } else {
        hip_check(hipMalloc(&slab_host, (size_t)slab_bytes),
                  "hipMalloc slab");
    }
    if (!ds4_tp_attach_slab(tp, slab_host, err, sizeof(err))) {
        std::fprintf(stderr, "FAIL ds4_tp_attach_slab: %s\n", err);
        return 1;
    }
    if (!ds4_tp_big_gate_is_rdma_capable(tp)) {
        std::fprintf(stderr, "FAIL big gate is not RDMA-capable\n");
        return 1;
    }

    void *send = nullptr;
    void *recv = nullptr;
    hip_check(hipMalloc(&send, (size_t)bytes), "hipMalloc send");
    hip_check(hipMalloc(&recv, (size_t)bytes), "hipMalloc recv");
    std::vector<uint8_t> local(bytes), peer(bytes);
    std::vector<double> samples;
    samples.reserve(iterations);
    const uint32_t rank = leader ? 0u : 1u;
    uint64_t mismatches = 0u;
    uint64_t row_seq = 1u;
    uint8_t *slab = (uint8_t *)slab_host;
    const char *gap_env = std::getenv("DS4_TP_VERIFY_TEST_GAP_US");
    const uint32_t gap_us = gap_env
        ? (uint32_t)std::strtoul(gap_env, nullptr, 10) : 0u;

    for (uint32_t it = 0; it < iterations + 4u; ++it) {
        if (gap_us) usleep(gap_us);
        for (uint64_t i = 0; i < bytes; ++i)
            local[(size_t)i] = payload_byte(rank, it, i);
        hip_check(hipMemcpy(send, local.data(), (size_t)bytes,
                            hipMemcpyHostToDevice), "upload payload");
        hip_check(hipMemset(recv, 0xa5, (size_t)bytes), "clear receive");
        const uint32_t batch_layer = it % kLayers;
        if (mode_batch) {
            hip_check(hipMemcpy(slab + ds4_tp_slab_batch_out_offset(tp, batch_layer),
                                local.data(), (size_t)bytes,
                                hipMemcpyHostToDevice), "upload batch payload");
        } else if (mode_rows) {
            for (uint32_t row = 0; row < rows; ++row) {
                const uint64_t seq = row_seq + row;
                const uint32_t slot = (uint32_t)((seq - 1u) % (kLayers * 2u));
                const uint32_t layer = slot / 2u;
                const uint32_t gate = slot % 2u;
                hip_check(hipMemcpy(slab + ds4_tp_slab_out_offset(tp, layer, gate),
                                    local.data() + (uint64_t)row * kWidth * sizeof(float),
                                    kWidth * sizeof(float), hipMemcpyHostToDevice),
                          "upload row payload");
            }
        }
        const double begin = now_us();
        const uint64_t seq = UINT64_C(1000) + it;
        bool exchange_ok = false;
        if (mode_big) {
            exchange_ok = ds4_tp_big_gate_exchange(tp, batch_layer, seq,
                                                   send, recv, bytes) != 0;
        } else if (mode_batch) {
            exchange_ok = ds4_tp_batch_gate_exchange(tp, batch_layer, rows,
                                                     seq) != 0;
        } else {
            exchange_ok = true;
            for (uint32_t row = 0; exchange_ok && row < rows; ++row) {
                const uint64_t rseq = row_seq + row;
                const uint32_t slot =
                    (uint32_t)((rseq - 1u) % (kLayers * 2u));
                exchange_ok = ds4_tp_gate_exchange(tp, slot / 2u, slot % 2u,
                                                   rseq) != 0;
            }
            row_seq += rows;
        }
        if (!exchange_ok) {
            std::fprintf(stderr, "FAIL exchange iteration=%u\n", it);
            return 1;
        }
        const double elapsed = now_us() - begin;
        if (mode_big) {
            hip_check(hipMemcpy(peer.data(), recv, (size_t)bytes,
                                hipMemcpyDeviceToHost), "download payload");
        } else if (mode_batch) {
            hip_check(hipMemcpy(peer.data(),
                                slab + ds4_tp_slab_batch_in_offset(tp, batch_layer),
                                (size_t)bytes, hipMemcpyDeviceToHost),
                      "download batch payload");
        } else {
            const uint64_t first_seq = row_seq - rows;
            for (uint32_t row = 0; row < rows; ++row) {
                const uint64_t rseq = first_seq + row;
                const uint32_t slot =
                    (uint32_t)((rseq - 1u) % (kLayers * 2u));
                hip_check(hipMemcpy(peer.data() +
                                        (uint64_t)row * kWidth * sizeof(float),
                                    slab + ds4_tp_slab_in_offset(tp, slot / 2u,
                                                                 slot % 2u),
                                    kWidth * sizeof(float), hipMemcpyDeviceToHost),
                          "download row payload");
            }
        }
        for (uint64_t i = 0; i < bytes; ++i) {
            const uint8_t want = payload_byte(rank ^ 1u, it, i);
            if (peer[(size_t)i] != want) ++mismatches;
        }
        if (it >= 4u) samples.push_back(elapsed);
    }

    double sum = 0.0;
    for (double v : samples) sum += v;
    std::printf(
        "TP_VERIFY_GATE rank=%u provider=%s mode=%s bytes=%llu iterations=%u "
        "mean_us=%.3f p50_us=%.3f p95_us=%.3f p99_us=%.3f "
        "mismatches=%llu\n",
        rank, device, mode, (unsigned long long)bytes, iterations,
        sum / (double)samples.size(), percentile(samples, 0.50),
        percentile(samples, 0.95), percentile(samples, 0.99),
        (unsigned long long)mismatches);
    std::printf("%s exact=%d rdma=1\n",
                mismatches == 0u ? "PASS" : "FAIL", mismatches == 0u);

    (void)hipFree(recv);
    (void)hipFree(send);
    ds4_tp_free(tp);
    if (host_slab) (void)hipHostFree(slab_host);
    else (void)hipFree(slab_host);
    return mismatches == 0u ? 0 : 1;
}
