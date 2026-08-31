/* Two-node transport oracle for GLM-5.3 prompt/decode phase transitions.
 *
 * It executes two complete bulk-prompt -> latency-decode cycles over the
 * production sparse 53-gate schedule.  Each payload is generated on the GPU,
 * copied through the same mapped-slab aliases used by the GLM executor, and
 * checked after the receive CQE.  The second cycle proves that draining and
 * rebuilding the persistent receive window does not retain an old sequence.
 *
 * Usage (start leader first):
 *   test_tp_glm5_phase_transition leader 0.0.0.0 15912 mlx5_0 3
 *   test_tp_glm5_phase_transition worker 192.168.99.1 15912 mlx5_1 3
 */

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "ds4_glm5_next_runtime.h"
#include "ds4_tp.h"
}

static constexpr uint32_t kLayers = 46u;
static constexpr uint32_t kWidth = 4096u;
static constexpr uint64_t kBytes = (uint64_t)kWidth * sizeof(float);
static constexpr uint32_t kCycles = 2u;
static constexpr uint32_t kDecodeTokens = 4u;

struct Gate {
    uint32_t layer;
    uint32_t gate;
};

__global__ static void fill_payload(float *out, uint32_t rank, uint64_t seq) {
    const float base = (float)(rank * 100000u + (uint32_t)seq);
    for (uint32_t i = threadIdx.x; i < kWidth; i += blockDim.x)
        out[i] = base + (float)(i & 63u);
}

static void hip_check(hipError_t rc, const char *what) {
    if (rc == hipSuccess) return;
    std::fprintf(stderr, "FAIL %s: %s\n", what, hipGetErrorString(rc));
    std::exit(1);
}

static std::vector<Gate> glm_schedule(bool kda_tp) {
    std::vector<Gate> gates;
    for (uint32_t layer = 0u;
         layer < DS4_GLM5_NEXT_TRUNK_COUNT; ++layer) {
        const bool routed = layer >= DS4_GLM5_NEXT_LEADING_DENSE;
        if ((kda_tp && !ds4_glm5_next_layer_is_mla(layer)) ||
            (routed && ds4_glm5_next_layer_is_mla(layer)))
            gates.push_back({layer, DS4_TP_GATE_ATTN});
        if (routed) gates.push_back({layer, DS4_TP_GATE_FFN});
    }
    return gates;
}

static bool verify_payload(const std::vector<float> &got,
                           uint32_t peer_rank, uint64_t seq) {
    const float base = (float)(peer_rank * 100000u + (uint32_t)seq);
    for (uint32_t i = 0u; i < kWidth; ++i) {
        const float want = base + (float)(i & 63u);
        if (got[i] != want) {
            std::fprintf(stderr,
                         "FAIL stale payload seq=%llu index=%u got=%g want=%g\n",
                         (unsigned long long)seq, i, got[i], want);
            return false;
        }
    }
    return true;
}

static void usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage: %s leader|worker ADDRESS PORT RDMA_DEVICE "
                 "GID_INDEX_OR_-1\n", argv0);
}

int main(int argc, char **argv) {
    if (argc != 6 ||
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
    if (port <= 0 || port > 65535 || !device[0]) {
        usage(argv[0]);
        return 2;
    }

    (void)setenv("DS4_TP_BIG_DIRECT", "1", 1);
    (void)setenv("DS4_TP_BIG_DIRECT_MAX_ROWS", "1", 1);
    const char *small_env = std::getenv("DS4_TEST_GLM5_SMALL_GATE");
    const bool small_gate = !small_env || std::strcmp(small_env, "1") == 0;
    if (small_env && std::strcmp(small_env, "0") != 0 &&
        std::strcmp(small_env, "1") != 0) {
        std::fprintf(stderr, "FAIL DS4_TEST_GLM5_SMALL_GATE must be 0 or 1\n");
        return 1;
    }

    ds4_tp_options options = {};
    options.role = leader ? DS4_TP_LEADER : DS4_TP_WORKER;
    options.requested = true;
    options.transport = DS4_TP_TRANSPORT_RDMA;
    options.rdma_device = device;
    options.rdma_gid_index = gid;
    options.rdma_gid_index_set = gid >= 0;
    if (leader) {
        options.listen_host = address;
        options.listen_port = port;
    } else {
        options.leader_host = address;
        options.leader_port = port;
    }

    ds4_tp_identity identity = {};
    identity.gguf_bytes = 1u;
    identity.model_id = 0x474c4d35u;
    identity.n_layer = kLayers;
    identity.n_embd = kWidth;
    identity.n_vocab = 1u;
    identity.quant_bits = 4u;
    identity.ctx_size = 16u;
    identity.runtime_features = small_gate
        ? DS4_TP_FEATURE_GLM5_SMALL_GATE : 0u;
    const char *kda_tp_env = std::getenv("DS4_GLM5_KDA_TP");
    const bool kda_tp = kda_tp_env && std::strcmp(kda_tp_env, "1") == 0;
    if (kda_tp_env && !kda_tp && std::strcmp(kda_tp_env, "0") != 0) {
        std::fprintf(stderr, "FAIL DS4_GLM5_KDA_TP must be 0 or 1\n");
        return 1;
    }
    if (kda_tp) identity.runtime_features |= DS4_TP_FEATURE_GLM5_KDA_TP;
    identity.gate_slot_start = kda_tp ? 0u :
        DS4_GLM5_NEXT_LEADING_DENSE * DS4_TP_GATES_PER_LAYER;
    identity.gate_slot_step = 1u;
    if (!ds4_glm5_next_build_tp_gate_mask(identity.gate_slot_mask,
                                           &identity.gates_per_token,
                                           identity.runtime_features)) {
        std::fprintf(stderr, "FAIL build GLM gate schedule\n");
        return 1;
    }
    const std::vector<Gate> schedule = glm_schedule(kda_tp);
    if (schedule.size() != identity.gates_per_token ||
        schedule.size() != (kda_tp ? 87u : 53u)) {
        std::fprintf(stderr, "FAIL schedule size=%zu\n", schedule.size());
        return 1;
    }

    char error[256] = {};
    ds4_tp *tp = nullptr;
    if (!ds4_tp_create(&tp, &options, &identity,
                       error, sizeof(error))) {
        std::fprintf(stderr, "FAIL create: %s\n", error);
        return 1;
    }
    if (!ds4_tp_is_rdma(tp)) {
        std::fprintf(stderr, "FAIL mandatory RDMA transport\n");
        ds4_tp_free(tp);
        return 1;
    }

    const uint64_t slab_bytes = ds4_tp_alloc_slab_bytes(tp);
    void *slab_host = nullptr;
    void *slab_device = nullptr;
    const bool mapped_host = ds4_tp_requires_host_slab(tp);
    if (mapped_host) {
        hip_check(hipHostMalloc(&slab_host, (size_t)slab_bytes,
                                hipHostMallocMapped), "slab host alloc");
        hip_check(hipHostGetDevicePointer(&slab_device, slab_host, 0),
                  "slab device alias");
    } else {
        hip_check(hipMalloc(&slab_device, (size_t)slab_bytes),
                  "slab device alloc");
        slab_host = slab_device;
    }
    hip_check(hipMemset(slab_device, 0, (size_t)slab_bytes), "slab clear");
    hip_check(hipDeviceSynchronize(), "slab clear sync");
    if (!ds4_tp_attach_slab(tp, slab_host, error, sizeof(error))) {
        std::fprintf(stderr, "FAIL attach: %s\n", error);
        ds4_tp_free(tp);
        if (mapped_host) (void)hipHostFree(slab_host);
        else (void)hipFree(slab_device);
        return 1;
    }
    if (!ds4_tp_big_gate_is_rdma_capable(tp)) {
        std::fprintf(stderr, "FAIL mandatory RDMA bulk capability\n");
        ds4_tp_free(tp);
        if (mapped_host) (void)hipHostFree(slab_host);
        else (void)hipFree(slab_device);
        return 1;
    }

    float *stage = nullptr;
    float *received = nullptr;
    hip_check(hipMalloc(&stage, (size_t)kBytes), "stage alloc");
    hip_check(hipMalloc(&received, (size_t)kBytes), "received alloc");
    std::vector<float> host(kWidth);
    const uint32_t rank = leader ? 0u : 1u;
    const uint32_t peer_rank = rank ^ 1u;
    uint64_t sequence = 0u;
    uint64_t checked = 0u;
    double prompt_ms = 0.0;
    double decode_ms = 0.0;
    bool ok = true;

    for (uint32_t cycle = 0u; ok && cycle < kCycles; ++cycle) {
        /* One multi-row prompt batch fires one bulk exchange per graph gate. */
        const auto prompt_start = std::chrono::steady_clock::now();
        for (const Gate &g : schedule) {
            const uint64_t seq = ++sequence;
            fill_payload<<<1, 256>>>(stage, rank, seq);
            hip_check(hipGetLastError(), "bulk fill launch");
            const uint64_t out_off = ds4_tp_slab_big_out_offset(tp);
            const uint64_t in_off = ds4_tp_slab_big_in_offset(tp);
            hip_check(hipMemcpyAsync((char *)slab_device + out_off, stage,
                                     (size_t)kBytes,
                                     hipMemcpyDeviceToDevice, 0),
                      "bulk mapped output copy");
            hip_check(hipDeviceSynchronize(), "bulk producer sync");
            if (!ds4_tp_big_gate_exchange(
                    tp, g.layer, seq,
                    (char *)slab_host + out_off,
                    (char *)slab_host + in_off, kBytes)) {
                std::fprintf(stderr,
                             "FAIL bulk exchange cycle=%u seq=%llu\n",
                             cycle, (unsigned long long)seq);
                ok = false;
                break;
            }
            hip_check(hipMemcpyAsync(received,
                                     (char *)slab_device + in_off,
                                     (size_t)kBytes,
                                     hipMemcpyDeviceToDevice, 0),
                      "bulk mapped input copy");
            hip_check(hipDeviceSynchronize(), "bulk consumer sync");
            hip_check(hipMemcpy(host.data(), received, (size_t)kBytes,
                                hipMemcpyDeviceToHost), "bulk result read");
            ok = verify_payload(host, peer_rank, seq);
            checked++;
            if (!ok) break;
        }

        prompt_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - prompt_start).count();
        const auto decode_start = std::chrono::steady_clock::now();
        for (uint32_t token = 0u; ok && token < kDecodeTokens; ++token) {
            for (const Gate &g : schedule) {
                const uint64_t seq = ++sequence;
                fill_payload<<<1, 256>>>(stage, rank, seq);
                hip_check(hipGetLastError(), "decode fill launch");
                const uint64_t out_off = small_gate
                    ? ds4_tp_slab_out_offset(tp, g.layer, g.gate)
                    : ds4_tp_slab_big_out_offset(tp);
                const uint64_t in_off = small_gate
                    ? ds4_tp_slab_in_offset(tp, g.layer, g.gate)
                    : ds4_tp_slab_big_in_offset(tp);
                hip_check(hipMemcpyAsync((char *)slab_device + out_off, stage,
                                         (size_t)kBytes,
                                         hipMemcpyDeviceToDevice, 0),
                          "decode mapped output copy");
                hip_check(hipDeviceSynchronize(), "decode producer sync");
                const bool exchanged = small_gate
                    ? ds4_tp_gate_exchange(tp, g.layer, g.gate, seq)
                    : ds4_tp_big_gate_exchange(
                          tp, g.layer, seq,
                          (char *)slab_host + out_off,
                          (char *)slab_host + in_off, kBytes);
                if (!exchanged) {
                    std::fprintf(stderr,
                                 "FAIL decode exchange cycle=%u token=%u "
                                 "seq=%llu\n", cycle, token,
                                 (unsigned long long)seq);
                    ok = false;
                    break;
                }
                hip_check(hipMemcpyAsync(received,
                                         (char *)slab_device + in_off,
                                         (size_t)kBytes,
                                         hipMemcpyDeviceToDevice, 0),
                          "decode mapped input copy");
                hip_check(hipDeviceSynchronize(), "decode consumer sync");
                hip_check(hipMemcpy(host.data(), received, (size_t)kBytes,
                                    hipMemcpyDeviceToHost),
                          "decode result read");
                ok = verify_payload(host, peer_rank, seq);
                checked++;
                if (!ok) break;
            }
        }
        decode_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - decode_start).count();
    }

    std::printf("provider_device=%s rank=%u allocator=%s small_gate=%u cycles=%u "
                "prompt_gates=%zu decode_tokens=%u checked=%llu "
                "final_seq=%llu prompt_ms=%.3f decode_ms=%.3f verdict=%s\n",
                device, rank, mapped_host ? "hipHostMallocMapped" : "hipMalloc",
                small_gate ? 1u : 0u,
                kCycles, schedule.size(), kDecodeTokens,
                (unsigned long long)checked,
                (unsigned long long)sequence, prompt_ms, decode_ms,
                ok ? "PASS" : "FAIL");

    (void)hipFree(received);
    (void)hipFree(stage);
    ds4_tp_free(tp);
    if (mapped_host) (void)hipHostFree(slab_host);
    else (void)hipFree(slab_device);
    return ok ? 0 : 1;
}
