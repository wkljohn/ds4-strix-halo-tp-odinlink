#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
extern "C" {
#include "ds4_tp.h"
}
#include "tests/glm5_gguf_test.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#define CHECK(expr, message) do {                                           \
    if (!(expr)) {                                                          \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__);    \
        return false;                                                       \
    }                                                                       \
} while (0)

namespace {

constexpr uint32_t kRows = 10u;
constexpr uint32_t kFirstValid = 1u;
constexpr uint32_t kHidden = 4096u;
constexpr uint32_t kHc = 4u;
constexpr uint32_t kHcMix = 24u;
constexpr uint32_t kQRank = 1536u;
constexpr uint32_t kHeads = 64u;
constexpr uint32_t kHeadDim = 256u;
constexpr uint32_t kKvLora = 512u;
constexpr uint32_t kIndexHeads = 32u;
constexpr uint32_t kIndexDim = 128u;
constexpr uint32_t kPools = 3u;
constexpr uint32_t kSelectedPools = 2u;
constexpr uint32_t kSelectedTokens = 9u;
constexpr uint32_t kTokenBudget = 2048u;
constexpr uint32_t kExpandedWidth = kTokenBudget + 3u;

struct TpGuard {
    ds4_tp *tp = nullptr;
    void *slab = nullptr;
    ~TpGuard() {
        if (tp) ds4_tp_free(tp);
        if (slab) {
            const hipError_t rc = hipHostFree(slab);
            if (rc != hipSuccess) {
                std::fprintf(stderr, "WARN hipHostFree MLA TP slab: %s\n",
                             hipGetErrorString(rc));
            }
        }
    }
};

uint64_t fnv1a64(const void *data, uint64_t bytes) {
    const auto *p = static_cast<const uint8_t *>(data);
    uint64_t hash = UINT64_C(1469598103934665603);
    for (uint64_t i = 0u; i < bytes; ++i) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

uint64_t fnv1a64_continue(uint64_t hash, const void *data, uint64_t bytes) {
    const auto *p = static_cast<const uint8_t *>(data);
    for (uint64_t i = 0u; i < bytes; ++i) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

template <typename T>
bool read_array(const std::string &path, size_t count, std::vector<T> &out) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return false;
    const std::streamoff size = input.tellg();
    if (size < 0 || (uint64_t)size != (uint64_t)count * sizeof(T)) return false;
    input.seekg(0);
    out.resize(count);
    return (bool)input.read((char *)out.data(), size);
}

template <typename T>
bool read_tensor(ds4_gpu_tensor *tensor, size_t count, std::vector<T> &out) {
    out.resize(count);
    return ds4_gpu_tensor_read(tensor, 0u, out.data(),
                               (uint64_t)count * sizeof(T));
}

bool compare_values(const char *name, const std::vector<float> &got,
                    const std::vector<float> &expected,
                    double max_abs_limit, double nmse_limit) {
    CHECK(got.size() == expected.size(), "MLA composition comparison shape");
    double maximum = 0.0, error2 = 0.0, reference2 = 0.0;
    double reference_max = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        CHECK(std::isfinite(got[i]) && std::isfinite(expected[i]),
              "finite MLA composition output");
        const double error = (double)got[i] - expected[i];
        maximum = std::max(maximum, std::fabs(error));
        error2 += error * error;
        reference2 += (double)expected[i] * expected[i];
        reference_max = std::max(reference_max,
                                 std::fabs((double)expected[i]));
    }
    const double nmse = error2 / std::max(reference2, 1.0e-30);
    std::fprintf(stderr,
                 "GLM5 MLA compose %-12s count=%zu max_abs=%.9g "
                 "reference_max=%.9g nmse=%.9g\n",
                 name, got.size(), maximum, reference_max, nmse);
    CHECK(reference_max >= 1.0e-6, "non-degenerate MLA composition reference");
    CHECK(maximum <= max_abs_limit && nmse <= nmse_limit,
          "MLA composition numerical envelope");
    return true;
}

uint16_t ordered_bf16(float value, bool &exact) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    exact = (bits & UINT32_C(0xffff)) == 0u;
    uint16_t bf16 = (uint16_t)(bits >> 16u);
    if ((bf16 & UINT16_C(0x7fff)) == 0u) bf16 = 0u;
    return (bf16 & UINT16_C(0x8000)) != 0u
        ? (uint16_t)~bf16 : (uint16_t)(bf16 | UINT16_C(0x8000));
}

bool compare_bf16_ulps(const char *name, const std::vector<float> &got,
                       const std::vector<float> &expected,
                       uint32_t ulp_limit) {
    CHECK(got.size() == expected.size(), "BF16 ULP comparison shape");
    uint32_t max_ulps = 0u;
    uint64_t mismatches = 0u;
    double max_abs = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        bool got_exact = false, expected_exact = false;
        const uint16_t got_ordered = ordered_bf16(got[i], got_exact);
        const uint16_t expected_ordered =
            ordered_bf16(expected[i], expected_exact);
        CHECK(got_exact && expected_exact,
              "BF16 ULP comparison values lie on BF16 grid");
        const uint32_t ulps = got_ordered > expected_ordered
            ? (uint32_t)(got_ordered - expected_ordered)
            : (uint32_t)(expected_ordered - got_ordered);
        max_ulps = std::max(max_ulps, ulps);
        mismatches += ulps != 0u;
        max_abs = std::max(max_abs,
                           std::fabs((double)got[i] - expected[i]));
    }
    std::fprintf(stderr,
                 "GLM5 MLA compose %-12s count=%zu max_bf16_ulps=%u "
                 "mismatches=%llu max_abs=%.9g\n",
                 name, got.size(), max_ulps,
                 (unsigned long long)mismatches, max_abs);
    CHECK(max_ulps <= ulp_limit, "BF16 ULP numerical envelope");
    return true;
}

bool run_roce_output(const Glm5TestGGUF &gguf,
                     const std::vector<float> &half0,
                     const std::vector<float> &half1,
                     const std::vector<float> &expected,
                     std::vector<float> &composed,
                     std::vector<float> &peer_received,
                     TpGuard *persistent_transport) {
    composed.clear();
    peer_received.clear();
    const char *role_value = std::getenv("DS4_GLM5_TP_ROLE");
    if (!role_value) return true;
    CHECK(std::strcmp(role_value, "leader") == 0 ||
          std::strcmp(role_value, "worker") == 0,
          "MLA TP role is leader or worker");
    const bool leader = std::strcmp(role_value, "leader") == 0;
    const char *host = std::getenv("DS4_GLM5_TP_HOST");
    const char *device = std::getenv("DS4_GLM5_TP_RDMA_DEVICE");
    const char *port_value = std::getenv("DS4_GLM5_TP_PORT");
    const char *timeout = std::getenv("DS4_GLM5_TP_CONNECT_TIMEOUT_SEC");
    if (!timeout || !timeout[0]) timeout = "120";
    CHECK(host && host[0] && device && device[0] && port_value && port_value[0],
          "MLA TP host, device and port are required");
    char *end = nullptr;
    const long port = std::strtol(port_value, &end, 10);
    CHECK(end && *end == '\0' && port >= 1024 && port <= 65535,
          "MLA TP port range");
    CHECK(setenv("DS4_TP_BIG_DIRECT", "1", 1) == 0 &&
          setenv("DS4_TP_BIG_DIRECT_MAX_ROWS", "1", 1) == 0 &&
          setenv("DS4_TP_CONNECT_TIMEOUT_SEC", timeout, 1) == 0 &&
          unsetenv("DS4_TP_VERBS_LIB") == 0,
          "select system-verbs direct RoCE slab");

    ds4_tp_options options = {};
    options.role = leader ? DS4_TP_LEADER : DS4_TP_WORKER;
    options.requested = true;
    options.listen_host = leader ? host : nullptr;
    options.listen_port = leader ? (int)port : 0;
    options.leader_host = leader ? nullptr : host;
    options.leader_port = leader ? 0 : (int)port;
    options.transport = DS4_TP_TRANSPORT_RDMA;
    options.rdma_device = device;
    options.rdma_gid_index = 3;
    options.rdma_gid_index_set = true;

    ds4_tp_identity identity = {};
    identity.gguf_bytes = gguf.size;
    identity.model_id = 3u;
    identity.n_layer = 46u;
    identity.n_embd = kHidden;
    identity.n_vocab = 154880u;
    identity.quant_bits = 4u;
    identity.ctx_size = 1u;
    identity.gate_slot_start = 3u * DS4_TP_GATES_PER_LAYER +
                               DS4_TP_GATE_ATTN;
    identity.gate_slot_step = DS4_TP_GATES_PER_LAYER;
    identity.gates_per_token = 42u;

    TpGuard local_transport;
    TpGuard &transport = persistent_transport ?
        *persistent_transport : local_transport;
    CHECK(!transport.tp && !transport.slab,
          "MLA persistent transport starts empty");
    char error[256] = {};
    CHECK(ds4_tp_create(&transport.tp, &options, &identity,
                        error, sizeof(error)), error);
    CHECK(ds4_tp_is_rdma(transport.tp) &&
          ds4_tp_requires_host_slab(transport.tp),
          "MLA TP selected mapped-host RDMA");
    const uint64_t slab_bytes = ds4_tp_alloc_slab_bytes(transport.tp);
    CHECK(slab_bytes != 0u &&
          hipHostMalloc(&transport.slab, slab_bytes,
                        hipHostMallocMapped) == hipSuccess,
          "allocate MLA RoCE mapped slab");
    CHECK(ds4_tp_attach_slab(transport.tp, transport.slab,
                             error, sizeof(error)), error);
    CHECK(ds4_tp_big_gate_is_rdma_capable(transport.tp),
          "MLA direct gate is RDMA capable");
    const uint64_t contract = fnv1a64(expected.data(),
                                      expected.size() * sizeof(float));
    CHECK(ds4_tp_hash_check(transport.tp, UINT64_C(0x474c4d4d4c410001),
                            contract, error, sizeof(error)) == 1,
          error);

    const std::vector<float> &local = leader ? half0 : half1;
    CHECK(local.size() == kHidden && half0.size() == half1.size() &&
          expected.size() == kHidden,
          "MLA RoCE partial shapes");
    const uint64_t bytes = (uint64_t)kHidden * sizeof(float);
    auto *base = static_cast<uint8_t *>(transport.slab);
    float *out = reinterpret_cast<float *>(
        base + ds4_tp_slab_big_out_offset(transport.tp));
    float *in = reinterpret_cast<float *>(
        base + ds4_tp_slab_big_in_offset(transport.tp));
    std::memcpy(out, local.data(), bytes);
    std::memset(in, 0, bytes);
    const bool direct =
        ds4_tp_big_gate_is_direct(transport.tp, out, in, bytes) != 0;
    CHECK(ds4_tp_big_capacity_rows(transport.tp) >= 1u && direct,
          "MLA partial lies in direct registered regions");
    CHECK(ds4_tp_big_gate_exchange(transport.tp, 3u, 1u,
                                   out, in, bytes),
          "exchange MLA attention partial over RoCE");
    composed.resize(kHidden);
    for (uint32_t i = 0u; i < kHidden; ++i) composed[i] = out[i] + in[i];
    peer_received.assign(in, in + kHidden);
    CHECK(compare_values("roce_tp_sum", composed, expected,
                         8.0e-6, 6.0e-12),
          "RoCE-composed attention output matches oracle");
    const uint64_t composed_hash = fnv1a64(composed.data(), bytes);
    CHECK(ds4_tp_hash_check(transport.tp, UINT64_C(0x474c4d4d4c410002),
                            composed_hash, error, sizeof(error)) == 1,
          error);
    std::fprintf(stderr,
        "GLM5 MLA TP RoCE role=%s device=%s bytes=%llu direct=%d "
        "local_fnv=%016llx peer_fnv=%016llx composed_fnv=%016llx\n",
        role_value, device, (unsigned long long)bytes, direct ? 1 : 0,
        (unsigned long long)fnv1a64(out, bytes),
        (unsigned long long)fnv1a64(in, bytes),
        (unsigned long long)composed_hash);
    return true;
}

bool run_second_gate_probe(TpGuard &transport, bool leader) {
    CHECK(transport.tp && transport.slab,
          "block-session probe has live TP transport and slab");
    constexpr uint64_t bytes = (uint64_t)kHidden * sizeof(float);
    auto *base = static_cast<uint8_t *>(transport.slab);
    float *out = reinterpret_cast<float *>(
        base + ds4_tp_slab_big_out_offset(transport.tp));
    float *in = reinterpret_cast<float *>(
        base + ds4_tp_slab_big_in_offset(transport.tp));
    std::vector<float> local(kHidden), expected_peer(kHidden), composed(kHidden);
    for (uint32_t i = 0u; i < kHidden; ++i) {
        const float leader_value =
            (float)((int)(i % 97u) - 48) * (1.0f / 128.0f);
        const float worker_value =
            (float)((int)(i % 89u) - 44) * (1.0f / 256.0f);
        local[i] = leader ? leader_value : worker_value;
        expected_peer[i] = leader ? worker_value : leader_value;
    }
    std::memcpy(out, local.data(), bytes);
    std::memset(in, 0xa5, bytes);
    CHECK(ds4_tp_big_gate_is_direct(transport.tp, out, in, bytes),
          "second block-stage payload reuses direct registered regions");
    char error[256] = {};
    const uint64_t contract =
        fnv1a64(local.data(), bytes) ^ fnv1a64(expected_peer.data(), bytes);
    CHECK(ds4_tp_hash_check(transport.tp, UINT64_C(0x474c4d3533420001),
                            contract, error, sizeof(error)) == 1,
          error);
    CHECK(ds4_tp_big_gate_exchange(transport.tp, 3u, 2u,
                                   out, in, bytes),
          "exchange second block-stage payload over same RoCE session");
    CHECK(std::memcmp(out, local.data(), bytes) == 0 &&
          std::memcmp(in, expected_peer.data(), bytes) == 0,
          "second exchange overwrites poison without corrupting local payload");
    for (uint32_t i = 0u; i < kHidden; ++i)
        composed[i] = out[i] + in[i];
    const uint64_t composed_hash = fnv1a64(composed.data(), bytes);
    CHECK(ds4_tp_hash_check(transport.tp, UINT64_C(0x474c4d3533420002),
                            composed_hash, error, sizeof(error)) == 1,
          error);
    std::fprintf(stderr,
        "GLM5 block-session RoCE role=%s layer=3 seq=2 bytes=%llu "
        "direct=1 local_fnv=%016llx peer_fnv=%016llx composed_fnv=%016llx\n",
        leader ? "leader" : "worker", (unsigned long long)bytes,
        (unsigned long long)fnv1a64(out, bytes),
        (unsigned long long)fnv1a64(in, bytes),
        (unsigned long long)composed_hash);
    return true;
}

struct FfnPrerouterState {
    ds4_gpu_tensor *hidden = nullptr;
    ds4_gpu_tensor *selected = nullptr;
    ds4_gpu_tensor *weights = nullptr;
    ~FfnPrerouterState() {
        ds4_gpu_tensor_free(weights);
        ds4_gpu_tensor_free(selected);
        ds4_gpu_tensor_free(hidden);
    }
};

bool run_ffn_prerouter(
        const Glm5TestGGUF &gguf, const ds4_gpu_tensor *carried,
        uint64_t hc_fn, uint64_t hc_scale, uint64_t hc_base,
        uint64_t ffn_norm, uint64_t router_weight, uint64_t router_bias,
        const std::vector<float> &expected_split,
        const std::vector<float> &expected_hidden,
        const std::vector<int32_t> &expected_ids,
        const std::vector<float> &expected_weights,
        const char *label, uint64_t &route_hash,
        FfnPrerouterState *retained) {
    ds4_gpu_tensor *flat = ds4_gpu_tensor_alloc(
        (uint64_t)kHc * kHidden * sizeof(float));
    ds4_gpu_tensor *mix = ds4_gpu_tensor_alloc(
        (uint64_t)kHcMix * sizeof(float));
    ds4_gpu_tensor *split = ds4_gpu_tensor_alloc(
        (uint64_t)kHcMix * sizeof(float));
    ds4_gpu_tensor *collapsed = ds4_gpu_tensor_alloc(
        (uint64_t)kHidden * sizeof(float));
    ds4_gpu_tensor *hidden = ds4_gpu_tensor_alloc(
        (uint64_t)kHidden * sizeof(float));
    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(288u * sizeof(float));
    ds4_gpu_tensor *probs = ds4_gpu_tensor_alloc(288u * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(8u * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(8u * sizeof(float));
    const auto release = [&]() {
        ds4_gpu_tensor_free(weights);
        ds4_gpu_tensor_free(selected);
        ds4_gpu_tensor_free(probs);
        ds4_gpu_tensor_free(logits);
        ds4_gpu_tensor_free(hidden);
        ds4_gpu_tensor_free(collapsed);
        ds4_gpu_tensor_free(split);
        ds4_gpu_tensor_free(mix);
        ds4_gpu_tensor_free(flat);
    };
    bool ok = flat && mix && split && collapsed && hidden && logits && probs &&
              selected && weights;
    if (ok) {
        ok = ds4_gpu_rms_norm_plain_rows_tensor(
                 flat, carried, kHc * kHidden, 1u, 1.0e-5f) &&
             ds4_gpu_matmul_bf16_tensor(
                 mix, gguf.map, gguf.size, hc_fn,
                 kHc * kHidden, kHcMix, flat, 1u) &&
             ds4_gpu_hc_split_weighted_sum_norm_tensor(
                 collapsed, hidden, split, mix, carried,
                 gguf.map, gguf.size, hc_scale, hc_base, ffn_norm,
                 kHidden, kHc, 20u, 1.0e-6f, 1.0e-5f) &&
             ds4_gpu_matmul_f32_tensor(
                 logits, gguf.map, gguf.size, router_weight,
                 kHidden, 288u, hidden, 1u) &&
             ds4_gpu_glm_router_select_batch_tensor(
                 selected, weights, probs, gguf.map, gguf.size,
                 router_bias, logits, 288u, 8u, 2.5f, 1u) &&
             ds4_gpu_synchronize();
    }
    std::vector<float> got_split, got_hidden, got_weights;
    std::vector<int32_t> got_ids;
    if (ok) {
        ok = read_tensor(split, kHcMix, got_split) &&
             read_tensor(hidden, kHidden, got_hidden) &&
             read_tensor(selected, 8u, got_ids) &&
             read_tensor(weights, 8u, got_weights);
    }
    if (ok) {
        ok = compare_values("ffn_split", got_split, expected_split,
                            2.0e-6, 1.0e-10) &&
             compare_values("ffn_hidden", got_hidden, expected_hidden,
                            3.0e-6, 1.0e-10) &&
             got_ids == expected_ids &&
             compare_values("router_weight", got_weights,
                            expected_weights, 2.0e-6, 1.0e-10);
    }
    if (ok) {
        const uint64_t split_hash = fnv1a64(
            got_split.data(), got_split.size() * sizeof(float));
        const uint64_t ids_hash = fnv1a64(
            got_ids.data(), got_ids.size() * sizeof(int32_t));
        const uint64_t weights_hash = fnv1a64(
            got_weights.data(), got_weights.size() * sizeof(float));
        route_hash = UINT64_C(1469598103934665603);
        route_hash = fnv1a64_continue(
            route_hash, got_split.data(),
            got_split.size() * sizeof(float));
        route_hash = fnv1a64_continue(
            route_hash, got_ids.data(),
            got_ids.size() * sizeof(int32_t));
        route_hash = fnv1a64_continue(
            route_hash, got_weights.data(),
            got_weights.size() * sizeof(float));
        std::fprintf(stderr,
            "GLM5 block FFN prerouter source=%s split_fnv=%016llx "
            "ids_fnv=%016llx weights_fnv=%016llx route_fnv=%016llx\n",
            label, (unsigned long long)split_hash,
            (unsigned long long)ids_hash,
            (unsigned long long)weights_hash,
            (unsigned long long)route_hash);
    }
    if (ok && retained) {
        CHECK(!retained->hidden && !retained->selected && !retained->weights,
              "retained FFN prerouter state starts empty");
        retained->hidden = hidden;
        retained->selected = selected;
        retained->weights = weights;
        hidden = nullptr;
        selected = nullptr;
        weights = nullptr;
    }
    if (!ok) (void)ds4_gpu_synchronize();
    release();
    CHECK(ok, "execute and validate FFN mHC/router boundary");
    return true;
}

bool run_shared_half_gpu(const Glm5TestGGUF &gguf,
                         const ds4_gpu_tensor *hidden,
                         uint64_t gate_offset, uint64_t up_offset,
                         uint64_t down_offset, uint32_t half,
                         ds4_gpu_tensor **output) {
    constexpr uint32_t full_mid = 2048u;
    constexpr uint32_t half_mid = 1024u;
    constexpr uint64_t q8_block_bytes = 34u;
    constexpr uint64_t q8_qk = 32u;
    CHECK(hidden && output && !*output && half < 2u,
          "shared-expert GPU half geometry");
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(
        (uint64_t)half_mid * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(
        (uint64_t)half_mid * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(
        (uint64_t)half_mid * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(
        (uint64_t)kHidden * sizeof(float));
    const uint64_t row_bytes =
        (kHidden / q8_qk) * q8_block_bytes;
    const uint64_t row_offset =
        (uint64_t)half * half_mid * row_bytes;
    bool ok = gate && up && mid && out;
    if (ok) {
        ok = ds4_gpu_matmul_q8_0_tensor(
                 gate, gguf.map, gguf.size, gate_offset + row_offset,
                 kHidden, half_mid, hidden, 1u) &&
             ds4_gpu_matmul_q8_0_tensor(
                 up, gguf.map, gguf.size, up_offset + row_offset,
                 kHidden, half_mid, hidden, 1u) &&
             ds4_gpu_swiglu_tensor(mid, gate, up, half_mid, 10.0f, 1.0f) &&
             ds4_gpu_matmul_q8_0_kslice_tensor(
                 out, gguf.map, gguf.size, down_offset, full_mid,
                 (uint64_t)half * half_mid, half_mid, kHidden, mid, 0u) &&
             ds4_gpu_synchronize();
    }
    if (!ok) (void)ds4_gpu_synchronize();
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(gate);
    if (!ok) ds4_gpu_tensor_free(out);
    CHECK(ok, "execute shared-expert Q8 rank half from GPU hidden state");
    *output = out;
    return true;
}

bool run_test() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    const char *oracle_prefix =
        std::getenv("DS4_GLM5_MLA_COMPOSE_ORACLE_PREFIX");
    CHECK(model && model[0] && oracle_prefix && oracle_prefix[0],
          "model and MLA composition oracle environment");
    const std::string base(oracle_prefix);

    std::vector<float> hc_residual, expected_hc_post, expected_hc_comb,
        expected_hc_collapsed, hidden, expected_hc_carried,
        expected_q_resid, expected_query,
        expected_kv_norm, expected_qk_low, expected_index_q,
        expected_index_key, expected_pool_gate, expected_head_weights,
        expected_pooled, expected_pool_scores, expected_heads,
        expected_attn_output, expected_ffn_split, expected_ffn_hidden,
        expected_router_weights, expected_shared_output;
    std::vector<uint32_t> valid, expected_pool_valid, expected_selected_pools;
    std::vector<int32_t> expected_pool_indices, expected_selected_tokens,
        expected_router_ids;
    CHECK(read_array(base + ".hc_residual.f32",
                     (uint64_t)kRows * kHc * kHidden, hc_residual) &&
          read_array(base + ".hc_post.f32", kRows * kHc,
                     expected_hc_post) &&
          read_array(base + ".hc_comb.f32", kRows * kHc * kHc,
                     expected_hc_comb) &&
          read_array(base + ".hc_collapsed.f32", kRows * kHidden,
                     expected_hc_collapsed) &&
          read_array(base + ".hidden.f32", kRows * kHidden, hidden) &&
          read_array(base + ".hc_carried.f32", kHc * kHidden,
                     expected_hc_carried) &&
          read_array(base + ".q_resid.f32", kQRank, expected_q_resid) &&
          read_array(base + ".query.f32", kHeads * kHeadDim,
                     expected_query) &&
          read_array(base + ".kv_norm.f32", kRows * kKvLora,
                     expected_kv_norm) &&
          read_array(base + ".qk_low.f32", kHeads * kKvLora,
                     expected_qk_low) &&
          read_array(base + ".valid.u32", kRows, valid) &&
          read_array(base + ".index_q.f32", kIndexHeads * kIndexDim,
                     expected_index_q) &&
          read_array(base + ".index_key.f32", kRows * kIndexDim,
                     expected_index_key) &&
          read_array(base + ".pool_gate.f32", kRows * kIndexDim,
                     expected_pool_gate) &&
          read_array(base + ".head_weights.f32", kIndexHeads,
                     expected_head_weights) &&
          read_array(base + ".pooled.f32", kPools * kIndexDim,
                     expected_pooled) &&
          read_array(base + ".pool_indices.i32", kPools * 4u,
                     expected_pool_indices) &&
          read_array(base + ".pool_valid.u32", kPools,
                     expected_pool_valid) &&
          read_array(base + ".pool_scores.f32", kPools,
                     expected_pool_scores) &&
          read_array(base + ".selected_pools.u32", kSelectedPools,
                     expected_selected_pools) &&
          read_array(base + ".selected_tokens.i32", kSelectedTokens,
                     expected_selected_tokens) &&
          read_array(base + ".heads.f32", kHeads * kHeadDim,
                     expected_heads) &&
          read_array(base + ".attn_output.f32", kHidden,
                     expected_attn_output) &&
          read_array(base + ".ffn_split.f32", kHcMix,
                     expected_ffn_split) &&
          read_array(base + ".ffn_hidden.f32", kHidden,
                     expected_ffn_hidden) &&
          read_array(base + ".router_ids.i32", 8u,
                     expected_router_ids) &&
          read_array(base + ".router_weights.f32", 8u,
                     expected_router_weights) &&
          read_array(base + ".shared_output.f32", kHidden,
                     expected_shared_output),
          "read sparse-MLA heads composition oracle dumps");

    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM5 GGUF directory");
    uint64_t hc_fn = 0u, hc_base = 0u, hc_scale = 0u, attn_norm = 0u,
             q_a = 0u, q_norm = 0u, q_b = 0u, kv_a = 0u,
             kv_norm_w = 0u, k_b = 0u, v_b = 0u, index_q_w = 0u,
             index_k_w = 0u, index_weight_w = 0u, pool_gate_w = 0u,
             pool_ape = 0u, index_norm_w = 0u, index_norm_b = 0u,
             attn_output_w = 0u, hc_ffn_fn = 0u, hc_ffn_base = 0u,
             hc_ffn_scale = 0u, ffn_norm = 0u, router_weight = 0u,
             router_bias = 0u, shared_gate = 0u, shared_up = 0u,
             shared_down = 0u;
    CHECK(gguf.tensor("blk.3.hc_attn_fn.weight", {16384u, 24u}, 30u,
                      hc_fn) &&
          gguf.tensor("blk.3.hc_attn_base.weight", {24u}, 0u, hc_base) &&
          gguf.tensor("blk.3.hc_attn_scale.weight", {3u}, 0u, hc_scale) &&
          gguf.tensor("blk.3.attn_norm.weight", {4096u}, 0u, attn_norm) &&
          gguf.tensor("blk.3.attn_q_a.weight", {4096u, 1536u}, 8u, q_a) &&
          gguf.tensor("blk.3.attn_q_a_norm.weight", {1536u}, 0u, q_norm) &&
          gguf.tensor("blk.3.attn_q_b.weight", {1536u, 16384u}, 8u, q_b) &&
          gguf.tensor("blk.3.attn_kv_a_mqa.weight", {4096u, 512u}, 8u, kv_a) &&
          gguf.tensor("blk.3.attn_kv_a_norm.weight", {512u}, 0u, kv_norm_w) &&
          gguf.tensor("blk.3.attn_k_b.weight", {256u, 512u, 64u}, 8u, k_b) &&
          gguf.tensor("blk.3.attn_v_b.weight", {512u, 256u, 64u}, 8u, v_b) &&
          gguf.tensor("blk.3.indexer.attn_q_b.weight", {1536u, 4096u}, 30u, index_q_w) &&
          gguf.tensor("blk.3.indexer.attn_k.weight", {4096u, 128u}, 30u, index_k_w) &&
          gguf.tensor("blk.3.indexer.proj.weight", {4096u, 32u}, 30u, index_weight_w) &&
          gguf.tensor("blk.3.indexer.pool_gate.weight", {4096u, 128u}, 30u, pool_gate_w) &&
          gguf.tensor("blk.3.indexer.pool_ape.weight", {128u, 4u}, 30u, pool_ape) &&
          gguf.tensor("blk.3.indexer.k_norm.weight", {128u}, 0u, index_norm_w) &&
          gguf.tensor("blk.3.indexer.k_norm.bias", {128u}, 0u, index_norm_b) &&
          gguf.tensor("blk.3.attn_output.weight", {16384u, 4096u}, 8u,
                      attn_output_w) &&
          gguf.tensor("blk.3.hc_ffn_fn.weight", {16384u, 24u}, 30u,
                      hc_ffn_fn) &&
          gguf.tensor("blk.3.hc_ffn_base.weight", {24u}, 0u,
                      hc_ffn_base) &&
          gguf.tensor("blk.3.hc_ffn_scale.weight", {3u}, 0u,
                      hc_ffn_scale) &&
          gguf.tensor("blk.3.ffn_norm.weight", {4096u}, 0u, ffn_norm) &&
          gguf.tensor("blk.3.ffn_gate_inp.weight", {4096u, 288u}, 0u,
                      router_weight) &&
          gguf.tensor("blk.3.exp_probs_b.bias", {288u}, 0u, router_bias),
          "bind real block-3 mHC and sparse-MLA tensors");
    CHECK(gguf.tensor("blk.3.ffn_gate_shexp.weight",
                      {4096u, 2048u}, 8u, shared_gate) &&
          gguf.tensor("blk.3.ffn_up_shexp.weight",
                      {4096u, 2048u}, 8u, shared_up) &&
          gguf.tensor("blk.3.ffn_down_shexp.weight",
                      {2048u, 4096u}, 8u, shared_down),
          "bind real block-3 shared Q8 expert tensors");

    ds4_gpu_config config = {};
    config.n_gpus = 1u;
    config.device_indices[0] = 0u;
    CHECK(ds4_gpu_init_multi(&config) &&
          ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map) &&
          ds4_gpu_set_model_map(gguf.map, gguf.size),
          "initialize gfx1151 and register model map");
    ds4_tp_test_reset_exchange_calls();
    const auto f32 = [](uint64_t n) {
        return ds4_gpu_tensor_alloc(n * sizeof(float));
    };
    ds4_gpu_tensor *d_hc_residual =
        f32((uint64_t)kRows * kHc * kHidden);
    ds4_gpu_tensor *d_hc_flat = f32((uint64_t)kRows * kHc * kHidden);
    ds4_gpu_tensor *d_hc_mix = f32((uint64_t)kRows * kHcMix);
    ds4_gpu_tensor *d_hc_split = f32((uint64_t)kRows * kHcMix);
    ds4_gpu_tensor *d_hc_collapsed = f32((uint64_t)kRows * kHidden);
    ds4_gpu_tensor *d_hidden = f32((uint64_t)kRows * kHidden);
    ds4_gpu_tensor *d_q_a = f32(kQRank);
    ds4_gpu_tensor *d_q_resid = f32(kQRank);
    ds4_gpu_tensor *d_query = f32((uint64_t)kHeads * kHeadDim);
    ds4_gpu_tensor *d_kv_raw = f32((uint64_t)kRows * kKvLora);
    ds4_gpu_tensor *d_kv_norm = f32((uint64_t)kRows * kKvLora);
    ds4_gpu_tensor *d_qk_low = f32((uint64_t)kHeads * kKvLora);
    ds4_gpu_tensor *d_cache = f32((uint64_t)kRows * kKvLora);
    ds4_gpu_tensor *d_index_q = f32((uint64_t)kIndexHeads * kIndexDim);
    ds4_gpu_tensor *d_index_k_raw = f32((uint64_t)kRows * kIndexDim);
    ds4_gpu_tensor *d_index_key = f32((uint64_t)kRows * kIndexDim);
    ds4_gpu_tensor *d_pool_gate = f32((uint64_t)kRows * kIndexDim);
    ds4_gpu_tensor *d_head_weights = f32(kIndexHeads);
    ds4_gpu_tensor *d_valid = ds4_gpu_tensor_alloc(kRows * sizeof(uint32_t));
    ds4_gpu_tensor *d_pooled = f32((uint64_t)kPools * kIndexDim);
    ds4_gpu_tensor *d_pool_indices = ds4_gpu_tensor_alloc(
        (uint64_t)kPools * 4u * sizeof(int32_t));
    ds4_gpu_tensor *d_pool_valid = ds4_gpu_tensor_alloc(
        kPools * sizeof(uint32_t));
    ds4_gpu_tensor *d_pool_scores = f32(kPools);
    ds4_gpu_tensor *d_selected_pools = ds4_gpu_tensor_alloc(
        kSelectedPools * sizeof(uint32_t));
    ds4_gpu_tensor *d_selected_tokens = ds4_gpu_tensor_alloc(
        (uint64_t)kExpandedWidth * sizeof(int32_t));
    ds4_gpu_tensor *d_heads = f32((uint64_t)kHeads * kHeadDim);
    ds4_gpu_tensor *d_attn_full = f32(kHidden);
    ds4_gpu_tensor *d_attn_half0 = f32(kHidden);
    ds4_gpu_tensor *d_attn_half1 = f32(kHidden);
    ds4_gpu_tensor *d_attn_sum = f32(kHidden);
    ds4_gpu_tensor *d_attn_reject = f32(kHidden);
    ds4_gpu_tensor *d_hc_carried_local = f32((uint64_t)kHc * kHidden);
    ds4_gpu_tensor *d_hc_carried_add = f32((uint64_t)kHc * kHidden);
    ds4_gpu_tensor *d_hc_carried_roce = f32((uint64_t)kHc * kHidden);
    CHECK(d_hc_residual && d_hc_flat && d_hc_mix && d_hc_split &&
          d_hc_collapsed && d_hidden && d_q_a && d_q_resid && d_query && d_kv_raw &&
          d_kv_norm && d_qk_low && d_cache && d_index_q &&
          d_index_k_raw && d_index_key && d_pool_gate && d_head_weights &&
          d_valid && d_pooled && d_pool_indices && d_pool_valid &&
          d_pool_scores && d_selected_pools && d_selected_tokens && d_heads &&
          d_attn_full && d_attn_half0 && d_attn_half1 && d_attn_sum &&
          d_attn_reject && d_hc_carried_local && d_hc_carried_add &&
          d_hc_carried_roce,
          "allocate bounded mHC-to-sparse-MLA composition tensors");
    ds4_gpu_tensor *d_last_hidden = ds4_gpu_tensor_view(
        d_hidden, (uint64_t)(kRows - 1u) * kHidden * sizeof(float),
        (uint64_t)kHidden * sizeof(float));
    ds4_gpu_tensor *d_last_hc_residual = ds4_gpu_tensor_view(
        d_hc_residual,
        (uint64_t)(kRows - 1u) * kHc * kHidden * sizeof(float),
        (uint64_t)kHc * kHidden * sizeof(float));
    ds4_gpu_tensor *d_last_hc_split = ds4_gpu_tensor_view(
        d_hc_split, (uint64_t)(kRows - 1u) * kHcMix * sizeof(float),
        (uint64_t)kHcMix * sizeof(float));
    ds4_gpu_tensor *d_last_hc_post = ds4_gpu_tensor_view(
        d_hc_split,
        ((uint64_t)(kRows - 1u) * kHcMix + kHc) * sizeof(float),
        (uint64_t)kHc * sizeof(float));
    ds4_gpu_tensor *d_last_hc_comb = ds4_gpu_tensor_view(
        d_hc_split,
        ((uint64_t)(kRows - 1u) * kHcMix + 2u * kHc) * sizeof(float),
        (uint64_t)kHc * kHc * sizeof(float));
    CHECK(d_last_hidden && d_last_hc_residual && d_last_hc_split &&
          d_last_hc_post && d_last_hc_comb &&
          ds4_gpu_tensor_write(d_hc_residual, 0u, hc_residual.data(),
                               (uint64_t)hc_residual.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(d_valid, 0u, valid.data(),
                               (uint64_t)valid.size() * sizeof(uint32_t)),
          "upload mHC residual and sparse-MLA validity inputs");

    CHECK(ds4_gpu_rms_norm_plain_rows_tensor(
              d_hc_flat, d_hc_residual, kHc * kHidden, kRows, 1.0e-5f) &&
          ds4_gpu_matmul_bf16_tensor(
              d_hc_mix, gguf.map, gguf.size, hc_fn,
              kHc * kHidden, kHcMix, d_hc_flat, kRows) &&
          ds4_gpu_hc_split_weighted_sum_norm_tensor(
              d_hc_collapsed, d_hidden, d_hc_split, d_hc_mix,
              d_hc_residual, gguf.map, gguf.size, hc_scale, hc_base,
              attn_norm, kHidden, kHc, 20u, 1.0e-6f, 1.0e-5f) &&
          ds4_gpu_matmul_q8_0_tensor(d_q_a, gguf.map, gguf.size, q_a,
                                     kHidden, kQRank, d_last_hidden, 1u) &&
          ds4_gpu_rms_norm_weight_tensor(d_q_resid, d_q_a, gguf.map,
                                         gguf.size, q_norm, kQRank, 1.0e-5f) &&
          ds4_gpu_matmul_q8_0_tensor(d_query, gguf.map, gguf.size, q_b,
                                     kQRank, kHeads * kHeadDim,
                                     d_q_resid, 1u) &&
          ds4_gpu_matmul_q8_0_tensor(d_kv_raw, gguf.map, gguf.size, kv_a,
                                     kHidden, kKvLora, d_hidden, kRows) &&
          ds4_gpu_glm_kv_lora_rms_norm_tensor(
              d_kv_norm, d_kv_raw, gguf.map, gguf.size, kv_norm_w,
              kRows, kKvLora, kKvLora, 1.0e-5f) &&
          ds4_gpu_glm_store_compact_kv_tensor(
              d_cache, nullptr, d_kv_norm, d_kv_raw, 0u, kRows, kRows,
              kKvLora, kKvLora, 0u, false) &&
          ds4_gpu_glm_qk_lowrank_typed_tensor(
              d_qk_low, d_query, gguf.map, gguf.size, k_b, 8u, kHeads,
              kKvLora, kHeadDim, kHeadDim),
          "execute real mHC pre-stage, Q/KV trunk and compact NoPE store");

    CHECK(ds4_gpu_matmul_bf16_tensor(
              d_index_q, gguf.map, gguf.size, index_q_w, kQRank,
              kIndexHeads * kIndexDim, d_q_resid, 1u) &&
          ds4_gpu_round_bf16_inplace_tensor(
              d_index_q, kIndexHeads * kIndexDim, 1.0f) &&
          ds4_gpu_matmul_bf16_tensor(
              d_index_k_raw, gguf.map, gguf.size, index_k_w, kHidden,
              kIndexDim, d_hidden, kRows) &&
          ds4_gpu_round_bf16_inplace_tensor(
              d_index_k_raw, (uint64_t)kRows * kIndexDim, 1.0f) &&
          ds4_gpu_glm_store_indexer_k_tensor(
              d_index_key, d_index_k_raw, gguf.map, gguf.size,
              index_norm_w, index_norm_b, 0u, kRows, kRows, kIndexDim,
              0u, 1u, 1.0e-6f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
              false) &&
          ds4_gpu_round_bf16_inplace_tensor(
              d_index_key, (uint64_t)kRows * kIndexDim, 1.0f) &&
          ds4_gpu_matmul_bf16_tensor(
              d_pool_gate, gguf.map, gguf.size, pool_gate_w, kHidden,
              kIndexDim, d_hidden, kRows) &&
          ds4_gpu_round_bf16_inplace_tensor(
              d_pool_gate, (uint64_t)kRows * kIndexDim, 1.0f) &&
          ds4_gpu_matmul_bf16_tensor(
              d_head_weights, gguf.map, gguf.size, index_weight_w,
              kHidden, kIndexHeads, d_last_hidden, 1u) &&
          ds4_gpu_round_bf16_inplace_tensor(
              d_head_weights, kIndexHeads, 1.0f) &&
          ds4_gpu_round_bf16_inplace_tensor(
              d_head_weights, kIndexHeads, 0.1767766952966369f) &&
          ds4_gpu_glm5_kpool_tensor(
              d_pooled, d_pool_indices, d_pool_valid, d_index_key,
              d_pool_gate, d_valid, gguf.map, gguf.size, pool_ape,
              kRows, kIndexDim, 4u, kFirstValid) &&
          ds4_gpu_glm_indexer_score_one_tensor(
              d_pool_scores, d_index_q, d_head_weights, d_pooled, kPools,
              kIndexHeads, kIndexDim, 0.08838834764831845f, false) &&
          ds4_gpu_glm5_mask_pool_scores_tensor(
              d_pool_scores, d_pool_valid, kPools) &&
          ds4_gpu_indexer_topk_tensor(
              d_selected_pools, d_pool_scores, kPools, 1u,
              kSelectedPools) &&
          ds4_gpu_glm5_expand_pool_selection_tensor(
              d_selected_tokens, d_selected_pools, d_pool_indices,
              d_pool_valid, d_valid, kPools, kSelectedPools, kRows,
              kFirstValid, kRows - kFirstValid, kTokenBudget, 4u),
          "execute coupled BF16 indexer and compact selection");

    CHECK(ds4_gpu_glm_attention_indexed_decode_typed_tensor(
              d_heads, d_query, d_qk_low, d_cache, nullptr,
              gguf.map, gguf.size, v_b, 8u, d_selected_tokens,
              kSelectedTokens, kRows, false, kHeads, kKvLora, kHeadDim,
              0u, kHeadDim, 0u, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f) &&
          ds4_gpu_matmul_q8_0_tensor(
              d_attn_full, gguf.map, gguf.size, attn_output_w,
              kHeads * kHeadDim, kHidden, d_heads, 1u) &&
          ds4_gpu_matmul_q8_0_kslice_tensor(
              d_attn_half0, gguf.map, gguf.size, attn_output_w,
              kHeads * kHeadDim, 0u, (kHeads * kHeadDim) / 2u,
              kHidden, d_heads, 0u) &&
          ds4_gpu_matmul_q8_0_kslice_tensor(
              d_attn_half1, gguf.map, gguf.size, attn_output_w,
              kHeads * kHeadDim, (kHeads * kHeadDim) / 2u,
              (kHeads * kHeadDim) / 2u, kHidden, d_heads,
              (kHeads * kHeadDim) / 2u) &&
          ds4_gpu_add_tensor(d_attn_sum, d_attn_half0, d_attn_half1,
                             kHidden) &&
          !ds4_gpu_matmul_q8_0_kslice_tensor(
              d_attn_reject, gguf.map, gguf.size, attn_output_w,
              kHeads * kHeadDim, 0u,
              (kHeads * kHeadDim) / 2u - 1u, kHidden, d_heads, 0u) &&
          ds4_gpu_hc_expand_split_tensor(
              d_hc_carried_local, d_attn_sum, d_last_hc_residual,
              d_last_hc_split, kHidden, kHc) &&
          ds4_gpu_hc_expand_add_tensor(
              d_hc_carried_add, d_attn_half0, d_attn_half1,
              d_last_hc_residual, d_last_hc_post, d_last_hc_comb,
              kHidden, kHc) &&
          ds4_gpu_synchronize(),
          "execute sparse MLA, output reduction and local mHC carry");

    std::vector<float> got_hc_split, got_hc_collapsed, got_hidden,
        got_hc_carried_local, got_hc_carried_add, got_q_resid, got_query,
        got_kv_norm, got_qk_low,
        got_index_q, got_index_key, got_pool_gate, got_head_weights,
        got_pooled, got_pool_scores, got_heads, got_attn_full,
        got_attn_half0, got_attn_half1, got_attn_sum;
    std::vector<uint32_t> got_pool_valid, got_selected_pools;
    std::vector<int32_t> got_pool_indices, got_expanded_tokens;
    CHECK(read_tensor(d_hc_split, (uint64_t)kRows * kHcMix,
                      got_hc_split) &&
          read_tensor(d_hc_collapsed, (uint64_t)kRows * kHidden,
                      got_hc_collapsed) &&
          read_tensor(d_hidden, (uint64_t)kRows * kHidden, got_hidden) &&
          read_tensor(d_hc_carried_local, (uint64_t)kHc * kHidden,
                      got_hc_carried_local) &&
          read_tensor(d_hc_carried_add, (uint64_t)kHc * kHidden,
                      got_hc_carried_add) &&
          read_tensor(d_q_resid, kQRank, got_q_resid) &&
          read_tensor(d_query, kHeads * kHeadDim, got_query) &&
          read_tensor(d_kv_norm, kRows * kKvLora, got_kv_norm) &&
          read_tensor(d_qk_low, kHeads * kKvLora, got_qk_low) &&
          read_tensor(d_index_q, kIndexHeads * kIndexDim, got_index_q) &&
          read_tensor(d_index_key, kRows * kIndexDim, got_index_key) &&
          read_tensor(d_pool_gate, kRows * kIndexDim, got_pool_gate) &&
          read_tensor(d_head_weights, kIndexHeads, got_head_weights) &&
          read_tensor(d_pooled, kPools * kIndexDim, got_pooled) &&
          read_tensor(d_pool_scores, kPools, got_pool_scores) &&
          read_tensor(d_pool_indices, kPools * 4u, got_pool_indices) &&
          read_tensor(d_pool_valid, kPools, got_pool_valid) &&
          read_tensor(d_selected_pools, kSelectedPools, got_selected_pools) &&
          read_tensor(d_selected_tokens, kExpandedWidth, got_expanded_tokens) &&
          read_tensor(d_heads, kHeads * kHeadDim, got_heads) &&
          read_tensor(d_attn_full, kHidden, got_attn_full) &&
          read_tensor(d_attn_half0, kHidden, got_attn_half0) &&
          read_tensor(d_attn_half1, kHidden, got_attn_half1) &&
          read_tensor(d_attn_sum, kHidden, got_attn_sum),
          "read every mHC-to-sparse-MLA composition boundary");

    std::vector<float> got_hc_post((size_t)kRows * kHc);
    std::vector<float> got_hc_comb((size_t)kRows * kHc * kHc);
    for (uint32_t row = 0u; row < kRows; ++row) {
        const float *split_row = got_hc_split.data() + (uint64_t)row * kHcMix;
        std::copy(split_row + kHc, split_row + 2u * kHc,
                  got_hc_post.data() + (uint64_t)row * kHc);
        std::copy(split_row + 2u * kHc, split_row + kHcMix,
                  got_hc_comb.data() + (uint64_t)row * kHc * kHc);
    }

    CHECK(compare_values("hc_post", got_hc_post, expected_hc_post,
                         2.0e-6, 1.0e-10) &&
          compare_values("hc_comb", got_hc_comb, expected_hc_comb,
                         2.0e-6, 1.0e-10) &&
          compare_values("hc_collapsed", got_hc_collapsed,
                         expected_hc_collapsed, 2.0e-6, 1.0e-10) &&
          compare_values("hc_attn_norm", got_hidden, hidden,
                         2.0e-6, 1.0e-10) &&
          compare_values("hc_carried_local", got_hc_carried_local,
                         expected_hc_carried, 1.0e-5, 1.0e-10) &&
          compare_values("hc_carried_add", got_hc_carried_add,
                         expected_hc_carried, 1.0e-5, 1.0e-10) &&
          compare_values("q_resid", got_q_resid, expected_q_resid,
                         5.0e-6, 2.0e-12) &&
          compare_values("query", got_query, expected_query,
                         1.0e-5, 2.0e-12) &&
          compare_values("kv_norm", got_kv_norm, expected_kv_norm,
                         4.0e-6, 2.0e-12) &&
          compare_values("qk_low", got_qk_low, expected_qk_low,
                         8.0e-6, 2.0e-12) &&
          compare_values("index_q", got_index_q, expected_index_q,
                         0.015625, 5.0e-10) &&
          compare_values("index_key", got_index_key, expected_index_key,
                         0.000244140625, 1.0e-8) &&
          compare_bf16_ulps("pool_gate", got_pool_gate,
                            expected_pool_gate, 1u) &&
          compare_values("head_weights", got_head_weights,
                         expected_head_weights, 1.0e-5, 1.0e-10) &&
          compare_values("pooled", got_pooled, expected_pooled,
                         0.03125, 1.0e-8),
          "sparse-MLA heads intermediate numerical gates");
    CHECK(got_pool_indices == expected_pool_indices &&
          got_pool_valid == expected_pool_valid &&
          got_selected_pools == expected_selected_pools,
          "exact pool structure and top-k order");
    CHECK(std::equal(expected_selected_tokens.begin(),
                     expected_selected_tokens.end(),
                     got_expanded_tokens.begin()),
          "exact compact selected-row prefix");
    for (uint32_t i = kSelectedTokens; i < kExpandedWidth; ++i) {
        CHECK(got_expanded_tokens[i] == -1,
              "fixed-width selection suffix is fail-closed padding");
    }
    CHECK(std::isfinite(got_pool_scores[0]) &&
          std::isfinite(got_pool_scores[1]) &&
          got_pool_scores[2] == -std::numeric_limits<float>::max(),
          "invalid tail pool remains finite-min masked");
    got_pool_scores.resize(2u);
    expected_pool_scores.resize(2u);
    CHECK(compare_values("pool_scores", got_pool_scores,
                         expected_pool_scores, 3.0e-5, 1.0e-12) &&
          compare_values("heads", got_heads, expected_heads,
                         1.0e-6, 5.0e-13),
          "selected score and final attention output gates");
    CHECK(compare_values("attn_full", got_attn_full, expected_attn_output,
                         8.0e-6, 6.0e-12) &&
          compare_values("attn_tp_sum", got_attn_sum, expected_attn_output,
                         8.0e-6, 6.0e-12) &&
          compare_values("sum_vs_full", got_attn_sum, got_attn_full,
                         3.0e-6, 6.0e-13),
          "full and two-half Q8 attention output composition");
    std::vector<float> roce_composed, roce_peer;
    TpGuard block_tp;
    CHECK(run_roce_output(gguf, got_attn_half0, got_attn_half1,
                          expected_attn_output, roce_composed, roce_peer,
                          &block_tp),
          "mandatory-RDMA MLA output composition");
    const char *tp_role = std::getenv("DS4_GLM5_TP_ROLE");
    const char *block_probe =
        std::getenv("DS4_GLM5_BLOCK_SESSION_PROBE");
    const char *ffn_prerouter =
        std::getenv("DS4_GLM5_BLOCK_FFN_PREROUTER");
    const char *ffn_shared =
        std::getenv("DS4_GLM5_BLOCK_FFN_SHARED");
    CHECK(!block_probe ||
              ((block_probe[0] == '0' || block_probe[0] == '1') &&
               block_probe[1] == '\0'),
          "block-session probe must be exactly 0 or 1 when set");
    CHECK(!ffn_prerouter ||
              ((ffn_prerouter[0] == '0' || ffn_prerouter[0] == '1') &&
               ffn_prerouter[1] == '\0'),
          "FFN prerouter mode must be exactly 0 or 1 when set");
    CHECK(!ffn_shared ||
              ((ffn_shared[0] == '0' || ffn_shared[0] == '1') &&
               ffn_shared[1] == '\0'),
          "FFN shared mode must be exactly 0 or 1 when set");
    const bool block_probe_enabled =
        block_probe && block_probe[0] == '1';
    const bool ffn_prerouter_enabled =
        ffn_prerouter && ffn_prerouter[0] == '1';
    const bool ffn_shared_enabled = ffn_shared && ffn_shared[0] == '1';
    CHECK(!block_probe_enabled || tp_role,
          "block-session probe requires a TP role");
    CHECK(!ffn_shared_enabled || (ffn_prerouter_enabled && tp_role),
          "FFN shared mode requires prerouter mode and a TP role");
    uint64_t serial_route_hash = 0u;
    FfnPrerouterState serial_ffn, roce_ffn;
    if (ffn_prerouter_enabled) {
        CHECK(run_ffn_prerouter(
                  gguf, d_hc_carried_add, hc_ffn_fn, hc_ffn_scale,
                  hc_ffn_base, ffn_norm, router_weight, router_bias,
                  expected_ffn_split, expected_ffn_hidden,
                  expected_router_ids, expected_router_weights,
                  "serial", serial_route_hash, &serial_ffn),
              "serial attention carry reaches FFN prerouter");
    }
    if (tp_role) {
        const bool leader = std::strcmp(tp_role, "leader") == 0;
        ds4_gpu_tensor *d_owned_partial =
            leader ? d_attn_half0 : d_attn_half1;
        ds4_gpu_tensor *d_peer_partial =
            leader ? d_attn_half1 : d_attn_half0;
        const std::vector<float> poison(kHidden, 0.0f);
        std::vector<float> poison_sum, poison_peer;
        CHECK(roce_composed.size() == kHidden && roce_peer.size() == kHidden &&
              ds4_gpu_tensor_write(d_attn_sum, 0u, poison.data(),
                                   (uint64_t)kHidden * sizeof(float)) &&
              ds4_gpu_tensor_write(d_peer_partial, 0u, poison.data(),
                                   (uint64_t)kHidden * sizeof(float)) &&
              ds4_gpu_synchronize() &&
              read_tensor(d_attn_sum, kHidden, poison_sum) &&
              read_tensor(d_peer_partial, kHidden, poison_peer) &&
              poison_sum == poison && poison_peer == poison,
              "poison prior local reduction and peer partial before upload");
        CHECK(ds4_gpu_tensor_write(d_attn_sum, 0u, roce_composed.data(),
                                   (uint64_t)kHidden * sizeof(float)) &&
              ds4_gpu_tensor_write(d_peer_partial, 0u, roce_peer.data(),
                                   (uint64_t)kHidden * sizeof(float)) &&
              ds4_gpu_synchronize(),
              "upload RoCE-composed row and received peer partial");
        std::vector<float> uploaded_sum, uploaded_peer;
        CHECK(read_tensor(d_attn_sum, kHidden, uploaded_sum) &&
              read_tensor(d_peer_partial, kHidden, uploaded_peer) &&
              uploaded_sum == roce_composed && uploaded_peer == roce_peer &&
              fnv1a64(uploaded_sum.data(),
                      (uint64_t)kHidden * sizeof(float)) ==
                  fnv1a64(roce_composed.data(),
                          (uint64_t)kHidden * sizeof(float)) &&
              fnv1a64(uploaded_peer.data(),
                      (uint64_t)kHidden * sizeof(float)) ==
                  fnv1a64(roce_peer.data(),
                          (uint64_t)kHidden * sizeof(float)) &&
              ds4_gpu_hc_expand_add_tensor(
                  d_hc_carried_roce, d_owned_partial, d_peer_partial,
                  d_last_hc_residual, d_last_hc_post, d_last_hc_comb,
                  kHidden, kHc) &&
              ds4_gpu_synchronize(),
              "consume received peer partial in production-form mHC add");
        std::fprintf(stderr,
                     "GLM5 MLA RoCE upload sum_fnv=%016llx peer_fnv=%016llx\n",
                     (unsigned long long)fnv1a64(
                         uploaded_sum.data(),
                         (uint64_t)kHidden * sizeof(float)),
                     (unsigned long long)fnv1a64(
                         uploaded_peer.data(),
                         (uint64_t)kHidden * sizeof(float)));
        std::vector<float> got_hc_carried_roce;
        CHECK(read_tensor(d_hc_carried_roce, (uint64_t)kHc * kHidden,
                          got_hc_carried_roce) &&
              compare_values("hc_carried_roce", got_hc_carried_roce,
                             expected_hc_carried, 1.0e-5, 1.0e-10),
              "RoCE attention output reaches correct mHC carry");
        if (ffn_prerouter_enabled) {
            uint64_t roce_route_hash = 0u;
            CHECK(run_ffn_prerouter(
                      gguf, d_hc_carried_roce, hc_ffn_fn, hc_ffn_scale,
                      hc_ffn_base, ffn_norm, router_weight, router_bias,
                      expected_ffn_split, expected_ffn_hidden,
                      expected_router_ids, expected_router_weights,
                      "roce", roce_route_hash, &roce_ffn) &&
                  serial_route_hash == roce_route_hash,
                  "serial and RoCE FFN route states are identical");
            char error[256] = {};
            CHECK(ds4_tp_hash_check(
                      block_tp.tp, UINT64_C(0x474c4d3533420010),
                      roce_route_hash, error, sizeof(error)) == 1,
                  error);
            if (ffn_shared_enabled) {
                ds4_gpu_tensor *serial_half0 = nullptr;
                ds4_gpu_tensor *serial_half1 = nullptr;
                ds4_gpu_tensor *serial_sum = ds4_gpu_tensor_alloc(
                    (uint64_t)kHidden * sizeof(float));
                ds4_gpu_tensor *roce_local = nullptr;
                const uint32_t local_half =
                    std::strcmp(tp_role, "leader") == 0 ? 0u : 1u;
                CHECK(serial_sum &&
                      run_shared_half_gpu(
                          gguf, serial_ffn.hidden, shared_gate, shared_up,
                          shared_down, 0u, &serial_half0) &&
                      run_shared_half_gpu(
                          gguf, serial_ffn.hidden, shared_gate, shared_up,
                          shared_down, 1u, &serial_half1) &&
                      ds4_gpu_add_tensor(serial_sum, serial_half0,
                                         serial_half1, kHidden) &&
                      run_shared_half_gpu(
                          gguf, roce_ffn.hidden, shared_gate, shared_up,
                          shared_down, local_half, &roce_local) &&
                      ds4_gpu_synchronize(),
                      "execute serial and rank-local shared Q8 expert");
                std::vector<float> got_shared, got_serial_local,
                    got_roce_local;
                CHECK(read_tensor(serial_sum, kHidden, got_shared) &&
                      read_tensor(local_half == 0u ? serial_half0 :
                                                     serial_half1,
                                  kHidden, got_serial_local) &&
                      read_tensor(roce_local, kHidden, got_roce_local) &&
                      compare_values("shared_full", got_shared,
                                     expected_shared_output,
                                     8.0e-7, 1.0e-10) &&
                      got_serial_local == got_roce_local,
                      "shared Q8 serial/full and RoCE-local numerical gates");
                std::fprintf(stderr,
                    "GLM5 block FFN shared role=%s local_half=%u "
                    "local_fnv=%016llx full_fnv=%016llx cache_bytes=0\n",
                    tp_role, local_half,
                    (unsigned long long)fnv1a64(
                        got_roce_local.data(),
                        got_roce_local.size() * sizeof(float)),
                    (unsigned long long)fnv1a64(
                        got_shared.data(),
                        got_shared.size() * sizeof(float)));
                ds4_gpu_tensor_free(roce_local);
                ds4_gpu_tensor_free(serial_sum);
                ds4_gpu_tensor_free(serial_half1);
                ds4_gpu_tensor_free(serial_half0);
            }
        }
        if (block_probe_enabled) {
            CHECK(run_second_gate_probe(
                      block_tp, std::strcmp(tp_role, "leader") == 0),
                  "reuse persistent RoCE session for second block stage");
        }
    }
    if (!std::getenv("DS4_GLM5_TP_ROLE")) {
        CHECK(ds4_tp_test_get_exchange_calls() == 0u,
              "rank-local sparse-MLA heads invokes no TP exchange");
    } else {
        CHECK(ds4_tp_test_get_exchange_calls() ==
                  (block_probe_enabled ? 2u : 1u),
              "RoCE block invokes the expected TP exchange count");
    }

    ds4_gpu_tensor_free(d_last_hc_comb);
    ds4_gpu_tensor_free(d_last_hc_post);
    ds4_gpu_tensor_free(d_last_hc_split);
    ds4_gpu_tensor_free(d_last_hc_residual);
    ds4_gpu_tensor_free(d_last_hidden);
    ds4_gpu_tensor_free(d_hc_carried_roce);
    ds4_gpu_tensor_free(d_hc_carried_add);
    ds4_gpu_tensor_free(d_hc_carried_local);
    ds4_gpu_tensor_free(d_attn_reject);
    ds4_gpu_tensor_free(d_attn_sum);
    ds4_gpu_tensor_free(d_attn_half1);
    ds4_gpu_tensor_free(d_attn_half0);
    ds4_gpu_tensor_free(d_attn_full);
    ds4_gpu_tensor_free(d_heads);
    ds4_gpu_tensor_free(d_selected_tokens);
    ds4_gpu_tensor_free(d_selected_pools);
    ds4_gpu_tensor_free(d_pool_scores);
    ds4_gpu_tensor_free(d_pool_valid);
    ds4_gpu_tensor_free(d_pool_indices);
    ds4_gpu_tensor_free(d_pooled);
    ds4_gpu_tensor_free(d_valid);
    ds4_gpu_tensor_free(d_head_weights);
    ds4_gpu_tensor_free(d_pool_gate);
    ds4_gpu_tensor_free(d_index_key);
    ds4_gpu_tensor_free(d_index_k_raw);
    ds4_gpu_tensor_free(d_index_q);
    ds4_gpu_tensor_free(d_cache);
    ds4_gpu_tensor_free(d_qk_low);
    ds4_gpu_tensor_free(d_kv_norm);
    ds4_gpu_tensor_free(d_kv_raw);
    ds4_gpu_tensor_free(d_query);
    ds4_gpu_tensor_free(d_q_resid);
    ds4_gpu_tensor_free(d_q_a);
    ds4_gpu_tensor_free(d_hidden);
    ds4_gpu_tensor_free(d_hc_collapsed);
    ds4_gpu_tensor_free(d_hc_split);
    ds4_gpu_tensor_free(d_hc_mix);
    ds4_gpu_tensor_free(d_hc_flat);
    ds4_gpu_tensor_free(d_hc_residual);
    ds4_gpu_cleanup();
    std::fprintf(stderr,
                 "PASS same-GGUF GLM5 block-3 mHC-to-sparse-MLA gate\n");
    return true;
}

}  // namespace

int main() { return run_test() ? 0 : 1; }
