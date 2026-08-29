#include "ds4_glm5_next_exec.h"
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
extern "C" {
#include "ds4.h"
#include "ds4_tp.h"
}
#include "tests/glm5_gguf_test.hpp"
#include "tests/glm5_next_real_offsets.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#define CHECK(expr, message) do {                                        \
    if (!(expr)) {                                                       \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__); \
        return false;                                                    \
    }                                                                    \
} while (0)

namespace {

constexpr uint32_t kWidth = 4096u;
constexpr uint32_t kHcWidth = 4u * kWidth;
constexpr uint64_t kPrefixFNV = UINT64_C(0x6b704c8b12a398ef);

uint64_t fnv64(const void *data, uint64_t bytes) {
    const auto *p = static_cast<const uint8_t *>(data);
    uint64_t hash = UINT64_C(1469598103934665603);
    for (uint64_t i = 0u; i < bytes; ++i) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

bool buffer_all_zero(const std::vector<uint8_t> &buffer) {
    return std::all_of(buffer.begin(), buffer.end(),
                       [](uint8_t value) { return value == 0u; });
}

bool kda_state_has_exact_local_ownership(
        const ds4_glm5_kda_layer_state &state, uint32_t rank,
        uint64_t *owned_fnv) {
    constexpr uint64_t kHistoryHalfBytes =
        (uint64_t)(DS4_GLM5_KDA_CHANNELS / 2u) *
        DS4_GLM5_KDA_HISTORY * sizeof(float);
    constexpr uint64_t kRecurrentHalfBytes =
        (uint64_t)(DS4_GLM5_KDA_HEADS / 2u) *
        DS4_GLM5_KDA_HEAD_DIM * DS4_GLM5_KDA_HEAD_DIM * sizeof(float);
    if (!state.valid || rank > 1u || !state.q_history || !state.k_history ||
        !state.v_history || !state.recurrent || !owned_fnv) return false;

    const ds4_gpu_tensor *tensors[] = {
        state.q_history, state.k_history, state.v_history, state.recurrent};
    const uint64_t half_bytes[] = {
        kHistoryHalfBytes, kHistoryHalfBytes, kHistoryHalfBytes,
        kRecurrentHalfBytes};
    uint64_t combined = UINT64_C(1469598103934665603);
    for (size_t i = 0u; i < 4u; ++i) {
        std::vector<uint8_t> owned(half_bytes[i]);
        std::vector<uint8_t> peer(half_bytes[i]);
        if (!ds4_gpu_tensor_read(
                tensors[i], (uint64_t)rank * half_bytes[i],
                owned.data(), half_bytes[i]) ||
            !ds4_gpu_tensor_read(
                tensors[i], (uint64_t)(1u - rank) * half_bytes[i],
                peer.data(), half_bytes[i]) ||
            buffer_all_zero(owned) || !buffer_all_zero(peer)) return false;
        const uint64_t part = fnv64(owned.data(), half_bytes[i]);
        combined ^= part;
        combined *= UINT64_C(1099511628211);
    }
    *owned_fnv = combined;
    return true;
}

struct VectorError {
    double nrmse = 0.0;
    double cosine = 0.0;
    double max_abs = 0.0;
};

VectorError vector_error_data(const float *reference, const float *candidate,
                              size_t count) {
    VectorError result;
    if (!reference || !candidate || count == 0u) {
        result.nrmse = INFINITY;
        result.cosine = -1.0;
        result.max_abs = INFINITY;
        return result;
    }
    long double diff2 = 0.0L, ref2 = 0.0L;
    long double dot = 0.0L, candidate2 = 0.0L;
    for (size_t i = 0u; i < count; ++i) {
        const long double a = reference[i];
        const long double b = candidate[i];
        const long double d = b - a;
        diff2 += d * d;
        ref2 += a * a;
        dot += a * b;
        candidate2 += b * b;
        result.max_abs = std::max(result.max_abs, std::fabs((double)d));
    }
    result.nrmse = ref2 > 0.0L ? std::sqrt((double)(diff2 / ref2)) :
        (diff2 == 0.0L ? 0.0 : INFINITY);
    result.cosine = ref2 > 0.0L && candidate2 > 0.0L ?
        (double)(dot / std::sqrt(ref2 * candidate2)) : 0.0;
    return result;
}

VectorError vector_error(const std::vector<float> &reference,
                         const std::vector<float> &candidate) {
    if (reference.size() != candidate.size()) {
        VectorError invalid;
        invalid.nrmse = INFINITY;
        invalid.cosine = -1.0;
        invalid.max_abs = INFINITY;
        return invalid;
    }
    return vector_error_data(
        reference.data(), candidate.data(), reference.size());
}

struct TensorGuard {
    ds4_gpu_tensor *value = nullptr;
    ~TensorGuard() { ds4_gpu_tensor_free(value); }
};

struct StateGuard {
    ds4_glm5_next_state value = {};
    ~StateGuard() { ds4_glm5_next_state_free(&value); }
};

struct WorkspaceGuard {
    ds4_glm5_next_workspace *value = nullptr;
    ~WorkspaceGuard() { ds4_glm5_next_workspace_destroy(value); }
};

struct CodecGuard {
    ds4_glm5_next_text_codec *value = nullptr;
    ~CodecGuard() { ds4_glm5_next_text_codec_close(value); }
};

struct TokensGuard {
    ds4_tokens value = {};
    ~TokensGuard() { ds4_tokens_free(&value); }
};

struct TpGuard {
    ds4_tp *tp = nullptr;
    ds4_gpu_tensor *slab = nullptr;
    ds4_gpu_tensor *big_out = nullptr;
    ds4_gpu_tensor *big_in = nullptr;
    ~TpGuard() {
        ds4_gpu_tensor_free(big_in);
        ds4_gpu_tensor_free(big_out);
        if (tp) ds4_tp_free(tp);
        ds4_gpu_tensor_free(slab);
    }
};

bool create_tp(const Glm5TestGGUF &gguf, bool leader,
               const char *host, const char *device, int port,
               uint32_t context_capacity,
               TpGuard &guard, ds4_glm5_next_exec_ctx &exec,
               uint64_t &sequence) {
    char direct_rows[32] = {};
    CHECK(std::snprintf(direct_rows, sizeof(direct_rows), "%u",
                        context_capacity) > 0,
          "format exact TP batch row capacity");
    CHECK(setenv("DS4_TP_BIG_DIRECT", "1", 1) == 0 &&
          setenv("DS4_TP_BIG_DIRECT_MAX_ROWS", direct_rows, 1) == 0,
          "select mandatory direct RDMA");
    ds4_tp_options options = {};
    options.role = leader ? DS4_TP_LEADER : DS4_TP_WORKER;
    options.requested = true;
    options.listen_host = leader ? host : nullptr;
    options.listen_port = leader ? port : 0;
    options.leader_host = leader ? nullptr : host;
    options.leader_port = leader ? 0 : port;
    options.transport = DS4_TP_TRANSPORT_RDMA;
    options.rdma_device = device;
    const char *gid_text = getenv("DS4_GLM5_TP_RDMA_GID_INDEX");
    if (gid_text && gid_text[0]) {
        char *end = nullptr;
        errno = 0;
        const long gid = strtol(gid_text, &end, 10);
        CHECK(errno == 0 && end && *end == '\0' && gid >= 0 && gid <= 255,
              "valid explicit RDMA GID index");
        options.rdma_gid_index = (int)gid;
        options.rdma_gid_index_set = true;
    }

    ds4_tp_identity identity = {};
    identity.gguf_bytes = gguf.size;
    identity.model_id = 3u;
    identity.n_layer = 46u;
    identity.n_embd = kWidth;
    identity.n_vocab = 154880u;
    identity.quant_bits = 4u;
    identity.ctx_size = context_capacity;
    const char *disable_small_gate =
        std::getenv("DS4_GLM5_DISABLE_SMALL_GATE");
    CHECK(!disable_small_gate ||
              std::strcmp(disable_small_gate, "0") == 0 ||
              std::strcmp(disable_small_gate, "1") == 0,
          "small-gate rollback selector is exactly 0 or 1");
    const bool small_gate_enabled = !disable_small_gate ||
        std::strcmp(disable_small_gate, "1") != 0;
    identity.runtime_features =
        DS4_TP_FEATURE_Q4K_WMMA | DS4_TP_FEATURE_Q4K_KSHARD |
        (small_gate_enabled ? DS4_TP_FEATURE_GLM5_SMALL_GATE : 0u);
    const char *kda_tp = std::getenv("DS4_GLM5_KDA_TP");
    CHECK(!kda_tp || std::strcmp(kda_tp, "0") == 0 ||
              std::strcmp(kda_tp, "1") == 0,
          "KDA-TP selector is exactly 0 or 1");
    if (kda_tp && std::strcmp(kda_tp, "1") == 0)
        identity.runtime_features |= DS4_TP_FEATURE_GLM5_KDA_TP;
    identity.gate_slot_start =
        (identity.runtime_features & DS4_TP_FEATURE_GLM5_KDA_TP) != 0u ?
            0u : 3u * DS4_TP_GATES_PER_LAYER;
    identity.gate_slot_step = 1u;
    CHECK(ds4_glm5_next_build_tp_gate_mask(identity.gate_slot_mask,
                                            &identity.gates_per_token,
                                            identity.runtime_features),
          "GLM5.3 hybrid TP gate schedule");

    char error[256] = {};
    CHECK(ds4_tp_create(&guard.tp, &options, &identity,
                        error, sizeof(error)), error);
    CHECK(ds4_tp_is_rdma(guard.tp) &&
          ds4_tp_requires_host_slab(guard.tp),
          "GLM5 layer3 test selected RoCE mapped-host RDMA");
    const uint64_t slab_bytes = ds4_tp_alloc_slab_bytes(guard.tp);
    guard.slab = ds4_gpu_tensor_alloc_rdma_host(slab_bytes);
    CHECK(guard.slab &&
          ds4_tp_attach_slab(guard.tp, ds4_gpu_tensor_contents(guard.slab),
                             error, sizeof(error)), error);
    const uint64_t row_bytes = (uint64_t)kWidth * sizeof(float);
    const uint64_t direct_bytes = (uint64_t)context_capacity * row_bytes;
    guard.big_out = ds4_gpu_tensor_view(
        guard.slab, ds4_tp_slab_big_out_offset(guard.tp), direct_bytes);
    guard.big_in = ds4_gpu_tensor_view(
        guard.slab, ds4_tp_slab_big_in_offset(guard.tp), direct_bytes);
    CHECK(guard.big_out && guard.big_in &&
          ds4_tp_big_gate_is_direct(
              guard.tp, ds4_gpu_tensor_contents(guard.big_out),
              ds4_gpu_tensor_contents(guard.big_in), direct_bytes),
          "GLM5 layer3 uses direct registered big-gate rows");

    exec.tp = guard.tp;
    exec.tp_rank = leader ? 0u : 1u;
    exec.tp_slab = guard.slab;
    exec.tp_big_out = guard.big_out;
    exec.tp_big_in = guard.big_in;
    exec.tp_big_out_host = ds4_gpu_tensor_contents(guard.big_out);
    exec.tp_big_in_host = ds4_gpu_tensor_contents(guard.big_in);
    exec.tp_sequence = &sequence;
    return true;
}

struct LayerTiming {
    double kda_dense_ms = 0.0;
    double kda_routed_ms = 0.0;
    double mla_routed_ms = 0.0;
    uint64_t kda_dense_calls = 0u;
    uint64_t kda_routed_calls = 0u;
    uint64_t mla_routed_calls = 0u;
};

bool execute_full_token(ds4_glm5_next_exec_ctx &exec,
                        ds4_glm5_next_state &state,
                        ds4_glm5_next_workspace *workspace,
                        uint32_t token,
                        ds4_gpu_tensor *&current,
                        ds4_gpu_tensor *&output,
                        LayerTiming *timing = nullptr) {
    CHECK(ds4_glm5_next_embed_token(&exec, token, current),
          "embed text token");
    for (uint32_t il = 0u; il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il) {
        const auto begin = std::chrono::steady_clock::now();
        CHECK(ds4_glm5_next_layer_forward(
                  &exec, il, &state, workspace, current, output),
              "execute text token through complete trunk");
        if (timing) {
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();
            const ds4_glm5_next_layer_offsets &layer = exec.model->layer[il];
            if (layer.attention == DS4_GLM5_NEXT_ATTN_KDA &&
                layer.ffn == DS4_GLM5_NEXT_FFN_DENSE) {
                timing->kda_dense_ms += ms;
                timing->kda_dense_calls++;
            } else if (layer.attention == DS4_GLM5_NEXT_ATTN_KDA &&
                       layer.ffn == DS4_GLM5_NEXT_FFN_ROUTED) {
                timing->kda_routed_ms += ms;
                timing->kda_routed_calls++;
            } else if (layer.attention == DS4_GLM5_NEXT_ATTN_MLA &&
                       layer.ffn == DS4_GLM5_NEXT_FFN_ROUTED) {
                timing->mla_routed_ms += ms;
                timing->mla_routed_calls++;
            }
        }
        std::swap(current, output);
    }
    return true;
}

bool execute_full_batch(ds4_glm5_next_exec_ctx &exec,
                        ds4_glm5_next_state &state,
                        ds4_glm5_next_workspace *workspace,
                        const ds4_gpu_tensor *tokens,
                        uint32_t n_tokens,
                        ds4_gpu_tensor *&current,
                        ds4_gpu_tensor *&output,
                        LayerTiming *timing = nullptr) {
    CHECK(ds4_glm5_next_embed_tokens(
              &exec, tokens, n_tokens, current),
          "embed text prompt batch");
    for (uint32_t il = 0u; il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il) {
        const auto begin = std::chrono::steady_clock::now();
        CHECK(ds4_glm5_next_layer_forward_batch(
                  &exec, il, &state, workspace, current, output, n_tokens),
              "execute text prompt batch through complete trunk");
        if (timing) {
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();
            const ds4_glm5_next_layer_offsets &layer = exec.model->layer[il];
            if (layer.attention == DS4_GLM5_NEXT_ATTN_KDA &&
                layer.ffn == DS4_GLM5_NEXT_FFN_DENSE) {
                timing->kda_dense_ms += ms;
                timing->kda_dense_calls++;
            } else if (layer.attention == DS4_GLM5_NEXT_ATTN_KDA &&
                       layer.ffn == DS4_GLM5_NEXT_FFN_ROUTED) {
                timing->kda_routed_ms += ms;
                timing->kda_routed_calls++;
            } else if (layer.attention == DS4_GLM5_NEXT_ATTN_MLA &&
                       layer.ffn == DS4_GLM5_NEXT_FFN_ROUTED) {
                timing->mla_routed_ms += ms;
                timing->mla_routed_calls++;
            }
        }
        std::swap(current, output);
    }
    return true;
}

bool finite_error(const VectorError &error) {
    return std::isfinite(error.nrmse) && std::isfinite(error.cosine) &&
           std::isfinite(error.max_abs);
}

bool finite_nonzero_vector(const std::vector<float> &values) {
    bool nonzero = false;
    for (float value : values) {
        if (!std::isfinite(value)) return false;
        nonzero = nonzero || value != 0.0f;
    }
    return nonzero;
}

bool compare_full_prompt_states(const char *role,
                                const ds4_glm5_next_state &reference,
                                const ds4_glm5_next_state &candidate,
                                uint32_t n_tokens) {
    CHECK(reference.valid && candidate.valid,
          "scalar and batch prompt states remain valid");
    constexpr uint32_t kKdaLayers[] = {0u, 20u};
    constexpr uint64_t kHistoryFloats =
        (uint64_t)DS4_GLM5_KDA_HISTORY * DS4_GLM5_KDA_CHANNELS;
    constexpr uint64_t kRecurrentFloats =
        (uint64_t)DS4_GLM5_KDA_HEADS * DS4_GLM5_KDA_HEAD_DIM *
        DS4_GLM5_KDA_HEAD_DIM;
    for (uint32_t il : kKdaLayers) {
        const ds4_glm5_kda_layer_state &ref = reference.kda.layer[il];
        const ds4_glm5_kda_layer_state &got = candidate.kda.layer[il];
        CHECK(ref.valid && got.valid && ref.token_count == n_tokens &&
                  got.token_count == n_tokens,
              "selected KDA prompt state has the complete token count");
        std::vector<float> ref_q(kHistoryFloats), got_q(kHistoryFloats);
        std::vector<float> ref_k(kHistoryFloats), got_k(kHistoryFloats);
        std::vector<float> ref_v(kHistoryFloats), got_v(kHistoryFloats);
        std::vector<float> ref_recurrent(kRecurrentFloats);
        std::vector<float> got_recurrent(kRecurrentFloats);
        CHECK(ds4_gpu_tensor_read(ref.q_history, 0u, ref_q.data(),
                                   ref_q.size() * sizeof(float)) &&
                  ds4_gpu_tensor_read(got.q_history, 0u, got_q.data(),
                                       got_q.size() * sizeof(float)) &&
                  ds4_gpu_tensor_read(ref.k_history, 0u, ref_k.data(),
                                       ref_k.size() * sizeof(float)) &&
                  ds4_gpu_tensor_read(got.k_history, 0u, got_k.data(),
                                       got_k.size() * sizeof(float)) &&
                  ds4_gpu_tensor_read(ref.v_history, 0u, ref_v.data(),
                                       ref_v.size() * sizeof(float)) &&
                  ds4_gpu_tensor_read(got.v_history, 0u, got_v.data(),
                                       got_v.size() * sizeof(float)) &&
                  ds4_gpu_tensor_read(ref.recurrent, 0u,
                                       ref_recurrent.data(),
                                       ref_recurrent.size() * sizeof(float)) &&
                  ds4_gpu_tensor_read(got.recurrent, 0u,
                                       got_recurrent.data(),
                                       got_recurrent.size() * sizeof(float)),
              "read selected scalar and batch KDA prompt state");
        const VectorError q_error = vector_error(ref_q, got_q);
        const VectorError k_error = vector_error(ref_k, got_k);
        const VectorError v_error = vector_error(ref_v, got_v);
        const VectorError recurrent_error =
            vector_error(ref_recurrent, got_recurrent);
        std::fprintf(stderr,
            "GLM5 complete batch KDA state role=%s layer=%u "
            "q_nrmse=%.9g k_nrmse=%.9g v_nrmse=%.9g "
            "recurrent_nrmse=%.9g recurrent_cosine=%.12g "
            "recurrent_max_abs=%.9g\n",
            role, il, q_error.nrmse, k_error.nrmse, v_error.nrmse,
            recurrent_error.nrmse, recurrent_error.cosine,
            recurrent_error.max_abs);
        /* Layer zero checks the batch KDA state almost directly. Layer 20
         * also contains the cumulative Lane-B drift from all preceding Q4
         * routed layers, so freeze a wider but still bounded envelope there. */
        const double history_nrmse_bound = il == 0u ? 1.0e-5 : 8.0e-2;
        const double recurrent_nrmse_bound = il == 0u ? 1.0e-5 : 5.5e-2;
        const double recurrent_cosine_bound = il == 0u ? 0.999999 : 0.9985;
        const double recurrent_max_abs_bound = il == 0u ? 1.0e-5 : 5.0e-2;
        CHECK(finite_error(q_error) && finite_error(k_error) &&
                  finite_error(v_error) && finite_error(recurrent_error) &&
                  q_error.nrmse <= history_nrmse_bound &&
                  k_error.nrmse <= history_nrmse_bound &&
                  v_error.nrmse <= history_nrmse_bound &&
                  recurrent_error.nrmse <= recurrent_nrmse_bound &&
                  recurrent_error.cosine >= recurrent_cosine_bound &&
                  recurrent_error.max_abs <= recurrent_max_abs_bound,
              "selected batch KDA prompt state stays in its Lane-B envelope");
    }

    constexpr uint32_t kMlaLayer = 3u;
    const ds4_glm5_next_mla_state &ref_mla = reference.mla[kMlaLayer];
    const ds4_glm5_next_mla_state &got_mla = candidate.mla[kMlaLayer];
    const uint64_t kv_count =
        (uint64_t)n_tokens * DS4_GLM5_NEXT_MLA_KV_WIDTH;
    const uint64_t pool_count =
        (uint64_t)(n_tokens / 4u) * DS4_GLM5_NEXT_INDEX_WIDTH;
    const uint64_t tail_count =
        (uint64_t)(n_tokens % 4u) * DS4_GLM5_NEXT_INDEX_WIDTH;
    CHECK(ref_mla.valid && got_mla.valid &&
              ref_mla.token_count == n_tokens &&
              got_mla.token_count == n_tokens &&
              ref_mla.complete_pools == n_tokens / 4u &&
              got_mla.complete_pools == n_tokens / 4u &&
              ref_mla.tail_count == n_tokens % 4u &&
              got_mla.tail_count == n_tokens % 4u,
          "selected MLA prompt state has exact counters");
    std::vector<float> ref_kv(kv_count), got_kv(kv_count);
    std::vector<float> ref_pool(pool_count), got_pool(pool_count);
    std::vector<float> ref_tail(tail_count), got_tail(tail_count);
    std::vector<float> ref_gate(tail_count), got_gate(tail_count);
    CHECK(ds4_gpu_tensor_read(ref_mla.compact_kv, 0u, ref_kv.data(),
                               ref_kv.size() * sizeof(float)) &&
              ds4_gpu_tensor_read(got_mla.compact_kv, 0u, got_kv.data(),
                                   got_kv.size() * sizeof(float)) &&
              (pool_count == 0u ||
               (ds4_gpu_tensor_read(ref_mla.index_pool, 0u,
                                     ref_pool.data(),
                                     ref_pool.size() * sizeof(float)) &&
                ds4_gpu_tensor_read(got_mla.index_pool, 0u,
                                     got_pool.data(),
                                     got_pool.size() * sizeof(float)))) &&
              (tail_count == 0u ||
               (ds4_gpu_tensor_read(ref_mla.index_tail, 0u,
                                     ref_tail.data(),
                                     ref_tail.size() * sizeof(float)) &&
                ds4_gpu_tensor_read(got_mla.index_tail, 0u,
                                     got_tail.data(),
                                     got_tail.size() * sizeof(float)) &&
                ds4_gpu_tensor_read(ref_mla.pool_gate_tail, 0u,
                                     ref_gate.data(),
                                     ref_gate.size() * sizeof(float)) &&
                ds4_gpu_tensor_read(got_mla.pool_gate_tail, 0u,
                                     got_gate.data(),
                                     got_gate.size() * sizeof(float)))),
          "read selected scalar and batch MLA prompt state");
    const VectorError kv_error = vector_error(ref_kv, got_kv);
    const VectorError pool_error = pool_count ?
        vector_error(ref_pool, got_pool) : VectorError{};
    const VectorError tail_error = tail_count ?
        vector_error(ref_tail, got_tail) : VectorError{};
    const VectorError gate_error = tail_count ?
        vector_error(ref_gate, got_gate) : VectorError{};
    std::fprintf(stderr,
        "GLM5 complete batch MLA state role=%s layer=%u "
        "kv_nrmse=%.9g pool_nrmse=%.9g tail_nrmse=%.9g "
        "gate_nrmse=%.9g\n",
        role, kMlaLayer, kv_error.nrmse, pool_error.nrmse,
        tail_error.nrmse, gate_error.nrmse);
    const double pool_nrmse_bound = n_tokens > 64u ? 5.0e-5 : 1.0e-5;
    CHECK(finite_error(kv_error) && kv_error.nrmse <= 1.0e-5 &&
              (!pool_count ||
               (finite_nonzero_vector(ref_pool) &&
                finite_error(pool_error) &&
                pool_error.nrmse <= pool_nrmse_bound)) &&
              (!tail_count ||
               (finite_error(tail_error) && tail_error.nrmse <= 1.0e-5 &&
                finite_error(gate_error) && gate_error.nrmse <= 1.0e-5)),
          "selected batch MLA prompt state stays in its Lane-B envelope");
    std::fprintf(stderr,
        "PASS GLM5 complete batch state comparison role=%s tokens=%u\n",
        role, n_tokens);
    return true;
}

bool read_argmax(ds4_glm5_next_exec_ctx &exec,
                 ds4_glm5_next_workspace *workspace,
                 ds4_gpu_tensor *current,
                 ds4_gpu_tensor *logits,
                 ds4_gpu_tensor *gpu_argmax,
                 bool validate_full_logits,
                 std::vector<float> &host_logits,
                 uint32_t &argmax,
                 uint64_t &logits_hash) {
    CHECK(ds4_glm5_next_output_logits(&exec, workspace, current, logits),
          "execute text vocabulary projection");
    if (!validate_full_logits) {
        int32_t gpu_token = -1;
        CHECK(gpu_argmax &&
                  ds4_gpu_argmax_tensor(gpu_argmax, logits, 154880u) &&
                  ds4_gpu_synchronize() &&
                  ds4_gpu_tensor_read(gpu_argmax, 0u, &gpu_token,
                                      sizeof(gpu_token)) &&
                  gpu_token >= 0 && gpu_token < 154880,
              "select text vocabulary argmax on GPU");
        argmax = (uint32_t)gpu_token;
        logits_hash = fnv64(&gpu_token, sizeof(gpu_token));
        return true;
    }
    CHECK(ds4_gpu_synchronize(), "synchronize text vocabulary projection");
    CHECK(ds4_gpu_tensor_read(logits, 0u, host_logits.data(),
                              host_logits.size() * sizeof(float)),
          "read text vocabulary logits");
    logits_hash = fnv64(host_logits.data(),
                        host_logits.size() * sizeof(float));
    argmax = 0u;
    CHECK(std::isfinite(host_logits[0]), "text logits start finite");
    for (uint32_t token = 1u; token < host_logits.size(); ++token) {
        CHECK(std::isfinite(host_logits[token]), "all text logits are finite");
        if (host_logits[token] > host_logits[argmax]) argmax = token;
    }
    return true;
}

bool run() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    const char *role = std::getenv("DS4_GLM5_TP_ROLE");
    const char *host = std::getenv("DS4_GLM5_TP_HOST");
    const char *device = std::getenv("DS4_GLM5_TP_RDMA_DEVICE");
    const char *port_text = std::getenv("DS4_GLM5_TP_PORT");
    CHECK(model && model[0] && role && host && host[0] && device && device[0] &&
          port_text && port_text[0], "required model and TP environment");
    const bool leader = std::strcmp(role, "leader") == 0;
    CHECK(leader || std::strcmp(role, "worker") == 0,
          "TP role is exactly leader or worker");
    char *end = nullptr;
    const long port_long = std::strtol(port_text, &end, 10);
    CHECK(end && *end == '\0' && port_long >= 1024 && port_long <= 65535,
          "TP port is valid");

    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM5 GGUF");
    ds4_glm5_next_model_offsets offsets = {};
    CHECK(glm5_next_bind_real_offsets(gguf, offsets),
          "bind complete real GLM5 offsets");
    const char *full_trunk_env = std::getenv("DS4_GLM5_FULL_TRUNK");
    const bool full_trunk = full_trunk_env &&
                            std::strcmp(full_trunk_env, "1") == 0;
    const char *kda_batch_env =
        std::getenv("DS4_GLM5_KDA_ROUTED_BATCH_TEST");
    const bool kda_batch_test = kda_batch_env &&
        std::strcmp(kda_batch_env, "1") == 0;
    CHECK(!kda_batch_env || kda_batch_test ||
              std::strcmp(kda_batch_env, "0") == 0,
          "KDA routed batch test selector is exactly 0 or 1");
    const char *kda_batch_rows_env =
        std::getenv("DS4_GLM5_KDA_ROUTED_BATCH_ROWS");
    char *kda_batch_rows_end = nullptr;
    const unsigned long kda_batch_rows_long = kda_batch_rows_env ?
        std::strtoul(kda_batch_rows_env, &kda_batch_rows_end, 10) : 3ul;
    CHECK((!kda_batch_rows_env ||
              (kda_batch_rows_end && *kda_batch_rows_end == '\0')) &&
              (kda_batch_rows_long == 1ul ||
               kda_batch_rows_long == 3ul || kda_batch_rows_long == 33ul),
          "KDA routed batch rows are the bounded 1, 3, or 33 fixture");
    const uint32_t kda_batch_rows = (uint32_t)kda_batch_rows_long;
    const char *kda_profile_repeats_env =
        std::getenv("DS4_GLM5_KDA_ROUTED_PROFILE_REPEATS");
    char *kda_profile_repeats_end = nullptr;
    const unsigned long kda_profile_repeats_long =
        kda_profile_repeats_env ?
        std::strtoul(kda_profile_repeats_env,
                     &kda_profile_repeats_end, 10) : 0ul;
    CHECK((!kda_profile_repeats_env ||
              (kda_profile_repeats_end &&
               *kda_profile_repeats_end == '\0')) &&
              kda_profile_repeats_long <= 16ul &&
              (kda_profile_repeats_long == 0ul ||
               (kda_batch_test && kda_batch_rows == 33u)),
          "KDA routed profile repeats are bounded and use 33 rows");
    const uint32_t kda_profile_repeats =
        (uint32_t)kda_profile_repeats_long;
    const char *kda_continuation_rows_env =
        std::getenv("DS4_GLM5_KDA_ROUTED_CONTINUATION_ROWS");
    char *kda_continuation_rows_end = nullptr;
    const unsigned long kda_continuation_rows_long =
        kda_continuation_rows_env ?
        std::strtoul(kda_continuation_rows_env,
                     &kda_continuation_rows_end, 10) : 1ul;
    CHECK((!kda_continuation_rows_env ||
              (kda_continuation_rows_end &&
               *kda_continuation_rows_end == '\0')) &&
              (kda_continuation_rows_long == 1ul ||
               kda_continuation_rows_long == 16ul) &&
              (kda_continuation_rows_long == 1ul ||
               (kda_batch_test && kda_batch_rows == 33u)),
          "KDA routed continuation rows are the bounded 1 or 16 fixture");
    const uint32_t kda_continuation_rows =
        (uint32_t)kda_continuation_rows_long;
    const char *mla_batch_env =
        std::getenv("DS4_GLM5_MLA_ROUTED_BATCH_TEST");
    const bool mla_batch_test = mla_batch_env &&
        std::strcmp(mla_batch_env, "1") == 0;
    CHECK(!mla_batch_env || mla_batch_test ||
              std::strcmp(mla_batch_env, "0") == 0,
          "MLA routed batch test selector is exactly 0 or 1");
    const char *mla_batch_rows_env =
        std::getenv("DS4_GLM5_MLA_ROUTED_BATCH_ROWS");
    char *mla_batch_rows_end = nullptr;
    const unsigned long mla_batch_rows_long = mla_batch_rows_env ?
        std::strtoul(mla_batch_rows_env, &mla_batch_rows_end, 10) : 3ul;
    CHECK((!mla_batch_rows_env ||
              (mla_batch_rows_end && *mla_batch_rows_end == '\0')) &&
              (mla_batch_rows_long == 3ul || mla_batch_rows_long == 5ul ||
               mla_batch_rows_long == 33ul),
          "MLA routed batch rows are the bounded 3, 5, or 33 fixture");
    const uint32_t mla_batch_rows = (uint32_t)mla_batch_rows_long;
    const char *mla_prefix_rows_env =
        std::getenv("DS4_GLM5_MLA_ROUTED_PREFIX_ROWS");
    char *mla_prefix_rows_end = nullptr;
    const unsigned long mla_prefix_rows_long = mla_prefix_rows_env ?
        std::strtoul(mla_prefix_rows_env, &mla_prefix_rows_end, 10) : 0ul;
    CHECK((!mla_prefix_rows_env ||
              (mla_prefix_rows_end && *mla_prefix_rows_end == '\0')) &&
              mla_prefix_rows_long <= 3ul,
          "MLA routed scalar prefix is bounded to 0..3 rows");
    const uint32_t mla_prefix_rows = (uint32_t)mla_prefix_rows_long;
    const char *mla_continuation_rows_env =
        std::getenv("DS4_GLM5_MLA_ROUTED_CONTINUATION_ROWS");
    char *mla_continuation_rows_end = nullptr;
    const unsigned long mla_continuation_rows_long =
        mla_continuation_rows_env ?
        std::strtoul(mla_continuation_rows_env,
                     &mla_continuation_rows_end, 10) : 1ul;
    CHECK((!mla_continuation_rows_env ||
              (mla_continuation_rows_end &&
               *mla_continuation_rows_end == '\0')) &&
              (mla_continuation_rows_long == 1ul ||
               mla_continuation_rows_long == 16ul) &&
              (mla_continuation_rows_long == 1ul ||
               (mla_batch_test && mla_batch_rows == 33u)),
          "MLA routed continuation rows are the bounded 1 or 16 fixture");
    const uint32_t mla_continuation_rows =
        (uint32_t)mla_continuation_rows_long;
    const char *full_tokens_env = std::getenv("DS4_GLM5_FULL_TOKENS");
    const uint32_t full_tokens = !full_tokens_env ? 1u :
        std::strcmp(full_tokens_env, "1") == 0 ? 1u :
        std::strcmp(full_tokens_env, "2") == 0 ? 2u : 0u;
    CHECK(full_tokens != 0u &&
              (full_trunk || full_tokens == 1u),
          "full-token count is bounded and requires full-trunk mode");
    const char *text_prompt = std::getenv("DS4_GLM5_TEXT_PROMPT");
    const bool text_mode = text_prompt != nullptr;
    const char *batch_prefill_env =
        std::getenv("DS4_GLM5_BATCH_PREFILL_TEST");
    const bool batch_prefill = batch_prefill_env &&
        std::strcmp(batch_prefill_env, "1") == 0;
    CHECK(!batch_prefill_env || batch_prefill ||
              std::strcmp(batch_prefill_env, "0") == 0,
          "batch prefill test selector is exactly 0 or 1");
    const char *batch_prefill_compare_env =
        std::getenv("DS4_GLM5_BATCH_PREFILL_COMPARE");
    const bool batch_prefill_compare = batch_prefill_compare_env &&
        std::strcmp(batch_prefill_compare_env, "1") == 0;
    CHECK(!batch_prefill_compare_env || batch_prefill_compare ||
              std::strcmp(batch_prefill_compare_env, "0") == 0,
          "batch prefill comparison selector is exactly 0 or 1");
    const char *perf_mode_env = std::getenv("DS4_GLM5_PERF_MODE");
    const bool perf_mode = perf_mode_env &&
                           std::strcmp(perf_mode_env, "1") == 0;
    CHECK(!perf_mode_env || perf_mode || std::strcmp(perf_mode_env, "0") == 0,
          "performance mode is exactly 0 or 1");
    const char *layer_timing_env = std::getenv("DS4_GLM5_LAYER_TIMING");
    const bool layer_timing = layer_timing_env &&
                              std::strcmp(layer_timing_env, "1") == 0;
    CHECK(!layer_timing_env || layer_timing ||
              std::strcmp(layer_timing_env, "0") == 0,
          "layer timing is exactly 0 or 1");
    const char *text_generate_env = std::getenv("DS4_GLM5_TEXT_GENERATE");
    char *text_generate_end = nullptr;
    const unsigned long text_generate_long = text_generate_env ?
        std::strtoul(text_generate_env, &text_generate_end, 10) : 4ul;
    const bool text_generate_valid = !text_generate_env ||
        (text_generate_end && *text_generate_end == '\0');
    CHECK(!text_mode ||
              (full_trunk && text_prompt[0] && text_generate_long >= 1ul &&
               text_generate_long <= 128ul && text_generate_valid),
          "real-text mode requires full trunk and 1..128 generated tokens");
    const uint32_t text_generate = (uint32_t)text_generate_long;
    const char *teacher_ids_env =
        std::getenv("DS4_GLM5_TEXT_TEACHER_IDS");
    std::vector<uint32_t> teacher_ids;
    if (teacher_ids_env && teacher_ids_env[0]) {
        const char *cursor = teacher_ids_env;
        while (*cursor) {
            char *teacher_end = nullptr;
            const unsigned long value =
                std::strtoul(cursor, &teacher_end, 10);
            CHECK(teacher_end != cursor && value < 154880ul &&
                      (*teacher_end == ',' || *teacher_end == '\0'),
                  "teacher token list is comma-separated vocabulary IDs");
            teacher_ids.push_back((uint32_t)value);
            cursor = *teacher_end == ',' ? teacher_end + 1 : teacher_end;
            CHECK(*cursor || *teacher_end == '\0',
                  "teacher token list has no empty trailing item");
        }
        CHECK(text_mode && teacher_ids.size() == text_generate,
              "teacher token count must equal generated-token bound");
    }
    CHECK(!perf_mode || (text_mode && teacher_ids.empty()),
          "performance mode requires greedy text generation");
    CHECK(!batch_prefill || text_mode,
          "batch prefill test requires real-text mode");
    CHECK(!batch_prefill_compare || batch_prefill,
          "batch prefill comparison requires the batch path");
    CHECK(!kda_batch_test || (full_trunk && !text_mode),
          "KDA routed batch test requires full trunk and no text mode");
    CHECK(!mla_batch_test || (full_trunk && !text_mode),
          "MLA routed batch test requires full trunk and no text mode");
    CHECK(!(kda_batch_test && mla_batch_test),
          "select only one isolated routed batch gate");
    CodecGuard codec;
    TokensGuard prompt_tokens;
    if (text_mode) {
        CHECK(ds4_glm5_next_text_codec_open(&codec.value, model) == 0 &&
                  codec.value,
              "open exact same-GGUF GLM5 text codec");
        ds4_glm5_next_text_codec_encode_chat(
            codec.value, nullptr, text_prompt, DS4_THINK_MAX,
            &prompt_tokens.value);
        CHECK(prompt_tokens.value.len > 0 && prompt_tokens.value.len <= 128,
              "real chat prompt token count is bounded to tested 1..128");
    }
    const uint32_t context_capacity = text_mode ?
        (uint32_t)prompt_tokens.value.len + text_generate :
        kda_batch_test ? kda_batch_rows + kda_continuation_rows :
        mla_batch_test ? mla_prefix_rows + mla_batch_rows +
                         mla_continuation_rows : 2u;
    Glm5NextKShardPlan kshard;
    CHECK(!full_trunk || glm5_next_build_kshard_plan(gguf, offsets, kshard),
          "build exact full-trunk compact residency plan");
    ds4_gpu_config config = {};
    config.n_gpus = 1u;
    config.device_indices[0] = 0u;
    CHECK(ds4_gpu_init_multi(&config) &&
          ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map) &&
          (full_trunk || ds4_gpu_set_model_map(gguf.map, gguf.size)),
          "initialize gfx1151 without broad full-trunk mmap registration");

    StateGuard state;
    WorkspaceGuard workspace;
    TensorGuard current, output, logits, gpu_argmax;
    CHECK(ds4_glm5_next_state_init(&state.value, &offsets,
                                   context_capacity, nullptr) &&
          (workspace.value = ds4_glm5_next_workspace_create()) != nullptr &&
          (current.value = ds4_gpu_tensor_alloc(
              (uint64_t)kHcWidth * sizeof(float))) != nullptr &&
          (output.value = ds4_gpu_tensor_alloc(
              (uint64_t)kHcWidth * sizeof(float))) != nullptr &&
          (!full_trunk ||
           (logits.value = ds4_gpu_tensor_alloc(
               UINT64_C(154880) * sizeof(float))) != nullptr) &&
          (!perf_mode ||
           (gpu_argmax.value = ds4_gpu_tensor_alloc(sizeof(int32_t))) != nullptr),
          "allocate resident state and production workspace");

    ds4_glm5_next_exec_ctx exec = {};
    exec.model_map = gguf.map;
    exec.model_size = gguf.size;
    exec.model = &offsets;
    exec.trace_prefix = std::getenv("DS4_GLM5_NEXT_TRACE_PREFIX");
    if (exec.trace_prefix) {
        const char *layer_text = std::getenv("DS4_GLM5_NEXT_TRACE_LAYER");
        const char *token_text = std::getenv("DS4_GLM5_NEXT_TRACE_TOKEN");
        char *layer_end = nullptr, *token_end = nullptr;
        const unsigned long layer = layer_text ?
            std::strtoul(layer_text, &layer_end, 10) : 3u;
        const bool trace_all = token_text && std::strcmp(token_text, "all") == 0;
        const unsigned long token = !token_text || trace_all ? 0u :
            std::strtoul(token_text, &token_end, 10);
        CHECK((!layer_text || (layer_end && *layer_end == '\0')) &&
              (!token_text || trace_all ||
               (token_end && *token_end == '\0')) &&
              layer < DS4_GLM5_NEXT_TRUNK_COUNT && token <= UINT32_MAX,
              "valid staged trace selector");
        exec.trace_layer = (uint32_t)layer;
        exec.trace_token = trace_all ? UINT32_MAX : (uint32_t)token;
    }
    uint64_t sequence = 0u;
    TpGuard tp;
    CHECK(create_tp(gguf, leader, host, device, (int)port_long,
                    context_capacity,
                    tp, exec, sequence),
          "create persistent GLM5 layer3 RoCE transport");
    const bool kda_tp_enabled =
        (ds4_tp_runtime_features(tp.tp) &
         DS4_TP_FEATURE_GLM5_KDA_TP) != 0u;
    uint64_t active_gate_mask[DS4_GLM5_NEXT_TP_GATE_MASK_WORDS] = {};
    uint32_t tp_gates_per_token = 0u;
    CHECK(ds4_glm5_next_build_tp_gate_mask(
              active_gate_mask, &tp_gates_per_token,
              ds4_tp_runtime_features(tp.tp)) &&
              tp_gates_per_token == (kda_tp_enabled ? 87u : 53u),
          "runtime feature set derives the exact active gate count");
    char rank_error[256] = {};
    CHECK(ds4_tp_hash_check(
              tp.tp, UINT64_C(0x474c4d3552414e4b),
              UINT64_C(0x52414e4b00000000) ^ exec.tp_rank,
              rank_error, sizeof(rank_error)) == -1,
          "TP roles must prove distinct rank identities");
    char mode_error[256] = {};
    const uint64_t prompt_hash = text_mode ?
        fnv64(prompt_tokens.value.v,
              (uint64_t)prompt_tokens.value.len * sizeof(int)) : 0u;
    const uint64_t teacher_hash = teacher_ids.empty() ? 0u :
        fnv64(teacher_ids.data(),
              teacher_ids.size() * sizeof(teacher_ids[0]));
    /* Hash complete, independently sized fields.  A packed XOR encoding can
     * silently collide when a row count overlaps a neighbouring mode bit,
     * defeating the fail-closed two-rank configuration check. */
    const uint64_t mode_fields[] = {
        UINT64_C(0x474c4d354d4f4400),
        (uint64_t)full_trunk,
        (uint64_t)full_tokens,
        (uint64_t)text_mode,
        (uint64_t)text_generate,
        (uint64_t)perf_mode,
        (uint64_t)layer_timing,
        (uint64_t)kda_batch_test,
        (uint64_t)kda_batch_rows,
        (uint64_t)kda_continuation_rows,
        (uint64_t)mla_batch_test,
        (uint64_t)mla_batch_rows,
        (uint64_t)mla_prefix_rows,
        (uint64_t)mla_continuation_rows,
        (uint64_t)batch_prefill,
        (uint64_t)batch_prefill_compare,
        prompt_hash,
        teacher_hash,
    };
    const uint64_t mode_hash = fnv64(mode_fields, sizeof(mode_fields));
    if (text_mode) {
        std::fprintf(stderr,
            "GLM5 text prompt role=%s tokens=%d token_fnv=%016llx ids=",
            role, prompt_tokens.value.len,
            (unsigned long long)prompt_hash);
        for (int i = 0; i < prompt_tokens.value.len; ++i) {
            std::fprintf(stderr, "%s%d", i ? "," : "",
                         prompt_tokens.value.v[i]);
        }
        std::fprintf(stderr, "\n");
    }
    CHECK(ds4_tp_hash_check(tp.tp, UINT64_C(0x474c4d354d4f4445),
                            mode_hash, mode_error, sizeof(mode_error)) == 1,
          "TP roles must negotiate the same full-trunk/text mode");
    if (full_trunk) {
        constexpr uint64_t install_headroom = UINT64_C(2) << 30u;
        size_t free_before_install = 0u, total_before_install = 0u;
        CHECK(hipMemGetInfo(&free_before_install, &total_before_install) ==
                  hipSuccess,
              "query bounded residency budget before install");
        const uint64_t packed_required = kshard.packed_total_bytes;
        const uint64_t residency_required =
            kshard.dense_total_bytes + packed_required;
        std::fprintf(stderr,
            "GLM5 full-trunk plan rank=%u dense_spans=%zu "
            "dense_bytes=%llu packed_bytes=%llu required_bytes=%llu "
            "headroom_bytes=%llu hip_free=%llu hip_total=%llu\n",
            exec.tp_rank, kshard.dense_offsets.size(),
            (unsigned long long)kshard.dense_total_bytes,
            (unsigned long long)packed_required,
            (unsigned long long)residency_required,
            (unsigned long long)install_headroom,
            (unsigned long long)free_before_install,
            (unsigned long long)total_before_install);
        CHECK(residency_required <= free_before_install &&
                  install_headroom <= free_before_install - residency_required,
              "full-trunk residency exceeds HIP-free GTT; reduce BIOS UMA "
              "carveout and retain at least 2 GiB install headroom");
        CHECK(ds4_gpu_q4k_kshard_install(
                  gguf.map, gguf.size, gguf.fd, exec.tp_rank,
                  kshard.dense_offsets.data(), kshard.dense_sizes.data(),
                  (uint32_t)kshard.dense_offsets.size(),
                  kshard.dense_max_tensor_bytes, kshard.layers.data(),
                  (uint32_t)kshard.layers.size()),
              "install bounded all-layer Q4 K-shard before model access");
        ds4_gpu_q4k_kshard_windows windows = {};
        CHECK(ds4_gpu_q4k_kshard_windows_get(&windows) &&
                  windows.rank == exec.tp_rank && windows.n_layers == 42u &&
                  windows.n_expert == 288u &&
                  windows.expert_in_dim == 4096u &&
                  windows.expert_mid_dim == 1024u &&
                  windows.out_dim == 4096u &&
                  windows.row_base == exec.tp_rank * 1024u &&
                  windows.row_count == 1024u &&
                  windows.down_column_byte_base ==
                      (uint64_t)exec.tp_rank * 4u * 144u &&
                  windows.down_column_byte_count == 4u * 144u,
              "installed Q4 shard geometry matches the negotiated rank");
        const uint64_t packed_from_windows =
            (uint64_t)windows.n_layers * windows.n_expert *
            (2u * windows.packed_gate_expert_bytes +
             windows.packed_down_expert_bytes);
        CHECK(packed_from_windows == packed_required &&
                  ds4_gpu_q4k_packed_slice_bytes() == packed_from_windows,
              "installed Q4 bytes derive exactly from published windows");
        size_t free_after_install = 0u, total_after_install = 0u;
        CHECK(hipMemGetInfo(&free_after_install, &total_after_install) ==
                  hipSuccess && total_after_install == total_before_install &&
                  free_after_install >= install_headroom,
              "full-trunk install must retain measured 2 GiB HIP headroom");
        std::fprintf(stderr,
            "GLM5 full-trunk residency rank=%u dense_spans=%zu "
            "dense_bytes=%llu packed_q4_bytes=%llu\n",
            exec.tp_rank, kshard.dense_offsets.size(),
            (unsigned long long)kshard.dense_total_bytes,
            (unsigned long long)ds4_gpu_q4k_packed_slice_bytes());
        std::fprintf(stderr,
            "GLM5 full-trunk post-install rank=%u hip_free=%llu "
            "hip_total=%llu installed_delta=%llu\n",
            exec.tp_rank, (unsigned long long)free_after_install,
            (unsigned long long)total_after_install,
            (unsigned long long)(free_before_install - free_after_install));
    }

    if (text_mode || kda_batch_test || mla_batch_test) {
        char ready_error[256] = {};
        CHECK(ds4_gpu_synchronize() &&
                  ds4_tp_hash_check(
                      tp.tp, UINT64_C(0x474c4d3552454144),
                      UINT64_C(0x46554c4c5452554e), ready_error,
                      sizeof(ready_error)) == 1,
              ready_error[0] ? ready_error :
                  "both ranks ready after full-trunk residency");
    }

    if (mla_batch_test) {
        using Clock = std::chrono::steady_clock;
        const uint32_t rows = mla_batch_rows;
        const uint32_t prefix_rows = mla_prefix_rows;
        const uint32_t committed_rows = prefix_rows + rows;
        const std::vector<uint32_t> ids = {
            42u, 154822u, 154824u, 154826u, 25062u, 287u, 29905u, 371u,
            25u, 7487u, 154827u, 675u, 279u, 11478u, 7735u, 369u,
            6623u, 323u, 279u, 3150u, 315u, 41907u, 323u, 4968u,
            18110u, 558u, 13u, 21754u, 304u, 825u, 11646u, 13u,
            154828u, 154841u, 17u, 287u,
        };
        CHECK(committed_rows <= ids.size(),
              "select exact MLA routed token fixture");
        constexpr uint32_t continuation_ids[16] = {
            17u, 287u, 315u, 279u, 371u, 13u, 825u, 304u,
            6623u, 323u, 25u, 7487u, 558u, 369u, 11478u, 7735u,
        };
        StateGuard sequential_state, batch_state, topk_reject_state;
        StateGuard attention_sequential_state, attention_batch_state;
        WorkspaceGuard batch_workspace;
        TensorGuard batch_ids, batch_input, batch_output, sequential_output;
        TensorGuard attention_batch_output, attention_sequential_output;
        TensorGuard continuation_input, sequential_continuation;
        TensorGuard batch_continuation;
        TensorGuard nope_reject_lora, nope_reject_q, nope_reject_qk;
        CHECK(ds4_glm5_next_state_init(
                  &sequential_state.value, &offsets,
                  committed_rows + mla_continuation_rows, nullptr) &&
              ds4_glm5_next_state_init(
                  &batch_state.value, &offsets,
                  committed_rows + mla_continuation_rows, nullptr) &&
              ds4_glm5_next_state_init(
                  &attention_sequential_state.value, &offsets,
                  committed_rows, nullptr) &&
              ds4_glm5_next_state_init(
                  &attention_batch_state.value, &offsets,
                  committed_rows, nullptr) &&
              ds4_glm5_next_state_init(
                  &topk_reject_state.value, &offsets,
                  DS4_GLM5_NEXT_INDEX_TOP_K + 1u, nullptr) &&
              (batch_workspace.value =
                   ds4_glm5_next_workspace_create_capacity(rows)) != nullptr &&
              (batch_ids.value = ds4_gpu_tensor_alloc(
                   (uint64_t)rows * sizeof(uint32_t))) != nullptr &&
              (batch_input.value = ds4_gpu_tensor_alloc(
                   (uint64_t)rows * kHcWidth * sizeof(float))) != nullptr &&
              (batch_output.value = ds4_gpu_tensor_alloc(
                   (uint64_t)rows * kHcWidth * sizeof(float))) != nullptr &&
              (sequential_output.value = ds4_gpu_tensor_alloc(
                   (uint64_t)rows * kHcWidth * sizeof(float))) != nullptr &&
              (attention_batch_output.value = ds4_gpu_tensor_alloc(
                   (uint64_t)rows * kHcWidth * sizeof(float))) != nullptr &&
              (attention_sequential_output.value = ds4_gpu_tensor_alloc(
                   (uint64_t)rows * kHcWidth * sizeof(float))) != nullptr &&
              (continuation_input.value = ds4_gpu_tensor_alloc(
                   (uint64_t)kHcWidth * sizeof(float))) != nullptr &&
              (sequential_continuation.value = ds4_gpu_tensor_alloc(
                   (uint64_t)kHcWidth * sizeof(float))) != nullptr &&
              (batch_continuation.value = ds4_gpu_tensor_alloc(
                   (uint64_t)kHcWidth * sizeof(float))) != nullptr &&
              (nope_reject_lora.value = ds4_gpu_tensor_alloc(
                   (uint64_t)64u * 512u * sizeof(float))) != nullptr &&
              (nope_reject_q.value = ds4_gpu_tensor_alloc(
                   (uint64_t)64u * 256u * sizeof(float))) != nullptr &&
              (nope_reject_qk.value = ds4_gpu_tensor_alloc(
                   (uint64_t)64u * 512u * sizeof(float))) != nullptr &&
              ds4_gpu_tensor_write(batch_ids.value, 0u,
                                    ids.data() + prefix_rows,
                                    (uint64_t)rows * sizeof(uint32_t)) &&
              ds4_glm5_next_embed_tokens(
                  &exec, batch_ids.value, rows, batch_input.value),
              "allocate exact MLA+routed row comparison");

        /* Public batch and NoPE entry points must reject malformed geometry
         * before mutating an otherwise reusable state. */
        CHECK(!ds4_glm5_next_layer_forward_batch(
                  &exec, 3u, &batch_state.value, workspace.value,
                  batch_input.value, batch_output.value, rows) &&
              batch_state.value.valid,
              "reject MLA batch workspace capacity mismatch");
        CHECK(!ds4_glm5_next_layer_forward_batch(
                  &exec, 3u, &batch_state.value, batch_workspace.value,
                  continuation_input.value, batch_continuation.value, rows) &&
              batch_state.value.valid,
              "reject MLA batch HC byte-size mismatch");
        ds4_gpu_tensor *saved_kda_mla =
            batch_state.value.mla[0].compact_kv;
        batch_state.value.mla[0].compact_kv =
            batch_state.value.mla[3].compact_kv;
        const int kda_with_mla = ds4_glm5_next_layer_forward_batch(
            &exec, 0u, &batch_state.value, batch_workspace.value,
            batch_input.value, batch_output.value, rows);
        batch_state.value.mla[0].compact_kv = saved_kda_mla;
        CHECK(!kda_with_mla && batch_state.value.valid,
              "reject KDA layer carrying contradictory MLA state");
        topk_reject_state.value.mla[3].token_count =
            DS4_GLM5_NEXT_INDEX_TOP_K;
        topk_reject_state.value.mla[3].complete_pools =
            DS4_GLM5_NEXT_INDEX_TOP_K / 4u;
        topk_reject_state.value.mla[3].tail_count = 0u;
        CHECK(!ds4_glm5_next_layer_forward_batch(
                  &exec, 3u, &topk_reject_state.value,
                  batch_workspace.value, batch_input.value,
                  batch_output.value, rows) &&
              topk_reject_state.value.valid,
              "reject dense-selection MLA batch beyond top-k boundary");
        CHECK(!ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
                  nope_reject_lora.value, nope_reject_q.value,
                  nope_reject_qk.value,
                  batch_state.value.mla[3].compact_kv,
                  batch_state.value.mla[3].compact_kv,
                  1u, 0u, 1u, committed_rows + mla_continuation_rows,
                  false, 64u, 512u, 256u, 0u, 0u,
                  10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f),
              "reject NoPE causal batch with non-NULL rope cache");

        for (uint32_t row = 0u; row < prefix_rows; ++row) {
            CHECK(ds4_glm5_next_embed_token(
                      &exec, ids[row], continuation_input.value) &&
                  ds4_glm5_next_mla_attention_forward_test(
                      &exec, 3u, &attention_sequential_state.value,
                      workspace.value, continuation_input.value,
                      sequential_continuation.value, 1u) &&
                  ds4_glm5_next_mla_attention_forward_test(
                      &exec, 3u, &attention_batch_state.value,
                      workspace.value, continuation_input.value,
                      batch_continuation.value, 1u) &&
                  ds4_glm5_next_layer_forward(
                      &exec, 3u, &sequential_state.value, workspace.value,
                      continuation_input.value,
                      sequential_continuation.value) &&
                  ds4_glm5_next_layer_forward(
                      &exec, 3u, &batch_state.value, workspace.value,
                      continuation_input.value, batch_continuation.value),
                  "seed identical scalar MLA tail before batch");
        }

        for (uint32_t row = 0u; row < rows; ++row) {
            ds4_gpu_tensor *row_input = ds4_gpu_tensor_view(
                batch_input.value,
                (uint64_t)row * kHcWidth * sizeof(float),
                (uint64_t)kHcWidth * sizeof(float));
            ds4_gpu_tensor *row_output = ds4_gpu_tensor_view(
                attention_sequential_output.value,
                (uint64_t)row * kHcWidth * sizeof(float),
                (uint64_t)kHcWidth * sizeof(float));
            const bool row_ok = row_input && row_output &&
                ds4_glm5_next_mla_attention_forward_test(
                    &exec, 3u, &attention_sequential_state.value,
                    workspace.value, row_input, row_output, 1u);
            ds4_gpu_tensor_free(row_output);
            ds4_gpu_tensor_free(row_input);
            CHECK(row_ok, "execute sequential MLA attention oracle row");
        }
        CHECK(ds4_glm5_next_mla_attention_forward_test(
                  &exec, 3u, &attention_batch_state.value,
                  batch_workspace.value, batch_input.value,
                  attention_batch_output.value, rows),
              "execute production MLA attention batch oracle");
        std::vector<float> attention_sequential(
            (uint64_t)rows * kHcWidth);
        std::vector<float> attention_batch((uint64_t)rows * kHcWidth);
        CHECK(ds4_gpu_tensor_read(
                  attention_sequential_output.value, 0u,
                  attention_sequential.data(),
                  attention_sequential.size() * sizeof(float)) &&
              ds4_gpu_tensor_read(
                  attention_batch_output.value, 0u,
                  attention_batch.data(),
                  attention_batch.size() * sizeof(float)),
              "read decomposed MLA attention outputs");
        const VectorError attention_error =
            vector_error(attention_sequential, attention_batch);
        std::fprintf(stderr,
            "GLM5 MLA attention batch measurement role=%s prefix=%u rows=%u "
            "nrmse=%.9g cosine=%.12g max_abs=%.9g\n",
            role, prefix_rows, rows, attention_error.nrmse,
            attention_error.cosine,
            attention_error.max_abs);
        CHECK(attention_error.nrmse <= 1.0e-6 &&
                  attention_error.cosine >= 0.999999999 &&
                  attention_error.max_abs <= 2.0e-6,
              "MLA attention batch matches sequential same-GGUF execution");

        const auto sequential_begin = Clock::now();
        for (uint32_t row = 0u; row < rows; ++row) {
            ds4_gpu_tensor *row_input = ds4_gpu_tensor_view(
                batch_input.value,
                (uint64_t)row * kHcWidth * sizeof(float),
                (uint64_t)kHcWidth * sizeof(float));
            ds4_gpu_tensor *row_output = ds4_gpu_tensor_view(
                sequential_output.value,
                (uint64_t)row * kHcWidth * sizeof(float),
                (uint64_t)kHcWidth * sizeof(float));
            const bool row_ok = row_input && row_output &&
                ds4_glm5_next_layer_forward(
                    &exec, 3u, &sequential_state.value,
                    workspace.value, row_input, row_output);
            ds4_gpu_tensor_free(row_output);
            ds4_gpu_tensor_free(row_input);
            CHECK(row_ok, "execute sequential MLA+routed comparison row");
        }
        const auto sequential_end = Clock::now();

        const auto batch_begin = Clock::now();
        CHECK(ds4_glm5_next_layer_forward_batch(
                  &exec, 3u, &batch_state.value, batch_workspace.value,
                  batch_input.value, batch_output.value, rows),
              "execute production MLA+routed batch");
        const auto batch_end = Clock::now();
        const double sequential_ms =
            std::chrono::duration<double, std::milli>(
                sequential_end - sequential_begin).count();
        const double batch_ms =
            std::chrono::duration<double, std::milli>(
                batch_end - batch_begin).count();

        std::vector<float> sequential((uint64_t)rows * kHcWidth);
        std::vector<float> batch((uint64_t)rows * kHcWidth);
        CHECK(ds4_gpu_tensor_read(
                  sequential_output.value, 0u, sequential.data(),
                  sequential.size() * sizeof(float)) &&
              ds4_gpu_tensor_read(batch_output.value, 0u, batch.data(),
                                  batch.size() * sizeof(float)),
              "read sequential and production MLA+routed rows");
        const VectorError output_error = vector_error(sequential, batch);

        const uint64_t kv_count = (uint64_t)committed_rows * 512u;
        const uint64_t pool_count =
            (uint64_t)(committed_rows / 4u) * 128u;
        const uint64_t tail_count =
            (uint64_t)(committed_rows % 4u) * 128u;
        std::vector<float> sequential_kv(kv_count), batch_kv(kv_count);
        std::vector<float> sequential_pool(pool_count), batch_pool(pool_count);
        std::vector<float> sequential_tail(tail_count), batch_tail(tail_count);
        std::vector<float> sequential_gate(tail_count), batch_gate(tail_count);
        CHECK(ds4_gpu_tensor_read(
                  sequential_state.value.mla[3].compact_kv, 0u,
                  sequential_kv.data(), kv_count * sizeof(float)) &&
              ds4_gpu_tensor_read(
                  batch_state.value.mla[3].compact_kv, 0u,
                  batch_kv.data(), kv_count * sizeof(float)) &&
              (pool_count == 0u ||
               (ds4_gpu_tensor_read(
                    sequential_state.value.mla[3].index_pool, 0u,
                    sequential_pool.data(), pool_count * sizeof(float)) &&
                ds4_gpu_tensor_read(
                    batch_state.value.mla[3].index_pool, 0u,
                    batch_pool.data(), pool_count * sizeof(float)))) &&
              (tail_count == 0u ||
               (ds4_gpu_tensor_read(
                    sequential_state.value.mla[3].index_tail, 0u,
                    sequential_tail.data(), tail_count * sizeof(float)) &&
                ds4_gpu_tensor_read(
                    batch_state.value.mla[3].index_tail, 0u,
                    batch_tail.data(), tail_count * sizeof(float)) &&
                ds4_gpu_tensor_read(
                    sequential_state.value.mla[3].pool_gate_tail, 0u,
                    sequential_gate.data(), tail_count * sizeof(float)) &&
                ds4_gpu_tensor_read(
                    batch_state.value.mla[3].pool_gate_tail, 0u,
                    batch_gate.data(), tail_count * sizeof(float)))),
              "read sequential and batch MLA resident state");
        const VectorError kv_error = vector_error(sequential_kv, batch_kv);
        const VectorError pool_error = pool_count ?
            vector_error(sequential_pool, batch_pool) : VectorError{};
        const VectorError tail_error = tail_count ?
            vector_error(sequential_tail, batch_tail) : VectorError{};
        const VectorError gate_error = tail_count ?
            vector_error(sequential_gate, batch_gate) : VectorError{};
        CHECK(sequential_state.value.mla[3].token_count == committed_rows &&
                  batch_state.value.mla[3].token_count == committed_rows &&
                  sequential_state.value.mla[3].complete_pools ==
                      committed_rows / 4u &&
                  batch_state.value.mla[3].complete_pools ==
                      committed_rows / 4u &&
                  sequential_state.value.mla[3].tail_count ==
                      committed_rows % 4u &&
                  batch_state.value.mla[3].tail_count ==
                      committed_rows % 4u,
              "MLA batch commits the exact pool/tail counters");
        std::fprintf(stderr,
            "GLM5 MLA+routed batch measurement role=%s prefix=%u rows=%u "
            "output_nrmse=%.9g output_cosine=%.12g output_max_abs=%.9g "
            "kv_nrmse=%.9g pool_nrmse=%.9g tail_nrmse=%.9g "
            "gate_nrmse=%.9g sequential_ms=%.3f batch_ms=%.3f "
            "speedup=%.3fx\n",
            role, prefix_rows, rows, output_error.nrmse, output_error.cosine,
            output_error.max_abs, kv_error.nrmse, pool_error.nrmse,
            tail_error.nrmse, gate_error.nrmse, sequential_ms, batch_ms,
            sequential_ms / batch_ms);
        /* Attention above has a scalar-equivalence envelope.  The routed-Q4
         * batch deliberately uses different reduction order than one-row
         * decode, so this layer-level bound is only a provisional Lane-B
         * containment gate.  Full-logit, teacher-forced, semantic and long
         * prompt gates decide whether that arithmetic can be promoted. */
        const double output_nrmse_bound = 1.0e-2;
        const double output_cosine_bound = 0.99995;
        const double output_max_abs_bound = 6.0e-2;
        CHECK(output_error.nrmse <= output_nrmse_bound &&
                  output_error.cosine >= output_cosine_bound &&
                  output_error.max_abs <= output_max_abs_bound &&
                  kv_error.nrmse <= 1.0e-5 &&
                  (!pool_count || pool_error.nrmse <= 1.0e-5) &&
                  (!tail_count || (tail_error.nrmse <= 1.0e-5 &&
                                   gate_error.nrmse <= 1.0e-5)),
              "MLA+routed batch matches sequential same-GGUF execution");

        std::vector<float> sequential_cont(
            (uint64_t)mla_continuation_rows * kHcWidth);
        std::vector<float> batch_cont(
            (uint64_t)mla_continuation_rows * kHcWidth);
        for (uint32_t step = 0u; step < mla_continuation_rows; ++step) {
            CHECK(ds4_glm5_next_embed_token(
                      &exec, continuation_ids[step],
                      continuation_input.value) &&
                  ds4_glm5_next_layer_forward(
                      &exec, 3u, &sequential_state.value, workspace.value,
                      continuation_input.value, sequential_continuation.value) &&
                  ds4_glm5_next_layer_forward(
                      &exec, 3u, &batch_state.value, workspace.value,
                      continuation_input.value, batch_continuation.value) &&
                  ds4_gpu_tensor_read(
                      sequential_continuation.value, 0u,
                      sequential_cont.data() + (uint64_t)step * kHcWidth,
                      (uint64_t)kHcWidth * sizeof(float)) &&
                  ds4_gpu_tensor_read(
                      batch_continuation.value, 0u,
                      batch_cont.data() + (uint64_t)step * kHcWidth,
                      (uint64_t)kHcWidth * sizeof(float)),
                  "ordinary scalar decode continues from MLA batch state");
        }
        const VectorError continuation_error =
            vector_error(sequential_cont, batch_cont);
        CHECK(continuation_error.nrmse <= 1.0e-2 &&
                  continuation_error.cosine >= 0.9999 &&
                  continuation_error.max_abs <= 1.0e-2 &&
                  sequential_state.value.mla[3].token_count ==
                      committed_rows + mla_continuation_rows &&
                  batch_state.value.mla[3].token_count ==
                      committed_rows + mla_continuation_rows,
              "MLA batch state supports ordinary scalar continuation");
        const uint64_t output_hash =
            fnv64(batch.data(), batch.size() * sizeof(float));
        const uint64_t continuation_hash = fnv64(
            batch_cont.data(), batch_cont.size() * sizeof(float));
        char hash_error[256] = {};
        CHECK(ds4_tp_hash_check(
                  tp.tp, UINT64_C(0x474c4d354d424154), output_hash,
                  hash_error, sizeof(hash_error)) == 1 &&
              ds4_tp_hash_check(
                  tp.tp, UINT64_C(0x474c4d354d434f4e), continuation_hash,
                  hash_error, sizeof(hash_error)) == 1,
              hash_error);
        std::fprintf(stderr,
            "PASS GLM5 MLA+routed batch role=%s prefix=%u rows=%u "
            "nrmse=%.9g cosine=%.12g max_abs=%.9g output=%016llx "
            "continue_nrmse=%.9g continue_cosine=%.12g "
            "continue_max_abs=%.9g continue_rows=%u continuation=%016llx "
            "sequential_ms=%.3f batch_ms=%.3f speedup=%.3fx "
            "tp_seq=%llu packed_q4_bytes=%llu rdma=1\n",
            role, prefix_rows, rows, output_error.nrmse,
            output_error.cosine,
            output_error.max_abs, (unsigned long long)output_hash,
            continuation_error.nrmse, continuation_error.cosine,
            continuation_error.max_abs, mla_continuation_rows,
            (unsigned long long)continuation_hash,
            sequential_ms, batch_ms, sequential_ms / batch_ms,
            (unsigned long long)sequence,
            (unsigned long long)ds4_gpu_q4k_packed_slice_bytes());
        return true;
    }

    if (kda_batch_test) {
        using Clock = std::chrono::steady_clock;
        const uint32_t rows = kda_batch_rows;
        std::vector<uint32_t> ids;
        if (rows == 1u) {
            ids = {42u};
        } else if (rows == 3u) {
            ids = {42u, 154822u, 154824u};
        } else {
            ids = {
                154822u, 154824u, 154826u, 25062u, 287u, 29905u, 371u,
                25u, 7487u, 154827u, 675u, 279u, 11478u, 7735u, 369u,
                6623u, 323u, 279u, 3150u, 315u, 41907u, 323u, 4968u,
                18110u, 558u, 13u, 21754u, 304u, 825u, 11646u, 13u,
                154828u, 154841u,
            };
        }
        CHECK(ids.size() == rows, "select exact KDA routed token fixture");
        constexpr uint32_t continuation_ids[16] = {
            17u, 287u, 315u, 279u, 371u, 13u, 825u, 304u,
            6623u, 323u, 25u, 7487u, 558u, 369u, 11478u, 7735u,
        };
        StateGuard sequential_state, batch_state;
        WorkspaceGuard batch_workspace;
        TensorGuard batch_ids, batch_input, batch_output, sequential_output;
        CHECK(ds4_glm5_next_state_init(
                  &sequential_state.value, &offsets, rows + 1u, nullptr) &&
              ds4_glm5_next_state_init(
                  &batch_state.value, &offsets, rows + 1u, nullptr) &&
              (batch_workspace.value =
                   ds4_glm5_next_workspace_create_capacity(rows)) != nullptr &&
              (batch_ids.value = ds4_gpu_tensor_alloc(
                   (uint64_t)rows * sizeof(uint32_t))) != nullptr &&
              (batch_input.value = ds4_gpu_tensor_alloc(
                   (uint64_t)rows * kHcWidth * sizeof(float))) != nullptr &&
              (batch_output.value = ds4_gpu_tensor_alloc(
                   (uint64_t)rows * kHcWidth * sizeof(float))) != nullptr &&
              (sequential_output.value = ds4_gpu_tensor_alloc(
                   (uint64_t)rows * kHcWidth * sizeof(float))) != nullptr &&
              ds4_gpu_tensor_write(batch_ids.value, 0u, ids.data(),
                                    (uint64_t)rows * sizeof(uint32_t)) &&
              ds4_glm5_next_embed_tokens(
                  &exec, batch_ids.value, rows, batch_input.value),
              "allocate exact KDA+routed row comparison");

        const auto sequential_begin = Clock::now();
        for (uint32_t row = 0u; row < rows; ++row) {
            ds4_gpu_tensor *row_input = ds4_gpu_tensor_view(
                batch_input.value,
                (uint64_t)row * kHcWidth * sizeof(float),
                (uint64_t)kHcWidth * sizeof(float));
            ds4_gpu_tensor *row_output = ds4_gpu_tensor_view(
                sequential_output.value,
                (uint64_t)row * kHcWidth * sizeof(float),
                (uint64_t)kHcWidth * sizeof(float));
            const bool row_ok = row_input && row_output &&
                ds4_glm5_next_layer_forward(
                    &exec, 4u, &sequential_state.value,
                    workspace.value, row_input, row_output);
            ds4_gpu_tensor_free(row_output);
            ds4_gpu_tensor_free(row_input);
            CHECK(row_ok,
                  "execute sequential KDA+routed comparison row");
        }
        const auto sequential_end = Clock::now();

        const auto batch_begin = Clock::now();
        CHECK(ds4_glm5_next_layer_forward_batch(
                  &exec, 4u, &batch_state.value, batch_workspace.value,
                  batch_input.value, batch_output.value, rows),
              "execute production KDA+routed batch");
        const auto batch_end = Clock::now();
        const double sequential_ms =
            std::chrono::duration<double, std::milli>(
                sequential_end - sequential_begin).count();
        const double batch_ms =
            std::chrono::duration<double, std::milli>(
                batch_end - batch_begin).count();
        std::vector<float> sequential((uint64_t)rows * kHcWidth);
        std::vector<float> batch((uint64_t)rows * kHcWidth);
        CHECK(ds4_gpu_tensor_read(
                  sequential_output.value, 0u, sequential.data(),
                  sequential.size() * sizeof(float)) &&
              ds4_gpu_tensor_read(batch_output.value, 0u, batch.data(),
                                  batch.size() * sizeof(float)),
              "read sequential and production KDA+routed rows");
        const VectorError batch_error = vector_error(sequential, batch);
        VectorError worst_row;
        worst_row.cosine = 1.0;
        uint32_t worst_nrmse_row = 0u, worst_cosine_row = 0u;
        uint32_t worst_max_abs_row = 0u;
        for (uint32_t row = 0u; row < rows; ++row) {
            const VectorError row_error = vector_error_data(
                sequential.data() + (uint64_t)row * kHcWidth,
                batch.data() + (uint64_t)row * kHcWidth, kHcWidth);
            if (row == 0u || row_error.nrmse > worst_row.nrmse) {
                worst_row.nrmse = row_error.nrmse;
                worst_nrmse_row = row;
            }
            if (row == 0u || row_error.cosine < worst_row.cosine) {
                worst_row.cosine = row_error.cosine;
                worst_cosine_row = row;
            }
            if (row == 0u || row_error.max_abs > worst_row.max_abs) {
                worst_row.max_abs = row_error.max_abs;
                worst_max_abs_row = row;
            }
        }
        std::fprintf(stderr,
            "GLM5 KDA+routed batch measurement role=%s rows=%u "
            "nrmse=%.9g cosine=%.12g max_abs=%.9g "
            "worst_row_nrmse=%.9g worst_nrmse_row=%u "
            "worst_row_cosine=%.12g worst_cosine_row=%u "
            "worst_row_max_abs=%.9g worst_max_abs_row=%u "
            "sequential_ms=%.3f batch_ms=%.3f speedup=%.3fx\n",
            role, rows, batch_error.nrmse, batch_error.cosine,
            batch_error.max_abs,
            worst_row.nrmse, worst_nrmse_row,
            worst_row.cosine, worst_cosine_row,
            worst_row.max_abs, worst_max_abs_row,
            sequential_ms, batch_ms,
            sequential_ms / batch_ms);
        /* These bounds are pinned from the independently recorded 33-row
         * measurement with approximately 3x headroom. The batch Q4_K WMMA
         * lane intentionally has a different reduction path; later full-logit
         * and teacher gates decide whether that arithmetic is promotable. */
        const double nrmse_bound = rows == 3u ? 4.1e-3 : 7.0e-3;
        const double cosine_bound = rows == 3u ? 0.999997 : 0.999993;
        const double max_abs_bound = rows == 3u ? 3.6e-4 : 2.5e-3;
        CHECK(batch_error.nrmse <= nrmse_bound &&
                  batch_error.cosine >= cosine_bound &&
                  batch_error.max_abs <= max_abs_bound,
              "KDA+routed batch matches sequential same-GGUF execution");
        const uint64_t batch_hash =
            fnv64(batch.data(), batch.size() * sizeof(float));
        char batch_hash_error[256] = {};
        CHECK(ds4_tp_hash_check(
                  tp.tp, UINT64_C(0x474c4d354b424154), batch_hash,
                  batch_hash_error, sizeof(batch_hash_error)) == 1,
              batch_hash_error);

        ds4_glm5_kda_digest sequential_digest = {};
        ds4_glm5_kda_digest batch_digest = {};
        CHECK(ds4_glm5_kda_layer_digest(
                  &sequential_state.value.kda.layer[4], batch_output.value,
                  1u, &sequential_digest) &&
              ds4_glm5_kda_layer_digest(
                  &batch_state.value.kda.layer[4], batch_output.value,
                  1u, &batch_digest),
              "read sequential and batch KDA recurrent digests");
        const bool recurrent_digest_equal =
            sequential_digest.q_history_fnv64 ==
                batch_digest.q_history_fnv64 &&
            sequential_digest.k_history_fnv64 ==
                batch_digest.k_history_fnv64 &&
            sequential_digest.v_history_fnv64 ==
                batch_digest.v_history_fnv64 &&
            sequential_digest.recurrent_fnv64 ==
                batch_digest.recurrent_fnv64 &&
            sequential_digest.token_count == batch_digest.token_count;
        const uint64_t batch_digest_hash =
            fnv64(&batch_digest, sizeof(batch_digest));
        char batch_digest_error[256] = {};
        uint64_t sequential_owned_fnv = 0u, batch_owned_fnv = 0u;
        CHECK(!kda_tp_enabled ||
                  (kda_state_has_exact_local_ownership(
                       sequential_state.value.kda.layer[4], exec.tp_rank,
                       &sequential_owned_fnv) &&
                   kda_state_has_exact_local_ownership(
                       batch_state.value.kda.layer[4], exec.tp_rank,
                       &batch_owned_fnv) &&
                   sequential_owned_fnv == batch_owned_fnv),
              "sharded KDA batch state owns exactly one matching head half");
        CHECK(kda_tp_enabled ||
                  ds4_tp_hash_check(
                      tp.tp, UINT64_C(0x474c4d354b444947),
                      batch_digest_hash, batch_digest_error,
                      sizeof(batch_digest_error)) == 1,
              batch_digest_error);
        std::fprintf(stderr,
            "GLM5 KDA+routed state role=%s recurrent_digest_equal=%d "
            "sequential_recurrent=%016llx batch_recurrent=%016llx "
            "batch_digest=%016llx local_owned=%016llx kda_tp=%d\n",
            role, recurrent_digest_equal ? 1 : 0,
            (unsigned long long)sequential_digest.recurrent_fnv64,
            (unsigned long long)batch_digest.recurrent_fnv64,
            (unsigned long long)batch_digest_hash,
            (unsigned long long)batch_owned_fnv,
            kda_tp_enabled ? 1 : 0);

        std::vector<float> sequential_continue(kHcWidth);
        std::vector<float> batch_continue(kHcWidth);
        std::vector<float> batch_continuations(
            (uint64_t)kda_continuation_rows * kHcWidth);
        VectorError continuation_error = {};
        continuation_error.cosine = 1.0;
        uint32_t worst_continue_nrmse_step = 0u;
        uint32_t worst_continue_cosine_step = 0u;
        uint32_t worst_continue_max_abs_step = 0u;
        for (uint32_t step = 0u; step < kda_continuation_rows; ++step) {
            CHECK(ds4_glm5_next_embed_token(
                      &exec, continuation_ids[step], current.value) &&
                  ds4_glm5_next_layer_forward(
                      &exec, 4u, &sequential_state.value, workspace.value,
                      current.value, output.value) &&
                  ds4_gpu_tensor_read(output.value, 0u,
                                       sequential_continue.data(),
                                       (uint64_t)kHcWidth * sizeof(float)) &&
                  ds4_glm5_next_layer_forward(
                      &exec, 4u, &batch_state.value, workspace.value,
                      current.value, output.value) &&
                  ds4_gpu_tensor_read(output.value, 0u,
                      batch_continue.data(),
                      (uint64_t)kHcWidth * sizeof(float)),
                  "execute continuation after KDA+routed batch");
            std::copy(batch_continue.begin(), batch_continue.end(),
                      batch_continuations.begin() +
                          (uint64_t)step * kHcWidth);
            const VectorError step_error =
                vector_error(sequential_continue, batch_continue);
            std::fprintf(stderr,
                "GLM5 KDA+routed continuation step role=%s step=%u "
                "token=%u nrmse=%.9g cosine=%.12g max_abs=%.9g\n",
                role, step, continuation_ids[step], step_error.nrmse,
                step_error.cosine, step_error.max_abs);
            if (step == 0u || step_error.nrmse > continuation_error.nrmse) {
                continuation_error.nrmse = step_error.nrmse;
                worst_continue_nrmse_step = step;
            }
            if (step == 0u || step_error.cosine < continuation_error.cosine) {
                continuation_error.cosine = step_error.cosine;
                worst_continue_cosine_step = step;
            }
            if (step == 0u || step_error.max_abs >
                                  continuation_error.max_abs) {
                continuation_error.max_abs = step_error.max_abs;
                worst_continue_max_abs_step = step;
            }
        }
        std::fprintf(stderr,
            "GLM5 KDA+routed continuation measurement role=%s "
            "rows=%u nrmse=%.9g worst_nrmse_step=%u "
            "cosine=%.12g worst_cosine_step=%u max_abs=%.9g "
            "worst_max_abs_step=%u\n",
            role, kda_continuation_rows, continuation_error.nrmse,
            worst_continue_nrmse_step, continuation_error.cosine,
            worst_continue_cosine_step, continuation_error.max_abs,
            worst_continue_max_abs_step);
        CHECK(continuation_error.nrmse <= 5.0e-4 &&
                  continuation_error.cosine >= 0.9999999 &&
                  continuation_error.max_abs <= 2.5e-5,
              "ordinary decode continues from KDA+routed batch state");
        const uint64_t continuation_hash = fnv64(
            batch_continuations.data(),
            batch_continuations.size() * sizeof(float));
        char continuation_hash_error[256] = {};
        CHECK(ds4_tp_hash_check(
                  tp.tp, UINT64_C(0x474c4d354b434f4e), continuation_hash,
                  continuation_hash_error,
                  sizeof(continuation_hash_error)) == 1,
              continuation_hash_error);
        CHECK(sequential_state.value.kda.layer[4].token_count ==
                  rows + kda_continuation_rows &&
                  batch_state.value.kda.layer[4].token_count ==
                      rows + kda_continuation_rows &&
                  sequential_state.value.valid && batch_state.value.valid,
              "both KDA+routed paths commit identical recurrent length");
        std::fprintf(stderr,
            "PASS GLM5 KDA+routed batch role=%s rows=%u layer=4 "
            "nrmse=%.9g cosine=%.12g max_abs=%.9g output=%016llx "
            "continue_nrmse=%.9g continue_cosine=%.12g "
            "continue_max_abs=%.9g continue_rows=%u continuation=%016llx "
            "sequential_ms=%.3f batch_ms=%.3f speedup=%.3fx "
            "tp_seq=%llu packed_q4_bytes=%llu rdma=1\n",
            role, rows, batch_error.nrmse, batch_error.cosine,
            batch_error.max_abs, (unsigned long long)batch_hash,
            continuation_error.nrmse, continuation_error.cosine,
            continuation_error.max_abs,
            kda_continuation_rows,
            (unsigned long long)continuation_hash,
            sequential_ms, batch_ms, sequential_ms / batch_ms,
            (unsigned long long)sequence,
            (unsigned long long)ds4_gpu_q4k_packed_slice_bytes());
        if (kda_profile_repeats > 0u) {
            std::vector<double> profile_ms;
            profile_ms.reserve(kda_profile_repeats > 1u ?
                               kda_profile_repeats - 1u : 1u);
            for (uint32_t repeat = 0u; repeat < kda_profile_repeats;
                 ++repeat) {
                StateGuard profile_state;
                CHECK(ds4_glm5_next_state_init(
                          &profile_state.value, &offsets, rows + 1u,
                          nullptr),
                      "allocate isolated KDA+routed profile state");
                const auto profile_begin = Clock::now();
                CHECK(ds4_glm5_next_layer_forward_batch(
                          &exec, 4u, &profile_state.value,
                          batch_workspace.value, batch_input.value,
                          batch_output.value, rows),
                      "execute isolated KDA+routed profile repeat");
                const double elapsed_ms =
                    std::chrono::duration<double, std::milli>(
                        Clock::now() - profile_begin).count();
                std::fprintf(stderr,
                    "GLM5 KDA+routed profile role=%s rows=%u "
                    "repeat=%u warmup=%d elapsed_ms=%.3f tp_seq=%llu\n",
                    role, rows, repeat, repeat == 0u ? 1 : 0,
                    elapsed_ms, (unsigned long long)sequence);
                if (repeat > 0u || kda_profile_repeats == 1u)
                    profile_ms.push_back(elapsed_ms);
            }
            std::sort(profile_ms.begin(), profile_ms.end());
            const size_t mid = profile_ms.size() / 2u;
            const double median_ms = profile_ms.size() % 2u ?
                profile_ms[mid] :
                0.5 * (profile_ms[mid - 1u] + profile_ms[mid]);
            std::fprintf(stderr,
                "GLM5 KDA+routed profile summary role=%s rows=%u "
                "measured=%zu median_ms=%.3f min_ms=%.3f max_ms=%.3f\n",
                role, rows, profile_ms.size(), median_ms,
                profile_ms.front(), profile_ms.back());
        }
        return true;
    }

    if (text_mode) {
        using Clock = std::chrono::steady_clock;
        WorkspaceGuard prompt_workspace;
        TensorGuard prompt_ids, prompt_current_guard, prompt_output_guard;
        StateGuard prompt_reference_state;
        TensorGuard reference_current_guard, reference_output_guard;
        TensorGuard reference_logits_guard;
        ds4_gpu_tensor *prompt_current = nullptr;
        ds4_gpu_tensor *prompt_output = nullptr;
        ds4_gpu_tensor *reference_current = nullptr;
        ds4_gpu_tensor *reference_output = nullptr;
        std::vector<float> reference_hidden;
        std::vector<float> reference_logits;
        if (batch_prefill) {
            const uint32_t prompt_rows =
                (uint32_t)prompt_tokens.value.len;
            /* NoPE prefill remains dense only while every row is inside the
             * model's 2048-row index-selection boundary. */
            CHECK(prompt_rows <= context_capacity &&
                  prompt_rows <= DS4_GLM5_NEXT_INDEX_TOP_K &&
                  (prompt_workspace.value =
                       ds4_glm5_next_workspace_create_capacity(
                           prompt_rows)) != nullptr &&
                  (prompt_ids.value = ds4_gpu_tensor_alloc(
                       (uint64_t)prompt_rows * sizeof(uint32_t))) != nullptr &&
                  (prompt_current_guard.value = ds4_gpu_tensor_alloc(
                       (uint64_t)prompt_rows * kHcWidth *
                       sizeof(float))) != nullptr &&
                  (prompt_output_guard.value = ds4_gpu_tensor_alloc(
                       (uint64_t)prompt_rows * kHcWidth *
                       sizeof(float))) != nullptr &&
                  ds4_gpu_tensor_write(
                      prompt_ids.value, 0u, prompt_tokens.value.v,
                      (uint64_t)prompt_rows * sizeof(uint32_t)),
                  "allocate exact-capacity complete prompt batch");
            prompt_current = prompt_current_guard.value;
            prompt_output = prompt_output_guard.value;
        }
        if (batch_prefill_compare) {
            CHECK(ds4_glm5_next_state_init(
                      &prompt_reference_state.value, &offsets,
                      context_capacity, nullptr) &&
                  (reference_current_guard.value = ds4_gpu_tensor_alloc(
                       (uint64_t)kHcWidth * sizeof(float))) != nullptr &&
                  (reference_output_guard.value = ds4_gpu_tensor_alloc(
                       (uint64_t)kHcWidth * sizeof(float))) != nullptr &&
                  (reference_logits_guard.value = ds4_gpu_tensor_alloc(
                       (uint64_t)154880u * sizeof(float))) != nullptr,
                  "allocate scalar prompt reference");
            reference_current = reference_current_guard.value;
            reference_output = reference_output_guard.value;
            const uint64_t reference_sequence_begin = sequence;
            for (int i = 0; i < prompt_tokens.value.len; ++i) {
                CHECK(execute_full_token(
                          exec, prompt_reference_state.value,
                          workspace.value,
                          (uint32_t)prompt_tokens.value.v[i],
                          reference_current, reference_output),
                      "execute scalar prompt reference");
            }
            reference_hidden.resize(kHcWidth);
            reference_logits.resize(154880u);
            CHECK(sequence == reference_sequence_begin +
                                  (uint64_t)prompt_tokens.value.len *
                                      tp_gates_per_token &&
                  ds4_gpu_tensor_read(
                      reference_current, 0u, reference_hidden.data(),
                      reference_hidden.size() * sizeof(float)) &&
                  ds4_glm5_next_output_logits(
                      &exec, workspace.value, reference_current,
                      reference_logits_guard.value) &&
                  ds4_gpu_synchronize() &&
                  ds4_gpu_tensor_read(
                      reference_logits_guard.value, 0u,
                      reference_logits.data(),
                      reference_logits.size() * sizeof(float)),
                  "capture scalar prompt hidden state and logits");
        }
        const auto prompt_begin = Clock::now();
        uint32_t executed = 0u;
        uint64_t expected_sequence = sequence;
        LayerTiming prompt_layer_timing;
        if (batch_prefill) {
            CHECK(execute_full_batch(
                      exec, state.value, prompt_workspace.value,
                      prompt_ids.value,
                      (uint32_t)prompt_tokens.value.len,
                      prompt_current, prompt_output,
                      layer_timing ? &prompt_layer_timing : nullptr) &&
                  ds4_gpu_tensor_copy(
                      current.value, 0u, prompt_current,
                      (uint64_t)(prompt_tokens.value.len - 1) *
                          kHcWidth * sizeof(float),
                      (uint64_t)kHcWidth * sizeof(float)) &&
                  ds4_gpu_synchronize(),
                  "execute complete batched chat prompt and retain last row");
            executed = (uint32_t)prompt_tokens.value.len;
            expected_sequence += tp_gates_per_token;
        } else {
            for (int i = 0; i < prompt_tokens.value.len; ++i) {
                CHECK(execute_full_token(
                          exec, state.value, workspace.value,
                          (uint32_t)prompt_tokens.value.v[i],
                          current.value, output.value,
                          layer_timing ? &prompt_layer_timing : nullptr),
                      "execute exact chat prompt token");
                executed++;
                expected_sequence += tp_gates_per_token;
            }
        }
        if (batch_prefill_compare) {
            std::vector<float> batch_hidden(kHcWidth);
            std::vector<float> batch_logits(154880u);
            CHECK(ds4_gpu_tensor_read(
                      current.value, 0u, batch_hidden.data(),
                      batch_hidden.size() * sizeof(float)) &&
                  ds4_glm5_next_output_logits(
                      &exec, workspace.value, current.value, logits.value) &&
                  ds4_gpu_synchronize() &&
                  ds4_gpu_tensor_read(
                      logits.value, 0u, batch_logits.data(),
                      batch_logits.size() * sizeof(float)),
                  "capture complete batch prompt hidden state and logits");
            const VectorError hidden_error =
                vector_error(reference_hidden, batch_hidden);
            const VectorError logits_error =
                vector_error(reference_logits, batch_logits);
            const uint32_t reference_top1 = (uint32_t)std::distance(
                reference_logits.begin(),
                std::max_element(reference_logits.begin(),
                                 reference_logits.end()));
            const uint32_t batch_top1 = (uint32_t)std::distance(
                batch_logits.begin(),
                std::max_element(batch_logits.begin(), batch_logits.end()));
            std::fprintf(stderr,
                "GLM5 complete batch prompt comparison role=%s rows=%d "
                "hidden_nrmse=%.9g hidden_cosine=%.12g "
                "hidden_max_abs=%.9g logits_nrmse=%.9g "
                "logits_cosine=%.12g logits_max_abs=%.9g "
                "reference_top1=%u batch_top1=%u\n",
                role, prompt_tokens.value.len,
                hidden_error.nrmse, hidden_error.cosine,
                hidden_error.max_abs, logits_error.nrmse,
                logits_error.cosine, logits_error.max_abs,
                reference_top1, batch_top1);
            /* Keep independent short and length-screen envelopes. The legal
             * batch reduction drift accumulates through the routed stack and
             * is measurably larger at 121 rows. These are research regression
             * ceilings, not production quality thresholds. */
            const bool long_compare = prompt_tokens.value.len > 64;
            const double prompt_nrmse_bound = long_compare ? 0.22 : 0.15;
            const double prompt_cosine_bound = long_compare ? 0.98 : 0.99;
            const double prompt_max_abs_bound = long_compare ? 2.5 : 2.0;
            CHECK(finite_error(hidden_error) &&
                      finite_error(logits_error) &&
                      hidden_error.nrmse <= prompt_nrmse_bound &&
                      hidden_error.cosine >= prompt_cosine_bound &&
                      hidden_error.max_abs <= prompt_max_abs_bound &&
                      logits_error.nrmse <= prompt_nrmse_bound &&
                      logits_error.cosine >= prompt_cosine_bound &&
                      logits_error.max_abs <= prompt_max_abs_bound &&
                      reference_top1 == batch_top1,
                  "batch prompt stays inside its provisional Lane-B envelope");
            CHECK(compare_full_prompt_states(
                      role, prompt_reference_state.value, state.value,
                      (uint32_t)prompt_tokens.value.len),
                  "batch prompt preserves selected recurrent states");
        }
        CHECK(state.value.valid &&
                  sequence == expected_sequence,
              "chat prompt commits one complete TP schedule per token");
        if (batch_prefill) {
            for (uint32_t il = 0u; il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il) {
                const ds4_glm5_next_layer_offsets &layer = offsets.layer[il];
                CHECK(layer.attention == DS4_GLM5_NEXT_ATTN_MLA ?
                          state.value.mla[il].token_count == executed :
                          state.value.kda.layer[il].token_count == executed,
                      "complete prompt batch commits every layer state");
            }
        }
        const auto prompt_end = Clock::now();
        if (layer_timing) {
            const double classified = prompt_layer_timing.kda_dense_ms +
                prompt_layer_timing.kda_routed_ms +
                prompt_layer_timing.mla_routed_ms;
            std::fprintf(stderr,
                "GLM5 prompt layer timing role=%s tokens=%d "
                "kda_dense_ms=%.3f kda_dense_calls=%llu "
                "kda_routed_ms=%.3f kda_routed_calls=%llu "
                "mla_routed_ms=%.3f mla_routed_calls=%llu "
                "classified_ms=%.3f\n",
                role, prompt_tokens.value.len,
                prompt_layer_timing.kda_dense_ms,
                (unsigned long long)prompt_layer_timing.kda_dense_calls,
                prompt_layer_timing.kda_routed_ms,
                (unsigned long long)prompt_layer_timing.kda_routed_calls,
                prompt_layer_timing.mla_routed_ms,
                (unsigned long long)prompt_layer_timing.mla_routed_calls,
                classified);
        }

        std::vector<float> host_logits(perf_mode ? 0u : 154880u);
        std::vector<uint32_t> generated;
        std::string text;
        double teacher_nll_sum = 0.0;
        double projection_ms = 0.0;
        double recurrent_ms = 0.0;
        const auto decode_begin = Clock::now();
        for (uint32_t step = 0u; step < text_generate; ++step) {
            uint32_t top1 = 0u;
            uint64_t logits_hash = 0u;
            const auto projection_begin = Clock::now();
            CHECK(read_argmax(exec, workspace.value, current.value,
                              logits.value, gpu_argmax.value, !perf_mode,
                              host_logits, top1, logits_hash),
                  "select real-text greedy token");
            const auto projection_end = Clock::now();
            projection_ms += std::chrono::duration<double, std::milli>(
                projection_end - projection_begin).count();
            char logits_error[256] = {};
            CHECK(ds4_tp_hash_check(
                      tp.tp,
                      UINT64_C(0x474c4d3554584c00) ^ step,
                      logits_hash, logits_error,
                      sizeof(logits_error)) == 1,
                  logits_error);
            const uint32_t token = teacher_ids.empty() ?
                top1 : teacher_ids[step];
            double teacher_nll = 0.0;
            if (!teacher_ids.empty()) {
                const float max_logit = *std::max_element(
                    host_logits.begin(), host_logits.end());
                double exp_sum = 0.0;
                for (float value : host_logits)
                    exp_sum += std::exp((double)value - max_logit);
                teacher_nll = (double)max_logit + std::log(exp_sum) -
                              host_logits[token];
                CHECK(std::isfinite(teacher_nll) && teacher_nll >= 0.0,
                      "teacher token has finite nonnegative NLL");
                teacher_nll_sum += teacher_nll;
            }
            generated.push_back(token);
            const bool stop = ds4_glm5_next_text_codec_token_is_stop(
                codec.value, (int)token);
            if (!stop) {
                size_t piece_len = 0u;
                char *piece = ds4_glm5_next_text_codec_token_text(
                    codec.value, (int)token, &piece_len);
                CHECK(piece != nullptr, "detokenize generated GLM5 token");
                text.append(piece, piece_len);
                free(piece);
            }
            if (teacher_ids.empty()) {
                std::fprintf(stderr,
                    "GLM5 text step role=%s step=%u token=%u top1=%u "
                    "logits=%016llx\n",
                    role, step, token, top1,
                    (unsigned long long)logits_hash);
            } else {
                std::fprintf(stderr,
                    "GLM5 text step role=%s step=%u token=%u top1=%u "
                    "teacher_nll=%.9g logits=%016llx\n",
                    role, step, token, top1, teacher_nll,
                    (unsigned long long)logits_hash);
            }
            if (stop) {
                CHECK(teacher_ids.empty() || step + 1u == teacher_ids.size(),
                      "teacher stop token must be the final prescribed token");
                break;
            }
            if (step + 1u < text_generate) {
                const auto recurrent_begin = Clock::now();
                CHECK(execute_full_token(exec, state.value, workspace.value,
                                         token, current.value, output.value),
                      "feed generated token into recurrent GLM5 state");
                const auto recurrent_end = Clock::now();
                recurrent_ms += std::chrono::duration<double, std::milli>(
                    recurrent_end - recurrent_begin).count();
                executed++;
                expected_sequence += tp_gates_per_token;
                CHECK(sequence == expected_sequence,
                      "generated token commits one complete TP schedule");
            }
        }
        const auto decode_end = Clock::now();
        CHECK(!generated.empty(), "real-text generation produced a token");
        CHECK(teacher_ids.empty() || generated.size() == teacher_ids.size(),
              "teacher-forced run consumed every prescribed token");
        const uint64_t generated_hash = fnv64(
            generated.data(), generated.size() * sizeof(generated[0]));
        char generated_error[256] = {};
        CHECK(ds4_tp_hash_check(tp.tp, UINT64_C(0x474c4d355458544f),
                                generated_hash, generated_error,
                                sizeof(generated_error)) == 1,
              generated_error);
        const double prompt_ms =
            std::chrono::duration<double, std::milli>(
                prompt_end - prompt_begin).count();
        const double decode_ms =
            std::chrono::duration<double, std::milli>(
                decode_end - decode_begin).count();
        std::fprintf(stderr,
            "GLM5 staged timing role=%s prompt_tokens=%d "
            "prompt_ms=%.3f prefill_tps=%.6f batch_prefill=%d generated=%zu "
            "decode_ms=%.3f decode_tps=%.6f projection_ms=%.3f "
            "recurrent_ms=%.3f full_logit_validation=%d\n",
            role, prompt_tokens.value.len, prompt_ms,
            prompt_tokens.value.len * 1000.0 / prompt_ms,
            batch_prefill ? 1 : 0, generated.size(), decode_ms,
            generated.size() * 1000.0 / decode_ms,
            projection_ms, recurrent_ms, perf_mode ? 0 : 1);
        if (!teacher_ids.empty()) {
            const uint64_t teacher_nll_hash =
                fnv64(&teacher_nll_sum, sizeof(teacher_nll_sum));
            char teacher_error[256] = {};
            CHECK(ds4_tp_hash_check(
                      tp.tp, UINT64_C(0x474c4d35544e4c4c),
                      teacher_nll_hash, teacher_error,
                      sizeof(teacher_error)) == 1,
                  teacher_error);
            std::fprintf(stderr,
                "PASS GLM5 teacher-forced role=%s tokens=%zu "
                "token_fnv=%016llx nll_sum=%.9g nll_mean=%.9g "
                "nll_fnv=%016llx\n",
                role, teacher_ids.size(),
                (unsigned long long)teacher_hash, teacher_nll_sum,
                teacher_nll_sum / teacher_ids.size(),
                (unsigned long long)teacher_nll_hash);
        }
        if (leader) {
            std::fprintf(stdout, "%s\n", text.c_str());
            std::fflush(stdout);
        }
        std::fprintf(stderr,
            "PASS GLM5 real-text role=%s prompt_tokens=%d generated=%zu "
            "generated_fnv=%016llx tp_seq=%llu packed_q4_bytes=%llu "
            "rdma=1\n",
            role, prompt_tokens.value.len, generated.size(),
            (unsigned long long)generated_hash,
            (unsigned long long)sequence,
            (unsigned long long)ds4_gpu_q4k_packed_slice_bytes());
        return true;
    }

    CHECK(ds4_glm5_next_embed_token(&exec, 42u, current.value),
          "embed token 42");
    for (uint32_t il = 0u; il < 3u; ++il) {
        CHECK(ds4_glm5_next_layer_forward(
                  &exec, il, &state.value, workspace.value,
                  current.value, output.value),
              "execute production dense prefix");
        std::swap(current.value, output.value);
    }
    std::vector<float> prefix(kHcWidth);
    CHECK(ds4_gpu_tensor_read(current.value, 0u, prefix.data(),
                              prefix.size() * sizeof(float)),
          "read production dense prefix");
    const uint64_t prefix_hash =
        fnv64(prefix.data(), prefix.size() * sizeof(float));
    if (prefix_hash != kPrefixFNV) {
        std::fprintf(stderr,
                     "dense-prefix fingerprint expected=%016llx observed=%016llx\n",
                     (unsigned long long)kPrefixFNV,
                     (unsigned long long)prefix_hash);
    }
    CHECK(prefix_hash == kPrefixFNV,
          "production prefix matches frozen real-GGUF fixture");

    if (full_trunk) {
        for (uint32_t il = 3u; il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il) {
            CHECK(ds4_glm5_next_layer_forward(
                      &exec, il, &state.value, workspace.value,
                      current.value, output.value),
                  "execute complete GLM5.3 trunk layer over RoCE");
            std::swap(current.value, output.value);
        }
        CHECK(state.value.valid && sequence == tp_gates_per_token,
              "complete trunk emits the exact negotiated gate schedule");
        for (uint32_t il = 0u; il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il) {
            if (ds4_glm5_next_layer_is_mla(il)) {
                CHECK(state.value.mla[il].token_count == 1u,
                      "full trunk commits one MLA cache row per MLA layer");
            } else {
                CHECK(state.value.kda.layer[il].token_count == 1u,
                      "full trunk commits one recurrent KDA step per KDA layer");
            }
        }
        std::vector<float> trunk(kHcWidth);
        CHECK(ds4_gpu_tensor_read(current.value, 0u, trunk.data(),
                                  trunk.size() * sizeof(float)),
              "read complete trunk output");
        const uint64_t trunk_hash = fnv64(
            trunk.data(), trunk.size() * sizeof(float));
        char trunk_error[256] = {};
        CHECK(ds4_tp_hash_check(tp.tp, UINT64_C(0x474c4d3546554c4c),
                                trunk_hash, trunk_error,
                                sizeof(trunk_error)) == 1,
              trunk_error);
        CHECK(ds4_glm5_next_output_logits(
                  &exec, workspace.value, current.value, logits.value) &&
              ds4_gpu_synchronize(),
              "execute replicated GLM5.3 output head");
        std::vector<float> full_logits(154880u);
        CHECK(ds4_gpu_tensor_read(
                  logits.value, 0u, full_logits.data(),
                  full_logits.size() * sizeof(float)),
              "read complete GLM5.3 vocabulary logits");
        const uint64_t logits_hash = fnv64(
            full_logits.data(), full_logits.size() * sizeof(float));
        char logits_error[256] = {};
        CHECK(ds4_tp_hash_check(tp.tp, UINT64_C(0x474c4d354c4f4749),
                                logits_hash, logits_error,
                                sizeof(logits_error)) == 1,
              logits_error);
        uint32_t argmax = 0u;
        float logit_min = full_logits[0], logit_max = full_logits[0];
        CHECK(std::isfinite(full_logits[0]), "all logits are finite");
        for (uint32_t token = 1u; token < full_logits.size(); ++token) {
            CHECK(std::isfinite(full_logits[token]), "all logits are finite");
            logit_min = std::min(logit_min, full_logits[token]);
            logit_max = std::max(logit_max, full_logits[token]);
            if (full_logits[token] > full_logits[argmax]) argmax = token;
        }
        CHECK(logit_max > logit_min,
              "complete vocabulary logits are nonconstant");
        const uint64_t packed = ds4_gpu_q4k_packed_slice_bytes();
        const uint64_t expected_packed = kshard.packed_total_bytes;
        CHECK(packed == expected_packed,
              "full trunk owns one compact Q4 shard per routed layer");
        std::fprintf(stderr,
            "PASS GLM5 prefix->layer3 full-trunk role=%s "
            "trunk_output=%016llx logits=%016llx argmax=%u "
            "tp_seq=%llu packed_q4_bytes=%llu rdma=1\n",
            role, (unsigned long long)trunk_hash,
            (unsigned long long)logits_hash, argmax,
            (unsigned long long)sequence, (unsigned long long)packed);
        if (full_tokens == 2u) {
            CHECK(argmax == 154822u,
                  "first greedy token matches the frozen two-token fixture");
            CHECK(ds4_glm5_next_embed_token(&exec, argmax, current.value),
                  "embed first greedy argmax for second full token");
            for (uint32_t il = 0u; il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il) {
                CHECK(ds4_glm5_next_layer_forward(
                          &exec, il, &state.value, workspace.value,
                          current.value, output.value),
                      "execute second greedy token through complete trunk");
                std::swap(current.value, output.value);
            }
            CHECK(state.value.valid &&
                      sequence == (uint64_t)tp_gates_per_token * 2u,
                  "two full tokens emit two exact negotiated gate schedules");
            for (uint32_t il = 0u; il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il) {
                if (ds4_glm5_next_layer_is_mla(il)) {
                    CHECK(state.value.mla[il].token_count == 2u,
                          "second token commits every MLA layer state");
                } else {
                    CHECK(state.value.kda.layer[il].token_count == 2u,
                          "second token commits every KDA layer state");
                }
            }
            std::vector<float> trunk2(kHcWidth);
            CHECK(ds4_gpu_tensor_read(current.value, 0u, trunk2.data(),
                                      trunk2.size() * sizeof(float)),
                  "read second greedy full-trunk output");
            const uint64_t trunk2_hash = fnv64(
                trunk2.data(), trunk2.size() * sizeof(float));
            char trunk2_error[256] = {};
            CHECK(ds4_tp_hash_check(tp.tp, UINT64_C(0x474c4d3547325452),
                                    trunk2_hash, trunk2_error,
                                    sizeof(trunk2_error)) == 1,
                  trunk2_error);
            CHECK(ds4_glm5_next_output_logits(
                      &exec, workspace.value, current.value, logits.value) &&
                  ds4_gpu_synchronize(),
                  "execute second greedy vocabulary projection");
            CHECK(ds4_gpu_tensor_read(
                      logits.value, 0u, full_logits.data(),
                      full_logits.size() * sizeof(float)),
                  "read second greedy vocabulary logits");
            const uint64_t logits2_hash = fnv64(
                full_logits.data(), full_logits.size() * sizeof(float));
            char logits2_error[256] = {};
            CHECK(ds4_tp_hash_check(tp.tp, UINT64_C(0x474c4d3547324c47),
                                    logits2_hash, logits2_error,
                                    sizeof(logits2_error)) == 1,
                  logits2_error);
            uint32_t argmax2 = 0u;
            float logit2_min = full_logits[0], logit2_max = full_logits[0];
            CHECK(std::isfinite(full_logits[0]),
                  "second-token logits start finite");
            for (uint32_t token = 1u; token < full_logits.size(); ++token) {
                CHECK(std::isfinite(full_logits[token]),
                      "all second-token logits are finite");
                logit2_min = std::min(logit2_min, full_logits[token]);
                logit2_max = std::max(logit2_max, full_logits[token]);
                if (full_logits[token] > full_logits[argmax2]) argmax2 = token;
            }
            CHECK(logit2_max > logit2_min &&
                      trunk2_hash != trunk_hash &&
                      trunk2_hash == UINT64_C(0x53f8b3cef301d00c) &&
                      logits2_hash == UINT64_C(0xb3fa54df32db51ec) &&
                      argmax2 == 20u &&
                      ds4_gpu_q4k_packed_slice_bytes() == expected_packed,
                  "second-token frozen fixture is stateful and cache-free");
            std::fprintf(stderr,
                "PASS GLM5 prefix->layer3 greedy2 role=%s "
                "greedy1=%u greedy2_trunk=%016llx "
                "greedy2_logits=%016llx greedy2=%u tp_seq=%llu "
                "packed_q4_bytes=%llu rdma=1\n",
                role, argmax, (unsigned long long)trunk2_hash,
                (unsigned long long)logits2_hash, argmax2,
                (unsigned long long)sequence,
                (unsigned long long)ds4_gpu_q4k_packed_slice_bytes());
        }
        return true;
    }

    CHECK(ds4_glm5_next_layer_forward(
              &exec, 3u, &state.value, workspace.value,
              current.value, output.value),
          "execute production MLA+routed layer3 over RoCE");
    const uint64_t partial_token0_after_layer3 =
        kda_tp_enabled ? 5u : 2u;
    CHECK(state.value.valid && state.value.mla[3].valid &&
          state.value.mla[3].token_count == 1u &&
          sequence == partial_token0_after_layer3,
          "layer3 commits exactly one token after both exchanges");

    std::vector<float> result(kHcWidth), kv(512u), index_key(128u),
        pool_gate(128u);
    CHECK(ds4_gpu_tensor_read(output.value, 0u, result.data(),
                              result.size() * sizeof(float)) &&
          ds4_gpu_tensor_read(state.value.mla[3].compact_kv, 0u,
                              kv.data(), kv.size() * sizeof(float)) &&
          ds4_gpu_tensor_read(state.value.mla[3].index_tail, 0u,
                              index_key.data(), index_key.size() * sizeof(float)) &&
          ds4_gpu_tensor_read(state.value.mla[3].pool_gate_tail, 0u,
                              pool_gate.data(), pool_gate.size() * sizeof(float)),
          "read committed layer3 output and resident token0 rows");
    const uint64_t output_hash = fnv64(
        result.data(), result.size() * sizeof(float));
    const uint64_t kv_hash = fnv64(kv.data(), kv.size() * sizeof(float));
    const uint64_t index_hash = fnv64(
        index_key.data(), index_key.size() * sizeof(float));
    const uint64_t gate_hash = fnv64(
        pool_gate.data(), pool_gate.size() * sizeof(float));
    const uint64_t layer3_packed_q4_bytes =
        ds4_gpu_q4k_packed_slice_bytes();
    char error[256] = {};
    CHECK(ds4_tp_hash_check(tp.tp, UINT64_C(0x474c4d35334f5554),
                            output_hash, error, sizeof(error)) == 1, error);
    CHECK(layer3_packed_q4_bytes != 0u,
          "layer3 owns compact Q4 rank residency");

    CHECK(ds4_glm5_next_layer_forward(
              &exec, 4u, &state.value, workspace.value,
              output.value, current.value),
          "compose production KDA+routed layer4 over RoCE");
    ds4_glm5_kda_digest kda4_token0 = {};
    const uint64_t partial_token0_after_layer4 =
        kda_tp_enabled ? 7u : 3u;
    CHECK(state.value.kda.layer[4].token_count == 1u &&
          sequence == partial_token0_after_layer4 &&
          ds4_glm5_kda_layer_digest(&state.value.kda.layer[4], current.value,
                                    kHcWidth, &kda4_token0),
          "layer4 commits one recurrent KDA step and one FFN exchange");
    const uint64_t packed_q4_bytes = ds4_gpu_q4k_packed_slice_bytes();
    uint64_t kda4_owned0 = 0u;
    CHECK((full_trunk ? packed_q4_bytes == layer3_packed_q4_bytes
                      : packed_q4_bytes > layer3_packed_q4_bytes) &&
          ((!kda_tp_enabled &&
            ds4_tp_hash_check(
                tp.tp, UINT64_C(0x474c4d35344b4430),
                fnv64(&kda4_token0, sizeof(kda4_token0)),
                error, sizeof(error)) == 1) ||
           (kda_tp_enabled &&
            kda_state_has_exact_local_ownership(
                state.value.kda.layer[4], exec.tp_rank, &kda4_owned0) &&
            ds4_tp_hash_check(
                tp.tp, UINT64_C(0x474c4d35344f5550),
                kda4_token0.output_fnv64, error, sizeof(error)) == 1)),
          "layer4 output agrees and state ownership matches TP mode");

    /* The legacy partial fixture deliberately stops after layer 4.  Once
     * every KDA attention site is in the negotiated gate mask, beginning a
     * second token here would violate the 87-gate full-token schedule by
     * skipping layers 5..45.  The exact 33+1 continuation oracle covers the
     * split recurrent state; multi-token transport is exercised only by the
     * complete-trunk path above. */
    if (kda_tp_enabled) {
        std::fprintf(stderr,
            "PASS GLM5 prefix->layer3 token0 role=%s output=%016llx "
            "layer4_token0=%016llx kda4_owned0=%016llx tp_seq=%llu "
            "layer3_packed_q4_bytes=%llu packed_q4_bytes=%llu "
            "kda_tp=1 partial_schedule=1 window_cache_bytes=0 rdma=1\n",
            role, (unsigned long long)output_hash,
            (unsigned long long)kda4_token0.output_fnv64,
            (unsigned long long)kda4_owned0,
            (unsigned long long)sequence,
            (unsigned long long)layer3_packed_q4_bytes,
            (unsigned long long)packed_q4_bytes);
        return true;
    }

    CHECK(ds4_glm5_next_embed_token(&exec, 43u, current.value),
          "embed token 43");
    for (uint32_t il = 0u; il < 3u; ++il) {
        CHECK(ds4_glm5_next_layer_forward(
                  &exec, il, &state.value, workspace.value,
                  current.value, output.value),
              "execute second-token dense prefix");
        std::swap(current.value, output.value);
    }
    CHECK(ds4_glm5_next_layer_forward(
              &exec, 3u, &state.value, workspace.value,
              current.value, output.value),
          "execute second-token MLA cache-read layer3 over RoCE");
    const uint64_t partial_token1_after_layer3 =
        partial_token0_after_layer4 + (kda_tp_enabled ? 5u : 2u);
    CHECK(state.value.valid && state.value.mla[3].token_count == 2u &&
          sequence == partial_token1_after_layer3 &&
          ds4_gpu_q4k_packed_slice_bytes() == packed_q4_bytes,
          "second token commits after two exchanges without Q4 duplication");

    std::vector<float> result1(kHcWidth), kv2(1024u), index2(256u),
        pool2(256u);
    CHECK(ds4_gpu_tensor_read(output.value, 0u, result1.data(),
                              result1.size() * sizeof(float)) &&
          ds4_gpu_tensor_read(state.value.mla[3].compact_kv, 0u,
                              kv2.data(), kv2.size() * sizeof(float)) &&
          ds4_gpu_tensor_read(state.value.mla[3].index_tail, 0u,
                              index2.data(), index2.size() * sizeof(float)) &&
          ds4_gpu_tensor_read(state.value.mla[3].pool_gate_tail, 0u,
                              pool2.data(), pool2.size() * sizeof(float)),
          "read second-token output and two resident cache rows");
    CHECK(std::memcmp(kv2.data(), kv.data(), kv.size() * sizeof(float)) == 0 &&
          std::memcmp(index2.data(), index_key.data(),
                      index_key.size() * sizeof(float)) == 0 &&
          std::memcmp(pool2.data(), pool_gate.data(),
                      pool_gate.size() * sizeof(float)) == 0,
          "second token preserves the committed token0 cache rows exactly");
    const uint64_t output1_hash = fnv64(
        result1.data(), result1.size() * sizeof(float));
    const uint64_t kv2_hash = fnv64(kv2.data(), kv2.size() * sizeof(float));
    const uint64_t index2_hash = fnv64(
        index2.data(), index2.size() * sizeof(float));
    const uint64_t gate2_hash = fnv64(
        pool2.data(), pool2.size() * sizeof(float));
    uint64_t state_hash = fnv64(&output1_hash, sizeof(output1_hash));
    state_hash ^= kv2_hash;
    state_hash ^= index2_hash;
    state_hash ^= gate2_hash;
    CHECK(output1_hash != output_hash &&
          std::memcmp(kv2.data(), kv2.data() + kv.size(),
                      kv.size() * sizeof(float)) != 0 &&
          ds4_tp_hash_check(tp.tp, UINT64_C(0x474c4d3533544f4b),
                            state_hash, error, sizeof(error)) == 1,
          "second-token output/state are nontrivial and rank-identical");

    CHECK(ds4_glm5_next_layer_forward(
              &exec, 4u, &state.value, workspace.value,
              output.value, current.value),
          "execute second-token KDA+routed layer4 over RoCE");
    ds4_glm5_kda_digest kda4_token1 = {};
    uint64_t kda4_owned1 = 0u;
    const uint64_t partial_token1_after_layer4 =
        partial_token1_after_layer3 + (kda_tp_enabled ? 2u : 1u);
    CHECK(state.value.valid && state.value.kda.layer[4].token_count == 2u &&
          sequence == partial_token1_after_layer4 &&
          ds4_gpu_q4k_packed_slice_bytes() == packed_q4_bytes &&
          ds4_glm5_kda_layer_digest(&state.value.kda.layer[4], current.value,
                                    kHcWidth, &kda4_token1) &&
          kda4_token1.token_count == 2u &&
          kda4_token1.output_fnv64 != kda4_token0.output_fnv64 &&
          ((!kda_tp_enabled &&
            ds4_tp_hash_check(
                tp.tp, UINT64_C(0x474c4d35344b4431),
                fnv64(&kda4_token1, sizeof(kda4_token1)),
                error, sizeof(error)) == 1) ||
           (kda_tp_enabled &&
            kda_state_has_exact_local_ownership(
                state.value.kda.layer[4], exec.tp_rank, &kda4_owned1) &&
            kda4_owned1 != kda4_owned0 &&
            ds4_tp_hash_check(
                tp.tp, UINT64_C(0x474c4d35344f5531),
                kda4_token1.output_fnv64, error, sizeof(error)) == 1)),
          "layer4 token1 advances recurrent state without Q4 duplication");
    std::fprintf(stderr,
        "PASS GLM5 prefix->layer3 token0 role=%s output=%016llx kv=%016llx "
        "index=%016llx pool_gate=%016llx token1_output=%016llx "
        "kv2=%016llx index2=%016llx pool2=%016llx "
        "layer4_token0=%016llx layer4_token1=%016llx tp_seq=%llu "
        "layer3_packed_q4_bytes=%llu packed_q4_bytes=%llu "
        "kda4_owned0=%016llx kda4_owned1=%016llx kda_tp=%d "
        "window_cache_bytes=0 rdma=1\n",
        role, (unsigned long long)output_hash,
        (unsigned long long)kv_hash, (unsigned long long)index_hash,
        (unsigned long long)gate_hash, (unsigned long long)output1_hash,
        (unsigned long long)kv2_hash, (unsigned long long)index2_hash,
        (unsigned long long)gate2_hash,
        (unsigned long long)kda4_token0.output_fnv64,
        (unsigned long long)kda4_token1.output_fnv64,
        (unsigned long long)sequence,
        (unsigned long long)layer3_packed_q4_bytes,
        (unsigned long long)packed_q4_bytes,
        (unsigned long long)kda4_owned0,
        (unsigned long long)kda4_owned1,
        kda_tp_enabled ? 1 : 0);
    return true;
}

}  // namespace

int main() {
    const bool ok = run();
    ds4_gpu_cleanup();
    return ok ? 0 : 1;
}
