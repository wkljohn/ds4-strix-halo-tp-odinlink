#include "ds4_glm5_kda.h"
#include "ds4_glm5_next_exec.h"
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
extern "C" {
#include "ds4_tp.h"
}
#include "tests/glm5_gguf_test.hpp"
#include "tests/glm5_next_real_offsets.hpp"

#include <cmath>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(expr, message) do {                                           \
    if (!(expr)) {                                                          \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__);    \
        return false;                                                       \
    }                                                                       \
} while (0)

namespace {
constexpr uint32_t kWidth = 4096u;
constexpr uint32_t kHc = 4u;
constexpr uint32_t kHcWidth = kWidth * kHc;
constexpr uint32_t kMix = 24u;
constexpr uint32_t kDense = 12288u;
constexpr uint32_t kVocab = 154880u;
/* Lane-C correction: these hashes use model rms_norm_eps=1e-5 in the KDA
 * gated output.  See the independent same-GGUF evidence before updating. */
constexpr uint64_t kExpectedBlock0FNV = UINT64_C(0x74ba1355cebb30b3);
constexpr uint64_t kExpectedPrefixFNV = UINT64_C(0x6b704c8b12a398ef);

bool parse_token_env(const char *name, uint32_t fallback, uint32_t &token,
                     bool *present = nullptr) {
    if (present) *present = false;
    token = fallback;
    const char *value = std::getenv(name);
    if (!value || !value[0]) return true;
    if (present) *present = true;
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (!end || end == value || *end != '\0' || parsed >= kVocab) return false;
    token = (uint32_t)parsed;
    return true;
}

bool selected_tokens(uint32_t &token, uint32_t &token2, bool &has_token2) {
    token = 42u;
    token2 = 0u;
    has_token2 = false;
    return parse_token_env("DS4_GLM5_DENSE_TOKEN", 42u, token) &&
           parse_token_env("DS4_GLM5_DENSE_TOKEN2", 0u, token2,
                           &has_token2);
}

bool selected_trace_layer(uint32_t &layer) {
    return parse_token_env("DS4_GLM5_DENSE_TRACE_LAYER", 0u, layer) &&
           layer < 3u;
}

struct DenseOffsets {
    uint64_t embedding = 0u;
    uint64_t attn_hc_fn = 0u, attn_hc_scale = 0u, attn_hc_base = 0u;
    uint64_t ffn_hc_fn = 0u, ffn_hc_scale = 0u, ffn_hc_base = 0u;
    uint64_t attn_norm = 0u, ffn_norm = 0u;
    uint64_t ffn_gate = 0u, ffn_up = 0u, ffn_down = 0u;
    ds4_glm5_kda_weight_offsets kda = {};
};

bool layer_tensor(const Glm5TestGGUF &g, uint32_t layer, const char *suffix,
                  std::initializer_list<uint64_t> dims, uint32_t type,
                  uint64_t &offset) {
    char name[96];
    const int n = std::snprintf(name, sizeof(name), "blk.%u.%s", layer, suffix);
    return n > 0 && (size_t)n < sizeof(name) &&
           g.tensor(name, dims, type, offset);
}

bool bind(const Glm5TestGGUF &g, uint32_t layer, DenseOffsets &w) {
    return g.tensor("token_embd.weight", {kWidth, kVocab}, 30u, w.embedding) &&
           layer_tensor(g, layer, "hc_attn_fn.weight", {kHcWidth, kMix}, 30u, w.attn_hc_fn) &&
           layer_tensor(g, layer, "hc_attn_scale.weight", {3u}, 0u, w.attn_hc_scale) &&
           layer_tensor(g, layer, "hc_attn_base.weight", {kMix}, 0u, w.attn_hc_base) &&
           layer_tensor(g, layer, "hc_ffn_fn.weight", {kHcWidth, kMix}, 30u, w.ffn_hc_fn) &&
           layer_tensor(g, layer, "hc_ffn_scale.weight", {3u}, 0u, w.ffn_hc_scale) &&
           layer_tensor(g, layer, "hc_ffn_base.weight", {kMix}, 0u, w.ffn_hc_base) &&
           layer_tensor(g, layer, "attn_norm.weight", {kWidth}, 0u, w.attn_norm) &&
           layer_tensor(g, layer, "ffn_norm.weight", {kWidth}, 0u, w.ffn_norm) &&
           layer_tensor(g, layer, "ffn_gate.weight", {kWidth, kDense}, 8u, w.ffn_gate) &&
           layer_tensor(g, layer, "ffn_up.weight", {kWidth, kDense}, 8u, w.ffn_up) &&
           layer_tensor(g, layer, "ffn_down.weight", {kDense, kWidth}, 8u, w.ffn_down) &&
           layer_tensor(g, layer, "kda_q.weight", {kWidth, 8192u}, 30u, w.kda.q) &&
           layer_tensor(g, layer, "kda_k.weight", {kWidth, 8192u}, 30u, w.kda.k) &&
           layer_tensor(g, layer, "kda_v.weight", {kWidth, 8192u}, 30u, w.kda.v) &&
           layer_tensor(g, layer, "kda_output.weight", {8192u, kWidth}, 30u, w.kda.output) &&
           layer_tensor(g, layer, "kda_q_conv.weight", {4u, 1u, 8192u}, 0u, w.kda.q_conv) &&
           layer_tensor(g, layer, "kda_k_conv.weight", {4u, 1u, 8192u}, 0u, w.kda.k_conv) &&
           layer_tensor(g, layer, "kda_v_conv.weight", {4u, 1u, 8192u}, 0u, w.kda.v_conv) &&
           layer_tensor(g, layer, "kda_f_a.weight", {kWidth, 128u}, 30u, w.kda.f_a) &&
           layer_tensor(g, layer, "kda_f_b.weight", {128u, 8192u}, 30u, w.kda.f_b) &&
           layer_tensor(g, layer, "kda_g_a.weight", {kWidth, 128u}, 30u, w.kda.g_a) &&
           layer_tensor(g, layer, "kda_g_b.weight", {128u, 8192u}, 30u, w.kda.g_b) &&
           layer_tensor(g, layer, "kda_beta.weight", {kWidth, 64u}, 30u, w.kda.beta) &&
           layer_tensor(g, layer, "kda_o_norm.weight", {128u}, 0u, w.kda.o_norm) &&
           layer_tensor(g, layer, "kda_dt_bias.weight", {8192u}, 0u, w.kda.dt_bias) &&
           layer_tensor(g, layer, "kda_a_log.weight", {64u}, 0u, w.kda.a_log) &&
           (w.kda.attn_norm = w.attn_norm) != 0u;
}

uint64_t fnv64(const std::vector<float> &values) {
    uint64_t hash = UINT64_C(1469598103934665603);
    const unsigned char *p = reinterpret_cast<const unsigned char *>(values.data());
    for (size_t i = 0u; i < values.size() * sizeof(float); ++i) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

bool close_vectors(const std::vector<float> &reference,
                   const std::vector<float> &candidate,
                   const char *label) {
    if (reference.size() != candidate.size() || reference.empty()) return false;
    double ref2 = 0.0, err2 = 0.0, dot = 0.0, cand2 = 0.0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(candidate[i]))
            return false;
        const double r = reference[i];
        const double c = candidate[i];
        const double e = c - r;
        ref2 += r * r;
        cand2 += c * c;
        dot += r * c;
        err2 += e * e;
        max_abs = std::fmax(max_abs, std::fabs(candidate[i] - reference[i]));
    }
    const double nrmse = std::sqrt(err2 / std::fmax(ref2, 1.0e-30));
    const double cosine = dot / std::sqrt(
        std::fmax(ref2 * cand2, 1.0e-30));
    std::fprintf(stderr,
                 "GLM5 dense batch compare %s n=%zu nrmse=%.9g "
                 "cosine=%.12g max_abs=%.9g\n",
                 label, reference.size(), nrmse, cosine, max_abs);
    /* This is a Lane-C arithmetic comparison: batched GEMMs may associate
     * FP32 terms differently, but the complete state must remain tightly
     * bounded and directionally indistinguishable. */
    return nrmse <= 1.0e-5 && cosine >= 0.999999 && max_abs <= 1.0e-5f;
}

bool write_f32_file(const char *path, const std::vector<float> &values) {
    if (!path || !path[0] || values.empty()) return false;
    FILE *fp = std::fopen(path, "wb");
    if (!fp) return false;
    const size_t written = std::fwrite(
        values.data(), sizeof(float), values.size(), fp);
    return std::fclose(fp) == 0 && written == values.size();
}

struct Tensors {
    ds4_gpu_tensor *cur = nullptr, *flat = nullptr, *mix = nullptr;
    ds4_gpu_tensor *split = nullptr, *collapsed = nullptr;
    ds4_gpu_tensor *attn = nullptr, *after_attn = nullptr;
    ds4_gpu_tensor *ffn_flat = nullptr, *ffn_mix = nullptr;
    ds4_gpu_tensor *ffn_split = nullptr, *ffn_collapsed = nullptr;
    ds4_gpu_tensor *ffn_hidden = nullptr, *gate = nullptr, *up = nullptr;
    ds4_gpu_tensor *mid = nullptr, *down = nullptr, *out = nullptr;
    void clear() {
        ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(down);
        ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(up);
        ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(ffn_hidden);
        ds4_gpu_tensor_free(ffn_collapsed); ds4_gpu_tensor_free(ffn_split);
        ds4_gpu_tensor_free(ffn_mix); ds4_gpu_tensor_free(ffn_flat);
        ds4_gpu_tensor_free(after_attn); ds4_gpu_tensor_free(attn);
        ds4_gpu_tensor_free(collapsed);
        ds4_gpu_tensor_free(split); ds4_gpu_tensor_free(mix);
        ds4_gpu_tensor_free(flat); ds4_gpu_tensor_free(cur);
        cur = flat = mix = split = collapsed = nullptr;
        attn = after_attn = ffn_flat = ffn_mix = nullptr;
        ffn_split = ffn_collapsed = ffn_hidden = nullptr;
        gate = up = mid = down = out = nullptr;
    }
    ~Tensors() { clear(); }
};

bool allocate(Tensors &t) {
    const auto f32 = [](uint64_t n) { return ds4_gpu_tensor_alloc(n * 4u); };
    t.cur = f32(kHcWidth); t.flat = f32(kHcWidth); t.mix = f32(kMix);
    t.split = f32(kMix); t.collapsed = f32(kWidth);
    t.attn = f32(kWidth); t.after_attn = f32(kHcWidth);
    t.ffn_flat = f32(kHcWidth); t.ffn_mix = f32(kMix);
    t.ffn_split = f32(kMix); t.ffn_collapsed = f32(kWidth);
    t.ffn_hidden = f32(kWidth); t.gate = f32(kDense); t.up = f32(kDense);
    t.mid = f32(kDense); t.down = f32(kWidth); t.out = f32(kHcWidth);
    return t.cur && t.flat && t.mix && t.split && t.collapsed &&
           t.attn && t.after_attn && t.ffn_flat && t.ffn_mix &&
           t.ffn_split && t.ffn_collapsed && t.ffn_hidden && t.gate &&
           t.up && t.mid && t.down && t.out;
}

bool execute(const Glm5TestGGUF &g, const DenseOffsets &w,
             ds4_glm5_kda_layer_state *state,
             ds4_glm5_kda_workspace *workspace, Tensors &t,
             std::vector<float> &result, float rms_norm_eps, float hc_eps,
             const char *trace_prefix = nullptr) {
    CHECK(ds4_gpu_rms_norm_plain_rows_tensor(
              t.flat, t.cur, kHcWidth, 1u, rms_norm_eps) &&
          ds4_gpu_matmul_bf16_tensor(
              t.mix, g.map, g.size, w.attn_hc_fn, kHcWidth, kMix,
              t.flat, 1u) &&
          ds4_gpu_hc_split_weighted_sum_tensor(
              t.collapsed, t.split, t.mix, t.cur,
              g.map, g.size, w.attn_hc_scale, w.attn_hc_base,
              kWidth, kHc, 20u, hc_eps) &&
          ds4_glm5_kda_layer_forward(
              state, workspace, &w.kda, g.map, g.size,
              t.collapsed, t.attn, 1u, rms_norm_eps) &&
          ds4_gpu_hc_expand_split_tensor(
              t.after_attn, t.attn, t.cur, t.split, kWidth, kHc) &&
          ds4_gpu_rms_norm_plain_rows_tensor(
              t.ffn_flat, t.after_attn, kHcWidth, 1u, rms_norm_eps) &&
          ds4_gpu_matmul_bf16_tensor(
              t.ffn_mix, g.map, g.size, w.ffn_hc_fn, kHcWidth, kMix,
              t.ffn_flat, 1u) &&
          ds4_gpu_hc_split_weighted_sum_norm_tensor(
              t.ffn_collapsed, t.ffn_hidden, t.ffn_split, t.ffn_mix,
              t.after_attn, g.map, g.size, w.ffn_hc_scale, w.ffn_hc_base,
              w.ffn_norm, kWidth, kHc, 20u, hc_eps, rms_norm_eps) &&
          ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
              t.gate, t.up, t.mid, g.map, g.size, w.ffn_gate, w.ffn_up,
              kWidth, kDense, t.ffn_hidden, 10.0f) &&
          ds4_gpu_matmul_q8_0_tensor(
              t.down, g.map, g.size, w.ffn_down,
              kDense, kWidth, t.mid, 1u) &&
          ds4_gpu_hc_expand_split_tensor(
              t.out, t.down, t.after_attn, t.ffn_split, kWidth, kHc) &&
          ds4_gpu_synchronize(),
          "execute embedding through complete dense block 0");
    result.resize(kHcWidth);
    CHECK(ds4_gpu_tensor_read(t.out, 0u, result.data(),
                              (uint64_t)result.size() * sizeof(float)),
          "read complete block-0 state");
    for (float value : result) CHECK(std::isfinite(value), "finite block-0 state");
    if (trace_prefix) {
        struct TraceTensor {
            const char *name;
            ds4_gpu_tensor *tensor;
            uint64_t count;
        } traces[] = {
            {"input_hc", t.cur, kHcWidth},
            {"attn_split", t.split, kMix},
            {"attn_collapsed", t.collapsed, kWidth},
            {"attn_output", t.attn, kWidth},
            {"after_attn", t.after_attn, kHcWidth},
            {"ffn_split", t.ffn_split, kMix},
            {"ffn_hidden", t.ffn_hidden, kWidth},
            {"ffn_mid", t.mid, kDense},
            {"ffn_down", t.down, kWidth},
            {"output_hc", t.out, kHcWidth},
        };
        for (const TraceTensor &trace : traces) {
            std::vector<float> values(trace.count);
            char path[512];
            const int n = std::snprintf(path, sizeof(path), "%s.%s.f32",
                                        trace_prefix, trace.name);
            CHECK(n > 0 && (size_t)n < sizeof(path) &&
                  ds4_gpu_tensor_read(trace.tensor, 0u, values.data(),
                                      trace.count * sizeof(float)),
                  "read external-oracle trace tensor");
            FILE *fp = std::fopen(path, "wb");
            CHECK(fp && std::fwrite(values.data(), sizeof(float), trace.count,
                                    fp) == trace.count &&
                  std::fclose(fp) == 0,
                  "write external-oracle trace tensor");
        }
    }
    return true;
}
}  // namespace

static bool run_test(void) {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    CHECK(model && model[0], "model environment");
    Glm5TestGGUF gguf;
    DenseOffsets weights[3];
    uint32_t token = 0u, token2 = 0u;
    uint32_t trace_layer = 0u;
    bool has_token2 = false;
    float rms_norm_eps = 0.0f;
    float hc_eps = 0.0f;
    CHECK(selected_tokens(token, token2, has_token2),
          "valid dense-prefix token override");
    CHECK(selected_trace_layer(trace_layer), "valid dense trace layer");
    CHECK(gguf.open_file(model) &&
              gguf.metadata(
                  "glm5-next.attention.layer_norm_rms_epsilon",
                  rms_norm_eps) &&
              rms_norm_eps == 1.0e-5f &&
              gguf.metadata("glm5-next.hyper_connection.epsilon", hc_eps) &&
              hc_eps == 1.0e-6f,
          "open real-GGUF model with expected RMS and mHC epsilon");
    for (uint32_t il = 0u; il < 3u; ++il) {
        CHECK(bind(gguf, il, weights[il]), "bind complete dense-prefix layer");
    }
    ds4_gpu_config config = {};
    config.n_gpus = 1u;
    config.device_indices[0] = 0u;
    CHECK(ds4_gpu_init_multi(&config) &&
          ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map) &&
          ds4_gpu_set_model_map(gguf.map, gguf.size),
          "initialize gfx1151 and register model map");
    ds4_tp_test_reset_exchange_calls();

    ds4_glm5_layer_kind schedule[3] = {
        {.layer = 0u, .is_kda = true},
        {.layer = 1u, .is_kda = true},
        {.layer = 2u, .is_kda = true},
    };
    ds4_glm5_kda_slot slot = {};
    ds4_glm5_kda_workspace workspace = {};
    Tensors tensors;
    CHECK(ds4_glm5_kda_slot_init(&slot, schedule, 3u, 1u, nullptr) &&
          ds4_glm5_kda_workspace_init(&workspace, 1u) && allocate(tensors),
          "allocate complete dense-prefix state and workspace");
    std::vector<float> block0, first, second;
    const auto execute_prefix = [&](uint32_t input_token,
                                    std::vector<float> &result,
                                    bool capture_block0) {
        if (!ds4_gpu_embed_token_hc_bf16_tensor(
                tensors.cur, gguf.map, gguf.size, weights[0].embedding,
                kVocab, input_token, kWidth, kHc)) return false;
        for (uint32_t il = 0u; il < 3u; ++il) {
            std::vector<float> layer_result;
            if (!execute(gguf, weights[il], &slot.layer[il], &workspace,
                         tensors, layer_result, rms_norm_eps, hc_eps,
                         capture_block0 && il == trace_layer ?
                             std::getenv("DS4_GLM5_DENSE_TRACE_PREFIX") :
                             nullptr)) return false;
            if (capture_block0 && il == 0u) block0 = layer_result;
            if (il + 1u < 3u) std::swap(tensors.cur, tensors.out);
            else result = std::move(layer_result);
        }
        return true;
    };
    CHECK(execute_prefix(token, first, !has_token2),
          "execute complete dense prefix first token");
    if (has_token2) {
        CHECK(execute_prefix(token2, first, true),
              "execute complete dense prefix second token");
    }
    for (uint32_t il = 0u; il < 3u; ++il) {
        CHECK(slot.layer[il].token_count == (has_token2 ? 2u : 1u),
              "dense-prefix KDA layer advances requested tokens");
    }
    CHECK(ds4_glm5_kda_slot_reset(&slot) &&
          execute_prefix(token, second, false) &&
          (!has_token2 || execute_prefix(token2, second, false)),
          "repeat complete dense-prefix token sequence");
    CHECK(first == second, "complete dense prefix is deterministic after reset");
    for (uint32_t il = 0u; il < 3u; ++il) {
        CHECK(slot.layer[il].token_count == (has_token2 ? 2u : 1u),
              "reset dense-prefix KDA layer advances requested tokens");
    }
    CHECK(ds4_tp_test_get_exchange_calls() == 0u,
          "replicated dense block invokes no TP exchange");
    const uint64_t block0_hash = fnv64(block0);
    if (!has_token2 && token == 42u &&
        (block0_hash != kExpectedBlock0FNV ||
         fnv64(first) != kExpectedPrefixFNV)) {
        std::fprintf(stderr,
                     "corrected dense fingerprints: block0=%016" PRIx64
                     " prefix=%016" PRIx64 "\n",
                     block0_hash, fnv64(first));
    }
    CHECK(has_token2 || token != 42u || block0_hash == kExpectedBlock0FNV,
          "default complete block-0 output matches the pinned same-GGUF fingerprint");
    const uint64_t hash = fnv64(first);
    CHECK(has_token2 || token != 42u || hash == kExpectedPrefixFNV,
          "default complete dense-prefix output matches the pinned same-GGUF fingerprint");

    ds4_glm5_next_model_offsets model_offsets = {};
    ds4_glm5_next_state resident = {};
    ds4_glm5_next_workspace *runtime_workspace = nullptr;
    CHECK(glm5_next_bind_real_offsets(gguf, model_offsets) &&
          ds4_glm5_next_state_init(&resident, &model_offsets, 1u, nullptr) &&
          (runtime_workspace = ds4_glm5_next_workspace_create()) != nullptr,
          "bind full model and allocate production resident runtime");
    ds4_glm5_next_exec_ctx exec = {};
    exec.model_map = gguf.map;
    exec.model_size = gguf.size;
    exec.model = &model_offsets;
    ds4_gpu_tensor *tiny_output = ds4_gpu_tensor_alloc(sizeof(float));
    CHECK(tiny_output && ds4_glm5_next_embed_token(&exec, token, tensors.cur) &&
          !ds4_glm5_next_layer_forward(
              &exec, 0u, &resident, runtime_workspace,
              tensors.cur, tiny_output) &&
          resident.valid && resident.kda.layer[0].token_count == 0u,
          "undersized output fails before resident state mutation");
    ds4_gpu_tensor_free(tiny_output);
    const auto execute_runtime_prefix = [&](uint32_t input_token,
                                            std::vector<float> &result) {
        if (!ds4_glm5_next_embed_token(&exec, input_token, tensors.cur)) return false;
        for (uint32_t il = 0u; il < 3u; ++il) {
            if (!ds4_glm5_next_layer_forward(
                    &exec, il, &resident, runtime_workspace,
                    tensors.cur, tensors.out)) return false;
            if (il + 1u < 3u) std::swap(tensors.cur, tensors.out);
        }
        result.resize(kHcWidth);
        return ds4_gpu_tensor_read(tensors.out, 0u, result.data(),
                                   (uint64_t)kHcWidth * sizeof(float)) != 0;
    };
    std::vector<float> runtime_first, runtime_second;
    CHECK(execute_runtime_prefix(token, runtime_first) &&
          (!has_token2 || execute_runtime_prefix(token2, runtime_first)) &&
          runtime_first == first,
          "production executor matches the independent dense-prefix harness");
    CHECK(ds4_glm5_next_state_reset(&resident) &&
          execute_runtime_prefix(token, runtime_second) &&
          (!has_token2 || execute_runtime_prefix(token2, runtime_second)) &&
          runtime_second == runtime_first,
          "production resident executor is deterministic after atomic reset");
    for (uint32_t il = 0u; il < 3u; ++il) {
        CHECK(resident.kda.layer[il].token_count == (has_token2 ? 2u : 1u),
              "production dense-prefix KDA state advances requested tokens");
    }
    CHECK(!ds4_glm5_next_layer_forward(
              &exec, 3u, &resident, runtime_workspace,
              tensors.cur, tensors.out) && resident.valid &&
          resident.mla[3].token_count == 0u,
          "unimplemented MLA+routed kind fails before resident mutation");

    /* Stateful batch gate: compare three batched prompt rows with the same
     * independent one-row harness, then consume a fourth token through the
     * ordinary production decode API to prove recurrent-state continuity. */
    const uint32_t batch_tokens[] = {42u, 154822u, 154824u, 17u};
    std::vector<float> sequential_rows;
    std::vector<float> sequential_continue;
    CHECK(ds4_glm5_kda_slot_reset(&slot),
          "reset independent dense-prefix oracle for batch sequence");
    for (size_t i = 0u; i < 4u; ++i) {
        std::vector<float> row;
        CHECK(execute_prefix(batch_tokens[i], row, false),
              "execute independent sequential batch oracle row");
        if (i < 3u) {
            sequential_rows.insert(sequential_rows.end(), row.begin(), row.end());
        } else {
            sequential_continue = std::move(row);
        }
    }

    ds4_glm5_next_workspace *batch_workspace =
        ds4_glm5_next_workspace_create_capacity(3u);
    ds4_gpu_tensor *batch_ids =
        ds4_gpu_tensor_alloc(3u * sizeof(uint32_t));
    ds4_gpu_tensor *batch_cur =
        ds4_gpu_tensor_alloc(3ull * kHcWidth * sizeof(float));
    ds4_gpu_tensor *batch_out =
        ds4_gpu_tensor_alloc(3ull * kHcWidth * sizeof(float));
    CHECK(batch_workspace && batch_ids && batch_cur && batch_out &&
          ds4_gpu_tensor_write(batch_ids, 0u, batch_tokens,
                               3u * sizeof(uint32_t)) &&
          ds4_glm5_next_state_reset(&resident) &&
          ds4_glm5_next_embed_tokens(&exec, batch_ids, 3u, batch_cur),
          "allocate and embed production dense-prefix batch");
    for (uint32_t il = 0u; il < 3u; ++il) {
        CHECK(ds4_glm5_next_layer_forward_batch(
                  &exec, il, &resident, batch_workspace,
                  batch_cur, batch_out, 3u),
              "execute production dense-prefix batch layer");
        if (il + 1u < 3u) std::swap(batch_cur, batch_out);
    }
    std::vector<float> batch_rows(3ull * kHcWidth);
    CHECK(ds4_gpu_tensor_read(batch_out, 0u, batch_rows.data(),
                              batch_rows.size() * sizeof(float)) &&
          close_vectors(sequential_rows, batch_rows, "three-row output"),
          "batched dense-prefix output matches sequential same-GGUF oracle");
    const char *batch_trace = std::getenv("DS4_GLM5_DENSE_BATCH_OUTPUT");
    CHECK(!batch_trace || write_f32_file(batch_trace, batch_rows),
          "write three-row dense-prefix batch for external oracle");
    for (uint32_t il = 0u; il < 3u; ++il) {
        CHECK(resident.kda.layer[il].token_count == 3u,
              "batched dense-prefix KDA state advances all rows");
    }
    std::vector<float> batch_continue;
    CHECK(execute_runtime_prefix(batch_tokens[3], batch_continue) &&
          close_vectors(sequential_continue, batch_continue,
                        "post-batch continuation"),
          "ordinary decode continues from batched KDA recurrent state");
    ds4_gpu_tensor_free(batch_out);
    ds4_gpu_tensor_free(batch_cur);
    ds4_gpu_tensor_free(batch_ids);
    ds4_glm5_next_workspace_destroy(batch_workspace);

    const uint32_t prompt33[] = {
        154822u, 154824u, 154826u, 25062u, 287u, 29905u, 371u, 25u,
        7487u, 154827u, 675u, 279u, 11478u, 7735u, 369u, 6623u, 323u,
        279u, 3150u, 315u, 41907u, 323u, 4968u, 18110u, 558u, 13u,
        21754u, 304u, 825u, 11646u, 13u, 154828u, 154841u,
    };
    constexpr uint32_t kPromptRows =
        (uint32_t)(sizeof(prompt33) / sizeof(prompt33[0]));
    using Clock = std::chrono::steady_clock;
    std::vector<float> prompt_reference;
    std::vector<float> prompt_reference_continue;
    CHECK(ds4_glm5_kda_slot_reset(&slot) && ds4_gpu_synchronize(),
          "reset independent 33-token dense-prefix oracle");
    const auto sequential_begin = Clock::now();
    for (uint32_t i = 0u; i <= kPromptRows; ++i) {
        std::vector<float> row;
        const uint32_t input_token = i < kPromptRows ? prompt33[i] : 17u;
        CHECK(execute_prefix(input_token, row, false),
              "execute independent 33-token dense-prefix oracle row");
        if (i < kPromptRows) {
            prompt_reference.insert(prompt_reference.end(),
                                    row.begin(), row.end());
        } else {
            prompt_reference_continue = std::move(row);
        }
    }
    CHECK(ds4_gpu_synchronize(), "finish sequential dense-prefix timing");
    const auto sequential_end = Clock::now();

    ds4_glm5_next_workspace *prompt_workspace =
        ds4_glm5_next_workspace_create_capacity(kPromptRows);
    ds4_gpu_tensor *prompt_ids = ds4_gpu_tensor_alloc(sizeof(prompt33));
    ds4_gpu_tensor *prompt_cur = ds4_gpu_tensor_alloc(
        (uint64_t)kPromptRows * kHcWidth * sizeof(float));
    ds4_gpu_tensor *prompt_out = ds4_gpu_tensor_alloc(
        (uint64_t)kPromptRows * kHcWidth * sizeof(float));
    CHECK(prompt_workspace && prompt_ids && prompt_cur && prompt_out &&
          ds4_gpu_tensor_write(prompt_ids, 0u, prompt33, sizeof(prompt33)) &&
          ds4_glm5_next_state_reset(&resident) &&
          ds4_glm5_next_embed_tokens(
              &exec, prompt_ids, kPromptRows, prompt_cur) &&
          ds4_gpu_synchronize(),
          "allocate and embed real 33-token dense-prefix batch");
    const auto batch_begin = Clock::now();
    for (uint32_t il = 0u; il < 3u; ++il) {
        CHECK(ds4_glm5_next_layer_forward_batch(
                  &exec, il, &resident, prompt_workspace,
                  prompt_cur, prompt_out, kPromptRows),
              "execute real 33-token dense-prefix batch layer");
        if (il + 1u < 3u) std::swap(prompt_cur, prompt_out);
    }
    CHECK(ds4_gpu_synchronize(), "finish batched dense-prefix timing");
    const auto batch_end = Clock::now();
    std::vector<float> prompt_batch((uint64_t)kPromptRows * kHcWidth);
    CHECK(ds4_gpu_tensor_read(prompt_out, 0u, prompt_batch.data(),
                              prompt_batch.size() * sizeof(float)) &&
          close_vectors(prompt_reference, prompt_batch,
                        "real 33-row prompt output"),
          "real prompt batch matches sequential same-GGUF oracle");
    for (uint32_t il = 0u; il < 3u; ++il) {
        CHECK(resident.kda.layer[il].token_count == kPromptRows,
              "real prompt batch advances every dense-prefix KDA row");
    }
    std::vector<float> prompt_batch_continue;
    CHECK(execute_runtime_prefix(17u, prompt_batch_continue) &&
          close_vectors(prompt_reference_continue, prompt_batch_continue,
                        "real-prompt post-batch continuation"),
          "decode continues from real 33-token batched KDA state");
    const double sequential_ms =
        std::chrono::duration<double, std::milli>(
            sequential_end - sequential_begin).count();
    const double batch_ms =
        std::chrono::duration<double, std::milli>(
            batch_end - batch_begin).count();
    std::fprintf(stderr,
                 "GLM5 dense-prefix timing rows=%u sequential_ms=%.3f "
                 "batch_ms=%.3f sequential_tps=%.3f batch_tps=%.3f "
                 "speedup=%.3fx\n",
                 kPromptRows, sequential_ms, batch_ms,
                 kPromptRows * 1000.0 / sequential_ms,
                 kPromptRows * 1000.0 / batch_ms,
                 sequential_ms / batch_ms);
    ds4_gpu_tensor_free(prompt_out);
    ds4_gpu_tensor_free(prompt_cur);
    ds4_gpu_tensor_free(prompt_ids);
    ds4_glm5_next_workspace_destroy(prompt_workspace);
    if (has_token2) {
        std::fprintf(stderr,
                     "PASS same-GGUF GLM5 dense prefix tokens=%u,%u block0=%016llx prefix=%016llx\n",
                     token, token2, (unsigned long long)block0_hash,
                     (unsigned long long)hash);
    } else {
        std::fprintf(stderr,
                     "PASS same-GGUF GLM5 dense prefix token=%u block0=%016llx prefix=%016llx\n",
                     token, (unsigned long long)block0_hash,
                     (unsigned long long)hash);
    }
    ds4_glm5_next_workspace_destroy(runtime_workspace);
    ds4_glm5_next_state_free(&resident);
    tensors.clear();
    ds4_glm5_kda_workspace_free(&workspace);
    ds4_glm5_kda_slot_free(&slot);
    ds4_gpu_cleanup();
    return true;
}

int main(void) { return run_test() ? 0 : 1; }
