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
constexpr uint64_t kExpectedBlock0FNV = UINT64_C(0xf055102b4604cdf3);
constexpr uint64_t kExpectedPrefixFNV = UINT64_C(0xb6c2590232ac924b);

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
             std::vector<float> &result, const char *trace_prefix = nullptr) {
    CHECK(ds4_gpu_rms_norm_plain_rows_tensor(
              t.flat, t.cur, kHcWidth, 1u, 1.0e-5f) &&
          ds4_gpu_matmul_bf16_tensor(
              t.mix, g.map, g.size, w.attn_hc_fn, kHcWidth, kMix,
              t.flat, 1u) &&
          ds4_gpu_hc_split_weighted_sum_tensor(
              t.collapsed, t.split, t.mix, t.cur,
              g.map, g.size, w.attn_hc_scale, w.attn_hc_base,
              kWidth, kHc, 20u, 1.0e-6f) &&
          ds4_glm5_kda_layer_forward(
              state, workspace, &w.kda, g.map, g.size,
              t.collapsed, t.attn, 1u) &&
          ds4_gpu_hc_expand_split_tensor(
              t.after_attn, t.attn, t.cur, t.split, kWidth, kHc) &&
          ds4_gpu_rms_norm_plain_rows_tensor(
              t.ffn_flat, t.after_attn, kHcWidth, 1u, 1.0e-5f) &&
          ds4_gpu_matmul_bf16_tensor(
              t.ffn_mix, g.map, g.size, w.ffn_hc_fn, kHcWidth, kMix,
              t.ffn_flat, 1u) &&
          ds4_gpu_hc_split_weighted_sum_norm_tensor(
              t.ffn_collapsed, t.ffn_hidden, t.ffn_split, t.ffn_mix,
              t.after_attn, g.map, g.size, w.ffn_hc_scale, w.ffn_hc_base,
              w.ffn_norm, kWidth, kHc, 20u, 1.0e-6f, 1.0e-5f) &&
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
    CHECK(gguf.open_file(model), "open real-GGUF model");
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
    const auto execute_prefix = [&](std::vector<float> &result,
                                    bool capture_block0) {
        if (!ds4_gpu_embed_token_hc_bf16_tensor(
                tensors.cur, gguf.map, gguf.size, weights[0].embedding,
                kVocab, 42u, kWidth, kHc)) return false;
        for (uint32_t il = 0u; il < 3u; ++il) {
            std::vector<float> layer_result;
            if (!execute(gguf, weights[il], &slot.layer[il], &workspace,
                         tensors, layer_result,
                         capture_block0 && il == 0u ?
                             std::getenv("DS4_GLM5_DENSE_TRACE_PREFIX") :
                             nullptr)) return false;
            if (capture_block0 && il == 0u) block0 = layer_result;
            if (il + 1u < 3u) std::swap(tensors.cur, tensors.out);
            else result = std::move(layer_result);
        }
        return true;
    };
    CHECK(execute_prefix(first, true), "execute complete dense prefix");
    for (uint32_t il = 0u; il < 3u; ++il) {
        CHECK(slot.layer[il].token_count == 1u,
              "dense-prefix KDA layer advances one token");
    }
    CHECK(ds4_glm5_kda_slot_reset(&slot) && execute_prefix(second, false),
          "repeat complete dense-prefix execution");
    CHECK(first == second, "complete dense prefix is deterministic after reset");
    for (uint32_t il = 0u; il < 3u; ++il) {
        CHECK(slot.layer[il].token_count == 1u,
              "reset dense-prefix KDA layer advances one token");
    }
    CHECK(ds4_tp_test_get_exchange_calls() == 0u,
          "replicated dense block invokes no TP exchange");
    const uint64_t block0_hash = fnv64(block0);
    CHECK(block0_hash == kExpectedBlock0FNV,
          "complete block-0 output matches the pinned same-GGUF fingerprint");
    const uint64_t hash = fnv64(first);
    CHECK(hash == kExpectedPrefixFNV,
          "complete dense-prefix output matches the pinned same-GGUF fingerprint");

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
    CHECK(tiny_output && ds4_glm5_next_embed_token(&exec, 42u, tensors.cur) &&
          !ds4_glm5_next_layer_forward(
              &exec, 0u, &resident, runtime_workspace,
              tensors.cur, tiny_output) &&
          resident.valid && resident.kda.layer[0].token_count == 0u,
          "undersized output fails before resident state mutation");
    ds4_gpu_tensor_free(tiny_output);
    const auto execute_runtime_prefix = [&](std::vector<float> &result) {
        if (!ds4_glm5_next_embed_token(&exec, 42u, tensors.cur)) return false;
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
    CHECK(execute_runtime_prefix(runtime_first) && runtime_first == first,
          "production executor matches the independent dense-prefix harness");
    CHECK(ds4_glm5_next_state_reset(&resident) &&
          execute_runtime_prefix(runtime_second) &&
          runtime_second == runtime_first,
          "production resident executor is deterministic after atomic reset");
    for (uint32_t il = 0u; il < 3u; ++il) {
        CHECK(resident.kda.layer[il].token_count == 1u,
              "production dense-prefix KDA state advances once");
    }
    CHECK(!ds4_glm5_next_layer_forward(
              &exec, 3u, &resident, runtime_workspace,
              tensors.cur, tensors.out) && resident.valid &&
          resident.mla[3].token_count == 0u,
          "unimplemented MLA+routed kind fails before resident mutation");
    std::fprintf(stderr,
                 "PASS same-GGUF GLM5 dense prefix block0=%016llx prefix=%016llx\n",
                 (unsigned long long)block0_hash,
                 (unsigned long long)hash);
    ds4_glm5_next_workspace_destroy(runtime_workspace);
    ds4_glm5_next_state_free(&resident);
    tensors.clear();
    ds4_glm5_kda_workspace_free(&workspace);
    ds4_glm5_kda_slot_free(&slot);
    ds4_gpu_cleanup();
    return true;
}

int main(void) { return run_test() ? 0 : 1; }
