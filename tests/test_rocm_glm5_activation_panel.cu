// Real KDA adapter/state comparison; selected GGUF, local TP-half composition.
#include "ds4_glm5_kda.h"
#include "ds4_gpu_mgpu.h"
#include "tests/glm5_gguf_test.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" int ds4_rocm_glm5_bf16_qkv_panel_tensor(
    ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const void *,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    const ds4_gpu_tensor *, uint64_t, ds4_gpu_tensor *);

static void check(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", what); std::exit(1); }
}
static void hip_check(hipError_t result, const char *what) {
    if (result != hipSuccess) {
        std::fprintf(stderr, "FAIL %s: %s\n", what, hipGetErrorString(result));
        std::exit(1);
    }
}
static ds4_gpu_tensor *allocate(uint64_t floats) {
    auto *t = ds4_gpu_tensor_alloc(floats * sizeof(float));
    check(t != nullptr, "allocate tensor");
    return t;
}
static void equal(const ds4_gpu_tensor *a, const ds4_gpu_tensor *b,
                  uint64_t floats, const char *what) {
    std::vector<float> x(floats), y(floats);
    check(ds4_gpu_tensor_read(a, 0, x.data(), floats * 4u) &&
          ds4_gpu_tensor_read(b, 0, y.data(), floats * 4u), "read comparison");
    for (uint64_t i = 0; i < floats; ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i]) ||
            std::memcmp(&x[i], &y[i], sizeof(float)) != 0) {
            std::fprintf(stderr, "FAIL %s index=%llu raw=%.9g panel=%.9g\n",
                         what, (unsigned long long)i, x[i], y[i]);
            std::exit(1);
        }
    }
}

static ds4_glm5_kda_weight_offsets bind(const Glm5TestGGUF &g) {
    ds4_glm5_kda_weight_offsets w = {};
    auto tensor = [&](const char *name, std::vector<uint64_t> dims,
                      uint32_t type, uint64_t &offset) {
        check(g.tensor(std::string("blk.0.") + name + ".weight", dims, type,
                       offset), name);
    };
    tensor("attn_norm", {4096}, 0, w.attn_norm);
    tensor("kda_q", {4096,8192}, 30, w.q);
    tensor("kda_k", {4096,8192}, 30, w.k);
    tensor("kda_v", {4096,8192}, 30, w.v);
    tensor("kda_output", {8192,4096}, 30, w.output);
    tensor("kda_q_conv", {4,1,8192}, 0, w.q_conv);
    tensor("kda_k_conv", {4,1,8192}, 0, w.k_conv);
    tensor("kda_v_conv", {4,1,8192}, 0, w.v_conv);
    tensor("kda_f_a", {4096,128}, 30, w.f_a);
    tensor("kda_f_b", {128,8192}, 30, w.f_b);
    tensor("kda_g_a", {4096,128}, 30, w.g_a);
    tensor("kda_g_b", {128,8192}, 30, w.g_b);
    tensor("kda_beta", {4096,64}, 30, w.beta);
    tensor("kda_o_norm", {128}, 0, w.o_norm);
    tensor("kda_dt_bias", {8192}, 0, w.dt_bias);
    tensor("kda_a_log", {64}, 0, w.a_log);
    w.q_type = w.k_type = w.v_type = w.output_type = 30u;
    w.f_a_type = w.f_b_type = w.g_a_type = w.g_b_type = w.beta_type = 30u;
    return w;
}

struct Arm {
    ds4_glm5_kda_slot slot = {};
    ds4_glm5_kda_workspace workspace = {};
    ds4_glm5_kda_layer_state half[2] = {};
    ds4_gpu_tensor *gated[2] = {}, *composed = nullptr, *output = nullptr;
    unsigned mode;
    explicit Arm(unsigned panel_mode) : mode(panel_mode) {
        setenv("DS4_ROCM_GLM5_BF16_QKV_ACTIVATION_PANEL", mode ? "1" : "0", 1);
        const ds4_glm5_layer_kind schedule = {0u, true};
        check(ds4_glm5_kda_slot_init(&slot, &schedule, 1, 1, nullptr) &&
              ds4_glm5_kda_workspace_init(&workspace, 256), "KDA owned workspace");
        check((workspace.qkv_activation_panel != nullptr) == (mode != 0u),
              "panel allocation matches selected arm");
        constexpr uint64_t history_bytes = UINT64_C(4096) * 3u * 4u;
        constexpr uint64_t state_bytes = UINT64_C(32) * 128u * 128u * 4u;
        for (unsigned rank = 0; rank < 2; ++rank) {
            half[rank].q_history = ds4_gpu_tensor_view(slot.layer[0].q_history, rank * history_bytes, history_bytes);
            half[rank].k_history = ds4_gpu_tensor_view(slot.layer[0].k_history, rank * history_bytes, history_bytes);
            half[rank].v_history = ds4_gpu_tensor_view(slot.layer[0].v_history, rank * history_bytes, history_bytes);
            half[rank].recurrent = ds4_gpu_tensor_view(slot.layer[0].recurrent, rank * state_bytes, state_bytes);
            half[rank].owner_slot = &slot;
            half[rank].valid = true;
            check(half[rank].q_history && half[rank].k_history &&
                  half[rank].v_history && half[rank].recurrent, "canonical half-state views");
            gated[rank] = allocate(UINT64_C(256) * 4096u);
        }
        composed = allocate(UINT64_C(256) * 8192u);
        output = allocate(UINT64_C(256) * 4096u);
    }
    ~Arm() {
        for (unsigned rank = 0; rank < 2; ++rank) {
            ds4_gpu_tensor_free(half[rank].q_history);
            ds4_gpu_tensor_free(half[rank].k_history);
            ds4_gpu_tensor_free(half[rank].v_history);
            ds4_gpu_tensor_free(half[rank].recurrent);
            ds4_gpu_tensor_free(gated[rank]);
        }
        ds4_gpu_tensor_free(composed);
        ds4_gpu_tensor_free(output);
        ds4_glm5_kda_workspace_free(&workspace);
        ds4_glm5_kda_slot_free(&slot);
    }
    void select() const {
        setenv("DS4_ROCM_GLM5_BF16_QKV_ACTIVATION_PANEL", mode ? "1" : "0", 1);
    }
    void reset() {
        check(ds4_glm5_kda_slot_reset(&slot), "reset state outside timing");
        for (auto &state : half) {
            state.pending_tokens = 0;
            state.token_count = 0;
            state.valid = true;
        }
    }
    void begin(unsigned rank, const Glm5TestGGUF &g,
               const ds4_glm5_kda_weight_offsets &w,
               const ds4_gpu_tensor *input, unsigned tokens, float eps) {
        select();
        check(ds4_glm5_kda_layer_begin(&half[rank], &workspace, &w,
              g.map, g.size, input, gated[rank], tokens, eps, rank * 32u, 32u),
              "real KDA layer begin");
    }
    void finish(const Glm5TestGGUF &g, const ds4_glm5_kda_weight_offsets &w,
                unsigned tokens) {
        select();
        check(ds4_glm5_kda_compose_head_halves(composed, gated[0], gated[1], tokens),
              "local canonical head composition");
        for (unsigned rank = 0; rank < 2; ++rank)
            check(ds4_glm5_kda_layer_finish(&half[rank], &w, g.map, g.size,
                  composed, output, tokens), "real KDA output projection and commit");
    }
};

static void workspace_equal(const Arm &a, const Arm &b, unsigned tokens) {
    const auto &x = a.workspace; const auto &y = b.workspace;
    equal(x.norm, y.norm, (uint64_t)tokens * 4096u, "RMS normalized input");
    equal(x.q, y.q, (uint64_t)tokens * 4096u, "Q after KDA begin");
    equal(x.k, y.k, (uint64_t)tokens * 4096u, "K after KDA begin");
    equal(x.v, y.v, (uint64_t)tokens * 4096u, "V after KDA begin");
    equal(x.forget, y.forget, (uint64_t)tokens * 4096u, "forget gate");
    equal(x.beta, y.beta, (uint64_t)tokens * 32u, "beta gate");
    equal(x.recurrent_out, y.recurrent_out, (uint64_t)tokens * 4096u, "recurrence output");
}

static void run(const Glm5TestGGUF &g, const ds4_glm5_kda_weight_offsets &w, float eps) {
    Arm raw(0), prepared(1);
    Arm *arms[] = {&raw, &prepared};
    auto *input = allocate(UINT64_C(256) * 4096u);
    std::vector<float> host(UINT64_C(256) * 4096u);
    for (unsigned pattern = 0; pattern < 3; ++pattern) {
        const unsigned tokens = pattern == 2 ? 1u : 256u;
        for (size_t i = 0; i < host.size(); ++i)
            host[i] = 0.23f * std::cos((double)((i + pattern * 137u) % 8191u) * 0.009) -
                      0.08f * std::sin((double)(i + pattern * 7u) * 0.004);
        check(ds4_gpu_tensor_write(input, 0, host.data(), host.size() * 4u), "changing layer input");
        for (unsigned rank = 0; rank < 2; ++rank) {
            for (auto *arm : arms) arm->begin(rank, g, w, input, tokens, eps);
            check(ds4_gpu_synchronize(), "synchronize layer comparison");
            workspace_equal(raw, prepared, tokens);
            equal(raw.gated[rank], prepared.gated[rank], (uint64_t)tokens * 4096u, "gated half output");
        }
        for (auto *arm : arms) arm->finish(g, w, tokens);
        check(ds4_gpu_synchronize(), "synchronize output/state comparison");
        equal(raw.output, prepared.output, (uint64_t)tokens * 4096u, "complete KDA output");
        equal(raw.slot.layer[0].q_history, prepared.slot.layer[0].q_history, 8192u * 3u, "Q history");
        equal(raw.slot.layer[0].k_history, prepared.slot.layer[0].k_history, 8192u * 3u, "K history");
        equal(raw.slot.layer[0].v_history, prepared.slot.layer[0].v_history, 8192u * 3u, "V history");
        equal(raw.slot.layer[0].recurrent, prepared.slot.layer[0].recurrent,
              64u * 128u * 128u, "complete recurrent state");
        const uint64_t expected = pattern == 0 ? 256u : pattern == 1 ? 512u : 513u;
        for (auto *arm : arms) for (const auto &half : arm->half)
            check(half.valid && half.pending_tokens == 0u && half.token_count == expected,
                  "successful continuation state");
        std::printf("PASS real layer0 tokens=%u pattern=%u cumulative=%llu both-rank-halves all-bytes-equal\n",
                    tokens, pattern, (unsigned long long)expected);
    }

    hipEvent_t start, end;
    hip_check(hipEventCreate(&start), "start event");
    hip_check(hipEventCreate(&end), "end event");
    std::vector<float> samples[2];
    for (unsigned sample = 0; sample < 9; ++sample) {
        for (unsigned order = 0; order < 2; ++order) {
            const unsigned arm = order ^ (sample & 1u);
            arms[arm]->reset();
            check(ds4_gpu_synchronize(), "finish reset outside timing");
            hip_check(hipEventRecord(start), "layer timing start");
            for (unsigned rank = 0; rank < 2; ++rank)
                arms[arm]->begin(rank, g, w, input, 256, eps);
            arms[arm]->finish(g, w, 256);
            hip_check(hipEventRecord(end), "layer timing end");
            hip_check(hipEventSynchronize(end), "layer timing wait");
            float ms = 0;
            hip_check(hipEventElapsedTime(&ms, start, end), "layer timing");
            samples[arm].push_back(ms);
            std::printf("sample complete-layer both-halves serial sample=%u arm=%u ms=%.6f\n", sample, arm, ms);
        }
    }
    for (unsigned arm = 0; arm < 2; ++arm) {
        std::sort(samples[arm].begin(), samples[arm].end());
        std::printf("timing complete-layer arm=%u median_ms=%.6f min_ms=%.6f max_ms=%.6f\n",
                    arm, samples[arm][4], samples[arm][0], samples[arm][8]);
    }
    // Reuse the actual F32 norm produced above, through the real validated
    // APIs and their registered GGUF ranges. Candidate timing includes pack.
    for (unsigned rank = 0; rank < 2; ++rank) {
        const uint64_t offset = (uint64_t)rank * 4096u * 4096u * 2u;
        auto project = [&](unsigned arm) {
            auto &ws = arms[arm]->workspace;
            const int ok = arm == 0u
                ? ds4_gpu_matmul_bf16_wmma_hilo_qkv_tensor(
                    ws.q, ws.k, ws.v, g.map, g.size,
                    w.q + offset, w.k + offset, w.v + offset,
                    4096, 4096, raw.workspace.norm, 256)
                : ds4_rocm_glm5_bf16_qkv_panel_tensor(
                    ws.q, ws.k, ws.v, g.map, g.size,
                    w.q + offset, w.k + offset, w.v + offset,
                    4096, 4096, raw.workspace.norm, 256, ws.qkv_activation_panel);
            check(ok == 1, "actual QKV API with RMS normalized input");
        };
        samples[0].clear(); samples[1].clear();
        project(0); project(1);
        for (unsigned sample = 0; sample < 9; ++sample) {
            for (unsigned order = 0; order < 2; ++order) {
                const unsigned arm = order ^ (sample & 1u);
                hip_check(hipEventRecord(start), "QKV timing start");
                for (unsigned repeat = 0; repeat < 3; ++repeat) project(arm);
                hip_check(hipEventRecord(end), "QKV timing end");
                hip_check(hipEventSynchronize(end), "QKV timing wait");
                float ms = 0;
                hip_check(hipEventElapsedTime(&ms, start, end), "QKV timing");
                samples[arm].push_back(ms / 3.0f);
                std::printf("sample actual-norm-QKV rank=%u sample=%u arm=%u ms=%.6f\n",
                            rank, sample, arm, ms / 3.0f);
            }
        }
        equal(raw.workspace.q, prepared.workspace.q, UINT64_C(256) * 4096u, "raw Q projection");
        equal(raw.workspace.k, prepared.workspace.k, UINT64_C(256) * 4096u, "raw K projection");
        equal(raw.workspace.v, prepared.workspace.v, UINT64_C(256) * 4096u, "raw V projection");
        for (unsigned arm = 0; arm < 2; ++arm) {
            std::sort(samples[arm].begin(), samples[arm].end());
            std::printf("timing actual-norm-QKV rank=%u arm=%u median_ms=%.6f min_ms=%.6f max_ms=%.6f\n",
                        rank, arm, samples[arm][4], samples[arm][0], samples[arm][8]);
        }
    }
    auto validate = [&](ds4_gpu_tensor *panel, unsigned tokens) {
        auto &ws = prepared.workspace;
        return ds4_rocm_glm5_bf16_qkv_panel_tensor(ws.q, ws.k, ws.v, g.map,
            g.size, w.q, w.k, w.v, 4096, 4096, raw.workspace.norm, tokens, panel);
    };
    auto *panel = prepared.workspace.qkv_activation_panel;
    check(validate(nullptr, 256) == 0, "missing scratch fails closed");
    check(validate(panel, 1) == -1 && validate(panel, 512) == -1, "off-shape panel decline");
    ds4_gpu_tensor bad = *panel;
    bad.bytes -= 4u;
    check(validate(&bad, 256) == 0, "undersized scratch rejection");
    bad = *panel; bad.device_id = -1;
    check(validate(&bad, 256) == 0, "scratch device mismatch rejection");
    for (auto *alias : {raw.workspace.norm, prepared.workspace.q,
                       prepared.workspace.k, prepared.workspace.v}) {
        bad = *panel; bad.ptr = alias->ptr;
        check(validate(&bad, 256) == 0, "scratch input/output alias rejection");
    }
    std::puts("PASS real QKV API scratch/shape/device/alias guards");
    hip_check(hipEventDestroy(start), "destroy start event");
    hip_check(hipEventDestroy(end), "destroy end event");
    ds4_gpu_tensor_free(input);
}

int main() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    check(model && *model, "DS4_GLM5_MODEL required");
    Glm5TestGGUF g;
    check(g.open_file(model), "selected GGUF");
    // bind() validates the complete KDA layer's shapes and quantization.
    // Metadata-only differences must not reject another compatible artifact.
    float eps = 0;
    check(g.metadata("glm5-next.attention.layer_norm_rms_epsilon", eps) && eps == 1.0e-5f,
          "GGUF RMS epsilon");
    const auto w = bind(g);
    unsetenv("DS4_ROCM_GLM5_BF16_KDA_SIX_PREFILL");
    unsetenv("DS4_ROCM_GLM5_BF16_KDA_SIX_MULTIPTR");
    unsetenv("DS4_ROCM_GLM5_BF16_QKV_SHARED_A_PREFILL");
    unsetenv("DS4_ROCM_DISABLE_BF16_BATCH_TOKTILE");
    setenv("DS4_ROCM_GLM5_BF16_WMMA_HILO", "1", 1);
    setenv("DS4_ROCM_GLM5_BF16_WMMA_QKV_FUSED", "1", 1);
    setenv("DS4_ROCM_GLM5_BF16_QKV_DECODE_MULTIPTR", "1", 1);
    /* Exercise the new producer only for the panel arm; Arm(0) ignores it.
     * The value is externally selectable so the frozen probe can compare the
     * incumbent two-pass producer and this fused producer in separate runs. */
    if (!std::getenv("DS4_ROCM_GLM5_BF16_QKV_ACTIVATION_PANEL_FUSED_NORM"))
        setenv("DS4_ROCM_GLM5_BF16_QKV_ACTIVATION_PANEL_FUSED_NORM", "1", 1);
    ds4_gpu_config config = {};
    config.n_gpus = 1; config.device_indices[0] = 0;
    check(ds4_gpu_init_multi(&config), "initialize ROCm");
    check(ds4_gpu_set_model_fd_for_map(g.fd, g.map) &&
          ds4_gpu_set_model_map(g.map, g.size), "register original GGUF");
    run(g, w, eps);
    ds4_gpu_cleanup();
    std::puts("PASS real KDA activation panel integration");
}
