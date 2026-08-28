#ifndef DS4_TESTS_GLM5_NEXT_REAL_OFFSETS_HPP
#define DS4_TESTS_GLM5_NEXT_REAL_OFFSETS_HPP

#include "ds4_glm5_next_runtime.h"
#include "tests/glm5_gguf_test.hpp"

#include <cstdio>
#include <initializer_list>
#include <algorithm>
#include <limits>
#include <string>
#include <vector>

static bool glm5_next_layer_tensor(
        const Glm5TestGGUF &g, uint32_t layer, const char *suffix,
        std::initializer_list<uint64_t> dims, uint32_t type,
        uint64_t &offset) {
    char name[112];
    const int n = std::snprintf(name, sizeof(name), "blk.%u.%s", layer, suffix);
    return n > 0 && (size_t)n < sizeof(name) &&
           g.tensor(name, dims, type, offset);
}

static bool glm5_next_bind_real_offsets(
        const Glm5TestGGUF &g, ds4_glm5_next_model_offsets &model) {
    model = {};
    model.layer_count = DS4_GLM5_NEXT_LAYER_COUNT;
    model.trunk_count = DS4_GLM5_NEXT_TRUNK_COUNT;
    model.nextn_count = 1u;
    if (!g.tensor("token_embd.weight", {4096u, 154880u}, 30u,
                  model.token_embd) ||
        !g.tensor("output_norm.weight", {4096u}, 0u, model.output_norm) ||
        !g.tensor("output.weight", {4096u, 154880u}, 30u, model.output) ||
        !glm5_next_layer_tensor(g, 45u, "nextn.eh_proj.weight",
                               {8192u, 4096u}, 30u,
                               model.nextn_eh_proj)) return false;

    for (uint32_t il = 0u; il < model.layer_count; ++il) {
        ds4_glm5_next_layer_offsets &layer = model.layer[il];
        layer.layer = il;
        layer.is_trunk = il < model.trunk_count;
        layer.attention = (il == model.trunk_count || (il & 3u) == 3u) ?
            DS4_GLM5_NEXT_ATTN_MLA : DS4_GLM5_NEXT_ATTN_KDA;
        layer.ffn = il < DS4_GLM5_NEXT_LEADING_DENSE ?
            DS4_GLM5_NEXT_FFN_DENSE : DS4_GLM5_NEXT_FFN_ROUTED;
        if (!glm5_next_layer_tensor(g, il, "attn_norm.weight", {4096u},
                                    0u, layer.attn_norm) ||
            !glm5_next_layer_tensor(g, il, "ffn_norm.weight", {4096u},
                                    0u, layer.ffn_norm)) return false;
        if (layer.attention == DS4_GLM5_NEXT_ATTN_KDA) {
            layer.kda.attn_norm = layer.attn_norm;
            if (!glm5_next_layer_tensor(g, il, "kda_q.weight",
                                        {4096u, 8192u}, 30u, layer.kda.q) ||
                !glm5_next_layer_tensor(g, il, "kda_k.weight",
                                        {4096u, 8192u}, 30u, layer.kda.k) ||
                !glm5_next_layer_tensor(g, il, "kda_v.weight",
                                        {4096u, 8192u}, 30u, layer.kda.v) ||
                !glm5_next_layer_tensor(g, il, "kda_output.weight",
                                        {8192u, 4096u}, 30u, layer.kda.output) ||
                !glm5_next_layer_tensor(g, il, "kda_q_conv.weight",
                                        {4u, 1u, 8192u}, 0u, layer.kda.q_conv) ||
                !glm5_next_layer_tensor(g, il, "kda_k_conv.weight",
                                        {4u, 1u, 8192u}, 0u, layer.kda.k_conv) ||
                !glm5_next_layer_tensor(g, il, "kda_v_conv.weight",
                                        {4u, 1u, 8192u}, 0u, layer.kda.v_conv) ||
                !glm5_next_layer_tensor(g, il, "kda_f_a.weight",
                                        {4096u, 128u}, 30u, layer.kda.f_a) ||
                !glm5_next_layer_tensor(g, il, "kda_f_b.weight",
                                        {128u, 8192u}, 30u, layer.kda.f_b) ||
                !glm5_next_layer_tensor(g, il, "kda_g_a.weight",
                                        {4096u, 128u}, 30u, layer.kda.g_a) ||
                !glm5_next_layer_tensor(g, il, "kda_g_b.weight",
                                        {128u, 8192u}, 30u, layer.kda.g_b) ||
                !glm5_next_layer_tensor(g, il, "kda_beta.weight",
                                        {4096u, 64u}, 30u, layer.kda.beta) ||
                !glm5_next_layer_tensor(g, il, "kda_o_norm.weight",
                                        {128u}, 0u, layer.kda.o_norm) ||
                !glm5_next_layer_tensor(g, il, "kda_dt_bias.weight",
                                        {8192u}, 0u, layer.kda.dt_bias) ||
                !glm5_next_layer_tensor(g, il, "kda_a_log.weight",
                                        {64u}, 0u, layer.kda.a_log)) return false;
        } else {
            ds4_glm5_next_mla_offsets &m = layer.mla;
            if (!glm5_next_layer_tensor(g, il, "attn_q_a.weight",
                                        {4096u, 1536u}, 8u, m.q_a) ||
                !glm5_next_layer_tensor(g, il, "attn_q_a_norm.weight",
                                        {1536u}, 0u, m.q_a_norm) ||
                !glm5_next_layer_tensor(g, il, "attn_q_b.weight",
                                        {1536u, 16384u}, 8u, m.q_b) ||
                !glm5_next_layer_tensor(g, il, "attn_kv_a_mqa.weight",
                                        {4096u, 512u}, 8u, m.kv_a_mqa) ||
                !glm5_next_layer_tensor(g, il, "attn_kv_a_norm.weight",
                                        {512u}, 0u, m.kv_a_norm) ||
                !glm5_next_layer_tensor(g, il, "attn_k_b.weight",
                                        {256u, 512u, 64u}, 8u, m.k_b) ||
                !glm5_next_layer_tensor(g, il, "attn_v_b.weight",
                                        {512u, 256u, 64u}, 8u, m.v_b) ||
                !glm5_next_layer_tensor(g, il, "attn_output.weight",
                                        {16384u, 4096u}, 8u, m.output) ||
                !glm5_next_layer_tensor(g, il, "indexer.attn_q_b.weight",
                                        {1536u, 4096u}, 30u, m.index_q_b) ||
                !glm5_next_layer_tensor(g, il, "indexer.attn_k.weight",
                                        {4096u, 128u}, 30u, m.index_k) ||
                !glm5_next_layer_tensor(g, il, "indexer.proj.weight",
                                        {4096u, 32u}, 30u, m.index_proj) ||
                !glm5_next_layer_tensor(g, il, "indexer.pool_ape.weight",
                                        {128u, 4u}, 30u, m.index_pool_ape) ||
                !glm5_next_layer_tensor(g, il, "indexer.pool_gate.weight",
                                        {4096u, 128u}, 30u, m.index_pool_gate) ||
                !glm5_next_layer_tensor(g, il, "indexer.k_norm.weight",
                                        {128u}, 0u, m.index_k_norm) ||
                !glm5_next_layer_tensor(g, il, "indexer.k_norm.bias",
                                        {128u}, 0u, m.index_k_norm_b)) return false;
        }

        ds4_glm5_next_ffn_offsets &f = layer.ffn_weight;
        if (layer.ffn == DS4_GLM5_NEXT_FFN_DENSE) {
            if (!glm5_next_layer_tensor(g, il, "ffn_gate.weight",
                                        {4096u, 12288u}, 8u, f.gate) ||
                !glm5_next_layer_tensor(g, il, "ffn_up.weight",
                                        {4096u, 12288u}, 8u, f.up) ||
                !glm5_next_layer_tensor(g, il, "ffn_down.weight",
                                        {12288u, 4096u}, 8u, f.down)) return false;
        } else {
            if (!glm5_next_layer_tensor(g, il, "ffn_gate_exps.weight",
                                        {4096u, 2048u, 288u}, 12u,
                                        f.gate_exps) ||
                !glm5_next_layer_tensor(g, il, "ffn_up_exps.weight",
                                        {4096u, 2048u, 288u}, 12u,
                                        f.up_exps) ||
                !glm5_next_layer_tensor(g, il, "ffn_down_exps.weight",
                                        {2048u, 4096u, 288u}, 12u,
                                        f.down_exps) ||
                !glm5_next_layer_tensor(g, il, "ffn_gate_inp.weight",
                                        {4096u, 288u}, 0u, f.gate_inp) ||
                !glm5_next_layer_tensor(g, il, "exp_probs_b.bias",
                                        {288u}, 0u, f.exp_probs_b) ||
                !glm5_next_layer_tensor(g, il, "ffn_gate_shexp.weight",
                                        {4096u, 2048u}, 8u, f.gate_shexp) ||
                !glm5_next_layer_tensor(g, il, "ffn_up_shexp.weight",
                                        {4096u, 2048u}, 8u, f.up_shexp) ||
                !glm5_next_layer_tensor(g, il, "ffn_down_shexp.weight",
                                        {2048u, 4096u}, 8u, f.down_shexp)) return false;
        }
        if (layer.is_trunk) {
            if (!glm5_next_layer_tensor(g, il, "hc_attn_fn.weight",
                                        {16384u, 24u}, 30u, layer.hc.attn_fn) ||
                !glm5_next_layer_tensor(g, il, "hc_ffn_fn.weight",
                                        {16384u, 24u}, 30u, layer.hc.ffn_fn) ||
                !glm5_next_layer_tensor(g, il, "hc_attn_base.weight",
                                        {24u}, 0u, layer.hc.attn_base) ||
                !glm5_next_layer_tensor(g, il, "hc_ffn_base.weight",
                                        {24u}, 0u, layer.hc.ffn_base) ||
                !glm5_next_layer_tensor(g, il, "hc_attn_scale.weight",
                                        {3u}, 0u, layer.hc.attn_scale) ||
                !glm5_next_layer_tensor(g, il, "hc_ffn_scale.weight",
                                        {3u}, 0u, layer.hc.ffn_scale)) return false;
        }
    }
    return ds4_glm5_next_model_offsets_validate(&model) != 0;
}

struct Glm5NextKShardPlan {
    std::vector<uint64_t> dense_offsets;
    std::vector<uint64_t> dense_sizes;
    std::vector<ds4_gpu_q4k_kshard_layer> layers;
    uint64_t dense_max_tensor_bytes = 0u;
    uint64_t dense_total_bytes = 0u;
    uint64_t packed_total_bytes = 0u;
};

static bool glm5_next_test_u64_mul(uint64_t a, uint64_t b, uint64_t &out) {
    if (a != 0u && b > UINT64_MAX / a) return false;
    out = a * b;
    return true;
}

static bool glm5_next_test_u64_add(uint64_t a, uint64_t b, uint64_t &out) {
    if (b > UINT64_MAX - a) return false;
    out = a + b;
    return true;
}

static bool glm5_next_test_tensor_bytes(const Glm5TestTensorInfo &tensor,
                                        uint64_t &bytes) {
    uint64_t elements = 1u;
    for (uint64_t dim : tensor.dims) {
        if (dim == 0u || elements > UINT64_MAX / dim) return false;
        elements *= dim;
    }
    uint64_t block_elements = 0u, block_bytes = 0u;
    switch (tensor.type) {
    case 0u:  block_elements = 1u;   block_bytes = 4u;   break; // F32
    case 8u:  block_elements = 32u;  block_bytes = 34u;  break; // Q8_0
    case 12u: block_elements = 256u; block_bytes = 144u; break; // Q4_K
    case 30u: block_elements = 1u;   block_bytes = 2u;   break; // BF16
    default: return false;
    }
    if ((elements % block_elements) != 0u ||
        elements / block_elements > UINT64_MAX / block_bytes) return false;
    bytes = elements / block_elements * block_bytes;
    return bytes != 0u;
}

static bool glm5_next_test_trunk_tensor(const std::string &name) {
    if (name == "token_embd.weight" || name == "output_norm.weight" ||
        name == "output.weight") return true;
    unsigned layer = 0u;
    int consumed = 0;
    return std::sscanf(name.c_str(), "blk.%u.%n", &layer, &consumed) == 1 &&
           consumed > 0 && layer < DS4_GLM5_NEXT_TRUNK_COUNT;
}

static bool glm5_next_test_routed_tensor(const std::string &name) {
    return name.find(".ffn_gate_exps.weight") != std::string::npos ||
           name.find(".ffn_up_exps.weight") != std::string::npos ||
           name.find(".ffn_down_exps.weight") != std::string::npos;
}

static bool glm5_next_build_kshard_plan(
        const Glm5TestGGUF &g, const ds4_glm5_next_model_offsets &model,
        Glm5NextKShardPlan &plan) {
    plan = {};
    struct Span { uint64_t off, end, tensor_bytes; };
    std::vector<Span> spans;
    for (const auto &entry : g.tensors) {
        if (!glm5_next_test_trunk_tensor(entry.first) ||
            glm5_next_test_routed_tensor(entry.first)) continue;
        uint64_t bytes = 0u;
        if (!glm5_next_test_tensor_bytes(entry.second, bytes) ||
            entry.second.relative_offset > UINT64_MAX - g.data_start) {
            return false;
        }
        const uint64_t off = g.data_start + entry.second.relative_offset;
        if (off > g.size || bytes > g.size - off) return false;
        spans.push_back({off, off + bytes, bytes});
    }
    if (spans.empty()) return false;
    std::sort(spans.begin(), spans.end(), [](const Span &a, const Span &b) {
        return a.off < b.off || (a.off == b.off && a.end < b.end);
    });
    for (const Span &span : spans) {
        if (!plan.dense_offsets.empty()) {
            const size_t last = plan.dense_offsets.size() - 1u;
            const uint64_t last_end =
                plan.dense_offsets[last] + plan.dense_sizes[last];
            if (span.off <= last_end) {
                if (span.end > last_end)
                    plan.dense_sizes[last] = span.end - plan.dense_offsets[last];
                continue;
            }
        }
        plan.dense_offsets.push_back(span.off);
        plan.dense_sizes.push_back(span.end - span.off);
    }
    for (uint64_t bytes : plan.dense_sizes) {
        if (plan.dense_total_bytes > UINT64_MAX - bytes) return false;
        plan.dense_total_bytes += bytes;
        plan.dense_max_tensor_bytes =
            std::max(plan.dense_max_tensor_bytes, bytes);
    }

    plan.layers.reserve(DS4_GLM5_NEXT_TRUNK_COUNT -
                        DS4_GLM5_NEXT_LEADING_DENSE);
    constexpr uint64_t q4_block_bytes = 144u;
    constexpr uint64_t gate_row_bytes = 16u * q4_block_bytes;
    constexpr uint64_t down_row_bytes = 8u * q4_block_bytes;
    for (uint32_t il = DS4_GLM5_NEXT_LEADING_DENSE;
         il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il) {
        const ds4_glm5_next_ffn_offsets &ffn = model.layer[il].ffn_weight;
        plan.layers.push_back({
            ffn.gate_exps, ffn.up_exps, ffn.down_exps, 288u,
            4096u, 2048u, 4096u,
            gate_row_bytes, gate_row_bytes, down_row_bytes,
        });
        uint64_t gate_expert_bytes = 0u, down_expert_bytes = 0u;
        uint64_t two_gate_bytes = 0u, packed_expert_bytes = 0u;
        uint64_t packed_layer_bytes = 0u;
        if (!glm5_next_test_u64_mul(1024u, gate_row_bytes,
                                    gate_expert_bytes) ||
            !glm5_next_test_u64_mul(4096u, down_row_bytes / 2u,
                                    down_expert_bytes) ||
            !glm5_next_test_u64_mul(2u, gate_expert_bytes,
                                    two_gate_bytes) ||
            !glm5_next_test_u64_add(two_gate_bytes, down_expert_bytes,
                                    packed_expert_bytes) ||
            !glm5_next_test_u64_mul(288u, packed_expert_bytes,
                                    packed_layer_bytes) ||
            !glm5_next_test_u64_add(plan.packed_total_bytes,
                                    packed_layer_bytes,
                                    plan.packed_total_bytes)) {
            return false;
        }
    }
    return plan.layers.size() ==
               DS4_GLM5_NEXT_TRUNK_COUNT - DS4_GLM5_NEXT_LEADING_DENSE &&
           plan.dense_offsets.size() == plan.dense_sizes.size();
}

#endif
