#ifndef DS4_TESTS_GLM5_NEXT_REAL_OFFSETS_HPP
#define DS4_TESTS_GLM5_NEXT_REAL_OFFSETS_HPP

#include "ds4_glm5_next_runtime.h"
#include "tests/glm5_gguf_test.hpp"

#include <cstdio>
#include <initializer_list>

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

#endif
