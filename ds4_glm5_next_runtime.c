#include "ds4_glm5_next_runtime.h"
#include "ds4_tp.h"

#include <string.h>

bool ds4_glm5_next_layer_is_mla(uint32_t layer) {
    return layer == DS4_GLM5_NEXT_TRUNK_COUNT || (layer & 3u) == 3u;
}

int ds4_glm5_next_mla_dense_selection_visible_for_topk(
        uint64_t token_count, uint32_t capacity_tokens, uint32_t top_k,
        uint32_t *visible) {
    if (!visible || capacity_tokens == 0u || top_k == 0u ||
        token_count >= capacity_tokens || token_count >= top_k) return 0;
    *visible = (uint32_t)token_count + 1u;
    return 1;
}

int ds4_glm5_next_mla_dense_selection_visible(
        uint64_t token_count, uint32_t capacity_tokens, uint32_t *visible) {
    return ds4_glm5_next_mla_dense_selection_visible_for_topk(
        token_count, capacity_tokens, DS4_GLM5_NEXT_INDEX_TOP_K, visible);
}

int ds4_glm5_next_build_tp_gate_mask(
        uint64_t mask[DS4_GLM5_NEXT_TP_GATE_MASK_WORDS],
        uint32_t *gate_count,
        uint32_t runtime_features) {
    if (!mask || !gate_count) return 0;
    memset(mask, 0,
           sizeof(uint64_t) * DS4_GLM5_NEXT_TP_GATE_MASK_WORDS);
    uint32_t count = 0;
    const bool kda_tp =
        (runtime_features & DS4_TP_FEATURE_GLM5_KDA_TP) != 0u;
    for (uint32_t il = 0u; il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il) {
        const bool routed = il >= DS4_GLM5_NEXT_LEADING_DENSE;
        if ((kda_tp && !ds4_glm5_next_layer_is_mla(il)) ||
            (routed && ds4_glm5_next_layer_is_mla(il))) {
            const uint32_t slot = il * 2u;
            mask[slot / 64u] |= UINT64_C(1) << (slot % 64u);
            count++;
        }
        if (routed) {
            const uint32_t slot = il * 2u + 1u;
            mask[slot / 64u] |= UINT64_C(1) << (slot % 64u);
            count++;
        }
    }
    *gate_count = count;
    return count == (kda_tp ? 87u : 53u);
}

static int kda_complete(const ds4_glm5_kda_weight_offsets *w) {
    return w->attn_norm && w->q && w->k && w->v && w->output &&
           w->q_conv && w->k_conv && w->v_conv && w->f_a && w->f_b &&
           w->g_a && w->g_b && w->beta && w->o_norm && w->dt_bias &&
           w->a_log;
}

static int kda_empty(const ds4_glm5_kda_weight_offsets *w) {
    return !w->attn_norm && !w->q && !w->k && !w->v && !w->output &&
           !w->q_conv && !w->k_conv && !w->v_conv && !w->f_a && !w->f_b &&
           !w->g_a && !w->g_b && !w->beta && !w->o_norm && !w->dt_bias &&
           !w->a_log;
}

static int mla_complete(const ds4_glm5_next_mla_offsets *w) {
    return w->q_a && w->q_a_norm && w->q_b && w->kv_a_mqa &&
           w->kv_a_norm && w->k_b && w->v_b && w->output && w->index_q_b &&
           w->index_k && w->index_proj && w->index_pool_ape &&
           w->index_pool_gate && w->index_k_norm && w->index_k_norm_b;
}

static int mla_empty(const ds4_glm5_next_mla_offsets *w) {
    return !w->q_a && !w->q_a_norm && !w->q_b && !w->kv_a_mqa &&
           !w->kv_a_norm && !w->k_b && !w->v_b && !w->output &&
           !w->index_q_b && !w->index_k && !w->index_proj &&
           !w->index_pool_ape && !w->index_pool_gate && !w->index_k_norm &&
           !w->index_k_norm_b;
}

static int dense_ffn_complete(const ds4_glm5_next_ffn_offsets *w) {
    return w->gate && w->up && w->down && !w->gate_exps && !w->up_exps &&
           !w->down_exps && !w->gate_inp && !w->exp_probs_b &&
           !w->gate_shexp && !w->up_shexp && !w->down_shexp;
}

static int routed_ffn_complete(const ds4_glm5_next_ffn_offsets *w) {
    return !w->gate && !w->up && !w->down && w->gate_exps && w->up_exps &&
           w->down_exps && w->gate_inp && w->exp_probs_b &&
           w->gate_shexp && w->up_shexp && w->down_shexp;
}

static int hc_complete(const ds4_glm5_next_hc_offsets *w) {
    return w->attn_fn && w->ffn_fn && w->attn_base && w->ffn_base &&
           w->attn_scale && w->ffn_scale;
}

static int hc_empty(const ds4_glm5_next_hc_offsets *w) {
    return !w->attn_fn && !w->ffn_fn && !w->attn_base && !w->ffn_base &&
           !w->attn_scale && !w->ffn_scale;
}

int ds4_glm5_next_model_offsets_validate(
        const ds4_glm5_next_model_offsets *model) {
    if (!model || model->layer_count != DS4_GLM5_NEXT_LAYER_COUNT ||
        model->trunk_count != DS4_GLM5_NEXT_TRUNK_COUNT ||
        model->nextn_count != 1u || !model->token_embd ||
        !model->output_norm || !model->output || !model->nextn_eh_proj ||
        !(model->rms_norm_eps > 0.0f) || model->rms_norm_eps > 1.0f ||
        !(model->hc_eps > 0.0f) || model->hc_eps > 1.0f) {
        return 0;
    }
    for (uint32_t il = 0u; il < model->layer_count; ++il) {
        const ds4_glm5_next_layer_offsets *layer = &model->layer[il];
        const bool trunk = il < model->trunk_count;
        if (layer->layer != il || layer->is_trunk != trunk ||
            !layer->attn_norm || !layer->ffn_norm) return 0;
        const bool want_mla = ds4_glm5_next_layer_is_mla(il);
        if (layer->attention != (want_mla ? DS4_GLM5_NEXT_ATTN_MLA :
                                           DS4_GLM5_NEXT_ATTN_KDA)) return 0;
        if (want_mla ? (!mla_complete(&layer->mla) || !kda_empty(&layer->kda))
                     : (!kda_complete(&layer->kda) || !mla_empty(&layer->mla)))
            return 0;
        const ds4_glm5_next_ffn_kind want_ffn =
            il < DS4_GLM5_NEXT_LEADING_DENSE ?
                DS4_GLM5_NEXT_FFN_DENSE : DS4_GLM5_NEXT_FFN_ROUTED;
        if (layer->ffn != want_ffn) return 0;
        if (want_ffn == DS4_GLM5_NEXT_FFN_DENSE) {
            if (!dense_ffn_complete(&layer->ffn_weight)) return 0;
        } else {
            if (!routed_ffn_complete(&layer->ffn_weight)) return 0;
        }
        if (trunk) {
            if (!hc_complete(&layer->hc)) return 0;
        } else if (!hc_empty(&layer->hc)) {
            return 0;
        }
    }
    return 1;
}
