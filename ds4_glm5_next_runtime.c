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

int ds4_glm5_next_mla_sparse_selection_plan(
        uint64_t token_count, uint32_t capacity_tokens, uint32_t top_k,
        uint32_t pool_size, uint32_t *visible, uint32_t *n_pools,
        uint32_t *selected_pools, uint32_t *selected_tokens) {
    if (!visible || !n_pools || !selected_pools || !selected_tokens ||
        capacity_tokens == 0u || top_k == 0u || pool_size == 0u ||
        top_k % pool_size != 0u || token_count >= capacity_tokens ||
        token_count < top_k || token_count >= UINT32_MAX) return 0;
    const uint32_t rows = (uint32_t)token_count + 1u;
    const uint32_t pools = rows / pool_size;
    const uint32_t pool_budget = top_k / pool_size;
    const uint32_t selected = pools < pool_budget ? pools : pool_budget;
    if (selected > (UINT32_MAX - rows % pool_size) / pool_size) return 0;
    *visible = rows;
    *n_pools = pools;
    *selected_pools = selected;
    *selected_tokens = selected * pool_size + rows % pool_size;
    return 1;
}

uint32_t ds4_glm5_next_prefill_chunk(
        uint32_t position, uint32_t remaining, uint32_t requested_batch) {
    if (remaining == 0u) return 0u;
    if (requested_batch < 2u ||
        position >= DS4_GLM5_NEXT_INDEX_TOP_K) return 1u;
    uint32_t chunk = remaining < requested_batch ? remaining : requested_batch;
    const uint32_t dense_remaining = DS4_GLM5_NEXT_INDEX_TOP_K - position;
    if (chunk > dense_remaining) chunk = dense_remaining;
    return chunk ? chunk : 1u;
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
    const int q_type = w->q_type == 0u || w->q_type == 12u ||
                       w->q_type == 30u;
    const int k_type = w->k_type == 0u || w->k_type == 12u ||
                       w->k_type == 30u;
    const int v_type = w->v_type == 0u || w->v_type == 8u ||
                       w->v_type == 30u;
    const int output_type = w->output_type == 0u || w->output_type == 8u ||
                            w->output_type == 30u;
    const int low_types =
        (w->f_a_type == 0u || w->f_a_type == 8u || w->f_a_type == 30u) &&
        (w->f_b_type == 0u || w->f_b_type == 8u || w->f_b_type == 30u) &&
        (w->g_a_type == 0u || w->g_a_type == 8u || w->g_a_type == 30u) &&
        (w->g_b_type == 0u || w->g_b_type == 8u || w->g_b_type == 30u) &&
        (w->beta_type == 0u || w->beta_type == 8u ||
         w->beta_type == 30u);
    return q_type && k_type && v_type && output_type && low_types &&
           w->attn_norm && w->q && w->k && w->v && w->output &&
           w->q_conv && w->k_conv && w->v_conv && w->f_a && w->f_b &&
           w->g_a && w->g_b && w->beta && w->o_norm && w->dt_bias &&
           w->a_log;
}

static int kda_empty(const ds4_glm5_kda_weight_offsets *w) {
    return !w->attn_norm && !w->q && !w->k && !w->v && !w->output &&
           !w->q_conv && !w->k_conv && !w->v_conv && !w->f_a && !w->f_b &&
           !w->g_a && !w->g_b && !w->beta && !w->o_norm && !w->dt_bias &&
           !w->a_log && !w->q_type && !w->k_type && !w->v_type &&
           !w->output_type && !w->f_a_type && !w->f_b_type &&
           !w->g_a_type && !w->g_b_type && !w->beta_type;
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
    if ((model->token_embd_type != 0u &&
         model->token_embd_type != 8u && model->token_embd_type != 30u) ||
        (model->output_type != 0u &&
         model->output_type != 8u && model->output_type != 30u)) {
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
