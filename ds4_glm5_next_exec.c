#include "ds4_glm5_next_exec.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_gpu.h"
#include "ds4_tp.h"

enum {
    GLM5_WIDTH = 4096,
    GLM5_HC = 4,
    GLM5_HC_WIDTH = GLM5_WIDTH * GLM5_HC,
    GLM5_HC_MIX = 24,
    GLM5_DENSE_MID = 12288,
    GLM5_VOCAB = 154880,
    GLM5_Q_RANK = 1536,
    GLM5_HEADS = 64,
    GLM5_HEAD_DIM = 256,
    GLM5_KV_LORA = 512,
    GLM5_INDEX_DIM = 128,
    GLM5_INDEX_TOP_K = 2048,
    GLM5_EXPERTS = 288,
    GLM5_EXPERTS_USED = 8,
    GLM5_ROUTED_MID = 2048,
    GLM5_RANK_MID = 1024,
    GLM5_Q4K_BLOCK_BYTES = 144,
    GLM5_Q4K_QK = 256,
    GLM5_Q8_BLOCK_BYTES = 34,
    GLM5_Q8_QK = 32,
};

struct ds4_glm5_next_workspace {
    ds4_gpu_tensor *hc_flat;
    ds4_gpu_tensor *hc_mix;
    ds4_gpu_tensor *hc_split;
    ds4_gpu_tensor *collapsed;
    ds4_gpu_tensor *attention;
    ds4_gpu_tensor *after_attention;
    ds4_gpu_tensor *ffn_flat;
    ds4_gpu_tensor *ffn_mix;
    ds4_gpu_tensor *ffn_split;
    ds4_gpu_tensor *ffn_collapsed;
    ds4_gpu_tensor *ffn_hidden;
    ds4_gpu_tensor *gate;
    ds4_gpu_tensor *up;
    ds4_gpu_tensor *mid;
    ds4_gpu_tensor *down;
    ds4_gpu_tensor *mla_q_a;
    ds4_gpu_tensor *mla_q_resid;
    ds4_gpu_tensor *mla_query;
    ds4_gpu_tensor *mla_kv_raw;
    ds4_gpu_tensor *mla_kv_norm;
    ds4_gpu_tensor *mla_qk_low;
    ds4_gpu_tensor *mla_index_k_raw;
    ds4_gpu_tensor *mla_pool_gate_raw;
    ds4_gpu_tensor *mla_selected_token;
    ds4_gpu_tensor *mla_heads;
    ds4_gpu_tensor *router_logits;
    ds4_gpu_tensor *router_probs;
    ds4_gpu_tensor *router_selected;
    ds4_gpu_tensor *router_weights;
    ds4_gpu_tensor *routed_gate;
    ds4_gpu_tensor *routed_up;
    ds4_gpu_tensor *routed_mid;
    ds4_gpu_tensor *routed_experts;
    ds4_gpu_tensor *routed_out;
    ds4_gpu_tensor *shared_gate;
    ds4_gpu_tensor *shared_up;
    ds4_gpu_tensor *shared_mid;
    ds4_gpu_tensor *shared_out;
    ds4_glm5_kda_workspace kda;
};

static ds4_gpu_tensor *f32(uint64_t count) {
    return ds4_gpu_tensor_alloc(count * sizeof(float));
}

void ds4_glm5_next_workspace_destroy(ds4_glm5_next_workspace *w) {
    if (!w) return;
    ds4_glm5_kda_workspace_free(&w->kda);
    ds4_gpu_tensor_free(w->shared_out);
    ds4_gpu_tensor_free(w->shared_mid);
    ds4_gpu_tensor_free(w->shared_up);
    ds4_gpu_tensor_free(w->shared_gate);
    ds4_gpu_tensor_free(w->routed_out);
    ds4_gpu_tensor_free(w->routed_experts);
    ds4_gpu_tensor_free(w->routed_mid);
    ds4_gpu_tensor_free(w->routed_up);
    ds4_gpu_tensor_free(w->routed_gate);
    ds4_gpu_tensor_free(w->router_weights);
    ds4_gpu_tensor_free(w->router_selected);
    ds4_gpu_tensor_free(w->router_probs);
    ds4_gpu_tensor_free(w->router_logits);
    ds4_gpu_tensor_free(w->mla_heads);
    ds4_gpu_tensor_free(w->mla_selected_token);
    ds4_gpu_tensor_free(w->mla_pool_gate_raw);
    ds4_gpu_tensor_free(w->mla_index_k_raw);
    ds4_gpu_tensor_free(w->mla_qk_low);
    ds4_gpu_tensor_free(w->mla_kv_norm);
    ds4_gpu_tensor_free(w->mla_kv_raw);
    ds4_gpu_tensor_free(w->mla_query);
    ds4_gpu_tensor_free(w->mla_q_resid);
    ds4_gpu_tensor_free(w->mla_q_a);
    ds4_gpu_tensor_free(w->down);
    ds4_gpu_tensor_free(w->mid);
    ds4_gpu_tensor_free(w->up);
    ds4_gpu_tensor_free(w->gate);
    ds4_gpu_tensor_free(w->ffn_hidden);
    ds4_gpu_tensor_free(w->ffn_collapsed);
    ds4_gpu_tensor_free(w->ffn_split);
    ds4_gpu_tensor_free(w->ffn_mix);
    ds4_gpu_tensor_free(w->ffn_flat);
    ds4_gpu_tensor_free(w->after_attention);
    ds4_gpu_tensor_free(w->attention);
    ds4_gpu_tensor_free(w->collapsed);
    ds4_gpu_tensor_free(w->hc_split);
    ds4_gpu_tensor_free(w->hc_mix);
    ds4_gpu_tensor_free(w->hc_flat);
    memset(w, 0, sizeof(*w));
    free(w);
}

ds4_glm5_next_workspace *ds4_glm5_next_workspace_create(void) {
    ds4_glm5_next_workspace *w = calloc(1u, sizeof(*w));
    if (!w) return NULL;
    w->hc_flat = f32(GLM5_HC_WIDTH);
    w->hc_mix = f32(GLM5_HC_MIX);
    w->hc_split = f32(GLM5_HC_MIX);
    w->collapsed = f32(GLM5_WIDTH);
    w->attention = f32(GLM5_WIDTH);
    w->after_attention = f32(GLM5_HC_WIDTH);
    w->ffn_flat = f32(GLM5_HC_WIDTH);
    w->ffn_mix = f32(GLM5_HC_MIX);
    w->ffn_split = f32(GLM5_HC_MIX);
    w->ffn_collapsed = f32(GLM5_WIDTH);
    w->ffn_hidden = f32(GLM5_WIDTH);
    w->gate = f32(GLM5_DENSE_MID);
    w->up = f32(GLM5_DENSE_MID);
    w->mid = f32(GLM5_DENSE_MID);
    w->down = f32(GLM5_WIDTH);
    w->mla_q_a = f32(GLM5_Q_RANK);
    w->mla_q_resid = f32(GLM5_Q_RANK);
    w->mla_query = f32((uint64_t)GLM5_HEADS * GLM5_HEAD_DIM);
    w->mla_kv_raw = f32(GLM5_KV_LORA);
    w->mla_kv_norm = f32(GLM5_KV_LORA);
    w->mla_qk_low = f32((uint64_t)GLM5_HEADS * GLM5_KV_LORA);
    w->mla_index_k_raw = f32(GLM5_INDEX_DIM);
    w->mla_pool_gate_raw = f32(GLM5_INDEX_DIM);
    w->mla_selected_token = ds4_gpu_tensor_alloc(
        (GLM5_INDEX_TOP_K + 3u) * sizeof(int32_t));
    w->mla_heads = f32((uint64_t)GLM5_HEADS * GLM5_HEAD_DIM);
    w->router_logits = f32(GLM5_EXPERTS);
    w->router_probs = f32(GLM5_EXPERTS);
    w->router_selected = ds4_gpu_tensor_alloc(
        GLM5_EXPERTS_USED * sizeof(int32_t));
    w->router_weights = f32(GLM5_EXPERTS_USED);
    w->routed_gate = f32((uint64_t)GLM5_EXPERTS_USED * GLM5_RANK_MID);
    w->routed_up = f32((uint64_t)GLM5_EXPERTS_USED * GLM5_RANK_MID);
    w->routed_mid = f32((uint64_t)GLM5_EXPERTS_USED * GLM5_RANK_MID);
    w->routed_experts = f32((uint64_t)GLM5_EXPERTS_USED * GLM5_WIDTH);
    w->routed_out = f32(GLM5_WIDTH);
    w->shared_gate = f32(GLM5_RANK_MID);
    w->shared_up = f32(GLM5_RANK_MID);
    w->shared_mid = f32(GLM5_RANK_MID);
    w->shared_out = f32(GLM5_WIDTH);
    if (!w->hc_flat || !w->hc_mix || !w->hc_split || !w->collapsed ||
        !w->attention || !w->after_attention || !w->ffn_flat ||
        !w->ffn_mix || !w->ffn_split || !w->ffn_collapsed ||
        !w->ffn_hidden || !w->gate || !w->up || !w->mid || !w->down ||
        !w->mla_q_a || !w->mla_q_resid || !w->mla_query ||
        !w->mla_kv_raw || !w->mla_kv_norm || !w->mla_qk_low ||
        !w->mla_index_k_raw || !w->mla_pool_gate_raw ||
        !w->mla_selected_token || !w->mla_heads || !w->router_logits ||
        !w->router_probs || !w->router_selected || !w->router_weights ||
        !w->routed_gate || !w->routed_up || !w->routed_mid ||
        !w->routed_experts || !w->routed_out || !w->shared_gate ||
        !w->shared_up || !w->shared_mid || !w->shared_out ||
        !ds4_glm5_kda_workspace_init(&w->kda, 1u)) {
        ds4_glm5_next_workspace_destroy(w);
        return NULL;
    }
    return w;
}

static int context_valid(const ds4_glm5_next_exec_ctx *ctx) {
    return ctx && ctx->model_map && ctx->model_size != 0u &&
           ds4_glm5_next_model_offsets_validate(ctx->model);
}

int ds4_glm5_next_embed_token(const ds4_glm5_next_exec_ctx *ctx,
                              uint32_t token,
                              ds4_gpu_tensor *hc_out) {
    return context_valid(ctx) && hc_out && token < GLM5_VOCAB &&
           ds4_gpu_embed_token_hc_bf16_tensor(
               hc_out, ctx->model_map, ctx->model_size,
               ctx->model->token_embd, GLM5_VOCAB, token,
               GLM5_WIDTH, GLM5_HC);
}

static int kda_attention_one(const ds4_glm5_next_exec_ctx *ctx,
                             uint32_t il,
                             ds4_glm5_next_state *state,
                             ds4_glm5_next_workspace *w,
                             const ds4_gpu_tensor *hc_in) {
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    ds4_glm5_kda_layer_state *kda = &state->kda.layer[il];
    return
        ds4_gpu_rms_norm_plain_rows_tensor(
            w->hc_flat, hc_in, GLM5_HC_WIDTH, 1u, 1.0e-5f) &&
        ds4_gpu_matmul_bf16_tensor(
            w->hc_mix, ctx->model_map, ctx->model_size,
            layer->hc.attn_fn, GLM5_HC_WIDTH, GLM5_HC_MIX,
            w->hc_flat, 1u) &&
        ds4_gpu_hc_split_weighted_sum_tensor(
            w->collapsed, w->hc_split, w->hc_mix, hc_in,
            ctx->model_map, ctx->model_size,
            layer->hc.attn_scale, layer->hc.attn_base,
            GLM5_WIDTH, GLM5_HC, 20u, 1.0e-6f) &&
        ds4_glm5_kda_layer_forward(
            kda, &w->kda, &layer->kda, ctx->model_map, ctx->model_size,
            w->collapsed, w->attention, 1u) &&
        ds4_gpu_hc_expand_split_tensor(
            w->after_attention, w->attention, hc_in, w->hc_split,
            GLM5_WIDTH, GLM5_HC);
}

static int dense_kda_forward(const ds4_glm5_next_exec_ctx *ctx,
                             uint32_t il,
                             ds4_glm5_next_state *state,
                             ds4_glm5_next_workspace *w,
                             const ds4_gpu_tensor *hc_in,
                             ds4_gpu_tensor *hc_out) {
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    const int ok =
        kda_attention_one(ctx, il, state, w, hc_in) &&
        ds4_gpu_rms_norm_plain_rows_tensor(
            w->ffn_flat, w->after_attention, GLM5_HC_WIDTH, 1u, 1.0e-5f) &&
        ds4_gpu_matmul_bf16_tensor(
            w->ffn_mix, ctx->model_map, ctx->model_size,
            layer->hc.ffn_fn, GLM5_HC_WIDTH, GLM5_HC_MIX,
            w->ffn_flat, 1u) &&
        ds4_gpu_hc_split_weighted_sum_norm_tensor(
            w->ffn_collapsed, w->ffn_hidden, w->ffn_split, w->ffn_mix,
            w->after_attention, ctx->model_map, ctx->model_size,
            layer->hc.ffn_scale, layer->hc.ffn_base, layer->ffn_norm,
            GLM5_WIDTH, GLM5_HC, 20u, 1.0e-6f, 1.0e-5f) &&
        ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
            w->gate, w->up, w->mid, ctx->model_map, ctx->model_size,
            layer->ffn_weight.gate, layer->ffn_weight.up,
            GLM5_WIDTH, GLM5_DENSE_MID, w->ffn_hidden, 10.0f) &&
        ds4_gpu_matmul_q8_0_tensor(
            w->down, ctx->model_map, ctx->model_size,
            layer->ffn_weight.down, GLM5_DENSE_MID, GLM5_WIDTH,
            w->mid, 1u) &&
        ds4_gpu_hc_expand_split_tensor(
            hc_out, w->down, w->after_attention, w->ffn_split,
            GLM5_WIDTH, GLM5_HC) &&
        ds4_gpu_synchronize();
    if (!ok) ds4_glm5_next_state_invalidate(state);
    return ok;
}

static int tp_context_valid(const ds4_glm5_next_exec_ctx *ctx) {
    const uint64_t bytes = (uint64_t)GLM5_WIDTH * sizeof(float);
    return ctx && ctx->tp && ctx->tp_sequence && ctx->tp_rank <= 1u &&
           (uint32_t)ds4_tp_rank(ctx->tp) == ctx->tp_rank &&
           ctx->tp_big_out && ctx->tp_big_in &&
           ds4_gpu_tensor_bytes(ctx->tp_big_out) >= bytes &&
           ds4_gpu_tensor_bytes(ctx->tp_big_in) >= bytes &&
           ctx->tp_big_out_host && ctx->tp_big_in_host &&
           ds4_tp_is_rdma(ctx->tp) &&
           ds4_tp_big_gate_is_rdma_capable(ctx->tp) &&
           ds4_tp_big_gate_is_direct(ctx->tp, ctx->tp_big_out_host,
                                     ctx->tp_big_in_host, bytes);
}

static int tp_exchange(const ds4_glm5_next_exec_ctx *ctx, uint32_t layer) {
    const uint64_t bytes = (uint64_t)GLM5_WIDTH * sizeof(float);
    if (!tp_context_valid(ctx) || *ctx->tp_sequence == UINT64_MAX ||
        !ds4_gpu_synchronize()) return 0;
    const uint64_t sequence = ++*ctx->tp_sequence;
    if (!ds4_tp_big_gate_exchange(ctx->tp, layer, sequence,
                                  ctx->tp_big_out_host,
                                  ctx->tp_big_in_host, bytes)) {
        ds4_tp_mark_failed(ctx->tp);
        return 0;
    }
    return 1;
}

static uint64_t fnv64_continue(uint64_t hash, const void *data,
                               uint64_t bytes) {
    const unsigned char *p = (const unsigned char *)data;
    for (uint64_t i = 0u; i < bytes; ++i) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int route_agrees(const ds4_glm5_next_exec_ctx *ctx, uint32_t layer,
                        uint32_t token_ordinal,
                        const ds4_gpu_tensor *selected,
                        const ds4_gpu_tensor *weights) {
    int32_t ids[GLM5_EXPERTS_USED];
    float route_weights[GLM5_EXPERTS_USED];
    if (!tp_context_valid(ctx) ||
        !ds4_gpu_tensor_read(selected, 0u, ids, sizeof(ids)) ||
        !ds4_gpu_tensor_read(weights, 0u, route_weights,
                             sizeof(route_weights))) return 0;
    for (uint32_t i = 0u; i < GLM5_EXPERTS_USED; ++i) {
        if (ids[i] < 0 || ids[i] >= GLM5_EXPERTS ||
            !isfinite(route_weights[i]) || route_weights[i] < 0.0f) return 0;
        for (uint32_t j = 0u; j < i; ++j)
            if (ids[i] == ids[j]) return 0;
    }
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = fnv64_continue(hash, ids, sizeof(ids));
    hash = fnv64_continue(hash, route_weights, sizeof(route_weights));
    const uint64_t check_sequence = UINT64_C(0x474c4d3500000000) ^
        (*ctx->tp_sequence << 16u) ^ ((uint64_t)layer << 8u) ^ token_ordinal;
    char error[256] = {0};
    const int rc = ds4_tp_hash_check(ctx->tp, check_sequence, hash,
                                     error, sizeof(error));
    if (rc != 1) {
        fprintf(stderr, "ds4: GLM5 route agreement failed: %s\n",
                error[0] ? error : "invalid local route");
        ds4_tp_mark_failed(ctx->tp);
        return 0;
    }
    return 1;
}

static int declare_local_q4k_half(const ds4_glm5_next_exec_ctx *ctx,
                                  const ds4_glm5_next_layer_offsets *layer) {
    const uint64_t gate_row_bytes =
        (GLM5_WIDTH / GLM5_Q4K_QK) * GLM5_Q4K_BLOCK_BYTES;
    const uint64_t down_row_bytes =
        (GLM5_ROUTED_MID / GLM5_Q4K_QK) * GLM5_Q4K_BLOCK_BYTES;
    const uint64_t down_half_bytes =
        (GLM5_RANK_MID / GLM5_Q4K_QK) * GLM5_Q4K_BLOCK_BYTES;
    const uint32_t row_base = ctx->tp_rank * GLM5_RANK_MID;
    const uint64_t column_base = (uint64_t)ctx->tp_rank * down_half_bytes;
    return ds4_gpu_q4k_packed_slice_declare(
               ctx->model_map, ctx->model_size, layer->ffn_weight.gate_exps,
               GLM5_EXPERTS, GLM5_ROUTED_MID, gate_row_bytes,
               row_base, GLM5_RANK_MID, 0u, gate_row_bytes,
               DS4_GPU_Q4K_PACKED_ROW_RANGE) &&
           ds4_gpu_q4k_packed_slice_declare(
               ctx->model_map, ctx->model_size, layer->ffn_weight.up_exps,
               GLM5_EXPERTS, GLM5_ROUTED_MID, gate_row_bytes,
               row_base, GLM5_RANK_MID, 0u, gate_row_bytes,
               DS4_GPU_Q4K_PACKED_ROW_RANGE) &&
           ds4_gpu_q4k_packed_slice_declare(
               ctx->model_map, ctx->model_size, layer->ffn_weight.down_exps,
               GLM5_EXPERTS, GLM5_WIDTH, down_row_bytes,
               0u, GLM5_WIDTH, column_base, down_half_bytes,
               DS4_GPU_Q4K_PACKED_K_RANGE) &&
           ds4_gpu_q4k_packed_slice_load(
               ctx->model_map, layer->ffn_weight.gate_exps,
               row_base, GLM5_RANK_MID, 0u, gate_row_bytes) &&
           ds4_gpu_q4k_packed_slice_load(
               ctx->model_map, layer->ffn_weight.up_exps,
               row_base, GLM5_RANK_MID, 0u, gate_row_bytes) &&
           ds4_gpu_q4k_packed_slice_load(
               ctx->model_map, layer->ffn_weight.down_exps,
               0u, GLM5_WIDTH, column_base, down_half_bytes);
}

/* Before the first complete pool-4 group exists, the official selection is
 * exactly the visible tail. Exercise that cache-read path independently before
 * adding the pooled long-context selector. */
static int mla_short_context_attention(const ds4_glm5_next_exec_ctx *ctx,
                                       uint32_t il,
                                       ds4_glm5_next_state *state,
                                       ds4_glm5_next_workspace *w,
                                       const ds4_gpu_tensor *hc_in) {
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    const ds4_glm5_next_mla_offsets *m = &layer->mla;
    ds4_glm5_next_mla_state *mla = &state->mla[il];
    const uint32_t pos = mla->token_count;
    const uint32_t visible = pos + 1u;
    const uint64_t half_heads =
        ((uint64_t)GLM5_HEADS * GLM5_HEAD_DIM) / 2u;
    return
        ds4_gpu_rms_norm_plain_rows_tensor(
            w->hc_flat, hc_in, GLM5_HC_WIDTH, 1u, 1.0e-5f) &&
        ds4_gpu_matmul_bf16_tensor(
            w->hc_mix, ctx->model_map, ctx->model_size,
            layer->hc.attn_fn, GLM5_HC_WIDTH, GLM5_HC_MIX,
            w->hc_flat, 1u) &&
        ds4_gpu_hc_split_weighted_sum_norm_tensor(
            w->collapsed, w->ffn_hidden, w->hc_split, w->hc_mix, hc_in,
            ctx->model_map, ctx->model_size,
            layer->hc.attn_scale, layer->hc.attn_base, layer->attn_norm,
            GLM5_WIDTH, GLM5_HC, 20u, 1.0e-6f, 1.0e-5f) &&
        ds4_gpu_matmul_q8_0_tensor(
            w->mla_q_a, ctx->model_map, ctx->model_size, m->q_a,
            GLM5_WIDTH, GLM5_Q_RANK, w->ffn_hidden, 1u) &&
        ds4_gpu_rms_norm_weight_tensor(
            w->mla_q_resid, w->mla_q_a, ctx->model_map, ctx->model_size,
            m->q_a_norm, GLM5_Q_RANK, 1.0e-5f) &&
        ds4_gpu_matmul_q8_0_tensor(
            w->mla_query, ctx->model_map, ctx->model_size, m->q_b,
            GLM5_Q_RANK, GLM5_HEADS * GLM5_HEAD_DIM,
            w->mla_q_resid, 1u) &&
        ds4_gpu_matmul_q8_0_tensor(
            w->mla_kv_raw, ctx->model_map, ctx->model_size, m->kv_a_mqa,
            GLM5_WIDTH, GLM5_KV_LORA, w->ffn_hidden, 1u) &&
        ds4_gpu_glm_kv_lora_rms_norm_tensor(
            w->mla_kv_norm, w->mla_kv_raw,
            ctx->model_map, ctx->model_size, m->kv_a_norm,
            1u, GLM5_KV_LORA, GLM5_KV_LORA, 1.0e-5f) &&
        ds4_gpu_glm_store_compact_kv_tensor(
            mla->compact_kv, NULL, w->mla_kv_norm, w->mla_kv_raw,
            pos, 1u, mla->capacity_tokens, GLM5_KV_LORA,
            GLM5_KV_LORA, 0u, false) &&
        ds4_gpu_glm_qk_lowrank_typed_tensor(
            w->mla_qk_low, w->mla_query,
            ctx->model_map, ctx->model_size, m->k_b, 8u,
            GLM5_HEADS, GLM5_KV_LORA, GLM5_HEAD_DIM, GLM5_HEAD_DIM) &&
        ds4_gpu_matmul_bf16_tensor(
            w->mla_index_k_raw, ctx->model_map, ctx->model_size, m->index_k,
            GLM5_WIDTH, GLM5_INDEX_DIM, w->ffn_hidden, 1u) &&
        ds4_gpu_glm_store_indexer_k_tensor(
            mla->index_key, w->mla_index_k_raw,
            ctx->model_map, ctx->model_size,
            m->index_k_norm, m->index_k_norm_b,
            pos, 1u, mla->capacity_tokens, GLM5_INDEX_DIM,
            0u, 1u, 1.0e-6f, 1.0f, 1.0f, 0.0f,
            1.0f, 0.0f, 0.0f, false) &&
        ds4_gpu_matmul_bf16_tensor(
            w->mla_pool_gate_raw, ctx->model_map, ctx->model_size,
            m->index_pool_gate, GLM5_WIDTH, GLM5_INDEX_DIM,
            w->ffn_hidden, 1u) &&
        ds4_gpu_tensor_copy(mla->pool_gate,
                            (uint64_t)pos * GLM5_INDEX_DIM * sizeof(float),
                            w->mla_pool_gate_raw,
                            0u, GLM5_INDEX_DIM * sizeof(float)) &&
        ds4_gpu_glm_fill_selected_range_tensor(w->mla_selected_token,
                                                visible) &&
        ds4_gpu_glm_attention_indexed_decode_typed_tensor(
            w->mla_heads, w->mla_query, w->mla_qk_low,
            mla->compact_kv, NULL, ctx->model_map, ctx->model_size,
            m->v_b, 8u, w->mla_selected_token, visible,
            mla->capacity_tokens, false, GLM5_HEADS, GLM5_KV_LORA,
            GLM5_HEAD_DIM, 0u, GLM5_HEAD_DIM, 0u,
            1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f) &&
        ds4_gpu_matmul_q8_0_kslice_tensor(
            ctx->tp_big_out, ctx->model_map, ctx->model_size, m->output,
            GLM5_HEADS * GLM5_HEAD_DIM,
            (uint64_t)ctx->tp_rank * half_heads, half_heads,
            GLM5_WIDTH, w->mla_heads,
            (uint64_t)ctx->tp_rank * half_heads) &&
        tp_exchange(ctx, il) &&
        ds4_gpu_add_tensor(w->attention, ctx->tp_big_out, ctx->tp_big_in,
                           GLM5_WIDTH) &&
        ds4_gpu_hc_expand_split_tensor(
            w->after_attention, w->attention, hc_in, w->hc_split,
            GLM5_WIDTH, GLM5_HC);
}

static int routed_ffn_one(const ds4_glm5_next_exec_ctx *ctx,
                          uint32_t il,
                          uint32_t token_ordinal,
                          ds4_glm5_next_workspace *w,
                          ds4_gpu_tensor *hc_out) {
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    const ds4_glm5_next_ffn_offsets *f = &layer->ffn_weight;
    const uint32_t rank_mid_base = ctx->tp_rank * GLM5_RANK_MID;
    const uint64_t q8_gate_row_bytes =
        (GLM5_WIDTH / GLM5_Q8_QK) * GLM5_Q8_BLOCK_BYTES;
    const uint64_t q8_down_row_bytes =
        (GLM5_ROUTED_MID / GLM5_Q8_QK) * GLM5_Q8_BLOCK_BYTES;
    const uint64_t q4_gate_row_bytes =
        (GLM5_WIDTH / GLM5_Q4K_QK) * GLM5_Q4K_BLOCK_BYTES;
    const uint64_t q4_down_row_bytes =
        (GLM5_ROUTED_MID / GLM5_Q4K_QK) * GLM5_Q4K_BLOCK_BYTES;
    const uint64_t q4_down_half_bytes =
        (GLM5_RANK_MID / GLM5_Q4K_QK) * GLM5_Q4K_BLOCK_BYTES;
    const uint64_t q4_down_base =
        (uint64_t)ctx->tp_rank * q4_down_half_bytes;
    const int ok =
        ds4_gpu_rms_norm_plain_rows_tensor(
            w->ffn_flat, w->after_attention, GLM5_HC_WIDTH, 1u, 1.0e-5f) &&
        ds4_gpu_matmul_bf16_tensor(
            w->ffn_mix, ctx->model_map, ctx->model_size,
            layer->hc.ffn_fn, GLM5_HC_WIDTH, GLM5_HC_MIX,
            w->ffn_flat, 1u) &&
        ds4_gpu_hc_split_weighted_sum_norm_tensor(
            w->ffn_collapsed, w->ffn_hidden, w->ffn_split, w->ffn_mix,
            w->after_attention, ctx->model_map, ctx->model_size,
            layer->hc.ffn_scale, layer->hc.ffn_base, layer->ffn_norm,
            GLM5_WIDTH, GLM5_HC, 20u, 1.0e-6f, 1.0e-5f) &&
        ds4_gpu_matmul_f32_tensor(
            w->router_logits, ctx->model_map, ctx->model_size, f->gate_inp,
            GLM5_WIDTH, GLM5_EXPERTS, w->ffn_hidden, 1u) &&
        ds4_gpu_glm_router_select_tensor(
            w->router_selected, w->router_weights, w->router_probs,
            ctx->model_map, ctx->model_size, f->exp_probs_b,
            w->router_logits, GLM5_EXPERTS, GLM5_EXPERTS_USED, 2.5f) &&
        route_agrees(ctx, il, token_ordinal,
                     w->router_selected, w->router_weights) &&
        declare_local_q4k_half(ctx, layer) &&
        ds4_gpu_routed_moe_one_packed_q4k_tensor(
            w->routed_out, w->routed_gate, w->routed_up, w->routed_mid,
            w->routed_experts, ctx->model_map, ctx->model_size,
            f->gate_exps, f->up_exps, f->down_exps, GLM5_EXPERTS,
            q4_gate_row_bytes, q4_down_row_bytes,
            rank_mid_base, GLM5_RANK_MID, q4_down_base,
            q4_down_half_bytes, w->router_selected, w->router_weights,
            GLM5_EXPERTS_USED, 10.0f, w->ffn_hidden, NULL, il) &&
        ds4_gpu_matmul_q8_0_tensor(
            w->shared_gate, ctx->model_map, ctx->model_size,
            f->gate_shexp + (uint64_t)rank_mid_base * q8_gate_row_bytes,
            GLM5_WIDTH, GLM5_RANK_MID, w->ffn_hidden, 1u) &&
        ds4_gpu_matmul_q8_0_tensor(
            w->shared_up, ctx->model_map, ctx->model_size,
            f->up_shexp + (uint64_t)rank_mid_base * q8_gate_row_bytes,
            GLM5_WIDTH, GLM5_RANK_MID, w->ffn_hidden, 1u) &&
        ds4_gpu_swiglu_tensor(
            w->shared_mid, w->shared_gate, w->shared_up,
            GLM5_RANK_MID, 10.0f, 1.0f) &&
        ds4_gpu_matmul_q8_0_kslice_tensor(
            w->shared_out, ctx->model_map, ctx->model_size, f->down_shexp,
            GLM5_ROUTED_MID, rank_mid_base, GLM5_RANK_MID,
            GLM5_WIDTH, w->shared_mid, 0u) &&
        ds4_gpu_add_tensor(ctx->tp_big_out, w->routed_out, w->shared_out,
                           GLM5_WIDTH) &&
        tp_exchange(ctx, il) &&
        ds4_gpu_add_tensor(w->down, ctx->tp_big_out, ctx->tp_big_in,
                           GLM5_WIDTH) &&
        ds4_gpu_hc_expand_split_tensor(
            hc_out, w->down, w->after_attention, w->ffn_split,
            GLM5_WIDTH, GLM5_HC) &&
        ds4_gpu_synchronize();
    (void)q8_down_row_bytes;
    return ok;
}

static int kda_routed_one_forward(const ds4_glm5_next_exec_ctx *ctx,
                                  uint32_t il,
                                  ds4_glm5_next_state *state,
                                  ds4_glm5_next_workspace *w,
                                  const ds4_gpu_tensor *hc_in,
                                  ds4_gpu_tensor *hc_out) {
    ds4_glm5_kda_layer_state *kda = &state->kda.layer[il];
    if (!tp_context_valid(ctx) || !kda->valid || !kda->recurrent ||
        kda->token_count > UINT32_MAX) return 0;
    const uint32_t token_ordinal = (uint32_t)kda->token_count;
    const int ok = kda_attention_one(ctx, il, state, w, hc_in) &&
                   routed_ffn_one(ctx, il, token_ordinal, w, hc_out);
    if (!ok) ds4_glm5_next_state_invalidate(state);
    return ok;
}

static int mla_routed_short_forward(const ds4_glm5_next_exec_ctx *ctx,
                                    uint32_t il,
                                    ds4_glm5_next_state *state,
                                    ds4_glm5_next_workspace *w,
                                    const ds4_gpu_tensor *hc_in,
                                    ds4_gpu_tensor *hc_out) {
    ds4_glm5_next_mla_state *mla = &state->mla[il];
    if (!tp_context_valid(ctx) || !mla->valid || !mla->compact_kv ||
        !mla->index_key || !mla->pool_gate || mla->owner != state ||
        mla->capacity_tokens == 0u ||
        mla->token_count >= mla->capacity_tokens ||
        mla->token_count >= 4u) return 0;
    const int ok = mla_short_context_attention(ctx, il, state, w, hc_in) &&
                   routed_ffn_one(ctx, il, mla->token_count, w, hc_out);
    if (!ok) {
        ds4_glm5_next_state_invalidate(state);
        return 0;
    }
    if (mla->token_count == 0u) mla->first_valid = 0u;
    mla->token_count++;
    return 1;
}

int ds4_glm5_next_layer_forward(const ds4_glm5_next_exec_ctx *ctx,
                                uint32_t il,
                                ds4_glm5_next_state *state,
                                ds4_glm5_next_workspace *w,
                                const ds4_gpu_tensor *hc_in,
                                ds4_gpu_tensor *hc_out) {
    const uint64_t hc_bytes = (uint64_t)GLM5_HC_WIDTH * sizeof(float);
    if (!context_valid(ctx) || !state || !state->valid || !w || !hc_in ||
        !hc_out || hc_in == hc_out ||
        ds4_gpu_tensor_bytes(hc_in) < hc_bytes ||
        ds4_gpu_tensor_bytes(hc_out) < hc_bytes ||
        il >= ctx->model->trunk_count ||
        state->layer_count != ctx->model->trunk_count ||
        state->kda.layer_count != ctx->model->trunk_count) return 0;
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    if (layer->attention == DS4_GLM5_NEXT_ATTN_KDA &&
        layer->ffn == DS4_GLM5_NEXT_FFN_DENSE &&
        state->kda.layer[il].recurrent && !state->mla[il].compact_kv) {
        return dense_kda_forward(ctx, il, state, w, hc_in, hc_out);
    }
    if (layer->attention == DS4_GLM5_NEXT_ATTN_MLA &&
        layer->ffn == DS4_GLM5_NEXT_FFN_ROUTED &&
        state->mla[il].compact_kv && !state->kda.layer[il].recurrent) {
        return mla_routed_short_forward(ctx, il, state, w, hc_in, hc_out);
    }
    if (layer->attention == DS4_GLM5_NEXT_ATTN_KDA &&
        layer->ffn == DS4_GLM5_NEXT_FFN_ROUTED &&
        state->kda.layer[il].recurrent && !state->mla[il].compact_kv) {
        return kda_routed_one_forward(ctx, il, state, w, hc_in, hc_out);
    }
    return 0;
}
