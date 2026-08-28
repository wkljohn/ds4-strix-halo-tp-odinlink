#include "ds4_glm5_next_exec.h"

#include <stdlib.h>
#include <string.h>

#include "ds4_gpu.h"

enum {
    GLM5_WIDTH = 4096,
    GLM5_HC = 4,
    GLM5_HC_WIDTH = GLM5_WIDTH * GLM5_HC,
    GLM5_HC_MIX = 24,
    GLM5_DENSE_MID = 12288,
    GLM5_VOCAB = 154880,
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
    ds4_glm5_kda_workspace kda;
};

static ds4_gpu_tensor *f32(uint64_t count) {
    return ds4_gpu_tensor_alloc(count * sizeof(float));
}

void ds4_glm5_next_workspace_destroy(ds4_glm5_next_workspace *w) {
    if (!w) return;
    ds4_glm5_kda_workspace_free(&w->kda);
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
    if (!w->hc_flat || !w->hc_mix || !w->hc_split || !w->collapsed ||
        !w->attention || !w->after_attention || !w->ffn_flat ||
        !w->ffn_mix || !w->ffn_split || !w->ffn_collapsed ||
        !w->ffn_hidden || !w->gate || !w->up || !w->mid || !w->down ||
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

static int dense_kda_forward(const ds4_glm5_next_exec_ctx *ctx,
                             uint32_t il,
                             ds4_glm5_next_state *state,
                             ds4_glm5_next_workspace *w,
                             const ds4_gpu_tensor *hc_in,
                             ds4_gpu_tensor *hc_out) {
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    ds4_glm5_kda_layer_state *kda = &state->kda.layer[il];
    const int ok =
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
            GLM5_WIDTH, GLM5_HC) &&
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
    return 0;
}
