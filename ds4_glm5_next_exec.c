#include "ds4_glm5_next_exec.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
    GLM5_INDEX_HEADS = 32,
    GLM5_INDEX_POOL = 4,
    GLM5_EXPERTS = 288,
    GLM5_EXPERTS_USED = 8,
    GLM5_ROUTED_MID = 2048,
    GLM5_RANK_MID = 1024,
    GLM5_Q4K_BLOCK_BYTES = 144,
    GLM5_Q4K_QK = 256,
    GLM5_Q8_BLOCK_BYTES = 34,
    GLM5_Q8_QK = 32,
    GLM5_KDA_LOCAL_CHANNELS = DS4_GLM5_KDA_CHANNELS / 2,
};

_Static_assert((DS4_GLM5_KDA_CHANNELS % 2u) == 0u,
               "KDA TP output K slices require an even channel count");

static int kda_output_kslice_contract(uint32_t rank, uint32_t n_tokens,
                                      uint64_t *k_off, uint64_t *k_cnt,
                                      uint64_t *local_bytes) {
    if (rank > 1u || n_tokens == 0u || !k_off || !k_cnt || !local_bytes)
        return 0;
    *k_cnt = GLM5_KDA_LOCAL_CHANNELS;
    *k_off = (uint64_t)rank * *k_cnt;
    *local_bytes =
        (uint64_t)n_tokens * *k_cnt * sizeof(float);
    return *k_off + *k_cnt <= DS4_GLM5_KDA_CHANNELS;
}

#ifdef DS4_TP_TEST_HOOKS
int ds4_glm5_next_kda_output_kslice_contract_test(
        uint32_t rank, uint32_t n_tokens, uint64_t *k_off,
        uint64_t *k_cnt, uint64_t *local_bytes) {
    return kda_output_kslice_contract(
        rank, n_tokens, k_off, k_cnt, local_bytes);
}
#endif

#ifdef DS4_TP_TEST_HOOKS
enum { GLM5_ROUTE_TRACE_LAYERS = 128 };
static uint64_t g_glm5_route_trace_hash[GLM5_ROUTE_TRACE_LAYERS];
static unsigned char g_glm5_route_trace_seen[GLM5_ROUTE_TRACE_LAYERS];
static int g_glm5_route_trace_registered;
static uint64_t g_glm5_hc_trace_hash[GLM5_ROUTE_TRACE_LAYERS];
static unsigned char g_glm5_hc_trace_seen[GLM5_ROUTE_TRACE_LAYERS];
static int g_glm5_hc_trace_registered;

static void glm5_route_trace_dump(void) {
    for (uint32_t layer = 0u; layer < GLM5_ROUTE_TRACE_LAYERS; ++layer) {
        if (g_glm5_route_trace_seen[layer]) {
            fprintf(stderr, "GLM5 batch route trace layer=%u hash=%016llx\n",
                    layer,
                    (unsigned long long)g_glm5_route_trace_hash[layer]);
        }
    }
}

static void glm5_hc_trace_dump(void) {
    for (uint32_t layer = 0u; layer < GLM5_ROUTE_TRACE_LAYERS; ++layer) {
        if (g_glm5_hc_trace_seen[layer]) {
            fprintf(stderr, "GLM5 batch HC trace layer=%u hash=%016llx\n",
                    layer,
                    (unsigned long long)g_glm5_hc_trace_hash[layer]);
        }
    }
}
#endif

/* ROCm may provide the existing strided F32xQ8 token-tile kernel for this
 * layout. Other backends return -1 and retain the scalar exact fallback. A
 * selected ROCm implementation returns 0 on failure so it cannot silently
 * fall back after engaging. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
int ds4_rocm_q8_kslice_f32_rows_strided(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t full_in_dim, uint64_t out_dim,
        uint64_t in_start, uint64_t in_count, const ds4_gpu_tensor *x,
        uint64_t x_elem_start, uint64_t n_tokens,
        uint64_t x_token_stride) {
    (void)out;
    (void)model_map;
    (void)model_size;
    (void)weight_offset;
    (void)full_in_dim;
    (void)out_dim;
    (void)in_start;
    (void)in_count;
    (void)x;
    (void)x_elem_start;
    (void)n_tokens;
    (void)x_token_stride;
    return -1;
}

/* ROCm-only RDMA cache-ordering probes. Other backends retain a fail-closed
 * weak implementation so selecting the diagnostic cannot silently do
 * nothing. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
int ds4_rocm_rdma_cache_release(void) { return 0; }

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
int ds4_rocm_rdma_cache_acquire(void) { return 0; }

/* The MLA batch path temporarily aliases routed_experts as its
 * [token][head][kv_lora] attention output.  Keep that reuse fail-closed if
 * either model geometry changes. */
_Static_assert(GLM5_EXPERTS_USED * GLM5_WIDTH >=
                   GLM5_HEADS * GLM5_KV_LORA,
               "MLA attention alias exceeds routed expert workspace");

struct ds4_glm5_next_workspace {
    uint32_t capacity_tokens;
    uint32_t sparse_pool_capacity;
    ds4_gpu_tensor *hc_mean_weights;
    ds4_gpu_tensor *output_hidden;
    ds4_gpu_tensor *output_norm;
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
    ds4_gpu_tensor *mla_index_k_norm;
    ds4_gpu_tensor *mla_pool_gate_raw;
    ds4_gpu_tensor *mla_pool_indices;
    ds4_gpu_tensor *mla_pool_valid;
    ds4_gpu_tensor *mla_tail_valid;
    ds4_gpu_tensor *mla_index_q;
    ds4_gpu_tensor *mla_index_weights;
    ds4_gpu_tensor *mla_pool_scores;
    ds4_gpu_tensor *mla_selected_pool;
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
    /* Optional decode-lifetime selected-expert windows.  Disabled unless
     * explicitly requested; retaining one eight-expert window per routed
     * layer is a diagnostic alternative to re-uploading every token. */
    ds4_gpu_q4k_window_cache *q4_window[DS4_GLM5_NEXT_LAYER_COUNT];
    ds4_gpu_q4k_window_cache *q4_window_scratch;
    bool decode_phase;
    ds4_glm5_kda_workspace kda;
};

static ds4_gpu_tensor *f32(uint64_t count) {
    if (count == 0u || count > UINT64_MAX / sizeof(float)) return NULL;
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(count * sizeof(float));
    if (t && getenv("DS4_GLM5_ZERO_WORKSPACE") != NULL &&
        !ds4_gpu_tensor_fill_f32(t, 0.0f, count)) {
        ds4_gpu_tensor_free(t);
        return NULL;
    }
    return t;
}

void ds4_glm5_next_workspace_begin_prefill(ds4_glm5_next_workspace *w) {
    if (w) w->decode_phase = false;
}

void ds4_glm5_next_workspace_begin_decode(ds4_glm5_next_workspace *w) {
    if (w) w->decode_phase = true;
}

static ds4_gpu_tensor *f32_rows(uint32_t rows, uint64_t width) {
    if (rows == 0u || width == 0u || width > UINT64_MAX / rows) return NULL;
    return f32((uint64_t)rows * width);
}

static ds4_gpu_tensor *bytes_rows(uint32_t rows, uint64_t row_bytes) {
    if (rows == 0u || row_bytes == 0u || row_bytes > UINT64_MAX / rows)
        return NULL;
    const uint64_t bytes = (uint64_t)rows * row_bytes;
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    if (t && getenv("DS4_GLM5_ZERO_WORKSPACE") != NULL &&
        ((bytes & 3u) != 0u ||
         !ds4_gpu_tensor_fill_f32(t, 0.0f, bytes / sizeof(float)))) {
        ds4_gpu_tensor_free(t);
        return NULL;
    }
    return t;
}

static int route_failure_stats(const char *name,
                               const ds4_gpu_tensor *tensor,
                               uint32_t count);

void ds4_glm5_next_workspace_destroy(ds4_glm5_next_workspace *w) {
    if (!w) return;
    if (w->q4_window_scratch) {
        ds4_gpu_q4k_window_cache_destroy(w->q4_window_scratch);
        w->q4_window_scratch = NULL;
    }
    for (uint32_t il = 0u; il < DS4_GLM5_NEXT_LAYER_COUNT; ++il) {
        if (w->q4_window[il]) {
            ds4_gpu_q4k_window_cache_destroy(w->q4_window[il]);
            w->q4_window[il] = NULL;
        }
    }
    ds4_glm5_kda_workspace_free(&w->kda);
    ds4_gpu_tensor_free(w->output_norm);
    ds4_gpu_tensor_free(w->output_hidden);
    ds4_gpu_tensor_free(w->hc_mean_weights);
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
    ds4_gpu_tensor_free(w->mla_selected_pool);
    ds4_gpu_tensor_free(w->mla_pool_scores);
    ds4_gpu_tensor_free(w->mla_index_weights);
    ds4_gpu_tensor_free(w->mla_index_q);
    ds4_gpu_tensor_free(w->mla_tail_valid);
    ds4_gpu_tensor_free(w->mla_pool_valid);
    ds4_gpu_tensor_free(w->mla_pool_indices);
    ds4_gpu_tensor_free(w->mla_pool_gate_raw);
    ds4_gpu_tensor_free(w->mla_index_k_norm);
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

ds4_glm5_next_workspace *ds4_glm5_next_workspace_create_capacity_context(
        uint32_t capacity_tokens, uint32_t context_capacity) {
    if (capacity_tokens == 0u || context_capacity < capacity_tokens) return NULL;
    ds4_glm5_next_workspace *w = calloc(1u, sizeof(*w));
    if (!w) return NULL;
    w->capacity_tokens = capacity_tokens;
    w->sparse_pool_capacity = context_capacity / 4u +
        (context_capacity % 4u != 0u);
    w->hc_mean_weights = f32(GLM5_HC);
    w->output_hidden = f32(GLM5_WIDTH);
    w->output_norm = f32(GLM5_WIDTH);
    w->hc_flat = f32_rows(capacity_tokens, GLM5_HC_WIDTH);
    w->hc_mix = f32_rows(capacity_tokens, GLM5_HC_MIX);
    w->hc_split = f32_rows(capacity_tokens, GLM5_HC_MIX);
    w->collapsed = f32_rows(capacity_tokens, GLM5_WIDTH);
    w->attention = f32_rows(capacity_tokens, GLM5_WIDTH);
    w->after_attention = f32_rows(capacity_tokens, GLM5_HC_WIDTH);
    w->ffn_flat = f32_rows(capacity_tokens, GLM5_HC_WIDTH);
    w->ffn_mix = f32_rows(capacity_tokens, GLM5_HC_MIX);
    w->ffn_split = f32_rows(capacity_tokens, GLM5_HC_MIX);
    w->ffn_collapsed = f32_rows(capacity_tokens, GLM5_WIDTH);
    w->ffn_hidden = f32_rows(capacity_tokens, GLM5_WIDTH);
    w->gate = f32_rows(capacity_tokens, GLM5_DENSE_MID);
    w->up = f32_rows(capacity_tokens, GLM5_DENSE_MID);
    w->mid = f32_rows(capacity_tokens, GLM5_DENSE_MID);
    w->down = f32_rows(capacity_tokens, GLM5_WIDTH);
    w->mla_q_a = f32_rows(capacity_tokens, GLM5_Q_RANK);
    w->mla_q_resid = f32_rows(capacity_tokens, GLM5_Q_RANK);
    w->mla_query = f32_rows(capacity_tokens,
                            (uint64_t)GLM5_HEADS * GLM5_HEAD_DIM);
    w->mla_kv_raw = f32_rows(capacity_tokens, GLM5_KV_LORA);
    w->mla_kv_norm = f32_rows(capacity_tokens, GLM5_KV_LORA);
    w->mla_qk_low = f32_rows(capacity_tokens,
                             (uint64_t)GLM5_HEADS * GLM5_KV_LORA);
    w->mla_index_k_raw = f32_rows(capacity_tokens, GLM5_INDEX_DIM);
    w->mla_index_k_norm = f32_rows(capacity_tokens, GLM5_INDEX_DIM);
    w->mla_pool_gate_raw = f32_rows(capacity_tokens, GLM5_INDEX_DIM);
    w->mla_pool_indices = bytes_rows(capacity_tokens,
                                     4u * sizeof(int32_t));
    w->mla_pool_valid = bytes_rows(capacity_tokens, sizeof(uint32_t));
    w->mla_tail_valid = bytes_rows(capacity_tokens,
                                   4u * sizeof(uint32_t));
    w->mla_index_q = f32((uint64_t)32u * GLM5_INDEX_DIM);
    w->mla_index_weights = f32(32u);
    w->mla_pool_scores = f32(w->sparse_pool_capacity);
    w->mla_selected_pool = bytes_rows(
        DS4_GLM5_NEXT_INDEX_TOP_K / 4u, sizeof(uint32_t));
    w->mla_selected_token = bytes_rows(
        capacity_tokens,
        (DS4_GLM5_NEXT_INDEX_TOP_K + 3u) * sizeof(int32_t));
    w->mla_heads = f32_rows(capacity_tokens,
                            (uint64_t)GLM5_HEADS * GLM5_HEAD_DIM);
    w->router_logits = f32_rows(capacity_tokens, GLM5_EXPERTS);
    w->router_probs = f32_rows(capacity_tokens, GLM5_EXPERTS);
    w->router_selected = bytes_rows(
        capacity_tokens, GLM5_EXPERTS_USED * sizeof(int32_t));
    w->router_weights = f32_rows(capacity_tokens, GLM5_EXPERTS_USED);
    /* Mixed IQ2_XXS/Q2_K keeps expert ownership sharded across ranks but each
     * owned expert still has the complete 2048-wide intermediate. Q4_K's
     * K-sharded path uses only the first 1024 values of these same temporary
     * buffers. This is tile/workspace scratch, never persistent weights. */
    w->routed_gate = f32_rows(
        capacity_tokens, (uint64_t)GLM5_EXPERTS_USED * GLM5_ROUTED_MID);
    w->routed_up = f32_rows(
        capacity_tokens, (uint64_t)GLM5_EXPERTS_USED * GLM5_ROUTED_MID);
    w->routed_mid = f32_rows(
        capacity_tokens, (uint64_t)GLM5_EXPERTS_USED * GLM5_ROUTED_MID);
    w->routed_experts = f32_rows(
        capacity_tokens, (uint64_t)GLM5_EXPERTS_USED * GLM5_WIDTH);
    w->routed_out = f32_rows(capacity_tokens, GLM5_WIDTH);
    w->shared_gate = f32_rows(capacity_tokens, GLM5_RANK_MID);
    w->shared_up = f32_rows(capacity_tokens, GLM5_RANK_MID);
    w->shared_mid = f32_rows(capacity_tokens, GLM5_RANK_MID);
    w->shared_out = f32_rows(capacity_tokens, GLM5_WIDTH);
    if (!w->hc_mean_weights || !w->output_hidden || !w->output_norm ||
        !w->hc_flat || !w->hc_mix || !w->hc_split || !w->collapsed ||
        !w->attention || !w->after_attention || !w->ffn_flat ||
        !w->ffn_mix || !w->ffn_split || !w->ffn_collapsed ||
        !w->ffn_hidden || !w->gate || !w->up || !w->mid || !w->down ||
        !w->mla_q_a || !w->mla_q_resid || !w->mla_query ||
        !w->mla_kv_raw || !w->mla_kv_norm || !w->mla_qk_low ||
        !w->mla_index_k_raw || !w->mla_index_k_norm ||
        !w->mla_pool_gate_raw ||
        !w->mla_pool_indices || !w->mla_pool_valid || !w->mla_tail_valid ||
        !w->mla_index_q || !w->mla_index_weights || !w->mla_pool_scores ||
        !w->mla_selected_pool ||
        !w->mla_selected_token || !w->mla_heads || !w->router_logits ||
        !w->router_probs || !w->router_selected || !w->router_weights ||
        !w->routed_gate || !w->routed_up || !w->routed_mid ||
        !w->routed_experts || !w->routed_out || !w->shared_gate ||
        !w->shared_up || !w->shared_mid || !w->shared_out ||
        !ds4_glm5_kda_workspace_init(&w->kda, capacity_tokens)) {
        ds4_glm5_next_workspace_destroy(w);
        return NULL;
    }
    const uint32_t tail_valid[4] = {1u, 1u, 1u, 1u};
    if (!ds4_gpu_tensor_fill_f32(w->hc_mean_weights,
                                 1.0f / (float)GLM5_HC,
                                 GLM5_HC) ||
        !ds4_gpu_tensor_write(w->mla_tail_valid, 0u, tail_valid,
                              sizeof(tail_valid))) {
        ds4_glm5_next_workspace_destroy(w);
        return NULL;
    }
    return w;
}

ds4_glm5_next_workspace *ds4_glm5_next_workspace_create_capacity(
        uint32_t capacity_tokens) {
    return ds4_glm5_next_workspace_create_capacity_context(
        capacity_tokens, capacity_tokens);
}

ds4_glm5_next_workspace *ds4_glm5_next_workspace_create(void) {
    return ds4_glm5_next_workspace_create_capacity(1u);
}

static int context_valid(const ds4_glm5_next_exec_ctx *ctx) {
    return ctx && ctx->model_map && ctx->model_size != 0u &&
           ds4_glm5_next_model_offsets_validate(ctx->model);
}

static int trace_tensor(const ds4_glm5_next_exec_ctx *ctx, uint32_t layer,
                        uint32_t token, const char *name,
                        const ds4_gpu_tensor *tensor, uint64_t bytes) {
    if (!ctx->trace_prefix ||
        (ctx->trace_layer != UINT32_MAX && layer != ctx->trace_layer) ||
        (ctx->trace_token != UINT32_MAX && token != ctx->trace_token)) return 1;
    if (!name || !tensor || ds4_gpu_tensor_bytes(tensor) < bytes ||
        bytes == 0u || bytes > SIZE_MAX) return 0;
    void *host = malloc((size_t)bytes);
    if (!host) return 0;
    char path[768];
    int n = 0;
    if (ctx->trace_layer == UINT32_MAX && ctx->trace_token == UINT32_MAX)
        n = snprintf(path, sizeof(path), "%s.l%u.t%u.%s",
                     ctx->trace_prefix, layer, token, name);
    else if (ctx->trace_layer == UINT32_MAX)
        n = snprintf(path, sizeof(path), "%s.l%u.%s",
                     ctx->trace_prefix, layer, name);
    else if (ctx->trace_token == UINT32_MAX)
        n = snprintf(path, sizeof(path), "%s.t%u.%s", ctx->trace_prefix,
                     token, name);
    else
        n = snprintf(path, sizeof(path), "%s.%s", ctx->trace_prefix, name);
    int ok = n > 0 && (size_t)n < sizeof(path) &&
             ds4_gpu_tensor_read(tensor, 0u, host, bytes);
    FILE *fp = ok ? fopen(path, "wb") : NULL;
    if (fp) {
        const size_t written = fwrite(host, 1u, (size_t)bytes, fp);
        const int close_rc = fclose(fp);
        ok = written == (size_t)bytes && close_rc == 0;
    } else {
        ok = 0;
    }
    free(host);
    return ok;
}

static int trace_tensor_row(const ds4_glm5_next_exec_ctx *ctx, uint32_t layer,
                            uint32_t token, const char *name,
                            const ds4_gpu_tensor *tensor, uint32_t row,
                            uint64_t row_bytes) {
    if (!ctx->trace_prefix ||
        (ctx->trace_layer != UINT32_MAX && layer != ctx->trace_layer) ||
        (ctx->trace_token != UINT32_MAX && token != ctx->trace_token)) return 1;
    if (!tensor || row_bytes == 0u ||
        ds4_gpu_tensor_bytes(tensor) < (uint64_t)(row + 1u) * row_bytes)
        return 0;
    void *host = malloc((size_t)row_bytes);
    if (!host) return 0;
    char path[768];
    const int n = ctx->trace_layer == UINT32_MAX ?
        snprintf(path, sizeof(path), "%s.batch.l%u.t%u.%s",
                 ctx->trace_prefix, layer, token, name) :
        snprintf(path, sizeof(path), "%s.batch.t%u.%s",
                 ctx->trace_prefix, token, name);
    int ok = n > 0 && (size_t)n < sizeof(path) &&
             ds4_gpu_tensor_read(tensor, (uint64_t)row * row_bytes,
                                 host, row_bytes);
    FILE *fp = ok ? fopen(path, "wb") : NULL;
    if (fp) {
        ok = fwrite(host, 1u, (size_t)row_bytes, fp) == (size_t)row_bytes &&
             fclose(fp) == 0;
    } else ok = 0;
    free(host);
    return ok;
}

static int trace_mla_attention(const ds4_glm5_next_exec_ctx *ctx,
                               uint32_t layer, uint32_t token,
                               const ds4_gpu_tensor *hc_in,
                               ds4_glm5_next_workspace *w) {
    return trace_tensor(ctx, layer, token, "input_hc.f32", hc_in,
                        (uint64_t)GLM5_HC_WIDTH * sizeof(float)) &&
           trace_tensor(ctx, layer, token, "attn_split.f32", w->hc_split,
                        (uint64_t)GLM5_HC_MIX * sizeof(float)) &&
           trace_tensor(ctx, layer, token, "attn_collapsed.f32", w->collapsed,
                        (uint64_t)GLM5_WIDTH * sizeof(float)) &&
           trace_tensor(ctx, layer, token, "attn_hidden.f32", w->ffn_hidden,
                        (uint64_t)GLM5_WIDTH * sizeof(float)) &&
           trace_tensor(ctx, layer, token, "q_resid.f32", w->mla_q_resid,
                        (uint64_t)GLM5_Q_RANK * sizeof(float)) &&
           trace_tensor(ctx, layer, token, "query.f32", w->mla_query,
                        (uint64_t)GLM5_HEADS * GLM5_HEAD_DIM * sizeof(float)) &&
           trace_tensor(ctx, layer, token, "kv_norm.f32", w->mla_kv_norm,
                        (uint64_t)GLM5_KV_LORA * sizeof(float)) &&
           trace_tensor(ctx, layer, token, "qk_low.f32", w->mla_qk_low,
                        (uint64_t)GLM5_HEADS * GLM5_KV_LORA * sizeof(float)) &&
           trace_tensor(ctx, layer, token, "heads.f32", w->mla_heads,
                        (uint64_t)GLM5_HEADS * GLM5_HEAD_DIM * sizeof(float)) &&
           trace_tensor(ctx, layer, token, "attn_output.f32", w->attention,
                        (uint64_t)GLM5_WIDTH * sizeof(float)) &&
           trace_tensor(ctx, layer, token, "after_attn.f32", w->after_attention,
                        (uint64_t)GLM5_HC_WIDTH * sizeof(float));
}

static int trace_routed_ffn(const ds4_glm5_next_exec_ctx *ctx,
                            uint32_t layer, uint32_t token,
                            const ds4_gpu_tensor *hc_out,
                            ds4_glm5_next_workspace *w) {
    return trace_tensor(ctx, layer, token, "routed_gate.f32", w->routed_gate,
                        (uint64_t)GLM5_EXPERTS_USED * GLM5_ROUTED_MID * sizeof(float)) &&
           trace_tensor(ctx, layer, token, "routed_up.f32", w->routed_up,
                        (uint64_t)GLM5_EXPERTS_USED * GLM5_ROUTED_MID * sizeof(float)) &&
           trace_tensor(ctx, layer, token, "routed_mid.f32", w->routed_mid,
                        (uint64_t)GLM5_EXPERTS_USED * GLM5_ROUTED_MID * sizeof(float)) &&
           trace_tensor(ctx, layer, token, "routed_experts.f32", w->routed_experts,
                        (uint64_t)GLM5_EXPERTS_USED * GLM5_WIDTH * sizeof(float)) &&
           trace_tensor(ctx, layer, token, "ffn_down.f32", w->down,
                        (uint64_t)GLM5_WIDTH * sizeof(float)) &&
           trace_tensor(ctx, layer, token, "ffn_split.f32", w->ffn_split,
                        (uint64_t)GLM5_HC_MIX * sizeof(float)) &&
           trace_tensor(ctx, layer, token, "ffn_hidden.f32", w->ffn_hidden,
                        (uint64_t)GLM5_WIDTH * sizeof(float)) &&
           trace_tensor(ctx, layer, token, "router_ids.i32", w->router_selected,
                        (uint64_t)GLM5_EXPERTS_USED * sizeof(int32_t)) &&
           trace_tensor(ctx, layer, token, "router_weights.f32", w->router_weights,
                        (uint64_t)GLM5_EXPERTS_USED * sizeof(float)) &&
           trace_tensor(ctx, layer, token, "routed_out.f32", w->routed_out,
                        (uint64_t)GLM5_WIDTH * sizeof(float)) &&
           trace_tensor(ctx, layer, token, "shared_out.f32", w->shared_out,
                        (uint64_t)GLM5_WIDTH * sizeof(float)) &&
           trace_tensor(ctx, layer, token, "ffn_down.f32", w->down,
                        (uint64_t)GLM5_WIDTH * sizeof(float)) &&
           trace_tensor(ctx, layer, token, "output_hc.f32", hc_out,
                        (uint64_t)GLM5_HC_WIDTH * sizeof(float));
}

int ds4_glm5_next_embed_token(const ds4_glm5_next_exec_ctx *ctx,
                              uint32_t token,
                              ds4_gpu_tensor *hc_out) {
    if (!context_valid(ctx) || !hc_out || token >= GLM5_VOCAB) return 0;
    const uint32_t type = ctx->model->token_embd_type;
    if (type == 8u) {
        return ds4_gpu_embed_token_hc_q8_0_tensor(
            hc_out, ctx->model_map, ctx->model_size,
            ctx->model->token_embd, GLM5_VOCAB, token,
            GLM5_WIDTH, GLM5_HC);
    }
    if (type == 0u || type == 30u) {
        return ds4_gpu_embed_token_hc_bf16_tensor(
            hc_out, ctx->model_map, ctx->model_size,
            ctx->model->token_embd, GLM5_VOCAB, token,
            GLM5_WIDTH, GLM5_HC);
    }
    return 0;
}

int ds4_glm5_next_embed_tokens(const ds4_glm5_next_exec_ctx *ctx,
                               const ds4_gpu_tensor *tokens,
                               uint32_t n_tokens,
                               ds4_gpu_tensor *hc_out) {
    const uint64_t row_bytes = (uint64_t)GLM5_HC_WIDTH * sizeof(float);
    if (!context_valid(ctx) || !tokens || !hc_out || n_tokens == 0u ||
        (uint64_t)n_tokens > UINT64_MAX / row_bytes ||
        ds4_gpu_tensor_bytes(tokens) <
            (uint64_t)n_tokens * sizeof(uint32_t) ||
        ds4_gpu_tensor_bytes(hc_out) != (uint64_t)n_tokens * row_bytes) {
        return 0;
    }
    const uint32_t type = ctx->model->token_embd_type;
    if (type == 8u) {
        return ds4_gpu_embed_tokens_hc_q8_0_tensor(
            hc_out, tokens, ctx->model_map, ctx->model_size,
            ctx->model->token_embd, GLM5_VOCAB, n_tokens,
            GLM5_WIDTH, GLM5_HC);
    }
    if (type == 0u || type == 30u) {
        return ds4_gpu_embed_tokens_hc_bf16_tensor(
            hc_out, tokens, ctx->model_map, ctx->model_size,
            ctx->model->token_embd, GLM5_VOCAB, n_tokens,
            GLM5_WIDTH, GLM5_HC);
    }
    return 0;
}

int ds4_glm5_next_output_logits(const ds4_glm5_next_exec_ctx *ctx,
                                ds4_glm5_next_workspace *w,
                                const ds4_gpu_tensor *hc_hidden,
                                ds4_gpu_tensor *logits_out) {
    const uint64_t hc_bytes =
        (uint64_t)GLM5_HC_WIDTH * sizeof(float);
    const uint64_t logits_bytes =
        (uint64_t)GLM5_VOCAB * sizeof(float);
    return context_valid(ctx) && w && hc_hidden && logits_out &&
           ds4_gpu_tensor_bytes(hc_hidden) >= hc_bytes &&
           ds4_gpu_tensor_bytes(logits_out) >= logits_bytes &&
           ds4_gpu_hc_weighted_sum_tensor(
               w->output_hidden, hc_hidden, w->hc_mean_weights,
               GLM5_WIDTH, GLM5_HC) &&
           ds4_gpu_rms_norm_weight_tensor(
               w->output_norm, w->output_hidden,
               ctx->model_map, ctx->model_size, ctx->model->output_norm,
               GLM5_WIDTH, ctx->model->rms_norm_eps) &&
           ((ctx->model->output_type == 8u &&
             ds4_gpu_matmul_q8_0_tensor(
                 logits_out, ctx->model_map, ctx->model_size,
                 ctx->model->output, GLM5_WIDTH, GLM5_VOCAB,
                 w->output_norm, 1u)) ||
            ((ctx->model->output_type == 0u ||
              ctx->model->output_type == 30u) &&
             ds4_gpu_matmul_bf16_tensor(
                 logits_out, ctx->model_map, ctx->model_size,
                 ctx->model->output, GLM5_WIDTH, GLM5_VOCAB,
                 w->output_norm, 1u)));
}

static int tp_exchange_rows(const ds4_glm5_next_exec_ctx *ctx,
                            uint32_t layer, uint32_t gate,
                            uint32_t n_tokens);
static int tp_context_valid_bytes(const ds4_glm5_next_exec_ctx *ctx,
                                  uint64_t bytes);

static void kda_half_state_free(ds4_glm5_kda_layer_state *local) {
    if (!local) return;
    ds4_gpu_tensor_free(local->recurrent);
    ds4_gpu_tensor_free(local->v_history);
    ds4_gpu_tensor_free(local->k_history);
    ds4_gpu_tensor_free(local->q_history);
    memset(local, 0, sizeof(*local));
}

static int kda_half_state_view(ds4_glm5_kda_layer_state *local,
                               ds4_glm5_kda_layer_state *full,
                               uint32_t rank) {
    const uint64_t history_bytes =
        (uint64_t)(DS4_GLM5_KDA_CHANNELS / 2u) *
        DS4_GLM5_KDA_HISTORY * sizeof(float);
    const uint64_t recurrent_bytes =
        (uint64_t)(DS4_GLM5_KDA_HEADS / 2u) *
        DS4_GLM5_KDA_HEAD_DIM * DS4_GLM5_KDA_HEAD_DIM * sizeof(float);
    if (!local || !full || rank > 1u || !full->valid ||
        !full->q_history || !full->k_history || !full->v_history ||
        !full->recurrent) return 0;
    memset(local, 0, sizeof(*local));
    local->q_history = ds4_gpu_tensor_view(
        full->q_history, (uint64_t)rank * history_bytes, history_bytes);
    local->k_history = ds4_gpu_tensor_view(
        full->k_history, (uint64_t)rank * history_bytes, history_bytes);
    local->v_history = ds4_gpu_tensor_view(
        full->v_history, (uint64_t)rank * history_bytes, history_bytes);
    local->recurrent = ds4_gpu_tensor_view(
        full->recurrent, (uint64_t)rank * recurrent_bytes, recurrent_bytes);
    if (!local->q_history || !local->k_history || !local->v_history ||
        !local->recurrent) {
        kda_half_state_free(local);
        return 0;
    }
    local->token_count = full->token_count;
    local->valid = true;
    local->owner_slot = full->owner_slot;
    return 1;
}

static int kda_attention_rows(const ds4_glm5_next_exec_ctx *ctx,
                              uint32_t il,
                              ds4_glm5_next_state *state,
                              ds4_glm5_next_workspace *w,
                              const ds4_gpu_tensor *hc_in,
                              uint32_t n_tokens) {
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    ds4_glm5_kda_layer_state *kda = &state->kda.layer[il];
    const int prefix_ok =
        ds4_gpu_rms_norm_plain_rows_tensor(
            w->hc_flat, hc_in, GLM5_HC_WIDTH, n_tokens,
            ctx->model->rms_norm_eps) &&
        ds4_gpu_matmul_bf16_tensor(
            w->hc_mix, ctx->model_map, ctx->model_size,
            layer->hc.attn_fn, GLM5_HC_WIDTH, GLM5_HC_MIX,
            w->hc_flat, n_tokens) &&
        ds4_gpu_hc_split_weighted_sum_tensor(
            w->collapsed, w->hc_split, w->hc_mix, hc_in,
            ctx->model_map, ctx->model_size,
            layer->hc.attn_scale, layer->hc.attn_base,
            GLM5_WIDTH, GLM5_HC, 20u, ctx->model->hc_eps);
    if (!prefix_ok) return 0;
#ifdef DS4_TP_TEST_HOOKS
    /* Differential-only boundary capture.  The batch-vs-tokenwise fixture
     * enables this to identify whether drift begins before the recurrent
     * kernel; production never sets the trace environment. */
    if (getenv("DS4_GLM5_KDA_STAGE_TRACE") != NULL && ctx->trace_prefix) {
        const uint64_t hc_row = (uint64_t)GLM5_WIDTH * sizeof(float);
        const uint64_t mix_row = (uint64_t)GLM5_HC_MIX * sizeof(float);
        for (uint32_t t = 0u; t < n_tokens; ++t) {
            const uint32_t token = (uint32_t)kda->token_count + t;
            if (!trace_tensor_row(ctx, il, token, "hc_split.f32",
                                  w->hc_split, t, mix_row) ||
                !trace_tensor_row(ctx, il, token, "hc_collapsed.f32",
                                  w->collapsed, t, hc_row)) return 0;
        }
    }
#endif

    const uint32_t tp_features = ctx->tp ?
        ds4_tp_runtime_features(ctx->tp) : 0u;
    const int head_sharded =
        (tp_features & DS4_TP_FEATURE_GLM5_KDA_TP) != 0u;
    if (!head_sharded) {
        return ds4_glm5_kda_layer_forward(
                   kda, &w->kda, &layer->kda,
                   ctx->model_map, ctx->model_size,
                   w->collapsed, w->attention, n_tokens,
                   ctx->model->rms_norm_eps) &&
               ds4_gpu_hc_expand_split_tensor(
                   w->after_attention, w->attention, hc_in, w->hc_split,
                   GLM5_WIDTH, GLM5_HC);
    }

    /* Keep the compact gated half in device memory for the decode K-slice.
     * Reading it directly from the NIC-registerable host-mapped slab would
     * make every output block restage the same 16 KiB across the host link. */
    const int output_kslice =
        (tp_features & DS4_TP_FEATURE_GLM5_KDA_OUTPUT_KSLICE) != 0u &&
        (n_tokens == 1u ||
         (getenv("DS4_ROCM_GLM5_BATCH_KSLICE_OUTPUT") != NULL &&
          strcmp(getenv("DS4_ROCM_GLM5_BATCH_KSLICE_OUTPUT"), "1") == 0));
    if (output_kslice) {
        static int logged_output_kslice[2] = {0, 0};
        const uint32_t rank = ctx->tp_rank < 2u ? ctx->tp_rank : 0u;
        if (!logged_output_kslice[rank]) {
            fprintf(stderr,
                    "ds4: GLM5 KDA output K-slice engaged rank=%u "
                    "tokens=%u layer=%u\n",
                    ctx->tp_rank, n_tokens, il);
            logged_output_kslice[rank] = 1;
        }
    }
    if (output_kslice && layer->kda.output_type != 30u) {
        ds4_glm5_next_state_invalidate(state);
        return 0;
    }
    uint64_t k_off = 0u, k_cnt = 0u, local_bytes = 0u;
    ds4_glm5_kda_layer_state local = {};
    if (!kda_output_kslice_contract(
            ctx->tp_rank, n_tokens, &k_off, &k_cnt, &local_bytes) ||
        !tp_context_valid_bytes(ctx, local_bytes) ||
        !kda_half_state_view(&local, kda, ctx->tp_rank)) {
        ds4_glm5_next_state_invalidate(state);
        return 0;
    }
    ds4_gpu_tensor *const local_gated =
        output_kslice ? w->kda.recurrent_out : ctx->tp_big_out;
    const int local_ok = ds4_glm5_kda_layer_begin(
        &local, &w->kda, &layer->kda, ctx->model_map, ctx->model_size,
        w->collapsed, local_gated, n_tokens,
        ctx->model->rms_norm_eps,
        ctx->tp_rank * (DS4_GLM5_KDA_HEADS / 2u),
        DS4_GLM5_KDA_HEADS / 2u);
#ifdef DS4_TP_TEST_HOOKS
    if (local_ok && getenv("DS4_GLM5_KDA_STAGE_TRACE") != NULL &&
        ctx->trace_prefix) {
        const uint64_t row_bytes =
            (uint64_t)(DS4_GLM5_KDA_HEADS / 2u) *
            DS4_GLM5_KDA_HEAD_DIM * sizeof(float);
        for (uint32_t t = 0u; t < n_tokens; ++t) {
            if (!trace_tensor_row(ctx, il, (uint32_t)kda->token_count + t,
                                  "kda_local_gated.f32", local_gated, t,
                                  row_bytes)) return 0;
        }
    }
#endif
#ifdef DS4_TP_TEST_HOOKS
    /* Capture the half-head output before either output-projection route or
     * RDMA exchange.  This is deliberately test-only: it localizes stacked
     * K-slice drift without changing the production graph or payload. */
    if (local_ok && n_tokens == 1u) {
        if (!trace_tensor(
                ctx, il, (uint32_t)kda->token_count,
                "kda_local_gated.f32", local_gated,
                (uint64_t)(DS4_GLM5_KDA_HEADS / 2u) *
                    DS4_GLM5_KDA_HEAD_DIM * sizeof(float))) {
            ds4_glm5_kda_layer_abort(&local);
            kda_half_state_free(&local);
            ds4_glm5_next_state_invalidate(state);
            return 0;
        }
    }
#endif
    /* The strided half-row projection is production-enabled for decode.  Its
     * batched form remains opt-in until the route-consistency and Lane-B
     * quality gates establish a new deterministic prompt/decode trajectory. */
    if (local_ok && output_kslice) {
        const uint64_t output_values =
            (uint64_t)n_tokens * GLM5_WIDTH;
        if (output_values > UINT32_MAX ||
            !ds4_gpu_matmul_bf16_kslice_rows_tensor(
                w->attention, ctx->model_map, ctx->model_size,
                layer->kda.output, DS4_GLM5_KDA_CHANNELS, GLM5_WIDTH,
                k_off, k_cnt, w->kda.recurrent_out, n_tokens) ||
            !ds4_gpu_tensor_copy(
                ctx->tp_big_out, 0u, w->attention, 0u,
                output_values * sizeof(float))) {
            ds4_glm5_kda_layer_abort(&local);
            kda_half_state_free(&local);
            ds4_glm5_next_state_invalidate(state);
            return 0;
        }
    }
    if (!local_ok ||
        !tp_exchange_rows(ctx, il, DS4_TP_GATE_ATTN, n_tokens)) {
        ds4_glm5_kda_layer_abort(&local);
        kda_half_state_free(&local);
        ds4_glm5_next_state_invalidate(state);
        return 0;
    }
    if (output_kslice) {
        const ds4_gpu_tensor *rank0_partial =
            ctx->tp_rank == 0u ? w->attention : ctx->tp_big_in;
        const ds4_gpu_tensor *rank1_partial =
            ctx->tp_rank == 0u ? ctx->tp_big_in : w->attention;
        const uint64_t output_values =
            (uint64_t)n_tokens * GLM5_WIDTH;
        const int suffix_ok =
            ds4_gpu_add_tensor(
                w->kda.recurrent_out, rank0_partial, rank1_partial,
                (uint32_t)output_values) &&
            ds4_glm5_kda_layer_commit(&local, n_tokens) &&
            ds4_gpu_hc_expand_split_tensor(
                w->after_attention, w->kda.recurrent_out,
                hc_in, w->hc_split,
                GLM5_WIDTH, GLM5_HC);
        if (suffix_ok) kda->token_count = local.token_count;
        else ds4_glm5_kda_layer_abort(&local);
        kda_half_state_free(&local);
        if (!suffix_ok) ds4_glm5_next_state_invalidate(state);
        return suffix_ok;
    }
    const ds4_gpu_tensor *rank0 =
        ctx->tp_rank == 0u ? ctx->tp_big_out : ctx->tp_big_in;
    const ds4_gpu_tensor *rank1 =
        ctx->tp_rank == 0u ? ctx->tp_big_in : ctx->tp_big_out;
    const int suffix_ok =
        ds4_glm5_kda_compose_head_halves(
            w->kda.recurrent_out, rank0, rank1, n_tokens) &&
        ds4_glm5_kda_layer_finish(
            &local, &layer->kda, ctx->model_map, ctx->model_size,
            w->kda.recurrent_out, w->attention, n_tokens) &&
        ds4_gpu_hc_expand_split_tensor(
            w->after_attention, w->attention, hc_in, w->hc_split,
            GLM5_WIDTH, GLM5_HC);
    if (suffix_ok) kda->token_count = local.token_count;
    else ds4_glm5_kda_layer_abort(&local);
    kda_half_state_free(&local);
    if (!suffix_ok) ds4_glm5_next_state_invalidate(state);
    return suffix_ok;
}

static int kda_attention_one(const ds4_glm5_next_exec_ctx *ctx,
                             uint32_t il,
                             ds4_glm5_next_state *state,
                             ds4_glm5_next_workspace *w,
                             const ds4_gpu_tensor *hc_in) {
    return kda_attention_rows(ctx, il, state, w, hc_in, 1u);
}

#ifdef DS4_TP_TEST_HOOKS
static const ds4_gpu_tensor *kda_attention_result_for_trace(
        const ds4_glm5_next_exec_ctx *ctx,
        const ds4_glm5_next_workspace *w,
        uint32_t n_tokens) {
    const char *batch_kslice =
        getenv("DS4_ROCM_GLM5_BATCH_KSLICE_OUTPUT");
    const int output_kslice = ctx && ctx->tp &&
        (ds4_tp_runtime_features(ctx->tp) &
         DS4_TP_FEATURE_GLM5_KDA_OUTPUT_KSLICE) != 0u &&
        (n_tokens == 1u ||
         (batch_kslice != NULL && strcmp(batch_kslice, "1") == 0));
    return output_kslice ? w->kda.recurrent_out : w->attention;
}
#endif

static int dense_kda_forward_rows(const ds4_glm5_next_exec_ctx *ctx,
                                  uint32_t il,
                                  ds4_glm5_next_state *state,
                                  ds4_glm5_next_workspace *w,
                                  const ds4_gpu_tensor *hc_in,
                                  ds4_gpu_tensor *hc_out,
                                  uint32_t n_tokens) {
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    const uint32_t token_ordinal =
        state->kda.layer[il].token_count <= UINT32_MAX ?
        (uint32_t)state->kda.layer[il].token_count : UINT32_MAX;
    const int finite_debug = getenv("DS4_GLM5_NEXT_VALIDATE_FINITE") != NULL;
    if (finite_debug) route_failure_stats("dense_hc_in", hc_in,
                                          n_tokens * GLM5_HC_WIDTH);
    int ok = kda_attention_rows(ctx, il, state, w, hc_in, n_tokens);
#ifdef DS4_TP_TEST_HOOKS
    if (ok && n_tokens == 1u) {
        ok = trace_tensor(
                 ctx, il, token_ordinal, "kda_out.f32",
                 kda_attention_result_for_trace(ctx, w, n_tokens),
                 (uint64_t)GLM5_WIDTH * sizeof(float)) &&
             trace_tensor(
                 ctx, il, token_ordinal, "after_attn.f32",
                 w->after_attention,
                 (uint64_t)GLM5_HC_WIDTH * sizeof(float));
    }
#endif
    if (finite_debug) {
        route_failure_stats("dense_attention", w->attention,
                            n_tokens * GLM5_WIDTH);
        route_failure_stats("dense_after_attention", w->after_attention,
                            n_tokens * GLM5_HC_WIDTH);
    }
    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(
            w->ffn_flat, w->after_attention, GLM5_HC_WIDTH, n_tokens,
            ctx->model->rms_norm_eps);
    if (ok) ok = ds4_gpu_matmul_bf16_tensor(
            w->ffn_mix, ctx->model_map, ctx->model_size,
            layer->hc.ffn_fn, GLM5_HC_WIDTH, GLM5_HC_MIX,
            w->ffn_flat, n_tokens);
    if (ok) ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(
            w->ffn_collapsed, w->ffn_hidden, w->ffn_split, w->ffn_mix,
            w->after_attention, ctx->model_map, ctx->model_size,
            layer->hc.ffn_scale, layer->hc.ffn_base, layer->ffn_norm,
            GLM5_WIDTH, GLM5_HC, 20u, ctx->model->hc_eps,
            ctx->model->rms_norm_eps);
    if (finite_debug) route_failure_stats("dense_ffn_hidden", w->ffn_hidden,
                                          n_tokens * GLM5_WIDTH);
    if (ok) ok = ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
            w->gate, w->up, w->mid, ctx->model_map, ctx->model_size,
            layer->ffn_weight.gate, layer->ffn_weight.up,
            GLM5_WIDTH, GLM5_DENSE_MID, w->ffn_hidden, n_tokens, 10.0f);
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(
            w->down, ctx->model_map, ctx->model_size,
            layer->ffn_weight.down, GLM5_DENSE_MID, GLM5_WIDTH,
            w->mid, n_tokens);
    if (finite_debug) route_failure_stats("dense_ffn_down", w->down,
                                          n_tokens * GLM5_WIDTH);
    if (ok) ok = ds4_gpu_hc_expand_split_tensor(
            hc_out, w->down, w->after_attention, w->ffn_split,
            GLM5_WIDTH, GLM5_HC);
#ifdef DS4_TP_TEST_HOOKS
    if (ok && n_tokens == 1u)
        ok = trace_tensor(
            ctx, il, token_ordinal, "output_hc.f32", hc_out,
            (uint64_t)GLM5_HC_WIDTH * sizeof(float));
#endif
    if (ok) ok = ds4_gpu_synchronize();
    if (!ok) ds4_glm5_next_state_invalidate(state);
    return ok;
}

static int dense_kda_forward(const ds4_glm5_next_exec_ctx *ctx,
                             uint32_t il,
                             ds4_glm5_next_state *state,
                             ds4_glm5_next_workspace *w,
                             const ds4_gpu_tensor *hc_in,
                             ds4_gpu_tensor *hc_out) {
    return dense_kda_forward_rows(ctx, il, state, w, hc_in, hc_out, 1u);
}

static int tp_context_valid_bytes(const ds4_glm5_next_exec_ctx *ctx,
                                  uint64_t bytes) {
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

static int tp_context_valid(const ds4_glm5_next_exec_ctx *ctx) {
    return tp_context_valid_bytes(
        ctx, (uint64_t)GLM5_WIDTH * sizeof(float));
}

static int tp_exchange_rows(const ds4_glm5_next_exec_ctx *ctx,
                            uint32_t layer, uint32_t gate,
                            uint32_t n_tokens) {
    if (n_tokens == 0u) return 0;
    const uint64_t bytes =
        (uint64_t)n_tokens * GLM5_WIDTH * sizeof(float);
    if (!tp_context_valid_bytes(ctx, bytes) || gate >= DS4_TP_GATES_PER_LAYER ||
        *ctx->tp_sequence == UINT64_MAX) return 0;

    const int small_gate_requested = n_tokens == 1u &&
        (ds4_tp_runtime_features(ctx->tp) &
         DS4_TP_FEATURE_GLM5_SMALL_GATE) != 0u;
    if (small_gate_requested) {
        if (!ctx->tp_slab || bytes != ds4_tp_vec_bytes(ctx->tp)) {
            ds4_tp_mark_failed(ctx->tp);
            return 0;
        }
        const uint64_t out_off =
            ds4_tp_slab_out_offset(ctx->tp, layer, gate);
        const uint64_t in_off =
            ds4_tp_slab_in_offset(ctx->tp, layer, gate);
        if (ds4_gpu_tensor_bytes(ctx->tp_slab) < out_off + bytes ||
            ds4_gpu_tensor_bytes(ctx->tp_slab) < in_off + bytes ||
            !ds4_gpu_tensor_copy(ctx->tp_slab, out_off,
                                 ctx->tp_big_out, 0u, bytes) ||
            !ds4_gpu_synchronize()) {
            ds4_tp_mark_failed(ctx->tp);
            return 0;
        }
        const uint64_t sequence = ++*ctx->tp_sequence;
        if (!ds4_tp_gate_exchange(ctx->tp, layer, gate, sequence)) {
            ds4_tp_mark_failed(ctx->tp);
            return 0;
        }
        if (!ds4_gpu_tensor_copy(ctx->tp_big_in, 0u,
                                 ctx->tp_slab, in_off, bytes)) {
            ds4_tp_mark_failed(ctx->tp);
            return 0;
        }
        static int reported;
        if (!reported) {
            fprintf(stderr,
                    "ds4: GLM5 one-token TP reductions use the "
                    "preposted latency gate (%llu bytes)\n",
                    (unsigned long long)bytes);
            reported = 1;
        }
        return 1;
    }

    if (!ds4_gpu_synchronize()) return 0;
    const char *cache_fence = getenv("DS4_ROCM_RDMA_CACHE_FENCE");
    const int fence_release = cache_fence &&
        (strcmp(cache_fence, "release") == 0 ||
         strcmp(cache_fence, "both") == 0);
    const int fence_acquire = cache_fence &&
        (strcmp(cache_fence, "acquire") == 0 ||
         strcmp(cache_fence, "both") == 0);
    if (cache_fence && !fence_release && !fence_acquire) return 0;
    if (fence_release && !ds4_rocm_rdma_cache_release()) return 0;
    const uint64_t sequence = ++*ctx->tp_sequence;
    if (!ds4_tp_big_gate_exchange(ctx->tp, layer, sequence,
                                  ctx->tp_big_out_host,
                                  ctx->tp_big_in_host, bytes)) {
        ds4_tp_mark_failed(ctx->tp);
        return 0;
    }
    if (fence_acquire && !ds4_rocm_rdma_cache_acquire()) {
        ds4_tp_mark_failed(ctx->tp);
        return 0;
    }
    return 1;
}

static int tp_exchange(const ds4_glm5_next_exec_ctx *ctx,
                       uint32_t layer, uint32_t gate) {
    return tp_exchange_rows(ctx, layer, gate, 1u);
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

#ifdef DS4_TP_TEST_HOOKS
typedef struct {
    const char *name;
    uint32_t row_floats;
} glm5_mla_capture_stage;

static const glm5_mla_capture_stage g_glm5_mla_capture_stages[] = {
    {"input_hc", GLM5_HC_WIDTH},
    {"compact_kv", GLM5_KV_LORA},
    {"qk_low", GLM5_HEADS * GLM5_KV_LORA},
    {"lora_out", GLM5_HEADS * GLM5_KV_LORA},
    {"heads", GLM5_HEADS * GLM5_HEAD_DIM},
    {"attn_local", GLM5_WIDTH},
    {"attn_peer", GLM5_WIDTH},
    {"attn_sum", GLM5_WIDTH},
    {"after_attn", GLM5_HC_WIDTH},
};

uint64_t ds4_glm5_next_mla_stage_capture_bytes(uint32_t n_tokens) {
    if (n_tokens == 0u) return 0u;
    uint64_t row_floats = 0u;
    for (uint32_t i = 0u;
         i < sizeof(g_glm5_mla_capture_stages) /
                 sizeof(g_glm5_mla_capture_stages[0]); ++i) {
        row_floats += g_glm5_mla_capture_stages[i].row_floats;
    }
    if (row_floats > UINT64_MAX / sizeof(float) ||
        (uint64_t)n_tokens > UINT64_MAX /
            (row_floats * sizeof(float))) return 0u;
    return (uint64_t)n_tokens * row_floats * sizeof(float);
}

int ds4_glm5_next_mla_stage_capture_dump(
        const ds4_gpu_tensor *capture, uint32_t n_tokens, FILE *stream) {
    const uint64_t bytes =
        ds4_glm5_next_mla_stage_capture_bytes(n_tokens);
    if (!capture || !stream || bytes == 0u || bytes > SIZE_MAX ||
        ds4_gpu_tensor_bytes(capture) != bytes) return 0;
    unsigned char *host = (unsigned char *)malloc((size_t)bytes);
    if (!host || !ds4_gpu_tensor_read(
            (ds4_gpu_tensor *)capture, 0u, host, bytes)) {
        free(host);
        return 0;
    }
    uint64_t offset = 0u;
    for (uint32_t i = 0u;
         i < sizeof(g_glm5_mla_capture_stages) /
                 sizeof(g_glm5_mla_capture_stages[0]); ++i) {
        const uint64_t row_bytes =
            (uint64_t)g_glm5_mla_capture_stages[i].row_floats *
            sizeof(float);
        const uint64_t stage_bytes = (uint64_t)n_tokens * row_bytes;
        fprintf(stream,
                "GLM5 device MLA stage capture stage=%s hash=%016llx\n",
                g_glm5_mla_capture_stages[i].name,
                (unsigned long long)fnv64_continue(
                    UINT64_C(1469598103934665603), host + offset,
                    stage_bytes));
        for (uint32_t row = 0u; row < n_tokens; ++row) {
            fprintf(stream,
                    "GLM5 device MLA stage row capture stage=%s row=%u "
                    "hash=%016llx\n",
                    g_glm5_mla_capture_stages[i].name, row,
                    (unsigned long long)fnv64_continue(
                        UINT64_C(1469598103934665603),
                        host + offset + (uint64_t)row * row_bytes,
                        row_bytes));
        }
        offset += stage_bytes;
    }
    free(host);
    return offset == bytes;
}

static int glm5_capture_mla_stages(
        const ds4_glm5_next_exec_ctx *ctx, uint32_t il,
        const ds4_glm5_next_mla_state *mla,
        const ds4_glm5_next_workspace *w,
        const ds4_gpu_tensor *hc_in, uint32_t pos0, uint32_t n_tokens) {
    if (!ctx->device_mla_stage_capture) return 1;
    const uint64_t expected =
        ds4_glm5_next_mla_stage_capture_bytes(n_tokens);
    if (il != ctx->device_mla_stage_capture_layer) return 1;
    if (!mla || !w || !hc_in || expected == 0u ||
        ctx->device_mla_stage_capture_bytes != expected ||
        ds4_gpu_tensor_bytes(ctx->device_mla_stage_capture) != expected)
        return 0;
    const ds4_gpu_tensor *sources[] = {
        hc_in, mla->compact_kv, w->mla_qk_low, w->routed_experts,
        w->mla_heads, ctx->tp_big_out, ctx->tp_big_in, w->attention,
        w->after_attention,
    };
    const uint32_t source_row_base[] = {
        0u, pos0, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    };
    uint64_t offset = 0u;
    for (uint32_t i = 0u;
         i < sizeof(g_glm5_mla_capture_stages) /
                 sizeof(g_glm5_mla_capture_stages[0]); ++i) {
        const uint64_t row_bytes =
            (uint64_t)g_glm5_mla_capture_stages[i].row_floats *
            sizeof(float);
        const uint64_t stage_bytes = (uint64_t)n_tokens * row_bytes;
        if (!sources[i] || ds4_gpu_tensor_bytes(sources[i]) <
                ((uint64_t)source_row_base[i] + n_tokens) * row_bytes ||
            !ds4_gpu_tensor_copy(ctx->device_mla_stage_capture, offset,
                                 sources[i],
                                 (uint64_t)source_row_base[i] * row_bytes,
                                 stage_bytes)) return 0;
        offset += stage_bytes;
    }
    return offset == expected;
}
#endif

#ifdef DS4_TP_TEST_HOOKS
static int hc_batch_hash_trace(const ds4_glm5_next_exec_ctx *ctx,
                               uint32_t layer,
                               const ds4_gpu_tensor *hc,
                               uint32_t n_tokens) {
    if (getenv("DS4_GLM5_HC_HASH_TRACE") == NULL) return 1;
    float row[GLM5_HC_WIDTH];
    const uint64_t row_bytes = sizeof(row);
    if (!ctx || !hc || layer >= GLM5_ROUTE_TRACE_LAYERS ||
        n_tokens == 0u || ds4_gpu_tensor_bytes(hc) <
            (uint64_t)n_tokens * row_bytes ||
        !ds4_gpu_tensor_read((ds4_gpu_tensor *)hc,
                             (uint64_t)(n_tokens - 1u) * row_bytes,
                             row, row_bytes)) return 0;
    if (!g_glm5_hc_trace_registered) {
        if (atexit(glm5_hc_trace_dump) != 0) return 0;
        g_glm5_hc_trace_registered = 1;
    }
    g_glm5_hc_trace_hash[layer] = fnv64_continue(
        UINT64_C(1469598103934665603), row, row_bytes);
    g_glm5_hc_trace_seen[layer] = 1u;
    return 1;
}

static int layer_completion_diagnostic(const ds4_gpu_tensor *hc,
                                       uint32_t n_tokens) {
    const char *mode = getenv("DS4_GLM5_LAYER_COMPLETION_DIAGNOSTIC");
    if (!mode) return 1;
    if (strcmp(mode, "sync") == 0) return ds4_gpu_synchronize();
    if (strcmp(mode, "read1") == 0) {
        float value = 0.0f;
        const uint64_t row_bytes =
            (uint64_t)GLM5_HC_WIDTH * sizeof(float);
        return hc && n_tokens != 0u && ds4_gpu_tensor_bytes(hc) >=
                   (uint64_t)n_tokens * row_bytes &&
               ds4_gpu_tensor_read(
                   hc, (uint64_t)(n_tokens - 1u) * row_bytes,
                   &value, sizeof(value));
    }
    if (strcmp(mode, "readfirst") == 0) {
        float value = 0.0f;
        return hc && n_tokens != 0u &&
               ds4_gpu_tensor_bytes(hc) >= sizeof(value) &&
               ds4_gpu_tensor_read(hc, 0u, &value, sizeof(value));
    }
    if (strcmp(mode, "readdummy") == 0) {
        static ds4_gpu_tensor *dummy;
        if (!dummy) {
            const float zero = 0.0f;
            dummy = ds4_gpu_tensor_alloc(sizeof(zero));
            if (!dummy || !ds4_gpu_tensor_write(
                    dummy, 0u, &zero, sizeof(zero))) return 0;
        }
        float value = 0.0f;
        return ds4_gpu_tensor_read(dummy, 0u, &value, sizeof(value));
    }
    if (strcmp(mode, "d2ddummy") == 0) {
        static ds4_gpu_tensor *src;
        static ds4_gpu_tensor *dst;
        if (!src || !dst) {
            const float zero = 0.0f;
            src = ds4_gpu_tensor_alloc(sizeof(zero));
            dst = ds4_gpu_tensor_alloc(sizeof(zero));
            if (!src || !dst || !ds4_gpu_tensor_write(
                    src, 0u, &zero, sizeof(zero))) return 0;
        }
        return ds4_gpu_tensor_copy(dst, 0u, src, 0u, sizeof(float)) &&
               ds4_gpu_synchronize();
    }
    if (strncmp(mode, "delay-us-", 9u) == 0) {
        char *end = NULL;
        const unsigned long usec = strtoul(mode + 9u, &end, 10);
        if (!end || *end != '\0' || usec == 0u || usec > 1000000u)
            return 0;
        const struct timespec delay = {
            .tv_sec = (time_t)(usec / 1000000u),
            .tv_nsec = (long)(usec % 1000000u) * 1000L,
        };
        return nanosleep(&delay, NULL) == 0;
    }
    return 0;
}
#endif

static int route_failure_stats(const char *name, const ds4_gpu_tensor *tensor,
                               uint32_t count) {
    float *values = malloc((size_t)count * sizeof(*values));
    if (!values || !ds4_gpu_tensor_read(tensor, 0u, values,
                                         (uint64_t)count * sizeof(*values))) {
        fprintf(stderr, "ds4: GLM5 route diagnostic %s unreadable\n", name);
        free(values);
        return 0;
    }
    uint32_t nonfinite = 0u;
    float min_value = INFINITY, max_value = -INFINITY;
    for (uint32_t i = 0u; i < count; ++i) {
        if (!isfinite(values[i])) { nonfinite++; continue; }
        if (values[i] < min_value) min_value = values[i];
        if (values[i] > max_value) max_value = values[i];
    }
    fprintf(stderr,
            "ds4: GLM5 route diagnostic %s count=%u nonfinite=%u min=%g max=%g\n",
            name, count, nonfinite, min_value, max_value);
    free(values);
    return nonfinite == 0u;
}

static int validate_layer_finite(const ds4_glm5_next_exec_ctx *ctx,
                                 uint32_t layer,
                                 const ds4_gpu_tensor *output) {
    if (!getenv("DS4_GLM5_NEXT_VALIDATE_FINITE")) return 1;
    const int finite = route_failure_stats("layer_hc_out", output,
                                           GLM5_HC_WIDTH);
    if (!finite) {
        fprintf(stderr,
                "ds4: GLM5 finite gate failed layer=%u rank=%u\n",
                layer, ctx->tp_rank);
    }
    return finite;
}

static int route_agrees(const ds4_glm5_next_exec_ctx *ctx, uint32_t layer,
                        uint32_t token_ordinal,
                        const ds4_gpu_tensor *selected,
                        const ds4_gpu_tensor *weights,
                        const ds4_gpu_tensor *logits,
                        const ds4_gpu_tensor *hidden) {
    int32_t ids[GLM5_EXPERTS_USED];
    float route_weights[GLM5_EXPERTS_USED];
    if (!tp_context_valid(ctx)) {
        fprintf(stderr, "ds4: GLM5 route agreement invalid TP context layer=%u\n", layer);
        return 0;
    }
    if (!ds4_gpu_tensor_read(selected, 0u, ids, sizeof(ids)) ||
        !ds4_gpu_tensor_read(weights, 0u, route_weights,
                             sizeof(route_weights))) {
        fprintf(stderr, "ds4: GLM5 route agreement tensor read failed layer=%u rank=%u\n",
                layer, ctx->tp_rank);
        return 0;
    }
    for (uint32_t i = 0u; i < GLM5_EXPERTS_USED; ++i) {
        if (ids[i] < 0 || ids[i] >= GLM5_EXPERTS ||
            !isfinite(route_weights[i]) || route_weights[i] < 0.0f) {
            fprintf(stderr,
                    "ds4: GLM5 route agreement invalid slot layer=%u rank=%u slot=%u id=%d weight=%g\n",
                    layer, ctx->tp_rank, i, ids[i], route_weights[i]);
            route_failure_stats("router_logits", logits, GLM5_EXPERTS);
            route_failure_stats("ffn_hidden", hidden, GLM5_WIDTH);
            return 0;
        }
        for (uint32_t j = 0u; j < i; ++j)
            if (ids[i] == ids[j]) {
                fprintf(stderr,
                        "ds4: GLM5 route agreement duplicate expert layer=%u rank=%u slots=%u/%u id=%d\n",
                        layer, ctx->tp_rank, j, i, ids[i]);
                return 0;
            }
    }
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = fnv64_continue(hash, ids, sizeof(ids));
    hash = fnv64_continue(hash, route_weights, sizeof(route_weights));
    const uint64_t sequence_fields[] = {
        UINT64_C(0x474c4d3500000000),
        *ctx->tp_sequence,
        (uint64_t)layer,
        (uint64_t)token_ordinal,
        UINT64_C(1),
    };
    const uint64_t check_sequence = fnv64_continue(
        UINT64_C(1469598103934665603), sequence_fields,
        sizeof(sequence_fields));
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

static int route_batch_agrees(const ds4_glm5_next_exec_ctx *ctx,
                              uint32_t layer, uint32_t token_ordinal,
                              const ds4_gpu_tensor *selected,
                              const ds4_gpu_tensor *weights,
                              uint32_t n_tokens) {
    enum { ROUTE_HASH_ROWS = 256 };
    int32_t ids[ROUTE_HASH_ROWS * GLM5_EXPERTS_USED];
    float route_weights[ROUTE_HASH_ROWS * GLM5_EXPERTS_USED];
    if (!tp_context_valid(ctx) || !selected || !weights || n_tokens == 0u ||
        token_ordinal > UINT32_MAX - (n_tokens - 1u)) return 0;
    uint64_t hash = UINT64_C(1469598103934665603);
    int ok = 1;
    for (uint32_t row = 0u; ok && row < n_tokens;
         row += ROUTE_HASH_ROWS) {
        const uint32_t rows = n_tokens - row < ROUTE_HASH_ROWS ?
            n_tokens - row : ROUTE_HASH_ROWS;
        const uint64_t count = (uint64_t)rows * GLM5_EXPERTS_USED;
        const uint64_t bytes = count * sizeof(ids[0]);
        const uint64_t offset =
            (uint64_t)row * GLM5_EXPERTS_USED * sizeof(ids[0]);
        if (!ds4_gpu_tensor_read(selected, offset, ids, bytes)) {
            ds4_tp_mark_failed(ctx->tp);
            return 0;
        }
        for (uint32_t t = 0u; ok && t < rows; ++t) {
            const uint32_t base = t * GLM5_EXPERTS_USED;
            for (uint32_t i = 0u; ok && i < GLM5_EXPERTS_USED; ++i) {
                const int32_t id = ids[base + i];
                if (id < 0 || id >= GLM5_EXPERTS) {
                    ok = 0;
                    break;
                }
                for (uint32_t j = 0u; j < i; ++j)
                    if (id == ids[base + j]) ok = 0;
            }
        }
        if (ok) hash = fnv64_continue(hash, ids, bytes);
    }
    for (uint32_t row = 0u; ok && row < n_tokens;
         row += ROUTE_HASH_ROWS) {
        const uint32_t rows = n_tokens - row < ROUTE_HASH_ROWS ?
            n_tokens - row : ROUTE_HASH_ROWS;
        const uint64_t count = (uint64_t)rows * GLM5_EXPERTS_USED;
        const uint64_t bytes = count * sizeof(route_weights[0]);
        const uint64_t offset =
            (uint64_t)row * GLM5_EXPERTS_USED *
            sizeof(route_weights[0]);
        if (!ds4_gpu_tensor_read(
                weights, offset, route_weights, bytes)) {
            ds4_tp_mark_failed(ctx->tp);
            return 0;
        }
        for (uint64_t i = 0u; ok && i < count; ++i) {
            if (!isfinite(route_weights[i]) || route_weights[i] < 0.0f)
                ok = 0;
        }
        if (ok) hash = fnv64_continue(hash, route_weights, bytes);
    }
    if (!ok) {
        ds4_tp_mark_failed(ctx->tp);
        return 0;
    }
#ifdef DS4_TP_TEST_HOOKS
    /* Record without adding a GPU operation or a per-layer stdio delay. The
     * route path already copied these bytes to the host for rank agreement;
     * dumping them at exit lets transport A/B runs localize the first changed
     * routing boundary without introducing another synchronization point. */
    if (n_tokens > 1u && layer < GLM5_ROUTE_TRACE_LAYERS &&
        getenv("DS4_GLM5_ROUTE_HASH_TRACE") != NULL) {
        if (!g_glm5_route_trace_registered) {
            if (atexit(glm5_route_trace_dump) == 0)
                g_glm5_route_trace_registered = 1;
        }
        g_glm5_route_trace_hash[layer] = hash;
        g_glm5_route_trace_seen[layer] = 1u;
    }
#endif
    const uint64_t sequence_fields[] = {
        UINT64_C(0x474c4d3542415400),
        *ctx->tp_sequence,
        (uint64_t)layer,
        (uint64_t)token_ordinal,
        (uint64_t)n_tokens,
    };
    const uint64_t check_sequence = fnv64_continue(
        UINT64_C(1469598103934665603), sequence_fields,
        sizeof(sequence_fields));
    char error[256] = {0};
    const int rc = ds4_tp_hash_check(ctx->tp, check_sequence, hash,
                                     error, sizeof(error));
    if (rc != 1) {
        fprintf(stderr, "ds4: GLM5 batch route agreement failed: %s\n",
                error[0] ? error : "invalid local batch route");
        ds4_tp_mark_failed(ctx->tp);
        return 0;
    }
    return 1;
}

static int declare_local_q4k_half_only(
        const ds4_glm5_next_exec_ctx *ctx,
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
               DS4_GPU_Q4K_PACKED_K_RANGE);
}

/* Return 1 only when the exact gate/up/down triple is already materialized,
 * 0 when none is resident, and -1 for a partial/inconsistent triple.  The
 * distinction is load-bearing: a layer-local streaming cache owns and releases
 * its descriptors, while full-trunk residency must survive every layer/token. */
static int local_q4k_half_residency(
        const ds4_glm5_next_exec_ctx *ctx,
        const ds4_glm5_next_layer_offsets *layer) {
    const uint64_t gate_row_bytes =
        (GLM5_WIDTH / GLM5_Q4K_QK) * GLM5_Q4K_BLOCK_BYTES;
    const uint64_t down_row_bytes =
        (GLM5_ROUTED_MID / GLM5_Q4K_QK) * GLM5_Q4K_BLOCK_BYTES;
    const uint64_t down_half_bytes =
        (GLM5_RANK_MID / GLM5_Q4K_QK) * GLM5_Q4K_BLOCK_BYTES;
    const uint32_t row_base = ctx->tp_rank * GLM5_RANK_MID;
    const uint64_t column_base = (uint64_t)ctx->tp_rank * down_half_bytes;
    const void *device = NULL;
    uint64_t packed = 0u, expert = 0u, row = 0u;
    const int gate = ds4_gpu_q4k_packed_slice_resolve(
        ctx->model_map, layer->ffn_weight.gate_exps, GLM5_EXPERTS,
        GLM5_ROUTED_MID, gate_row_bytes, row_base, GLM5_RANK_MID,
        0u, gate_row_bytes, DS4_GPU_Q4K_PACKED_ROW_RANGE,
        &device, &packed, &expert, &row);
    const int up = ds4_gpu_q4k_packed_slice_resolve(
        ctx->model_map, layer->ffn_weight.up_exps, GLM5_EXPERTS,
        GLM5_ROUTED_MID, gate_row_bytes, row_base, GLM5_RANK_MID,
        0u, gate_row_bytes, DS4_GPU_Q4K_PACKED_ROW_RANGE,
        &device, &packed, &expert, &row);
    const int down = ds4_gpu_q4k_packed_slice_resolve(
        ctx->model_map, layer->ffn_weight.down_exps, GLM5_EXPERTS,
        GLM5_WIDTH, down_row_bytes, 0u, GLM5_WIDTH,
        column_base, down_half_bytes, DS4_GPU_Q4K_PACKED_K_RANGE,
        &device, &packed, &expert, &row);
    const int loaded = gate + up + down;
    return loaded == 0 ? 0 : loaded == 3 ? 1 : -1;
}

static int declare_local_q4k_half(const ds4_glm5_next_exec_ctx *ctx,
                                  const ds4_glm5_next_layer_offsets *layer) {
    const uint64_t gate_row_bytes =
        (GLM5_WIDTH / GLM5_Q4K_QK) * GLM5_Q4K_BLOCK_BYTES;
    const uint64_t down_half_bytes =
        (GLM5_RANK_MID / GLM5_Q4K_QK) * GLM5_Q4K_BLOCK_BYTES;
    const uint32_t row_base = ctx->tp_rank * GLM5_RANK_MID;
    const uint64_t column_base = (uint64_t)ctx->tp_rank * down_half_bytes;
    return declare_local_q4k_half_only(ctx, layer) &&
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

static int mla_publish_completed_pool(
        const ds4_glm5_next_exec_ctx *ctx,
        const ds4_glm5_next_mla_offsets *offsets,
        ds4_glm5_next_mla_state *mla,
        ds4_glm5_next_workspace *w,
        uint32_t pool,
        bool publish_pool) {
    if (!publish_pool) return 1;
    if (!mla->index_pool || pool >= mla->capacity_pools) return 0;
    const uint64_t row_bytes = (uint64_t)GLM5_INDEX_DIM * sizeof(float);
    ds4_gpu_tensor *output = ds4_gpu_tensor_view(
        mla->index_pool, (uint64_t)pool * row_bytes, row_bytes);
    if (!output) return 0;
    const int ok = ds4_gpu_glm5_kpool_tensor(
        output, w->mla_pool_indices, w->mla_pool_valid,
        mla->index_tail, mla->pool_gate_tail, w->mla_tail_valid,
        ctx->model_map, ctx->model_size, offsets->index_pool_ape,
        4u, GLM5_INDEX_DIM, 4u, 0u);
    ds4_gpu_tensor_free(output);
    if (!ok || !mla->index_pool_ids || !mla->index_pool_valid)
        return 0;
    /* A committed pool is always the four contiguous rows immediately before
     * the current tail.  Record this only after the pooled key kernel has
     * completed successfully, preserving a single publication boundary. */
    return ds4_gpu_glm5_fill_pool_members_tensor(
        mla->index_pool_ids, mla->index_pool_valid, pool, pool * 4u,
        mla->capacity_tokens);
}

/* The official selector uses the full visible range through top-k. Pooled
 * selection begins only when visible exceeds 2048 and is a separate path. */
static int mla_dense_selection_attention(const ds4_glm5_next_exec_ctx *ctx,
                                       uint32_t il,
                                       ds4_glm5_next_state *state,
                                       ds4_glm5_next_workspace *w,
                                       const ds4_gpu_tensor *hc_in,
                                       uint32_t visible,
                                       uint32_t tail_slot,
                                       uint32_t pool_index,
                                       bool publish_pool) {
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    const ds4_glm5_next_mla_offsets *m = &layer->mla;
    ds4_glm5_next_mla_state *mla = &state->mla[il];
    const uint32_t pos = mla->token_count;
    const uint64_t half_heads =
        ((uint64_t)GLM5_HEADS * GLM5_HEAD_DIM) / 2u;
    return
        ds4_gpu_rms_norm_plain_rows_tensor(
            w->hc_flat, hc_in, GLM5_HC_WIDTH, 1u,
            ctx->model->rms_norm_eps) &&
        ds4_gpu_matmul_bf16_tensor(
            w->hc_mix, ctx->model_map, ctx->model_size,
            layer->hc.attn_fn, GLM5_HC_WIDTH, GLM5_HC_MIX,
            w->hc_flat, 1u) &&
        ds4_gpu_hc_split_weighted_sum_norm_tensor(
            w->collapsed, w->ffn_hidden, w->hc_split, w->hc_mix, hc_in,
            ctx->model_map, ctx->model_size,
            layer->hc.attn_scale, layer->hc.attn_base, layer->attn_norm,
            GLM5_WIDTH, GLM5_HC, 20u, ctx->model->hc_eps,
            ctx->model->rms_norm_eps) &&
        ds4_gpu_matmul_q8_0_tensor(
            w->mla_q_a, ctx->model_map, ctx->model_size, m->q_a,
            GLM5_WIDTH, GLM5_Q_RANK, w->ffn_hidden, 1u) &&
        ds4_gpu_rms_norm_weight_tensor(
            w->mla_q_resid, w->mla_q_a, ctx->model_map, ctx->model_size,
            m->q_a_norm, GLM5_Q_RANK, ctx->model->rms_norm_eps) &&
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
            1u, GLM5_KV_LORA, GLM5_KV_LORA,
            ctx->model->rms_norm_eps) &&
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
            mla->index_tail, w->mla_index_k_raw,
            ctx->model_map, ctx->model_size,
            m->index_k_norm, m->index_k_norm_b,
            tail_slot, 1u, 4u, GLM5_INDEX_DIM,
            0u, 1u, 1.0e-6f, 1.0f, 1.0f, 0.0f,
            1.0f, 0.0f, 0.0f, false) &&
        ds4_gpu_matmul_bf16_tensor(
            w->mla_pool_gate_raw, ctx->model_map, ctx->model_size,
            m->index_pool_gate, GLM5_WIDTH, GLM5_INDEX_DIM,
            w->ffn_hidden, 1u) &&
        ds4_gpu_tensor_copy(mla->pool_gate_tail,
                            (uint64_t)tail_slot * GLM5_INDEX_DIM * sizeof(float),
                            w->mla_pool_gate_raw,
                            0u, GLM5_INDEX_DIM * sizeof(float)) &&
        mla_publish_completed_pool(
            ctx, m, mla, w, pool_index, publish_pool) &&
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
        tp_exchange(ctx, il, DS4_TP_GATE_ATTN) &&
        ds4_gpu_add_tensor(w->attention, ctx->tp_big_out, ctx->tp_big_in,
                           GLM5_WIDTH) &&
        ds4_gpu_hc_expand_split_tensor(
            w->after_attention, w->attention, hc_in, w->hc_split,
            GLM5_WIDTH, GLM5_HC);
}

/* Beyond the model's 2048-row index budget, score completed pool-4 keys,
 * retain the best 512 pools, expand them back to raw rows, and append the
 * current incomplete tail.  All selector intermediates remain device-local;
 * only the established attention output slice crosses RDMA. */
static int mla_sparse_selection_attention(
        const ds4_glm5_next_exec_ctx *ctx,
        uint32_t il,
        ds4_glm5_next_state *state,
        ds4_glm5_next_workspace *w,
        const ds4_gpu_tensor *hc_in,
        uint32_t tail_slot,
        uint32_t pool_index,
        bool publish_pool,
        uint32_t top_k) {
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    const ds4_glm5_next_mla_offsets *m = &layer->mla;
    ds4_glm5_next_mla_state *mla = &state->mla[il];
    const uint32_t pos = mla->token_count;
    uint32_t visible = 0u, n_pools = 0u, selected_pools = 0u;
    uint32_t selected_tokens = 0u;
    const uint64_t half_heads =
        ((uint64_t)GLM5_HEADS * GLM5_HEAD_DIM) / 2u;
    if (!ds4_glm5_next_mla_sparse_selection_plan(
            pos, mla->capacity_tokens, top_k,
            GLM5_INDEX_POOL, &visible, &n_pools, &selected_pools,
            &selected_tokens) ||
        n_pools == 0u || n_pools > mla->capacity_pools ||
        n_pools > w->sparse_pool_capacity || selected_pools == 0u ||
        top_k > DS4_GLM5_NEXT_INDEX_TOP_K ||
        selected_pools > top_k / GLM5_INDEX_POOL ||
        selected_tokens > top_k +
                              GLM5_INDEX_POOL - 1u) {
        return 0;
    }
    return
        ds4_gpu_rms_norm_plain_rows_tensor(
            w->hc_flat, hc_in, GLM5_HC_WIDTH, 1u,
            ctx->model->rms_norm_eps) &&
        ds4_gpu_matmul_bf16_tensor(
            w->hc_mix, ctx->model_map, ctx->model_size,
            layer->hc.attn_fn, GLM5_HC_WIDTH, GLM5_HC_MIX,
            w->hc_flat, 1u) &&
        ds4_gpu_hc_split_weighted_sum_norm_tensor(
            w->collapsed, w->ffn_hidden, w->hc_split, w->hc_mix, hc_in,
            ctx->model_map, ctx->model_size,
            layer->hc.attn_scale, layer->hc.attn_base, layer->attn_norm,
            GLM5_WIDTH, GLM5_HC, 20u, ctx->model->hc_eps,
            ctx->model->rms_norm_eps) &&
        ds4_gpu_matmul_q8_0_tensor(
            w->mla_q_a, ctx->model_map, ctx->model_size, m->q_a,
            GLM5_WIDTH, GLM5_Q_RANK, w->ffn_hidden, 1u) &&
        ds4_gpu_rms_norm_weight_tensor(
            w->mla_q_resid, w->mla_q_a, ctx->model_map, ctx->model_size,
            m->q_a_norm, GLM5_Q_RANK, ctx->model->rms_norm_eps) &&
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
            1u, GLM5_KV_LORA, GLM5_KV_LORA,
            ctx->model->rms_norm_eps) &&
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
            mla->index_tail, w->mla_index_k_raw,
            ctx->model_map, ctx->model_size,
            m->index_k_norm, m->index_k_norm_b,
            tail_slot, 1u, GLM5_INDEX_POOL, GLM5_INDEX_DIM,
            0u, 1u, 1.0e-6f, 1.0f, 1.0f, 0.0f,
            1.0f, 0.0f, 0.0f, false) &&
        ds4_gpu_matmul_bf16_tensor(
            w->mla_pool_gate_raw, ctx->model_map, ctx->model_size,
            m->index_pool_gate, GLM5_WIDTH, GLM5_INDEX_DIM,
            w->ffn_hidden, 1u) &&
        ds4_gpu_tensor_copy(
            mla->pool_gate_tail,
            (uint64_t)tail_slot * GLM5_INDEX_DIM * sizeof(float),
            w->mla_pool_gate_raw, 0u, GLM5_INDEX_DIM * sizeof(float)) &&
        mla_publish_completed_pool(
            ctx, m, mla, w, pool_index, publish_pool) &&
        ds4_gpu_matmul_bf16_tensor(
            w->mla_index_q, ctx->model_map, ctx->model_size, m->index_q_b,
            GLM5_Q_RANK, GLM5_INDEX_HEADS * GLM5_INDEX_DIM,
            w->mla_q_resid, 1u) &&
        ds4_gpu_matmul_bf16_tensor(
            w->mla_index_weights, ctx->model_map, ctx->model_size,
            m->index_proj, GLM5_WIDTH, GLM5_INDEX_HEADS,
            w->ffn_hidden, 1u) &&
        ds4_gpu_glm_indexer_score_one_tensor(
            w->mla_pool_scores, w->mla_index_q, w->mla_index_weights,
            mla->index_pool, n_pools, GLM5_INDEX_HEADS, GLM5_INDEX_DIM,
            0.015625f, false) &&
        ds4_gpu_glm5_mask_pool_scores_tensor(
            w->mla_pool_scores, mla->index_pool_valid, n_pools) &&
        ds4_gpu_indexer_topk_tensor(
            w->mla_selected_pool, w->mla_pool_scores, n_pools, 1u,
            selected_pools) &&
        ds4_gpu_glm5_expand_pool_selection_tensor(
            w->mla_selected_token, w->mla_selected_pool,
            mla->index_pool_ids, mla->index_pool_valid,
            mla->index_valid_keys, n_pools, selected_pools,
            mla->capacity_tokens, mla->first_valid, visible,
            top_k, GLM5_INDEX_POOL) &&
        ds4_gpu_glm_attention_indexed_decode_typed_tensor(
            w->mla_heads, w->mla_query, w->mla_qk_low,
            mla->compact_kv, NULL, ctx->model_map, ctx->model_size,
            m->v_b, 8u, w->mla_selected_token, selected_tokens,
            mla->capacity_tokens, false, GLM5_HEADS, GLM5_KV_LORA,
            GLM5_HEAD_DIM, 0u, GLM5_HEAD_DIM, 0u,
            1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f) &&
        ds4_gpu_matmul_q8_0_kslice_tensor(
            ctx->tp_big_out, ctx->model_map, ctx->model_size, m->output,
            GLM5_HEADS * GLM5_HEAD_DIM,
            (uint64_t)ctx->tp_rank * half_heads, half_heads,
            GLM5_WIDTH, w->mla_heads,
            (uint64_t)ctx->tp_rank * half_heads) &&
        tp_exchange(ctx, il, DS4_TP_GATE_ATTN) &&
        ds4_gpu_add_tensor(
            w->attention, ctx->tp_big_out, ctx->tp_big_in, GLM5_WIDTH) &&
        ds4_gpu_hc_expand_split_tensor(
            w->after_attention, w->attention, hc_in, w->hc_split,
            GLM5_WIDTH, GLM5_HC);
}

/* Preserve the scalar four-row pool state machine while the expensive MLA
 * projections execute as one row batch.  The normalized key and pool-gate
 * rows are copied into the resident tail in token order; every fourth row is
 * published before its slot can be reused.  State counters are committed only
 * after the complete attention+FFN stage succeeds. */
static int mla_stage_index_rows(const ds4_glm5_next_exec_ctx *ctx,
                                const ds4_glm5_next_mla_offsets *offsets,
                                ds4_glm5_next_mla_state *mla,
                                ds4_glm5_next_workspace *w,
                                uint32_t pos0,
                                uint32_t n_tokens) {
    const uint64_t row_bytes =
        (uint64_t)GLM5_INDEX_DIM * sizeof(float);
    if (!ctx || !offsets || !mla || !w || n_tokens == 0u ||
        mla->token_count != pos0 || mla->complete_pools != pos0 / 4u ||
        mla->tail_count != pos0 % 4u) return 0;
    for (uint32_t t = 0u; t < n_tokens; ++t) {
        const uint32_t token = pos0 + t;
        const uint32_t tail_slot = token % 4u;
        const uint32_t pool_index = token / 4u;
        const bool publish_pool = tail_slot == 3u;
        if (!ds4_gpu_tensor_copy(
                mla->index_tail, (uint64_t)tail_slot * row_bytes,
                w->mla_index_k_norm, (uint64_t)t * row_bytes, row_bytes) ||
            !ds4_gpu_tensor_copy(
                mla->pool_gate_tail, (uint64_t)tail_slot * row_bytes,
                w->mla_pool_gate_raw, (uint64_t)t * row_bytes, row_bytes) ||
            !mla_publish_completed_pool(ctx, offsets, mla, w, pool_index,
                                        publish_pool)) {
            return 0;
        }
    }
    return 1;
}

static int mla_value_project_rows_batch(
        const ds4_glm5_next_exec_ctx *ctx,
        const ds4_glm5_next_mla_offsets *offsets,
        ds4_glm5_next_workspace *w,
        uint32_t n_tokens) {
    return ds4_gpu_glm_value_project_typed_batch_heads_tensor(
        w->mla_heads, w->routed_experts, ctx->model_map, ctx->model_size,
        offsets->v_b, 8u, n_tokens, GLM5_HEADS,
        GLM5_KV_LORA, GLM5_HEAD_DIM);
}

static int mla_output_project_rows_batch(
        const ds4_glm5_next_exec_ctx *ctx,
        const ds4_glm5_next_mla_offsets *offsets,
        ds4_glm5_next_workspace *w,
        uint32_t n_tokens,
        int force_serial) {
    const uint64_t full_heads =
        (uint64_t)GLM5_HEADS * GLM5_HEAD_DIM;
    const uint64_t half_heads = full_heads / 2u;
    const uint64_t in_start = (uint64_t)ctx->tp_rank * half_heads;
    /* A single row keeps the established decode implementation. Besides
     * avoiding a 64 KiB token-tile launch for one row, this makes the hook's
     * 0 return unambiguously mean that an engaged batch path failed. */
    if (n_tokens == 1u) {
        ds4_gpu_tensor *out = ds4_gpu_tensor_view(
            ctx->tp_big_out, 0u,
            (uint64_t)GLM5_WIDTH * sizeof(float));
        const int ok = out && ds4_gpu_matmul_q8_0_kslice_tensor(
            out, ctx->model_map, ctx->model_size, offsets->output,
            full_heads, in_start, half_heads, GLM5_WIDTH, w->mla_heads,
            in_start);
        ds4_gpu_tensor_free(out);
        return ok;
    }
    /* Q4's multi-row MLA output kernel was intermittently non-deterministic
     * across both RDMA providers.  Keep Q2's validated batch path, but make
     * Q4 use the established row kernel until a standalone oracle proves a
     * replacement. */
    const int batch = force_serial &&
        getenv("DS4_GLM5_ALLOW_Q4_BATCH_MLA_OUTPUT") == NULL ? -1 :
        ds4_rocm_q8_kslice_f32_rows_strided(
            ctx->tp_big_out, ctx->model_map, ctx->model_size,
            offsets->output, full_heads, GLM5_WIDTH, in_start, half_heads,
            w->mla_heads, in_start, n_tokens, full_heads);
    if (batch >= 0) return batch;
    /* Preserve the exact one-row implementation on backends without the
     * strided token-tile entry point. */
    for (uint32_t t = 0u; t < n_tokens; ++t) {
        ds4_gpu_tensor *out = ds4_gpu_tensor_view(
            ctx->tp_big_out,
            (uint64_t)t * GLM5_WIDTH * sizeof(float),
            (uint64_t)GLM5_WIDTH * sizeof(float));
        const int ok = out && ds4_gpu_matmul_q8_0_kslice_tensor(
            out, ctx->model_map, ctx->model_size, offsets->output,
            full_heads, in_start, half_heads, GLM5_WIDTH, w->mla_heads,
            (uint64_t)t * full_heads + in_start);
        ds4_gpu_tensor_free(out);
        if (!ok) return 0;
    }
    return 1;
}

static int mla_dense_selection_attention_rows(
        const ds4_glm5_next_exec_ctx *ctx,
        uint32_t il,
        ds4_glm5_next_state *state,
        ds4_glm5_next_workspace *w,
        const ds4_gpu_tensor *hc_in,
        uint32_t n_tokens) {
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    const ds4_glm5_next_mla_offsets *m = &layer->mla;
    ds4_glm5_next_mla_state *mla = &state->mla[il];
    const uint32_t pos0 = mla->token_count;
    const uint32_t n_selected = pos0 + n_tokens;
    const uint64_t full_heads =
        (uint64_t)GLM5_HEADS * GLM5_HEAD_DIM;
    const uint64_t elements = (uint64_t)n_tokens * GLM5_WIDTH;
    if (n_tokens == 0u || n_selected < pos0 ||
        n_selected > DS4_GLM5_NEXT_INDEX_TOP_K ||
        n_selected > mla->capacity_tokens ||
        elements > UINT32_MAX) return 0;

    int ok =
        ds4_gpu_rms_norm_plain_rows_tensor(
            w->hc_flat, hc_in, GLM5_HC_WIDTH, n_tokens,
            ctx->model->rms_norm_eps) &&
        ds4_gpu_matmul_bf16_tensor(
            w->hc_mix, ctx->model_map, ctx->model_size,
            layer->hc.attn_fn, GLM5_HC_WIDTH, GLM5_HC_MIX,
            w->hc_flat, n_tokens) &&
        ds4_gpu_hc_split_weighted_sum_norm_tensor(
            w->collapsed, w->ffn_hidden, w->hc_split, w->hc_mix, hc_in,
            ctx->model_map, ctx->model_size,
            layer->hc.attn_scale, layer->hc.attn_base, layer->attn_norm,
            GLM5_WIDTH, GLM5_HC, 20u, ctx->model->hc_eps,
            ctx->model->rms_norm_eps) &&
        ds4_gpu_matmul_q8_0_tensor(
            w->mla_q_a, ctx->model_map, ctx->model_size, m->q_a,
            GLM5_WIDTH, GLM5_Q_RANK, w->ffn_hidden, n_tokens) &&
        ds4_gpu_rms_norm_weight_rows_tensor(
            w->mla_q_resid, w->mla_q_a, ctx->model_map, ctx->model_size,
            m->q_a_norm, GLM5_Q_RANK, n_tokens,
            ctx->model->rms_norm_eps) &&
        ds4_gpu_matmul_q8_0_tensor(
            w->mla_query, ctx->model_map, ctx->model_size, m->q_b,
            GLM5_Q_RANK, full_heads, w->mla_q_resid, n_tokens) &&
        ds4_gpu_matmul_q8_0_tensor(
            w->mla_kv_raw, ctx->model_map, ctx->model_size, m->kv_a_mqa,
            GLM5_WIDTH, GLM5_KV_LORA, w->ffn_hidden, n_tokens) &&
        ds4_gpu_glm_kv_lora_rms_norm_tensor(
            w->mla_kv_norm, w->mla_kv_raw,
            ctx->model_map, ctx->model_size, m->kv_a_norm,
            n_tokens, GLM5_KV_LORA, GLM5_KV_LORA,
            ctx->model->rms_norm_eps) &&
        ds4_gpu_glm_store_compact_kv_tensor(
            mla->compact_kv, NULL, w->mla_kv_norm, w->mla_kv_raw,
            pos0, n_tokens, mla->capacity_tokens, GLM5_KV_LORA,
            GLM5_KV_LORA, 0u, false) &&
        ds4_gpu_glm_qk_lowrank_typed_batch_tensor(
            w->mla_qk_low, w->mla_query,
            ctx->model_map, ctx->model_size, m->k_b, 8u, n_tokens,
            GLM5_HEADS, GLM5_KV_LORA, GLM5_HEAD_DIM, GLM5_HEAD_DIM) &&
        ds4_gpu_matmul_bf16_tensor(
            w->mla_index_k_raw, ctx->model_map, ctx->model_size, m->index_k,
            GLM5_WIDTH, GLM5_INDEX_DIM, w->ffn_hidden, n_tokens) &&
        ds4_gpu_glm_store_indexer_k_tensor(
            w->mla_index_k_norm, w->mla_index_k_raw,
            ctx->model_map, ctx->model_size,
            m->index_k_norm, m->index_k_norm_b,
            0u, n_tokens, n_tokens, GLM5_INDEX_DIM,
            0u, 1u, 1.0e-6f, 1.0f, 1.0f, 0.0f,
            1.0f, 0.0f, 0.0f, false) &&
        ds4_gpu_matmul_bf16_tensor(
            w->mla_pool_gate_raw, ctx->model_map, ctx->model_size,
            m->index_pool_gate, GLM5_WIDTH, GLM5_INDEX_DIM,
            w->ffn_hidden, n_tokens) &&
        mla_stage_index_rows(ctx, m, mla, w, pos0, n_tokens) &&
        ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
            w->routed_experts, w->mla_query, w->mla_qk_low,
            mla->compact_kv, NULL, n_tokens, pos0, n_selected,
            mla->capacity_tokens, false, GLM5_HEADS, GLM5_KV_LORA,
            GLM5_HEAD_DIM, 0u, 0u,
            1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f) &&
        mla_value_project_rows_batch(ctx, m, w, n_tokens) &&
        mla_output_project_rows_batch(
            ctx, m, w, n_tokens,
            !(layer->ffn_weight.gate_exps_type == 16u &&
              layer->ffn_weight.up_exps_type == 16u &&
              layer->ffn_weight.down_exps_type == 10u));
    return ok &&
        tp_exchange_rows(ctx, il, DS4_TP_GATE_ATTN, n_tokens) &&
        ds4_gpu_add_tensor(w->attention, ctx->tp_big_out, ctx->tp_big_in,
                           (uint32_t)elements) &&
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
    const bool mixed_q2 = f->gate_exps_type == 16u &&
                          f->up_exps_type == 16u &&
                          f->down_exps_type == 10u;
    ds4_gpu_q4k_window_cache *cache = NULL;
    bool overlap_prefetched = false;
    const int q4_residency = mixed_q2 ? 0 :
        local_q4k_half_residency(ctx, layer);
    int ok = ds4_gpu_rms_norm_plain_rows_tensor(
            w->ffn_flat, w->after_attention, GLM5_HC_WIDTH, 1u,
            ctx->model->rms_norm_eps);
    if (!ok) { fprintf(stderr, "ds4: GLM5 routed layer %u failed at FFN HC norm rank=%u\n", il, ctx->tp_rank); goto routed_one_done; }
    ok = ds4_gpu_matmul_bf16_tensor(
            w->ffn_mix, ctx->model_map, ctx->model_size,
            layer->hc.ffn_fn, GLM5_HC_WIDTH, GLM5_HC_MIX,
            w->ffn_flat, 1u);
    if (!ok) { fprintf(stderr, "ds4: GLM5 routed layer %u failed at FFN HC projection rank=%u\n", il, ctx->tp_rank); goto routed_one_done; }
    ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(
            w->ffn_collapsed, w->ffn_hidden, w->ffn_split, w->ffn_mix,
            w->after_attention, ctx->model_map, ctx->model_size,
            layer->hc.ffn_scale, layer->hc.ffn_base, layer->ffn_norm,
            GLM5_WIDTH, GLM5_HC, 20u, ctx->model->hc_eps,
            ctx->model->rms_norm_eps);
    if (!ok) { fprintf(stderr, "ds4: GLM5 routed layer %u failed at FFN HC split rank=%u\n", il, ctx->tp_rank); goto routed_one_done; }
    ok = ds4_gpu_matmul_f32_tensor(
            w->router_logits, ctx->model_map, ctx->model_size, f->gate_inp,
            GLM5_WIDTH, GLM5_EXPERTS, w->ffn_hidden, 1u);
    if (!ok) { fprintf(stderr, "ds4: GLM5 routed layer %u failed at router projection rank=%u\n", il, ctx->tp_rank); goto routed_one_done; }
    ok = ds4_gpu_glm_router_select_tensor(
            w->router_selected, w->router_weights, w->router_probs,
            ctx->model_map, ctx->model_size, f->exp_probs_b,
            w->router_logits, GLM5_EXPERTS, GLM5_EXPERTS_USED, 2.5f);
    if (!ok) { fprintf(stderr, "ds4: GLM5 routed layer %u failed at top-8 select rank=%u\n", il, ctx->tp_rank); goto routed_one_done; }
    ok = route_agrees(ctx, il, token_ordinal,
                      w->router_selected, w->router_weights,
                      w->router_logits, w->ffn_hidden);
    if (!ok) {
        route_failure_stats("after_attention", w->after_attention,
                            GLM5_HC_WIDTH);
        route_failure_stats("ffn_mix", w->ffn_mix, GLM5_HC_MIX);
        route_failure_stats("ffn_collapsed", w->ffn_collapsed, GLM5_WIDTH);
        fprintf(stderr, "ds4: GLM5 routed layer %u failed at router prelude rank=%u\n",
                il, ctx->tp_rank);
        goto routed_one_done;
    }
    if (q4_residency < 0) {
        fprintf(stderr,
                "ds4: GLM5 routed layer %u has partial Q4_K residency rank=%u\n",
                il, ctx->tp_rank);
        ok = 0;
    }
    if (ok && mixed_q2) {
        /* The mixed GLM Q2 layout is handled by the existing type-aware
         * routed launcher. It consumes the original GGUF tables and uses
         * tile-local/staged ranges; this branch must not enter Q4_K slice
         * declarations, whose 144-byte geometry is incompatible with 66/84.
         */
        ok = ds4_gpu_routed_moe_one_tensor(
            w->routed_out, w->routed_gate, w->routed_up, w->routed_mid,
            w->routed_experts, ctx->model_map, ctx->model_size,
            f->gate_exps, f->up_exps, f->down_exps,
            f->gate_exps_type, f->down_exps_type,
            (uint64_t)GLM5_ROUTED_MID * (GLM5_WIDTH / 256u) * 66u,
            (GLM5_WIDTH / 256u) * 66u,
            (uint64_t)GLM5_WIDTH * (GLM5_ROUTED_MID / 256u) * 84u,
            (GLM5_ROUTED_MID / 256u) * 84u,
            GLM5_WIDTH, GLM5_ROUTED_MID, GLM5_WIDTH,
            w->router_selected, w->router_weights,
            GLM5_EXPERTS, GLM5_EXPERTS_USED, 10.0f,
            w->ffn_hidden, NULL, il, false);
    } else if (ok && q4_residency == 1) {
        ok = ds4_gpu_routed_moe_one_packed_q4k_tensor(
                 w->routed_out, w->routed_gate, w->routed_up, w->routed_mid,
                 w->routed_experts, ctx->model_map, ctx->model_size,
                 f->gate_exps, f->up_exps, f->down_exps, GLM5_EXPERTS,
                 q4_gate_row_bytes, q4_down_row_bytes,
                 rank_mid_base, GLM5_RANK_MID, q4_down_base,
                 q4_down_half_bytes, w->router_selected, w->router_weights,
                 GLM5_EXPERTS_USED, 10.0f, w->ffn_hidden, NULL, il);
    } else if (ok) {
        const char *scratch_windows = getenv("DS4_ROCM_GLM5_WINDOW_SCRATCH");
        const bool use_scratch = w->decode_phase && scratch_windows &&
            strcmp(scratch_windows, "1") == 0;
        uint32_t window_slots = use_scratch ? 8u : GLM5_EXPERTS_USED;
        const char *window_slots_env =
            getenv("DS4_ROCM_GLM5_WINDOW_SLOTS");
        if (!use_scratch && window_slots_env && *window_slots_env) {
            char *end = NULL;
            const unsigned long parsed = strtoul(window_slots_env, &end, 10);
            if (end != window_slots_env && *end == '\0' &&
                parsed >= GLM5_EXPERTS_USED && parsed <= 64u)
                window_slots = (uint32_t)parsed;
        }
        const ds4_gpu_q4k_window_cache_config config = {
            .model_map = ctx->model_map,
            .gate_offset = f->gate_exps,
            .up_offset = f->up_exps,
            .down_offset = f->down_exps,
            .n_expert = GLM5_EXPERTS,
            .gate_row_base = rank_mid_base,
            .gate_row_count = GLM5_RANK_MID,
            .gate_column_byte_base = 0u,
            .gate_column_byte_count = q4_gate_row_bytes,
            .down_row_base = 0u,
            .down_row_count = GLM5_WIDTH,
            .down_column_byte_base = q4_down_base,
            .down_column_byte_count = q4_down_half_bytes,
            .slots = window_slots,
        };
        ok = declare_local_q4k_half_only(ctx, layer);
        const char *persist_windows =
            getenv("DS4_ROCM_GLM5_WINDOW_PERSIST");
        if (ok && use_scratch && il < DS4_GLM5_NEXT_LAYER_COUNT) {
            if (!w->q4_window_scratch)
                w->q4_window_scratch =
                    ds4_gpu_q4k_window_cache_create(&config);
            else
                ok = ds4_gpu_q4k_window_cache_rebind(
                    w->q4_window_scratch, &config);
            cache = w->q4_window_scratch;
        } else if (ok && persist_windows && strcmp(persist_windows, "1") == 0 &&
            il < DS4_GLM5_NEXT_LAYER_COUNT) {
            cache = w->q4_window[il];
            if (!cache) {
                cache = ds4_gpu_q4k_window_cache_create(&config);
                if (cache) w->q4_window[il] = cache;
            }
        } else {
            cache = ok ? ds4_gpu_q4k_window_cache_create(&config) : NULL;
        }
        const char *overlap_env = getenv("DS4_ROCM_GLM5_WINDOW_OVERLAP");
        overlap_prefetched = ok && use_scratch && overlap_env &&
            strcmp(overlap_env, "1") == 0;
        if (overlap_prefetched) {
            /* Start selected-expert uploads before the shared Q8 path.  The
             * routed helper repeats the metadata check and waits on the
             * cache event immediately before its consumer kernel. */
            ok = ds4_gpu_q4k_window_cache_prefetch(
                cache, w->router_selected, w->router_weights,
                GLM5_EXPERTS_USED);
        }
        if (ok && overlap_prefetched) ok =
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
                GLM5_RANK_MID, 10.0f, 1.0f);
        ok = cache && ds4_gpu_routed_moe_one_packed_q4k_window_tensor(
                 w->routed_out, w->routed_gate, w->routed_up, w->routed_mid,
                 w->routed_experts, cache, w->router_selected,
                 w->router_weights, GLM5_EXPERTS_USED, 10.0f,
                 w->ffn_hidden, NULL, il);
    }
    if (!ok) {
        fprintf(stderr, "ds4: GLM5 routed layer %u failed at routed MoE rank=%u mixed_q2=%d\n",
                il, ctx->tp_rank, mixed_q2 ? 1 : 0);
        goto routed_one_done;
    }
    if (ok && !overlap_prefetched) ok = ds4_gpu_matmul_q8_0_tensor(
            w->shared_gate, ctx->model_map, ctx->model_size,
            f->gate_shexp + (uint64_t)rank_mid_base * q8_gate_row_bytes,
            GLM5_WIDTH, GLM5_RANK_MID, w->ffn_hidden, 1u);
    if (!ok) { fprintf(stderr, "ds4: GLM5 routed layer %u failed at shared gate rank=%u\n", il, ctx->tp_rank); goto routed_one_done; }
    if (ok && !overlap_prefetched) ok = ds4_gpu_matmul_q8_0_tensor(
            w->shared_up, ctx->model_map, ctx->model_size,
            f->up_shexp + (uint64_t)rank_mid_base * q8_gate_row_bytes,
            GLM5_WIDTH, GLM5_RANK_MID, w->ffn_hidden, 1u);
    if (!ok) { fprintf(stderr, "ds4: GLM5 routed layer %u failed at shared up rank=%u\n", il, ctx->tp_rank); goto routed_one_done; }
    if (ok && !overlap_prefetched) ok = ds4_gpu_swiglu_tensor(
            w->shared_mid, w->shared_gate, w->shared_up,
            GLM5_RANK_MID, 10.0f, 1.0f);
    if (!ok) { fprintf(stderr, "ds4: GLM5 routed layer %u failed at shared SwiGLU rank=%u\n", il, ctx->tp_rank); goto routed_one_done; }
    if (ok) ok = ds4_gpu_matmul_q8_0_kslice_tensor(
            w->shared_out, ctx->model_map, ctx->model_size, f->down_shexp,
            GLM5_ROUTED_MID, rank_mid_base, GLM5_RANK_MID,
            GLM5_WIDTH, w->shared_mid, 0u);
    if (!ok) { fprintf(stderr, "ds4: GLM5 routed layer %u failed at shared down rank=%u\n", il, ctx->tp_rank); goto routed_one_done; }
    if (ok) ok = ds4_gpu_add_tensor(ctx->tp_big_out, w->routed_out,
                                    w->shared_out, GLM5_WIDTH);
    if (!ok) { fprintf(stderr, "ds4: GLM5 routed layer %u failed at local compose rank=%u\n", il, ctx->tp_rank); goto routed_one_done; }
    if (ok) ok = tp_exchange(ctx, il, DS4_TP_GATE_FFN);
    if (!ok) { fprintf(stderr, "ds4: GLM5 routed layer %u failed at TP exchange rank=%u\n", il, ctx->tp_rank); goto routed_one_done; }
    if (ok) ok = ds4_gpu_add_tensor(w->down, ctx->tp_big_out, ctx->tp_big_in,
                                    GLM5_WIDTH);
    if (!ok) { fprintf(stderr, "ds4: GLM5 routed layer %u failed at all-rank compose rank=%u\n", il, ctx->tp_rank); goto routed_one_done; }
    if (ok) ok = ds4_gpu_hc_expand_split_tensor(
            hc_out, w->down, w->after_attention, w->ffn_split,
            GLM5_WIDTH, GLM5_HC);
    if (!ok) { fprintf(stderr, "ds4: GLM5 routed layer %u failed at mHC expand rank=%u\n", il, ctx->tp_rank); goto routed_one_done; }
    if (ok) ok = ds4_gpu_synchronize();
    if (!ok) fprintf(stderr, "ds4: GLM5 routed layer %u failed at synchronize rank=%u\n", il, ctx->tp_rank);
    /* The packed slices are layer-scoped.  The final synchronize above makes
     * it safe to release them before the next layer is declared. */
routed_one_done:
    if (!ok) ds4_gpu_synchronize();
    const char *persist_windows = getenv("DS4_ROCM_GLM5_WINDOW_PERSIST");
    const char *scratch_windows = getenv("DS4_ROCM_GLM5_WINDOW_SCRATCH");
    const bool use_scratch = w->decode_phase && scratch_windows &&
        strcmp(scratch_windows, "1") == 0;
    if (!mixed_q2 && q4_residency == 0 &&
        !(persist_windows && strcmp(persist_windows, "1") == 0) &&
        !use_scratch)
        ds4_gpu_q4k_packed_slice_release_all();
    (void)q8_down_row_bytes;
    return ok;
}

#ifdef DS4_TP_TEST_HOOKS
static int trace_same_input_routed_ffn(
        const ds4_glm5_next_exec_ctx *ctx,
        uint32_t il,
        uint32_t token_ordinal,
        const ds4_gpu_tensor *after_attention_rows,
        uint32_t row) {
    if (!ctx->trace_prefix || il != ctx->trace_layer ||
        getenv("DS4_GLM5_TRACE_FFN_SAME_INPUT") == NULL) return 1;
    const uint64_t hc_row_bytes =
        (uint64_t)GLM5_HC_WIDTH * sizeof(float);
    ds4_glm5_next_workspace *probe =
        ds4_glm5_next_workspace_create_capacity(1u);
    ds4_gpu_tensor *probe_out = ds4_gpu_tensor_alloc(hc_row_bytes);
    char prefix[640];
    const int prefix_len = snprintf(
        prefix, sizeof(prefix), "%s.same_input", ctx->trace_prefix);
    ds4_glm5_next_exec_ctx probe_ctx = *ctx;
    probe_ctx.trace_prefix = prefix;
    int ok = probe && probe_out && prefix_len > 0 &&
        (size_t)prefix_len < sizeof(prefix) &&
        ds4_gpu_tensor_copy(probe->after_attention, 0u,
                            after_attention_rows,
                            (uint64_t)row * hc_row_bytes,
                            hc_row_bytes) &&
        routed_ffn_one(&probe_ctx, il, token_ordinal, probe, probe_out) &&
        trace_routed_ffn(
            &probe_ctx, il, token_ordinal, probe_out, probe);
    ds4_gpu_tensor_free(probe_out);
    ds4_glm5_next_workspace_destroy(probe);
    return ok;
}
#endif

static int routed_ffn_rows(const ds4_glm5_next_exec_ctx *ctx,
                           uint32_t il,
                           uint32_t token_ordinal,
                           ds4_glm5_next_workspace *w,
                           ds4_gpu_tensor *hc_out,
                           uint32_t n_tokens) {
    /* Exact-capacity is load-bearing: mHC helpers infer the row count from
     * tensor byte sizes. This internal entry is valid only after the public
     * batch entry has proved workspace capacity and exact HC tensor sizes. */
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    const ds4_glm5_next_ffn_offsets *f = &layer->ffn_weight;
    const uint32_t rank_mid_base = ctx->tp_rank * GLM5_RANK_MID;
    const uint64_t q8_gate_row_bytes =
        (GLM5_WIDTH / GLM5_Q8_QK) * GLM5_Q8_BLOCK_BYTES;
    const uint64_t q4_gate_row_bytes =
        (GLM5_WIDTH / GLM5_Q4K_QK) * GLM5_Q4K_BLOCK_BYTES;
    const uint64_t q4_down_row_bytes =
        (GLM5_ROUTED_MID / GLM5_Q4K_QK) * GLM5_Q4K_BLOCK_BYTES;
    const uint64_t q4_down_half_bytes =
        (GLM5_RANK_MID / GLM5_Q4K_QK) * GLM5_Q4K_BLOCK_BYTES;
    const uint64_t q4_down_base =
        (uint64_t)ctx->tp_rank * q4_down_half_bytes;
    uint64_t elements = 0u;
    bool routed_mid_is_f16 = true;
    if (n_tokens == 0u ||
        (uint64_t)n_tokens > UINT32_MAX / GLM5_WIDTH ||
        (uint64_t)n_tokens > UINT32_MAX / GLM5_RANK_MID) return 0;
    elements = (uint64_t)n_tokens * GLM5_WIDTH;
    int ok = 1;
    /* Diagnostic for partially-written or asynchronously-reused routed-FFN
     * outputs.  Workspace creation-time clearing cannot expose that class of
     * bug because these tensors are reused by every layer.  Keep this opt-in:
     * it deliberately adds several large device clears per layer. */
    const char *zero_moe = getenv("DS4_GLM5_ZERO_MOE_EACH_LAYER");
    if (zero_moe != NULL) {
        const uint64_t routed_mid_elems = (uint64_t)n_tokens *
            GLM5_EXPERTS_USED * GLM5_ROUTED_MID;
        const uint64_t routed_out_elems = (uint64_t)n_tokens *
            GLM5_EXPERTS_USED * GLM5_WIDTH;
        const uint64_t shared_mid_elems =
            (uint64_t)n_tokens * GLM5_RANK_MID;
        const int zero_all = strcmp(zero_moe, "1") == 0 ||
                             strcmp(zero_moe, "all") == 0;
        const int zero_routed = zero_all || strcmp(zero_moe, "routed") == 0;
        const int zero_shared = zero_all || strcmp(zero_moe, "shared") == 0;
        ok = (zero_routed || zero_shared) &&
            routed_mid_elems <= UINT32_MAX &&
            routed_out_elems <= UINT32_MAX &&
            shared_mid_elems <= UINT32_MAX &&
            elements <= UINT32_MAX;
        if (ok && zero_routed) ok =
            ds4_gpu_tensor_fill_f32(w->routed_gate, 0.0f,
                                    (uint32_t)routed_mid_elems) &&
            ds4_gpu_tensor_fill_f32(w->routed_up, 0.0f,
                                    (uint32_t)routed_mid_elems) &&
            ds4_gpu_tensor_fill_f32(w->routed_mid, 0.0f,
                                    (uint32_t)routed_mid_elems) &&
            ds4_gpu_tensor_fill_f32(w->routed_experts, 0.0f,
                                    (uint32_t)routed_out_elems) &&
            ds4_gpu_tensor_fill_f32(w->routed_out, 0.0f,
                                    (uint32_t)elements);
        if (ok && zero_shared) ok =
            ds4_gpu_tensor_fill_f32(w->shared_gate, 0.0f,
                                    (uint32_t)shared_mid_elems) &&
            ds4_gpu_tensor_fill_f32(w->shared_up, 0.0f,
                                    (uint32_t)shared_mid_elems) &&
            ds4_gpu_tensor_fill_f32(w->shared_mid, 0.0f,
                                    (uint32_t)shared_mid_elems) &&
            ds4_gpu_tensor_fill_f32(w->shared_out, 0.0f,
                                    (uint32_t)elements);
    }
    ok = ok &&
        ds4_gpu_rms_norm_plain_rows_tensor(
            w->ffn_flat, w->after_attention, GLM5_HC_WIDTH, n_tokens,
            ctx->model->rms_norm_eps) &&
        ds4_gpu_matmul_bf16_tensor(
            w->ffn_mix, ctx->model_map, ctx->model_size,
            layer->hc.ffn_fn, GLM5_HC_WIDTH, GLM5_HC_MIX,
            w->ffn_flat, n_tokens) &&
        ds4_gpu_hc_split_weighted_sum_norm_tensor(
            w->ffn_collapsed, w->ffn_hidden, w->ffn_split, w->ffn_mix,
            w->after_attention, ctx->model_map, ctx->model_size,
            layer->hc.ffn_scale, layer->hc.ffn_base, layer->ffn_norm,
            GLM5_WIDTH, GLM5_HC, 20u, ctx->model->hc_eps,
            ctx->model->rms_norm_eps) &&
        ds4_gpu_matmul_f32_tensor(
            w->router_logits, ctx->model_map, ctx->model_size, f->gate_inp,
            GLM5_WIDTH, GLM5_EXPERTS, w->ffn_hidden, n_tokens) &&
        ds4_gpu_glm_router_select_batch_tensor(
            w->router_selected, w->router_weights, w->router_probs,
            ctx->model_map, ctx->model_size, f->exp_probs_b,
            w->router_logits, GLM5_EXPERTS, GLM5_EXPERTS_USED, 2.5f,
            n_tokens) &&
        route_batch_agrees(ctx, il, token_ordinal,
                           w->router_selected, w->router_weights,
                           n_tokens);
    const bool mixed_q2 = f->gate_exps_type == 16u &&
                          f->up_exps_type == 16u &&
                          f->down_exps_type == 10u;
    const int q4_residency = mixed_q2 ? 0 :
        local_q4k_half_residency(ctx, layer);
    if (q4_residency < 0) {
        fprintf(stderr,
                "ds4: GLM5 batch layer %u has partial Q4_K residency rank=%u\n",
                il, ctx->tp_rank);
        ok = 0;
    }
    ds4_gpu_q4k_window_cache *cache = NULL;
    if (ok && mixed_q2) {
        routed_mid_is_f16 = false;
        if (getenv("DS4_ROCM_GLM5_BATCH_MOE_SERIAL") != NULL) {
            const uint64_t width_bytes = (uint64_t)GLM5_WIDTH * sizeof(float);
            const uint64_t mid_bytes = (uint64_t)GLM5_EXPERTS_USED *
                GLM5_ROUTED_MID * sizeof(float);
            const uint64_t pair_out_bytes = (uint64_t)GLM5_EXPERTS_USED *
                GLM5_WIDTH * sizeof(float);
            for (uint32_t t = 0u; t < n_tokens && ok; ++t) {
                ds4_gpu_tensor *ov = ds4_gpu_tensor_view(
                    w->routed_out, (uint64_t)t * width_bytes, width_bytes);
                ds4_gpu_tensor *gv = ds4_gpu_tensor_view(
                    w->routed_gate, (uint64_t)t * mid_bytes, mid_bytes);
                ds4_gpu_tensor *uv = ds4_gpu_tensor_view(
                    w->routed_up, (uint64_t)t * mid_bytes, mid_bytes);
                ds4_gpu_tensor *mv = ds4_gpu_tensor_view(
                    w->routed_mid, (uint64_t)t * mid_bytes, mid_bytes);
                ds4_gpu_tensor *dv = ds4_gpu_tensor_view(
                    w->routed_experts, (uint64_t)t * pair_out_bytes,
                    pair_out_bytes);
                ds4_gpu_tensor *xv = ds4_gpu_tensor_view(
                    w->ffn_hidden, (uint64_t)t * width_bytes, width_bytes);
                ds4_gpu_tensor *sv = ds4_gpu_tensor_view(
                    w->router_selected,
                    (uint64_t)t * GLM5_EXPERTS_USED * sizeof(int32_t),
                    (uint64_t)GLM5_EXPERTS_USED * sizeof(int32_t));
                ds4_gpu_tensor *wv = ds4_gpu_tensor_view(
                    w->router_weights,
                    (uint64_t)t * GLM5_EXPERTS_USED * sizeof(float),
                    (uint64_t)GLM5_EXPERTS_USED * sizeof(float));
                ok = ov && gv && uv && mv && dv && xv && sv && wv &&
                    ds4_gpu_routed_moe_one_tensor(
                        ov, gv, uv, mv, dv, ctx->model_map, ctx->model_size,
                        f->gate_exps, f->up_exps, f->down_exps,
                        f->gate_exps_type, f->down_exps_type,
                        (uint64_t)GLM5_ROUTED_MID * (GLM5_WIDTH / 256u) * 66u,
                        (GLM5_WIDTH / 256u) * 66u,
                        (uint64_t)GLM5_WIDTH * (GLM5_ROUTED_MID / 256u) * 84u,
                        (GLM5_ROUTED_MID / 256u) * 84u,
                        GLM5_WIDTH, GLM5_ROUTED_MID, GLM5_WIDTH,
                        sv, wv, GLM5_EXPERTS, GLM5_EXPERTS_USED, 10.0f,
                        xv, NULL, il, false);
                ds4_gpu_tensor_free(ov); ds4_gpu_tensor_free(gv);
                ds4_gpu_tensor_free(uv); ds4_gpu_tensor_free(mv);
                ds4_gpu_tensor_free(dv); ds4_gpu_tensor_free(xv);
                ds4_gpu_tensor_free(sv); ds4_gpu_tensor_free(wv);
            }
        } else ok = ds4_gpu_routed_moe_batch_tensor(
            w->routed_out, w->routed_gate, w->routed_up, w->routed_mid,
            w->routed_experts, ctx->model_map, ctx->model_size,
            f->gate_exps, f->up_exps, f->down_exps,
            f->gate_exps_type, f->down_exps_type,
            (uint64_t)GLM5_ROUTED_MID * (GLM5_WIDTH / 256u) * 66u,
            (GLM5_WIDTH / 256u) * 66u,
            (uint64_t)GLM5_WIDTH * (GLM5_ROUTED_MID / 256u) * 84u,
            (GLM5_ROUTED_MID / 256u) * 84u,
            GLM5_WIDTH, GLM5_ROUTED_MID, GLM5_WIDTH,
            w->router_selected, w->router_weights,
            GLM5_EXPERTS, GLM5_EXPERTS_USED, 10.0f,
            w->ffn_hidden, il, n_tokens, &routed_mid_is_f16, false);
    } else if (ok && q4_residency == 0) {
        ok = declare_local_q4k_half_only(ctx, layer);
    }
    if (ok && !mixed_q2 && q4_residency == 1) {
        ok = ds4_gpu_routed_moe_batch_packed_q4k_tensor(
            w->routed_out, w->routed_gate, w->routed_up, w->routed_mid,
            w->routed_experts, ctx->model_map, ctx->model_size,
            f->gate_exps, f->up_exps, f->down_exps, GLM5_EXPERTS,
            q4_gate_row_bytes, q4_down_row_bytes,
            rank_mid_base, GLM5_RANK_MID, q4_down_base,
            q4_down_half_bytes, w->router_selected, w->router_weights,
            GLM5_EXPERTS_USED, 10.0f, w->ffn_hidden, il, n_tokens,
            &routed_mid_is_f16);
    } else if (ok && !mixed_q2) {
        const ds4_gpu_q4k_window_cache_config config = {
            .model_map = ctx->model_map,
            .gate_offset = f->gate_exps,
            .up_offset = f->up_exps,
            .down_offset = f->down_exps,
            .n_expert = GLM5_EXPERTS,
            .gate_row_base = rank_mid_base,
            .gate_row_count = GLM5_RANK_MID,
            .gate_column_byte_base = 0u,
            .gate_column_byte_count = q4_gate_row_bytes,
            .down_row_base = 0u,
            .down_row_count = GLM5_WIDTH,
            .down_column_byte_base = q4_down_base,
            .down_column_byte_count = q4_down_half_bytes,
            /* Preserve the established batch sizing.  The bounded scratch
             * cache is selected only by the scalar decode path above. */
            .slots = n_tokens > GLM5_EXPERTS / GLM5_EXPERTS_USED ?
                GLM5_EXPERTS : n_tokens * GLM5_EXPERTS_USED,
        };
        cache = ds4_gpu_q4k_window_cache_create(&config);
        ok = cache && ds4_gpu_routed_moe_batch_packed_q4k_window_tensor(
            w->routed_out, w->routed_gate, w->routed_up, w->routed_mid,
            w->routed_experts, cache, w->router_selected,
            w->router_weights, GLM5_EXPERTS_USED, 10.0f,
            w->ffn_hidden, il, n_tokens, &routed_mid_is_f16);
    }
    int shared_projection_ok = 1;
    if (ok && !routed_mid_is_f16 &&
        getenv("DS4_ROCM_GLM5_BATCH_SHARED_SERIAL") != NULL) {
        const uint64_t hc_row_bytes = (uint64_t)GLM5_WIDTH * sizeof(float);
        const uint64_t mid_row_bytes = (uint64_t)GLM5_RANK_MID * sizeof(float);
        for (uint32_t t = 0u; t < n_tokens && shared_projection_ok; ++t) {
            ds4_gpu_tensor *xv = ds4_gpu_tensor_view(
                w->ffn_hidden, (uint64_t)t * hc_row_bytes, hc_row_bytes);
            ds4_gpu_tensor *gv = ds4_gpu_tensor_view(
                w->shared_gate, (uint64_t)t * mid_row_bytes, mid_row_bytes);
            ds4_gpu_tensor *uv = ds4_gpu_tensor_view(
                w->shared_up, (uint64_t)t * mid_row_bytes, mid_row_bytes);
            shared_projection_ok = xv && gv && uv &&
                ds4_gpu_matmul_q8_0_pair_tensor(
                    gv, uv, ctx->model_map, ctx->model_size,
                    f->gate_shexp + (uint64_t)rank_mid_base * q8_gate_row_bytes,
                    f->up_shexp + (uint64_t)rank_mid_base * q8_gate_row_bytes,
                    GLM5_WIDTH, GLM5_RANK_MID, GLM5_RANK_MID, xv, 1u);
            ds4_gpu_tensor_free(xv);
            ds4_gpu_tensor_free(gv);
            ds4_gpu_tensor_free(uv);
        }
    } else if (ok && !routed_mid_is_f16) {
        shared_projection_ok = ds4_gpu_matmul_q8_0_pair_tensor(
            w->shared_gate, w->shared_up,
            ctx->model_map, ctx->model_size,
            f->gate_shexp + (uint64_t)rank_mid_base * q8_gate_row_bytes,
            f->up_shexp + (uint64_t)rank_mid_base * q8_gate_row_bytes,
            GLM5_WIDTH, GLM5_RANK_MID, GLM5_RANK_MID,
            w->ffn_hidden, n_tokens);
    }
    ok = ok && !routed_mid_is_f16 && shared_projection_ok &&
        ds4_gpu_swiglu_tensor(
            w->shared_mid, w->shared_gate, w->shared_up,
            n_tokens * GLM5_RANK_MID, 10.0f, 1.0f);
    int shared_down_ok = ok;
    if (shared_down_ok &&
        getenv("DS4_ROCM_GLM5_BATCH_SHARED_DOWN_SERIAL") != NULL) {
        const uint64_t mid_row_bytes =
            (uint64_t)GLM5_RANK_MID * sizeof(float);
        const uint64_t out_row_bytes =
            (uint64_t)GLM5_WIDTH * sizeof(float);
        for (uint32_t t = 0u; t < n_tokens && shared_down_ok; ++t) {
            ds4_gpu_tensor *xv = ds4_gpu_tensor_view(
                w->shared_mid, (uint64_t)t * mid_row_bytes, mid_row_bytes);
            ds4_gpu_tensor *ov = ds4_gpu_tensor_view(
                w->shared_out, (uint64_t)t * out_row_bytes, out_row_bytes);
            shared_down_ok = xv && ov &&
                ds4_gpu_matmul_q8_0_kslice_tensor(
                    ov, ctx->model_map, ctx->model_size, f->down_shexp,
                    GLM5_ROUTED_MID, rank_mid_base, GLM5_RANK_MID,
                    GLM5_WIDTH, xv, 0u);
            ds4_gpu_tensor_free(xv);
            ds4_gpu_tensor_free(ov);
        }
    } else if (shared_down_ok &&
               getenv("DS4_ROCM_GLM5_ENABLE_SHARED_DOWN_F32") != NULL &&
               getenv("DS4_ROCM_GLM5_DISABLE_SHARED_DOWN_F32") == NULL) {
        const int batch = ds4_rocm_q8_kslice_f32_rows_strided(
            w->shared_out, ctx->model_map, ctx->model_size,
            f->down_shexp, GLM5_ROUTED_MID, GLM5_WIDTH,
            rank_mid_base, GLM5_RANK_MID, w->shared_mid,
            0u, n_tokens, GLM5_RANK_MID);
        if (batch >= 0) {
            shared_down_ok = batch;
        } else {
            /* A backend may not expose the multi-row F32 entry point. Keep
             * the established scalar-F32 arithmetic rather than failing the
             * whole GLM-5 batch or silently switching to Q8 activations. */
            const uint64_t mid_row_bytes =
                (uint64_t)GLM5_RANK_MID * sizeof(float);
            const uint64_t out_row_bytes =
                (uint64_t)GLM5_WIDTH * sizeof(float);
            for (uint32_t t = 0u; t < n_tokens && shared_down_ok; ++t) {
                ds4_gpu_tensor *xv = ds4_gpu_tensor_view(
                    w->shared_mid, (uint64_t)t * mid_row_bytes,
                    mid_row_bytes);
                ds4_gpu_tensor *ov = ds4_gpu_tensor_view(
                    w->shared_out, (uint64_t)t * out_row_bytes,
                    out_row_bytes);
                shared_down_ok = xv && ov &&
                    ds4_gpu_matmul_q8_0_kslice_tensor(
                        ov, ctx->model_map, ctx->model_size,
                        f->down_shexp, GLM5_ROUTED_MID, rank_mid_base,
                        GLM5_RANK_MID, GLM5_WIDTH, xv, 0u);
                ds4_gpu_tensor_free(xv);
                ds4_gpu_tensor_free(ov);
            }
        }
    } else if (shared_down_ok) {
        shared_down_ok = ds4_gpu_matmul_q8_0_kslice_rows_tensor(
            w->shared_out, ctx->model_map, ctx->model_size,
            f->down_shexp, GLM5_ROUTED_MID, GLM5_WIDTH,
            rank_mid_base, GLM5_RANK_MID, w->shared_mid, n_tokens);
    }
    ok = ok && shared_down_ok &&
        ds4_gpu_add_tensor(ctx->tp_big_out, w->routed_out, w->shared_out,
                           (uint32_t)elements) &&
        tp_exchange_rows(ctx, il, DS4_TP_GATE_FFN, n_tokens) &&
        ds4_gpu_add_tensor(w->down, ctx->tp_big_out, ctx->tp_big_in,
                           (uint32_t)elements) &&
        ds4_gpu_hc_expand_split_tensor(
            hc_out, w->down, w->after_attention, w->ffn_split,
            GLM5_WIDTH, GLM5_HC) &&
        ds4_gpu_synchronize();
    if (!ok) ds4_gpu_synchronize();
    /* Only the streaming branch owns these layer-local descriptors/cache.
     * Full-trunk packed residency is process-scoped and must survive. */
    if (!mixed_q2 && q4_residency == 0)
        ds4_gpu_q4k_packed_slice_release_all();
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
    int ok = kda_attention_one(ctx, il, state, w, hc_in);
#ifdef DS4_TP_TEST_HOOKS
    if (ok) ok =
        trace_tensor(ctx, il, token_ordinal, "kda_out.f32",
                     kda_attention_result_for_trace(ctx, w, 1u),
                     (uint64_t)GLM5_WIDTH * sizeof(float)) &&
        trace_tensor(ctx, il, token_ordinal, "input_hc.f32", hc_in,
                     (uint64_t)GLM5_HC_WIDTH * sizeof(float)) &&
        trace_tensor(ctx, il, token_ordinal, "after_attn.f32",
                     w->after_attention,
                     (uint64_t)GLM5_HC_WIDTH * sizeof(float));
#endif
    if (ok) ok = routed_ffn_one(ctx, il, token_ordinal, w, hc_out);
#ifdef DS4_TP_TEST_HOOKS
    if (ok) ok = trace_routed_ffn(
        ctx, il, token_ordinal, hc_out, w);
#endif
    if (!ok) {
        route_failure_stats("layer_hc_in", hc_in, GLM5_HC_WIDTH);
        route_failure_stats("attention_local", w->attention, GLM5_WIDTH);
        route_failure_stats("attention_hc_out", w->after_attention,
                            GLM5_HC_WIDTH);
        ds4_glm5_next_state_invalidate(state);
    }
    return ok;
}

static int kda_routed_rows_forward(const ds4_glm5_next_exec_ctx *ctx,
                                   uint32_t il,
                                   ds4_glm5_next_state *state,
                                   ds4_glm5_next_workspace *w,
                                   const ds4_gpu_tensor *hc_in,
                                   ds4_gpu_tensor *hc_out,
                                   uint32_t n_tokens) {
    ds4_glm5_kda_layer_state *kda = &state->kda.layer[il];
    if (!tp_context_valid_bytes(
            ctx, (uint64_t)n_tokens * GLM5_WIDTH * sizeof(float)) ||
        !kda->valid || !kda->recurrent ||
        kda->token_count > UINT32_MAX ||
        n_tokens > UINT32_MAX - (uint32_t)kda->token_count) return 0;
    const uint32_t token_ordinal = (uint32_t)kda->token_count;
    int ok = kda_attention_rows(ctx, il, state, w, hc_in, n_tokens);
#ifdef DS4_TP_TEST_HOOKS
    uint32_t trace_token = token_ordinal + n_tokens - 1u;
    uint32_t trace_row = n_tokens - 1u;
    if (ctx->trace_token != UINT32_MAX &&
        ctx->trace_token >= token_ordinal &&
        ctx->trace_token - token_ordinal < n_tokens) {
        trace_token = ctx->trace_token;
        trace_row = ctx->trace_token - token_ordinal;
    }
    const int trace_post_only =
        getenv("DS4_GLM5_TRACE_POST_ONLY") != NULL;
    if (ok && !trace_post_only) ok =
        trace_tensor_row(ctx, il, trace_token, "input_hc.f32", hc_in,
                         trace_row,
                         (uint64_t)GLM5_HC_WIDTH * sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "after_attn.f32",
                         w->after_attention, trace_row,
                         (uint64_t)GLM5_HC_WIDTH * sizeof(float));
    if (ok && !trace_post_only) ok = trace_same_input_routed_ffn(
        ctx, il, trace_token,
        w->after_attention, trace_row);
#endif
    if (ok) ok = routed_ffn_rows(
        ctx, il, token_ordinal, w, hc_out, n_tokens);
#ifdef DS4_TP_TEST_HOOKS
    if (ok) ok = layer_completion_diagnostic(hc_out, n_tokens);
    if (ok) ok = hc_batch_hash_trace(ctx, il, hc_out, n_tokens);
#endif
#ifdef DS4_TP_TEST_HOOKS
    if (ok) ok =
        trace_tensor_row(ctx, il, trace_token, "ffn_hidden.f32",
                         w->ffn_hidden, trace_row,
                         (uint64_t)GLM5_WIDTH * sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "routed_gate.f32",
                         w->routed_gate, trace_row,
                         (uint64_t)GLM5_EXPERTS_USED * GLM5_ROUTED_MID *
                             sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "routed_up.f32",
                         w->routed_up, trace_row,
                         (uint64_t)GLM5_EXPERTS_USED * GLM5_ROUTED_MID *
                             sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "routed_mid.f32",
                         w->routed_mid, trace_row,
                         (uint64_t)GLM5_EXPERTS_USED * GLM5_ROUTED_MID *
                             sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "routed_experts.f32",
                         w->routed_experts, trace_row,
                         (uint64_t)GLM5_EXPERTS_USED * GLM5_WIDTH *
                             sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "router_ids.i32",
                         w->router_selected, trace_row,
                         (uint64_t)GLM5_EXPERTS_USED * sizeof(int32_t)) &&
        trace_tensor_row(ctx, il, trace_token, "router_weights.f32",
                         w->router_weights, trace_row,
                         (uint64_t)GLM5_EXPERTS_USED * sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "routed_out.f32",
                         w->routed_out, trace_row,
                         (uint64_t)GLM5_WIDTH * sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "shared_out.f32",
                         w->shared_out, trace_row,
                         (uint64_t)GLM5_WIDTH * sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "output_hc.f32", hc_out,
                         trace_row,
                         (uint64_t)GLM5_HC_WIDTH * sizeof(float));
#endif
    if (!ok) ds4_glm5_next_state_invalidate(state);
    return ok;
}

static int mla_routed_dense_selection_forward(const ds4_glm5_next_exec_ctx *ctx,
                                    uint32_t il,
                                    ds4_glm5_next_state *state,
                                    ds4_glm5_next_workspace *w,
                                    const ds4_gpu_tensor *hc_in,
                                    ds4_gpu_tensor *hc_out) {
    ds4_glm5_next_mla_state *mla = &state->mla[il];
    uint32_t visible = 0u;
    uint32_t tail_slot = 0u, pool_index = 0u;
    bool publish_pool = false;
    if (!tp_context_valid(ctx) || !mla->valid || !mla->compact_kv ||
        !mla->index_pool || !mla->index_tail || !mla->pool_gate_tail ||
        mla->owner != state || mla->first_valid != 0u ||
        !ds4_glm5_next_mla_append_plan(
            mla, &tail_slot, &pool_index, &publish_pool)) return 0;
    const uint32_t token = mla->token_count;
    const int dense = ds4_glm5_next_mla_dense_selection_visible(
        mla->token_count, mla->capacity_tokens, &visible);
    const int attention_ok = dense ?
        mla_dense_selection_attention(
            ctx, il, state, w, hc_in, visible, tail_slot,
            pool_index, publish_pool) :
        mla_sparse_selection_attention(
            ctx, il, state, w, hc_in, tail_slot,
            pool_index, publish_pool, DS4_GLM5_NEXT_INDEX_TOP_K);
    const int ok = attention_ok &&
                   trace_mla_attention(ctx, il, token, hc_in, w) &&
                   routed_ffn_one(ctx, il, token, w, hc_out) &&
                   trace_routed_ffn(ctx, il, token, hc_out, w);
    if (!ok) {
        route_failure_stats("mla_layer_hc_in", hc_in, GLM5_HC_WIDTH);
        route_failure_stats("mla_attention_local", w->attention, GLM5_WIDTH);
        route_failure_stats("mla_attention_hc_out", w->after_attention,
                            GLM5_HC_WIDTH);
        ds4_glm5_next_state_invalidate(state);
        return 0;
    }
    if (!ds4_glm5_next_mla_append_commit(mla)) {
        ds4_glm5_next_state_invalidate(state);
        return 0;
    }
    return 1;
}

static int mla_routed_dense_selection_rows_forward(
        const ds4_glm5_next_exec_ctx *ctx,
        uint32_t il,
        ds4_glm5_next_state *state,
        ds4_glm5_next_workspace *w,
        const ds4_gpu_tensor *hc_in,
        ds4_gpu_tensor *hc_out,
        uint32_t n_tokens) {
    ds4_glm5_next_mla_state *mla = &state->mla[il];
    if (!tp_context_valid_bytes(
            ctx, (uint64_t)n_tokens * GLM5_WIDTH * sizeof(float)) ||
        !mla->valid || !mla->compact_kv || !mla->index_pool ||
        !mla->index_tail || !mla->pool_gate_tail || mla->owner != state ||
        mla->first_valid != 0u || mla->token_count > UINT32_MAX ||
        mla->token_count > mla->capacity_tokens ||
        mla->token_count > DS4_GLM5_NEXT_INDEX_TOP_K ||
        n_tokens == 0u ||
        n_tokens > mla->capacity_tokens - mla->token_count ||
        n_tokens > DS4_GLM5_NEXT_INDEX_TOP_K - mla->token_count) {
        return 0;
    }
    const uint32_t token_ordinal = mla->token_count;
#ifdef DS4_TP_TEST_HOOKS
    uint32_t trace_token = token_ordinal + n_tokens - 1u;
    uint32_t trace_row = n_tokens - 1u;
    if (ctx->trace_token != UINT32_MAX &&
        ctx->trace_token >= token_ordinal &&
        ctx->trace_token - token_ordinal < n_tokens) {
        trace_token = ctx->trace_token;
        trace_row = ctx->trace_token - token_ordinal;
    }
#else
    const uint32_t trace_token = token_ordinal;
    const uint32_t trace_row = n_tokens - 1u;
#endif
    const int ok =
        mla_dense_selection_attention_rows(
            ctx, il, state, w, hc_in, n_tokens) &&
#ifdef DS4_TP_TEST_HOOKS
        glm5_capture_mla_stages(
            ctx, il, mla, w, hc_in, token_ordinal, n_tokens) &&
#endif
        trace_tensor_row(ctx, il, trace_token, "input_hc.f32",
                         hc_in, trace_row,
                         (uint64_t)GLM5_HC_WIDTH * sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "attn_flat.f32",
                         w->hc_flat, trace_row,
                         (uint64_t)GLM5_HC_WIDTH * sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "attn_mix.f32",
                         w->hc_mix, trace_row,
                         (uint64_t)GLM5_HC_MIX * sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "attn_split.f32",
                         w->hc_split, trace_row,
                         (uint64_t)GLM5_HC_MIX * sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "attn_collapsed.f32",
                         w->collapsed, trace_row,
                         (uint64_t)GLM5_WIDTH * sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "attn_hidden.f32",
                         w->ffn_hidden, trace_row,
                         (uint64_t)GLM5_WIDTH * sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "mla_q_a.f32",
                         w->mla_q_a, trace_row,
                         (uint64_t)GLM5_Q_RANK * sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "mla_q_resid.f32",
                         w->mla_q_resid, trace_row,
                         (uint64_t)GLM5_Q_RANK * sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "mla_query.f32",
                         w->mla_query, trace_row,
                         (uint64_t)GLM5_HEADS * GLM5_HEAD_DIM *
                             sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "mla_kv_raw.f32",
                         w->mla_kv_raw, trace_row,
                         (uint64_t)GLM5_KV_LORA * sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "mla_kv_norm.f32",
                         w->mla_kv_norm, trace_row,
                         (uint64_t)GLM5_KV_LORA * sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "mla_compact_kv.f32",
                         mla->compact_kv, token_ordinal + trace_row,
                         (uint64_t)GLM5_KV_LORA * sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "mla_qk_low.f32",
                         w->mla_qk_low, trace_row,
                         (uint64_t)GLM5_HEADS * GLM5_KV_LORA *
                             sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "mla_lora_out.f32",
                         w->routed_experts, trace_row,
                         (uint64_t)GLM5_HEADS * GLM5_KV_LORA *
                             sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "mla_heads.f32",
                         w->mla_heads, trace_row,
                         (uint64_t)GLM5_HEADS * GLM5_HEAD_DIM *
                             sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "attn_local.f32",
                         ctx->tp_big_out, trace_row,
                         (uint64_t)GLM5_WIDTH * sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "attn_peer.f32",
                         ctx->tp_big_in, trace_row,
                         (uint64_t)GLM5_WIDTH * sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "attn_sum.f32",
                         w->attention, trace_row,
                         (uint64_t)GLM5_WIDTH * sizeof(float)) &&
        trace_tensor_row(ctx, il, trace_token, "after_attn.f32",
                         w->after_attention, trace_row,
                         (uint64_t)GLM5_HC_WIDTH * sizeof(float)) &&
        routed_ffn_rows(
            ctx, il, token_ordinal, w, hc_out, n_tokens);
#ifdef DS4_TP_TEST_HOOKS
    if (ok && !layer_completion_diagnostic(hc_out, n_tokens))
        return 0;
    if (ok && !hc_batch_hash_trace(ctx, il, hc_out, n_tokens))
        return 0;
#endif
    if (ok && !trace_tensor_row(ctx, il, trace_token, "output_hc.f32",
                                hc_out, trace_row,
                                (uint64_t)GLM5_HC_WIDTH * sizeof(float)))
        return 0;
    if (ok && !trace_tensor_row(ctx, il, trace_token, "routed_out.f32",
                                w->routed_out, trace_row,
                                (uint64_t)GLM5_WIDTH * sizeof(float)))
        return 0;
    if (ok && !trace_tensor_row(ctx, il, trace_token, "routed_gate.f32",
                                w->routed_gate, trace_row,
                                (uint64_t)GLM5_EXPERTS_USED * GLM5_ROUTED_MID * sizeof(float)))
        return 0;
    if (ok && !trace_tensor_row(ctx, il, trace_token, "routed_up.f32",
                                w->routed_up, trace_row,
                                (uint64_t)GLM5_EXPERTS_USED * GLM5_ROUTED_MID * sizeof(float)))
        return 0;
    if (ok && !trace_tensor_row(ctx, il, trace_token, "routed_mid.f32",
                                w->routed_mid, trace_row,
                                (uint64_t)GLM5_EXPERTS_USED * GLM5_ROUTED_MID * sizeof(float)))
        return 0;
    if (ok && !trace_tensor_row(ctx, il, trace_token, "routed_experts.f32",
                                w->routed_experts, trace_row,
                                (uint64_t)GLM5_EXPERTS_USED * GLM5_WIDTH * sizeof(float)))
        return 0;
    if (ok && !trace_tensor_row(ctx, il, trace_token,
                                "ffn_down.f32", w->down, trace_row,
                                (uint64_t)GLM5_WIDTH * sizeof(float)))
        return 0;
    if (ok && !trace_tensor_row(ctx, il, trace_token, "shared_out.f32",
                                w->shared_out, trace_row,
                                (uint64_t)GLM5_WIDTH * sizeof(float)))
        return 0;
    if (ok && !trace_tensor_row(ctx, il, trace_token, "ffn_hidden.f32",
                                w->ffn_hidden, trace_row,
                                (uint64_t)GLM5_WIDTH * sizeof(float)))
        return 0;
    if (ok && !trace_tensor_row(ctx, il, trace_token, "router_ids.i32",
                                w->router_selected, trace_row,
                                (uint64_t)GLM5_EXPERTS_USED * sizeof(int32_t)))
        return 0;
    if (ok && !trace_tensor_row(ctx, il, trace_token,
                                "router_weights.f32", w->router_weights,
                                trace_row,
                                (uint64_t)GLM5_EXPERTS_USED * sizeof(float)))
        return 0;
    if (!ok) {
        ds4_glm5_next_state_invalidate(state);
        return 0;
    }
    for (uint32_t t = 0u; t < n_tokens; ++t) {
        if (!ds4_glm5_next_mla_append_commit(mla)) {
            ds4_glm5_next_state_invalidate(state);
            return 0;
        }
    }
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
        const int ok = dense_kda_forward(ctx, il, state, w, hc_in, hc_out);
        return ok && validate_layer_finite(ctx, il, hc_out);
    }
    if (layer->attention == DS4_GLM5_NEXT_ATTN_MLA &&
        layer->ffn == DS4_GLM5_NEXT_FFN_ROUTED &&
        state->mla[il].compact_kv && !state->kda.layer[il].recurrent) {
        const int ok = mla_routed_dense_selection_forward(
            ctx, il, state, w, hc_in, hc_out);
        return ok && validate_layer_finite(ctx, il, hc_out);
    }
    if (layer->attention == DS4_GLM5_NEXT_ATTN_KDA &&
        layer->ffn == DS4_GLM5_NEXT_FFN_ROUTED &&
        state->kda.layer[il].recurrent && !state->mla[il].compact_kv) {
        const int ok = kda_routed_one_forward(ctx, il, state, w, hc_in, hc_out);
        return ok && validate_layer_finite(ctx, il, hc_out);
    }
    return 0;
}

int ds4_glm5_next_layer_forward_batch(const ds4_glm5_next_exec_ctx *ctx,
                                      uint32_t il,
                                      ds4_glm5_next_state *state,
                                      ds4_glm5_next_workspace *w,
                                      const ds4_gpu_tensor *hc_in,
                                      ds4_gpu_tensor *hc_out,
                                      uint32_t n_tokens) {
    const uint64_t hc_row_bytes =
        (uint64_t)GLM5_HC_WIDTH * sizeof(float);
    if (!context_valid(ctx) || !state || !state->valid || !w || !hc_in ||
        !hc_out || hc_in == hc_out || n_tokens == 0u ||
        w->capacity_tokens != n_tokens ||
        (uint64_t)n_tokens > UINT64_MAX / hc_row_bytes ||
        ds4_gpu_tensor_bytes(hc_in) != (uint64_t)n_tokens * hc_row_bytes ||
        ds4_gpu_tensor_bytes(hc_out) != (uint64_t)n_tokens * hc_row_bytes ||
        il >= ctx->model->trunk_count ||
        state->layer_count != ctx->model->trunk_count ||
        state->kda.layer_count != ctx->model->trunk_count) return 0;
    if (n_tokens == 1u) {
        return ds4_glm5_next_layer_forward(
            ctx, il, state, w, hc_in, hc_out);
    }
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    if (layer->attention == DS4_GLM5_NEXT_ATTN_MLA &&
        layer->ffn == DS4_GLM5_NEXT_FFN_ROUTED &&
        state->mla[il].compact_kv && !state->kda.layer[il].recurrent) {
        return mla_routed_dense_selection_rows_forward(
            ctx, il, state, w, hc_in, hc_out, n_tokens);
    }
    if (layer->attention != DS4_GLM5_NEXT_ATTN_KDA ||
        !state->kda.layer[il].recurrent || state->mla[il].compact_kv)
        return 0;
    if (layer->ffn == DS4_GLM5_NEXT_FFN_DENSE) {
        return dense_kda_forward_rows(
            ctx, il, state, w, hc_in, hc_out, n_tokens);
    }
    if (layer->ffn == DS4_GLM5_NEXT_FFN_ROUTED) {
        return kda_routed_rows_forward(
            ctx, il, state, w, hc_in, hc_out, n_tokens);
    }
    return 0;
}

#ifdef DS4_TP_TEST_HOOKS
int ds4_glm5_next_kda_attention_forward_test(
        const ds4_glm5_next_exec_ctx *ctx,
        uint32_t il,
        ds4_glm5_next_state *state,
        ds4_glm5_next_workspace *w,
        const ds4_gpu_tensor *hc_in,
        ds4_gpu_tensor *hc_out,
        uint32_t n_tokens) {
    const uint64_t row_bytes =
        (uint64_t)GLM5_HC_WIDTH * sizeof(float);
    if (!context_valid(ctx) || !state || !state->valid || !w || !hc_in ||
        !hc_out || hc_in == hc_out || n_tokens == 0u ||
        w->capacity_tokens != n_tokens || il >= ctx->model->trunk_count ||
        ctx->model->layer[il].attention != DS4_GLM5_NEXT_ATTN_KDA ||
        !state->kda.layer[il].recurrent || state->mla[il].compact_kv ||
        ds4_gpu_tensor_bytes(hc_in) != (uint64_t)n_tokens * row_bytes ||
        ds4_gpu_tensor_bytes(hc_out) != (uint64_t)n_tokens * row_bytes) {
        return 0;
    }
    const int ok = kda_attention_rows(
        ctx, il, state, w, hc_in, n_tokens) &&
        ds4_gpu_tensor_copy(hc_out, 0u, w->after_attention, 0u,
                            (uint64_t)n_tokens * row_bytes) &&
        ds4_gpu_synchronize();
    if (!ok) ds4_glm5_next_state_invalidate(state);
    return ok;
}

int ds4_glm5_next_mla_attention_forward_test(
        const ds4_glm5_next_exec_ctx *ctx,
        uint32_t il,
        ds4_glm5_next_state *state,
        ds4_glm5_next_workspace *w,
        const ds4_gpu_tensor *hc_in,
        ds4_gpu_tensor *hc_out,
        uint32_t n_tokens) {
    const uint64_t row_bytes =
        (uint64_t)GLM5_HC_WIDTH * sizeof(float);
    if (!context_valid(ctx) || !state || !state->valid || !w || !hc_in ||
        !hc_out || hc_in == hc_out || n_tokens == 0u ||
        w->capacity_tokens != n_tokens ||
        il >= ctx->model->trunk_count ||
        ctx->model->layer[il].attention != DS4_GLM5_NEXT_ATTN_MLA ||
        !state->mla[il].compact_kv || state->kda.layer[il].recurrent ||
        ds4_gpu_tensor_bytes(hc_in) != (uint64_t)n_tokens * row_bytes ||
        ds4_gpu_tensor_bytes(hc_out) != (uint64_t)n_tokens * row_bytes) {
        return 0;
    }
    ds4_glm5_next_mla_state *mla = &state->mla[il];
    int ok = 0;
    if (n_tokens == 1u) {
        uint32_t visible = 0u, tail_slot = 0u, pool_index = 0u;
        bool publish_pool = false;
        ok = ds4_glm5_next_mla_append_plan(
                 mla, &tail_slot, &pool_index, &publish_pool);
        if (ok && ds4_glm5_next_mla_dense_selection_visible(
                      mla->token_count, mla->capacity_tokens, &visible)) {
            ok = mla_dense_selection_attention(
                ctx, il, state, w, hc_in, visible, tail_slot,
                pool_index, publish_pool);
        } else if (ok) {
            ok = mla_sparse_selection_attention(
                ctx, il, state, w, hc_in, tail_slot,
                pool_index, publish_pool, DS4_GLM5_NEXT_INDEX_TOP_K);
        }
    } else {
        ok = mla_dense_selection_attention_rows(
            ctx, il, state, w, hc_in, n_tokens);
    }
    if (ok) {
        ok = ds4_gpu_tensor_copy(
                 hc_out, 0u, w->after_attention, 0u,
                 (uint64_t)n_tokens * row_bytes) &&
             ds4_gpu_synchronize();
    }
    if (!ok) {
        ds4_glm5_next_state_invalidate(state);
        return 0;
    }
    for (uint32_t t = 0u; t < n_tokens; ++t) {
        if (!ds4_glm5_next_mla_append_commit(mla)) {
            ds4_glm5_next_state_invalidate(state);
            return 0;
        }
    }
    return 1;
}

int ds4_glm5_next_mla_sparse_attention_forward_test(
        const ds4_glm5_next_exec_ctx *ctx,
        uint32_t il,
        ds4_glm5_next_state *state,
        ds4_glm5_next_workspace *w,
        const ds4_gpu_tensor *hc_in,
        ds4_gpu_tensor *hc_out,
        uint32_t top_k) {
    const uint64_t row_bytes =
        (uint64_t)GLM5_HC_WIDTH * sizeof(float);
    if (!context_valid(ctx) || !state || !state->valid || !w || !hc_in ||
        !hc_out || hc_in == hc_out || w->capacity_tokens != 1u ||
        il >= ctx->model->trunk_count || top_k == 0u ||
        top_k > DS4_GLM5_NEXT_INDEX_TOP_K ||
        top_k % GLM5_INDEX_POOL != 0u ||
        ctx->model->layer[il].attention != DS4_GLM5_NEXT_ATTN_MLA ||
        !state->mla[il].compact_kv || state->kda.layer[il].recurrent ||
        ds4_gpu_tensor_bytes(hc_in) != row_bytes ||
        ds4_gpu_tensor_bytes(hc_out) != row_bytes) {
        return 0;
    }
    ds4_glm5_next_mla_state *mla = &state->mla[il];
    uint32_t tail_slot = 0u, pool_index = 0u;
    bool publish_pool = false;
    const int ok = ds4_glm5_next_mla_append_plan(
                       mla, &tail_slot, &pool_index, &publish_pool) &&
                   mla_sparse_selection_attention(
                       ctx, il, state, w, hc_in, tail_slot, pool_index,
                       publish_pool, top_k) &&
                   ds4_gpu_tensor_copy(
                       hc_out, 0u, w->after_attention, 0u, row_bytes) &&
                   ds4_gpu_synchronize();
    if (!ok || !ds4_glm5_next_mla_append_commit(mla)) {
        ds4_glm5_next_state_invalidate(state);
        return 0;
    }
    return 1;
}

int ds4_glm5_next_mla_sparse_selection_read_test(
        const ds4_glm5_next_workspace *w,
        int32_t *selected,
        uint32_t count) {
    if (!w || !w->mla_selected_token || !selected || count == 0u ||
        count > DS4_GLM5_NEXT_INDEX_TOP_K + GLM5_INDEX_POOL - 1u) {
        return 0;
    }
    return ds4_gpu_tensor_read(
        w->mla_selected_token, 0u, selected,
        (uint64_t)count * sizeof(*selected));
}

int ds4_glm5_next_mla_sparse_indexer_read_test(
        const ds4_glm5_next_workspace *w,
        uint32_t n_pools,
        uint32_t selected_count,
        float *query,
        float *weights,
        float *scores,
        uint32_t *selected_pools) {
    if (!w || !query || !weights || !scores || !selected_pools ||
        n_pools == 0u ||
        n_pools > w->sparse_pool_capacity || selected_count == 0u ||
        selected_count > n_pools || !w->mla_index_q ||
        !w->mla_index_weights || !w->mla_pool_scores ||
        !w->mla_selected_pool) return 0;
    const uint64_t q_bytes =
        (uint64_t)GLM5_INDEX_HEADS * GLM5_INDEX_DIM * sizeof(float);
    const uint64_t weight_bytes =
        (uint64_t)GLM5_INDEX_HEADS * sizeof(float);
    const uint64_t score_bytes = (uint64_t)n_pools * sizeof(float);
    const uint64_t selected_bytes =
        (uint64_t)selected_count * sizeof(uint32_t);
    return ds4_gpu_tensor_read(w->mla_index_q, 0u, query, q_bytes) &&
           ds4_gpu_tensor_read(w->mla_index_weights, 0u, weights,
                               weight_bytes) &&
           ds4_gpu_tensor_read(w->mla_pool_scores, 0u, scores,
                               score_bytes) &&
           ds4_gpu_tensor_read(w->mla_selected_pool, 0u, selected_pools,
                               selected_bytes);
}
#endif
