#include "ds4_glm5_next_exec.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ds4_gpu.h"
#include "ds4_tp.h"
#include "ds4_glm5_route_profile.h"
#include "ds4_glm5_expert_pairs.h"
#ifdef DS4_ROCM_BUILD
#include "ds4_gpu_mgpu.h"
#endif

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
    GLM5_SELECTED_STRIDE =
        DS4_GLM5_NEXT_INDEX_TOP_K + GLM5_INDEX_POOL - 1,
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
_Static_assert(GLM5_EXPERTS == DS4_GLM5_ROUTE_PROFILE_EXPERTS &&
               GLM5_EXPERTS_USED == DS4_GLM5_ROUTE_PROFILE_USED,
               "route diagnostics must match the engine expert layout");

static double glm5_exec_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
int ds4_rocm_glm5_expert_six_admit(ds4_glm5_expert_six_plan *p,
        const ds4_glm5_expert_six_args *a) { (void)p; (void)a; return 0; }
#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
int ds4_rocm_glm5_expert_six_begin(const ds4_glm5_expert_six_plan *p,
        const ds4_glm5_expert_groups *g, const int32_t *ids, const float *weights) {
    (void)p; (void)g; (void)ids; (void)weights; return 0;
}
#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
int ds4_rocm_glm5_expert_six_down_row(const ds4_glm5_expert_six_plan *p, uint32_t row) {
    (void)p; (void)row; return 0;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
int ds4_rocm_glm5_dense_q8_small_m(
        ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
        const void *model_map, uint64_t model_size,
        uint64_t offset0, uint64_t offset1,
        uint32_t in_dim, uint32_t out_dim,
        const ds4_gpu_tensor *x, uint32_t tokens) {
    (void)out0; (void)out1; (void)model_map; (void)model_size;
    (void)offset0; (void)offset1; (void)in_dim; (void)out_dim;
    (void)x; (void)tokens;
    return 0;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
int ds4_rocm_glm5_shared_q8_small_m(
        ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
        const void *model_map, uint64_t model_size,
        uint64_t offset0, uint64_t offset1,
        uint32_t in_dim, uint32_t out_dim, uint64_t row_bytes,
        uint32_t k_first, const ds4_gpu_tensor *x, uint32_t tokens) {
    (void)out0; (void)out1; (void)model_map; (void)model_size;
    (void)offset0; (void)offset1; (void)in_dim; (void)out_dim;
    (void)row_bytes; (void)k_first; (void)x; (void)tokens;
    return 0;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
int ds4_rocm_glm5_mla_output_q8_small_m(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t offset, uint32_t full_in_dim, uint32_t k_first,
        uint32_t in_dim, uint32_t out_dim, uint64_t row_bytes,
        const ds4_gpu_tensor *x, uint32_t tokens) {
    (void)out; (void)model_map; (void)model_size; (void)offset;
    (void)full_in_dim; (void)k_first; (void)in_dim; (void)out_dim;
    (void)row_bytes; (void)x; (void)tokens;
    return 0;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
int ds4_rocm_glm5_mla_output_q8_small_m_supported(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t offset, uint32_t full_in_dim, uint32_t k_first,
        uint32_t in_dim, uint32_t out_dim, uint64_t row_bytes,
        const ds4_gpu_tensor *x, uint32_t tokens) {
    (void)out; (void)model_map; (void)model_size; (void)offset;
    (void)full_in_dim; (void)k_first; (void)in_dim; (void)out_dim;
    (void)row_bytes; (void)x; (void)tokens;
    return 0;
}

static void glm5_phase_trace(const ds4_glm5_next_exec_ctx *ctx,
                             const char *phase, uint32_t layer,
                             uint32_t n_tokens) {
    const char *enabled = getenv("DS4_GLM5_PHASE_TRACE");
    if (!enabled || strcmp(enabled, "1") != 0 || layer != 0u) return;
    fprintf(stderr,
            "ds4: GLM5 phase rank=%u time=%.9f phase=%s layer=%u rows=%u seq=%llu\n",
            ctx ? ctx->tp_rank : UINT32_MAX, glm5_exec_now_sec(), phase,
            layer, n_tokens,
            (unsigned long long)(ctx && ctx->tp_sequence ?
                                 *ctx->tp_sequence : 0u));
    fflush(stderr);
}

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

/* ROCm-only selected-list sparse attention.  A positive return means the
 * exact head-shared rows path ran, zero is a hard failure, and -1 means the
 * backend does not provide this research specialization. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
int ds4_rocm_glm5_sparse_attention_exact_rows(
        ds4_gpu_tensor *lora_out,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *selected,
        uint32_t n_tokens,
        uint32_t selected_stride,
        uint32_t cache_cap) {
    (void)lora_out;
    (void)qk_low;
    (void)kv_lora_cache;
    (void)selected;
    (void)n_tokens;
    (void)selected_stride;
    (void)cache_cap;
    return -1;
}

/* Lane-B diagnostic counterpart. It stays unreachable unless the explicit
 * sparse-attention comparison switch is selected. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
int ds4_rocm_glm5_sparse_attention_f16_gemm_rows(
        ds4_gpu_tensor *lora_out,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *selected,
        uint32_t n_tokens,
        uint32_t selected_stride,
        uint32_t cache_cap) {
    (void)lora_out;
    (void)qk_low;
    (void)kv_lora_cache;
    (void)selected;
    (void)n_tokens;
    (void)selected_stride;
    (void)cache_cap;
    return -1;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
int ds4_rocm_glm5_sparse_attention_f16_gemm_reserve(void) {
    return -1;
}

/* Optimized builds can fold the non-ROCm selector arm, but debug and Metal
 * builds must not acquire a hard link dependency on the ROCm/CUDA scorer. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
int ds4_gpu_glm_indexer_scores_pool_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache,
        const ds4_gpu_tensor *pool_valid,
        uint32_t n_pools, uint32_t score_stride, uint32_t n_tokens,
        uint32_t pos0, uint32_t pool_size, uint32_t n_head,
        uint32_t head_dim, float scale, bool cache_f16) {
    (void)scores;
    (void)q;
    (void)weights;
    (void)indexer_key_cache;
    (void)pool_valid;
    (void)n_pools;
    (void)score_stride;
    (void)n_tokens;
    (void)pos0;
    (void)pool_size;
    (void)n_head;
    (void)head_dim;
    (void)scale;
    (void)cache_f16;
    return 0;
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
_Static_assert(GLM5_EXPERTS_USED * GLM5_WIDTH ==
                   GLM5_HEADS * GLM5_KV_LORA,
               "MLA attention alias must exactly match routed expert row");

struct ds4_glm5_next_workspace {
    uint32_t capacity_tokens;
    uint64_t expert_six_calls[2], expert_tail_calls[2], expert_pair_reads[2];
    uint32_t expert_pair_rank, expert_pair_mode;
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
    bool draft_only;
    /* One dedicated workspace stays with one state/model/link for its lifetime. */
    const ds4_glm5_next_state *draft_owner;
    ds4_glm5_next_exec_ctx draft_context;
    uint32_t draft_runtime_features;
    uint64_t draft_prefill_config;
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

/* The decode probe relies on same-default-stream ordering.  Keep it
 * explicitly opt-in so prefill and any path with a different lifetime or
 * stream contract retain the historical fence. */
static int glm5_decode_async_layer_enabled(
        const ds4_glm5_next_workspace *w, uint32_t n_tokens) {
    const char *enabled = getenv("DS4_ROCM_GLM5_DECODE_ASYNC_LAYER");
    return w && w->decode_phase && n_tokens == 1u && enabled &&
           strcmp(enabled, "1") == 0;
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
    if (w->expert_pair_mode) fprintf(stderr,
        "VERIFY_EXPERT_PAIRS rank=%u mode=%u workspace=%u kda_six=%llu mla_six=%llu kda_tail=%llu mla_tail=%llu kda_pairs=%llu mla_pairs=%llu\n",
        w->expert_pair_rank, w->expert_pair_mode, w->capacity_tokens,
        (unsigned long long)w->expert_six_calls[0], (unsigned long long)w->expert_six_calls[1],
        (unsigned long long)w->expert_tail_calls[0], (unsigned long long)w->expert_tail_calls[1],
        (unsigned long long)w->expert_pair_reads[0], (unsigned long long)w->expert_pair_reads[1]);
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
    w->mla_index_q = f32_rows(
        capacity_tokens, (uint64_t)GLM5_INDEX_HEADS * GLM5_INDEX_DIM);
    w->mla_index_weights = f32_rows(capacity_tokens, GLM5_INDEX_HEADS);
    /* Pool scores are row-major [query tile][compact pool].  Keeping the
     * column width at the full context lets a tile score newly published
     * pools without allocating a persistent weight/cache copy; scalar
     * workspaces have one row and retain the old footprint. */
    w->mla_pool_scores = f32_rows(capacity_tokens,
                                  w->sparse_pool_capacity);
    w->mla_selected_pool = bytes_rows(
        DS4_GLM5_NEXT_INDEX_TOP_K / 4u, sizeof(uint32_t));
    w->mla_selected_token = bytes_rows(
        capacity_tokens,
        GLM5_SELECTED_STRIDE * sizeof(int32_t));
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

uint32_t ds4_glm5_next_workspace_capacity(
        const ds4_glm5_next_workspace *workspace) {
    return workspace ? workspace->capacity_tokens : 0u;
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
    char ranked_prefix[768];
    const char *prefix = ctx->trace_prefix;
    if (getenv("DS4_GLM5_TRACE_RANKED") != NULL) {
        const int pn = snprintf(ranked_prefix, sizeof(ranked_prefix),
                                "%s.r%u", ctx->trace_prefix, ctx->tp_rank);
        if (pn <= 0 || (size_t)pn >= sizeof(ranked_prefix)) {
            free(host);
            return 0;
        }
        prefix = ranked_prefix;
    }
    int n = 0;
    if (ctx->trace_layer == UINT32_MAX && ctx->trace_token == UINT32_MAX)
        n = snprintf(path, sizeof(path), "%s.l%u.t%u.%s",
                     prefix, layer, token, name);
    else if (ctx->trace_layer == UINT32_MAX)
        n = snprintf(path, sizeof(path), "%s.l%u.%s",
                     prefix, layer, name);
    else if (ctx->trace_token == UINT32_MAX)
        n = snprintf(path, sizeof(path), "%s.t%u.%s", prefix,
                     token, name);
    else
        n = snprintf(path, sizeof(path), "%s.%s", prefix, name);
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
    char ranked_prefix[768];
    const char *prefix = ctx->trace_prefix;
    if (getenv("DS4_GLM5_TRACE_RANKED") != NULL) {
        const int pn = snprintf(ranked_prefix, sizeof(ranked_prefix),
                                "%s.r%u", ctx->trace_prefix, ctx->tp_rank);
        if (pn <= 0 || (size_t)pn >= sizeof(ranked_prefix)) {
            free(host);
            return 0;
        }
        prefix = ranked_prefix;
    }
    const int n = ctx->trace_layer == UINT32_MAX ?
        snprintf(path, sizeof(path), "%s.batch.l%u.t%u.%s",
                 prefix, layer, token, name) :
        snprintf(path, sizeof(path), "%s.batch.t%u.%s",
                 prefix, token, name);
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
static int tp_exchange_bytes(const ds4_glm5_next_exec_ctx *ctx,
                             uint32_t layer, uint32_t gate,
                             uint64_t bytes);
static int tp_exchange_aux_bytes(const ds4_glm5_next_exec_ctx *ctx,
                                 uint32_t layer, uint64_t bytes);
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

static int kda_prefix_rows(const ds4_glm5_next_exec_ctx *ctx, uint32_t il,
                           ds4_glm5_next_workspace *w,
                           const ds4_gpu_tensor *hc_in, uint32_t n_tokens,
                           bool serial_mix) {
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    if (!ds4_gpu_rms_norm_plain_rows_tensor(
            w->hc_flat, hc_in, GLM5_HC_WIDTH, n_tokens,
            ctx->model->rms_norm_eps)) return 0;
    if (serial_mix) {
        const uint64_t in_bytes = (uint64_t)GLM5_HC_WIDTH * sizeof(float);
        const uint64_t out_bytes = (uint64_t)GLM5_HC_MIX * sizeof(float);
        for (uint32_t t = 0; t < n_tokens; ++t) {
            ds4_gpu_tensor *in = ds4_gpu_tensor_view(w->hc_flat, t * in_bytes, in_bytes);
            ds4_gpu_tensor *out = ds4_gpu_tensor_view(w->hc_mix, t * out_bytes, out_bytes);
            const int ok = in && out && ds4_gpu_matmul_bf16_tensor(
                out, ctx->model_map, ctx->model_size, layer->hc.attn_fn,
                GLM5_HC_WIDTH, GLM5_HC_MIX, in, 1u);
            ds4_gpu_tensor_free(out);
            ds4_gpu_tensor_free(in);
            if (!ok) return 0;
        }
    } else if (!ds4_gpu_matmul_bf16_tensor(
            w->hc_mix, ctx->model_map, ctx->model_size,
            layer->hc.attn_fn, GLM5_HC_WIDTH, GLM5_HC_MIX,
            w->hc_flat, n_tokens)) return 0;
    return ds4_gpu_hc_split_weighted_sum_tensor(
        w->collapsed, w->hc_split, w->hc_mix, hc_in,
        ctx->model_map, ctx->model_size,
        layer->hc.attn_scale, layer->hc.attn_base,
        GLM5_WIDTH, GLM5_HC, 20u, ctx->model->hc_eps);
}

static int kda_attention_rows(const ds4_glm5_next_exec_ctx *ctx,
                              uint32_t il,
                              ds4_glm5_next_state *state,
                              ds4_glm5_next_workspace *w,
                              const ds4_gpu_tensor *hc_in,
                              uint32_t n_tokens) {
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    ds4_glm5_kda_layer_state *kda = &state->kda.layer[il];
    if (state->kda.pending_verifications) return 0;
    glm5_phase_trace(ctx, "kda_prefix_enter", il, n_tokens);
    const int prefix_ok = kda_prefix_rows(ctx, il, w, hc_in, n_tokens, false);
    if (!prefix_ok) return 0;
    glm5_phase_trace(ctx, "kda_prefix_done", il, n_tokens);
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
    const int output_rowslice =
        (tp_features & DS4_TP_FEATURE_GLM5_KDA_OUTPUT_ROWSLICE) != 0u &&
        n_tokens == 1u;
    if (output_kslice && output_rowslice) {
        ds4_glm5_next_state_invalidate(state);
        return 0;
    }
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
    if (output_rowslice && layer->kda.output_type != 30u) {
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
    glm5_phase_trace(ctx, "kda_begin_enter", il, n_tokens);
    const int local_ok = ds4_glm5_kda_layer_begin(
        &local, &w->kda, &layer->kda, ctx->model_map, ctx->model_size,
        w->collapsed, local_gated, n_tokens,
        ctx->model->rms_norm_eps,
        ctx->tp_rank * (DS4_GLM5_KDA_HEADS / 2u),
        DS4_GLM5_KDA_HEADS / 2u);
    glm5_phase_trace(ctx, local_ok ? "kda_begin_done" : "kda_begin_failed",
                     il, n_tokens);
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
    glm5_phase_trace(ctx, "kda_gate_enter", il, n_tokens);
    if (!local_ok ||
        !tp_exchange_rows(ctx, il, DS4_TP_GATE_ATTN, n_tokens)) {
        ds4_glm5_kda_layer_abort(&local);
        kda_half_state_free(&local);
        ds4_glm5_next_state_invalidate(state);
        return 0;
    }
    glm5_phase_trace(ctx, "kda_gate_done", il, n_tokens);
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
    if (output_rowslice) {
        const uint64_t row_bytes =
            (uint64_t)(GLM5_WIDTH / 2u) * sizeof(float);
        const int rowslice_local =
            getenv("DS4_GLM5_KDA_OUTPUT_ROWSLICE_LOCAL") != NULL &&
            strcmp(getenv("DS4_GLM5_KDA_OUTPUT_ROWSLICE_LOCAL"), "1") == 0;
        const int rowslice_full_gemm = rowslice_local &&
            getenv("DS4_GLM5_KDA_OUTPUT_ROWSLICE_FULL_GEMM") != NULL &&
            strcmp(getenv("DS4_GLM5_KDA_OUTPUT_ROWSLICE_FULL_GEMM"), "1") == 0;
        const int rowslice_sync = rowslice_local &&
            getenv("DS4_GLM5_KDA_OUTPUT_ROWSLICE_SYNC") != NULL &&
            strcmp(getenv("DS4_GLM5_KDA_OUTPUT_ROWSLICE_SYNC"), "1") == 0;
        const uint64_t row_weight_bytes =
            (uint64_t)(GLM5_WIDTH / 2u) * DS4_GLM5_KDA_CHANNELS *
            sizeof(uint16_t);
        const uint64_t row_weight_offset =
            layer->kda.output + (uint64_t)ctx->tp_rank * row_weight_bytes;
        const ds4_gpu_tensor *rank0_gated =
            ctx->tp_rank == 0u ? ctx->tp_big_out : ctx->tp_big_in;
        const ds4_gpu_tensor *rank1_gated =
            ctx->tp_rank == 0u ? ctx->tp_big_in : ctx->tp_big_out;
        if (!ds4_glm5_kda_compose_head_halves(
                w->kda.recurrent_out, rank0_gated, rank1_gated,
                n_tokens) ||
            !trace_tensor(ctx, il, (uint32_t)kda->token_count,
                          "kda_composed_gated.f32", w->kda.recurrent_out,
                          (uint64_t)DS4_GLM5_KDA_CHANNELS * sizeof(float)) ||
            (rowslice_full_gemm ?
                 (!ds4_gpu_matmul_bf16_tensor(
                      w->down, ctx->model_map, ctx->model_size,
                      layer->kda.output, DS4_GLM5_KDA_CHANNELS,
                      GLM5_WIDTH, w->kda.recurrent_out, n_tokens) ||
                  !ds4_gpu_tensor_copy(w->attention, 0u, w->down,
                                       0u, row_bytes) ||
                  !ds4_gpu_tensor_copy(w->attention, row_bytes, w->down,
                                       row_bytes, row_bytes)) :
                 !ds4_gpu_matmul_bf16_tensor(
                      w->attention, ctx->model_map, ctx->model_size,
                      row_weight_offset, DS4_GLM5_KDA_CHANNELS,
                      GLM5_WIDTH / 2u, w->kda.recurrent_out, n_tokens)) ||
            (rowslice_local ?
                 (rowslice_full_gemm ? 0 : !ds4_gpu_matmul_bf16_tensor(
                      /* Keep the locally computed peer half device-resident
                       * until it is copied into the final output.  The TP
                       * slab may be mapped host memory and is reserved for
                       * the transport exchange; using it as a second GEMM
                       * destination made long-context local mode diverge. */
                      w->down, ctx->model_map, ctx->model_size,
                      layer->kda.output + (uint64_t)(1u - ctx->tp_rank) *
                          row_weight_bytes,
                      DS4_GLM5_KDA_CHANNELS, GLM5_WIDTH / 2u,
                      w->kda.recurrent_out, n_tokens)) :
                 (!ds4_gpu_tensor_copy(
                       ctx->tp_slab,
                       ds4_tp_slab_aux_out_payload_offset(ctx->tp, il),
                       w->attention, 0u, row_bytes) ||
                  !tp_exchange_aux_bytes(ctx, il, row_bytes) ||
                  (ctx->tp_rank == 0u ?
                       !ds4_gpu_tensor_copy(w->attention, row_bytes,
                            ctx->tp_slab,
                            ds4_tp_slab_aux_in_payload_offset(ctx->tp, il),
                            row_bytes) :
                       (!ds4_gpu_tensor_copy(w->attention, 0u,
                            ctx->tp_slab,
                            ds4_tp_slab_aux_in_payload_offset(ctx->tp, il),
                            row_bytes) ||
                        !ds4_gpu_tensor_copy(w->attention, row_bytes,
                            ctx->tp_slab,
                            ds4_tp_slab_aux_out_payload_offset(ctx->tp, il),
                            row_bytes))))) ||
            (rowslice_local && !rowslice_full_gemm &&
             (ctx->tp_rank == 0u ?
                  !ds4_gpu_tensor_copy(w->attention, row_bytes, w->down,
                                       0u, row_bytes) :
                  (!ds4_gpu_tensor_copy(w->up, 0u, w->attention, 0u,
                                        row_bytes) ||
                   !ds4_gpu_tensor_copy(w->attention, 0u, w->down, 0u,
                                        row_bytes) ||
                   !ds4_gpu_tensor_copy(w->attention, row_bytes, w->up, 0u,
                                        row_bytes)))) ||
            (rowslice_sync && !ds4_gpu_synchronize()) ||
            !trace_tensor(ctx, il, (uint32_t)kda->token_count,
                          "kda_projected.f32", w->attention,
                          (uint64_t)GLM5_WIDTH * sizeof(float)) ||
            !ds4_glm5_kda_layer_commit(&local, n_tokens) ||
            !ds4_gpu_hc_expand_split_tensor(
                w->after_attention, w->attention, hc_in, w->hc_split,
                GLM5_WIDTH, GLM5_HC)) {
            ds4_glm5_kda_layer_abort(&local);
            kda_half_state_free(&local);
            ds4_glm5_next_state_invalidate(state);
            return 0;
        }
        kda->token_count = local.token_count;
        kda_half_state_free(&local);
        static int rowslice_reported;
        if (!rowslice_reported) {
            fprintf(stderr,
                    "ds4: GLM5 KDA output row-slice engaged (decode-only)\n");
            rowslice_reported = 1;
        }
        return 1;
    }
    const ds4_gpu_tensor *rank0 =
        ctx->tp_rank == 0u ? ctx->tp_big_out : ctx->tp_big_in;
    const ds4_gpu_tensor *rank1 =
        ctx->tp_rank == 0u ? ctx->tp_big_in : ctx->tp_big_out;
    const int suffix_ok =
        ds4_glm5_kda_compose_head_halves(
            w->kda.recurrent_out, rank0, rank1, n_tokens) &&
        trace_tensor(ctx, il, (uint32_t)kda->token_count,
                     "kda_composed_gated.f32", w->kda.recurrent_out,
                     (uint64_t)DS4_GLM5_KDA_CHANNELS * sizeof(float)) &&
        ds4_glm5_kda_layer_finish(
            &local, &layer->kda, ctx->model_map, ctx->model_size,
            w->kda.recurrent_out, w->attention, n_tokens) &&
        trace_tensor(ctx, il, (uint32_t)kda->token_count,
                     "kda_projected.f32", w->attention,
                     (uint64_t)GLM5_WIDTH * sizeof(float)) &&
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

static int dense_ffn_prefix(const ds4_glm5_next_exec_ctx *ctx, uint32_t il,
                            ds4_glm5_next_workspace *w,
                            uint32_t n_tokens, int finite_debug) {
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    int ok = ds4_gpu_rms_norm_plain_rows_tensor(
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
            ctx->model->rms_norm_eps);
    if (finite_debug) route_failure_stats("dense_ffn_hidden", w->ffn_hidden,
                                          n_tokens * GLM5_WIDTH);
    return ok;
}

static int dense_ffn_rows(const ds4_glm5_next_exec_ctx *ctx, uint32_t il,
                          ds4_glm5_next_workspace *w, ds4_gpu_tensor *hc_out,
                          uint32_t n_tokens, int finite_debug) {
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    int ok = dense_ffn_prefix(ctx, il, w, n_tokens, finite_debug);
    if (ok) ok = ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
        w->gate, w->up, w->mid, ctx->model_map, ctx->model_size,
        layer->ffn_weight.gate, layer->ffn_weight.up,
        GLM5_WIDTH, GLM5_DENSE_MID, w->ffn_hidden, n_tokens, 10.0f) &&
        ds4_gpu_matmul_q8_0_tensor(
            w->down, ctx->model_map, ctx->model_size,
            layer->ffn_weight.down, GLM5_DENSE_MID, GLM5_WIDTH,
            w->mid, n_tokens);
    if (finite_debug) route_failure_stats("dense_ffn_down", w->down,
                                          n_tokens * GLM5_WIDTH);
    return ok && ds4_gpu_hc_expand_split_tensor(
        hc_out, w->down, w->after_attention, w->ffn_split,
        GLM5_WIDTH, GLM5_HC);
}

static int dense_kda_forward_rows(const ds4_glm5_next_exec_ctx *ctx,
                                  uint32_t il,
                                  ds4_glm5_next_state *state,
                                  ds4_glm5_next_workspace *w,
                                  const ds4_gpu_tensor *hc_in,
                                  ds4_gpu_tensor *hc_out,
                                  uint32_t n_tokens) {
    const int finite_debug = getenv("DS4_GLM5_NEXT_VALIDATE_FINITE") != NULL;
    const int async_terminal =
        glm5_decode_async_layer_enabled(w, n_tokens);
    if (finite_debug) route_failure_stats("dense_hc_in", hc_in,
                                          n_tokens * GLM5_HC_WIDTH);
    int ok = kda_attention_rows(ctx, il, state, w, hc_in, n_tokens);
#ifdef DS4_TP_TEST_HOOKS
    const uint32_t token_ordinal =
        state->kda.layer[il].token_count <= UINT32_MAX ?
        (uint32_t)state->kda.layer[il].token_count : UINT32_MAX;
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
    if (ok) ok = dense_ffn_rows(ctx, il, w, hc_out, n_tokens, finite_debug);
#ifdef DS4_TP_TEST_HOOKS
    if (ok && n_tokens == 1u)
        ok = trace_tensor(
            ctx, il, token_ordinal, "output_hc.f32", hc_out,
            (uint64_t)GLM5_HC_WIDTH * sizeof(float));
#endif
    if (ok && !async_terminal) ok = ds4_gpu_synchronize();
    if (ok && async_terminal) {
        static int logged_async_terminal[2] = {0, 0};
        const uint32_t rank = ctx->tp_rank < 2u ? ctx->tp_rank : 0u;
        if (!logged_async_terminal[rank]) {
            fprintf(stderr,
                    "ds4: GLM5 decode terminal layer fence omitted "
                    "on same default stream rank=%u\n", ctx->tp_rank);
            logged_async_terminal[rank] = 1;
        }
    }
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

static int tp_exchange_bytes(const ds4_glm5_next_exec_ctx *ctx,
                             uint32_t layer, uint32_t gate,
                             uint64_t bytes) {
    if (bytes == 0u) return 0;
    if (!tp_context_valid_bytes(ctx, bytes) || gate >= DS4_TP_GATES_PER_LAYER ||
        *ctx->tp_sequence == UINT64_MAX) return 0;

    const char *cache_fence = getenv("DS4_ROCM_RDMA_CACHE_FENCE");
    const int fence_release = cache_fence &&
        (strcmp(cache_fence, "release") == 0 ||
         strcmp(cache_fence, "both") == 0);
    const int fence_acquire = cache_fence &&
        (strcmp(cache_fence, "acquire") == 0 ||
         strcmp(cache_fence, "both") == 0);
    if (cache_fence && !fence_release && !fence_acquire) return 0;

    const int small_gate_requested =
        !ctx->force_bulk_gates && bytes == ds4_tp_vec_bytes(ctx->tp) &&
        (ds4_tp_runtime_features(ctx->tp) &
         DS4_TP_FEATURE_GLM5_SMALL_GATE) != 0u;
    if (small_gate_requested) {
        if (!ctx->tp_slab) {
            ds4_tp_mark_failed(ctx->tp);
            return 0;
        }
        const uint64_t out_off =
            ds4_tp_slab_out_offset(ctx->tp, layer, gate);
        const uint64_t in_off =
            ds4_tp_slab_in_offset(ctx->tp, layer, gate);
        const int gpu_row_gate =
            (ds4_tp_runtime_features(ctx->tp) &
             DS4_TP_FEATURE_GLM5_GPU_ROW_GATE) != 0u;
        const int direct_send = !gpu_row_gate &&
            (ds4_tp_runtime_features(ctx->tp) &
             DS4_TP_FEATURE_GLM5_SMALL_GATE_DIRECT_SEND) != 0u;
        if (ds4_gpu_tensor_bytes(ctx->tp_slab) < out_off + bytes ||
            ds4_gpu_tensor_bytes(ctx->tp_slab) < in_off + bytes ||
            (!direct_send &&
             !ds4_gpu_tensor_copy(ctx->tp_slab, out_off,
                                  ctx->tp_big_out, 0u, bytes))) {
            ds4_tp_mark_failed(ctx->tp);
            return 0;
        }
        if (gpu_row_gate) {
            if (fence_release && !ds4_rocm_rdma_cache_release()) {
                ds4_tp_mark_failed(ctx->tp);
                return 0;
            }
            /* Keep the GLM control/hash sequence advancing exactly as the
             * legacy path does.  The GPU row channel owns a separate RDMA
             * sequence; conflating the two made a later prefill/decode
             * transition depend on whichever gate arrived first. */
            ++*ctx->tp_sequence;
            if (!ds4_gpu_tp_gate_encode(layer, gate) ||
                !ds4_gpu_tensor_copy(ctx->tp_big_in, 0u,
                                     ctx->tp_slab, in_off, bytes)) {
                ds4_tp_mark_failed(ctx->tp);
                return 0;
            }
            /* This mode is intentionally tested with host-synchronous gates
             * first.  In that mode encode returns only after RDMA completion,
             * so acquire fencing is ordered before the consumer copy. */
            if (fence_acquire && !ds4_rocm_rdma_cache_acquire()) {
                ds4_tp_mark_failed(ctx->tp);
                return 0;
            }
            static int gpu_reported;
            if (!gpu_reported) {
                fprintf(stderr,
                        "ds4: GLM5 one-token TP reductions use the "
                        "GPU-ordered preposted latency gate (%llu bytes)\n",
                        (unsigned long long)bytes);
                gpu_reported = 1;
            }
            return 1;
        }
        if (!ds4_gpu_synchronize()) {
            ds4_tp_mark_failed(ctx->tp);
            return 0;
        }
        if (fence_release && !ds4_rocm_rdma_cache_release()) {
            ds4_tp_mark_failed(ctx->tp);
            return 0;
        }
        const uint64_t sequence = ++*ctx->tp_sequence;
        const bool native_session = ds4_tp_glm5_native_rows(ds4_tp_prefill_config(ctx->tp)) != 0u;
        const int exchanged = native_session ?
            ds4_tp_native_gate_exchange_next(ctx->tp, layer, gate,
                direct_send ? ctx->tp_big_out_host : NULL) : direct_send ?
            ds4_tp_gate_exchange_from_registered(
                ctx->tp, layer, gate, sequence, ctx->tp_big_out_host) :
            ds4_tp_gate_exchange(ctx->tp, layer, gate, sequence);
        if (!exchanged) {
            ds4_tp_mark_failed(ctx->tp);
            return 0;
        }
        if (fence_acquire && !ds4_rocm_rdma_cache_acquire()) {
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
                    "ds4: GLM5 one-token TP reductions use the preposted "
                    "latency gate (%llu bytes, direct_send=%d)\n",
                    (unsigned long long)bytes, direct_send);
            reported = 1;
        }
        return 1;
    }

    const uint64_t trace_row_bytes = (uint64_t)GLM5_WIDTH * sizeof(float);
    const uint32_t trace_rows = (uint32_t)(
        (bytes + trace_row_bytes - 1u) / trace_row_bytes);
    glm5_phase_trace(ctx, "gate_gpu_sync_enter", layer, trace_rows);
    if (!ds4_gpu_synchronize()) return 0;
    glm5_phase_trace(ctx, "gate_gpu_sync_done", layer, trace_rows);
    if (fence_release && !ds4_rocm_rdma_cache_release()) return 0;
    const uint64_t sequence = ++*ctx->tp_sequence;
    glm5_phase_trace(ctx, "gate_rdma_enter", layer, trace_rows);
    if (!ds4_tp_big_gate_exchange(ctx->tp, layer, sequence,
                                  ctx->tp_big_out_host,
                                  ctx->tp_big_in_host, bytes)) {
        ds4_tp_mark_failed(ctx->tp);
        return 0;
    }
    glm5_phase_trace(ctx, "gate_rdma_done", layer, trace_rows);
    if (fence_acquire && !ds4_rocm_rdma_cache_acquire()) {
        ds4_tp_mark_failed(ctx->tp);
        return 0;
    }
    return 1;
}

static int tp_exchange_rows(const ds4_glm5_next_exec_ctx *ctx,
                            uint32_t layer, uint32_t gate,
                            uint32_t n_tokens) {
    if (n_tokens == 0u) return 0;
    return tp_exchange_bytes(ctx, layer, gate,
                             (uint64_t)n_tokens * GLM5_WIDTH * sizeof(float));
}

/* Exchange a second dependent payload without consuming a logical decode-gate
 * slot.  Output-row KDA sharding needs the normal gated-head exchange first,
 * then this small registered RDMA payload.  The header sequence is the
 * current logical sequence on both ranks; it is deliberately not incremented
 * so the existing attention/FFN receive-window schedule remains unchanged. */
static int tp_exchange_aux_bytes(const ds4_glm5_next_exec_ctx *ctx,
                                 uint32_t layer, uint64_t bytes) {
    if (!ctx || !ctx->tp || !ctx->tp_sequence ||
        *ctx->tp_sequence == UINT64_MAX || bytes == 0u ||
        !ctx->tp_slab || bytes != ds4_tp_aux_payload_bytes(ctx->tp)) return 0;
    const uint64_t out_off =
        ds4_tp_slab_aux_out_payload_offset(ctx->tp, layer);
    const uint64_t in_off =
        ds4_tp_slab_aux_in_payload_offset(ctx->tp, layer);
    if (out_off == UINT64_MAX || in_off == UINT64_MAX ||
        ds4_gpu_tensor_bytes(ctx->tp_slab) < out_off + bytes ||
        ds4_gpu_tensor_bytes(ctx->tp_slab) < in_off + bytes) return 0;
    const char *cache_fence = getenv("DS4_ROCM_RDMA_CACHE_FENCE");
    const int fence_release = cache_fence &&
        (strcmp(cache_fence, "release") == 0 ||
         strcmp(cache_fence, "both") == 0);
    const int fence_acquire = cache_fence &&
        (strcmp(cache_fence, "acquire") == 0 ||
         strcmp(cache_fence, "both") == 0);
    if (cache_fence && !fence_release && !fence_acquire) return 0;
    if (!ds4_gpu_synchronize()) return 0;
    if (fence_release && !ds4_rocm_rdma_cache_release()) return 0;
    const int ok = ds4_tp_aux_gate_exchange(ctx->tp, layer);
    if (!ok) return 0;
    if (fence_acquire && !ds4_rocm_rdma_cache_acquire()) return 0;
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

/* Independent of routed IDs/weights. Queue the incumbent shared arithmetic
 * before waiting on peer agreement; all outputs use separate shared scratch.
 * The caller drains outstanding work on any later failure. */
static int shared_ffn_before_route_check(
        const ds4_glm5_next_exec_ctx *ctx, uint32_t layer,
        ds4_glm5_next_workspace *w) {
    const ds4_glm5_next_ffn_offsets *f = &ctx->model->layer[layer].ffn_weight;
    const uint32_t base = ctx->tp_rank * GLM5_RANK_MID;
    const uint64_t row_bytes =
        (GLM5_WIDTH / GLM5_Q8_QK) * GLM5_Q8_BLOCK_BYTES;
    return ds4_gpu_matmul_q8_0_tensor(
               w->shared_gate, ctx->model_map, ctx->model_size,
               f->gate_shexp + (uint64_t)base * row_bytes,
               GLM5_WIDTH, GLM5_RANK_MID, w->ffn_hidden, 1u) &&
           ds4_gpu_matmul_q8_0_tensor(
               w->shared_up, ctx->model_map, ctx->model_size,
               f->up_shexp + (uint64_t)base * row_bytes,
               GLM5_WIDTH, GLM5_RANK_MID, w->ffn_hidden, 1u) &&
           ds4_gpu_swiglu_tensor(w->shared_mid, w->shared_gate, w->shared_up,
                                GLM5_RANK_MID, 10.0f, 1.0f) &&
           ds4_gpu_matmul_q8_0_kslice_tensor(
               w->shared_out, ctx->model_map, ctx->model_size, f->down_shexp,
               GLM5_ROUTED_MID, base, GLM5_RANK_MID,
               GLM5_WIDTH, w->shared_mid, 0u);
}

static int route_agrees(const ds4_glm5_next_exec_ctx *ctx, uint32_t layer,
                        uint32_t token_ordinal,
                        const ds4_gpu_tensor *selected,
                        const ds4_gpu_tensor *weights,
                        const ds4_gpu_tensor *logits,
                        const ds4_gpu_tensor *hidden,
                        ds4_glm5_next_workspace *early_shared) {
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
    if (early_shared && !shared_ffn_before_route_check(ctx, layer, early_shared)) {
        ds4_tp_mark_failed(ctx->tp);
        return 0;
    }
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

/* Return -1 for an invalid requested decode configuration, never silently
 * substitute another schedule. The switch deliberately does not alter prefill. */
static int shared_route_overlap_mode(const ds4_glm5_next_exec_ctx *ctx,
                                      const ds4_glm5_next_workspace *w,
                                      int q4_residency) {
    const char *value = getenv("DS4_ROCM_GLM5_SHARED_ROUTE_OVERLAP");
    if (!value || strcmp(value, "0") == 0) return 0;
    if (strcmp(value, "1") != 0 || !ctx || !w) return -1;
    if (!w->decode_phase) return 0;
#ifdef DS4_ROCM_BUILD
    const char *paired = getenv("DS4_ROCM_GLM5_SHARED_Q8_PAIR_DECODE");
    const char *window_overlap = getenv("DS4_ROCM_GLM5_WINDOW_OVERLAP");
    const char *window_scratch = getenv("DS4_ROCM_GLM5_WINDOW_SCRATCH");
    /* This schedule is qualified only for resident experts. Reject the
     * separate window/scratch overlap recipe explicitly rather than letting
     * residency silently mask an untested combination of scheduling modes. */
    if (window_overlap && strcmp(window_overlap, "1") == 0 &&
        window_scratch && strcmp(window_scratch, "1") == 0) return -1;
    if (q4_residency == 1 && ctx->tp_rank < 2u && tp_context_valid(ctx) &&
        (!paired || strcmp(paired, "0") == 0)) return 1;
#else
    (void)q4_residency;
#endif
    return -1;
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

/* Compute the stateless and token-addressed sparse MLA prelude as one row
 * batch. Pool publication normally remains outside this helper. The optional
 * pooled-score lane publishes a complete aligned tile before scoring, while
 * its score mask still exposes only floor((pos0+t+1)/4) pools to query t. */
static int mla_sparse_prelude_rows(
        const ds4_glm5_next_exec_ctx *ctx,
        uint32_t il,
        ds4_glm5_next_state *state,
        ds4_glm5_next_workspace *w,
        const ds4_gpu_tensor *hc_in,
        uint32_t n_tokens) {
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    ds4_glm5_next_mla_offsets local_m = layer->mla;
    const ds4_glm5_next_mla_offsets *m = &local_m;
    ds4_glm5_next_mla_state *mla = &state->mla[il];
    const uint32_t pos0 = mla->token_count;
    const uint64_t full_heads =
        (uint64_t)GLM5_HEADS * GLM5_HEAD_DIM;
    if (n_tokens == 0u || pos0 > mla->capacity_tokens ||
        n_tokens > mla->capacity_tokens - pos0) return 0;
    return
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
        ds4_gpu_matmul_bf16_tensor(
            w->mla_index_q, ctx->model_map, ctx->model_size, m->index_q_b,
            GLM5_Q_RANK, GLM5_INDEX_HEADS * GLM5_INDEX_DIM,
            w->mla_q_resid, n_tokens) &&
        ds4_gpu_matmul_bf16_tensor(
            w->mla_index_weights, ctx->model_map, ctx->model_size,
            m->index_proj, GLM5_WIDTH, GLM5_INDEX_HEADS,
            w->ffn_hidden, n_tokens);
}

/* Scalar MLA owns the same contiguous heads as its existing output K slice.
 * Prefill consumers retain full heads; the verifier's deferred tail explicitly
 * keeps this scalar layout. Only native Q8 pointer offsets change here.
 */
static int mla_scalar_head_layout(const ds4_glm5_next_exec_ctx *ctx,
                                  bool scalar_consumer,
                                  ds4_glm5_next_mla_offsets *m,
                                  uint32_t *heads,
                                  uint64_t *output_input_start) {
    const char *value = getenv("DS4_GLM5_MLA_OWNED_HEADS");
    if (value && strcmp(value, "0") != 0 && strcmp(value, "1") != 0)
        return 0;
    *heads = GLM5_HEADS;
    *output_input_start =
        (uint64_t)ctx->tp_rank * (GLM5_HEADS / 2u) * GLM5_HEAD_DIM;
    if (!scalar_consumer || !value || strcmp(value, "1") != 0) return 1;
    /* Full-layout tensor dumps have a separate contract; never emit a stale
     * unowned half as if this were a complete 64-head projection. */
    if (ctx->tp_rank > 1u || ctx->trace_prefix) return 0;
    const uint64_t first_head = (uint64_t)ctx->tp_rank * (GLM5_HEADS / 2u);
    const uint64_t q_offset = first_head * GLM5_HEAD_DIM *
        (GLM5_Q_RANK / GLM5_Q8_QK) * GLM5_Q8_BLOCK_BYTES;
    const uint64_t k_offset = first_head * GLM5_KV_LORA *
        (GLM5_HEAD_DIM / GLM5_Q8_QK) * GLM5_Q8_BLOCK_BYTES;
    const uint64_t v_offset = first_head * GLM5_HEAD_DIM *
        (GLM5_KV_LORA / GLM5_Q8_QK) * GLM5_Q8_BLOCK_BYTES;
    if (m->q_b > UINT64_MAX - q_offset ||
        m->k_b > UINT64_MAX - k_offset ||
        m->v_b > UINT64_MAX - v_offset) return 0;
    m->q_b += q_offset;
    m->k_b += k_offset;
    m->v_b += v_offset;
    *heads = GLM5_HEADS / 2u;
    *output_input_start = 0u;
    static int reported[2];
    if (!reported[ctx->tp_rank]) {
        fprintf(stderr, "ds4: GLM5 scalar MLA owned heads active "
                "rank=%u first=%llu heads=%u indexer_heads=%u\n",
                ctx->tp_rank, (unsigned long long)first_head,
                *heads, GLM5_INDEX_HEADS);
        reported[ctx->tp_rank] = 1;
    }
    return 1;
}

static int mla_scalar_prepare(const ds4_glm5_next_exec_ctx *ctx,
        const ds4_glm5_next_layer_offsets *layer, ds4_glm5_next_workspace *w,
        const ds4_gpu_tensor *input) {
    if (!layer->is_trunk)
        return ds4_gpu_rms_norm_weight_tensor(w->ffn_hidden, input,
            ctx->model_map, ctx->model_size, layer->attn_norm,
            GLM5_WIDTH, ctx->model->rms_norm_eps);
    return ds4_gpu_rms_norm_plain_rows_tensor(w->hc_flat, input,
            GLM5_HC_WIDTH, 1u, ctx->model->rms_norm_eps) &&
        ds4_gpu_matmul_bf16_tensor(w->hc_mix, ctx->model_map, ctx->model_size,
            layer->hc.attn_fn, GLM5_HC_WIDTH, GLM5_HC_MIX, w->hc_flat, 1u) &&
        ds4_gpu_hc_split_weighted_sum_norm_tensor(w->collapsed, w->ffn_hidden,
            w->hc_split, w->hc_mix, input, ctx->model_map, ctx->model_size,
            layer->hc.attn_scale, layer->hc.attn_base, layer->attn_norm,
            GLM5_WIDTH, GLM5_HC, 20u, ctx->model->hc_eps, ctx->model->rms_norm_eps);
}

static int mla_scalar_residual(const ds4_glm5_next_layer_offsets *layer,
        ds4_glm5_next_workspace *w, const ds4_gpu_tensor *input) {
    return layer->is_trunk ? ds4_gpu_hc_expand_split_tensor(
            w->after_attention, w->attention, input, w->hc_split, GLM5_WIDTH, GLM5_HC) :
        ds4_gpu_add_tensor(w->after_attention, input, w->attention, GLM5_WIDTH);
}

/* Verifier-only diagnostic. NULL preserves the original asynchronous call
 * sequence. Completed spans add fences and include host/peer waiting; they
 * are not active GPU time or a pure network latency measurement. */
enum {
    MLA_PROFILE_ENTRY, MLA_PROFILE_PREPARE_QKV, MLA_PROFILE_KV_STORE,
    MLA_PROFILE_LOWRANK, MLA_PROFILE_INDEX_STATE, MLA_PROFILE_SELECT,
    MLA_PROFILE_ATTENTION, MLA_PROFILE_OUTPUT, MLA_PROFILE_EXCHANGE,
    MLA_PROFILE_RESIDUAL, MLA_PROFILE_COUNT
};
typedef struct {
    double last, seconds[MLA_PROFILE_COUNT];
    uint32_t dense_rows, sparse_rows;
} glm5_mla_profile;

static int mla_profile_mark(glm5_mla_profile *p, unsigned stage) {
    if (!p) return 1;
    if (!ds4_gpu_synchronize()) return 0;
    const double now = glm5_exec_now_sec();
    p->seconds[stage] += now - p->last;
    p->last = now;
    return 1;
}

static int mla_profile_begin(glm5_mla_profile *p) {
    if (!p) return 1;
    p->last = glm5_exec_now_sec();
    return mla_profile_mark(p, MLA_PROFILE_ENTRY);
}

/* The official selector uses the full visible range through top-k. Pooled
 * selection begins only when visible exceeds 2048 and is a separate path. */
static int mla_dense_selection_attention_impl(const ds4_glm5_next_exec_ctx *ctx,
                                       uint32_t il,
                                       ds4_glm5_next_mla_state *mla,
                                       ds4_glm5_next_workspace *w,
                                       const ds4_gpu_tensor *hc_in,
                                       uint32_t visible,
                                       uint32_t tail_slot,
                                       uint32_t pool_index,
                                       bool publish_pool,
                                       bool defer_tail,
                                       glm5_mla_profile *profile) {
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    ds4_glm5_next_mla_offsets local_m = layer->mla;
    const ds4_glm5_next_mla_offsets *m = &local_m;
    uint32_t heads = GLM5_HEADS;
    uint64_t output_input_start = 0u;
    const uint32_t pos = mla->token_count;
    const uint64_t half_heads =
        ((uint64_t)GLM5_HEADS * GLM5_HEAD_DIM) / 2u;
    if (!mla_scalar_head_layout(ctx, true, &local_m,
                                &heads, &output_input_start)) return 0;
    int ok = mla_profile_begin(profile) &&
        mla_scalar_prepare(ctx, layer, w, hc_in) &&
        ds4_gpu_matmul_q8_0_tensor(
            w->mla_q_a, ctx->model_map, ctx->model_size, m->q_a,
            GLM5_WIDTH, GLM5_Q_RANK, w->ffn_hidden, 1u) &&
        ds4_gpu_rms_norm_weight_tensor(
            w->mla_q_resid, w->mla_q_a, ctx->model_map, ctx->model_size,
            m->q_a_norm, GLM5_Q_RANK, ctx->model->rms_norm_eps) &&
        ds4_gpu_matmul_q8_0_tensor(
            w->mla_query, ctx->model_map, ctx->model_size, m->q_b,
            GLM5_Q_RANK, heads * GLM5_HEAD_DIM,
            w->mla_q_resid, 1u) &&
        ds4_gpu_matmul_q8_0_tensor(
            w->mla_kv_raw, ctx->model_map, ctx->model_size, m->kv_a_mqa,
            GLM5_WIDTH, GLM5_KV_LORA, w->ffn_hidden, 1u) &&
        ds4_gpu_glm_kv_lora_rms_norm_tensor(
            w->mla_kv_norm, w->mla_kv_raw,
            ctx->model_map, ctx->model_size, m->kv_a_norm,
            1u, GLM5_KV_LORA, GLM5_KV_LORA,
            ctx->model->rms_norm_eps) &&
        mla_profile_mark(profile, MLA_PROFILE_PREPARE_QKV) &&
        ds4_gpu_glm_store_compact_kv_tensor(
            mla->compact_kv, NULL, w->mla_kv_norm, w->mla_kv_raw,
            pos, 1u, mla->capacity_tokens, GLM5_KV_LORA,
            GLM5_KV_LORA, 0u, false) &&
        mla_profile_mark(profile, MLA_PROFILE_KV_STORE) &&
        ds4_gpu_glm_qk_lowrank_typed_tensor(
            w->mla_qk_low, w->mla_query,
            ctx->model_map, ctx->model_size, m->k_b, 8u,
            heads, GLM5_KV_LORA, GLM5_HEAD_DIM, GLM5_HEAD_DIM) &&
        mla_profile_mark(profile, MLA_PROFILE_LOWRANK) &&
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
        ds4_glm5_next_mla_verify_record(
            mla, mla->token_count, mla->index_tail,
            (uint64_t)tail_slot * GLM5_INDEX_DIM * sizeof(float),
            w->mla_pool_gate_raw, 0u, 1u) &&
        mla_publish_completed_pool(
            ctx, m, mla, w, pool_index, publish_pool) &&
        mla_profile_mark(profile, MLA_PROFILE_INDEX_STATE) &&
        ds4_gpu_glm_fill_selected_range_tensor(w->mla_selected_token,
                                                visible) &&
        mla_profile_mark(profile, MLA_PROFILE_SELECT) &&
        ds4_gpu_glm_attention_indexed_decode_typed_tensor(
            w->mla_heads, w->mla_query, w->mla_qk_low,
            mla->compact_kv, NULL, ctx->model_map, ctx->model_size,
            m->v_b, 8u, w->mla_selected_token, visible,
            mla->capacity_tokens, false, heads, GLM5_KV_LORA,
            GLM5_HEAD_DIM, 0u, GLM5_HEAD_DIM, 0u,
            1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f) &&
        mla_profile_mark(profile, MLA_PROFILE_ATTENTION);
    if (defer_tail) {
        if (ok && profile) ++profile->dense_rows;
        return ok;
    }
    ok = ok &&
        ds4_gpu_matmul_q8_0_kslice_tensor(
            ctx->tp_big_out, ctx->model_map, ctx->model_size, m->output,
            GLM5_HEADS * GLM5_HEAD_DIM,
            (uint64_t)ctx->tp_rank * half_heads, half_heads,
            GLM5_WIDTH, w->mla_heads,
            output_input_start) &&
        mla_profile_mark(profile, MLA_PROFILE_OUTPUT) &&
        tp_exchange(ctx, il, DS4_TP_GATE_ATTN) &&
        mla_profile_mark(profile, MLA_PROFILE_EXCHANGE) &&
        ds4_gpu_add_tensor(w->attention, ctx->tp_big_out, ctx->tp_big_in,
                           GLM5_WIDTH) &&
        mla_scalar_residual(layer, w, hc_in) &&
        mla_profile_mark(profile, MLA_PROFILE_RESIDUAL);
    if (ok && profile) ++profile->dense_rows;
    return ok;
}

static int mla_dense_selection_attention(const ds4_glm5_next_exec_ctx *ctx,
                                       uint32_t il,
                                       ds4_glm5_next_mla_state *mla,
                                       ds4_glm5_next_workspace *w,
                                       const ds4_gpu_tensor *hc_in,
                                       uint32_t visible, uint32_t tail_slot,
                                       uint32_t pool_index, bool publish_pool) {
    return mla_dense_selection_attention_impl(ctx, il, mla, w, hc_in,
        visible, tail_slot, pool_index, publish_pool, false, NULL);
}

/* Beyond the model's 2048-row index budget, score completed pool-4 keys,
 * retain the best 512 pools, expand them back to raw rows, and append the
 * current incomplete tail.  All selector intermediates remain device-local;
 * only the established attention output slice crosses RDMA. */
static int mla_sparse_selection_attention_impl(
        const ds4_glm5_next_exec_ctx *ctx,
        uint32_t il,
        ds4_glm5_next_mla_state *mla,
        ds4_glm5_next_workspace *w,
        const ds4_gpu_tensor *hc_in,
        uint32_t tail_slot,
        uint32_t pool_index,
        bool publish_pool,
        uint32_t top_k,
        ds4_gpu_tensor *local_output,
        bool project_output,
        bool finish_attention,
        bool defer_tail,
        glm5_mla_profile *profile) {
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    ds4_glm5_next_mla_offsets local_m = layer->mla;
    const ds4_glm5_next_mla_offsets *m = &local_m;
    uint32_t heads = GLM5_HEADS;
    uint64_t output_input_start = 0u;
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
    if ((project_output &&
         (!local_output ||
          ds4_gpu_tensor_bytes(local_output) <
              (uint64_t)GLM5_WIDTH * sizeof(float))) ||
        (finish_attention &&
         (!project_output || local_output != ctx->tp_big_out))) {
        return 0;
    }
    if (!mla_scalar_head_layout(ctx, finish_attention, &local_m,
                                &heads, &output_input_start)) return 0;
    int ok = mla_profile_begin(profile) &&
        mla_scalar_prepare(ctx, layer, w, hc_in) &&
        ds4_gpu_matmul_q8_0_tensor(
            w->mla_q_a, ctx->model_map, ctx->model_size, m->q_a,
            GLM5_WIDTH, GLM5_Q_RANK, w->ffn_hidden, 1u) &&
        ds4_gpu_rms_norm_weight_tensor(
            w->mla_q_resid, w->mla_q_a, ctx->model_map, ctx->model_size,
            m->q_a_norm, GLM5_Q_RANK, ctx->model->rms_norm_eps) &&
        ds4_gpu_matmul_q8_0_tensor(
            w->mla_query, ctx->model_map, ctx->model_size, m->q_b,
            GLM5_Q_RANK, heads * GLM5_HEAD_DIM,
            w->mla_q_resid, 1u) &&
        ds4_gpu_matmul_q8_0_tensor(
            w->mla_kv_raw, ctx->model_map, ctx->model_size, m->kv_a_mqa,
            GLM5_WIDTH, GLM5_KV_LORA, w->ffn_hidden, 1u) &&
        ds4_gpu_glm_kv_lora_rms_norm_tensor(
            w->mla_kv_norm, w->mla_kv_raw,
            ctx->model_map, ctx->model_size, m->kv_a_norm,
            1u, GLM5_KV_LORA, GLM5_KV_LORA,
            ctx->model->rms_norm_eps) &&
        mla_profile_mark(profile, MLA_PROFILE_PREPARE_QKV) &&
        ds4_gpu_glm_store_compact_kv_tensor(
            mla->compact_kv, NULL, w->mla_kv_norm, w->mla_kv_raw,
            pos, 1u, mla->capacity_tokens, GLM5_KV_LORA,
            GLM5_KV_LORA, 0u, false) &&
        mla_profile_mark(profile, MLA_PROFILE_KV_STORE) &&
        ds4_gpu_glm_qk_lowrank_typed_tensor(
            w->mla_qk_low, w->mla_query,
            ctx->model_map, ctx->model_size, m->k_b, 8u,
            heads, GLM5_KV_LORA, GLM5_HEAD_DIM, GLM5_HEAD_DIM) &&
        mla_profile_mark(profile, MLA_PROFILE_LOWRANK) &&
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
        ds4_glm5_next_mla_verify_record(
            mla, mla->token_count, mla->index_tail,
            (uint64_t)tail_slot * GLM5_INDEX_DIM * sizeof(float),
            w->mla_pool_gate_raw, 0u, 1u) &&
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
        mla_profile_mark(profile, MLA_PROFILE_INDEX_STATE);
    ok = ok &&
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
        mla_profile_mark(profile, MLA_PROFILE_SELECT);
    ok = ok &&
        ds4_gpu_glm_attention_indexed_decode_typed_tensor(
            w->mla_heads, w->mla_query, w->mla_qk_low,
            mla->compact_kv, NULL, ctx->model_map, ctx->model_size,
            m->v_b, 8u, w->mla_selected_token, selected_tokens,
            mla->capacity_tokens, false, heads, GLM5_KV_LORA,
            GLM5_HEAD_DIM, 0u, GLM5_HEAD_DIM, 0u,
            1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f) &&
        mla_profile_mark(profile, MLA_PROFILE_ATTENTION);
    if (defer_tail) {
        if (ok && profile) ++profile->sparse_rows;
        return ok;
    }
    if (!project_output) return ok;
    ok = ok &&
        ds4_gpu_matmul_q8_0_kslice_tensor(
            local_output, ctx->model_map, ctx->model_size, m->output,
            GLM5_HEADS * GLM5_HEAD_DIM,
            (uint64_t)ctx->tp_rank * half_heads, half_heads,
            GLM5_WIDTH, w->mla_heads,
            output_input_start) &&
        mla_profile_mark(profile, MLA_PROFILE_OUTPUT);
    if (!finish_attention) return ok;
    ok = ok && tp_exchange(ctx, il, DS4_TP_GATE_ATTN) &&
        mla_profile_mark(profile, MLA_PROFILE_EXCHANGE);
    ok = ok &&
        ds4_gpu_add_tensor(
            w->attention, ctx->tp_big_out, ctx->tp_big_in, GLM5_WIDTH) &&
        mla_scalar_residual(layer, w, hc_in) &&
        mla_profile_mark(profile, MLA_PROFILE_RESIDUAL);
    if (ok && profile) ++profile->sparse_rows;
    return ok;
}

static int mla_sparse_selection_attention(
        const ds4_glm5_next_exec_ctx *ctx, uint32_t il,
        ds4_glm5_next_mla_state *mla, ds4_glm5_next_workspace *w,
        const ds4_gpu_tensor *hc_in, uint32_t tail_slot, uint32_t pool_index,
        bool publish_pool, uint32_t top_k, ds4_gpu_tensor *local_output,
        bool project_output, bool finish_attention) {
    return mla_sparse_selection_attention_impl(ctx, il, mla, w, hc_in,
        tail_slot, pool_index, publish_pool, top_k, local_output,
        project_output, finish_attention, false, NULL);
}

static int mla_output_small_m(const ds4_glm5_next_exec_ctx *ctx, uint32_t il,
        const ds4_glm5_next_workspace *batch, uint32_t rows, bool launch) {
    const uint32_t full = GLM5_HEADS * GLM5_HEAD_DIM, local = full / 2u;
    return (launch ? ds4_rocm_glm5_mla_output_q8_small_m :
        ds4_rocm_glm5_mla_output_q8_small_m_supported)(ctx->tp_big_out,
        ctx->model_map, ctx->model_size, ctx->model->layer[il].mla.output,
        full, ctx->tp_rank * local, local, GLM5_WIDTH,
        (uint64_t)full / GLM5_Q8_QK * GLM5_Q8_BLOCK_BYTES, batch->mla_heads, rows);
}

int ds4_glm5_next_mla_output_batch_supported(const ds4_glm5_next_exec_ctx *ctx,
        const ds4_glm5_next_workspace *batch) {
    if (!context_valid(ctx) || !batch || !batch->decode_phase || ctx->tp_rank > 1u ||
        (batch->capacity_tokens != 2u && batch->capacity_tokens != 4u && batch->capacity_tokens != 6u))
        return 0;
    unsigned count = 0u;
    for (uint32_t il = 0u; il < ctx->model->trunk_count; ++il) {
        if (ctx->model->layer[il].attention != DS4_GLM5_NEXT_ATTN_MLA) continue;
        if (!mla_output_small_m(ctx, il, batch, batch->capacity_tokens, false)) return 0;
        ++count;
    }
    return count == 11u;
}

/* Keep causal attention and the M1 projection/residual arithmetic unchanged.
 * Only the output reduction is deferred. Heads and mHC coefficients live in
 * existing batch scratch; do not reuse the scalar row's overwritten split. */
static int verify_mla_attention_handoff(const ds4_glm5_next_exec_ctx *ctx,
        uint32_t il, ds4_glm5_next_mla_state *mla,
        ds4_glm5_next_workspace *batch, ds4_glm5_next_workspace *scalar,
        const ds4_gpu_tensor *hc_in, uint32_t frontier, uint32_t rows,
        bool row_sync, bool batch_output, int ok, glm5_mla_profile *profile) {
    const uint64_t hc_row = (uint64_t)GLM5_HC_WIDTH * sizeof(float);
    const uint64_t split_row = (uint64_t)GLM5_HC_MIX * sizeof(float);
    const uint64_t head_elements = (uint64_t)(GLM5_HEADS / 2u) * GLM5_HEAD_DIM;
    const uint64_t head_row = head_elements * sizeof(float);
    const uint64_t out_row = (uint64_t)GLM5_WIDTH * sizeof(float);
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    char error[128] = {0};
    for (uint32_t t = 0u; ok && t < rows; ++t) {
        ds4_glm5_next_workspace w = *scalar;
        ds4_gpu_tensor *in = ds4_gpu_tensor_view(hc_in, t * hc_row, hc_row);
        w.mla_heads = ds4_gpu_tensor_view(batch->mla_heads, t * head_row, head_row);
        w.hc_split = ds4_gpu_tensor_view(batch->hc_split, t * split_row, split_row);
        uint32_t slot = 0u, pool = 0u, visible = 0u;
        bool publish = false;
        ok = in && w.mla_heads && w.hc_split &&
            ds4_glm5_next_mla_append_plan(mla, &slot, &pool, &publish);
        const bool dense = ok && ds4_glm5_next_mla_dense_selection_visible(
            mla->token_count, mla->capacity_tokens, &visible);
        /* finish_attention stays true: it selects the scalar owned32 layout.
         * defer_tail is an independent stop point after the attention kernel. */
        if (ok) ok = dense ? mla_dense_selection_attention_impl(ctx, il, mla,
            &w, in, visible, slot, pool, publish, true, profile) :
            mla_sparse_selection_attention_impl(ctx, il, mla, &w, in, slot,
                pool, publish, DS4_GLM5_NEXT_INDEX_TOP_K, ctx->tp_big_out,
                true, true, true, profile);
        if (ok && row_sync) ok = ds4_gpu_synchronize();
        if (ok) ok = ds4_glm5_next_mla_append_commit(mla);
        ds4_gpu_tensor_free(in);
        ds4_gpu_tensor_free(w.mla_heads);
        ds4_gpu_tensor_free(w.hc_split);
    }
    if (profile) profile->last = glm5_exec_now_sec();
    if (ok && batch_output) {
        ok = mla_output_small_m(ctx, il, batch, rows, true);
    }
    for (uint32_t t = 0u; ok && !batch_output && t < rows; ++t) {
        ds4_gpu_tensor *heads = ds4_gpu_tensor_view(batch->mla_heads, t * head_row, head_row);
        ds4_gpu_tensor *out = ds4_gpu_tensor_view(ctx->tp_big_out, t * out_row, out_row);
        ok = heads && out && ds4_gpu_matmul_q8_0_kslice_tensor(out,
            ctx->model_map, ctx->model_size, layer->mla.output,
            GLM5_HEADS * GLM5_HEAD_DIM, ctx->tp_rank * head_elements,
            head_elements, GLM5_WIDTH, heads, 0u);
        ds4_gpu_tensor_free(heads);
        ds4_gpu_tensor_free(out);
    }
    /* Drain even after partial enqueue failure, then agree before posting
     * payload. Phase2 is distinct from the subsequent FFN phases0/1. */
    const int completed = ds4_gpu_synchronize();
    ok = completed && ok;
    if (profile) {
        const double now = glm5_exec_now_sec();
        profile->seconds[MLA_PROFILE_OUTPUT] += now - profile->last;
        profile->last = now;
    }
    if (!ds4_tp_verify_layer_agree(ctx->tp, *ctx->tp_sequence, il, frontier,
            rows, 2u, 0u, ok, error, sizeof(error))) {
        ds4_tp_mark_failed(ctx->tp);
        fprintf(stderr, "ds4: native MLA attention agreement failed rank=%u layer=%u: %s\n",
            ctx->tp_rank, il, error);
        return 0;
    }
    ok = ok && tp_exchange_rows(ctx, il, DS4_TP_GATE_ATTN, rows) &&
        mla_profile_mark(profile, MLA_PROFILE_EXCHANGE);
    for (uint32_t t = 0u; ok && t < rows; ++t) {
        ds4_glm5_next_workspace w = *scalar;
        ds4_gpu_tensor *in = ds4_gpu_tensor_view(hc_in, t * hc_row, hc_row);
        ds4_gpu_tensor *local = ds4_gpu_tensor_view(ctx->tp_big_out, t * out_row, out_row);
        ds4_gpu_tensor *peer = ds4_gpu_tensor_view(ctx->tp_big_in, t * out_row, out_row);
        w.hc_split = ds4_gpu_tensor_view(batch->hc_split, t * split_row, split_row);
        w.attention = ds4_gpu_tensor_view(batch->attention, t * out_row, out_row);
        w.after_attention = ds4_gpu_tensor_view(batch->after_attention, t * hc_row, hc_row);
        ok = in && local && peer && w.hc_split && w.attention && w.after_attention &&
            ds4_gpu_add_tensor(w.attention, local, peer, GLM5_WIDTH) &&
            mla_scalar_residual(layer, &w, in);
        ds4_gpu_tensor_free(in);
        ds4_gpu_tensor_free(local);
        ds4_gpu_tensor_free(peer);
        ds4_gpu_tensor_free(w.hc_split);
        ds4_gpu_tensor_free(w.attention);
        ds4_gpu_tensor_free(w.after_attention);
    }
    ok = ok && mla_profile_mark(profile, MLA_PROFILE_RESIDUAL);
    if (!ok) {
        ds4_gpu_synchronize();
        ds4_tp_mark_failed(ctx->tp);
    }
    return ok;
}

/* Consume one query from a previously batched sparse prelude. The selector
 * and gathered attention remain scalar and query ordered, preserving the
 * accepted causal pool state machine while removing repeated projection
 * launches. */
static int mla_sparse_selection_from_prelude_row(
        const ds4_glm5_next_exec_ctx *ctx,
        uint32_t il,
        ds4_glm5_next_state *state,
        ds4_glm5_next_workspace *batch_w,
        ds4_glm5_next_workspace *scalar_w,
        uint32_t row,
        uint32_t tail_slot,
        uint32_t pool_index,
        bool publish_pool,
        uint32_t top_k,
        ds4_gpu_tensor *selected_out,
        bool run_attention,
        const ds4_gpu_tensor *batched_scores,
        uint32_t batched_n_pools) {
    const ds4_glm5_next_mla_offsets *m = &ctx->model->layer[il].mla;
    ds4_glm5_next_mla_state *mla = &state->mla[il];
    const uint32_t pos = mla->token_count;
    uint32_t visible = 0u, n_pools = 0u, selected_pools = 0u;
    uint32_t selected_tokens = 0u;
    const uint64_t query_row =
        (uint64_t)GLM5_HEADS * GLM5_HEAD_DIM * sizeof(float);
    const uint64_t qk_row =
        (uint64_t)GLM5_HEADS * GLM5_KV_LORA * sizeof(float);
    const uint64_t index_q_row =
        (uint64_t)GLM5_INDEX_HEADS * GLM5_INDEX_DIM * sizeof(float);
    const uint64_t index_weights_row =
        (uint64_t)GLM5_INDEX_HEADS * sizeof(float);
    const uint64_t index_k_row =
        (uint64_t)GLM5_INDEX_DIM * sizeof(float);
    if (!ds4_glm5_next_mla_sparse_selection_plan(
            pos, mla->capacity_tokens, top_k,
            GLM5_INDEX_POOL, &visible, &n_pools, &selected_pools,
            &selected_tokens) ||
        row >= batch_w->capacity_tokens || n_pools == 0u ||
        n_pools > mla->capacity_pools ||
        n_pools > scalar_w->sparse_pool_capacity || selected_pools == 0u ||
        top_k > DS4_GLM5_NEXT_INDEX_TOP_K ||
        selected_pools > top_k / GLM5_INDEX_POOL ||
        selected_tokens > top_k + GLM5_INDEX_POOL - 1u ||
        (batched_scores && (batched_n_pools != n_pools ||
            ds4_gpu_tensor_bytes(batched_scores) <
                (uint64_t)n_pools * sizeof(float)))) {
        return 0;
    }
    ds4_gpu_tensor *query = run_attention ? ds4_gpu_tensor_view(
        batch_w->mla_query, (uint64_t)row * query_row, query_row) : NULL;
    ds4_gpu_tensor *qk_low = run_attention ? ds4_gpu_tensor_view(
        batch_w->mla_qk_low, (uint64_t)row * qk_row, qk_row) : NULL;
    ds4_gpu_tensor *index_q = ds4_gpu_tensor_view(
        batch_w->mla_index_q, (uint64_t)row * index_q_row, index_q_row);
    ds4_gpu_tensor *index_weights = ds4_gpu_tensor_view(
        batch_w->mla_index_weights,
        (uint64_t)row * index_weights_row, index_weights_row);
    ds4_gpu_tensor *selected = selected_out ? selected_out :
        scalar_w->mla_selected_token;
    int ok = (!run_attention || (query && qk_low)) && selected &&
        index_q && index_weights &&
        (batched_scores ||
         (ds4_gpu_tensor_copy(
              mla->index_tail, (uint64_t)tail_slot * index_k_row,
              batch_w->mla_index_k_norm, (uint64_t)row * index_k_row,
              index_k_row) &&
          ds4_gpu_tensor_copy(
              mla->pool_gate_tail, (uint64_t)tail_slot * index_k_row,
              batch_w->mla_pool_gate_raw, (uint64_t)row * index_k_row,
              index_k_row) &&
          ds4_glm5_next_mla_verify_record(
              mla, mla->token_count, batch_w->mla_index_k_norm,
              (uint64_t)row * index_k_row, batch_w->mla_pool_gate_raw,
              (uint64_t)row * index_k_row, 1u) &&
          mla_publish_completed_pool(
              ctx, m, mla, scalar_w, pool_index, publish_pool))) &&
        (batched_scores ?
         ds4_gpu_indexer_topk_tensor(
             scalar_w->mla_selected_pool, batched_scores,
             batched_n_pools, 1u, selected_pools) :
         (ds4_gpu_glm_indexer_score_one_tensor(
              scalar_w->mla_pool_scores, index_q, index_weights,
              mla->index_pool, n_pools, GLM5_INDEX_HEADS, GLM5_INDEX_DIM,
              0.015625f, false) &&
          ds4_gpu_glm5_mask_pool_scores_tensor(
              scalar_w->mla_pool_scores, mla->index_pool_valid, n_pools) &&
          ds4_gpu_indexer_topk_tensor(
              scalar_w->mla_selected_pool, scalar_w->mla_pool_scores,
              n_pools, 1u, selected_pools))) &&
        ds4_gpu_glm5_expand_pool_selection_tensor(
            selected, scalar_w->mla_selected_pool,
            mla->index_pool_ids, mla->index_pool_valid,
            mla->index_valid_keys, n_pools, selected_pools,
            mla->capacity_tokens, mla->first_valid, visible,
            top_k, GLM5_INDEX_POOL) &&
        (!run_attention || ds4_gpu_glm_attention_indexed_decode_typed_tensor(
            scalar_w->mla_heads, query, qk_low,
            mla->compact_kv, NULL, ctx->model_map, ctx->model_size,
            m->v_b, 8u, selected, selected_tokens,
            mla->capacity_tokens, false, GLM5_HEADS, GLM5_KV_LORA,
            GLM5_HEAD_DIM, 0u, GLM5_HEAD_DIM, 0u,
            1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f));
    ds4_gpu_tensor_free(index_weights);
    ds4_gpu_tensor_free(index_q);
    ds4_gpu_tensor_free(qk_low);
    ds4_gpu_tensor_free(query);
    return ok;
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
    if (!ds4_glm5_next_mla_verify_record(
            mla, pos0, w->mla_index_k_norm, 0u, w->mla_pool_gate_raw, 0u,
            n_tokens)) return 0;
#ifdef DS4_ROCM_BUILD
    const char *batch_pool_value =
        getenv("DS4_ROCM_GLM5_BATCH_POOL_STAGE");
    const int batch_pool_stage =
        (batch_pool_value != NULL && strcmp(batch_pool_value, "1") == 0) ||
        (ctx->tp && (ds4_tp_prefill_config(ctx->tp) &
                     DS4_TP_PREFILL_CONFIG_GLM5_INDEXER_SCORE_BATCH) != 0u);
    if (batch_pool_value != NULL && strcmp(batch_pool_value, "0") != 0 &&
        strcmp(batch_pool_value, "1") != 0) {
        return 0;
    }
    if (batch_pool_stage && pos0 <= mla->capacity_tokens &&
        n_tokens <= mla->capacity_tokens - pos0 &&
        (pos0 & 3u) == 0u && (n_tokens & 3u) == 0u) {
        static int logged_batch_pool_stage[2] = {0, 0};
        const uint32_t rank = ctx->tp_rank < 2u ? ctx->tp_rank : 0u;
        if (!logged_batch_pool_stage[rank]) {
            fprintf(stderr,
                    "ds4: GLM5 aligned batch pool publication engaged "
                    "rank=%u pos0=%u tokens=%u\n",
                    ctx->tp_rank, pos0, n_tokens);
            logged_batch_pool_stage[rank] = 1;
        }
        return ds4_gpu_glm5_publish_pools_batch_tensor(
            mla->index_pool, mla->index_pool_ids, mla->index_pool_valid,
            w->mla_index_k_norm, w->mla_pool_gate_raw,
            ctx->model_map, ctx->model_size, offsets->index_pool_ape,
            pos0 / 4u, n_tokens, GLM5_INDEX_DIM);
    }
#endif
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
    const uint64_t full_bytes = (uint64_t)n_tokens * GLM5_HEADS *
        GLM5_HEAD_DIM * sizeof(float);
    const uint32_t heads = ds4_gpu_tensor_bytes(w->mla_heads) ==
        full_bytes / 2u ? GLM5_HEADS / 2u : GLM5_HEADS;
    return ds4_gpu_glm_value_project_typed_batch_heads_tensor(
        w->mla_heads, w->routed_experts, ctx->model_map, ctx->model_size,
        offsets->v_b, 8u, n_tokens, heads,
        GLM5_KV_LORA, GLM5_HEAD_DIM);
}

/* Diagnostic-only comparison on inputs produced by the real executor. The
 * candidate output never enters inference, and the switch is rejected by the
 * benchmark candidate gate. */
static int mla_sparse_attention_compare_f16(
        const ds4_glm5_next_mla_state *mla,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *reference,
        uint32_t layer,
        uint32_t pos0,
        uint32_t n_tokens) {
    fprintf(stderr,
            "ds4: GLM5 sparse F16 real-row compare begin layer=%u "
            "pos0=%u rows=%u\n",
            layer, pos0, n_tokens);
    const uint64_t out_count =
        (uint64_t)n_tokens * GLM5_HEADS * GLM5_KV_LORA;
    const uint64_t qk_count = out_count;
    const uint64_t selected_count =
        (uint64_t)n_tokens * GLM5_SELECTED_STRIDE;
    const uint64_t cache_count =
        (uint64_t)mla->capacity_tokens * GLM5_KV_LORA;
    if (out_count > SIZE_MAX / sizeof(float) ||
        selected_count > SIZE_MAX / sizeof(int32_t) ||
        cache_count > SIZE_MAX / sizeof(float)) return 0;
    ds4_gpu_tensor *candidate = ds4_gpu_tensor_alloc(
        out_count * sizeof(float));
    float *reference_h = (float *)malloc((size_t)out_count * sizeof(float));
    float *candidate_h = (float *)malloc((size_t)out_count * sizeof(float));
    float *qk_h = (float *)malloc((size_t)qk_count * sizeof(float));
    float *cache_h = (float *)malloc((size_t)cache_count * sizeof(float));
    int32_t *selected_h = (int32_t *)malloc(
        (size_t)selected_count * sizeof(int32_t));
    int ok = candidate && reference_h && candidate_h && qk_h && cache_h &&
        selected_h && ds4_gpu_synchronize() &&
        ds4_rocm_glm5_sparse_attention_f16_gemm_rows(
            candidate, qk_low, mla->compact_kv, selected, n_tokens,
            GLM5_SELECTED_STRIDE, mla->capacity_tokens) > 0 &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read((ds4_gpu_tensor *)reference, 0u, reference_h,
                            out_count * sizeof(float)) &&
        ds4_gpu_tensor_read(candidate, 0u, candidate_h,
                            out_count * sizeof(float)) &&
        ds4_gpu_tensor_read((ds4_gpu_tensor *)qk_low, 0u, qk_h,
                            qk_count * sizeof(float)) &&
        ds4_gpu_tensor_read((ds4_gpu_tensor *)selected, 0u, selected_h,
                            selected_count * sizeof(int32_t)) &&
        ds4_gpu_tensor_read(mla->compact_kv, 0u, cache_h,
                            cache_count * sizeof(float));
    if (ok) {
        long double error2 = 0.0L, reference2 = 0.0L;
        long double candidate2 = 0.0L, dot = 0.0L;
        long double qk2 = 0.0L, kv2 = 0.0L;
        double max_abs = 0.0, qk_max = 0.0, kv_max = 0.0;
        uint64_t nonfinite = 0u, kv_samples = 0u;
        for (uint64_t i = 0u; i < out_count; ++i) {
            const double a = reference_h[i];
            const double b = candidate_h[i];
            if (!isfinite(a) || !isfinite(b)) {
                nonfinite++;
                continue;
            }
            const double e = b - a;
            error2 += (long double)e * e;
            reference2 += (long double)a * a;
            candidate2 += (long double)b * b;
            dot += (long double)a * b;
            if (fabs(e) > max_abs) max_abs = fabs(e);
        }
        for (uint64_t i = 0u; i < qk_count; ++i) {
            const double value = qk_h[i];
            if (!isfinite(value)) {
                nonfinite++;
                continue;
            }
            qk2 += (long double)value * value;
            if (fabs(value) > qk_max) qk_max = fabs(value);
        }
        for (uint64_t i = 0u; i < selected_count; ++i) {
            const int32_t row = selected_h[i];
            if (row < 0 || (uint32_t)row >= mla->capacity_tokens) continue;
            const float *kv = cache_h + (uint64_t)(uint32_t)row * GLM5_KV_LORA;
            for (uint32_t j = 0u; j < GLM5_KV_LORA; ++j) {
                const double value = kv[j];
                if (!isfinite(value)) {
                    nonfinite++;
                    continue;
                }
                kv2 += (long double)value * value;
                if (fabs(value) > kv_max) kv_max = fabs(value);
                kv_samples++;
            }
        }
        const double nmse = (double)(
            error2 / fmaxl(reference2, 1.0e-30L));
        const double cosine = reference2 > 0.0L && candidate2 > 0.0L ?
            (double)(dot / sqrtl(reference2 * candidate2)) : 0.0;
        const double ref_rms = sqrt((double)(reference2 / out_count));
        const double qk_rms = sqrt((double)(qk2 / qk_count));
        const double kv_rms = kv_samples ?
            sqrt((double)(kv2 / kv_samples)) : 0.0;
        fprintf(stderr,
                "ds4: GLM5 sparse F16 real-row compare layer=%u "
                "pos0=%u rows=%u "
                "max_abs=%.9g nmse=%.9g cosine=%.12g ref_rms=%.9g "
                "qk_max=%.9g qk_rms=%.9g kv_max=%.9g kv_rms=%.9g "
                "nonfinite=%llu\n",
                layer, pos0, n_tokens, max_abs, nmse, cosine, ref_rms,
                qk_max, qk_rms, kv_max, kv_rms,
                (unsigned long long)nonfinite);
        ok = nonfinite == 0u && nmse <= 1.0e-6 &&
             max_abs <= 0.01 * ref_rms && cosine >= 0.99999;
    }
    free(selected_h);
    free(cache_h);
    free(qk_h);
    free(candidate_h);
    free(reference_h);
    ds4_gpu_tensor_free(candidate);
    if (!ok) {
        fprintf(stderr,
                "ds4: GLM5 sparse F16 real-row compare failed "
                "layer=%u pos0=%u rows=%u\n", layer, pos0, n_tokens);
    }
    return ok;
}

static int mla_sparse_attention_rows_deferred(
        const ds4_glm5_next_exec_ctx *ctx,
        const ds4_glm5_next_mla_state *mla,
        ds4_glm5_next_workspace *w,
        uint32_t layer,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t *selected_min_out,
        uint32_t *selected_max_out) {
    const uint64_t query_row_bytes =
        (uint64_t)GLM5_HEADS * GLM5_HEAD_DIM * sizeof(float);
    const uint64_t qk_row_bytes =
        (uint64_t)GLM5_HEADS * GLM5_KV_LORA * sizeof(float);
    const uint64_t selected_row_bytes =
        (uint64_t)GLM5_SELECTED_STRIDE * sizeof(int32_t);
    if (!ctx || !mla || !w || n_tokens == 0u ||
        !selected_min_out || !selected_max_out ||
        ds4_gpu_tensor_bytes(w->routed_experts) <
            (uint64_t)n_tokens * qk_row_bytes ||
        ds4_gpu_tensor_bytes(w->mla_selected_token) <
            (uint64_t)n_tokens * selected_row_bytes) {
        return 0;
    }
    uint32_t selected_min = UINT32_MAX;
    uint32_t selected_max = 0u;
    const bool head_shared_rows =
        getenv("DS4_ROCM_GLM5_SPARSE_ATTN_HEAD_SHARED") != NULL &&
        strcmp(getenv("DS4_ROCM_GLM5_SPARSE_ATTN_HEAD_SHARED"), "1") == 0;
    const bool f16_gemm_rows = ctx->tp &&
        (ds4_tp_runtime_features(ctx->tp) &
         DS4_TP_FEATURE_GLM5_SPARSE_ATTN_F16_GEMM) != 0u;
    if (f16_gemm_rows && !head_shared_rows) return 0;
    static int f16_compare_claimed;
    const bool compare_f16 = head_shared_rows && !f16_gemm_rows &&
        !f16_compare_claimed &&
        getenv("DS4_ROCM_GLM5_SPARSE_ATTN_COMPARE_F16") != NULL &&
        strcmp(getenv("DS4_ROCM_GLM5_SPARSE_ATTN_COMPARE_F16"), "1") == 0;
    uint32_t compare_layer = UINT32_MAX;
    uint32_t compare_pos = UINT32_MAX;
    if (compare_f16) {
        const char *layer_env = getenv(
            "DS4_ROCM_GLM5_SPARSE_ATTN_COMPARE_F16_LAYER");
        const char *pos_env = getenv(
            "DS4_ROCM_GLM5_SPARSE_ATTN_COMPARE_F16_POS");
        char *end = NULL;
        if (layer_env && layer_env[0]) {
            errno = 0;
            const unsigned long parsed = strtoul(layer_env, &end, 10);
            if (errno != 0 || !end || *end != '\0' || parsed > UINT32_MAX)
                return 0;
            compare_layer = (uint32_t)parsed;
        }
        if (pos_env && pos_env[0]) {
            errno = 0;
            const unsigned long parsed = strtoul(pos_env, &end, 10);
            if (errno != 0 || !end || *end != '\0' || parsed > UINT32_MAX)
                return 0;
            compare_pos = (uint32_t)parsed;
        }
    }
    for (uint32_t row0 = 0u; row0 < n_tokens; ++row0) {
        uint32_t visible = 0u, n_pools = 0u, selected_pools = 0u;
        uint32_t selected_tokens = 0u;
        if (!ds4_glm5_next_mla_sparse_selection_plan(
                (uint64_t)pos0 + row0, mla->capacity_tokens,
                DS4_GLM5_NEXT_INDEX_TOP_K, GLM5_INDEX_POOL,
                &visible, &n_pools, &selected_pools, &selected_tokens)) {
            return 0;
        }
        if (selected_tokens > GLM5_SELECTED_STRIDE) return 0;
        if (selected_tokens < selected_min) selected_min = selected_tokens;
        if (selected_tokens > selected_max) selected_max = selected_tokens;
        if (!head_shared_rows) {
            ds4_gpu_tensor *lora = ds4_gpu_tensor_view(
                w->routed_experts, (uint64_t)row0 * qk_row_bytes,
                qk_row_bytes);
            ds4_gpu_tensor *query = ds4_gpu_tensor_view(
                w->mla_query, (uint64_t)row0 * query_row_bytes,
                query_row_bytes);
            ds4_gpu_tensor *qk_low = ds4_gpu_tensor_view(
                w->mla_qk_low, (uint64_t)row0 * qk_row_bytes,
                qk_row_bytes);
            ds4_gpu_tensor *selected = ds4_gpu_tensor_view(
                w->mla_selected_token, (uint64_t)row0 * selected_row_bytes,
                selected_row_bytes);
            const int ok = lora && query && qk_low && selected &&
                ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
                    lora, query, qk_low, mla->compact_kv, NULL, selected,
                    1u, selected_tokens,
                    mla->capacity_tokens, false, GLM5_HEADS, GLM5_KV_LORA,
                    GLM5_HEAD_DIM, 0u, 0u,
                    1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            ds4_gpu_tensor_free(selected);
            ds4_gpu_tensor_free(qk_low);
            ds4_gpu_tensor_free(query);
            ds4_gpu_tensor_free(lora);
            if (!ok) return 0;
        }
    }
    if (head_shared_rows) {
        for (uint32_t row0 = 0u; row0 < n_tokens; row0 += 16u) {
            const uint32_t rows = n_tokens - row0 < 16u ?
                n_tokens - row0 : 16u;
            ds4_gpu_tensor *lora = ds4_gpu_tensor_view(
                w->routed_experts, (uint64_t)row0 * qk_row_bytes,
                (uint64_t)rows * qk_row_bytes);
            ds4_gpu_tensor *qk_low = ds4_gpu_tensor_view(
                w->mla_qk_low, (uint64_t)row0 * qk_row_bytes,
                (uint64_t)rows * qk_row_bytes);
            ds4_gpu_tensor *selected = ds4_gpu_tensor_view(
                w->mla_selected_token, (uint64_t)row0 * selected_row_bytes,
                (uint64_t)rows * selected_row_bytes);
            const int rc = lora && qk_low && selected ?
                (f16_gemm_rows ?
                    ds4_rocm_glm5_sparse_attention_f16_gemm_rows(
                        lora, qk_low, mla->compact_kv, selected, rows,
                        GLM5_SELECTED_STRIDE, mla->capacity_tokens) :
                    ds4_rocm_glm5_sparse_attention_exact_rows(
                        lora, qk_low, mla->compact_kv, selected, rows,
                        GLM5_SELECTED_STRIDE, mla->capacity_tokens)) : 0;
            const uint32_t tile_pos = pos0 + row0;
            const bool compare_tile = compare_f16 &&
                (compare_layer == UINT32_MAX || compare_layer == layer) &&
                (compare_pos == UINT32_MAX ? row0 == 0u :
                 compare_pos == tile_pos);
            const int compare_ok = rc > 0 && compare_tile ?
                mla_sparse_attention_compare_f16(
                    mla, qk_low, selected, lora, layer, tile_pos, rows) : 1;
            if (compare_ok && compare_tile) {
                f16_compare_claimed = 1;
            }
            ds4_gpu_tensor_free(selected);
            ds4_gpu_tensor_free(qk_low);
            ds4_gpu_tensor_free(lora);
            if (rc <= 0 || !compare_ok) return 0;
        }
    }
    *selected_min_out = selected_min;
    *selected_max_out = selected_max;
    return 1;
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
    const bool owned = ds4_gpu_tensor_bytes(w->mla_heads) ==
        (uint64_t)n_tokens * half_heads * sizeof(float);
    const uint64_t activation_start = owned ? 0u : in_start;
    const uint64_t activation_stride = owned ? half_heads : full_heads;
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
            activation_start);
        ds4_gpu_tensor_free(out);
        return ok;
    }
    /* The strided Q8 output projection is byte-identical to the one-row Q4
     * path in the production-shape oracle and frozen teacher/full-run gates.
     * Keep the old row loop as an explicit rollback, while Q2 continues to
     * use the same batched entry point unconditionally. */
    const int batch = force_serial &&
        getenv("DS4_GLM5_DISABLE_Q4_BATCH_MLA_OUTPUT") != NULL ? -1 :
        ds4_rocm_q8_kslice_f32_rows_strided(
            ctx->tp_big_out, ctx->model_map, ctx->model_size,
            offsets->output, full_heads, GLM5_WIDTH, in_start, half_heads,
            w->mla_heads, activation_start, n_tokens, activation_stride);
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
            (uint64_t)t * activation_stride + activation_start);
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
    ds4_glm5_next_mla_offsets local_m = layer->mla;
    const ds4_glm5_next_mla_offsets *m = &local_m;
    ds4_glm5_next_mla_state *mla = &state->mla[il];
    const uint32_t pos0 = mla->token_count;
    const uint32_t n_selected = pos0 + n_tokens;
    const bool owned = n_tokens >= 16u &&
        getenv("DS4_GLM5_MLA_BATCH_OWNED_HEADS") != NULL &&
        strcmp(getenv("DS4_GLM5_MLA_BATCH_OWNED_HEADS"), "1") == 0 &&
        ctx->tp_rank < 2u && ctx->trace_prefix == NULL;
    const uint32_t attention_heads = owned ? GLM5_HEADS / 2u : GLM5_HEADS;
    ds4_gpu_tensor *query_view = NULL;
    ds4_gpu_tensor *qk_view = NULL;
    ds4_gpu_tensor *heads_view = NULL;
    ds4_glm5_next_workspace owned_w = *w;
    ds4_gpu_tensor *query = w->mla_query;
    ds4_gpu_tensor *qk_low = w->mla_qk_low;
    if (owned) {
        const uint64_t first_head = (uint64_t)ctx->tp_rank * attention_heads;
        const uint64_t q_offset = first_head * GLM5_HEAD_DIM *
            (GLM5_Q_RANK / GLM5_Q8_QK) * GLM5_Q8_BLOCK_BYTES;
        const uint64_t k_offset = first_head * GLM5_KV_LORA *
            (GLM5_HEAD_DIM / GLM5_Q8_QK) * GLM5_Q8_BLOCK_BYTES;
        const uint64_t v_offset = first_head * GLM5_HEAD_DIM *
            (GLM5_KV_LORA / GLM5_Q8_QK) * GLM5_Q8_BLOCK_BYTES;
        if (m->q_b > UINT64_MAX - q_offset ||
            m->k_b > UINT64_MAX - k_offset ||
            m->v_b > UINT64_MAX - v_offset) return 0;
        local_m.q_b += q_offset;
        local_m.k_b += k_offset;
        local_m.v_b += v_offset;
        query_view = ds4_gpu_tensor_view(
            w->mla_query, 0u,
            (uint64_t)n_tokens * attention_heads * GLM5_HEAD_DIM *
                sizeof(float));
        qk_view = ds4_gpu_tensor_view(
            w->mla_qk_low, 0u,
            (uint64_t)n_tokens * attention_heads * GLM5_KV_LORA *
                sizeof(float));
        heads_view = ds4_gpu_tensor_view(
            w->mla_heads, 0u,
            (uint64_t)n_tokens * attention_heads * GLM5_HEAD_DIM *
                sizeof(float));
        if (!query_view || !qk_view || !heads_view) {
            ds4_gpu_tensor_free(heads_view);
            ds4_gpu_tensor_free(qk_view);
            ds4_gpu_tensor_free(query_view);
            return 0;
        }
        query = query_view;
        qk_low = qk_view;
        owned_w.mla_query = query_view;
        owned_w.mla_qk_low = qk_view;
        owned_w.mla_heads = heads_view;
        static int reported[2];
        if (!reported[ctx->tp_rank]) {
            fprintf(stderr,
                    "ds4: GLM5 batched MLA owned heads active rank=%u heads=%u\n",
                    ctx->tp_rank, attention_heads);
            reported[ctx->tp_rank] = 1;
        }
    }
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
            query, ctx->model_map, ctx->model_size, m->q_b,
            GLM5_Q_RANK, (uint64_t)attention_heads * GLM5_HEAD_DIM,
            w->mla_q_resid, n_tokens) &&
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
            qk_low, query,
            ctx->model_map, ctx->model_size, m->k_b, 8u, n_tokens,
            attention_heads, GLM5_KV_LORA, GLM5_HEAD_DIM, GLM5_HEAD_DIM) &&
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
            w->routed_experts, query, qk_low,
            mla->compact_kv, NULL, n_tokens, pos0, n_selected,
            mla->capacity_tokens, false, attention_heads, GLM5_KV_LORA,
            GLM5_HEAD_DIM, 0u, 0u,
            1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f) &&
        mla_value_project_rows_batch(ctx, m, owned ? &owned_w : w, n_tokens) &&
        mla_output_project_rows_batch(
            ctx, m, owned ? &owned_w : w, n_tokens,
            !(layer->ffn_weight.gate_exps_type == 16u &&
              layer->ffn_weight.up_exps_type == 16u &&
              layer->ffn_weight.down_exps_type == 10u));
    ds4_gpu_tensor_free(heads_view);
    ds4_gpu_tensor_free(qk_view);
    ds4_gpu_tensor_free(query_view);
    return ok &&
        tp_exchange_rows(ctx, il, DS4_TP_GATE_ATTN, n_tokens) &&
        ds4_gpu_add_tensor(w->attention, ctx->tp_big_out, ctx->tp_big_in,
                           (uint32_t)elements) &&
        ds4_gpu_hc_expand_split_tensor(
            w->after_attention, w->attention, hc_in, w->hc_split,
            GLM5_WIDTH, GLM5_HC);
}

static int routed_ffn_prefix(const ds4_glm5_next_exec_ctx *ctx, uint32_t il,
                             ds4_glm5_next_workspace *w) {
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    const bool native = !layer->is_trunk && il == DS4_GLM5_NEXT_TRUNK_COUNT;
    if (w->draft_only != native) return 0;
    if (native) return ds4_gpu_rms_norm_weight_tensor(w->ffn_hidden, w->after_attention,
        ctx->model_map, ctx->model_size, layer->ffn_norm,
        GLM5_WIDTH, ctx->model->rms_norm_eps);
    return ds4_gpu_rms_norm_plain_rows_tensor(w->ffn_flat, w->after_attention,
            GLM5_HC_WIDTH, 1u, ctx->model->rms_norm_eps) &&
        ds4_gpu_matmul_bf16_tensor(w->ffn_mix, ctx->model_map, ctx->model_size,
            layer->hc.ffn_fn, GLM5_HC_WIDTH, GLM5_HC_MIX, w->ffn_flat, 1u) &&
        ds4_gpu_hc_split_weighted_sum_norm_tensor(
            w->ffn_collapsed, w->ffn_hidden, w->ffn_split, w->ffn_mix,
            w->after_attention, ctx->model_map, ctx->model_size,
            layer->hc.ffn_scale, layer->hc.ffn_base, layer->ffn_norm,
            GLM5_WIDTH, GLM5_HC, 20u, ctx->model->hc_eps, ctx->model->rms_norm_eps);
}

static int routed_ffn_one_impl(const ds4_glm5_next_exec_ctx *ctx,
                          uint32_t il,
                          uint32_t token_ordinal,
                          ds4_glm5_next_workspace *w,
                          ds4_gpu_tensor *hc_out,
                          const ds4_glm5_next_workspace *prepared,
                          uint32_t prepared_row) {
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    const ds4_glm5_next_ffn_offsets *f = &layer->ffn_weight;
    const bool native = !layer->is_trunk && il == DS4_GLM5_NEXT_TRUNK_COUNT;
    if (w->draft_only != native) return 0;
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
    if (prepared && (native || !layer->is_trunk || !ctx->force_bulk_gates ||
        layer->attention != DS4_GLM5_NEXT_ATTN_KDA || !w->decode_phase ||
        q4_residency != 1 || prepared == w ||
        prepared_row >= prepared->capacity_tokens)) return 0;
    /* A nonresident layer releases packed descriptors below; retain its
     * historical fence even when the diagnostic switch is enabled. */
    const int async_terminal = q4_residency == 1 &&
        glm5_decode_async_layer_enabled(w, 1u);
    const int shared_route_setting = native ? 0 :
        shared_route_overlap_mode(ctx, w, q4_residency);
    const int shared_route_overlap = prepared ? 0 : shared_route_setting;
    if (shared_route_setting < 0) {
        fprintf(stderr, "ds4: GLM5 shared route overlap unsupported configuration rank=%u\n",
                ctx->tp_rank);
        ds4_tp_mark_failed(ctx->tp);
        return 0;
    }
    int ok = 1;
    if (prepared) {
        const uint64_t hidden_row = (uint64_t)GLM5_WIDTH * sizeof(float);
        const uint64_t split_row = (uint64_t)GLM5_HC_MIX * sizeof(float);
        ok = ds4_gpu_tensor_copy(w->ffn_hidden, 0u, prepared->ffn_hidden,
                prepared_row * hidden_row, hidden_row) &&
            ds4_gpu_tensor_copy(w->ffn_split, 0u, prepared->ffn_split,
                prepared_row * split_row, split_row) &&
            ds4_gpu_tensor_copy(w->shared_out, 0u, prepared->shared_out,
                prepared_row * hidden_row, hidden_row);
    } else {
        ok = routed_ffn_prefix(ctx, il, w);
    }
    if (!ok) { fprintf(stderr, "ds4: GLM5 routed layer %u failed at FFN prefix rank=%u\n", il, ctx->tp_rank); goto routed_one_done; }
    ok = ok && ds4_gpu_matmul_f32_tensor(
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
                      w->router_logits, w->ffn_hidden,
                      shared_route_overlap ? w : NULL);
    if (!ok) {
        route_failure_stats("after_attention", w->after_attention,
                            GLM5_HC_WIDTH);
        /* Prepared rows restore the live operands, not prefix diagnostic
         * temporaries, which belong to the last prepared row. */
        if (!prepared) {
            route_failure_stats("ffn_mix", w->ffn_mix, GLM5_HC_MIX);
            route_failure_stats("ffn_collapsed", w->ffn_collapsed, GLM5_WIDTH);
        }
        fprintf(stderr, "ds4: GLM5 routed layer %u failed at router prelude rank=%u\n",
                il, ctx->tp_rank);
        goto routed_one_done;
    }
    if (shared_route_overlap) {
        static int logged[2];
        if (!logged[ctx->tp_rank]) {
            fprintf(stderr, "ds4: GLM5 shared FFN queued before route agreement rank=%u weights=original cache_bytes=0\n",
                    ctx->tp_rank);
            logged[ctx->tp_rank] = 1;
        }
    }
    if (q4_residency < 0) {
        fprintf(stderr,
                "ds4: GLM5 routed layer %u has partial Q4_K residency rank=%u\n",
                il, ctx->tp_rank);
        ok = 0;
    }
    const char *shared_q8_pair_env =
        getenv("DS4_ROCM_GLM5_SHARED_Q8_PAIR_DECODE");
    const int shared_q8_pair_decode =
        w->decode_phase && shared_q8_pair_env &&
        strcmp(shared_q8_pair_env, "1") == 0 &&
        GLM5_WIDTH == 4096u && GLM5_RANK_MID == 1024u;
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
        const bool use_scratch = native || (w->decode_phase && scratch_windows &&
            strcmp(scratch_windows, "1") == 0);
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
        const char *persist_prefill =
            getenv("DS4_ROCM_GLM5_WINDOW_PERSIST_PREFILL");
        const bool persist_this_call =
            (persist_windows && strcmp(persist_windows, "1") == 0) ||
            (!w->decode_phase && persist_prefill &&
             strcmp(persist_prefill, "1") == 0);
        if (ok && use_scratch && il < DS4_GLM5_NEXT_LAYER_COUNT) {
            const char *reuse = getenv("DS4_GLM5_NATIVE_WINDOW_REUSE");
            const bool reuse_native = native && reuse && strcmp(reuse, "1") == 0;
            if (!w->q4_window_scratch)
                w->q4_window_scratch =
                    ds4_gpu_q4k_window_cache_create(&config);
            else if (!reuse_native)
                ok = ds4_gpu_q4k_window_cache_rebind(
                    w->q4_window_scratch, &config);
            /* A native workspace is bound to immutable layer45/model/rank
             * before entry and its previous consumer has synchronized.
             * Keep the same eight slots; prepare handles routing changes.
             * Generic scratch workspaces still rebind across trunk layers. */
            cache = w->q4_window_scratch;
        } else if (ok && persist_this_call &&
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
        overlap_prefetched = ok && !native && use_scratch && overlap_env &&
            strcmp(overlap_env, "1") == 0;
        if (overlap_prefetched) {
            /* Start selected-expert uploads before the shared Q8 path.  The
             * routed helper repeats the metadata check and waits on the
             * cache event immediately before its consumer kernel. */
            ok = ds4_gpu_q4k_window_cache_prefetch(
                cache, w->router_selected, w->router_weights,
                GLM5_EXPERTS_USED);
        }
        if (ok && overlap_prefetched) {
            if (shared_q8_pair_decode) {
                ok = ds4_gpu_matmul_q8_0_pair_tensor(
                    w->shared_gate, w->shared_up, ctx->model_map,
                    ctx->model_size,
                    f->gate_shexp + (uint64_t)rank_mid_base * q8_gate_row_bytes,
                    f->up_shexp + (uint64_t)rank_mid_base * q8_gate_row_bytes,
                    GLM5_WIDTH, GLM5_RANK_MID, GLM5_RANK_MID,
                    w->ffn_hidden, 1u);
            } else {
                ok = ds4_gpu_matmul_q8_0_tensor(
                    w->shared_gate, ctx->model_map, ctx->model_size,
                    f->gate_shexp + (uint64_t)rank_mid_base * q8_gate_row_bytes,
                    GLM5_WIDTH, GLM5_RANK_MID, w->ffn_hidden, 1u) &&
                    ds4_gpu_matmul_q8_0_tensor(
                    w->shared_up, ctx->model_map, ctx->model_size,
                    f->up_shexp + (uint64_t)rank_mid_base * q8_gate_row_bytes,
                    GLM5_WIDTH, GLM5_RANK_MID, w->ffn_hidden, 1u);
            }
            if (ok) ok = ds4_gpu_swiglu_tensor(
                w->shared_mid, w->shared_gate, w->shared_up,
                GLM5_RANK_MID, 10.0f, 1.0f);
            if (ok && shared_q8_pair_decode) {
                static int logged_shared_q8_pair[2] = {0, 0};
                const uint32_t rank = ctx->tp_rank < 2u ? ctx->tp_rank : 0u;
                if (!logged_shared_q8_pair[rank]) {
                    fprintf(stderr,
                            "ds4: GLM5 shared gate/up Q8 decode pair engaged "
                            "rank=%u overlap=1\n", ctx->tp_rank);
                    logged_shared_q8_pair[rank] = 1;
                }
            }
        }
        ok = ok && cache && ds4_gpu_routed_moe_one_packed_q4k_window_tensor(
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
    if (ok && !prepared && !shared_route_overlap && !overlap_prefetched && shared_q8_pair_decode) ok =
        ds4_gpu_matmul_q8_0_pair_tensor(
            w->shared_gate, w->shared_up, ctx->model_map, ctx->model_size,
            f->gate_shexp + (uint64_t)rank_mid_base * q8_gate_row_bytes,
            f->up_shexp + (uint64_t)rank_mid_base * q8_gate_row_bytes,
            GLM5_WIDTH, GLM5_RANK_MID, GLM5_RANK_MID,
            w->ffn_hidden, 1u);
    if (ok && !prepared && !shared_route_overlap && !overlap_prefetched && !shared_q8_pair_decode) ok =
        ds4_gpu_matmul_q8_0_tensor(
            w->shared_gate, ctx->model_map, ctx->model_size,
            f->gate_shexp + (uint64_t)rank_mid_base * q8_gate_row_bytes,
            GLM5_WIDTH, GLM5_RANK_MID, w->ffn_hidden, 1u);
    if (!ok) { fprintf(stderr, "ds4: GLM5 routed layer %u failed at shared gate rank=%u\n", il, ctx->tp_rank); goto routed_one_done; }
    if (ok && !prepared && !shared_route_overlap && !overlap_prefetched && !shared_q8_pair_decode) ok = ds4_gpu_matmul_q8_0_tensor(
            w->shared_up, ctx->model_map, ctx->model_size,
            f->up_shexp + (uint64_t)rank_mid_base * q8_gate_row_bytes,
            GLM5_WIDTH, GLM5_RANK_MID, w->ffn_hidden, 1u);
    if (!ok) { fprintf(stderr, "ds4: GLM5 routed layer %u failed at shared up rank=%u\n", il, ctx->tp_rank); goto routed_one_done; }
    if (ok && !prepared && !shared_route_overlap && !overlap_prefetched) ok = ds4_gpu_swiglu_tensor(
            w->shared_mid, w->shared_gate, w->shared_up,
            GLM5_RANK_MID, 10.0f, 1.0f);
    if (ok && !prepared && !overlap_prefetched && shared_q8_pair_decode) {
        static int logged_shared_q8_pair_serial[2] = {0, 0};
        const uint32_t rank = ctx->tp_rank < 2u ? ctx->tp_rank : 0u;
        if (!logged_shared_q8_pair_serial[rank]) {
            fprintf(stderr,
                    "ds4: GLM5 shared gate/up Q8 decode pair engaged "
                    "rank=%u overlap=0\n", ctx->tp_rank);
            logged_shared_q8_pair_serial[rank] = 1;
        }
    }
    if (!ok) { fprintf(stderr, "ds4: GLM5 routed layer %u failed at shared SwiGLU rank=%u\n", il, ctx->tp_rank); goto routed_one_done; }
    if (ok && !prepared && !shared_route_overlap) ok = ds4_gpu_matmul_q8_0_kslice_tensor(
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
    if (ok) ok = native ?
        ds4_gpu_add_tensor(hc_out, w->after_attention, w->down, GLM5_WIDTH) :
        ds4_gpu_hc_expand_split_tensor(hc_out, w->down, w->after_attention,
            w->ffn_split, GLM5_WIDTH, GLM5_HC);
    if (!ok) { fprintf(stderr, "ds4: GLM5 routed layer %u failed at mHC expand rank=%u\n", il, ctx->tp_rank); goto routed_one_done; }
    if (ok && !async_terminal) ok = ds4_gpu_synchronize();
    if (ok && async_terminal) {
        static int logged_async_terminal[2] = {0, 0};
        const uint32_t rank = ctx->tp_rank < 2u ? ctx->tp_rank : 0u;
        if (!logged_async_terminal[rank]) {
            fprintf(stderr,
                    "ds4: GLM5 routed decode terminal layer fence "
                    "omitted with resident Q4_K rank=%u\n", ctx->tp_rank);
            logged_async_terminal[rank] = 1;
        }
    }
    if (!ok) fprintf(stderr, "ds4: GLM5 routed layer %u failed at synchronize rank=%u\n", il, ctx->tp_rank);
    /* The packed slices are layer-scoped.  The final synchronize above makes
     * it safe to release them before the next layer is declared. */
routed_one_done:
    if (!ok) ds4_gpu_synchronize();
    const char *persist_windows = getenv("DS4_ROCM_GLM5_WINDOW_PERSIST");
    const char *persist_prefill =
        getenv("DS4_ROCM_GLM5_WINDOW_PERSIST_PREFILL");
    const bool persist_this_call =
        (persist_windows && strcmp(persist_windows, "1") == 0) ||
        (!w->decode_phase && persist_prefill &&
         strcmp(persist_prefill, "1") == 0);
    const char *scratch_windows = getenv("DS4_ROCM_GLM5_WINDOW_SCRATCH");
    const bool use_scratch = native || (w->decode_phase && scratch_windows &&
        strcmp(scratch_windows, "1") == 0);
    if (!mixed_q2 && q4_residency == 0 &&
        !persist_this_call &&
        !use_scratch)
        ds4_gpu_q4k_packed_slice_release_all();
    (void)q8_down_row_bytes;
    return ok;
}

static int routed_ffn_one(const ds4_glm5_next_exec_ctx *ctx,
                          uint32_t il, uint32_t token_ordinal,
                          ds4_glm5_next_workspace *w, ds4_gpu_tensor *hc_out) {
    return routed_ffn_one_impl(ctx, il, token_ordinal, w, hc_out, NULL, 0u);
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
    int shared_projection_fused = 0;
    /* The paired WMMA path owns both Q8 projections and the SwiGLU
     * epilogue.  Keep it explicitly opt-in: unsupported shapes and mixed
     * quantization remain on the established pair-GEMM route. */
    const char *shared_wmma_env =
        getenv("DS4_ROCM_SHARED_GU_WMMA_BATCH");
    const int shared_wmma_batch =
        shared_wmma_env && strcmp(shared_wmma_env, "1") == 0;
    if (ok && !routed_mid_is_f16 && shared_wmma_batch && n_tokens >= 64u) {
        shared_projection_ok = ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
            w->shared_gate, w->shared_up, w->shared_mid,
            ctx->model_map, ctx->model_size,
            f->gate_shexp + (uint64_t)rank_mid_base * q8_gate_row_bytes,
            f->up_shexp + (uint64_t)rank_mid_base * q8_gate_row_bytes,
            GLM5_WIDTH, GLM5_RANK_MID, w->ffn_hidden, n_tokens, 10.0f);
        shared_projection_fused = shared_projection_ok;
        if (shared_projection_fused) {
            static int logged_shared_wmma[2] = {0, 0};
            const uint32_t rank = ctx->tp_rank < 2u ? ctx->tp_rank : 0u;
            if (!logged_shared_wmma[rank]) {
                fprintf(stderr,
                        "ds4: GLM5 shared gate/up paired WMMA batch engaged "
                        "rank=%u tokens=%u tile=%s\n",
                        ctx->tp_rank, n_tokens,
                        getenv("DS4_ROCM_SHARED_GU_WMMA_BATCH_TILE") &&
                        strcmp(getenv("DS4_ROCM_SHARED_GU_WMMA_BATCH_TILE"),
                               "256") == 0 ? "256" : "128");
                logged_shared_wmma[rank] = 1;
            }
        }
    } else if (ok && !routed_mid_is_f16 &&
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
        (shared_projection_fused || ds4_gpu_swiglu_tensor(
            w->shared_mid, w->shared_gate, w->shared_up,
            n_tokens * GLM5_RANK_MID, 10.0f, 1.0f));
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
    const bool phase_profile = getenv("DS4_GLM5_PHASE_PROFILE") != NULL;
    const double phase_t0 = phase_profile ? glm5_exec_now_sec() : 0.0;
    int ok = kda_attention_rows(ctx, il, state, w, hc_in, n_tokens);
    const double phase_t1 = phase_profile ? glm5_exec_now_sec() : 0.0;
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
    if (phase_profile) {
        fprintf(stderr,
                "ds4: GLM5 phase layer=%u rows=%u attention_ms=%.3f ffn_ms=%.3f ok=%d\n",
                il, n_tokens, (phase_t1 - phase_t0) * 1000.0,
                (glm5_exec_now_sec() - phase_t1) * 1000.0, ok ? 1 : 0);
    }
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
            ctx, il, mla, w, hc_in, visible, tail_slot,
            pool_index, publish_pool) :
        mla_sparse_selection_attention(
            ctx, il, mla, w, hc_in, tail_slot,
            pool_index, publish_pool, DS4_GLM5_NEXT_INDEX_TOP_K,
            ctx->tp_big_out, true, true);
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
    const int phase_profile = getenv("DS4_GLM5_PHASE_PROFILE") != NULL;
    const double phase_t0 = phase_profile ? glm5_exec_now_sec() : 0.0;
    const int attn_ok = mla_dense_selection_attention_rows(
            ctx, il, state, w, hc_in, n_tokens);
    const double phase_t1 = phase_profile ? glm5_exec_now_sec() : 0.0;
    const int ok = attn_ok &&
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
    if (phase_profile) {
        fprintf(stderr,
                "ds4: GLM5 phase mla_layer=%u rows=%u token=%u "
                "attention_ms=%.3f ffn_ms=%.3f attn_ok=%d ok=%d\n",
                il, n_tokens, token_ordinal,
                (phase_t1 - phase_t0) * 1000.0,
                (glm5_exec_now_sec() - phase_t1) * 1000.0,
                attn_ok ? 1 : 0, ok ? 1 : 0);
    }
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

static int verify_layer_valid(const ds4_glm5_next_exec_ctx *ctx, uint32_t il,
                               const ds4_glm5_next_state *state) {
    return context_valid(ctx) && state && state->valid && ctx->tp_rank < 2u &&
        il < ctx->model->trunk_count && state->layer_count == ctx->model->trunk_count &&
        state->kda.layer && state->kda.layer_count == ctx->model->trunk_count;
}

static int verify_kda_layout(const ds4_glm5_next_exec_ctx *ctx, uint32_t il) {
    const uint32_t features = ctx->tp ? ds4_tp_runtime_features(ctx->tp) : 0u;
    const uint32_t required = DS4_TP_FEATURE_GLM5_KDA_TP |
        DS4_TP_FEATURE_GLM5_KDA_OUTPUT_ROWSLICE;
    const ds4_glm5_kda_weight_offsets *k = &ctx->model->layer[il].kda;
    const char *exact = getenv("DS4_ROCM_GLM5_BF16_SMALL_M_EXACT");
    return (features & required) == required &&
        !(features & DS4_TP_FEATURE_GLM5_KDA_OUTPUT_KSLICE) &&
        exact && strcmp(exact, "1") == 0 && k->output_type == 30u &&
        k->q_type == 30u && k->k_type == 30u && k->v_type == 30u &&
        k->f_a_type == 30u && k->f_b_type == 30u &&
        k->g_a_type == 30u && k->g_b_type == 30u && k->beta_type == 30u;
}

int ds4_glm5_next_layer_verify_reserve(const ds4_glm5_next_exec_ctx *ctx,
                                       uint32_t il,
                                       ds4_glm5_next_state *state,
                                       uint32_t capacity) {
    if (!verify_layer_valid(ctx, il, state) || state->verification.tokens ||
        state->kda.pending_verifications ||
        state->pending_mla_verifications ||
        !ds4_tp_glm5_native_width_valid(capacity)) return 0;
    if (ctx->model->layer[il].attention == DS4_GLM5_NEXT_ATTN_KDA)
        return verify_kda_layout(ctx, il) &&
            ds4_glm5_kda_replay_reserve(&state->kda.layer[il], capacity, ctx->tp_rank);
    return ds4_glm5_next_mla_replay_reserve(&state->mla[il], capacity);
}

static int verify_kda_attention(const ds4_glm5_next_exec_ctx *ctx, uint32_t il,
                                ds4_glm5_next_state *state,
                                ds4_glm5_next_workspace *w,
                                const ds4_gpu_tensor *hc_in, uint32_t n_tokens) {
    const ds4_glm5_kda_weight_offsets *weights = &ctx->model->layer[il].kda;
    const uint64_t half_row = (uint64_t)(GLM5_WIDTH / 2u) * sizeof(float);
    const uint64_t weight_half =
        (uint64_t)(GLM5_WIDTH / 2u) * DS4_GLM5_KDA_CHANNELS * sizeof(uint16_t);
    if (!kda_prefix_rows(ctx, il, w, hc_in, n_tokens, true) ||
        !ds4_glm5_kda_verify_begin(&state->kda.layer[il], &w->kda, weights,
            ctx->model_map, ctx->model_size, w->collapsed, ctx->tp_big_out,
            n_tokens, ctx->model->rms_norm_eps) ||
        !tp_exchange_rows(ctx, il, DS4_TP_GATE_ATTN, n_tokens)) return 0;
    const ds4_gpu_tensor *rank0 = ctx->tp_rank == 0u ? ctx->tp_big_out : ctx->tp_big_in;
    const ds4_gpu_tensor *rank1 = ctx->tp_rank == 0u ? ctx->tp_big_in : ctx->tp_big_out;
    if (!ds4_glm5_kda_compose_head_halves(w->kda.recurrent_out, rank0, rank1, n_tokens) ||
        !ds4_gpu_matmul_bf16_tensor(w->down, ctx->model_map, ctx->model_size,
            weights->output + ctx->tp_rank * weight_half,
            DS4_GLM5_KDA_CHANNELS, GLM5_WIDTH / 2u, w->kda.recurrent_out, n_tokens) ||
        !ds4_gpu_tensor_copy(ctx->tp_big_out, 0u, w->down, 0u, n_tokens * half_row) ||
        !tp_exchange_bytes(ctx, il, DS4_TP_GATE_ATTN, n_tokens * half_row)) return 0;
    /* The latency-QP auxiliary receive belongs to one preceding scalar gate.
     * Batch halves use one bulk exchange instead; unpack into full token rows
     * without overwriting subsequent packed local rows. */
    for (uint32_t t = 0; t < n_tokens; ++t) {
        const uint64_t base = (uint64_t)t * half_row * 2u;
        if (!ds4_gpu_tensor_copy(w->attention, base + ctx->tp_rank * half_row,
                w->down, t * half_row, half_row) ||
            !ds4_gpu_tensor_copy(w->attention, base + (1u - ctx->tp_rank) * half_row,
                ctx->tp_big_in, t * half_row, half_row)) return 0;
    }
    const int ok = ds4_gpu_hc_expand_split_tensor(w->after_attention, w->attention,
        hc_in, w->hc_split, GLM5_WIDTH, GLM5_HC);
    static uint32_t reported_rows[2];
    if (ok && !(reported_rows[ctx->tp_rank] & (1u << n_tokens))) {
        reported_rows[ctx->tp_rank] |= 1u << n_tokens;
        fprintf(stderr, "ds4: GLM5 native KDA verifier batch engaged rank=%u rows=%u qkv=bf16-small-m-exact\n",
            ctx->tp_rank, n_tokens);
    }
    return ok;
}

static int verify_dense_ffn(const ds4_glm5_next_exec_ctx *ctx, uint32_t il,
                            ds4_glm5_next_workspace *batch_w,
                            ds4_glm5_next_workspace *scalar_w,
                            ds4_gpu_tensor *hc_out, uint32_t tokens) {
    const uint64_t hc_row = (uint64_t)GLM5_HC_WIDTH * sizeof(float);
    const uint64_t hidden_row = (uint64_t)GLM5_WIDTH * sizeof(float);
    const uint64_t split_row = (uint64_t)GLM5_HC_MIX * sizeof(float);
    const ds4_glm5_next_ffn_offsets *f = &ctx->model->layer[il].ffn_weight;
    /* mHC's M1 reduction order is load-bearing. Batch only after scalar
     * preparation; all destinations are existing owned activation scratch. */
    for (uint32_t t = 0; t < tokens; ++t) {
        if (!ds4_gpu_tensor_copy(scalar_w->after_attention, 0u,
                batch_w->after_attention, t * hc_row, hc_row) ||
            !dense_ffn_prefix(ctx, il, scalar_w, 1u, 0) ||
            !ds4_gpu_tensor_copy(batch_w->ffn_hidden, t * hidden_row,
                scalar_w->ffn_hidden, 0u, hidden_row) ||
            !ds4_gpu_tensor_copy(batch_w->ffn_split, t * split_row,
                scalar_w->ffn_split, 0u, split_row)) return 0;
    }
    return ds4_rocm_glm5_dense_q8_small_m(batch_w->gate, batch_w->up,
            ctx->model_map, ctx->model_size, f->gate, f->up,
            GLM5_WIDTH, GLM5_DENSE_MID, batch_w->ffn_hidden, tokens) &&
        ds4_gpu_swiglu_tensor(batch_w->mid, batch_w->gate, batch_w->up,
            tokens * GLM5_DENSE_MID, 10.0f, 1.0f) &&
        ds4_rocm_glm5_dense_q8_small_m(batch_w->down, NULL,
            ctx->model_map, ctx->model_size, f->down, 0u,
            GLM5_DENSE_MID, GLM5_WIDTH, batch_w->mid, tokens) &&
        ds4_gpu_hc_expand_split_tensor(hc_out, batch_w->down,
            batch_w->after_attention, batch_w->ffn_split, GLM5_WIDTH, GLM5_HC);
}

static int verify_shared_ffn_prepare(const ds4_glm5_next_exec_ctx *ctx,
                                     uint32_t il,
                                     ds4_glm5_next_workspace *batch_w,
                                     ds4_glm5_next_workspace *scalar_w,
                                     uint32_t tokens) {
    const uint64_t hc_row = (uint64_t)GLM5_HC_WIDTH * sizeof(float);
    const uint64_t hidden_row = (uint64_t)GLM5_WIDTH * sizeof(float);
    const uint64_t split_row = (uint64_t)GLM5_HC_MIX * sizeof(float);
    const ds4_glm5_next_ffn_offsets *f = &ctx->model->layer[il].ffn_weight;
    const uint64_t gate_half = (uint64_t)ctx->tp_rank * GLM5_RANK_MID * 4352u;
    /* Same M1 prefix/reduction as ordinary decode. Existing batch activation
     * scratch retains the only operands needed by routed_ffn_one_impl. */
    for (uint32_t t = 0u; t < tokens; ++t) {
        if (!ds4_gpu_tensor_copy(scalar_w->after_attention, 0u,
                batch_w->after_attention, t * hc_row, hc_row) ||
            !routed_ffn_prefix(ctx, il, scalar_w) ||
            !ds4_gpu_tensor_copy(batch_w->ffn_hidden, t * hidden_row,
                scalar_w->ffn_hidden, 0u, hidden_row) ||
            !ds4_gpu_tensor_copy(batch_w->ffn_split, t * split_row,
                scalar_w->ffn_split, 0u, split_row)) return 0;
    }
    return ds4_rocm_glm5_shared_q8_small_m(batch_w->shared_gate, batch_w->shared_up,
            ctx->model_map, ctx->model_size, f->gate_shexp + gate_half,
            f->up_shexp + gate_half, GLM5_WIDTH, GLM5_RANK_MID, 4352u, 0u,
            batch_w->ffn_hidden, tokens) &&
        ds4_gpu_swiglu_tensor(batch_w->shared_mid, batch_w->shared_gate,
            batch_w->shared_up, tokens * GLM5_RANK_MID, 10.0f, 1.0f) &&
        ds4_rocm_glm5_shared_q8_small_m(batch_w->shared_out, NULL,
            ctx->model_map, ctx->model_size, f->down_shexp, 0u,
            GLM5_RANK_MID, GLM5_WIDTH, 2176u, ctx->tp_rank * GLM5_RANK_MID,
            batch_w->shared_mid, tokens);
}

/* All attention/prefix/shared rows have already been prepared. Retain the M1
 * router and packed expert dispatch: only their route and output handoffs are
 * batched. No prefill GEMM, alternate activation codec or route reordering. */
static int expert_six_admit(const ds4_glm5_next_exec_ctx *ctx, uint32_t il,
        const ds4_glm5_next_workspace *batch, ds4_glm5_expert_six_plan *plan) {
    if (!batch || !batch->decode_phase || batch->draft_only || batch->capacity_tokens != 6u ||
        !ctx->model->layer[il].is_trunk || local_q4k_half_residency(ctx, &ctx->model->layer[il]) != 1)
        return 0;
    const ds4_glm5_next_ffn_offsets *f = &ctx->model->layer[il].ffn_weight;
    const ds4_glm5_expert_six_args a = {
        .out = batch->routed_out, .mid = batch->routed_mid,
        .input_q8 = batch->routed_experts, .mid_q8 = batch->routed_gate,
        .descriptors = batch->routed_up, .input = batch->ffn_hidden,
        .selected = batch->router_selected, .weights = batch->router_weights,
        .model_map = ctx->model_map, .model_size = ctx->model_size,
        .gate_offset = f->gate_exps, .up_offset = f->up_exps, .down_offset = f->down_exps,
        .rank = ctx->tp_rank, .rows = 6u
    };
    return ds4_rocm_glm5_expert_six_admit(plan, &a);
}

int ds4_glm5_next_expert_pairs_supported(const ds4_glm5_next_exec_ctx *ctx,
        const ds4_glm5_next_workspace *batch) {
    if (!context_valid(ctx) || !batch || ctx->tp_rank > 1u) return 0;
    unsigned count = 0;
    for (uint32_t il = 0; il < ctx->model->trunk_count; ++il) {
        if (ctx->model->layer[il].ffn != DS4_GLM5_NEXT_FFN_ROUTED) continue;
        ds4_glm5_expert_six_plan plan;
        if (!expert_six_admit(ctx, il, batch, &plan)) return 0;
        ++count;
    }
    return count == 42u;
}

static int verify_ffn_handoff(const ds4_glm5_next_exec_ctx *ctx, uint32_t il,
                             ds4_glm5_next_workspace *batch,
                             ds4_glm5_next_workspace *scalar,
                             ds4_gpu_tensor *hc_out, uint32_t frontier,
                             uint32_t rows, int ok) {
    const uint64_t hidden_row = (uint64_t)GLM5_WIDTH * sizeof(float);
    const uint64_t route_row = GLM5_EXPERTS_USED * sizeof(uint32_t);
    const uint64_t hc_row = (uint64_t)GLM5_HC_WIDTH * sizeof(float);
    const uint64_t split_row = (uint64_t)GLM5_HC_MIX * sizeof(float);
    const uint64_t q4_gate_row = (GLM5_WIDTH / GLM5_Q4K_QK) * GLM5_Q4K_BLOCK_BYTES;
    const uint64_t q4_down_row = (GLM5_ROUTED_MID / GLM5_Q4K_QK) * GLM5_Q4K_BLOCK_BYTES;
    const uint64_t q4_down_half = (GLM5_RANK_MID / GLM5_Q4K_QK) * GLM5_Q4K_BLOCK_BYTES;
    const ds4_glm5_next_ffn_offsets *f = &ctx->model->layer[il].ffn_weight;
    const uint64_t sequence = *ctx->tp_sequence;
    const bool queue_experts = (ds4_tp_prefill_config(ctx->tp) &
        DS4_TP_CONFIG_GLM5_VERIFY_FFN_QUEUE) != 0u;
    const uint32_t pair_mode = ds4_tp_glm5_expert_pairs_mode(ds4_tp_prefill_config(ctx->tp));
    const bool six = pair_mode && rows == 6u;
    ds4_glm5_expert_six_plan plan = {0};
    ds4_glm5_expert_groups groups = {0};
    const bool profile = getenv("DS4_GLM5_VERIFY_PROFILE") != NULL;
    const char *routes_env = getenv("DS4_GLM5_VERIFY_ROUTE_PROFILE");
    const bool route_profile = routes_env && strcmp(routes_env, "1") == 0;
    const double begin = profile ? glm5_exec_now_sec() : 0.0;
    int32_t ids[8u * GLM5_EXPERTS_USED] = {0};
    float weights[8u * GLM5_EXPERTS_USED] = {0};
    uint64_t hash = UINT64_C(1469598103934665603);
    char error[128] = {0};
    ok = ok && shared_route_overlap_mode(ctx, scalar, 1) >= 0;
    for (uint32_t t = 0; ok && t < rows; ++t) {
        ok = ds4_gpu_tensor_copy(scalar->ffn_hidden, 0, batch->ffn_hidden,
                t * hidden_row, hidden_row) &&
            ds4_gpu_matmul_f32_tensor(scalar->router_logits, ctx->model_map,
                ctx->model_size, f->gate_inp, GLM5_WIDTH, GLM5_EXPERTS,
                scalar->ffn_hidden, 1u) &&
            ds4_gpu_glm_router_select_tensor(scalar->router_selected,
                scalar->router_weights, scalar->router_probs, ctx->model_map,
                ctx->model_size, f->exp_probs_b, scalar->router_logits,
                GLM5_EXPERTS, GLM5_EXPERTS_USED, 2.5f) &&
            ds4_gpu_tensor_copy(batch->router_selected, t * route_row,
                scalar->router_selected, 0, route_row) &&
            ds4_gpu_tensor_copy(batch->router_weights, t * route_row,
                scalar->router_weights, 0, route_row);
    }
    if (ok) ok = ds4_gpu_tensor_read(batch->router_selected, 0, ids, rows * route_row) &&
        ds4_gpu_tensor_read(batch->router_weights, 0, weights, rows * route_row);
    for (uint32_t t = 0; ok && t < rows; ++t)
        for (uint32_t i = 0; ok && i < GLM5_EXPERTS_USED; ++i) {
            const uint32_t index = t * GLM5_EXPERTS_USED + i;
            ok = ids[index] >= 0 && ids[index] < GLM5_EXPERTS &&
                isfinite(weights[index]) && weights[index] >= 0.0f;
            for (uint32_t j = 0; ok && j < i; ++j)
                ok = ids[index] != ids[t * GLM5_EXPERTS_USED + j];
        }
    if (ok) {
        hash = fnv64_continue(hash, ids, rows * route_row);
        hash = fnv64_continue(hash, weights, rows * route_row);
    }
    if (ok && six) ok = expert_six_admit(ctx, il, batch, &plan) &&
        ds4_glm5_expert_groups_build(&groups, ids, weights, pair_mode);
    const double routes_done = profile ? glm5_exec_now_sec() : 0.0;
    if (!ds4_tp_verify_layer_agree(ctx->tp, sequence, il, frontier, rows, 0u,
            hash, ok, error, sizeof(error))) goto failed;
    const double route_agree_done = profile ? glm5_exec_now_sec() : 0.0;

    /* The completed router readbacks above also finish earlier stream0 MLA
     * consumers of routed_experts. It can now hold input Q8_K safely. */
    if (six) ok = ds4_rocm_glm5_expert_six_begin(&plan, &groups, ids, weights);
    for (uint32_t t = 0; ok && t < rows; ++t) {
        ds4_gpu_tensor *local = ds4_gpu_tensor_view(ctx->tp_big_out,
            t * hidden_row, hidden_row);
        if (six) {
            ds4_gpu_tensor *routed = ds4_gpu_tensor_view(batch->routed_out,
                t * hidden_row, hidden_row);
            ds4_gpu_tensor *shared = ds4_gpu_tensor_view(batch->shared_out,
                t * hidden_row, hidden_row);
            ok = local && routed && shared &&
                ds4_rocm_glm5_expert_six_down_row(&plan, t) &&
                ds4_gpu_add_tensor(local, routed, shared, GLM5_WIDTH) &&
                (queue_experts || ds4_gpu_synchronize());
            ds4_gpu_tensor_free(routed); ds4_gpu_tensor_free(shared);
            ds4_gpu_tensor_free(local);
            continue;
        }
        ok = local && ds4_gpu_tensor_copy(scalar->ffn_hidden, 0,
                batch->ffn_hidden, t * hidden_row, hidden_row) &&
            ds4_gpu_tensor_copy(scalar->router_selected, 0,
                batch->router_selected, t * route_row, route_row) &&
            ds4_gpu_tensor_copy(scalar->router_weights, 0,
                batch->router_weights, t * route_row, route_row) &&
            ds4_gpu_tensor_copy(scalar->shared_out, 0,
                batch->shared_out, t * hidden_row, hidden_row) &&
            ds4_gpu_routed_moe_one_packed_q4k_tensor(scalar->routed_out,
                scalar->routed_gate, scalar->routed_up, scalar->routed_mid,
                scalar->routed_experts, ctx->model_map, ctx->model_size,
                f->gate_exps, f->up_exps, f->down_exps, GLM5_EXPERTS,
                q4_gate_row, q4_down_row, ctx->tp_rank * GLM5_RANK_MID, GLM5_RANK_MID,
                ctx->tp_rank * q4_down_half, q4_down_half, scalar->router_selected,
                scalar->router_weights, GLM5_EXPERTS_USED, 10.0f,
                scalar->ffn_hidden, NULL, il) &&
            ds4_gpu_add_tensor(local, scalar->routed_out,
                scalar->shared_out, GLM5_WIDTH) &&
            (queue_experts || ds4_gpu_synchronize());
        ds4_gpu_tensor_free(local);
    }
    if (queue_experts || (six && !ok)) {
        /* All copies, packed kernels and adds use stream0. The row view owns
         * no storage. Drain even on an enqueue failure, and report completion
         * status before phase1 agreement; the later payload fence is too late. */
        const int completed = ds4_gpu_synchronize();
        ok = ok && completed;
    }
    const double experts_done = profile ? glm5_exec_now_sec() : 0.0;
    /* Failure is always exchanged before either side posts the bulk payload. */
    if (!ds4_tp_verify_layer_agree(ctx->tp, sequence, il, frontier, rows, 1u,
            hash, ok, error, sizeof(error))) goto failed;
    const double compute_agree_done = profile ? glm5_exec_now_sec() : 0.0;
    if (!tp_exchange_rows(ctx, il, DS4_TP_GATE_FFN, rows)) goto failed;
    const double bulk_done = profile ? glm5_exec_now_sec() : 0.0;
    for (uint32_t t = 0; ok && t < rows; ++t) {
        ds4_gpu_tensor *local = ds4_gpu_tensor_view(ctx->tp_big_out,
            t * hidden_row, hidden_row);
        ds4_gpu_tensor *remote = ds4_gpu_tensor_view(ctx->tp_big_in,
            t * hidden_row, hidden_row);
        ds4_gpu_tensor *out = ds4_gpu_tensor_view(hc_out, t * hc_row, hc_row);
        ok = local && remote && out &&
            ds4_gpu_tensor_copy(scalar->after_attention, 0,
                batch->after_attention, t * hc_row, hc_row) &&
            ds4_gpu_tensor_copy(scalar->ffn_split, 0,
                batch->ffn_split, t * split_row, split_row) &&
            ds4_gpu_add_tensor(scalar->down, local, remote, GLM5_WIDTH) &&
            ds4_gpu_hc_expand_split_tensor(out, scalar->down,
                scalar->after_attention, scalar->ffn_split, GLM5_WIDTH, GLM5_HC);
        ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(remote); ds4_gpu_tensor_free(local);
    }
    if (ok) ok = ds4_gpu_synchronize();
    if (!ok) goto failed;
    if (pair_mode) {
        const unsigned kind = ctx->model->layer[il].attention == DS4_GLM5_NEXT_ATTN_MLA;
        batch->expert_pair_mode = pair_mode; batch->expert_pair_rank = ctx->tp_rank;
        if (six) {
            ++batch->expert_six_calls[kind];
            batch->expert_pair_reads[kind] += groups.doubles;
        } else ++batch->expert_tail_calls[kind];
    }
    const double done = profile ? glm5_exec_now_sec() : 0.0;
    if (profile) fprintf(stderr,
        "VERIFY_FFN rank=%u layer=%u frontier=%u m=%u sequence=%llu queue=%u route_ms=%.6f route_agree_ms=%.6f expert_ms=%.6f compute_agree_ms=%.6f bulk_ms=%.6f tail_ms=%.6f total_ms=%.6f\n",
        ctx->tp_rank, il, frontier, rows, (unsigned long long)sequence, queue_experts,
        (routes_done - begin) * 1000.0, (route_agree_done - routes_done) * 1000.0,
        (experts_done - route_agree_done) * 1000.0,
        (compute_agree_done - experts_done) * 1000.0,
        (bulk_done - compute_agree_done) * 1000.0, (done - bulk_done) * 1000.0,
        (done - begin) * 1000.0);
    if (route_profile) {
        ds4_glm5_route_profile p = {0};
        const int valid = ds4_glm5_route_profile_count(ids, rows, &p);
        unsigned zero_weights = 0;
        for (unsigned i = 0; i < rows * GLM5_EXPERTS_USED; ++i)
            zero_weights += weights[i] == 0.0f;
        fprintf(stderr,
            "VERIFY_ROUTES rank=%u layer=%u frontier=%u m=%u sequence=%llu hash=%016llx valid=%d unique2=%u unique4=%u unique_all=%u h1=%u h2=%u h3=%u h4=%u h5=%u h6=%u h7=%u h8=%u zero_weights=%u\n",
            ctx->tp_rank, il, frontier, rows, (unsigned long long)sequence,
            (unsigned long long)hash, valid, p.unique2, p.unique4, p.unique_all,
            p.multiplicity[0], p.multiplicity[1], p.multiplicity[2], p.multiplicity[3],
            p.multiplicity[4], p.multiplicity[5], p.multiplicity[6], p.multiplicity[7], zero_weights);
    }
    return 1;
failed:
    ds4_gpu_synchronize();
    ds4_tp_mark_failed(ctx->tp);
    fprintf(stderr, "ds4: native FFN handoff failed rank=%u layer=%u: %s\n",
        ctx->tp_rank, il, error[0] ? error : "local operation or payload exchange");
    return 0;
}

static int layer_verify_run(const ds4_glm5_next_exec_ctx *ctx,
                               uint32_t il, ds4_glm5_next_state *state,
                               ds4_glm5_next_workspace *batch_w,
                               ds4_glm5_next_workspace *scalar_w,
                               const ds4_gpu_tensor *hc_in,
                               ds4_gpu_tensor *hc_out, uint32_t n_tokens) {
    const uint64_t row = (uint64_t)GLM5_HC_WIDTH * sizeof(float);
    if (!verify_layer_valid(ctx, il, state) || !batch_w || !scalar_w ||
        batch_w == scalar_w || !scalar_w->decode_phase ||
        batch_w->capacity_tokens != n_tokens || scalar_w->capacity_tokens != 1u ||
        !ds4_tp_glm5_native_width_valid(n_tokens) ||
        !hc_in || !hc_out || hc_in == hc_out || ctx->trace_prefix ||
        ds4_gpu_tensor_bytes(hc_in) != n_tokens * row ||
        ds4_gpu_tensor_bytes(hc_out) != n_tokens * row ||
        !tp_context_valid_bytes(ctx, (uint64_t)n_tokens * GLM5_WIDTH * sizeof(float)))
        return 0;
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    const bool is_kda = layer->attention == DS4_GLM5_NEXT_ATTN_KDA;
    const char *mla_profile_option = getenv("DS4_GLM5_VERIFY_MLA_PROFILE");
    if (mla_profile_option && strcmp(mla_profile_option, "0") &&
        strcmp(mla_profile_option, "1")) {
        fprintf(stderr, "ds4: invalid GLM5 MLA verification profile selector\n");
        return 0;
    }
    glm5_mla_profile mla_timings = {0};
    glm5_mla_profile *mla_profile = (!is_kda && mla_profile_option &&
        strcmp(mla_profile_option, "1") == 0) ? &mla_timings : NULL;
    const char *dense_option = getenv("DS4_ROCM_GLM5_VERIFY_DENSE_Q8");
    if (dense_option && strcmp(dense_option, "0") != 0 && strcmp(dense_option, "1") != 0) {
        fprintf(stderr, "ds4: invalid GLM5 dense Q8 verification selector\n");
        return 0;
    }
    const bool batch_dense = is_kda && layer->ffn == DS4_GLM5_NEXT_FFN_DENSE &&
        dense_option && strcmp(dense_option, "1") == 0;
    const char *shared_option = getenv("DS4_ROCM_GLM5_VERIFY_SHARED_Q8");
    if (shared_option && strcmp(shared_option, "0") != 0 && strcmp(shared_option, "1") != 0) {
        fprintf(stderr, "ds4: invalid GLM5 shared Q8 verification selector\n");
        return 0;
    }
    const char *mla_option = getenv("DS4_ROCM_GLM5_VERIFY_MLA_FFN_HANDOFF");
    const bool mla_requested = mla_option && strcmp(mla_option, "1") == 0;
    const uint64_t config = ds4_tp_prefill_config(ctx->tp);
    const uint32_t pair_mode = ds4_tp_glm5_expert_pairs_parse(
        getenv("DS4_ROCM_GLM5_VERIFY_EXPERT_PAIRS"));
    if (pair_mode != ds4_tp_glm5_expert_pairs_mode(config) ||
        !ds4_tp_glm5_expert_pairs_config_valid(config) ||
        (pair_mode && (n_tokens > 6u || (n_tokens != 2u && n_tokens != 4u && n_tokens != 6u)))) {
        ds4_tp_mark_failed(ctx->tp);
        fprintf(stderr, "ds4: native expert pair selector/hello/width mismatch\n");
        return 0;
    }
    const char *queue_option = getenv("DS4_ROCM_GLM5_VERIFY_FFN_QUEUE");
    const bool queue_requested = queue_option && strcmp(queue_option, "1") == 0;
    if ((queue_option && strcmp(queue_option, "0") && strcmp(queue_option, "1")) ||
        queue_requested != ((config & DS4_TP_CONFIG_GLM5_VERIFY_FFN_QUEUE) != 0u) ||
        !ds4_tp_glm5_ffn_queue_config_valid(config)) {
        fprintf(stderr, "ds4: native FFN queue selector/hello mismatch or missing prerequisites\n");
        return 0;
    }
    if ((mla_option && strcmp(mla_option, "0") && strcmp(mla_option, "1")) ||
        mla_requested != ((config & DS4_TP_CONFIG_GLM5_VERIFY_MLA_FFN_HANDOFF) != 0u) ||
        !ds4_tp_glm5_mla_handoff_config_valid(config)) {
        fprintf(stderr, "ds4: native MLA FFN verifier selector/hello mismatch or missing prerequisites\n");
        return 0;
    }
    const bool batch_mla = !is_kda && mla_requested;
    const bool batch_shared = (is_kda || batch_mla) && layer->ffn == DS4_GLM5_NEXT_FFN_ROUTED &&
        shared_option && strcmp(shared_option, "1") == 0;
    const char *handoff_option = getenv("DS4_ROCM_GLM5_VERIFY_FFN_HANDOFF");
    if (handoff_option && strcmp(handoff_option, "0") && strcmp(handoff_option, "1"))
        return 0;
    const bool handoff_requested = handoff_option && strcmp(handoff_option, "1") == 0;
    if (handoff_requested != ((ds4_tp_prefill_config(ctx->tp) &
            DS4_TP_CONFIG_GLM5_VERIFY_FFN_HANDOFF) != 0u) ||
        (handoff_requested && (!shared_option || strcmp(shared_option, "1")))) return 0;
    const bool batch_handoff = handoff_requested && batch_shared;
    if (batch_mla && !batch_handoff) return 0;
    if (pair_mode && layer->ffn == DS4_GLM5_NEXT_FFN_ROUTED) {
        ds4_glm5_expert_six_plan plan;
        if (!batch_handoff || (n_tokens == 6u && !expert_six_admit(ctx, il, batch_w, &plan))) {
            ds4_tp_mark_failed(ctx->tp);
            fprintf(stderr, "ds4: native expert pair admission failed\n");
            return 0;
        }
    }
    const char *attn_option = getenv("DS4_ROCM_GLM5_VERIFY_MLA_ATTN_HANDOFF");
    const bool attn_requested = attn_option && !strcmp(attn_option, "1");
    const char *row_sync_option = getenv("DS4_GLM5_VERIFY_MLA_ROW_SYNC");
    const bool row_sync = row_sync_option && !strcmp(row_sync_option, "1");
    const char *owned = getenv("DS4_GLM5_MLA_OWNED_HEADS");
    if ((attn_option && strcmp(attn_option, "0") && strcmp(attn_option, "1")) ||
        attn_requested != ((config & DS4_TP_CONFIG_GLM5_VERIFY_MLA_ATTN_HANDOFF) != 0u) ||
        !ds4_tp_glm5_mla_attn_handoff_config_valid(config) ||
        (attn_requested && (!owned || strcmp(owned, "1") || n_tokens > 6u ||
            n_tokens > ds4_tp_glm5_native_rows(config) || (!is_kda && !batch_handoff))) ||
        (row_sync_option && strcmp(row_sync_option, "0") && strcmp(row_sync_option, "1")) ||
        (row_sync && !attn_requested)) {
        fprintf(stderr, "ds4: native MLA attention handoff selector/hello/layout mismatch\n");
        return 0;
    }
    const bool batch_attn = batch_mla && attn_requested;
    const char *output_option = getenv("DS4_ROCM_GLM5_VERIFY_MLA_OUTPUT_BATCH");
    const bool output_requested = output_option && !strcmp(output_option, "1");
    if ((output_option && strcmp(output_option, "0") && strcmp(output_option, "1")) ||
        (output_requested && !attn_requested)) {
        fprintf(stderr, "ds4: native MLA output batch requires attention handoff and a valid selector\n");
        return 0;
    }
    const bool batch_output = batch_attn && output_requested;
    /* Model binding already requires each MLA output to be Q8_0 K16384/N4096.
     * Check modes, resident range and scratch before starting private replay.
     * The causal loop writes packed owned32 heads, not full64-head rows. */
    if (batch_output && !mla_output_small_m(ctx, il, batch_w, n_tokens, false)) {
        ds4_tp_mark_failed(ctx->tp);
        fprintf(stderr, "ds4: native MLA output batch unsupported resident layout/settings\n");
        return 0;
    }
    if (batch_shared) {
        const char *pair = getenv("DS4_ROCM_GLM5_SHARED_Q8_PAIR_DECODE");
        if (!layer->is_trunk || scalar_w->draft_only ||
            local_q4k_half_residency(ctx, layer) != 1 ||
            (pair && strcmp(pair, "0") != 0)) {
            fprintf(stderr, "ds4: GLM5 shared Q8 verification unsupported layout/settings\n");
            return 0;
        }
    }
    const uint64_t frontier = is_kda ? state->kda.layer[il].token_count :
        state->mla[il].token_count;
    if (frontier > state->context_capacity ||
        n_tokens > state->context_capacity - frontier ||
        (is_kda && (!verify_kda_layout(ctx, il) ||
                    !ds4_glm5_kda_verify_ready(&state->kda.layer[il], n_tokens, ctx->tp_rank))) ||
        (!is_kda && (layer->ffn != DS4_GLM5_NEXT_FFN_ROUTED ||
                    !ds4_glm5_next_mla_verify_ready(&state->mla[il], n_tokens)))) return 0;
    ds4_glm5_next_exec_ctx bulk = *ctx;
    bulk.force_bulk_gates = true;
    const bool profile = getenv("DS4_GLM5_VERIFY_PROFILE") != NULL;
    double phase_start = profile ? glm5_exec_now_sec() : 0.0;
    double attention_sec = 0.0, ffn_sec = 0.0, shared_sec = 0.0, handoff_sec = 0.0;
    ds4_glm5_next_mla_state *mla = NULL;
    int ok = is_kda ? verify_kda_attention(&bulk, il, state, batch_w, hc_in, n_tokens) :
        ds4_glm5_next_mla_verify_begin(&state->mla[il], n_tokens, &mla);
    if (batch_attn) {
        ok = verify_mla_attention_handoff(&bulk, il, mla, batch_w, scalar_w,
            hc_in, (uint32_t)frontier, n_tokens, row_sync, batch_output, ok, mla_profile);
        if (!ok) {
            ds4_glm5_next_state_invalidate(state);
            return 0;
        }
        static int reported[2][2];
        if (!reported[ctx->tp_rank][batch_output]) {
            fprintf(stderr, "ds4: native MLA attention handoff active rank=%u owned_heads=32 output=%s row_sync=%u cache_bytes=0\n",
                ctx->tp_rank, batch_output ? "batch" : "M1", row_sync);
            reported[ctx->tp_rank][batch_output] = 1;
        }
    }
    /* Each layer input is already available for every verification row. Its
     * FFN does not feed the same layer's next attention row. Keep MLA append,
     * pool selection and attention arithmetic serial, then save the complete
     * residual before scalar scratch is reused by the next attention/FFN.
     * These are private replay appends; accepted-prefix publication is still
     * performed by target_verify_finish, after every layer succeeds. */
    for (uint32_t t = 0u; ok && batch_mla && !batch_attn && t < n_tokens; ++t) {
        ds4_gpu_tensor *in = ds4_gpu_tensor_view(hc_in, t * row, row);
        uint32_t slot = 0u, pool = 0u, visible = 0u;
        bool publish = false;
        ok = in && ds4_glm5_next_mla_append_plan(mla, &slot, &pool, &publish);
        const bool dense = ds4_glm5_next_mla_dense_selection_visible(
            mla->token_count, mla->capacity_tokens, &visible);
        if (ok) ok = dense ? mla_dense_selection_attention_impl(
            &bulk, il, mla, scalar_w, in, visible, slot, pool, publish, false, mla_profile) :
            mla_sparse_selection_attention_impl(&bulk, il, mla, scalar_w, in,
                slot, pool, publish, DS4_GLM5_NEXT_INDEX_TOP_K,
                bulk.tp_big_out, true, true, false, mla_profile);
        if (ok) ok = ds4_gpu_tensor_copy(batch_w->after_attention, t * row,
                scalar_w->after_attention, 0u, row) &&
            ds4_glm5_next_mla_append_commit(mla);
        ds4_gpu_tensor_free(in);
    }
    if (profile) {
        if (ok) ok = ds4_gpu_synchronize();
        attention_sec += glm5_exec_now_sec() - phase_start;
    }
    if (ok && batch_dense) {
        if (profile) phase_start = glm5_exec_now_sec();
        ok = verify_dense_ffn(&bulk, il, batch_w, scalar_w, hc_out, n_tokens);
        if (profile) {
            if (ok) ok = ds4_gpu_synchronize();
            ffn_sec += glm5_exec_now_sec() - phase_start;
        }
    }
    if (ok && batch_shared) {
        if (profile) phase_start = glm5_exec_now_sec();
        ok = verify_shared_ffn_prepare(&bulk, il, batch_w, scalar_w, n_tokens);
        if (profile) {
            if (ok) ok = ds4_gpu_synchronize();
            shared_sec = glm5_exec_now_sec() - phase_start;
            ffn_sec += shared_sec;
        }
        if (ok) {
            static int reported[2];
            if (!reported[ctx->tp_rank]) {
                fprintf(stderr, "ds4: GLM5 exact shared Q8 verifier batch active rank=%u weights=original new_weight_cache_bytes=0\n",
                        ctx->tp_rank);
                reported[ctx->tp_rank] = 1;
            }
        }
    }
    if (batch_handoff) {
        if (profile) phase_start = glm5_exec_now_sec();
        ok = verify_ffn_handoff(&bulk, il, batch_w, scalar_w, hc_out,
            (uint32_t)frontier, n_tokens, ok);
        if (profile) {
            handoff_sec = glm5_exec_now_sec() - phase_start;
            ffn_sec += handoff_sec;
        }
        if (ok) {
            static int reported[2][2];
            if (!reported[ctx->tp_rank][batch_mla]) {
                fprintf(stderr, "ds4: native %s FFN handoff batch active rank=%u arithmetic=M1 weights=original cache_bytes=0 expert_completion=%s\n",
                    batch_mla ? "MLA" : "KDA", ctx->tp_rank, queue_requested ? "layer" : "row");
                reported[ctx->tp_rank][batch_mla] = 1;
            }
        }
    }
    for (uint32_t t = 0u; ok && !batch_dense && !batch_handoff && t < n_tokens; ++t) {
        if (profile) phase_start = glm5_exec_now_sec();
        ds4_gpu_tensor *out = ds4_gpu_tensor_view(hc_out, t * row, row);
        ds4_gpu_tensor *in = is_kda ? NULL : ds4_gpu_tensor_view(hc_in, t * row, row);
        ok = out != NULL;
        if (ok && is_kda) {
            ok = ds4_gpu_tensor_copy(scalar_w->after_attention, 0u,
                batch_w->after_attention, t * row, row);
        } else if (ok) {
            uint32_t slot = 0u, pool = 0u, visible = 0u;
            bool publish = false;
            ok = in && ds4_glm5_next_mla_append_plan(mla, &slot, &pool, &publish);
            const bool dense = ds4_glm5_next_mla_dense_selection_visible(
                mla->token_count, mla->capacity_tokens, &visible);
            if (ok) ok = dense ? mla_dense_selection_attention_impl(
                &bulk, il, mla, scalar_w, in, visible, slot, pool, publish, false, mla_profile) :
                mla_sparse_selection_attention_impl(&bulk, il, mla, scalar_w, in,
                    slot, pool, publish, DS4_GLM5_NEXT_INDEX_TOP_K,
                    bulk.tp_big_out, true, true, false, mla_profile);
        }
        if (profile) {
            if (ok) ok = ds4_gpu_synchronize();
            attention_sec += glm5_exec_now_sec() - phase_start;
            phase_start = glm5_exec_now_sec();
        }
        if (ok) ok = layer->ffn == DS4_GLM5_NEXT_FFN_DENSE ?
            dense_ffn_rows(&bulk, il, scalar_w, out, 1u, 0) :
            routed_ffn_one_impl(&bulk, il, (uint32_t)frontier + t, scalar_w, out,
                batch_shared ? batch_w : NULL, t);
        if (ok && !is_kda) ok = ds4_glm5_next_mla_append_commit(mla);
        if (profile) {
            if (ok) ok = ds4_gpu_synchronize();
            ffn_sec += glm5_exec_now_sec() - phase_start;
        }
        ds4_gpu_tensor_free(in);
        ds4_gpu_tensor_free(out);
    }
    if (ok) ok = ds4_gpu_synchronize();
    if (mla_profile) {
        double total = 0.0;
        for (unsigned i = 0; i < MLA_PROFILE_COUNT; ++i)
            total += mla_timings.seconds[i];
        fprintf(stderr,
            "VERIFY_MLA rank=%u layer=%u frontier=%llu m=%u dense=%u sparse=%u ok=%d entry_ms=%.6f prepare_qkv_ms=%.6f kv_store_ms=%.6f lowrank_ms=%.6f index_state_ms=%.6f select_ms=%.6f attention_ms=%.6f output_ms=%.6f exchange_ms=%.6f residual_ms=%.6f total_ms=%.6f\n",
            ctx->tp_rank, il, (unsigned long long)frontier, n_tokens,
            mla_timings.dense_rows, mla_timings.sparse_rows, ok,
            mla_timings.seconds[MLA_PROFILE_ENTRY] * 1000.0,
            mla_timings.seconds[MLA_PROFILE_PREPARE_QKV] * 1000.0,
            mla_timings.seconds[MLA_PROFILE_KV_STORE] * 1000.0,
            mla_timings.seconds[MLA_PROFILE_LOWRANK] * 1000.0,
            mla_timings.seconds[MLA_PROFILE_INDEX_STATE] * 1000.0,
            mla_timings.seconds[MLA_PROFILE_SELECT] * 1000.0,
            mla_timings.seconds[MLA_PROFILE_ATTENTION] * 1000.0,
            mla_timings.seconds[MLA_PROFILE_OUTPUT] * 1000.0,
            mla_timings.seconds[MLA_PROFILE_EXCHANGE] * 1000.0,
            mla_timings.seconds[MLA_PROFILE_RESIDUAL] * 1000.0,
            total * 1000.0);
    }
    if (profile) fprintf(stderr,
        "VERIFY_PROFILE rank=%u layer=%u kind=%s m=%u attention_ms=%.6f ffn_ms=%.6f ok=%d frontier=%llu shared_ms=%.6f handoff_ms=%.6f\n",
        ctx->tp_rank, il, is_kda ? "kda" : "mla", n_tokens,
        attention_sec * 1000.0, ffn_sec * 1000.0, ok, (unsigned long long)frontier,
        shared_sec * 1000.0, handoff_sec * 1000.0);
    if (!ok) ds4_glm5_next_state_invalidate(state);
    return ok;
}

int ds4_glm5_next_layer_verify(const ds4_glm5_next_exec_ctx *ctx,
                               uint32_t il, ds4_glm5_next_state *state,
                               ds4_glm5_next_workspace *batch_w,
                               ds4_glm5_next_workspace *scalar_w,
                               const ds4_gpu_tensor *hc_in,
                               ds4_gpu_tensor *hc_out, uint32_t n_tokens) {
    return state && !state->verification.tokens &&
        layer_verify_run(ctx, il, state, batch_w, scalar_w, hc_in, hc_out, n_tokens);
}

int ds4_glm5_next_layer_verify_finish(const ds4_glm5_next_exec_ctx *ctx,
                                      uint32_t il, ds4_glm5_next_state *state,
                                      uint32_t accepted_inputs) {
    if (!verify_layer_valid(ctx, il, state) || state->verification.tokens) return 0;
    return ctx->model->layer[il].attention == DS4_GLM5_NEXT_ATTN_KDA ?
        ds4_glm5_kda_verify_finish(&state->kda.layer[il], accepted_inputs) :
        ds4_glm5_next_mla_verify_finish(&state->mla[il], accepted_inputs);
}

static int target_verify_ready(const ds4_glm5_next_exec_ctx *ctx,
                                 const ds4_glm5_next_state *state,
                                 uint32_t tokens) {
    if (!verify_layer_valid(ctx, 0u, state) || state->verification.tokens ||
        state->kda.pending_verifications || state->pending_mla_verifications ||
        state->kda.kda_count != 34u || state->mla_count != DS4_GLM5_NEXT_MLA_COUNT ||
        !ds4_tp_glm5_native_width_valid(tokens) || ctx->trace_prefix ||
        !tp_context_valid_bytes(ctx, (uint64_t)tokens * GLM5_WIDTH * sizeof(float)))
        return 0;
    const uint64_t frontier = state->kda.layer[0].token_count;
    if (frontier > state->context_capacity || tokens > state->context_capacity - frontier)
        return 0;
    for (uint32_t il = 0; il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il) {
        if (ctx->model->layer[il].attention == DS4_GLM5_NEXT_ATTN_KDA) {
            if (state->kda.layer[il].token_count != frontier ||
                !verify_kda_layout(ctx, il) ||
                !ds4_glm5_kda_verify_ready(&state->kda.layer[il], tokens, ctx->tp_rank))
                return 0;
        } else if (state->mla[il].token_count != frontier ||
                   !ds4_glm5_next_mla_verify_ready(&state->mla[il], tokens)) return 0;
    }
    return 1;
}

int ds4_glm5_next_target_verify_reserve(const ds4_glm5_next_exec_ctx *ctx,
                                        ds4_glm5_next_state *state,
                                        uint32_t capacity) {
    for (uint32_t il = 0; il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il)
        if (!ds4_glm5_next_layer_verify_reserve(ctx, il, state, capacity)) return 0;
    return target_verify_ready(ctx, state, capacity);
}

static int target_buffers_disjoint(ds4_gpu_tensor *a, ds4_gpu_tensor *b) {
#ifdef DS4_ROCM_BUILD
    /* The ROCm tensor ABI exposes addresses as host metadata. contents()
     * synchronizes the device; alias preflight must not add those fences. */
    const uintptr_t ap = a ? (uintptr_t)a->ptr : 0u;
    const uintptr_t bp = b ? (uintptr_t)b->ptr : 0u;
#else
    const uintptr_t ap = (uintptr_t)ds4_gpu_tensor_contents(a);
    const uintptr_t bp = (uintptr_t)ds4_gpu_tensor_contents(b);
#endif
    if (!ap || !bp) return 0;
    return ap <= bp ? ds4_gpu_tensor_bytes(a) <= bp - ap :
                     ds4_gpu_tensor_bytes(b) <= ap - bp;
}

static int target_binding_matches(const ds4_glm5_next_exec_ctx *ctx,
                                    const ds4_glm5_next_state *state) {
    if (!verify_layer_valid(ctx, 0u, state)) return 0;
    const ds4_glm5_next_verification *v = &state->verification;
    return v->tokens && v->model == ctx->model && v->model_map == ctx->model_map &&
        v->model_size == ctx->model_size && v->tp == ctx->tp && v->rank == ctx->tp_rank &&
        v->tp_slab == ctx->tp_slab && v->tp_big_out == ctx->tp_big_out &&
        v->tp_big_in == ctx->tp_big_in && v->tp_big_out_host == ctx->tp_big_out_host &&
        v->tp_big_in_host == ctx->tp_big_in_host && v->tp_sequence == ctx->tp_sequence &&
        tp_context_valid_bytes(ctx, (uint64_t)v->tokens * GLM5_WIDTH * sizeof(float)) &&
        v->sequence_end == *ctx->tp_sequence &&
        v->runtime_features == ds4_tp_runtime_features(ctx->tp) &&
        v->prefill_config == ds4_tp_prefill_config(ctx->tp);
}

static int native_draft_settings(void) {
    const char *reuse = getenv("DS4_GLM5_NATIVE_WINDOW_REUSE");
    if (reuse && strcmp(reuse, "0") != 0 && strcmp(reuse, "1") != 0) return 0;
    const char *disabled[] = {"DS4_ROCM_GLM5_WINDOW_PERSIST",
        "DS4_ROCM_GLM5_WINDOW_PERSIST_PREFILL", "DS4_ROCM_GLM5_WINDOW_SCRATCH",
        "DS4_ROCM_GLM5_WINDOW_ASYNC", "DS4_ROCM_GLM5_WINDOW_OVERLAP"};
    for (size_t i = 0; i < sizeof(disabled) / sizeof(disabled[0]); ++i) {
        const char *value = getenv(disabled[i]);
        if (value && strcmp(value, "0") != 0) return 0;
    }
    const char *slots = getenv("DS4_ROCM_GLM5_WINDOW_SLOTS");
    return !slots || strcmp(slots, "8") == 0;
}

ds4_glm5_next_workspace *ds4_glm5_next_draft_workspace_create_rows(
        uint32_t tokens, uint32_t capacity) {
    if (!tokens || tokens > 256u || !native_draft_settings()) return NULL;
    ds4_glm5_next_workspace *w = ds4_glm5_next_workspace_create_capacity_context(tokens, capacity);
    if (w) { w->draft_only = true; ds4_glm5_next_workspace_begin_decode(w); }
    return w;
}

ds4_glm5_next_workspace *ds4_glm5_next_draft_workspace_create(uint32_t capacity) {
    return ds4_glm5_next_draft_workspace_create_rows(1u, capacity);
}

static int draft_binding_matches(const ds4_glm5_next_exec_ctx *ctx,
        const ds4_glm5_next_state *owner, const ds4_glm5_next_workspace *w) {
    if (!w->draft_owner) return true;
    const ds4_glm5_next_exec_ctx *b = &w->draft_context;
    return w->draft_owner == owner && b->model == ctx->model &&
        b->model_map == ctx->model_map && b->model_size == ctx->model_size &&
        b->tp == ctx->tp && b->tp_rank == ctx->tp_rank && b->tp_slab == ctx->tp_slab &&
        b->tp_big_out == ctx->tp_big_out && b->tp_big_in == ctx->tp_big_in &&
        b->tp_big_out_host == ctx->tp_big_out_host && b->tp_big_in_host == ctx->tp_big_in_host &&
        b->tp_sequence == ctx->tp_sequence &&
        w->draft_runtime_features == ds4_tp_runtime_features(ctx->tp) &&
        w->draft_prefill_config == ds4_tp_prefill_config(ctx->tp);
}

int ds4_glm5_next_draft_target_hidden(const ds4_glm5_next_exec_ctx *ctx,
        ds4_glm5_next_workspace *w, const ds4_gpu_tensor *hc, ds4_gpu_tensor *hidden) {
    return ds4_glm5_next_draft_target_hidden_rows(ctx, w, hc, hidden,
        w ? w->capacity_tokens : 0u);
}

int ds4_glm5_next_draft_target_hidden_rows(const ds4_glm5_next_exec_ctx *ctx,
        ds4_glm5_next_workspace *w, const ds4_gpu_tensor *hc,
        ds4_gpu_tensor *hidden, uint32_t n) {
    if (!context_valid(ctx) || !w || !n || n > w->capacity_tokens ||
        ds4_gpu_tensor_bytes(hc) != (uint64_t)n * GLM5_HC_WIDTH * sizeof(float) ||
        ds4_gpu_tensor_bytes(hidden) != (uint64_t)n * GLM5_WIDTH * sizeof(float) ||
        !target_buffers_disjoint((ds4_gpu_tensor *)hc, hidden)) return 0;
    ds4_gpu_tensor *means = ds4_gpu_tensor_view(n == 1u ? w->hc_mean_weights : w->hc_flat,
        0, (uint64_t)n * GLM5_HC * sizeof(float));
    ds4_gpu_tensor *contracted = n == 1u ? w->output_hidden :
        ds4_gpu_tensor_view(w->collapsed, 0, (uint64_t)n * GLM5_WIDTH * sizeof(float));
    int ok = means && contracted && (n == 1u || ds4_gpu_tensor_fill_f32(means,
        1.0f / GLM5_HC, (uint64_t)n * GLM5_HC));
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(contracted, hc, means,
            GLM5_WIDTH, GLM5_HC) &&
        (n == 1u ? ds4_gpu_rms_norm_weight_tensor(hidden, contracted,
            ctx->model_map, ctx->model_size, ctx->model->output_norm,
            GLM5_WIDTH, ctx->model->rms_norm_eps) :
        ds4_gpu_rms_norm_weight_rows_tensor(hidden, contracted,
            ctx->model_map, ctx->model_size, ctx->model->output_norm,
            GLM5_WIDTH, n, ctx->model->rms_norm_eps));
    if (n != 1u) ds4_gpu_tensor_free(contracted);
    ds4_gpu_tensor_free(means);
    return ok;
}

static void draft_bind(const ds4_glm5_next_exec_ctx *ctx,
        ds4_glm5_next_state *owner, ds4_glm5_next_workspace *w) {
    if (w->draft_owner) return;
    w->draft_owner = owner;
    w->draft_context = *ctx;
    w->draft_runtime_features = ds4_tp_runtime_features(ctx->tp);
    w->draft_prefill_config = ds4_tp_prefill_config(ctx->tp);
}

/* Only activation tiles are concatenated. M1 keeps the original scalar
 * primitive calls; M>1 uses ordinary GGUF projection dispatch. */
static int native_project_inputs(const ds4_glm5_next_exec_ctx *ctx,
        ds4_glm5_next_workspace *w, const ds4_gpu_tensor *previous,
        const uint32_t *tokens, uint32_t n) {
    const uint64_t row = GLM5_WIDTH * sizeof(float);
    ds4_gpu_tensor *concat = ds4_gpu_tensor_view(w->hc_flat, 0, n * 2u * row);
    ds4_gpu_tensor *enorm = n == 1u ? ds4_gpu_tensor_view(w->hc_flat, 0, row) : NULL;
    ds4_gpu_tensor *hnorm = n == 1u ? ds4_gpu_tensor_view(w->hc_flat, row, row) : NULL;
    int ok = concat && (n != 1u || (enorm && hnorm));
    if (ok && n == 1u) {
        ok = ctx->model->token_embd_type == 8u ?
            ds4_gpu_embed_token_q8_0_tensor(w->ffn_collapsed, ctx->model_map,
                ctx->model_size, ctx->model->token_embd, GLM5_VOCAB, tokens[0], GLM5_WIDTH) :
            ds4_gpu_embed_token_hc_bf16_tensor(w->ffn_collapsed, ctx->model_map,
                ctx->model_size, ctx->model->token_embd, GLM5_VOCAB, tokens[0], GLM5_WIDTH, 1u);
        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(enorm, w->ffn_collapsed,
                ctx->model_map, ctx->model_size, ctx->model->nextn_enorm,
                GLM5_WIDTH, ctx->model->rms_norm_eps) &&
            ds4_gpu_rms_norm_weight_tensor(hnorm, previous,
                ctx->model_map, ctx->model_size, ctx->model->nextn_hnorm,
                GLM5_WIDTH, ctx->model->rms_norm_eps);
    } else if (ok) {
        /* Query selection is unused during warming: borrow its existing
         * integer storage for the small shifted-token tile. */
        ds4_gpu_tensor *ids = ds4_gpu_tensor_view(w->mla_selected_token, 0, n * 4u);
        ok = ids && ds4_gpu_tensor_write(ids, 0, tokens, n * 4u);
        if (ok) ok = ctx->model->token_embd_type == 8u ?
            ds4_gpu_embed_tokens_q8_0_tensor(w->ffn_collapsed, ids, ctx->model_map,
                ctx->model_size, ctx->model->token_embd, GLM5_VOCAB, n, GLM5_WIDTH) :
            ds4_gpu_embed_tokens_hc_bf16_tensor(w->ffn_collapsed, ids, ctx->model_map,
                ctx->model_size, ctx->model->token_embd, GLM5_VOCAB, n, GLM5_WIDTH, 1u);
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(w->ffn_hidden, w->ffn_collapsed,
                ctx->model_map, ctx->model_size, ctx->model->nextn_enorm,
                GLM5_WIDTH, n, ctx->model->rms_norm_eps) &&
            ds4_gpu_rms_norm_weight_rows_tensor(w->down, previous,
                ctx->model_map, ctx->model_size, ctx->model->nextn_hnorm,
                GLM5_WIDTH, n, ctx->model->rms_norm_eps);
        for (uint32_t t = 0; ok && t < n; ++t)
            ok = ds4_gpu_tensor_copy(concat, 2u * t * row, w->ffn_hidden, t * row, row) &&
                 ds4_gpu_tensor_copy(concat, (2u * t + 1u) * row, w->down, t * row, row);
        ds4_gpu_tensor_free(ids);
    }
    if (ok) ok = ds4_gpu_matmul_bf16_tensor(w->collapsed, ctx->model_map, ctx->model_size,
        ctx->model->nextn_eh_proj, 2u * GLM5_WIDTH, GLM5_WIDTH, concat, n);
    ds4_gpu_tensor_free(hnorm); ds4_gpu_tensor_free(enorm); ds4_gpu_tensor_free(concat);
    return ok;
}

int ds4_glm5_next_draft_warm_rows(const ds4_glm5_next_exec_ctx *ctx,
        ds4_glm5_next_state *owner, ds4_glm5_next_workspace *w,
        const ds4_gpu_tensor *previous, const uint32_t *tokens, uint32_t n) {
    uint32_t slot = 0, pool = 0;
    bool publish = false;
    if (!context_valid(ctx) || !tp_context_valid(ctx) || ctx->trace_prefix ||
        !owner || !owner->draft_only || owner->draft_model != ctx->model ||
        owner->pending_mla_verifications ||
        !w || !w->draft_only || !tokens || !n || n > 256u || n != w->capacity_tokens ||
        !draft_binding_matches(ctx, owner, w) || !native_draft_settings() ||
        ds4_gpu_tensor_bytes(previous) != (uint64_t)n * GLM5_WIDTH * sizeof(float)) return 0;
    ds4_glm5_next_mla_state *s = &owner->mla[DS4_GLM5_NEXT_TRUNK_COUNT];
    if (w->sparse_pool_capacity < s->capacity_pools ||
        !ds4_glm5_next_mla_append_plan(s, &slot, &pool, &publish) ||
        n > s->capacity_tokens - s->token_count ||
        !target_buffers_disjoint((ds4_gpu_tensor *)previous, ctx->tp_big_out) ||
        !target_buffers_disjoint((ds4_gpu_tensor *)previous, ctx->tp_big_in) ||
        (ctx->tp_slab && !target_buffers_disjoint((ds4_gpu_tensor *)previous, ctx->tp_slab))) return 0;
    for (uint32_t t = 0; t < n; ++t) if (tokens[t] >= GLM5_VOCAB) return 0;
    draft_bind(ctx, owner, w);
    const uint32_t pos = s->token_count;
    const ds4_glm5_next_layer_offsets *l = &ctx->model->layer[DS4_GLM5_NEXT_TRUNK_COUNT];
    const ds4_glm5_next_mla_offsets *m = &l->mla;
    int ok = native_project_inputs(ctx, w, previous, tokens, n);
    if (ok) ok = n == 1u ? ds4_gpu_rms_norm_weight_tensor(w->ffn_hidden, w->collapsed,
        ctx->model_map, ctx->model_size, l->attn_norm, GLM5_WIDTH, ctx->model->rms_norm_eps) :
        ds4_gpu_rms_norm_weight_rows_tensor(w->ffn_hidden, w->collapsed,
        ctx->model_map, ctx->model_size, l->attn_norm, GLM5_WIDTH, n, ctx->model->rms_norm_eps);
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(w->mla_kv_raw, ctx->model_map, ctx->model_size,
            m->kv_a_mqa, GLM5_WIDTH, GLM5_KV_LORA, w->ffn_hidden, n) &&
        ds4_gpu_glm_kv_lora_rms_norm_tensor(w->mla_kv_norm, w->mla_kv_raw,
            ctx->model_map, ctx->model_size, m->kv_a_norm, n, GLM5_KV_LORA,
            GLM5_KV_LORA, ctx->model->rms_norm_eps) &&
        ds4_gpu_glm_store_compact_kv_tensor(s->compact_kv, NULL, w->mla_kv_norm,
            w->mla_kv_raw, pos, n, s->capacity_tokens, GLM5_KV_LORA,
            GLM5_KV_LORA, 0u, false) &&
        ds4_gpu_matmul_bf16_tensor(w->mla_index_k_raw, ctx->model_map, ctx->model_size,
            m->index_k, GLM5_WIDTH, GLM5_INDEX_DIM, w->ffn_hidden, n) &&
        ds4_gpu_glm_store_indexer_k_tensor(w->mla_index_k_norm, w->mla_index_k_raw,
            ctx->model_map, ctx->model_size, m->index_k_norm, m->index_k_norm_b,
            0u, n, n, GLM5_INDEX_DIM, 0u, 1u, 1.0e-6f, 1, 1, 0, 1, 0, 0, false) &&
        ds4_gpu_matmul_bf16_tensor(w->mla_pool_gate_raw, ctx->model_map, ctx->model_size,
            m->index_pool_gate, GLM5_WIDTH, GLM5_INDEX_DIM, w->ffn_hidden, n) &&
        mla_stage_index_rows(ctx, m, s, w, pos, n);
    /* The aligned pooled helper can omit inactive tails. Preserve their
     * physical scalar bytes too, so future accepted-prefix journals match. */
    const uint64_t index_row = GLM5_INDEX_DIM * sizeof(float);
    for (uint32_t t = n > 4u ? n - 4u : 0u; ok && t < n; ++t)
        ok = ds4_gpu_tensor_copy(s->index_tail, ((pos + t) % 4u) * index_row,
                w->mla_index_k_norm, t * index_row, index_row) &&
             ds4_gpu_tensor_copy(s->pool_gate_tail, ((pos + t) % 4u) * index_row,
                w->mla_pool_gate_raw, t * index_row, index_row);
    if (ok) ok = ds4_gpu_synchronize();
    for (uint32_t t = 0; ok && t < n; ++t) ok = ds4_glm5_next_mla_append_commit(s);
    if (!ok) { ds4_gpu_synchronize(); ds4_glm5_next_state_invalidate(owner); }
    return ok;
}

int ds4_glm5_next_draft_step(const ds4_glm5_next_exec_ctx *ctx,
        ds4_glm5_next_mla_state *s, ds4_glm5_next_workspace *w,
        const ds4_gpu_tensor *previous, uint32_t token,
        ds4_gpu_tensor *hidden, ds4_gpu_tensor *logits) {
    const char *profile_option = getenv("DS4_GLM5_NATIVE_DRAFT_PROFILE");
    const bool profile = profile_option && strcmp(profile_option, "1") == 0;
    const double begin = profile ? glm5_exec_now_sec() : 0.0;
    double last = begin, admission_sec = 0.0, input_sec = 0.0;
    double attention_sec = 0.0, ffn_sec = 0.0, head_sec = 0.0;
    uint32_t slot = 0, pool = 0, visible = 0;
    bool publish = false;
    if (!context_valid(ctx) || !tp_context_valid(ctx) || ctx->trace_prefix ||
        (ds4_tp_runtime_features(ctx->tp) & DS4_TP_FEATURE_GLM5_GPU_ROW_GATE) ||
        !s || !s->owner || !s->owner->valid || !s->owner->draft_only ||
        s->owner->draft_model != ctx->model ||
        s->owner->layer_count != DS4_GLM5_NEXT_LAYER_COUNT || s->owner->mla_count != 1u ||
        !w || !w->draft_only || !w->decode_phase || w->capacity_tokens != 1u ||
        !draft_binding_matches(ctx, s->owner, w) ||
        w->sparse_pool_capacity < s->capacity_pools || !native_draft_settings() ||
        token >= GLM5_VOCAB || !ds4_glm5_next_mla_append_plan(s, &slot, &pool, &publish) ||
        ds4_gpu_tensor_bytes(previous) != GLM5_WIDTH * sizeof(float) ||
        ds4_gpu_tensor_bytes(hidden) != GLM5_WIDTH * sizeof(float) ||
        ds4_gpu_tensor_bytes(logits) != GLM5_VOCAB * sizeof(float) ||
        !target_buffers_disjoint((ds4_gpu_tensor *)previous, hidden) ||
        !target_buffers_disjoint((ds4_gpu_tensor *)previous, logits) ||
        !target_buffers_disjoint(hidden, logits)) return 0;
    ds4_gpu_tensor *buffers[] = {(ds4_gpu_tensor *)previous, hidden, logits};
    for (unsigned i = 0; i < 3; ++i)
        if (!target_buffers_disjoint(buffers[i], ctx->tp_big_out) ||
            !target_buffers_disjoint(buffers[i], ctx->tp_big_in) ||
            (ctx->tp_slab && !target_buffers_disjoint(buffers[i], ctx->tp_slab))) return 0;
    const unsigned il = DS4_GLM5_NEXT_TRUNK_COUNT;
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    const ds4_glm5_next_ffn_offsets *f = &layer->ffn_weight;
    if (f->gate_exps_type != 12u || f->up_exps_type != 12u ||
        f->down_exps_type != 12u || local_q4k_half_residency(ctx, layer) != 0) return 0;
    /* A target streaming-layer cleanup would destroy retained window handles.
     * Never allow a private draft to run across that ownership change. */
    for (unsigned trunk = 3; trunk < DS4_GLM5_NEXT_TRUNK_COUNT; ++trunk)
        if (local_q4k_half_residency(ctx, &ctx->model->layer[trunk]) != 1) return 0;
    draft_bind(ctx, s->owner, w);
    if (profile) { last = glm5_exec_now_sec(); admission_sec = last - begin; }
    int ok = native_project_inputs(ctx, w, previous, &token, 1u);
    /* Diagnostic-only completion fences. The off path keeps the existing
     * enqueue and terminal-completion schedule, including error handling. */
#define GLM5_DRAFT_PROFILE_MARK(field) do { \
    if (profile) { \
        if (ok) ok = ds4_gpu_synchronize(); \
        const double now = glm5_exec_now_sec(); \
        field = now - last; last = now; \
    } \
} while (0)
    GLM5_DRAFT_PROFILE_MARK(input_sec);
    ds4_glm5_next_exec_ctx bulk = *ctx;
    bulk.force_bulk_gates = true;
    const uint32_t pos = s->token_count;
    if (ok) ok = ds4_glm5_next_mla_dense_selection_visible(pos, s->capacity_tokens, &visible) ?
        mla_dense_selection_attention(&bulk, il, s, w, w->collapsed, visible, slot, pool, publish) :
        mla_sparse_selection_attention(&bulk, il, s, w, w->collapsed, slot, pool,
            publish, DS4_GLM5_NEXT_INDEX_TOP_K, bulk.tp_big_out, true, true);
    GLM5_DRAFT_PROFILE_MARK(attention_sec);
    if (ok) ok = routed_ffn_one(&bulk, il, pos, w, w->output_hidden);
    GLM5_DRAFT_PROFILE_MARK(ffn_sec);
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(hidden, w->output_hidden,
            ctx->model_map, ctx->model_size, ctx->model->nextn_shared_head_norm,
            GLM5_WIDTH, ctx->model->rms_norm_eps);
    if (ok) ok = ctx->model->output_type == 8u ?
        ds4_gpu_matmul_q8_0_tensor(logits, ctx->model_map, ctx->model_size,
            ctx->model->output, GLM5_WIDTH, GLM5_VOCAB, hidden, 1u) :
        ds4_gpu_matmul_bf16_tensor(logits, ctx->model_map, ctx->model_size,
            ctx->model->output, GLM5_WIDTH, GLM5_VOCAB, hidden, 1u);
    if (ok) ok = ds4_gpu_synchronize() && ds4_glm5_next_mla_append_commit(s);
    if (!ok) { ds4_gpu_synchronize(); ds4_glm5_next_state_invalidate(s->owner); }
    if (profile) {
        const double end = glm5_exec_now_sec();
        head_sec = end - last;
        fprintf(stderr, "NATIVE_DRAFT rank=%u pos=%u token=%u ok=%d admission_ms=%.6f input_ms=%.6f attention_ms=%.6f ffn_ms=%.6f norm_head_publish_ms=%.6f total_ms=%.6f\n",
            ctx->tp_rank, pos, token, ok, admission_sec * 1000.0,
            input_sec * 1000.0, attention_sec * 1000.0, ffn_sec * 1000.0,
            head_sec * 1000.0, (end - begin) * 1000.0);
    }
#undef GLM5_DRAFT_PROFILE_MARK
    return ok;
}

int ds4_glm5_next_draft_refresh(const ds4_glm5_next_exec_ctx *ctx,
        ds4_glm5_next_state *owner, ds4_glm5_next_workspace *w,
        const ds4_gpu_tensor *target_hidden, const uint32_t *inputs,
        uint32_t prefix, uint32_t rows, uint32_t accepted,
        ds4_gpu_tensor *previous) {
    const uint64_t row = GLM5_WIDTH * sizeof(float);
    if (!context_valid(ctx) || !tp_context_valid(ctx) || ctx->trace_prefix ||
        !owner || !owner->valid || !owner->draft_only || owner->draft_model != ctx->model ||
        !w || !w->draft_only || !w->decode_phase || w->capacity_tokens != 1u ||
        !draft_binding_matches(ctx, owner, w) || !native_draft_settings() ||
        !inputs || !prefix || !ds4_tp_glm5_native_width_valid(rows) ||
        !accepted || accepted > rows || prefix > owner->context_capacity ||
        rows > owner->context_capacity - prefix ||
        ds4_gpu_tensor_bytes(target_hidden) != rows * row ||
        ds4_gpu_tensor_bytes(previous) != row ||
        !target_buffers_disjoint((ds4_gpu_tensor *)target_hidden, previous)) return 0;
    ds4_glm5_next_mla_state *s = &owner->mla[DS4_GLM5_NEXT_TRUNK_COUNT];
    if (owner->pending_mla_verifications != 1u ||
        w->sparse_pool_capacity < s->capacity_pools ||
        !ds4_glm5_next_mla_verify_pending(s, prefix - 1u, rows - 1u)) return 0;
    ds4_gpu_tensor *buffers[] = {(ds4_gpu_tensor *)target_hidden, previous};
    for (unsigned i = 0; i < 2u; ++i)
        if (!target_buffers_disjoint(buffers[i], ctx->tp_big_out) ||
            !target_buffers_disjoint(buffers[i], ctx->tp_big_in) ||
            (ctx->tp_slab && !target_buffers_disjoint(buffers[i], ctx->tp_slab))) return 0;
    for (uint32_t t = 0; t < rows; ++t) if (inputs[t] >= GLM5_VOCAB) return 0;

    /* Create metadata views before retiring the journal so allocation refusal
     * leaves it usable. No GPU storage is allocated or copied here. */
    ds4_gpu_tensor *hidden[8] = {0};
    uint32_t prepared = 0;
    while (prepared < accepted) {
        hidden[prepared] = ds4_gpu_tensor_view((ds4_gpu_tensor *)target_hidden,
            prepared * row, row);
        if (!hidden[prepared]) break;
        ++prepared;
    }
    if (prepared != accepted) {
        for (uint32_t t = 0; t < prepared; ++t) ds4_gpu_tensor_free(hidden[t]);
        return 0;
    }
    draft_bind(ctx, owner, w);
    int ok = ds4_glm5_next_mla_verify_finish(s, 0u);
    for (uint32_t t = 0; ok && t < accepted; ++t)
        ok = ds4_glm5_next_draft_warm_rows(ctx, owner, w,
            t ? hidden[t - 1u] : previous, &inputs[t], 1u);
    if (ok) ok = ds4_gpu_tensor_copy(previous, 0, hidden[accepted - 1u], 0, row) &&
        ds4_gpu_synchronize();
    for (uint32_t t = 0; t < prepared; ++t) ds4_gpu_tensor_free(hidden[t]);
    if (!ok) { ds4_gpu_synchronize(); ds4_glm5_next_state_invalidate(owner); }
    return ok;
}

static int target_output_logits_rows(const ds4_glm5_next_exec_ctx *ctx,
                                    ds4_glm5_next_workspace *batch_w,
                                    ds4_glm5_next_workspace *scalar_w,
                                    const ds4_gpu_tensor *hc_hidden,
                                    ds4_gpu_tensor *logits, uint32_t tokens) {
    const uint64_t row = (uint64_t)GLM5_HC_WIDTH * sizeof(float);
    const uint64_t norm_row = (uint64_t)GLM5_WIDTH * sizeof(float);
    /* Trunk execution has finished. Reuse its collapsed activation scratch;
     * journals own their replay inputs independently of this workspace.
     * Collapse and normalization retain scalar dispatch and arithmetic. */
    for (uint32_t t = 0; t < tokens; ++t) {
        ds4_gpu_tensor *hidden = ds4_gpu_tensor_view(hc_hidden, t * row, row);
        ds4_gpu_tensor *norm = ds4_gpu_tensor_view(batch_w->collapsed, t * norm_row, norm_row);
        const int ok = hidden && norm &&
            ds4_gpu_hc_weighted_sum_tensor(scalar_w->output_hidden, hidden,
                scalar_w->hc_mean_weights, GLM5_WIDTH, GLM5_HC) &&
            ds4_gpu_rms_norm_weight_tensor(norm, scalar_w->output_hidden,
                ctx->model_map, ctx->model_size, ctx->model->output_norm,
                GLM5_WIDTH, ctx->model->rms_norm_eps);
        ds4_gpu_tensor_free(norm);
        ds4_gpu_tensor_free(hidden);
        if (!ok) return 0;
    }
    return ds4_gpu_matmul_bf16_tensor(logits, ctx->model_map, ctx->model_size,
        ctx->model->output, GLM5_WIDTH, GLM5_VOCAB, batch_w->collapsed, tokens);
}

int ds4_glm5_next_target_verify(const ds4_glm5_next_exec_ctx *ctx,
                                ds4_glm5_next_state *state,
                                ds4_glm5_next_workspace *batch_w,
                                ds4_glm5_next_workspace *scalar_w,
                                const uint32_t *input_tokens, uint32_t tokens,
                                ds4_gpu_tensor *hc_scratch,
                                ds4_gpu_tensor *hc_out,
                                ds4_gpu_tensor *logits_out) {
    const uint64_t row = (uint64_t)GLM5_HC_WIDTH * sizeof(float);
    const uint64_t logits_row = (uint64_t)GLM5_VOCAB * sizeof(float);
    const char *head_option = getenv("DS4_ROCM_GLM5_BF16_VERIFY_HEAD");
    if (head_option && strcmp(head_option, "0") != 0 &&
        strcmp(head_option, "1") != 0) return 0;
    if (!input_tokens || !target_verify_ready(ctx, state, tokens) ||
        !batch_w || !scalar_w || batch_w == scalar_w || !scalar_w->decode_phase ||
        scalar_w->capacity_tokens != 1u || batch_w->capacity_tokens != tokens ||
        scalar_w->sparse_pool_capacity < state->context_capacity / 4u +
            (state->context_capacity % 4u != 0u) ||
        ds4_gpu_tensor_bytes(hc_scratch) != tokens * row ||
        ds4_gpu_tensor_bytes(hc_out) != tokens * row ||
        ds4_gpu_tensor_bytes(logits_out) != tokens * logits_row ||
        !target_buffers_disjoint(hc_scratch, hc_out) ||
        !target_buffers_disjoint(hc_scratch, logits_out) ||
        !target_buffers_disjoint(hc_out, logits_out)) return 0;
    ds4_gpu_tensor *buffers[] = {hc_scratch, hc_out, logits_out};
    for (uint32_t i = 0; i < 3u; ++i)
        if (!target_buffers_disjoint(buffers[i], ctx->tp_big_out) ||
            !target_buffers_disjoint(buffers[i], ctx->tp_big_in) ||
            (ctx->tp_slab && !target_buffers_disjoint(buffers[i], ctx->tp_slab))) return 0;
    for (uint32_t t = 0; t < tokens; ++t) if (input_tokens[t] >= GLM5_VOCAB) return 0;
    ds4_glm5_next_verification *v = &state->verification;
    *v = (ds4_glm5_next_verification){
        .model=ctx->model, .model_map=ctx->model_map, .model_size=ctx->model_size,
        .tp=ctx->tp, .tp_slab=ctx->tp_slab, .tp_big_out=ctx->tp_big_out,
        .tp_big_in=ctx->tp_big_in, .tp_big_out_host=ctx->tp_big_out_host,
        .tp_big_in_host=ctx->tp_big_in_host, .tp_sequence=ctx->tp_sequence,
        .prefill_config=ds4_tp_prefill_config(ctx->tp),
        .runtime_features=ds4_tp_runtime_features(ctx->tp), .rank=ctx->tp_rank,
        .frontier=state->kda.layer[0].token_count, .tokens=tokens,
    };
    memcpy(v->input_tokens, input_tokens, tokens * sizeof(*input_tokens));
    const bool profile = getenv("DS4_GLM5_VERIFY_PROFILE") != NULL;
    const double begin_sec = profile ? glm5_exec_now_sec() : 0.0;
    int ok = 1;
    for (uint32_t t = 0; ok && t < tokens; ++t) {
        ds4_gpu_tensor *out = ds4_gpu_tensor_view(hc_scratch, t * row, row);
        ok = out && ds4_glm5_next_embed_token(ctx, v->input_tokens[t], out);
        ds4_gpu_tensor_free(out);
    }
    if (profile && ok) ok = ds4_gpu_synchronize();
    const double embed_sec = profile ? glm5_exec_now_sec() : 0.0;
    ds4_gpu_tensor *in = hc_scratch, *out = hc_out;
    for (uint32_t il = 0; ok && il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il) {
        ok = layer_verify_run(ctx, il, state, batch_w, scalar_w, in, out, tokens);
        if (ok) v->next_layer = il + 1u;
        ds4_gpu_tensor *swap = in; in = out; out = swap;
    }
    const double trunk_sec = profile ? glm5_exec_now_sec() : 0.0;
    /* The 45-layer trunk is odd: final hidden rows are always in hc_out. */
    const bool batch_head = head_option && strcmp(head_option, "1") == 0 &&
        ctx->model->output_type == 30u;
    if (ok && batch_head)
        ok = target_output_logits_rows(ctx, batch_w, scalar_w, hc_out, logits_out, tokens);
    for (uint32_t t = 0; ok && !batch_head && t < tokens; ++t) {
        ds4_gpu_tensor *hidden = ds4_gpu_tensor_view(hc_out, t * row, row);
        ds4_gpu_tensor *logits = ds4_gpu_tensor_view(logits_out, t * logits_row, logits_row);
        ok = hidden && logits && ds4_glm5_next_output_logits(ctx, scalar_w, hidden, logits);
        ds4_gpu_tensor_free(logits);
        ds4_gpu_tensor_free(hidden);
    }
    if (ok) ok = ds4_gpu_synchronize();
    if (profile) fprintf(stderr,
        "VERIFY_PROFILE rank=%u target m=%u embed_ms=%.6f trunk_ms=%.6f head_ms=%.6f ok=%d\n",
        ctx->tp_rank, tokens, (embed_sec - begin_sec) * 1000.0,
        (trunk_sec - embed_sec) * 1000.0,
        (glm5_exec_now_sec() - trunk_sec) * 1000.0, ok);
    if (!ok) {
        ds4_glm5_next_state_invalidate(state);
        return 0;
    }
    v->sequence_end = *ctx->tp_sequence;
    v->complete = true;
    return 1;
}

int ds4_glm5_next_target_verify_finish(const ds4_glm5_next_exec_ctx *ctx,
                                       ds4_glm5_next_state *state,
                                       uint32_t accepted) {
    if (!target_binding_matches(ctx, state)) return 0;
    const bool profile = getenv("DS4_GLM5_VERIFY_PROFILE") != NULL;
    const double begin_sec = profile ? glm5_exec_now_sec() : 0.0;
    const ds4_glm5_next_verification *v = &state->verification;
    if (!v->complete || v->next_layer != DS4_GLM5_NEXT_TRUNK_COUNT ||
        accepted > v->tokens || state->kda.pending_verifications != 34u ||
        state->pending_mla_verifications != DS4_GLM5_NEXT_MLA_COUNT) return 0;
    /* Check every journal before a single committed counter or byte changes. */
    for (uint32_t il = 0; il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il) {
        const int ready = ctx->model->layer[il].attention == DS4_GLM5_NEXT_ATTN_KDA ?
            ds4_glm5_kda_verify_pending(&state->kda.layer[il], v->frontier, v->tokens, v->rank) :
            ds4_glm5_next_mla_verify_pending(&state->mla[il], (uint32_t)v->frontier, v->tokens);
        if (!ready) return 0;
    }
    for (uint32_t il = 0; il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il) {
        const int ok = ctx->model->layer[il].attention == DS4_GLM5_NEXT_ATTN_KDA ?
            ds4_glm5_kda_verify_finish(&state->kda.layer[il], accepted) :
            ds4_glm5_next_mla_verify_finish(&state->mla[il], accepted);
        if (!ok) {
            ds4_glm5_next_state_invalidate(state);
            return 0;
        }
    }
    if (!ds4_gpu_synchronize()) {
        ds4_glm5_next_state_invalidate(state);
        return 0;
    }
    memset(&state->verification, 0, sizeof(state->verification));
    if (profile) fprintf(stderr,
        "VERIFY_PROFILE rank=%u finish accepted=%u commit_ms=%.6f ok=1\n",
        ctx->tp_rank, accepted, (glm5_exec_now_sec() - begin_sec) * 1000.0);
    return 1;
}

int ds4_glm5_next_layer_forward(const ds4_glm5_next_exec_ctx *ctx,
                                uint32_t il,
                                ds4_glm5_next_state *state,
                                ds4_glm5_next_workspace *w,
                                const ds4_gpu_tensor *hc_in,
                                ds4_gpu_tensor *hc_out) {
    const uint64_t hc_bytes = (uint64_t)GLM5_HC_WIDTH * sizeof(float);
    if (!context_valid(ctx) || !state || !state->valid ||
        state->verification.tokens || state->pending_mla_verifications ||
        state->kda.pending_verifications || !w || !hc_in ||
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
    if (!context_valid(ctx) || !state || !state->valid ||
        state->verification.tokens || state->pending_mla_verifications ||
        state->kda.pending_verifications || !w || !hc_in ||
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

int ds4_glm5_next_layer_forward_batch_sparse_bridge(
        const ds4_glm5_next_exec_ctx *ctx,
        uint32_t il,
        ds4_glm5_next_state *state,
        ds4_glm5_next_workspace *batch_w,
        ds4_glm5_next_workspace *scalar_w,
        const ds4_gpu_tensor *hc_in,
        ds4_gpu_tensor *hc_out,
        uint32_t n_tokens) {
    if (!context_valid(ctx) || !state || !state->valid ||
        state->verification.tokens || state->pending_mla_verifications ||
        state->kda.pending_verifications || !batch_w ||
        !scalar_w || batch_w == scalar_w || !hc_in || !hc_out ||
        hc_in == hc_out || n_tokens == 0u ||
        batch_w->capacity_tokens != n_tokens ||
        scalar_w->capacity_tokens != 1u || il >= ctx->model->trunk_count) {
        return 0;
    }
    const ds4_glm5_next_layer_offsets *layer = &ctx->model->layer[il];
    ds4_glm5_next_mla_state *mla = &state->mla[il];
    const bool sparse_mla =
        layer->attention == DS4_GLM5_NEXT_ATTN_MLA &&
        layer->ffn == DS4_GLM5_NEXT_FFN_ROUTED && mla->compact_kv &&
        !state->kda.layer[il].recurrent &&
        mla->token_count >= DS4_GLM5_NEXT_INDEX_TOP_K;
    if (!sparse_mla) {
        return ds4_glm5_next_layer_forward_batch(
            ctx, il, state, batch_w, hc_in, hc_out, n_tokens);
    }

    const uint64_t row_bytes = (uint64_t)GLM5_HC_WIDTH * sizeof(float);
    const uint64_t split_row_bytes =
        (uint64_t)GLM5_HC_MIX * sizeof(float);
    const uint64_t local_row_bytes =
        (uint64_t)GLM5_WIDTH * sizeof(float);
    const uint64_t heads_row_bytes =
        (uint64_t)GLM5_HEADS * GLM5_HEAD_DIM * sizeof(float);
    const uint64_t elements = (uint64_t)n_tokens * GLM5_WIDTH;
    const bool batch_output =
        (ds4_tp_runtime_features(ctx->tp) &
         DS4_TP_FEATURE_GLM5_SPARSE_BATCH_OUTPUT) != 0u;
    const bool batch_prelude =
        (ds4_tp_runtime_features(ctx->tp) &
         DS4_TP_FEATURE_GLM5_SPARSE_BATCH_PRELUDE) != 0u;
    const bool batch_value = n_tokens > 1u &&
        (ds4_tp_runtime_features(ctx->tp) &
         DS4_TP_FEATURE_GLM5_SPARSE_BATCH_VALUE) != 0u;
    const uint64_t selected_row_bytes =
        (uint64_t)GLM5_SELECTED_STRIDE * sizeof(int32_t);
    if ((uint64_t)n_tokens > UINT64_MAX / row_bytes ||
        ds4_gpu_tensor_bytes(hc_in) != (uint64_t)n_tokens * row_bytes ||
        ds4_gpu_tensor_bytes(hc_out) != (uint64_t)n_tokens * row_bytes ||
        elements > UINT32_MAX || ctx->trace_prefix != NULL ||
        mla->token_count > mla->capacity_tokens ||
        n_tokens > mla->capacity_tokens - mla->token_count) {
        return 0;
    }
    const uint32_t token_ordinal = mla->token_count;
    const bool batch_indexer_score = batch_prelude &&
        ctx->tp &&
        (ds4_tp_prefill_config(ctx->tp) &
         DS4_TP_PREFILL_CONFIG_GLM5_INDEXER_SCORE_BATCH) != 0u &&
        mla->first_valid == 0u && (token_ordinal & 3u) == 0u &&
        (n_tokens & 3u) == 0u &&
        (uint64_t)token_ordinal + n_tokens <= UINT32_MAX;
    const uint32_t batch_score_pools = batch_indexer_score
        ? (uint32_t)(((uint64_t)token_ordinal + n_tokens) / 4u) : 0u;
    const int phase_profile = getenv("DS4_GLM5_PHASE_PROFILE") != NULL;
    const double phase_t0 = phase_profile ? glm5_exec_now_sec() : 0.0;
    if (batch_prelude && !mla_sparse_prelude_rows(
            ctx, il, state, batch_w, hc_in, n_tokens)) {
        ds4_glm5_next_state_invalidate(state);
        return 0;
    }
    /* The prelude must populate normalized index rows before pool publication
     * and scoring. Aligned tiles may publish all complete pools up front; the
     * score kernel applies the per-query causal mask below. */
    if (batch_indexer_score &&
        (batch_score_pools == 0u ||
         batch_score_pools > mla->capacity_pools ||
         batch_score_pools > batch_w->sparse_pool_capacity ||
         ds4_gpu_tensor_bytes(batch_w->mla_pool_scores) <
             (uint64_t)n_tokens * batch_w->sparse_pool_capacity *
                 sizeof(float) ||
         !mla_stage_index_rows(ctx, &layer->mla, mla, batch_w,
                                token_ordinal, n_tokens) ||
         !ds4_gpu_glm_indexer_scores_pool_batch_tensor(
             batch_w->mla_pool_scores, batch_w->mla_index_q,
             batch_w->mla_index_weights, mla->index_pool,
             mla->index_pool_valid, batch_score_pools,
             batch_w->sparse_pool_capacity, n_tokens,
             token_ordinal, GLM5_INDEX_POOL, GLM5_INDEX_HEADS,
             GLM5_INDEX_DIM, 0.015625f, false))) {
        ds4_glm5_next_state_invalidate(state);
        return 0;
    }
    if (batch_indexer_score) {
        static int logged_batch_indexer_score[2] = {0, 0};
        const uint32_t rank = ctx->tp_rank < 2u ? ctx->tp_rank : 0u;
        if (!logged_batch_indexer_score[rank]) {
            fprintf(stderr,
                    "ds4: GLM5 pooled indexer score batch engaged "
                    "rank=%u pos0=%u rows=%u pools=%u\n",
                    ctx->tp_rank, token_ordinal, n_tokens,
                    batch_score_pools);
            logged_batch_indexer_score[rank] = 1;
        }
    }
    for (uint32_t t = 0u; t < n_tokens; ++t) {
        ds4_gpu_tensor *in_row = batch_prelude ? NULL :
            ds4_gpu_tensor_view(
                hc_in, (uint64_t)t * row_bytes, row_bytes);
        ds4_gpu_tensor *local_row = batch_output ? NULL :
            ds4_gpu_tensor_view(
                ctx->tp_big_out, (uint64_t)t * local_row_bytes,
                local_row_bytes);
        ds4_glm5_next_exec_ctx scalar_ctx = *ctx;
        ds4_gpu_tensor *selected_row = batch_value ?
            ds4_gpu_tensor_view(
                batch_w->mla_selected_token,
                (uint64_t)t * selected_row_bytes,
                selected_row_bytes) : NULL;
        const uint64_t row_end64 = (uint64_t)token_ordinal + t + 1u;
        const uint32_t row_pools = row_end64 <= UINT32_MAX
            ? (uint32_t)(row_end64 / GLM5_INDEX_POOL) : 0u;
        ds4_gpu_tensor *score_row = batch_indexer_score && row_pools != 0u
            ? ds4_gpu_tensor_view(
                batch_w->mla_pool_scores,
                (uint64_t)t * batch_w->sparse_pool_capacity *
                    sizeof(float),
                (uint64_t)row_pools * sizeof(float)) : NULL;
        uint32_t tail_slot = 0u, pool_index = 0u;
        bool publish_pool = false;
        const int planned = ds4_glm5_next_mla_append_plan(
            mla, &tail_slot, &pool_index, &publish_pool);
        const int attention_ok = planned &&
            (!batch_value || selected_row) && (batch_prelude ?
            mla_sparse_selection_from_prelude_row(
                &scalar_ctx, il, state, batch_w, scalar_w, t, tail_slot,
                pool_index, publish_pool, DS4_GLM5_NEXT_INDEX_TOP_K,
                selected_row, !batch_value, score_row, row_pools) :
            (in_row && mla_sparse_selection_attention(
                &scalar_ctx, il, mla, scalar_w, in_row, tail_slot,
                pool_index, publish_pool, DS4_GLM5_NEXT_INDEX_TOP_K,
                local_row, !batch_output, false)));
        const int row_ok = (batch_prelude || in_row) &&
            (batch_output || local_row) && attention_ok &&
            (!batch_output || batch_value || ds4_gpu_tensor_copy(
                batch_w->mla_heads, (uint64_t)t * heads_row_bytes,
                scalar_w->mla_heads, 0u, heads_row_bytes)) &&
            (batch_prelude || ds4_gpu_tensor_copy(
                batch_w->hc_split, (uint64_t)t * split_row_bytes,
                scalar_w->hc_split, 0u, split_row_bytes)) &&
            ds4_glm5_next_mla_append_commit(mla);
        ds4_gpu_tensor_free(local_row);
        ds4_gpu_tensor_free(in_row);
        ds4_gpu_tensor_free(selected_row);
        ds4_gpu_tensor_free(score_row);
        if (!row_ok) {
            ds4_glm5_next_state_invalidate(state);
            return 0;
        }
    }
    if (batch_value) {
        uint32_t selected_min = 0u, selected_max = 0u;
        if (!(mla_sparse_attention_rows_deferred(
                  ctx, mla, batch_w, il, token_ordinal, n_tokens,
                  &selected_min, &selected_max) &&
          mla_value_project_rows_batch(
              ctx, &layer->mla, batch_w, n_tokens))) {
            ds4_glm5_next_state_invalidate(state);
            return 0;
        }
        static int logged_batch_value[2] = {0, 0};
        const uint32_t rank = ctx->tp_rank < 2u ? ctx->tp_rank : 0u;
        if (!logged_batch_value[rank]) {
            const bool head_shared_rows =
                getenv("DS4_ROCM_GLM5_SPARSE_ATTN_HEAD_SHARED") != NULL &&
                strcmp(getenv("DS4_ROCM_GLM5_SPARSE_ATTN_HEAD_SHARED"),
                       "1") == 0;
            const bool f16_gemm_rows =
                (ds4_tp_runtime_features(ctx->tp) &
                 DS4_TP_FEATURE_GLM5_SPARSE_ATTN_F16_GEMM) != 0u;
            fprintf(stderr,
                    "ds4: GLM5 sparse deferred attention and batch "
                    "value projection engaged rank=%u layer=%u pos0=%u "
                    "rows=%u selected_stride=%u selected_live_min=%u "
                    "selected_live_max=%u attention=%s\n",
                    ctx->tp_rank, il, token_ordinal, n_tokens,
                    GLM5_SELECTED_STRIDE, selected_min, selected_max,
                    f16_gemm_rows ? "f16-gemm-r16" :
                    (head_shared_rows ? "head-shared-r16" : "scalar-rows"));
            logged_batch_value[rank] = 1;
        }
    }
    if (batch_output && !mla_output_project_rows_batch(
            ctx, &layer->mla, batch_w, n_tokens, 0)) {
        ds4_glm5_next_state_invalidate(state);
        return 0;
    }
    const double phase_t1 = phase_profile ? glm5_exec_now_sec() : 0.0;
    int ok = tp_exchange_rows(ctx, il, DS4_TP_GATE_ATTN, n_tokens) &&
        ds4_gpu_add_tensor(
            batch_w->attention, ctx->tp_big_out, ctx->tp_big_in,
            (uint32_t)elements) &&
        ds4_gpu_hc_expand_split_tensor(
            batch_w->after_attention, batch_w->attention, hc_in,
            batch_w->hc_split, GLM5_WIDTH, GLM5_HC);
    const double phase_t2 = phase_profile ? glm5_exec_now_sec() : 0.0;
    if (ok) ok = routed_ffn_rows(
        ctx, il, token_ordinal, batch_w, hc_out, n_tokens);
    if (phase_profile) {
        fprintf(stderr,
                "ds4: GLM5 phase sparse_bridge_layer=%u rows=%u token=%u "
                "scalar_attention_ms=%.3f exchange_post_ms=%.3f "
                "ffn_ms=%.3f ok=%d\n",
                il, n_tokens, token_ordinal,
                (phase_t1 - phase_t0) * 1000.0,
                (phase_t2 - phase_t1) * 1000.0,
                (glm5_exec_now_sec() - phase_t2) * 1000.0, ok ? 1 : 0);
    }
#ifdef DS4_TP_TEST_HOOKS
    if (ok && !layer_completion_diagnostic(hc_out, n_tokens)) {
        ds4_glm5_next_state_invalidate(state);
        return 0;
    }
    if (ok && !hc_batch_hash_trace(ctx, il, hc_out, n_tokens)) {
        ds4_glm5_next_state_invalidate(state);
        return 0;
    }
#endif
    if (!ok) ds4_glm5_next_state_invalidate(state);
    return ok;
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
    if (!context_valid(ctx) || !state || !state->valid ||
        state->verification.tokens || state->pending_mla_verifications ||
        state->kda.pending_verifications || !w || !hc_in ||
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
    if (!context_valid(ctx) || !state || !state->valid ||
        state->verification.tokens || state->pending_mla_verifications ||
        state->kda.pending_verifications || !w || !hc_in ||
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
                ctx, il, mla, w, hc_in, visible, tail_slot,
                pool_index, publish_pool);
        } else if (ok) {
            ok = mla_sparse_selection_attention(
                ctx, il, mla, w, hc_in, tail_slot,
                pool_index, publish_pool, DS4_GLM5_NEXT_INDEX_TOP_K,
                ctx->tp_big_out, true, true);
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
    if (!context_valid(ctx) || !state || !state->valid ||
        state->verification.tokens || state->pending_mla_verifications ||
        state->kda.pending_verifications || !w || !hc_in ||
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
                       ctx, il, mla, w, hc_in, tail_slot, pool_index,
                       publish_pool, top_k, ctx->tp_big_out, true, true) &&
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
