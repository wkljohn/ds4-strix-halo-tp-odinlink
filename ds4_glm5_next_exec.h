#ifndef DS4_GLM5_NEXT_EXEC_H
#define DS4_GLM5_NEXT_EXEC_H

#include <stdint.h>

#include "ds4_glm5_next_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ds4_tp;
typedef struct ds4_glm5_next_workspace ds4_glm5_next_workspace;

/* Reserve the bounded sparse-attention Lane-B tile workspace before timed
 * prefill. A positive return means reserved, zero is a hard backend failure,
 * and -1 means the backend does not provide the specialization. */
int ds4_rocm_glm5_sparse_attention_f16_gemm_reserve(void);

typedef struct {
    const void *model_map;
    uint64_t model_size;
    const ds4_glm5_next_model_offsets *model;
    struct ds4_tp *tp;
    uint32_t tp_rank;
    ds4_gpu_tensor *tp_slab;
    ds4_gpu_tensor *tp_big_out;
    ds4_gpu_tensor *tp_big_in;
    void *tp_big_out_host;
    void *tp_big_in_host;
    /* Research-only staged trace selector. NULL keeps the production path
     * free of device reads and filesystem activity. */
    const char *trace_prefix;
    uint32_t trace_layer;
    uint32_t trace_token;
#ifdef DS4_TP_TEST_HOOKS
    /* Optional GPU-only MLA stage capture. The executor enqueues copies into
     * this independent buffer and the harness performs one terminal read, so
     * diagnosis does not insert a synchronization between MLA and its FFN. */
    ds4_gpu_tensor *device_mla_stage_capture;
    uint64_t device_mla_stage_capture_bytes;
    uint32_t device_mla_stage_capture_layer;
#endif
    /* Transport-global, monotonically increasing big-gate sequence.  It is
     * deliberately not reset with a model sequence while the TP link lives. */
    uint64_t *tp_sequence;
} ds4_glm5_next_exec_ctx;

/* The default workspace preserves the one-token decode ABI.  Prefill callers
 * must request their exact row capacity and pass tensors with that exact row
 * count; this prevents byte-sized scratch buffers from silently selecting a
 * different number of mHC rows. */
ds4_glm5_next_workspace *ds4_glm5_next_workspace_create(void);
ds4_glm5_next_workspace *ds4_glm5_next_workspace_create_capacity(
        uint32_t capacity_tokens);
uint32_t ds4_glm5_next_workspace_capacity(
        const ds4_glm5_next_workspace *workspace);
/* Decode sparse-MLA scratch is sized from the session context independently
 * of the token-tile capacity. */
ds4_glm5_next_workspace *ds4_glm5_next_workspace_create_capacity_context(
        uint32_t capacity_tokens, uint32_t context_capacity);
void ds4_glm5_next_workspace_destroy(ds4_glm5_next_workspace *workspace);
/* Cache ownership follows the caller's execution phase. Prefill never uses
 * the bounded decode scratch, including scalar prompt tails. */
void ds4_glm5_next_workspace_begin_prefill(
        ds4_glm5_next_workspace *workspace);
void ds4_glm5_next_workspace_begin_decode(
        ds4_glm5_next_workspace *workspace);

#ifdef DS4_TP_TEST_HOOKS
uint64_t ds4_glm5_next_mla_stage_capture_bytes(uint32_t n_tokens);
int ds4_glm5_next_mla_stage_capture_dump(
        const ds4_gpu_tensor *capture, uint32_t n_tokens, FILE *stream);
/* Returns the exact compact activation slice and byte requirement used by
 * the production KDA TP output projection. This exists so a 4096/8192
 * dimension substitution cannot escape the production-shaped test gate. */
int ds4_glm5_next_kda_output_kslice_contract_test(
        uint32_t rank, uint32_t n_tokens, uint64_t *k_off,
        uint64_t *k_cnt, uint64_t *local_bytes);
/* Execute the production sparse selector at a smaller test-only top-k so the
 * pool crossover can be proven without a 2,048-token setup. */
int ds4_glm5_next_mla_sparse_attention_forward_test(
        const ds4_glm5_next_exec_ctx *ctx,
        uint32_t layer,
        ds4_glm5_next_state *state,
        ds4_glm5_next_workspace *workspace,
        const ds4_gpu_tensor *hc_in,
        ds4_gpu_tensor *hc_out,
        uint32_t top_k);
int ds4_glm5_next_mla_sparse_selection_read_test(
        const ds4_glm5_next_workspace *workspace,
        int32_t *selected,
        uint32_t count);
int ds4_glm5_next_mla_sparse_indexer_read_test(
        const ds4_glm5_next_workspace *workspace,
        uint32_t n_pools,
        uint32_t selected_count,
        float *query,
        float *weights,
        float *scores,
        uint32_t *selected_pools);
#endif

int ds4_glm5_next_embed_token(const ds4_glm5_next_exec_ctx *ctx,
                              uint32_t token,
                              ds4_gpu_tensor *hc_out);
int ds4_glm5_next_embed_tokens(const ds4_glm5_next_exec_ctx *ctx,
                               const ds4_gpu_tensor *tokens,
                               uint32_t n_tokens,
                               ds4_gpu_tensor *hc_out);

/* One-token output head. GLM-5.3 has no learned mHC output combiner: collapse
 * the four streams by their arithmetic mean, apply the model's F32 RMS norm,
 * and execute the replicated BF16 vocabulary projection. No TP exchange is
 * performed. Success means the kernels were launched; synchronize before
 * consuming logits or before reporting an execution failure. */
int ds4_glm5_next_output_logits(const ds4_glm5_next_exec_ctx *ctx,
                                ds4_glm5_next_workspace *workspace,
                                const ds4_gpu_tensor *hc_hidden,
                                ds4_gpu_tensor *logits_out);

/* Execute exactly one trunk layer. Unsupported kind combinations fail closed.
 * No workspace allocation is performed inside this call. Preconditions fail
 * without mutation; any backend failure invalidates the complete sequence. */
int ds4_glm5_next_layer_forward(const ds4_glm5_next_exec_ctx *ctx,
                                uint32_t layer,
                                ds4_glm5_next_state *state,
                                ds4_glm5_next_workspace *workspace,
                                const ds4_gpu_tensor *hc_in,
                                ds4_gpu_tensor *hc_out);

/* Exact-capacity multi-row execution. Independently validated layer kinds are
 * enabled one at a time; unsupported combinations fail before mutating
 * resident state. */
int ds4_glm5_next_layer_forward_batch(const ds4_glm5_next_exec_ctx *ctx,
                                      uint32_t layer,
                                      ds4_glm5_next_state *state,
                                      ds4_glm5_next_workspace *workspace,
                                      const ds4_gpu_tensor *hc_in,
                                      ds4_gpu_tensor *hc_out,
                                      uint32_t n_tokens);

/* Exact sparse-boundary bridge. Dense MLA and KDA layers retain the ordinary
 * batch entry. Sparse MLA attention rows use scalar_workspace in causal order,
 * then their stateless routed FFN executes in batch_workspace as one tile. The
 * scalar workspace must have token capacity one and full context capacity. */
int ds4_glm5_next_layer_forward_batch_sparse_bridge(
        const ds4_glm5_next_exec_ctx *ctx,
        uint32_t layer,
        ds4_glm5_next_state *state,
        ds4_glm5_next_workspace *batch_workspace,
        ds4_glm5_next_workspace *scalar_workspace,
        const ds4_gpu_tensor *hc_in,
        ds4_gpu_tensor *hc_out,
        uint32_t n_tokens);

#ifdef DS4_TP_TEST_HOOKS
/* Test-only decomposition gate for KDA batch recurrence. It commits exactly
 * the KDA attention half and copies the expanded mHC rows to hc_out. */
int ds4_glm5_next_kda_attention_forward_test(
        const ds4_glm5_next_exec_ctx *ctx,
        uint32_t layer,
        ds4_glm5_next_state *state,
        ds4_glm5_next_workspace *workspace,
        const ds4_gpu_tensor *hc_in,
        ds4_gpu_tensor *hc_out,
        uint32_t n_tokens);

/* Test-only decomposition gate. Executes and commits exactly the MLA
 * attention half of a metadata-validated MLA layer, copying the expanded mHC
 * result to hc_out. Production callers must use the complete layer entries. */
int ds4_glm5_next_mla_attention_forward_test(
        const ds4_glm5_next_exec_ctx *ctx,
        uint32_t layer,
        ds4_glm5_next_state *state,
        ds4_glm5_next_workspace *workspace,
        const ds4_gpu_tensor *hc_in,
        ds4_gpu_tensor *hc_out,
        uint32_t n_tokens);
#endif

#ifdef __cplusplus
}
#endif

#endif
