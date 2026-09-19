#ifndef DS4_GLM5_NEXT_EXEC_H
#define DS4_GLM5_NEXT_EXEC_H

#include <stdint.h>

#include "ds4_glm5_next_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ds4_tp;
typedef struct ds4_glm5_next_workspace ds4_glm5_next_workspace;

/* Research leaf for unchanged Q8_0 dense FFNs, M=2/4/6/8: paired K4096/N12288
 * gate/up (out1 non-NULL), or single K12288/N4096 down (out1 NULL).
 * Requires production Q8 tile mode 1 and exact shared-X prefetch=8 settings.
 * Independent contiguous F32 tensors; no GPU allocation or weight expansion.
 * Unsupported shapes/settings/backends return zero, with no retry. */
int ds4_rocm_glm5_dense_q8_small_m(
        ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
        const void *model_map, uint64_t model_size,
        uint64_t offset0, uint64_t offset1,
        uint32_t in_dim, uint32_t out_dim,
        const ds4_gpu_tensor *x, uint32_t tokens);

/* Production-unused shared-expert Q8_0 leaf, M=2/4/6/8, same settings as above.
 * Pair: rank-local gate/up K4096/N1024, row_bytes=4352, k_first=0.
 * Single: down K1024/N4096, full row_bytes=2176, k_first=0 or1024;
 * offset0 names the unsliced tensor, offset1 must be zero. Original strides
 * and scalar scale*code rounding are retained; no persistent weight copy.
 * Caller must establish Q8_0 tensor types. Invalid shapes/bounds/overlap refuse. */
int ds4_rocm_glm5_shared_q8_small_m(
        ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
        const void *model_map, uint64_t model_size,
        uint64_t offset0, uint64_t offset1,
        uint32_t in_dim, uint32_t out_dim, uint64_t row_bytes,
        uint32_t k_first, const ds4_gpu_tensor *x, uint32_t tokens);

/* Production-unused MLA input-projection leaf, M2/4/6, original resident Q8_0.
 * Q_a: K4096/N1536; KV_a: K4096/N512; both full rows, row_first=0.
 * Q_b: K1536/fullN16384, localN8192, row_first=rank*8192. Contiguous
 * F32 inputs/outputs and original full Q8 row_bytes are mandatory. Caller
 * supplies independently validated GGUF weight_type=8. No allocation, retry
 * or weight cache; no automatic production dispatch. The supported query
 * performs the same admission without launching or mutating output. */
int ds4_rocm_glm5_mla_prelude_q8_small_m_supported(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t offset, uint32_t weight_type, uint32_t in_dim,
        uint32_t full_out_dim, uint32_t row_first, uint32_t out_dim,
        uint64_t row_bytes, const ds4_gpu_tensor *x, uint32_t tokens);
int ds4_rocm_glm5_mla_prelude_q8_small_m(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t offset, uint32_t weight_type, uint32_t in_dim,
        uint32_t full_out_dim, uint32_t row_first, uint32_t out_dim,
        uint64_t row_bytes, const ds4_gpu_tensor *x, uint32_t tokens);

/* Research MLA output leaf: resident original Q8_0, M=2/4/6 only.
 * Full K16384 row_bytes17408, local K8192/N4096, k_first0/8192.
 * x must contain packed contiguous local rows, not full 64-head rows.
 * Caller proves Q8_0 tensor type and k_first == rank*8192. Requires the
 * exact mode1 recipe above; no allocation, retry or M8 dispatch. */
int ds4_rocm_glm5_mla_output_q8_small_m(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t offset, uint32_t full_in_dim, uint32_t k_first,
        uint32_t in_dim, uint32_t out_dim, uint64_t row_bytes,
        const ds4_gpu_tensor *x, uint32_t tokens);
/* Same admission without a kernel launch or lazy weight registration/copy.
 * Call before private-state mutation; a later launch failure is terminal. */
int ds4_rocm_glm5_mla_output_q8_small_m_supported(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t offset, uint32_t full_in_dim, uint32_t k_first,
        uint32_t in_dim, uint32_t out_dim, uint64_t row_bytes,
        const ds4_gpu_tensor *x, uint32_t tokens);

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
    /* Internal verifier phase: every payload uses the registered bulk path.
     * Scalar decode's auxiliary receive is paired with its latency gate and
     * cannot service batched verification. Zero preserves ordinary dispatch. */
    bool force_bulk_gates;
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
/* Session admission after resident spans and exact verifier workspaces exist.
 * Checks every trunk MLA output without enqueue, allocation or state mutation. */
int ds4_glm5_next_mla_output_batch_supported(const ds4_glm5_next_exec_ctx *ctx,
        const ds4_glm5_next_workspace *workspace);
/* Checks every resident routed layer for the exact six-row expert schedule. */
int ds4_glm5_next_expert_pairs_supported(const ds4_glm5_next_exec_ctx *ctx,
        const ds4_glm5_next_workspace *workspace);

/* Explicit native block45 workspace: one scalar activation set and one
 * bounded eight-expert packed window, invalidated before each reuse. */
ds4_glm5_next_workspace *ds4_glm5_next_draft_workspace_create(uint32_t context_capacity);
/* Teacher warming uses one exact row tile (1..256); no expert window is
 * allocated by warming. A one-row workspace can subsequently draft. */
ds4_glm5_next_workspace *ds4_glm5_next_draft_workspace_create_rows(
        uint32_t tokens, uint32_t context_capacity);
/* Target mHC -> contracted, post-output-norm hidden. Chained draft hidden is
 * already post-shared-head-norm and must not pass through this conversion.
 * Input/output contain exactly workspace-capacity rows. */
int ds4_glm5_next_draft_target_hidden(const ds4_glm5_next_exec_ctx *ctx,
        ds4_glm5_next_workspace *workspace, const ds4_gpu_tensor *hc_hidden,
        ds4_gpu_tensor *normalized_hidden);
/* Bounded view of the same conversion, using 1..workspace-capacity rows.
 * Exact-size input/output tiles let teacher warming stay bounded independently
 * of the target prefill batch. Allocates view metadata only, no GPU storage. */
int ds4_glm5_next_draft_target_hidden_rows(const ds4_glm5_next_exec_ctx *ctx,
        ds4_glm5_next_workspace *workspace, const ds4_gpu_tensor *hc_hidden,
        ds4_gpu_tensor *normalized_hidden, uint32_t rows);
/* Consume shifted teacher rows into live private KV/index state only. No
 * attention query/output, FFN, vocabulary head, payload exchange or proposal
 * selection. Exact-size normalized predecessor rows and host token IDs.
 * Scalar warming matches a full teacher draft step bitwise; multirow GEMM
 * may reorder draft arithmetic. A pending speculative journal is refused.
 * Same owner/model/link binding and invalidation rules as draft_step. */
int ds4_glm5_next_draft_warm_rows(const ds4_glm5_next_exec_ctx *ctx,
        ds4_glm5_next_state *draft_state, ds4_glm5_next_workspace *workspace,
        const ds4_gpu_tensor *normalized_predecessors,
        const uint32_t *shifted_tokens, uint32_t tokens);
/* A shifted token plus normalized predecessor hidden consumes one private
 * MLA row, returning normalized draft hidden and full logits. Caller owns
 * distinct F32 input/hidden/logit buffers and the draft-only state. Pass its
 * live mla[45] for teacher warming or a journal view for a speculative chain.
 * Registered bulk exchanges only; all trunk experts must remain resident.
 * One workspace binds to one owner/model/link on its first step, including
 * across state resets. Model offsets and weights remain immutable; destroy
 * the workspace before freeing that state or releasing model residency.
 * On any execution failure the entire private owner is invalidated. This
 * interface does not publish session history or select an accepted prefix. */
int ds4_glm5_next_draft_step(const ds4_glm5_next_exec_ctx *ctx,
        ds4_glm5_next_mla_state *draft_state, ds4_glm5_next_workspace *workspace,
        const ds4_gpu_tensor *previous_normalized_hidden, uint32_t shifted_token,
        ds4_gpu_tensor *normalized_hidden, ds4_gpu_tensor *logits);

/* Retire a completed M-1 native proposal journal and rebuild K accepted
 * input rows from actual normalized target hidden, using scalar teacher
 * arithmetic. Before the cycle target consumed N inputs, private draft N-1.
 * target_hidden contains exactly M rows (M=2/4/6/8), input_tokens the verified
 * [root, proposals...], and K includes root (1..M). previous_hidden initially
 * holds target h[N-1]. Warm [previous, target_hidden[0..K-2]] with inputs[0..K-1],
 * then replace previous_hidden with target_hidden[K-1]. The correction/bonus
 * token remains unconsumed. A scalar draft-only workspace and independent
 * target/previous buffers are required; no GPU storage or payload exchange.
 * Preflight refusal preserves the journal. Any execution failure invalidates
 * the private owner; caller must also invalidate its target/session. This
 * rank-local helper does not select K, verify tokens, commit target state or
 * publish history. Both ranks must agree on those outcomes before continuing.
 * Keep target hidden and input tokens immutable throughout the call. */
int ds4_glm5_next_draft_refresh(const ds4_glm5_next_exec_ctx *ctx,
        ds4_glm5_next_state *draft_state, ds4_glm5_next_workspace *workspace,
        const ds4_gpu_tensor *target_hidden, const uint32_t *input_tokens,
        uint32_t target_prefix, uint32_t verified_rows, uint32_t accepted_inputs,
        ds4_gpu_tensor *previous_hidden);

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

/* Research target verifier. Explicit reservations precede execution; no
 * journal allocation is performed by layer_verify. Batch only the proven
 * serial-exact KDA projections/recurrence; other stages retain scalar decode
 * arithmetic. KDA requires BF16, SMALL_M_EXACT=1 and TP output row sharding.
 * Success leaves a pending layer with live state unchanged. The all-layer
 * caller must coordinate one accepted input count, then finish every layer;
 * on any failed pass invalidate the whole state instead of accepting it.
 * This interface does not draft tokens or commit session/logit history. */
int ds4_glm5_next_layer_verify_reserve(const ds4_glm5_next_exec_ctx *ctx,
                                       uint32_t layer,
                                       ds4_glm5_next_state *state,
                                       uint32_t capacity);
int ds4_glm5_next_layer_verify(const ds4_glm5_next_exec_ctx *ctx,
                               uint32_t layer,
                               ds4_glm5_next_state *state,
                               ds4_glm5_next_workspace *batch_workspace,
                               ds4_glm5_next_workspace *scalar_workspace,
                               const ds4_gpu_tensor *hc_in,
                               ds4_gpu_tensor *hc_out,
                               uint32_t n_tokens);
int ds4_glm5_next_layer_verify_finish(const ds4_glm5_next_exec_ctx *ctx,
                                      uint32_t layer,
                                      ds4_glm5_next_state *state,
                                      uint32_t accepted_inputs);

/* Research full-target transaction. Reserve every trunk journal first;
 * partial reservation failure retains bounded storage owned/freed by state.
 * Caller owns exact M-row hidden scratch, hidden output and vocabulary logits
 * (M=2/4/6/8), plus distinct scalar/decode and M-row workspaces. The wrapper
 * allocates no GPU storage; scalar FFNs retain their ordinary residency policy.
 * Preload the production compressed slices before an allocation-free timed
 * pass. Caller buffers must be allocated independently of workspace/state
 * storage. The API rejects overlapping supplied tensor ranges and TP ranges;
 * it does not inspect every workspace/state allocation or parent allocation.
 * Input IDs are validated before any work; successful verification computes
 * all 45 layers and every output head, synchronizes, and leaves live state
 * pending at its original common frontier. The model, settings and transport
 * resources must stay alive and unchanged through finish. Ordinary calls and
 * individual layer verification/finish are excluded while active.
 *
 * Both ranks must supply the same accepted input count (root + accepted
 * drafts); the correction/bonus prediction is not yet consumed. Zero discards
 * a completed pass. Finish validates all journals before committing any;
 * backend failure invalidates the whole sequence. Reset/invalidation discards
 * an incomplete pass. Neither operation rewinds transport sequence counters.
 * A preflight refusal leaves the pass pending: restore its binding to finish
 * (including finish(0)), or reset the whole context. No context-preserving
 * abort is supported after a lost binding or an incomplete pass.
 * These are rank-local transactions. Before publishing history/logits/tokens
 * or resuming inference, the coordinator must establish that BOTH ranks
 * verified successfully and then finished successfully with its chosen count.
 * Any rank failure, disagreement or lost acknowledgement invalidates BOTH
 * sequence states; a local success alone never permits publication.
 * Session history, EOS, drafting and coordinator acceptance are caller work;
 * these APIs alone do not enable speculative session execution.
 * DS4_ROCM_GLM5_BF16_VERIFY_HEAD=1 opts into exact batched BF16 vocabulary
 * projection, reusing batch activation scratch. Other head types stay scalar;
 * the ordinary single-token output API is unchanged.
 * DS4_ROCM_GLM5_VERIFY_DENSE_Q8=1 batches the three dense FFNs' Q8 matrices
 * after scalar mHC preparation. Requires the research leaf settings above;
 * uses existing batch scratch and leaves ordinary/prefill selection intact. */
int ds4_glm5_next_target_verify_reserve(const ds4_glm5_next_exec_ctx *ctx,
                                        ds4_glm5_next_state *state,
                                        uint32_t capacity);
int ds4_glm5_next_target_verify(const ds4_glm5_next_exec_ctx *ctx,
                                ds4_glm5_next_state *state,
                                ds4_glm5_next_workspace *batch_workspace,
                                ds4_glm5_next_workspace *scalar_workspace,
                                const uint32_t *input_tokens, uint32_t n_tokens,
                                ds4_gpu_tensor *hc_scratch,
                                ds4_gpu_tensor *hc_out,
                                ds4_gpu_tensor *logits_out);
int ds4_glm5_next_target_verify_finish(const ds4_glm5_next_exec_ctx *ctx,
                                       ds4_glm5_next_state *state,
                                       uint32_t accepted_inputs);

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
