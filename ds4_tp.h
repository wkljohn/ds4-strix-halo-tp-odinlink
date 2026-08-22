#ifndef DS4_TP_H
#define DS4_TP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "ds4.h"

/* Tensor-parallel transport and lockstep protocol.
 *
 * Two ranks run the same logical model, each with one contiguous half of the
 * routed experts resident. Rank 0 (leader) is a normal frontend session that
 * mirrors every ds4_session_sync()/ds4_session_eval() call to rank 1 (worker)
 * over a TCP control socket, so both engines execute the identical graph
 * sequence.
 * Inside each decoded token, partial block outputs are exchanged through a
 * registered memory slab: two-sided RDMA SEND/RECV when RDMA over
 * Thunderbolt is available, or a full-duplex TCP exchange as fallback.
 *
 * Layering: ds4.c calls the session-mirroring and slab entry points;
 * ds4_metal.m only ever sees ds4_tp_gate_exchange() through a callback
 * registered with the GPU gate machinery.  Nothing here touches tensors.
 */

typedef struct ds4_tp ds4_tp;

enum {
    DS4_TP_GATE_ATTN = 0,
    DS4_TP_GATE_FFN = 1,
    DS4_TP_GATES_PER_LAYER = 2,
    /* Max rows in a verify-block batch gate (speculative blocks are <=5). */
    DS4_TP_BATCH_MAX_ROWS = 8,
};

enum {
    DS4_TP_FEATURE_Q4K_WMMA = UINT32_C(1) << 0,
    /* The verifier attention head split changes the per-layer big-gate
     * schedule.  Put it in the exact-matched hello word so independently
     * launched ranks cannot enter different schedules and deadlock. */
    DS4_TP_FEATURE_BATCH_ATTN_HEAD_SPLIT = UINT32_C(1) << 1,
    /* OdinLink's stream-backed provider retains an arrived message until a
     * posted receive consumes it.  This permits the batch verifier to stay on
     * its RDMA channel after the transition barrier instead of rendezvousing
     * over TCP before every layer.  Real verbs providers retain their normal
     * receive-before-send protocol; the transport checks the device name. */
    DS4_TP_FEATURE_ODINLINK_BATCH_ASYNC = UINT32_C(1) << 2,
    /* Exchange the exact vocab halves through the generic verbs bulk path.
     * This preserves full logits for sampling/logprobs while keeping the
     * per-token 256 KiB payload off the management/control socket. */
    DS4_TP_FEATURE_RDMA_LOGITS = UINT32_C(1) << 3,
    /* Rank 0 computes the exact full vocabulary head and rank 1 skips it.
     * Useful when duplicating half a Q8 head is cheaper than transferring a
     * full FP32 vocabulary half over the inter-node control/data path. */
    DS4_TP_FEATURE_RANK0_FULL_LOGITS = UINT32_C(1) << 4,
    /* Exact greedy mode: retain the row-sharded output head and exchange the
     * worker's best two (id, value) pairs instead of its full FP32 half. */
    DS4_TP_FEATURE_GREEDY_TOP2 = UINT32_C(1) << 5,
    /* gfx1151 IQ2_XXS gate/up uses dynamically quantized Q8_1 activations and
     * native integer WMMA.  Keep this in the exact-matched hello word: an
     * independently launched rank must never select different MoE arithmetic. */
    DS4_TP_FEATURE_IQ2_I8_WMMA = UINT32_C(1) << 6,
    /* Both ranks must defer the same compressor projection rows.  A mismatch
     * would preserve transport health while advancing different recurrent
     * attention states. */
    DS4_TP_FEATURE_TEMPORAL_COMPRESSOR = UINT32_C(1) << 7,
    /* Bits 8..15 carry the first expert owned by rank 1.  Including the
     * ownership boundary in the already exact-matched hello features makes
     * an asymmetric placement fail closed if the two independent nodes were
     * launched with different settings. */
    DS4_TP_FEATURE_EXPERT_SPLIT_SHIFT = 8,
    DS4_TP_FEATURE_EXPERT_SPLIT_MASK = UINT32_C(0xff) <<
                                           DS4_TP_FEATURE_EXPERT_SPLIT_SHIFT,
    /* Fusing the Q4_K hot-tile SwiGLU epilogue changes which buffers are
     * materialized.  Independently launched ranks must select it together. */
    DS4_TP_FEATURE_Q4K_FUSED_MID = UINT32_C(1) << 16,
    /* The one-token HC pre-chain changes from three launches to one
     * cooperative launch while preserving the established arithmetic and
     * scratch buffers.  Require both independently launched ranks to select
     * the same path so asymmetric environment or device support fails closed. */
    DS4_TP_FEATURE_HC_STAGE_EXACT_COOP = UINT32_C(1) << 17,
    /* Above 8192 compressed index rows, gfx1151 can replace the exact
     * bitonic chunk tree with an exact packed-key radix tree.  Negotiate it
     * because independently launched ranks must select identical index rows. */
    DS4_TP_FEATURE_INDEXER_TOPK_RADIX_TREE = UINT32_C(1) << 18,
    /* DSpark verifier rows use the low-latency RC SEND/RECV framing instead
     * of the synchronous bulk protocol. Exact-match in the TP hello prevents
     * independently launched ranks from choosing different wire schedules. */
    DS4_TP_FEATURE_VERIFY_ROW_BATCH = UINT32_C(1) << 19,
    /* The DSpark attention output projection writes its rank partial directly
     * into the registered verifier slab and consumes the peer slab in the
     * rank-ordered add. This changes the fixed gate/buffer schedule and must
     * therefore match exactly across independently launched ranks. */
    DS4_TP_FEATURE_VERIFY_ATTN_SLAB = UINT32_C(1) << 20,
    /* DSpark Q4_K verifier batches deduplicate live expert gate/up reads and
     * preserve the canonical six-slot down fold. Both independently launched
     * ranks must enter this arithmetic/scratch schedule together. */
    DS4_TP_FEATURE_Q4K_VERIFY_FIRST_OWNER = UINT32_C(1) << 21,
    /* DSpark verifier attention-output Q8_0 projections reuse each owned
     * weight tile across rows while retaining the one-row arithmetic tree.
     * Negotiate the scratch/launch schedule across independent ranks. */
    DS4_TP_FEATURE_Q8_ATTN_OUT_WEIGHT_OUTER = UINT32_C(1) << 22,
    /* The exact DSpark indexed-attention verifier packs two independent
     * 256-thread head reductions into one gfx1151 workgroup and batches the
     * verifier rows.  Both ranks must select the same launch schedule. */
    DS4_TP_FEATURE_DSPARK_EXACT_ATTN_HEAD2 = UINT32_C(1) << 23,
    /* The DSpark target head replaces a serial vocabulary top-1 scan with
     * one exact parallel reduction per verifier row.  Negotiate it so both
     * independently launched ranks retain one fail-closed execution profile. */
    DS4_TP_FEATURE_DSPARK_BATCH_ARGMAX = UINT32_C(1) << 24,
    /* DSpark Q4_K verifier down projection groups repeated expert rows while
     * retaining the shipped two-half wave reduction and slot fold. */
    DS4_TP_FEATURE_Q4K_VERIFY_DOWN_FIRST_OWNER = UINT32_C(1) << 25,
    /* DSpark TP verifier shared-Q8 rows reuse compact weights across exact
     * gate/up/SwiGLU and rank K-sliced down projections. */
    DS4_TP_FEATURE_DSPARK_SHARED_Q8_ROWS_EXACT = UINT32_C(1) << 26,
};

static inline uint32_t ds4_tp_feature_expert_split(uint32_t first_rank1) {
    return (first_rank1 & UINT32_C(0xff)) <<
           DS4_TP_FEATURE_EXPERT_SPLIT_SHIFT;
}

static inline uint32_t ds4_tp_feature_expert_split_value(uint32_t features) {
    return (features & DS4_TP_FEATURE_EXPERT_SPLIT_MASK) >>
           DS4_TP_FEATURE_EXPERT_SPLIT_SHIFT;
}

static inline bool ds4_tp_runtime_features_equal(uint32_t local,
                                                  uint32_t peer) {
    return local == peer;
}

/* Engine identity exchanged in the hello so a mismatched pair aborts before
 * any inference runs. */
typedef struct {
    uint64_t gguf_bytes;
    uint32_t model_id;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_vocab;
    uint32_t quant_bits;
    uint32_t ctx_size;
    uint32_t runtime_features;
    /* Decode gate schedule, used to place RDMA recvs into the right slab
     * slot: slot(seq) = start + ((seq-1) % per_token) * step.
     * per_token 0 falls back to the identity mapping over all slots
     * (DS4: every layer fires ATTN then FFN). GLM fires one FFN gate per
     * sparse layer only, so its schedule skips the dense prefix and the
     * ATTN slots. Exchanged in the hello; both sides must agree. */
    uint32_t gate_slot_start;
    uint32_t gate_slot_step;
    uint32_t gates_per_token;
} ds4_tp_identity;

/* DS4-TP-gfx1151 (patch 21): device-copy hook for big-gate staging.
 *
 * The big gate stages through the registered slab with memcpy when the caller's
 * buffers are outside it (`direct == 0`, which is ALWAYS the case for prefill:
 * batch_routed_out is an ordinary graph tensor). Those memcpys are CPU reads of
 * hipMalloc device memory, which on this UMA APU is host-reachable but
 * write-combining. MEASURED: 200 MB/s, 64% of all big-gate time.
 *
 * The backend registers a device-side copy here so the GPU's DMA engines do it
 * instead - they are idle, since the GPU is parked in hipStreamWaitValue64.
 * Returns non-zero on success; ds4_tp.c falls back to memcpy when unset or on
 * failure, so this stays optional and platform-free. */
typedef int (*ds4_tp_devcopy_fn)(void *dst, const void *src, uint64_t bytes);
void ds4_tp_set_devcopy(ds4_tp_devcopy_fn fn);

bool ds4_tp_enabled(const ds4_tp_options *opt);

typedef enum {
    DS4_TP_CLI_ERROR = -1,
    DS4_TP_CLI_NOT_MATCHED = 0,
    DS4_TP_CLI_MATCHED = 1,
} ds4_tp_cli_parse_result;

/* CLI parsing, same contract as ds4_dist_parse_cli_arg(): returns 1 when the
 * argument was consumed, 0 when not matched, -1 on error (err filled). */
int ds4_tp_parse_cli_arg(
        const char *arg,
        int *index,
        int argc,
        char **argv,
        ds4_tp_options *opt,
        char *err,
        size_t errlen);
int ds4_tp_adopt_distributed_options(
        ds4_tp_options *tp,
        ds4_distributed_options *dist,
        char *err,
        size_t errlen);
void ds4_tp_usage(FILE *fp);

/* Validates option combinations that TP cannot run with (SSD streaming,
 * distributed mode, MTP drafting, CPU backend). */
int ds4_tp_validate_engine_options(
        const ds4_engine_options *opt,
        char *err,
        size_t errlen);

/* Connection bring-up.  The leader listens and accepts one worker; the
 * worker dials with retry.  Both then exchange and validate identities.
 * Blocking; call after the engine is loaded (identity needs the shape). */
int ds4_tp_create(
        ds4_tp **out,
        const ds4_tp_options *opt,
        const ds4_tp_identity *id,
        char *err,
        size_t errlen);
void ds4_tp_free(ds4_tp *tp);

int ds4_tp_rank(const ds4_tp *tp);
bool ds4_tp_is_rdma(const ds4_tp *tp);
/* True only when large batch gates use verbs instead of TCP fallback. */
bool ds4_tp_big_gate_is_rdma_capable(const ds4_tp *tp);
/* True only when the selected provider requires a host-pinned GPU-visible
 * slab for NIC registration. OdinLink keeps its existing device allocation. */
bool ds4_tp_requires_host_slab(const ds4_tp *tp);
uint32_t ds4_tp_peer_ctx(const ds4_tp *tp);
uint32_t ds4_tp_runtime_features(const ds4_tp *tp);
#ifdef DS4_TP_TEST_HOOKS
int ds4_tp_test_hello_validate_runtime_features(uint32_t local, uint32_t peer,
                                                char *err, size_t errlen);
int ds4_tp_test_select_transport(ds4_tp_transport requested,
                                 int local_rdma_ok,
                                 int peer_rdma_ok,
                                 int *rdma_active,
                                 char *err,
                                 size_t errlen);
uint32_t ds4_tp_test_rdma_provider_decode_max_msg(const char *device_name);
uint32_t ds4_tp_test_rdma_negotiate_decode_max_msg(uint32_t local,
                                                   uint32_t peer);
uint64_t ds4_tp_test_connect_timeout_sec(void);
#endif
bool ds4_tp_failed(const ds4_tp *tp);
void ds4_tp_mark_failed(ds4_tp *tp);

/* Gate slab.  The engine allocates one shared GPU-visible block and hands
 * its base VA here; ds4_tp registers it with the NIC (RDMA) and exchanges
 * remote keys.  Layout, all offsets from base, S = n_layer * 2 slots:
 *
 *   out vectors   S * vec_bytes   written by local GPU kernels
 *   in  vectors   S * vec_bytes   RDMA/TCP-written with the peer partials
 *   in  seq flags S * 8           written strictly after each in vector
 *   token slot    16              {seq u64, token i32, pad} leader->worker
 *   (gpu flags, then batch out/in: n_layer * BATCH_MAX_ROWS * vec_bytes
 *    each, row partials for the speculative verify-block gates)
 *   (big out/in: DS4_TP_BIG_DIRECT_MAX_ROWS * vec_bytes each, opt-in via
 *    DS4_TP_BIG_DIRECT=1, zero bytes otherwise - see ds4_tp.c)
 *
 * vec_bytes = n_embd * 4 (f32 partials, never quantized on the wire). */
uint64_t ds4_tp_slab_bytes(uint32_t n_layer, uint32_t n_embd);
/* Provider-adjusted size after ds4_tp_create() selects a verbs device. */
uint64_t ds4_tp_alloc_slab_bytes(const ds4_tp *tp);
uint64_t ds4_tp_slab_out_offset(const ds4_tp *tp, uint32_t layer, uint32_t gate);
uint64_t ds4_tp_slab_in_offset(const ds4_tp *tp, uint32_t layer, uint32_t gate);
uint64_t ds4_tp_slab_batch_out_offset(const ds4_tp *tp, uint32_t layer);
uint64_t ds4_tp_slab_batch_in_offset(const ds4_tp *tp, uint32_t layer);
uint64_t ds4_tp_slab_gpu_flags_offset(const ds4_tp *tp);
/* direct=1 prefill big-gate regions (opt-in, DS4_TP_BIG_DIRECT=1). A single
 * region each, not per-layer: the big gate reuses one buffer across layers
 * within a prefill chunk, same as the ordinary batch_routed_out/batch_ffn_out
 * tensors it replaces. ds4_tp_big_capacity_rows() returns 0 when the feature
 * is off; callers must not use the offsets in that case. */
uint64_t ds4_tp_slab_big_out_offset(const ds4_tp *tp);
uint64_t ds4_tp_slab_big_in_offset(const ds4_tp *tp);
uint32_t ds4_tp_big_capacity_rows(const ds4_tp *tp);
int ds4_tp_attach_slab(ds4_tp *tp, void *base, char *err, size_t errlen);

/* Exchange one gate: send out[layer][gate] to the peer's in[layer][gate]
 * and wait until the peer's partial for `seq` has fully landed locally.
 * Called from the GPU gate service thread.  Returns 0 on failure. */
int ds4_tp_gate_exchange(ds4_tp *tp, uint32_t layer, uint32_t gate, uint64_t seq);

/* Verify-block batch gate: exchange `rows` row partials for one layer in one
 * bulk RDMA transfer, with a symmetric TCP transfer as fallback. Called from
 * the GPU gate service thread. */
int ds4_tp_batch_gate_exchange(ds4_tp *tp, uint32_t layer, uint32_t rows,
                               uint64_t seq);

/* Prefill batch gate: arbitrary-size symmetric payload exchange over bulk
 * RDMA, with interleaved 2MB TCP rounds as fallback (see ds4_tp.c). */
int ds4_tp_big_gate_exchange(ds4_tp *tp, uint32_t layer, uint64_t seq,
                             const void *out, void *in, uint64_t bytes);
typedef void (*ds4_tp_big_wave_ready_fn)(void *ud, uint32_t wave);
int ds4_tp_big_gate_exchange_waves(ds4_tp *tp, uint32_t layer, uint64_t seq,
                                   const void *out, void *in, uint64_t bytes,
                                   uint64_t wave_bytes, uint32_t waves,
                                   ds4_tp_big_wave_ready_fn ready,
                                   void *ready_ud);

/* Lockstep mirroring (leader side) and worker loop primitives. */
typedef struct {
    uint64_t session_id;
    int32_t token;
    uint32_t reserved;
} ds4_tp_batch_item;

int ds4_tp_send_session_create(ds4_tp *tp, uint64_t session_id, int ctx_size);
int ds4_tp_send_session_destroy(ds4_tp *tp, uint64_t session_id);
int ds4_tp_send_sync(ds4_tp *tp, uint64_t session_id,
                     const int *tokens, uint32_t n_tokens);
int ds4_tp_send_eval(ds4_tp *tp, uint64_t session_id,
                     uint64_t seq, int token);
int ds4_tp_send_rewind(ds4_tp *tp, uint64_t session_id, int pos);
int ds4_tp_send_invalidate(ds4_tp *tp, uint64_t session_id);
int ds4_tp_send_eval_batch(ds4_tp *tp, const ds4_tp_batch_item *items,
                           uint32_t count);
int ds4_tp_send_mixed_batch(ds4_tp *tp, uint64_t prefill_session_id,
                            const int *prompt, uint32_t prompt_count,
                            const ds4_tp_batch_item *items,
                            uint32_t count);
int ds4_tp_send_command_ack(ds4_tp *tp, uint64_t session_id, int status);
int ds4_tp_wait_command_ack(ds4_tp *tp, uint64_t session_id,
                            const char *operation, char *err, size_t errlen);
int ds4_tp_send_stop(ds4_tp *tp);

/* Worker: blocks for the next mirrored command.  Frame types below; for
 * DS4_TP_FRAME_SYNC the token array is returned in *tokens / *n_tokens
 * (malloc'd, caller frees), for DS4_TP_FRAME_EVAL seq/token are filled. */
typedef enum {
    DS4_TP_FRAME_ERROR = -1,
    DS4_TP_FRAME_SYNC = 1,
    DS4_TP_FRAME_EVAL = 2,
    DS4_TP_FRAME_REWIND = 3,
    DS4_TP_FRAME_INVALIDATE = 4,
    DS4_TP_FRAME_STOP = 5,
    DS4_TP_FRAME_HASH = 6,
    DS4_TP_FRAME_RDMA_INFO = 7,
    DS4_TP_FRAME_SYNC_ACK = 8,
    DS4_TP_FRAME_RDMA_READY = 9,
    DS4_TP_FRAME_LOGITS = 10,
    DS4_TP_FRAME_VERIFY = 11,
    DS4_TP_FRAME_VERIFY_COMMIT = 12,
    DS4_TP_FRAME_SESSION_CREATE = 13,
    DS4_TP_FRAME_SESSION_DESTROY = 14,
    DS4_TP_FRAME_EVAL_BATCH = 15,
    DS4_TP_FRAME_MIXED_BATCH = 16,
    DS4_TP_FRAME_COMMAND_ACK = 17,
    DS4_TP_FRAME_LOGITS_TOP2 = 18,
} ds4_tp_frame_type;

typedef struct {
    int32_t id[2];
    float value[2];
} ds4_tp_logits_top2;

typedef struct {
    ds4_tp_frame_type type;
    uint64_t session_id;
    uint64_t seq;
    int value;
    int *tokens;
    uint32_t n_tokens;
    ds4_tp_batch_item *items;
    uint32_t n_items;
} ds4_tp_command;

int ds4_tp_recv_command(
        ds4_tp *tp,
        ds4_tp_command *command,
        char *err,
        size_t errlen);
void ds4_tp_command_free(ds4_tp_command *command);

/* Debug lockstep check: both sides send their hidden-state hash for a token
 * and compare.  Returns 0 on transport failure, -1 on hash mismatch. */
int ds4_tp_hash_check(ds4_tp *tp, uint64_t seq, uint64_t hash, char *err, size_t errlen);

/* Vocab-split output head: the worker ships its logits half to the leader
 * after every eval (and after a sync) on the control socket. */
int ds4_tp_send_logits_half(ds4_tp *tp, const float *half, uint32_t count);
int ds4_tp_recv_logits_half(ds4_tp *tp, float *half, uint32_t count);
int ds4_tp_exchange_logits_halves(ds4_tp *tp, float *logits,
                                  uint32_t half_count);
int ds4_tp_send_logits_top2(ds4_tp *tp, const ds4_tp_logits_top2 *top2);
int ds4_tp_recv_logits_top2(ds4_tp *tp, ds4_tp_logits_top2 *top2);

/* Speculative verify mirroring.  The leader announces a draft block right
 * before both ranks run the expert-split batch verify; the worker then blocks
 * on the commit frame, which carries the leader's decision: full_accept keeps
 * the pushed rows, otherwise both sides roll back and replay replay_n tokens
 * through the gated single-token decode in lockstep. */
int ds4_tp_send_verify(ds4_tp *tp, uint64_t session_id,
                       const int *drafts, uint32_t n);
int ds4_tp_send_verify_commit(ds4_tp *tp, int32_t full_accept, int32_t replay_n);
int ds4_tp_recv_verify_commit(ds4_tp *tp, int32_t *full_accept, int32_t *replay_n);

/* Standalone worker mode entry. Loads nothing itself: the engine is already
 * open. */
int ds4_tp_worker_run(ds4_engine *engine, const ds4_tp_options *opt);

#endif
