#ifndef DS4_GLM5_NEXT_RUNTIME_H
#define DS4_GLM5_NEXT_RUNTIME_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "ds4_glm5_kda.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    DS4_GLM5_NEXT_LAYER_COUNT = 46,
    DS4_GLM5_NEXT_TRUNK_COUNT = 45,
    DS4_GLM5_NEXT_LEADING_DENSE = 3,
    DS4_GLM5_NEXT_MLA_COUNT = 11,
    DS4_GLM5_NEXT_TP_GATE_MASK_WORDS = 3,
    /* GLM-5.3 NoPE stores only the normalized 512-wide KV-LoRA row. */
    DS4_GLM5_NEXT_MLA_KV_WIDTH = 512,
    DS4_GLM5_NEXT_INDEX_WIDTH = 128,
};

typedef enum {
    DS4_GLM5_NEXT_ATTN_INVALID = 0,
    DS4_GLM5_NEXT_ATTN_KDA = 1,
    DS4_GLM5_NEXT_ATTN_MLA = 2,
} ds4_glm5_next_attention_kind;

typedef enum {
    DS4_GLM5_NEXT_FFN_INVALID = 0,
    DS4_GLM5_NEXT_FFN_DENSE = 1,
    DS4_GLM5_NEXT_FFN_ROUTED = 2,
} ds4_glm5_next_ffn_kind;

typedef struct {
    uint64_t q_a, q_a_norm, q_b;
    uint64_t kv_a_mqa, kv_a_norm, k_b, v_b, output;
    uint64_t index_q_b, index_k, index_proj;
    uint64_t index_pool_ape, index_pool_gate;
    uint64_t index_k_norm, index_k_norm_b;
} ds4_glm5_next_mla_offsets;

typedef struct {
    uint64_t gate, up, down;
    uint64_t gate_exps, up_exps, down_exps;
    uint64_t gate_inp, exp_probs_b;
    uint64_t gate_shexp, up_shexp, down_shexp;
} ds4_glm5_next_ffn_offsets;

typedef struct {
    uint64_t attn_fn, ffn_fn;
    uint64_t attn_base, ffn_base;
    uint64_t attn_scale, ffn_scale;
} ds4_glm5_next_hc_offsets;

typedef struct {
    uint32_t layer;
    bool is_trunk;
    ds4_glm5_next_attention_kind attention;
    ds4_glm5_next_ffn_kind ffn;
    uint64_t attn_norm;
    uint64_t ffn_norm;
    ds4_glm5_kda_weight_offsets kda;
    ds4_glm5_next_mla_offsets mla;
    ds4_glm5_next_ffn_offsets ffn_weight;
    ds4_glm5_next_hc_offsets hc;
} ds4_glm5_next_layer_offsets;

typedef struct {
    uint64_t token_embd;
    uint64_t output_norm;
    uint64_t output;
    uint64_t nextn_eh_proj;
    uint32_t layer_count;
    uint32_t trunk_count;
    uint32_t nextn_count;
    ds4_glm5_next_layer_offsets layer[DS4_GLM5_NEXT_LAYER_COUNT];
} ds4_glm5_next_model_offsets;

struct ds4_glm5_next_state;

typedef struct {
    ds4_gpu_tensor *compact_kv;
    ds4_gpu_tensor *index_key;
    ds4_gpu_tensor *pool_gate;
    uint32_t capacity_tokens;
    uint32_t token_count;
    /* Pool alignment origin for this sequence. Raw key/gate caches are kept
     * so IndexPool can be rebuilt when this value changes across requests. */
    uint32_t first_valid;
    bool valid;
    struct ds4_glm5_next_state *owner;
} ds4_glm5_next_mla_state;

typedef struct ds4_glm5_next_state {
    ds4_glm5_kda_slot kda;
    ds4_glm5_next_mla_state mla[DS4_GLM5_NEXT_LAYER_COUNT];
    uint32_t layer_count;
    uint32_t context_capacity;
    uint32_t mla_count;
    uint64_t bytes;
    bool valid;
} ds4_glm5_next_state;

bool ds4_glm5_next_layer_is_mla(uint32_t layer);
int ds4_glm5_next_build_tp_gate_mask(
        uint64_t mask[DS4_GLM5_NEXT_TP_GATE_MASK_WORDS],
        uint32_t *gate_count);

int ds4_glm5_next_model_offsets_validate(
        const ds4_glm5_next_model_offsets *model);
int ds4_glm5_next_state_bytes(const ds4_glm5_next_model_offsets *model,
                              uint32_t context_capacity,
                              uint64_t *bytes);
int ds4_glm5_next_state_init(ds4_glm5_next_state *state,
                             const ds4_glm5_next_model_offsets *model,
                             uint32_t context_capacity,
                             FILE *accounting);
int ds4_glm5_next_state_reset(ds4_glm5_next_state *state);
void ds4_glm5_next_state_invalidate(ds4_glm5_next_state *state);
void ds4_glm5_next_state_free(ds4_glm5_next_state *state);

#ifdef __cplusplus
}
#endif

#endif
