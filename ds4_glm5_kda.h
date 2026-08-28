#ifndef DS4_GLM5_KDA_H
#define DS4_GLM5_KDA_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "ds4_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    DS4_GLM5_KDA_CHANNELS = 8192,
    DS4_GLM5_KDA_HISTORY = 3,
    DS4_GLM5_KDA_HEADS = 64,
    DS4_GLM5_KDA_HEAD_DIM = 128,
    DS4_GLM5_KDA_MAX_SLOTS = 1,
};

typedef struct {
    uint32_t layer;
    bool is_kda;
} ds4_glm5_layer_kind;

struct ds4_glm5_kda_slot;

typedef struct {
    ds4_gpu_tensor *q_history;
    ds4_gpu_tensor *k_history;
    ds4_gpu_tensor *v_history;
    ds4_gpu_tensor *recurrent;
    uint64_t token_count;
    bool valid;
    struct ds4_glm5_kda_slot *owner_slot;
} ds4_glm5_kda_layer_state;

typedef struct ds4_glm5_kda_slot {
    ds4_glm5_kda_layer_state *layer;
    uint32_t layer_count;
    uint32_t kda_count;
    uint64_t bytes;
    bool valid;
} ds4_glm5_kda_slot;

typedef struct {
    uint64_t attn_norm, q, k, v, output;
    uint64_t q_conv, k_conv, v_conv;
    uint64_t f_a, f_b, g_a, g_b, beta, o_norm, dt_bias, a_log;
} ds4_glm5_kda_weight_offsets;

typedef struct {
    ds4_gpu_tensor *norm, *q, *k, *v, *f_low, *forget;
    ds4_gpu_tensor *beta, *recurrent_out;
    uint32_t capacity_tokens;
    uint64_t bytes;
} ds4_glm5_kda_workspace;

typedef struct {
    const ds4_glm5_kda_weight_offsets *weights;
    const void *model_map;
    uint64_t model_size;
    ds4_glm5_kda_layer_state *state;
    ds4_glm5_kda_workspace *workspace;
    const ds4_gpu_tensor *input;
    ds4_gpu_tensor *output;
    uint32_t n_tokens;
    float norm_eps;
} ds4_glm5_kda_device_args;

/* Test-only state comparison payload. Producing it performs device readback;
 * ordinary inference must never call this path. */
typedef struct {
    uint64_t output_fnv64;
    uint64_t q_history_fnv64;
    uint64_t k_history_fnv64;
    uint64_t v_history_fnv64;
    uint64_t recurrent_fnv64;
    uint64_t token_count;
} ds4_glm5_kda_digest;

int ds4_glm5_kda_state_bytes(uint64_t kda_count, uint32_t slot_count,
                             uint64_t *bytes);
int ds4_glm5_kda_build_schedule(ds4_glm5_layer_kind *out,
                                uint32_t capacity,
                                const bool *has_kda_q,
                                const bool *has_mla_q,
                                uint32_t layer_count,
                                uint32_t *kda_count);
int ds4_glm5_kda_slot_init(ds4_glm5_kda_slot *slot,
                           const ds4_glm5_layer_kind *schedule,
                           uint32_t layer_count,
                           uint32_t slot_count,
                           FILE *accounting);
int ds4_glm5_kda_slot_reset(ds4_glm5_kda_slot *slot);
void ds4_glm5_kda_slot_invalidate(ds4_glm5_kda_slot *slot);
void ds4_glm5_kda_slot_free(ds4_glm5_kda_slot *slot);
int ds4_glm5_kda_workspace_init(ds4_glm5_kda_workspace *workspace,
                                uint32_t capacity_tokens);
int ds4_glm5_kda_workspace_bytes(uint32_t capacity_tokens, uint64_t *bytes);
void ds4_glm5_kda_workspace_free(ds4_glm5_kda_workspace *workspace);
int ds4_glm5_kda_layer_forward(ds4_glm5_kda_layer_state *state,
                               ds4_glm5_kda_workspace *workspace,
                               const ds4_glm5_kda_weight_offsets *weights,
                               const void *model_map,
                               uint64_t model_size,
                               const ds4_gpu_tensor *input,
                               ds4_gpu_tensor *output,
                               uint32_t n_tokens,
                               float norm_eps);
int ds4_glm5_kda_layer_digest(const ds4_glm5_kda_layer_state *state,
                              const ds4_gpu_tensor *output,
                              uint64_t output_floats,
                              ds4_glm5_kda_digest *digest);
int ds4_glm5_kda_digest_equal(const ds4_glm5_kda_digest *rank0,
                              const ds4_glm5_kda_digest *rank1);

/* Internal backend adapter. Non-ROCm builds resolve the weak fail-closed
 * implementation in ds4_glm5_kda.c. */
int ds4_rocm_glm5_kda_layer_execute(
        const ds4_glm5_kda_device_args *args);

#if defined(DS4_GLM5_KDA_TEST_HOOKS)
enum {
    DS4_GLM5_KDA_FAIL_NONE = 0,
    DS4_GLM5_KDA_FAIL_INPUT_NORM,
    DS4_GLM5_KDA_FAIL_Q_PROJECTION,
    DS4_GLM5_KDA_FAIL_K_PROJECTION,
    DS4_GLM5_KDA_FAIL_V_PROJECTION,
    DS4_GLM5_KDA_FAIL_Q_CONV,
    DS4_GLM5_KDA_FAIL_K_CONV,
    DS4_GLM5_KDA_FAIL_V_CONV,
    DS4_GLM5_KDA_FAIL_GATE_PREP,
    DS4_GLM5_KDA_FAIL_RECURRENCE,
    DS4_GLM5_KDA_FAIL_GATED_NORM,
    DS4_GLM5_KDA_FAIL_OUTPUT_PROJECTION,
};
void ds4_glm5_kda_test_fail_after(uint32_t stage);
int ds4_glm5_kda_test_should_fail(uint32_t stage);
#else
static inline int ds4_glm5_kda_test_should_fail(uint32_t stage) {
    (void)stage;
    return 0;
}
#endif

#ifdef __cplusplus
}
#endif

#endif
