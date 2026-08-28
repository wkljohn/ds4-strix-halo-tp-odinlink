#ifndef DS4_GLM5_NEXT_EXEC_H
#define DS4_GLM5_NEXT_EXEC_H

#include <stdint.h>

#include "ds4_glm5_next_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ds4_tp;
typedef struct ds4_glm5_next_workspace ds4_glm5_next_workspace;

typedef struct {
    const void *model_map;
    uint64_t model_size;
    const ds4_glm5_next_model_offsets *model;
    struct ds4_tp *tp;
    uint32_t tp_rank;
    ds4_gpu_tensor *tp_big_out;
    ds4_gpu_tensor *tp_big_in;
    void *tp_big_out_host;
    void *tp_big_in_host;
    /* Transport-global, monotonically increasing big-gate sequence.  It is
     * deliberately not reset with a model sequence while the TP link lives. */
    uint64_t *tp_sequence;
} ds4_glm5_next_exec_ctx;

/* The initial production slice is a one-token decode workspace.  A later
 * prefill slice must parameterize capacity instead of silently reusing it. */
ds4_glm5_next_workspace *ds4_glm5_next_workspace_create(void);
void ds4_glm5_next_workspace_destroy(ds4_glm5_next_workspace *workspace);

int ds4_glm5_next_embed_token(const ds4_glm5_next_exec_ctx *ctx,
                              uint32_t token,
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

#ifdef __cplusplus
}
#endif

#endif
