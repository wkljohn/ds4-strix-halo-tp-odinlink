#ifndef DS4_GLM5_KDA_H
#define DS4_GLM5_KDA_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "ds4_gpu.h"

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

typedef struct {
    ds4_gpu_tensor *q_history;
    ds4_gpu_tensor *k_history;
    ds4_gpu_tensor *v_history;
    ds4_gpu_tensor *recurrent;
    uint64_t token_count;
    bool valid;
} ds4_glm5_kda_layer_state;

typedef struct {
    ds4_glm5_kda_layer_state *layer;
    uint32_t layer_count;
    uint32_t kda_count;
    uint64_t bytes;
    bool valid;
} ds4_glm5_kda_slot;

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

#endif
