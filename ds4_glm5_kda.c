#include "ds4_glm5_kda.h"

#include <inttypes.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

static int mul_u64(uint64_t a, uint64_t b, uint64_t *result) {
    if (!result || (a != 0 && b > UINT64_MAX / a)) return 0;
    *result = a * b;
    return 1;
}

static int add_u64(uint64_t a, uint64_t b, uint64_t *result) {
    if (!result || b > UINT64_MAX - a) return 0;
    *result = a + b;
    return 1;
}

static int layer_bytes(uint64_t *history_bytes,
                       uint64_t *recurrent_bytes,
                       uint64_t *total_bytes) {
    uint64_t history_values = 0;
    uint64_t recurrent_values = 0;
    uint64_t all_histories = 0;
    if (!mul_u64(DS4_GLM5_KDA_CHANNELS, DS4_GLM5_KDA_HISTORY,
                 &history_values) ||
        !mul_u64(history_values, sizeof(float), history_bytes) ||
        !mul_u64(DS4_GLM5_KDA_HEADS, DS4_GLM5_KDA_HEAD_DIM,
                 &recurrent_values) ||
        !mul_u64(recurrent_values, DS4_GLM5_KDA_HEAD_DIM,
                 &recurrent_values) ||
        !mul_u64(recurrent_values, sizeof(float), recurrent_bytes) ||
        !mul_u64(*history_bytes, 3, &all_histories) ||
        !add_u64(all_histories, *recurrent_bytes, total_bytes)) {
        return 0;
    }
    return 1;
}

int ds4_glm5_kda_state_bytes(uint64_t kda_count, uint32_t slot_count,
                             uint64_t *bytes) {
    uint64_t history = 0;
    uint64_t recurrent = 0;
    uint64_t per_layer = 0;
    uint64_t per_slot = 0;
    if (!bytes || slot_count != DS4_GLM5_KDA_MAX_SLOTS ||
        !layer_bytes(&history, &recurrent, &per_layer) ||
        !mul_u64(per_layer, kda_count, &per_slot) ||
        !mul_u64(per_slot, slot_count, bytes)) {
        return 0;
    }
    return 1;
}

int ds4_glm5_kda_build_schedule(ds4_glm5_layer_kind *out,
                                uint32_t capacity,
                                const bool *has_kda_q,
                                const bool *has_mla_q,
                                uint32_t layer_count,
                                uint32_t *kda_count) {
    if (!out || !has_kda_q || !has_mla_q || !kda_count ||
        layer_count == 0u || capacity < layer_count) {
        return 0;
    }
    uint32_t count = 0;
    for (uint32_t layer = 0; layer < layer_count; ++layer) {
        if (has_kda_q[layer] == has_mla_q[layer]) return 0;
        if (has_kda_q[layer]) ++count;
    }
    for (uint32_t layer = 0; layer < layer_count; ++layer) {
        out[layer].layer = layer;
        out[layer].is_kda = has_kda_q[layer];
    }
    *kda_count = count;
    return 1;
}

void ds4_glm5_kda_slot_free(ds4_glm5_kda_slot *slot) {
    if (!slot) return;
    if (slot->layer) {
        for (uint32_t i = 0; i < slot->layer_count; ++i) {
            ds4_gpu_tensor_free(slot->layer[i].q_history);
            ds4_gpu_tensor_free(slot->layer[i].k_history);
            ds4_gpu_tensor_free(slot->layer[i].v_history);
            ds4_gpu_tensor_free(slot->layer[i].recurrent);
        }
        free(slot->layer);
    }
    memset(slot, 0, sizeof(*slot));
}

void ds4_glm5_kda_slot_invalidate(ds4_glm5_kda_slot *slot) {
    if (!slot) return;
    slot->valid = false;
    for (uint32_t i = 0; i < slot->layer_count; ++i) {
        if (slot->layer[i].recurrent) slot->layer[i].valid = false;
    }
}

int ds4_glm5_kda_slot_reset(ds4_glm5_kda_slot *slot) {
    if (!slot || (slot->layer_count != 0 && !slot->layer)) return 0;
    const uint64_t history_values =
        (uint64_t)DS4_GLM5_KDA_CHANNELS * DS4_GLM5_KDA_HISTORY;
    const uint64_t recurrent_values =
        (uint64_t)DS4_GLM5_KDA_HEADS * DS4_GLM5_KDA_HEAD_DIM *
        DS4_GLM5_KDA_HEAD_DIM;
    for (uint32_t i = 0; i < slot->layer_count; ++i) {
        ds4_glm5_kda_layer_state *state = &slot->layer[i];
        if (!state->recurrent) continue;
        if (!ds4_gpu_tensor_fill_f32(state->q_history, 0.0f, history_values) ||
            !ds4_gpu_tensor_fill_f32(state->k_history, 0.0f, history_values) ||
            !ds4_gpu_tensor_fill_f32(state->v_history, 0.0f, history_values) ||
            !ds4_gpu_tensor_fill_f32(state->recurrent, 0.0f,
                                     recurrent_values)) {
            ds4_glm5_kda_slot_invalidate(slot);
            return 0;
        }
        state->token_count = 0;
        state->valid = true;
    }
    slot->valid = true;
    return 1;
}

int ds4_glm5_kda_slot_init(ds4_glm5_kda_slot *slot,
                           const ds4_glm5_layer_kind *schedule,
                           uint32_t layer_count,
                           uint32_t slot_count,
                           FILE *accounting) {
    uint64_t history_bytes = 0;
    uint64_t recurrent_bytes = 0;
    uint64_t per_layer = 0;
    uint64_t total_bytes = 0;
    uint64_t kda_count = 0;
    if (!slot || (layer_count != 0 && !schedule) || slot->layer ||
        !layer_bytes(&history_bytes, &recurrent_bytes, &per_layer)) {
        return 0;
    }
    for (uint32_t i = 0; i < layer_count; ++i) {
        if (schedule[i].is_kda) ++kda_count;
    }
    if (kda_count > UINT32_MAX ||
        !ds4_glm5_kda_state_bytes(kda_count, slot_count, &total_bytes)) {
        return 0;
    }
    FILE *stream = accounting ? accounting : stderr;
    fprintf(stream,
            "ds4: GLM5 KDA resident state: slots=%u layers=%" PRIu64
            " bytes=%" PRIu64 " MiB=%.2f\n",
            slot_count, kda_count, total_bytes,
            (double)total_bytes / (1024.0 * 1024.0));
    fflush(stream);

    slot->layer = calloc(layer_count ? layer_count : 1u, sizeof(*slot->layer));
    if (!slot->layer) return 0;
    slot->layer_count = layer_count;
    slot->kda_count = (uint32_t)kda_count;
    slot->bytes = total_bytes;
    for (uint32_t i = 0; i < layer_count; ++i) {
        if (!schedule[i].is_kda) continue;
        ds4_glm5_kda_layer_state *state = &slot->layer[i];
        state->q_history = ds4_gpu_tensor_alloc(history_bytes);
        if (!state->q_history) goto fail;
        state->k_history = ds4_gpu_tensor_alloc(history_bytes);
        if (!state->k_history) goto fail;
        state->v_history = ds4_gpu_tensor_alloc(history_bytes);
        if (!state->v_history) goto fail;
        state->recurrent = ds4_gpu_tensor_alloc(recurrent_bytes);
        if (!state->recurrent) goto fail;
    }
    if (!ds4_glm5_kda_slot_reset(slot)) goto fail;
    return 1;

fail:
    ds4_glm5_kda_slot_free(slot);
    return 0;
}
