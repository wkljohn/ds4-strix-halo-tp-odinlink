#include "ds4_glm5_next_runtime.h"

#include <inttypes.h>
#include <string.h>

static int add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || a > UINT64_MAX - b) return 0;
    *out = a + b;
    return 1;
}

static int mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || (a != 0u && b > UINT64_MAX / a)) return 0;
    *out = a * b;
    return 1;
}

static int mla_layer_bytes(uint32_t context_capacity, uint64_t *bytes) {
    const uint64_t values_per_token = DS4_GLM5_NEXT_MLA_KV_WIDTH +
        2u * DS4_GLM5_NEXT_INDEX_WIDTH;
    uint64_t values = 0u;
    return context_capacity != 0u &&
           mul_u64(context_capacity, values_per_token, &values) &&
           mul_u64(values, sizeof(float), bytes);
}

int ds4_glm5_next_state_bytes(const ds4_glm5_next_model_offsets *model,
                              uint32_t context_capacity,
                              uint64_t *bytes) {
    uint64_t kda_bytes = 0u, per_mla = 0u, mla_bytes = 0u;
    uint32_t kda_count = 0u, mla_count = 0u;
    if (ds4_glm5_next_model_offsets_validate(model)) {
        for (uint32_t il = 0u; il < model->trunk_count; ++il) {
            if (model->layer[il].attention == DS4_GLM5_NEXT_ATTN_KDA)
                ++kda_count;
            else
                ++mla_count;
        }
    }
    if (!bytes || !ds4_glm5_next_model_offsets_validate(model) ||
        kda_count != 34u || mla_count != DS4_GLM5_NEXT_MLA_COUNT ||
        !mla_layer_bytes(context_capacity, &per_mla) ||
        !ds4_glm5_kda_state_bytes(kda_count, DS4_GLM5_KDA_MAX_SLOTS,
                                  &kda_bytes) ||
        !mul_u64(per_mla, mla_count, &mla_bytes) ||
        !add_u64(kda_bytes, mla_bytes, bytes)) return 0;
    return 1;
}

void ds4_glm5_next_state_invalidate(ds4_glm5_next_state *state) {
    if (!state) return;
    state->valid = false;
    ds4_glm5_kda_slot_invalidate(&state->kda);
    for (uint32_t il = 0u; il < DS4_GLM5_NEXT_LAYER_COUNT; ++il) {
        if (state->mla[il].compact_kv) state->mla[il].valid = false;
    }
}

void ds4_glm5_next_state_free(ds4_glm5_next_state *state) {
    if (!state) return;
    ds4_glm5_kda_slot_free(&state->kda);
    for (uint32_t il = 0u; il < DS4_GLM5_NEXT_LAYER_COUNT; ++il) {
        ds4_gpu_tensor_free(state->mla[il].pool_gate);
        ds4_gpu_tensor_free(state->mla[il].index_key);
        ds4_gpu_tensor_free(state->mla[il].compact_kv);
    }
    memset(state, 0, sizeof(*state));
}

int ds4_glm5_next_state_reset(ds4_glm5_next_state *state) {
    if (!state || state->layer_count != DS4_GLM5_NEXT_TRUNK_COUNT ||
        state->context_capacity == 0u || state->mla_count !=
        DS4_GLM5_NEXT_MLA_COUNT || !ds4_glm5_kda_slot_reset(&state->kda)) {
        ds4_glm5_next_state_invalidate(state);
        return 0;
    }
    uint32_t live_mla = 0u;
    for (uint32_t il = 0u; il < state->layer_count; ++il) {
        ds4_glm5_next_mla_state *mla = &state->mla[il];
        const bool is_kda = state->kda.layer &&
                            state->kda.layer[il].recurrent != NULL;
        if (is_kda) {
            if (mla->compact_kv || mla->index_key || mla->pool_gate)
                goto invalid;
            continue;
        }
        if (!mla->index_key || !mla->pool_gate ||
            !mla->compact_kv ||
            mla->capacity_tokens != state->context_capacity ||
            mla->owner != state) goto invalid;
        mla->token_count = 0u;
        mla->first_valid = 0u;
        mla->valid = true;
        ++live_mla;
    }
    if (live_mla != state->mla_count ||
        state->mla[DS4_GLM5_NEXT_TRUNK_COUNT].compact_kv ||
        state->mla[DS4_GLM5_NEXT_TRUNK_COUNT].index_key ||
        state->mla[DS4_GLM5_NEXT_TRUNK_COUNT].pool_gate) goto invalid;
    state->valid = true;
    return 1;

invalid:
    ds4_glm5_next_state_invalidate(state);
    return 0;
}

static int state_is_empty(const ds4_glm5_next_state *state) {
    if (!state || state->layer_count || state->context_capacity ||
        state->mla_count || state->bytes || state->valid || state->kda.layer)
        return 0;
    for (uint32_t il = 0u; il < DS4_GLM5_NEXT_LAYER_COUNT; ++il) {
        if (state->mla[il].compact_kv || state->mla[il].index_key ||
            state->mla[il].pool_gate) return 0;
    }
    return 1;
}

int ds4_glm5_next_state_init(ds4_glm5_next_state *state,
                             const ds4_glm5_next_model_offsets *model,
                             uint32_t context_capacity,
                             FILE *accounting) {
    uint64_t total_bytes = 0u, per_mla = 0u;
    if (!state_is_empty(state) ||
        !ds4_glm5_next_state_bytes(model, context_capacity, &total_bytes) ||
        !mla_layer_bytes(context_capacity, &per_mla)) return 0;

    FILE *stream = accounting ? accounting : stderr;
    fprintf(stream,
            "ds4: GLM5-next resident state: context=%u kda=34 mla=%u "
            "bytes=%" PRIu64 " MiB=%.2f\n",
            context_capacity, DS4_GLM5_NEXT_MLA_COUNT, total_bytes,
            (double)total_bytes / (1024.0 * 1024.0));
    fflush(stream);

    ds4_glm5_layer_kind schedule[DS4_GLM5_NEXT_TRUNK_COUNT];
    for (uint32_t il = 0u; il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il) {
        schedule[il].layer = il;
        schedule[il].is_kda =
            model->layer[il].attention == DS4_GLM5_NEXT_ATTN_KDA;
    }
    state->layer_count = DS4_GLM5_NEXT_TRUNK_COUNT;
    state->context_capacity = context_capacity;
    state->mla_count = DS4_GLM5_NEXT_MLA_COUNT;
    state->bytes = total_bytes;
    if (!ds4_glm5_kda_slot_init(&state->kda, schedule,
                                 DS4_GLM5_NEXT_TRUNK_COUNT,
                                 DS4_GLM5_KDA_MAX_SLOTS, stream)) goto fail;

    uint64_t compact_bytes = 0u, index_bytes = 0u;
    if (!mul_u64(context_capacity, DS4_GLM5_NEXT_MLA_KV_WIDTH,
                 &compact_bytes) ||
        !mul_u64(compact_bytes, sizeof(float), &compact_bytes) ||
        !mul_u64(context_capacity, DS4_GLM5_NEXT_INDEX_WIDTH,
                 &index_bytes) ||
        !mul_u64(index_bytes, sizeof(float), &index_bytes)) goto fail;
    uint32_t allocated_mla = 0u;
    for (uint32_t il = 0u; il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il) {
        if (model->layer[il].attention != DS4_GLM5_NEXT_ATTN_MLA) continue;
        ds4_glm5_next_mla_state *mla = &state->mla[il];
        mla->owner = state;
        mla->capacity_tokens = context_capacity;
        mla->compact_kv = ds4_gpu_tensor_alloc(compact_bytes);
        if (!mla->compact_kv) goto fail;
        mla->index_key = ds4_gpu_tensor_alloc(index_bytes);
        if (!mla->index_key) goto fail;
        mla->pool_gate = ds4_gpu_tensor_alloc(index_bytes);
        if (!mla->pool_gate) goto fail;
        ++allocated_mla;
    }
    if (allocated_mla != DS4_GLM5_NEXT_MLA_COUNT) goto fail;
    for (uint32_t il = 0u; il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il) {
        if (!state->mla[il].compact_kv) continue;
        state->mla[il].token_count = 0u;
        state->mla[il].first_valid = 0u;
        state->mla[il].valid = true;
    }
    state->valid = true;
    return 1;

fail:
    ds4_glm5_next_state_free(state);
    return 0;
}
