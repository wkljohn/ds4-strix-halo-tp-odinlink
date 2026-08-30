#include "ds4_glm5_next_runtime.h"

#include <inttypes.h>
#include <stdlib.h>
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
    uint64_t compact_values = 0u, pool_values = 0u, tail_values = 0u;
    uint64_t pool_id_values = 0u, pool_valid_values = 0u, valid_key_values = 0u;
    uint64_t compact_bytes = 0u, pool_bytes = 0u, tail_bytes = 0u;
    uint64_t pool_id_bytes = 0u, pool_valid_bytes = 0u, valid_key_bytes = 0u;
    const uint64_t pool_capacity = context_capacity / 4u +
        (context_capacity % 4u != 0u);
    return context_capacity != 0u &&
           mul_u64(context_capacity, DS4_GLM5_NEXT_MLA_KV_WIDTH,
                   &compact_values) &&
           mul_u64(pool_capacity, DS4_GLM5_NEXT_INDEX_WIDTH, &pool_values) &&
           mul_u64(8u, DS4_GLM5_NEXT_INDEX_WIDTH, &tail_values) &&
           mul_u64(pool_capacity, 4u, &pool_id_values) &&
           mul_u64(pool_capacity, 1u, &pool_valid_values) &&
           mul_u64(context_capacity, 1u, &valid_key_values) &&
           mul_u64(compact_values, sizeof(float), &compact_bytes) &&
           mul_u64(pool_values, sizeof(float), &pool_bytes) &&
           mul_u64(tail_values, sizeof(float), &tail_bytes) &&
           mul_u64(pool_id_values, sizeof(int32_t), &pool_id_bytes) &&
           mul_u64(pool_valid_values, sizeof(uint32_t), &pool_valid_bytes) &&
           mul_u64(valid_key_values, sizeof(uint32_t), &valid_key_bytes) &&
           add_u64(compact_bytes, pool_bytes, bytes) &&
           add_u64(*bytes, tail_bytes, bytes) &&
           add_u64(*bytes, pool_id_bytes, bytes) &&
           add_u64(*bytes, pool_valid_bytes, bytes) &&
           add_u64(*bytes, valid_key_bytes, bytes);
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
        ds4_gpu_tensor_free(state->mla[il].pool_gate_tail);
        ds4_gpu_tensor_free(state->mla[il].index_tail);
        ds4_gpu_tensor_free(state->mla[il].index_pool_valid);
        ds4_gpu_tensor_free(state->mla[il].index_pool_ids);
        ds4_gpu_tensor_free(state->mla[il].index_valid_keys);
        ds4_gpu_tensor_free(state->mla[il].index_pool);
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
            if (mla->compact_kv || mla->index_pool || mla->index_pool_ids ||
                mla->index_pool_valid || mla->index_valid_keys ||
                mla->index_tail || mla->pool_gate_tail)
                goto invalid;
            continue;
        }
        const uint32_t expected_pools = state->context_capacity / 4u +
            (state->context_capacity % 4u != 0u);
        if (!mla->index_pool || !mla->index_pool_ids ||
            !mla->index_pool_valid || !mla->index_valid_keys ||
            !mla->index_tail || !mla->pool_gate_tail ||
            !mla->compact_kv || mla->capacity_pools != expected_pools ||
            mla->capacity_tokens != state->context_capacity ||
            mla->owner != state) goto invalid;
        mla->token_count = 0u;
        mla->complete_pools = 0u;
        mla->tail_count = 0u;
        mla->first_valid = 0u;
        mla->valid = true;
        ++live_mla;
    }
    if (live_mla != state->mla_count ||
        state->mla[DS4_GLM5_NEXT_TRUNK_COUNT].compact_kv ||
        state->mla[DS4_GLM5_NEXT_TRUNK_COUNT].index_pool ||
        state->mla[DS4_GLM5_NEXT_TRUNK_COUNT].index_pool_ids ||
        state->mla[DS4_GLM5_NEXT_TRUNK_COUNT].index_pool_valid ||
        state->mla[DS4_GLM5_NEXT_TRUNK_COUNT].index_valid_keys ||
        state->mla[DS4_GLM5_NEXT_TRUNK_COUNT].index_tail ||
        state->mla[DS4_GLM5_NEXT_TRUNK_COUNT].pool_gate_tail) goto invalid;
    state->valid = true;
    return 1;

invalid:
    ds4_glm5_next_state_invalidate(state);
    return 0;
}

int ds4_glm5_next_mla_append_plan(
        const ds4_glm5_next_mla_state *mla, uint32_t *tail_slot,
        uint32_t *pool_index, bool *publish_pool) {
    if (!mla || !tail_slot || !pool_index || !publish_pool || !mla->valid ||
        !mla->owner || !mla->compact_kv || !mla->index_pool ||
        !mla->index_pool_ids || !mla->index_pool_valid ||
        !mla->index_valid_keys ||
        !mla->index_tail || !mla->pool_gate_tail ||
        mla->capacity_tokens == 0u || mla->capacity_pools == 0u ||
        mla->token_count >= mla->capacity_tokens || mla->first_valid != 0u ||
        mla->complete_pools != mla->token_count / 4u ||
        mla->tail_count != mla->token_count % 4u) return 0;
    const uint32_t slot = mla->token_count % 4u;
    const uint32_t pool = mla->token_count / 4u;
    if (pool >= mla->capacity_pools) return 0;
    *tail_slot = slot;
    *pool_index = pool;
    *publish_pool = slot == 3u;
    return 1;
}

int ds4_glm5_next_mla_append_commit(ds4_glm5_next_mla_state *mla) {
    uint32_t tail_slot = 0u, pool_index = 0u;
    bool publish_pool = false;
    if (!ds4_glm5_next_mla_append_plan(
            mla, &tail_slot, &pool_index, &publish_pool)) return 0;
    (void)tail_slot;
    (void)pool_index;
    (void)publish_pool;
    const uint32_t next = mla->token_count + 1u;
    mla->token_count = next;
    mla->complete_pools = next / 4u;
    mla->tail_count = next % 4u;
    return 1;
}

static int state_is_empty(const ds4_glm5_next_state *state) {
    if (!state || state->layer_count || state->context_capacity ||
        state->mla_count || state->bytes || state->valid || state->kda.layer)
        return 0;
    for (uint32_t il = 0u; il < DS4_GLM5_NEXT_LAYER_COUNT; ++il) {
        if (state->mla[il].compact_kv || state->mla[il].index_pool ||
            state->mla[il].index_pool_ids || state->mla[il].index_pool_valid ||
            state->mla[il].index_valid_keys ||
            state->mla[il].index_tail || state->mla[il].pool_gate_tail)
            return 0;
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

    uint64_t compact_bytes = 0u, pool_bytes = 0u, tail_bytes = 0u;
    const uint32_t pool_capacity = context_capacity / 4u +
        (context_capacity % 4u != 0u);
    if (!mul_u64(context_capacity, DS4_GLM5_NEXT_MLA_KV_WIDTH,
                 &compact_bytes) ||
        !mul_u64(compact_bytes, sizeof(float), &compact_bytes) ||
        !mul_u64(pool_capacity, DS4_GLM5_NEXT_INDEX_WIDTH, &pool_bytes) ||
        !mul_u64(pool_bytes, sizeof(float), &pool_bytes) ||
        !mul_u64(4u, DS4_GLM5_NEXT_INDEX_WIDTH, &tail_bytes) ||
        !mul_u64(tail_bytes, sizeof(float), &tail_bytes)) goto fail;
    uint32_t allocated_mla = 0u;
    for (uint32_t il = 0u; il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il) {
        if (model->layer[il].attention != DS4_GLM5_NEXT_ATTN_MLA) continue;
        ds4_glm5_next_mla_state *mla = &state->mla[il];
        mla->owner = state;
        mla->capacity_tokens = context_capacity;
        mla->capacity_pools = pool_capacity;
        mla->compact_kv = ds4_gpu_tensor_alloc(compact_bytes);
        if (!mla->compact_kv) goto fail;
        mla->index_pool = ds4_gpu_tensor_alloc(pool_bytes);
        if (!mla->index_pool) goto fail;
        mla->index_pool_ids = ds4_gpu_tensor_alloc(
            (uint64_t)pool_capacity * 4u * sizeof(int32_t));
        if (!mla->index_pool_ids) goto fail;
        mla->index_pool_valid = ds4_gpu_tensor_alloc(
            (uint64_t)pool_capacity * sizeof(uint32_t));
        if (!mla->index_pool_valid) goto fail;
        mla->index_valid_keys = ds4_gpu_tensor_alloc(
            (uint64_t)context_capacity * sizeof(uint32_t));
        if (!mla->index_valid_keys) goto fail;
        /* The selector treats valid_keys as a u32 nonzero mask.  The generic
         * tensor fill writes the same nonzero bit pattern for every row and
         * avoids a host readback or a second persistent allocation. */
        if (!ds4_gpu_tensor_fill_f32(
                mla->index_valid_keys, 1.0f, context_capacity)) goto fail;
        mla->index_tail = ds4_gpu_tensor_alloc(tail_bytes);
        if (!mla->index_tail) goto fail;
        mla->pool_gate_tail = ds4_gpu_tensor_alloc(tail_bytes);
        if (!mla->pool_gate_tail) goto fail;
        /* Diagnostic for a read-before-write in the persistent sparse-MLA
         * state.  All sizes are four-byte aligned.  Keep this opt-in until a
         * narrower buffer contract is proven; clearing the full KV capacity
         * at startup is intentionally not a production default. */
        if (getenv("DS4_GLM5_ZERO_PERSISTENT_MLA") != NULL &&
            (!ds4_gpu_tensor_fill_f32(
                 mla->compact_kv, 0.0f,
                 (uint32_t)(compact_bytes / sizeof(float))) ||
             !ds4_gpu_tensor_fill_f32(
                 mla->index_pool, 0.0f,
                 (uint32_t)(pool_bytes / sizeof(float))) ||
             !ds4_gpu_tensor_fill_f32(
                 mla->index_pool_ids, 0.0f,
                 pool_capacity * 4u) ||
             !ds4_gpu_tensor_fill_f32(
                 mla->index_pool_valid, 0.0f,
                 pool_capacity) ||
             !ds4_gpu_tensor_fill_f32(
                 mla->index_tail, 0.0f,
                 (uint32_t)(tail_bytes / sizeof(float))) ||
             !ds4_gpu_tensor_fill_f32(
                 mla->pool_gate_tail, 0.0f,
                 (uint32_t)(tail_bytes / sizeof(float))))) goto fail;
        ++allocated_mla;
    }
    if (allocated_mla != DS4_GLM5_NEXT_MLA_COUNT) goto fail;
    for (uint32_t il = 0u; il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il) {
        if (!state->mla[il].compact_kv) continue;
        state->mla[il].token_count = 0u;
        state->mla[il].complete_pools = 0u;
        state->mla[il].tail_count = 0u;
        state->mla[il].first_valid = 0u;
        state->mla[il].valid = true;
    }
    state->valid = true;
    return 1;

fail:
    ds4_glm5_next_state_free(state);
    return 0;
}
