#include "ds4_glm5_next_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ds4_gpu_tensor {
    uint64_t bytes;
};

static int alloc_calls;
static int free_calls;
static int fill_calls;
static int fail_alloc_call;
static FILE *accounting_stream;
static int accounting_seen_before_alloc;

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    ++alloc_calls;
    if (accounting_stream) {
        fflush(accounting_stream);
        accounting_seen_before_alloc = ftell(accounting_stream) > 0;
    }
    if (fail_alloc_call > 0 && alloc_calls == fail_alloc_call) return NULL;
    ds4_gpu_tensor *tensor = calloc(1, sizeof(*tensor));
    if (tensor) tensor->bytes = bytes;
    return tensor;
}

void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    ++free_calls;
    free(tensor);
}

int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value,
                            uint64_t count) {
    (void)value;
    if (!tensor || count > tensor->bytes / sizeof(float)) return 0;
    ++fill_calls;
    return 1;
}

uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0u;
}

int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset,
                        void *data, uint64_t bytes) {
    (void)tensor;
    (void)offset;
    (void)data;
    (void)bytes;
    return 0;
}

#define CHECK(expr, message) do { \
    if (!(expr)) { fprintf(stderr, "FAIL %s\n", message); return 0; } \
} while (0)

static void reset_fakes(void) {
    alloc_calls = free_calls = fill_calls = 0;
    fail_alloc_call = 0;
    accounting_stream = NULL;
    accounting_seen_before_alloc = 0;
}

static void fill_words(void *value, size_t bytes, uint64_t *next) {
    uint64_t *word = value;
    for (size_t i = 0u; i < bytes / sizeof(uint64_t); ++i)
        word[i] = (*next)++;
}

static void make_valid(ds4_glm5_next_model_offsets *model) {
    memset(model, 0, sizeof(*model));
    uint64_t next = 1u;
    model->token_embd = next++;
    model->output_norm = next++;
    model->output = next++;
    model->nextn_eh_proj = next++;
    model->layer_count = DS4_GLM5_NEXT_LAYER_COUNT;
    model->trunk_count = DS4_GLM5_NEXT_TRUNK_COUNT;
    model->nextn_count = 1u;
    model->rms_norm_eps = 1.0e-5f;
    model->hc_eps = 1.0e-6f;
    for (uint32_t il = 0u; il < model->layer_count; ++il) {
        ds4_glm5_next_layer_offsets *layer = &model->layer[il];
        layer->layer = il;
        layer->is_trunk = il < model->trunk_count;
        layer->attn_norm = next++;
        layer->ffn_norm = next++;
        const bool mla = il == DS4_GLM5_NEXT_TRUNK_COUNT ||
                         (il & 3u) == 3u;
        layer->attention = mla ? DS4_GLM5_NEXT_ATTN_MLA :
                                 DS4_GLM5_NEXT_ATTN_KDA;
        if (mla) fill_words(&layer->mla, sizeof(layer->mla), &next);
        else {
            fill_words(&layer->kda, sizeof(layer->kda), &next);
            layer->kda.q_type = layer->kda.k_type = 30u;
            layer->kda.v_type = layer->kda.output_type = 30u;
            layer->kda.f_a_type = layer->kda.f_b_type = 30u;
            layer->kda.g_a_type = layer->kda.g_b_type = 30u;
            layer->kda.beta_type = 30u;
        }
        if (il < DS4_GLM5_NEXT_LEADING_DENSE) {
            layer->ffn = DS4_GLM5_NEXT_FFN_DENSE;
            layer->ffn_weight.gate = next++;
            layer->ffn_weight.up = next++;
            layer->ffn_weight.down = next++;
        } else {
            layer->ffn = DS4_GLM5_NEXT_FFN_ROUTED;
            fill_words(&layer->ffn_weight.gate_exps,
                       sizeof(layer->ffn_weight) - 3u * sizeof(uint64_t),
                       &next);
        }
        if (layer->is_trunk)
            fill_words(&layer->hc, sizeof(layer->hc), &next);
    }
}

static int test_bytes(void) {
    ds4_glm5_next_model_offsets model;
    make_valid(&model);
    uint64_t bytes = 0u;
    CHECK(ds4_glm5_next_state_bytes(&model, 8u, &bytes),
          "8-token state size accepted");
    CHECK(bytes == UINT64_C(152870680), "8-token compact state size exact");
    CHECK(ds4_glm5_next_state_bytes(&model, 9u, &bytes) &&
          bytes == UINT64_C(152899104),
          "9-token state uses ceil pool capacity exactly");
    CHECK(ds4_glm5_next_state_bytes(&model, 262144u, &bytes),
          "256K state size accepted");
    CHECK(bytes == UINT64_C(6453309440), "256K compact state size exact");
    CHECK(!ds4_glm5_next_state_bytes(&model, 0u, &bytes),
          "zero context rejected");
    model.layer[3].attention = DS4_GLM5_NEXT_ATTN_KDA;
    CHECK(!ds4_glm5_next_state_bytes(&model, 8u, &bytes),
          "invalid schedule rejected before accounting");
    return 1;
}

static int test_lifecycle(void) {
    reset_fakes();
    ds4_glm5_next_model_offsets model;
    make_valid(&model);
    ds4_glm5_next_state state = {0};
    char *text = NULL;
    size_t text_size = 0u;
    accounting_stream = open_memstream(&text, &text_size);
    CHECK(accounting_stream, "open accounting stream");
    CHECK(ds4_glm5_next_state_init(&state, &model, 8u,
                                   accounting_stream),
          "initialize complete mixed-attention state");
    fflush(accounting_stream);
    CHECK(accounting_seen_before_alloc, "accounting precedes allocation");
    CHECK(strstr(text, "context=8 kda=34 mla=11 bytes=152870680") != NULL,
          "combined accounting exact");
    CHECK(state.valid && state.kda.valid && state.layer_count == 45u &&
          state.mla_count == 11u && state.context_capacity == 8u,
          "complete state starts valid");
    CHECK(alloc_calls == 213 && fill_calls == 147,
          "34 KDA and 11 four-buffer compact MLA states allocated once");
    CHECK(state.mla[3].valid && state.mla[3].owner == &state &&
          state.mla[3].capacity_tokens == 8u &&
          state.mla[3].capacity_pools == 2u &&
          state.mla[3].index_pool && state.mla[3].index_tail &&
          state.mla[3].pool_gate_tail && state.mla[3].index_valid_keys &&
          !state.mla[45].compact_kv,
          "trunk MLA owned and nextn MLA excluded");
    uint32_t rejected_tail = 0u, rejected_pool = 0u;
    bool rejected_publish = false;
    state.mla[3].first_valid = 1u;
    CHECK(!ds4_glm5_next_mla_append_plan(
              &state.mla[3], &rejected_tail, &rejected_pool,
              &rejected_publish),
          "compact append rejects a shifted sequence origin");
    state.mla[3].first_valid = 0u;
    state.mla[3].tail_count = 1u;
    CHECK(!ds4_glm5_next_mla_append_plan(
              &state.mla[3], &rejected_tail, &rejected_pool,
              &rejected_publish),
          "compact append rejects inconsistent counters");
    state.mla[3].tail_count = 0u;
    ds4_gpu_tensor *append_tail = state.mla[3].index_tail;
    state.mla[3].index_tail = NULL;
    CHECK(!ds4_glm5_next_mla_append_plan(
              &state.mla[3], &rejected_tail, &rejected_pool,
              &rejected_publish),
          "compact append rejects a missing tail buffer");
    state.mla[3].index_tail = append_tail;
    for (uint32_t pos = 0u; pos < 8u; ++pos) {
        uint32_t tail_slot = UINT32_MAX, pool_index = UINT32_MAX;
        bool publish_pool = false;
        CHECK(ds4_glm5_next_mla_append_plan(
                  &state.mla[3], &tail_slot, &pool_index, &publish_pool) &&
              tail_slot == pos % 4u && pool_index == pos / 4u &&
              publish_pool == (pos % 4u == 3u),
              "compact MLA append plan follows pool/tail lifecycle");
        CHECK(ds4_glm5_next_mla_append_commit(&state.mla[3]) &&
              state.mla[3].token_count == pos + 1u &&
              state.mla[3].complete_pools == (pos + 1u) / 4u &&
              state.mla[3].tail_count == (pos + 1u) % 4u,
              "compact MLA append commits atomically");
    }
    uint32_t tail_slot = 0u, pool_index = 0u;
    bool publish_pool = false;
    CHECK(!ds4_glm5_next_mla_append_plan(
              &state.mla[3], &tail_slot, &pool_index, &publish_pool),
          "compact MLA append fails closed at context capacity");
    CHECK(ds4_glm5_next_state_reset(&state),
          "state resets after compact append lifecycle test");
    CHECK(!ds4_glm5_next_state_init(&state, &model, 8u,
                                    accounting_stream),
          "live state cannot be initialized twice");
    state.mla[3].token_count = 7u;
    state.mla[3].complete_pools = 1u;
    state.mla[3].tail_count = 3u;
    ds4_glm5_next_state_invalidate(&state);
    CHECK(!state.valid && !state.kda.valid && !state.mla[3].valid,
          "mixed state invalidates atomically");
    CHECK(ds4_glm5_next_state_reset(&state) && state.valid &&
          state.kda.valid && state.mla[3].valid &&
          state.mla[3].token_count == 0u &&
          state.mla[3].complete_pools == 0u &&
          state.mla[3].tail_count == 0u &&
          state.mla[3].first_valid == 0u && fill_calls == 419,
          "mixed state resets atomically");
    state.mla[3].owner = NULL;
    CHECK(!ds4_glm5_next_state_reset(&state) && !state.valid,
          "corrupt MLA ownership fails closed");
    state.mla[3].owner = &state;
    CHECK(ds4_glm5_next_state_reset(&state),
          "restored MLA ownership resets");
    ds4_gpu_tensor *saved_compact = state.mla[3].compact_kv;
    state.mla[3].compact_kv = NULL;
    CHECK(!ds4_glm5_next_state_reset(&state) && !state.valid,
          "missing MLA buffer fails closed");
    state.mla[3].compact_kv = saved_compact;
    ds4_gpu_tensor *saved_pool = state.mla[3].index_pool;
    state.mla[3].index_pool = NULL;
    CHECK(!ds4_glm5_next_state_reset(&state) && !state.valid,
          "missing compact MLA pool buffer fails closed");
    state.mla[3].index_pool = saved_pool;
    CHECK(ds4_glm5_next_state_reset(&state),
          "restored compact MLA pool buffer resets");
    ds4_glm5_next_state_free(&state);
    CHECK(free_calls == 213 && !state.valid && !state.kda.layer &&
          !state.mla[3].compact_kv,
          "complete mixed state freed exactly once");
    ds4_glm5_next_state_free(&state);
    CHECK(free_calls == 213 && !ds4_glm5_next_state_reset(&state),
          "double free is harmless and reset-after-free fails closed");
    fclose(accounting_stream);
    accounting_stream = NULL;
    free(text);
    return 1;
}

static int test_partial_failure(void) {
    reset_fakes();
    ds4_glm5_next_model_offsets model;
    make_valid(&model);
    ds4_glm5_next_state state = {0};
    fail_alloc_call = 150;
    FILE *stream = tmpfile();
    CHECK(stream, "open failure accounting stream");
    CHECK(!ds4_glm5_next_state_init(&state, &model, 8u, stream),
          "injected MLA allocation failure propagates");
    CHECK(alloc_calls == 150 && free_calls == 149 && !state.valid &&
          !state.kda.layer && state.bytes == 0u,
          "partial KDA/MLA state is fully unwound");
    fclose(stream);

    reset_fakes();
    make_valid(&model);
    fail_alloc_call = 10;
    stream = tmpfile();
    CHECK(stream, "open KDA failure accounting stream");
    CHECK(!ds4_glm5_next_state_init(&state, &model, 8u, stream),
          "injected KDA allocation failure propagates");
    CHECK(alloc_calls == 10 && free_calls == 9 && !state.valid &&
          !state.kda.layer && state.bytes == 0u,
          "partial KDA state is fully unwound");
    fclose(stream);
    return 1;
}

int main(void) {
    int ok = test_bytes();
    ok &= test_lifecycle();
    ok &= test_partial_failure();
    if (ok) fprintf(stderr, "PASS GLM5-next atomic resident state lifecycle\n");
    return ok ? 0 : 1;
}
