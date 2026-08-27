#include "ds4_glm5_kda.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ds4_gpu_tensor {
    float *data;
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
    if (!tensor) return NULL;
    tensor->data = malloc(bytes ? (size_t)bytes : 1u);
    if (!tensor->data) {
        free(tensor);
        return NULL;
    }
    memset(tensor->data, 0x7f, (size_t)bytes);
    tensor->bytes = bytes;
    return tensor;
}

int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value,
                            uint64_t count) {
    if (!tensor || count > tensor->bytes / sizeof(float)) return 0;
    ++fill_calls;
    for (uint64_t i = 0; i < count; ++i) tensor->data[i] = value;
    return 1;
}

void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    ++free_calls;
    free(tensor->data);
    free(tensor);
}

uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset,
                        void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes ||
        bytes > tensor->bytes - offset) return 0;
    memcpy(data, (const unsigned char *)tensor->data + offset, (size_t)bytes);
    return 1;
}

static void reset_fakes(void) {
    alloc_calls = 0;
    free_calls = 0;
    fill_calls = 0;
    fail_alloc_call = 0;
    accounting_stream = NULL;
    accounting_seen_before_alloc = 0;
}

static int all_zero(const ds4_gpu_tensor *tensor) {
    for (uint64_t i = 0; i < tensor->bytes / sizeof(float); ++i) {
        if (tensor->data[i] != 0.0f) return 0;
    }
    return 1;
}

#define CHECK(expr, message) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL %s\n", message); \
        return 0; \
    } \
} while (0)

static int test_state_byte_contract(void) {
    uint64_t bytes = 0;
    CHECK(ds4_glm5_kda_state_bytes(1, 1, &bytes), "one-layer size accepted");
    CHECK(bytes == UINT64_C(4489216), "one-layer size exact");
    CHECK(ds4_glm5_kda_state_bytes(34, 1, &bytes), "34-layer size accepted");
    CHECK(bytes == UINT64_C(152633344), "34-layer size exact");
    CHECK(!ds4_glm5_kda_state_bytes(UINT64_MAX, 1, &bytes),
          "layer multiplication overflow rejected");
    CHECK(!ds4_glm5_kda_state_bytes(1, 0, &bytes), "zero slots rejected");
    CHECK(!ds4_glm5_kda_state_bytes(1, 2, &bytes), "second slot rejected");
    return 1;
}

static int test_one_layer_lifecycle(void) {
    reset_fakes();
    ds4_glm5_layer_kind schedule[3] = {
        {.layer = 0, .is_kda = false},
        {.layer = 1, .is_kda = true},
        {.layer = 2, .is_kda = false},
    };
    ds4_glm5_kda_slot slot = {0};
    char *text = NULL;
    size_t text_size = 0;
    accounting_stream = open_memstream(&text, &text_size);
    CHECK(accounting_stream != NULL, "open accounting stream");
    CHECK(ds4_glm5_kda_slot_init(&slot, schedule, 3, 1,
                                 accounting_stream), "initialize slot");
    fflush(accounting_stream);
    CHECK(accounting_seen_before_alloc, "accounting printed before allocation");
    CHECK(strstr(text, "slots=1 layers=1 bytes=4489216 MiB=4.28") != NULL,
          "accounting values exact");
    CHECK(slot.layer_count == 3 && slot.kda_count == 1,
          "derived schedule counts");
    CHECK(slot.bytes == UINT64_C(4489216), "slot bytes exact");
    CHECK(slot.valid && slot.layer[1].valid, "slot starts valid");
    CHECK(alloc_calls == 4 && fill_calls == 4, "four tensors allocated and zeroed");
    CHECK(all_zero(slot.layer[1].q_history) &&
          all_zero(slot.layer[1].k_history) &&
          all_zero(slot.layer[1].v_history) &&
          all_zero(slot.layer[1].recurrent), "all resident state zeroed");
    slot.layer[1].token_count = 73;
    slot.layer[1].q_history->data[0] = 9.0f;
    ds4_glm5_kda_slot_invalidate(&slot);
    CHECK(!slot.valid && !slot.layer[1].valid, "invalidation propagates");
    CHECK(ds4_glm5_kda_slot_reset(&slot), "reset invalid slot");
    CHECK(slot.valid && slot.layer[1].valid &&
          slot.layer[1].token_count == 0, "reset restores validity and counter");
    CHECK(all_zero(slot.layer[1].q_history), "reset clears history");
    ds4_glm5_kda_slot_free(&slot);
    CHECK(free_calls == 4, "all tensors freed");
    CHECK(slot.layer == NULL && slot.layer_count == 0 && !slot.valid,
          "free clears owner");
    fclose(accounting_stream);
    accounting_stream = NULL;
    free(text);
    return 1;
}

static int test_partial_allocation_failure_cleans_up(void) {
    reset_fakes();
    ds4_glm5_layer_kind schedule[2] = {
        {.layer = 7, .is_kda = true},
        {.layer = 9, .is_kda = true},
    };
    ds4_glm5_kda_slot slot = {0};
    fail_alloc_call = 6;
    FILE *stream = tmpfile();
    CHECK(stream != NULL, "open failure accounting stream");
    CHECK(!ds4_glm5_kda_slot_init(&slot, schedule, 2, 1, stream),
          "injected allocation fails");
    CHECK(alloc_calls == 6 && free_calls == 5,
          "all successful partial allocations freed");
    CHECK(slot.layer == NULL && slot.bytes == 0 && !slot.valid,
          "failed slot remains empty and invalid");
    fclose(stream);
    return 1;
}

static int test_zero_kda_schedule_is_valid_and_empty(void) {
    reset_fakes();
    ds4_glm5_layer_kind schedule[2] = {
        {.layer = 4, .is_kda = false},
        {.layer = 8, .is_kda = false},
    };
    ds4_glm5_kda_slot slot = {0};
    FILE *stream = tmpfile();
    CHECK(stream != NULL, "open zero schedule stream");
    CHECK(ds4_glm5_kda_slot_init(&slot, schedule, 2, 1, stream),
          "zero KDA schedule accepted");
    CHECK(slot.kda_count == 0 && slot.bytes == 0 && alloc_calls == 0,
          "zero KDA schedule allocates no device state");
    ds4_glm5_kda_slot_free(&slot);
    fclose(stream);
    return 1;
}

int main(void) {
    int ok = 1;
    ok &= test_state_byte_contract();
    ok &= test_one_layer_lifecycle();
    ok &= test_partial_allocation_failure_cleans_up();
    ok &= test_zero_kda_schedule_is_valid_and_empty();
    if (ok) fprintf(stderr, "PASS GLM5 resident KDA state lifecycle\n");
    return ok ? 0 : 1;
}
