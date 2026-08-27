#include "ds4_glm5_kda.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ds4_gpu_tensor {
    unsigned char *data;
    uint64_t bytes;
};

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    ds4_gpu_tensor *tensor = calloc(1, sizeof(*tensor));
    if (!tensor) return NULL;
    tensor->data = calloc(bytes ? (size_t)bytes : 1u, 1u);
    if (!tensor->data) { free(tensor); return NULL; }
    tensor->bytes = bytes;
    return tensor;
}

void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    free(tensor->data);
    free(tensor);
}

int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value,
                            uint64_t count) {
    if (!tensor || count > tensor->bytes / sizeof(float)) return 0;
    float *values = (float *)tensor->data;
    for (uint64_t i = 0; i < count; ++i) values[i] = value;
    return 1;
}

uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset,
                        void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes ||
        bytes > tensor->bytes - offset) return 0;
    memcpy(data, tensor->data + offset, (size_t)bytes);
    return 1;
}

#define CHECK(expr, message) do { \
    if (!(expr)) { fprintf(stderr, "FAIL %s\n", message); return 0; } \
} while (0)

static void seed_tensor(ds4_gpu_tensor *tensor, uint32_t salt) {
    for (uint64_t i = 0; i < tensor->bytes; ++i)
        tensor->data[i] = (unsigned char)((i * 131u + salt * 17u) & 0xffu);
}

static int run_digest_test(void) {
    ds4_glm5_layer_kind schedule = {.layer = 0u, .is_kda = true};
    ds4_glm5_kda_slot rank0 = {0}, rank1 = {0};
    FILE *sink = tmpfile();
    CHECK(sink != NULL, "open accounting sink");
    CHECK(ds4_glm5_kda_slot_init(&rank0, &schedule, 1u, 1u, sink) &&
          ds4_glm5_kda_slot_init(&rank1, &schedule, 1u, 1u, sink),
          "allocate two independent rank states");
    ds4_gpu_tensor *out0 = ds4_gpu_tensor_alloc(2u * 4096u * sizeof(float));
    ds4_gpu_tensor *out1 = ds4_gpu_tensor_alloc(2u * 4096u * sizeof(float));
    CHECK(out0 && out1, "allocate rank outputs");

    ds4_gpu_tensor *rank0_tensors[] = {
        out0, rank0.layer[0].q_history, rank0.layer[0].k_history,
        rank0.layer[0].v_history, rank0.layer[0].recurrent};
    ds4_gpu_tensor *rank1_tensors[] = {
        out1, rank1.layer[0].q_history, rank1.layer[0].k_history,
        rank1.layer[0].v_history, rank1.layer[0].recurrent};
    for (uint32_t i = 0; i < 5u; ++i) {
        seed_tensor(rank0_tensors[i], i + 1u);
        memcpy(rank1_tensors[i]->data, rank0_tensors[i]->data,
               (size_t)rank0_tensors[i]->bytes);
    }
    rank0.layer[0].token_count = rank1.layer[0].token_count = 2u;
    ds4_glm5_kda_digest digest0 = {0}, digest1 = {0};
    CHECK(ds4_glm5_kda_layer_digest(&rank0.layer[0], out0, 8192u, &digest0) &&
          ds4_glm5_kda_layer_digest(&rank1.layer[0], out1, 8192u, &digest1) &&
          ds4_glm5_kda_digest_equal(&digest0, &digest1),
          "identical ranks produce equal complete digests");

    for (uint32_t i = 0; i < 5u; ++i) {
        rank1_tensors[i]->data[rank1_tensors[i]->bytes / 2u] ^= 1u;
        CHECK(ds4_glm5_kda_layer_digest(
                  &rank1.layer[0], out1, 8192u, &digest1) &&
              !ds4_glm5_kda_digest_equal(&digest0, &digest1),
              "one-byte tensor corruption rejected");
        rank1_tensors[i]->data[rank1_tensors[i]->bytes / 2u] ^= 1u;
    }
    rank1.layer[0].token_count++;
    CHECK(ds4_glm5_kda_layer_digest(&rank1.layer[0], out1, 8192u, &digest1) &&
          !ds4_glm5_kda_digest_equal(&digest0, &digest1),
          "token-count mismatch rejected");
    ds4_gpu_tensor_free(out1);
    ds4_gpu_tensor_free(out0);
    ds4_glm5_kda_slot_free(&rank1);
    ds4_glm5_kda_slot_free(&rank0);
    fclose(sink);
    fprintf(stderr, "PASS GLM5 KDA complete rank digest and corruption gates\n");
    return 1;
}

int main(void) { return run_digest_test() ? 0 : 1; }
