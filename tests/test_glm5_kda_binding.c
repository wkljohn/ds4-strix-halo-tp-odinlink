#include "ds4_glm5_kda.h"

#include <stdio.h>
#include <string.h>

/* The binding test exercises only schedule construction.  These stubs keep
 * the state-owner translation unit linkable without allocating a GPU. */
ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    (void)bytes;
    return NULL;
}

int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value,
                            uint64_t count) {
    (void)tensor;
    (void)value;
    (void)count;
    return 0;
}

void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    (void)tensor;
}

uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    (void)tensor;
    return 0;
}

#define CHECK(expr, message) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL %s\n", message); \
        return 0; \
    } \
} while (0)

static void make_real_presence(bool *has_kda, bool *has_mla,
                               uint32_t layers) {
    memset(has_kda, 0, layers * sizeof(*has_kda));
    memset(has_mla, 0, layers * sizeof(*has_mla));
    for (uint32_t layer = 0; layer < layers; ++layer) {
        const bool mla = layer == 45u || (layer & 3u) == 3u;
        has_mla[layer] = mla;
        has_kda[layer] = !mla;
    }
}

static int test_real_and_shifted_presence(void) {
    enum { layers = 46 };
    bool has_kda[layers], has_mla[layers];
    ds4_glm5_layer_kind schedule[layers];
    uint32_t kda_count = UINT32_MAX;
    make_real_presence(has_kda, has_mla, layers);
    CHECK(ds4_glm5_kda_build_schedule(schedule, layers, has_kda, has_mla,
                                      layers, &kda_count),
          "real tensor-presence schedule accepted");
    CHECK(kda_count == 34u, "real tensor-presence KDA count");
    for (uint32_t layer = 0; layer < layers; ++layer) {
        CHECK(schedule[layer].layer == layer, "schedule records layer index");
        CHECK(schedule[layer].is_kda == has_kda[layer],
              "schedule follows tensor presence");
    }

    /* Rotate the exact presence pattern by one layer.  The resulting layer
     * numbers deliberately violate the source GGUF's modulo pattern. */
    bool shifted_kda[layers], shifted_mla[layers];
    for (uint32_t layer = 0; layer < layers; ++layer) {
        const uint32_t source = (layer + layers - 1u) % layers;
        shifted_kda[layer] = has_kda[source];
        shifted_mla[layer] = has_mla[source];
    }
    memset(schedule, 0xa5, sizeof(schedule));
    kda_count = UINT32_MAX;
    CHECK(ds4_glm5_kda_build_schedule(schedule, layers,
                                      shifted_kda, shifted_mla,
                                      layers, &kda_count),
          "shifted non-modulo schedule accepted");
    CHECK(kda_count == 34u, "shifted schedule preserves KDA count");
    for (uint32_t layer = 0; layer < layers; ++layer) {
        CHECK(schedule[layer].is_kda == shifted_kda[layer],
              "shifted schedule follows presence rather than modulo");
    }
    return 1;
}

static int test_all_kda_and_rejections(void) {
    enum { layers = 7 };
    bool has_kda[layers], has_mla[layers];
    ds4_glm5_layer_kind schedule[layers];
    memset(has_kda, 1, sizeof(has_kda));
    memset(has_mla, 0, sizeof(has_mla));
    uint32_t kda_count = 0;
    CHECK(ds4_glm5_kda_build_schedule(schedule, layers, has_kda, has_mla,
                                      layers, &kda_count),
          "all-KDA schedule accepted");
    CHECK(kda_count == layers, "all-KDA count derived");

    ds4_glm5_layer_kind before[layers];
    memcpy(before, schedule, sizeof(before));
    const uint32_t before_count = kda_count;
    has_mla[2] = true;
    CHECK(!ds4_glm5_kda_build_schedule(schedule, layers, has_kda, has_mla,
                                       layers, &kda_count),
          "layer with both attention families rejected");
    CHECK(memcmp(schedule, before, sizeof(before)) == 0 &&
          kda_count == before_count,
          "both-family rejection leaves outputs unchanged");

    has_kda[2] = false;
    has_mla[2] = false;
    CHECK(!ds4_glm5_kda_build_schedule(schedule, layers, has_kda, has_mla,
                                       layers, &kda_count),
          "layer with neither attention family rejected");
    CHECK(memcmp(schedule, before, sizeof(before)) == 0 &&
          kda_count == before_count,
          "missing-family rejection leaves outputs unchanged");
    has_kda[2] = true;
    CHECK(!ds4_glm5_kda_build_schedule(schedule, layers - 1u,
                                       has_kda, has_mla, layers, &kda_count),
          "short output capacity rejected");
    CHECK(!ds4_glm5_kda_build_schedule(NULL, layers, has_kda, has_mla,
                                       layers, &kda_count),
          "null output rejected");
    CHECK(!ds4_glm5_kda_build_schedule(schedule, layers, NULL, has_mla,
                                       layers, &kda_count),
          "null KDA presence rejected");
    CHECK(!ds4_glm5_kda_build_schedule(schedule, layers, has_kda, NULL,
                                       layers, &kda_count),
          "null MLA presence rejected");
    CHECK(!ds4_glm5_kda_build_schedule(schedule, layers, has_kda, has_mla,
                                       layers, NULL),
          "null KDA count rejected");
    return 1;
}

int main(void) {
    const int ok = test_real_and_shifted_presence() &&
                   test_all_kda_and_rejections();
    if (ok) fprintf(stderr, "PASS GLM5 tensor-derived KDA schedule\n");
    return ok ? 0 : 1;
}
