/* Exact controls for the dense shared-expert and HC pieces needed by the
 * cache-free Q4_K output-row shard. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

enum {
    IN_DIM = 2048,
    OUT_DIM = 4096,
    K_HALF = IN_DIM / 2,
    OUT_HALF = OUT_DIM / 2,
    N_HC = 4,
    Q8_0_BLOCK_BYTES = 34,
};

#define CHECK(cond, msg) do {                                                \
    if (!(cond)) {                                                           \
        fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);           \
        return 1;                                                            \
    }                                                                        \
} while (0)

static int alloc_tensor(ds4_gpu_tensor *t, uint64_t bytes) {
    memset(t, 0, sizeof(*t));
    return ds4_gpu_tensor_alloc_on(t, 0, bytes) == 0;
}

static int upload(ds4_gpu_tensor *t, const void *src, uint64_t bytes) {
    return ds4_gpu_tensor_write(t, 0, src, bytes) != 0;
}

static uint64_t fnv1a64(const void *data, size_t bytes) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < bytes; i++) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static void pack_q8_0(unsigned char *model) {
    const uint32_t blocks = IN_DIM / 32u;
    for (uint32_t row = 0; row < OUT_DIM; row++) {
        for (uint32_t block = 0; block < blocks; block++) {
            unsigned char *dst = model +
                ((uint64_t)row * blocks + block) * Q8_0_BLOCK_BYTES;
            /* IEEE fp16 1/64, little-endian. */
            dst[0] = 0x00u;
            dst[1] = 0x24u;
            for (uint32_t k = 0; k < 32u; k++) {
                dst[2u + k] = (unsigned char)(int8_t)(
                    (int)((row * 13u + block * 7u + k * 5u) % 31u) - 15);
            }
        }
    }
}

int main(void) {
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm backend");

    const uint64_t row_bytes = (uint64_t)(IN_DIM / 32u) * Q8_0_BLOCK_BYTES;
    const uint64_t model_bytes = (uint64_t)OUT_DIM * row_bytes;
    unsigned char *model = (unsigned char *)malloc((size_t)model_bytes);
    float *x_host[2] = {
        (float *)malloc((size_t)K_HALF * sizeof(float)),
        (float *)malloc((size_t)K_HALF * sizeof(float))};
    float *full_host = (float *)malloc((size_t)OUT_DIM * sizeof(float));
    float *compact_host = (float *)malloc((size_t)OUT_HALF * sizeof(float));
    float *residual_host =
        (float *)malloc((size_t)N_HC * OUT_DIM * sizeof(float));
    float split_host[24];
    float *hc_ref_host =
        (float *)malloc((size_t)N_HC * OUT_DIM * sizeof(float));
    float *hc_got_host =
        (float *)malloc((size_t)N_HC * OUT_DIM * sizeof(float));
    CHECK(model && x_host[0] && x_host[1] && full_host && compact_host &&
          residual_host && hc_ref_host && hc_got_host,
          "allocate host controls");
    pack_q8_0(model);
    for (uint32_t rank = 0; rank < 2u; rank++) {
        for (uint32_t k = 0; k < K_HALF; k++) {
            x_host[rank][k] =
                (float)((int)((k * 11u + rank * 17u) % 67u) - 33) / 512.0f;
        }
    }
    for (uint32_t i = 0; i < N_HC * OUT_DIM; i++) {
        residual_host[i] =
            (float)((int)((i * 3u + 19u) % 53u) - 26) / 1024.0f;
    }
    for (uint32_t i = 0; i < 24u; i++) {
        split_host[i] =
            (float)((int)((i * 7u + 5u) % 29u) - 14) / 128.0f;
    }

    ds4_gpu_tensor x[2] = {}, full_partial[2] = {}, full_sum = {};
    ds4_gpu_tensor compact_partial[2] = {}, compact_sum[2] = {};
    ds4_gpu_tensor residual = {}, split = {}, hc_ref = {}, hc_got = {};
    const uint64_t full_bytes = (uint64_t)OUT_DIM * sizeof(float);
    const uint64_t half_bytes = (uint64_t)OUT_HALF * sizeof(float);
    const uint64_t hc_bytes = (uint64_t)N_HC * OUT_DIM * sizeof(float);
    CHECK(ds4_gpu_set_model_map(model, model_bytes), "install Q8 model map");
    for (uint32_t rank = 0; rank < 2u; rank++) {
        CHECK(alloc_tensor(&x[rank], (uint64_t)K_HALF * sizeof(float)) &&
              alloc_tensor(&full_partial[rank], full_bytes) &&
              alloc_tensor(&compact_partial[rank], half_bytes) &&
              alloc_tensor(&compact_sum[rank], half_bytes) &&
              upload(&x[rank], x_host[rank],
                     (uint64_t)K_HALF * sizeof(float)),
              "allocate/upload Q8 slice tensors");
    }
    CHECK(alloc_tensor(&full_sum, full_bytes) &&
          alloc_tensor(&residual, hc_bytes) &&
          alloc_tensor(&split, sizeof(split_host)) &&
          alloc_tensor(&hc_ref, hc_bytes) && alloc_tensor(&hc_got, hc_bytes) &&
          upload(&residual, residual_host, hc_bytes) &&
          upload(&split, split_host, sizeof(split_host)),
          "allocate/upload HC tensors");

    for (uint32_t rank = 0; rank < 2u; rank++) {
        CHECK(ds4_gpu_matmul_q8_0_kslice_rows_tensor(
                  &full_partial[rank], model, model_bytes, 0u,
                  IN_DIM, OUT_DIM, (uint64_t)rank * K_HALF, K_HALF,
                  &x[rank], 1u),
              "full-output Q8 K-slice reference");
    }
    CHECK(ds4_gpu_add_tensor(&full_sum, &full_partial[0], &full_partial[1],
                             OUT_DIM),
          "canonical full shared partial sum");
    CHECK(ds4_gpu_tensor_read(&full_sum, 0, full_host, full_bytes),
          "read full shared reference");

    for (uint32_t half = 0; half < 2u; half++) {
        for (uint32_t rank = 0; rank < 2u; rank++) {
            CHECK(ds4_gpu_matmul_q8_0_kslice_output_rows_tensor(
                      &compact_partial[rank], model, model_bytes, 0u,
                      IN_DIM, (uint64_t)rank * K_HALF, K_HALF, OUT_DIM,
                      (uint64_t)half * OUT_HALF, OUT_HALF, &x[rank], 1u),
                  "compact-output Q8 K-slice");
        }
        CHECK(ds4_gpu_add_tensor(&compact_sum[half], &compact_partial[0],
                                 &compact_partial[1], OUT_HALF) &&
              ds4_gpu_tensor_read(&compact_sum[half], 0, compact_host,
                                  half_bytes),
              "read compact shared sum");
        CHECK(memcmp(compact_host, full_host + (uint64_t)half * OUT_HALF,
                     (size_t)half_bytes) == 0,
              "compact shared output rows are bitwise exact");
    }
    CHECK(!ds4_gpu_matmul_q8_0_kslice_output_rows_tensor(
              &compact_partial[0], model, model_bytes, 0u,
              IN_DIM, 0u, K_HALF, OUT_DIM, OUT_DIM - 1u, 2u, &x[0], 1u),
          "reject output row overflow");

    CHECK(ds4_gpu_hc_expand_split_tensor(
              &hc_ref, &full_sum, &residual, &split, OUT_DIM, N_HC) &&
          ds4_gpu_hc_expand_split_two_halves_tensor(
              &hc_got, &compact_sum[0], &compact_sum[1],
              &residual, &split, OUT_DIM, N_HC) &&
          ds4_gpu_tensor_read(&hc_ref, 0, hc_ref_host, hc_bytes) &&
          ds4_gpu_tensor_read(&hc_got, 0, hc_got_host, hc_bytes),
          "run/read HC two-half control");
    CHECK(memcmp(hc_ref_host, hc_got_host, (size_t)hc_bytes) == 0,
          "two-half HC expansion is bitwise exact");
    printf("q4k_row_shard_support shared_fnv64=%016llx "
           "hc_fnv64=%016llx exact=1\n",
           (unsigned long long)fnv1a64(full_host, (size_t)full_bytes),
           (unsigned long long)fnv1a64(hc_got_host, (size_t)hc_bytes));

    ds4_gpu_tensor_free_in_place(&hc_got);
    ds4_gpu_tensor_free_in_place(&hc_ref);
    ds4_gpu_tensor_free_in_place(&split);
    ds4_gpu_tensor_free_in_place(&residual);
    for (uint32_t rank = 0; rank < 2u; rank++) {
        ds4_gpu_tensor_free_in_place(&compact_sum[rank]);
        ds4_gpu_tensor_free_in_place(&compact_partial[rank]);
        ds4_gpu_tensor_free_in_place(&full_partial[rank]);
        ds4_gpu_tensor_free_in_place(&x[rank]);
    }
    ds4_gpu_tensor_free_in_place(&full_sum);
    free(hc_got_host);
    free(hc_ref_host);
    free(residual_host);
    free(compact_host);
    free(full_host);
    free(x_host[1]);
    free(x_host[0]);
    free(model);
    ds4_gpu_cleanup();
    return 0;
}
