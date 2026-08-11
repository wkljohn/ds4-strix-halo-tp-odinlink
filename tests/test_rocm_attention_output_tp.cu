/* Regression test for the compact-head contract of TP attention output.
 *
 * Each TP rank stores only its owned attention groups at heads[0].  group0
 * selects the corresponding weight rows.  Offsetting heads by group0 again
 * silently projects unwritten storage on rank 1 while all tensor shapes remain
 * valid, so compare two compact rank projections with the full projection. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg) do {                                                \
    if (!(cond)) {                                                           \
        fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);           \
        return 1;                                                            \
    }                                                                        \
} while (0)

/* The backend registers TP device-copy callbacks when a transport is bound.
 * This focused single-device kernel test intentionally has no transport. */
typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

static void pack_q8(unsigned char *weights, uint64_t in_dim, uint64_t out_dim) {
    const uint64_t blocks = (in_dim + 31u) / 32u;
    for (uint64_t row = 0; row < out_dim; row++) {
        for (uint64_t block = 0; block < blocks; block++) {
            unsigned char *dst = weights + (row * blocks + block) * 34u;
            dst[0] = 0x00u;
            dst[1] = 0x3cu; /* fp16 1.0 */
            for (uint64_t lane = 0; lane < 32u; lane++) {
                const uint64_t col = block * 32u + lane;
                const int value = col < in_dim ?
                    (int)((row * 17u + col * 5u + 11u) % 23u) - 11 : 0;
                dst[2u + lane] = (unsigned char)(int8_t)value;
            }
        }
    }
}

int main(void) {
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm backend");

    const uint64_t group_dim = 64;
    const uint64_t rank = 64;
    const uint32_t groups = 4;
    const uint32_t owned = 2;
    const uint64_t out_dim = 48;
    const uint64_t low_dim = (uint64_t)groups * rank;
    const uint64_t a_bytes = (uint64_t)groups * rank *
                             ((group_dim + 31u) / 32u) * 34u;
    const uint64_t b_offset = a_bytes;
    const uint64_t model_size = a_bytes + out_dim *
                                ((low_dim + 31u) / 32u) * 34u;
    unsigned char *model = (unsigned char *)malloc((size_t)model_size);
    float *host_heads = (float *)malloc((size_t)groups * group_dim * sizeof(float));
    float *host_full = (float *)malloc((size_t)out_dim * sizeof(float));
    float *host_sum = (float *)malloc((size_t)out_dim * sizeof(float));
    CHECK(model && host_heads && host_full && host_sum, "allocate host buffers");
    pack_q8(model, group_dim, (uint64_t)groups * rank);
    pack_q8(model + b_offset, low_dim, out_dim);
    for (uint64_t i = 0; i < (uint64_t)groups * group_dim; i++) {
        host_heads[i] = (float)((int)(i % 29u) - 14) * 0.0625f;
    }
    CHECK(ds4_gpu_set_model_map(model, model_size), "install synthetic model");

    ds4_gpu_tensor heads = {}, low = {}, low0 = {}, low1 = {};
    ds4_gpu_tensor full = {}, part0 = {}, part1 = {}, sum = {};
    CHECK(ds4_gpu_tensor_alloc_on(&heads, 0,
            (uint64_t)groups * group_dim * sizeof(float)) == 0, "heads");
    CHECK(ds4_gpu_tensor_alloc_on(&low, 0, low_dim * sizeof(float)) == 0, "low");
    CHECK(ds4_gpu_tensor_alloc_on(&low0, 0,
            (uint64_t)owned * rank * sizeof(float)) == 0, "low0");
    CHECK(ds4_gpu_tensor_alloc_on(&low1, 0,
            (uint64_t)owned * rank * sizeof(float)) == 0, "low1");
    CHECK(ds4_gpu_tensor_alloc_on(&full, 0, out_dim * sizeof(float)) == 0, "full");
    CHECK(ds4_gpu_tensor_alloc_on(&part0, 0, out_dim * sizeof(float)) == 0, "part0");
    CHECK(ds4_gpu_tensor_alloc_on(&part1, 0, out_dim * sizeof(float)) == 0, "part1");
    CHECK(ds4_gpu_tensor_alloc_on(&sum, 0, out_dim * sizeof(float)) == 0, "sum");
    CHECK(ds4_gpu_tensor_write(&heads, 0, host_heads,
            (uint64_t)groups * group_dim * sizeof(float)), "upload heads");
    ds4_gpu_tensor *heads0 = ds4_gpu_tensor_view(
            &heads, 0, (uint64_t)owned * group_dim * sizeof(float));
    ds4_gpu_tensor *heads1 = ds4_gpu_tensor_view(
            &heads, (uint64_t)owned * group_dim * sizeof(float),
            (uint64_t)owned * group_dim * sizeof(float));
    CHECK(heads0 && heads1, "compact rank head views");

    CHECK(ds4_gpu_attention_output_q8_batch_tensor(
              &full, &low, NULL, NULL, model, model_size, 0, b_offset,
              group_dim, rank, groups, out_dim, &heads, 1) &&
          ds4_gpu_attention_output_q8_tp_tensor(
              &part0, &low0, model, model_size, 0, b_offset,
              group_dim, rank, groups, 0, owned, out_dim, heads0) &&
          ds4_gpu_attention_output_q8_tp_tensor(
              &part1, &low1, model, model_size, 0, b_offset,
              group_dim, rank, groups, owned, owned, out_dim, heads1) &&
          ds4_gpu_add_tensor(&sum, &part0, &part1, (uint32_t)out_dim) &&
          ds4_gpu_tensor_read(&full, 0, host_full, out_dim * sizeof(float)) &&
          ds4_gpu_tensor_read(&sum, 0, host_sum, out_dim * sizeof(float)),
          "run full and compact TP projections");

    float max_diff = 0.0f;
    float max_ref = 0.0f;
    uint64_t max_row = 0;
    double sq_diff = 0.0;
    double sq_ref = 0.0;
    for (uint64_t i = 0; i < out_dim; i++) {
        const float diff = fabsf(host_full[i] - host_sum[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_row = i;
        }
        if (fabsf(host_full[i]) > max_ref) max_ref = fabsf(host_full[i]);
        sq_diff += (double)diff * diff;
        sq_ref += (double)host_full[i] * host_full[i];
    }
    fprintf(stderr,
            "test_rocm_attention_output_tp: max_diff=%g max_ref=%g "
            "rel_rms=%g row=%llu full=%g split=%g\n",
            max_diff, max_ref, sqrt(sq_diff / sq_ref),
            (unsigned long long)max_row,
            host_full[max_row], host_sum[max_row]);
    /* The full-batch kernel quantizes all four activation groups together,
     * while each TP half chooses its own Q8 activation scale.  They therefore
     * need not be bit-exact; the measured envelope is about 0.32% RMS.  The
     * compact rank-1 call itself would fail closed under the old double-offset
     * implementation because heads1 is intentionally only half-width. */
    CHECK(sqrt(sq_diff / sq_ref) <= 0.005 && max_diff <= max_ref * 0.005,
          "compact TP sum must track full projection within Q8 scale drift");
    fprintf(stderr, "test_rocm_attention_output_tp: PASS\n");

    ds4_gpu_tensor_free(heads0);
    ds4_gpu_tensor_free(heads1);
    ds4_gpu_tensor_free_in_place(&heads);
    ds4_gpu_tensor_free_in_place(&low);
    ds4_gpu_tensor_free_in_place(&low0);
    ds4_gpu_tensor_free_in_place(&low1);
    ds4_gpu_tensor_free_in_place(&full);
    ds4_gpu_tensor_free_in_place(&part0);
    ds4_gpu_tensor_free_in_place(&part1);
    ds4_gpu_tensor_free_in_place(&sum);
    free(model);
    free(host_heads);
    free(host_full);
    free(host_sum);
    ds4_gpu_cleanup();
    return 0;
}
