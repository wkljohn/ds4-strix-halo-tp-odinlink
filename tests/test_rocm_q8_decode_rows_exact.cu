/* Exactness and timing oracle for dense-Q8 verifier QKV projections. */

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

typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

static void pack_q8(unsigned char *weights, uint64_t in_dim,
                    uint64_t out_dim, uint32_t salt) {
    const uint64_t blocks = in_dim / 32u;
    for (uint64_t row = 0; row < out_dim; row++) {
        for (uint64_t block = 0; block < blocks; block++) {
            unsigned char *dst = weights + (row * blocks + block) * 34u;
            const uint16_t scale = (uint16_t)(0x2800u +
                ((row * 3u + block * 5u + salt) & 0x03ffu));
            dst[0] = (unsigned char)(scale & 0xffu);
            dst[1] = (unsigned char)(scale >> 8u);
            for (uint64_t lane = 0; lane < 32u; lane++) {
                const int value = (int)((row * 17u + block * 13u +
                    lane * 5u + salt) % 127u) - 63;
                dst[2u + lane] = (unsigned char)(int8_t)value;
            }
        }
    }
}

static void fill_f32(float *dst, uint64_t count, uint32_t salt) {
    for (uint64_t i = 0; i < count; i++) {
        uint32_t bits = UINT32_C(0x3e800000) |
            ((uint32_t)(i * UINT64_C(2654435761) + salt) &
             UINT32_C(0x007fffff));
        if ((i * 17u + salt) & 1u) bits |= UINT32_C(0x80000000);
        memcpy(dst + i, &bits, sizeof(bits));
    }
}

static ds4_gpu_tensor tensor_row(ds4_gpu_tensor *base, uint64_t row,
                                 uint64_t elems) {
    ds4_gpu_tensor result = *base;
    result.ptr = (char *)base->ptr + row * elems * sizeof(float);
    result.bytes = elems * sizeof(float);
    result.owner = 0;
    return result;
}

static uint64_t mismatches(const float *a, const float *b, uint64_t count,
                           uint64_t *first) {
    uint64_t n = 0u;
    for (uint64_t i = 0; i < count; i++) {
        if (memcmp(a + i, b + i, sizeof(float)) != 0) {
            if (n++ == 0u) *first = i;
        }
    }
    return n;
}

static uint32_t reference_argmax(const float *row, uint32_t count) {
    uint32_t best = 0u;
    for (uint32_t i = 1u; i < count; i++) {
        if (row[i] > row[best]) best = i;
    }
    return best;
}

/* Merge in rank order. Keeping rank 0 unless rank 1 is strictly better also
 * preserves the shipped fail-closed behavior when global logit zero is NaN. */
static uint32_t merge_split_top1(uint32_t rank0_id, float rank0_value,
                                 uint32_t rank1_id, float rank1_value) {
    return rank1_value > rank0_value ||
           (rank1_value == rank0_value && rank1_id < rank0_id)
        ? rank1_id : rank0_id;
}

int main(int argc, char **argv) {
    const uint32_t width = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : 5u;
    const bool output_head_shape =
        argc > 2 && strcmp(argv[2], "output-head") == 0;
    CHECK(width >= 2u && width <= 5u, "width must be 2..5");
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm backend");

    const uint64_t pair_in = 4096u, pair_out0 = 1024u, pair_out1 = 512u;
    const uint64_t one_in = output_head_shape ? 7168u : 1024u;
    const uint64_t one_out = output_head_shape ? 129280u : 16384u;
    const uint64_t pair0_bytes = pair_out0 * (pair_in / 32u) * 34u;
    const uint64_t pair1_bytes = pair_out1 * (pair_in / 32u) * 34u;
    const uint64_t one_bytes = one_out * (one_in / 32u) * 34u;
    const uint64_t pair0_off = 0u;
    const uint64_t pair1_off = pair0_off + pair0_bytes;
    const uint64_t one_off = pair1_off + pair1_bytes;
    const uint64_t model_size = one_off + one_bytes;
    unsigned char *model = (unsigned char *)malloc((size_t)model_size);
    CHECK(model != NULL, "allocate synthetic model");
    pack_q8(model + pair0_off, pair_in, pair_out0, 11u);
    pack_q8(model + pair1_off, pair_in, pair_out1, 23u);
    pack_q8(model + one_off, one_in, one_out, 37u);
    CHECK(ds4_gpu_set_model_map(model, model_size), "install synthetic model");

    const uint64_t pair_x_count = (uint64_t)width * pair_in;
    const uint64_t pair_y0_count = (uint64_t)width * pair_out0;
    const uint64_t pair_y1_count = (uint64_t)width * pair_out1;
    const uint64_t one_x_count = (uint64_t)width * one_in;
    const uint64_t one_y_count = (uint64_t)width * one_out;
    float *host_pair_x = (float *)malloc(pair_x_count * sizeof(float));
    float *host_one_x = (float *)malloc(one_x_count * sizeof(float));
    float *host_ref = (float *)malloc(one_y_count * sizeof(float));
    float *host_got = (float *)malloc(one_y_count * sizeof(float));
    const uint64_t half_out = one_out / 2u;
    const uint64_t half_y_count = (uint64_t)width * half_out;
    float *host_half0 = output_head_shape ?
        (float *)malloc(half_y_count * sizeof(float)) : NULL;
    float *host_half1 = output_head_shape ?
        (float *)malloc(half_y_count * sizeof(float)) : NULL;
    CHECK(host_pair_x && host_one_x && host_ref && host_got &&
          (!output_head_shape || (host_half0 && host_half1)),
          "allocate host buffers");
    fill_f32(host_pair_x, pair_x_count, 0x5a17u);
    fill_f32(host_one_x, one_x_count, 0x913du);

    ds4_gpu_tensor pair_x = {}, pair_ref0 = {}, pair_ref1 = {};
    ds4_gpu_tensor pair_got0 = {}, pair_got1 = {}, one_x = {};
    ds4_gpu_tensor one_ref = {}, one_got = {};
    ds4_gpu_tensor half0 = {}, half1 = {}, half_ids0 = {}, half_ids1 = {};
    ds4_gpu_tensor half_vals0 = {}, half_vals1 = {};
    CHECK(ds4_gpu_tensor_alloc_on(&pair_x, 0, pair_x_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&pair_ref0, 0, pair_y0_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&pair_ref1, 0, pair_y1_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&pair_got0, 0, pair_y0_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&pair_got1, 0, pair_y1_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&one_x, 0, one_x_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&one_ref, 0, one_y_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&one_got, 0, one_y_count * sizeof(float)) == 0,
          "allocate device tensors");
    if (output_head_shape) {
        CHECK(ds4_gpu_tensor_alloc_on(&half0, 0,
                                      half_y_count * sizeof(float)) == 0 &&
              ds4_gpu_tensor_alloc_on(&half1, 0,
                                      half_y_count * sizeof(float)) == 0 &&
              ds4_gpu_tensor_alloc_on(&half_ids0, 0,
                                      width * sizeof(uint32_t)) == 0 &&
              ds4_gpu_tensor_alloc_on(&half_ids1, 0,
                                      width * sizeof(uint32_t)) == 0 &&
              ds4_gpu_tensor_alloc_on(&half_vals0, 0,
                                      width * sizeof(float)) == 0 &&
              ds4_gpu_tensor_alloc_on(&half_vals1, 0,
                                      width * sizeof(float)) == 0,
              "allocate split output-head tensors");
    }
    CHECK(ds4_gpu_tensor_write(&pair_x, 0, host_pair_x,
                               pair_x_count * sizeof(float)) &&
          ds4_gpu_tensor_write(&one_x, 0, host_one_x,
                               one_x_count * sizeof(float)), "upload inputs");

    for (uint32_t row = 0u; row < width; row++) {
        ds4_gpu_tensor xr = tensor_row(&pair_x, row, pair_in);
        ds4_gpu_tensor y0 = tensor_row(&pair_ref0, row, pair_out0);
        ds4_gpu_tensor y1 = tensor_row(&pair_ref1, row, pair_out1);
        CHECK(ds4_gpu_matmul_q8_0_pair_tensor(
                  &y0, &y1, model, model_size, pair0_off, pair1_off,
                  pair_in, pair_out0, pair_out1, &xr, 1u),
              "reference pair row");
        xr = tensor_row(&one_x, row, one_in);
        ds4_gpu_tensor y = tensor_row(&one_ref, row, one_out);
        CHECK(ds4_gpu_matmul_q8_0_tensor(&y, model, model_size, one_off,
                                         one_in, one_out, &xr, 1u),
              "reference single row");
    }
    CHECK(ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor(
              &pair_got0, &pair_got1, model, model_size, pair0_off, pair1_off,
              pair_in, pair_out0, pair_out1, &pair_x, width),
          "candidate pair rows");
    CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
              &one_got, model, model_size, one_off, one_in, one_out,
              &one_x, width), "candidate single rows");
    if (output_head_shape) {
        const uint64_t row_bytes = (one_in / 32u) * 34u;
        CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                  &half0, model, model_size, one_off, one_in, half_out,
                  &one_x, width), "candidate lower output-head half");
        CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                  &half1, model, model_size,
                  one_off + half_out * row_bytes,
                  one_in, half_out, &one_x, width),
              "candidate upper output-head half");
        CHECK(ds4_gpu_argmax_rows_value_tensor(
                  &half_ids0, &half_vals0, &half0,
                  (uint32_t)half_out, width, 0u),
              "lower output-head local top1");
        CHECK(ds4_gpu_argmax_rows_value_tensor(
                  &half_ids1, &half_vals1, &half1,
                  (uint32_t)half_out, width, (uint32_t)half_out),
              "upper output-head local top1");
    }

    uint64_t first = 0u, mismatch = 0u;
    CHECK(ds4_gpu_tensor_read(&pair_ref0, 0, host_ref,
                              pair_y0_count * sizeof(float)) &&
          ds4_gpu_tensor_read(&pair_got0, 0, host_got,
                              pair_y0_count * sizeof(float)), "read pair0");
    mismatch = mismatches(host_ref, host_got, pair_y0_count, &first);
    if (mismatch) fprintf(stderr, "pair0 mismatch=%llu first=%llu ref=%a got=%a\n",
                          (unsigned long long)mismatch,
                          (unsigned long long)first, host_ref[first], host_got[first]);
    CHECK(mismatch == 0u, "pair0 must be bit-exact");
    CHECK(ds4_gpu_tensor_read(&pair_ref1, 0, host_ref,
                              pair_y1_count * sizeof(float)) &&
          ds4_gpu_tensor_read(&pair_got1, 0, host_got,
                              pair_y1_count * sizeof(float)), "read pair1");
    first = 0u;
    mismatch = mismatches(host_ref, host_got, pair_y1_count, &first);
    if (mismatch) fprintf(stderr, "pair1 mismatch=%llu first=%llu ref=%a got=%a\n",
                          (unsigned long long)mismatch,
                          (unsigned long long)first, host_ref[first], host_got[first]);
    CHECK(mismatch == 0u, "pair1 must be bit-exact");
    CHECK(ds4_gpu_tensor_read(&one_ref, 0, host_ref,
                              one_y_count * sizeof(float)) &&
          ds4_gpu_tensor_read(&one_got, 0, host_got,
                              one_y_count * sizeof(float)), "read single");
    first = 0u;
    mismatch = mismatches(host_ref, host_got, one_y_count, &first);
    if (mismatch) fprintf(stderr, "single mismatch=%llu first=%llu ref=%a got=%a\n",
                          (unsigned long long)mismatch,
                          (unsigned long long)first, host_ref[first], host_got[first]);
    CHECK(mismatch == 0u, "single must be bit-exact");

    if (output_head_shape) {
        uint32_t ids0[5] = {}, ids1[5] = {};
        float vals0[5] = {}, vals1[5] = {};
        CHECK(ds4_gpu_tensor_read(&half0, 0, host_half0,
                                  half_y_count * sizeof(float)) &&
              ds4_gpu_tensor_read(&half1, 0, host_half1,
                                  half_y_count * sizeof(float)) &&
              ds4_gpu_tensor_read(&half_ids0, 0, ids0,
                                  width * sizeof(uint32_t)) &&
              ds4_gpu_tensor_read(&half_ids1, 0, ids1,
                                  width * sizeof(uint32_t)) &&
              ds4_gpu_tensor_read(&half_vals0, 0, vals0,
                                  width * sizeof(float)) &&
              ds4_gpu_tensor_read(&half_vals1, 0, vals1,
                                  width * sizeof(float)),
              "read split output-head oracle");
        for (uint32_t row = 0u; row < width; row++) {
            uint64_t split_first = 0u;
            CHECK(mismatches(host_got + (uint64_t)row * one_out,
                             host_half0 + (uint64_t)row * half_out,
                             half_out, &split_first) == 0u,
                  "lower half must equal full-head slice");
            split_first = 0u;
            CHECK(mismatches(host_got + (uint64_t)row * one_out + half_out,
                             host_half1 + (uint64_t)row * half_out,
                             half_out, &split_first) == 0u,
                  "upper half must equal full-head slice");
            const uint32_t full_top = reference_argmax(
                    host_got + (uint64_t)row * one_out,
                    (uint32_t)one_out);
            const uint32_t merged = merge_split_top1(
                    ids0[row], vals0[row], ids1[row], vals1[row]);
            CHECK(merged == full_top,
                  "merged split top1 must equal full argmax");
            CHECK(memcmp(host_got + (uint64_t)row * one_out + ids0[row],
                         vals0 + row, sizeof(float)) == 0,
                  "lower top1 value must equal full-head value");
            CHECK(memcmp(host_got + (uint64_t)row * one_out + ids1[row],
                         vals1 + row, sizeof(float)) == 0,
                  "upper top1 value must equal full-head value");
        }
        CHECK(merge_split_top1(7u, 1.0f,
                               (uint32_t)half_out + 3u, 1.0f) == 7u,
              "rank-ordered exact tie");
        CHECK(merge_split_top1(0u, NAN,
                               (uint32_t)half_out + 1u, 9.0f) == 0u,
              "global-zero NaN fail-closed merge");
        {
            float local0[16], local1[16];
            for (uint32_t i = 0u; i < 16u; i++) {
                local0[i] = -INFINITY;
                local1[i] = -INFINITY;
            }
            local0[0] = NAN;
            local0[8u + 7u] = 3.0f;
            local1[0] = NAN;
            local1[1] = 9.0f;
            local1[8u + 3u] = 3.0f;
            ds4_gpu_tensor tiny0 = {}, tiny1 = {};
            CHECK(ds4_gpu_tensor_alloc_on(&tiny0, 0, sizeof(local0)) == 0 &&
                  ds4_gpu_tensor_alloc_on(&tiny1, 0, sizeof(local1)) == 0 &&
                  ds4_gpu_tensor_write(&tiny0, 0, local0, sizeof(local0)) &&
                  ds4_gpu_tensor_write(&tiny1, 0, local1, sizeof(local1)),
                  "prepare split argmax edge cases");
            CHECK(ds4_gpu_argmax_rows_value_tensor(
                      &half_ids0, &half_vals0, &tiny0, 8u, 2u, 0u) &&
                  ds4_gpu_argmax_rows_value_tensor(
                      &half_ids1, &half_vals1, &tiny1, 8u, 2u, 8u),
                  "run split argmax edge cases");
            uint32_t edge_ids0[2] = {}, edge_ids1[2] = {};
            float edge_vals0[2] = {}, edge_vals1[2] = {};
            CHECK(ds4_gpu_tensor_read(&half_ids0, 0, edge_ids0,
                                      sizeof(edge_ids0)) &&
                  ds4_gpu_tensor_read(&half_ids1, 0, edge_ids1,
                                      sizeof(edge_ids1)) &&
                  ds4_gpu_tensor_read(&half_vals0, 0, edge_vals0,
                                      sizeof(edge_vals0)) &&
                  ds4_gpu_tensor_read(&half_vals1, 0, edge_vals1,
                                      sizeof(edge_vals1)),
                  "read split argmax edge cases");
            CHECK(edge_ids0[0] == 0u && isnan(edge_vals0[0]) &&
                  edge_ids1[0] == 9u && edge_vals1[0] == 9.0f &&
                  merge_split_top1(edge_ids0[0], edge_vals0[0],
                                   edge_ids1[0], edge_vals1[0]) == 0u,
                  "global-zero NaN semantics");
            CHECK(edge_ids0[1] == 7u && edge_ids1[1] == 11u &&
                  merge_split_top1(edge_ids0[1], edge_vals0[1],
                                   edge_ids1[1], edge_vals1[1]) == 7u,
                  "cross-shard exact tie semantics");
        }
        const uint32_t selected_row = width - 1u;
        memcpy(host_ref,
               host_half0 + (uint64_t)selected_row * half_out,
               half_out * sizeof(float));
        memcpy(host_ref + half_out,
               host_half1 + (uint64_t)selected_row * half_out,
               half_out * sizeof(float));
        uint64_t reconstructed_first = 0u;
        CHECK(mismatches(host_ref,
                         host_got + (uint64_t)selected_row * one_out,
                         one_out, &reconstructed_first) == 0u,
              "reconstructed selected row must equal full row");
    }

    hipEvent_t start = nullptr, stop = nullptr;
    CHECK(hipEventCreate(&start) == hipSuccess &&
          hipEventCreate(&stop) == hipSuccess, "create timing events");
    const uint32_t warmup = 4u, iterations = 30u;
    float elapsed_ms[4] = {};
    for (uint32_t phase = 0u; phase < 4u; phase++) {
        for (uint32_t i = 0u; i < warmup + iterations; i++) {
            if (i == warmup) CHECK(hipEventRecord(start) == hipSuccess,
                                   "record timing start");
            if (phase == 0u) {
                for (uint32_t row = 0u; row < width; row++) {
                    ds4_gpu_tensor xr = tensor_row(&pair_x, row, pair_in);
                    ds4_gpu_tensor y0 = tensor_row(&pair_ref0, row, pair_out0);
                    ds4_gpu_tensor y1 = tensor_row(&pair_ref1, row, pair_out1);
                    CHECK(ds4_gpu_matmul_q8_0_pair_tensor(
                              &y0, &y1, model, model_size, pair0_off, pair1_off,
                              pair_in, pair_out0, pair_out1, &xr, 1u),
                          "time serial pair");
                }
            } else if (phase == 1u) {
                CHECK(ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor(
                          &pair_got0, &pair_got1, model, model_size,
                          pair0_off, pair1_off, pair_in, pair_out0, pair_out1,
                          &pair_x, width), "time exact pair");
            } else if (phase == 2u) {
                for (uint32_t row = 0u; row < width; row++) {
                    ds4_gpu_tensor xr = tensor_row(&one_x, row, one_in);
                    ds4_gpu_tensor y = tensor_row(&one_ref, row, one_out);
                    CHECK(ds4_gpu_matmul_q8_0_tensor(
                              &y, model, model_size, one_off, one_in, one_out,
                              &xr, 1u), "time serial single");
                }
            } else {
                CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                          &one_got, model, model_size, one_off, one_in,
                          one_out, &one_x, width), "time exact single");
            }
        }
        CHECK(hipEventRecord(stop) == hipSuccess &&
              hipEventSynchronize(stop) == hipSuccess, "finish timing");
        CHECK(hipEventElapsedTime(&elapsed_ms[phase], start, stop) == hipSuccess,
              "read timing");
        elapsed_ms[phase] /= iterations;
    }
    fprintf(stderr,
            "shape=%s width=%u pair_serial_ms=%.6f pair_exact_ms=%.6f "
            "pair_speedup=%.3fx "
            "single_serial_ms=%.6f single_exact_ms=%.6f single_speedup=%.3fx exact=yes\n",
            output_head_shape ? "output-head" : "qkv",
            width, elapsed_ms[0], elapsed_ms[1], elapsed_ms[0] / elapsed_ms[1],
            elapsed_ms[2], elapsed_ms[3], elapsed_ms[2] / elapsed_ms[3]);

    if (output_head_shape) {
        const uint64_t row_bytes = (one_in / 32u) * 34u;
        float split_ms[3] = {};
        for (uint32_t phase = 0u; phase < 3u; phase++) {
            for (uint32_t i = 0u; i < warmup + iterations; i++) {
                if (i == warmup) CHECK(hipEventRecord(start) == hipSuccess,
                                       "record split timing start");
                if (phase == 0u) {
                    CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                              &one_got, model, model_size, one_off,
                              one_in, one_out, &one_x, width),
                          "time full output head");
                    CHECK(ds4_gpu_argmax_rows_value_tensor(
                              &half_ids0, &half_vals0, &one_got,
                              (uint32_t)one_out, width, 0u),
                          "time full output top1");
                } else {
                    ds4_gpu_tensor *half = phase == 1u ? &half0 : &half1;
                    ds4_gpu_tensor *ids = phase == 1u ? &half_ids0 : &half_ids1;
                    ds4_gpu_tensor *vals = phase == 1u ? &half_vals0 : &half_vals1;
                    const uint64_t off = phase == 1u ? one_off :
                        one_off + half_out * row_bytes;
                    CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                              half, model, model_size, off,
                              one_in, half_out, &one_x, width),
                          "time split output head");
                    CHECK(ds4_gpu_argmax_rows_value_tensor(
                              ids, vals, half, (uint32_t)half_out, width,
                              phase == 1u ? 0u : (uint32_t)half_out),
                          "time split output top1");
                }
            }
            CHECK(hipEventRecord(stop) == hipSuccess &&
                  hipEventSynchronize(stop) == hipSuccess,
                  "finish split timing");
            CHECK(hipEventElapsedTime(&split_ms[phase], start, stop) == hipSuccess,
                  "read split timing");
            split_ms[phase] /= iterations;
        }
        const float slower_half = split_ms[1] > split_ms[2] ?
            split_ms[1] : split_ms[2];
        const float split_speedup = split_ms[0] / slower_half;
        fprintf(stderr,
                "output_head_split width=%u full_head_top1_ms=%.6f "
                "lower_head_top1_ms=%.6f upper_head_top1_ms=%.6f "
                "parallel_speedup=%.3fx exact=yes\n",
                width, split_ms[0], split_ms[1], split_ms[2], split_speedup);
        CHECK(split_speedup >= 1.6f,
              "split output-head isolated speedup must reach 1.6x");
    }
    return 0;
}
