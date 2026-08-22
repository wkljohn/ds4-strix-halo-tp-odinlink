/* Exactness and timing oracle for dense-Q8 verifier QKV projections. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <hip/hip_runtime.h>
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
    CHECK(host_pair_x && host_one_x && host_ref && host_got,
          "allocate host buffers");
    fill_f32(host_pair_x, pair_x_count, 0x5a17u);
    fill_f32(host_one_x, one_x_count, 0x913du);

    ds4_gpu_tensor pair_x = {}, pair_ref0 = {}, pair_ref1 = {};
    ds4_gpu_tensor pair_got0 = {}, pair_got1 = {}, one_x = {};
    ds4_gpu_tensor one_ref = {}, one_got = {};
    CHECK(ds4_gpu_tensor_alloc_on(&pair_x, 0, pair_x_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&pair_ref0, 0, pair_y0_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&pair_ref1, 0, pair_y1_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&pair_got0, 0, pair_y0_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&pair_got1, 0, pair_y1_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&one_x, 0, one_x_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&one_ref, 0, one_y_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&one_got, 0, one_y_count * sizeof(float)) == 0,
          "allocate device tensors");
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
    return 0;
}
