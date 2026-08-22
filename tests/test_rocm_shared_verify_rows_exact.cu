/* Exactness and timing oracle for batched TP verifier shared-Q8 rows. */

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

static ds4_gpu_tensor tensor_row(ds4_gpu_tensor *base, uint64_t row,
                                 uint64_t elems) {
    ds4_gpu_tensor result = *base;
    result.ptr = (char *)base->ptr + row * elems * sizeof(float);
    result.bytes = elems * sizeof(float);
    result.owner = 0;
    return result;
}

static uint64_t mismatch_count(const float *a, const float *b,
                               uint64_t count, uint64_t *first) {
    uint64_t mismatches = 0u;
    for (uint64_t i = 0; i < count; i++) {
        if (memcmp(a + i, b + i, sizeof(float)) != 0) {
            if (mismatches == 0u) *first = i;
            mismatches++;
        }
    }
    return mismatches;
}

int main(int argc, char **argv) {
    const uint32_t width = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : 5u;
    CHECK(width >= 2u && width <= 5u, "width must be 2..5");
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm backend");

    const uint64_t gu_in = 4096u, gu_out = 1024u;
    const uint64_t down_full = 2048u, down_slice = 1024u, down_out = 4096u;
    const uint64_t gu_weight_bytes = gu_out * (gu_in / 32u) * 34u;
    const uint64_t down_weight_bytes =
        down_out * (down_full / 32u) * 34u;
    const uint64_t gate_offset = 0u;
    const uint64_t up_offset = gu_weight_bytes;
    const uint64_t down_offset = 2u * gu_weight_bytes;
    const uint64_t model_size = down_offset + down_weight_bytes;
    unsigned char *model = (unsigned char *)malloc((size_t)model_size);
    CHECK(model != NULL, "allocate synthetic model");
    pack_q8(model + gate_offset, gu_in, gu_out, 11u);
    pack_q8(model + up_offset, gu_in, gu_out, 23u);
    pack_q8(model + down_offset, down_full, down_out, 37u);
    CHECK(ds4_gpu_set_model_map(model, model_size), "install synthetic model");

    const uint64_t gu_x_count = (uint64_t)width * gu_in;
    const uint64_t gu_y_count = (uint64_t)width * gu_out;
    const uint64_t down_x_count = (uint64_t)width * down_slice;
    const uint64_t down_y_count = (uint64_t)width * down_out;
    float *host_gu_x = (float *)malloc(gu_x_count * sizeof(float));
    float *host_down_x = (float *)malloc(down_x_count * sizeof(float));
    float *host_ref = (float *)malloc(down_y_count * sizeof(float));
    float *host_got = (float *)malloc(down_y_count * sizeof(float));
    CHECK(host_gu_x && host_down_x && host_ref && host_got,
          "allocate host buffers");
    for (uint64_t i = 0; i < gu_x_count; i++) {
        uint32_t bits = UINT32_C(0x3f000000) |
            ((uint32_t)(i * UINT64_C(2654435761) + 0x5a17u) &
             UINT32_C(0x007fffff));
        if ((i * 17u + 3u) & 1u) bits |= UINT32_C(0x80000000);
        memcpy(host_gu_x + i, &bits, sizeof(bits));
    }
    for (uint64_t i = 0; i < down_x_count; i++) {
        uint32_t bits = UINT32_C(0x3e800000) |
            ((uint32_t)(i * UINT64_C(2246822519) + 0x913du) &
             UINT32_C(0x007fffff));
        if ((i * 29u + 5u) & 1u) bits |= UINT32_C(0x80000000);
        memcpy(host_down_x + i, &bits, sizeof(bits));
    }

    ds4_gpu_tensor gu_x = {}, down_x = {}, gate = {}, up = {};
    ds4_gpu_tensor mid_ref = {}, mid_got = {}, down_ref = {}, down_got = {};
    CHECK(ds4_gpu_tensor_alloc_on(&gu_x, 0, gu_x_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&down_x, 0, down_x_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&gate, 0, gu_y_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&up, 0, gu_y_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&mid_ref, 0, gu_y_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&mid_got, 0, gu_y_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&down_ref, 0, down_y_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&down_got, 0, down_y_count * sizeof(float)) == 0,
          "allocate device tensors");
    CHECK(ds4_gpu_tensor_write(&gu_x, 0, host_gu_x,
                               gu_x_count * sizeof(float)) &&
          ds4_gpu_tensor_write(&down_x, 0, host_down_x,
                               down_x_count * sizeof(float)), "upload inputs");

    const float clamp = 7.0f;
    for (uint32_t row = 0u; row < width; row++) {
        ds4_gpu_tensor xr = tensor_row(&gu_x, row, gu_in);
        ds4_gpu_tensor gr = tensor_row(&gate, row, gu_out);
        ds4_gpu_tensor ur = tensor_row(&up, row, gu_out);
        ds4_gpu_tensor mr = tensor_row(&mid_ref, row, gu_out);
        CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                  &gr, &ur, &mr, model, model_size, gate_offset, up_offset,
                  gu_in, gu_out, &xr, clamp), "reference gate/up row");
    }
    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_verify_rows_exact_tensor(
              &gate, &up, &mid_got, model, model_size, gate_offset, up_offset,
              gu_in, gu_out, &gu_x, width, clamp), "candidate gate/up rows");
    CHECK(ds4_gpu_tensor_read(&mid_ref, 0, host_ref,
                              gu_y_count * sizeof(float)) &&
          ds4_gpu_tensor_read(&mid_got, 0, host_got,
                              gu_y_count * sizeof(float)), "read gate/up outputs");
    uint64_t first = 0u;
    uint64_t mismatches = mismatch_count(host_ref, host_got, gu_y_count, &first);
    if (mismatches) {
        fprintf(stderr, "gate/up mismatch=%llu first=%llu ref=%a got=%a\n",
                (unsigned long long)mismatches, (unsigned long long)first,
                host_ref[first], host_got[first]);
    }
    CHECK(mismatches == 0u, "gate/up rows must be bit-exact");

    for (uint64_t k_off = 0u; k_off < down_full; k_off += down_slice) {
        for (uint32_t row = 0u; row < width; row++) {
            ds4_gpu_tensor xr = tensor_row(&down_x, row, down_slice);
            ds4_gpu_tensor yr = tensor_row(&down_ref, row, down_out);
            CHECK(ds4_gpu_matmul_q8_0_kslice_tensor(
                      &yr, model, model_size, down_offset, down_full, k_off,
                      down_slice, down_out, &xr, 0u), "reference down row");
        }
        CHECK(ds4_gpu_matmul_q8_0_kslice_verify_rows_exact_tensor(
                  &down_got, model, model_size, down_offset, down_full, k_off,
                  down_slice, down_out, &down_x, width),
              "candidate down rows");
        CHECK(ds4_gpu_tensor_read(&down_ref, 0, host_ref,
                                  down_y_count * sizeof(float)) &&
              ds4_gpu_tensor_read(&down_got, 0, host_got,
                                  down_y_count * sizeof(float)),
              "read down outputs");
        first = 0u;
        mismatches = mismatch_count(host_ref, host_got, down_y_count, &first);
        if (mismatches) {
            fprintf(stderr,
                    "down k_off=%llu mismatch=%llu first=%llu ref=%a got=%a\n",
                    (unsigned long long)k_off,
                    (unsigned long long)mismatches,
                    (unsigned long long)first,
                    host_ref[first], host_got[first]);
        }
        CHECK(mismatches == 0u, "down rows must be bit-exact");
    }

    hipEvent_t start = nullptr, stop = nullptr;
    CHECK(hipEventCreate(&start) == hipSuccess &&
          hipEventCreate(&stop) == hipSuccess, "create timing events");
    const uint32_t warmup = 5u, iterations = 50u;
    float serial_ms = 0.0f, batch_ms = 0.0f;
    for (uint32_t phase = 0u; phase < 2u; phase++) {
        for (uint32_t i = 0u; i < warmup + iterations; i++) {
            if (i == warmup) CHECK(hipEventRecord(start) == hipSuccess,
                                   "record timing start");
            if (phase == 0u) {
                for (uint32_t row = 0u; row < width; row++) {
                    ds4_gpu_tensor xr = tensor_row(&gu_x, row, gu_in);
                    ds4_gpu_tensor gr = tensor_row(&gate, row, gu_out);
                    ds4_gpu_tensor ur = tensor_row(&up, row, gu_out);
                    ds4_gpu_tensor mr = tensor_row(&mid_ref, row, gu_out);
                    ds4_gpu_tensor dx = tensor_row(&down_x, row, down_slice);
                    ds4_gpu_tensor dy = tensor_row(&down_ref, row, down_out);
                    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                              &gr, &ur, &mr, model, model_size, gate_offset,
                              up_offset, gu_in, gu_out, &xr, clamp) &&
                          ds4_gpu_matmul_q8_0_kslice_tensor(
                              &dy, model, model_size, down_offset, down_full,
                              0u, down_slice, down_out, &dx, 0u),
                          "time serial rows");
                }
            } else {
                CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_verify_rows_exact_tensor(
                          &gate, &up, &mid_got, model, model_size, gate_offset,
                          up_offset, gu_in, gu_out, &gu_x, width, clamp) &&
                      ds4_gpu_matmul_q8_0_kslice_verify_rows_exact_tensor(
                          &down_got, model, model_size, down_offset, down_full,
                          0u, down_slice, down_out, &down_x, width),
                      "time batched rows");
            }
        }
        CHECK(hipEventRecord(stop) == hipSuccess &&
              hipEventSynchronize(stop) == hipSuccess, "finish timing");
        float elapsed = 0.0f;
        CHECK(hipEventElapsedTime(&elapsed, start, stop) == hipSuccess,
              "read timing");
        if (phase == 0u) serial_ms = elapsed / iterations;
        else batch_ms = elapsed / iterations;
    }
    fprintf(stderr,
            "width=%u serial_ms=%.6f batch_ms=%.6f speedup=%.3fx exact=yes\n",
            width, serial_ms, batch_ms, serial_ms / batch_ms);

    float stage_ms[4] = {};
    for (uint32_t phase = 0u; phase < 4u; phase++) {
        for (uint32_t i = 0u; i < warmup + iterations; i++) {
            if (i == warmup) CHECK(hipEventRecord(start) == hipSuccess,
                                   "record stage timing start");
            if (phase == 0u) {
                for (uint32_t row = 0u; row < width; row++) {
                    ds4_gpu_tensor xr = tensor_row(&gu_x, row, gu_in);
                    ds4_gpu_tensor gr = tensor_row(&gate, row, gu_out);
                    ds4_gpu_tensor ur = tensor_row(&up, row, gu_out);
                    ds4_gpu_tensor mr = tensor_row(&mid_ref, row, gu_out);
                    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                              &gr, &ur, &mr, model, model_size, gate_offset,
                              up_offset, gu_in, gu_out, &xr, clamp),
                          "time serial gate/up rows");
                }
            } else if (phase == 1u) {
                CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_verify_rows_exact_tensor(
                          &gate, &up, &mid_got, model, model_size, gate_offset,
                          up_offset, gu_in, gu_out, &gu_x, width, clamp),
                      "time batched gate/up rows");
            } else if (phase == 2u) {
                for (uint32_t row = 0u; row < width; row++) {
                    ds4_gpu_tensor dx = tensor_row(&down_x, row, down_slice);
                    ds4_gpu_tensor dy = tensor_row(&down_ref, row, down_out);
                    CHECK(ds4_gpu_matmul_q8_0_kslice_tensor(
                              &dy, model, model_size, down_offset, down_full,
                              0u, down_slice, down_out, &dx, 0u),
                          "time serial down rows");
                }
            } else {
                CHECK(ds4_gpu_matmul_q8_0_kslice_verify_rows_exact_tensor(
                          &down_got, model, model_size, down_offset, down_full,
                          0u, down_slice, down_out, &down_x, width),
                      "time batched down rows");
            }
        }
        CHECK(hipEventRecord(stop) == hipSuccess &&
              hipEventSynchronize(stop) == hipSuccess,
              "finish stage timing");
        float elapsed = 0.0f;
        CHECK(hipEventElapsedTime(&elapsed, start, stop) == hipSuccess,
              "read stage timing");
        stage_ms[phase] = elapsed / iterations;
    }
    fprintf(stderr,
            "width=%u gate_serial_ms=%.6f gate_batch_ms=%.6f gate_speedup=%.3fx "
            "down_serial_ms=%.6f down_batch_ms=%.6f down_speedup=%.3fx\n",
            width, stage_ms[0], stage_ms[1], stage_ms[0] / stage_ms[1],
            stage_ms[2], stage_ms[3], stage_ms[2] / stage_ms[3]);

    CHECK(hipEventDestroy(start) == hipSuccess &&
          hipEventDestroy(stop) == hipSuccess, "destroy events");
    ds4_gpu_tensor_free_in_place(&gu_x);
    ds4_gpu_tensor_free_in_place(&down_x);
    ds4_gpu_tensor_free_in_place(&gate);
    ds4_gpu_tensor_free_in_place(&up);
    ds4_gpu_tensor_free_in_place(&mid_ref);
    ds4_gpu_tensor_free_in_place(&mid_got);
    ds4_gpu_tensor_free_in_place(&down_ref);
    ds4_gpu_tensor_free_in_place(&down_got);
    free(model);
    free(host_gu_x);
    free(host_down_x);
    free(host_ref);
    free(host_got);
    ds4_gpu_cleanup();
    return 0;
}
