/* Small-step oracle for cache-free multi-row packed-Q8 K-slice projection. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(expr, message) do {                                      \
    if (!(expr)) {                                                     \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__); \
        return 1;                                                      \
    }                                                                 \
} while (0)

typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

static void pack_q8(std::vector<unsigned char> &weights,
                    uint32_t in_dim, uint32_t out_dim) {
    const uint32_t blocks = in_dim / 32u;
    for (uint32_t row = 0u; row < out_dim; ++row) {
        for (uint32_t block = 0u; block < blocks; ++block) {
            unsigned char *dst = weights.data() +
                ((uint64_t)row * blocks + block) * 34u;
            dst[0] = 0x00u;
            dst[1] = 0x34u; /* IEEE fp16 0.25 */
            for (uint32_t lane = 0u; lane < 32u; ++lane) {
                const int value = (int)((row * 17u + block * 13u +
                                         lane * 5u + 11u) % 31u) - 15;
                dst[2u + lane] = (unsigned char)(int8_t)value;
            }
        }
    }
}

int main() {
    constexpr uint32_t full_in = 2048u;
    constexpr uint32_t slice_begin = 512u;
    constexpr uint32_t slice_dim = 1024u;
    constexpr uint32_t out_dim = 256u;
    constexpr uint32_t rows3 = 3u;
    constexpr uint32_t rows9 = 9u;
    const uint64_t full_row_bytes = (full_in / 32u) * 34u;
    const uint64_t slice_row_bytes = (slice_dim / 32u) * 34u;
    const uint64_t full_bytes = (uint64_t)out_dim * full_row_bytes;
    const uint64_t slice_bytes = (uint64_t)out_dim * slice_row_bytes;
    std::vector<unsigned char> model(full_bytes + slice_bytes);
    std::vector<unsigned char> full(full_bytes);
    pack_q8(full, full_in, out_dim);
    std::memcpy(model.data(), full.data(), full_bytes);
    const uint32_t first_block = slice_begin / 32u;
    const uint32_t slice_blocks = slice_dim / 32u;
    for (uint32_t row = 0u; row < out_dim; ++row) {
        std::memcpy(model.data() + full_bytes +
                        (uint64_t)row * slice_row_bytes,
                    full.data() + (uint64_t)row * full_row_bytes +
                        (uint64_t)first_block * 34u,
                    (uint64_t)slice_blocks * 34u);
    }
    std::vector<float> input9((uint64_t)rows9 * slice_dim);
    for (uint64_t i = 0u; i < input9.size(); ++i) {
        input9[i] = (float)((int)(i * 19u % 127u) - 63) / 64.0f;
    }

    CHECK(setenv("DS4_ROCM_Q8_SMALL_BATCH_DP4A", "1", 1) == 0,
          "enable established Q8 DP4A token tile");
    ds4_gpu_config config = {};
    config.n_gpus = 1u;
    config.device_indices[0] = 0u;
    CHECK(ds4_gpu_init_multi(&config) &&
              ds4_gpu_set_model_map(model.data(), model.size()),
          "initialize ROCm with synthetic packed-Q8 model");

    ds4_gpu_tensor *x3 = ds4_gpu_tensor_alloc(
        (uint64_t)rows3 * slice_dim * sizeof(float));
    ds4_gpu_tensor *reference3 = ds4_gpu_tensor_alloc(
        (uint64_t)rows3 * out_dim * sizeof(float));
    ds4_gpu_tensor *candidate3 = ds4_gpu_tensor_alloc(
        (uint64_t)rows3 * out_dim * sizeof(float));
    ds4_gpu_tensor *decode1 = ds4_gpu_tensor_alloc(
        (uint64_t)out_dim * sizeof(float));
    CHECK(x3 && reference3 && candidate3 && decode1 &&
              ds4_gpu_tensor_write(x3, 0u, input9.data(),
                                    (uint64_t)rows3 * slice_dim *
                                        sizeof(float)) &&
              ds4_gpu_matmul_q8_0_tensor(
                  reference3, model.data(), model.size(), full_bytes,
                  slice_dim, out_dim, x3, rows3) &&
              ds4_gpu_matmul_q8_0_kslice_rows_tensor(
                  candidate3, model.data(), model.size(), 0u,
                  full_in, out_dim, slice_begin, slice_dim, x3, rows3),
          "run contiguous and strided three-row projections");
    std::vector<float> host_reference3((uint64_t)rows3 * out_dim);
    std::vector<float> host_candidate3((uint64_t)rows3 * out_dim);
    CHECK(ds4_gpu_tensor_read(reference3, 0u, host_reference3.data(),
                              host_reference3.size() * sizeof(float)) &&
              ds4_gpu_tensor_read(candidate3, 0u, host_candidate3.data(),
                                  host_candidate3.size() * sizeof(float)) &&
              std::memcmp(host_reference3.data(), host_candidate3.data(),
                          host_reference3.size() * sizeof(float)) == 0,
          "three-row K-slice is bit-identical to repacked compact Q8");
    CHECK(ds4_gpu_matmul_q8_0_kslice_rows_tensor(
              decode1, model.data(), model.size(), 0u,
              full_in, out_dim, slice_begin, slice_dim, x3, 1u),
          "run one-row f32-activation decode seam");
    std::vector<float> host_decode1(out_dim);
    CHECK(ds4_gpu_tensor_read(decode1, 0u, host_decode1.data(),
                              host_decode1.size() * sizeof(float)),
          "read one-row decode seam");
    double seam_diff2 = 0.0, seam_ref2 = 0.0, seam_max_abs = 0.0;
    for (uint32_t i = 0u; i < out_dim; ++i) {
        const double reference = host_decode1[i];
        const double delta = (double)host_candidate3[i] - reference;
        seam_diff2 += delta * delta;
        seam_ref2 += reference * reference;
        seam_max_abs = std::max(seam_max_abs, std::fabs(delta));
    }
    const double seam_nrmse = std::sqrt(seam_diff2 / seam_ref2);
    CHECK(std::isfinite(seam_nrmse) && seam_nrmse <= 1.6e-2 &&
              seam_max_abs <= 5.0e-1,
          "dynamic-INT8 batch seam remains numerically bounded");

    ds4_gpu_tensor *x9 = ds4_gpu_tensor_alloc(
        (uint64_t)rows9 * slice_dim * sizeof(float));
    ds4_gpu_tensor *out9 = ds4_gpu_tensor_alloc(
        (uint64_t)rows9 * out_dim * sizeof(float));
    ds4_gpu_tensor *x8 = ds4_gpu_tensor_alloc(
        UINT64_C(8) * slice_dim * sizeof(float));
    ds4_gpu_tensor *out8 = ds4_gpu_tensor_alloc(
        UINT64_C(8) * out_dim * sizeof(float));
    ds4_gpu_tensor *x2 = ds4_gpu_tensor_alloc(
        UINT64_C(2) * slice_dim * sizeof(float));
    ds4_gpu_tensor *out2 = ds4_gpu_tensor_alloc(
        UINT64_C(2) * out_dim * sizeof(float));
    std::vector<float> input2(UINT64_C(2) * slice_dim);
    std::memcpy(input2.data(), input9.data(), slice_dim * sizeof(float));
    std::memcpy(input2.data() + slice_dim,
                input9.data() + UINT64_C(8) * slice_dim,
                slice_dim * sizeof(float));
    CHECK(x9 && out9 && x8 && out8 && x2 && out2 &&
              ds4_gpu_tensor_write(x9, 0u, input9.data(),
                                    input9.size() * sizeof(float)) &&
              ds4_gpu_tensor_write(x8, 0u, input9.data(),
                                    UINT64_C(8) * slice_dim * sizeof(float)) &&
              ds4_gpu_tensor_write(x2, 0u, input2.data(),
                                    input2.size() * sizeof(float)) &&
              ds4_gpu_matmul_q8_0_kslice_rows_tensor(
                  out9, model.data(), model.size(), 0u,
                  full_in, out_dim, slice_begin, slice_dim, x9, rows9) &&
              ds4_gpu_matmul_q8_0_kslice_rows_tensor(
                  out8, model.data(), model.size(), 0u,
                  full_in, out_dim, slice_begin, slice_dim, x8, 8u) &&
              ds4_gpu_matmul_q8_0_kslice_rows_tensor(
                  out2, model.data(), model.size(), 0u,
                  full_in, out_dim, slice_begin, slice_dim, x2, 2u),
          "run bounded 8+1 chunk coverage");
    std::vector<float> host9((uint64_t)rows9 * out_dim);
    std::vector<float> host8(UINT64_C(8) * out_dim);
    std::vector<float> host2(UINT64_C(2) * out_dim);
    CHECK(ds4_gpu_tensor_read(out9, 0u, host9.data(),
                              host9.size() * sizeof(float)) &&
              ds4_gpu_tensor_read(out8, 0u, host8.data(),
                                  host8.size() * sizeof(float)) &&
              ds4_gpu_tensor_read(out2, 0u, host2.data(),
                                  host2.size() * sizeof(float)) &&
              std::memcmp(host9.data(), host8.data(),
                          host8.size() * sizeof(float)) == 0 &&
              std::memcmp(host9.data() + UINT64_C(8) * out_dim,
                          host2.data() + out_dim,
                          out_dim * sizeof(float)) == 0,
          "nine rows preserve exact output across the internal chunk edge");
    CHECK(!ds4_gpu_matmul_q8_0_kslice_rows_tensor(
              candidate3, model.data(), model.size(), 0u,
              full_in, out_dim, slice_begin + 1u, slice_dim, x3, rows3),
          "misaligned Q8 K slice fails closed");

    std::vector<float> input_strided((uint64_t)rows3 * full_in, -7.0f);
    for (uint32_t row = 0u; row < rows3; ++row) {
        std::memcpy(input_strided.data() + (uint64_t)row * full_in +
                        slice_begin,
                    input9.data() + (uint64_t)row * slice_dim,
                    (uint64_t)slice_dim * sizeof(float));
    }
    ds4_gpu_tensor *x_strided = ds4_gpu_tensor_alloc(
        input_strided.size() * sizeof(float));
    ds4_gpu_tensor *strided_reference = ds4_gpu_tensor_alloc(
        (uint64_t)rows3 * out_dim * sizeof(float));
    ds4_gpu_tensor *strided_candidate = ds4_gpu_tensor_alloc(
        (uint64_t)rows3 * out_dim * sizeof(float));
    CHECK(x_strided && strided_reference && strided_candidate &&
              ds4_gpu_tensor_write(x_strided, 0u, input_strided.data(),
                                    input_strided.size() * sizeof(float)),
          "allocate physical-stride K-slice fixture");
    for (uint32_t row = 0u; row < rows3; ++row) {
        ds4_gpu_tensor *out_row = ds4_gpu_tensor_view(
            strided_reference, (uint64_t)row * out_dim * sizeof(float),
            (uint64_t)out_dim * sizeof(float));
        const int ok = out_row && ds4_gpu_matmul_q8_0_kslice_tensor(
            out_row, model.data(), model.size(), 0u,
            full_in, slice_begin, slice_dim, out_dim, x_strided,
            (uint64_t)row * full_in + slice_begin);
        ds4_gpu_tensor_free(out_row);
        CHECK(ok, "run scalar F32 reference for physical-stride row");
    }
    CHECK(ds4_rocm_q8_kslice_f32_rows_strided(
              strided_candidate, model.data(), model.size(), 0u,
              full_in, out_dim, slice_begin, slice_dim, x_strided,
              rows3, full_in) == 1 &&
              !ds4_rocm_q8_kslice_f32_rows_strided(
                  strided_candidate, model.data(), model.size(), 0u,
                  full_in, out_dim, slice_begin, slice_dim, x_strided,
                  rows3, slice_begin + slice_dim - 1u),
          "run strided F32 token tile and reject an undersized row stride");
    std::vector<float> host_strided_reference((uint64_t)rows3 * out_dim);
    std::vector<float> host_strided_candidate((uint64_t)rows3 * out_dim);
    CHECK(ds4_gpu_tensor_read(
              strided_reference, 0u, host_strided_reference.data(),
              host_strided_reference.size() * sizeof(float)) &&
              ds4_gpu_tensor_read(
                  strided_candidate, 0u, host_strided_candidate.data(),
                  host_strided_candidate.size() * sizeof(float)),
          "read physical-stride F32 token-tile comparison");
    double strided_diff2 = 0.0, strided_ref2 = 0.0;
    double strided_max_abs = 0.0;
    for (uint64_t i = 0u; i < host_strided_reference.size(); ++i) {
        const double reference = host_strided_reference[i];
        const double delta = host_strided_candidate[i] - reference;
        strided_diff2 += delta * delta;
        strided_ref2 += reference * reference;
        strided_max_abs = std::max(strided_max_abs, std::fabs(delta));
    }
    const double strided_nrmse = std::sqrt(strided_diff2 / strided_ref2);
    CHECK(std::isfinite(strided_nrmse) &&
              std::memcmp(host_strided_candidate.data(),
                          host_strided_reference.data(),
                          host_strided_reference.size() * sizeof(float)) == 0,
          "strided F32 token tile bit-matches one-row F32 K-slice arithmetic");

    std::fprintf(stderr,
                 "PASS ROCm Q8 K-slice rows: repacked_exact=1 "
                 "chunk_8_plus_1_exact=1 rows=3,9 "
                 "decode_batch_seam_nrmse=%.9g "
                 "decode_batch_seam_max_abs=%.9g "
                 "strided_f32_nrmse=%.9g strided_f32_max_abs=%.9g\n",
                 seam_nrmse, seam_max_abs,
                 strided_nrmse, strided_max_abs);
    ds4_gpu_tensor_free(strided_candidate);
    ds4_gpu_tensor_free(strided_reference);
    ds4_gpu_tensor_free(x_strided);
    ds4_gpu_tensor_free(out2);
    ds4_gpu_tensor_free(x2);
    ds4_gpu_tensor_free(out8);
    ds4_gpu_tensor_free(x8);
    ds4_gpu_tensor_free(out9);
    ds4_gpu_tensor_free(x9);
    ds4_gpu_tensor_free(candidate3);
    ds4_gpu_tensor_free(reference3);
    ds4_gpu_tensor_free(decode1);
    ds4_gpu_tensor_free(x3);
    ds4_gpu_cleanup();
    return 0;
}
