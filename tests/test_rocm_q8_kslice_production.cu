/* Production-shape oracle for the GLM-5.3 Q4 MLA output projection. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
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

static int run_case(const std::vector<unsigned char> &model,
                    uint64_t full_bytes, uint32_t full_in, uint32_t out_dim,
                    uint32_t in_start, uint32_t rows) {
    const uint64_t slice_dim = full_in / 2u;
    const uint64_t stride = full_in;
    const uint64_t input_elems = (uint64_t)rows * stride;
    std::vector<float> input(input_elems, -17.0f);
    for (uint32_t row = 0u; row < rows; ++row) {
        for (uint32_t col = 0u; col < slice_dim; ++col)
            input[(uint64_t)row * stride + in_start + col] =
                (float)((int)((row * 29u + col * 7u + 3u) % 127u) - 63) / 64.0f;
    }
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(input_elems * sizeof(float));
    ds4_gpu_tensor *reference = ds4_gpu_tensor_alloc(
        (uint64_t)rows * out_dim * sizeof(float));
    ds4_gpu_tensor *candidate = ds4_gpu_tensor_alloc(
        (uint64_t)rows * out_dim * sizeof(float));
    CHECK(x && reference && candidate &&
              ds4_gpu_tensor_write(x, 0u, input.data(),
                                   input.size() * sizeof(float)),
          "allocate production-shape fixture");

    for (uint32_t row = 0u; row < rows; ++row) {
        ds4_gpu_tensor *out_row = ds4_gpu_tensor_view(
            reference, (uint64_t)row * out_dim * sizeof(float),
            (uint64_t)out_dim * sizeof(float));
        CHECK(out_row && ds4_gpu_matmul_q8_0_kslice_tensor(
                  out_row, model.data(), model.size(), 0u,
                  full_in, in_start, slice_dim, out_dim, x,
                  (uint64_t)row * stride + in_start),
              "run scalar production reference");
        ds4_gpu_tensor_free(out_row);
    }

    std::vector<float> host_reference((uint64_t)rows * out_dim);
    std::vector<float> host_candidate(host_reference.size());
    CHECK(ds4_gpu_tensor_read(reference, 0u, host_reference.data(),
                              host_reference.size() * sizeof(float)),
          "read production reference");
    for (int repeat = 0; repeat < 10; ++repeat) {
        CHECK(ds4_rocm_q8_kslice_f32_rows_strided(
                  candidate, model.data(), model.size(), 0u,
                  full_in, out_dim, in_start, slice_dim, x,
                  in_start, rows, stride) == 1,
              "run production strided tile");
        CHECK(ds4_gpu_tensor_read(candidate, 0u, host_candidate.data(),
                                  host_candidate.size() * sizeof(float)) &&
                  std::memcmp(host_reference.data(), host_candidate.data(),
                              host_reference.size() * sizeof(float)) == 0,
              "production strided tile is bit-identical");
    }

    std::fprintf(stderr, "PASS production Q8 strided oracle: start=%u rows=%u repeats=10\n",
                 in_start, rows);
    ds4_gpu_tensor_free(candidate);
    ds4_gpu_tensor_free(reference);
    ds4_gpu_tensor_free(x);
    return 0;
}

int main() {
    constexpr uint32_t full_in = 16384u;
    constexpr uint32_t out_dim = 4096u;
    const uint64_t full_bytes = (uint64_t)out_dim * (full_in / 32u) * 34u;
    std::vector<unsigned char> model(full_bytes);
    pack_q8(model, full_in, out_dim);

    ds4_gpu_config config = {};
    config.n_gpus = 1u;
    config.device_indices[0] = 0u;
    CHECK(ds4_gpu_init_multi(&config) &&
              ds4_gpu_set_model_map(model.data(), model.size()),
          "initialize ROCm production oracle");
    CHECK(run_case(model, full_bytes, full_in, out_dim, 0u, 512u) == 0,
          "rank 0 512-row production oracle");
    CHECK(run_case(model, full_bytes, full_in, out_dim, 8192u, 512u) == 0,
          "rank 1 512-row production oracle");
    CHECK(run_case(model, full_bytes, full_in, out_dim, 0u, 88u) == 0,
          "rank 0 residual production oracle");
    CHECK(run_case(model, full_bytes, full_in, out_dim, 8192u, 25u) == 0,
          "rank 1 residual production oracle");
    ds4_gpu_cleanup();
    return 0;
}
