#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "tests/glm5_gguf_test.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CHECK(expr, message) do {                                         \
    if (!(expr)) {                                                        \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__); \
        return false;                                                     \
    }                                                                     \
} while (0)

namespace {

constexpr uint32_t kHidden = 4096u;
constexpr uint32_t kHc = 4u;
constexpr uint32_t kHcMix = 24u;
constexpr uint32_t kShared = 2048u;
constexpr uint32_t kHalf = 1024u;
constexpr uint32_t kQk = 32u;
constexpr uint32_t kQ8Block = 34u;
constexpr float kClamp = 10.0f;

struct RuntimeGuard {
    bool active = false;
    ~RuntimeGuard() { if (active) ds4_gpu_cleanup(); }
};

struct Tensors {
    std::vector<ds4_gpu_tensor *> values;
    ~Tensors() {
        for (auto it = values.rbegin(); it != values.rend(); ++it)
            ds4_gpu_tensor_free(*it);
    }
    ds4_gpu_tensor *f32(uint64_t count) {
        ds4_gpu_tensor *value = ds4_gpu_tensor_alloc(count * sizeof(float));
        if (value) values.push_back(value);
        return value;
    }
};

template <typename T>
bool read_array(const std::string &path, size_t count, std::vector<T> &out) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return false;
    const std::streamoff size = input.tellg();
    if (size < 0 || (uint64_t)size != (uint64_t)count * sizeof(T)) return false;
    input.seekg(0);
    out.resize(count);
    return (bool)input.read((char *)out.data(), size);
}

bool read_tensor(ds4_gpu_tensor *tensor, size_t count,
                 std::vector<float> &out) {
    out.resize(count);
    return ds4_gpu_tensor_read(tensor, 0u, out.data(),
                               (uint64_t)count * sizeof(float));
}

uint64_t fnv1a64(const void *data, uint64_t bytes) {
    const auto *p = static_cast<const uint8_t *>(data);
    uint64_t hash = UINT64_C(1469598103934665603);
    for (uint64_t i = 0; i < bytes; ++i) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

bool compare(const char *name, const std::vector<float> &got,
             const std::vector<float> &expected, double max_abs_limit,
             double nmse_limit) {
    CHECK(got.size() == expected.size(), "FFN shared comparison shape");
    double maximum = 0.0, error2 = 0.0, reference2 = 0.0;
    double reference_max = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        CHECK(std::isfinite(got[i]) && std::isfinite(expected[i]),
              "finite FFN shared output");
        const double error = (double)got[i] - expected[i];
        maximum = std::max(maximum, std::fabs(error));
        error2 += error * error;
        reference2 += (double)expected[i] * expected[i];
        reference_max = std::max(reference_max,
                                 std::fabs((double)expected[i]));
    }
    const double nmse = error2 / std::max(reference2, 1.0e-30);
    std::fprintf(stderr,
                 "GLM5 FFN shared %-18s count=%zu max_abs=%.9g "
                 "reference_max=%.9g nmse=%.9g\n",
                 name, got.size(), maximum, reference_max, nmse);
    CHECK(reference_max >= 1.0e-6, "non-degenerate FFN shared reference");
    CHECK(maximum <= max_abs_limit && nmse <= nmse_limit,
          "FFN shared numerical envelope");
    return true;
}

bool run_test() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    const char *oracle = std::getenv("DS4_GLM5_MLA_COMPOSE_ORACLE_PREFIX");
    CHECK(model && model[0] && oracle && oracle[0],
          "model and composed oracle prefix are required");
    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM5 GGUF");

    uint64_t fn = 0u, base = 0u, scale = 0u, norm = 0u;
    uint64_t gate_w = 0u, up_w = 0u, down_w = 0u;
    CHECK(gguf.tensor("blk.3.hc_ffn_fn.weight", {16384u, 24u}, 30u, fn) &&
          gguf.tensor("blk.3.hc_ffn_base.weight", {24u}, 0u, base) &&
          gguf.tensor("blk.3.hc_ffn_scale.weight", {3u}, 0u, scale) &&
          gguf.tensor("blk.3.ffn_norm.weight", {kHidden}, 0u, norm) &&
          gguf.tensor("blk.3.ffn_gate_shexp.weight",
                      {kHidden, kShared}, 8u, gate_w) &&
          gguf.tensor("blk.3.ffn_up_shexp.weight",
                      {kHidden, kShared}, 8u, up_w) &&
          gguf.tensor("blk.3.ffn_down_shexp.weight",
                      {kShared, kHidden}, 8u, down_w),
          "bind block-3 FFN mHC and shared Q8 tensors");

    const std::string prefix(oracle);
    std::vector<float> hc_carried, expected_split, expected_post, expected_comb;
    std::vector<float> expected_collapsed, expected_hidden;
    std::vector<float> expected_gate, expected_up, expected_mid;
    std::vector<float> expected_output;
    CHECK(read_array(prefix + ".hc_carried.f32", kHc * kHidden, hc_carried) &&
          read_array(prefix + ".ffn_split.f32", kHcMix, expected_split) &&
          read_array(prefix + ".ffn_post.f32", kHc, expected_post) &&
          read_array(prefix + ".ffn_comb.f32", kHc * kHc, expected_comb) &&
          read_array(prefix + ".ffn_collapsed.f32", kHidden,
                     expected_collapsed) &&
          read_array(prefix + ".ffn_hidden.f32", kHidden, expected_hidden) &&
          read_array(prefix + ".shared_gate.f32", kShared, expected_gate) &&
          read_array(prefix + ".shared_up.f32", kShared, expected_up) &&
          read_array(prefix + ".shared_mid.f32", kShared, expected_mid) &&
          read_array(prefix + ".shared_output.f32", kHidden,
                     expected_output),
          "read FFN mHC/shared independent oracle arrays");

    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "gfx1151 device available");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    RuntimeGuard runtime;
    CHECK(ds4_gpu_init_multi(&config), "initialize gfx1151");
    runtime.active = true;

    Tensors t;
    ds4_gpu_tensor *d_residual = t.f32((uint64_t)kHc * kHidden);
    ds4_gpu_tensor *d_flat = t.f32((uint64_t)kHc * kHidden);
    ds4_gpu_tensor *d_mix = t.f32(kHcMix);
    ds4_gpu_tensor *d_split = t.f32(kHcMix);
    ds4_gpu_tensor *d_collapsed = t.f32(kHidden);
    ds4_gpu_tensor *d_hidden = t.f32(kHidden);
    ds4_gpu_tensor *d_gate = t.f32(kShared);
    ds4_gpu_tensor *d_up = t.f32(kShared);
    ds4_gpu_tensor *d_mid = t.f32(kShared);
    ds4_gpu_tensor *d_full = t.f32(kHidden);
    ds4_gpu_tensor *d_gate_half[2] = {t.f32(kHalf), t.f32(kHalf)};
    ds4_gpu_tensor *d_up_half[2] = {t.f32(kHalf), t.f32(kHalf)};
    ds4_gpu_tensor *d_mid_half[2] = {t.f32(kHalf), t.f32(kHalf)};
    ds4_gpu_tensor *d_out_half[2] = {t.f32(kHidden), t.f32(kHidden)};
    ds4_gpu_tensor *d_sum = t.f32(kHidden);
    CHECK(d_residual && d_flat && d_mix && d_split && d_collapsed &&
          d_hidden && d_gate && d_up && d_mid && d_full && d_gate_half[0] &&
          d_gate_half[1] && d_up_half[0] && d_up_half[1] && d_mid_half[0] &&
          d_mid_half[1] && d_out_half[0] && d_out_half[1] && d_sum,
          "allocate bounded FFN shared tensors");
    CHECK(ds4_gpu_tensor_write(d_residual, 0u, hc_carried.data(),
                               (uint64_t)hc_carried.size() * sizeof(float)) &&
          ds4_gpu_rms_norm_plain_rows_tensor(
              d_flat, d_residual, kHc * kHidden, 1u, 1.0e-5f) &&
          ds4_gpu_matmul_bf16_tensor(d_mix, gguf.map, gguf.size, fn,
                                     kHc * kHidden, kHcMix, d_flat, 1u) &&
          ds4_gpu_hc_split_weighted_sum_norm_tensor(
              d_collapsed, d_hidden, d_split, d_mix, d_residual,
              gguf.map, gguf.size, scale, base, norm, kHidden, kHc, 20u,
              1.0e-6f, 1.0e-5f),
          "execute real block-3 FFN mHC pre-stage");

    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
              d_gate, d_up, d_mid, gguf.map, gguf.size, gate_w, up_w,
              kHidden, kShared, d_hidden, kClamp) &&
          ds4_gpu_matmul_q8_0_tensor(d_full, gguf.map, gguf.size, down_w,
                                     kShared, kHidden, d_mid, 1u),
          "execute full shared Q8 expert control");

    const uint64_t gate_row_bytes = (kHidden / kQk) * kQ8Block;
    for (uint32_t half = 0u; half < 2u; ++half) {
        const uint64_t row_offset = (uint64_t)half * kHalf * gate_row_bytes;
        CHECK(ds4_gpu_matmul_q8_0_tensor(
                  d_gate_half[half], gguf.map, gguf.size,
                  gate_w + row_offset, kHidden, kHalf, d_hidden, 1u) &&
              ds4_gpu_matmul_q8_0_tensor(
                  d_up_half[half], gguf.map, gguf.size,
                  up_w + row_offset, kHidden, kHalf, d_hidden, 1u) &&
              ds4_gpu_swiglu_tensor(d_mid_half[half], d_gate_half[half],
                                    d_up_half[half], kHalf, kClamp, 1.0f) &&
              ds4_gpu_matmul_q8_0_kslice_tensor(
                  d_out_half[half], gguf.map, gguf.size, down_w, kShared,
                  (uint64_t)half * kHalf, kHalf, kHidden, d_mid_half[half],
                  0u),
              "execute one shared-expert 1024-wide rank slice");
    }
    CHECK(ds4_gpu_add_tensor(d_sum, d_out_half[0], d_out_half[1], kHidden) &&
          ds4_gpu_synchronize(), "sum shared-expert rank slices");

    std::vector<float> got_split, got_collapsed, got_hidden, got_gate, got_up;
    std::vector<float> got_mid, got_full, got_sum, got_half0, got_half1;
    CHECK(read_tensor(d_split, kHcMix, got_split) &&
          read_tensor(d_collapsed, kHidden, got_collapsed) &&
          read_tensor(d_hidden, kHidden, got_hidden) &&
          read_tensor(d_gate, kShared, got_gate) &&
          read_tensor(d_up, kShared, got_up) &&
          read_tensor(d_mid, kShared, got_mid) &&
          read_tensor(d_full, kHidden, got_full) &&
          read_tensor(d_sum, kHidden, got_sum) &&
          read_tensor(d_out_half[0], kHidden, got_half0) &&
          read_tensor(d_out_half[1], kHidden, got_half1),
          "read FFN mHC and shared-expert boundaries");
    std::vector<float> got_post(got_split.begin() + kHc,
                                got_split.begin() + 2u * kHc);
    std::vector<float> got_comb(got_split.begin() + 2u * kHc,
                                got_split.end());
    CHECK(compare("ffn_split", got_split, expected_split, 2.0e-6, 1.0e-10) &&
          compare("ffn_post", got_post, expected_post, 2.0e-6, 1.0e-10) &&
          compare("ffn_comb", got_comb, expected_comb, 2.0e-6, 1.0e-10) &&
          compare("ffn_collapsed", got_collapsed, expected_collapsed,
                  2.0e-6, 1.0e-10) &&
          compare("ffn_hidden", got_hidden, expected_hidden,
                  2.0e-6, 1.0e-10) &&
          compare("shared_gate", got_gate, expected_gate,
                  1.0e-5, 2.0e-12) &&
          compare("shared_up", got_up, expected_up, 1.0e-5, 2.0e-12) &&
          compare("shared_mid", got_mid, expected_mid, 2.0e-5, 3.0e-12) &&
          compare("shared_full", got_full, expected_output,
                  2.0e-5, 6.0e-12) &&
          compare("shared_split", got_sum, expected_output,
                  2.0e-5, 6.0e-12) &&
          compare("split_vs_full", got_sum, got_full,
                  3.0e-6, 6.0e-13),
          "independent FFN mHC and shared-expert gates");
    double half0_l2 = 0.0, half1_l2 = 0.0;
    for (uint32_t i = 0; i < kHidden; ++i) {
        half0_l2 += (double)got_half0[i] * got_half0[i];
        half1_l2 += (double)got_half1[i] * got_half1[i];
    }
    CHECK(half0_l2 > 1.0e-12 && half1_l2 > 1.0e-12,
          "both shared-expert rank slices are non-degenerate");
    std::fprintf(stderr,
                 "PASS GLM5 block-3 FFN mHC/shared split control "
                 "split_fnv=%016llx shared_fnv=%016llx\n",
                 (unsigned long long)fnv1a64(
                     got_split.data(), got_split.size() * sizeof(float)),
                 (unsigned long long)fnv1a64(
                     got_sum.data(), got_sum.size() * sizeof(float)));
    return true;
}

}  // namespace

int main() { return run_test() ? 0 : 1; }
