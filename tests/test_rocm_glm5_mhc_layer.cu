#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
extern "C" {
#include "ds4_tp.h"
}
#include "tests/glm5_gguf_test.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(expr, message) do {                                           \
    if (!(expr)) {                                                          \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__);    \
        return false;                                                       \
    }                                                                       \
} while (0)

struct ErrorStats {
    double max_abs = 0.0;
    long double error_sq = 0.0;
    long double reference_sq = 0.0;
    bool finite = true;
};

static ErrorStats compare(const std::vector<float> &reference,
                          const std::vector<float> &candidate) {
    ErrorStats stats;
    if (reference.size() != candidate.size()) {
        stats.finite = false;
        return stats;
    }
    for (size_t i = 0; i < reference.size(); ++i) {
        const double r = reference[i], c = candidate[i];
        if (!std::isfinite(r) || !std::isfinite(c)) stats.finite = false;
        const double e = c - r;
        stats.max_abs = std::max(stats.max_abs, std::abs(e));
        stats.error_sq += (long double)e * e;
        stats.reference_sq += (long double)r * r;
    }
    return stats;
}

static double nmse(const ErrorStats &stats) {
    return stats.reference_sq == 0.0L
        ? (double)stats.error_sq
        : (double)(stats.error_sq / stats.reference_sq);
}

static bool read_f32(const std::string &path, uint64_t count,
                     std::vector<float> &values) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    CHECK(fp != nullptr, "open mHC oracle dump");
    values.resize((size_t)count);
    const size_t got = std::fread(values.data(), sizeof(float),
                                  (size_t)count, fp);
    const int extra = std::fgetc(fp);
    const int close_rc = std::fclose(fp);
    CHECK(got == count && extra == EOF && close_rc == 0,
          "read exact mHC oracle dump");
    return true;
}

static bool run_test(void) {
    constexpr uint32_t tokens = 3u, width = 4096u, hc = 4u, mix_width = 24u;
    const char *model = std::getenv("DS4_GLM5_MODEL");
    const char *prefix = std::getenv("DS4_GLM5_MHC_ORACLE_PREFIX");
    CHECK(model && model[0] && prefix && prefix[0],
          "model and mHC oracle prefix environment");

    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM5 GGUF directory");
    uint64_t fn = 0, base = 0, scale = 0, norm_weight = 0;
    CHECK(gguf.tensor("blk.0.hc_attn_fn.weight", {16384, 24}, 30, fn) &&
          gguf.tensor("blk.0.hc_attn_base.weight", {24}, 0, base) &&
          gguf.tensor("blk.0.hc_attn_scale.weight", {3}, 0, scale) &&
          gguf.tensor("blk.0.attn_norm.weight", {4096}, 0, norm_weight),
          "bind same-GGUF layer-0 mHC tensors");

    std::vector<float> input, post_ref, comb_ref, collapsed_ref;
    CHECK(read_f32(std::string(prefix) + ".input.f32",
                   (uint64_t)tokens * hc * width, input) &&
          read_f32(std::string(prefix) + ".post.f32",
                   (uint64_t)tokens * hc, post_ref) &&
          read_f32(std::string(prefix) + ".comb.f32",
                   (uint64_t)tokens * hc * hc, comb_ref) &&
          read_f32(std::string(prefix) + ".collapsed.f32",
                   (uint64_t)tokens * width, collapsed_ref),
          "load same-GGUF mHC oracle arrays");

    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config) &&
          ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map) &&
          ds4_gpu_set_model_map(gguf.map, gguf.size),
          "initialize gfx1151 and register model map");
    ds4_tp_test_reset_exchange_calls();

    const uint64_t residual_values = (uint64_t)tokens * hc * width;
    const uint64_t flat_values = residual_values;
    const uint64_t mix_values = (uint64_t)tokens * mix_width;
    const uint64_t row_values = (uint64_t)tokens * width;
    ds4_gpu_tensor *residual = ds4_gpu_tensor_alloc(residual_values * 4u);
    ds4_gpu_tensor *flat = ds4_gpu_tensor_alloc(flat_values * 4u);
    ds4_gpu_tensor *mix = ds4_gpu_tensor_alloc(mix_values * 4u);
    ds4_gpu_tensor *split = ds4_gpu_tensor_alloc(mix_values * 4u);
    ds4_gpu_tensor *collapsed = ds4_gpu_tensor_alloc(row_values * 4u);
    ds4_gpu_tensor *norm = ds4_gpu_tensor_alloc(row_values * 4u);
    CHECK(residual && flat && mix && split && collapsed && norm &&
          ds4_gpu_tensor_write(residual, 0, input.data(),
                               residual_values * 4u),
          "allocate and upload mHC tensors");
    CHECK(ds4_gpu_rms_norm_plain_rows_tensor(flat, residual, hc * width,
                                              tokens, 1.0e-5f) &&
          ds4_gpu_matmul_bf16_tensor(mix, gguf.map, gguf.size, fn,
                                     hc * width, mix_width, flat, tokens) &&
          ds4_gpu_hc_split_weighted_sum_norm_tensor(
              collapsed, norm, split, mix, residual,
              gguf.map, gguf.size, scale, base, norm_weight,
              width, hc, 20u, 1.0e-6f, 1.0e-5f) &&
          ds4_gpu_synchronize(),
          "execute same-GGUF BF16 mHC pre-stage");

    std::vector<float> split_got((size_t)mix_values);
    std::vector<float> collapsed_got((size_t)row_values);
    CHECK(ds4_gpu_tensor_read(split, 0, split_got.data(), mix_values * 4u) &&
          ds4_gpu_tensor_read(collapsed, 0, collapsed_got.data(),
                              row_values * 4u),
          "read mHC component results");
    std::vector<float> post_got((size_t)tokens * hc);
    std::vector<float> comb_got((size_t)tokens * hc * hc);
    for (uint32_t token = 0; token < tokens; ++token) {
        const float *row = split_got.data() + (size_t)token * mix_width;
        std::copy(row + hc, row + 2u * hc,
                  post_got.data() + (size_t)token * hc);
        std::copy(row + 2u * hc, row + mix_width,
                  comb_got.data() + (size_t)token * hc * hc);
    }
    const ErrorStats post_error = compare(post_ref, post_got);
    const ErrorStats comb_error = compare(comb_ref, comb_got);
    const ErrorStats collapsed_error = compare(collapsed_ref, collapsed_got);
    std::fprintf(stderr,
        "GLM5 mHC layer0 post_abs=%.9g post_nmse=%.9g comb_abs=%.9g "
        "comb_nmse=%.9g collapsed_abs=%.9g collapsed_nmse=%.9g\n",
        post_error.max_abs, nmse(post_error), comb_error.max_abs,
        nmse(comb_error), collapsed_error.max_abs, nmse(collapsed_error));
    CHECK(post_error.finite && comb_error.finite && collapsed_error.finite &&
          post_error.max_abs <= 5.0e-4 && nmse(post_error) <= 1.0e-7 &&
          comb_error.max_abs <= 5.0e-4 && nmse(comb_error) <= 1.0e-7 &&
          collapsed_error.max_abs <= 5.0e-5 &&
          nmse(collapsed_error) <= 1.0e-7,
          "same-GGUF mHC numerical envelope");
    CHECK(ds4_tp_test_get_exchange_calls() == 0u,
          "mHC pre-stage invokes no TP exchange API");

    ds4_gpu_tensor_free(norm);
    ds4_gpu_tensor_free(collapsed);
    ds4_gpu_tensor_free(split);
    ds4_gpu_tensor_free(mix);
    ds4_gpu_tensor_free(flat);
    ds4_gpu_tensor_free(residual);
    ds4_gpu_cleanup();
    std::fprintf(stderr, "PASS same-GGUF GLM5 BF16 mHC pre-stage\n");
    return true;
}

int main(void) { return run_test() ? 0 : 1; }
