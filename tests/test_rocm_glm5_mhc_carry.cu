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
#include <utility>
#include <vector>

#define CHECK(expr, message) do {                                           \
    if (!(expr)) {                                                          \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__);    \
        return false;                                                       \
    }                                                                       \
} while (0)

namespace {

constexpr uint32_t kWidth = 4096u;
constexpr uint32_t kHc = 4u;
constexpr uint32_t kMix = 24u;
constexpr uint32_t kSites = 3u;

struct ErrorStats {
    double max_abs = 0.0;
    long double error_sq = 0.0L;
    long double reference_sq = 0.0L;
    uint64_t nonfinite = 0u;

    double nmse() const {
        return reference_sq == 0.0L ? (double)error_sq
                                    : (double)(error_sq / reference_sq);
    }
};

ErrorStats compare(const std::vector<float> &reference,
                   const std::vector<float> &candidate) {
    ErrorStats stats;
    if (reference.size() != candidate.size()) {
        stats.nonfinite = UINT64_MAX;
        return stats;
    }
    for (size_t i = 0; i < reference.size(); ++i) {
        const double ref = reference[i], got = candidate[i];
        if (!std::isfinite(ref) || !std::isfinite(got)) ++stats.nonfinite;
        const double error = got - ref;
        stats.max_abs = std::max(stats.max_abs, std::fabs(error));
        stats.error_sq += (long double)error * error;
        stats.reference_sq += (long double)ref * ref;
    }
    return stats;
}

bool read_f32(const std::string &path, uint64_t count,
              std::vector<float> &values) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    CHECK(fp, "open mHC carry oracle dump");
    values.resize((size_t)count);
    const size_t got = std::fread(values.data(), sizeof(float),
                                  (size_t)count, fp);
    const int extra = std::fgetc(fp);
    const int close_rc = std::fclose(fp);
    CHECK(got == count && extra == EOF && close_rc == 0,
          "read exact mHC carry oracle dump");
    return true;
}

struct Site {
    uint32_t layer;
    const char *name;
    uint64_t fn = 0u, base = 0u, scale = 0u, norm = 0u;
};

bool bind_site(const Glm5TestGGUF &gguf, Site &site) {
    const std::string prefix = "blk." + std::to_string(site.layer) +
                               ".hc_" + site.name;
    const std::string norm = "blk." + std::to_string(site.layer) +
        (std::string(site.name) == "attn" ? ".attn_norm.weight"
                                          : ".ffn_norm.weight");
    return gguf.tensor(prefix + "_fn.weight", {16384u, 24u}, 30u, site.fn) &&
           gguf.tensor(prefix + "_base.weight", {24u}, 0u, site.base) &&
           gguf.tensor(prefix + "_scale.weight", {3u}, 0u, site.scale) &&
           gguf.tensor(norm, {4096u}, 0u, site.norm);
}

bool run_test() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    const char *prefix = std::getenv("DS4_GLM5_MHC_CARRY_ORACLE_PREFIX");
    CHECK(model && model[0] && prefix && prefix[0],
          "model and mHC carry oracle prefix environment");
    const char *tokens_env = std::getenv("DS4_GLM5_MHC_CARRY_TOKENS");
    const unsigned long parsed_tokens = tokens_env ? std::strtoul(tokens_env, nullptr, 10) : 3ul;
    CHECK(parsed_tokens >= 1ul && parsed_tokens <= UINT32_MAX,
          "valid mHC carry token count");
    const uint32_t tokens = (uint32_t)parsed_tokens;

    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM5 GGUF directory");
    Site sites[kSites] = {{0u, "attn"}, {0u, "ffn"}, {1u, "attn"}};
    for (Site &site : sites)
        CHECK(bind_site(gguf, site), "bind consecutive same-GGUF mHC site");

    const uint64_t hc_values = (uint64_t)tokens * kHc * kWidth;
    const uint64_t row_values = (uint64_t)tokens * kWidth;
    const uint64_t mix_values = (uint64_t)tokens * kMix;
    std::vector<float> initial;
    CHECK(read_f32(std::string(prefix) + ".input.f32", hc_values, initial),
          "load mHC carry initial streams");

    ds4_gpu_config config = {};
    config.n_gpus = 1u;
    config.device_indices[0] = 0u;
    CHECK(ds4_gpu_init_multi(&config) &&
          ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map) &&
          ds4_gpu_set_model_map(gguf.map, gguf.size),
          "initialize gfx1151 and register GLM5 model map");
    ds4_tp_test_reset_exchange_calls();

    ds4_gpu_tensor *residual = ds4_gpu_tensor_alloc(hc_values * sizeof(float));
    ds4_gpu_tensor *next = ds4_gpu_tensor_alloc(hc_values * sizeof(float));
    ds4_gpu_tensor *flat = ds4_gpu_tensor_alloc(hc_values * sizeof(float));
    ds4_gpu_tensor *mix = ds4_gpu_tensor_alloc(mix_values * sizeof(float));
    ds4_gpu_tensor *split = ds4_gpu_tensor_alloc(mix_values * sizeof(float));
    ds4_gpu_tensor *collapsed = ds4_gpu_tensor_alloc(row_values * sizeof(float));
    ds4_gpu_tensor *norm = ds4_gpu_tensor_alloc(row_values * sizeof(float));
    ds4_gpu_tensor *branch = ds4_gpu_tensor_alloc(row_values * sizeof(float));
    CHECK(residual && next && flat && mix && split && collapsed && norm &&
          branch && ds4_gpu_tensor_write(residual, 0u, initial.data(),
                                         hc_values * sizeof(float)),
          "allocate mHC carry tensors and upload initial streams");

    for (uint32_t ordinal = 0; ordinal < kSites; ++ordinal) {
        const Site &site = sites[ordinal];
        const std::string stem = std::string(prefix) + ".site" +
                                 std::to_string(ordinal);
        std::vector<float> post_ref, comb_ref, collapsed_ref;
        std::vector<float> branch_ref, carried_ref;
        CHECK(read_f32(stem + ".post.f32", (uint64_t)tokens * kHc,
                       post_ref) &&
              read_f32(stem + ".comb.f32",
                       (uint64_t)tokens * kHc * kHc, comb_ref) &&
              read_f32(stem + ".collapsed.f32", row_values, collapsed_ref) &&
              read_f32(stem + ".branch.f32", row_values, branch_ref) &&
              read_f32(stem + ".carried.f32", hc_values, carried_ref),
              "load one mHC carry site oracle");
        CHECK(ds4_gpu_tensor_write(branch, 0u, branch_ref.data(),
                                   row_values * sizeof(float)) &&
              ds4_gpu_rms_norm_plain_rows_tensor(
                  flat, residual, kHc * kWidth, tokens, 1.0e-5f) &&
              ds4_gpu_matmul_bf16_tensor(
                  mix, gguf.map, gguf.size, site.fn,
                  kHc * kWidth, kMix, flat, tokens) &&
              ds4_gpu_hc_split_weighted_sum_norm_tensor(
                  collapsed, norm, split, mix, residual,
                  gguf.map, gguf.size, site.scale, site.base, site.norm,
                  kWidth, kHc, 20u, 1.0e-6f, 1.0e-5f) &&
              ds4_gpu_hc_expand_split_tensor(
                  next, branch, residual, split, kWidth, kHc) &&
              ds4_gpu_synchronize(),
              "execute consecutive mHC pre-stage and four-stream carry");

        std::vector<float> split_got((size_t)mix_values);
        std::vector<float> collapsed_got((size_t)row_values);
        std::vector<float> carried_got((size_t)hc_values);
        CHECK(ds4_gpu_tensor_read(split, 0u, split_got.data(),
                                  mix_values * sizeof(float)) &&
              ds4_gpu_tensor_read(collapsed, 0u, collapsed_got.data(),
                                  row_values * sizeof(float)) &&
              ds4_gpu_tensor_read(next, 0u, carried_got.data(),
                                  hc_values * sizeof(float)),
              "read consecutive mHC carry outputs");
        std::vector<float> post_got((size_t)tokens * kHc);
        std::vector<float> comb_got((size_t)tokens * kHc * kHc);
        double row_sum_error = 0.0, column_sum_error = 0.0;
        for (uint32_t token = 0; token < tokens; ++token) {
            const float *row = split_got.data() + (uint64_t)token * kMix;
            std::copy(row + kHc, row + 2u * kHc,
                      post_got.data() + (uint64_t)token * kHc);
            std::copy(row + 2u * kHc, row + kMix,
                      comb_got.data() + (uint64_t)token * kHc * kHc);
            for (uint32_t src = 0; src < kHc; ++src) {
                double sum = 0.0;
                for (uint32_t dst = 0; dst < kHc; ++dst)
                    sum += row[2u * kHc + src * kHc + dst];
                row_sum_error = std::max(row_sum_error, std::fabs(sum - 1.0));
            }
            for (uint32_t dst = 0; dst < kHc; ++dst) {
                double sum = 0.0;
                for (uint32_t src = 0; src < kHc; ++src)
                    sum += row[2u * kHc + src * kHc + dst];
                column_sum_error =
                    std::max(column_sum_error, std::fabs(sum - 1.0));
            }
        }
        const ErrorStats post_error = compare(post_ref, post_got);
        const ErrorStats comb_error = compare(comb_ref, comb_got);
        const ErrorStats collapsed_error = compare(collapsed_ref, collapsed_got);
        const ErrorStats carried_error = compare(carried_ref, carried_got);
        std::fprintf(stderr,
            "GLM5 mHC carry site=%u layer=%u kind=%s "
            "post_abs=%.9g comb_abs=%.9g collapsed_abs=%.9g "
            "collapsed_nmse=%.9g carried_abs=%.9g carried_nmse=%.9g "
            "row_sum_error=%.9g column_sum_error=%.9g\n",
            ordinal, site.layer, site.name, post_error.max_abs,
            comb_error.max_abs, collapsed_error.max_abs,
            collapsed_error.nmse(), carried_error.max_abs,
            carried_error.nmse(), row_sum_error, column_sum_error);
        CHECK(post_error.nonfinite == 0u && comb_error.nonfinite == 0u &&
              collapsed_error.nonfinite == 0u &&
              carried_error.nonfinite == 0u &&
              post_error.max_abs <= 2.0e-6 &&
              comb_error.max_abs <= 2.0e-6 &&
              collapsed_error.max_abs <= 2.0e-6 &&
              collapsed_error.nmse() <= 1.0e-10 &&
              carried_error.max_abs <= 2.0e-6 &&
              carried_error.nmse() <= 1.0e-10 &&
              row_sum_error <= 5.0e-3 && column_sum_error <= 2.0e-6,
              "consecutive mHC carry numerical and stochastic envelope");
        std::swap(residual, next);
    }
    CHECK(ds4_tp_test_get_exchange_calls() == 0u,
          "mHC carry invokes no TP exchange API");

    ds4_gpu_tensor_free(branch);
    ds4_gpu_tensor_free(norm);
    ds4_gpu_tensor_free(collapsed);
    ds4_gpu_tensor_free(split);
    ds4_gpu_tensor_free(mix);
    ds4_gpu_tensor_free(flat);
    ds4_gpu_tensor_free(next);
    ds4_gpu_tensor_free(residual);
    ds4_gpu_cleanup();
    std::fprintf(stderr,
                 "PASS same-GGUF GLM5 %u-token three-site four-stream mHC carry\n",
                 tokens);
    return true;
}

}  // namespace

int main() { return run_test() ? 0 : 1; }
