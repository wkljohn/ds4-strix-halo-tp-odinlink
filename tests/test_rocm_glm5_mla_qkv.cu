#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
extern "C" {
#include "ds4_tp.h"
}
#include "tests/glm5_gguf_test.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define CHECK(expr, message) do {                                           \
    if (!(expr)) {                                                          \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__);    \
        return false;                                                       \
    }                                                                       \
} while (0)

namespace {

constexpr uint32_t kRows = 10u;
constexpr uint32_t kHidden = 4096u;
constexpr uint32_t kQRank = 1536u;
constexpr uint32_t kHeads = 64u;
constexpr uint32_t kHeadDim = 256u;
constexpr uint32_t kKvLora = 512u;

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

bool compare_values(const char *name, const std::vector<float> &got,
                    const std::vector<float> &expected,
                    double max_abs_limit, double nmse_limit) {
    CHECK(got.size() == expected.size(), "MLA QKV comparison shape");
    double maximum = 0.0;
    double error2 = 0.0;
    double reference2 = 0.0;
    double reference_max = 0.0;
    uint64_t mismatches = 0u;
    for (size_t i = 0; i < got.size(); ++i) {
        CHECK(std::isfinite(got[i]) && std::isfinite(expected[i]),
              "finite MLA QKV component output");
        const double error = (double)got[i] - expected[i];
        maximum = std::max(maximum, std::fabs(error));
        error2 += error * error;
        reference2 += (double)expected[i] * expected[i];
        reference_max = std::max(reference_max,
                                 std::fabs((double)expected[i]));
        mismatches += got[i] != expected[i];
    }
    const double nmse = error2 / std::max(reference2, 1.0e-30);
    std::fprintf(stderr,
                 "GLM5 MLA %-10s count=%zu mismatch=%llu max_abs=%.9g "
                 "reference_max=%.9g nmse=%.9g\n", name, got.size(),
                 (unsigned long long)mismatches, maximum, reference_max,
                 nmse);
    CHECK(reference_max >= 1.0e-6, "non-degenerate MLA QKV reference");
    CHECK(maximum <= max_abs_limit && nmse <= nmse_limit,
          "MLA QKV numerical envelope");
    return true;
}

bool compare_absorbed_scores(const std::vector<float> &query,
                             const std::vector<float> &qk_low,
                             const std::vector<float> &kv_norm,
                             const std::vector<float> &k_nope) {
    double maximum = 0.0;
    double error2 = 0.0;
    double reference2 = 0.0;
    for (uint32_t token = 0u; token < kRows; ++token) {
        for (uint32_t head = 0u; head < kHeads; ++head) {
            double absorbed = 0.0;
            double explicit_key = 0.0;
            for (uint32_t j = 0u; j < kKvLora; ++j) {
                absorbed +=
                    (double)qk_low[(uint64_t)head * kKvLora + j] *
                    kv_norm[(uint64_t)token * kKvLora + j];
            }
            for (uint32_t j = 0u; j < kHeadDim; ++j) {
                explicit_key +=
                    (double)query[(uint64_t)head * kHeadDim + j] *
                    k_nope[((uint64_t)token * kHeads + head) * kHeadDim + j];
            }
            const double error = absorbed - explicit_key;
            maximum = std::max(maximum, std::fabs(error));
            error2 += error * error;
            reference2 += explicit_key * explicit_key;
        }
    }
    const double nmse = error2 / std::max(reference2, 1.0e-30);
    std::fprintf(stderr,
                 "GLM5 MLA absorption count=%u max_abs=%.9g nmse=%.9g\n",
                 kRows * kHeads, maximum, nmse);
    CHECK(reference2 >= 1.0e-6, "non-degenerate MLA absorption reference");
    CHECK(maximum <= 5.0e-5 && nmse <= 2.0e-13,
          "MLA absorbed and explicit key scores agree");
    return true;
}

bool run_test() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    const char *oracle_prefix = std::getenv("DS4_GLM5_MLA_QKV_ORACLE_PREFIX");
    CHECK(model && model[0] && oracle_prefix && oracle_prefix[0],
          "model and MLA QKV oracle environment");

    const std::string base(oracle_prefix);
    std::vector<float> hidden, expected_q_a, expected_q_resid, expected_query,
        expected_kv_raw, expected_kv_norm, expected_qk_low;
    CHECK(read_array(base + ".hidden.f32", kRows * kHidden, hidden) &&
          read_array(base + ".q_a.f32", kQRank, expected_q_a) &&
          read_array(base + ".q_resid.f32", kQRank, expected_q_resid) &&
          read_array(base + ".query.f32", kHeads * kHeadDim,
                     expected_query) &&
          read_array(base + ".kv_raw.f32", kRows * kKvLora,
                     expected_kv_raw) &&
          read_array(base + ".kv_norm.f32", kRows * kKvLora,
                     expected_kv_norm) &&
          read_array(base + ".qk_low.f32", kHeads * kKvLora,
                     expected_qk_low),
          "read same-GGUF MLA QKV oracle dumps");

    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM5 GGUF directory");
    uint64_t q_a_offset = 0u, q_norm_offset = 0u, q_b_offset = 0u,
             kv_offset = 0u, kv_norm_offset = 0u, k_b_offset = 0u;
    CHECK(gguf.tensor("blk.3.attn_q_a.weight", {4096u, 1536u}, 8u,
                      q_a_offset) &&
          gguf.tensor("blk.3.attn_q_a_norm.weight", {1536u}, 0u,
                      q_norm_offset) &&
          gguf.tensor("blk.3.attn_q_b.weight", {1536u, 16384u}, 8u,
                      q_b_offset) &&
          gguf.tensor("blk.3.attn_kv_a_mqa.weight", {4096u, 512u}, 8u,
                      kv_offset) &&
          gguf.tensor("blk.3.attn_kv_a_norm.weight", {512u}, 0u,
                      kv_norm_offset) &&
          gguf.tensor("blk.3.attn_k_b.weight", {256u, 512u, 64u}, 8u,
                      k_b_offset),
          "bind real block-3 MLA QKV tensors");

    ds4_gpu_config config = {};
    config.n_gpus = 1u;
    config.device_indices[0] = 0u;
    CHECK(ds4_gpu_init_multi(&config) &&
          ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map) &&
          ds4_gpu_set_model_map(gguf.map, gguf.size),
          "initialize gfx1151 and register model map");
    ds4_tp_test_reset_exchange_calls();

    const auto alloc_f32 = [](uint64_t count) {
        return ds4_gpu_tensor_alloc(count * sizeof(float));
    };
    ds4_gpu_tensor *d_hidden = alloc_f32((uint64_t)kRows * kHidden);
    ds4_gpu_tensor *d_q_a = alloc_f32(kQRank);
    ds4_gpu_tensor *d_q_resid = alloc_f32(kQRank);
    ds4_gpu_tensor *d_query = alloc_f32((uint64_t)kHeads * kHeadDim);
    ds4_gpu_tensor *d_kv_raw = alloc_f32((uint64_t)kRows * kKvLora);
    ds4_gpu_tensor *d_kv_norm = alloc_f32((uint64_t)kRows * kKvLora);
    ds4_gpu_tensor *d_qk_low = alloc_f32((uint64_t)kHeads * kKvLora);
    ds4_gpu_tensor *d_k_nope = alloc_f32(
        (uint64_t)kRows * kHeads * kHeadDim);
    CHECK(d_hidden && d_q_a && d_q_resid && d_query && d_kv_raw &&
          d_kv_norm && d_qk_low && d_k_nope,
          "allocate bounded MLA QKV component tensors");
    ds4_gpu_tensor *d_query_hidden = ds4_gpu_tensor_view(
        d_hidden, (uint64_t)(kRows - 1u) * kHidden * sizeof(float),
        (uint64_t)kHidden * sizeof(float));
    CHECK(d_query_hidden &&
          ds4_gpu_tensor_write(d_hidden, 0u, hidden.data(),
                               (uint64_t)hidden.size() * sizeof(float)),
          "upload MLA QKV hidden rows");

    CHECK(ds4_gpu_matmul_q8_0_tensor(
              d_q_a, gguf.map, gguf.size, q_a_offset,
              kHidden, kQRank, d_query_hidden, 1u) &&
          ds4_gpu_rms_norm_weight_tensor(
              d_q_resid, d_q_a, gguf.map, gguf.size, q_norm_offset,
              kQRank, 1.0e-5f) &&
          ds4_gpu_matmul_q8_0_tensor(
              d_query, gguf.map, gguf.size, q_b_offset,
              kQRank, (uint64_t)kHeads * kHeadDim, d_q_resid, 1u) &&
          ds4_gpu_matmul_q8_0_tensor(
              d_kv_raw, gguf.map, gguf.size, kv_offset,
              kHidden, kKvLora, d_hidden, kRows) &&
          ds4_gpu_glm_kv_lora_rms_norm_tensor(
              d_kv_norm, d_kv_raw, gguf.map, gguf.size, kv_norm_offset,
              kRows, kKvLora, kKvLora, 1.0e-5f) &&
          ds4_gpu_glm_qk_lowrank_typed_tensor(
              d_qk_low, d_query, gguf.map, gguf.size, k_b_offset, 8u,
              kHeads, kKvLora, kHeadDim, kHeadDim) &&
          ds4_gpu_glm_k_b_project_typed_tensor(
              d_k_nope, d_kv_norm, gguf.map, gguf.size, k_b_offset, 8u,
              kRows, kKvLora, kHeadDim, kHeads) &&
          ds4_gpu_synchronize(),
          "execute real block-3 MLA Q/KV trunk");

    std::vector<float> got_q_a, got_q_resid, got_query, got_kv_raw,
        got_kv_norm, got_qk_low, got_k_nope;
    CHECK(read_tensor(d_q_a, kQRank, got_q_a) &&
          read_tensor(d_q_resid, kQRank, got_q_resid) &&
          read_tensor(d_query, kHeads * kHeadDim, got_query) &&
          read_tensor(d_kv_raw, kRows * kKvLora, got_kv_raw) &&
          read_tensor(d_kv_norm, kRows * kKvLora, got_kv_norm) &&
          read_tensor(d_qk_low, kHeads * kKvLora, got_qk_low) &&
          read_tensor(d_k_nope, (uint64_t)kRows * kHeads * kHeadDim,
                      got_k_nope),
          "read MLA QKV component outputs");

    /* Both independent gfx1151 nodes produced identical observations.  Keep
       a little over 2x headroom above those maxima without turning this into
       an architecture-independent bit-identity gate. */
    CHECK(compare_values("q_a", got_q_a, expected_q_a, 1.0e-6, 2.0e-12) &&
          compare_values("q_resid", got_q_resid, expected_q_resid,
                         5.0e-6, 2.0e-12) &&
          compare_values("query", got_query, expected_query,
                         1.0e-5, 2.0e-12) &&
          compare_values("kv_raw", got_kv_raw, expected_kv_raw,
                         2.0e-6, 2.0e-12) &&
          compare_values("kv_norm", got_kv_norm, expected_kv_norm,
                         4.0e-6, 2.0e-12) &&
          compare_values("qk_low", got_qk_low, expected_qk_low,
                         8.0e-6, 2.0e-12),
          "MLA QKV component numerical gates");
    CHECK(compare_absorbed_scores(got_query, got_qk_low, got_kv_norm,
                                  got_k_nope),
          "MLA independent key-absorption identity");
    CHECK(ds4_tp_test_get_exchange_calls() == 0u,
          "rank-local MLA QKV invokes no TP exchange");

    ds4_gpu_tensor_free(d_query_hidden);
    ds4_gpu_tensor_free(d_k_nope);
    ds4_gpu_tensor_free(d_qk_low);
    ds4_gpu_tensor_free(d_kv_norm);
    ds4_gpu_tensor_free(d_kv_raw);
    ds4_gpu_tensor_free(d_query);
    ds4_gpu_tensor_free(d_q_resid);
    ds4_gpu_tensor_free(d_q_a);
    ds4_gpu_tensor_free(d_hidden);
    ds4_gpu_cleanup();
    std::fprintf(stderr,
                 "PASS same-GGUF GLM5 block-3 MLA Q/KV trunk gate\n");
    return true;
}

}  // namespace

int main() { return run_test() ? 0 : 1; }
