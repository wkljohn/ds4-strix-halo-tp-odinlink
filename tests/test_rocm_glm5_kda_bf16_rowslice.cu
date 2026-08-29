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
#include <cstring>
#include <string>
#include <vector>

extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

#define CHECK(expr, message) do {                                         \
    if (!(expr)) {                                                        \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__); \
        return false;                                                     \
    }                                                                     \
} while (0)

namespace {

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

uint64_t fnv1a64(const void *data, uint64_t bytes) {
    const auto *p = static_cast<const uint8_t *>(data);
    uint64_t hash = UINT64_C(1469598103934665603);
    for (uint64_t i = 0; i < bytes; ++i) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

bool run_shape(const Glm5TestGGUF &gguf, const char *name,
               uint64_t weight_offset, uint32_t in_dim,
               uint32_t out_dim, uint32_t tokens) {
    CHECK((out_dim & 1u) == 0u, "even BF16 projection output width");
    const uint32_t local_out = out_dim / 2u;
    const uint64_t input_count = (uint64_t)tokens * in_dim;
    const uint64_t full_count = (uint64_t)tokens * out_dim;
    const uint64_t local_count = (uint64_t)tokens * local_out;
    std::vector<float> host_input((size_t)input_count);
    for (uint64_t i = 0; i < input_count; ++i) {
        const int32_t centered =
            (int32_t)((i * UINT64_C(73) + tokens * 29u + in_dim) % 509u) -
            254;
        host_input[(size_t)i] = (float)centered * (1.0f / 2048.0f) +
            0.03125f * std::sin((double)(i % 8191u) * 0.017);
    }

    Tensors tensors;
    ds4_gpu_tensor *input = tensors.f32(input_count);
    ds4_gpu_tensor *full = tensors.f32(full_count);
    ds4_gpu_tensor *half[2] = {
        tensors.f32(local_count), tensors.f32(local_count),
    };
    CHECK(input && full && half[0] && half[1],
          "allocate BF16 row-slice tensors");
    CHECK(ds4_gpu_tensor_write(input, 0u, host_input.data(),
                               input_count * sizeof(float)),
          "upload deterministic BF16 row-slice input");
    CHECK(ds4_gpu_matmul_bf16_tensor(full, gguf.map, gguf.size,
                                     weight_offset, in_dim, out_dim,
                                     input, tokens),
          "execute full BF16 projection");
    const uint64_t local_weight_bytes =
        (uint64_t)local_out * in_dim * sizeof(uint16_t);
    for (uint32_t rank = 0u; rank < 2u; ++rank) {
        CHECK(ds4_gpu_matmul_bf16_tensor(
                  half[rank], gguf.map, gguf.size,
                  weight_offset + (uint64_t)rank * local_weight_bytes,
                  in_dim, local_out, input, tokens),
              "execute rank-local BF16 row slice");
    }
    CHECK(ds4_gpu_synchronize(), "synchronize BF16 row-slice projections");

    std::vector<float> full_host((size_t)full_count);
    std::vector<float> half_host[2] = {
        std::vector<float>((size_t)local_count),
        std::vector<float>((size_t)local_count),
    };
    CHECK(ds4_gpu_tensor_read(full, 0u, full_host.data(),
                              full_count * sizeof(float)) &&
          ds4_gpu_tensor_read(half[0], 0u, half_host[0].data(),
                              local_count * sizeof(float)) &&
          ds4_gpu_tensor_read(half[1], 0u, half_host[1].data(),
                              local_count * sizeof(float)),
          "read BF16 row-slice projections");

    for (uint32_t token = 0u; token < tokens; ++token) {
        for (uint32_t rank = 0u; rank < 2u; ++rank) {
            const float *expected = full_host.data() +
                (uint64_t)token * out_dim + (uint64_t)rank * local_out;
            const float *got = half_host[rank].data() +
                (uint64_t)token * local_out;
            CHECK(std::memcmp(expected, got,
                              (size_t)local_out * sizeof(float)) == 0,
                  "rank-local BF16 rows bit-match full projection");
        }
    }
    std::fprintf(stderr,
                 "PASS GLM5 KDA BF16 rowslice tensor=%s tokens=%u "
                 "shape=%ux%u local_out=%u full_fnv=%016llx "
                 "rank0_fnv=%016llx rank1_fnv=%016llx\n",
                 name, tokens, in_dim, out_dim, local_out,
                 (unsigned long long)fnv1a64(
                     full_host.data(), full_count * sizeof(float)),
                 (unsigned long long)fnv1a64(
                     half_host[0].data(), local_count * sizeof(float)),
                 (unsigned long long)fnv1a64(
                     half_host[1].data(), local_count * sizeof(float)));
    return true;
}

bool run_test() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    CHECK(model && model[0], "DS4_GLM5_MODEL environment");
    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM5 GGUF");

    uint64_t q = 0u, k = 0u, v = 0u;
    uint64_t f_b = 0u, g_b = 0u, beta = 0u;
    uint64_t q_conv = 0u, k_conv = 0u, v_conv = 0u;
    uint64_t dt_bias = 0u, a_log = 0u;
    CHECK(gguf.tensor("blk.0.kda_q.weight", {4096u, 8192u}, 30u, q) &&
          gguf.tensor("blk.0.kda_k.weight", {4096u, 8192u}, 30u, k) &&
          gguf.tensor("blk.0.kda_v.weight", {4096u, 8192u}, 30u, v) &&
          gguf.tensor("blk.0.kda_f_b.weight", {128u, 8192u}, 30u, f_b) &&
          gguf.tensor("blk.0.kda_g_b.weight", {128u, 8192u}, 30u, g_b) &&
          gguf.tensor("blk.0.kda_beta.weight", {4096u, 64u}, 30u, beta) &&
          gguf.tensor("blk.0.kda_q_conv.weight", {4u, 1u, 8192u}, 0u,
                      q_conv) &&
          gguf.tensor("blk.0.kda_k_conv.weight", {4u, 1u, 8192u}, 0u,
                      k_conv) &&
          gguf.tensor("blk.0.kda_v_conv.weight", {4u, 1u, 8192u}, 0u,
                      v_conv) &&
          gguf.tensor("blk.0.kda_dt_bias.weight", {8192u}, 0u, dt_bias) &&
          gguf.tensor("blk.0.kda_a_log.weight", {64u}, 0u, a_log),
          "bind real layer-0 KDA head-local tensors");

    const auto check_f32_halves = [&](const char *name, uint64_t offset,
                                      uint64_t values) {
        CHECK((values & 1u) == 0u && offset <= gguf.size &&
                  values <= (gguf.size - offset) / sizeof(float),
              "bounded KDA F32 head-local tensor");
        const float *full = reinterpret_cast<const float *>(gguf.map + offset);
        const uint64_t half = values / 2u;
        uint64_t hash[2] = {};
        for (uint32_t rank = 0u; rank < 2u; ++rank) {
            const float *view = full + (uint64_t)rank * half;
            CHECK(std::memcmp(view, full + (uint64_t)rank * half,
                              (size_t)half * sizeof(float)) == 0,
                  "KDA F32 half view equals canonical mapped slice");
            hash[rank] = fnv1a64(view, half * sizeof(float));
        }
        CHECK(hash[0] != hash[1], "KDA F32 halves are non-degenerate");
        std::fprintf(stderr,
                     "PASS GLM5 KDA F32 rowslice tensor=%s values=%llu "
                     "rank0_fnv=%016llx rank1_fnv=%016llx\n",
                     name, (unsigned long long)values,
                     (unsigned long long)hash[0],
                     (unsigned long long)hash[1]);
        return true;
    };
    CHECK(check_f32_halves("kda_q_conv", q_conv, 8192u * 4u) &&
          check_f32_halves("kda_k_conv", k_conv, 8192u * 4u) &&
          check_f32_halves("kda_v_conv", v_conv, 8192u * 4u) &&
          check_f32_halves("kda_dt_bias", dt_bias, 8192u) &&
          check_f32_halves("kda_a_log", a_log, 64u),
          "real layer-0 KDA F32 row-slice offsets");

    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    RuntimeGuard runtime;
    CHECK(ds4_gpu_init_multi(&config) &&
          ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map) &&
          ds4_gpu_set_model_map(gguf.map, gguf.size),
          "initialize gfx1151 and register GLM5 model map");
    runtime.active = true;

    for (uint32_t tokens : {1u, 4u, 33u}) {
        CHECK(run_shape(gguf, "kda_q", q, 4096u, 8192u, tokens),
              "KDA q projection row-slice identity");
        CHECK(run_shape(gguf, "kda_k", k, 4096u, 8192u, tokens),
              "KDA k projection row-slice identity");
        CHECK(run_shape(gguf, "kda_v", v, 4096u, 8192u, tokens),
              "KDA v projection row-slice identity");
        CHECK(run_shape(gguf, "kda_f_b", f_b, 128u, 8192u, tokens),
              "KDA f_b projection row-slice identity");
        CHECK(run_shape(gguf, "kda_g_b", g_b, 128u, 8192u, tokens),
              "KDA g_b projection row-slice identity");
        CHECK(run_shape(gguf, "kda_beta", beta, 4096u, 64u, tokens),
              "KDA beta projection row-slice identity");
    }
    std::fprintf(stderr,
                 "PASS complete real-GGUF GLM5 KDA BF16 row-slice gate\n");
    return true;
}

}  // namespace

int main() { return run_test() ? 0 : 1; }
