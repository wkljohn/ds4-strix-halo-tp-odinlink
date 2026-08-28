#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "tests/glm5_gguf_test.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(expr, message) do {                                           \
    if (!(expr)) {                                                          \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__);    \
        return false;                                                       \
    }                                                                       \
} while (0)

static float bf16_to_f32(uint16_t bits) {
    uint32_t word = (uint32_t)bits << 16u;
    float value = 0.0f;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

static bool run_test(void) {
    constexpr uint32_t width = 4096u, hc = 4u, token = 42u;
    constexpr uint32_t vocab = 154880u;
    constexpr uint32_t batch = 4u;
    const char *model = std::getenv("DS4_GLM5_MODEL");
    CHECK(model && model[0], "model environment");

    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM5 GGUF");
    uint64_t embedding = 0u;
    CHECK(gguf.tensor("token_embd.weight", {width, vocab}, 30, embedding),
          "bind BF16 token embedding");

    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config) &&
          ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map) &&
          ds4_gpu_set_model_map(gguf.map, gguf.size),
          "initialize gfx1151 and register model map");

    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)width * hc * 4u);
    CHECK(out && ds4_gpu_embed_token_hc_bf16_tensor(
                    out, gguf.map, gguf.size, embedding, vocab, token,
                    width, hc) && ds4_gpu_synchronize(),
          "execute BF16 embedding expansion");

    std::vector<float> got((size_t)width * hc);
    CHECK(ds4_gpu_tensor_read(out, 0u, got.data(), got.size() * sizeof(float)),
          "read BF16 embedding expansion");
    const uint16_t *weights = reinterpret_cast<const uint16_t *>(
        reinterpret_cast<const char *>(gguf.map) + embedding);
    const uint16_t *row = weights + (uint64_t)token * width;
    for (uint32_t h = 0u; h < hc; ++h) {
        for (uint32_t d = 0u; d < width; ++d) {
            const float want = bf16_to_f32(row[d]);
            const float have = got[(uint64_t)h * width + d];
            CHECK(std::isfinite(have) && std::memcmp(&want, &have, 4u) == 0,
                  "exact BF16 conversion and four-stream replication");
        }
    }

    const int32_t token_ids[batch] = {42, 7, 154879, -1};
    ds4_gpu_tensor *tokens = ds4_gpu_tensor_alloc(batch * sizeof(int32_t));
    ds4_gpu_tensor *batch_out = ds4_gpu_tensor_alloc(
        (uint64_t)batch * hc * width * sizeof(float));
    CHECK(tokens && batch_out &&
          ds4_gpu_tensor_write(tokens, 0u, token_ids, sizeof(token_ids)) &&
          ds4_gpu_embed_tokens_hc_bf16_tensor(
              batch_out, tokens, gguf.map, gguf.size, embedding, vocab,
              batch, width, hc) && ds4_gpu_synchronize(),
          "execute batched BF16 embedding expansion");
    std::vector<float> batch_got((size_t)batch * hc * width);
    CHECK(ds4_gpu_tensor_read(batch_out, 0u, batch_got.data(),
                              batch_got.size() * sizeof(float)),
          "read batched BF16 embedding expansion");
    for (uint32_t t = 0u; t < batch; ++t) {
        const uint32_t resolved = token_ids[t] < 0 ? 0u : (uint32_t)token_ids[t];
        const uint16_t *batch_row = weights + (uint64_t)resolved * width;
        for (uint32_t h = 0u; h < hc; ++h) {
            for (uint32_t d = 0u; d < width; ++d) {
                const float want = bf16_to_f32(batch_row[d]);
                const float have = batch_got[((uint64_t)t * hc + h) * width + d];
                CHECK(std::isfinite(have) && std::memcmp(&want, &have, 4u) == 0,
                      "exact batched BF16 conversion and stream replication");
            }
        }
    }

    ds4_gpu_tensor_free(batch_out);
    ds4_gpu_tensor_free(tokens);
    ds4_gpu_tensor_free(out);
    ds4_gpu_cleanup();
    std::fprintf(stderr,
                 "PASS same-GGUF GLM5 BF16 token embedding token=%u\n",
                 token);
    return true;
}

int main(void) { return run_test() ? 0 : 1; }
