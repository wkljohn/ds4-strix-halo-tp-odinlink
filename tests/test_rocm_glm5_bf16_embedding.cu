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

static float f16_to_f32(uint16_t value) {
    const uint32_t sign = (uint32_t)(value & 0x8000u) << 16u;
    uint32_t exponent = (value >> 10u) & 0x1fu;
    uint32_t fraction = value & 0x03ffu;
    uint32_t bits = 0u;
    if (exponent == 0u) {
        if (fraction == 0u) {
            bits = sign;
        } else {
            int shift = 0;
            while ((fraction & 0x0400u) == 0u) {
                fraction <<= 1u;
                ++shift;
            }
            fraction &= 0x03ffu;
            bits = sign | (uint32_t)(113 - shift) << 23u |
                   fraction << 13u;
        }
    } else if (exponent == 31u) {
        bits = sign | 0x7f800000u | fraction << 13u;
    } else {
        bits = sign | (exponent + 112u) << 23u | fraction << 13u;
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static bool run_test(void) {
    constexpr uint32_t width = 4096u, hc = 4u, token = 42u;
    constexpr uint32_t vocab = 154880u;
    constexpr uint32_t batch = 4u;
    const char *model = std::getenv("DS4_GLM5_MODEL");
    CHECK(model && model[0], "model environment");

    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM5 GGUF");
    const auto found = gguf.tensors.find("token_embd.weight");
    CHECK(found != gguf.tensors.end() &&
          found->second.dims == std::vector<uint64_t>({width, vocab}) &&
          (found->second.type == 8u || found->second.type == 30u) &&
          found->second.relative_offset <= UINT64_MAX - gguf.data_start,
          "bind supported GLM5 token embedding");
    const uint32_t embedding_type = found->second.type;
    const uint64_t embedding = gguf.data_start + found->second.relative_offset;

    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config) &&
          ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map) &&
          ds4_gpu_set_model_map(gguf.map, gguf.size),
          "initialize gfx1151 and register model map");

    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)width * hc * 4u);
    const bool one_ok = embedding_type == 30u ?
        ds4_gpu_embed_token_hc_bf16_tensor(
            out, gguf.map, gguf.size, embedding, vocab, token, width, hc) :
        ds4_gpu_embed_token_hc_q8_0_tensor(
            out, gguf.map, gguf.size, embedding, vocab, token, width, hc);
    CHECK(out && one_ok && ds4_gpu_synchronize(),
          "execute typed embedding expansion");

    std::vector<float> got((size_t)width * hc);
    CHECK(ds4_gpu_tensor_read(out, 0u, got.data(), got.size() * sizeof(float)),
          "read BF16 embedding expansion");
    const uint8_t *weights = reinterpret_cast<const uint8_t *>(gguf.map) + embedding;
    const uint64_t q8_row_bytes = (width / 32u) * 34u;
    for (uint32_t h = 0u; h < hc; ++h) {
        for (uint32_t d = 0u; d < width; ++d) {
            float want = 0.0f;
            if (embedding_type == 30u) {
                const uint16_t *row = reinterpret_cast<const uint16_t *>(weights) +
                                      (uint64_t)token * width;
                want = bf16_to_f32(row[d]);
            } else {
                const uint8_t *block = weights + (uint64_t)token * q8_row_bytes +
                                       (uint64_t)(d / 32u) * 34u;
                uint16_t scale_bits = 0u;
                std::memcpy(&scale_bits, block, sizeof(scale_bits));
                const int8_t q = reinterpret_cast<const int8_t *>(block + 2u)[d % 32u];
                want = f16_to_f32(scale_bits) * (float)q;
            }
            const float have = got[(uint64_t)h * width + d];
            CHECK(std::isfinite(have) && std::memcmp(&want, &have, 4u) == 0,
                  "exact typed conversion and four-stream replication");
        }
    }

    const int32_t token_ids[batch] = {42, 7, 154879, -1};
    ds4_gpu_tensor *tokens = ds4_gpu_tensor_alloc(batch * sizeof(int32_t));
    ds4_gpu_tensor *batch_out = ds4_gpu_tensor_alloc(
        (uint64_t)batch * hc * width * sizeof(float));
    CHECK(tokens && batch_out &&
          ds4_gpu_tensor_write(tokens, 0u, token_ids, sizeof(token_ids)),
          "upload batched token ids");
    const bool batch_ok = embedding_type == 30u ?
        ds4_gpu_embed_tokens_hc_bf16_tensor(
            batch_out, tokens, gguf.map, gguf.size, embedding, vocab,
            batch, width, hc) :
        ds4_gpu_embed_tokens_hc_q8_0_tensor(
            batch_out, tokens, gguf.map, gguf.size, embedding, vocab,
            batch, width, hc);
    CHECK(batch_ok && ds4_gpu_synchronize(),
          "execute batched typed embedding expansion");
    std::vector<float> batch_got((size_t)batch * hc * width);
    CHECK(ds4_gpu_tensor_read(batch_out, 0u, batch_got.data(),
                              batch_got.size() * sizeof(float)),
          "read batched BF16 embedding expansion");
    for (uint32_t t = 0u; t < batch; ++t) {
        const uint32_t resolved = token_ids[t] < 0 ? 0u : (uint32_t)token_ids[t];
        for (uint32_t h = 0u; h < hc; ++h) {
            for (uint32_t d = 0u; d < width; ++d) {
                float want = 0.0f;
                if (embedding_type == 30u) {
                    const uint16_t *batch_row =
                        reinterpret_cast<const uint16_t *>(weights) +
                        (uint64_t)resolved * width;
                    want = bf16_to_f32(batch_row[d]);
                } else {
                    const uint8_t *block = weights +
                        (uint64_t)resolved * q8_row_bytes +
                        (uint64_t)(d / 32u) * 34u;
                    uint16_t scale_bits = 0u;
                    std::memcpy(&scale_bits, block, sizeof(scale_bits));
                    const int8_t q =
                        reinterpret_cast<const int8_t *>(block + 2u)[d % 32u];
                    want = f16_to_f32(scale_bits) * (float)q;
                }
                const float have = batch_got[((uint64_t)t * hc + h) * width + d];
                CHECK(std::isfinite(have) && std::memcmp(&want, &have, 4u) == 0,
                      "exact batched typed conversion and stream replication");
            }
        }
    }

    ds4_gpu_tensor_free(batch_out);
    ds4_gpu_tensor_free(tokens);
    ds4_gpu_tensor_free(out);
    ds4_gpu_cleanup();
    std::fprintf(stderr,
                 "PASS same-GGUF GLM5 token embedding type=%u token=%u\n",
                 embedding_type, token);
    return true;
}

int main(void) { return run_test() ? 0 : 1; }
