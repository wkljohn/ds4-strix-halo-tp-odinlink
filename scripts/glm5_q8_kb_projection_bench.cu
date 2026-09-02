#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "ds4_glm5_next_runtime.h"
#include "tests/glm5_gguf_test.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(expr, message) do {                                         \
    if (!(expr)) {                                                        \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__); \
        return 1;                                                         \
    }                                                                     \
} while (0)

namespace {

constexpr uint32_t kHeads = 64u;
constexpr uint32_t kInDim = 256u;
constexpr uint32_t kOutDim = 512u;
constexpr uint32_t kSamples = 5u;
constexpr uint32_t kRepeats = 50u;

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2u];
}

uint64_t fnv64(const std::vector<float> &values) {
    uint64_t hash = UINT64_C(1469598103934665603);
    const auto *bytes = reinterpret_cast<const unsigned char *>(values.data());
    for (size_t i = 0; i < values.size() * sizeof(float); ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

}  // namespace

int main() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    CHECK(model && model[0], "DS4_GLM5_MODEL is required");

    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM5 GGUF directory");
    std::vector<uint64_t> offsets;
    std::vector<uint32_t> layers;
    for (uint32_t layer = 0u; layer < DS4_GLM5_NEXT_TRUNK_COUNT; ++layer) {
        uint64_t offset = 0u;
        const std::string name =
            "blk." + std::to_string(layer) + ".attn_k_b.weight";
        if (gguf.tensor(name, {kInDim, kOutDim, kHeads}, 8u, offset)) {
            layers.push_back(layer);
            offsets.push_back(offset);
        }
    }
    std::fprintf(stderr, "GLM5 real k_b layers=%zu", offsets.size());
    for (uint32_t layer : layers) std::fprintf(stderr, " %u", layer);
    std::fprintf(stderr, "\n");
    CHECK(offsets.size() == 11u, "bind all eleven real GLM5 k_b tensors");

    ds4_gpu_config config = {};
    config.n_gpus = 1u;
    config.device_indices[0] = 0u;
    CHECK(ds4_gpu_init_multi(&config) &&
          ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map) &&
          ds4_gpu_set_model_map(gguf.map, gguf.size),
          "initialize gfx1151 and register model map");

    ds4_gpu_tensor *query = ds4_gpu_tensor_alloc(
        (uint64_t)kHeads * kInDim * sizeof(float));
    ds4_gpu_tensor *output = ds4_gpu_tensor_alloc(
        (uint64_t)kHeads * kOutDim * sizeof(float));
    CHECK(query && output, "allocate bounded k_b benchmark tensors");
    std::vector<float> input((uint64_t)kHeads * kInDim);
    for (size_t i = 0; i < input.size(); ++i) {
        const int value = (int)((i * 73u + (i >> 4u) * 29u) % 1021u) - 510;
        input[i] = (float)value / 1024.0f;
    }
    CHECK(ds4_gpu_tensor_write(query, 0u, input.data(),
                               input.size() * sizeof(float)),
          "upload deterministic k_b input");

    std::vector<std::vector<float>> references(offsets.size());
    std::vector<float> candidate((uint64_t)kHeads * kOutDim);
    unsetenv("DS4_ROCM_GLM5_QK_LOW_LDS_EXACT");
    for (size_t i = 0; i < offsets.size(); ++i) {
        CHECK(ds4_gpu_glm_qk_lowrank_typed_tensor(
                  output, query, gguf.map, gguf.size, offsets[i], 8u,
                  kHeads, kOutDim, kInDim, kInDim) &&
              ds4_gpu_synchronize(),
              "run incumbent real-tensor k_b projection");
        references[i].resize(candidate.size());
        CHECK(ds4_gpu_tensor_read(output, 0u, references[i].data(),
                                  references[i].size() * sizeof(float)),
              "read incumbent real-tensor k_b output");
    }
    CHECK(setenv("DS4_ROCM_GLM5_QK_LOW_LDS_EXACT", "1", 1) == 0,
          "enable exact LDS k_b candidate");
    for (size_t i = 0; i < offsets.size(); ++i) {
        CHECK(ds4_gpu_glm_qk_lowrank_typed_tensor(
                  output, query, gguf.map, gguf.size, offsets[i], 8u,
                  kHeads, kOutDim, kInDim, kInDim) &&
              ds4_gpu_synchronize() &&
              ds4_gpu_tensor_read(output, 0u, candidate.data(),
                                  candidate.size() * sizeof(float)),
              "run exact LDS real-tensor k_b projection");
        if (std::memcmp(references[i].data(), candidate.data(),
                        candidate.size() * sizeof(float)) != 0) {
            std::fprintf(stderr,
                         "FAIL layer %u exact output mismatch "
                         "reference=%016llx candidate=%016llx\n",
                         layers[i],
                         (unsigned long long)fnv64(references[i]),
                         (unsigned long long)fnv64(candidate));
            return 1;
        }
    }

    const auto run_stream = [&](bool exact, uint32_t repeats) -> double {
        if (exact) {
            setenv("DS4_ROCM_GLM5_QK_LOW_LDS_EXACT", "1", 1);
        } else {
            unsetenv("DS4_ROCM_GLM5_QK_LOW_LDS_EXACT");
        }
        if (!ds4_gpu_synchronize()) {
            std::fprintf(stderr, "FAIL synchronize before k_b timing\n");
            std::exit(1);
        }
        const auto begin = std::chrono::steady_clock::now();
        for (uint32_t repeat = 0u; repeat < repeats; ++repeat) {
            for (uint64_t offset : offsets) {
                if (!ds4_gpu_glm_qk_lowrank_typed_tensor(
                        output, query, gguf.map, gguf.size, offset, 8u,
                        kHeads, kOutDim, kInDim, kInDim)) {
                    std::fprintf(stderr,
                                 "FAIL launch timed real-tensor k_b "
                                 "projection\n");
                    std::exit(1);
                }
            }
        }
        if (!ds4_gpu_synchronize()) {
            std::fprintf(stderr, "FAIL synchronize after k_b timing\n");
            std::exit(1);
        }
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(end - begin).count() /
               repeats;
    };

    (void)run_stream(false, 3u);
    (void)run_stream(true, 3u);
    std::vector<double> incumbent_ms, candidate_ms;
    for (uint32_t sample = 0u; sample < kSamples; ++sample) {
        incumbent_ms.push_back(run_stream(false, kRepeats));
        candidate_ms.push_back(run_stream(true, kRepeats));
    }
    const double incumbent = median(incumbent_ms);
    const double exact = median(candidate_ms);
    std::printf("layers,%zu\n", offsets.size());
    std::printf("stream_bytes,%llu\n",
                (unsigned long long)offsets.size() * kHeads * kOutDim *
                    (kInDim / 32u) * 34u);
    std::printf("incumbent_ms_per_11_layer_stream,%.6f\n", incumbent);
    std::printf("lds_exact_ms_per_11_layer_stream,%.6f\n", exact);
    std::printf("speedup,%.6f\n", incumbent / exact);
    std::printf("saved_ms_per_token,%.6f\n", incumbent - exact);
    std::printf("exact_all_layers,1\n");

    unsetenv("DS4_ROCM_GLM5_QK_LOW_LDS_EXACT");
    ds4_gpu_tensor_free(output);
    ds4_gpu_tensor_free(query);
    ds4_gpu_cleanup();
    return 0;
}
