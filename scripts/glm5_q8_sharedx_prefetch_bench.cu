#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "tests/glm5_gguf_test.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(expr, message) do {                                      \
    if (!(expr)) {                                                     \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__); \
        return 1;                                                      \
    }                                                                 \
} while (0)

namespace {

struct Target {
    std::string name;
    uint64_t offset;
    uint64_t full_in;
    uint64_t full_out;
    uint64_t in_start;
    uint64_t in_count;
    uint64_t out_start;
    uint64_t out_count;
};

struct Variant {
    uint32_t prefetch;
    int nontemporal;
    uint32_t rows;
};

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

bool add_tensor(const Glm5TestGGUF &gguf, std::vector<Target> &targets,
                const std::string &name, const std::vector<uint64_t> &dims,
                uint64_t in_start, uint64_t in_count,
                uint64_t out_start, uint64_t out_count) {
    uint64_t offset = 0u;
    if (!gguf.tensor(name, dims, 8u, offset)) return false;
    targets.push_back({name, offset, dims[0], dims[1], in_start, in_count,
                       out_start, out_count});
    return true;
}

}  // namespace

int main() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    CHECK(model && model[0], "DS4_GLM5_MODEL is required");
    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM5 GGUF directory");

    std::vector<Target> targets;
    for (uint32_t layer = 0u; layer < 45u; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        (void)add_tensor(gguf, targets, prefix + "attn_q_a.weight",
                         {4096u, 1536u}, 0u, 4096u, 0u, 1536u);
        (void)add_tensor(gguf, targets, prefix + "attn_q_b.weight",
                         {1536u, 16384u}, 0u, 1536u, 0u, 16384u);
        (void)add_tensor(gguf, targets, prefix + "attn_kv_a_mqa.weight",
                         {4096u, 512u}, 0u, 4096u, 0u, 512u);
        (void)add_tensor(gguf, targets, prefix + "attn_output.weight",
                         {16384u, 4096u}, 0u, 8192u, 0u, 4096u);
        (void)add_tensor(gguf, targets, prefix + "ffn_gate_shexp.weight",
                         {4096u, 2048u}, 0u, 4096u, 0u, 1024u);
        (void)add_tensor(gguf, targets, prefix + "ffn_up_shexp.weight",
                         {4096u, 2048u}, 0u, 4096u, 0u, 1024u);
        (void)add_tensor(gguf, targets, prefix + "ffn_down_shexp.weight",
                         {2048u, 4096u}, 0u, 1024u, 0u, 4096u);
        (void)add_tensor(gguf, targets, prefix + "ffn_down.weight",
                         {12288u, 4096u}, 0u, 12288u, 0u, 4096u);
    }
    CHECK(targets.size() == 173u,
          "bind the 173 Q8 shared-X decode projections from the profile");

    ds4_gpu_config config = {};
    config.n_gpus = 1u;
    config.device_indices[0] = 0u;
    CHECK(ds4_gpu_init_multi(&config) &&
          ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map) &&
          ds4_gpu_set_model_map(gguf.map, gguf.size),
          "initialize gfx1151 and register model map");

    ds4_gpu_tensor *input = ds4_gpu_tensor_alloc(16384u * sizeof(float));
    ds4_gpu_tensor *output = ds4_gpu_tensor_alloc(16384u * sizeof(float));
    CHECK(input && output, "allocate bounded projection buffers");
    std::vector<float> host_input(16384u);
    for (size_t i = 0; i < host_input.size(); ++i) {
        const int value = (int)((i * 73u + (i >> 4u) * 29u) % 1021u) - 510;
        host_input[i] = (float)value / 1024.0f;
    }
    CHECK(ds4_gpu_tensor_write(input, 0u, host_input.data(),
                               host_input.size() * sizeof(float)),
          "upload deterministic input");

    const auto launch = [&](const Target &target, const Variant &variant) {
        return ds4_gpu_rocm_q8_sharedx_prefetch_tensor(
            output, gguf.map, gguf.size, target.offset, target.full_in,
            target.full_out, target.in_start, target.in_count,
            target.out_start, target.out_count, input, variant.prefetch,
            variant.nontemporal, variant.rows) != 0;
    };

    const Variant incumbent = {0u, 0, 32u};
    std::vector<std::vector<float>> references(targets.size());
    std::vector<uint64_t> reference_hashes(targets.size());
    for (size_t i = 0; i < targets.size(); ++i) {
        CHECK(launch(targets[i], incumbent) && ds4_gpu_synchronize(),
              "launch incumbent real Q8 tensor");
        references[i].resize(targets[i].out_count);
        CHECK(ds4_gpu_tensor_read(output, 0u, references[i].data(),
                                  references[i].size() * sizeof(float)),
              "read incumbent Q8 output");
        reference_hashes[i] = fnv64(references[i]);
    }

    std::vector<Variant> variants;
    variants.push_back(incumbent);
    for (uint32_t prefetch : {4u, 8u, 16u, 32u})
        for (int nt : {0, 1})
            for (uint32_t rows : {8u, 16u, 32u})
                variants.push_back({prefetch, nt, rows});

    for (size_t v = 1u; v < variants.size(); ++v) {
        for (size_t i = 0; i < targets.size(); ++i) {
            CHECK(launch(targets[i], variants[v]) && ds4_gpu_synchronize(),
                  "launch candidate real Q8 tensor");
            std::vector<float> candidate(targets[i].out_count);
            CHECK(ds4_gpu_tensor_read(output, 0u, candidate.data(),
                                      candidate.size() * sizeof(float)),
                  "read candidate Q8 output");
            if (std::memcmp(references[i].data(), candidate.data(),
                            candidate.size() * sizeof(float)) != 0) {
                std::fprintf(stderr,
                    "FAIL exact mismatch tensor=%s reference=%016llx candidate=%016llx\n",
                    targets[i].name.c_str(),
                    (unsigned long long)reference_hashes[i],
                    (unsigned long long)fnv64(candidate));
                return 1;
            }
        }
    }

    constexpr uint32_t samples = 5u;
    constexpr uint32_t streams_per_sample = 20u;
    std::vector<std::vector<double>> timings(variants.size());
    for (uint32_t sample = 0u; sample < samples; ++sample) {
        for (size_t step = 0u; step < variants.size(); ++step) {
            const size_t v = (step + sample * 7u) % variants.size();
            CHECK(ds4_gpu_synchronize(), "synchronize before timing");
            const auto begin = std::chrono::steady_clock::now();
            for (uint32_t repeat = 0u; repeat < streams_per_sample; ++repeat)
                for (const Target &target : targets)
                    CHECK(launch(target, variants[v]),
                          "launch timed Q8 projection stream");
            CHECK(ds4_gpu_synchronize(), "synchronize after timing");
            const auto end = std::chrono::steady_clock::now();
            timings[v].push_back(
                std::chrono::duration<double, std::milli>(end - begin).count() /
                streams_per_sample);
        }
    }

    const double incumbent_ms = median(timings[0]);
    uint64_t stream_bytes = 0u;
    for (const Target &target : targets)
        stream_bytes += target.out_count * (target.in_count / 32u) * 34u;
    std::printf("targets,%zu\nstream_bytes,%llu\n",
                targets.size(), (unsigned long long)stream_bytes);
    std::printf("prefetch,nontemporal,rows,median_ms,speedup,saved_ms,exact\n");
    for (size_t v = 0u; v < variants.size(); ++v) {
        const double value = median(timings[v]);
        std::printf("%u,%d,%u,%.6f,%.6f,%.6f,1\n",
                    variants[v].prefetch, variants[v].nontemporal,
                    variants[v].rows, value, incumbent_ms / value,
                    incumbent_ms - value);
    }

    ds4_gpu_tensor_free(output);
    ds4_gpu_tensor_free(input);
    ds4_gpu_cleanup();
    return 0;
}
