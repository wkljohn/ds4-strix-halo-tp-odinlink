#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
extern "C" {
#include "ds4_tp.h"
}
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

extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

#define CHECK(expr, message) do {                                      \
    if (!(expr)) {                                                     \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__); \
        return false;                                                  \
    }                                                                 \
} while (0)

namespace {

constexpr uint32_t kFull = 8192u;
constexpr uint32_t kHalf = 4096u;
constexpr uint32_t kOut = 4096u;

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

struct Stats {
    double max_abs = 0.0;
    long double abs_sum = 0.0L;
    long double error_sq = 0.0L;
    long double reference_sq = 0.0L;
    long double candidate_sq = 0.0L;
    long double dot = 0.0L;
    long double signed_sum = 0.0L;
    uint64_t positive = 0u;
    uint64_t count = 0u;
    bool finite = true;
};

template <typename Reference>
Stats compare(const std::vector<Reference> &reference,
              const std::vector<float> &candidate) {
    Stats stats;
    if (reference.size() != candidate.size()) {
        stats.finite = false;
        return stats;
    }
    stats.count = reference.size();
    for (size_t i = 0u; i < reference.size(); ++i) {
        const long double r = (long double)reference[i];
        const long double c = (long double)candidate[i];
        if (!std::isfinite((double)r) || !std::isfinite((double)c))
            stats.finite = false;
        const long double error = c - r;
        stats.max_abs = std::max(stats.max_abs, std::fabs((double)error));
        stats.abs_sum += std::fabs(error);
        stats.error_sq += error * error;
        stats.reference_sq += r * r;
        stats.candidate_sq += c * c;
        stats.dot += r * c;
        stats.signed_sum += error;
        if (error > 0.0L) ++stats.positive;
    }
    return stats;
}

double nmse(const Stats &stats) {
    return (double)(stats.error_sq /
        std::max(stats.reference_sq, (long double)1.0e-30));
}

double cosine(const Stats &stats) {
    const long double denominator = std::sqrt(std::max(
        stats.reference_sq * stats.candidate_sq, (long double)1.0e-60));
    return (double)(stats.dot / denominator);
}

uint64_t fnv1a64(const void *data, size_t bytes) {
    const unsigned char *p = static_cast<const unsigned char *>(data);
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0u; i < bytes; ++i) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

void print_stats(const char *arm, const char *comparison,
                 const Stats &stats) {
    std::printf(
        "METRIC arm=%s comparison=%s count=%llu max_abs=%.17g "
        "mae=%.17g rmse=%.17g nmse=%.17g cosine=%.17g "
        "mean_signed_error=%.17g positive_error_fraction=%.17g finite=%d\n",
        arm, comparison, (unsigned long long)stats.count, stats.max_abs,
        stats.count ? (double)(stats.abs_sum / stats.count) : 0.0,
        stats.count ? std::sqrt((double)(stats.error_sq / stats.count)) : 0.0,
        nmse(stats), cosine(stats),
        stats.count ? (double)(stats.signed_sum / stats.count) : 0.0,
        stats.count ? (double)stats.positive / stats.count : 0.0,
        stats.finite ? 1 : 0);
}

void print_exact(const char *arm, const std::vector<float> &reference,
                 const std::vector<float> &candidate) {
    uint64_t mismatches = 0u;
    if (reference.size() != candidate.size()) {
        mismatches = std::max(reference.size(), candidate.size());
    } else {
        for (size_t i = 0u; i < reference.size(); ++i) {
            if (std::memcmp(&reference[i], &candidate[i], sizeof(float)) != 0)
                ++mismatches;
        }
    }
    std::printf("EXACT arm=%s count=%llu mismatches=%llu identical=%d\n",
                arm, (unsigned long long)reference.size(),
                (unsigned long long)mismatches, mismatches == 0u ? 1 : 0);
}

bool read_f32(const std::string &path, uint64_t count,
              std::vector<float> &out) {
    std::ifstream fp(path, std::ios::binary | std::ios::ate);
    if (!fp || fp.tellg() != (std::streamoff)(count * sizeof(float)))
        return false;
    fp.seekg(0);
    out.resize((size_t)count);
    return (bool)fp.read(reinterpret_cast<char *>(out.data()),
                         (std::streamsize)(count * sizeof(float)));
}

float bf16_to_f32(uint16_t value) {
    const uint32_t bits = (uint32_t)value << 16u;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

struct Arm {
    std::string name;
    std::vector<float> input;
    std::vector<float> captured;
    std::vector<long double> sequential;
    std::vector<float> half_grouped;
    std::vector<float> gpu_full;
    std::vector<float> gpu_kslice;
    std::vector<float> gpu_rowslice;
};

bool load_arm(const std::string &root, const char *name, Arm &arm) {
    arm.name = name;
    std::vector<float> half0, half1, output0, output1;
    const std::string prefix = root + "/raw/" + name;
    CHECK(read_f32(prefix + "-rank0-kda_local_gated.f32", kHalf, half0) &&
          read_f32(prefix + "-rank1-kda_local_gated.f32", kHalf, half1) &&
          read_f32(prefix + "-rank0-kda_out.f32", kOut, output0) &&
          read_f32(prefix + "-rank1-kda_out.f32", kOut, output1),
          "read two-rank real activation traces");
    CHECK(std::memcmp(output0.data(), output1.data(),
                      kOut * sizeof(float)) == 0,
          "rank outputs agree byte-for-byte");
    CHECK(std::memcmp(half0.data(), half1.data(),
                      kHalf * sizeof(float)) != 0,
          "rank input halves are distinct");
    arm.input.reserve(kFull);
    arm.input.insert(arm.input.end(), half0.begin(), half0.end());
    arm.input.insert(arm.input.end(), half1.begin(), half1.end());
    arm.captured = std::move(output0);
    return true;
}

void build_oracles(const uint16_t *weight, Arm &arm) {
    arm.sequential.resize(kOut);
    arm.half_grouped.resize(kOut);
    for (uint32_t row = 0u; row < kOut; ++row) {
        const uint16_t *wr = weight + (uint64_t)row * kFull;
        long double sequential = 0.0L;
        long double half0 = 0.0L;
        long double half1 = 0.0L;
        for (uint32_t k = 0u; k < kFull; ++k) {
            const long double product =
                (long double)bf16_to_f32(wr[k]) * arm.input[k];
            sequential += product;
            if (k < kHalf) half0 += product;
            else half1 += product;
        }
        arm.sequential[row] = sequential;
        arm.half_grouped[row] = (float)half0 + (float)half1;
    }
}

bool replay_gpu(const Glm5TestGGUF &gguf, uint64_t weight_offset, Arm &arm) {
    Tensors tensors;
    ds4_gpu_tensor *full_x = tensors.f32(kFull);
    ds4_gpu_tensor *half_x[2] = {tensors.f32(kHalf), tensors.f32(kHalf)};
    ds4_gpu_tensor *full_y = tensors.f32(kOut);
    ds4_gpu_tensor *partial[2] = {tensors.f32(kOut), tensors.f32(kOut)};
    ds4_gpu_tensor *sum = tensors.f32(kOut);
    ds4_gpu_tensor *row_half[2] = {tensors.f32(kOut / 2u),
                                   tensors.f32(kOut / 2u)};
    CHECK(full_x && half_x[0] && half_x[1] && full_y && partial[0] &&
          partial[1] && sum && row_half[0] && row_half[1],
          "allocate real activation replay tensors");
    CHECK(ds4_gpu_tensor_write(full_x, 0u, arm.input.data(),
                               kFull * sizeof(float)) &&
          ds4_gpu_tensor_write(half_x[0], 0u, arm.input.data(),
                               kHalf * sizeof(float)) &&
          ds4_gpu_tensor_write(half_x[1], 0u, arm.input.data() + kHalf,
                               kHalf * sizeof(float)),
          "upload real activation replay input");
    CHECK(ds4_gpu_matmul_bf16_tensor(
              full_y, gguf.map, gguf.size, weight_offset,
              kFull, kOut, full_x, 1u) &&
          ds4_gpu_matmul_bf16_kslice_rows_tensor(
              partial[0], gguf.map, gguf.size, weight_offset,
              kFull, kOut, 0u, kHalf, half_x[0], 1u) &&
          ds4_gpu_matmul_bf16_kslice_rows_tensor(
              partial[1], gguf.map, gguf.size, weight_offset,
              kFull, kOut, kHalf, kHalf, half_x[1], 1u) &&
          /* Output-row oracle: use the existing full-K kernel twice, with
           * only a row-aligned weight/output offset.  This intentionally
           * contains no TP or production-graph change. */
          ds4_gpu_matmul_bf16_tensor(
              row_half[0], gguf.map, gguf.size, weight_offset,
              kFull, kOut / 2u, full_x, 1u) &&
          ds4_gpu_matmul_bf16_tensor(
              row_half[1], gguf.map, gguf.size,
              weight_offset + (uint64_t)(kOut / 2u) * kFull * sizeof(uint16_t),
              kFull, kOut / 2u, full_x, 1u) &&
          ds4_gpu_add_tensor(sum, partial[0], partial[1], kOut) &&
          ds4_gpu_synchronize(),
          "execute real activation full and K-slice replay");
    arm.gpu_full.resize(kOut);
    arm.gpu_kslice.resize(kOut);
    arm.gpu_rowslice.resize(kOut);
    std::vector<float> row0(kOut / 2u), row1(kOut / 2u);
    CHECK(ds4_gpu_tensor_read(full_y, 0u, arm.gpu_full.data(),
                              kOut * sizeof(float)) &&
          ds4_gpu_tensor_read(sum, 0u, arm.gpu_kslice.data(),
                              kOut * sizeof(float)) &&
          ds4_gpu_tensor_read(row_half[0], 0u, row0.data(),
                              (kOut / 2u) * sizeof(float)) &&
          ds4_gpu_tensor_read(row_half[1], 0u, row1.data(),
                              (kOut / 2u) * sizeof(float)),
          "read real activation GPU replay");
    std::copy(row0.begin(), row0.end(), arm.gpu_rowslice.begin());
    std::copy(row1.begin(), row1.end(), arm.gpu_rowslice.begin() + kOut / 2u);
    if (std::getenv("DS4_GLM5_KDA_BENCHMARK") != nullptr) {
        constexpr uint32_t kWarmup = 8u;
        constexpr uint32_t kRepeats = 64u;
        double elapsed_ms[2] = {};
        double row_elapsed_ms[2] = {};
        for (uint32_t rank = 0u; rank < 2u; ++rank) {
            for (uint32_t i = 0u; i < kWarmup; ++i) {
                CHECK(ds4_gpu_matmul_bf16_kslice_rows_tensor(
                          partial[rank], gguf.map, gguf.size, weight_offset,
                          kFull, kOut, rank * kHalf, kHalf, half_x[rank], 1u),
                      "warm real activation K-slice benchmark");
            }
            hipEvent_t begin = nullptr, end = nullptr;
            CHECK(hipEventCreate(&begin) == hipSuccess &&
                  hipEventCreate(&end) == hipSuccess &&
                  hipEventRecord(begin) == hipSuccess,
                  "create real activation K-slice benchmark events");
            for (uint32_t i = 0u; i < kRepeats; ++i) {
                CHECK(ds4_gpu_matmul_bf16_kslice_rows_tensor(
                          partial[rank], gguf.map, gguf.size, weight_offset,
                          kFull, kOut, rank * kHalf, kHalf, half_x[rank], 1u),
                      "launch real activation K-slice benchmark");
            }
            CHECK(hipEventRecord(end) == hipSuccess &&
                  hipEventSynchronize(end) == hipSuccess,
                  "complete real activation K-slice benchmark events");
            float total_ms = 0.0f;
            CHECK(hipEventElapsedTime(&total_ms, begin, end) == hipSuccess &&
                  hipEventDestroy(end) == hipSuccess &&
                  hipEventDestroy(begin) == hipSuccess,
                  "read real activation K-slice benchmark events");
            elapsed_ms[rank] = (double)total_ms / kRepeats;
        }
        for (uint32_t half = 0u; half < 2u; ++half) {
            for (uint32_t i = 0u; i < kWarmup; ++i) {
                CHECK(ds4_gpu_matmul_bf16_tensor(
                          row_half[half], gguf.map, gguf.size,
                          weight_offset + (uint64_t)half * (kOut / 2u) *
                              kFull * sizeof(uint16_t),
                          kFull, kOut / 2u, full_x, 1u),
                      "warm real activation output-row benchmark");
            }
            hipEvent_t begin = nullptr, end = nullptr;
            CHECK(hipEventCreate(&begin) == hipSuccess &&
                  hipEventCreate(&end) == hipSuccess &&
                  hipEventRecord(begin) == hipSuccess,
                  "create real activation output-row benchmark events");
            for (uint32_t i = 0u; i < kRepeats; ++i) {
                CHECK(ds4_gpu_matmul_bf16_tensor(
                          row_half[half], gguf.map, gguf.size,
                          weight_offset + (uint64_t)half * (kOut / 2u) *
                              kFull * sizeof(uint16_t),
                          kFull, kOut / 2u, full_x, 1u),
                      "launch real activation output-row benchmark");
            }
            CHECK(hipEventRecord(end) == hipSuccess &&
                  hipEventSynchronize(end) == hipSuccess,
                  "complete real activation output-row benchmark events");
            float total_ms = 0.0f;
            CHECK(hipEventElapsedTime(&total_ms, begin, end) == hipSuccess &&
                  hipEventDestroy(end) == hipSuccess &&
                  hipEventDestroy(begin) == hipSuccess,
                  "read real activation output-row benchmark events");
            row_elapsed_ms[half] = (double)total_ms / kRepeats;
        }
        std::printf(
            "BENCH KDA_KSLICE rows=%u k=%u rank0_ms=%.9g rank1_ms=%.9g "
            "critical_ms=%.9g rowslice0_ms=%.9g rowslice1_ms=%.9g "
            "rowslice_sum_ms=%.9g repeats=%u\n",
            kOut, kHalf, elapsed_ms[0], elapsed_ms[1],
            std::max(elapsed_ms[0], elapsed_ms[1]), row_elapsed_ms[0],
            row_elapsed_ms[1], row_elapsed_ms[0] + row_elapsed_ms[1],
            kRepeats);
    }
    return true;
}

bool replay_synthetic_scales(const Glm5TestGGUF &gguf,
                             uint64_t weight_offset) {
    for (const float scale : {1.0e-4f, 1.0f, 1024.0f}) {
        Arm synthetic;
        synthetic.name = "synthetic";
        synthetic.input.resize(kFull);
        uint32_t state = UINT32_C(0x243f6a88);
        for (uint32_t i = 0u; i < kFull; ++i) {
            state = state * UINT32_C(1664525) + UINT32_C(1013904223);
            const int32_t centered = (int32_t)((state >> 8u) & 0xffffu) -
                                     32768;
            synthetic.input[i] = scale * (float)centered *
                                 (1.0f / 32768.0f);
        }
        CHECK(replay_gpu(gguf, weight_offset, synthetic),
              "replay synthetic-scale KDA input");
        const bool identical =
            synthetic.gpu_full.size() == synthetic.gpu_kslice.size() &&
            std::memcmp(synthetic.gpu_full.data(),
                        synthetic.gpu_kslice.data(),
                        synthetic.gpu_full.size() * sizeof(float)) == 0;
        std::printf(
            "SYNTHETIC_EXACT scale=%.9g count=%u identical=%d "
            "full_fnv=%016llx kslice_fnv=%016llx\n",
            scale, kOut, identical ? 1 : 0,
            (unsigned long long)fnv1a64(
                synthetic.gpu_full.data(),
                synthetic.gpu_full.size() * sizeof(float)),
            (unsigned long long)fnv1a64(
                synthetic.gpu_kslice.data(),
                synthetic.gpu_kslice.size() * sizeof(float)));
        CHECK(identical,
              "synthetic split-order full and K-slice are byte-identical");
    }
    return true;
}

bool report_arm(const Arm &arm, bool captured_is_kslice) {
    const bool require_exact =
        std::getenv("DS4_GLM5_KDA_REQUIRE_EXACT") != nullptr;
    print_stats(arm.name.c_str(), "captured-vs-sequential",
                compare(arm.sequential, arm.captured));
    print_stats(arm.name.c_str(), "full-vs-sequential",
                compare(arm.sequential, arm.gpu_full));
    print_stats(arm.name.c_str(), "kslice-vs-sequential",
                compare(arm.sequential, arm.gpu_kslice));
    print_stats(arm.name.c_str(), "full-vs-half-grouped",
                compare(arm.half_grouped, arm.gpu_full));
    print_stats(arm.name.c_str(), "kslice-vs-half-grouped",
                compare(arm.half_grouped, arm.gpu_kslice));
    print_stats(arm.name.c_str(), "rowslice-vs-full",
                compare(arm.gpu_full, arm.gpu_rowslice));
    print_stats(arm.name.c_str(), "full-vs-kslice",
                compare(arm.gpu_full, arm.gpu_kslice));
    print_exact(arm.name.c_str(), arm.gpu_full, arm.gpu_kslice);
    print_exact(arm.name.c_str(), arm.gpu_full, arm.gpu_rowslice);
    const std::vector<float> &replay =
        captured_is_kslice ? arm.gpu_kslice : arm.gpu_full;
    const Stats captured_replay = compare(replay, arm.captured);
    print_stats(arm.name.c_str(), "captured-vs-replay", captured_replay);

    uint32_t kslice_closer = 0u, full_closer = 0u, tied = 0u;
    for (uint32_t row = 0u; row < kOut; ++row) {
        const long double full_error = std::fabs(
            (long double)arm.gpu_full[row] - arm.sequential[row]);
        const long double kslice_error = std::fabs(
            (long double)arm.gpu_kslice[row] - arm.sequential[row]);
        if (kslice_error < full_error) ++kslice_closer;
        else if (full_error < kslice_error) ++full_closer;
        else ++tied;
    }
    std::printf(
        "CLOSER arm=%s oracle=sequential kslice=%u full=%u tied=%u\n",
        arm.name.c_str(), kslice_closer, full_closer, tied);
    /* Exact split-order mode intentionally replaces the archived full-path
     * reduction with the K-slice reduction.  Its layer-44 one-ULP-scale drift
     * from the old capture is expected; the strict assertion in this mode is
     * full-vs-K-slice identity below, not identity with obsolete arithmetic. */
    const double capture_tolerance = require_exact ? 2.0e-6 : 2.0e-7;
    CHECK(captured_replay.finite &&
          captured_replay.max_abs <= capture_tolerance,
          "captured path agrees with current same-input replay");
    if (require_exact) {
        CHECK(arm.gpu_full.size() == arm.gpu_kslice.size() &&
              std::memcmp(arm.gpu_full.data(), arm.gpu_kslice.data(),
                          arm.gpu_full.size() * sizeof(float)) == 0,
              "split-order full and K-slice replays are byte-identical");
    }
    return true;
}

bool run() {
    const char *model_path = std::getenv("DS4_GLM5_MODEL");
    const char *root = std::getenv("DS4_GLM5_KDA_ORACLE_ROOT");
    const char *layer_env = std::getenv("DS4_GLM5_KDA_ORACLE_LAYER");
    CHECK(model_path && model_path[0] && root && root[0] &&
          layer_env && layer_env[0], "real activation oracle environment");
    char *end = nullptr;
    const unsigned long layer_value = std::strtoul(layer_env, &end, 10);
    CHECK(end && !*end && layer_value < 45u,
          "valid real activation oracle layer");

    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model_path), "open GLM5 GGUF");
    uint64_t weight_offset = 0u;
    const std::string tensor_name = "blk." + std::to_string(layer_value) +
                                    ".kda_output.weight";
    CHECK(gguf.tensor(tensor_name, {kFull, kOut}, 30u, weight_offset),
          "locate layer BF16 KDA output tensor");
    CHECK(weight_offset <= gguf.size &&
          (uint64_t)kFull * kOut * sizeof(uint16_t) <=
              gguf.size - weight_offset,
          "validate layer BF16 KDA output tensor range");

    Arm full, kslice;
    CHECK(load_arm(root, "full", full) && load_arm(root, "kslice", kslice),
          "load real activation arms");
    print_stats("paths", "input-full-vs-kslice",
                compare(full.input, kslice.input));
    print_stats("paths", "captured-full-vs-kslice",
                compare(full.captured, kslice.captured));

    const uint16_t *weight = reinterpret_cast<const uint16_t *>(
        gguf.map + weight_offset);
    build_oracles(weight, full);
    build_oracles(weight, kslice);

    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    RuntimeGuard runtime;
    CHECK(ds4_gpu_init_multi(&config) &&
          ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map) &&
          ds4_gpu_set_model_map(gguf.map, gguf.size),
          "initialize gfx1151 for real activation replay");
    runtime.active = true;
    CHECK(replay_gpu(gguf, weight_offset, full) &&
          replay_gpu(gguf, weight_offset, kslice),
          "replay both real activation arms");
    if (std::getenv("DS4_GLM5_KDA_REQUIRE_EXACT") != nullptr) {
        CHECK(replay_synthetic_scales(gguf, weight_offset),
              "replay synthetic-scale exactness matrix");
    }
    CHECK(report_arm(full, false) && report_arm(kslice, true),
          "report real activation oracle metrics");
    std::printf(
        "PASS GLM5 KDA real activation oracle layer=%lu tensor=%s "
        "weight_offset=%llu\n",
        layer_value, tensor_name.c_str(),
        (unsigned long long)weight_offset);
    return true;
}

}  // namespace

int main() { return run() ? 0 : 1; }
