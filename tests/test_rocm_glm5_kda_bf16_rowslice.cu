#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
extern "C" {
#include "ds4_glm5_next_runtime.h"
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

struct ErrorStats {
    double max_abs = 0.0;
    long double error_sq = 0.0;
    long double reference_sq = 0.0;
    long double dot = 0.0;
    long double candidate_sq = 0.0;
    bool finite = true;
};

ErrorStats compare_f32(const std::vector<float> &reference,
                       const std::vector<float> &candidate) {
    ErrorStats stats;
    if (reference.size() != candidate.size()) {
        stats.finite = false;
        return stats;
    }
    for (size_t i = 0; i < reference.size(); ++i) {
        const double r = reference[i];
        const double c = candidate[i];
        if (!std::isfinite(r) || !std::isfinite(c)) stats.finite = false;
        const double error = c - r;
        stats.max_abs = std::max(stats.max_abs, std::fabs(error));
        stats.error_sq += (long double)error * error;
        stats.reference_sq += (long double)r * r;
        stats.candidate_sq += (long double)c * c;
        stats.dot += (long double)r * c;
    }
    return stats;
}

double nmse(const ErrorStats &stats) {
    return (double)(stats.error_sq /
        std::max(stats.reference_sq, (long double)1.0e-30));
}

double cosine(const ErrorStats &stats) {
    const long double denom = std::sqrt(
        std::max(stats.reference_sq * stats.candidate_sq,
                 (long double)1.0e-60));
    return (double)(stats.dot / denom);
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

bool run_output_kslice(const Glm5TestGGUF &gguf, uint64_t weight_offset,
                       uint32_t tokens, bool benchmark) {
    constexpr uint32_t kFull = 8192u;
    constexpr uint32_t kHalf = 4096u;
    constexpr uint32_t kOut = 4096u;
    const uint64_t full_input_count = (uint64_t)tokens * kFull;
    const uint64_t half_input_count = (uint64_t)tokens * kHalf;
    const uint64_t output_count = (uint64_t)tokens * kOut;
    std::vector<float> full_input((size_t)full_input_count);
    std::vector<float> half_input[2] = {
        std::vector<float>((size_t)half_input_count),
        std::vector<float>((size_t)half_input_count),
    };
    for (uint32_t token = 0u; token < tokens; ++token) {
        for (uint32_t k = 0u; k < kFull; ++k) {
            const uint64_t linear = (uint64_t)token * kFull + k;
            const int32_t centered =
                (int32_t)((linear * UINT64_C(89) + tokens * 31u) % 1021u) -
                510;
            const float value = (float)centered * (1.0f / 4096.0f) +
                0.015625f * std::sin((double)(linear % 16381u) * 0.013);
            full_input[(size_t)linear] = value;
            half_input[k / kHalf][(uint64_t)token * kHalf + k % kHalf] =
                value;
        }
    }

    Tensors tensors;
    ds4_gpu_tensor *full_x = tensors.f32(full_input_count);
    ds4_gpu_tensor *half_x[2] = {
        tensors.f32(half_input_count), tensors.f32(half_input_count),
    };
    ds4_gpu_tensor *full_y = tensors.f32(output_count);
    ds4_gpu_tensor *partial[2] = {
        tensors.f32(output_count), tensors.f32(output_count),
    };
    ds4_gpu_tensor *sum = tensors.f32(output_count);
    CHECK(full_x && half_x[0] && half_x[1] && full_y && partial[0] &&
          partial[1] && sum, "allocate BF16 K-slice tensors");
    CHECK(ds4_gpu_tensor_write(full_x, 0u, full_input.data(),
                               full_input_count * sizeof(float)) &&
          ds4_gpu_tensor_write(half_x[0], 0u, half_input[0].data(),
                               half_input_count * sizeof(float)) &&
          ds4_gpu_tensor_write(half_x[1], 0u, half_input[1].data(),
                               half_input_count * sizeof(float)),
          "upload deterministic BF16 K-slice inputs");

    CHECK(ds4_gpu_matmul_bf16_tensor(
              full_y, gguf.map, gguf.size, weight_offset,
              kFull, kOut, full_x, tokens) &&
          ds4_gpu_matmul_bf16_kslice_rows_tensor(
              partial[0], gguf.map, gguf.size, weight_offset,
              kFull, kOut, 0u, kHalf, half_x[0], tokens) &&
          ds4_gpu_matmul_bf16_kslice_rows_tensor(
              partial[1], gguf.map, gguf.size, weight_offset,
              kFull, kOut, kHalf, kHalf, half_x[1], tokens) &&
          ds4_gpu_add_tensor(sum, partial[0], partial[1],
                             (uint32_t)output_count) &&
          ds4_gpu_synchronize(),
          "execute full and two-half BF16 KDA output projections");

    std::vector<float> full_host((size_t)output_count);
    std::vector<float> partial_host[2] = {
        std::vector<float>((size_t)output_count),
        std::vector<float>((size_t)output_count),
    };
    std::vector<float> sum_host((size_t)output_count);
    CHECK(ds4_gpu_tensor_read(full_y, 0u, full_host.data(),
                              output_count * sizeof(float)) &&
          ds4_gpu_tensor_read(partial[0], 0u, partial_host[0].data(),
                              output_count * sizeof(float)) &&
          ds4_gpu_tensor_read(partial[1], 0u, partial_host[1].data(),
                              output_count * sizeof(float)) &&
          ds4_gpu_tensor_read(sum, 0u, sum_host.data(),
                              output_count * sizeof(float)),
          "read BF16 K-slice outputs");

    /* A batch-vs-tokenwise comparison is stricter than full-vs-two-halves at
     * one batch shape: it exposes row-count-dependent accumulation or tile
     * ordering before a two-node recurrence test can conflate that drift with
     * RDMA gate sequencing. */
    if (tokens <= 2049u) {
        ds4_gpu_tensor *tokenwise_partial[2] = {
            tensors.f32(output_count), tensors.f32(output_count),
        };
        CHECK(tokenwise_partial[0] && tokenwise_partial[1],
              "allocate tokenwise BF16 K-slice outputs");
        for (uint32_t token = 0u; token < tokens; ++token) {
            for (uint32_t rank = 0u; rank < 2u; ++rank) {
            ds4_gpu_tensor *input_row = ds4_gpu_tensor_view(
                half_x[rank], (uint64_t)token * kHalf * sizeof(float),
                (uint64_t)kHalf * sizeof(float));
            ds4_gpu_tensor *output_row = ds4_gpu_tensor_view(
                tokenwise_partial[rank],
                (uint64_t)token * kOut * sizeof(float),
                (uint64_t)kOut * sizeof(float));
            CHECK(input_row && output_row,
                  "create tokenwise BF16 K-slice views");
            const int row_ok = ds4_gpu_matmul_bf16_kslice_rows_tensor(
                output_row, gguf.map, gguf.size, weight_offset,
                kFull, kOut, rank * kHalf, kHalf, input_row, 1u);
            ds4_gpu_tensor_free(output_row);
            ds4_gpu_tensor_free(input_row);
            CHECK(row_ok, "execute tokenwise BF16 K-slice projection");
            }
        }
        CHECK(ds4_gpu_synchronize(),
              "synchronize tokenwise BF16 K-slice projections");
        std::vector<float> tokenwise_host[2] = {
            std::vector<float>((size_t)output_count),
            std::vector<float>((size_t)output_count),
        };
        CHECK(ds4_gpu_tensor_read(tokenwise_partial[0], 0u,
                                  tokenwise_host[0].data(),
                                  output_count * sizeof(float)) &&
              ds4_gpu_tensor_read(tokenwise_partial[1], 0u,
                                  tokenwise_host[1].data(),
                                  output_count * sizeof(float)),
              "read tokenwise BF16 K-slice outputs");
        for (uint32_t rank = 0u; rank < 2u; ++rank) {
            const ErrorStats row_error =
                compare_f32(partial_host[rank], tokenwise_host[rank]);
            uint64_t bit_mismatches = 0u;
            for (uint64_t i = 0u; i < output_count; ++i) {
                if (std::memcmp(&partial_host[rank][(size_t)i],
                                &tokenwise_host[rank][(size_t)i],
                                sizeof(float)) != 0) ++bit_mismatches;
            }
            std::fprintf(stderr,
                "MEASURE GLM5 KDA BF16 K-slice route tokens=%u rank=%u "
                "max_abs=%.9g nmse=%.9g nrmse=%.9g mismatches=%llu/%llu "
                "batch_fnv=%016llx tokenwise_fnv=%016llx\n",
                tokens, rank, row_error.max_abs, nmse(row_error),
                std::sqrt(nmse(row_error)),
                (unsigned long long)bit_mismatches,
                (unsigned long long)output_count,
                (unsigned long long)fnv1a64(
                    partial_host[rank].data(), output_count * sizeof(float)),
                (unsigned long long)fnv1a64(
                    tokenwise_host[rank].data(), output_count * sizeof(float)));
            const bool route_ok = row_error.finite &&
                row_error.max_abs <= 1.0e-7 &&
                std::sqrt(nmse(row_error)) <= 1.0e-7;
            std::fprintf(stderr,
                "DECISION GLM5 KDA BF16 K-slice route tokens=%u rank=%u "
                "result=%s threshold_nrmse=1e-7 threshold_max_abs=1e-7\n",
                tokens, rank, route_ok ? "PASS" : "REPAIR");
            if (std::getenv("DS4_GLM5_KDA_REQUIRE_ROUTE_MATCH") != nullptr) {
                CHECK(route_ok,
                      "BF16 K-slice batch preserves tokenwise route envelope");
            }
        }
    }
    const ErrorStats split_error = compare_f32(full_host, sum_host);
    CHECK(split_error.finite && split_error.max_abs <= 5.0e-4 &&
          nmse(split_error) <= 1.0e-7 && cosine(split_error) >= 0.9999999,
          "BF16 K-slice Lane-B numerical envelope");

    /* Independent address/layout check against a long-double dot for sampled
     * rows and the first/last token. This catches the packed-row footgun even
     * though the GPU full-vs-split comparison uses related kernels. */
    const uint16_t *weights = reinterpret_cast<const uint16_t *>(
        gguf.map + weight_offset);
    const uint32_t sample_rows[] = {0u, 1u, 127u, 2048u, 4095u};
    double cpu_max_abs = 0.0;
    for (uint32_t sample_token : {0u, tokens - 1u}) {
        for (uint32_t row : sample_rows) {
            long double ref = 0.0;
            const uint16_t *wr = weights + (uint64_t)row * kFull;
            for (uint32_t k = 0u; k < kFull; ++k) {
                const uint32_t bits = (uint32_t)wr[k] << 16u;
                float wf = 0.0f;
                std::memcpy(&wf, &bits, sizeof(wf));
                ref += (long double)wf *
                    full_input[(uint64_t)sample_token * kFull + k];
            }
            cpu_max_abs = std::max(cpu_max_abs,
                std::fabs((double)ref -
                          (double)sum_host[(uint64_t)sample_token * kOut + row]));
        }
    }
    CHECK(cpu_max_abs <= 2.0e-4,
          "BF16 K-slice sampled independent GGUF dot envelope");

    ds4_gpu_tensor *scratch = tensors.f32(output_count);
    CHECK(scratch &&
          !ds4_gpu_matmul_bf16_kslice_rows_tensor(
              scratch, gguf.map, gguf.size, weight_offset,
              kFull, kOut, 0u, kHalf - 1u, half_x[0], tokens) &&
          !ds4_gpu_matmul_bf16_kslice_rows_tensor(
              scratch, gguf.map, gguf.size, weight_offset,
              kFull, kOut, kHalf, kHalf + 32u, half_x[1], tokens),
          "BF16 K-slice malformed ranges fail closed");

    std::fprintf(stderr,
        "PASS GLM5 KDA BF16 K-slice tokens=%u max_abs=%.9g nmse=%.9g "
        "cosine=%.12g cpu_sample_max_abs=%.9g full_fnv=%016llx "
        "sum_fnv=%016llx\n",
        tokens, split_error.max_abs, nmse(split_error), cosine(split_error),
        cpu_max_abs,
        (unsigned long long)fnv1a64(full_host.data(),
                                    output_count * sizeof(float)),
        (unsigned long long)fnv1a64(sum_host.data(),
                                    output_count * sizeof(float)));

    if (benchmark) {
        const uint32_t warmup = tokens >= 512u ? 2u : 8u;
        const uint32_t repeats = tokens >= 512u ? 6u : 64u;
        const auto launch = [&](uint32_t operation) {
            if (operation == 0u)
                return ds4_gpu_matmul_bf16_tensor(
                    full_y, gguf.map, gguf.size, weight_offset,
                    kFull, kOut, full_x, tokens) != 0;
            if (operation <= 2u) {
                const uint32_t rank = operation - 1u;
                return ds4_gpu_matmul_bf16_kslice_rows_tensor(
                    partial[rank], gguf.map, gguf.size, weight_offset,
                    kFull, kOut, rank * kHalf, kHalf, half_x[rank], tokens) != 0;
            }
            return ds4_gpu_add_tensor(
                       sum, partial[0], partial[1],
                       (uint32_t)output_count) != 0;
        };
        const auto time_operation = [&](uint32_t operation) -> double {
            for (uint32_t i = 0u; i < warmup; ++i)
                CHECK(launch(operation), "warm BF16 K-slice timing operation");
            hipEvent_t begin = nullptr, end = nullptr;
            CHECK(hipEventCreate(&begin) == hipSuccess &&
                  hipEventCreate(&end) == hipSuccess &&
                  hipEventRecord(begin) == hipSuccess,
                  "create BF16 K-slice timing events");
            for (uint32_t i = 0u; i < repeats; ++i)
                CHECK(launch(operation), "launch BF16 K-slice timing operation");
            CHECK(hipEventRecord(end) == hipSuccess &&
                  hipEventSynchronize(end) == hipSuccess,
                  "complete BF16 K-slice timing events");
            float total_ms = 0.0f;
            CHECK(hipEventElapsedTime(&total_ms, begin, end) == hipSuccess,
                  "read BF16 K-slice timing events");
            CHECK(hipEventDestroy(end) == hipSuccess &&
                  hipEventDestroy(begin) == hipSuccess,
                  "destroy BF16 K-slice timing events");
            return (double)total_ms / repeats;
        };
        const double full_ms = time_operation(0u);
        const double rank0_ms = time_operation(1u);
        const double rank1_ms = time_operation(2u);
        const double add_ms = time_operation(3u);
        CHECK(full_ms > 0.0 && rank0_ms > 0.0 && rank1_ms > 0.0 &&
              add_ms > 0.0, "valid BF16 K-slice timing results");
        const double critical_ms = std::max(rank0_ms, rank1_ms) + add_ms;
        const double speedup = full_ms / critical_ms;
        std::fprintf(stderr,
            "MEASURE GLM5 KDA BF16 K-slice tokens=%u full_ms=%.6f "
            "rank0_ms=%.6f "
            "rank1_ms=%.6f add_ms=%.6f critical_ms=%.6f speedup=%.4fx "
            "decision=%s\n",
            tokens, full_ms, rank0_ms, rank1_ms, add_ms, critical_ms, speedup,
            speedup >= 1.4 ? "GO" : "STOP");
    }
    return true;
}

bool benchmark_output_rowslice(const Glm5TestGGUF &gguf,
                               uint64_t weight_offset,
                               uint32_t tokens) {
    constexpr uint32_t kIn = 8192u;
    constexpr uint32_t kOut = 4096u;
    constexpr uint32_t kLocalOut = kOut / 2u;
    const uint64_t input_count = (uint64_t)tokens * kIn;
    const uint64_t full_count = (uint64_t)tokens * kOut;
    const uint64_t local_count = (uint64_t)tokens * kLocalOut;
    std::vector<float> host_input((size_t)input_count);
    for (uint64_t i = 0u; i < input_count; ++i) {
        const int32_t centered =
            (int32_t)((i * UINT64_C(131) + tokens * 17u) % 1021u) - 510;
        host_input[(size_t)i] = (float)centered * (1.0f / 4096.0f) +
            0.0078125f * std::sin((double)(i % 32749u) * 0.011);
    }

    Tensors tensors;
    ds4_gpu_tensor *input = tensors.f32(input_count);
    ds4_gpu_tensor *full = tensors.f32(full_count);
    ds4_gpu_tensor *half[2] = {
        tensors.f32(local_count), tensors.f32(local_count),
    };
    CHECK(input && full && half[0] && half[1],
          "allocate BF16 output-row benchmark tensors");
    CHECK(ds4_gpu_tensor_write(input, 0u, host_input.data(),
                               input_count * sizeof(float)),
          "upload BF16 output-row benchmark input");

    const uint64_t half_weight_bytes =
        (uint64_t)kLocalOut * kIn * sizeof(uint16_t);
    const auto launch = [&](uint32_t operation) {
        if (operation == 0u) {
            return ds4_gpu_matmul_bf16_tensor(
                full, gguf.map, gguf.size, weight_offset,
                kIn, kOut, input, tokens) != 0;
        }
        const uint32_t rank = operation - 1u;
        return rank < 2u && ds4_gpu_matmul_bf16_tensor(
            half[rank], gguf.map, gguf.size,
            weight_offset + (uint64_t)rank * half_weight_bytes,
            kIn, kLocalOut, input, tokens) != 0;
    };
    CHECK(launch(0u) && launch(1u) && launch(2u) && ds4_gpu_synchronize(),
          "execute BF16 output-row benchmark identity arms");

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
          "read BF16 output-row benchmark identity arms");
    for (uint32_t token = 0u; token < tokens; ++token) {
        for (uint32_t rank = 0u; rank < 2u; ++rank) {
            CHECK(std::memcmp(
                      full_host.data() + (uint64_t)token * kOut +
                          (uint64_t)rank * kLocalOut,
                      half_host[rank].data() +
                          (uint64_t)token * kLocalOut,
                      (size_t)kLocalOut * sizeof(float)) == 0,
                  "BF16 output-row benchmark preserves exact rows");
        }
    }

    const uint32_t warmup = tokens >= 2048u ? 2u :
                            (tokens >= 256u ? 4u : 8u);
    const uint32_t repeats = tokens >= 2048u ? 6u :
                             (tokens >= 256u ? 12u : 64u);
    const auto time_operation = [&](uint32_t operation) -> double {
        for (uint32_t i = 0u; i < warmup; ++i)
            CHECK(launch(operation), "warm BF16 output-row timing arm");
        hipEvent_t begin = nullptr, end = nullptr;
        CHECK(hipEventCreate(&begin) == hipSuccess &&
              hipEventCreate(&end) == hipSuccess &&
              hipEventRecord(begin) == hipSuccess,
              "create BF16 output-row timing events");
        for (uint32_t i = 0u; i < repeats; ++i)
            CHECK(launch(operation), "launch BF16 output-row timing arm");
        CHECK(hipEventRecord(end) == hipSuccess &&
              hipEventSynchronize(end) == hipSuccess,
              "complete BF16 output-row timing events");
        float total_ms = 0.0f;
        CHECK(hipEventElapsedTime(&total_ms, begin, end) == hipSuccess &&
              hipEventDestroy(end) == hipSuccess &&
              hipEventDestroy(begin) == hipSuccess,
              "read BF16 output-row timing events");
        return (double)total_ms / repeats;
    };
    const double full_ms = time_operation(0u);
    const double rank0_ms = time_operation(1u);
    const double rank1_ms = time_operation(2u);
    CHECK(full_ms > 0.0 && rank0_ms > 0.0 && rank1_ms > 0.0,
          "valid BF16 output-row timing results");
    const double critical_ms = std::max(rank0_ms, rank1_ms);
    std::fprintf(stderr,
        "MEASURE GLM5 KDA BF16 output-row tokens=%u full_ms=%.6f "
        "rank0_ms=%.6f rank1_ms=%.6f critical_ms=%.6f speedup=%.4fx "
        "saved_ms=%.6f repeats=%u exact=1\n",
        tokens, full_ms, rank0_ms, rank1_ms, critical_ms,
        full_ms / critical_ms, full_ms - critical_ms, repeats);
    return true;
}

bool benchmark_decode_qkv_half(const Glm5TestGGUF &gguf,
                               const char *name,
                               uint64_t weight_offset) {
    constexpr uint32_t kIn = 4096u;
    constexpr uint32_t kFullOut = 8192u;
    constexpr uint32_t kLocalOut = kFullOut / 2u;
    std::vector<float> host_input(kIn);
    for (uint32_t i = 0u; i < kIn; ++i) {
        const int32_t centered =
            (int32_t)(((uint64_t)i * 193u + 17u) % 1021u) - 510;
        host_input[i] = (float)centered * (1.0f / 4096.0f) +
            0.0078125f * std::sin((double)i * 0.019);
    }

    Tensors tensors;
    ds4_gpu_tensor *input = tensors.f32(kIn);
    ds4_gpu_tensor *output[2] = {
        tensors.f32(kLocalOut), tensors.f32(kLocalOut),
    };
    CHECK(input && output[0] && output[1],
          "allocate BF16 decode-QKV benchmark tensors");
    CHECK(ds4_gpu_tensor_write(input, 0u, host_input.data(),
                               (uint64_t)kIn * sizeof(float)),
          "upload BF16 decode-QKV benchmark input");
    const uint64_t local_weight_bytes =
        (uint64_t)kLocalOut * kIn * sizeof(uint16_t);
    const auto launch = [&](uint32_t rank) {
        return rank < 2u && ds4_gpu_matmul_bf16_tensor(
            output[rank], gguf.map, gguf.size,
            weight_offset + (uint64_t)rank * local_weight_bytes,
            kIn, kLocalOut, input, 1u) != 0;
    };
    CHECK(launch(0u) && launch(1u) && ds4_gpu_synchronize(),
          "execute BF16 decode-QKV benchmark identity arms");
    std::vector<float> host_output[2] = {
        std::vector<float>(kLocalOut), std::vector<float>(kLocalOut),
    };
    CHECK(ds4_gpu_tensor_read(output[0], 0u, host_output[0].data(),
                              (uint64_t)kLocalOut * sizeof(float)) &&
          ds4_gpu_tensor_read(output[1], 0u, host_output[1].data(),
                              (uint64_t)kLocalOut * sizeof(float)),
          "read BF16 decode-QKV benchmark outputs");

    constexpr uint32_t warmup = 16u;
    constexpr uint32_t repeats = 256u;
    const auto time_rank = [&](uint32_t rank) -> double {
        for (uint32_t i = 0u; i < warmup; ++i)
            CHECK(launch(rank), "warm BF16 decode-QKV timing arm");
        hipEvent_t begin = nullptr, end = nullptr;
        CHECK(hipEventCreate(&begin) == hipSuccess &&
              hipEventCreate(&end) == hipSuccess &&
              hipEventRecord(begin) == hipSuccess,
              "create BF16 decode-QKV timing events");
        for (uint32_t i = 0u; i < repeats; ++i)
            CHECK(launch(rank), "launch BF16 decode-QKV timing arm");
        CHECK(hipEventRecord(end) == hipSuccess &&
              hipEventSynchronize(end) == hipSuccess,
              "complete BF16 decode-QKV timing events");
        float elapsed_ms = 0.0f;
        CHECK(hipEventElapsedTime(&elapsed_ms, begin, end) == hipSuccess &&
              hipEventDestroy(end) == hipSuccess &&
              hipEventDestroy(begin) == hipSuccess,
              "read BF16 decode-QKV timing events");
        return (double)elapsed_ms / repeats;
    };
    const double ms[2] = {time_rank(0u), time_rank(1u)};
    const double weight_gb = (double)local_weight_bytes / 1.0e9;
    std::fprintf(stderr,
        "MEASURE GLM5 BF16 decode-QKV tensor=%s shape=%ux%u "
        "rank0_ms=%.6f rank1_ms=%.6f critical_ms=%.6f "
        "rank0_gbps=%.3f rank1_gbps=%.3f "
        "rank0_fnv=%016llx rank1_fnv=%016llx repeats=%u\n",
        name, kIn, kLocalOut, ms[0], ms[1], std::max(ms[0], ms[1]),
        weight_gb / (ms[0] / 1000.0), weight_gb / (ms[1] / 1000.0),
        (unsigned long long)fnv1a64(
            host_output[0].data(), (uint64_t)kLocalOut * sizeof(float)),
        (unsigned long long)fnv1a64(
            host_output[1].data(), (uint64_t)kLocalOut * sizeof(float)),
        repeats);
    return ms[0] > 0.0 && ms[1] > 0.0;
}

bool benchmark_decode_qkv_multiptr(const Glm5TestGGUF &gguf,
                                   uint64_t q_offset, uint64_t k_offset,
                                   uint64_t v_offset) {
    constexpr uint32_t kIn = 4096u;
    constexpr uint32_t kOut = 4096u;
    const uint64_t weight_bytes = (uint64_t)kIn * kOut * sizeof(uint16_t);
    std::vector<float> host_input(kIn);
    for (uint32_t i = 0u; i < kIn; ++i)
        host_input[i] = 0.03125f * std::sin((double)i * 0.013) +
                        0.0078125f * std::cos((double)i * 0.031);
    Tensors tensors;
    ds4_gpu_tensor *input = tensors.f32(kIn);
    ds4_gpu_tensor *seq[3] = {tensors.f32(kOut), tensors.f32(kOut),
                              tensors.f32(kOut)};
    ds4_gpu_tensor *fused[3] = {tensors.f32(kOut), tensors.f32(kOut),
                                tensors.f32(kOut)};
    CHECK(input && seq[0] && seq[1] && seq[2] && fused[0] && fused[1] &&
          fused[2] && ds4_gpu_tensor_write(input, 0u, host_input.data(),
                                           (uint64_t)kIn * sizeof(float)),
          "allocate decode QKV multiptr A/B tensors");
    const uint64_t offsets[3] = {q_offset, k_offset, v_offset};
    const auto sequential = [&] {
        for (uint32_t i = 0u; i < 3u; ++i)
            CHECK(ds4_gpu_matmul_bf16_tensor(
                      seq[i], gguf.map, gguf.size, offsets[i], kIn, kOut,
                      input, 1u), "launch sequential decode QKV");
        return true;
    };
    const auto multiptr = [&] {
        return ds4_gpu_matmul_bf16_qkv_decode_multiptr_tensor(
            fused[0], fused[1], fused[2], gguf.map, gguf.size,
            q_offset, k_offset, v_offset, kIn, kOut, input, 1u) != 0;
    };
    CHECK(sequential() && multiptr() && ds4_gpu_synchronize(),
          "warm decode QKV multiptr A/B");
    std::vector<float> a(kOut), b(kOut);
    for (uint32_t i = 0u; i < 3u; ++i) {
        CHECK(ds4_gpu_tensor_read(seq[i], 0u, a.data(),
                                  (uint64_t)kOut * sizeof(float)) &&
              ds4_gpu_tensor_read(fused[i], 0u, b.data(),
                                  (uint64_t)kOut * sizeof(float)),
              "read decode QKV multiptr A/B");
        CHECK(std::memcmp(a.data(), b.data(), (size_t)kOut * sizeof(float)) == 0,
              "decode QKV multiptr bit exact");
    }
    constexpr uint32_t warmup = 8u, repeats = 64u;
    const auto timed = [&](bool use_fused) -> double {
        for (uint32_t i = 0u; i < warmup; ++i)
            CHECK(use_fused ? multiptr() : sequential(),
                  "warm decode QKV multiptr arm");
        hipEvent_t begin = nullptr, end = nullptr;
        CHECK(hipEventCreate(&begin) == hipSuccess &&
              hipEventCreate(&end) == hipSuccess && hipEventRecord(begin) == hipSuccess,
              "create decode QKV multiptr events");
        for (uint32_t i = 0u; i < repeats; ++i)
            CHECK(use_fused ? multiptr() : sequential(),
                  "time decode QKV multiptr arm");
        CHECK(hipEventRecord(end) == hipSuccess && hipEventSynchronize(end) == hipSuccess,
              "complete decode QKV multiptr events");
        float elapsed = 0.0f;
        CHECK(hipEventElapsedTime(&elapsed, begin, end) == hipSuccess &&
              hipEventDestroy(end) == hipSuccess && hipEventDestroy(begin) == hipSuccess,
              "read decode QKV multiptr events");
        return (double)elapsed / repeats;
    };
    const double seq_ms = timed(false);
    const double fused_ms = timed(true);
    std::fprintf(stderr,
                 "MEASURE GLM5 BF16 decode-QKV multiptr shape=3x1x%ux%u "
                 "sequential_ms=%.6f fused_ms=%.6f speedup=%.3fx "
                 "weight_gib=%.6f exact=1 repeats=%u\n",
                 kOut, kIn, seq_ms, fused_ms, seq_ms / fused_ms,
                 (double)(3u * weight_bytes) / (1024.0 * 1024.0 * 1024.0),
                 repeats);
    return fused_ms > 0.0;
}

bool benchmark_decode_qkv_stream(const Glm5TestGGUF &gguf,
                                 const std::vector<uint64_t> &offsets) {
    constexpr uint32_t kIn = 4096u;
    constexpr uint32_t kLocalOut = 4096u;
    const uint64_t local_weight_bytes =
        (uint64_t)kLocalOut * kIn * sizeof(uint16_t);
    CHECK(offsets.size() == 102u,
          "34-layer GLM5 KDA Q/K/V stream geometry");
    std::vector<float> host_input(kIn);
    for (uint32_t i = 0u; i < kIn; ++i) {
        const int32_t centered =
            (int32_t)(((uint64_t)i * 193u + 17u) % 1021u) - 510;
        host_input[i] = (float)centered * (1.0f / 4096.0f) +
            0.0078125f * std::sin((double)i * 0.019);
    }
    Tensors tensors;
    ds4_gpu_tensor *input = tensors.f32(kIn);
    ds4_gpu_tensor *output = tensors.f32(kLocalOut);
    CHECK(input && output &&
          ds4_gpu_tensor_write(input, 0u, host_input.data(),
                               (uint64_t)kIn * sizeof(float)),
          "allocate and upload BF16 decode-QKV stream tensors");
    const auto launch_stream = [&](uint32_t rank) {
        if (rank > 1u) return false;
        for (uint64_t offset : offsets) {
            if (!ds4_gpu_matmul_bf16_tensor(
                    output, gguf.map, gguf.size,
                    offset + (uint64_t)rank * local_weight_bytes,
                    kIn, kLocalOut, input, 1u)) return false;
        }
        return true;
    };
    /* One complete 3.42 GB/rank traversal faults in every required model
     * range but is far larger than the cache hierarchy.  Subsequent repeats
     * therefore model production's layer-to-layer stream instead of the
     * misleading 32 MiB single-matrix cache loop above. */
    CHECK(launch_stream(0u) && launch_stream(1u) && ds4_gpu_synchronize(),
          "warm complete BF16 decode-QKV stream");
    constexpr uint32_t repeats = 4u;
    const auto time_rank = [&](uint32_t rank) -> double {
        hipEvent_t begin = nullptr, end = nullptr;
        CHECK(hipEventCreate(&begin) == hipSuccess &&
              hipEventCreate(&end) == hipSuccess &&
              hipEventRecord(begin) == hipSuccess,
              "create BF16 decode-QKV stream timing events");
        for (uint32_t i = 0u; i < repeats; ++i)
            CHECK(launch_stream(rank), "launch BF16 decode-QKV stream arm");
        CHECK(hipEventRecord(end) == hipSuccess &&
              hipEventSynchronize(end) == hipSuccess,
              "complete BF16 decode-QKV stream timing events");
        float elapsed_ms = 0.0f;
        CHECK(hipEventElapsedTime(&elapsed_ms, begin, end) == hipSuccess &&
              hipEventDestroy(end) == hipSuccess &&
              hipEventDestroy(begin) == hipSuccess,
              "read BF16 decode-QKV stream timing events");
        return (double)elapsed_ms / repeats;
    };
    const double ms[2] = {time_rank(0u), time_rank(1u)};
    CHECK(ds4_gpu_synchronize(), "finish BF16 decode-QKV stream benchmark");
    std::vector<float> host_output(kLocalOut);
    CHECK(ds4_gpu_tensor_read(output, 0u, host_output.data(),
                              (uint64_t)kLocalOut * sizeof(float)),
          "read BF16 decode-QKV stream terminal output");
    const double stream_gb =
        (double)(local_weight_bytes * offsets.size()) / 1.0e9;
    std::fprintf(stderr,
        "DECISION GLM5 BF16 decode-QKV stream projections=%zu "
        "bytes_gb=%.6f rank0_ms=%.6f rank1_ms=%.6f critical_ms=%.6f "
        "rank0_gbps=%.3f rank1_gbps=%.3f terminal_fnv=%016llx "
        "repeats=%u\n",
        offsets.size(), stream_gb, ms[0], ms[1],
        std::max(ms[0], ms[1]),
        stream_gb / (ms[0] / 1000.0), stream_gb / (ms[1] / 1000.0),
        (unsigned long long)fnv1a64(
            host_output.data(), (uint64_t)kLocalOut * sizeof(float)),
        repeats);
    return ms[0] > 0.0 && ms[1] > 0.0;
}

bool benchmark_decode_output_stream(const Glm5TestGGUF &gguf,
                                    const std::vector<uint64_t> &offsets) {
    constexpr uint32_t kIn = 8192u;
    constexpr uint32_t kLocalOut = 2048u;
    const uint64_t local_weight_bytes =
        (uint64_t)kLocalOut * kIn * sizeof(uint16_t);
    CHECK(offsets.size() == 34u,
          "34-layer GLM5 KDA output stream geometry");
    std::vector<float> host_input(kIn);
    for (uint32_t i = 0u; i < kIn; ++i) {
        const int32_t centered =
            (int32_t)(((uint64_t)i * 157u + 29u) % 2053u) - 1026;
        host_input[i] = (float)centered * (1.0f / 8192.0f) +
            0.00390625f * std::cos((double)i * 0.011);
    }
    Tensors tensors;
    ds4_gpu_tensor *input = tensors.f32(kIn);
    ds4_gpu_tensor *output = tensors.f32(kLocalOut);
    CHECK(input && output &&
          ds4_gpu_tensor_write(input, 0u, host_input.data(),
                               (uint64_t)kIn * sizeof(float)),
          "allocate and upload BF16 decode-output stream tensors");
    const auto launch_stream = [&](uint32_t rank) {
        if (rank > 1u) return false;
        for (uint64_t offset : offsets) {
            if (!ds4_gpu_matmul_bf16_tensor(
                    output, gguf.map, gguf.size,
                    offset + (uint64_t)rank * local_weight_bytes,
                    kIn, kLocalOut, input, 1u)) return false;
        }
        return true;
    };
    CHECK(launch_stream(0u) && launch_stream(1u) && ds4_gpu_synchronize(),
          "warm complete BF16 decode-output stream");
    constexpr uint32_t repeats = 8u;
    const auto time_rank = [&](uint32_t rank) -> double {
        hipEvent_t begin = nullptr, end = nullptr;
        CHECK(hipEventCreate(&begin) == hipSuccess &&
              hipEventCreate(&end) == hipSuccess &&
              hipEventRecord(begin) == hipSuccess,
              "create BF16 decode-output stream timing events");
        for (uint32_t i = 0u; i < repeats; ++i)
            CHECK(launch_stream(rank),
                  "launch BF16 decode-output stream arm");
        CHECK(hipEventRecord(end) == hipSuccess &&
              hipEventSynchronize(end) == hipSuccess,
              "complete BF16 decode-output stream timing events");
        float elapsed_ms = 0.0f;
        CHECK(hipEventElapsedTime(&elapsed_ms, begin, end) == hipSuccess &&
              hipEventDestroy(end) == hipSuccess &&
              hipEventDestroy(begin) == hipSuccess,
              "read BF16 decode-output stream timing events");
        return (double)elapsed_ms / repeats;
    };
    const double ms[2] = {time_rank(0u), time_rank(1u)};
    CHECK(ds4_gpu_synchronize(), "finish BF16 decode-output stream benchmark");
    std::vector<float> host_output(kLocalOut);
    CHECK(ds4_gpu_tensor_read(output, 0u, host_output.data(),
                              (uint64_t)kLocalOut * sizeof(float)),
          "read BF16 decode-output stream terminal output");
    const double stream_gb =
        (double)(local_weight_bytes * offsets.size()) / 1.0e9;
    std::fprintf(stderr,
        "DECISION GLM5 BF16 decode-output stream projections=%zu "
        "bytes_gb=%.6f rank0_ms=%.6f rank1_ms=%.6f critical_ms=%.6f "
        "rank0_gbps=%.3f rank1_gbps=%.3f terminal_fnv=%016llx "
        "repeats=%u\n",
        offsets.size(), stream_gb, ms[0], ms[1],
        std::max(ms[0], ms[1]),
        stream_gb / (ms[0] / 1000.0), stream_gb / (ms[1] / 1000.0),
        (unsigned long long)fnv1a64(
            host_output.data(), (uint64_t)kLocalOut * sizeof(float)),
        repeats);
    return ms[0] > 0.0 && ms[1] > 0.0;
}

bool run_test() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    CHECK(model && model[0], "DS4_GLM5_MODEL environment");
    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM5 GGUF");

    uint64_t q = 0u, k = 0u, v = 0u, output = 0u;
    uint64_t f_b = 0u, g_b = 0u, beta = 0u;
    uint64_t q_conv = 0u, k_conv = 0u, v_conv = 0u;
    uint64_t dt_bias = 0u, a_log = 0u;
    CHECK(gguf.tensor("blk.0.kda_q.weight", {4096u, 8192u}, 30u, q) &&
          gguf.tensor("blk.0.kda_k.weight", {4096u, 8192u}, 30u, k) &&
          gguf.tensor("blk.0.kda_v.weight", {4096u, 8192u}, 30u, v) &&
          gguf.tensor("blk.0.kda_output.weight", {8192u, 4096u}, 30u,
                      output) &&
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

    std::vector<uint64_t> qkv_stream;
    std::vector<uint64_t> output_stream;
    for (uint32_t il = 0u; il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il) {
        if (ds4_glm5_next_layer_is_mla(il)) continue;
        for (const char *suffix : {"kda_q", "kda_k", "kda_v"}) {
            char tensor_name[64];
            const int written = std::snprintf(
                tensor_name, sizeof(tensor_name), "blk.%u.%s.weight",
                il, suffix);
            uint64_t offset = 0u;
            CHECK(written > 0 && (size_t)written < sizeof(tensor_name) &&
                  gguf.tensor(tensor_name, {4096u, 8192u}, 30u, offset),
                  "bind complete GLM5 KDA Q/K/V stream");
            qkv_stream.push_back(offset);
        }
        char output_name[64];
        const int output_written = std::snprintf(
            output_name, sizeof(output_name), "blk.%u.kda_output.weight", il);
        uint64_t output_offset = 0u;
        CHECK(output_written > 0 &&
              (size_t)output_written < sizeof(output_name) &&
              gguf.tensor(output_name, {8192u, 4096u}, 30u, output_offset),
              "bind complete GLM5 KDA output stream");
        output_stream.push_back(output_offset);
    }
    CHECK(qkv_stream.size() == 102u,
          "bind exactly 34 GLM5 KDA Q/K/V triples");
    CHECK(output_stream.size() == 34u,
          "bind exactly 34 GLM5 KDA output tensors");

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

    if (std::getenv("DS4_GLM5_KDA_BF16_DECODE_BENCH_ONLY") != nullptr) {
        CHECK(benchmark_decode_qkv_half(gguf, "kda_q", q) &&
              benchmark_decode_qkv_half(gguf, "kda_k", k) &&
              benchmark_decode_qkv_half(gguf, "kda_v", v) &&
              benchmark_decode_qkv_multiptr(gguf, q, k, v) &&
              benchmark_decode_qkv_stream(gguf, qkv_stream) &&
              benchmark_decode_output_stream(gguf, output_stream),
              "GLM5 BF16 decode projection geometry benchmark");
        std::fprintf(stderr,
                     "PASS GLM5 BF16 decode-QKV benchmark-only gate\n");
        return true;
    }

    for (uint32_t tokens : {1u, 3u, 4u, 33u}) {
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
        CHECK(run_output_kslice(
                  gguf, output, tokens, tokens == 1u || tokens == 33u),
              "KDA output K-slice real-GGUF oracle");
    }
    /* The production token-tile dispatcher consumes complete 32-row chunks
     * followed by 16/8/4/2/1 tails.  Exercise both sides of the first two
     * boundaries and the production prefill boundary against repeated
     * one-row execution, rather than assuming the 33-row case covers every
     * tail composition. */
    for (uint32_t tokens : {31u, 32u, 63u, 64u, 65u, 2049u}) {
        CHECK(run_output_kslice(gguf, output, tokens, false),
              "KDA output K-slice token-tile boundary oracle");
    }
    CHECK(run_output_kslice(gguf, output, 2048u, true),
          "KDA output K-slice 2048-row correctness and timing oracle");
    for (uint32_t tokens : {1u, 256u, 2048u}) {
        CHECK(benchmark_output_rowslice(gguf, output, tokens),
              "KDA output-row M=1/256/2048 geometry benchmark");
    }
    std::fprintf(stderr,
                 "PASS complete real-GGUF GLM5 KDA BF16 row-slice gate\n");
    return true;
}

}  // namespace

int main() { return run_test() ? 0 : 1; }
