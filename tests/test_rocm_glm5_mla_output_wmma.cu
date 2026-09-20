#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "tests/glm5_gguf_test.hpp"

static void require(bool ok, const char *message) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", message); std::exit(1); }
}

int main(int argc, char **argv) {
    uint32_t M = 256u;
    if (argc != 1) {
        require(argc == 3 && std::strcmp(argv[1], "--rows") == 0 &&
                (std::strcmp(argv[2], "256") == 0 ||
                 std::strcmp(argv[2], "1024") == 0),
                "usage: test_rocm_glm5_mla_output_wmma [--rows 256|1024]");
        M = std::strcmp(argv[2], "1024") == 0 ? 1024u : 256u;
    }
    const char *model = std::getenv("DS4_GLM5_MODEL");
    Glm5TestGGUF gguf;
    require(model && gguf.open_file(model), "open real GGUF");
    ds4_gpu_config config{};
    config.n_gpus = 1;
    require(ds4_gpu_init_multi(&config) &&
            ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map) &&
            ds4_gpu_set_model_map(gguf.map, gguf.size), "initialize mapped model");
    constexpr uint32_t FullK = 16384, K = 8192, N = 4096;
    const uint64_t OutBytes = (uint64_t)M * N * sizeof(float);
    constexpr uint64_t GuardBytes = 64u * sizeof(float);
    constexpr float Canary = -1234567.0f;
    const uint64_t row_bytes = (uint64_t)(FullK / 32u) * 34u;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)M * FullK * sizeof(float));
    ds4_gpu_tensor *packed = ds4_gpu_tensor_alloc((uint64_t)M * K * sizeof(float));
    ds4_gpu_tensor *out_storage = ds4_gpu_tensor_alloc(OutBytes + 2u * GuardBytes);
    ds4_gpu_tensor *out = out_storage ?
        ds4_gpu_tensor_view(out_storage, GuardBytes, OutBytes) : nullptr;
    require(x && packed && out, "bounded activation/output scratch");
    const std::vector<float> guards(64u, Canary);
    require(ds4_gpu_tensor_write(out_storage, 0, guards.data(), GuardBytes) &&
            ds4_gpu_tensor_write(out_storage, GuardBytes + OutBytes,
                                 guards.data(), GuardBytes), "write output canaries");
    auto check_guards = [&]() {
        std::vector<float> before(64u), after(64u);
        require(ds4_gpu_tensor_read(out_storage, 0, before.data(), GuardBytes) &&
                ds4_gpu_tensor_read(out_storage, GuardBytes + OutBytes,
                                     after.data(), GuardBytes), "read output canaries");
        require(before == guards && after == guards, "output canaries intact");
    };
    std::printf("fixture rows=%u layers=11 slices=2 weights=original synthetic_activations=1 quality_admission=0\n", M);
    std::vector<float> host((size_t)M * FullK), gather((size_t)M * K);
    for (size_t i = 0; i < host.size(); ++i)
        host[i] = 0.2f * std::sin((double)i * 0.037) +
                  (float)((int)((i * 191u) % 997u) - 498) / 3171.0f;
    require(ds4_gpu_tensor_write(x, 0, host.data(), host.size() * sizeof(float)),
            "write input with distinct TP halves");
    std::vector<float> baseline((size_t)M * N), candidate(baseline.size()), oracle(baseline.size());
    for (uint32_t layer = 3u; layer < 45u; layer += 4u) {
        uint64_t offset = 0;
        require(gguf.tensor("blk." + std::to_string(layer) + ".attn_output.weight",
                            {FullK, N}, 8u, offset), "Q8 MLA output shape");
        for (uint32_t rank : {0u, 1u}) {
            const uint32_t start = rank * K;
            auto run = [&](bool wmma, bool gathered, uint32_t rows = 0u) {
                if (rows == 0u) rows = M;
                require(setenv("DS4_ROCM_GLM5_MLA_OUTPUT_WMMA", wmma ? "1" : "0", 1) == 0,
                        "set selector");
                return ds4_rocm_q8_kslice_f32_rows_strided(
                    out, gguf.map, gguf.size, offset, FullK, N, start, K,
                    gathered ? packed : x, gathered ? 0u : start,
                    rows, gathered ? K : FullK);
            };
            require(run(true, false, 1u) == -1, "M1 declines prefill accelerator");
            for (uint32_t t = 0; t < M; ++t)
                std::memcpy(gather.data() + (size_t)t * K,
                            host.data() + (size_t)t * FullK + start, K * sizeof(float));
            require(ds4_gpu_tensor_write(packed, 0, gather.data(), gather.size() * sizeof(float)),
                    "upload gathered activation oracle");
            require(run(false, false) == 1 && ds4_gpu_synchronize() &&
                    ds4_gpu_tensor_read(out, 0, baseline.data(), OutBytes), "incumbent output");
            require(run(true, false) == 1 && ds4_gpu_synchronize() &&
                    ds4_gpu_tensor_read(out, 0, candidate.data(), OutBytes), "strided WMMA output");
            require(run(true, true) == 1 && ds4_gpu_synchronize() &&
                    ds4_gpu_tensor_read(out, 0, oracle.data(), OutBytes), "gathered WMMA oracle");
            require(std::memcmp(candidate.data(), oracle.data(), OutBytes) == 0,
                    "strided and gathered output bytes match");
            check_guards();
            std::vector<float> errors(candidate.size());
            double sq_error = 0, sq_reference = 0;
            for (size_t i = 0; i < candidate.size(); ++i) {
                require(std::isfinite(candidate[i]) && std::isfinite(baseline[i]), "finite outputs");
                errors[i] = std::fabs(candidate[i] - baseline[i]);
                sq_error += (double)errors[i] * errors[i];
                sq_reference += (double)baseline[i] * baseline[i];
            }
            std::sort(errors.begin(), errors.end());
            require(errors.back() > 0, "candidate differs from F32 rollback (engagement)");
            // An analytic rounding envelope detects indexing/codec errors in
            // independent sampled CPU dots. This is not a full-model quality gate.
            double cpu_max = 0;
            for (uint32_t sample = 0; sample < 64; ++sample) {
                const uint32_t t = sample * 197u % M, n = sample * 719u % N;
                const unsigned char *wr = gguf.map + offset + n * row_bytes + (start / 32u) * 34u;
                long double dot = 0, magnitude = 0;
                for (uint32_t k = 0; k < K; ++k) {
                    const unsigned char *block = wr + (k / 32u) * 34u;
                    _Float16 scale;
                    std::memcpy(&scale, block, 2);
                    const float w = (float)scale * (float)((const int8_t *)(block + 2))[k % 32u];
                    const long double product = (long double)w * gather[(size_t)t * K + k];
                    dot += product;
                    magnitude += std::fabs(product);
                }
                const double error = std::fabs((double)dot - candidate[(size_t)t * N + n]);
                cpu_max = std::max(cpu_max, error);
                require(error <= (double)magnitude * 0.003 + 1.0e-6,
                        "canonical sampled Q8 dot rounding envelope");
            }
            std::printf("numerical layer=%u rank=%u rows=%u exact_gather=1 max_abs=%g p99_abs=%g nmse=%.9g cpu_max=%g\n",
                        layer, rank, M, errors.back(), errors[errors.size() * 99u / 100u],
                        sq_error / std::max(sq_reference, 1.0e-30), cpu_max);
            require(run(false, false) == 1 && ds4_gpu_synchronize() &&
                    ds4_gpu_tensor_read(out, 0, oracle.data(), OutBytes) &&
                    std::memcmp(baseline.data(), oracle.data(), OutBytes) == 0,
                    "selector rollback bytes");
            check_guards();
            std::vector<double> times[2];
            for (uint32_t sample = 0; sample < 5; ++sample) {
                for (uint32_t order = 0; order < 2; ++order) {
                    const uint32_t arm = order ^ (sample & 1u);
                    require(ds4_gpu_synchronize(), "timing fence");
                    const auto begin = std::chrono::steady_clock::now();
                    for (uint32_t repeat = 0; repeat < 3; ++repeat)
                        require(run(arm != 0u, false) == 1, "timed projection");
                    require(ds4_gpu_synchronize(), "complete timing");
                    times[arm].push_back(std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - begin).count() / 3.0);
                }
            }
            for (auto &v : times) std::sort(v.begin(), v.end());
            std::printf("timing layer=%u rank=%u rows=%u control_ms=%.6f candidate_ms=%.6f speedup=%.3f\n",
                        layer, rank, M, times[0][2], times[1][2], times[0][2] / times[1][2]);
            check_guards();
            std::fflush(stdout);
        }
    }
    require(setenv("DS4_ROCM_GLM5_MLA_OUTPUT_WMMA", "1", 1) == 0, "set bounds selector");
    uint64_t off = 0;
    require(gguf.tensor("blk.3.attn_output.weight", {FullK, N}, 8u, off), "bounds weight");
    require(ds4_rocm_q8_kslice_f32_rows_strided(out, gguf.map, gguf.size, off,
                FullK, N, 0, K, x, UINT64_MAX, M, FullK) == 0,
            "overflowed activation offset rejected");
    ds4_gpu_tensor *short_x = ds4_gpu_tensor_view(x, 0, ((uint64_t)M * FullK - 1u) * sizeof(float));
    require(short_x && ds4_rocm_q8_kslice_f32_rows_strided(out, gguf.map, gguf.size, off,
                FullK, N, K, K, short_x, K, M, FullK) == 0, "short last row rejected");
    ds4_gpu_tensor_free(short_x);
    for (unsigned arm = 0; arm < 2; ++arm) {
        require(setenv("DS4_ROCM_GLM5_MLA_OUTPUT_WMMA", arm ? "1" : "0", 1) == 0,
                "set tail selector");
        require(ds4_rocm_q8_kslice_f32_rows_strided(out, gguf.map, gguf.size, off,
                    FullK, N, K, K, x, K, M - 1u, FullK) == 1 &&
                ds4_gpu_synchronize() && ds4_gpu_tensor_read(out, 0,
                    arm ? candidate.data() : baseline.data(), (uint64_t)(M - 1u) * N * sizeof(float)),
                "nonmultiple-of-256 incumbent fallback");
    }
    require(std::memcmp(baseline.data(), candidate.data(),
                        (size_t)(M - 1u) * N * sizeof(float)) == 0,
            "nonmultiple-of-256 selector retains incumbent bytes");
    require(setenv("DS4_ROCM_GLM5_MLA_OUTPUT_WMMA", "invalid", 1) == 0 &&
            ds4_rocm_q8_kslice_f32_rows_strided(out, gguf.map, gguf.size, off,
                FullK, N, K, K, x, K, M, FullK) == 0, "invalid selector rejected");
    unsetenv("DS4_ROCM_GLM5_MLA_OUTPUT_WMMA");
    check_guards();
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(out_storage);
    ds4_gpu_tensor_free(packed); ds4_gpu_tensor_free(x);
    std::puts("PASS real-GGUF MLA output WMMA indexing, rounding and rollback");
}
