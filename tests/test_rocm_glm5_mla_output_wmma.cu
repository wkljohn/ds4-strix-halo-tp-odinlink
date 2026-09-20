#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <climits>
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

static void read_f32(const std::string &path, std::vector<float> &values) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    require(fp != nullptr, path.c_str());
    require(std::fread(values.data(), sizeof(float), values.size(), fp) == values.size(),
            "complete capture file");
    require(std::fgetc(fp) == EOF && !std::ferror(fp), "exact capture size");
    require(std::fclose(fp) == 0, "close capture");
}

static void check_capture_meta(const std::string &path, uint32_t rank,
        uint32_t layer, uint32_t position, uint64_t weight_offset) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    require(fp != nullptr, "open capture metadata");
    char line[1024];
    std::unordered_map<std::string, std::string> fields;
    while (std::fgets(line, sizeof(line), fp)) {
        std::string text(line);
        require(!text.empty() && text.back() == '\n', "complete metadata line");
        text.pop_back();
        const auto split = text.find('=');
        require(split != std::string::npos && fields.emplace(
            text.substr(0, split), text.substr(split + 1)).second, "unique metadata key");
    }
    require(!std::ferror(fp) && std::fclose(fp) == 0, "read metadata");
    require(fields.size() == 13u && fields.at("schema") == "1" && fields.at("wmma") == "0" &&
            !fields.at("run_id").empty() && fields.at("rank") == std::to_string(rank) &&
            fields.at("layer") == std::to_string(layer) &&
            fields.at("pos") == std::to_string(position) && fields.at("rows") == "1024" &&
            fields.at("stride") == "16384" && fields.at("local_k") == "8192" &&
            fields.at("out_dim") == "4096" &&
            fields.at("weight_offset") == std::to_string(weight_offset) &&
            fields.at("heads_bytes") == "67108864" &&
            fields.at("output_bytes") == "16777216", "capture identity and layout");
    std::printf("capture_identity layer=%u rank=%u pos=%u run_id=%s weight_offset=%llu\n",
                layer, rank, position, fields.at("run_id").c_str(),
                (unsigned long long)weight_offset);
}

int main(int argc, char **argv) {
    uint32_t M = 256u;
    const char *capture = nullptr;
    uint32_t position = 0u;
    if (argc == 4 && std::strcmp(argv[1], "--replay") == 0) {
        require(std::strcmp(argv[3], "3072") == 0 ||
                std::strcmp(argv[3], "7168") == 0, "replay position 3072|7168");
        capture = argv[2];
        position = (uint32_t)std::strtoul(argv[3], nullptr, 10);
        M = 1024u;
    } else if (argc != 1) {
        require(argc == 3 && std::strcmp(argv[1], "--rows") == 0 &&
                (std::strcmp(argv[2], "256") == 0 ||
                 std::strcmp(argv[2], "1024") == 0),
                "usage: test_rocm_glm5_mla_output_wmma [--rows 256|1024 | --replay PREFIX 3072|7168]");
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
    std::printf("fixture rows=%u layers=%u slices=2 weights=original synthetic_activations=%u quality_admission=0\n",
                M, capture ? 3u : 11u, capture ? 0u : 1u);
    std::vector<float> host((size_t)M * FullK), gather((size_t)M * K);
    for (size_t i = 0; i < host.size(); ++i)
        host[i] = 0.2f * std::sin((double)i * 0.037) +
                  (float)((int)((i * 191u) % 997u) - 498) / 3171.0f;
    require(ds4_gpu_tensor_write(x, 0, host.data(), host.size() * sizeof(float)),
            "write input with distinct TP halves");
    std::vector<float> baseline((size_t)M * N), candidate(baseline.size()), oracle(baseline.size());
    for (uint32_t layer = 3u; layer < 45u; layer += 4u) {
        if (capture && layer != 3u && layer != 23u && layer != 43u) continue;
        uint64_t offset = 0;
        require(gguf.tensor("blk." + std::to_string(layer) + ".attn_output.weight",
                            {FullK, N}, 8u, offset), "Q8 MLA output shape");
        for (uint32_t rank : {0u, 1u}) {
            const uint32_t start = rank * K;
            std::string capture_stem;
            if (capture) {
                capture_stem = std::string(capture) + ".r" + std::to_string(rank) +
                    ".l" + std::to_string(layer) + ".p" + std::to_string(position) +
                    ".m" + std::to_string(M);
                check_capture_meta(capture_stem + ".meta", rank, layer, position, offset);
                read_f32(capture_stem + ".heads.f32", host);
                require(ds4_gpu_tensor_write(x, 0, host.data(), host.size() * sizeof(float)),
                        "upload captured full head rows");
            }
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
            double activation_max = 0, activation_round_max = 0;
            uint64_t subnormal = 0, underflow = 0, overflow = 0, nonfinite = 0;
            size_t activation_argmax = 0, first_bad = gather.size();
            for (size_t i = 0; i < gather.size(); ++i) {
                const float value = gather[i], rounded = (float)(_Float16)value;
                if (!std::isfinite(value)) ++nonfinite;
                else if (!std::isfinite(rounded)) ++overflow;
                if ((!std::isfinite(value) || !std::isfinite(rounded)) && first_bad == gather.size()) first_bad = i;
                if (std::isfinite(value) && std::fabs(value) > activation_max) {
                    activation_max = std::fabs(value);
                    activation_argmax = i;
                }
                if (std::isfinite(value) && std::isfinite(rounded))
                    activation_round_max = std::max(activation_round_max, (double)std::fabs(value - rounded));
                if (rounded != 0 && std::fabs(rounded) < 0x1p-14f) ++subnormal;
                if (value != 0 && rounded == 0) ++underflow;
            }
            std::printf("activation layer=%u rank=%u rows=%u max_abs=%g max_round_error=%g "
                        "nonfinite=%llu fp16_overflow=%llu fp16_subnormal=%llu fp16_underflow=%llu "
                        "argmax_row=%zu argmax_k=%zu first_bad_index=%llu\n",
                        layer, rank, M, activation_max, activation_round_max,
                        (unsigned long long)nonfinite, (unsigned long long)overflow,
                        (unsigned long long)subnormal, (unsigned long long)underflow,
                        activation_argmax / K, activation_argmax % K,
                        first_bad == gather.size() ? ULLONG_MAX : (unsigned long long)first_bad);
            std::fflush(stdout);
            require(nonfinite == 0 && overflow == 0, "finite original and FP16-rounded activation range");
            require(run(false, false) == 1 && ds4_gpu_synchronize() &&
                    ds4_gpu_tensor_read(out, 0, baseline.data(), OutBytes), "incumbent output");
            if (capture) {
                read_f32(capture_stem + ".output.f32", oracle);
                require(std::memcmp(baseline.data(), oracle.data(), OutBytes) == 0,
                        "replay matches captured production output exactly");
            }
            require(run(true, false) == 1 && ds4_gpu_synchronize() &&
                    ds4_gpu_tensor_read(out, 0, candidate.data(), OutBytes), "strided WMMA output");
            require(run(true, true) == 1 && ds4_gpu_synchronize() &&
                    ds4_gpu_tensor_read(out, 0, oracle.data(), OutBytes), "gathered WMMA oracle");
            require(std::memcmp(candidate.data(), oracle.data(), OutBytes) == 0,
                    "strided and gathered output bytes match");
            check_guards();
            std::vector<float> errors(candidate.size());
            double sq_error = 0, sq_reference = 0;
            size_t error_argmax = 0;
            size_t output_nonfinite = 0;
            for (size_t i = 0; i < candidate.size(); ++i)
                if (!std::isfinite(candidate[i]) || !std::isfinite(baseline[i])) ++output_nonfinite;
            std::printf("output_safety layer=%u rank=%u nonfinite=%zu\n", layer, rank, output_nonfinite);
            std::fflush(stdout);
            require(output_nonfinite == 0, "finite outputs");
            for (size_t i = 0; i < candidate.size(); ++i) {
                errors[i] = std::fabs(candidate[i] - baseline[i]);
                if (errors[i] > errors[error_argmax]) error_argmax = i;
                sq_error += (double)errors[i] * errors[i];
                sq_reference += (double)baseline[i] * baseline[i];
            }
            std::sort(errors.begin(), errors.end());
            require(errors.back() > 0, "candidate differs from F32 rollback (engagement)");
            // An analytic rounding envelope detects indexing/codec errors in
            // independent sampled CPU dots. This is not a full-model quality gate.
            double cpu_max = 0, rounded_max = 0, rounded_magnitude_ratio = 0;
            uint32_t envelope_violations = 0;
            for (uint32_t sample = 0; sample < 66; ++sample) {
                const uint32_t t = sample == 64u ? error_argmax / N :
                    sample == 65u ? activation_argmax / K : sample * 197u % M;
                const uint32_t n = sample >= 64u ? error_argmax % N : sample * 719u % N;
                const unsigned char *wr = gguf.map + offset + n * row_bytes + (start / 32u) * 34u;
                long double dot = 0, magnitude = 0, rounded_dot = 0, rounded_mag = 0;
                for (uint32_t k = 0; k < K; ++k) {
                    const unsigned char *block = wr + (k / 32u) * 34u;
                    _Float16 scale;
                    std::memcpy(&scale, block, 2);
                    const float w = (float)scale * (float)((const int8_t *)(block + 2))[k % 32u];
                    const long double product = (long double)w * gather[(size_t)t * K + k];
                    const float rw = (float)(_Float16)w;
                    const float rx = (float)(_Float16)gather[(size_t)t * K + k];
                    require(std::isfinite(rw), "finite rounded weight");
                    const long double rounded_product = (long double)rw * rx;
                    dot += product;
                    magnitude += std::fabs(product);
                    rounded_dot += rounded_product;
                    rounded_mag += std::fabs(rounded_product);
                }
                const double error = std::fabs((double)dot - candidate[(size_t)t * N + n]);
                cpu_max = std::max(cpu_max, error);
                const double rounded_error = std::fabs((double)rounded_dot - candidate[(size_t)t * N + n]);
                rounded_max = std::max(rounded_max, rounded_error);
                rounded_magnitude_ratio = std::max(rounded_magnitude_ratio,
                    rounded_error / std::max((double)rounded_mag, 1.0e-30));
                if (error > (double)magnitude * 0.003 + 1.0e-6) ++envelope_violations;
            }
            std::printf("numerical layer=%u rank=%u rows=%u exact_gather=1 max_abs=%g p99_abs=%g nmse=%.9g cpu_max=%g\n",
                        layer, rank, M, errors.back(), errors[errors.size() * 99u / 100u],
                        sq_error / std::max(sq_reference, 1.0e-30), cpu_max);
            std::printf("operands layer=%u rank=%u rows=%u production_replay=%u "
                        "max_abs=%g max_round_error=%g fp16_subnormal=%llu "
                        "fp16_underflow=%llu rounded_dot_max=%g rounded_error_over_l1=%g "
                        "envelope_violations=%u error_argmax_row=%zu error_argmax_n=%zu\n",
                        layer, rank, M, capture ? 1u : 0u, activation_max,
                        activation_round_max, (unsigned long long)subnormal,
                        (unsigned long long)underflow, rounded_max, rounded_magnitude_ratio,
                        envelope_violations, error_argmax / N, error_argmax % N);
            std::fflush(stdout);
            if (!capture) require(envelope_violations == 0,
                                  "canonical synthetic Q8 dot rounding envelope");
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
