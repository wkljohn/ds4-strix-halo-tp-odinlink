// Selected-GGUF QKV differential probe. Time includes activation preparation.
// No engine selector or weight copy. See the canonical candidate dossier.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "tests/glm5_gguf_test.hpp"
#include "rocm/ds4_rocm_bf16_toktile.cuh"

static void require(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", what); std::exit(1); }
}
static void hip_check(hipError_t error, const char *what) {
    if (error != hipSuccess) {
        std::fprintf(stderr, "FAIL %s: %s\n", what, hipGetErrorString(error));
        std::exit(1);
    }
}

// Each view registers independent read-only pages from the unchanged GGUF.
struct WeightView {
    void *mapping = nullptr;
    size_t bytes = 0;
    const uint16_t *device = nullptr;
    void bind(int fd, uint64_t offset, uint64_t count) {
        const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
        const uint64_t base = offset / page * page;
        bytes = (size_t)((offset - base + count * 2u + page - 1u) / page * page);
        mapping = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, (off_t)base);
        require(mapping != MAP_FAILED, "map original weight slice");
        hip_check(hipHostRegister(mapping, bytes, hipHostRegisterMapped), "register weights");
        void *gpu = nullptr;
        hip_check(hipHostGetDevicePointer(&gpu, mapping, 0), "weight pointer");
        device = (const uint16_t *)((const char *)gpu + offset - base);
    }
    ~WeightView() {
        if (mapping && mapping != MAP_FAILED) {
            hip_check(hipHostUnregister(mapping), "unregister weights");
            require(munmap(mapping, bytes) == 0, "unmap weights");
        }
    }
};

// Preserve the original loader's two separate stores to check packing and
// exceptional term bits independently of any resulting NaN in the MMA output.
__global__ static void reference_terms(uint16_t *high, uint16_t *low,
                                      const float *x, uint32_t count) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float xv = x[i];
    const uint16_t hi = ds4_bf16_rne_bits(xv);
    const float hi_f = __uint_as_float((uint32_t)hi << 16u);
    high[i] = hi;
    low[i] = ds4_bf16_rne_bits(xv - hi_f);
}

static void check_terms(float *x, uint32_t *panel) {
    // Ties, signed zero, subnormal boundaries, overflow and signed NaN payloads.
    const uint32_t bits[] = {
        0u, 0x80000000u, 1u, 0x80000001u, 0x007fffffu, 0x00800000u,
        0x3f807fffu, 0x3f808000u, 0x3f808001u, 0x3f818000u,
        0xbf808000u, 0xbf818000u, 0x7f7fffffu, 0xff7fffffu,
        0x7f800000u, 0xff800000u, 0x7fc12345u, 0xffc12345u,
        0x7f800001u, 0xff800001u, 0x00008000u, 0x80008000u
    };
    constexpr uint32_t count = sizeof(bits) / sizeof(bits[0]);
    uint16_t *terms = nullptr;
    hip_check(hipMalloc(&terms, 2u * count * sizeof(uint16_t)), "term scratch");
    hip_check(hipMemcpy(x, bits, sizeof(bits), hipMemcpyHostToDevice), "edge input");
    reference_terms<<<1, 256>>>(terms, terms + count, x, count);
    ds4_bf16_hilo_prepare_kernel<<<1, 256>>>(panel, x, count);
    hip_check(hipGetLastError(), "term launches");
    std::vector<uint16_t> expected(2u * count);
    std::vector<uint32_t> packed(count);
    hip_check(hipMemcpy(expected.data(), terms, expected.size() * 2u,
                        hipMemcpyDeviceToHost), "reference terms");
    hip_check(hipMemcpy(packed.data(), panel, packed.size() * 4u,
                        hipMemcpyDeviceToHost), "packed terms");
    for (uint32_t i = 0; i < count; ++i)
        require((uint16_t)packed[i] == expected[i] &&
                (uint16_t)(packed[i] >> 16u) == expected[count + i],
                "exact exceptional term bits");
    hip_check(hipFree(terms), "free term scratch");
    std::printf("PASS exceptional packed terms count=%u\n", count);
}

int main() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    require(model && *model, "DS4_GLM5_MODEL required");
    Glm5TestGGUF gguf;
    require(gguf.open_file(model), "open selected GGUF");
    // Validate every mapped tensor's shape/type below. Container byte size
    // also reflects chat metadata and cannot identify compatible weights.
    hipDeviceProp_t props{};
    hip_check(hipGetDeviceProperties(&props, 0), "device properties");
    require(std::strstr(props.gcnArchName, "gfx1151") && props.warpSize == 32,
            "gfx1151 wave32 required");
    constexpr uint32_t M = 256, K = 4096, N = 4096;
    constexpr size_t values = (size_t)M * K, outputs = (size_t)3 * M * N;
    constexpr size_t padded_values = values + (size_t)(K / 16u) * 16u;
    float *x = nullptr, *out[4] = {};
    uint32_t *panel = nullptr;
    hip_check(hipMalloc(&x, values * 4u), "input scratch");
    hip_check(hipMalloc(&panel, padded_values * 4u), "activation panel");
    for (auto &p : out) hip_check(hipMalloc(&p, outputs * 4u), "output scratch");
    check_terms(x, panel);
    std::vector<float> input(values), reference(outputs), candidate(outputs);
    hipEvent_t begin, end;
    hip_check(hipEventCreate(&begin), "begin event");
    hip_check(hipEventCreate(&end), "end event");
    const char *names[] = {"kda_q.weight", "kda_k.weight", "kda_v.weight"};
    std::printf("workload=selected-GGUF-QKV model=%s model_bytes=%llu M=256 K=4096 N=4096 inputs=changing-synthetic\n",
                model, (unsigned long long)gguf.size);
    std::puts("arm=0 raw-QKV arm=1 prepare-row-major+QKV arm=2 prepare-tiled+QKV scratch_bytes=4194304");
    std::puts("arm=3 prepare-padded-tiled+QKV scratch_bytes=4210688");
    for (uint32_t layer : {0u, 4u, 20u, 44u}) {
        for (uint32_t rank : {0u, 1u}) {
            WeightView weights[3];
            for (uint32_t p = 0; p < 3; ++p) {
                uint64_t offset = 0;
                require(gguf.tensor("blk." + std::to_string(layer) + "." + names[p],
                                    {K, 2u * N}, 30u, offset), "BF16 QKV geometry");
                weights[p].bind(gguf.fd, offset + (uint64_t)rank * K * N * 2u,
                                (uint64_t)K * N);
            }
            auto launch = [&](uint32_t arm, uint32_t component = 0u) {
                if (arm == 0u)
                    matmul_bf16_f32_wmma_hilo_qkv_multiptr_kernel<<<
                        dim3(3u * (N / 32u), 1), 512>>>(
                        out[arm], out[arm] + M * N, out[arm] + 2u * M * N,
                        weights[0].device, weights[1].device, weights[2].device,
                        x, K, N, M);
                else if (arm == 1u) {
                    if (component != 2u)
                    ds4_bf16_hilo_prepare_kernel<<<(values + 255u) / 256u, 256>>>(panel, x, values);
                    if (component != 1u)
                    matmul_bf16_f32_wmma_hilo_qkv_multiptr_kernel<true><<<
                        dim3(3u * (N / 32u), 1), 512>>>(
                        out[arm], out[arm] + M * N, out[arm] + 2u * M * N,
                        weights[0].device, weights[1].device, weights[2].device,
                        x, K, N, M, panel);
                } else if (arm == 2u) {
                    if (component != 2u)
                    ds4_bf16_hilo_prepare_kernel<true><<<
                        (values + 255u) / 256u, 256>>>(panel, x, values, K);
                    if (component != 1u)
                    matmul_bf16_f32_wmma_hilo_qkv_multiptr_kernel<2u><<<
                        dim3(3u * (N / 32u), 1), 512>>>(
                        out[arm], out[arm] + M * N, out[arm] + 2u * M * N,
                        weights[0].device, weights[1].device, weights[2].device,
                        x, K, N, M, panel);
                } else {
                    if (component != 2u)
                    ds4_bf16_hilo_prepare_kernel<true, 16u><<<
                        (padded_values + 255u) / 256u, 256>>>(panel, x, padded_values, K);
                    if (component != 1u)
                    matmul_bf16_f32_wmma_hilo_qkv_multiptr_kernel<3u><<<
                        dim3(3u * (N / 32u), 1), 512>>>(
                        out[arm], out[arm] + M * N, out[arm] + 2u * M * N,
                        weights[0].device, weights[1].device, weights[2].device,
                        x, K, N, M, panel);
                }
                hip_check(hipGetLastError(), "QKV launch");
            };
            for (uint32_t pattern = 0; pattern < 3; ++pattern) {
                for (size_t i = 0; i < values; ++i)
                    input[i] = (pattern == 1u ? 2.3f : 0.23f) *
                        std::cos((double)((i + 37u * layer + rank) % 8191u) * 0.009) -
                        0.08f * std::sin((double)(i + pattern * 131u) * 0.004);
                // A separate full-output exceptional test. Each affected row
                // is compared bitwise, including the resulting NaN payload.
                if (pattern == 2u) {
                    const uint32_t exceptional[] = {0x80000000u, 0x7f800000u,
                        0xff800000u, 0x7fc12345u, 0x7f7fffffu, 1u};
                    for (size_t i = 0; i < sizeof(exceptional) / 4u; ++i)
                        std::memcpy(&input[i * K + 17u], &exceptional[i], 4u);
                }
                hip_check(hipMemcpy(x, input.data(), values * 4u, hipMemcpyHostToDevice), "changing input");
                // Also verify the templated raw path against the independent,
                // unchanged single-projection kernel used before grid fusion.
                for (uint32_t p = 0; p < 3; ++p)
                    matmul_bf16_f32_wmma_hilo_m256_kernel<2u><<<
                        dim3(N / 32u, 1), 512>>>(
                        out[1] + p * M * N, weights[p].device, x, K, N, M);
                hip_check(hipGetLastError(), "legacy sequential launch");
                launch(0);
                hip_check(hipMemcpy(reference.data(), out[0], outputs * 4u, hipMemcpyDeviceToHost), "raw output");
                hip_check(hipMemcpy(candidate.data(), out[1], outputs * 4u, hipMemcpyDeviceToHost), "legacy output");
                require(std::memcmp(reference.data(), candidate.data(), outputs * 4u) == 0,
                        "raw QKV matches unchanged sequential arithmetic");
                for (uint32_t arm = 1u; arm < 4u; ++arm) {
                    launch(arm);
                    hip_check(hipMemcpy(candidate.data(), out[arm], outputs * 4u, hipMemcpyDeviceToHost), "prepared output");
                    size_t mismatches = 0;
                    for (size_t i = 0; i < outputs; ++i) {
                        mismatches += std::memcmp(&reference[i], &candidate[i], 4u) != 0;
                        if (pattern != 2u) require(std::isfinite(reference[i]) &&
                                                   std::isfinite(candidate[i]), "finite output");
                    }
                    std::printf("compare layer=%u rank=%u pattern=%u arm=%u mismatches=%zu\n",
                                layer, rank, pattern, arm, mismatches);
                    require(mismatches == 0u, "complete QKV bit equality");
                }
                if (pattern != 0u) continue;
                std::vector<float> samples[4];
                for (uint32_t sample = 0; sample < 9; ++sample) {
                    for (uint32_t order = 0; order < 4; ++order) {
                        const uint32_t arm = (sample + order) % 4u;
                        hip_check(hipEventRecord(begin), "timing begin");
                        for (uint32_t repeat = 0; repeat < 3; ++repeat) launch(arm);
                        hip_check(hipEventRecord(end), "timing end");
                        hip_check(hipEventSynchronize(end), "timing wait");
                        float ms = 0;
                        hip_check(hipEventElapsedTime(&ms, begin, end), "elapsed time");
                        samples[arm].push_back(ms / 3.0f);
                        std::printf("sample layer=%u rank=%u repeat=%u arm=%u ms=%.6f\n",
                                    layer, rank, sample, arm, ms / 3.0f);
                    }
                }
                for (uint32_t arm = 0; arm < 4; ++arm) {
                    std::sort(samples[arm].begin(), samples[arm].end());
                    std::printf("timing layer=%u rank=%u arm=%u median_ms=%.6f min_ms=%.6f max_ms=%.6f\n",
                                layer, rank, arm, samples[arm][4], samples[arm][0], samples[arm][8]);
                }
                // Attribution only: these component times never substitute
                // for the complete preparation-plus-consumer measurement.
                for (uint32_t arm = 1; arm < 4; ++arm) {
                    launch(arm);
                    for (uint32_t component = 1; component < 3; ++component) {
                        hip_check(hipEventRecord(begin), "component begin");
                        for (uint32_t repeat = 0; repeat < 5; ++repeat)
                            launch(arm, component);
                        hip_check(hipEventRecord(end), "component end");
                        hip_check(hipEventSynchronize(end), "component wait");
                        float ms = 0;
                        hip_check(hipEventElapsedTime(&ms, begin, end), "component elapsed");
                        std::printf("component layer=%u rank=%u arm=%u part=%s mean_ms=%.6f\n",
                                    layer, rank, arm,
                                    component == 1u ? "prepare" : "QKV", ms / 5.0f);
                    }
                }
                std::fflush(stdout);
            }
        }
    }
    hip_check(hipEventDestroy(begin), "destroy begin");
    hip_check(hipEventDestroy(end), "destroy end");
    for (auto p : out) hip_check(hipFree(p), "free output");
    hip_check(hipFree(panel), "free activation panel");
    hip_check(hipFree(x), "free input");
    std::puts("PASS original-GGUF activation panel differential");
}
