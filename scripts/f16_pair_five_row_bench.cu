// Small-step oracle for the paired F16 compressor projections used by
// DeepSeek V4 Flash's five-row DSpark verifier. It rotates through multiple
// weight pairs to avoid measuring only a cache-hot layer.
//
// Build:
//   /opt/rocm/bin/hipcc -O3 -ffast-math -fno-finite-math-only \
//     --offload-arch=gfx1151 scripts/f16_pair_five_row_bench.cu \
//     -lhipblas -o /tmp/f16_pair_five_row_bench

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static constexpr uint32_t kTokens = 5;
static constexpr uint32_t kInput = 4096;
#ifndef DS4_BENCH_OUTPUT
#define DS4_BENCH_OUTPUT 256
#endif
static constexpr uint32_t kOutput = DS4_BENCH_OUTPUT;
#ifndef DS4_BENCH_WEIGHT_SETS
#define DS4_BENCH_WEIGHT_SETS 16
#endif
static constexpr uint32_t kWeightSets = DS4_BENCH_WEIGHT_SETS;

static void check(hipError_t rc, const char *what) {
    if (rc != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(rc));
        std::exit(1);
    }
}

static void check_blas(hipblasStatus_t rc, const char *what) {
    if (rc != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: hipBLAS status %d\n", what, (int)rc);
        std::exit(1);
    }
}

__global__ static void f32_to_f16(__half *dst, const float *src, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

template <uint32_t TOKENS>
__global__ static void f16_pair_five_row_kernel(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const __half *x,
        uint32_t in_dim,
        uint32_t out_dim) {
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t row = blockIdx.x * rows_per_block + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;

    float a0[TOKENS];
    float a1[TOKENS];
#pragma unroll
    for (uint32_t t = 0; t < TOKENS; ++t) {
        a0[t] = 0.0f;
        a1[t] = 0.0f;
    }
    const __half *wr0 = w0 + (uint64_t)row * in_dim;
    const __half *wr1 = w1 + (uint64_t)row * in_dim;
    for (uint32_t k = lane; k < in_dim; k += 32u) {
        const float fw0 = __half2float(wr0[k]);
        const float fw1 = __half2float(wr1[k]);
#pragma unroll
        for (uint32_t t = 0; t < TOKENS; ++t) {
            const float xv = __half2float(x[(uint64_t)t * in_dim + k]);
            a0[t] = fmaf(fw0, xv, a0[t]);
            a1[t] = fmaf(fw1, xv, a1[t]);
        }
    }
#pragma unroll
    for (uint32_t t = 0; t < TOKENS; ++t) {
        for (uint32_t delta = 16; delta != 0; delta >>= 1u) {
            a0[t] += __shfl_down(a0[t], delta, 32);
            a1[t] += __shfl_down(a1[t], delta, 32);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (uint32_t t = 0; t < TOKENS; ++t) {
            out0[(uint64_t)t * out_dim + row] = a0[t];
            out1[(uint64_t)t * out_dim + row] = a1[t];
        }
    }
}

struct bench {
    hipblasHandle_t handle{};
    std::vector<__half> w0, w1;
    std::vector<float> x;
    __half *dw0{}, *dw1{}, *dxh{};
    float *dx{}, *base0{}, *base1{}, *cand0{}, *cand1{};

    bench() : w0((uint64_t)kWeightSets * kOutput * kInput),
              w1((uint64_t)kWeightSets * kOutput * kInput),
              x((uint64_t)kTokens * kInput) {
        std::mt19937 gen(17);
        std::uniform_real_distribution<float> dist(-0.08f, 0.08f);
        for (auto &v : w0) v = __float2half(dist(gen));
        for (auto &v : w1) v = __float2half(dist(gen));
        for (auto &v : x) v = dist(gen);
        const size_t wb = w0.size() * sizeof(__half);
        const size_t xb = x.size() * sizeof(float);
        const size_t xhb = x.size() * sizeof(__half);
        const size_t ob = (size_t)kTokens * kOutput * sizeof(float);
        check(hipMalloc(&dw0, wb), "malloc w0");
        check(hipMalloc(&dw1, wb), "malloc w1");
        check(hipMalloc(&dx, xb), "malloc x");
        check(hipMalloc(&dxh, xhb), "malloc xh");
        check(hipMalloc(&base0, ob), "malloc base0");
        check(hipMalloc(&base1, ob), "malloc base1");
        check(hipMalloc(&cand0, ob), "malloc cand0");
        check(hipMalloc(&cand1, ob), "malloc cand1");
        check(hipMemcpy(dw0, w0.data(), wb, hipMemcpyHostToDevice), "copy w0");
        check(hipMemcpy(dw1, w1.data(), wb, hipMemcpyHostToDevice), "copy w1");
        check(hipMemcpy(dx, x.data(), xb, hipMemcpyHostToDevice), "copy x");
        check_blas(hipblasCreate(&handle), "hipblasCreate");
    }

    ~bench() {
        hipblasDestroy(handle);
        (void)hipFree(dw0); (void)hipFree(dw1); (void)hipFree(dx); (void)hipFree(dxh);
        (void)hipFree(base0); (void)hipFree(base1); (void)hipFree(cand0); (void)hipFree(cand1);
    }

    void baseline(uint32_t set) {
        const uint64_t xn = (uint64_t)kTokens * kInput;
        const uint64_t weight_stride = (uint64_t)kOutput * kInput;
        const __half *sw0 = dw0 + (uint64_t)(set % kWeightSets) * weight_stride;
        const __half *sw1 = dw1 + (uint64_t)(set % kWeightSets) * weight_stride;
        const float alpha = 1.0f, beta = 0.0f;
        f32_to_f16<<<(xn + 255u) / 256u, 256>>>(dxh, dx, xn);
        check_blas(hipblasGemmEx(handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                                 kOutput, kTokens, kInput,
                                 &alpha, sw0, HIPBLAS_R_16F, kInput,
                                 dxh, HIPBLAS_R_16F, kInput,
                                 &beta, base0, HIPBLAS_R_32F, kOutput,
                                 HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT), "gemm0");
        f32_to_f16<<<(xn + 255u) / 256u, 256>>>(dxh, dx, xn);
        check_blas(hipblasGemmEx(handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                                 kOutput, kTokens, kInput,
                                 &alpha, sw1, HIPBLAS_R_16F, kInput,
                                 dxh, HIPBLAS_R_16F, kInput,
                                 &beta, base1, HIPBLAS_R_32F, kOutput,
                                 HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT), "gemm1");
    }

    void candidate(uint32_t set) {
        const uint64_t xn = (uint64_t)kTokens * kInput;
        const uint64_t weight_stride = (uint64_t)kOutput * kInput;
        const __half *sw0 = dw0 + (uint64_t)(set % kWeightSets) * weight_stride;
        const __half *sw1 = dw1 + (uint64_t)(set % kWeightSets) * weight_stride;
        f32_to_f16<<<(xn + 255u) / 256u, 256>>>(dxh, dx, xn);
        f16_pair_five_row_kernel<kTokens><<<(kOutput + 7u) / 8u, 256>>>(
                cand0, cand1, sw0, sw1, dxh, kInput, kOutput);
    }
};

template <typename F>
static float time_ms(F fn, int iterations) {
    hipEvent_t begin{}, end{};
    check(hipEventCreate(&begin), "event begin");
    check(hipEventCreate(&end), "event end");
    for (int i = 0; i < 3; ++i) fn(i);
    check(hipDeviceSynchronize(), "warm sync");
    check(hipEventRecord(begin), "record begin");
    for (int i = 0; i < iterations; ++i) fn(i);
    check(hipEventRecord(end), "record end");
    check(hipEventSynchronize(end), "sync end");
    float elapsed = 0.0f;
    check(hipEventElapsedTime(&elapsed, begin, end), "elapsed");
    (void)hipEventDestroy(begin); (void)hipEventDestroy(end);
    return elapsed / iterations;
}

int main() {
    bench b;
    b.baseline(0); b.candidate(0);
    check(hipDeviceSynchronize(), "correctness sync");
    const size_t n = (size_t)kTokens * kOutput;
    std::vector<float> base0(n), base1(n), cand0(n), cand1(n);
    check(hipMemcpy(base0.data(), b.base0, n * sizeof(float), hipMemcpyDeviceToHost), "read base0");
    check(hipMemcpy(base1.data(), b.base1, n * sizeof(float), hipMemcpyDeviceToHost), "read base1");
    check(hipMemcpy(cand0.data(), b.cand0, n * sizeof(float), hipMemcpyDeviceToHost), "read cand0");
    check(hipMemcpy(cand1.data(), b.cand1, n * sizeof(float), hipMemcpyDeviceToHost), "read cand1");
    double max_abs = 0.0, max_rel = 0.0, rms = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (int p = 0; p < 2; ++p) {
            const double ref = p ? base1[i] : base0[i];
            const double got = p ? cand1[i] : cand0[i];
            const double d = std::fabs(ref - got);
            max_abs = std::max(max_abs, d);
            max_rel = std::max(max_rel, d / std::max(1e-6, std::fabs(ref)));
            rms += d * d;
        }
    }
    rms = std::sqrt(rms / (2.0 * n));
    const float base_ms = time_ms([&](int i) { b.baseline((uint32_t)i); }, 100);
    const float cand_ms = time_ms([&](int i) { b.candidate((uint32_t)i); }, 100);
    std::printf("shape tokens=%u input=%u output_pair=%u weight_sets=%u working_set_mib=%.1f\n",
                kTokens, kInput, kOutput, kWeightSets,
                (double)2u * kWeightSets * kOutput * kInput * sizeof(__half) / 1048576.0);
    std::printf("correctness max_abs=%.9g max_rel=%.9g rms=%.9g\n",
                max_abs, max_rel, rms);
    std::printf("baseline_ms=%.4f candidate_ms=%.4f change=%+.1f%%\n",
                base_ms, cand_ms, 100.0f * (cand_ms / base_ms - 1.0f));
    return max_abs > 2e-3 ? 2 : 0;
}
