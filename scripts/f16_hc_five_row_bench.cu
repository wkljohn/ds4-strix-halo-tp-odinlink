// Shape-accurate oracle for the F16 HC control projection used by the
// five-row DeepSeek V4 Flash DSpark verifier.
//
// Build:
//   /opt/rocm/bin/hipcc -O3 -ffast-math -fno-finite-math-only \
//     --offload-arch=gfx1151 scripts/f16_hc_five_row_bench.cu \
//     -lhipblas -o /tmp/f16_hc_five_row_bench

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
static constexpr uint32_t kInput = 16384;
static constexpr uint32_t kOutput = 24;
static constexpr uint32_t kThreads = 256;
static constexpr uint32_t kWarps = kThreads / 32;
static constexpr uint32_t kWeightSets = 64;

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

static __device__ __forceinline__ float warp_sum(float v) {
    for (uint32_t delta = 16; delta != 0; delta >>= 1u) {
        v += __shfl_down(v, delta, 32);
    }
    return v;
}

// One output row per block. All 256 threads cooperate on K, so the exact
// 16384x24 shape launches 24 blocks rather than only three blocks with the
// compressor kernel's eight-output-row layout. Each F16 weight is loaded once
// and reused across all five activation rows.
template <uint32_t TOKENS>
__global__ static void f16_hc_five_row_kernel(
        float *out,
        const __half *w,
        const __half *x,
        uint32_t in_dim,
        uint32_t out_dim) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (row >= out_dim) return;

    float acc[TOKENS] = {0.0f};
    const __half *wr = w + (uint64_t)row * in_dim;
    for (uint32_t k = tid; k < in_dim; k += blockDim.x) {
        const float fw = __half2float(wr[k]);
#pragma unroll
        for (uint32_t t = 0; t < TOKENS; ++t) {
            acc[t] = fmaf(fw, __half2float(x[(uint64_t)t * in_dim + k]), acc[t]);
        }
    }

    __shared__ float partial[TOKENS][kWarps];
#pragma unroll
    for (uint32_t t = 0; t < TOKENS; ++t) {
        acc[t] = warp_sum(acc[t]);
        if (lane == 0) partial[t][warp] = acc[t];
    }
    __syncthreads();

    if (warp == 0) {
#pragma unroll
        for (uint32_t t = 0; t < TOKENS; ++t) {
            float v = lane < kWarps ? partial[t][lane] : 0.0f;
            v = warp_sum(v);
            if (lane == 0) out[(uint64_t)t * out_dim + row] = v;
        }
    }
}

// Two neighboring output rows per block reuse each activation load. This
// halves the block count and raises register pressure, so the oracle measures
// it independently instead of assuming the trade is favorable on gfx1151.
template <uint32_t TOKENS>
__global__ static void f16_hc_pair_rows_five_row_kernel(
        float *out,
        const __half *w,
        const __half *x,
        uint32_t in_dim,
        uint32_t out_dim) {
    const uint32_t row0 = blockIdx.x * 2u;
    const uint32_t row1 = row0 + 1u;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (row0 >= out_dim) return;

    float acc0[TOKENS] = {0.0f};
    float acc1[TOKENS] = {0.0f};
    const __half *wr0 = w + (uint64_t)row0 * in_dim;
    const __half *wr1 = w + (uint64_t)row1 * in_dim;
    for (uint32_t k = tid; k < in_dim; k += blockDim.x) {
        const float fw0 = __half2float(wr0[k]);
        const float fw1 = row1 < out_dim ? __half2float(wr1[k]) : 0.0f;
#pragma unroll
        for (uint32_t t = 0; t < TOKENS; ++t) {
            const float xv = __half2float(x[(uint64_t)t * in_dim + k]);
            acc0[t] = fmaf(fw0, xv, acc0[t]);
            acc1[t] = fmaf(fw1, xv, acc1[t]);
        }
    }

    __shared__ float partial[2][TOKENS][kWarps];
#pragma unroll
    for (uint32_t t = 0; t < TOKENS; ++t) {
        acc0[t] = warp_sum(acc0[t]);
        acc1[t] = warp_sum(acc1[t]);
        if (lane == 0) {
            partial[0][t][warp] = acc0[t];
            partial[1][t][warp] = acc1[t];
        }
    }
    __syncthreads();

    if (warp == 0) {
#pragma unroll
        for (uint32_t t = 0; t < TOKENS; ++t) {
            float v0 = lane < kWarps ? partial[0][t][lane] : 0.0f;
            float v1 = lane < kWarps ? partial[1][t][lane] : 0.0f;
            v0 = warp_sum(v0);
            v1 = warp_sum(v1);
            if (lane == 0) {
                out[(uint64_t)t * out_dim + row0] = v0;
                if (row1 < out_dim) out[(uint64_t)t * out_dim + row1] = v1;
            }
        }
    }
}

// Control alternatives: preserve one independent reduction per token/output
// while removing hipBLAS setup. They reread the F16 weights five times, but
// may track hipBLAS numerics more closely than cross-token reuse.
__global__ static void f16_hc_token_block_kernel(
        float *out, const __half *w, const __half *x,
        uint32_t in_dim, uint32_t out_dim) {
    const uint32_t row = blockIdx.x;
    const uint32_t tok = blockIdx.y;
    float sum = 0.0f;
    const __half *wr = w + (uint64_t)row * in_dim;
    const __half *xr = x + (uint64_t)tok * in_dim;
    for (uint32_t k = threadIdx.x; k < in_dim; k += blockDim.x) {
        sum = fmaf(__half2float(wr[k]), __half2float(xr[k]), sum);
    }
    __shared__ float partial[kThreads];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride != 0; stride >>= 1u) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[(uint64_t)tok * out_dim + row] = partial[0];
}

__global__ static void f16_hc_token_ordered32_kernel(
        float *out, const __half *w, const __half *x,
        uint32_t in_dim, uint32_t out_dim) {
    const uint32_t row = blockIdx.x;
    const uint32_t tok = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t chunk = (in_dim + 31u) / 32u;
    const uint32_t k0 = tid * chunk;
    const uint32_t k1 = min(k0 + chunk, in_dim);
    const __half *wr = w + (uint64_t)row * in_dim;
    const __half *xr = x + (uint64_t)tok * in_dim;
    float sum = 0.0f;
    for (uint32_t k = k0; k < k1; ++k) {
        sum = fmaf(__half2float(wr[k]), __half2float(xr[k]), sum);
    }
    __shared__ float partial[32];
    partial[tid] = sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (uint32_t i = 0; i < 32u; ++i) total += partial[i];
        out[(uint64_t)tok * out_dim + row] = total;
    }
}

struct bench {
    hipblasHandle_t handle{};
    std::vector<__half> w;
    std::vector<float> x;
    __half *dw{}, *dxh{};
    float *dx{}, *base{}, *single{}, *paired{}, *token256{}, *ordered32{};

    bench() : w((uint64_t)kWeightSets * kOutput * kInput),
              x((uint64_t)kTokens * kInput) {
        std::mt19937 gen(71);
        std::uniform_real_distribution<float> dist(-0.08f, 0.08f);
        for (auto &v : w) v = __float2half(dist(gen));
        for (auto &v : x) v = dist(gen);
        const size_t wb = w.size() * sizeof(__half);
        const size_t xb = x.size() * sizeof(float);
        const size_t xhb = x.size() * sizeof(__half);
        const size_t ob = (size_t)kTokens * kOutput * sizeof(float);
        check(hipMalloc(&dw, wb), "malloc weights");
        check(hipMalloc(&dx, xb), "malloc x");
        check(hipMalloc(&dxh, xhb), "malloc xh");
        check(hipMalloc(&base, ob), "malloc base");
        check(hipMalloc(&single, ob), "malloc single");
        check(hipMalloc(&paired, ob), "malloc paired");
        check(hipMalloc(&token256, ob), "malloc token256");
        check(hipMalloc(&ordered32, ob), "malloc ordered32");
        check(hipMemcpy(dw, w.data(), wb, hipMemcpyHostToDevice), "copy weights");
        check(hipMemcpy(dx, x.data(), xb, hipMemcpyHostToDevice), "copy x");
        check_blas(hipblasCreate(&handle), "hipblasCreate");
    }

    ~bench() {
        hipblasDestroy(handle);
        (void)hipFree(dw); (void)hipFree(dx); (void)hipFree(dxh);
        (void)hipFree(base); (void)hipFree(single); (void)hipFree(paired);
        (void)hipFree(token256); (void)hipFree(ordered32);
    }

    const __half *weights(uint32_t set) const {
        return dw + (uint64_t)(set % kWeightSets) * kOutput * kInput;
    }

    void baseline(uint32_t set) {
        const uint64_t xn = (uint64_t)kTokens * kInput;
        const float alpha = 1.0f, beta = 0.0f;
        f32_to_f16<<<(xn + 255u) / 256u, 256>>>(dxh, dx, xn);
        check_blas(hipblasGemmEx(handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                                 kOutput, kTokens, kInput,
                                 &alpha, weights(set), HIPBLAS_R_16F, kInput,
                                 dxh, HIPBLAS_R_16F, kInput,
                                 &beta, base, HIPBLAS_R_32F, kOutput,
                                 HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT), "gemm");
    }

    void candidate_single(uint32_t set) {
        const uint64_t xn = (uint64_t)kTokens * kInput;
        f32_to_f16<<<(xn + 255u) / 256u, 256>>>(dxh, dx, xn);
        f16_hc_five_row_kernel<kTokens><<<kOutput, kThreads>>>(
                single, weights(set), dxh, kInput, kOutput);
    }

    void candidate_paired(uint32_t set) {
        const uint64_t xn = (uint64_t)kTokens * kInput;
        f32_to_f16<<<(xn + 255u) / 256u, 256>>>(dxh, dx, xn);
        f16_hc_pair_rows_five_row_kernel<kTokens><<<(kOutput + 1u) / 2u, kThreads>>>(
                paired, weights(set), dxh, kInput, kOutput);
    }

    void candidate_token256(uint32_t set) {
        const uint64_t xn = (uint64_t)kTokens * kInput;
        f32_to_f16<<<(xn + 255u) / 256u, 256>>>(dxh, dx, xn);
        f16_hc_token_block_kernel<<<dim3(kOutput, kTokens), kThreads>>>(
                token256, weights(set), dxh, kInput, kOutput);
    }

    void candidate_ordered32(uint32_t set) {
        const uint64_t xn = (uint64_t)kTokens * kInput;
        f32_to_f16<<<(xn + 255u) / 256u, 256>>>(dxh, dx, xn);
        f16_hc_token_ordered32_kernel<<<dim3(kOutput, kTokens), 32>>>(
                ordered32, weights(set), dxh, kInput, kOutput);
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
    (void)hipEventDestroy(begin);
    (void)hipEventDestroy(end);
    return elapsed / iterations;
}

struct error_stats { double max_abs{}, max_rel{}, rms{}; };

static error_stats compare(const std::vector<float> &ref,
                           const std::vector<float> &got) {
    error_stats s{};
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = std::fabs((double)ref[i] - got[i]);
        s.max_abs = std::max(s.max_abs, d);
        s.max_rel = std::max(s.max_rel, d / std::max(1e-6, std::fabs((double)ref[i])));
        s.rms += d * d;
    }
    s.rms = std::sqrt(s.rms / ref.size());
    return s;
}

int main() {
    bench b;
    b.baseline(0);
    b.candidate_single(0);
    b.candidate_paired(0);
    b.candidate_token256(0);
    b.candidate_ordered32(0);
    check(hipDeviceSynchronize(), "correctness sync");

    const size_t n = (size_t)kTokens * kOutput;
    std::vector<float> base(n), single(n), paired(n), token256(n), ordered32(n);
    check(hipMemcpy(base.data(), b.base, n * sizeof(float), hipMemcpyDeviceToHost), "read base");
    check(hipMemcpy(single.data(), b.single, n * sizeof(float), hipMemcpyDeviceToHost), "read single");
    check(hipMemcpy(paired.data(), b.paired, n * sizeof(float), hipMemcpyDeviceToHost), "read paired");
    check(hipMemcpy(token256.data(), b.token256, n * sizeof(float), hipMemcpyDeviceToHost), "read token256");
    check(hipMemcpy(ordered32.data(), b.ordered32, n * sizeof(float), hipMemcpyDeviceToHost), "read ordered32");
    const error_stats es = compare(base, single);
    const error_stats ep = compare(base, paired);
    const error_stats et = compare(base, token256);
    const error_stats eo = compare(base, ordered32);

    const float base_ms = time_ms([&](int i) { b.baseline((uint32_t)i); }, 256);
    const float single_ms = time_ms([&](int i) { b.candidate_single((uint32_t)i); }, 256);
    const float paired_ms = time_ms([&](int i) { b.candidate_paired((uint32_t)i); }, 256);
    const float token256_ms = time_ms([&](int i) { b.candidate_token256((uint32_t)i); }, 256);
    const float ordered32_ms = time_ms([&](int i) { b.candidate_ordered32((uint32_t)i); }, 256);
    std::printf("shape tokens=%u input=%u output=%u weight_sets=%u working_set_mib=%.1f\n",
                kTokens, kInput, kOutput, kWeightSets,
                (double)kWeightSets * kOutput * kInput * sizeof(__half) / 1048576.0);
    std::printf("single correctness max_abs=%.9g max_rel=%.9g rms=%.9g\n",
                es.max_abs, es.max_rel, es.rms);
    std::printf("paired correctness max_abs=%.9g max_rel=%.9g rms=%.9g\n",
                ep.max_abs, ep.max_rel, ep.rms);
    std::printf("token256 correctness max_abs=%.9g max_rel=%.9g rms=%.9g\n",
                et.max_abs, et.max_rel, et.rms);
    std::printf("ordered32 correctness max_abs=%.9g max_rel=%.9g rms=%.9g\n",
                eo.max_abs, eo.max_rel, eo.rms);
    std::printf("baseline_ms=%.4f single_ms=%.4f change=%+.1f%% paired_ms=%.4f change=%+.1f%%\n",
                base_ms, single_ms, 100.0f * (single_ms / base_ms - 1.0f),
                paired_ms, 100.0f * (paired_ms / base_ms - 1.0f));
    std::printf("token256_ms=%.4f change=%+.1f%% ordered32_ms=%.4f change=%+.1f%%\n",
                token256_ms, 100.0f * (token256_ms / base_ms - 1.0f),
                ordered32_ms, 100.0f * (ordered32_ms / base_ms - 1.0f));
    return (es.max_abs > 2e-3 || ep.max_abs > 2e-3 ||
            et.max_abs > 2e-3 || eo.max_abs > 2e-3) ? 2 : 0;
}
