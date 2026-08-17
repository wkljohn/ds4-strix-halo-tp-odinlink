// Exact-arithmetic shape oracle for temporally batching ordinary one-token
// F16 compressor projections. Four independent FP32 activation rows reuse
// each F16 weight while retaining the production lane and reduction order.
//
// Build all production widths, for example:
//   hipcc -O3 -ffast-math -fno-finite-math-only --offload-arch=gfx1151 \
//     -DDS4_BENCH_OUTPUT=1024 scripts/f16_pair_temporal_batch_bench.cu \
//     -o /tmp/f16_pair_temporal_batch_bench

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static constexpr uint32_t kInput = 4096;
static constexpr uint32_t kTokens = 4;
#ifndef DS4_BENCH_OUTPUT
#define DS4_BENCH_OUTPUT 1024
#endif
static constexpr uint32_t kOutput = DS4_BENCH_OUTPUT;
static constexpr uint32_t kWeightSets = 8;

static void check(hipError_t rc, const char *what) {
    if (rc != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(rc));
        std::exit(1);
    }
}

__device__ static float warp_sum(float v) {
    for (uint32_t delta = 16u; delta != 0u; delta >>= 1u) {
        v += __shfl_down(v, delta, 32);
    }
    return v;
}

// Byte-for-byte copy of the production arithmetic structure for one token.
__global__ static void pair_one_token(
        float *out0, float *out1, const __half *w0, const __half *w1,
        const float *x) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    for (uint32_t i = tid; i < kInput; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    const uint32_t row = blockIdx.x * rows_per_block + wave;
    if (row >= kOutput) return;
    const __half *wr0 = w0 + (uint64_t)row * kInput;
    const __half *wr1 = w1 + (uint64_t)row * kInput;
    float acc0 = 0.0f, acc1 = 0.0f;
    uint32_t i = lane;
    for (; i + 224u < kInput; i += 256u) {
        float xv = shx[i];
        acc0 += __half2float(wr0[i]) * xv;
        acc1 += __half2float(wr1[i]) * xv;
        xv = shx[i + 32u];
        acc0 += __half2float(wr0[i + 32u]) * xv;
        acc1 += __half2float(wr1[i + 32u]) * xv;
        xv = shx[i + 64u];
        acc0 += __half2float(wr0[i + 64u]) * xv;
        acc1 += __half2float(wr1[i + 64u]) * xv;
        xv = shx[i + 96u];
        acc0 += __half2float(wr0[i + 96u]) * xv;
        acc1 += __half2float(wr1[i + 96u]) * xv;
        xv = shx[i + 128u];
        acc0 += __half2float(wr0[i + 128u]) * xv;
        acc1 += __half2float(wr1[i + 128u]) * xv;
        xv = shx[i + 160u];
        acc0 += __half2float(wr0[i + 160u]) * xv;
        acc1 += __half2float(wr1[i + 160u]) * xv;
        xv = shx[i + 192u];
        acc0 += __half2float(wr0[i + 192u]) * xv;
        acc1 += __half2float(wr1[i + 192u]) * xv;
        xv = shx[i + 224u];
        acc0 += __half2float(wr0[i + 224u]) * xv;
        acc1 += __half2float(wr1[i + 224u]) * xv;
    }
    for (; i < kInput; i += 32u) {
        const float xv = shx[i];
        acc0 += __half2float(wr0[i]) * xv;
        acc1 += __half2float(wr1[i]) * xv;
    }
    acc0 = warp_sum(acc0);
    acc1 = warp_sum(acc1);
    if (lane == 0u) {
        out0[row] = acc0;
        out1[row] = acc1;
    }
}

// Four independent accumulators per weight matrix. The k iteration and warp
// reduction order observed by each token exactly mirror pair_one_token.
__global__ static void pair_four_tokens(
        float *out0, float *out1, const __half *w0, const __half *w1,
        const float *x) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    for (uint32_t i = tid; i < kTokens * kInput; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    const uint32_t row = blockIdx.x * rows_per_block + wave;
    if (row >= kOutput) return;
    const __half *wr0 = w0 + (uint64_t)row * kInput;
    const __half *wr1 = w1 + (uint64_t)row * kInput;
    float acc0[kTokens] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[kTokens] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t i = lane;
    for (; i + 224u < kInput; i += 256u) {
#pragma unroll
        for (uint32_t u = 0; u < 8u; u++) {
            const uint32_t k = i + u * 32u;
            const float fw0 = __half2float(wr0[k]);
            const float fw1 = __half2float(wr1[k]);
#pragma unroll
            for (uint32_t t = 0; t < kTokens; t++) {
                const float xv = shx[t * kInput + k];
                acc0[t] += fw0 * xv;
                acc1[t] += fw1 * xv;
            }
        }
    }
    for (; i < kInput; i += 32u) {
        const float fw0 = __half2float(wr0[i]);
        const float fw1 = __half2float(wr1[i]);
#pragma unroll
        for (uint32_t t = 0; t < kTokens; t++) {
            const float xv = shx[t * kInput + i];
            acc0[t] += fw0 * xv;
            acc1[t] += fw1 * xv;
        }
    }
#pragma unroll
    for (uint32_t t = 0; t < kTokens; t++) {
        acc0[t] = warp_sum(acc0[t]);
        acc1[t] = warp_sum(acc1[t]);
    }
    if (lane == 0u) {
#pragma unroll
        for (uint32_t t = 0; t < kTokens; t++) {
            out0[(uint64_t)t * kOutput + row] = acc0[t];
            out1[(uint64_t)t * kOutput + row] = acc1[t];
        }
    }
}

struct buffers {
    __half *w0{}, *w1{};
    float *x{}, *base0{}, *base1{}, *cand0{}, *cand1{};

    buffers() {
        const uint64_t weights = (uint64_t)kWeightSets * kOutput * kInput;
        const uint64_t outputs = (uint64_t)kTokens * kOutput;
        std::mt19937 gen(29);
        std::uniform_real_distribution<float> dist(-0.08f, 0.08f);
        std::vector<__half> hw0(weights), hw1(weights);
        std::vector<float> hx((uint64_t)kTokens * kInput);
        for (auto &v : hw0) v = __float2half(dist(gen));
        for (auto &v : hw1) v = __float2half(dist(gen));
        for (auto &v : hx) v = dist(gen);
        check(hipMalloc(&w0, weights * sizeof(__half)), "malloc w0");
        check(hipMalloc(&w1, weights * sizeof(__half)), "malloc w1");
        check(hipMalloc(&x, hx.size() * sizeof(float)), "malloc x");
        check(hipMalloc(&base0, outputs * sizeof(float)), "malloc base0");
        check(hipMalloc(&base1, outputs * sizeof(float)), "malloc base1");
        check(hipMalloc(&cand0, outputs * sizeof(float)), "malloc cand0");
        check(hipMalloc(&cand1, outputs * sizeof(float)), "malloc cand1");
        check(hipMemcpy(w0, hw0.data(), weights * sizeof(__half), hipMemcpyHostToDevice), "copy w0");
        check(hipMemcpy(w1, hw1.data(), weights * sizeof(__half), hipMemcpyHostToDevice), "copy w1");
        check(hipMemcpy(x, hx.data(), hx.size() * sizeof(float), hipMemcpyHostToDevice), "copy x");
    }

    ~buffers() {
        (void)hipFree(w0); (void)hipFree(w1); (void)hipFree(x);
        (void)hipFree(base0); (void)hipFree(base1);
        (void)hipFree(cand0); (void)hipFree(cand1);
    }

    void baseline(uint32_t iteration) {
        const uint64_t stride = (uint64_t)kOutput * kInput;
        const uint32_t set = iteration % kWeightSets;
        for (uint32_t t = 0; t < kTokens; t++) {
            pair_one_token<<<(kOutput + 31u) / 32u, 1024u,
                             kInput * sizeof(float)>>>(
                base0 + (uint64_t)t * kOutput,
                base1 + (uint64_t)t * kOutput,
                w0 + (uint64_t)set * stride, w1 + (uint64_t)set * stride,
                x + (uint64_t)t * kInput);
        }
    }

    // The same layer is revisited only after every other layer and expert in
    // a decode token, so its weights are cold on the next token. Rotate a
    // distinct model-sized pair for each timed token instead of granting the
    // sequential control three unrealistic immediate cache hits.
    void baseline_cold(uint32_t iteration) {
        const uint64_t stride = (uint64_t)kOutput * kInput;
        for (uint32_t t = 0; t < kTokens; t++) {
            const uint32_t set = (iteration * kTokens + t) % kWeightSets;
            pair_one_token<<<(kOutput + 31u) / 32u, 1024u,
                             kInput * sizeof(float)>>>(
                base0 + (uint64_t)t * kOutput,
                base1 + (uint64_t)t * kOutput,
                w0 + (uint64_t)set * stride, w1 + (uint64_t)set * stride,
                x + (uint64_t)t * kInput);
        }
    }

    void candidate(uint32_t iteration) {
        const uint64_t stride = (uint64_t)kOutput * kInput;
        const uint32_t set = iteration % kWeightSets;
        pair_four_tokens<<<(kOutput + 31u) / 32u, 1024u,
                           kTokens * kInput * sizeof(float)>>>(
            cand0, cand1, w0 + (uint64_t)set * stride,
            w1 + (uint64_t)set * stride, x);
    }
};

template <typename F>
static float time_us(F fn, uint32_t iterations) {
    hipEvent_t begin{}, end{};
    check(hipEventCreate(&begin), "event begin");
    check(hipEventCreate(&end), "event end");
    for (uint32_t i = 0; i < kWeightSets; i++) fn(i);
    check(hipDeviceSynchronize(), "warm sync");
    check(hipEventRecord(begin), "record begin");
    for (uint32_t i = 0; i < iterations; i++) fn(i);
    check(hipEventRecord(end), "record end");
    check(hipEventSynchronize(end), "event sync");
    float ms = 0.0f;
    check(hipEventElapsedTime(&ms, begin, end), "event elapsed");
    (void)hipEventDestroy(begin); (void)hipEventDestroy(end);
    return ms * 1000.0f / iterations;
}

int main() {
    buffers b;
    b.baseline(3); b.candidate(3);
    check(hipDeviceSynchronize(), "correctness sync");
    const uint64_t n = (uint64_t)kTokens * kOutput;
    std::vector<float> base(n), cand(n);
    check(hipMemcpy(base.data(), b.base0, n * sizeof(float), hipMemcpyDeviceToHost), "read base0");
    check(hipMemcpy(cand.data(), b.cand0, n * sizeof(float), hipMemcpyDeviceToHost), "read cand0");
    const bool out0_equal = std::memcmp(base.data(), cand.data(), n * sizeof(float)) == 0;
    check(hipMemcpy(base.data(), b.base1, n * sizeof(float), hipMemcpyDeviceToHost), "read base1");
    check(hipMemcpy(cand.data(), b.cand1, n * sizeof(float), hipMemcpyDeviceToHost), "read cand1");
    const bool out1_equal = std::memcmp(base.data(), cand.data(), n * sizeof(float)) == 0;
    const float base_hot_us = time_us([&](uint32_t i) { b.baseline(i); }, 128);
    const float base_us = time_us([&](uint32_t i) { b.baseline_cold(i); }, 128);
    const float cand_us = time_us([&](uint32_t i) { b.candidate(i); }, 128);
    std::printf("shape=4x4096x%u pair rotating_weight_mib=%.1f "
                "bitwise=%d/%d sequential_hot_us=%.3f sequential_cold_us=%.3f "
                "tiled_us=%.3f cold_speedup=%.3fx\n",
                kOutput,
                2.0 * kWeightSets * kOutput * kInput * sizeof(__half) / 1048576.0,
                out0_equal, out1_equal, base_hot_us, base_us, cand_us,
                base_us / cand_us);
    return out0_equal && out1_equal ? 0 : 2;
}
