// Small-step oracle for fusing the ordinary one-token paired F16 compressor
// projection with its state-store epilogue. It uses the production reduction
// order on both sides, so every projected/state value must match bit-for-bit.
//
// Build, varying DS4_BENCH_OUTPUT across 256, 512, and 1024:
//   hipcc -O3 --offload-arch=gfx1151 \
//     scripts/f16_pair_compressor_store_bench.cu \
//     -o /tmp/f16_pair_compressor_store_bench

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static constexpr uint32_t kInput = 4096;
#ifndef DS4_BENCH_OUTPUT
#define DS4_BENCH_OUTPUT 1024
#endif
static constexpr uint32_t kOutput = DS4_BENCH_OUTPUT;
static constexpr uint32_t kRatio = kOutput == 512 ? 128 : 4;
static constexpr uint32_t kWeightSets = 8;

static void check(hipError_t rc, const char *what) {
    if (rc != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(rc));
        std::exit(1);
    }
}

__device__ static float warp_sum(float v) {
    for (uint32_t delta = 16; delta != 0; delta >>= 1u) {
        v += __shfl_down(v, delta, 32);
    }
    return v;
}

template <bool FUSED>
__global__ static void projection(
        float *out0, float *out1, float *state0, float *state1,
        const __half *w0, const __half *w1, const float *x,
        const __half *ape, uint32_t pos) {
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
#pragma unroll
        for (uint32_t u = 0; u < 8; u++) {
            const uint32_t k = i + u * 32u;
            const float xv = shx[k];
            acc0 += __half2float(wr0[k]) * xv;
            acc1 += __half2float(wr1[k]) * xv;
        }
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
        if constexpr (FUSED) {
            const uint32_t dst = kRatio == 4 ? kRatio + pos % kRatio : pos % kRatio;
            state0[(uint64_t)dst * kOutput + row] = acc0;
            state1[(uint64_t)dst * kOutput + row] =
                acc1 + __half2float(ape[(uint64_t)(pos % kRatio) * kOutput + row]);
        }
    }
}

// Two output rows per wave keeps each row's exact reduction order, but uses
// four accumulators per lane and two 512-thread blocks per CU instead of one
// 1024-thread block. This tests the occupancy hypothesis independently from
// the store fusion.
__global__ static void projection_two_row(
        float *out0, float *out1, const __half *w0, const __half *w1,
        const float *x) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    for (uint32_t i = tid; i < kInput; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    const uint32_t row0 = blockIdx.x * 32u + wave * 2u;
    const uint32_t row1 = row0 + 1u;
    if (row0 >= kOutput) return;
    const __half *w00 = w0 + (uint64_t)row0 * kInput;
    const __half *w10 = w1 + (uint64_t)row0 * kInput;
    const __half *w01 = w0 + (uint64_t)row1 * kInput;
    const __half *w11 = w1 + (uint64_t)row1 * kInput;
    float a00 = 0.0f, a10 = 0.0f, a01 = 0.0f, a11 = 0.0f;
    uint32_t i = lane;
    for (; i + 224u < kInput; i += 256u) {
#pragma unroll
        for (uint32_t u = 0; u < 8; u++) {
            const uint32_t k = i + u * 32u;
            const float xv = shx[k];
            a00 += __half2float(w00[k]) * xv;
            a10 += __half2float(w10[k]) * xv;
            a01 += __half2float(w01[k]) * xv;
            a11 += __half2float(w11[k]) * xv;
        }
    }
    for (; i < kInput; i += 32u) {
        const float xv = shx[i];
        a00 += __half2float(w00[i]) * xv;
        a10 += __half2float(w10[i]) * xv;
        a01 += __half2float(w01[i]) * xv;
        a11 += __half2float(w11[i]) * xv;
    }
    a00 = warp_sum(a00); a10 = warp_sum(a10);
    a01 = warp_sum(a01); a11 = warp_sum(a11);
    if (lane == 0u) {
        out0[row0] = a00; out1[row0] = a10;
        out0[row1] = a01; out1[row1] = a11;
    }
}

__global__ static void store(const float *out0, const float *out1,
                             float *state0, float *state1,
                             const __half *ape, uint32_t pos) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= kOutput) return;
    const uint32_t dst = kRatio == 4 ? kRatio + pos % kRatio : pos % kRatio;
    state0[(uint64_t)dst * kOutput + row] = out0[row];
    state1[(uint64_t)dst * kOutput + row] =
        out1[row] + __half2float(ape[(uint64_t)(pos % kRatio) * kOutput + row]);
}

struct buffers {
    __half *w0{}, *w1{}, *ape{};
    float *x{}, *base0{}, *base1{}, *cand0{}, *cand1{};
    float *base_state0{}, *base_state1{}, *cand_state0{}, *cand_state1{};

    buffers() {
        std::mt19937 gen(17);
        std::uniform_real_distribution<float> dist(-0.08f, 0.08f);
        const uint64_t weights = (uint64_t)kWeightSets * kOutput * kInput;
        const uint64_t states = (uint64_t)(kRatio == 4 ? 8 : kRatio) * kOutput;
        std::vector<__half> hw0(weights), hw1(weights), hape((uint64_t)kRatio * kOutput);
        std::vector<float> hx(kInput);
        for (auto &v : hw0) v = __float2half(dist(gen));
        for (auto &v : hw1) v = __float2half(dist(gen));
        for (auto &v : hape) v = __float2half(dist(gen));
        for (auto &v : hx) v = dist(gen);
#define MALLOC(P, N, LABEL) check(hipMalloc(&(P), (N)), LABEL)
        MALLOC(w0, weights * sizeof(__half), "malloc w0");
        MALLOC(w1, weights * sizeof(__half), "malloc w1");
        MALLOC(ape, hape.size() * sizeof(__half), "malloc ape");
        MALLOC(x, kInput * sizeof(float), "malloc x");
        MALLOC(base0, kOutput * sizeof(float), "malloc base0");
        MALLOC(base1, kOutput * sizeof(float), "malloc base1");
        MALLOC(cand0, kOutput * sizeof(float), "malloc cand0");
        MALLOC(cand1, kOutput * sizeof(float), "malloc cand1");
        MALLOC(base_state0, states * sizeof(float), "malloc base state0");
        MALLOC(base_state1, states * sizeof(float), "malloc base state1");
        MALLOC(cand_state0, states * sizeof(float), "malloc cand state0");
        MALLOC(cand_state1, states * sizeof(float), "malloc cand state1");
#undef MALLOC
        check(hipMemcpy(w0, hw0.data(), weights * sizeof(__half), hipMemcpyHostToDevice), "copy w0");
        check(hipMemcpy(w1, hw1.data(), weights * sizeof(__half), hipMemcpyHostToDevice), "copy w1");
        check(hipMemcpy(ape, hape.data(), hape.size() * sizeof(__half), hipMemcpyHostToDevice), "copy ape");
        check(hipMemcpy(x, hx.data(), kInput * sizeof(float), hipMemcpyHostToDevice), "copy x");
        check(hipMemset(base_state0, 0, states * sizeof(float)), "clear base state0");
        check(hipMemset(base_state1, 0, states * sizeof(float)), "clear base state1");
        check(hipMemset(cand_state0, 0, states * sizeof(float)), "clear cand state0");
        check(hipMemset(cand_state1, 0, states * sizeof(float)), "clear cand state1");
    }

    ~buffers() {
        (void)hipFree(w0); (void)hipFree(w1); (void)hipFree(ape); (void)hipFree(x);
        (void)hipFree(base0); (void)hipFree(base1); (void)hipFree(cand0); (void)hipFree(cand1);
        (void)hipFree(base_state0); (void)hipFree(base_state1);
        (void)hipFree(cand_state0); (void)hipFree(cand_state1);
    }

    void baseline(uint32_t iteration) {
        const uint64_t stride = (uint64_t)kOutput * kInput;
        const uint32_t set = iteration % kWeightSets;
        const uint32_t pos = iteration % kRatio;
        projection<false><<<(kOutput + 31u) / 32u, 1024u, kInput * sizeof(float)>>>(
            base0, base1, nullptr, nullptr, w0 + set * stride, w1 + set * stride,
            x, ape, pos);
        store<<<(kOutput + 255u) / 256u, 256u>>>(
            base0, base1, base_state0, base_state1, ape, pos);
    }

    void candidate(uint32_t iteration) {
        const uint64_t stride = (uint64_t)kOutput * kInput;
        const uint32_t set = iteration % kWeightSets;
        const uint32_t pos = iteration % kRatio;
        projection<true><<<(kOutput + 31u) / 32u, 1024u, kInput * sizeof(float)>>>(
            cand0, cand1, cand_state0, cand_state1, w0 + set * stride,
            w1 + set * stride, x, ape, pos);
    }

    void two_row(uint32_t iteration) {
        const uint64_t stride = (uint64_t)kOutput * kInput;
        const uint32_t set = iteration % kWeightSets;
        const uint32_t pos = iteration % kRatio;
        projection_two_row<<<(kOutput + 31u) / 32u, 512u,
                             kInput * sizeof(float)>>>(
            cand0, cand1, w0 + set * stride, w1 + set * stride, x);
        store<<<(kOutput + 255u) / 256u, 256u>>>(
            cand0, cand1, cand_state0, cand_state1, ape, pos);
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
    b.baseline(3); b.two_row(3);
    check(hipDeviceSynchronize(), "correctness sync");
    const uint64_t states = (uint64_t)(kRatio == 4 ? 8 : kRatio) * kOutput;
    std::vector<float> a(states), c(states);
    check(hipMemcpy(a.data(), b.base_state0, states * sizeof(float), hipMemcpyDeviceToHost), "read base0");
    check(hipMemcpy(c.data(), b.cand_state0, states * sizeof(float), hipMemcpyDeviceToHost), "read cand0");
    const bool two_state0_equal = std::memcmp(a.data(), c.data(), states * sizeof(float)) == 0;
    check(hipMemcpy(a.data(), b.base_state1, states * sizeof(float), hipMemcpyDeviceToHost), "read base1");
    check(hipMemcpy(c.data(), b.cand_state1, states * sizeof(float), hipMemcpyDeviceToHost), "read cand1");
    const bool two_state1_equal = std::memcmp(a.data(), c.data(), states * sizeof(float)) == 0;
    b.candidate(3);
    check(hipDeviceSynchronize(), "fused correctness sync");
    check(hipMemcpy(c.data(), b.cand_state0, states * sizeof(float), hipMemcpyDeviceToHost), "read fused0");
    check(hipMemcpy(a.data(), b.base_state0, states * sizeof(float), hipMemcpyDeviceToHost), "reread base0");
    const bool fused_state0_equal = std::memcmp(a.data(), c.data(), states * sizeof(float)) == 0;
    check(hipMemcpy(c.data(), b.cand_state1, states * sizeof(float), hipMemcpyDeviceToHost), "read fused1");
    check(hipMemcpy(a.data(), b.base_state1, states * sizeof(float), hipMemcpyDeviceToHost), "reread base1");
    const bool fused_state1_equal = std::memcmp(a.data(), c.data(), states * sizeof(float)) == 0;
    const float base_us = time_us([&](uint32_t i) { b.baseline(i); }, 256);
    const float two_us = time_us([&](uint32_t i) { b.two_row(i); }, 256);
    const float cand_us = time_us([&](uint32_t i) { b.candidate(i); }, 256);
    std::printf("shape=4096x%u pair ratio=%u rotating_weight_mib=%.1f\n",
                kOutput, kRatio,
                2.0 * kWeightSets * kOutput * kInput * sizeof(__half) / 1048576.0);
    std::printf("two_row_bitwise=%d/%d fused_bitwise=%d/%d baseline_us=%.3f "
                "two_row_us=%.3f two_row_change=%+.2f%% fused_us=%.3f "
                "fused_change=%+.2f%%\n", two_state0_equal, two_state1_equal,
                fused_state0_equal, fused_state1_equal, base_us, two_us,
                100.0f * (two_us / base_us - 1.0f), cand_us,
                100.0f * (cand_us / base_us - 1.0f));
    return two_state0_equal && two_state1_equal &&
           fused_state0_equal && fused_state1_equal ? 0 : 2;
}
