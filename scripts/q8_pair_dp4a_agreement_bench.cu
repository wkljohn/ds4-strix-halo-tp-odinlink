// Arithmetic-agreement oracle for the paired Q8_0 projection used by
// committed one-row decode and the 2--5-row DSpark verifier.
//
// Build:
//   /opt/rocm/bin/hipcc -O3 -ffast-math -fno-finite-math-only \
//     --offload-arch=gfx1151 scripts/q8_pair_dp4a_agreement_bench.cu \
//     -o /tmp/q8_pair_dp4a_agreement_bench

#include "../ds4_rocm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static constexpr uint32_t kIn = 4096;
static constexpr uint32_t kOut0 = 1024;
static constexpr uint32_t kOut1 = 512;
static constexpr uint32_t kBlocks = kIn / 32;
static constexpr uint64_t kRowBytes = (uint64_t)kBlocks * 34u;

static void check(hipError_t rc, const char *what) {
    if (rc != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(rc));
        std::exit(1);
    }
}

__device__ __forceinline__ static float warp_max(float v) {
    for (int d = 16; d; d >>= 1) v = fmaxf(v, __shfl_down(v, d, 32));
    return v;
}

__device__ __forceinline__ static float warp_sum(float v) {
    for (int d = 16; d; d >>= 1) v += __shfl_down(v, d, 32);
    return v;
}

__global__ static void quantize_q8_0(
        int8_t *q, float *scale, const float *x, uint32_t n_tok) {
    const uint32_t b = blockIdx.x;
    const uint32_t t = blockIdx.y;
    if (b >= kBlocks || t >= n_tok) return;
    const float *src = x + (uint64_t)t * kIn + b * 32u;
    float a = warp_max(fabsf(src[threadIdx.x]));
    const float d = __shfl(a, 0, 32) / 127.0f;
    const float id = d == 0.0f ? 0.0f : 1.0f / d;
    const uint64_t qb = (uint64_t)t * kBlocks + b;
    if (threadIdx.x == 0) scale[qb] = d;
    int v = __float2int_rn(src[threadIdx.x] * id);
    v = max(-128, min(127, v));
    q[qb * 32u + threadIdx.x] = (int8_t)v;
}

template <uint32_t TOK_TILE>
__global__ static void pair_dp4a(
        float *out, const unsigned char *w, const int8_t *q,
        const float *scale, uint32_t out_dim, uint32_t n_tok) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t row = blockIdx.x * rows_per_block + wave;
    if (row >= out_dim) return;
    const unsigned char *wr = w + (uint64_t)row * kRowBytes;
    float acc[TOK_TILE];
#pragma unroll
    for (uint32_t t = 0; t < TOK_TILE; ++t) acc[t] = 0.0f;
    for (uint32_t b = lane; b < kBlocks; b += 32u) {
        const unsigned char *bp = wr + (uint64_t)b * 34u;
        const float ws = __half2float(*reinterpret_cast<const __half *>(bp));
        int32_t qw[8];
#pragma unroll
        for (uint32_t j = 0; j < 8u; ++j) {
            const uint8_t *p = bp + 2u + j * 4u;
            qw[j] = (int32_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                              ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
        }
#pragma unroll
        for (uint32_t t = 0; t < TOK_TILE; ++t) {
            if (t >= n_tok) continue;
            const uint64_t qb = (uint64_t)t * kBlocks + b;
            const int8_t *xq = q + qb * 32u;
            int32_t dot = 0;
#pragma unroll
            for (uint32_t j = 0; j < 8u; ++j) {
                int32_t xw;
                __builtin_memcpy(&xw, xq + j * 4u, sizeof(xw));
                dot = __dp4a(qw[j], xw, dot);
            }
            acc[t] += ws * scale[qb] * (float)dot;
        }
    }
#pragma unroll
    for (uint32_t t = 0; t < TOK_TILE; ++t) acc[t] = warp_sum(acc[t]);
    if (lane == 0) {
#pragma unroll
        for (uint32_t t = 0; t < TOK_TILE; ++t) {
            if (t < n_tok) out[(uint64_t)t * out_dim + row] = acc[t];
        }
    }
}

static void launch_rows(float *out, const unsigned char *w, const int8_t *q,
                        const float *s, uint32_t out_dim, uint32_t n_tok) {
    const dim3 grid((out_dim + 7u) / 8u);
    if (n_tok <= 2u) pair_dp4a<2><<<grid, 256>>>(out, w, q, s, out_dim, n_tok);
    else if (n_tok == 3u) pair_dp4a<3><<<grid, 256>>>(out, w, q, s, out_dim, n_tok);
    else if (n_tok == 4u) pair_dp4a<4><<<grid, 256>>>(out, w, q, s, out_dim, n_tok);
    else pair_dp4a<5><<<grid, 256>>>(out, w, q, s, out_dim, n_tok);
}

struct Buffers {
    std::vector<unsigned char> h_w0, h_w1;
    std::vector<float> h_x;
    unsigned char *w0 = nullptr, *w1 = nullptr;
    float *x = nullptr, *s_batch = nullptr, *s_one = nullptr;
    int8_t *q_batch = nullptr, *q_one = nullptr;
    float *batch0 = nullptr, *batch1 = nullptr, *one0 = nullptr, *one1 = nullptr;

    Buffers() : h_w0((uint64_t)kOut0 * kRowBytes),
                h_w1((uint64_t)kOut1 * kRowBytes), h_x(5u * kIn) {
        std::mt19937 rng(0x51445034u);
        std::uniform_real_distribution<float> xf(-1.5f, 1.5f);
        std::uniform_real_distribution<float> sf(0.001f, 0.04f);
        std::uniform_int_distribution<int> qi(-127, 127);
        auto fill_w = [&](std::vector<unsigned char> &w, uint32_t rows) {
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t b = 0; b < kBlocks; ++b) {
                    unsigned char *p = w.data() + (uint64_t)r * kRowBytes + b * 34u;
                    const __half hs = __float2half(sf(rng));
                    std::memcpy(p, &hs, sizeof(hs));
                    for (uint32_t i = 0; i < 32; ++i) p[2u + i] = (unsigned char)(int8_t)qi(rng);
                }
            }
        };
        fill_w(h_w0, kOut0); fill_w(h_w1, kOut1);
        for (float &v : h_x) v = xf(rng);
        check(hipMalloc(&w0, h_w0.size()), "w0");
        check(hipMalloc(&w1, h_w1.size()), "w1");
        check(hipMalloc(&x, h_x.size() * sizeof(float)), "x");
        const size_t qbytes = 5ull * kBlocks * 32u;
        const size_t sbytes = 5ull * kBlocks * sizeof(float);
        check(hipMalloc(&q_batch, qbytes), "q_batch");
        check(hipMalloc(&q_one, qbytes), "q_one");
        check(hipMalloc(&s_batch, sbytes), "s_batch");
        check(hipMalloc(&s_one, sbytes), "s_one");
        check(hipMalloc(&batch0, 5ull * kOut0 * sizeof(float)), "batch0");
        check(hipMalloc(&batch1, 5ull * kOut1 * sizeof(float)), "batch1");
        check(hipMalloc(&one0, 5ull * kOut0 * sizeof(float)), "one0");
        check(hipMalloc(&one1, 5ull * kOut1 * sizeof(float)), "one1");
        check(hipMemcpy(w0, h_w0.data(), h_w0.size(), hipMemcpyHostToDevice), "copy w0");
        check(hipMemcpy(w1, h_w1.data(), h_w1.size(), hipMemcpyHostToDevice), "copy w1");
        check(hipMemcpy(x, h_x.data(), h_x.size() * sizeof(float), hipMemcpyHostToDevice), "copy x");
    }
    ~Buffers() {
        hipFree(w0); hipFree(w1); hipFree(x); hipFree(q_batch); hipFree(q_one);
        hipFree(s_batch); hipFree(s_one); hipFree(batch0); hipFree(batch1);
        hipFree(one0); hipFree(one1);
    }
};

static void run_batch(Buffers &b, uint32_t n) {
    quantize_q8_0<<<dim3(kBlocks, n), 32>>>(b.q_batch, b.s_batch, b.x, n);
    launch_rows(b.batch0, b.w0, b.q_batch, b.s_batch, kOut0, n);
    launch_rows(b.batch1, b.w1, b.q_batch, b.s_batch, kOut1, n);
}

static void run_one_by_one(Buffers &b, uint32_t n) {
    for (uint32_t t = 0; t < n; ++t) {
        quantize_q8_0<<<dim3(kBlocks, 1), 32>>>(
            b.q_one + (uint64_t)t * kBlocks * 32u,
            b.s_one + (uint64_t)t * kBlocks,
            b.x + (uint64_t)t * kIn, 1);
        launch_rows(b.one0 + (uint64_t)t * kOut0, b.w0,
                    b.q_one + (uint64_t)t * kBlocks * 32u,
                    b.s_one + (uint64_t)t * kBlocks, kOut0, 1);
        launch_rows(b.one1 + (uint64_t)t * kOut1, b.w1,
                    b.q_one + (uint64_t)t * kBlocks * 32u,
                    b.s_one + (uint64_t)t * kBlocks, kOut1, 1);
    }
}

static double elapsed_ms(void (*fn)(Buffers &, uint32_t), Buffers &b,
                         uint32_t n, int iters) {
    hipEvent_t start, stop;
    check(hipEventCreate(&start), "event start");
    check(hipEventCreate(&stop), "event stop");
    for (int i = 0; i < 3; ++i) fn(b, n);
    check(hipDeviceSynchronize(), "warmup");
    check(hipEventRecord(start), "record start");
    for (int i = 0; i < iters; ++i) fn(b, n);
    check(hipEventRecord(stop), "record stop");
    check(hipEventSynchronize(stop), "sync stop");
    float ms = 0.0f;
    check(hipEventElapsedTime(&ms, start, stop), "elapsed");
    hipEventDestroy(start); hipEventDestroy(stop);
    return ms / iters;
}

int main() {
    Buffers b;
    bool all_ok = true;
    for (uint32_t n = 1; n <= 5; ++n) {
        run_batch(b, n); run_one_by_one(b, n);
        check(hipDeviceSynchronize(), "agreement run");
        const size_t qbytes = (size_t)n * kBlocks * 32u;
        const size_t scount = (size_t)n * kBlocks;
        std::vector<int8_t> qb(qbytes), qo(qbytes);
        std::vector<float> sb(scount), so(scount);
        check(hipMemcpy(qb.data(), b.q_batch, qbytes, hipMemcpyDeviceToHost), "read qb");
        check(hipMemcpy(qo.data(), b.q_one, qbytes, hipMemcpyDeviceToHost), "read qo");
        check(hipMemcpy(sb.data(), b.s_batch, scount * sizeof(float), hipMemcpyDeviceToHost), "read sb");
        check(hipMemcpy(so.data(), b.s_one, scount * sizeof(float), hipMemcpyDeviceToHost), "read so");
        size_t qdiff = 0, sdiff = 0, odiff = 0;
        for (size_t i = 0; i < qbytes; ++i) qdiff += qb[i] != qo[i];
        for (size_t i = 0; i < scount; ++i) sdiff += std::memcmp(&sb[i], &so[i], 4) != 0;
        double max_abs = 0.0;
        auto compare_out = [&](float *a_dev, float *o_dev, size_t count) {
            std::vector<float> a(count), o(count);
            check(hipMemcpy(a.data(), a_dev, count * sizeof(float), hipMemcpyDeviceToHost), "read batch out");
            check(hipMemcpy(o.data(), o_dev, count * sizeof(float), hipMemcpyDeviceToHost), "read one out");
            for (size_t i = 0; i < count; ++i) {
                odiff += std::memcmp(&a[i], &o[i], 4) != 0;
                max_abs = std::max(max_abs, std::fabs((double)a[i] - o[i]));
            }
        };
        compare_out(b.batch0, b.one0, (size_t)n * kOut0);
        compare_out(b.batch1, b.one1, (size_t)n * kOut1);
        const bool ok = qdiff == 0 && sdiff == 0 && odiff == 0;
        all_ok &= ok;
        std::printf("rows=%u quant_diff=%zu scale_bitdiff=%zu output_bitdiff=%zu max_abs=%.9g %s\n",
                    n, qdiff, sdiff, odiff, max_abs, ok ? "PASS" : "FAIL");
    }
    const double serial = elapsed_ms(run_one_by_one, b, 5, 50);
    const double batch = elapsed_ms(run_batch, b, 5, 50);
    std::printf("five_rows serial_pair_ms=%.4f batch_pair_ms=%.4f change=%+.1f%% agreement=%s\n",
                serial, batch, 100.0 * (batch / serial - 1.0), all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 2;
}
