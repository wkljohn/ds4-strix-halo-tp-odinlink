// Small-step harness for gfx1151 DSpark verifier Q8_0 projections.
// Compares the current Q8-weight/F32-activation shape with dynamic per-block
// Q8 activations plus packed signed INT8 dot products.  No production code is
// selected by this file.
//
// Build:
//   hipcc -O3 -ffast-math --offload-arch=gfx1151 \
//     scripts/q8_small_batch_dp4a_bench.cu -o /tmp/q8-small-dp4a

#include "../ds4_rocm.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static void ck(hipError_t rc, const char *where) {
    if (rc != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, hipGetErrorString(rc));
        std::exit(1);
    }
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int d = 16; d; d >>= 1) v += __shfl_down(v, d, 32);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int d = 16; d; d >>= 1) v = fmaxf(v, __shfl_down(v, d, 32));
    return v;
}
__device__ __forceinline__ int32_t load_i32_unaligned(const int8_t *p) {
    const uint8_t *u = reinterpret_cast<const uint8_t *>(p);
    return (int32_t)((uint32_t)u[0] | ((uint32_t)u[1] << 8) |
                     ((uint32_t)u[2] << 16) | ((uint32_t)u[3] << 24));
}

__global__ void quantize_q8_blocks(const float *x, int8_t *q, float *scale,
                                   uint32_t tokens, uint32_t blocks) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t item = blockIdx.x * (blockDim.x >> 5u) + warp;
    if (item >= tokens * blocks) return;
    const uint32_t token = item / blocks;
    const uint32_t block = item - token * blocks;
    const uint64_t off = ((uint64_t)token * blocks + block) * 32u;
    const float v = x[off + lane];
    const float amax = warp_max(fabsf(v));
    const float d = __shfl(amax, 0, 32) * (1.0f / 127.0f);
    const float inv = d > 0.0f ? 1.0f / d : 0.0f;
    int iv = __float2int_rn(v * inv);
    iv = max(-127, min(127, iv));
    q[off + lane] = (int8_t)iv;
    if (lane == 0) scale[item] = d;
}

template <uint32_t TOKENS>
__global__ void q8q8_dp4a(float *out, const uint8_t *w, const int8_t *q,
                          const float *xs, uint32_t blocks,
                          uint32_t out_dim, uint64_t row_bytes) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t row = blockIdx.x * (blockDim.x >> 5u) + wave;
    if (row >= out_dim) return;
    const uint8_t *wr = w + (uint64_t)row * row_bytes;
    float acc[TOKENS];
#pragma unroll
    for (uint32_t t = 0; t < TOKENS; ++t) acc[t] = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const uint8_t *bp = wr + (uint64_t)b * 34u;
        const float ws = __half2float(*reinterpret_cast<const __half *>(bp));
        int32_t qw[8];
#pragma unroll
        for (uint32_t j = 0; j < 8u; ++j)
            qw[j] = load_i32_unaligned(reinterpret_cast<const int8_t *>(bp + 2u) + 4u * j);
#pragma unroll
        for (uint32_t t = 0; t < TOKENS; ++t) {
            const uint64_t xb = ((uint64_t)t * blocks + b);
            const int8_t *xq = q + xb * 32u;
            int32_t dot = 0;
#pragma unroll
            for (uint32_t j = 0; j < 8u; ++j)
                dot = __dp4a(qw[j], *reinterpret_cast<const int32_t *>(xq + 4u * j), dot);
            acc[t] += ws * xs[xb] * (float)dot;
        }
    }
#pragma unroll
    for (uint32_t t = 0; t < TOKENS; ++t) acc[t] = warp_sum(acc[t]);
    if (lane == 0) {
#pragma unroll
        for (uint32_t t = 0; t < TOKENS; ++t)
            out[(uint64_t)t * out_dim + row] = acc[t];
    }
}

template <uint32_t TOKENS>
__global__ void q8f32_reference(float *out, const uint8_t *w, const float *x,
                                uint32_t blocks, uint32_t out_dim,
                                uint64_t row_bytes) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t row = blockIdx.x * (blockDim.x >> 5u) + wave;
    if (row >= out_dim) return;
    const uint8_t *wr = w + (uint64_t)row * row_bytes;
    float acc[TOKENS];
#pragma unroll
    for (uint32_t t = 0; t < TOKENS; ++t) acc[t] = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *bp = wr + (uint64_t)b * 34u;
        const float ws = __half2float(*reinterpret_cast<const __half *>(bp));
        const float wv = ws * (float)reinterpret_cast<const int8_t *>(bp + 2u)[lane];
#pragma unroll
        for (uint32_t t = 0; t < TOKENS; ++t)
            acc[t] += wv * x[((uint64_t)t * blocks + b) * 32u + lane];
    }
#pragma unroll
    for (uint32_t t = 0; t < TOKENS; ++t) acc[t] = warp_sum(acc[t]);
    if (lane == 0) {
#pragma unroll
        for (uint32_t t = 0; t < TOKENS; ++t)
            out[(uint64_t)t * out_dim + row] = acc[t];
    }
}

struct Buffers {
    static constexpr uint32_t tokens = 5;
    uint32_t in_dim, out_dim, blocks;
    uint64_t row_bytes;
    std::vector<uint8_t> hw;
    std::vector<float> hx;
    uint8_t *w = nullptr;
    float *x = nullptr, *xs = nullptr, *ref = nullptr, *got = nullptr;
    int8_t *xq = nullptr;
    Buffers(uint32_t k, uint32_t m)
        : in_dim(k), out_dim(m), blocks(k / 32u),
          row_bytes((uint64_t)blocks * 34u), hw((uint64_t)m * row_bytes),
          hx((uint64_t)tokens * k) {
        std::mt19937 rng(71);
        std::uniform_real_distribution<float> xf(-1.25f, 1.25f);
        std::uniform_int_distribution<int> qi(-127, 127);
        for (uint32_t r = 0; r < m; ++r) for (uint32_t b = 0; b < blocks; ++b) {
            uint8_t *p = hw.data() + (uint64_t)r * row_bytes + (uint64_t)b * 34u;
            const __half d = __float2half(0.002f + 0.02f * fabsf(xf(rng)));
            __builtin_memcpy(p, &d, 2);
            for (uint32_t j = 0; j < 32; ++j) reinterpret_cast<int8_t *>(p + 2)[j] = (int8_t)qi(rng);
        }
        for (float &v : hx) v = xf(rng);
        const size_t xb = hx.size() * sizeof(float), ob = (size_t)tokens * m * sizeof(float);
        ck(hipMalloc(&w, hw.size()), "w"); ck(hipMalloc(&x, xb), "x");
        ck(hipMalloc(&xq, hx.size()), "xq");
        ck(hipMalloc(&xs, (size_t)tokens * blocks * sizeof(float)), "xs");
        ck(hipMalloc(&ref, ob), "ref"); ck(hipMalloc(&got, ob), "got");
        ck(hipMemcpy(w, hw.data(), hw.size(), hipMemcpyHostToDevice), "copy w");
        ck(hipMemcpy(x, hx.data(), xb, hipMemcpyHostToDevice), "copy x");
    }
    ~Buffers() { hipFree(w); hipFree(x); hipFree(xq); hipFree(xs); hipFree(ref); hipFree(got); }
};

static void launch_ref(Buffers &b) {
    q8f32_reference<5><<<dim3((b.out_dim + 7u) / 8u), 256>>>(b.ref, b.w, b.x, b.blocks, b.out_dim, b.row_bytes);
}
static void launch_dp(Buffers &b, uint32_t threads = 256u) {
    quantize_q8_blocks<<<dim3((Buffers::tokens * b.blocks + 7u) / 8u), 256>>>(b.x, b.xq, b.xs, Buffers::tokens, b.blocks);
    const uint32_t rows = threads / 32u;
    q8q8_dp4a<5><<<dim3((b.out_dim + rows - 1u) / rows), threads>>>(b.got, b.w, b.xq, b.xs, b.blocks, b.out_dim, b.row_bytes);
}
static float elapsed(Buffers &b, bool dp, int iters, uint32_t threads = 256u) {
    hipEvent_t a, z; ck(hipEventCreate(&a), "event"); ck(hipEventCreate(&z), "event");
    for (int i = 0; i < 3; ++i) dp ? launch_dp(b, threads) : launch_ref(b);
    ck(hipDeviceSynchronize(), "warm"); ck(hipEventRecord(a), "record");
    for (int i = 0; i < iters; ++i) dp ? launch_dp(b, threads) : launch_ref(b);
    ck(hipEventRecord(z), "record"); ck(hipEventSynchronize(z), "sync");
    float ms = 0; ck(hipEventElapsedTime(&ms, a, z), "elapsed");
    hipEventDestroy(a); hipEventDestroy(z); return ms / iters;
}
static void run(const char *name, uint32_t k, uint32_t m) {
    Buffers b(k, m); launch_ref(b); launch_dp(b); ck(hipDeviceSynchronize(), "correctness");
    const size_t n = (size_t)Buffers::tokens * m;
    std::vector<float> a(n), z(n);
    ck(hipMemcpy(a.data(), b.ref, n * sizeof(float), hipMemcpyDeviceToHost), "read ref");
    ck(hipMemcpy(z.data(), b.got, n * sizeof(float), hipMemcpyDeviceToHost), "read got");
    double mae = 0, rmse = 0, max_abs = 0, ref_rms = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double)z[i] - a[i]; mae += fabs(d); rmse += d*d;
        ref_rms += (double)a[i]*a[i]; max_abs = std::max(max_abs, fabs(d));
    }
    mae /= n; rmse = sqrt(rmse/n); ref_rms = sqrt(ref_rms/n);
    const float r = elapsed(b, false, 20);
    const float d4 = elapsed(b, true, 20, 128u);
    const float d8 = elapsed(b, true, 20, 256u);
    const float d16 = elapsed(b, true, 20, 512u);
    const float d32 = elapsed(b, true, 20, 1024u);
    const float d = std::min({d4, d8, d16, d32});
    std::printf("shape=%s k=%u m=%u ref_ms=%.4f dp4a_rpb4=%.4f rpb8=%.4f rpb16=%.4f rpb32=%.4f best_speedup=%.2fx mae=%.6g nrmse=%.6g max_abs=%.6g\n",
                name, k, m, r, d4, d8, d16, d32, r/d, mae,
                rmse/std::max(1e-12,ref_rms), max_abs);
}
int main() {
    run("attn_output_a", 4096, 8192);
    run("attn_output_b", 8192, 4096);
    run("attn_q_b", 1024, 32768);
    run("fc", 12288, 4096);
}
