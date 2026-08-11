// Standalone gfx1151 oracle for the DSpark N=5 Q8_0 indexer projection.
// Compares DS4's current block-per-output-row DP4A schedule with a persistent
// wavefront-split-K schedule inspired by vLLM's skinny-GEMM design.
//
// Build:
//   /opt/rocm/bin/hipcc -O3 -ffast-math -fno-finite-math-only \
//     --offload-arch=gfx1151 scripts/dspark_q8_wvsplitk_bench.cu \
//     -o /tmp/dspark_q8_wvsplitk_bench
#include "../ds4_rocm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

#define HIP_OK(expr) do {                                                   \
    hipError_t e_ = (expr);                                                 \
    if (e_ != hipSuccess) {                                                 \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,            \
                     hipGetErrorString(e_));                                \
        std::exit(1);                                                       \
    }                                                                       \
} while (0)

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
    for (uint32_t mask = 16; mask != 0; mask >>= 1) {
        v += __shfl_xor(v, mask, 32);
    }
    return v;
}

__global__ void quantize_q8_0_f32(
        int8_t *xq, float *xs, const float *x, uint32_t in_dim) {
    const uint32_t b = blockIdx.x;
    const uint32_t t = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    const uint32_t blocks = in_dim / 32;
    const float xv = x[(uint64_t)t * in_dim + b * 32 + lane];
    float a = fabsf(xv);
#pragma unroll
    for (uint32_t mask = 16; mask != 0; mask >>= 1) {
        a = fmaxf(a, __shfl_xor(a, mask, 32));
    }
    const float d = __shfl(a, 0, 32) / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (lane == 0) xs[(uint64_t)t * blocks + b] = d;
    int q = __float2int_rn(xv * id);
    q = q > 127 ? 127 : (q < -128 ? -128 : q);
    xq[((uint64_t)t * blocks + b) * 32 + lane] = (int8_t)q;
}

template <uint32_t NTOK>
__global__ void current_dp4a(
        float *out, const uint8_t *w, const int8_t *xq, const float *xs,
        uint32_t blocks, uint32_t out_dim) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t row = blockIdx.x * rows_per_block + wave;
    if (row >= out_dim) return;
    const uint8_t *wr = w + (uint64_t)row * blocks * 34;
    float acc[NTOK] = {};
    for (uint32_t b = lane; b < blocks; b += 32) {
        const uint8_t *bp = wr + (uint64_t)b * 34;
        const float ws = __half2float(*reinterpret_cast<const __half *>(bp));
        int32_t qw[8];
#pragma unroll
        for (uint32_t j = 0; j < 8; ++j) {
            __builtin_memcpy(&qw[j], bp + 2 + 4 * j, sizeof(qw[j]));
        }
#pragma unroll
        for (uint32_t t = 0; t < NTOK; ++t) {
            const uint64_t xb = (uint64_t)t * blocks + b;
            const int8_t *q = xq + xb * 32;
            int32_t dot = 0;
#pragma unroll
            for (uint32_t j = 0; j < 8; ++j) {
                int32_t xw;
                __builtin_memcpy(&xw, q + 4 * j, sizeof(xw));
                dot = __dp4a(qw[j], xw, dot);
            }
            acc[t] += ws * xs[xb] * (float)dot;
        }
    }
#pragma unroll
    for (uint32_t t = 0; t < NTOK; ++t) acc[t] = wave_sum(acc[t]);
    if (lane == 0) {
#pragma unroll
        for (uint32_t t = 0; t < NTOK; ++t) {
            out[(uint64_t)t * out_dim + row] = acc[t];
        }
    }
}

template <uint32_t YTILE>
__launch_bounds__(512, 1)
__global__ void persistent_dp4a_n5(
        float *out, const uint8_t *w, const int8_t *xq, const float *xs,
        uint32_t blocks, uint32_t out_dim) {
    __shared__ __align__(16) int8_t sxq[5 * 1536];
    __shared__ float sxs[5 * 48];
    const uint32_t lane = threadIdx.x;
    const uint32_t wave = threadIdx.y;
    const uint32_t linear = wave * 32 + lane;
    const uint32_t qbytes = 5 * blocks * 32;
    for (uint32_t i = linear; i < qbytes; i += 512) sxq[i] = xq[i];
    for (uint32_t i = linear; i < 5 * blocks; i += 512) sxs[i] = xs[i];
    __syncthreads();

    uint32_t row0 = (blockIdx.x * 16 + wave) * YTILE;
    const uint32_t row_stride = gridDim.x * 16 * YTILE;
    while (row0 < out_dim) {
        float acc[5][YTILE] = {};
        for (uint32_t b = lane; b < blocks; b += 32) {
            int32_t xword[5][8];
#pragma unroll
            for (uint32_t t = 0; t < 5; ++t) {
#pragma unroll
                for (uint32_t j = 0; j < 8; ++j) {
                    __builtin_memcpy(&xword[t][j],
                        sxq + ((uint64_t)t * blocks + b) * 32 + 4 * j,
                        sizeof(int32_t));
                }
            }
#pragma unroll
            for (uint32_t y = 0; y < YTILE; ++y) {
                const uint32_t row = min(row0 + y, out_dim - 1);
                const uint8_t *bp = w + ((uint64_t)row * blocks + b) * 34;
                const float ws = __half2float(
                    *reinterpret_cast<const __half *>(bp));
                int32_t qw[8];
#pragma unroll
                for (uint32_t j = 0; j < 8; ++j) {
                    __builtin_memcpy(&qw[j], bp + 2 + 4 * j, sizeof(qw[j]));
                }
#pragma unroll
                for (uint32_t t = 0; t < 5; ++t) {
                    int32_t dot = 0;
#pragma unroll
                    for (uint32_t j = 0; j < 8; ++j) {
                        dot = __dp4a(qw[j], xword[t][j], dot);
                    }
                    acc[t][y] += ws * sxs[t * blocks + b] * (float)dot;
                }
            }
        }
#pragma unroll
        for (uint32_t t = 0; t < 5; ++t) {
#pragma unroll
            for (uint32_t y = 0; y < YTILE; ++y) {
                acc[t][y] = wave_sum(acc[t][y]);
            }
        }
        if (lane == 0) {
#pragma unroll
            for (uint32_t t = 0; t < 5; ++t) {
#pragma unroll
                for (uint32_t y = 0; y < YTILE; ++y) {
                    if (row0 + y < out_dim) {
                        out[(uint64_t)t * out_dim + row0 + y] = acc[t][y];
                    }
                }
            }
        }
        row0 += row_stride;
    }
}

template <typename Launch>
double time_kernel(Launch launch, int warmup = 20, int iters = 200) {
    for (int i = 0; i < warmup; ++i) launch();
    HIP_OK(hipDeviceSynchronize());
    hipEvent_t begin, end;
    HIP_OK(hipEventCreate(&begin));
    HIP_OK(hipEventCreate(&end));
    HIP_OK(hipEventRecord(begin));
    for (int i = 0; i < iters; ++i) launch();
    HIP_OK(hipEventRecord(end));
    HIP_OK(hipEventSynchronize(end));
    float ms = 0;
    HIP_OK(hipEventElapsedTime(&ms, begin, end));
    HIP_OK(hipEventDestroy(begin));
    HIP_OK(hipEventDestroy(end));
    return ms / iters;
}

int main() {
    constexpr uint32_t ntok = 5, in_dim = 1536, out_dim = 8192;
    constexpr uint32_t blocks = in_dim / 32;
    hipDeviceProp_t prop{};
    HIP_OK(hipGetDeviceProperties(&prop, 0));
    std::printf("device=%s arch=%s CUs=%d shape=N%u K%u M%u\n",
                prop.name, prop.gcnArchName, prop.multiProcessorCount,
                ntok, in_dim, out_dim);

    std::mt19937 rng(20260811);
    std::normal_distribution<float> dist(0.0f, 0.25f);
    std::vector<float> hx((size_t)ntok * in_dim);
    std::vector<uint8_t> hw((size_t)out_dim * blocks * 34);
    for (float &v : hx) v = dist(rng);
    for (uint32_t r = 0; r < out_dim; ++r) {
        for (uint32_t b = 0; b < blocks; ++b) {
            uint8_t *p = hw.data() + ((uint64_t)r * blocks + b) * 34;
            const __half scale = __float2half(0.01f + 0.02f * std::fabs(dist(rng)));
            std::memcpy(p, &scale, sizeof(scale));
            for (uint32_t j = 0; j < 32; ++j) p[2 + j] = (uint8_t)(int8_t)(rng() % 255 - 127);
        }
    }

    float *dx, *dout0, *dout1;
    int8_t *dxq;
    float *dxs;
    uint8_t *dw;
    HIP_OK(hipMalloc(&dx, hx.size() * sizeof(float)));
    HIP_OK(hipMalloc(&dw, hw.size()));
    HIP_OK(hipMalloc(&dxq, (size_t)ntok * in_dim));
    HIP_OK(hipMalloc(&dxs, (size_t)ntok * blocks * sizeof(float)));
    HIP_OK(hipMalloc(&dout0, (size_t)ntok * out_dim * sizeof(float)));
    HIP_OK(hipMalloc(&dout1, (size_t)ntok * out_dim * sizeof(float)));
    HIP_OK(hipMemcpy(dx, hx.data(), hx.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dw, hw.data(), hw.size(), hipMemcpyHostToDevice));
    quantize_q8_0_f32<<<dim3(blocks, ntok), 32>>>(dxq, dxs, dx, in_dim);
    HIP_OK(hipGetLastError());

    auto current = [&] {
        current_dp4a<5><<<(out_dim + 7) / 8, 256>>>(
            dout0, dw, dxq, dxs, blocks, out_dim);
    };
    const double t0 = time_kernel(current);
    double best = 1.0e30;
    uint32_t best_y = 0, best_grid = 0;
    for (uint32_t grid_mul : {1u, 2u, 4u}) {
        const uint32_t grid = prop.multiProcessorCount * grid_mul;
        const double t1 = time_kernel([&] {
            persistent_dp4a_n5<1><<<grid, dim3(32, 16)>>>(
                dout1, dw, dxq, dxs, blocks, out_dim);
        });
        const double t2 = time_kernel([&] {
            persistent_dp4a_n5<2><<<grid, dim3(32, 16)>>>(
                dout1, dw, dxq, dxs, blocks, out_dim);
        });
        const double t4 = time_kernel([&] {
            persistent_dp4a_n5<4><<<grid, dim3(32, 16)>>>(
                dout1, dw, dxq, dxs, blocks, out_dim);
        });
        std::printf("grid=%u y1_ms=%.6f y2_ms=%.6f y4_ms=%.6f "
                    "speedups=%.3fx/%.3fx/%.3fx\n",
                    grid, t1, t2, t4, t0 / t1, t0 / t2, t0 / t4);
        for (const auto &v : {std::pair<uint32_t, double>{1, t1},
                              {2, t2}, {4, t4}}) {
            if (v.second < best) {
                best = v.second;
                best_y = v.first;
                best_grid = grid;
            }
        }
    }
    if (best_y == 1) {
        persistent_dp4a_n5<1><<<best_grid, dim3(32, 16)>>>(
            dout1, dw, dxq, dxs, blocks, out_dim);
    } else if (best_y == 2) {
        persistent_dp4a_n5<2><<<best_grid, dim3(32, 16)>>>(
            dout1, dw, dxq, dxs, blocks, out_dim);
    } else {
        persistent_dp4a_n5<4><<<best_grid, dim3(32, 16)>>>(
            dout1, dw, dxq, dxs, blocks, out_dim);
    }
    HIP_OK(hipDeviceSynchronize());
    std::vector<float> h0((size_t)ntok * out_dim), h1(h0.size());
    HIP_OK(hipMemcpy(h0.data(), dout0, h0.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(h1.data(), dout1, h1.size() * sizeof(float), hipMemcpyDeviceToHost));
    float max_abs = 0, max_rel = 0;
    size_t bit_diff = 0;
    for (size_t i = 0; i < h0.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(h0[i] - h1[i]));
        max_rel = std::max(max_rel, std::fabs(h0[i] - h1[i]) /
                                   std::max(1.0e-6f, std::fabs(h0[i])));
        uint32_t a, b;
        std::memcpy(&a, &h0[i], 4);
        std::memcpy(&b, &h1[i], 4);
        bit_diff += a != b;
    }
    std::printf("current_ms=%.6f best_ms=%.6f best_y=%u best_grid=%u "
                "speedup=%.3fx\n", t0, best, best_y, best_grid, t0 / best);
    std::printf("fidelity_best bit_diff=%zu/%zu max_abs=%.9g max_rel=%.9g\n",
                bit_diff, h0.size(), max_abs, max_rel);
    return 0;
}
