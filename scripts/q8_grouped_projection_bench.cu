// Standalone correctness and timing harness for DS4's grouped Q8_0
// attention-output A projection. It compares the shipping F32 shared-X tile
// with the dormant tile-local Q8->FP16 rocWMMA implementation without loading
// a model or allocating a persistent expanded-weight cache.
//
// Build:
//   /opt/rocm/bin/hipcc -O3 -ffast-math -fno-finite-math-only \
//     --offload-arch=gfx1151 scripts/q8_grouped_projection_bench.cu \
//     -o /tmp/q8_grouped_projection_bench

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static void hip_check(hipError_t rc, const char *where) {
    if (rc != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, hipGetErrorString(rc));
        std::exit(1);
    }
}

__device__ static float warp_sum_f32(float v) {
    for (unsigned d = 16; d; d >>= 1) v += __shfl_down(v, d, 32);
    return v;
}

__device__ static float q8_scale(const unsigned char *blk) {
    const float d = __half2float(*reinterpret_cast<const __half *>(blk));
    return __shfl(d, 0, 32);
}

template <uint32_t TOK_TILE, uint32_t BLOCKS_TILE>
__global__ static void sharedx_kernel(float *low, const unsigned char *w,
        const float *heads, uint32_t n_tokens, uint32_t n_groups,
        uint32_t n_blocks, uint32_t rank, uint64_t row_bytes) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x, lane = tid & 31u, wave = tid >> 5u;
    const uint32_t rows = blockDim.x >> 5u;
    const uint32_t row_blocks = (rank + rows - 1u) / rows;
    const uint32_t g = blockIdx.x / row_blocks;
    const uint32_t row = (blockIdx.x - g * row_blocks) * rows + wave;
    const uint32_t t0 = blockIdx.y * TOK_TILE;
    if (g >= n_groups || t0 >= n_tokens) return;
    const uint32_t group_dim = n_blocks * 32u;
    const bool valid = row < rank;
    const unsigned char *wr = w + ((uint64_t)g * rank + (valid ? row : 0u)) * row_bytes;
    float acc[TOK_TILE];
#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; ++u) acc[u] = 0.0f;
    for (uint32_t b0 = 0; b0 < n_blocks; b0 += BLOCKS_TILE) {
        const uint32_t bc = min(BLOCKS_TILE, n_blocks - b0);
        for (uint32_t j = tid; j < TOK_TILE * BLOCKS_TILE * 32u; j += blockDim.x) {
            const uint32_t u = j / (BLOCKS_TILE * 32u);
            const uint32_t r = j - u * BLOCKS_TILE * 32u;
            const uint32_t bb = r >> 5u, k = r & 31u, t = t0 + u;
            const uint64_t off = ((uint64_t)t * n_groups + g) * group_dim
                               + ((uint64_t)(b0 + bb) << 5u) + k;
            shx[j] = (t < n_tokens && bb < bc) ? heads[off] : 0.0f;
        }
        __syncthreads();
        if (valid) {
            for (uint32_t bb = 0; bb < bc; ++bb) {
                const unsigned char *blk = wr + (uint64_t)(b0 + bb) * 34u;
                const float wv = q8_scale(blk) * (float)reinterpret_cast<const int8_t *>(blk + 2)[lane];
#pragma unroll
                for (uint32_t u = 0; u < TOK_TILE; ++u)
                    acc[u] += wv * shx[(u * BLOCKS_TILE + bb) * 32u + lane];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; ++u) acc[u] = warp_sum_f32(acc[u]);
    if (lane == 0 && valid) {
#pragma unroll
        for (uint32_t u = 0; u < TOK_TILE; ++u) {
            const uint32_t t = t0 + u;
            if (t < n_tokens) low[((uint64_t)t * n_groups + g) * rank + row] = acc[u];
        }
    }
}

template <int TILES_N = 8, int BM = 16, int BN = 16, int BK = 16>
__global__ static void wmma_kernel(float *low, const unsigned char *w,
        const float *heads, uint32_t n_tokens, uint32_t n_groups,
        uint32_t group_dim, uint32_t rank, uint64_t row_bytes) {
    extern __shared__ unsigned char raw[];
    __half *shA = reinterpret_cast<__half *>(raw);
    __half *shB = shA + BM * BK;
    float *shC = reinterpret_cast<float *>(shB + TILES_N * BK * BN);
    const uint32_t tid = threadIdx.x, wave = tid >> 5u;
    const uint32_t rtpg = (rank + TILES_N * BN - 1u) / (TILES_N * BN);
    const uint32_t g = blockIdx.x / rtpg;
    const uint32_t row0 = (blockIdx.x - g * rtpg) * TILES_N * BN;
    const uint32_t t0 = blockIdx.y * BM;
    if (g >= n_groups) return;
    using fa = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, __half, rocwmma::row_major>;
    using fb = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, __half, rocwmma::row_major>;
    using fc = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    fa a; fb b; fc acc;
    if (wave < TILES_N) rocwmma::fill_fragment(acc, 0.0f);
    for (uint32_t k0 = 0; k0 < group_dim; k0 += BK) {
        for (uint32_t j = tid; j < BM * BK; j += blockDim.x) {
            const uint32_t m = j / BK, kk = j % BK, t = t0 + m, k = k0 + kk;
            shA[j] = (t < n_tokens && k < group_dim)
                ? __float2half(heads[((uint64_t)t * n_groups + g) * group_dim + k])
                : __float2half(0.0f);
        }
        for (uint32_t j = tid; j < TILES_N * BK * BN; j += blockDim.x) {
            const uint32_t tn = j / (BK * BN), rem = j % (BK * BN);
            const uint32_t kk = rem / BN, nn = rem % BN;
            const uint32_t row = row0 + tn * BN + nn, k = k0 + kk;
            if (row < rank && k < group_dim) {
                const unsigned char *blk = w + ((uint64_t)g * rank + row) * row_bytes
                                           + (uint64_t)(k >> 5u) * 34u;
                shB[j] = __float2half(__half2float(*reinterpret_cast<const __half *>(blk))
                                     * (float)reinterpret_cast<const int8_t *>(blk + 2)[k & 31u]);
            } else shB[j] = __float2half(0.0f);
        }
        __syncthreads();
        if (wave < TILES_N) {
            rocwmma::load_matrix_sync(a, shA, BK);
            rocwmma::load_matrix_sync(b, shB + wave * BK * BN, BN);
            rocwmma::mma_sync(acc, a, b, acc);
        }
        __syncthreads();
    }
    if (wave < TILES_N)
        rocwmma::store_matrix_sync(shC + wave * BM * BN, acc, BN, rocwmma::mem_row_major);
    __syncthreads();
    for (uint32_t j = tid; j < TILES_N * BM * BN; j += blockDim.x) {
        const uint32_t tn = j / (BM * BN), rem = j % (BM * BN);
        const uint32_t m = rem / BN, nn = rem % BN;
        const uint32_t t = t0 + m, row = row0 + tn * BN + nn;
        if (t < n_tokens && row < rank)
            low[((uint64_t)t * n_groups + g) * rank + row] = shC[j];
    }
}

struct Buffers {
    uint32_t tokens, groups, dim, rank, blocks;
    uint64_t row_bytes;
    std::vector<unsigned char> w;
    std::vector<float> x;
    float *dw = nullptr, *dx = nullptr, *da = nullptr, *db = nullptr;
    Buffers(uint32_t t, uint32_t g, uint32_t d, uint32_t r)
        : tokens(t), groups(g), dim(d), rank(r), blocks(d / 32), row_bytes((uint64_t)blocks * 34u),
          w((uint64_t)g * r * row_bytes), x((uint64_t)t * g * d) {
        std::mt19937 rng(7); std::uniform_real_distribution<float> rf(-1.0f, 1.0f);
        std::uniform_int_distribution<int> ri(-127, 127);
        for (uint64_t row = 0; row < (uint64_t)groups * rank; ++row)
            for (uint32_t b = 0; b < blocks; ++b) {
                unsigned char *p = w.data() + row * row_bytes + (uint64_t)b * 34u;
                *reinterpret_cast<__half *>(p) = __float2half(0.002f + 0.02f * std::fabs(rf(rng)));
                for (int i = 0; i < 32; ++i) reinterpret_cast<int8_t *>(p + 2)[i] = (int8_t)ri(rng);
            }
        for (float &v : x) v = rf(rng);
        const size_t out = (size_t)tokens * groups * rank * sizeof(float);
        hip_check(hipMalloc(&dw, w.size()), "malloc w"); hip_check(hipMalloc(&dx, x.size() * sizeof(float)), "malloc x");
        hip_check(hipMalloc(&da, out), "malloc a"); hip_check(hipMalloc(&db, out), "malloc b");
        hip_check(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice), "copy w");
        hip_check(hipMemcpy(dx, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice), "copy x");
    }
    ~Buffers() { hipFree(dw); hipFree(dx); hipFree(da); hipFree(db); }
};

static void launch_shared(Buffers &b) {
    dim3 grid(b.groups * ((b.rank + 31u) / 32u), (b.tokens + 31u) / 32u);
    sharedx_kernel<32, 16><<<grid, 1024, 65536>>>(b.da, reinterpret_cast<unsigned char *>(b.dw), b.dx,
        b.tokens, b.groups, b.blocks, b.rank, b.row_bytes);
}

static void launch_wmma(Buffers &b) {
    constexpr uint32_t rows = 8u * 16u;
    dim3 grid(b.groups * ((b.rank + rows - 1u) / rows), (b.tokens + 15u) / 16u);
    constexpr size_t smem = 16u * 16u * sizeof(__half) + 8u * 16u * 16u * sizeof(__half)
                          + 8u * 16u * 16u * sizeof(float);
    wmma_kernel<<<grid, 256, smem>>>(b.db, reinterpret_cast<unsigned char *>(b.dw), b.dx,
        b.tokens, b.groups, b.dim, b.rank, b.row_bytes);
}

static float elapsed_ms(Buffers &b, bool wmma, int iters) {
    hipEvent_t s, e; hipEventCreate(&s); hipEventCreate(&e);
    for (int i = 0; i < 3; ++i) wmma ? launch_wmma(b) : launch_shared(b);
    hip_check(hipDeviceSynchronize(), "warmup"); hipEventRecord(s);
    for (int i = 0; i < iters; ++i) wmma ? launch_wmma(b) : launch_shared(b);
    hipEventRecord(e); hipEventSynchronize(e); float ms = 0; hipEventElapsedTime(&ms, s, e);
    hipEventDestroy(s); hipEventDestroy(e); return ms / iters;
}

int main() {
    Buffers small(17, 2, 64, 32); launch_shared(small); launch_wmma(small);
    hip_check(hipDeviceSynchronize(), "correctness kernels");
    const size_t n = (size_t)small.tokens * small.groups * small.rank;
    std::vector<float> a(n), b(n); hipMemcpy(a.data(), small.da, n * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(b.data(), small.db, n * sizeof(float), hipMemcpyDeviceToHost);
    double max_abs = 0, max_rel = 0, rms = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double)a[i] - b[i]); max_abs = std::max(max_abs, d);
        max_rel = std::max(max_rel, d / std::max(1e-6, std::fabs((double)a[i]))); rms += d * d;
    }
    rms = std::sqrt(rms / n);
    std::printf("correctness sharedx-vs-wmma max_abs=%.6g max_rel=%.6g rms=%.6g\n", max_abs, max_rel, rms);
    for (uint32_t t : {128u, 256u, 512u, 1024u, 2048u}) {
        Buffers real(t, 16, 4096, 512); const int iters = t <= 512 ? 10 : 4;
        const float fs = elapsed_ms(real, false, iters), fw = elapsed_ms(real, true, iters);
        std::printf("tokens=%u sharedx_ms=%.4f wmma_ms=%.4f wmma_change=%+.1f%%\n",
                    t, fs, fw, 100.0f * (fw / fs - 1.0f));
    }
    return (max_rel > 0.10 && max_abs > 0.10) ? 2 : 0;
}
