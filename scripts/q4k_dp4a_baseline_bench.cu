// Stage -0A: ds4 Q4_K DP4A baseline characterization (no WMMA yet).
//
// Standalone harness that copies ds4's production Q4_K/Q8_K block layouts
// and its tile4/tile8 DP4A routed-MoE gate/up kernels verbatim (source:
// ~/Desktop/cc/ds4-upstream/rocm/ds4_rocm_moe.cuh, dev_dot_q4_K_q8_K_block4
// at :291, dev_dot_q4_K_q8_K_block8 at :321,
// moe_gate_up_mid_q4K_expert_tile4_row32_kernel at :1671,
// moe_gate_up_mid_q4K_expert_tile8_row32_kernel at :1754; launch config
// from ds4_rocm_moe_launch.cuh:1134-1152), MIT license, ds4 (DwarfStar) by
// antirez et al.
//
// Purpose: measure how much reuse ds4's existing tile8 DP4A kernel already
// captures across routed-token buckets, BEFORE any WMMA code is written.
// See ds4-strix-halo-tp/docs/Q4K-WMMA-PLAN.md, "Stage -0A".
//
// Does NOT touch ds4-upstream/ - new file only. Build with the exact
// production ROCM_CFLAGS (ds4-upstream/Makefile:43) plus -Wall -Wextra:
//
//   /opt/rocm-7.2.0/bin/hipcc -O3 -ffast-math -g -fno-finite-math-only \
//     -pthread -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument \
//     --offload-arch=gfx1151 -Wall -Wextra \
//     q4k_dp4a_baseline_bench.cu -o q4k_dp4a_baseline_bench -lm -pthread
//
// Usage: ./q4k_dp4a_baseline_bench [--skewed]
//   Sweeps buckets {1,4,5,6,8,16,17,22,32,33,48,64,96,128} at
//   K=4096 (xq_blocks=16), expert_mid_dim=2048, 256 experts / 6 used,
//   both tile4 and tile8, reporting kernel-only time (HIP events,
//   5 warmup + 20 timed iterations, median/p10/p90).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <string>
#include <cmath>

#define MASK_T uint64_t
#define CUDA_QK_K 256

// __dp4a HIP compat shim, verbatim from ds4-upstream/ds4_rocm.h:117-123 -
// gfx11-class AMD GPUs expose this as v_dot4_i32_i8 via amd_mixed_dot.
static __device__ __forceinline__ int32_t __dp4a(int32_t a, int32_t b, int32_t c) {
    union ds4_i8x4_bits { int32_t i; char4 v; } av, bv;
    av.i = a;
    bv.i = b;
    return amd_mixed_dot(av.v, bv.v, c, false);
}

// ---------------------------------------------------------------------
// Verbatim from ds4-upstream/ds4_rocm.cu:65-83
// ---------------------------------------------------------------------
typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[CUDA_QK_K / 2];
} cuda_block_q4_K;

typedef struct {
    float d;
    int8_t qs[CUDA_QK_K];
    int16_t bsums[CUDA_QK_K / 16];
} cuda_block_q8_K;

static_assert(sizeof(cuda_block_q4_K) == 144, "cuda_block_q4_K layout drifted from production");
static_assert(sizeof(cuda_block_q8_K) == 292, "cuda_block_q8_K layout drifted from production");
static_assert(offsetof(cuda_block_q4_K, scales) == 4, "cuda_block_q4_K.scales offset drifted");
static_assert(offsetof(cuda_block_q4_K, qs) == 16, "cuda_block_q4_K.qs offset drifted");
static_assert(offsetof(cuda_block_q8_K, bsums) == 260, "cuda_block_q8_K.bsums offset drifted");

// ---------------------------------------------------------------------
// Stage -0B: Q4_K x Q8_1 integer-WMMA arm.
//
// Ported/adapted from llama.cpp commit 1a064ab0921238c1daa397d6f4a900ef33884de2
// (MIT, Copyright (c) Georgi Gerganov and contributors):
//   ggml/src/ggml-cuda/mmq-load-tiles.cuh:612-690
//   ggml/src/ggml-cuda/mmq-vec-dot.cuh:315-359
//   ggml/src/ggml-cuda/mma.cuh:81-93,530-560,803-834,1310-1333
//   ggml/src/ggml-cuda/quantize.cu:458-549
// I=64 is an experiment requested by Q4K-WMMA-PLAN.md. llama.cpp's gfx1151
// table (mmq-config-rdna4.cuh:108-119) uses I=128; this is not claimed to
// be an upstream configuration.
// ---------------------------------------------------------------------

struct q8_1_mmq_block {
    half2 ds[4];                 // (d, unquantized sum), one pair per 32 values
    int8_t qs[128];
};
static_assert(sizeof(q8_1_mmq_block) == 144, "unexpected Q8_1 MMQ block size");

// Q8_K already contains int8 quants, so conversion is a repack rather than
// requantizing floats. The normal mode necessarily rounds d and d*sum to
// fp16, exactly as MMQ_Q8_1_DS_LAYOUT_DS4 does. Blocks are transposed as
// [K/128][token], matching mul_mat_q_process_tile's global-memory layout.
__global__ static void q8_K_to_q8_1_mmq_kernel(
        const cuda_block_q8_K *src, q8_1_mmq_block *dst,
        uint32_t ntokens, uint32_t xq_blocks) {
    const uint32_t token = blockIdx.x;
    const uint32_t q8k_b = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    if (token >= ntokens || q8k_b >= xq_blocks || lane >= 32) return;
    const cuda_block_q8_K &s = src[(uint64_t)token*xq_blocks + q8k_b];
    #pragma unroll
    for (uint32_t half = 0; half < 2; ++half) {
        q8_1_mmq_block &d = dst[((uint64_t)q8k_b*2 + half)*ntokens + token];
        const uint32_t i = half*128 + lane*4;
        *reinterpret_cast<int32_t *>(d.qs + lane*4) =
            *reinterpret_cast<const int32_t *>(s.qs + i);
        if ((lane & 7u) == 0) {
            const uint32_t sub = half*4 + lane/8;
            const int32_t isum = (int32_t)s.bsums[2*sub] + s.bsums[2*sub + 1];
            d.ds[lane/8] = __floats2half2_rn(s.d, s.d*(float)isum);
        }
    }
}

using i32x4 = int32_t __attribute__((ext_vector_type(4)));
using i32x8 = int32_t __attribute__((ext_vector_type(8)));

struct wmma_ab_frag { int32_t x[8]; };

static __device__ __forceinline__ wmma_ab_frag load_rdna3_mirrored_16x8(
        const int32_t *p, int stride) {
    // mma.cuh:811-834: DATA_LAYOUT_I_MAJOR_MIRRORED, 8 int/lane, two
    // 16-byte copies from columns 0..3 and 4..7 of lane%16's row.
    wmma_ab_frag r;
    const int row = (int)(threadIdx.x & 15u);
    *reinterpret_cast<i32x4 *>(&r.x[0]) = *reinterpret_cast<const i32x4 *>(p + row*stride + 0);
    *reinterpret_cast<i32x4 *>(&r.x[4]) = *reinterpret_cast<const i32x4 *>(p + row*stride + 4);
    return r;
}

static __device__ __forceinline__ i32x8 wmma_i8_16x16x16(
        const wmma_ab_frag &a, const wmma_ab_frag &b, i32x8 c) {
    // mma.cuh:1326-1332: RDNA3 consumes each mirrored 16x8 fragment as
    // two int32x4 operands, issuing two K=16 WMMA instructions for K=32.
    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
        true, *reinterpret_cast<const i32x4 *>(&a.x[0]),
        true, *reinterpret_cast<const i32x4 *>(&b.x[0]), c, true);
    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
        true, *reinterpret_cast<const i32x4 *>(&a.x[4]),
        true, *reinterpret_cast<const i32x4 *>(&b.x[4]), c, true);
    return c;
}

static __device__ __forceinline__ int unpack_q4k_scales(const int *scales, int ksc) {
    return ((scales[(ksc%2) + (ksc != 0)] >> (4*(ksc & (ksc/2)))) & 0x0f0f0f0f) |
           ((scales[ksc/2] >> (2*(ksc%2))) & 0x30303030);
}

template<int J>
__global__ __launch_bounds__(256) static void q4k_wmma_i64_kernel(
        const cuda_block_q4_K *weights, const q8_1_mmq_block *acts,
        float *out, uint32_t ntokens, uint32_t xq_blocks, uint32_t nrows) {
    constexpr int I = 64;
    constexpr int XS = 76;       // mmq.cuh Q8_1 SRAM stride
    constexpr int YS = 36;       // 32 quant ints + four ds ints
    extern __shared__ int32_t smem[];
    int32_t *sy = smem;
    int32_t *sx = sy + J*YS;
    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const uint32_t row0 = blockIdx.x*I;
    const uint32_t tok0 = blockIdx.y*J;
    float acc[J/16][8] = {};

    for (uint32_t kb = 0; kb < xq_blocks; ++kb) {
        // Exact Q4_K MMA loader shape from mmq-load-tiles.cuh:622-690.
        if (wave < 8) {
            const int r = wave*8 + lane/4;
            const int txi = (lane%4)*8;
            if (row0 + r < nrows) {
                const cuda_block_q4_K &w = weights[(uint64_t)(row0+r)*xq_blocks + kb];
                #pragma unroll
                for (int q = 0; q < 8; ++q) {
                    const int v = *reinterpret_cast<const int32_t *>(w.qs + 4*(txi+q));
                    sx[r*XS + 16*((txi+q)/8) + (txi+q)%8 + 0] = (v >> 0) & 0x0f0f0f0f;
                    sx[r*XS + 16*((txi+q)/8) + (txi+q)%8 + 8] = (v >> 4) & 0x0f0f0f0f;
                }
            }
        }
        if (wave < 4) {
            const int r = wave*16 + lane/2;
            if (row0 + r < nrows) {
                const cuda_block_q4_K &w = weights[(uint64_t)(row0+r)*xq_blocks + kb];
                const int ksc = lane & 1;
                const int sc32 = unpack_q4k_scales(reinterpret_cast<const int *>(w.scales), ksc);
                const int m32  = unpack_q4k_scales(reinterpret_cast<const int *>(w.scales), ksc+2);
                const uint8_t *sc = reinterpret_cast<const uint8_t *>(&sc32);
                const uint8_t *mn = reinterpret_cast<const uint8_t *>(&m32);
                const float wd = __half2float(*reinterpret_cast<const half *>(&w.d));
                const float wm = __half2float(*reinterpret_cast<const half *>(&w.dmin));
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    reinterpret_cast<half2 *>(sx + r*XS + 64)[4*ksc+l] =
                        __floats2half2_rn(wd*sc[l], -wm*mn[l]);
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            for (int l = tid; l < J*YS; l += 256) {
                const int j = l/YS, e = l%YS;
                if (tok0 + j < ntokens) {
                    const int32_t *src = reinterpret_cast<const int32_t *>(
                        &acts[((uint64_t)kb*2+half)*ntokens + tok0+j]);
                    sy[l] = src[e];
                } else sy[l] = 0;
            }
            __syncthreads();
            if (wave < 4) {
                #pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    wmma_ab_frag A = load_rdna3_mirrored_16x8(sx + wave*16*XS + half*32 + kk*8, XS);
                    #pragma unroll
                    for (int j0 = 0; j0 < J; j0 += 16) {
                        wmma_ab_frag B = load_rdna3_mirrored_16x8(sy + j0*YS + 4 + kk*8, YS);
                        i32x8 c = {};
                        c = wmma_i8_16x16x16(A, B, c);
                        #pragma unroll
                        for (int l = 0; l < 8; ++l) {
                            // mma.cuh DATA_LAYOUT_J_MAJOR swaps the I-major
                            // accumulator coordinates.  On RDNA3, for a
                            // tile<16,16,int>, C[l] is therefore
                            //   i = 2*l + lane/16, j = lane%16
                            // (not i = lane%16, j = 2*l + lane/16).
                            const int i = 2*l + lane/16;
                            const int j = j0 + lane%16;
                            const float2 bd = __half22float2(reinterpret_cast<const half2 *>(sy + j*YS)[kk]);
                            const float2 ad = __half22float2(reinterpret_cast<const half2 *>(sx + (wave*16 + i)*XS + 64)[half*4+kk]);
                            acc[j0/16][l] += ad.x*bd.x*(float)c[l] + ad.y*bd.y;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
    if (wave < 4) {
        #pragma unroll
        for (int j0 = 0; j0 < J; j0 += 16) {
            #pragma unroll
            for (int l = 0; l < 8; ++l) {
                const uint32_t row = row0 + wave*16 + 2*l + lane/16;
                const uint32_t tok = tok0 + j0 + lane%16;
                if (row < nrows && tok < ntokens) out[(uint64_t)tok*nrows + row] = acc[j0/16][l];
            }
        }
    }
}

// ---------------------------------------------------------------------
// Verbatim from ds4-upstream/rocm/ds4_rocm_moe.cuh
// ---------------------------------------------------------------------
__device__ static float dev_f16_to_f32(uint16_t v) {
    return __half2float(*reinterpret_cast<const __half *>(&v));
}

__device__ static void dev_q4_K_get_scale_min(
        uint32_t j,
        const uint8_t *scales,
        uint8_t *d_out,
        uint8_t *m_out) {
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
        *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u);
    }
}

__device__ __forceinline__ static int32_t dev_dot_q4_32(const uint8_t *qs, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (*(const int32_t *)(qs + i) >> shift) & 0x0f0f0f0f;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

__device__ static void dev_dot_q4_K_q8_K_block4(
        const cuda_block_q4_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const cuda_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum[4] = {0, 0, 0, 0};
    int summs[4] = {0, 0, 0, 0};
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        for (uint32_t p = 0; p < n; p++) {
            if (!ys[p]) continue;
            summs[p] += (int)m * (int)(ys[p]->bsums[2u * j] + ys[p]->bsums[2u * j + 1u]);
            isum[p] += (int)sc * dev_dot_q4_32(x->qs + byte_off, ys[p]->qs + j * 32u, shift);
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        if (ys[p]) acc[p] += ys[p]->d * xd * (float)isum[p] - ys[p]->d * xmin * (float)summs[p];
    }
}

__device__ static void dev_dot_q4_K_q8_K_block8(
        const cuda_block_q4_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        for (uint32_t p = 0; p < n; p++) {
            if (!ys[p]) continue;
            summs[p] += (int)m * (int)(ys[p]->bsums[2u * j] + ys[p]->bsums[2u * j + 1u]);
            isum[p] += (int)sc * dev_dot_q4_32(x->qs + byte_off, ys[p]->qs + j * 32u, shift);
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        if (ys[p]) acc[p] += ys[p]->d * xd * (float)isum[p] - ys[p]->d * xmin * (float)summs[p];
    }
}

// Single-projection correctness reference. Its arithmetic is the existing
// dev_dot_q4_K_q8_K_block8 verbatim; only the surrounding output shell is new.
__global__ static void q4k_dp4a_single_reference_kernel(
        const cuda_block_q4_K *weights, const cuda_block_q8_K *acts,
        float *out, uint32_t ntokens, uint32_t xq_blocks, uint32_t nrows) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x*32u + threadIdx.x/8u;
    const uint32_t tok0 = blockIdx.y*8u;
    if (row >= nrows) return;
    float acc[8] = {};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        const cuda_block_q8_K *a[8] = {};
        #pragma unroll
        for (uint32_t p = 0; p < 8; ++p) if (tok0+p < ntokens) a[p] = acts + (uint64_t)(tok0+p)*xq_blocks+b;
        dev_dot_q4_K_q8_K_block8(weights + (uint64_t)row*xq_blocks+b,
            a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7], min(8u,ntokens-tok0),acc);
    }
    const uint32_t mask = 0xffu << (threadIdx.x & 24u);
    #pragma unroll
    for (uint32_t p = 0; p < 8; ++p) {
        float v = acc[p];
        for (int off=4; off>0; off>>=1) v += __shfl_down_sync((MASK_T)mask,v,off,8);
        if (lane == 0 && tok0+p < ntokens) out[(uint64_t)(tok0+p)*nrows+row] = v;
    }
}

__device__ static float quarter_warp_sum_f32(float v, uint32_t lane8) {
    uint32_t mask = 0xffu << (threadIdx.x & 24u);
    for (int offset = 4; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(static_cast<MASK_T>(mask), v, offset, 8);
    }
    (void)lane8;
    return v;
}

// ---------------------------------------------------------------------
// Verbatim from ds4-upstream/rocm/ds4_rocm_moe.cuh:1671 (tile4) and
// :1754 (tile8), unchanged including the documented generic-pointer LDS
// bug - this bench is measuring the SHIPPING kernel, not a fixed one.
// ---------------------------------------------------------------------
__global__ static void moe_gate_up_mid_q4K_expert_tile4_row32_kernel(
        float *gate_out, float *up_out, float *mid_out,
        const char *gate_base, const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs, const uint32_t *offsets, const uint32_t *counts,
        const uint32_t *tile_total, const uint32_t *tile_experts, const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint32_t xq_blocks, uint32_t expert_mid_dim, uint32_t n_expert,
        uint32_t max_count, uint32_t write_aux, float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t count = counts[expert];
    if (max_count != 0u && count >= max_count) return;
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[4][16];
    uint32_t pair[4] = {0, 0, 0, 0};
    uint32_t tok[4] = {0, 0, 0, 0};
    uint32_t slot[4] = {0, 0, 0, 0};
    const cuda_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= count) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_q4_K_q8_K_block4(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, gate);
        dev_dot_q4_K_q8_K_block4(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, up);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}

__global__ static void moe_gate_up_mid_q4K_expert_tile8_row32_kernel(
        float *gate_out, float *up_out, float *mid_out,
        const char *gate_base, const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs, const uint32_t *offsets, const uint32_t *counts,
        const uint32_t *tile_total, const uint32_t *tile_experts, const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint32_t xq_blocks, uint32_t expert_mid_dim, uint32_t n_expert,
        uint32_t max_count, uint32_t write_aux, float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t count = counts[expert];
    if (max_count != 0u && count >= max_count) return;
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= count) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_q4_K_q8_K_block8(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate);
        dev_dot_q4_K_q8_K_block8(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}

// ---------------------------------------------------------------------
// Harness (new code, not a production copy): synthetic routing metadata,
// valid Q4_K/Q8_K block generation, HIP-event timing.
// ---------------------------------------------------------------------

static void hip_check(hipError_t e, const char *what) {
    if (e != hipSuccess) {
        fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
        exit(2);
    }
}

// __float2half_rn returns a `half` VALUE; cuda_block_q4_K.d/.dmin are raw
// uint16_t bit-storage fields (matching ds4's production layout). Assigning
// a half directly to a uint16_t performs a VALUE conversion (0.02f -> 0),
// not a bit-reinterpret - discovered because it silently zeroed every
// synthetic weight block's d/dmin, making every WMMA-vs-DP4A "exact match"
// below vacuous (both arms were computing 0*sc*nibble - 0*m = 0 for every
// row). Every uint16_t d/dmin assignment must go through this.
static uint16_t f16_bits(float f) {
    half h = __float2half_rn(f);
    uint16_t bits;
    __builtin_memcpy(&bits, &h, sizeof(bits));
    return bits;
}

// Fill one Q4_K block with valid (non-degenerate) scale/min/nibble data.
static void fill_q4_K_block(cuda_block_q4_K *b, std::mt19937 &rng) {
    std::uniform_int_distribution<int> byte_dist(0, 255);
    b->d = f16_bits(0.02f + 0.01f * (byte_dist(rng) / 255.0f));
    b->dmin = f16_bits(0.01f + 0.005f * (byte_dist(rng) / 255.0f));
    for (int i = 0; i < 12; i++) b->scales[i] = (uint8_t)byte_dist(rng);
    for (int i = 0; i < CUDA_QK_K / 2; i++) b->qs[i] = (uint8_t)byte_dist(rng);
}

// Fill one Q8_K block with valid data (bsums consistent with qs so the
// dot product isn't operating on nonsense, though exact correctness
// isn't the point of this bench - Stage 0 owns correctness testing).
static void fill_q8_K_block(cuda_block_q8_K *b, std::mt19937 &rng) {
    std::uniform_int_distribution<int> q8_dist(-127, 127);
    b->d = 0.001f + 0.0005f * (q8_dist(rng) + 127) / 254.0f;
    for (int j = 0; j < CUDA_QK_K / 16; j++) {
        int32_t s = 0;
        for (int k = 0; k < 16; k++) {
            int8_t v = (int8_t)q8_dist(rng);
            b->qs[j * 16 + k] = v;
            s += v;
        }
        b->bsums[j] = (int16_t)s;
    }
}

struct BenchResult {
    double median_us, p10_us, p90_us;
};

struct WmmaBenchResult {
    BenchResult convert, core;
    double max_abs, max_rel;
    uint64_t bad;
    int J;
};

template<class Launch>
static BenchResult time_launch(Launch launch) {
    hipEvent_t start, stop;
    hip_check(hipEventCreate(&start), "event create start");
    hip_check(hipEventCreate(&stop), "event create stop");
    for (int i=0;i<5;++i) launch();
    hip_check(hipDeviceSynchronize(), "warmup sync");
    hip_check(hipGetLastError(), "warmup launch");
    std::vector<double> s(20);
    for (int i=0;i<20;++i) {
        hip_check(hipEventRecord(start), "record start"); launch();
        hip_check(hipEventRecord(stop), "record stop");
        hip_check(hipEventSynchronize(stop), "sync stop");
        float ms=0; hip_check(hipEventElapsedTime(&ms,start,stop), "elapsed"); s[i]=ms*1000.0;
    }
    std::sort(s.begin(),s.end());
    (void)hipEventDestroy(start); (void)hipEventDestroy(stop);
    return {s[10],s[2],s[17]};
}

static WmmaBenchResult time_wmma(uint32_t ntokens, uint32_t xq_blocks, uint32_t nrows, bool equalize_scale) {
    std::mt19937 rng(0x514b0000u + ntokens + (equalize_scale ? 0u : 0x9e3779b9u));
    std::vector<cuda_block_q4_K> hw((size_t)nrows*xq_blocks);
    std::vector<cuda_block_q8_K> ha((size_t)ntokens*xq_blocks);
    for (auto &b:hw) fill_q4_K_block(&b,rng);
    for (auto &b:ha) {
        fill_q8_K_block(&b,rng);
        if (equalize_scale) {
            // Scale-equalized mode: make the source DP4A scale exactly equal
            // to the fp16 value representable by Q8_1 ds4. This isolates
            // fragment, nibble and scale/min transcription errors from
            // format conversion. A power-of-two scale also makes
            // d*integer_sum exactly representable for the random test
            // range, so fp16 sum storage does not loosen the transcription
            // test. This mode alone cannot exercise fp16-sum precision loss
            // (Integration fact 3) since every scale is identical and
            // exactly representable - see the non-equalized pass below.
            b.d = 1.0f/1024.0f;
        }
    }
    cuda_block_q4_K *dw=nullptr; cuda_block_q8_K *da=nullptr;
    q8_1_mmq_block *dq=nullptr; float *do_wmma=nullptr,*do_ref=nullptr;
    const size_t osz=(size_t)ntokens*nrows*sizeof(float);
    hip_check(hipMalloc(&dw,hw.size()*sizeof(*dw)),"malloc wmma weights");
    hip_check(hipMalloc(&da,ha.size()*sizeof(*da)),"malloc wmma acts");
    hip_check(hipMalloc(&dq,(size_t)ntokens*xq_blocks*2*sizeof(*dq)),"malloc q8_1");
    hip_check(hipMalloc(&do_wmma,osz),"malloc wmma out");
    hip_check(hipMalloc(&do_ref,osz),"malloc ref out");
    hip_check(hipMemcpy(dw,hw.data(),hw.size()*sizeof(*dw),hipMemcpyHostToDevice),"copy weights");
    hip_check(hipMemcpy(da,ha.data(),ha.size()*sizeof(*da),hipMemcpyHostToDevice),"copy acts");
    auto cvt=[&](){q8_K_to_q8_1_mmq_kernel<<<dim3(ntokens,xq_blocks),32>>>(da,dq,ntokens,xq_blocks);};
    BenchResult cr=time_launch(cvt); cvt(); hip_check(hipDeviceSynchronize(),"conversion sync");
    // J=64/128 were compiled during the static desk-check but spilled 120 B
    // and 1172 B per lane respectively. Keep the runnable first draft on the
    // two zero-private-segment specializations; larger buckets tile in J.
    int J=ntokens<=16?16:32;
    dim3 grid((nrows+63)/64,(ntokens+J-1)/J);
    const size_t sh=((size_t)J*36+64*76)*sizeof(int32_t);
    auto core=[&](){
        switch(J){
            case 16:q4k_wmma_i64_kernel<16><<<grid,256,sh>>>(dw,dq,do_wmma,ntokens,xq_blocks,nrows);break;
            case 32:q4k_wmma_i64_kernel<32><<<grid,256,sh>>>(dw,dq,do_wmma,ntokens,xq_blocks,nrows);break;
            default:q4k_wmma_i64_kernel<32><<<grid,256,sh>>>(dw,dq,do_wmma,ntokens,xq_blocks,nrows);break;
        }
    };
    BenchResult wr=time_launch(core);
    q4k_dp4a_single_reference_kernel<<<dim3((nrows+31)/32,(ntokens+7)/8),256>>>(dw,da,do_ref,ntokens,xq_blocks,nrows);
    hip_check(hipDeviceSynchronize(),"correctness kernels"); hip_check(hipGetLastError(),"correctness launch");
    std::vector<float> ow((size_t)ntokens*nrows),orr(ow.size());
    hip_check(hipMemcpy(ow.data(),do_wmma,osz,hipMemcpyDeviceToHost),"copy wmma out");
    hip_check(hipMemcpy(orr.data(),do_ref,osz,hipMemcpyDeviceToHost),"copy ref out");
    // Equalized mode is exact (same power-of-two scale on every block, no
    // fp16-sum rounding), so it can use a tight tolerance. Non-equalized
    // mode has real per-block scale diversity and genuinely different
    // rounding paths (fp16 Q8_1 (d, d*sum) pair vs fp32 Q8_K accumulation)
    // between the two arms, so it needs a looser but still meaningful bound.
    const double atol = equalize_scale ? 1e-4 : 5e-2;
    const double rtol = equalize_scale ? 1e-4 : 5e-3;
    double ma=0,mr=0; uint64_t bad=0;
    for(size_t i=0;i<ow.size();++i){
        const double ae=fabs((double)ow[i]-orr[i]); const double re=ae/std::max(1e-30,fabs((double)orr[i]));
        ma=std::max(ma,ae); mr=std::max(mr,re);
        if(!std::isfinite(ow[i]) || ae>atol+rtol*fabs((double)orr[i])) ++bad;
    }
    if (getenv("Q4K_DEBUG_SAMPLE")) {
        fprintf(stderr, "[debug ntokens=%u eq=%d] sample outputs (idx, wmma, ref):\n", ntokens, (int)equalize_scale);
        for (size_t i = 0; i < ow.size() && i < 8; ++i)
            fprintf(stderr, "  [%zu] wmma=%.9g ref=%.9g\n", i, (double)ow[i], (double)orr[i]);
        fprintf(stderr, "  ow.size()=%zu nonzero_wmma=%zu nonzero_ref=%zu\n", ow.size(),
            (size_t)std::count_if(ow.begin(),ow.end(),[](float v){return v!=0.0f;}),
            (size_t)std::count_if(orr.begin(),orr.end(),[](float v){return v!=0.0f;}));
    }
    (void)hipFree(dw);(void)hipFree(da);(void)hipFree(dq);(void)hipFree(do_wmma);(void)hipFree(do_ref);
    return {cr,wr,ma,mr,bad,J};
}

// Elementwise SwiGLU + per-token weight combine, matching the shipping
// tile8 kernel's formula exactly (ds4_rocm_moe.cuh:530, clamp omitted -
// production calls in this harness always pass clamp=0.0f i.e. disabled):
//   mid = (gate/(1+exp(-gate))) * up * weights[tok]
__global__ static void swiglu_weight_kernel(
        const float *gate, const float *up, const float *tok_weight,
        float *mid, uint32_t ntokens, uint32_t nrows) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)ntokens * nrows;
    if (i >= total) return;
    uint32_t tok = (uint32_t)(i / nrows);
    float g = gate[i], u = up[i];
    mid[i] = (g / (1.0f + expf(-g))) * u * tok_weight[tok];
}

struct WmmaPipelineResult {
    BenchResult full_pipe; // convert + gate-core + up-core + swiglu, one timed unit
    double max_abs, max_rel;
    uint64_t bad;
    int J;
};

// True apples-to-apples arm: two independent-weight WMMA GEMMs (gate, up)
// sharing one Q8_1 conversion, combined by the same SwiGLU formula tile8
// applies internally. Compared against the DP4A reference computed the
// same way (two single-projection DP4A passes + the same combine kernel),
// not against the WMMA-vs-WMMA self-consistency already checked in
// time_wmma().
static WmmaPipelineResult time_wmma_pipeline(uint32_t ntokens, uint32_t xq_blocks, uint32_t nrows) {
    std::mt19937 rng(0x5a5a0000u + ntokens);
    std::vector<cuda_block_q4_K> hwg((size_t)nrows*xq_blocks), hwu((size_t)nrows*xq_blocks);
    std::vector<cuda_block_q8_K> ha((size_t)ntokens*xq_blocks);
    for (auto &b:hwg) fill_q4_K_block(&b,rng);
    for (auto &b:hwu) fill_q4_K_block(&b,rng);
    for (auto &b:ha) fill_q8_K_block(&b,rng);
    std::uniform_real_distribution<float> wdist(0.5f,1.5f);
    std::vector<float> h_tokw(ntokens);
    for (auto &w:h_tokw) w = wdist(rng);

    cuda_block_q4_K *dwg=nullptr,*dwu=nullptr; cuda_block_q8_K *da=nullptr;
    q8_1_mmq_block *dq=nullptr;
    float *do_gate=nullptr,*do_up=nullptr,*do_mid=nullptr;
    float *do_ref_gate=nullptr,*do_ref_up=nullptr,*do_ref_mid=nullptr,*d_tokw=nullptr;
    const size_t osz=(size_t)ntokens*nrows*sizeof(float);
    hip_check(hipMalloc(&dwg,hwg.size()*sizeof(*dwg)),"malloc gate w");
    hip_check(hipMalloc(&dwu,hwu.size()*sizeof(*dwu)),"malloc up w");
    hip_check(hipMalloc(&da,ha.size()*sizeof(*da)),"malloc acts");
    hip_check(hipMalloc(&dq,(size_t)ntokens*xq_blocks*2*sizeof(*dq)),"malloc q8_1");
    hip_check(hipMalloc(&do_gate,osz),"malloc gate out");
    hip_check(hipMalloc(&do_up,osz),"malloc up out");
    hip_check(hipMalloc(&do_mid,osz),"malloc mid out");
    hip_check(hipMalloc(&do_ref_gate,osz),"malloc ref gate");
    hip_check(hipMalloc(&do_ref_up,osz),"malloc ref up");
    hip_check(hipMalloc(&do_ref_mid,osz),"malloc ref mid");
    hip_check(hipMalloc(&d_tokw,std::max<size_t>(ntokens,1)*sizeof(float)),"malloc tokw");
    hip_check(hipMemcpy(dwg,hwg.data(),hwg.size()*sizeof(*dwg),hipMemcpyHostToDevice),"cpy gate w");
    hip_check(hipMemcpy(dwu,hwu.data(),hwu.size()*sizeof(*dwu),hipMemcpyHostToDevice),"cpy up w");
    hip_check(hipMemcpy(da,ha.data(),ha.size()*sizeof(*da),hipMemcpyHostToDevice),"cpy acts");
    hip_check(hipMemcpy(d_tokw,h_tokw.data(),h_tokw.size()*sizeof(float),hipMemcpyHostToDevice),"cpy tokw");

    int J=ntokens<=16?16:32;
    dim3 grid((nrows+63)/64,(ntokens+J-1)/J);
    const size_t sh=((size_t)J*36+64*76)*sizeof(int32_t);
    const unsigned swi_threads=256;
    const unsigned swi_blocks=(unsigned)(((size_t)ntokens*nrows+swi_threads-1)/swi_threads);

    auto full_pipeline=[&](){
        q8_K_to_q8_1_mmq_kernel<<<dim3(ntokens,xq_blocks),32>>>(da,dq,ntokens,xq_blocks);
        if (J==16) {
            q4k_wmma_i64_kernel<16><<<grid,256,sh>>>(dwg,dq,do_gate,ntokens,xq_blocks,nrows);
            q4k_wmma_i64_kernel<16><<<grid,256,sh>>>(dwu,dq,do_up,ntokens,xq_blocks,nrows);
        } else {
            q4k_wmma_i64_kernel<32><<<grid,256,sh>>>(dwg,dq,do_gate,ntokens,xq_blocks,nrows);
            q4k_wmma_i64_kernel<32><<<grid,256,sh>>>(dwu,dq,do_up,ntokens,xq_blocks,nrows);
        }
        swiglu_weight_kernel<<<swi_blocks,swi_threads>>>(do_gate,do_up,d_tokw,do_mid,ntokens,nrows);
    };
    BenchResult pipe=time_launch(full_pipeline);

    q4k_dp4a_single_reference_kernel<<<dim3((nrows+31)/32,(ntokens+7)/8),256>>>(dwg,da,do_ref_gate,ntokens,xq_blocks,nrows);
    q4k_dp4a_single_reference_kernel<<<dim3((nrows+31)/32,(ntokens+7)/8),256>>>(dwu,da,do_ref_up,ntokens,xq_blocks,nrows);
    swiglu_weight_kernel<<<swi_blocks,swi_threads>>>(do_ref_gate,do_ref_up,d_tokw,do_ref_mid,ntokens,nrows);
    hip_check(hipDeviceSynchronize(),"pipeline correctness sync");
    hip_check(hipGetLastError(),"pipeline correctness launch");

    std::vector<float> om((size_t)ntokens*nrows), orm(om.size());
    hip_check(hipMemcpy(om.data(),do_mid,osz,hipMemcpyDeviceToHost),"copy wmma mid");
    hip_check(hipMemcpy(orm.data(),do_ref_mid,osz,hipMemcpyDeviceToHost),"copy ref mid");
    // SiLU is smooth but not linear, so gate-side noise doesn't scale
    // predictably into mid-side noise near gate~0 - keep the same
    // "realistic" order-of-magnitude tolerance as time_wmma(), with a
    // slightly looser rtol since SwiGLU adds one more multiply.
    const double atol=5e-2, rtol=1e-2;
    double ma=0,mr=0; uint64_t bad=0;
    for (size_t i=0;i<om.size();++i){
        double ae=fabs((double)om[i]-orm[i]); double re=ae/std::max(1e-30,fabs((double)orm[i]));
        ma=std::max(ma,ae); mr=std::max(mr,re);
        if(!std::isfinite(om[i]) || ae>atol+rtol*fabs((double)orm[i])) ++bad;
    }
    (void)hipFree(dwg);(void)hipFree(dwu);(void)hipFree(da);(void)hipFree(dq);
    (void)hipFree(do_gate);(void)hipFree(do_up);(void)hipFree(do_mid);
    (void)hipFree(do_ref_gate);(void)hipFree(do_ref_up);(void)hipFree(do_ref_mid);(void)hipFree(d_tokw);
    return {pipe,ma,mr,bad,J};
}

static BenchResult time_kernel(int tile_m, uint32_t bucket_n, uint32_t xq_blocks,
                                uint32_t expert_mid_dim, uint32_t n_expert_total,
                                uint32_t n_expert_used) {
    // One expert receiving exactly `bucket_n` routed pairs; all other
    // experts empty. Isolates the per-bucket-size kernel cost cleanly.
    std::mt19937 rng(12345 + bucket_n * 7 + tile_m);

    const uint32_t n_tokens = bucket_n; // one pair per token for this expert
    std::vector<cuda_block_q4_K> h_gate_w(expert_mid_dim * xq_blocks);
    std::vector<cuda_block_q4_K> h_up_w(expert_mid_dim * xq_blocks);
    for (auto &b : h_gate_w) fill_q4_K_block(&b, rng);
    for (auto &b : h_up_w) fill_q4_K_block(&b, rng);

    std::vector<cuda_block_q8_K> h_xq(std::max(n_tokens, 1u) * xq_blocks);
    for (auto &b : h_xq) fill_q8_K_block(&b, rng);

    std::vector<uint32_t> h_sorted_pairs(bucket_n);
    std::vector<uint32_t> h_counts(n_expert_total, 0);
    std::vector<uint32_t> h_offsets(n_expert_total, 0);
    h_counts[0] = bucket_n;
    for (uint32_t i = 0; i < bucket_n; i++) h_sorted_pairs[i] = i * n_expert_used; // token i, slot 0 -> expert 0
    std::vector<float> h_weights(n_tokens * n_expert_used, 1.0f);

    uint32_t tile_m_u = (uint32_t)tile_m;
    uint32_t n_tiles = (bucket_n + tile_m_u - 1u) / tile_m_u;
    if (n_tiles == 0) n_tiles = 0;
    std::vector<uint32_t> h_tile_experts(n_tiles, 0u);
    std::vector<uint32_t> h_tile_starts(n_tiles);
    for (uint32_t t = 0; t < n_tiles; t++) h_tile_starts[t] = t * tile_m_u;
    uint32_t h_tile_total = n_tiles;

    cuda_block_q4_K *d_gate_w, *d_up_w;
    cuda_block_q8_K *d_xq;
    uint32_t *d_sorted_pairs, *d_counts, *d_offsets, *d_tile_experts, *d_tile_starts, *d_tile_total;
    float *d_weights, *d_gate_out, *d_up_out, *d_mid_out;

    hip_check(hipMalloc(&d_gate_w, h_gate_w.size() * sizeof(cuda_block_q4_K)), "malloc gate_w");
    hip_check(hipMalloc(&d_up_w, h_up_w.size() * sizeof(cuda_block_q4_K)), "malloc up_w");
    hip_check(hipMalloc(&d_xq, h_xq.size() * sizeof(cuda_block_q8_K)), "malloc xq");
    hip_check(hipMalloc(&d_sorted_pairs, std::max<size_t>(h_sorted_pairs.size(), 1) * sizeof(uint32_t)), "malloc pairs");
    hip_check(hipMalloc(&d_counts, h_counts.size() * sizeof(uint32_t)), "malloc counts");
    hip_check(hipMalloc(&d_offsets, h_offsets.size() * sizeof(uint32_t)), "malloc offsets");
    hip_check(hipMalloc(&d_tile_experts, std::max<size_t>(h_tile_experts.size(), 1) * sizeof(uint32_t)), "malloc tile_experts");
    hip_check(hipMalloc(&d_tile_starts, std::max<size_t>(h_tile_starts.size(), 1) * sizeof(uint32_t)), "malloc tile_starts");
    hip_check(hipMalloc(&d_tile_total, sizeof(uint32_t)), "malloc tile_total");
    hip_check(hipMalloc(&d_weights, std::max<size_t>(h_weights.size(), 1) * sizeof(float)), "malloc weights");
    hip_check(hipMalloc(&d_gate_out, (size_t)n_tokens * n_expert_used * expert_mid_dim * sizeof(float) + sizeof(float)), "malloc gate_out");
    hip_check(hipMalloc(&d_up_out, (size_t)n_tokens * n_expert_used * expert_mid_dim * sizeof(float) + sizeof(float)), "malloc up_out");
    hip_check(hipMalloc(&d_mid_out, (size_t)n_tokens * n_expert_used * expert_mid_dim * sizeof(float) + sizeof(float)), "malloc mid_out");

    hip_check(hipMemcpy(d_gate_w, h_gate_w.data(), h_gate_w.size() * sizeof(cuda_block_q4_K), hipMemcpyHostToDevice), "cpy gate_w");
    hip_check(hipMemcpy(d_up_w, h_up_w.data(), h_up_w.size() * sizeof(cuda_block_q4_K), hipMemcpyHostToDevice), "cpy up_w");
    hip_check(hipMemcpy(d_xq, h_xq.data(), h_xq.size() * sizeof(cuda_block_q8_K), hipMemcpyHostToDevice), "cpy xq");
    if (!h_sorted_pairs.empty())
        hip_check(hipMemcpy(d_sorted_pairs, h_sorted_pairs.data(), h_sorted_pairs.size() * sizeof(uint32_t), hipMemcpyHostToDevice), "cpy pairs");
    hip_check(hipMemcpy(d_counts, h_counts.data(), h_counts.size() * sizeof(uint32_t), hipMemcpyHostToDevice), "cpy counts");
    hip_check(hipMemcpy(d_offsets, h_offsets.data(), h_offsets.size() * sizeof(uint32_t), hipMemcpyHostToDevice), "cpy offsets");
    if (!h_tile_experts.empty()) {
        hip_check(hipMemcpy(d_tile_experts, h_tile_experts.data(), h_tile_experts.size() * sizeof(uint32_t), hipMemcpyHostToDevice), "cpy tile_experts");
        hip_check(hipMemcpy(d_tile_starts, h_tile_starts.data(), h_tile_starts.size() * sizeof(uint32_t), hipMemcpyHostToDevice), "cpy tile_starts");
    }
    hip_check(hipMemcpy(d_tile_total, &h_tile_total, sizeof(uint32_t), hipMemcpyHostToDevice), "cpy tile_total");
    if (!h_weights.empty())
        hip_check(hipMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(float), hipMemcpyHostToDevice), "cpy weights");

    const uint64_t gate_row_bytes = (uint64_t)xq_blocks * sizeof(cuda_block_q4_K);
    const uint64_t gate_expert_bytes = gate_row_bytes * expert_mid_dim;
    dim3 tgrid((expert_mid_dim + 31u) / 32u, h_tile_total, 1);

    hipEvent_t start, stop;
    hip_check(hipEventCreate(&start), "event create start");
    hip_check(hipEventCreate(&stop), "event create stop");

    auto launch = [&]() {
        if (tile_m == 8) {
            moe_gate_up_mid_q4K_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                d_gate_out, d_up_out, d_mid_out,
                (const char *)d_gate_w, (const char *)d_up_w, d_xq,
                d_sorted_pairs, d_offsets, d_counts,
                d_tile_total, d_tile_experts, d_tile_starts, d_weights,
                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert_total,
                0u, 1u, 0.0f);
        } else {
            moe_gate_up_mid_q4K_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                d_gate_out, d_up_out, d_mid_out,
                (const char *)d_gate_w, (const char *)d_up_w, d_xq,
                d_sorted_pairs, d_offsets, d_counts,
                d_tile_total, d_tile_experts, d_tile_starts, d_weights,
                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert_total,
                0u, 1u, 0.0f);
        }
    };

    for (int i = 0; i < 5; i++) launch();
    hip_check(hipDeviceSynchronize(), "warmup sync");
    hip_check(hipGetLastError(), "warmup launch error");

    const int ITERS = 20;
    std::vector<double> samples(ITERS);
    for (int i = 0; i < ITERS; i++) {
        hip_check(hipEventRecord(start), "record start");
        launch();
        hip_check(hipEventRecord(stop), "record stop");
        hip_check(hipEventSynchronize(stop), "sync stop");
        float ms = 0.0f;
        hip_check(hipEventElapsedTime(&ms, start, stop), "elapsed");
        samples[i] = (double)ms * 1000.0; // us
    }
    hip_check(hipGetLastError(), "timed launch error");

    std::sort(samples.begin(), samples.end());
    BenchResult r;
    r.median_us = samples[ITERS / 2];
    r.p10_us = samples[ITERS / 10];
    r.p90_us = samples[ITERS - 1 - ITERS / 10];

    (void)hipEventDestroy(start); (void)hipEventDestroy(stop);
    (void)hipFree(d_gate_w); (void)hipFree(d_up_w); (void)hipFree(d_xq);
    (void)hipFree(d_sorted_pairs); (void)hipFree(d_counts); (void)hipFree(d_offsets);
    (void)hipFree(d_tile_experts); (void)hipFree(d_tile_starts); (void)hipFree(d_tile_total);
    (void)hipFree(d_weights); (void)hipFree(d_gate_out); (void)hipFree(d_up_out); (void)hipFree(d_mid_out);
    return r;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    int dev_count = 0;
    hip_check(hipGetDeviceCount(&dev_count), "get device count");
    if (dev_count < 1) { fprintf(stderr, "no ROCm device found\n"); return 2; }
    hipDeviceProp_t prop;
    hip_check(hipGetDeviceProperties(&prop, 0), "get device props");
    printf("Device 0: %s (gcnArch %s)\n", prop.name, prop.gcnArchName);

    const uint32_t xq_blocks = 16;         // K=4096 / CUDA_QK_K(256)
    const uint32_t expert_mid_dim = 2048;  // production N
    const uint32_t n_expert_total = 256;
    const uint32_t n_expert_used = 6;

    const uint32_t buckets[] = {1, 4, 5, 6, 8, 16, 17, 22, 32, 33, 48, 64, 96, 128};

    printf("K=4096 (xq_blocks=%u), expert_mid_dim=%u, experts=%u/%u used\n",
           xq_blocks, expert_mid_dim, n_expert_used, n_expert_total);
    printf("WMMA-core ratios (d8/wmma) are diagnostic: one GEMM vs tile8's fused gate+up+SwiGLU.\n");
    printf("pipe2/dp4a8 is the real apples-to-apples number: 2x WMMA GEMM (gate+up, shared\n");
    printf("Q8_1 conversion) + SwiGLU combine, vs tile8's single fused launch.\n");
    printf("%7s %10s %10s %10s %10s %10s %7s %11s %14s %14s %10s %11s %14s\n",
           "bucket","dp4a4_us","dp4a8_us","wmma_us","conv_us","pipe_us","J","d8/wmma",
           "eq(max_abs/bad)","real(max_abs/bad)","pipe2_us","dp4a8/pipe2","pipe2(max_abs/bad)");
    uint64_t total_bad_eq = 0, total_bad_real = 0, total_bad_pipe2 = 0;
    for (uint32_t bucket : buckets) {
        BenchResult r4 = time_kernel(4, bucket, xq_blocks, expert_mid_dim, n_expert_total, n_expert_used);
        BenchResult r8 = time_kernel(8, bucket, xq_blocks, expert_mid_dim, n_expert_total, n_expert_used);
        WmmaBenchResult rw = time_wmma(bucket,xq_blocks,expert_mid_dim,/*equalize_scale=*/true);
        WmmaBenchResult rn = time_wmma(bucket,xq_blocks,expert_mid_dim,/*equalize_scale=*/false);
        WmmaPipelineResult rp = time_wmma_pipeline(bucket,xq_blocks,expert_mid_dim);
        total_bad_eq += rw.bad; total_bad_real += rn.bad; total_bad_pipe2 += rp.bad;
        printf("%7u %10.2f %10.2f %10.2f %10.2f %10.2f %7d %10.3fx %8.3g/%-5llu %8.3g/%-5llu %10.2f %10.3fx %8.3g/%-5llu\n",
               bucket,r4.median_us,r8.median_us,rw.core.median_us,rw.convert.median_us,
               rw.core.median_us+rw.convert.median_us,rw.J,r8.median_us/rw.core.median_us,
               rw.max_abs,(unsigned long long)rw.bad,rn.max_abs,(unsigned long long)rn.bad,
               rp.full_pipe.median_us,r8.median_us/rp.full_pipe.median_us,rp.max_abs,(unsigned long long)rp.bad);
    }
    printf("\ntotal bad: equalized=%llu realistic-scale=%llu pipe2(fused gate+up+swiglu)=%llu\n",
           (unsigned long long)total_bad_eq, (unsigned long long)total_bad_real, (unsigned long long)total_bad_pipe2);

    return 0;
}
