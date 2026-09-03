#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocwmma/rocwmma.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../rocm/ds4_rocm_bf16_toktile.cuh"

namespace {

constexpr uint32_t kTokens = 33u;
constexpr uint32_t kPrefillTokens = 256u;
constexpr uint32_t kThreads = kDs4Bf16ToktileThreads;

/* Benchmark-only launch-collapse arm for the three equal-shape KDA Q/K/V
 * projections.  It deliberately preserves the production kernel's per-block
 * arithmetic and LDS geometry: blockIdx.x merely selects one of three
 * independent weight/output pointers before running the same 32-column tile.
 * This isolates scheduler tail/launch cost without concatenating weights or
 * changing the numerical operation. */
__global__ __launch_bounds__(16u * 32u, 1)
void bf16_f32_wmma_hilo_m256_qkv_multiptr_kernel(
        float *out_q, float *out_k, float *out_v,
        const uint16_t *weight_q, const uint16_t *weight_k,
        const uint16_t *weight_v, const float *x,
        uint32_t in_dim, uint32_t out_dim) {
    constexpr uint32_t BM = 16u;
    constexpr uint32_t BN = 16u;
    constexpr uint32_t BK = 16u;
    constexpr uint32_t MTile = 256u;
    constexpr uint32_t MTiles = MTile / BM;
    constexpr uint32_t NTilesN = 2u;
    constexpr uint32_t NThreads = MTiles * 32u;
    const uint32_t blocks_per_projection = (out_dim + 31u) / 32u;
    const uint32_t projection = blockIdx.x / blocks_per_projection;
    const uint32_t nblock = blockIdx.x % blocks_per_projection;
    if (projection >= 3u) return;
    float *out = projection == 0u ? out_q :
                 projection == 1u ? out_k : out_v;
    const uint16_t *weight = projection == 0u ? weight_q :
                             projection == 1u ? weight_k : weight_v;
    __shared__ uint16_t sh_a_hi[MTile * BK];
    __shared__ uint16_t sh_a_lo[MTile * BK];
    __shared__ uint16_t sh_b[NTilesN * BK * BN];
    const uint32_t tid = threadIdx.x;
    const uint32_t mt = tid >> 5u;
    const uint32_t nbase = nblock * NTilesN * BN;

    using Bf16 = rocwmma::bfloat16_t;
    using FragA = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK,
                                     Bf16, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK,
                                     Bf16, rocwmma::row_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK,
                                     float>;
    FragA a;
    FragB b;
    FragC acc[NTilesN];
#pragma unroll
    for (uint32_t nt = 0u; nt < NTilesN; ++nt)
        rocwmma::fill_fragment(acc[nt], 0.0f);
    for (uint32_t k0 = 0u; k0 < in_dim; k0 += BK) {
        for (uint32_t j = tid; j < MTile * BK; j += NThreads) {
            const uint32_t m = j / BK;
            const uint32_t kk = j % BK;
            const float xv = x[(uint64_t)m * in_dim + k0 + kk];
            const uint16_t hi = ds4_bf16_rne_bits(xv);
            const float hi_f = __uint_as_float((uint32_t)hi << 16u);
            sh_a_hi[j] = hi;
            sh_a_lo[j] = ds4_bf16_rne_bits(xv - hi_f);
        }
        for (uint32_t j = tid; j < NTilesN * BK * BN; j += NThreads) {
            const uint32_t nt = j / (BK * BN);
            const uint32_t rem = j % (BK * BN);
            const uint32_t kk = rem / BN;
            const uint32_t nn = rem % BN;
            const uint32_t n = nbase + nt * BN + nn;
            sh_b[j] = n < out_dim
                ? weight[(uint64_t)n * in_dim + k0 + kk]
                : 0u;
        }
        __syncthreads();
#pragma unroll
        for (uint32_t nt = 0u; nt < NTilesN; ++nt) {
            rocwmma::load_matrix_sync(
                b, reinterpret_cast<const Bf16 *>(
                    sh_b + nt * BK * BN), BN);
            rocwmma::load_matrix_sync(
                a, reinterpret_cast<const Bf16 *>(
                    sh_a_hi + mt * BM * BK), BK);
            rocwmma::mma_sync(acc[nt], a, b, acc[nt]);
            rocwmma::load_matrix_sync(
                a, reinterpret_cast<const Bf16 *>(
                    sh_a_lo + mt * BM * BK), BK);
            rocwmma::mma_sync(acc[nt], a, b, acc[nt]);
        }
        __syncthreads();
    }
#pragma unroll
    for (uint32_t nt = 0u; nt < NTilesN; ++nt) {
        const uint32_t n0 = nbase + nt * BN;
        if (n0 < out_dim)
            rocwmma::store_matrix_sync(
                out + (uint64_t)(mt * BM) * out_dim + n0,
                acc[nt], out_dim, rocwmma::mem_row_major);
    }
}

[[noreturn]] void fail(const char *what) {
    std::fprintf(stderr, "FAIL %s\n", what);
    std::exit(1);
}

void hip_ok(hipError_t status, const char *what) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "FAIL %s: %s\n", what,
                     hipGetErrorString(status));
        std::exit(1);
    }
}

void blas_ok(hipblasStatus_t status, const char *what) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL %s: hipBLAS status %d\n", what,
                     (int)status);
        std::exit(1);
    }
}

uint16_t bf16_rne(float value) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u)
        return (uint16_t)((bits >> 16u) | 0x0040u);
    const uint32_t tie_to_even = (bits >> 16u) & 1u;
    return (uint16_t)((bits + 0x00007fffu + tie_to_even) >> 16u);
}

__device__ uint16_t bf16_rne_device(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u)
        return (uint16_t)((bits >> 16u) | 0x0040u);
    const uint32_t tie_to_even = (bits >> 16u) & 1u;
    return (uint16_t)((bits + 0x00007fffu + tie_to_even) >> 16u);
}

__global__ void f32_to_bf16_kernel(uint16_t *out, const float *x,
                                    uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = bf16_rne_device(x[i]);
}

/* Diagnostic reference copied from the old production scalar kernel. The
 * candidate body itself is shared from the production header above; this arm
 * intentionally measures the pre-candidate arithmetic, not an external oracle. */
__global__ void current_bf16_f32_kernel(float *out, const uint16_t *weight,
                                         const float *x, uint32_t in_dim,
                                         uint32_t out_dim,
                                         uint32_t tokens) {
    const uint32_t row = blockIdx.x;
    const uint32_t token = blockIdx.y;
    if (row >= out_dim || token >= tokens) return;
    const uint16_t *wr = weight + (uint64_t)row * in_dim;
    const float *xr = x + (uint64_t)token * in_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < in_dim; i += blockDim.x)
        sum += __uint_as_float((uint32_t)wr[i] << 16u) * xr[i];
    __shared__ float partial[kThreads];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride)
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0u)
        out[(uint64_t)token * out_dim + row] = partial[0];
}

/* M=256 diagnostic for the production prefill geometry. One block owns one
 * output row. It loads each 256-value weight slice once into LDS, then the
 * eight wave32 groups consume that slice for all 256 activation rows before
 * advancing K. This deliberately changes the reduction grouping, but retains
 * BF16 weights and F32 activations so it can distinguish weight rereads from
 * hipBLAS' additional F32->BF16 activation conversion. */
__global__ void weight_once_bf16_f32_m256_kernel(
        float *out, const uint16_t *weight, const float *x,
        uint32_t in_dim, uint32_t out_dim) {
    constexpr uint32_t kKTile = 256u;
    constexpr uint32_t kWaves = kThreads / 32u;
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    if (row >= out_dim) return;
    __shared__ uint16_t weight_tile[kKTile];
    __shared__ float token_sums[kPrefillTokens];
    token_sums[tid] = 0.0f;
    __syncthreads();
    const uint16_t *weight_row = weight + (uint64_t)row * in_dim;
    for (uint32_t k_base = 0u; k_base < in_dim; k_base += kKTile) {
        const uint32_t k = k_base + tid;
        weight_tile[tid] = k < in_dim ? weight_row[k] : 0u;
        __syncthreads();
#pragma unroll
        for (uint32_t group = 0u;
             group < kPrefillTokens / kWaves; ++group) {
            const uint32_t token = wave + group * kWaves;
            float partial = 0.0f;
#pragma unroll
            for (uint32_t item = 0u; item < kKTile / 32u; ++item) {
                const uint32_t local_k = lane + item * 32u;
                if (k_base + local_k < in_dim) {
                    const float w = __uint_as_float(
                        (uint32_t)weight_tile[local_k] << 16u);
                    partial += w * x[(uint64_t)token * in_dim +
                                     k_base + local_k];
                }
            }
#pragma unroll
            for (uint32_t offset = 16u; offset > 0u; offset >>= 1u)
                partial += __shfl_down(partial, offset, 32u);
            if (lane == 0u) token_sums[token] += partial;
        }
        __syncthreads();
    }
    out[(uint64_t)tid * out_dim + row] = token_sums[tid];
}

/* Benchmark-only native gfx11 BF16 WMMA ceiling.  Each wave owns one 16x16
 * C tile, so every weight element is fetched once per 16-token M tile instead
 * of once per 32-token production chunk.  Eight waves share a workgroup and
 * cover adjacent N tiles without staging or retaining a second weight copy.
 * This deliberately consumes BF16 activations and is therefore a Lane-B
 * numerical ceiling, not a production candidate. */
template <uint32_t NWaves>
__global__ __launch_bounds__(NWaves * 32u, 1)
void bf16_wmma_m256_ceiling_kernel(
        float *out, const uint16_t *weight, const uint16_t *x,
        uint32_t in_dim, uint32_t out_dim) {
    constexpr uint32_t BM = 16u;
    constexpr uint32_t BN = 16u;
    constexpr uint32_t BK = 16u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t m0 = blockIdx.y * BM;
    const uint32_t n0 = (blockIdx.x * NWaves + wave) * BN;
    if (wave >= NWaves || m0 >= kPrefillTokens || n0 >= out_dim) return;

    using Bf16 = rocwmma::bfloat16_t;
    using FragA = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK,
                                     Bf16, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK,
                                     Bf16, rocwmma::col_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK,
                                     float>;
    FragA a;
    FragB b;
    FragC acc;
    rocwmma::fill_fragment(acc, 0.0f);
    const Bf16 *xb = reinterpret_cast<const Bf16 *>(x);
    const Bf16 *wb = reinterpret_cast<const Bf16 *>(weight);
    for (uint32_t k0 = 0u; k0 < in_dim; k0 += BK) {
        rocwmma::load_matrix_sync(a, xb + (uint64_t)m0 * in_dim + k0,
                                  in_dim);
        rocwmma::load_matrix_sync(b, wb + (uint64_t)n0 * in_dim + k0,
                                  in_dim);
        rocwmma::mma_sync(acc, a, b, acc);
    }
    rocwmma::store_matrix_sync(out + (uint64_t)m0 * out_dim + n0,
                               acc, out_dim, rocwmma::mem_row_major);
}

/* Weight-once cooperative WMMA geometry.  Sixteen M waves consume the same
 * staged B tile, covering all 256 prompt rows before it is discarded.  An
 * optional second N tile shares the same 256x16 A panel; testing 16 and 32
 * waves makes the occupancy/extra-reuse tradeoff explicit on gfx1151. */
template <uint32_t NTilesN>
__global__ __launch_bounds__(16u * NTilesN * 32u, 1)
void bf16_wmma_m256_weight_once_kernel(
        float *out, const uint16_t *weight, const uint16_t *x,
        uint32_t in_dim, uint32_t out_dim) {
    constexpr uint32_t BM = 16u;
    constexpr uint32_t BN = 16u;
    constexpr uint32_t BK = 16u;
    constexpr uint32_t MTiles = kPrefillTokens / BM;
    constexpr uint32_t NWaves = MTiles * NTilesN;
    __shared__ uint16_t sh_a[kPrefillTokens * BK];
    __shared__ uint16_t sh_b[NTilesN * BK * BN];
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t mt = wave % MTiles;
    const uint32_t nt = wave / MTiles;
    const uint32_t n0 = (blockIdx.x * NTilesN + nt) * BN;

    using Bf16 = rocwmma::bfloat16_t;
    using FragA = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK,
                                     Bf16, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK,
                                     Bf16, rocwmma::row_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK,
                                     float>;
    FragA a;
    FragB b;
    FragC acc;
    rocwmma::fill_fragment(acc, 0.0f);
    const uint16_t *xb = x;
    for (uint32_t k0 = 0u; k0 < in_dim; k0 += BK) {
        for (uint32_t j = tid; j < kPrefillTokens * BK; j += NWaves * 32u) {
            const uint32_t m = j / BK;
            const uint32_t kk = j % BK;
            sh_a[j] = xb[(uint64_t)m * in_dim + k0 + kk];
        }
        for (uint32_t j = tid; j < NTilesN * BK * BN; j += NWaves * 32u) {
            const uint32_t tile_n = j / (BK * BN);
            const uint32_t rem = j % (BK * BN);
            const uint32_t kk = rem / BN;
            const uint32_t nn = rem % BN;
            const uint32_t n = (blockIdx.x * NTilesN + tile_n) * BN + nn;
            sh_b[j] = n < out_dim
                ? weight[(uint64_t)n * in_dim + k0 + kk]
                : 0u;
        }
        __syncthreads();
        rocwmma::load_matrix_sync(
            a, reinterpret_cast<const Bf16 *>(sh_a + mt * BM * BK), BK);
        rocwmma::load_matrix_sync(
            b, reinterpret_cast<const Bf16 *>(sh_b + nt * BK * BN), BN);
        rocwmma::mma_sync(acc, a, b, acc);
        __syncthreads();
    }
    if (n0 < out_dim) {
        rocwmma::store_matrix_sync(
            out + (uint64_t)(mt * BM) * out_dim + n0,
            acc, out_dim, rocwmma::mem_row_major);
    }
}

/* Same 16-wave M coverage, with each wave retaining multiple adjacent N
 * accumulators.  This shares A across N without forcing a 1,024-thread block;
 * N=2 and N=4 expose the VGPR-versus-input-traffic knee. */
template <uint32_t NTilesN>
__global__ __launch_bounds__(16u * 32u, 1)
void bf16_wmma_m256_multin_kernel(
        float *out, const uint16_t *weight, const uint16_t *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t tokens) {
    constexpr uint32_t BM = 16u;
    constexpr uint32_t BN = 16u;
    constexpr uint32_t BK = 16u;
    constexpr uint32_t MTiles = kPrefillTokens / BM;
    constexpr uint32_t NThreads = MTiles * 32u;
    __shared__ uint16_t sh_a[kPrefillTokens * BK];
    __shared__ uint16_t sh_b[NTilesN * BK * BN];
    const uint32_t tid = threadIdx.x;
    const uint32_t mt = tid >> 5u;
    const uint32_t nbase = blockIdx.x * NTilesN * BN;
    const uint32_t mbase = blockIdx.y * kPrefillTokens;
    if (mbase >= tokens) return;

    using Bf16 = rocwmma::bfloat16_t;
    using FragA = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK,
                                     Bf16, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK,
                                     Bf16, rocwmma::row_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK,
                                     float>;
    FragA a;
    FragB b;
    FragC acc[NTilesN];
#pragma unroll
    for (uint32_t nt = 0u; nt < NTilesN; ++nt)
        rocwmma::fill_fragment(acc[nt], 0.0f);
    for (uint32_t k0 = 0u; k0 < in_dim; k0 += BK) {
        for (uint32_t j = tid; j < kPrefillTokens * BK; j += NThreads) {
            const uint32_t m = j / BK;
            const uint32_t kk = j % BK;
            const uint32_t global_m = mbase + m;
            sh_a[j] = global_m < tokens
                ? x[(uint64_t)global_m * in_dim + k0 + kk]
                : 0u;
        }
        for (uint32_t j = tid; j < NTilesN * BK * BN; j += NThreads) {
            const uint32_t nt = j / (BK * BN);
            const uint32_t rem = j % (BK * BN);
            const uint32_t kk = rem / BN;
            const uint32_t nn = rem % BN;
            const uint32_t n = nbase + nt * BN + nn;
            sh_b[j] = n < out_dim
                ? weight[(uint64_t)n * in_dim + k0 + kk]
                : 0u;
        }
        __syncthreads();
        rocwmma::load_matrix_sync(
            a, reinterpret_cast<const Bf16 *>(sh_a + mt * BM * BK), BK);
#pragma unroll
        for (uint32_t nt = 0u; nt < NTilesN; ++nt) {
            rocwmma::load_matrix_sync(
                b, reinterpret_cast<const Bf16 *>(sh_b + nt * BK * BN), BN);
            rocwmma::mma_sync(acc[nt], a, b, acc[nt]);
        }
        __syncthreads();
    }
#pragma unroll
    for (uint32_t nt = 0u; nt < NTilesN; ++nt) {
        const uint32_t n0 = nbase + nt * BN;
        if (n0 < out_dim && mbase + mt * BM < tokens)
            rocwmma::store_matrix_sync(
                out + (uint64_t)(mbase + mt * BM) * out_dim + n0,
                acc[nt], out_dim, rocwmma::mem_row_major);
    }
}

/* Precision-repaired WMMA ceiling: x = bf16_hi(x) + bf16_lo(x-hi).
 * BF16 weights are exact model storage, so two WMMA products recover most of
 * the F32-activation path's precision while retaining weight-once execution.
 * Both activation fragments are tile-local; there is no expanded persistent
 * activation or weight allocation. */
template <uint32_t NTilesN>
__global__ __launch_bounds__(16u * 32u, 1)
void bf16_wmma_m256_multin_hilo_kernel(
        float *out, const uint16_t *weight, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t tokens) {
    constexpr uint32_t BM = 16u;
    constexpr uint32_t BN = 16u;
    constexpr uint32_t BK = 16u;
    constexpr uint32_t MTiles = kPrefillTokens / BM;
    constexpr uint32_t NThreads = MTiles * 32u;
    __shared__ uint16_t sh_a_hi[kPrefillTokens * BK];
    __shared__ uint16_t sh_a_lo[kPrefillTokens * BK];
    __shared__ uint16_t sh_b[NTilesN * BK * BN];
    const uint32_t tid = threadIdx.x;
    const uint32_t mt = tid >> 5u;
    const uint32_t nbase = blockIdx.x * NTilesN * BN;
    const uint32_t mbase = blockIdx.y * kPrefillTokens;
    if (mbase >= tokens) return;

    using Bf16 = rocwmma::bfloat16_t;
    using FragA = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK,
                                     Bf16, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK,
                                     Bf16, rocwmma::row_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK,
                                     float>;
    FragA a;
    FragB b;
    FragC acc[NTilesN];
#pragma unroll
    for (uint32_t nt = 0u; nt < NTilesN; ++nt)
        rocwmma::fill_fragment(acc[nt], 0.0f);
    for (uint32_t k0 = 0u; k0 < in_dim; k0 += BK) {
        for (uint32_t j = tid; j < kPrefillTokens * BK; j += NThreads) {
            const uint32_t m = j / BK;
            const uint32_t kk = j % BK;
            const uint32_t global_m = mbase + m;
            if (global_m < tokens) {
                const float xv = x[(uint64_t)global_m * in_dim + k0 + kk];
                const uint16_t hi = bf16_rne_device(xv);
                const float hi_f = __uint_as_float((uint32_t)hi << 16u);
                sh_a_hi[j] = hi;
                sh_a_lo[j] = bf16_rne_device(xv - hi_f);
            } else {
                sh_a_hi[j] = 0u;
                sh_a_lo[j] = 0u;
            }
        }
        for (uint32_t j = tid; j < NTilesN * BK * BN; j += NThreads) {
            const uint32_t nt = j / (BK * BN);
            const uint32_t rem = j % (BK * BN);
            const uint32_t kk = rem / BN;
            const uint32_t nn = rem % BN;
            const uint32_t n = nbase + nt * BN + nn;
            sh_b[j] = n < out_dim
                ? weight[(uint64_t)n * in_dim + k0 + kk]
                : 0u;
        }
        __syncthreads();
#pragma unroll
        for (uint32_t nt = 0u; nt < NTilesN; ++nt) {
            rocwmma::load_matrix_sync(
                b, reinterpret_cast<const Bf16 *>(sh_b + nt * BK * BN), BN);
            rocwmma::load_matrix_sync(
                a, reinterpret_cast<const Bf16 *>(sh_a_hi + mt * BM * BK), BK);
            rocwmma::mma_sync(acc[nt], a, b, acc[nt]);
            rocwmma::load_matrix_sync(
                a, reinterpret_cast<const Bf16 *>(sh_a_lo + mt * BM * BK), BK);
            rocwmma::mma_sync(acc[nt], a, b, acc[nt]);
        }
        __syncthreads();
    }
#pragma unroll
    for (uint32_t nt = 0u; nt < NTilesN; ++nt) {
        const uint32_t n0 = nbase + nt * BN;
        if (n0 < out_dim && mbase + mt * BM < tokens)
            rocwmma::store_matrix_sync(
                out + (uint64_t)(mbase + mt * BM) * out_dim + n0,
                acc[nt], out_dim, rocwmma::mem_row_major);
    }
}

void launch_segmented_tiled(float *out, const uint16_t *weight,
                            const float *x, uint32_t in_dim,
                            uint32_t out_dim, uint32_t tokens) {
    uint32_t first = 0u;
    const uint32_t chunks32 = tokens / 32u;
    if (chunks32 > 0u) {
        matmul_bf16_f32_toktile_w32_kernel<32u><<<
            dim3(out_dim, chunks32), kThreads>>>(
            out, weight, x, in_dim, out_dim);
        hip_ok(hipGetLastError(), "segmented tile32 launch");
        first += chunks32 * 32u;
    }
#define LAUNCH_TAIL(T) do {                                                \
        if ((tokens - first) >= (T)) {                                     \
            matmul_bf16_f32_toktile_w32_kernel<(T)><<<out_dim, kThreads>>>(\
                out + (uint64_t)first * out_dim, weight,                   \
                x + (uint64_t)first * in_dim, in_dim, out_dim);           \
            hip_ok(hipGetLastError(), "segmented tail launch");           \
            first += (T);                                                  \
        }                                                                  \
    } while (0)
    LAUNCH_TAIL(16u);
    LAUNCH_TAIL(8u);
    LAUNCH_TAIL(4u);
    LAUNCH_TAIL(2u);
    LAUNCH_TAIL(1u);
#undef LAUNCH_TAIL
    if (first != tokens) fail("segmented tile decomposition");
}

void launch_tail25_tiled(float *out, const uint16_t *weight,
                         const float *x, uint32_t in_dim,
                         uint32_t out_dim, uint32_t tokens) {
    const uint32_t chunks32 = tokens / 32u;
    const uint32_t first = chunks32 * 32u;
    if (chunks32 > 0u) {
        matmul_bf16_f32_toktile_w32_kernel<32u><<<
            dim3(out_dim, chunks32), kThreads>>>(
            out, weight, x, in_dim, out_dim);
        hip_ok(hipGetLastError(), "tail25 tile32 launch");
    }
    if (tokens - first != 25u) fail("tail25 requires a 25-token tail");
    matmul_bf16_f32_toktile_w32_kernel<25u><<<out_dim, kThreads>>>(
        out + (uint64_t)first * out_dim, weight,
        x + (uint64_t)first * in_dim, in_dim, out_dim);
    hip_ok(hipGetLastError(), "tail25 fused launch");
}

void launch_lowrank128_tiled(float *out, const uint16_t *weight,
                             const float *x, uint32_t out_dim,
                             uint32_t tokens) {
    uint32_t first = 0u;
    const uint32_t chunks32 = tokens / 32u;
    if (chunks32 > 0u) {
        matmul_bf16_f32_lowrank128_toktile_w32_kernel<32u><<<
            dim3(out_dim, chunks32), 32u>>>(out, weight, x, out_dim);
        hip_ok(hipGetLastError(), "low-rank tile32 launch");
        first = chunks32 * 32u;
    }
#define LAUNCH_LOWRANK_TAIL(T) do {                                      \
        if ((tokens - first) >= (T)) {                                   \
            matmul_bf16_f32_lowrank128_toktile_w32_kernel<(T)><<<        \
                out_dim, 32u>>>(                                         \
                out + (uint64_t)first * out_dim, weight,                 \
                x + (uint64_t)first * 128u, out_dim);                    \
            hip_ok(hipGetLastError(), "low-rank tail launch");          \
            first += (T);                                                \
        }                                                                \
    } while (0)
    LAUNCH_LOWRANK_TAIL(16u);
    LAUNCH_LOWRANK_TAIL(8u);
    LAUNCH_LOWRANK_TAIL(4u);
    LAUNCH_LOWRANK_TAIL(2u);
    LAUNCH_LOWRANK_TAIL(1u);
#undef LAUNCH_LOWRANK_TAIL
    if (first != tokens) fail("low-rank tile decomposition");
}

struct Error {
    double nrmse;
    double cosine;
    double max_abs;
};

Error compare(const std::vector<float> &reference,
              const std::vector<float> &candidate) {
    if (reference.size() != candidate.size()) fail("output size mismatch");
    double square_error = 0.0, square_reference = 0.0;
    double dot = 0.0, square_candidate = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(candidate[i]))
            fail("non-finite output");
        const double error = (double)candidate[i] - reference[i];
        square_error += error * error;
        square_reference += (double)reference[i] * reference[i];
        square_candidate += (double)candidate[i] * candidate[i];
        dot += (double)reference[i] * candidate[i];
        max_abs = std::max(max_abs, std::abs(error));
    }
    if (!(square_reference > 0.0) || !(square_candidate > 0.0))
        fail("degenerate output norm");
    return {
        std::sqrt(square_error / square_reference),
        dot / std::sqrt(square_reference * square_candidate),
        max_abs,
    };
}

template <typename Launch>
float time_ms(Launch launch) {
    hipEvent_t begin = nullptr, end = nullptr;
    hip_ok(hipEventCreate(&begin), "create begin event");
    hip_ok(hipEventCreate(&end), "create end event");
    for (int i = 0; i < 2; ++i) launch();
    hip_ok(hipDeviceSynchronize(), "warm synchronize");
    hip_ok(hipEventRecord(begin), "record begin");
    for (int i = 0; i < 5; ++i) launch();
    hip_ok(hipEventRecord(end), "record end");
    hip_ok(hipEventSynchronize(end), "wait end");
    float elapsed = 0.0f;
    hip_ok(hipEventElapsedTime(&elapsed, begin, end), "elapsed time");
    hip_ok(hipEventDestroy(end), "destroy end event");
    hip_ok(hipEventDestroy(begin), "destroy begin event");
    return elapsed / 5.0f;
}

void run_shape(hipblasHandle_t handle, const uint16_t *weight,
               uint32_t in_dim, uint32_t out_dim) {
    const uint64_t x_count = (uint64_t)kTokens * in_dim;
    const uint64_t out_count = (uint64_t)kTokens * out_dim;
    std::vector<float> x(x_count);
    for (uint64_t i = 0; i < x_count; ++i)
        x[i] = 0.7f * std::cos((double)(i % 3571u) * 0.013) +
               0.03f * std::sin((double)i * 0.001);

    float *d_x = nullptr, *d_reference = nullptr;
    float *d_blas = nullptr, *d_exact33 = nullptr;
    float *d_segmented = nullptr, *d_lowrank = nullptr;
    uint16_t *d_x_bf16 = nullptr;
    hip_ok(hipMalloc(&d_x, x_count * sizeof(float)), "allocate F32 input");
    hip_ok(hipMalloc(&d_x_bf16, x_count * sizeof(uint16_t)),
           "allocate BF16 input");
    hip_ok(hipMalloc(&d_reference, out_count * sizeof(float)),
           "allocate reference output");
    hip_ok(hipMalloc(&d_blas, out_count * sizeof(float)),
           "allocate hipBLAS output");
    hip_ok(hipMalloc(&d_exact33, out_count * sizeof(float)),
           "allocate exact33 output");
    hip_ok(hipMalloc(&d_segmented, out_count * sizeof(float)),
           "allocate segmented output");
    if (in_dim == 128u)
        hip_ok(hipMalloc(&d_lowrank, out_count * sizeof(float)),
               "allocate low-rank output");
    hip_ok(hipMemcpy(d_x, x.data(), x_count * sizeof(float),
                     hipMemcpyHostToDevice), "copy input");

    const auto current = [&] {
        current_bf16_f32_kernel<<<dim3(out_dim, kTokens), kThreads>>>(
            d_reference, weight, d_x, in_dim, out_dim, kTokens);
        hip_ok(hipGetLastError(), "current BF16xF32 launch");
    };
    const auto exact33 = [&] {
        matmul_bf16_f32_toktile_w32_kernel<33u><<<out_dim, kThreads>>>(
            d_exact33, weight, d_x, in_dim, out_dim);
        hip_ok(hipGetLastError(), "exact33 BF16xF32 launch");
    };
    const auto segmented = [&] {
        launch_segmented_tiled(d_segmented, weight, d_x, in_dim,
                               out_dim, kTokens);
    };
    const auto lowrank = [&] {
        launch_lowrank128_tiled(d_lowrank, weight, d_x, out_dim, kTokens);
    };
    const auto blas = [&] {
        f32_to_bf16_kernel<<<(x_count + 255u) / 256u, 256u>>>(
            d_x_bf16, d_x, x_count);
        hip_ok(hipGetLastError(), "BF16 conversion launch");
        const float alpha = 1.0f;
        const float beta = 0.0f;
        blas_ok(hipblasGemmEx(
                    handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                    (int)out_dim, (int)kTokens, (int)in_dim,
                    &alpha, weight, HIP_R_16BF, (int)in_dim,
                    d_x_bf16, HIP_R_16BF, (int)in_dim,
                    &beta, d_blas, HIP_R_32F, (int)out_dim,
                    HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
                "BF16 hipBLAS GEMM");
    };

    const float current_ms = time_ms(current);
    const float exact33_ms = time_ms(exact33);
    const float segmented_ms = time_ms(segmented);
    const float lowrank_ms = in_dim == 128u ? time_ms(lowrank) : 0.0f;
    const float blas_ms = time_ms(blas);
    current();
    exact33();
    segmented();
    if (in_dim == 128u) lowrank();
    blas();
    hip_ok(hipDeviceSynchronize(), "output synchronize");
    std::vector<float> reference(out_count), exact33_output(out_count);
    std::vector<float> segmented_output(out_count);
    std::vector<float> lowrank_output;
    std::vector<float> blas_output(out_count);
    hip_ok(hipMemcpy(reference.data(), d_reference,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read reference");
    hip_ok(hipMemcpy(exact33_output.data(), d_exact33,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read exact33 output");
    hip_ok(hipMemcpy(segmented_output.data(), d_segmented,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read segmented output");
    if (in_dim == 128u) {
        lowrank_output.resize(out_count);
        hip_ok(hipMemcpy(lowrank_output.data(), d_lowrank,
                         out_count * sizeof(float), hipMemcpyDeviceToHost),
               "read low-rank output");
    }
    hip_ok(hipMemcpy(blas_output.data(), d_blas,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read hipBLAS output");
    const Error exact33_error = compare(reference, exact33_output);
    const Error segmented_error = compare(reference, segmented_output);
    if (in_dim == 128u) {
        const Error lowrank_error = compare(reference, lowrank_output);
        if (std::memcmp(reference.data(), lowrank_output.data(),
                        out_count * sizeof(float)) != 0)
            fail("low-rank token tile must bit-match scalar reduction");
        std::printf(
            "  lowrank128_ms=%.4f speedup=%.3fx nrmse=%.9g "
            "cosine=%.12g max_abs=%.9g bit_exact=1\n",
            lowrank_ms, current_ms / lowrank_ms, lowrank_error.nrmse,
            lowrank_error.cosine, lowrank_error.max_abs);
    }
    const Error blas_error = compare(reference, blas_output);
    std::printf(
        "BF16 batch GEMM shape=%ux%ux%u residency=host-registered "
        "current_ms=%.4f exact33_ms=%.4f segmented32_ms=%.4f "
        "bf16_blas_ms=%.4f exact33_speedup=%.3fx "
        "segmented_speedup=%.3fx blas_speedup=%.3fx\n",
        kTokens, out_dim, in_dim, current_ms, exact33_ms,
        segmented_ms, blas_ms, current_ms / exact33_ms,
        current_ms / segmented_ms, current_ms / blas_ms);
    std::printf(
        "  exact33 nrmse=%.9g cosine=%.12g max_abs=%.9g\n",
        exact33_error.nrmse, exact33_error.cosine,
        exact33_error.max_abs);
    std::printf(
        "  segmented32 nrmse=%.9g cosine=%.12g max_abs=%.9g\n",
        segmented_error.nrmse, segmented_error.cosine,
        segmented_error.max_abs);
    std::printf(
        "  bf16_blas nrmse=%.9g cosine=%.12g max_abs=%.9g\n",
        blas_error.nrmse, blas_error.cosine, blas_error.max_abs);

    if (d_lowrank) hip_ok(hipFree(d_lowrank), "free low-rank output");
    hip_ok(hipFree(d_segmented), "free segmented output");
    hip_ok(hipFree(d_exact33), "free exact33 output");
    hip_ok(hipFree(d_blas), "free hipBLAS output");
    hip_ok(hipFree(d_reference), "free reference output");
    hip_ok(hipFree(d_x_bf16), "free BF16 input");
    hip_ok(hipFree(d_x), "free F32 input");
}

void run_prefill_shape(hipblasHandle_t handle, const uint16_t *weight,
                       uint32_t in_dim, uint32_t out_dim) {
    const uint64_t x_count = (uint64_t)kPrefillTokens * in_dim;
    const uint64_t out_count = (uint64_t)kPrefillTokens * out_dim;
    std::vector<float> x(x_count);
    for (uint64_t i = 0u; i < x_count; ++i)
        x[i] = 0.7f * std::cos((double)(i % 3571u) * 0.013) +
               0.03f * std::sin((double)i * 0.001);

    float *d_x = nullptr, *d_reference = nullptr;
    float *d_production = nullptr, *d_blas = nullptr;
    float *d_weight_once = nullptr;
    float *d_wmma = nullptr, *d_wmma_weight_once_1 = nullptr;
    float *d_wmma_weight_once_2 = nullptr;
    float *d_wmma_multin_2 = nullptr, *d_wmma_multin_4 = nullptr;
    float *d_wmma_hilo_2 = nullptr;
    float *d_lowrank = nullptr;
    uint16_t *d_x_bf16 = nullptr;
    hip_ok(hipMalloc(&d_x, x_count * sizeof(float)),
           "allocate M256 F32 input");
    hip_ok(hipMalloc(&d_x_bf16, x_count * sizeof(uint16_t)),
           "allocate M256 BF16 input");
    hip_ok(hipMalloc(&d_reference, out_count * sizeof(float)),
           "allocate M256 reference output");
    hip_ok(hipMalloc(&d_production, out_count * sizeof(float)),
           "allocate M256 production output");
    hip_ok(hipMalloc(&d_blas, out_count * sizeof(float)),
           "allocate M256 hipBLAS output");
    hip_ok(hipMalloc(&d_weight_once, out_count * sizeof(float)),
           "allocate M256 weight-once output");
    hip_ok(hipMalloc(&d_wmma, out_count * sizeof(float)),
           "allocate M256 WMMA output");
    hip_ok(hipMalloc(&d_wmma_weight_once_1, out_count * sizeof(float)),
           "allocate M256 cooperative WMMA N1 output");
    hip_ok(hipMalloc(&d_wmma_weight_once_2, out_count * sizeof(float)),
           "allocate M256 cooperative WMMA N2 output");
    hip_ok(hipMalloc(&d_wmma_multin_2, out_count * sizeof(float)),
           "allocate M256 multi-N WMMA N2 output");
    hip_ok(hipMalloc(&d_wmma_multin_4, out_count * sizeof(float)),
           "allocate M256 multi-N WMMA N4 output");
    hip_ok(hipMalloc(&d_wmma_hilo_2, out_count * sizeof(float)),
           "allocate M256 hi/lo WMMA N2 output");
    if (in_dim == 128u)
        hip_ok(hipMalloc(&d_lowrank, out_count * sizeof(float)),
               "allocate M256 low-rank output");
    hip_ok(hipMemcpy(d_x, x.data(), x_count * sizeof(float),
                     hipMemcpyHostToDevice), "copy M256 input");
    f32_to_bf16_kernel<<<(x_count + 255u) / 256u, 256u>>>(
        d_x_bf16, d_x, x_count);
    hip_ok(hipGetLastError(), "M256 BF16 conversion launch");

    const auto production = [&] {
        launch_segmented_tiled(d_production, weight, d_x, in_dim,
                               out_dim, kPrefillTokens);
    };
    const auto weight_once = [&] {
        weight_once_bf16_f32_m256_kernel<<<out_dim, kThreads>>>(
            d_weight_once, weight, d_x, in_dim, out_dim);
        hip_ok(hipGetLastError(), "M256 weight-once launch");
    };
    const auto lowrank = [&] {
        launch_lowrank128_tiled(d_lowrank, weight, d_x, out_dim,
                                kPrefillTokens);
    };
    const auto blas = [&] {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        blas_ok(hipblasGemmEx(
                    handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                    (int)out_dim, (int)kPrefillTokens, (int)in_dim,
                    &alpha, weight, HIP_R_16BF, (int)in_dim,
                    d_x_bf16, HIP_R_16BF, (int)in_dim,
                    &beta, d_blas, HIP_R_32F, (int)out_dim,
                    HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
                "M256 BF16 hipBLAS GEMM");
    };
    const auto wmma = [&] {
        constexpr uint32_t kWmmaWaves = 8u;
        bf16_wmma_m256_ceiling_kernel<kWmmaWaves><<<
            dim3((out_dim + kWmmaWaves * 16u - 1u) /
                     (kWmmaWaves * 16u),
                 (kPrefillTokens + 15u) / 16u),
            kWmmaWaves * 32u>>>(d_wmma, weight, d_x_bf16,
                                in_dim, out_dim);
        hip_ok(hipGetLastError(), "M256 BF16 WMMA launch");
    };
    const auto wmma_weight_once_1 = [&] {
        bf16_wmma_m256_weight_once_kernel<1u><<<
            (out_dim + 15u) / 16u, 16u * 32u>>>(
                d_wmma_weight_once_1, weight, d_x_bf16,
                in_dim, out_dim);
        hip_ok(hipGetLastError(), "M256 cooperative WMMA N1 launch");
    };
    const auto wmma_weight_once_2 = [&] {
        bf16_wmma_m256_weight_once_kernel<2u><<<
            (out_dim + 31u) / 32u, 32u * 32u>>>(
                d_wmma_weight_once_2, weight, d_x_bf16,
                in_dim, out_dim);
        hip_ok(hipGetLastError(), "M256 cooperative WMMA N2 launch");
    };
    const auto wmma_multin_2 = [&] {
        bf16_wmma_m256_multin_kernel<2u><<<
            dim3((out_dim + 31u) / 32u, 1u), 16u * 32u>>>(
                d_wmma_multin_2, weight, d_x_bf16, in_dim, out_dim,
                kPrefillTokens);
        hip_ok(hipGetLastError(), "M256 multi-N WMMA N2 launch");
    };
    const auto wmma_multin_4 = [&] {
        bf16_wmma_m256_multin_kernel<4u><<<
            dim3((out_dim + 63u) / 64u, 1u), 16u * 32u>>>(
                d_wmma_multin_4, weight, d_x_bf16, in_dim, out_dim,
                kPrefillTokens);
        hip_ok(hipGetLastError(), "M256 multi-N WMMA N4 launch");
    };
    const auto wmma_hilo_2 = [&] {
        matmul_bf16_f32_wmma_hilo_m256_kernel<2u><<<
            dim3((out_dim + 31u) / 32u, 1u), 16u * 32u>>>(
                d_wmma_hilo_2, weight, d_x, in_dim, out_dim,
                kPrefillTokens);
        hip_ok(hipGetLastError(), "M256 hi/lo WMMA N2 launch");
    };

    current_bf16_f32_kernel<<<dim3(out_dim, kPrefillTokens), kThreads>>>(
        d_reference, weight, d_x, in_dim, out_dim, kPrefillTokens);
    hip_ok(hipGetLastError(), "M256 scalar reference launch");
    const float production_ms = time_ms(production);
    const float weight_once_ms = time_ms(weight_once);
    const float lowrank_ms = in_dim == 128u ? time_ms(lowrank) : 0.0f;
    const float blas_ms = time_ms(blas);
    const float wmma_ms = time_ms(wmma);
    const float wmma_weight_once_1_ms = time_ms(wmma_weight_once_1);
    const float wmma_weight_once_2_ms = time_ms(wmma_weight_once_2);
    const float wmma_multin_2_ms = time_ms(wmma_multin_2);
    const float wmma_multin_4_ms = time_ms(wmma_multin_4);
    const float wmma_hilo_2_ms = time_ms(wmma_hilo_2);
    production();
    weight_once();
    if (in_dim == 128u) lowrank();
    blas();
    wmma();
    wmma_weight_once_1();
    wmma_weight_once_2();
    wmma_multin_2();
    wmma_multin_4();
    wmma_hilo_2();
    hip_ok(hipDeviceSynchronize(), "M256 output synchronize");

    std::vector<float> reference(out_count), production_output(out_count);
    std::vector<float> weight_once_output(out_count), blas_output(out_count);
    std::vector<float> wmma_output(out_count);
    std::vector<float> wmma_weight_once_1_output(out_count);
    std::vector<float> wmma_weight_once_2_output(out_count);
    std::vector<float> wmma_multin_2_output(out_count);
    std::vector<float> wmma_multin_4_output(out_count);
    std::vector<float> wmma_hilo_2_output(out_count);
    std::vector<float> lowrank_output;
    hip_ok(hipMemcpy(reference.data(), d_reference,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read M256 reference");
    hip_ok(hipMemcpy(production_output.data(), d_production,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read M256 production");
    hip_ok(hipMemcpy(weight_once_output.data(), d_weight_once,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read M256 weight-once");
    if (in_dim == 128u) {
        lowrank_output.resize(out_count);
        hip_ok(hipMemcpy(lowrank_output.data(), d_lowrank,
                         out_count * sizeof(float), hipMemcpyDeviceToHost),
               "read M256 low-rank");
    }
    hip_ok(hipMemcpy(blas_output.data(), d_blas,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read M256 hipBLAS");
    hip_ok(hipMemcpy(wmma_output.data(), d_wmma,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read M256 WMMA");
    hip_ok(hipMemcpy(wmma_weight_once_1_output.data(), d_wmma_weight_once_1,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read M256 cooperative WMMA N1");
    hip_ok(hipMemcpy(wmma_weight_once_2_output.data(), d_wmma_weight_once_2,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read M256 cooperative WMMA N2");
    hip_ok(hipMemcpy(wmma_multin_2_output.data(), d_wmma_multin_2,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read M256 multi-N WMMA N2");
    hip_ok(hipMemcpy(wmma_multin_4_output.data(), d_wmma_multin_4,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read M256 multi-N WMMA N4");
    hip_ok(hipMemcpy(wmma_hilo_2_output.data(), d_wmma_hilo_2,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read M256 hi/lo WMMA N2");
    const Error production_error = compare(reference, production_output);
    const Error weight_once_error = compare(reference, weight_once_output);
    const Error blas_error = compare(reference, blas_output);
    const Error wmma_error = compare(reference, wmma_output);
    const Error wmma_weight_once_1_error =
        compare(reference, wmma_weight_once_1_output);
    const Error wmma_weight_once_2_error =
        compare(reference, wmma_weight_once_2_output);
    const Error wmma_multin_2_error = compare(reference, wmma_multin_2_output);
    const Error wmma_multin_4_error = compare(reference, wmma_multin_4_output);
    const Error wmma_hilo_2_error = compare(reference, wmma_hilo_2_output);
    std::printf(
        "BF16 prefill shape=%ux%ux%u residency=host-registered "
        "production_ms=%.4f weight_once_f32_ms=%.4f bf16_blas_ms=%.4f "
        "wmma_bf16_ms=%.4f wmma_weight_once_n1_ms=%.4f "
        "wmma_weight_once_n2_ms=%.4f weight_once_speedup=%.3fx "
        "blas_speedup=%.3fx wmma_speedup=%.3fx "
        "wmma_weight_once_n1_speedup=%.3fx "
        "wmma_weight_once_n2_speedup=%.3fx\n",
        kPrefillTokens, out_dim, in_dim, production_ms, weight_once_ms,
        blas_ms, wmma_ms, wmma_weight_once_1_ms, wmma_weight_once_2_ms,
        production_ms / weight_once_ms, production_ms / blas_ms,
        production_ms / wmma_ms, production_ms / wmma_weight_once_1_ms,
        production_ms / wmma_weight_once_2_ms);
    std::printf(
        "  wmma_multin_n2_ms=%.4f speedup=%.3fx "
        "wmma_multin_n4_ms=%.4f speedup=%.3fx "
        "wmma_hilo_n2_ms=%.4f speedup=%.3fx\n",
        wmma_multin_2_ms, production_ms / wmma_multin_2_ms,
        wmma_multin_4_ms, production_ms / wmma_multin_4_ms,
        wmma_hilo_2_ms, production_ms / wmma_hilo_2_ms);
    std::printf(
        "  production nrmse=%.9g cosine=%.12g max_abs=%.9g\n",
        production_error.nrmse, production_error.cosine,
        production_error.max_abs);
    std::printf(
        "  weight_once_f32 nrmse=%.9g cosine=%.12g max_abs=%.9g\n",
        weight_once_error.nrmse, weight_once_error.cosine,
        weight_once_error.max_abs);
    if (in_dim == 128u) {
        const Error lowrank_error = compare(reference, lowrank_output);
        if (std::memcmp(reference.data(), lowrank_output.data(),
                        out_count * sizeof(float)) != 0)
            fail("M256 low-rank token tile must bit-match scalar reduction");
        std::printf(
            "  lowrank128_ms=%.4f speedup=%.3fx nrmse=%.9g "
            "cosine=%.12g max_abs=%.9g bit_exact=1\n",
            lowrank_ms, production_ms / lowrank_ms, lowrank_error.nrmse,
            lowrank_error.cosine, lowrank_error.max_abs);
    }
    std::printf(
        "  bf16_blas nrmse=%.9g cosine=%.12g max_abs=%.9g\n",
        blas_error.nrmse, blas_error.cosine, blas_error.max_abs);
    std::printf(
        "  wmma_bf16 nrmse=%.9g cosine=%.12g max_abs=%.9g\n",
        wmma_error.nrmse, wmma_error.cosine, wmma_error.max_abs);
    std::printf(
        "  wmma_weight_once_n1 nrmse=%.9g cosine=%.12g max_abs=%.9g\n",
        wmma_weight_once_1_error.nrmse,
        wmma_weight_once_1_error.cosine,
        wmma_weight_once_1_error.max_abs);
    std::printf(
        "  wmma_weight_once_n2 nrmse=%.9g cosine=%.12g max_abs=%.9g\n",
        wmma_weight_once_2_error.nrmse,
        wmma_weight_once_2_error.cosine,
        wmma_weight_once_2_error.max_abs);
    std::printf(
        "  wmma_multin_n2 nrmse=%.9g cosine=%.12g max_abs=%.9g\n",
        wmma_multin_2_error.nrmse, wmma_multin_2_error.cosine,
        wmma_multin_2_error.max_abs);
    std::printf(
        "  wmma_multin_n4 nrmse=%.9g cosine=%.12g max_abs=%.9g\n",
        wmma_multin_4_error.nrmse, wmma_multin_4_error.cosine,
        wmma_multin_4_error.max_abs);
    std::printf(
        "  wmma_hilo_n2 nrmse=%.9g cosine=%.12g max_abs=%.9g\n",
        wmma_hilo_2_error.nrmse, wmma_hilo_2_error.cosine,
        wmma_hilo_2_error.max_abs);

    if (d_lowrank)
        hip_ok(hipFree(d_lowrank), "free M256 low-rank output");
    hip_ok(hipFree(d_wmma), "free M256 WMMA output");
    hip_ok(hipFree(d_wmma_weight_once_1),
           "free M256 cooperative WMMA N1 output");
    hip_ok(hipFree(d_wmma_weight_once_2),
           "free M256 cooperative WMMA N2 output");
    hip_ok(hipFree(d_wmma_multin_2), "free M256 multi-N WMMA N2 output");
    hip_ok(hipFree(d_wmma_multin_4), "free M256 multi-N WMMA N4 output");
    hip_ok(hipFree(d_wmma_hilo_2), "free M256 hi/lo WMMA N2 output");
    hip_ok(hipFree(d_weight_once), "free M256 weight-once output");
    hip_ok(hipFree(d_blas), "free M256 hipBLAS output");
    hip_ok(hipFree(d_production), "free M256 production output");
    hip_ok(hipFree(d_reference), "free M256 reference output");
    hip_ok(hipFree(d_x_bf16), "free M256 BF16 input");
    hip_ok(hipFree(d_x), "free M256 F32 input");
}

void launch_skinny_exact_tiled(float *out, const uint16_t *weight,
                               const float *x, uint32_t in_dim,
                               uint32_t out_dim, uint32_t tokens) {
    uint32_t first = 0u;
    const uint32_t chunks32 = tokens / 32u;
    if (chunks32 > 0u) {
        matmul_bf16_f32_skinny_exact_toktile_kernel<32u><<<
            dim3(out_dim, chunks32), kThreads>>>(
            out, weight, x, in_dim, out_dim);
        hip_ok(hipGetLastError(), "skinny exact tile32 launch");
        first = chunks32 * 32u;
    }
#define LAUNCH_SKINNY_EXACT_TAIL(T) do {                                \
        if (tokens - first >= (T)) {                                    \
            matmul_bf16_f32_skinny_exact_toktile_kernel<(T)><<<         \
                out_dim, kThreads>>>(                                   \
                out + (uint64_t)first * out_dim, weight,                \
                x + (uint64_t)first * in_dim, in_dim, out_dim);         \
            hip_ok(hipGetLastError(), "skinny exact tail launch");     \
            first += (T);                                               \
        }                                                               \
    } while (0)
    LAUNCH_SKINNY_EXACT_TAIL(16u);
    LAUNCH_SKINNY_EXACT_TAIL(8u);
    LAUNCH_SKINNY_EXACT_TAIL(4u);
    LAUNCH_SKINNY_EXACT_TAIL(2u);
    LAUNCH_SKINNY_EXACT_TAIL(1u);
#undef LAUNCH_SKINNY_EXACT_TAIL
    if (first != tokens) fail("skinny exact token decomposition");
}

void run_skinny_exact_shape(const uint16_t *weight, uint32_t in_dim,
                            uint32_t out_dim, uint32_t tokens) {
    const uint64_t x_count = (uint64_t)tokens * in_dim;
    const uint64_t out_count = (uint64_t)tokens * out_dim;
    std::vector<float> x(x_count);
    for (uint64_t i = 0u; i < x_count; ++i)
        x[i] = 0.41f * std::cos((double)(i % 4093u) * 0.011) -
               0.07f * std::sin((double)i * 0.003);
    float *d_x = nullptr, *d_reference = nullptr, *d_candidate = nullptr;
    float *d_candidate16 = nullptr, *d_candidate8 = nullptr;
    hip_ok(hipMalloc(&d_x, x_count * sizeof(float)),
           "allocate skinny exact input");
    hip_ok(hipMalloc(&d_reference, out_count * sizeof(float)),
           "allocate skinny exact reference");
    hip_ok(hipMalloc(&d_candidate, out_count * sizeof(float)),
           "allocate skinny exact candidate");
    hip_ok(hipMalloc(&d_candidate16, out_count * sizeof(float)),
           "allocate skinny exact candidate16");
    hip_ok(hipMalloc(&d_candidate8, out_count * sizeof(float)),
           "allocate skinny exact candidate8");
    hip_ok(hipMemcpy(d_x, x.data(), x_count * sizeof(float),
                     hipMemcpyHostToDevice), "copy skinny exact input");
    const auto reference = [&] {
        current_bf16_f32_kernel<<<
            dim3(out_dim, tokens), kThreads>>>(
            d_reference, weight, d_x, in_dim, out_dim, tokens);
        hip_ok(hipGetLastError(), "skinny exact reference launch");
    };
    const auto candidate = [&] {
        launch_skinny_exact_tiled(d_candidate, weight, d_x, in_dim,
                                  out_dim, tokens);
    };
    const auto candidate16 = [&] {
        if ((tokens % 16u) != 0u) fail("skinny exact tile16 divisibility");
        matmul_bf16_f32_skinny_exact_toktile_kernel<16u><<<
            dim3(out_dim, tokens / 16u), kThreads>>>(
            d_candidate16, weight, d_x, in_dim, out_dim);
        hip_ok(hipGetLastError(), "skinny exact candidate16 launch");
    };
    const auto candidate8 = [&] {
        if ((tokens % 8u) != 0u) fail("skinny exact tile8 divisibility");
        matmul_bf16_f32_skinny_exact_toktile_kernel<8u><<<
            dim3(out_dim, tokens / 8u), kThreads>>>(
            d_candidate8, weight, d_x, in_dim, out_dim);
        hip_ok(hipGetLastError(), "skinny exact candidate8 launch");
    };
    const float reference_ms = time_ms(reference);
    const float candidate_ms = time_ms(candidate);
    const float candidate16_ms = (tokens % 16u) == 0u
        ? time_ms(candidate16) : 0.0f;
    const float candidate8_ms = (tokens % 8u) == 0u
        ? time_ms(candidate8) : 0.0f;
    reference();
    candidate();
    if ((tokens % 16u) == 0u) candidate16();
    if ((tokens % 8u) == 0u) candidate8();
    hip_ok(hipDeviceSynchronize(), "skinny exact synchronize");
    std::vector<float> host_reference(out_count), host_candidate(out_count);
    std::vector<float> host_candidate16(out_count), host_candidate8(out_count);
    hip_ok(hipMemcpy(host_reference.data(), d_reference,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read skinny exact reference");
    hip_ok(hipMemcpy(host_candidate.data(), d_candidate,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read skinny exact candidate");
    if ((tokens % 16u) == 0u)
        hip_ok(hipMemcpy(host_candidate16.data(), d_candidate16,
                         out_count * sizeof(float), hipMemcpyDeviceToHost),
               "read skinny exact candidate16");
    if ((tokens % 8u) == 0u)
        hip_ok(hipMemcpy(host_candidate8.data(), d_candidate8,
                         out_count * sizeof(float), hipMemcpyDeviceToHost),
               "read skinny exact candidate8");
    const Error error = compare(host_reference, host_candidate);
    const bool bit_exact =
        std::memcmp(host_reference.data(), host_candidate.data(),
                    out_count * sizeof(float)) == 0;
    const bool bit_exact16 = (tokens % 16u) != 0u ||
        std::memcmp(host_reference.data(), host_candidate16.data(),
                    out_count * sizeof(float)) == 0;
    const bool bit_exact8 = (tokens % 8u) != 0u ||
        std::memcmp(host_reference.data(), host_candidate8.data(),
                    out_count * sizeof(float)) == 0;
    std::printf(
        "BF16 skinny exact shape=%ux%ux%u residency=host-registered "
        "generic_ms=%.4f tile32_ms=%.4f tile16_ms=%.4f tile8_ms=%.4f "
        "tile32_speedup=%.3fx tile16_speedup=%.3fx tile8_speedup=%.3fx "
        "nrmse=%.9g cosine=%.12g max_abs=%.9g "
        "bit_exact32=%d bit_exact16=%d bit_exact8=%d\n",
        tokens, out_dim, in_dim, reference_ms, candidate_ms,
        candidate16_ms, candidate8_ms, reference_ms / candidate_ms,
        candidate16_ms > 0.0f ? reference_ms / candidate16_ms : 0.0f,
        candidate8_ms > 0.0f ? reference_ms / candidate8_ms : 0.0f,
        error.nrmse, error.cosine, error.max_abs, bit_exact ? 1 : 0,
        bit_exact16 ? 1 : 0, bit_exact8 ? 1 : 0);
    if (!bit_exact || !bit_exact16 || !bit_exact8)
        fail("skinny exact token tile bit identity");
    hip_ok(hipFree(d_candidate8), "free skinny exact candidate8");
    hip_ok(hipFree(d_candidate16), "free skinny exact candidate16");
    hip_ok(hipFree(d_candidate), "free skinny exact candidate");
    hip_ok(hipFree(d_reference), "free skinny exact reference");
    hip_ok(hipFree(d_x), "free skinny exact input");
}

void run_rowtile_shape(const uint16_t *weight, uint32_t in_dim,
                       uint32_t out_dim) {
    constexpr uint32_t tokens = 256u;
    const uint64_t x_count = (uint64_t)tokens * in_dim;
    const uint64_t out_count = (uint64_t)tokens * out_dim;
    std::vector<float> x(x_count);
    for (uint64_t i = 0u; i < x_count; ++i)
        x[i] = 0.29f * std::cos((double)(i % 8191u) * 0.007) +
               0.11f * std::sin((double)i * 0.005);
    float *d_x = nullptr, *d_reference = nullptr;
    float *d_row2 = nullptr, *d_row4 = nullptr, *d_row8 = nullptr;
    hip_ok(hipMalloc(&d_x, x_count * sizeof(float)),
           "allocate rowtile input");
    hip_ok(hipMalloc(&d_reference, out_count * sizeof(float)),
           "allocate rowtile reference");
    hip_ok(hipMalloc(&d_row2, out_count * sizeof(float)),
           "allocate rowtile 2x16 output");
    hip_ok(hipMalloc(&d_row4, out_count * sizeof(float)),
           "allocate rowtile 4x8 output");
    hip_ok(hipMalloc(&d_row8, out_count * sizeof(float)),
           "allocate rowtile 8x4 output");
    hip_ok(hipMemcpy(d_x, x.data(), x_count * sizeof(float),
                     hipMemcpyHostToDevice), "copy rowtile input");
    const auto reference = [&] {
        launch_segmented_tiled(d_reference, weight, d_x, in_dim,
                               out_dim, tokens);
    };
    const auto row2 = [&] {
        matmul_bf16_f32_rowtile_w32_kernel<2u, 16u><<<
            dim3((out_dim + 1u) / 2u, tokens / 16u), kThreads>>>(
            d_row2, weight, d_x, in_dim, out_dim);
        hip_ok(hipGetLastError(), "rowtile 2x16 launch");
    };
    const auto row4 = [&] {
        matmul_bf16_f32_rowtile_w32_kernel<4u, 8u><<<
            dim3((out_dim + 3u) / 4u, tokens / 8u), kThreads>>>(
            d_row4, weight, d_x, in_dim, out_dim);
        hip_ok(hipGetLastError(), "rowtile 4x8 launch");
    };
    const auto row8 = [&] {
        matmul_bf16_f32_rowtile_w32_kernel<8u, 4u><<<
            dim3((out_dim + 7u) / 8u, tokens / 4u), kThreads>>>(
            d_row8, weight, d_x, in_dim, out_dim);
        hip_ok(hipGetLastError(), "rowtile 8x4 launch");
    };
    const float reference_ms = time_ms(reference);
    const float row2_ms = time_ms(row2);
    const float row4_ms = time_ms(row4);
    const float row8_ms = time_ms(row8);
    reference();
    row2();
    row4();
    row8();
    hip_ok(hipDeviceSynchronize(), "rowtile synchronize");
    std::vector<float> host_reference(out_count), host_row2(out_count);
    std::vector<float> host_row4(out_count), host_row8(out_count);
    hip_ok(hipMemcpy(host_reference.data(), d_reference,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read rowtile reference");
    hip_ok(hipMemcpy(host_row2.data(), d_row2,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read rowtile 2x16");
    hip_ok(hipMemcpy(host_row4.data(), d_row4,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read rowtile 4x8");
    hip_ok(hipMemcpy(host_row8.data(), d_row8,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read rowtile 8x4");
    const bool exact2 = std::memcmp(host_reference.data(), host_row2.data(),
                                    out_count * sizeof(float)) == 0;
    const bool exact4 = std::memcmp(host_reference.data(), host_row4.data(),
                                    out_count * sizeof(float)) == 0;
    const bool exact8 = std::memcmp(host_reference.data(), host_row8.data(),
                                    out_count * sizeof(float)) == 0;
    const Error error2 = compare(host_reference, host_row2);
    const Error error4 = compare(host_reference, host_row4);
    const Error error8 = compare(host_reference, host_row8);
    std::printf(
        "BF16 rowtile shape=%ux%ux%u residency=host-registered "
        "reference_ms=%.4f row2x16_ms=%.4f row4x8_ms=%.4f "
        "row8x4_ms=%.4f speedup2=%.3fx speedup4=%.3fx speedup8=%.3fx "
        "exact2=%d exact4=%d exact8=%d nrmse2=%.9g nrmse4=%.9g "
        "nrmse8=%.9g\n",
        tokens, out_dim, in_dim, reference_ms, row2_ms, row4_ms, row8_ms,
        reference_ms / row2_ms, reference_ms / row4_ms,
        reference_ms / row8_ms, exact2 ? 1 : 0, exact4 ? 1 : 0,
        exact8 ? 1 : 0, error2.nrmse, error4.nrmse, error8.nrmse);
    if (!exact2 || !exact4 || !exact8)
        fail("BF16 adjacent-row token tile bit identity");
    hip_ok(hipFree(d_row8), "free rowtile 8x4 output");
    hip_ok(hipFree(d_row4), "free rowtile 4x8 output");
    hip_ok(hipFree(d_row2), "free rowtile 2x16 output");
    hip_ok(hipFree(d_reference), "free rowtile reference");
    hip_ok(hipFree(d_x), "free rowtile input");
}

void run_rowtile_m2048_guard(const uint16_t *weight, uint32_t in_dim,
                             uint32_t out_dim) {
    constexpr uint32_t tokens = 2048u;
    const uint64_t x_count = (uint64_t)tokens * in_dim;
    const uint64_t out_count = (uint64_t)tokens * out_dim;
    std::vector<float> x(x_count);
    for (uint64_t i = 0u; i < x_count; ++i)
        x[i] = 0.17f * std::cos((double)(i % 12289u) * 0.003) -
               0.13f * std::sin((double)i * 0.002);
    float *d_x = nullptr, *d_reference = nullptr, *d_candidate = nullptr;
    float *d_wmma = nullptr;
    float *d_wmma_hilo = nullptr;
    uint16_t *d_x_bf16 = nullptr;
    hip_ok(hipMalloc(&d_x, x_count * sizeof(float)),
           "allocate M2048 rowtile input");
    hip_ok(hipMalloc(&d_reference, out_count * sizeof(float)),
           "allocate M2048 rowtile reference");
    hip_ok(hipMalloc(&d_candidate, out_count * sizeof(float)),
           "allocate M2048 rowtile candidate");
    hip_ok(hipMalloc(&d_wmma, out_count * sizeof(float)),
           "allocate M2048 multi-N WMMA output");
    hip_ok(hipMalloc(&d_wmma_hilo, out_count * sizeof(float)),
           "allocate M2048 hi/lo WMMA output");
    hip_ok(hipMalloc(&d_x_bf16, x_count * sizeof(uint16_t)),
           "allocate M2048 BF16 input");
    hip_ok(hipMemcpy(d_x, x.data(), x_count * sizeof(float),
                     hipMemcpyHostToDevice), "copy M2048 rowtile input");
    f32_to_bf16_kernel<<<(x_count + 255u) / 256u, 256u>>>(
        d_x_bf16, d_x, x_count);
    hip_ok(hipGetLastError(), "M2048 BF16 conversion launch");
    const auto reference = [&] {
        launch_segmented_tiled(d_reference, weight, d_x, in_dim,
                               out_dim, tokens);
    };
    const auto candidate = [&] {
        matmul_bf16_f32_rowtile_w32_kernel<2u, 16u><<<
            dim3((out_dim + 1u) / 2u, tokens / 16u), kThreads>>>(
            d_candidate, weight, d_x, in_dim, out_dim);
        hip_ok(hipGetLastError(), "M2048 rowtile 2x16 launch");
    };
    const auto wmma = [&] {
        bf16_wmma_m256_multin_kernel<2u><<<
            dim3((out_dim + 31u) / 32u,
                 (tokens + kPrefillTokens - 1u) / kPrefillTokens),
            16u * 32u>>>(d_wmma, weight, d_x_bf16,
                          in_dim, out_dim, tokens);
        hip_ok(hipGetLastError(), "M2048 multi-N WMMA launch");
    };
    const auto wmma_hilo = [&] {
        matmul_bf16_f32_wmma_hilo_m256_kernel<2u><<<
            dim3((out_dim + 31u) / 32u,
                 (tokens + kPrefillTokens - 1u) / kPrefillTokens),
            16u * 32u>>>(d_wmma_hilo, weight, d_x,
                          in_dim, out_dim, tokens);
        hip_ok(hipGetLastError(), "M2048 hi/lo WMMA launch");
    };
    const float reference_ms = time_ms(reference);
    const float candidate_ms = time_ms(candidate);
    const float wmma_ms = time_ms(wmma);
    const float wmma_hilo_ms = time_ms(wmma_hilo);
    reference();
    candidate();
    wmma();
    wmma_hilo();
    hip_ok(hipDeviceSynchronize(), "M2048 rowtile synchronize");
    std::vector<float> host_reference(out_count), host_candidate(out_count);
    std::vector<float> host_wmma(out_count);
    std::vector<float> host_wmma_hilo(out_count);
    hip_ok(hipMemcpy(host_reference.data(), d_reference,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read M2048 rowtile reference");
    hip_ok(hipMemcpy(host_candidate.data(), d_candidate,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read M2048 rowtile candidate");
    hip_ok(hipMemcpy(host_wmma.data(), d_wmma,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read M2048 multi-N WMMA output");
    hip_ok(hipMemcpy(host_wmma_hilo.data(), d_wmma_hilo,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read M2048 hi/lo WMMA output");
    const bool exact = std::memcmp(host_reference.data(),
                                   host_candidate.data(),
                                   out_count * sizeof(float)) == 0;
    const Error error = compare(host_reference, host_candidate);
    const Error wmma_error = compare(host_reference, host_wmma);
    const Error wmma_hilo_error = compare(host_reference, host_wmma_hilo);
    std::printf(
        "BF16 rowtile guard shape=%ux%ux%u residency=host-registered "
        "reference_ms=%.4f row2x16_ms=%.4f speedup=%.3fx exact=%d "
        "nrmse=%.9g wmma_multin_n2_ms=%.4f wmma_speedup=%.3fx "
        "wmma_nrmse=%.9g wmma_max_abs=%.9g "
        "wmma_hilo_ms=%.4f wmma_hilo_speedup=%.3fx "
        "wmma_hilo_nrmse=%.9g wmma_hilo_max_abs=%.9g\n",
        tokens, out_dim, in_dim, reference_ms, candidate_ms,
        reference_ms / candidate_ms, exact ? 1 : 0, error.nrmse,
        wmma_ms, reference_ms / wmma_ms, wmma_error.nrmse,
        wmma_error.max_abs, wmma_hilo_ms, reference_ms / wmma_hilo_ms,
        wmma_hilo_error.nrmse, wmma_hilo_error.max_abs);
    if (!exact) fail("BF16 M2048 adjacent-row token tile bit identity");
    hip_ok(hipFree(d_x_bf16), "free M2048 BF16 input");
    hip_ok(hipFree(d_wmma), "free M2048 multi-N WMMA output");
    hip_ok(hipFree(d_wmma_hilo), "free M2048 hi/lo WMMA output");
    hip_ok(hipFree(d_candidate), "free M2048 rowtile candidate");
    hip_ok(hipFree(d_reference), "free M2048 rowtile reference");
    hip_ok(hipFree(d_x), "free M2048 rowtile input");
}

void run_rowtile_dispatch_guard() {
    if (!ds4_bf16_rowtile2x16_dispatch_allowed(
            true, false, 8192u, 4096u, 256u, false, false))
        fail("BF16 rowtile expected dispatch");
    if (!ds4_bf16_rowtile2x16_dispatch_allowed(
            true, false, 4096u, 4096u, 256u, false, false))
        fail("BF16 rowtile expected TP=2 QKV dispatch");
    if (ds4_bf16_rowtile2x16_dispatch_allowed(
            true, true, 8192u, 4096u, 256u, false, false))
        fail("BF16 rowtile family rollback must disable dispatch");
    if (ds4_bf16_rowtile2x16_dispatch_allowed(
            false, false, 8192u, 4096u, 256u, false, false))
        fail("BF16 rowtile selector rollback must disable dispatch");
    if (ds4_bf16_rowtile2x16_dispatch_allowed(
            true, false, 8192u, 4096u, 255u, false, false))
        fail("BF16 rowtile partial tile must refuse dispatch");
    if (ds4_bf16_rowtile2x16_dispatch_allowed(
            true, false, 8192u, 4096u, 256u, true, false) ||
        ds4_bf16_rowtile2x16_dispatch_allowed(
            true, false, 8192u, 4096u, 256u, false, true))
        fail("BF16 rowtile diagnostic modes must refuse dispatch");
    if (ds4_bf16_rowtile2x16_dispatch_allowed(
            true, false, 4096u, 12288u, 256u, false, false))
        fail("BF16 rowtile unknown projection must refuse dispatch");
    std::printf("BF16 rowtile dispatch and rollback guards pass\n");
}

void run_wmma_hilo_dispatch_guard() {
    if (!ds4_bf16_wmma_hilo_dispatch_allowed(
            true, false, 4096u, 8192u, 256u, false, false) ||
        !ds4_bf16_wmma_hilo_dispatch_allowed(
            true, false, 8192u, 4096u, 2048u, false, false))
        fail("BF16 WMMA hi/lo expected dispatch");
    if (ds4_bf16_wmma_hilo_dispatch_allowed(
            false, false, 4096u, 8192u, 256u, false, false) ||
        ds4_bf16_wmma_hilo_dispatch_allowed(
            true, true, 4096u, 8192u, 256u, false, false) ||
        ds4_bf16_wmma_hilo_dispatch_allowed(
            true, false, 4096u, 8192u, 255u, false, false) ||
        ds4_bf16_wmma_hilo_dispatch_allowed(
            true, false, 4096u, 8192u, 257u, false, false) ||
        ds4_bf16_wmma_hilo_dispatch_allowed(
            true, false, 4096u, 8192u, 300u, false, false) ||
        ds4_bf16_wmma_hilo_dispatch_allowed(
            true, false, 4096u, 8192u, 2051u, false, false) ||
        ds4_bf16_wmma_hilo_dispatch_allowed(
            true, false, 4096u, 8192u, 256u, true, false) ||
        ds4_bf16_wmma_hilo_dispatch_allowed(
            true, false, 4096u, 8192u, 256u, false, true) ||
        ds4_bf16_wmma_hilo_dispatch_allowed(
            true, false, 4096u, 2048u, 256u, false, false))
        fail("BF16 WMMA hi/lo rollback/shape guard");
    if (!ds4_bf16_wmma_hilo_dispatch_allowed(
            true, false, 4096u, 8192u, 4096u, false, false))
        fail("BF16 WMMA hi/lo 4096-row dispatch");
    std::printf("BF16 WMMA hi/lo dispatch and rollback guards pass\n");
}

void run_wmma_hilo_qkv_multiptr(const uint16_t *weight) {
    constexpr uint32_t tokens = 256u;
    constexpr uint32_t in_dim = 4096u;
    constexpr uint32_t out_dim = 4096u;
    constexpr uint64_t weight_count = (uint64_t)in_dim * out_dim;
    constexpr uint64_t x_count = (uint64_t)tokens * in_dim;
    constexpr uint64_t out_count = (uint64_t)tokens * out_dim;
    const uint16_t *weight_q = weight;
    const uint16_t *weight_k = weight + weight_count;
    const uint16_t *weight_v = weight + 2u * weight_count;
    std::vector<float> x(x_count);
    for (uint64_t i = 0u; i < x_count; ++i)
        x[i] = 0.23f * std::cos((double)(i % 8191u) * 0.009) -
               0.08f * std::sin((double)i * 0.004);

    float *d_x = nullptr;
    float *d_seq_q = nullptr, *d_seq_k = nullptr, *d_seq_v = nullptr;
    float *d_fused_q = nullptr, *d_fused_k = nullptr, *d_fused_v = nullptr;
    hip_ok(hipMalloc(&d_x, x_count * sizeof(float)),
           "allocate QKV multiptr input");
    hip_ok(hipMalloc(&d_seq_q, out_count * sizeof(float)),
           "allocate QKV sequential q");
    hip_ok(hipMalloc(&d_seq_k, out_count * sizeof(float)),
           "allocate QKV sequential k");
    hip_ok(hipMalloc(&d_seq_v, out_count * sizeof(float)),
           "allocate QKV sequential v");
    hip_ok(hipMalloc(&d_fused_q, out_count * sizeof(float)),
           "allocate QKV multiptr q");
    hip_ok(hipMalloc(&d_fused_k, out_count * sizeof(float)),
           "allocate QKV multiptr k");
    hip_ok(hipMalloc(&d_fused_v, out_count * sizeof(float)),
           "allocate QKV multiptr v");
    hip_ok(hipMemcpy(d_x, x.data(), x_count * sizeof(float),
                     hipMemcpyHostToDevice), "copy QKV multiptr input");

    const auto sequential = [&] {
        matmul_bf16_f32_wmma_hilo_m256_kernel<2u><<<
            dim3(out_dim / 32u, 1u), 16u * 32u>>>(
                d_seq_q, weight_q, d_x, in_dim, out_dim, tokens);
        matmul_bf16_f32_wmma_hilo_m256_kernel<2u><<<
            dim3(out_dim / 32u, 1u), 16u * 32u>>>(
                d_seq_k, weight_k, d_x, in_dim, out_dim, tokens);
        matmul_bf16_f32_wmma_hilo_m256_kernel<2u><<<
            dim3(out_dim / 32u, 1u), 16u * 32u>>>(
                d_seq_v, weight_v, d_x, in_dim, out_dim, tokens);
        hip_ok(hipGetLastError(), "QKV sequential WMMA launches");
    };
    const auto multiptr = [&] {
        bf16_f32_wmma_hilo_m256_qkv_multiptr_kernel<<<
            3u * (out_dim / 32u), 16u * 32u>>>(
                d_fused_q, d_fused_k, d_fused_v,
                weight_q, weight_k, weight_v, d_x, in_dim, out_dim);
        hip_ok(hipGetLastError(), "QKV multiptr WMMA launch");
    };
    const float sequential_ms = time_ms(sequential);
    const float multiptr_ms = time_ms(multiptr);
    sequential();
    multiptr();
    hip_ok(hipDeviceSynchronize(), "QKV multiptr synchronize");

    std::vector<float> seq(out_count), fused(out_count);
    bool exact = true;
    const float *seq_ptrs[] = {d_seq_q, d_seq_k, d_seq_v};
    const float *fused_ptrs[] = {d_fused_q, d_fused_k, d_fused_v};
    double max_nrmse = 0.0;
    for (uint32_t projection = 0u; projection < 3u; ++projection) {
        hip_ok(hipMemcpy(seq.data(), seq_ptrs[projection],
                         out_count * sizeof(float), hipMemcpyDeviceToHost),
               "read QKV sequential output");
        hip_ok(hipMemcpy(fused.data(), fused_ptrs[projection],
                         out_count * sizeof(float), hipMemcpyDeviceToHost),
               "read QKV multiptr output");
        exact = exact && std::memcmp(seq.data(), fused.data(),
                                     out_count * sizeof(float)) == 0;
        max_nrmse = std::max(max_nrmse, compare(seq, fused).nrmse);
    }
    std::printf(
        "BF16 QKV multiptr shape=3x%ux%ux%u residency=host-registered "
        "sequential_ms=%.4f multiptr_ms=%.4f speedup=%.3fx "
        "bit_exact=%d max_nrmse=%.9g\n",
        tokens, out_dim, in_dim, sequential_ms, multiptr_ms,
        sequential_ms / multiptr_ms, exact ? 1 : 0, max_nrmse);
    if (!exact) fail("QKV multiptr launch collapse must match WMMA outputs");

    hip_ok(hipFree(d_fused_v), "free QKV multiptr v");
    hip_ok(hipFree(d_fused_k), "free QKV multiptr k");
    hip_ok(hipFree(d_fused_q), "free QKV multiptr q");
    hip_ok(hipFree(d_seq_v), "free QKV sequential v");
    hip_ok(hipFree(d_seq_k), "free QKV sequential k");
    hip_ok(hipFree(d_seq_q), "free QKV sequential q");
    hip_ok(hipFree(d_x), "free QKV multiptr input");
}

void run_lowrank128_tail_coverage(const uint16_t *weight, uint32_t tokens) {
    constexpr uint32_t in_dim = 128u;
    constexpr uint32_t out_dim = 8192u;
    const uint64_t x_count = (uint64_t)tokens * in_dim;
    const uint64_t out_count = (uint64_t)tokens * out_dim;
    std::vector<float> x(x_count);
    for (uint64_t i = 0u; i < x_count; ++i)
        x[i] = 0.3f * std::cos((double)(i % 1877u) * 0.019) +
               0.04f * std::sin((double)i * 0.005);
    float *d_x = nullptr, *d_reference = nullptr, *d_candidate = nullptr;
    hip_ok(hipMalloc(&d_x, x_count * sizeof(float)),
           "allocate low-rank tail input");
    hip_ok(hipMalloc(&d_reference, out_count * sizeof(float)),
           "allocate low-rank tail reference");
    hip_ok(hipMalloc(&d_candidate, out_count * sizeof(float)),
           "allocate low-rank tail candidate");
    hip_ok(hipMemcpy(d_x, x.data(), x_count * sizeof(float),
                     hipMemcpyHostToDevice), "copy low-rank tail input");
    current_bf16_f32_kernel<<<dim3(out_dim, tokens), kThreads>>>(
        d_reference, weight, d_x, in_dim, out_dim, tokens);
    hip_ok(hipGetLastError(), "low-rank tail reference launch");
    launch_lowrank128_tiled(d_candidate, weight, d_x, out_dim, tokens);
    hip_ok(hipDeviceSynchronize(), "low-rank tail synchronize");
    std::vector<float> reference(out_count), candidate(out_count);
    hip_ok(hipMemcpy(reference.data(), d_reference,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read low-rank tail reference");
    hip_ok(hipMemcpy(candidate.data(), d_candidate,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read low-rank tail candidate");
    if (std::memcmp(reference.data(), candidate.data(),
                    out_count * sizeof(float)) != 0)
        fail("low-rank binary-tail bit identity");
    std::printf(
        "BF16 low-rank token-tile tail tokens=%u decomposition=32+16+8+4+2+1 "
        "bit_exact=1\n", tokens);
    hip_ok(hipFree(d_candidate), "free low-rank tail candidate");
    hip_ok(hipFree(d_reference), "free low-rank tail reference");
    hip_ok(hipFree(d_x), "free low-rank tail input");
}

void run_tail25_candidate(const uint16_t *weight, uint32_t in_dim,
                          uint32_t out_dim, uint32_t tokens) {
    const uint64_t x_count = (uint64_t)tokens * in_dim;
    const uint64_t out_count = (uint64_t)tokens * out_dim;
    std::vector<float> x(x_count);
    for (uint64_t i = 0u; i < x_count; ++i)
        x[i] = 0.6f * std::cos((double)(i % 3251u) * 0.007) -
               0.05f * std::sin((double)i * 0.002);
    float *d_x = nullptr, *d_reference = nullptr, *d_tail25 = nullptr;
    hip_ok(hipMalloc(&d_x, x_count * sizeof(float)),
           "allocate tail25 input");
    hip_ok(hipMalloc(&d_reference, out_count * sizeof(float)),
           "allocate tail25 reference");
    hip_ok(hipMalloc(&d_tail25, out_count * sizeof(float)),
           "allocate tail25 candidate");
    hip_ok(hipMemcpy(d_x, x.data(), x_count * sizeof(float),
                     hipMemcpyHostToDevice), "copy tail25 input");
    const auto reference = [&] {
        launch_segmented_tiled(d_reference, weight, d_x, in_dim, out_dim,
                               tokens);
    };
    const auto tail25 = [&] {
        launch_tail25_tiled(d_tail25, weight, d_x, in_dim, out_dim, tokens);
    };
    const float reference_ms = time_ms(reference);
    const float tail25_ms = time_ms(tail25);
    reference();
    tail25();
    hip_ok(hipDeviceSynchronize(), "tail25 synchronize");
    std::vector<float> host_reference(out_count), host_tail25(out_count);
    hip_ok(hipMemcpy(host_reference.data(), d_reference,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read tail25 reference");
    hip_ok(hipMemcpy(host_tail25.data(), d_tail25,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read tail25 candidate");
    if (std::memcmp(host_reference.data(), host_tail25.data(),
                    out_count * sizeof(float)) != 0)
        fail("tail25 candidate must bit-match segmented token tile");
    std::printf(
        "BF16 fused-tail shape=%ux%ux%u segmented_ms=%.4f "
        "tail25_ms=%.4f tail25_speedup=%.3fx bit_exact=1\n",
        tokens, out_dim, in_dim, reference_ms, tail25_ms,
        reference_ms / tail25_ms);
    hip_ok(hipFree(d_tail25), "free tail25 candidate");
    hip_ok(hipFree(d_reference), "free tail25 reference");
    hip_ok(hipFree(d_x), "free tail25 input");
}

void run_tail_coverage(const uint16_t *weight, uint32_t tokens) {
    constexpr uint32_t in_dim = 1024u;
    constexpr uint32_t out_dim = 1024u;
    const uint64_t x_count = (uint64_t)tokens * in_dim;
    const uint64_t out_count = (uint64_t)tokens * out_dim;
    std::vector<float> x(x_count);
    for (uint64_t i = 0; i < x_count; ++i)
        x[i] = 0.4f * std::cos((double)(i % 1291u) * 0.011) -
               0.02f * std::sin((double)i * 0.003);
    float *d_x = nullptr, *d_reference = nullptr, *d_segmented = nullptr;
    hip_ok(hipMalloc(&d_x, x_count * sizeof(float)),
           "allocate tail F32 input");
    hip_ok(hipMalloc(&d_reference, out_count * sizeof(float)),
           "allocate tail reference");
    hip_ok(hipMalloc(&d_segmented, out_count * sizeof(float)),
           "allocate tail segmented output");
    hip_ok(hipMemcpy(d_x, x.data(), x_count * sizeof(float),
                     hipMemcpyHostToDevice), "copy tail input");
    current_bf16_f32_kernel<<<dim3(out_dim, tokens), kThreads>>>(
        d_reference, weight, d_x, in_dim, out_dim, tokens);
    hip_ok(hipGetLastError(), "tail reference launch");
    launch_segmented_tiled(d_segmented, weight, d_x, in_dim, out_dim,
                           tokens);
    hip_ok(hipDeviceSynchronize(), "tail output synchronize");
    std::vector<float> reference(out_count), segmented(out_count);
    hip_ok(hipMemcpy(reference.data(), d_reference,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read tail reference");
    hip_ok(hipMemcpy(segmented.data(), d_segmented,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read tail segmented output");
    const Error error = compare(reference, segmented);
    if (error.nrmse > 2.0e-7 || error.cosine < 0.999999999 ||
        error.max_abs > 1.0e-5)
        fail("segmented binary-tail accuracy");
    std::printf(
        "BF16 token-tile tail tokens=%u nrmse=%.9g cosine=%.12g "
        "max_abs=%.9g\n",
        tokens, error.nrmse, error.cosine, error.max_abs);
    hip_ok(hipFree(d_segmented), "free tail segmented output");
    hip_ok(hipFree(d_reference), "free tail reference");
    hip_ok(hipFree(d_x), "free tail input");
}

}  // namespace

int main() {
    constexpr uint64_t weight_count = (uint64_t)3u * 4096u * 4096u;
    constexpr uint64_t weight_bytes = weight_count * sizeof(uint16_t);
    void *host_allocation = nullptr;
    if (posix_memalign(&host_allocation, 4096u, weight_bytes) != 0 ||
        !host_allocation)
        fail("allocate page-aligned host weights");
    auto *host_weight = static_cast<uint16_t *>(host_allocation);
    for (uint64_t i = 0; i < weight_count; ++i) {
        const float value = 0.035f * std::sin((double)(i % 7919u) * 0.017);
        host_weight[i] = bf16_rne(value);
    }
    hip_ok(hipHostRegister(host_weight, weight_bytes,
                           hipHostRegisterMapped | hipHostRegisterReadOnly),
           "register mapped read-only weights");
    uint16_t *device_weight = nullptr;
    hip_ok(hipHostGetDevicePointer(
               reinterpret_cast<void **>(&device_weight), host_weight, 0u),
           "resolve mapped weight pointer");
    hipblasHandle_t handle = nullptr;
    blas_ok(hipblasCreate(&handle), "create hipBLAS handle");

    /* GLM-5.3 KDA f_b/g_b expand a 128-wide low-rank state into all 8192
     * channels. This shape was intentionally below the original production
     * token-tile gate, so keep it in the three-arm diagnostic before changing
     * dispatch. */
    run_shape(handle, device_weight, 128u, 8192u);
    run_shape(handle, device_weight, 128u, 4096u);
    run_shape(handle, device_weight, 4096u, 8192u);
    run_shape(handle, device_weight, 8192u, 4096u);
    run_prefill_shape(handle, device_weight, 4096u, 8192u);
    run_prefill_shape(handle, device_weight, 8192u, 4096u);
    run_prefill_shape(handle, device_weight, 128u, 4096u);
    run_prefill_shape(handle, device_weight, 4096u, 64u);
    run_prefill_shape(handle, device_weight, 4096u, 128u);
    run_skinny_exact_shape(device_weight, 16384u, 24u, 256u);
    run_skinny_exact_shape(device_weight, 4096u, 32u, 256u);
    run_skinny_exact_shape(device_weight, 4096u, 128u, 256u);
    run_skinny_exact_shape(device_weight, 4096u, 128u, 63u);
    run_rowtile_shape(device_weight, 4096u, 8192u);
    run_rowtile_shape(device_weight, 8192u, 4096u);
    /* TP=2 presents each 4096->8192 KDA Q/K/V weight as one contiguous
     * 4096-row rank view.  Measure that live square geometry before widening
     * the production dispatch. */
    run_rowtile_shape(device_weight, 4096u, 4096u);
    run_rowtile_m2048_guard(device_weight, 4096u, 8192u);
    run_rowtile_m2048_guard(device_weight, 8192u, 4096u);
    run_rowtile_m2048_guard(device_weight, 4096u, 4096u);
    run_rowtile_dispatch_guard();
    run_wmma_hilo_dispatch_guard();
    run_wmma_hilo_qkv_multiptr(device_weight);
    run_lowrank128_tail_coverage(device_weight, 63u);
    for (const uint32_t tokens : {25u, 57u, 121u, 153u}) {
        run_tail25_candidate(device_weight, 4096u, 8192u, tokens);
        run_tail25_candidate(device_weight, 8192u, 4096u, tokens);
    }
    for (const uint32_t tokens : {16u, 17u, 31u, 47u, 64u})
        run_tail_coverage(device_weight, tokens);

    blas_ok(hipblasDestroy(handle), "destroy hipBLAS handle");
    hip_ok(hipHostUnregister(host_weight), "unregister weights");
    std::free(host_weight);
    std::puts("PASS BF16 real-shape three-arm diagnostic");
    return 0;
}
