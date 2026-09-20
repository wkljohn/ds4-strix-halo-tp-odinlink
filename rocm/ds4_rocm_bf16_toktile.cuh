// Cache-free multi-row BF16-weight projection shared by production and its
// standalone real-shape/tail diagnostic. The including translation unit must
// provide HIP builtins and fixed-width integer types.

#include <rocwmma/rocwmma.hpp>

static constexpr uint32_t kDs4Bf16ToktileThreads = 256u;
static constexpr uint32_t kDs4Bf16ToktileWaves =
    kDs4Bf16ToktileThreads / 32u;
static_assert(kDs4Bf16ToktileThreads % 32u == 0u,
              "BF16 token tile requires complete wave32 groups");

static inline bool ds4_bf16_wmma_hilo_dispatch_allowed(
        bool selector_enabled,
        bool batch_toktile_disabled,
        uint32_t in_dim,
        uint32_t out_dim,
        uint32_t n_tok,
        bool quality_mode,
        bool graph_dump) {
    const bool supported_shape =
        (in_dim == 4096u && out_dim == 4096u) ||
        (in_dim == 4096u && out_dim == 8192u) ||
        (in_dim == 8192u && out_dim == 4096u);
    return selector_enabled && !batch_toktile_disabled && supported_shape &&
        (in_dim % 16u) == 0u && (out_dim % 32u) == 0u &&
        n_tok >= 256u && (n_tok % 256u) == 0u &&
        !quality_mode && !graph_dump;
}

static __device__ __forceinline__ uint16_t ds4_bf16_rne_bits(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u)
        return (uint16_t)((bits >> 16u) | 0x0040u);
    const uint32_t tie_to_even = (bits >> 16u) & 1u;
    return (uint16_t)((bits + 0x00007fffu + tie_to_even) >> 16u);
}

// Separate contiguous hi/lo panels for a vendor GEMM; no weight conversion.
__global__ static void ds4_bf16_hilo_prepare_planar_kernel(
        uint16_t *hi, uint16_t *lo, const float *x, uint64_t count) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float value = x[i];
    const uint16_t high = ds4_bf16_rne_bits(value);
    hi[i] = high;
    lo[i] = ds4_bf16_rne_bits(value - __uint_as_float(uint32_t(high) << 16u));
}

// Transient activation preparation only. The caller owns count * 4 bytes,
// keeps the F32 input alive, and prepares again whenever that input changes.
// This must use exactly the same high/residual conversion as the raw loader.
template <bool Tiled = false, uint32_t Padding = 0u>
__global__ static void ds4_bf16_hilo_prepare_kernel(
        uint32_t *pairs, const float *x, uint64_t count,
        uint32_t in_dim = 4096u) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    // Tiled contract: complete M256/K16 panels, checked by the caller.
    // Write contiguous [M/256,K/16,256,16] pairs for every consumer CTA.
    uint64_t xi = i;
    if (Tiled) {
        constexpr uint32_t slice_values = 4096u + Padding;
        const uint64_t panel_values = (uint64_t)slice_values * (in_dim / 16u);
        const uint64_t within = i % panel_values;
        const uint32_t j = within % slice_values;
        if (j >= 4096u) { pairs[i] = 0u; return; }
        const uint64_t row = (i / panel_values) * 256u + j / 16u;
        const uint64_t col = (within / slice_values) * 16u + j % 16u;
        xi = row * in_dim + col;
    }
    const float xv = x[xi];
    const uint16_t hi = ds4_bf16_rne_bits(xv);
    const float hi_f = __uint_as_float((uint32_t)hi << 16u);
    const uint16_t lo = ds4_bf16_rne_bits(xv - hi_f);
    pairs[i] = (uint32_t)hi | ((uint32_t)lo << 16u);
}

/* gfx1151 BF16-WMMA projection candidate.  One 512-thread workgroup covers a
 * 256-row activation panel and two adjacent 16-column output tiles.  The
 * weight tile is loaded once for all 16 M waves.  Splitting each F32
 * activation into BF16 high and residual terms retains substantially more of
 * the incumbent F32-activation accuracy without a persistent conversion
 * buffer.  Production dispatch remains explicit and shape checked. */
template <uint32_t NTilesN, bool NativeActivation = false,
          bool CoalescedWeights = false>
__global__ __launch_bounds__(16u * 32u, 1)
static void matmul_bf16_f32_wmma_hilo_m256_kernel(
        float *out,
        const uint16_t *weight,
        const float *x,
        uint32_t in_dim,
        uint32_t out_dim,
        uint32_t tokens) {
    static_assert(NTilesN == 2u,
                  "validated BF16 WMMA candidate uses two N tiles");
    constexpr uint32_t BM = 16u;
    constexpr uint32_t BN = 16u;
    constexpr uint32_t BK = 16u;
    constexpr uint32_t MTile = 256u;
    constexpr uint32_t MTiles = MTile / BM;
    constexpr uint32_t NThreads = MTiles * 32u;
    __shared__ uint16_t sh_a_hi[MTile * BK];
    __shared__ uint16_t sh_a_lo[MTile * BK];
    __shared__ uint16_t sh_b[NTilesN * BK * BN];
    const uint32_t tid = threadIdx.x;
    const uint32_t mt = tid >> 5u;
    const uint32_t nbase = blockIdx.x * NTilesN * BN;
    const uint32_t mbase = blockIdx.y * MTile;
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
        for (uint32_t j = tid; j < MTile * BK; j += NThreads) {
            const uint32_t m = j / BK;
            const uint32_t kk = j % BK;
            const uint32_t global_m = mbase + m;
            if (global_m < tokens) {
                const float xv = x[(uint64_t)global_m * in_dim + k0 + kk];
                const uint16_t hi = ds4_bf16_rne_bits(xv);
                const float hi_f = __uint_as_float((uint32_t)hi << 16u);
                sh_a_hi[j] = hi;
                sh_a_lo[j] = NativeActivation ? 0u :
                    ds4_bf16_rne_bits(xv - hi_f);
            } else {
                sh_a_hi[j] = 0u;
                sh_a_lo[j] = 0u;
            }
        }
        if constexpr (CoalescedWeights) {
            // Two adjacent BF16 words per aligned load; identical LDS tile.
            for (uint32_t j = tid; j < NTilesN * BK * BN / 2u; j += NThreads) {
                const uint32_t nt = j / (BK * BN / 2u);
                const uint32_t rem = j % (BK * BN / 2u);
                const uint32_t kk = (rem % (BK / 2u)) * 2u;
                const uint32_t nn = rem / (BK / 2u);
                const uint32_t n = nbase + nt * BN + nn;
                const uint32_t bits = n < out_dim ?
                    *reinterpret_cast<const uint32_t *>(
                        weight + (uint64_t)n * in_dim + k0 + kk) : 0u;
                sh_b[nt * BK * BN + kk * BN + nn] = (uint16_t)bits;
                sh_b[nt * BK * BN + (kk + 1u) * BN + nn] = (uint16_t)(bits >> 16u);
            }
        } else {
            for (uint32_t j = tid; j < NTilesN * BK * BN; j += NThreads) {
                const uint32_t nt = j / (BK * BN);
                const uint32_t rem = j % (BK * BN);
                const uint32_t kk = rem / BN;
                const uint32_t nn = rem % BN;
                const uint32_t n = nbase + nt * BN + nn;
                sh_b[j] = n < out_dim ?
                    weight[(uint64_t)n * in_dim + k0 + kk] : 0u;
            }
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
            if (!NativeActivation) {
                rocwmma::load_matrix_sync(
                    a, reinterpret_cast<const Bf16 *>(
                        sh_a_lo + mt * BM * BK), BK);
                rocwmma::mma_sync(acc[nt], a, b, acc[nt]);
            }
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

// Wider N and deeper staging for the same hi/lo arithmetic. Each wave owns
// M16/N64, with K32 staged once and consumed as K16 high/residual pairs in
// exactly the incumbent order. No prepared weight or activation allocation.
__global__ __launch_bounds__(8u * 32u, 1)
static void matmul_bf16_f32_wmma_hilo_m128n64k32_kernel(
        float *out, const uint16_t *weight, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t tokens) {
    constexpr uint32_t BM = 16u, BN = 16u, BK = 16u;
    constexpr uint32_t MTile = 128u, KStage = 32u, NTilesN = 4u;
    constexpr uint32_t NThreads = 256u;
    __shared__ uint16_t sh_a_hi[MTile * KStage];
    __shared__ uint16_t sh_a_lo[MTile * KStage];
    __shared__ uint16_t sh_b[NTilesN * KStage * BN];
    const uint32_t tid = threadIdx.x;
    const uint32_t mt = tid >> 5u;
    const uint32_t nbase = blockIdx.x * NTilesN * BN;
    const uint32_t mbase = blockIdx.y * MTile;
    if (mbase >= tokens) return;
    using Bf16 = rocwmma::bfloat16_t;
    using FragA = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK,
                                   Bf16, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK,
                                   Bf16, rocwmma::col_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    FragA a_hi, a_lo;
    FragB b;
    FragC acc[NTilesN];
#pragma unroll
    for (uint32_t nt = 0u; nt < NTilesN; ++nt)
        rocwmma::fill_fragment(acc[nt], 0.0f);
    for (uint32_t k0 = 0u; k0 < in_dim; k0 += KStage) {
        for (uint32_t j = tid; j < MTile * KStage; j += NThreads) {
            const uint32_t m = mbase + j / KStage;
            const uint32_t kk = j % KStage;
            const float xv = m < tokens ? x[(uint64_t)m * in_dim + k0 + kk] : 0.0f;
            const uint16_t hi = ds4_bf16_rne_bits(xv);
            sh_a_hi[j] = hi;
            sh_a_lo[j] = ds4_bf16_rne_bits(
                xv - __uint_as_float((uint32_t)hi << 16u));
        }
        // Keep original [N,K] order in LDS. Column-major B consumes this
        // directly, avoiding stride-16 BF16 stores across the K lanes.
        for (uint32_t j = tid; j < NTilesN * KStage * BN; j += NThreads) {
            const uint32_t nlocal = j / KStage;
            const uint32_t kk = j % KStage;
            const uint32_t n = nbase + nlocal;
            sh_b[j] = n < out_dim ?
                weight[(uint64_t)n * in_dim + k0 + kk] : 0u;
        }
        __syncthreads();
#pragma unroll
        for (uint32_t ks = 0u; ks < KStage; ks += BK) {
            rocwmma::load_matrix_sync(a_hi, reinterpret_cast<const Bf16 *>(
                sh_a_hi + mt * BM * KStage + ks), KStage);
            rocwmma::load_matrix_sync(a_lo, reinterpret_cast<const Bf16 *>(
                sh_a_lo + mt * BM * KStage + ks), KStage);
#pragma unroll
            for (uint32_t nt = 0u; nt < NTilesN; ++nt) {
                rocwmma::load_matrix_sync(b, reinterpret_cast<const Bf16 *>(
                    sh_b + nt * KStage * BN + ks), KStage);
                rocwmma::mma_sync(acc[nt], a_hi, b, acc[nt]);
                rocwmma::mma_sync(acc[nt], a_lo, b, acc[nt]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (uint32_t nt = 0u; nt < NTilesN; ++nt) {
        const uint32_t n = nbase + nt * BN;
        if (n < out_dim && mbase + mt * BM < tokens)
            rocwmma::store_matrix_sync(
                out + (uint64_t)(mbase + mt * BM) * out_dim + n,
                acc[nt], out_dim, rocwmma::mem_row_major);
    }
}

// Smaller workgroup geometry. Each wave owns three independent
// 16x16 output tiles. Retain the incumbent K16 high/residual update sequence,
// including at M96 panel tails. Caller admits whole M16/N32/K32 multiples.
template <bool Prepared = false, uint32_t LdsPad = 0u, bool VectorLoads = false,
          bool MultiPointer = false>
__global__ __launch_bounds__(128, 1)
static void matmul_bf16_f32_wmma_hilo_m96n32k32_kernel(
        float *out, const uint16_t *weight, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t tokens,
        const uint32_t *prepared = nullptr,
        float *out_k = nullptr, float *out_v = nullptr,
        const uint16_t *weight_k = nullptr, const uint16_t *weight_v = nullptr) {
    constexpr uint32_t BM = 16u, BN = 16u, BK = 16u;
    constexpr uint32_t MTile = 96u, NTile = 32u, KStage = 32u;
    constexpr uint32_t Ld = KStage+LdsPad;
    static_assert(LdsPad == 0u || LdsPad == 16u,"measured LDS layouts only");
    constexpr uint32_t NThreads = 128u;
    __shared__ __align__(16) uint16_t a_hi[MTile*Ld], a_lo[MTile*Ld];
    __shared__ __align__(16) uint16_t b_tile[NTile*Ld];
    const uint32_t tid = threadIdx.x, wave = tid >> 5u;
    const uint32_t wave_m = (wave & 1u)*3u, wave_n = wave >> 1u;
    uint32_t nblock = blockIdx.x;
    if constexpr (MultiPointer) {
        const uint32_t blocks_per_projection = out_dim/NTile;
        const uint32_t projection = nblock/blocks_per_projection;
        if (projection >= 3u) return;
        nblock %= blocks_per_projection;
        if (projection == 1u) { out = out_k; weight = weight_k; }
        if (projection == 2u) { out = out_v; weight = weight_v; }
    }
    const uint32_t mbase = blockIdx.y*MTile, nbase = nblock*NTile;
    if (mbase >= tokens) return;
    using Bf16 = rocwmma::bfloat16_t;
    using FragA = rocwmma::fragment<rocwmma::matrix_a,BM,BN,BK,Bf16,rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b,BM,BN,BK,Bf16,rocwmma::col_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator,BM,BN,BK,float>;
    FragC acc[3];
#pragma unroll
    for (uint32_t t=0; t<3; ++t) rocwmma::fill_fragment(acc[t],0.0f);
    for (uint32_t k0=0; k0<in_dim; k0+=KStage) {
        if constexpr (Prepared && VectorLoads) {
            for (uint32_t j=tid*4u; j<MTile*KStage; j+=NThreads*4u) {
                const uint32_t m=mbase+j/KStage, k=k0+j%KStage;
                const uint32_t dest=(j/KStage)*Ld+j%KStage;
                const uint4 v = m < tokens ?
                    *reinterpret_cast<const uint4 *>(prepared+uint64_t(m)*in_dim+k) : make_uint4(0,0,0,0);
                *reinterpret_cast<uint2 *>(a_hi+dest) = make_uint2(
                    (v.x&0xffffu)|(v.y<<16u),(v.z&0xffffu)|(v.w<<16u));
                *reinterpret_cast<uint2 *>(a_lo+dest) = make_uint2(
                    (v.x>>16u)|(v.y&0xffff0000u),(v.z>>16u)|(v.w&0xffff0000u));
            }
        } else for (uint32_t j=tid; j<MTile*KStage; j+=NThreads) {
            const uint32_t m = mbase+j/KStage, k = k0+j%KStage;
            const uint32_t dest=(j/KStage)*Ld+j%KStage;
            if constexpr (Prepared) {
                const uint32_t bits = m < tokens ? prepared[uint64_t(m)*in_dim+k] : 0u;
                a_hi[dest] = uint16_t(bits);
                a_lo[dest] = uint16_t(bits>>16u);
            } else {
                const float value = m < tokens ? x[uint64_t(m)*in_dim+k] : 0.0f;
                const uint16_t high = ds4_bf16_rne_bits(value);
                a_hi[dest] = high;
                a_lo[dest] = ds4_bf16_rne_bits(value-__uint_as_float(uint32_t(high)<<16u));
            }
        }
        if constexpr (VectorLoads) {
            for (uint32_t j=tid*8u; j<NTile*KStage; j+=NThreads*8u) {
                const uint32_t n=nbase+j/KStage, k=k0+j%KStage;
                *reinterpret_cast<uint4 *>(b_tile+(j/KStage)*Ld+j%KStage) = n < out_dim ?
                    *reinterpret_cast<const uint4 *>(weight+uint64_t(n)*in_dim+k) : make_uint4(0,0,0,0);
            }
        } else for (uint32_t j=tid; j<NTile*KStage; j+=NThreads) {
            const uint32_t n=nbase+j/KStage, k=k0+j%KStage;
            b_tile[(j/KStage)*Ld+j%KStage] = n < out_dim ? weight[uint64_t(n)*in_dim+k] : 0u;
        }
        __syncthreads();
#pragma unroll 1
        for (uint32_t ks=0; ks<KStage; ks+=BK) {
            FragB b;
            rocwmma::load_matrix_sync(b,reinterpret_cast<const Bf16 *>(b_tile+wave_n*BN*Ld+ks),Ld);
#pragma unroll
            for (uint32_t t=0; t<3; ++t) {
                const uint32_t mt=wave_m+t;
                FragA a;
                rocwmma::load_matrix_sync(a,reinterpret_cast<const Bf16 *>(a_hi+mt*BM*Ld+ks),Ld);
                rocwmma::mma_sync(acc[t],a,b,acc[t]);
                rocwmma::load_matrix_sync(a,reinterpret_cast<const Bf16 *>(a_lo+mt*BM*Ld+ks),Ld);
                rocwmma::mma_sync(acc[t],a,b,acc[t]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (uint32_t t=0; t<3; ++t) {
        const uint32_t m=mbase+(wave_m+t)*BM, n=nbase+wave_n*BN;
        if (m < tokens && n < out_dim)
            rocwmma::store_matrix_sync(out+uint64_t(m)*out_dim+n,acc[t],out_dim,rocwmma::mem_row_major);
    }
}

/* Collapse the three equal-shape GLM KDA Q/K/V projections into one grid.
 * Each workgroup retains the validated single-projection arithmetic and owns
 * exactly one weight/output pointer; removing the two kernel boundaries lets
 * gfx1151 fill the final occupancy wave with work from the next projection.
 * No weight concatenation or persistent cache is used. */
template <uint32_t PreparedLayout = 0u>
__global__ __launch_bounds__(16u * 32u, 1)
static void matmul_bf16_f32_wmma_hilo_qkv_multiptr_kernel(
        float *out_q, float *out_k, float *out_v,
        const uint16_t *weight_q, const uint16_t *weight_k,
        const uint16_t *weight_v, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t tokens,
        const uint32_t *prepared = nullptr) {
    constexpr uint32_t BM = 16u;
    constexpr uint32_t BN = 16u;
    constexpr uint32_t BK = 16u;
    constexpr uint32_t MTile = 256u;
    constexpr uint32_t MTiles = MTile / BM;
    constexpr uint32_t NTilesN = 2u;
    constexpr uint32_t NThreads = MTiles * 32u;
    static_assert(PreparedLayout <= 3u, "raw, row-major, tiled or padded activation panel");
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
    const uint32_t mbase = blockIdx.y * MTile;
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
        for (uint32_t j = tid; j < MTile * BK; j += NThreads) {
            const uint32_t m = j / BK;
            const uint32_t kk = j % BK;
            const uint32_t global_m = mbase + m;
            if (global_m < tokens) {
                const uint64_t xi = (uint64_t)global_m * in_dim + k0 + kk;
                if (PreparedLayout != 0u) {
                    constexpr uint32_t panel_stride =
                        MTile * BK + (PreparedLayout == 3u ? 16u : 0u);
                    const uint64_t pi = PreparedLayout >= 2u
                        ? (uint64_t)(mbase / MTile) * (in_dim / BK) * panel_stride +
                          (uint64_t)(k0 / BK) * panel_stride + j
                        : xi;
                    const uint32_t pair = prepared[pi];
                    sh_a_hi[j] = (uint16_t)pair;
                    sh_a_lo[j] = (uint16_t)(pair >> 16u);
                } else {
                    const float xv = x[xi];
                    const uint16_t hi = ds4_bf16_rne_bits(xv);
                    const float hi_f = __uint_as_float((uint32_t)hi << 16u);
                    sh_a_hi[j] = hi;
                    sh_a_lo[j] = ds4_bf16_rne_bits(xv - hi_f);
                }
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
        if (n0 < out_dim && mbase + mt * BM < tokens)
            rocwmma::store_matrix_sync(
                out + (uint64_t)(mbase + mt * BM) * out_dim + n0,
                acc[nt], out_dim, rocwmma::mem_row_major);
    }
}

/* True shared-A variant of the square Q/K/V prefill path.  The older
 * multipointer kernel above combines three grids, but each workgroup still
 * owns one projection and therefore stages the activation independently.  In
 * this variant a 24-wave workgroup is partitioned into three eight-wave
 * projection groups.  Every group consumes the same 128-row high/residual A
 * tile while retaining an independent GGUF weight pointer and accumulator.
 *
 * The smaller M tile is deliberate: gfx1151 cannot launch the 48 waves that
 * would be needed to give all three projections the incumbent 256-row tile.
 * Keeping one accumulator set per wave avoids a three-way register multiply;
 * the trade is two y-grid blocks for a 256-token prompt.  The host gate only
 * admits complete 128-row tiles, so rocWMMA stores never touch a partial row.
 * No concatenated or expanded weight storage is used. */
__global__ __launch_bounds__(24u * 32u, 1)
static void matmul_bf16_f32_wmma_hilo_qkv_shared_a_kernel(
        float *out_q, float *out_k, float *out_v,
        const uint16_t *weight_q, const uint16_t *weight_k,
        const uint16_t *weight_v, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t tokens) {
    constexpr uint32_t BM = 16u;
    constexpr uint32_t BN = 16u;
    constexpr uint32_t BK = 16u;
    constexpr uint32_t Projections = 3u;
    constexpr uint32_t ProjectionWaves = 8u;
    constexpr uint32_t ActiveWaves = Projections * ProjectionWaves;
    constexpr uint32_t MTile = ProjectionWaves * BM;
    constexpr uint32_t NTilesN = 2u;
    constexpr uint32_t BTileElems = NTilesN * BK * BN;
    constexpr uint32_t NThreads = ActiveWaves * 32u;
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const bool active = wave < ActiveWaves;
    const uint32_t projection = active ? wave / ProjectionWaves : 0u;
    const uint32_t mt = active ? wave % ProjectionWaves : 0u;
    const uint32_t nbase = blockIdx.x * NTilesN * BN;
    const uint32_t mbase = blockIdx.y * MTile;
    if (mbase >= tokens) return;

    using Bf16 = rocwmma::bfloat16_t;
    using FragA = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK,
                                     Bf16, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK,
                                     Bf16, rocwmma::row_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK,
                                     float>;
    __shared__ uint16_t sh_a_hi[MTile * BK];
    __shared__ uint16_t sh_a_lo[MTile * BK];
    __shared__ uint16_t sh_b[Projections * BTileElems];
    FragA a;
    FragB b;
    FragC acc[NTilesN];
#pragma unroll
    for (uint32_t nt = 0u; nt < NTilesN; ++nt)
        rocwmma::fill_fragment(acc[nt], 0.0f);

    const uint16_t *weights[Projections] = {
        weight_q, weight_k, weight_v,
    };
    float *outputs[Projections] = {out_q, out_k, out_v};
    const uint16_t *weight = weights[projection];
    float *out = outputs[projection];

    for (uint32_t k0 = 0u; k0 < in_dim; k0 += BK) {
        /* One conversion of each activation serves all three projection
         * groups.  The two halves retain the incumbent hi/lo arithmetic. */
        for (uint32_t j = tid; tid < NThreads && j < MTile * BK;
             j += NThreads) {
            const uint32_t m = j / BK;
            const uint32_t kk = j % BK;
            const uint32_t global_m = mbase + m;
            if (global_m < tokens) {
                const float xv = x[(uint64_t)global_m * in_dim + k0 + kk];
                const uint16_t hi = ds4_bf16_rne_bits(xv);
                const float hi_f = __uint_as_float((uint32_t)hi << 16u);
                sh_a_hi[j] = hi;
                sh_a_lo[j] = ds4_bf16_rne_bits(xv - hi_f);
            } else {
                sh_a_hi[j] = 0u;
                sh_a_lo[j] = 0u;
            }
        }
        for (uint32_t j = tid; tid < NThreads &&
             j < Projections * BTileElems; j += NThreads) {
            const uint32_t p = j / BTileElems;
            const uint32_t rem = j % BTileElems;
            const uint32_t nt = rem / (BK * BN);
            const uint32_t tile_rem = rem % (BK * BN);
            const uint32_t kk = tile_rem / BN;
            const uint32_t nn = tile_rem % BN;
            const uint32_t n = nbase + nt * BN + nn;
            sh_b[j] = n < out_dim
                ? weights[p][(uint64_t)n * in_dim + k0 + kk]
                : 0u;
        }
        __syncthreads();
#pragma unroll
        for (uint32_t nt = 0u; nt < NTilesN; ++nt) {
            if (active) {
                rocwmma::load_matrix_sync(
                    b, reinterpret_cast<const Bf16 *>(
                        sh_b + projection * BTileElems + nt * BK * BN), BN);
                rocwmma::load_matrix_sync(
                    a, reinterpret_cast<const Bf16 *>(
                        sh_a_hi + mt * BM * BK), BK);
                rocwmma::mma_sync(acc[nt], a, b, acc[nt]);
                rocwmma::load_matrix_sync(
                    a, reinterpret_cast<const Bf16 *>(
                        sh_a_lo + mt * BM * BK), BK);
                rocwmma::mma_sync(acc[nt], a, b, acc[nt]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (uint32_t nt = 0u; nt < NTilesN; ++nt) {
        if (active) {
            const uint32_t n0 = nbase + nt * BN;
            if (n0 < out_dim && mbase + mt * BM < tokens)
                rocwmma::store_matrix_sync(
                    out + (uint64_t)(mbase + mt * BM) * out_dim + n0,
                    acc[nt], out_dim, rocwmma::mem_row_major);
        }
    }
}

/* Full-M=256 shared-A variant.  A workgroup keeps the incumbent 16-wave
 * geometry, but each wave owns one Q/K/V accumulator set for the same token
 * rows.  This avoids the padding and second y-grid block of the 128-row
 * partition above.  It is intentionally a separate experiment because the
 * three accumulator sets can raise VGPR pressure on some ROCm toolchains. */
__global__ __launch_bounds__(16u * 32u, 1)
static void matmul_bf16_f32_wmma_hilo_qkv_shared_a_m256_kernel(
        float *out_q, float *out_k, float *out_v,
        const uint16_t *weight_q, const uint16_t *weight_k,
        const uint16_t *weight_v, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t tokens) {
    constexpr uint32_t BM = 16u;
    constexpr uint32_t BN = 16u;
    constexpr uint32_t BK = 16u;
    constexpr uint32_t Projections = 3u;
    constexpr uint32_t MTile = 256u;
    constexpr uint32_t MTiles = MTile / BM;
    constexpr uint32_t NTilesN = 2u;
    constexpr uint32_t BTileElems = NTilesN * BK * BN;
    constexpr uint32_t NThreads = MTiles * 32u;
    const uint32_t tid = threadIdx.x;
    const uint32_t mt = tid >> 5u;
    const uint32_t nbase = blockIdx.x * NTilesN * BN;
    const uint32_t mbase = blockIdx.y * MTile;
    if (mbase >= tokens) return;

    using Bf16 = rocwmma::bfloat16_t;
    using FragA = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK,
                                     Bf16, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK,
                                     Bf16, rocwmma::row_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK,
                                     float>;
    __shared__ uint16_t sh_a_hi[MTile * BK];
    __shared__ uint16_t sh_a_lo[MTile * BK];
    __shared__ uint16_t sh_b[Projections * BTileElems];
    const uint16_t *weights[Projections] = {
        weight_q, weight_k, weight_v,
    };
    float *outputs[Projections] = {out_q, out_k, out_v};
    FragA a;
    FragB b;
    FragC acc[Projections][NTilesN];
#pragma unroll
    for (uint32_t p = 0u; p < Projections; ++p)
#pragma unroll
        for (uint32_t nt = 0u; nt < NTilesN; ++nt)
            rocwmma::fill_fragment(acc[p][nt], 0.0f);

    for (uint32_t k0 = 0u; k0 < in_dim; k0 += BK) {
        for (uint32_t j = tid; j < MTile * BK; j += NThreads) {
            const uint32_t m = j / BK;
            const uint32_t kk = j % BK;
            const uint32_t global_m = mbase + m;
            if (global_m < tokens) {
                const float xv = x[(uint64_t)global_m * in_dim + k0 + kk];
                const uint16_t hi = ds4_bf16_rne_bits(xv);
                const float hi_f = __uint_as_float((uint32_t)hi << 16u);
                sh_a_hi[j] = hi;
                sh_a_lo[j] = ds4_bf16_rne_bits(xv - hi_f);
            } else {
                sh_a_hi[j] = 0u;
                sh_a_lo[j] = 0u;
            }
        }
        for (uint32_t j = tid; j < Projections * BTileElems;
             j += NThreads) {
            const uint32_t p = j / BTileElems;
            const uint32_t rem = j % BTileElems;
            const uint32_t nt = rem / (BK * BN);
            const uint32_t tile_rem = rem % (BK * BN);
            const uint32_t kk = tile_rem / BN;
            const uint32_t nn = tile_rem % BN;
            const uint32_t n = nbase + nt * BN + nn;
            sh_b[j] = n < out_dim
                ? weights[p][(uint64_t)n * in_dim + k0 + kk]
                : 0u;
        }
        __syncthreads();
#pragma unroll
        for (uint32_t p = 0u; p < Projections; ++p) {
#pragma unroll
            for (uint32_t nt = 0u; nt < NTilesN; ++nt) {
                rocwmma::load_matrix_sync(
                    b, reinterpret_cast<const Bf16 *>(
                        sh_b + p * BTileElems + nt * BK * BN), BN);
                rocwmma::load_matrix_sync(
                    a, reinterpret_cast<const Bf16 *>(
                        sh_a_hi + mt * BM * BK), BK);
                rocwmma::mma_sync(acc[p][nt], a, b, acc[p][nt]);
                rocwmma::load_matrix_sync(
                    a, reinterpret_cast<const Bf16 *>(
                        sh_a_lo + mt * BM * BK), BK);
                rocwmma::mma_sync(acc[p][nt], a, b, acc[p][nt]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (uint32_t p = 0u; p < Projections; ++p) {
#pragma unroll
        for (uint32_t nt = 0u; nt < NTilesN; ++nt) {
            const uint32_t n0 = nbase + nt * BN;
            if (n0 < out_dim && mbase + mt * BM < tokens)
                rocwmma::store_matrix_sync(
                    outputs[p] + (uint64_t)(mbase + mt * BM) * out_dim + n0,
                    acc[p][nt], out_dim, rocwmma::mem_row_major);
        }
    }
}

/* Six-pointer prefill probe.  One workgroup owns a 256-row Q/K/V tile and
 * stages its activation once, then computes the two-row f_a/g_a/beta tiles in
 * the same launch using the incumbent exact F32 reduction.  The skinny phase
 * intentionally reads the original activation pointer so its per-lane
 * stride-256 accumulation order is unchanged; this is a launch/weight-pointer
 * fusion experiment, not a new arithmetic path.  All six weights remain
 * independent views into the original GGUF mapping. */
template <bool NativeQKV = false>
__global__ __launch_bounds__(16u * 32u, 1)
static void matmul_bf16_f32_wmma_hilo_kda_six_fused_shared_a_m256_kernel(
        float *out_q, float *out_k, float *out_v,
        float *out_f, float *out_g, float *out_beta,
        const uint16_t *weight_q, const uint16_t *weight_k,
        const uint16_t *weight_v, const uint16_t *weight_f,
        const uint16_t *weight_g, const uint16_t *weight_beta,
        const float *x, uint32_t in_dim, uint32_t q_rows,
        uint32_t low_rows, uint32_t beta_rows, uint32_t tokens) {
    constexpr uint32_t BM = 16u;
    constexpr uint32_t BN = 16u;
    constexpr uint32_t BK = 16u;
    constexpr uint32_t Projections = 3u;
    constexpr uint32_t MTile = 256u;
    constexpr uint32_t MTiles = MTile / BM;
    constexpr uint32_t NTilesN = 2u;
    constexpr uint32_t BTileElems = NTilesN * BK * BN;
    constexpr uint32_t NThreads = MTiles * 32u;
    const uint32_t tid = threadIdx.x;
    const uint32_t mt = tid >> 5u;
    const uint32_t nblock = blockIdx.x;
    const uint32_t q_blocks = (q_rows + 31u) / 32u;
    const uint32_t low_blocks = (low_rows + 1u) / 2u;
    const uint32_t beta_blocks = (beta_rows + 1u) / 2u;
    const uint32_t mbase = blockIdx.y * MTile;
    if (mbase >= tokens) return;

    using Bf16 = rocwmma::bfloat16_t;
    using FragA = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK,
                                     Bf16, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK,
                                     Bf16, rocwmma::row_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK,
                                     float>;
    __shared__ uint16_t sh_a_hi[MTile * BK];
    __shared__ uint16_t sh_a_lo[MTile * BK];
    __shared__ uint16_t sh_b[Projections * BTileElems];
    __shared__ float sh_reduce[8u][NThreads];
    FragA a;
    FragB b;
    FragC acc[Projections][NTilesN];
#pragma unroll
    for (uint32_t p = 0u; p < Projections; ++p)
#pragma unroll
        for (uint32_t nt = 0u; nt < NTilesN; ++nt)
            rocwmma::fill_fragment(acc[p][nt], 0.0f);

    const uint16_t *weights[Projections] = {
        weight_q, weight_k, weight_v,
    };
    float *outputs[Projections] = {out_q, out_k, out_v};
    const uint32_t q_nbase = nblock * NTilesN * BN;
    for (uint32_t k0 = 0u; k0 < in_dim; k0 += BK) {
        for (uint32_t j = tid; j < MTile * BK; j += NThreads) {
            const uint32_t m = j / BK;
            const uint32_t kk = j % BK;
            const uint32_t global_m = mbase + m;
            if (global_m < tokens) {
                const float xv = x[(uint64_t)global_m * in_dim + k0 + kk];
                const uint16_t hi = ds4_bf16_rne_bits(xv);
                const float hi_f = __uint_as_float((uint32_t)hi << 16u);
                sh_a_hi[j] = hi;
                sh_a_lo[j] = NativeQKV ? 0u :
                    ds4_bf16_rne_bits(xv - hi_f);
            } else {
                sh_a_hi[j] = 0u;
                sh_a_lo[j] = 0u;
            }
        }
        for (uint32_t j = tid; j < Projections * BTileElems;
             j += NThreads) {
            const uint32_t p = j / BTileElems;
            const uint32_t rem = j % BTileElems;
            const uint32_t nt = rem / (BK * BN);
            const uint32_t tile_rem = rem % (BK * BN);
            const uint32_t kk = tile_rem / BN;
            const uint32_t nn = tile_rem % BN;
            const uint32_t n = q_nbase + nt * BN + nn;
            sh_b[j] = n < q_rows
                ? weights[p][(uint64_t)n * in_dim + k0 + kk] : 0u;
        }
        __syncthreads();
#pragma unroll
        for (uint32_t p = 0u; p < Projections; ++p)
#pragma unroll
            for (uint32_t nt = 0u; nt < NTilesN; ++nt) {
                rocwmma::load_matrix_sync(
                    b, reinterpret_cast<const Bf16 *>(
                        sh_b + p * BTileElems + nt * BK * BN), BN);
                rocwmma::load_matrix_sync(
                    a, reinterpret_cast<const Bf16 *>(
                        sh_a_hi + mt * BM * BK), BK);
                rocwmma::mma_sync(acc[p][nt], a, b, acc[p][nt]);
                if constexpr (!NativeQKV) {
                    rocwmma::load_matrix_sync(
                        a, reinterpret_cast<const Bf16 *>(
                            sh_a_lo + mt * BM * BK), BK);
                    rocwmma::mma_sync(acc[p][nt], a, b, acc[p][nt]);
                }
            }
        __syncthreads();
    }
#pragma unroll
    for (uint32_t p = 0u; p < Projections; ++p)
#pragma unroll
        for (uint32_t nt = 0u; nt < NTilesN; ++nt) {
            const uint32_t n0 = q_nbase + nt * BN;
            if (n0 < q_rows && mbase + mt * BM < tokens)
                rocwmma::store_matrix_sync(
                    outputs[p] + (uint64_t)(mbase + mt * BM) * q_rows + n0,
                    acc[p][nt], q_rows, rocwmma::mem_row_major);
        }

    /* The q-block grid is at least as wide as either skinny grid for the
     * admitted GLM shape.  Reuse its first low/beta blocks for exact gates. */
    const uint32_t lane = tid & 255u;
    const uint32_t low_row = nblock * 2u + (tid >> 8u);
    const uint32_t beta_row = low_row;
    for (uint32_t p = 0u; p < 3u; ++p) {
        const uint32_t rows = p == 2u ? beta_rows : low_rows;
        const uint32_t row = p == 2u ? beta_row : low_row;
        if (nblock >= (p == 2u ? beta_blocks : low_blocks)) continue;
        const uint16_t *weight = p == 0u ? weight_f :
                                 p == 1u ? weight_g : weight_beta;
        float *out = p == 0u ? out_f : p == 1u ? out_g : out_beta;
        for (uint32_t first = 0u; first < MTile; first += 8u) {
            float sums[8] = {};
            for (uint32_t k = lane; k < in_dim; k += 256u) {
                const float w = row < rows ? __uint_as_float(
                    (uint32_t)weight[(uint64_t)row * in_dim + k] << 16u)
                    : 0.0f;
#pragma unroll
                for (uint32_t t = 0u; t < 8u; ++t)
                    sums[t] += w * x[(uint64_t)(mbase + first + t) *
                                     in_dim + k];
            }
#pragma unroll
            for (uint32_t t = 0u; t < 8u; ++t)
                sh_reduce[t][tid] = sums[t];
            __syncthreads();
            for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
                if (lane < stride) {
#pragma unroll
                    for (uint32_t t = 0u; t < 8u; ++t)
                        sh_reduce[t][tid] += sh_reduce[t][tid + stride];
                }
                __syncthreads();
            }
            if (lane == 0u && row < rows) {
#pragma unroll
                for (uint32_t t = 0u; t < 8u; ++t)
                    out[(uint64_t)(mbase + first + t) * rows + row] =
                        sh_reduce[t][tid];
            }
            __syncthreads();
        }
    }
}

/* Low-register shared-A geometry for the Q/K/V experiment.  Five wave32
 * groups own one 16-row tile for each projection (80 rows total); the
 * sixteenth wave participates in activation/weight staging and then stays
 * idle during MMA.  Unlike the M=256 arm, every active wave retains only one
 * projection's two N-tile accumulators.  The common 80-row tile is intentional
 * and the launcher's y-grid covers the final partial tile with zero padding.
 * This is diagnostic/default-off until a production sequence shows that the
 * reduced per-block parallelism is worth the lower VGPR footprint. */
__global__ __launch_bounds__(16u * 32u, 1)
static void matmul_bf16_f32_wmma_hilo_qkv_shared_a_m80_kernel(
        float *out_q, float *out_k, float *out_v,
        const uint16_t *weight_q, const uint16_t *weight_k,
        const uint16_t *weight_v, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t tokens) {
    constexpr uint32_t BM = 16u;
    constexpr uint32_t BN = 16u;
    constexpr uint32_t BK = 16u;
    constexpr uint32_t Projections = 3u;
    constexpr uint32_t ProjectionWaves = 5u;
    constexpr uint32_t MTile = ProjectionWaves * BM;
    constexpr uint32_t NTilesN = 2u;
    constexpr uint32_t BTileElems = NTilesN * BK * BN;
    constexpr uint32_t NThreads = 16u * 32u;
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t projection = wave / ProjectionWaves;
    const uint32_t mt = wave % ProjectionWaves;
    const uint32_t nbase = blockIdx.x * NTilesN * BN;
    const uint32_t mbase = blockIdx.y * MTile;
    if (mbase >= tokens) return;

    using Bf16 = rocwmma::bfloat16_t;
    using FragA = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK,
                                     Bf16, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK,
                                     Bf16, rocwmma::row_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK,
                                     float>;
    __shared__ uint16_t sh_a_hi[MTile * BK];
    __shared__ uint16_t sh_a_lo[MTile * BK];
    __shared__ uint16_t sh_b[Projections * BTileElems];
    FragA a;
    FragB b;
    FragC acc[NTilesN];
    if (projection < Projections) {
#pragma unroll
        for (uint32_t nt = 0u; nt < NTilesN; ++nt)
            rocwmma::fill_fragment(acc[nt], 0.0f);
    }

    const uint16_t *weights[Projections] = {
        weight_q, weight_k, weight_v,
    };
    float *outputs[Projections] = {out_q, out_k, out_v};
    for (uint32_t k0 = 0u; k0 < in_dim; k0 += BK) {
        /* All 16 waves cooperate here, including the otherwise idle wave. */
        for (uint32_t j = tid; j < MTile * BK; j += NThreads) {
            const uint32_t m = j / BK;
            const uint32_t kk = j % BK;
            const uint32_t global_m = mbase + m;
            if (global_m < tokens) {
                const float xv = x[(uint64_t)global_m * in_dim + k0 + kk];
                const uint16_t hi = ds4_bf16_rne_bits(xv);
                const float hi_f = __uint_as_float((uint32_t)hi << 16u);
                sh_a_hi[j] = hi;
                sh_a_lo[j] = ds4_bf16_rne_bits(xv - hi_f);
            } else {
                sh_a_hi[j] = 0u;
                sh_a_lo[j] = 0u;
            }
        }
        for (uint32_t j = tid; j < Projections * BTileElems;
             j += NThreads) {
            const uint32_t p = j / BTileElems;
            const uint32_t rem = j % BTileElems;
            const uint32_t nt = rem / (BK * BN);
            const uint32_t tile_rem = rem % (BK * BN);
            const uint32_t kk = tile_rem / BN;
            const uint32_t nn = tile_rem % BN;
            const uint32_t n = nbase + nt * BN + nn;
            sh_b[j] = n < out_dim
                ? weights[p][(uint64_t)n * in_dim + k0 + kk]
                : 0u;
        }
        __syncthreads();
        if (projection < Projections) {
#pragma unroll
            for (uint32_t nt = 0u; nt < NTilesN; ++nt) {
                rocwmma::load_matrix_sync(
                    b, reinterpret_cast<const Bf16 *>(
                        sh_b + projection * BTileElems + nt * BK * BN), BN);
                rocwmma::load_matrix_sync(
                    a, reinterpret_cast<const Bf16 *>(
                        sh_a_hi + mt * BM * BK), BK);
                rocwmma::mma_sync(acc[nt], a, b, acc[nt]);
                rocwmma::load_matrix_sync(
                    a, reinterpret_cast<const Bf16 *>(
                        sh_a_lo + mt * BM * BK), BK);
                rocwmma::mma_sync(acc[nt], a, b, acc[nt]);
            }
        }
        __syncthreads();
    }

    if (projection < Projections) {
        float *out = outputs[projection];
#pragma unroll
        for (uint32_t nt = 0u; nt < NTilesN; ++nt) {
            const uint32_t n0 = nbase + nt * BN;
            if (n0 < out_dim && mbase + mt * BM < tokens)
                rocwmma::store_matrix_sync(
                    out + (uint64_t)(mbase + mt * BM) * out_dim + n0,
                    acc[nt], out_dim, rocwmma::mem_row_major);
        }
    }
}

/* M=256 shared-A variant with a single 16-column tile per wave.  The
 * two-column arm above keeps three projection accumulators *and* two N tiles
 * live (six fragments); this arm keeps only three fragments and doubles the
 * x-grid instead.  It is a useful occupancy probe on gfx1151: all three
 * projections still consume the same converted A panel, and no weight cache
 * or concatenated pointer is introduced. */
__global__ __launch_bounds__(16u * 32u, 1)
static void matmul_bf16_f32_wmma_hilo_qkv_shared_a_m256_n1_kernel(
        float *out_q, float *out_k, float *out_v,
        const uint16_t *weight_q, const uint16_t *weight_k,
        const uint16_t *weight_v, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t tokens) {
    constexpr uint32_t BM = 16u;
    constexpr uint32_t BN = 16u;
    constexpr uint32_t BK = 16u;
    constexpr uint32_t Projections = 3u;
    constexpr uint32_t MTile = 256u;
    constexpr uint32_t MTiles = MTile / BM;
    constexpr uint32_t BTileElems = BK * BN;
    constexpr uint32_t NThreads = MTiles * 32u;
    const uint32_t tid = threadIdx.x;
    const uint32_t mt = tid >> 5u;
    const uint32_t nbase = blockIdx.x * BN;
    const uint32_t mbase = blockIdx.y * MTile;
    if (mbase >= tokens) return;

    using Bf16 = rocwmma::bfloat16_t;
    using FragA = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK,
                                     Bf16, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK,
                                     Bf16, rocwmma::row_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK,
                                     float>;
    __shared__ uint16_t sh_a_hi[MTile * BK];
    __shared__ uint16_t sh_a_lo[MTile * BK];
    __shared__ uint16_t sh_b[Projections * BTileElems];
    const uint16_t *weights[Projections] = {
        weight_q, weight_k, weight_v,
    };
    float *outputs[Projections] = {out_q, out_k, out_v};
    FragA a;
    FragB b;
    FragC acc[Projections];
#pragma unroll
    for (uint32_t p = 0u; p < Projections; ++p)
        rocwmma::fill_fragment(acc[p], 0.0f);

    for (uint32_t k0 = 0u; k0 < in_dim; k0 += BK) {
        for (uint32_t j = tid; j < MTile * BK; j += NThreads) {
            const uint32_t m = j / BK;
            const uint32_t kk = j % BK;
            const uint32_t global_m = mbase + m;
            if (global_m < tokens) {
                const float xv = x[(uint64_t)global_m * in_dim + k0 + kk];
                const uint16_t hi = ds4_bf16_rne_bits(xv);
                const float hi_f = __uint_as_float((uint32_t)hi << 16u);
                sh_a_hi[j] = hi;
                sh_a_lo[j] = ds4_bf16_rne_bits(xv - hi_f);
            } else {
                sh_a_hi[j] = 0u;
                sh_a_lo[j] = 0u;
            }
        }
        for (uint32_t j = tid; j < Projections * BTileElems;
             j += NThreads) {
            const uint32_t p = j / BTileElems;
            const uint32_t rem = j % BTileElems;
            const uint32_t kk = rem / BN;
            const uint32_t nn = rem % BN;
            const uint32_t n = nbase + nn;
            sh_b[j] = n < out_dim
                ? weights[p][(uint64_t)n * in_dim + k0 + kk]
                : 0u;
        }
        __syncthreads();
#pragma unroll
        for (uint32_t p = 0u; p < Projections; ++p) {
            rocwmma::load_matrix_sync(
                b, reinterpret_cast<const Bf16 *>(
                    sh_b + p * BTileElems), BN);
            rocwmma::load_matrix_sync(
                a, reinterpret_cast<const Bf16 *>(
                    sh_a_hi + mt * BM * BK), BK);
            rocwmma::mma_sync(acc[p], a, b, acc[p]);
            rocwmma::load_matrix_sync(
                a, reinterpret_cast<const Bf16 *>(
                    sh_a_lo + mt * BM * BK), BK);
            rocwmma::mma_sync(acc[p], a, b, acc[p]);
        }
        __syncthreads();
    }

#pragma unroll
    for (uint32_t p = 0u; p < Projections; ++p) {
        if (nbase < out_dim && mbase + mt * BM < tokens)
            rocwmma::store_matrix_sync(
                outputs[p] + (uint64_t)(mbase + mt * BM) * out_dim + nbase,
                acc[p], out_dim, rocwmma::mem_row_major);
    }
}

/* One-launch six-pointer KDA prefill candidate. QKV keeps hi/lo WMMA;
 * f_a/g_a/beta retain the incumbent 256-lane F32 chains and butterfly.
 * Those small recurrent gates must not silently inherit WMMA arithmetic.
 * Each exact workgroup handles two output rows and reuses each weight over
 * eight tokens. Its reduction storage aliases the existing WMMA panels.
 * All physical GGUF weights remain independent, unchanged pointers.
 * NativeQkv is a component-only Lane B probe: QKV uses BF16-rounded input,
 * while the skinny gates below retain their original F32 arithmetic. */
template <bool CoalescedWeights = false, bool SkinnyOnly = false,
          bool NativeQkv = false>
__global__ __launch_bounds__(16u * 32u, 1)
static void matmul_bf16_f32_wmma_hilo_kda_six_multiptr_kernel(
        float *out_q, float *out_k, float *out_v,
        float *out_f, float *out_g, float *out_beta,
        const uint16_t *weight_q, const uint16_t *weight_k,
        const uint16_t *weight_v, const uint16_t *weight_f,
        const uint16_t *weight_g, const uint16_t *weight_beta,
        const float *x, uint32_t in_dim, uint32_t q_rows,
        uint32_t low_rows, uint32_t beta_rows, uint32_t tokens) {
    static_assert(!NativeQkv || (!SkinnyOnly && !CoalescedWeights),
                  "native QKV probe cannot combine geometry experiments");
    constexpr uint32_t BM = 16u;
    constexpr uint32_t BN = 16u;
    constexpr uint32_t BK = 16u;
    constexpr uint32_t MTile = 256u;
    constexpr uint32_t MTiles = MTile / BM;
    constexpr uint32_t NTilesN = 2u;
    constexpr uint32_t NThreads = MTiles * 32u;
    const uint32_t q_blocks = (q_rows + 31u) / 32u;
    const uint32_t low_blocks = (low_rows + 1u) / 2u;
    const uint32_t beta_blocks = (beta_rows + 1u) / 2u;
    // The standalone split-scheduling probe launches only the small gates
    // here, retaining their original token/reduction order and pointers.
    const uint32_t bx = blockIdx.x + (SkinnyOnly ? 3u*q_blocks : 0u);
    uint32_t projection = 0u;
    uint32_t nblock = 0u;
    uint32_t out_dim = 0u;
    if (bx < 3u * q_blocks) {
        projection = bx / q_blocks;
        nblock = bx % q_blocks;
        out_dim = q_rows;
    } else if (bx < 3u * q_blocks + 2u * low_blocks) {
        const uint32_t local = bx - 3u * q_blocks;
        projection = 3u + local / low_blocks;
        nblock = local % low_blocks;
        out_dim = low_rows;
    } else {
        const uint32_t local = bx - 3u * q_blocks - 2u * low_blocks;
        projection = 5u;
        nblock = local;
        out_dim = beta_rows;
    }
    if (projection >= 6u || nblock >=
            (projection < 3u ? q_blocks :
             projection < 5u ? low_blocks : beta_blocks)) return;
    float *out = projection == 0u ? out_q :
                 projection == 1u ? out_k :
                 projection == 2u ? out_v :
                 projection == 3u ? out_f :
                 projection == 4u ? out_g : out_beta;
    const uint16_t *weight = projection == 0u ? weight_q :
                             projection == 1u ? weight_k :
                             projection == 2u ? weight_v :
                             projection == 3u ? weight_f :
                             projection == 4u ? weight_g : weight_beta;
    const uint32_t tid = threadIdx.x;
    const uint32_t mt = tid >> 5u;
    const uint32_t nbase = nblock * NTilesN * BN;
    const uint32_t mbase = blockIdx.y * MTile;
    if (mbase >= tokens) return;

    union alignas(16) Shared {
        struct {
            uint16_t a_hi[MTile * BK];
            uint16_t a_lo[MTile * BK];
            uint16_t b[NTilesN * BK * BN];
        } wmma;
        float exact[8u][NThreads];
    };
    static_assert(sizeof(Shared) ==
                  (2u * MTile * BK + NTilesN * BK * BN) * sizeof(uint16_t),
                  "exact recurrent gates must not grow the WMMA LDS footprint");
    __shared__ Shared tile;
    if (projection >= 3u) {
        // The sole production launcher rejects incomplete M256 batches.
        // These exact groups therefore need no token-tail masking; any new
        // launcher must retain that precondition or mask its final group.
        const uint32_t lane = tid & 255u;
        const uint32_t row = nblock * 2u + (tid >> 8u);
        for (uint32_t first = 0u; first < MTile; first += 8u) {
            float sums[8] = {};
            for (uint32_t k = lane; k < in_dim; k += 256u) {
                const float w = row < out_dim ? __uint_as_float(
                    (uint32_t)weight[(uint64_t)row * in_dim + k] << 16u) : 0.0f;
#pragma unroll
                for (uint32_t t = 0u; t < 8u; ++t)
                    sums[t] += w * x[(uint64_t)(mbase + first + t) * in_dim + k];
            }
#pragma unroll
            for (uint32_t t = 0u; t < 8u; ++t) tile.exact[t][tid] = sums[t];
            __syncthreads();
            for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
                if (lane < stride) {
#pragma unroll
                    for (uint32_t t = 0u; t < 8u; ++t)
                        tile.exact[t][tid] += tile.exact[t][tid + stride];
                }
                __syncthreads();
            }
            if (lane == 0u && row < out_dim) {
#pragma unroll
                for (uint32_t t = 0u; t < 8u; ++t)
                    out[(uint64_t)(mbase + first + t) * out_dim + row] =
                        tile.exact[t][tid];
            }
            __syncthreads();
        }
        return;
    }

    using Bf16 = rocwmma::bfloat16_t;
    using FragA = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK,
                                     Bf16, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK,
                                     Bf16, rocwmma::row_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK,
                                     float>;
    uint16_t *sh_a_hi = tile.wmma.a_hi;
    uint16_t *sh_a_lo = tile.wmma.a_lo;
    uint16_t *sh_b = tile.wmma.b;
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
            const uint32_t global_m = mbase + m;
            if (global_m < tokens) {
                const float xv = x[(uint64_t)global_m * in_dim + k0 + kk];
                const uint16_t hi = ds4_bf16_rne_bits(xv);
                sh_a_hi[j] = hi;
                if constexpr (!NativeQkv) {
                    const float hi_f = __uint_as_float((uint32_t)hi << 16u);
                    sh_a_lo[j] = ds4_bf16_rne_bits(xv - hi_f);
                }
            } else {
                sh_a_hi[j] = 0u;
                if constexpr (!NativeQkv) sh_a_lo[j] = 0u;
            }
        }
        if constexpr (CoalescedWeights) {
            // Two adjacent BF16 words per aligned load; identical LDS tile.
            for (uint32_t j = tid; j < NTilesN * BK * BN / 2u; j += NThreads) {
                const uint32_t nt = j / (BK * BN / 2u);
                const uint32_t rem = j % (BK * BN / 2u);
                const uint32_t kk = (rem % (BK / 2u)) * 2u;
                const uint32_t nn = rem / (BK / 2u);
                const uint32_t n = nbase + nt * BN + nn;
                const uint32_t bits = n < out_dim ?
                    *reinterpret_cast<const uint32_t *>(
                        weight + (uint64_t)n * in_dim + k0 + kk) : 0u;
                sh_b[nt * BK * BN + kk * BN + nn] = (uint16_t)bits;
                sh_b[nt * BK * BN + (kk + 1u) * BN + nn] = (uint16_t)(bits >> 16u);
            }
        } else {
            for (uint32_t j = tid; j < NTilesN * BK * BN; j += NThreads) {
                const uint32_t nt = j / (BK * BN);
                const uint32_t rem = j % (BK * BN);
                const uint32_t kk = rem / BN;
                const uint32_t nn = rem % BN;
                const uint32_t n = nbase + nt * BN + nn;
                sh_b[j] = n < out_dim ?
                    weight[(uint64_t)n * in_dim + k0 + kk] : 0u;
            }
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
            if constexpr (!NativeQkv) {
                rocwmma::load_matrix_sync(
                    a, reinterpret_cast<const Bf16 *>(
                        sh_a_lo + mt * BM * BK), BK);
                rocwmma::mma_sync(acc[nt], a, b, acc[nt]);
            }
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

static inline bool ds4_bf16_rowtile2x16_dispatch_allowed(
        bool selector_enabled,
        bool batch_toktile_disabled,
        uint32_t in_dim,
        uint32_t out_dim,
        uint32_t n_tok,
        bool quality_mode,
        bool graph_dump) {
    const bool supported_shape =
        (in_dim == 4096u && out_dim == 4096u) ||
        (in_dim == 4096u && out_dim == 8192u) ||
        (in_dim == 8192u && out_dim == 4096u);
    return selector_enabled && !batch_toktile_disabled && supported_shape &&
        n_tok >= 16u && (n_tok % 16u) == 0u && !quality_mode && !graph_dump;
}

static __device__ __forceinline__ float ds4_bf16_ordered_mul(float a,
                                                              float b) {
    float out;
    asm("v_mul_f32 %0, %1, %2" : "=v"(out) : "v"(a), "v"(b));
    return out;
}

static __device__ __forceinline__ float ds4_bf16_ordered_add(float a,
                                                              float b) {
    float out;
    asm("v_add_f32 %0, %1, %2" : "=v"(out) : "v"(a), "v"(b));
    return out;
}

template <uint32_t TokenTile>
__global__ static void matmul_bf16_f32_toktile_w32_kernel(
        float *out,
        const uint16_t *w,
        const float *x,
        uint32_t in_dim,
        uint32_t out_dim) {
    const uint32_t row = blockIdx.x;
    const uint32_t token_base = blockIdx.y * TokenTile;
    if (row >= out_dim) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint16_t *wr = w + (uint64_t)row * in_dim;
    float sums[TokenTile] = {};
    for (uint32_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        const float weight =
            __uint_as_float((uint32_t)wr[i] << 16u);
#pragma unroll
        for (uint32_t token = 0u; token < TokenTile; ++token) {
            sums[token] += weight *
                x[(uint64_t)(token_base + token) * in_dim + i];
        }
    }
    __shared__ float wave_sums[TokenTile][kDs4Bf16ToktileWaves];
#pragma unroll
    for (uint32_t token = 0u; token < TokenTile; ++token) {
        float value = sums[token];
#pragma unroll
        for (uint32_t offset = 16u; offset > 0u; offset >>= 1u)
            value += __shfl_down(value, offset, 32u);
        if (lane == 0u) wave_sums[token][wave] = value;
    }
    __syncthreads();
    if (threadIdx.x < TokenTile) {
        float value = 0.0f;
#pragma unroll
        for (uint32_t wv = 0u; wv < kDs4Bf16ToktileWaves; ++wv)
            value += wave_sums[threadIdx.x][wv];
        out[(uint64_t)(token_base + threadIdx.x) * out_dim + row] = value;
    }
}

/* Reuse each activation value across several adjacent output rows while
 * retaining the token-tile kernel's per-thread K chains and wave32 reduction
 * order.  Keep RowTile*TokenTile fixed at 32 so the accumulator footprint is
 * comparable to the incumbent 1x32 kernel. Production dispatch keeps this
 * behind an explicit, shape-checked research selector until its provider and
 * diverse-prompt promotion gates are complete. */
template <uint32_t RowTile, uint32_t TokenTile>
__global__ static void matmul_bf16_f32_rowtile_w32_kernel(
        float *out,
        const uint16_t *w,
        const float *x,
        uint32_t in_dim,
        uint32_t out_dim) {
    static_assert(RowTile >= 1u && TokenTile >= 1u,
                  "BF16 row/token tiles must be nonzero");
    static_assert(RowTile * TokenTile == 32u,
                  "BF16 row/token tile keeps 32 accumulators");
    const uint32_t row_base = blockIdx.x * RowTile;
    const uint32_t token_base = blockIdx.y * TokenTile;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    float sums[RowTile][TokenTile] = {};
    for (uint32_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        float activations[TokenTile];
#pragma unroll
        for (uint32_t token = 0u; token < TokenTile; ++token)
            activations[token] =
                x[(uint64_t)(token_base + token) * in_dim + i];
#pragma unroll
        for (uint32_t row = 0u; row < RowTile; ++row) {
            const uint32_t output_row = row_base + row;
            const float weight = output_row < out_dim
                ? __uint_as_float((uint32_t)w[
                    (uint64_t)output_row * in_dim + i] << 16u)
                : 0.0f;
#pragma unroll
            for (uint32_t token = 0u; token < TokenTile; ++token)
                sums[row][token] += weight * activations[token];
        }
    }
    __shared__ float
        wave_sums[RowTile][TokenTile][kDs4Bf16ToktileWaves];
#pragma unroll
    for (uint32_t row = 0u; row < RowTile; ++row) {
#pragma unroll
        for (uint32_t token = 0u; token < TokenTile; ++token) {
            float value = sums[row][token];
#pragma unroll
            for (uint32_t offset = 16u; offset > 0u; offset >>= 1u)
                value += __shfl_down(value, offset, 32u);
            if (lane == 0u) wave_sums[row][token][wave] = value;
        }
    }
    __syncthreads();
    if (threadIdx.x < RowTile * TokenTile) {
        const uint32_t row = threadIdx.x / TokenTile;
        const uint32_t token = threadIdx.x % TokenTile;
        const uint32_t output_row = row_base + row;
        if (output_row < out_dim) {
            float value = 0.0f;
#pragma unroll
            for (uint32_t wv = 0u; wv < kDs4Bf16ToktileWaves; ++wv)
                value += wave_sums[row][token][wv];
            out[(uint64_t)(token_base + token) * out_dim + output_row] =
                value;
        }
    }
}

/* Exact-order token tiling for narrow-output BF16 projections.  The generic
 * production kernel owns one [row, token] per block and reduces its 256
 * thread-local K chains through a 256-wide LDS butterfly.  Preserve those
 * chains and that butterfly independently for each token, but reuse every
 * weight load across a 32-token prompt tile.  This is deliberately distinct
 * from the lower-LDS wave reduction above, whose association differs from the
 * generic kernel. */
template <uint32_t TokenTile>
__global__ static void matmul_bf16_f32_skinny_exact_toktile_kernel(
        float *out,
        const uint16_t *w,
        const float *x,
        uint32_t in_dim,
        uint32_t out_dim) {
    static_assert(TokenTile >= 1u && TokenTile <= 32u,
                  "skinny exact BF16 token tile must be 1..32");
    const uint32_t row = blockIdx.x;
    const uint32_t token_base = blockIdx.y * TokenTile;
    const uint32_t tid = threadIdx.x;
    if (row >= out_dim || tid >= kDs4Bf16ToktileThreads) return;
    const uint16_t *wr = w + (uint64_t)row * in_dim;
    float sums[TokenTile] = {};
    for (uint32_t i = tid; i < in_dim; i += kDs4Bf16ToktileThreads) {
        const float weight = __uint_as_float((uint32_t)wr[i] << 16u);
#pragma unroll
        for (uint32_t token = 0u; token < TokenTile; ++token)
            sums[token] += weight *
                x[(uint64_t)(token_base + token) * in_dim + i];
    }
    __shared__ float partial[TokenTile][kDs4Bf16ToktileThreads];
#pragma unroll
    for (uint32_t token = 0u; token < TokenTile; ++token)
        partial[token][tid] = sums[token];
    __syncthreads();
    for (uint32_t stride = kDs4Bf16ToktileThreads >> 1u;
         stride > 0u; stride >>= 1u) {
        if (tid < stride) {
#pragma unroll
            for (uint32_t token = 0u; token < TokenTile; ++token)
                partial[token][tid] += partial[token][tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0u) {
#pragma unroll
        for (uint32_t token = 0u; token < TokenTile; ++token)
            out[(uint64_t)(token_base + token) * out_dim + row] =
                partial[token][0];
    }
}

template <uint32_t TokenTile>
__global__ static void matmul_bf16_f32_kslice_toktile_w32_kernel(
        float *out,
        const uint16_t *w,
        const float *x,
        uint32_t full_in_dim,
        uint32_t k_off,
        uint32_t k_cnt,
        uint32_t out_dim) {
    const uint32_t row = blockIdx.x;
    const uint32_t token_base = blockIdx.y * TokenTile;
    if (row >= out_dim) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint16_t *wr = w + (uint64_t)row * full_in_dim + k_off;
    float sums[TokenTile] = {};
    for (uint32_t i = threadIdx.x; i < k_cnt; i += blockDim.x) {
        const float weight = __uint_as_float((uint32_t)wr[i] << 16u);
#pragma unroll
        for (uint32_t token = 0u; token < TokenTile; ++token) {
            sums[token] += weight *
                x[(uint64_t)(token_base + token) * k_cnt + i];
        }
    }
    __shared__ float wave_sums[TokenTile][kDs4Bf16ToktileWaves];
#pragma unroll
    for (uint32_t token = 0u; token < TokenTile; ++token) {
        float value = sums[token];
#pragma unroll
        for (uint32_t offset = 16u; offset > 0u; offset >>= 1u)
            value += __shfl_down(value, offset, 32u);
        if (lane == 0u) wave_sums[token][wave] = value;
    }
    __syncthreads();
    if (threadIdx.x < TokenTile) {
        float value = 0.0f;
#pragma unroll
        for (uint32_t wv = 0u; wv < kDs4Bf16ToktileWaves; ++wv)
            value += wave_sums[threadIdx.x][wv];
        out[(uint64_t)(token_base + threadIdx.x) * out_dim + row] = value;
    }
}

/* Exact-order GLM-5 KDA low-rank expansion. The scalar 256-thread kernel has
 * only 128 active lanes for this shape. Its first effective reduction steps
 * are (x[lane] + x[lane+64]) + (x[lane+32] + x[lane+96]), followed by the
 * ordinary 32-lane tree. Preserve that order while sharing four BF16 weights
 * across a tile of prompt rows. */
template <uint32_t TokenTile>
__global__ static void matmul_bf16_f32_lowrank128_toktile_w32_kernel(
        float *out,
        const uint16_t *w,
        const float *x,
        uint32_t out_dim) {
    static_assert(TokenTile >= 1u && TokenTile <= 32u,
                  "low-rank BF16 token tile must fit one wave");
    const uint32_t row = blockIdx.x;
    const uint32_t token_base = blockIdx.y * TokenTile;
    const uint32_t lane = threadIdx.x;
    if (row >= out_dim || lane >= 32u) return;
    const uint16_t *wr = w + (uint64_t)row * 128u;
    const float w0 = __uint_as_float((uint32_t)wr[lane] << 16u);
    const float w1 = __uint_as_float((uint32_t)wr[lane + 32u] << 16u);
    const float w2 = __uint_as_float((uint32_t)wr[lane + 64u] << 16u);
    const float w3 = __uint_as_float((uint32_t)wr[lane + 96u] << 16u);
    float sums[TokenTile];
#pragma unroll
    for (uint32_t token = 0u; token < TokenTile; ++token) {
        const float *xr = x + (uint64_t)(token_base + token) * 128u;
        const float p0 = ds4_bf16_ordered_mul(w0, xr[lane]);
        const float p1 = ds4_bf16_ordered_mul(w1, xr[lane + 32u]);
        const float p2 = ds4_bf16_ordered_mul(w2, xr[lane + 64u]);
        const float p3 = ds4_bf16_ordered_mul(w3, xr[lane + 96u]);
        const float pair02 = ds4_bf16_ordered_add(p0, p2);
        const float pair13 = ds4_bf16_ordered_add(p1, p3);
        sums[token] = ds4_bf16_ordered_add(pair02, pair13);
    }
#pragma unroll
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
#pragma unroll
        for (uint32_t token = 0u; token < TokenTile; ++token)
            sums[token] = ds4_bf16_ordered_add(
                sums[token], __shfl_down(sums[token], offset, 32u));
    }
    if (lane == 0u) {
#pragma unroll
        for (uint32_t token = 0u; token < TokenTile; ++token)
            out[(uint64_t)(token_base + token) * out_dim + row] =
                sums[token];
    }
}
