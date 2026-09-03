// DS4 ROCm common embedding/dense-matmul kernels and device helpers.
//
// Included from ds4_cuda.cu before more specialized modules; these helpers are
// intentionally kept static in the single translation unit.

__global__ static void fill_f32_kernel(float *x, uint64_t n, float v) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}

__global__ static void embed_token_hc_kernel(float *out, const unsigned short *w, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_embd * n_hc;
    if (i >= n) return;
    uint32_t e = i % n_embd;
    out[i] = __half2float(reinterpret_cast<const __half *>(w)[(uint64_t)token * n_embd + e]);
}

__global__ static void embed_token_hc_bf16_kernel(float *out, const uint16_t *w,
                                                   uint32_t token, uint32_t n_embd,
                                                   uint32_t n_hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_embd * n_hc;
    if (i >= n) return;
    uint32_t e = i % n_embd;
    out[i] = __uint_as_float((uint32_t)w[(uint64_t)token * n_embd + e] << 16u);
}

__device__ static float embed_q8_0_scale(const unsigned char *blk) {
    const uint16_t bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    return __half2float(__ushort_as_half((unsigned short)bits));
}

__global__ static void embed_token_hc_q8_0_kernel(
        float *out,
        const unsigned char *w,
        uint32_t token,
        uint32_t n_embd,
        uint32_t n_hc) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_hc;
    if (gid >= n) return;
    const uint32_t d = gid % n_embd;
    const uint32_t blocks = (n_embd + 31u) / 32u;
    const uint32_t b = d >> 5u;
    const uint32_t j = d & 31u;
    const unsigned char *blk = w + ((uint64_t)token * blocks + b) * 34u;
    out[gid] = embed_q8_0_scale(blk) * (float)((const int8_t *)(blk + 2u))[j];
}

__global__ static void embed_tokens_hc_q8_0_kernel(
        float *out,
        const int32_t *tokens,
        const unsigned char *w,
        uint32_t n_vocab,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_hc) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n) return;
    const uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    const uint32_t t = tmp / n_hc;
    int32_t tok_i = tokens[t];
    uint32_t tok = tok_i < 0 ? 0u : (uint32_t)tok_i;
    if (tok >= n_vocab) tok = 0;
    const uint32_t blocks = (n_embd + 31u) / 32u;
    const uint32_t b = d >> 5u;
    const uint32_t j = d & 31u;
    const unsigned char *blk = w + ((uint64_t)tok * blocks + b) * 34u;
    out[gid] = embed_q8_0_scale(blk) * (float)((const int8_t *)(blk + 2u))[j];
}

__global__ static void embed_tokens_hc_kernel(
        float *out,
        const int32_t *tokens,
        const __half *w,
        uint32_t n_vocab,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_hc) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t t = tmp / n_hc;
    int32_t tok_i = tokens[t];
    uint32_t tok = tok_i < 0 ? 0u : (uint32_t)tok_i;
    if (tok >= n_vocab) tok = 0;
    out[gid] = __half2float(w[(uint64_t)tok * n_embd + d]);
}

__global__ static void embed_tokens_hc_bf16_kernel(
        float *out, const int32_t *tokens, const uint16_t *w,
        uint32_t n_vocab, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t t = tmp / n_hc;
    int32_t tok_i = tokens[t];
    uint32_t tok = tok_i < 0 ? 0u : (uint32_t)tok_i;
    if (tok >= n_vocab) tok = 0u;
    out[gid] = __uint_as_float((uint32_t)w[(uint64_t)tok * n_embd + d] << 16u);
}

__device__ static float warp_sum_f32(float v);
__device__ static float warp_sum_f32_compensated(float v);

__global__ static void matmul_f16_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += __half2float(wr[i]) * xr[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void matmul_bf16_kernel(
        float *out,
        const uint16_t *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    const uint64_t row = (uint64_t)blockIdx.x;
    const uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const uint16_t *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += __uint_as_float((uint32_t)wr[i] << 16) * xr[i];
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

/* BF16 K-slice over a row-major [out_dim, full_in_dim] matrix.  The input is
 * compact [n_tok, k_cnt], but weight rows retain their full physical stride.
 * This is intentionally distinct from output-row slicing: treating the
 * 8192-wide KDA output weight as a packed 4096-wide matrix would make row 1
 * consume row 0's second half. */
__global__ static void matmul_bf16_kslice_kernel(
        float *out,
        const uint16_t *w,
        const float *x,
        uint32_t full_in_dim,
        uint32_t k_off,
        uint32_t k_cnt,
        uint32_t out_dim,
        uint32_t n_tok) {
    const uint32_t row = blockIdx.x;
    const uint32_t tok = blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const uint16_t *wr = w + (uint64_t)row * full_in_dim + k_off;
    const float *xr = x + (uint64_t)tok * k_cnt;
    for (uint32_t i = threadIdx.x; i < k_cnt; i += blockDim.x)
        sum += __uint_as_float((uint32_t)wr[i] << 16u) * xr[i];
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride)
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0u)
        out[(uint64_t)tok * out_dim + row] = partial[0];
}

#include "ds4_rocm_bf16_toktile.cuh"

__global__ static void matmul_f16_ordered_chunks_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    __shared__ float partial[32];
    const uint32_t tid = threadIdx.x;
    float sum = 0.0f;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = k0; i < k1; i++) {
        sum += __half2float(wr[i]) * xr[i];
    }
    partial[tid] = sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) total += partial[i];
        out[tok * out_dim + row] = total;
    }
}

__global__ static void matmul_f16_f32_sharedx_warp_rows_w32_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint32_t in_dim,
        uint64_t out_dim) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    for (uint32_t i = tid; i < in_dim; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    const uint64_t row = (uint64_t)blockIdx.x * rows_per_block + wave;
    if (row >= out_dim) return;
    const __half *wr = w + row * (uint64_t)in_dim;
    float acc = 0.0f;
    uint32_t i = lane;
    for (; i + 224u < in_dim; i += 256u) {
        acc += __half2float(wr[i]) * shx[i];
        acc += __half2float(wr[i + 32u]) * shx[i + 32u];
        acc += __half2float(wr[i + 64u]) * shx[i + 64u];
        acc += __half2float(wr[i + 96u]) * shx[i + 96u];
        acc += __half2float(wr[i + 128u]) * shx[i + 128u];
        acc += __half2float(wr[i + 160u]) * shx[i + 160u];
        acc += __half2float(wr[i + 192u]) * shx[i + 192u];
        acc += __half2float(wr[i + 224u]) * shx[i + 224u];
    }
    for (; i < in_dim; i += 32u) {
        acc += __half2float(wr[i]) * shx[i];
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) out[row] = acc;
}

/* Decode-sized BF16 projection: one wave32 per output row, with the input
 * vector shared by all waves in the block.  The generic BF16 kernel launches
 * one 256-thread block per row; for DSpark's 256 -> vocab Markov head that is
 * 129,280 tiny blocks and is dominated by scheduling overhead. */
__global__ static void matmul_bf16_f32_sharedx_warp_rows_w32_kernel(
        float *out,
        const uint16_t *w,
        const float *x,
        uint32_t in_dim,
        uint64_t out_dim) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    for (uint32_t i = tid; i < in_dim; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    const uint64_t row = (uint64_t)blockIdx.x * rows_per_block + wave;
    if (row >= out_dim) return;
    const uint16_t *wr = w + row * (uint64_t)in_dim;
    float acc = 0.0f;
    for (uint32_t i = lane; i < in_dim; i += 32u) {
        acc += __uint_as_float((uint32_t)wr[i] << 16) * shx[i];
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) out[row] = acc;
}

/* Decode-only Q/K/V experiment.  A 768-thread block owns eight rows of each
 * independent projection (24 wave32 rows total), stages the shared activation
 * vector once, and then applies the incumbent lane-major BF16->F32 reduction
 * to each output.  The GGUF matrices are never concatenated. */
__global__ static void matmul_bf16_f32_sharedx_qkv_multiptr_decode_kernel(
        float *out_q, float *out_k, float *out_v,
        const uint16_t *weight_q, const uint16_t *weight_k,
        const uint16_t *weight_v, const float *x,
        uint32_t in_dim, uint32_t out_dim) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    constexpr uint32_t rows_per_projection = 8u;
    constexpr uint32_t projections = 3u;
    const uint32_t projection = wave / rows_per_projection;
    const uint32_t row_in_block = wave % rows_per_projection;
    for (uint32_t i = tid; i < in_dim; i += blockDim.x) shx[i] = x[i];
    __syncthreads();
    if (projection >= projections) return;
    const uint64_t row = (uint64_t)blockIdx.x * rows_per_projection +
                         row_in_block;
    if (row >= out_dim) return;
    const uint16_t *weight = projection == 0u ? weight_q :
                             projection == 1u ? weight_k : weight_v;
    float *out = projection == 0u ? out_q :
                 projection == 1u ? out_k : out_v;
    const uint16_t *wr = weight + row * (uint64_t)in_dim;
    float acc = 0.0f;
    for (uint32_t i = lane; i < in_dim; i += 32u)
        acc += __uint_as_float((uint32_t)wr[i] << 16u) * shx[i];
    acc = warp_sum_f32(acc);
    if (lane == 0u) out[row] = acc;
}

/* Exact-order gfx1151 decode matvec.  Each lane retains the original
 * lane,lane+32,... accumulation sequence, but batches 64 independent weight
 * and LDS loads before consuming them.  This raises memory-level parallelism
 * without adding another accumulator or changing the wave reduction tree. */
__global__ static void matmul_bf16_f32_sharedx_mlp64_warp_rows_w32_kernel(
        float *out,
        const uint16_t *w,
        const float *x,
        uint32_t in_dim,
        uint64_t out_dim,
        int split_order) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    for (uint32_t i = tid; i < in_dim; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    const uint64_t row = (uint64_t)blockIdx.x * rows_per_block + wave;
    if (row >= out_dim) return;
    const uint16_t *wr = w + row * (uint64_t)in_dim;
    float acc = 0.0f;
    float acc_hi = 0.0f;
    uint32_t i = lane;
    for (; i + 63u * 32u < in_dim; i += 64u * 32u) {
        uint16_t packed_w[64];
        float packed_x[64];
#pragma unroll
        for (uint32_t u = 0u; u < 64u; ++u) {
            const uint32_t index = i + u * 32u;
            packed_w[u] = wr[index];
            packed_x[u] = shx[index];
        }
#pragma unroll
        for (uint32_t u = 0u; u < 64u; ++u) {
            const float product =
                __uint_as_float((uint32_t)packed_w[u] << 16) * packed_x[u];
            if (split_order && i + u * 32u >= 4096u) acc_hi += product;
            else acc += product;
        }
    }
    for (; i < in_dim; i += 32u) {
        const float product =
            __uint_as_float((uint32_t)wr[i] << 16) * shx[i];
        if (split_order && i >= 4096u) acc_hi += product;
        else acc += product;
    }
    acc = split_order ? warp_sum_f32(acc) + warp_sum_f32(acc_hi)
                      : warp_sum_f32(acc);
    if (lane == 0u) out[row] = acc;
}

/* Research-only exact-order alternative for the GLM-5.3 KDA Q/K/V decode
 * shape.  The compile-time load window and non-temporal policy specialize
 * away the incumbent's runtime split-order branch and unused second
 * accumulator.  Each lane still consumes i=lane+32*k in the identical order,
 * and the wave reduction is unchanged, so a successful arm must bit-match
 * the incumbent. */
template <bool NONTEMPORAL>
__device__ __forceinline__ static uint16_t
matmul_bf16_exact_load(const uint16_t *ptr);

template <>
__device__ __forceinline__ uint16_t
matmul_bf16_exact_load<false>(const uint16_t *ptr) {
    return *ptr;
}

template <>
__device__ __forceinline__ uint16_t
matmul_bf16_exact_load<true>(const uint16_t *ptr) {
    return __builtin_nontemporal_load(ptr);
}

template <uint32_t PREFETCH, bool NONTEMPORAL>
__global__ static void matmul_bf16_f32_sharedx_exact_prefetch_warp_rows_w32_kernel(
        float *out,
        const uint16_t *w,
        const float *x,
        uint32_t in_dim,
        uint64_t out_dim) {
    static_assert(PREFETCH > 0u, "nonzero BF16 prefetch window");
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    for (uint32_t i = tid; i < in_dim; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    const uint64_t row = (uint64_t)blockIdx.x * rows_per_block + wave;
    if (row >= out_dim) return;
    const uint16_t *wr = w + row * (uint64_t)in_dim;
    float acc = 0.0f;
    uint32_t i = lane;
    for (; i + (PREFETCH - 1u) * 32u < in_dim;
         i += PREFETCH * 32u) {
        uint16_t packed_w[PREFETCH];
        float packed_x[PREFETCH];
#pragma unroll
        for (uint32_t u = 0u; u < PREFETCH; ++u) {
            const uint32_t index = i + u * 32u;
            packed_w[u] = matmul_bf16_exact_load<NONTEMPORAL>(&wr[index]);
            packed_x[u] = shx[index];
        }
#pragma unroll
        for (uint32_t u = 0u; u < PREFETCH; ++u) {
            acc += __uint_as_float((uint32_t)packed_w[u] << 16) * packed_x[u];
        }
    }
    for (; i < in_dim; i += 32u) {
        acc += __uint_as_float(
                   (uint32_t)matmul_bf16_exact_load<NONTEMPORAL>(&wr[i]) << 16) *
               shx[i];
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) out[row] = acc;
}

/* Keep the arithmetic body shared by the diagnostic full-width split-order
 * kernel and the candidate 4096-wide K-slice kernel.  The two paths are
 * intended to differ only in ownership and transport, so giving the compiler
 * one source expression for each half prevents surrounding optional modes
 * from silently changing contraction or reassociation. */
__device__ __forceinline__ static float warp_sum_f32_ordered_w32(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float peer = __shfl_down(value, offset, 32);
        float next = 0.0f;
        asm volatile("v_add_f32_e32 %0, %1, %2"
                     : "=v"(next) : "v"(value), "v"(peer));
        value = next;
    }
    return value;
}

template <uint32_t KCount>
__device__ __forceinline__ static float
matmul_bf16_f32_mlp64_half_sum_w32(
        const uint16_t *wr, const float *x, uint32_t lane) {
    float acc = 0.0f;
    uint32_t i = lane;
    for (; i + 63u * 32u < KCount; i += 64u * 32u) {
        uint16_t packed_w[64];
        float packed_x[64];
#pragma unroll
        for (uint32_t u = 0u; u < 64u; ++u) {
            const uint32_t index = i + u * 32u;
            packed_w[u] = wr[index];
            packed_x[u] = x[index];
        }
#pragma unroll
        for (uint32_t u = 0u; u < 64u; ++u) {
            acc = fmaf(__uint_as_float((uint32_t)packed_w[u] << 16u),
                       packed_x[u], acc);
        }
    }
    for (; i < KCount; i += 32u) {
        acc = fmaf(__uint_as_float((uint32_t)wr[i] << 16u), x[i], acc);
    }
    return warp_sum_f32_ordered_w32(acc);
}

__global__ static void
matmul_bf16_f32_sharedx_mlp64_split4096_warp_rows_w32_kernel(
        float *out,
        const uint16_t *w,
        const float *x,
        uint64_t out_dim) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    for (uint32_t i = tid; i < 8192u; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    const uint64_t row = (uint64_t)blockIdx.x * rows_per_block + wave;
    if (row >= out_dim) return;
    const uint16_t *wr = w + row * 8192u;
    const float low = matmul_bf16_f32_mlp64_half_sum_w32<4096u>(
        wr, shx, lane);
    const float high = matmul_bf16_f32_mlp64_half_sum_w32<4096u>(
        wr + 4096u, shx + 4096u, lane);
    if (lane == 0u) out[row] = low + high;
}

/* Decode form of the strided BF16 K-slice above.  Each rank stages only its
 * compact activation half in LDS and reads the matching half of every full
 * matrix row.  Reduction order is deterministic within a slice, but summing
 * two slice outputs is Lane-B-equivalent rather than bit-identical to one
 * 8192-element reduction. */
__global__ static void
matmul_bf16_f32_sharedx_mlp64_kslice_warp_rows_w32_kernel(
        float *out,
        const uint16_t *w,
        const float *x,
        uint32_t full_in_dim,
        uint32_t k_off,
        uint32_t k_cnt,
        uint32_t out_dim,
        int compensated_reduce,
        int pair_accum,
        int kahan_accum) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    for (uint32_t i = tid; i < k_cnt; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    const uint32_t row = blockIdx.x * rows_per_block + wave;
    if (row >= out_dim) return;
    const uint16_t *wr = w + (uint64_t)row * full_in_dim + k_off;
    float acc = 0.0f;
    float acc_pair = 0.0f;
    float acc_c = 0.0f;
    uint32_t i = lane;
    for (; i + 63u * 32u < k_cnt; i += 64u * 32u) {
        uint16_t packed_w[64];
        float packed_x[64];
#pragma unroll
        for (uint32_t u = 0u; u < 64u; ++u) {
            const uint32_t index = i + u * 32u;
            packed_w[u] = wr[index];
            packed_x[u] = shx[index];
        }
#pragma unroll
        for (uint32_t u = 0u; u < 64u; ++u) {
            const float product =
                __uint_as_float((uint32_t)packed_w[u] << 16u) * packed_x[u];
            if (pair_accum && (u & 1u)) acc_pair += product;
            else if (kahan_accum) {
                const float y = product - acc_c;
                const float t = acc + y;
                acc_c = (t - acc) - y;
                acc = t;
            } else acc += product;
        }
    }
    for (; i < k_cnt; i += 32u) {
        const float product =
            __uint_as_float((uint32_t)wr[i] << 16u) * shx[i];
        if (pair_accum && ((i >> 5u) & 1u)) acc_pair += product;
        else if (kahan_accum) {
            const float y = product - acc_c;
            const float t = acc + y;
            acc_c = (t - acc) - y;
            acc = t;
        } else acc += product;
    }
    acc += acc_pair + acc_c;
    acc = compensated_reduce ? warp_sum_f32_compensated(acc)
                             : warp_sum_f32(acc);
    if (lane == 0u) out[row] = acc;
}

__global__ static void
matmul_bf16_f32_sharedx_mlp64_kslice_exact4096_warp_rows_w32_kernel(
        float *out,
        const uint16_t *w,
        const float *x,
        uint32_t full_in_dim,
        uint32_t k_off,
        uint32_t out_dim) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    for (uint32_t i = tid; i < 4096u; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    const uint32_t row = blockIdx.x * rows_per_block + wave;
    if (row >= out_dim) return;
    const uint16_t *wr = w + (uint64_t)row * full_in_dim + k_off;
    const float acc = matmul_bf16_f32_mlp64_half_sum_w32<4096u>(
        wr, shx, lane);
    if (lane == 0u) out[row] = acc;
}

__global__ static void matmul_f16_pair_f32_sharedx_warp_rows_w32_kernel(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const float *x,
        uint32_t in_dim,
        uint64_t out_dim) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    for (uint32_t i = tid; i < in_dim; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    const uint64_t row = (uint64_t)blockIdx.x * rows_per_block + wave;
    if (row >= out_dim) return;
    const __half *wr0 = w0 + row * (uint64_t)in_dim;
    const __half *wr1 = w1 + row * (uint64_t)in_dim;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    uint32_t i = lane;
    for (; i + 224u < in_dim; i += 256u) {
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
    for (; i < in_dim; i += 32u) {
        const float xv = shx[i];
        acc0 += __half2float(wr0[i]) * xv;
        acc1 += __half2float(wr1[i]) * xv;
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0u) {
        out0[row] = acc0;
        out1[row] = acc1;
    }
}

/* Exact temporal compressor projection.  Each token keeps an independent
 * FP32 accumulator and observes the same k-order and wave reduction as the
 * ordinary one-row kernel above.  Reusing the two F16 weight rows across up
 * to four activation rows removes repeated cold model-weight reads without
 * creating an expanded or repacked weight cache. */
template <uint32_t TOKENS>
__global__ static void matmul_f16_pair_f32_temporal_rows_w32_kernel(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const float *x,
        uint32_t in_dim,
        uint32_t out_dim) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    for (uint32_t i = tid; i < TOKENS * in_dim; i += blockDim.x) {
        shx[i] = x[i];
    }
    __syncthreads();

    const uint32_t row = blockIdx.x * rows_per_block + wave;
    if (row >= out_dim) return;
    const __half *wr0 = w0 + (uint64_t)row * in_dim;
    const __half *wr1 = w1 + (uint64_t)row * in_dim;
    float acc0[TOKENS] = {};
    float acc1[TOKENS] = {};
    uint32_t i = lane;
    for (; i + 224u < in_dim; i += 256u) {
#pragma unroll
        for (uint32_t u = 0; u < 8u; u++) {
            const uint32_t k = i + u * 32u;
            const float fw0 = __half2float(wr0[k]);
            const float fw1 = __half2float(wr1[k]);
#pragma unroll
            for (uint32_t t = 0; t < TOKENS; t++) {
                const float xv = shx[(uint64_t)t * in_dim + k];
                acc0[t] += fw0 * xv;
                acc1[t] += fw1 * xv;
            }
        }
    }
    for (; i < in_dim; i += 32u) {
        const float fw0 = __half2float(wr0[i]);
        const float fw1 = __half2float(wr1[i]);
#pragma unroll
        for (uint32_t t = 0; t < TOKENS; t++) {
            const float xv = shx[(uint64_t)t * in_dim + i];
            acc0[t] += fw0 * xv;
            acc1[t] += fw1 * xv;
        }
    }
#pragma unroll
    for (uint32_t t = 0; t < TOKENS; t++) {
        acc0[t] = warp_sum_f32(acc0[t]);
        acc1[t] = warp_sum_f32(acc1[t]);
    }
    if (lane == 0u) {
#pragma unroll
        for (uint32_t t = 0; t < TOKENS; t++) {
            out0[(uint64_t)t * out_dim + row] = acc0[t];
            out1[(uint64_t)t * out_dim + row] = acc1[t];
        }
    }
}

/* DSpark verifier microbatch: reuse each pair of compressor weights across
 * all five rows.  hipBLAS handles these as two skinny GEMMs; on gfx1151 their
 * launch/setup cost and poor N=5 utilization dominate.  The input is converted
 * to F16 once by the caller so this follows the established hipBLAS path's
 * activation precision while keeping each row's reduction independent. */
template <uint32_t TOKENS>
__global__ static void matmul_f16_pair_five_row_kernel(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const __half *x,
        uint32_t in_dim,
        uint32_t out_dim) {
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t row = blockIdx.x * rows_per_block + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;

    float acc0[TOKENS];
    float acc1[TOKENS];
#pragma unroll
    for (uint32_t t = 0; t < TOKENS; t++) {
        acc0[t] = 0.0f;
        acc1[t] = 0.0f;
    }
    const __half *wr0 = w0 + (uint64_t)row * in_dim;
    const __half *wr1 = w1 + (uint64_t)row * in_dim;
    for (uint32_t k = lane; k < in_dim; k += 32u) {
        const float fw0 = __half2float(wr0[k]);
        const float fw1 = __half2float(wr1[k]);
#pragma unroll
        for (uint32_t t = 0; t < TOKENS; t++) {
            const float xv = __half2float(x[(uint64_t)t * in_dim + k]);
            acc0[t] = fmaf(fw0, xv, acc0[t]);
            acc1[t] = fmaf(fw1, xv, acc1[t]);
        }
    }
#pragma unroll
    for (uint32_t t = 0; t < TOKENS; t++) {
        acc0[t] = warp_sum_f32(acc0[t]);
        acc1[t] = warp_sum_f32(acc1[t]);
    }
    if (lane == 0u) {
#pragma unroll
        for (uint32_t t = 0; t < TOKENS; t++) {
            out0[(uint64_t)t * out_dim + row] = acc0[t];
            out1[(uint64_t)t * out_dim + row] = acc1[t];
        }
    }
}

/* DeepSeek V4 Flash HC control projection for the five-row DSpark verifier.
 * The matrix is only 16384x24, so the generic skinny hipBLAS GEMM is dominated
 * by setup and under-utilization.  One 256-thread block cooperates on each
 * output row and reuses every F16 weight across all five activation rows.
 * Twenty-four blocks provide substantially more parallelism than adapting the
 * compressor kernel's eight-output-row block layout to this tiny output. */
template <uint32_t TOKENS>
__global__ static void matmul_f16_hc_five_row_kernel(
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
        for (uint32_t t = 0; t < TOKENS; t++) {
            acc[t] = fmaf(fw,
                          __half2float(x[(uint64_t)t * in_dim + k]),
                          acc[t]);
        }
    }

    enum { WARPS = 8 };
    __shared__ float partial[TOKENS][WARPS];
#pragma unroll
    for (uint32_t t = 0; t < TOKENS; t++) {
        acc[t] = warp_sum_f32(acc[t]);
        if (lane == 0u) partial[t][warp] = acc[t];
    }
    __syncthreads();

    if (warp == 0u) {
#pragma unroll
        for (uint32_t t = 0; t < TOKENS; t++) {
            float v = lane < WARPS ? partial[t][lane] : 0.0f;
            v = warp_sum_f32(v);
            if (lane == 0u) {
                out[(uint64_t)t * out_dim + row] = v;
            }
        }
    }
}

/* Independent token/output reductions for the same HC shape.  This rereads
 * each weight row for every token but exposes 120 blocks, avoids hipBLAS
 * setup, and provides a second reduction order for numerical-path testing. */
__global__ static void matmul_f16_hc_token_block_kernel(
        float *out,
        const __half *w,
        const __half *x,
        uint32_t in_dim,
        uint32_t out_dim) {
    const uint32_t row = blockIdx.x;
    const uint32_t tok = blockIdx.y;
    float sum = 0.0f;
    const __half *wr = w + (uint64_t)row * in_dim;
    const __half *xr = x + (uint64_t)tok * in_dim;
    for (uint32_t k = threadIdx.x; k < in_dim; k += blockDim.x) {
        sum = fmaf(__half2float(wr[k]), __half2float(xr[k]), sum);
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        out[(uint64_t)tok * out_dim + row] = partial[0];
    }
}

/* Up-to-five-row speculative indexer Q projection (K=1024, M=8192 on the
 * tested V4 Flash GGUF; K=1536 is retained for compatible variants).
 * One persistent block per gfx1151 CU streams output rows while all 16 waves
 * reuse up to five F16 activation rows from LDS.  Every token keeps its own
 * wave reduction; the only numerical change versus hipBLAS is the dot-product
 * reduction order.  The caller restricts this research kernel to gfx1151 and
 * N=1..5. */
#if defined(__HIP_PLATFORM_AMD__)
union alignas(16) ds4_f16_pack8h {
    __half h[8];
    float4 raw;
    float packed2[4];
};

static __device__ __forceinline__ void ds4_f16_dot2_acc(
        float &acc, float a, float b) {
    asm("v_dot2_f32_f16 %0, %1, %2, %0"
        : "+v"(acc) : "v"(a), "v"(b));
}

template <uint32_t TOKENS, uint32_t YTILE = 4u, uint32_t UNROLL = 2u>
__launch_bounds__(512, 1)
__global__ static void matmul_f16_indexer_q_wvsplit_kernel(
        float *out,
        const __half *weights,
        const __half *x,
        uint32_t kdim,
        uint32_t mdim) {
    __shared__ __align__(16) __half sx[5u * 1536u];
    const uint32_t lane = threadIdx.x;
    const uint32_t wave = threadIdx.y;
    const uint32_t linear = wave * 32u + lane;

    const uint32_t x_elems = TOKENS * kdim;
    for (uint32_t i = linear * 8u; i < x_elems; i += 512u * 8u) {
        reinterpret_cast<float4 *>(sx)[i / 8u] =
            reinterpret_cast<const float4 *>(x)[i / 8u];
    }
    __syncthreads();

    uint32_t m0 = (blockIdx.x * 16u + wave) * YTILE;
    const uint32_t m_stride = gridDim.x * 16u * YTILE;
    while (m0 < mdim) {
        float sum[TOKENS][YTILE] = {};
        for (uint32_t k0 = 0; k0 < kdim; k0 += 32u * 8u * UNROLL) {
            ds4_f16_pack8h av[TOKENS][UNROLL];
            ds4_f16_pack8h wv[YTILE][UNROLL];
#pragma unroll
            for (uint32_t u = 0; u < UNROLL; ++u) {
                const uint32_t k = k0 + u * 32u * 8u + lane * 8u;
                if (k < kdim) {
#pragma unroll
                    for (uint32_t t = 0; t < TOKENS; ++t) {
                        av[t][u].raw = reinterpret_cast<const float4 *>(
                            sx + (uint64_t)t * kdim + k)[0];
                    }
#pragma unroll
                    for (uint32_t y = 0; y < YTILE; ++y) {
                        const uint32_t m = min(m0 + y, mdim - 1u);
                        wv[y][u].raw = reinterpret_cast<const float4 *>(
                            weights + (uint64_t)m * kdim + k)[0];
                    }
                } else {
#pragma unroll
                    for (uint32_t t = 0; t < TOKENS; ++t) av[t][u].raw = {};
#pragma unroll
                    for (uint32_t y = 0; y < YTILE; ++y) wv[y][u].raw = {};
                }
            }
#pragma unroll
            for (uint32_t u = 0; u < UNROLL; ++u) {
#pragma unroll
                for (uint32_t t = 0; t < TOKENS; ++t) {
#pragma unroll
                    for (uint32_t y = 0; y < YTILE; ++y) {
#pragma unroll
                        for (uint32_t p = 0; p < 4u; ++p) {
                            ds4_f16_dot2_acc(sum[t][y], av[t][u].packed2[p],
                                             wv[y][u].packed2[p]);
                        }
                    }
                }
            }
        }
#pragma unroll
        for (uint32_t mask = 16u; mask != 0u; mask >>= 1u) {
#pragma unroll
            for (uint32_t t = 0; t < TOKENS; ++t) {
#pragma unroll
                for (uint32_t y = 0; y < YTILE; ++y) {
                    sum[t][y] += __shfl_xor(sum[t][y], mask, 32);
                }
            }
        }
        if (lane == 0u) {
#pragma unroll
            for (uint32_t t = 0; t < TOKENS; ++t) {
#pragma unroll
                for (uint32_t y = 0; y < YTILE; ++y) {
                    const uint32_t m = m0 + y;
                    if (m < mdim) out[(uint64_t)t * mdim + m] = sum[t][y];
                }
            }
        }
        m0 += m_stride;
    }
}
#endif

__global__ static void matmul_f16_pair_ordered_chunks_kernel(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const float *x,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim) {
    uint64_t row = (uint64_t)blockIdx.x;
    if (row >= out0_dim && row >= out1_dim) return;

    __shared__ float partial0[32];
    __shared__ float partial1[32];
    const uint32_t tid = threadIdx.x;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr0 = row < out0_dim ? w0 + row * in_dim : w0;
    const __half *wr1 = row < out1_dim ? w1 + row * in_dim : w1;
    for (uint64_t i = k0; i < k1; i++) {
        const float xv = x[i];
        if (row < out0_dim) sum0 += __half2float(wr0[i]) * xv;
        if (row < out1_dim) sum1 += __half2float(wr1[i]) * xv;
    }
    partial0[tid] = sum0;
    partial1[tid] = sum1;
    __syncthreads();
    if (tid == 0) {
        float total0 = 0.0f;
        float total1 = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) {
            total0 += partial0[i];
            total1 += partial1[i];
        }
        if (row < out0_dim) out0[row] = total0;
        if (row < out1_dim) out1[row] = total1;
    }
}

__global__ static void matmul_f32_kernel(
        float *out,
        const float *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const float *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += wr[i] * xr[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void repeat_hc_kernel(float *out, const float *row, uint32_t n_embd, uint32_t n_hc) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_hc;
    if (i >= n) return;
    out[i] = row[i % n_embd];
}

__global__ static void repeat_hc_rows_kernel(float *out, const float *rows, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (i >= n) return;

    uint64_t hc_row = (uint64_t)n_hc * n_embd;
    uint64_t tok = i / hc_row;
    uint64_t embd = i % n_embd;
    out[i] = rows[tok * n_embd + embd];
}

__global__ static void pack_slot_rows_f32_kernel(float *out, const float *slots, uint32_t n_rows, uint32_t width, uint32_t n_slots, uint32_t slot_cap) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_rows * n_slots * width;
    if (i >= n) return;

    uint64_t col = i % width;
    uint64_t slot = (i / width) % n_slots;
    uint64_t row = i / ((uint64_t)n_slots * width);
    out[i] = slots[((slot * slot_cap) + row) * width + col];
}

__global__ static void f32_to_f16_kernel(__half *out, const float *x, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(x[i]);
}

__device__ static uint16_t f32_to_bf16_bits_rne(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) {
        /* Preserve sign/payload high bits while forcing a quiet BF16 NaN.
         * This also keeps a NaN whose payload lives only in the discarded
         * F32 mantissa bits from becoming infinity. */
        return (uint16_t)((bits >> 16u) | 0x0040u);
    }
    const uint32_t tie_to_even = (bits >> 16u) & 1u;
    return (uint16_t)((bits + 0x00007fffu + tie_to_even) >> 16u);
}

__device__ static float f32_round_bf16_rne(float value) {
    return __uint_as_float((uint32_t)f32_to_bf16_bits_rne(value) << 16u);
}

__global__ static void round_bf16_inplace_kernel(
        float *values, uint64_t count, float post_scale) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    values[i] = f32_round_bf16_rne(values[i]) * post_scale;
}

__device__ static float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        v += __shfl_down(v, offset, 32);
#else
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset, 32);
#endif
    }
    return v;
}

/* Error-compensated wave reduction for split-K experiments.  The production
 * path intentionally keeps warp_sum_f32's established arithmetic.  This
 * helper preserves a low component through the fixed shuffle tree, allowing
 * us to measure whether finalization error (rather than the per-lane dot
 * accumulation) is responsible for K-slice drift. */
__device__ static float warp_sum_f32_compensated(float v) {
    float c = 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        const float other = __shfl_down(v, offset, 32);
#else
        const float other = __shfl_down_sync(FULL_WARP_MASK, v, offset, 32);
#endif
        const float t = v + other;
        if (fabsf(v) >= fabsf(other)) c += (v - t) + other;
        else c += (other - t) + v;
        v = t;
    }
    return v + c;
}

__device__ static float warp_max_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        v = fmaxf(v, __shfl_down(v, offset, 32));
#else
        v = fmaxf(v, __shfl_down_sync(FULL_WARP_MASK, v, offset, 32));
#endif
    }
    return v;
}

__device__ static uint16_t f32_to_f16_bits_hip_round(float f) {
    union { float f; uint32_t u; } v;
    v.f = f;
    uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

__device__ static float f16_bits_to_f32(uint16_t bits) {
    return __half2float(__ushort_as_half((unsigned short)bits));
}

__device__ static float dot4_f32(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
