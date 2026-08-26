// DS4 ROCm attention kernels (prefill/decode, raw/mixed KV).
//
// Included from ds4_cuda.cu in the same translation unit to keep launch/API
// glue unchanged while kernel implementations are split into modules.

#define DS4_ROCM_ATTENTION_PREFILL_MIXED_SCORE_CAP 2048u
#define DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP 1024u
#define DS4_ROCM_ATTENTION_INDEXED_SCORE_CAP \
    (256u + DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP)

__global__ static void attention_prefill_raw_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        uint32_t n_q,
        uint32_t q_row0,
        uint32_t window,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t_local = blockIdx.x;
    uint32_t t_abs = q_row0 + t_local;
    uint32_t h = blockIdx.y;
    if (t_local >= n_q || h >= n_head) return;
    uint32_t raw_count = (window != 0u && t_abs + 1u > window) ? window : t_abs + 1u;
    uint32_t raw_start = t_abs + 1u - raw_count;
    const float *qh = q + ((uint64_t)t_local * n_head + h) * head_dim;
    __shared__ float scores[256];
    __shared__ float partial[128];
    __shared__ float max_s;
    __shared__ float denom;
    float scale = rsqrtf((float)head_dim);
    float local_max = sinks[h];
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        const float *kv = raw_kv + (uint64_t)(raw_start + r) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
        scores[r] = dot * scale;
        local_max = fmaxf(local_max, scores[r]);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    if (threadIdx.x == 0) {
        float den = expf(sinks[h] - max_s);
        for (uint32_t r = 0; r < raw_count; r++) {
            scores[r] = expf(scores[r] - max_s);
            den += scores[r];
        }
        denom = den;
    }
    __syncthreads();
    float *oh = heads + ((uint64_t)t_local * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            acc += raw_kv[(uint64_t)(raw_start + r) * head_dim + d] * scores[r];
        }
        oh[d] = acc / denom;
    }
}

__global__ static void attention_prefill_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_q,
        uint32_t q_row0,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t_local = blockIdx.x;
    uint32_t t_abs = q_row0 + t_local;
    uint32_t h = blockIdx.y;
    if (t_local >= n_q || h >= n_head) return;
    const float *qh = q + ((uint64_t)t_local * n_head + h) * head_dim;
    uint32_t raw_start = (window != 0 && t_abs + 1u > window) ? t_abs + 1u - window : 0u;
    uint32_t raw_count = t_abs + 1u - raw_start;
    uint32_t visible_comp = (t_abs + 1u) / ratio;
    if (visible_comp > n_comp) visible_comp = n_comp;
    __shared__ float scores[DS4_ROCM_ATTENTION_PREFILL_MIXED_SCORE_CAP];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float scale = rsqrtf((float)head_dim);
    float local_max = sinks[h];
    uint32_t n_score = raw_count + visible_comp;
    if (n_score > DS4_ROCM_ATTENTION_PREFILL_MIXED_SCORE_CAP) return;

    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        const float *kvrow = raw_kv + (uint64_t)(raw_start + r) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
        scores[r] = dot * scale;
        local_max = fmaxf(local_max, scores[r]);
    }
    for (uint32_t c = threadIdx.x; c < visible_comp; c += blockDim.x) {
        float add = use_comp_mask ? comp_mask[(uint64_t)t_abs * n_comp + c] : 0.0f;
        float s = -INFINITY;
        if (add > -1.0e20f) {
            const float *kvrow = comp_kv + (uint64_t)c * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            s = dot * scale + add;
        }
        scores[raw_count + c] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t_local * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)(raw_start + r) * head_dim + d] * scores[r];
        for (uint32_t c = 0; c < visible_comp; c++) acc += comp_kv[(uint64_t)c * head_dim + d] * scores[raw_count + c];
        oh[d] = acc / denom;
    }
}

__global__ static void attention_prefill_raw_softmax_kernel(
        float *scores,
        const float *sinks,
        uint32_t n_q,
        uint32_t q_row0,
        uint32_t window,
        uint32_t n_keys) {
    uint32_t t_local = blockIdx.x;
    uint32_t t_abs = q_row0 + t_local;
    uint32_t h = blockIdx.y;
    if (t_local >= n_q) return;
    float *row = scores + ((uint64_t)h * n_q + t_local) * n_keys;
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        bool valid = k <= t_abs && (window == 0 || t_abs - k < window);
        float s = valid ? row[k] : -INFINITY;
        row[k] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float p = isfinite(row[k]) ? expf(row[k] - max_s) : 0.0f;
        row[k] = p;
        den_local += p;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) row[k] /= denom;
}

__global__ static void attention_prefill_mixed_softmax_kernel(
        float *scores,
        const float *sinks,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_q,
        uint32_t q_row0,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_keys) {
    uint32_t t_local = blockIdx.x;
    uint32_t t_abs = q_row0 + t_local;
    uint32_t h = blockIdx.y;
    if (t_local >= n_q || ratio == 0) return;
    float *row = scores + ((uint64_t)h * n_q + t_local) * n_keys;
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    const uint32_t visible_comp = (t_abs + 1u) / ratio;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float s = -INFINITY;
        if (k < n_tokens) {
            if (k <= t_abs && (window == 0 || t_abs - k < window)) s = row[k];
        } else {
            uint32_t c = k - n_tokens;
            if (c < n_comp && c < visible_comp) {
                float add = use_comp_mask ? comp_mask[(uint64_t)t_abs * n_comp + c] : 0.0f;
                if (add > -1.0e20f) s = row[k] + add;
            }
        }
        row[k] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float p = isfinite(row[k]) ? expf(row[k] - max_s) : 0.0f;
        row[k] = p;
        den_local += p;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) row[k] /= denom;
}

__global__ static void attention_prefill_mixed_softmax_tile_kernel(
        float *scores,
        const float *sinks,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t raw_tokens,
        uint32_t q_row0,
        uint32_t tile_start,
        uint32_t tile_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_keys) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= tile_tokens || ratio == 0) return;
    const uint32_t global_t = q_row0 + tile_start + t;
    float *row = scores + ((uint64_t)h * tile_tokens + t) * n_keys;
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    const uint32_t visible_comp = (global_t + 1u) / ratio;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float s = -INFINITY;
        if (k < raw_tokens) {
            if (k <= global_t && (window == 0 || global_t - k < window)) s = row[k];
        } else {
            uint32_t c = k - raw_tokens;
            if (c < n_comp && c < visible_comp) {
                float add = use_comp_mask ? comp_mask[(uint64_t)global_t * n_comp + c] : 0.0f;
                if (add > -1.0e20f) s = row[k] + add;
            }
        }
        row[k] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float p = isfinite(row[k]) ? expf(row[k] - max_s) : 0.0f;
        row[k] = p;
        den_local += p;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) row[k] /= denom;
}

__global__ static void attention_prefill_pack_mixed_kv_kernel(
        float *dst,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)(n_tokens + n_comp) * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint32_t r = gid / head_dim;
    dst[gid] = r < n_tokens ? raw_kv[(uint64_t)r * head_dim + d]
                             : comp_kv[(uint64_t)(r - n_tokens) * head_dim + d];
}

__global__ static void attention_prefill_unpack_heads_kernel(
        float *heads,
        const float *tmp,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint64_t q = gid / head_dim;
    uint32_t h = q % n_head;
    uint32_t t = q / n_head;
    heads[gid] = tmp[((uint64_t)h * n_tokens + t) * head_dim + d];
}

__global__ static void attention_pack_group_heads_f16_kernel(
        __half *dst,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t group_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_groups * n_tokens * group_dim;
    if (gid >= n) return;
    uint32_t d = gid % group_dim;
    uint64_t q = gid / group_dim;
    uint32_t t = q % n_tokens;
    uint32_t g = q / n_tokens;
    dst[gid] = __float2half(heads[((uint64_t)t * n_groups + g) * group_dim + d]);
}

__global__ static void attention_unpack_group_low_kernel(
        float *low,
        const float *tmp,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t rank) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_groups * n_tokens * rank;
    if (gid >= n) return;
    uint32_t r = gid % rank;
    uint64_t q = gid / rank;
    uint32_t t = q % n_tokens;
    uint32_t g = q / n_tokens;
    uint32_t low_dim = n_groups * rank;
    low[(uint64_t)t * low_dim + (uint64_t)g * rank + r] = tmp[gid];
}

__device__ static float attention_warp_sum_oldhip_w32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        v += __shfl_down(v, offset, 32);
#else
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset, 32);
#endif
    }
    return v;
}

__device__ static float attention_warp_max_oldhip_w32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        v = fmaxf(v, __shfl_down(v, offset, 32));
#else
        v = fmaxf(v, __shfl_down_sync(FULL_WARP_MASK, v, offset, 32));
#endif
    }
    return v;
}

__device__ static float attention_block_sum_oldhip_w32(float v) {
    __shared__ float sh[32];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wid = tid >> 5u;
    const uint32_t nwarp = (blockDim.x + 31u) >> 5u;
    v = attention_warp_sum_oldhip_w32(v);
    if (lane == 0u) sh[wid] = v;
    __syncthreads();
    v = (tid < nwarp) ? sh[lane] : 0.0f;
    if (wid == 0u) v = attention_warp_sum_oldhip_w32(v);
    if (tid == 0u) sh[0] = v;
    __syncthreads();
    return sh[0];
}

__device__ static float attention_block_max_oldhip_w32(float v) {
    __shared__ float sh[32];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wid = tid >> 5u;
    const uint32_t nwarp = (blockDim.x + 31u) >> 5u;
    v = attention_warp_max_oldhip_w32(v);
    if (lane == 0u) sh[wid] = v;
    __syncthreads();
    v = (tid < nwarp) ? sh[lane] : -3.4e38f;
    if (wid == 0u) v = attention_warp_max_oldhip_w32(v);
    if (tid == 0u) sh[0] = v;
    __syncthreads();
    return sh[0];
}

__device__ __forceinline__ static float attention_dot_f32_vec4_oldhip(const float *a, const float *b, uint32_t n) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    const uint32_t n4 = n >> 2u;
    const float4 *a4 = (const float4 *)a;
    const float4 *b4 = (const float4 *)b;
    for (uint32_t i = 0; i < n4; i++) {
        const float4 av = a4[i];
        const float4 bv = b4[i];
        s0 += av.x * bv.x;
        s1 += av.y * bv.y;
        s2 += av.z * bv.z;
        s3 += av.w * bv.w;
    }
    float s = (s0 + s1) + (s2 + s3);
    for (uint32_t i = n4 << 2u; i < n; i++) s += a[i] * b[i];
    return s;
}

__global__ static void attention_decode_mixed_one_fast_oldhip_kernel(
        float *heads,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        const float *sinks,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t use_mask,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t use_vec4) {
    const uint32_t h = (uint32_t)blockIdx.x;
    if (h >= n_head) return;
    extern __shared__ float scores[];
    const uint32_t tid = threadIdx.x;
    const uint32_t n_rows = n_raw + n_comp;
    const float *qh = q + (uint64_t)h * head_dim;
    const float scale = rsqrtf((float)head_dim);

    float local_max = sinks[h];
    for (uint32_t r = tid; r < n_raw; r += blockDim.x) {
        const uint32_t row = raw_cap ? ((raw_start + r) % raw_cap) : r;
        const float *kv = raw_kv + (uint64_t)row * head_dim;
        float s = use_vec4 ? attention_dot_f32_vec4_oldhip(qh, kv, head_dim) : 0.0f;
        if (!use_vec4) {
            for (uint32_t i = 0; i < head_dim; i++) s += qh[i] * kv[i];
        }
        s *= scale;
        scores[r] = s;
        local_max = fmaxf(local_max, s);
    }
    for (uint32_t c = tid; c < n_comp; c += blockDim.x) {
        float s = -3.4e38f;
        if (!(use_mask && comp_mask && comp_mask[c] <= -5.0e29f)) {
            const float *kv = comp_kv + (uint64_t)c * head_dim;
            float dot = use_vec4 ? attention_dot_f32_vec4_oldhip(qh, kv, head_dim) : 0.0f;
            if (!use_vec4) {
                for (uint32_t i = 0; i < head_dim; i++) dot += qh[i] * kv[i];
            }
            s = dot * scale;
            if (use_mask && comp_mask) s += comp_mask[c];
        }
        scores[n_raw + c] = s;
        local_max = fmaxf(local_max, s);
    }
    const float max_score = attention_block_max_oldhip_w32(local_max);

    float local_sum = 0.0f;
    for (uint32_t r = tid; r < n_rows; r += blockDim.x) {
        const float w = expf(scores[r] - max_score);
        scores[r] = w;
        local_sum += w;
    }
    if (tid == 0u) local_sum += expf(sinks[h] - max_score);
    const float denom = attention_block_sum_oldhip_w32(local_sum);
    const float inv_denom = 1.0f / denom;

    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < n_raw; r++) {
            const uint32_t row = raw_cap ? ((raw_start + r) % raw_cap) : r;
            acc += scores[r] * raw_kv[(uint64_t)row * head_dim + d];
        }
        for (uint32_t c = 0; c < n_comp; c++) {
            acc += scores[n_raw + c] * comp_kv[(uint64_t)c * head_dim + d];
        }
        heads[(uint64_t)h * head_dim + d] = acc * inv_denom;
    }
}

__global__ static void attention_decode_indexed_mixed_one_fast_oldhip_kernel(
        float *heads,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        const float *sinks,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t pos0,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t use_vec4) {
    const uint32_t h = (uint32_t)blockIdx.x;
    if (h >= n_head) return;
    extern __shared__ float scores[];
    __shared__ uint32_t comp_rows[DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP];
    __shared__ uint32_t comp_count_s;
    const uint32_t tid = threadIdx.x;
    const float *qh = q + (uint64_t)h * head_dim;
    const float scale = rsqrtf((float)head_dim);

    uint32_t visible_comp = n_comp;
    if (ratio != 0u) {
        visible_comp = (pos0 + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }
    if (tid == 0u) {
        comp_count_s = 0;
        for (uint32_t i = 0;
             i < top_k && comp_count_s < DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP;
             i++) {
            const int32_t ci = topk[i];
            if (ci < 0) continue;
            const uint32_t c = (uint32_t)ci;
            if (c < n_comp && c < visible_comp) comp_rows[comp_count_s++] = c;
        }
    }
    __syncthreads();
    const uint32_t comp_count = comp_count_s;
    const uint32_t n_rows = n_raw + comp_count;

    float local_max = sinks[h];
    for (uint32_t r = tid; r < n_raw; r += blockDim.x) {
        const uint32_t row = raw_cap ? ((raw_start + r) % raw_cap) : r;
        const float *kv = raw_kv + (uint64_t)row * head_dim;
        float s = use_vec4 ? attention_dot_f32_vec4_oldhip(qh, kv, head_dim) : 0.0f;
        if (!use_vec4) {
            for (uint32_t i = 0; i < head_dim; i++) s += qh[i] * kv[i];
        }
        s *= scale;
        scores[r] = s;
        local_max = fmaxf(local_max, s);
    }
    for (uint32_t c = tid; c < comp_count; c += blockDim.x) {
        const uint32_t row = comp_rows[c];
        const float *kv = comp_kv + (uint64_t)row * head_dim;
        float dot = use_vec4 ? attention_dot_f32_vec4_oldhip(qh, kv, head_dim) : 0.0f;
        if (!use_vec4) {
            for (uint32_t i = 0; i < head_dim; i++) dot += qh[i] * kv[i];
        }
        const float s = dot * scale;
        scores[n_raw + c] = s;
        local_max = fmaxf(local_max, s);
    }
    const float max_score = attention_block_max_oldhip_w32(local_max);

    float local_sum = 0.0f;
    for (uint32_t r = tid; r < n_rows; r += blockDim.x) {
        const float w = expf(scores[r] - max_score);
        scores[r] = w;
        local_sum += w;
    }
    if (tid == 0u) local_sum += expf(sinks[h] - max_score);
    const float denom = attention_block_sum_oldhip_w32(local_sum);
    const float inv_denom = 1.0f / denom;

    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < n_raw; r++) {
            const uint32_t row = raw_cap ? ((raw_start + r) % raw_cap) : r;
            acc += scores[r] * raw_kv[(uint64_t)row * head_dim + d];
        }
        for (uint32_t c = 0; c < comp_count; c++) {
            const uint32_t row = comp_rows[c];
            acc += scores[n_raw + c] * comp_kv[(uint64_t)row * head_dim + d];
        }
        heads[(uint64_t)h * head_dim + d] = acc * inv_denom;
    }
}

/* Two independent 256-thread copies of the shipped old-HIP reduction tree in
 * one workgroup.  blockDim.x must be exactly 512.  The hard-coded eight-wave
 * second stage is what makes each subgroup bit-identical to a 256-thread
 * attention_block_{max,sum}_oldhip_w32 call; deriving nwarp from blockDim.x
 * would mix the two heads. */
__device__ __forceinline__ static float attention_subgroup_max_oldhip_w32(
        float v, float *sh, uint32_t sub_tid) {
    const uint32_t lane = sub_tid & 31u;
    const uint32_t wid = sub_tid >> 5u;
    v = attention_warp_max_oldhip_w32(v);
    if (lane == 0u) sh[wid] = v;
    __syncthreads();
    v = (sub_tid < 8u) ? sh[lane] : -3.4e38f;
    if (wid == 0u) v = attention_warp_max_oldhip_w32(v);
    if (sub_tid == 0u) sh[0] = v;
    __syncthreads();
    return sh[0];
}

__device__ __forceinline__ static float attention_subgroup_sum_oldhip_w32(
        float v, float *sh, uint32_t sub_tid) {
    const uint32_t lane = sub_tid & 31u;
    const uint32_t wid = sub_tid >> 5u;
    v = attention_warp_sum_oldhip_w32(v);
    if (lane == 0u) sh[wid] = v;
    __syncthreads();
    v = (sub_tid < 8u) ? sh[lane] : 0.0f;
    if (wid == 0u) v = attention_warp_sum_oldhip_w32(v);
    if (sub_tid == 0u) sh[0] = v;
    __syncthreads();
    return sh[0];
}

__device__ __forceinline__ static uint4 attention_exact_state_for_token(
        uint32_t t, uint4 s0, uint4 s1, uint4 s2, uint4 s3, uint4 s4) {
    switch (t) {
        case 0u: return s0;
        case 1u: return s1;
        case 2u: return s2;
        case 3u: return s3;
        default: return s4;
    }
}

/* Exact H=2 indexed-attention batch prototype. Each token remains a separate
 * grid row; each 256-thread subgroup owns one head and retains the shipped
 * row striding, serial float4 dot, softmax reduction, and raw-then-comp output
 * recurrence. The raw top-k order is compacted once per CTA and shared by the
 * two heads; it must never pass through the sorted multi-token path. */
__global__ static void attention_decode_indexed_exact_head2_batch_kernel(
        float *heads,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        const float *sinks,
        uint32_t n_tokens,
        uint32_t raw_cap,
        uint32_t top_k,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t score_stride,
        uint4 state0,
        uint4 state1,
        uint4 state2,
        uint4 state3,
        uint4 state4) {
    const uint32_t t = blockIdx.x;
    const uint32_t subgroup = threadIdx.x >> 8u;
    const uint32_t sub_tid = threadIdx.x & 255u;
    const uint32_t h = blockIdx.y * 2u + subgroup;
    if (t >= n_tokens || blockDim.x != 512u || head_dim != 512u) return;

    extern __shared__ int32_t smem[];
    uint32_t *comp_rows = (uint32_t *)smem;
    float *scores = (float *)(comp_rows + top_k);
    float *reduce_max = scores + 2u * score_stride;
    float *reduce_sum = reduce_max + 16u;
    float *head_scores = scores + subgroup * score_stride;
    float *head_reduce_max = reduce_max + subgroup * 8u;
    float *head_reduce_sum = reduce_sum + subgroup * 8u;

    const uint4 state = attention_exact_state_for_token(
        t, state0, state1, state2, state3, state4);
    const uint32_t n_raw = state.x;
    const uint32_t raw_start = state.y;
    const uint32_t n_comp = state.z;
    const uint32_t visible_comp = state.w;

    __shared__ uint32_t comp_count_s;
    if (threadIdx.x == 0u) {
        comp_count_s = 0u;
        for (uint32_t i = 0u;
             i < top_k && comp_count_s < DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP;
             i++) {
            const int32_t ci = topk[(uint64_t)t * top_k + i];
            if (ci < 0) continue;
            const uint32_t c = (uint32_t)ci;
            if (c < n_comp && c < visible_comp) {
                comp_rows[comp_count_s++] = c;
            }
        }
    }
    __syncthreads();
    const uint32_t comp_count = comp_count_s;
    const uint32_t n_rows = n_raw + comp_count;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    const float scale = rsqrtf((float)head_dim);

    float local_max = sinks[h];
    for (uint32_t r = sub_tid; r < n_raw; r += 256u) {
        const uint32_t row = raw_cap ? ((raw_start + r) % raw_cap) : r;
        const float *kv = raw_kv + (uint64_t)row * head_dim;
        float s = attention_dot_f32_vec4_oldhip(qh, kv, head_dim);
        s *= scale;
        head_scores[r] = s;
        local_max = fmaxf(local_max, s);
    }
    for (uint32_t c = sub_tid; c < comp_count; c += 256u) {
        const uint32_t row = comp_rows[c];
        const float *kv = comp_kv + (uint64_t)row * head_dim;
        const float s = attention_dot_f32_vec4_oldhip(qh, kv, head_dim) * scale;
        head_scores[n_raw + c] = s;
        local_max = fmaxf(local_max, s);
    }
    const float max_score = attention_subgroup_max_oldhip_w32(
        local_max, head_reduce_max, sub_tid);

    float local_sum = 0.0f;
    for (uint32_t r = sub_tid; r < n_rows; r += 256u) {
        const float w = expf(head_scores[r] - max_score);
        head_scores[r] = w;
        local_sum += w;
    }
    if (sub_tid == 0u) local_sum += expf(sinks[h] - max_score);
    const float denom = attention_subgroup_sum_oldhip_w32(
        local_sum, head_reduce_sum, sub_tid);
    const float inv_denom = 1.0f / denom;

    for (uint32_t d = sub_tid; d < head_dim; d += 256u) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < n_raw; r++) {
            const uint32_t row = raw_cap ? ((raw_start + r) % raw_cap) : r;
            acc += head_scores[r] * raw_kv[(uint64_t)row * head_dim + d];
        }
        for (uint32_t c = 0; c < comp_count; c++) {
            const uint32_t row = comp_rows[c];
            acc += head_scores[n_raw + c] *
                   comp_kv[(uint64_t)row * head_dim + d];
        }
        heads[((uint64_t)t * n_head + h) * head_dim + d] = acc * inv_denom;
    }
}

/* Regrid independent QK dots as 32 logical rows by eight head-contiguous
 * lanes. Each lane calls the shipped serial float4 helper verbatim; only the
 * mapping of independent (row, head) scores changes. Compressed scores remain
 * indexed by their original top-k slot so softmax can compact them later
 * without an atomic planning pass. */
__global__ static void attention_decode_indexed_exact_head8_qk_kernel(
        float *scores,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t raw_cap,
        uint32_t top_k,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t raw_score_stride,
        uint4 state0,
        uint4 state1,
        uint4 state2,
        uint4 state3,
        uint4 state4) {
    const uint32_t t = blockIdx.z;
    const uint32_t local_row = threadIdx.x >> 3u;
    const uint32_t local_head = threadIdx.x & 7u;
    const uint32_t logical_row = blockIdx.x * 32u + local_row;
    const uint32_t h = blockIdx.y * 8u + local_head;
    if (t >= n_tokens || h >= n_head || blockDim.x != 256u ||
        head_dim != 512u || logical_row >= raw_score_stride) return;

    const uint4 state = attention_exact_state_for_token(
        t, state0, state1, state2, state3, state4);
    const uint32_t raw_stride = raw_score_stride - top_k;
    const float *kv = NULL;
    if (logical_row < raw_stride) {
        if (logical_row >= state.x) return;
        const uint32_t row = (state.y + logical_row) % raw_cap;
        kv = raw_kv + (uint64_t)row * head_dim;
    } else {
        const uint32_t slot = logical_row - raw_stride;
        const int32_t ci = topk[(uint64_t)t * top_k + slot];
        if (ci < 0) return;
        const uint32_t row = (uint32_t)ci;
        if (row >= state.z || row >= state.w) return;
        kv = comp_kv + (uint64_t)row * head_dim;
    }
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    const float dot = attention_dot_f32_vec4_oldhip(qh, kv, head_dim);
    scores[((uint64_t)t * raw_score_stride + logical_row) * n_head + h] =
        dot * rsqrtf((float)head_dim);
}

/* Preserve the shipped head-pair max/sum partition exactly, replacing only
 * QK dot calls with score loads. The separate raw/comp max loops and single
 * concatenated sum loop are numerically significant. */
__global__ static void attention_decode_indexed_exact_head2_softmax_kernel(
        float *weights,
        float *inv_denoms,
        uint32_t *token_comp_rows,
        uint32_t *token_comp_counts,
        const float *scores,
        const int32_t *topk,
        const float *sinks,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t n_head,
        uint32_t raw_score_stride,
        uint32_t weight_stride,
        uint4 state0,
        uint4 state1,
        uint4 state2,
        uint4 state3,
        uint4 state4) {
    const uint32_t t = blockIdx.x;
    const uint32_t subgroup = threadIdx.x >> 8u;
    const uint32_t sub_tid = threadIdx.x & 255u;
    const uint32_t h = blockIdx.y * 2u + subgroup;
    if (t >= n_tokens || h >= n_head || blockDim.x != 512u) return;

    extern __shared__ unsigned char smem_bytes[];
    uint32_t *comp_rows = (uint32_t *)smem_bytes;
    uint16_t *comp_slots = (uint16_t *)(comp_rows + top_k);
    uintptr_t reduce_addr = (uintptr_t)(comp_slots + top_k);
    reduce_addr = (reduce_addr + 3u) & ~(uintptr_t)3u;
    float *reduce_max = (float *)reduce_addr;
    float *reduce_sum = reduce_max + 16u;
    __shared__ uint32_t comp_count_s;

    const uint4 state = attention_exact_state_for_token(
        t, state0, state1, state2, state3, state4);
    const uint32_t n_raw = state.x;
    const uint32_t raw_stride = raw_score_stride - top_k;
    if (threadIdx.x == 0u) {
        comp_count_s = 0u;
        for (uint32_t i = 0u;
             i < top_k && comp_count_s < DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP;
             i++) {
            const int32_t ci = topk[(uint64_t)t * top_k + i];
            if (ci < 0) continue;
            const uint32_t row = (uint32_t)ci;
            if (row < state.z && row < state.w) {
                comp_rows[comp_count_s] = row;
                comp_slots[comp_count_s] = (uint16_t)i;
                comp_count_s++;
            }
        }
        token_comp_counts[t] = comp_count_s;
        for (uint32_t c = 0u; c < comp_count_s; c++) {
            token_comp_rows[(uint64_t)t * top_k + c] = comp_rows[c];
        }
    }
    __syncthreads();
    const uint32_t comp_count = comp_count_s;
    const float *token_scores =
        scores + (uint64_t)t * raw_score_stride * n_head;

    float local_max = sinks[h];
    for (uint32_t r = sub_tid; r < n_raw; r += 256u) {
        local_max = fmaxf(local_max,
            token_scores[(uint64_t)r * n_head + h]);
    }
    for (uint32_t c = sub_tid; c < comp_count; c += 256u) {
        const uint32_t slot = comp_slots[c];
        local_max = fmaxf(local_max, token_scores[
            (uint64_t)(raw_stride + slot) * n_head + h]);
    }
    float *head_reduce_max = reduce_max + subgroup * 8u;
    float *head_reduce_sum = reduce_sum + subgroup * 8u;
    const float max_score = attention_subgroup_max_oldhip_w32(
        local_max, head_reduce_max, sub_tid);

    const uint32_t n_rows = n_raw + comp_count;
    float local_sum = 0.0f;
    for (uint32_t r = sub_tid; r < n_rows; r += 256u) {
        const float s = r < n_raw
            ? token_scores[(uint64_t)r * n_head + h]
            : token_scores[(uint64_t)(raw_stride + comp_slots[r - n_raw]) *
                           n_head + h];
        const float w = expf(s - max_score);
        weights[((uint64_t)t * weight_stride + r) * n_head + h] = w;
        local_sum += w;
    }
    if (sub_tid == 0u) local_sum += expf(sinks[h] - max_score);
    const float denom = attention_subgroup_sum_oldhip_w32(
        local_sum, head_reduce_sum, sub_tid);
    if (sub_tid == 0u) {
        inv_denoms[(uint64_t)t * n_head + h] = 1.0f / denom;
    }
}

/* A token-grid, head-contiguous PV layout. Each lane owns four consecutive
 * dimensions and retains one ordered raw-then-compressed recurrence per
 * output. Row ids are staged once per CTA. */
__global__ static void attention_decode_indexed_exact_head16_pv_vec4_kernel(
        float *heads,
        const float *weights,
        const float *inv_denoms,
        const uint32_t *token_comp_rows,
        const uint32_t *token_comp_counts,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t raw_cap,
        uint32_t top_k,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t weight_stride,
        uint4 state0,
        uint4 state1,
        uint4 state2,
        uint4 state3,
        uint4 state4) {
    const uint32_t t = blockIdx.z;
    if (t >= n_tokens || blockDim.x != 256u || head_dim != 512u) return;
    const uint4 state = attention_exact_state_for_token(
        t, state0, state1, state2, state3, state4);
    const uint32_t n_raw = state.x;
    const uint32_t comp_count = token_comp_counts[t];
    extern __shared__ uint32_t rows[];
    for (uint32_t r = threadIdx.x; r < n_raw; r += blockDim.x) {
        rows[r] = (state.y + r) % raw_cap;
    }
    for (uint32_t c = threadIdx.x; c < comp_count; c += blockDim.x) {
        rows[n_raw + c] = token_comp_rows[(uint64_t)t * top_k + c];
    }
    __syncthreads();

    const uint32_t h = blockIdx.x * 16u + (threadIdx.x & 15u);
    const uint32_t d =
        (blockIdx.y * 16u + (threadIdx.x >> 4u)) * 4u;
    if (h >= n_head || d + 3u >= head_dim) return;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (uint32_t r = 0u; r < n_raw; r++) {
        const float w =
            weights[((uint64_t)t * weight_stride + r) * n_head + h];
        const float4 v = *(const float4 *)(raw_kv +
            (uint64_t)rows[r] * head_dim + d);
        acc0 = fmaf(w, v.x, acc0);
        acc1 = fmaf(w, v.y, acc1);
        acc2 = fmaf(w, v.z, acc2);
        acc3 = fmaf(w, v.w, acc3);
    }
    for (uint32_t c = 0u; c < comp_count; c++) {
        const float w = weights[
            ((uint64_t)t * weight_stride + n_raw + c) * n_head + h];
        const float4 v = *(const float4 *)(comp_kv +
            (uint64_t)rows[n_raw + c] * head_dim + d);
        acc0 = fmaf(w, v.x, acc0);
        acc1 = fmaf(w, v.y, acc1);
        acc2 = fmaf(w, v.z, acc2);
        acc3 = fmaf(w, v.w, acc3);
    }
    const float inv = inv_denoms[(uint64_t)t * n_head + h];
    const uint64_t out = ((uint64_t)t * n_head + h) * head_dim + d;
    heads[out + 0u] = acc0 * inv;
    heads[out + 1u] = acc1 * inv;
    heads[out + 2u] = acc2 * inv;
    heads[out + 3u] = acc3 * inv;
}

__global__ static void attention_decode_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const bool single_all = (n_tokens == 1u && ratio == 0u);
    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    if (visible_comp > n_comp) visible_comp = n_comp;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[DS4_ROCM_ATTENTION_SCORE_CAP];
    __shared__ uint32_t raw_rows[256];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (single_all) {
                raw_count = n_raw > 256u ? 256u : n_raw;
            } else if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();
    uint32_t n_score = raw_count + visible_comp;
    float local_max = sinks[h];
    if (visible_comp == 0 || n_tokens == 1u) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
            const float *kvrow = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            scores[r] = dot * scale;
            local_max = fmaxf(local_max, scores[r]);
        }
        for (uint32_t c = threadIdx.x; c < visible_comp; c += blockDim.x) {
            float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
            float s = -INFINITY;
            if (add > -1.0e20f) {
                const float *kvrow = comp_kv + (uint64_t)c * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
                s = dot * scale + add;
            }
            scores[raw_count + c] = s;
            local_max = fmaxf(local_max, s);
        }
    } else {
        uint32_t qlane = threadIdx.x & 7u;
        uint32_t qgroup = threadIdx.x >> 3u;
        for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
            uint32_t row = row0 + qgroup;
            if (row < n_score) {
                float add = 0.0f;
                const float *kvrow = NULL;
                if (row < raw_count) {
                    kvrow = raw_kv + (uint64_t)raw_rows[row] * head_dim;
                } else {
                    uint32_t c = row - raw_count;
                    add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                    if (add > -1.0e20f) kvrow = comp_kv + (uint64_t)c * head_dim;
                }
                float s = -INFINITY;
                if (kvrow) {
                    float dot = 0.0f;
                    for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * kvrow[d];
                    const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                    for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                        dot += __shfl_down_sync(static_cast<MASK_T>(mask), dot, off, 8);
                    }
                    s = dot * scale + add;
                }
                if (qlane == 0) scores[row] = s;
            }
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
            local_max = fmaxf(local_max, scores[i]);
        }
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x == 256u) {
        uint32_t d0 = threadIdx.x;
        uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)c * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + d] * scores[r];
            for (uint32_t c = 0; c < visible_comp; c++) acc += comp_kv[(uint64_t)c * head_dim + d] * scores[raw_count + c];
            oh[d] = acc / denom;
        }
    }
}

__global__ static void attention_indexed_mixed_scalar_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_tokens * n_head * head_dim;
    if (idx >= total) return;
    const uint32_t d = (uint32_t)(idx % head_dim);
    const uint64_t th = idx / head_dim;
    const uint32_t h = (uint32_t)(th % n_head);
    const uint32_t t = (uint32_t)(th / n_head);
    const uint32_t qpos = pos0 + t;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    const float scale = rsqrtf((float)head_dim);
    const uint32_t visible = ratio ? (qpos + 1u) / ratio : n_comp;
    float max_score = sinks[h];

    for (uint32_t r = 0; r < n_raw; r++) {
        const uint32_t kpos = first_raw_pos + r;
        if (kpos > qpos) continue;
        if (window != 0 && qpos - kpos >= window) continue;
        const uint32_t row = raw_cap ? ((raw_start + r) % raw_cap) : r;
        const float *kv = raw_kv + (uint64_t)row * head_dim;
        float s = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) s += qh[i] * kv[i];
        s *= scale;
        if (s > max_score) max_score = s;
    }
    for (uint32_t u = 0; u < top_k; u++) {
        const int32_t ci = topk[(uint64_t)t * top_k + u];
        if (ci < 0) continue;
        const uint32_t c = (uint32_t)ci;
        if (c >= n_comp || c >= visible) continue;
        const float *kv = comp_kv + (uint64_t)c * head_dim;
        float s = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) s += qh[i] * kv[i];
        s *= scale;
        if (s > max_score) max_score = s;
    }

    float denom = expf(sinks[h] - max_score);
    float acc = 0.0f;
    for (uint32_t r = 0; r < n_raw; r++) {
        const uint32_t kpos = first_raw_pos + r;
        if (kpos > qpos) continue;
        if (window != 0 && qpos - kpos >= window) continue;
        const uint32_t row = raw_cap ? ((raw_start + r) % raw_cap) : r;
        const float *kv = raw_kv + (uint64_t)row * head_dim;
        float s = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) s += qh[i] * kv[i];
        const float w = expf(s * scale - max_score);
        denom += w;
        acc += w * kv[d];
    }
    for (uint32_t u = 0; u < top_k; u++) {
        const int32_t ci = topk[(uint64_t)t * top_k + u];
        if (ci < 0) continue;
        const uint32_t c = (uint32_t)ci;
        if (c >= n_comp || c >= visible) continue;
        const float *kv = comp_kv + (uint64_t)c * head_dim;
        float s = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) s += qh[i] * kv[i];
        s *= scale;
        const float w = expf(s - max_score);
        denom += w;
        acc += w * kv[d];
    }
    heads[idx] = acc / denom;
}

__global__ static void attention_indexed_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[DS4_ROCM_ATTENTION_INDEXED_SCORE_CAP];
    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count;
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        comp_count = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    if (threadIdx.x == 0) {
        for (uint32_t i = 0;
             i < top_k && comp_count < DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP;
             i++) {
            int32_t c = topk[(uint64_t)t * top_k + i];
            if (c >= 0 && (uint32_t)c < visible_comp) comp_rows[comp_count++] = (uint32_t)c;
        }
    }
    __syncthreads();
    uint32_t n_score = raw_count + comp_count;
    float local_max = sinks[h];
    if (comp_count == 0) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
            const float *kvrow = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            scores[r] = dot * scale;
            local_max = fmaxf(local_max, scores[r]);
        }
    } else {
        uint32_t qlane = threadIdx.x & 7u;
        uint32_t qgroup = threadIdx.x >> 3u;
        for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
            uint32_t row = row0 + qgroup;
            if (row < n_score) {
                const float *kvrow = row < raw_count
                    ? raw_kv + (uint64_t)raw_rows[row] * head_dim
                    : comp_kv + (uint64_t)comp_rows[row - raw_count] * head_dim;
                float dot = 0.0f;
                for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * kvrow[d];
                const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                    dot += __shfl_down_sync(static_cast<MASK_T>(mask), dot, off, 8);
                }
                if (qlane == 0) scores[row] = dot * scale;
            }
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
            local_max = fmaxf(local_max, scores[i]);
        }
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x == 256u) {
        uint32_t d0 = threadIdx.x;
        uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        for (uint32_t c = 0; c < comp_count; c++) {
            float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)comp_rows[c] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + d] * scores[r];
            for (uint32_t s = 0; s < comp_count; s++) acc += comp_kv[(uint64_t)comp_rows[s] * head_dim + d] * scores[raw_count + s];
            oh[d] = acc / denom;
        }
    }
}

template <uint32_t ROWS_PER_STAGE, uint32_t HEADS_PER_GROUP>
__global__ static void attention_indexed_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * HEADS_PER_GROUP + warp;
    const bool valid_head = head < n_head;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP];
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count_s;
    __shared__ float4 kv_shared[ROWS_PER_STAGE * 128];

    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }

    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        comp_count_s = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
        for (uint32_t i = 0;
             i < top_k && comp_count_s < DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP;
             i++) {
            const int32_t ci = topk[(uint64_t)t * top_k + i];
            if (ci < 0) continue;
            const uint32_t c = (uint32_t)ci;
            if (c < n_comp && c < visible_comp) comp_rows[comp_count_s++] = c;
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();

    const uint32_t comp_count = comp_count_s;
    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = valid_head ? sinks[head] : -INFINITY;
    float sum_s = valid_head ? 1.0f : 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += ROWS_PER_STAGE) {
        const uint32_t nr = n_score - row0 < ROWS_PER_STAGE ? n_score - row0 : ROWS_PER_STAGE;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)comp_rows[sr - raw_count] * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(FULL_WARP_MASK, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

__global__ static void attention_static_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_q,
        uint32_t q_row0,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t_local = blockIdx.x;
    uint32_t t_abs = q_row0 + t_local;
    uint32_t head_group = blockIdx.y;
    if (t_local >= n_q || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    __shared__ float4 kv_shared[4 * 128];

    const uint32_t raw_count = window != 0u && t_abs + 1u > window ? window : t_abs + 1u;
    const uint32_t raw_start = t_abs + 1u - raw_count;
    uint32_t comp_count = 0;
    if (n_comp != 0u && ratio != 0u) {
        comp_count = (t_abs + 1u) / ratio;
        if (comp_count > n_comp) comp_count = n_comp;
    }
    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);

    /* Keep the fast heads8 staging, but use the same two-pass score/softmax
     * shape as the old-HIP warprows path.  The previous online recurrence was
     * close, but crossed greedy near-ties on long prompts. */
    __shared__ float scores[8 * 768];
    if (n_score > 768u) return;

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)(raw_start + sr) * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            const float *qh = q + ((uint64_t)t_local * n_head + head) * head_dim;
            const float *kvf = (const float *)kv_shared;
            for (uint32_t rr = 0; rr < nr; rr++) {
                float dot = 0.0f;
#pragma unroll 16
                for (uint32_t d = lane; d < 512u; d += 32u) dot += qh[d] * kvf[(uint64_t)rr * 512u + d];
                dot = warp_sum_f32(dot);
                if (lane == 0u) scores[warp * 768u + row0 + rr] = dot * scale;
            }
        }
        __syncthreads();
    }

    float max_s = valid_head ? sinks[head] : -INFINITY;
    if (valid_head) {
        const float *score_row = scores + warp * 768u;
        for (uint32_t i = lane; i < n_score; i += 32u) max_s = fmaxf(max_s, score_row[i]);
        max_s = warp_max_f32(max_s);
        max_s = __shfl_sync(FULL_WARP_MASK, max_s, 0);
    }
    float den = 0.0f;
    if (valid_head) {
        float *score_row = scores + warp * 768u;
        for (uint32_t i = lane; i < n_score; i += 32u) {
            float p = expf(score_row[i] - max_s);
            score_row[i] = p;
            den += p;
        }
        den = warp_sum_f32(den);
        den += expf(sinks[head] - max_s);
        den = __shfl_sync(FULL_WARP_MASK, den, 0);
    }

    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;
    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)(raw_start + sr) * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            const float *score_row = scores + warp * 768u;
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float p = den == 0.0f ? 0.0f : score_row[row0 + rr] / den;
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                o0.x += k0.x * p; o0.y += k0.y * p; o0.z += k0.z * p; o0.w += k0.w * p;
                o1.x += k1.x * p; o1.y += k1.y * p; o1.z += k1.z * p; o1.w += k1.w * p;
                o2.x += k2.x * p; o2.y += k2.y * p; o2.z += k2.z * p; o2.w += k2.w * p;
                o3.x += k3.x * p; o3.y += k3.y * p; o3.z += k3.z * p; o3.w += k3.w * p;
            }
        }
        __syncthreads();
    }
    if (valid_head) {
        float4 *out4 = (float4 *)(heads + ((uint64_t)t_local * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

/* Single-pass flash-style form of the static mixed prefill kernel. Its online
 * softmax changes FP32 reduction order, but avoids the 24 KiB score tile and
 * the second global KV traversal. Dispatch defaults to this form on gfx1151
 * and retains an explicit legacy fallback in the launch helper. */
__global__ static void attention_static_mixed_heads8_flash_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_q,
        uint32_t q_row0,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t t_local = blockIdx.x;
    const uint32_t t_abs = q_row0 + t_local;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = blockIdx.y * 8u + warp;
    if (t_local >= n_q || head_dim != 512u) return;
    const bool valid_head = head < n_head;

    __shared__ float4 kv_shared[4 * 128];

    const uint32_t raw_count = window != 0u && t_abs + 1u > window ? window : t_abs + 1u;
    const uint32_t raw_start = t_abs + 1u - raw_count;
    uint32_t comp_count = 0u;
    if (n_comp != 0u && ratio != 0u) {
        comp_count = (t_abs + 1u) / ratio;
        if (comp_count > n_comp) comp_count = n_comp;
    }
    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);

    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t_local * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = valid_head ? sinks[head] : -INFINITY;
    float sum_s = valid_head ? 1.0f : 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)(raw_start + sr) * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                const float4 k0 = kv4[lane +  0u];
                const float4 k1 = kv4[lane + 32u];
                const float4 k2 = kv4[lane + 64u];
                const float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) + dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) + dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(FULL_WARP_MASK, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
#define DS4_FLASH_ACCUM4(o, k) \
                do { \
                    (o).x = (o).x * old_scale + (k).x * row_scale; \
                    (o).y = (o).y * old_scale + (k).y * row_scale; \
                    (o).z = (o).z * old_scale + (k).z * row_scale; \
                    (o).w = (o).w * old_scale + (k).w * row_scale; \
                } while (0)
                DS4_FLASH_ACCUM4(o0, k0);
                DS4_FLASH_ACCUM4(o1, k1);
                DS4_FLASH_ACCUM4(o2, k2);
                DS4_FLASH_ACCUM4(o3, k3);
#undef DS4_FLASH_ACCUM4
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t_local * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

__global__ static void attention_decode_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t raw_count_s;
    __shared__ uint32_t raw_first_idx_s;
    __shared__ float4 kv_shared[4 * 128];

    const uint32_t qpos = pos0 + t;
    const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t comp_count = 0;
    if (n_comp != 0u) {
        if (n_tokens == 1u && ratio == 0u) {
            comp_count = n_comp;
        } else if (ratio != 0u) {
            comp_count = (qpos + 1u) / ratio;
            if (comp_count > n_comp) comp_count = n_comp;
        }
    }
    if (threadIdx.x == 0) {
        uint32_t raw_count = 0;
        uint32_t raw_first_idx = 0;
        if (n_raw != 0u) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0u && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
        raw_count_s = raw_count;
        raw_first_idx_s = raw_first_idx;
    }
    __syncthreads();
    const uint32_t raw_count = raw_count_s;
    const uint32_t raw_first_idx = raw_first_idx_s;
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();

    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = valid_head ? sinks[head] : -INFINITY;

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(FULL_WARP_MASK, score, 0);

                max_s = fmaxf(max_s, score);
            }
        }
        __syncthreads();
    }

    float sum_s = valid_head ? expf(sinks[head] - max_s) : 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(FULL_WARP_MASK, score, 0);

                const float row_scale = expf(score - max_s);
                sum_s += row_scale;
                o0.x += k0.x * row_scale; o0.y += k0.y * row_scale; o0.z += k0.z * row_scale; o0.w += k0.w * row_scale;
                o1.x += k1.x * row_scale; o1.y += k1.y * row_scale; o1.z += k1.z * row_scale; o1.w += k1.w * row_scale;
                o2.x += k2.x * row_scale; o2.y += k2.y * row_scale; o2.z += k2.z * row_scale; o2.w += k2.w * row_scale;
                o3.x += k3.x * row_scale; o3.y += k3.y * row_scale; o3.z += k3.z * row_scale; o3.w += k3.w * row_scale;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}
