// Shape oracle for the sparse QK stage in the five-row DSpark verifier.
//
// Build:
//   /opt/rocm/bin/hipcc -O3 -ffast-math -fno-finite-math-only \
//     --offload-arch=gfx1151 scripts/dspark_sparse_qk_wmma_bench.cu \
//     -o /tmp/dspark_sparse_qk_wmma_bench

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static constexpr uint32_t kTokens = 5;
static constexpr uint32_t kHeads = 64;
static constexpr uint32_t kDim = 512;
static constexpr uint32_t kRows = 768;
static constexpr uint32_t kKvRows = 4096;
static constexpr float kScale = 0.04419417382415922f;

static void check(hipError_t rc, const char *where) {
    if (rc != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, hipGetErrorString(rc));
        std::exit(1);
    }
}

static __device__ __forceinline__ float dot4(float4 a, float4 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

static __device__ __forceinline__ float warp_sum(float v) {
    for (uint32_t d = 16; d != 0; d >>= 1u) v += __shfl_down(v, d, 32);
    return v;
}

static __device__ __forceinline__ float warp_max(float v) {
    for (uint32_t d = 16; d != 0; d >>= 1u) v = fmaxf(v, __shfl_down(v, d, 32));
    return v;
}

// This preserves the production kernel's one-warp-per-head computation and
// eight-row shared KV staging, but writes QK scores instead of doing softmax.
__launch_bounds__(1024, 1)
__global__ static void qk_warp_f32(
        float *scores,
        const float *q,
        const float *kv,
        const int32_t *indices) {
    const uint32_t token = blockIdx.x;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = blockIdx.y * 32u + warp;
    __shared__ float4 tile[8u * (kDim / 4u)];

    const float4 *q4 = reinterpret_cast<const float4 *>(
        q + ((uint64_t)token * kHeads + head) * kDim);
    const float4 q0 = q4[lane + 0u];
    const float4 q1 = q4[lane + 32u];
    const float4 q2 = q4[lane + 64u];
    const float4 q3 = q4[lane + 96u];

    for (uint32_t row0 = 0; row0 < kRows; row0 += 8u) {
        for (uint32_t off = threadIdx.x; off < 8u * (kDim / 4u); off += blockDim.x) {
            const uint32_t row = row0 + off / (kDim / 4u);
            const uint32_t c4 = off % (kDim / 4u);
            const uint32_t kv_row = (uint32_t)indices[(uint64_t)token * kRows + row];
            tile[off] = reinterpret_cast<const float4 *>(kv + (uint64_t)kv_row * kDim)[c4];
        }
        __syncthreads();
#pragma unroll
        for (uint32_t r = 0; r < 8u; ++r) {
            const float4 *k4 = tile + r * (kDim / 4u);
            float v = dot4(q0, k4[lane + 0u]) +
                      dot4(q1, k4[lane + 32u]) +
                      dot4(q2, k4[lane + 64u]) +
                      dot4(q3, k4[lane + 96u]);
            v = warp_sum(v) * kScale;
            if (lane == 0u) {
                scores[((uint64_t)token * kHeads + head) * kRows + row0 + r] = v;
            }
        }
        __syncthreads();
    }
}

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// One wave computes a 16-head x 16-row score tile. The selected KV rows are
// loaded directly, matching AITER's sparse MLA dot shape without a gather
// buffer. This stage intentionally excludes softmax and PV accumulation.
__launch_bounds__(256, 1)
__global__ static void qk_wmma_f16(
        float *scores,
        const float *q,
        const float *kv,
        const int32_t *indices) {
    const uint32_t token = blockIdx.x;
    const uint32_t head0 = blockIdx.y * 16u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t lane16 = lane & 15u;

    for (uint32_t tile = wave; tile < kRows / 16u; tile += 8u) {
        float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const uint32_t head = head0 + lane16;
        const uint32_t row = tile * 16u + lane16;
        const uint32_t kv_row = (uint32_t)indices[(uint64_t)token * kRows + row];
        const float *qh = q + ((uint64_t)token * kHeads + head) * kDim;
        const float *kr = kv + (uint64_t)kv_row * kDim;

#pragma unroll 1
        for (uint32_t k0 = 0; k0 < kDim; k0 += 16u) {
            half16_t a;
            half16_t b;
#pragma unroll
            for (uint32_t i = 0; i < 16u; ++i) {
                a[i] = (_Float16)qh[k0 + i];
                b[i] = (_Float16)kr[k0 + i];
            }
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
        }

#pragma unroll
        for (uint32_t j = 0; j < 8u; ++j) {
            const uint32_t out_head = head0 + 2u*j + (lane >> 4u);
            scores[((uint64_t)token * kHeads + out_head) * kRows + row] = acc[j] * kScale;
        }
    }
}

// Production-shaped fused reference: one warp per head, shared KV staging,
// and online softmax/value accumulation.
__launch_bounds__(1024, 1)
__global__ static void attention_warp_f32(
        float *out,
        const float *q,
        const float *kv,
        const int32_t *indices,
        const float *sinks) {
    const uint32_t token = blockIdx.x;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = blockIdx.y * 32u + warp;
    __shared__ float4 tile[8u * (kDim / 4u)];

    const float4 *q4 = reinterpret_cast<const float4 *>(
        q + ((uint64_t)token * kHeads + head) * kDim);
    const float4 q0 = q4[lane + 0u];
    const float4 q1 = q4[lane + 32u];
    const float4 q2 = q4[lane + 64u];
    const float4 q3 = q4[lane + 96u];
    float max_s = sinks[head];
    float sum_s = 1.0f;
    float4 o0 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < kRows; row0 += 8u) {
        for (uint32_t off = threadIdx.x; off < 8u * (kDim / 4u); off += blockDim.x) {
            const uint32_t row = row0 + off / (kDim / 4u);
            const uint32_t c4 = off % (kDim / 4u);
            const uint32_t kv_row = (uint32_t)indices[(uint64_t)token * kRows + row];
            tile[off] = reinterpret_cast<const float4 *>(kv + (uint64_t)kv_row * kDim)[c4];
        }
        __syncthreads();
#pragma unroll
        for (uint32_t r = 0; r < 8u; ++r) {
            const float4 *k4 = tile + r * (kDim / 4u);
            const float4 k0 = k4[lane + 0u];
            const float4 k1 = k4[lane + 32u];
            const float4 k2 = k4[lane + 64u];
            const float4 k3 = k4[lane + 96u];
            float score = dot4(q0, k0) + dot4(q1, k1) + dot4(q2, k2) + dot4(q3, k3);
            score = __shfl(warp_sum(score) * kScale, 0, 32);
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
        __syncthreads();
    }
    const float inv = 1.0f / sum_s;
    float4 *dst = reinterpret_cast<float4 *>(
        out + ((uint64_t)token * kHeads + head) * kDim);
    o0.x *= inv; o0.y *= inv; o0.z *= inv; o0.w *= inv;
    o1.x *= inv; o1.y *= inv; o1.z *= inv; o1.w *= inv;
    o2.x *= inv; o2.y *= inv; o2.z *= inv; o2.w *= inv;
    o3.x *= inv; o3.y *= inv; o3.z *= inv; o3.w *= inv;
    dst[lane + 0u] = o0;
    dst[lane + 32u] = o1;
    dst[lane + 64u] = o2;
    dst[lane + 96u] = o3;
}

// AITER-style 16-head sparse attention: WMMA QK, FP32 softmax, then WMMA PV.
// The 48 KiB score tile stays block-local and no persistent gather buffer is
// introduced.
template <bool CORRECTED>
__launch_bounds__(256, 1)
__global__ static void attention_wmma_f16(
        float *out,
        const float *q,
        const float *kv,
        const int32_t *indices,
        const float *sinks) {
    const uint32_t token = blockIdx.x;
    const uint32_t head0 = blockIdx.y * 16u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t lane16 = lane & 15u;
    __shared__ float probs[16u * kRows];

    for (uint32_t tile = wave; tile < kRows / 16u; tile += 8u) {
        float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const uint32_t head = head0 + lane16;
        const uint32_t row = tile * 16u + lane16;
        const uint32_t kv_row = (uint32_t)indices[(uint64_t)token * kRows + row];
        const float *qh = q + ((uint64_t)token * kHeads + head) * kDim;
        const float *kr = kv + (uint64_t)kv_row * kDim;
#pragma unroll 1
        for (uint32_t k0 = 0; k0 < kDim; k0 += 16u) {
            half16_t a;
            half16_t b;
            half16_t ar;
            half16_t br;
#pragma unroll
            for (uint32_t i = 0; i < 16u; ++i) {
                const float af = qh[k0 + i];
                const float bf = kr[k0 + i];
                a[i] = (_Float16)af;
                b[i] = (_Float16)bf;
                ar[i] = (_Float16)(af - (float)a[i]);
                br[i] = (_Float16)(bf - (float)b[i]);
            }
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
            if constexpr (CORRECTED) {
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ar, b, acc);
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, br, acc);
            }
        }
#pragma unroll
        for (uint32_t j = 0; j < 8u; ++j) {
            const uint32_t h = 2u*j + (lane >> 4u);
            probs[(uint64_t)h * kRows + row] = acc[j] * kScale;
        }
    }
    __syncthreads();

#pragma unroll
    for (uint32_t pass = 0; pass < 2u; ++pass) {
        const uint32_t h = wave + pass * 8u;
        float m = sinks[head0 + h];
        for (uint32_t r = lane; r < kRows; r += 32u) {
            m = fmaxf(m, probs[(uint64_t)h * kRows + r]);
        }
        m = __shfl(warp_max(m), 0, 32);
        float sum = lane == 0u ? expf(sinks[head0 + h] - m) : 0.0f;
        for (uint32_t r = lane; r < kRows; r += 32u) {
            sum += expf(probs[(uint64_t)h * kRows + r] - m);
        }
        sum = __shfl(warp_sum(sum), 0, 32);
        for (uint32_t r = lane; r < kRows; r += 32u) {
            probs[(uint64_t)h * kRows + r] = expf(probs[(uint64_t)h * kRows + r] - m) / sum;
        }
    }
    __syncthreads();

    for (uint32_t out_tile = wave; out_tile < kDim / 16u; out_tile += 8u) {
        float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const uint32_t h = lane16;
        const uint32_t d = out_tile * 16u + lane16;
#pragma unroll 1
        for (uint32_t row0 = 0; row0 < kRows; row0 += 16u) {
            half16_t a;
            half16_t b;
            half16_t ar;
            half16_t br;
#pragma unroll
            for (uint32_t i = 0; i < 16u; ++i) {
                const uint32_t kv_row = (uint32_t)indices[
                    (uint64_t)token * kRows + row0 + i];
                const float af = probs[(uint64_t)h * kRows + row0 + i];
                const float bf = kv[(uint64_t)kv_row * kDim + d];
                a[i] = (_Float16)af;
                b[i] = (_Float16)bf;
                ar[i] = (_Float16)(af - (float)a[i]);
                br[i] = (_Float16)(bf - (float)b[i]);
            }
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
            if constexpr (CORRECTED) {
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ar, b, acc);
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, br, acc);
            }
        }
#pragma unroll
        for (uint32_t j = 0; j < 8u; ++j) {
            const uint32_t out_head = head0 + 2u*j + (lane >> 4u);
            out[((uint64_t)token * kHeads + out_head) * kDim + d] = acc[j];
        }
    }
}

// Keep WMMA only for corrected QK. Softmax and PV use one FP32 warp per head
// in the original selected-row order.
__launch_bounds__(512, 1)
__global__ static void attention_wmma_qk_f32_pv(
        float *out,
        const float *q,
        const float *kv,
        const int32_t *indices,
        const float *sinks) {
    const uint32_t token = blockIdx.x;
    const uint32_t head0 = blockIdx.y * 16u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t lane16 = lane & 15u;
    __shared__ float probs[16u * kRows];

    for (uint32_t tile = wave; tile < kRows / 16u; tile += 16u) {
        float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const uint32_t head = head0 + lane16;
        const uint32_t row = tile * 16u + lane16;
        const uint32_t kv_row = (uint32_t)indices[(uint64_t)token * kRows + row];
        const float *qh = q + ((uint64_t)token * kHeads + head) * kDim;
        const float *kr = kv + (uint64_t)kv_row * kDim;
#pragma unroll 1
        for (uint32_t k0 = 0; k0 < kDim; k0 += 16u) {
            half16_t a, b, ar, br;
#pragma unroll
            for (uint32_t i = 0; i < 16u; ++i) {
                const float af = qh[k0 + i];
                const float bf = kr[k0 + i];
                a[i] = (_Float16)af;
                b[i] = (_Float16)bf;
                ar[i] = (_Float16)(af - (float)a[i]);
                br[i] = (_Float16)(bf - (float)b[i]);
            }
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ar, b, acc);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, br, acc);
        }
#pragma unroll
        for (uint32_t j = 0; j < 8u; ++j) {
            const uint32_t h = 2u*j + (lane >> 4u);
            probs[(uint64_t)h * kRows + row] = acc[j] * kScale;
        }
    }
    __syncthreads();

    const uint32_t h = wave;
    float m = sinks[head0 + h];
    for (uint32_t r = lane; r < kRows; r += 32u) {
        m = fmaxf(m, probs[(uint64_t)h * kRows + r]);
    }
    m = __shfl(warp_max(m), 0, 32);
    float sum = lane == 0u ? expf(sinks[head0 + h] - m) : 0.0f;
    for (uint32_t r = lane; r < kRows; r += 32u) {
        sum += expf(probs[(uint64_t)h * kRows + r] - m);
    }
    sum = __shfl(warp_sum(sum), 0, 32);
    for (uint32_t r = lane; r < kRows; r += 32u) {
        probs[(uint64_t)h * kRows + r] = expf(probs[(uint64_t)h * kRows + r] - m) / sum;
    }
    __syncthreads();

    float4 acc0 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 acc1 = acc0, acc2 = acc0, acc3 = acc0;
    for (uint32_t r = 0; r < kRows; ++r) {
        const uint32_t kv_row = (uint32_t)indices[(uint64_t)token * kRows + r];
        const float4 *v = reinterpret_cast<const float4 *>(kv + (uint64_t)kv_row * kDim);
        const float p = probs[(uint64_t)h * kRows + r];
        const float4 v0 = v[lane + 0u];
        const float4 v1 = v[lane + 32u];
        const float4 v2 = v[lane + 64u];
        const float4 v3 = v[lane + 96u];
        acc0.x = fmaf(v0.x, p, acc0.x); acc0.y = fmaf(v0.y, p, acc0.y);
        acc0.z = fmaf(v0.z, p, acc0.z); acc0.w = fmaf(v0.w, p, acc0.w);
        acc1.x = fmaf(v1.x, p, acc1.x); acc1.y = fmaf(v1.y, p, acc1.y);
        acc1.z = fmaf(v1.z, p, acc1.z); acc1.w = fmaf(v1.w, p, acc1.w);
        acc2.x = fmaf(v2.x, p, acc2.x); acc2.y = fmaf(v2.y, p, acc2.y);
        acc2.z = fmaf(v2.z, p, acc2.z); acc2.w = fmaf(v2.w, p, acc2.w);
        acc3.x = fmaf(v3.x, p, acc3.x); acc3.y = fmaf(v3.y, p, acc3.y);
        acc3.z = fmaf(v3.z, p, acc3.z); acc3.w = fmaf(v3.w, p, acc3.w);
    }
    float4 *dst = reinterpret_cast<float4 *>(out +
        ((uint64_t)token * kHeads + head0 + h) * kDim);
    dst[lane + 0u] = acc0;
    dst[lane + 32u] = acc1;
    dst[lane + 64u] = acc2;
    dst[lane + 96u] = acc3;
}

struct buffers {
    std::vector<float> q;
    std::vector<float> kv;
    std::vector<int32_t> indices;
    float *dq = nullptr;
    float *dkv = nullptr;
    float *dbase = nullptr;
    float *dwmma = nullptr;
    float *dattn_base = nullptr;
    float *dattn_wmma = nullptr;
    float *dattn_corrected = nullptr;
    float *dattn_qk_only = nullptr;
    float *dsinks = nullptr;
    int32_t *dindices = nullptr;

    buffers()
        : q((uint64_t)kTokens * kHeads * kDim),
          kv((uint64_t)kKvRows * kDim),
          indices((uint64_t)kTokens * kRows) {
        std::mt19937 rng(1151);
        std::uniform_real_distribution<float> value(-0.125f, 0.125f);
        for (float &v : q) v = value(rng);
        for (float &v : kv) v = value(rng);
        for (uint32_t t = 0; t < kTokens; ++t) {
            std::vector<int32_t> rows(kKvRows);
            for (uint32_t i = 0; i < kKvRows; ++i) rows[i] = (int32_t)i;
            std::shuffle(rows.begin(), rows.end(), rng);
            rows.resize(kRows);
            std::sort(rows.begin(), rows.end());
            std::copy(rows.begin(), rows.end(), indices.begin() + (uint64_t)t * kRows);
        }
        const size_t score_bytes = (size_t)kTokens * kHeads * kRows * sizeof(float);
        check(hipMalloc(&dq, q.size() * sizeof(float)), "q allocation");
        check(hipMalloc(&dkv, kv.size() * sizeof(float)), "kv allocation");
        check(hipMalloc(&dindices, indices.size() * sizeof(int32_t)), "indices allocation");
        check(hipMalloc(&dbase, score_bytes), "baseline allocation");
        check(hipMalloc(&dwmma, score_bytes), "wmma allocation");
        check(hipMalloc(&dattn_base, q.size() * sizeof(float)), "baseline attention allocation");
        check(hipMalloc(&dattn_wmma, q.size() * sizeof(float)), "wmma attention allocation");
        check(hipMalloc(&dattn_corrected, q.size() * sizeof(float)), "corrected attention allocation");
        check(hipMalloc(&dattn_qk_only, q.size() * sizeof(float)), "QK-only attention allocation");
        check(hipMalloc(&dsinks, kHeads * sizeof(float)), "sink allocation");
        check(hipMemcpy(dq, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice), "q upload");
        check(hipMemcpy(dkv, kv.data(), kv.size() * sizeof(float), hipMemcpyHostToDevice), "kv upload");
        check(hipMemcpy(dindices, indices.data(), indices.size() * sizeof(int32_t), hipMemcpyHostToDevice), "indices upload");
        std::vector<float> sinks(kHeads, -1.25f);
        check(hipMemcpy(dsinks, sinks.data(), sinks.size() * sizeof(float), hipMemcpyHostToDevice), "sink upload");
    }

    ~buffers() {
        (void)hipFree(dq);
        (void)hipFree(dkv);
        (void)hipFree(dindices);
        (void)hipFree(dbase);
        (void)hipFree(dwmma);
        (void)hipFree(dattn_base);
        (void)hipFree(dattn_wmma);
        (void)hipFree(dattn_corrected);
        (void)hipFree(dattn_qk_only);
        (void)hipFree(dsinks);
    }
};

static void launch_base(buffers &b) {
    qk_warp_f32<<<dim3(kTokens, kHeads / 32u), 1024>>>(b.dbase, b.dq, b.dkv, b.dindices);
}

static void launch_wmma(buffers &b) {
    qk_wmma_f16<<<dim3(kTokens, kHeads / 16u), 256>>>(b.dwmma, b.dq, b.dkv, b.dindices);
}

static void launch_attention_base(buffers &b) {
    attention_warp_f32<<<dim3(kTokens, kHeads / 32u), 1024>>>(
        b.dattn_base, b.dq, b.dkv, b.dindices, b.dsinks);
}

static void launch_attention_wmma(buffers &b) {
    attention_wmma_f16<false><<<dim3(kTokens, kHeads / 16u), 256>>>(
        b.dattn_wmma, b.dq, b.dkv, b.dindices, b.dsinks);
}

static void launch_attention_corrected(buffers &b) {
    attention_wmma_f16<true><<<dim3(kTokens, kHeads / 16u), 256>>>(
        b.dattn_corrected, b.dq, b.dkv, b.dindices, b.dsinks);
}

static void launch_attention_qk_only(buffers &b) {
    attention_wmma_qk_f32_pv<<<dim3(kTokens, kHeads / 16u), 512>>>(
        b.dattn_qk_only, b.dq, b.dkv, b.dindices, b.dsinks);
}

template <typename Launch>
static float time_ms(Launch launch, uint32_t iters) {
    hipEvent_t begin, end;
    check(hipEventCreate(&begin), "event begin");
    check(hipEventCreate(&end), "event end");
    for (uint32_t i = 0; i < 5u; ++i) launch();
    check(hipDeviceSynchronize(), "warmup");
    check(hipEventRecord(begin), "record begin");
    for (uint32_t i = 0; i < iters; ++i) launch();
    check(hipEventRecord(end), "record end");
    check(hipEventSynchronize(end), "wait end");
    float elapsed = 0.0f;
    check(hipEventElapsedTime(&elapsed, begin, end), "elapsed");
    check(hipEventDestroy(begin), "destroy begin");
    check(hipEventDestroy(end), "destroy end");
    return elapsed / (float)iters;
}

int main() {
    buffers b;
    launch_base(b);
    launch_wmma(b);
    launch_attention_base(b);
    launch_attention_wmma(b);
    launch_attention_corrected(b);
    launch_attention_qk_only(b);
    check(hipDeviceSynchronize(), "correctness kernels");

    const size_t n = (size_t)kTokens * kHeads * kRows;
    std::vector<float> base(n), wmma(n);
    check(hipMemcpy(base.data(), b.dbase, n * sizeof(float), hipMemcpyDeviceToHost), "baseline download");
    check(hipMemcpy(wmma.data(), b.dwmma, n * sizeof(float), hipMemcpyDeviceToHost), "wmma download");
    double sum_sq = 0.0;
    double ref_sq = 0.0;
    float max_abs = 0.0f;
    uint64_t bad = 0;
    for (size_t i = 0; i < n; ++i) {
        const float d = std::fabs(base[i] - wmma[i]);
        max_abs = std::max(max_abs, d);
        sum_sq += (double)d * d;
        ref_sq += (double)base[i] * base[i];
        if (!std::isfinite(wmma[i])) bad++;
    }

    std::vector<float> attn_base(b.q.size()), attn_wmma(b.q.size());
    std::vector<float> attn_corrected(b.q.size());
    std::vector<float> attn_qk_only(b.q.size());
    check(hipMemcpy(attn_base.data(), b.dattn_base, attn_base.size() * sizeof(float), hipMemcpyDeviceToHost), "baseline attention download");
    check(hipMemcpy(attn_wmma.data(), b.dattn_wmma, attn_wmma.size() * sizeof(float), hipMemcpyDeviceToHost), "wmma attention download");
    check(hipMemcpy(attn_corrected.data(), b.dattn_corrected, attn_corrected.size() * sizeof(float), hipMemcpyDeviceToHost), "corrected attention download");
    check(hipMemcpy(attn_qk_only.data(), b.dattn_qk_only, attn_qk_only.size() * sizeof(float), hipMemcpyDeviceToHost), "QK-only attention download");
    double attn_sum_sq = 0.0;
    double attn_ref_sq = 0.0;
    float attn_max_abs = 0.0f;
    uint64_t attn_bad = 0;
    double corrected_sum_sq = 0.0;
    float corrected_max_abs = 0.0f;
    uint64_t corrected_bad = 0;
    double qk_only_sum_sq = 0.0;
    float qk_only_max_abs = 0.0f;
    uint64_t qk_only_bad = 0;
    for (size_t i = 0; i < attn_base.size(); ++i) {
        const float d = std::fabs(attn_base[i] - attn_wmma[i]);
        attn_max_abs = std::max(attn_max_abs, d);
        attn_sum_sq += (double)d * d;
        attn_ref_sq += (double)attn_base[i] * attn_base[i];
        if (!std::isfinite(attn_wmma[i])) attn_bad++;
        const float dc = std::fabs(attn_base[i] - attn_corrected[i]);
        corrected_max_abs = std::max(corrected_max_abs, dc);
        corrected_sum_sq += (double)dc * dc;
        if (!std::isfinite(attn_corrected[i])) corrected_bad++;
        const float dqk = std::fabs(attn_base[i] - attn_qk_only[i]);
        qk_only_max_abs = std::max(qk_only_max_abs, dqk);
        qk_only_sum_sq += (double)dqk * dqk;
        if (!std::isfinite(attn_qk_only[i])) qk_only_bad++;
    }

    constexpr uint32_t iters = 200;
    const float base_ms = time_ms([&] { launch_base(b); }, iters);
    const float wmma_ms = time_ms([&] { launch_wmma(b); }, iters);
    const float attn_base_ms = time_ms([&] { launch_attention_base(b); }, iters);
    const float attn_wmma_ms = time_ms([&] { launch_attention_wmma(b); }, iters);
    const float attn_corrected_ms = time_ms([&] { launch_attention_corrected(b); }, iters);
    const float attn_qk_only_ms = time_ms([&] { launch_attention_qk_only(b); }, iters);
    std::printf("shape tokens=%u heads=%u dim=%u selected_rows=%u kv_rows=%u\n",
                kTokens, kHeads, kDim, kRows, kKvRows);
    std::printf("correctness max_abs=%.9g rms=%.9g rel_rms=%.9g nonfinite=%llu\n",
                max_abs, std::sqrt(sum_sq / n),
                ref_sq == 0.0 ? 0.0 : std::sqrt(sum_sq / ref_sq),
                (unsigned long long)bad);
    std::printf("qk warp_f32_ms=%.6f wmma_f16_ms=%.6f speedup=%.3fx\n",
                base_ms, wmma_ms, base_ms / wmma_ms);
    std::printf("attention correctness max_abs=%.9g rms=%.9g rel_rms=%.9g nonfinite=%llu\n",
                attn_max_abs, std::sqrt(attn_sum_sq / attn_base.size()),
                attn_ref_sq == 0.0 ? 0.0 : std::sqrt(attn_sum_sq / attn_ref_sq),
                (unsigned long long)attn_bad);
    std::printf("attention warp_f32_ms=%.6f wmma_f16_ms=%.6f speedup=%.3fx\n",
                attn_base_ms, attn_wmma_ms, attn_base_ms / attn_wmma_ms);
    std::printf("corrected attention max_abs=%.9g rms=%.9g rel_rms=%.9g nonfinite=%llu\n",
                corrected_max_abs,
                std::sqrt(corrected_sum_sq / attn_base.size()),
                attn_ref_sq == 0.0 ? 0.0 : std::sqrt(corrected_sum_sq / attn_ref_sq),
                (unsigned long long)corrected_bad);
    std::printf("corrected attention_ms=%.6f speedup=%.3fx\n",
                attn_corrected_ms, attn_base_ms / attn_corrected_ms);
    std::printf("QK-only max_abs=%.9g rms=%.9g rel_rms=%.9g nonfinite=%llu\n",
                qk_only_max_abs, std::sqrt(qk_only_sum_sq / attn_base.size()),
                attn_ref_sq == 0.0 ? 0.0 : std::sqrt(qk_only_sum_sq / attn_ref_sq),
                (unsigned long long)qk_only_bad);
    std::printf("QK-only attention_ms=%.6f speedup=%.3fx\n",
                attn_qk_only_ms, attn_base_ms / attn_qk_only_ms);
    return bad == 0 && attn_bad == 0 && corrected_bad == 0 && qk_only_bad == 0 ? 0 : 1;
}
