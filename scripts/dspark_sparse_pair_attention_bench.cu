// Exact paired-query sparse-attention oracle for the DSpark five-row verifier.
// Adjacent queries share 127/128 raw rows and 511/512 compressed rows, matching
// the live layer-20 capture at position 2052.  Each paired block merges the two
// sorted row lists, stages every union row once, and preserves each query's
// original online-softmax visitation order.
//
// Build:
//   /opt/rocm/bin/hipcc -O3 -ffast-math -fno-finite-math-only \
//     --offload-arch=gfx1151 scripts/dspark_sparse_pair_attention_bench.cu \
//     -o /tmp/dspark_sparse_pair_attention_bench

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static constexpr uint32_t kTokens = 5;
static constexpr uint32_t kPairs = (kTokens + 1u) / 2u;
static constexpr uint32_t kHeads = 64;
static constexpr uint32_t kDim = 512;
static constexpr uint32_t kRaw = 128;
static constexpr uint32_t kComp = 512;
static constexpr uint32_t kRows = kRaw + kComp;
static constexpr uint32_t kPairCap = 2u * kRows;
static constexpr uint32_t kKvRows = 2048;
static constexpr uint32_t kStageRows = 8;
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
#pragma unroll
    for (uint32_t d = 16; d != 0; d >>= 1u) v += __shfl_down(v, d, 32);
    return v;
}

static __device__ __forceinline__ void online_update(
        float &m, float &sum, float4 out[4], float score,
        float4 v0, float4 v1, float4 v2, float4 v3) {
    const float nm = fmaxf(m, score);
    const float old_scale = expf(m - nm);
    const float row_scale = expf(score - nm);
    sum = sum * old_scale + row_scale;
#define DS4_PAIR_ACC(o_, v_) do {                                      \
    (o_).x = (o_).x * old_scale + (v_).x * row_scale;                  \
    (o_).y = (o_).y * old_scale + (v_).y * row_scale;                  \
    (o_).z = (o_).z * old_scale + (v_).z * row_scale;                  \
    (o_).w = (o_).w * old_scale + (v_).w * row_scale;                  \
} while (0)
    DS4_PAIR_ACC(out[0], v0);
    DS4_PAIR_ACC(out[1], v1);
    DS4_PAIR_ACC(out[2], v2);
    DS4_PAIR_ACC(out[3], v3);
#undef DS4_PAIR_ACC
    m = nm;
}

// Shipping-shape reference: one token x 32 heads per 1024-thread block.
__launch_bounds__(1024, 1)
__global__ static void attention_single_query(
        float *out, const float *q, const float *kv,
        const int32_t *indices, const float *sinks) {
    const uint32_t token = blockIdx.x;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = blockIdx.y * 32u + warp;
    __shared__ float4 tile[kStageRows * (kDim / 4u)];

    const float4 *q4 = reinterpret_cast<const float4 *>(
        q + ((uint64_t)token * kHeads + head) * kDim);
    const float4 qv[4] = {
        q4[lane], q4[lane + 32u], q4[lane + 64u], q4[lane + 96u]
    };
    float m = sinks[head], sum = 1.0f;
    float4 acc[4] = {};

    for (uint32_t row0 = 0; row0 < kRows; row0 += kStageRows) {
        for (uint32_t off = threadIdx.x;
             off < kStageRows * (kDim / 4u); off += blockDim.x) {
            const uint32_t rr = off / (kDim / 4u);
            const uint32_t c4 = off % (kDim / 4u);
            const uint32_t row = (uint32_t)indices[
                (uint64_t)token * kRows + row0 + rr];
            tile[off] = reinterpret_cast<const float4 *>(
                kv + (uint64_t)row * kDim)[c4];
        }
        __syncthreads();
#pragma unroll
        for (uint32_t rr = 0; rr < kStageRows; ++rr) {
            const float4 *v = tile + rr * (kDim / 4u);
            const float4 v0 = v[lane], v1 = v[lane + 32u];
            const float4 v2 = v[lane + 64u], v3 = v[lane + 96u];
            float score = dot4(qv[0], v0) + dot4(qv[1], v1) +
                          dot4(qv[2], v2) + dot4(qv[3], v3);
            score = __shfl(warp_sum(score) * kScale, 0, 32);
            online_update(m, sum, acc, score, v0, v1, v2, v3);
        }
        __syncthreads();
    }
    const float inv = 1.0f / sum;
    float4 *dst = reinterpret_cast<float4 *>(
        out + ((uint64_t)token * kHeads + head) * kDim);
#pragma unroll
    for (uint32_t i = 0; i < 4u; ++i) {
        acc[i].x *= inv; acc[i].y *= inv;
        acc[i].z *= inv; acc[i].w *= inv;
        dst[lane + 32u*i] = acc[i];
    }
}

template <uint32_t HEADS_PER_GROUP>
__launch_bounds__(32u * HEADS_PER_GROUP, 1)
__global__ static void attention_pair_query(
        float *out, const float *q, const float *kv,
        const int32_t *pair_rows, const uint8_t *pair_masks,
        const uint32_t *pair_counts, const float *sinks) {
    const uint32_t pair = blockIdx.x;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = blockIdx.y * HEADS_PER_GROUP + warp;
    const bool valid_head = head < kHeads;
    const uint32_t token0 = 2u * pair;
    const uint32_t token1 = token0 + 1u;
    const bool valid_token1 = token1 < kTokens;
    const uint32_t count = pair_counts[pair];

    __shared__ float4 tile[kStageRows * (kDim / 4u)];
    __shared__ int32_t row_ids[kStageRows];
    __shared__ uint8_t row_masks[kStageRows];

    float4 qv[2][4] = {};
    if (valid_head) {
        const float4 *q0 = reinterpret_cast<const float4 *>(
            q + ((uint64_t)token0 * kHeads + head) * kDim);
#pragma unroll
        for (uint32_t i = 0; i < 4u; ++i) qv[0][i] = q0[lane + 32u*i];
        if (valid_token1) {
            const float4 *q1 = reinterpret_cast<const float4 *>(
                q + ((uint64_t)token1 * kHeads + head) * kDim);
#pragma unroll
            for (uint32_t i = 0; i < 4u; ++i) qv[1][i] = q1[lane + 32u*i];
        }
    }
    float m[2] = {
        valid_head ? sinks[head] : -INFINITY,
        valid_head && valid_token1 ? sinks[head] : -INFINITY
    };
    float sum[2] = {valid_head ? 1.0f : 0.0f,
                    valid_head && valid_token1 ? 1.0f : 0.0f};
    float4 acc[2][4] = {};

    for (uint32_t row0 = 0; row0 < count; row0 += kStageRows) {
        const uint32_t nr = min(kStageRows, count - row0);
        if (threadIdx.x < nr) {
            row_ids[threadIdx.x] = pair_rows[(uint64_t)pair*kPairCap + row0 + threadIdx.x];
            row_masks[threadIdx.x] = pair_masks[(uint64_t)pair*kPairCap + row0 + threadIdx.x];
        }
        __syncthreads();
        for (uint32_t off = threadIdx.x;
             off < nr * (kDim / 4u); off += blockDim.x) {
            const uint32_t rr = off / (kDim / 4u);
            const uint32_t c4 = off % (kDim / 4u);
            tile[off] = reinterpret_cast<const float4 *>(
                kv + (uint64_t)(uint32_t)row_ids[rr] * kDim)[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; ++rr) {
                const float4 *v = tile + rr * (kDim / 4u);
                const float4 v0 = v[lane], v1 = v[lane + 32u];
                const float4 v2 = v[lane + 64u], v3 = v[lane + 96u];
                const uint8_t mask = row_masks[rr];
#pragma unroll
                for (uint32_t t = 0; t < 2u; ++t) {
                    if ((mask & (1u << t)) == 0u) continue;
                    float score = dot4(qv[t][0], v0) + dot4(qv[t][1], v1) +
                                  dot4(qv[t][2], v2) + dot4(qv[t][3], v3);
                    score = __shfl(warp_sum(score) * kScale, 0, 32);
                    online_update(m[t], sum[t], acc[t], score, v0, v1, v2, v3);
                }
            }
        }
        __syncthreads();
    }
    if (valid_head) {
#pragma unroll
        for (uint32_t t = 0; t < 2u; ++t) {
            const uint32_t token = token0 + t;
            if (token >= kTokens) continue;
            const float inv = 1.0f / sum[t];
            float4 *dst = reinterpret_cast<float4 *>(
                out + ((uint64_t)token * kHeads + head) * kDim);
#pragma unroll
            for (uint32_t i = 0; i < 4u; ++i) {
                acc[t][i].x *= inv; acc[t][i].y *= inv;
                acc[t][i].z *= inv; acc[t][i].w *= inv;
                dst[lane + 32u*i] = acc[t][i];
            }
        }
    }
}

struct buffers {
    std::vector<float> q, kv;
    std::vector<int32_t> indices, pair_rows;
    std::vector<uint8_t> pair_masks;
    std::vector<uint32_t> pair_counts;
    float *dq = nullptr, *dkv = nullptr, *dref = nullptr, *dcand = nullptr;
    float *dsinks = nullptr;
    int32_t *dindices = nullptr, *dpair_rows = nullptr;
    uint8_t *dpair_masks = nullptr;
    uint32_t *dpair_counts = nullptr;

    buffers()
        : q((uint64_t)kTokens*kHeads*kDim), kv((uint64_t)kKvRows*kDim),
          indices((uint64_t)kTokens*kRows), pair_rows((uint64_t)kPairs*kPairCap, -1),
          pair_masks((uint64_t)kPairs*kPairCap, 0), pair_counts(kPairs, 0) {
        std::mt19937 rng(1151);
        std::uniform_real_distribution<float> value(-0.125f, 0.125f);
        for (float &v : q) v = value(rng);
        for (float &v : kv) v = value(rng);
        for (uint32_t t = 0; t < kTokens; ++t) {
            int32_t *dst = indices.data() + (uint64_t)t*kRows;
            for (uint32_t r = 0; r < kRaw; ++r) dst[r] = (int32_t)(t + r);
            for (uint32_t r = 0; r < kComp - 1u; ++r) dst[kRaw + r] = (int32_t)(1024u + r);
            dst[kRows - 1u] = (int32_t)(1024u + kComp - 1u + t);
        }
        for (uint32_t p = 0; p < kPairs; ++p) {
            const uint32_t t0 = 2u*p, t1 = t0 + 1u;
            uint32_t a = 0, b = 0, n = 0;
            while (a < kRows || (t1 < kTokens && b < kRows)) {
                const int32_t va = a < kRows ? indices[(uint64_t)t0*kRows + a] : INT32_MAX;
                const int32_t vb = t1 < kTokens && b < kRows ?
                    indices[(uint64_t)t1*kRows + b] : INT32_MAX;
                const int32_t row = va < vb ? va : vb;
                uint8_t mask = 0;
                if (va == row) { mask |= 1u; ++a; }
                if (vb == row) { mask |= 2u; ++b; }
                pair_rows[(uint64_t)p*kPairCap + n] = row;
                pair_masks[(uint64_t)p*kPairCap + n] = mask;
                ++n;
            }
            pair_counts[p] = n;
        }
        const size_t out_bytes = q.size()*sizeof(float);
        check(hipMalloc(&dq, out_bytes), "q alloc");
        check(hipMalloc(&dkv, kv.size()*sizeof(float)), "kv alloc");
        check(hipMalloc(&dindices, indices.size()*sizeof(int32_t)), "indices alloc");
        check(hipMalloc(&dpair_rows, pair_rows.size()*sizeof(int32_t)), "pair rows alloc");
        check(hipMalloc(&dpair_masks, pair_masks.size()*sizeof(uint8_t)), "pair masks alloc");
        check(hipMalloc(&dpair_counts, pair_counts.size()*sizeof(uint32_t)), "pair counts alloc");
        check(hipMalloc(&dref, out_bytes), "ref alloc");
        check(hipMalloc(&dcand, out_bytes), "candidate alloc");
        check(hipMalloc(&dsinks, kHeads*sizeof(float)), "sinks alloc");
        check(hipMemcpy(dq, q.data(), out_bytes, hipMemcpyHostToDevice), "q upload");
        check(hipMemcpy(dkv, kv.data(), kv.size()*sizeof(float), hipMemcpyHostToDevice), "kv upload");
        check(hipMemcpy(dindices, indices.data(), indices.size()*sizeof(int32_t), hipMemcpyHostToDevice), "indices upload");
        check(hipMemcpy(dpair_rows, pair_rows.data(), pair_rows.size()*sizeof(int32_t), hipMemcpyHostToDevice), "pair rows upload");
        check(hipMemcpy(dpair_masks, pair_masks.data(), pair_masks.size()*sizeof(uint8_t), hipMemcpyHostToDevice), "pair masks upload");
        check(hipMemcpy(dpair_counts, pair_counts.data(), pair_counts.size()*sizeof(uint32_t), hipMemcpyHostToDevice), "pair counts upload");
        std::vector<float> sinks(kHeads, -1.25f);
        check(hipMemcpy(dsinks, sinks.data(), kHeads*sizeof(float), hipMemcpyHostToDevice), "sinks upload");
    }
    ~buffers() {
        (void)hipFree(dq); (void)hipFree(dkv); (void)hipFree(dindices);
        (void)hipFree(dpair_rows); (void)hipFree(dpair_masks); (void)hipFree(dpair_counts);
        (void)hipFree(dref); (void)hipFree(dcand); (void)hipFree(dsinks);
    }
};

static void launch_ref(buffers &b) {
    attention_single_query<<<dim3(kTokens, kHeads/32u), 1024>>>(
        b.dref, b.dq, b.dkv, b.dindices, b.dsinks);
}

template <uint32_t HPG>
static void launch_pair(buffers &b) {
    attention_pair_query<HPG><<<dim3(kPairs, (kHeads + HPG - 1u)/HPG), 32u*HPG>>>(
        b.dcand, b.dq, b.dkv, b.dpair_rows, b.dpair_masks,
        b.dpair_counts, b.dsinks);
}

template <typename Launch>
static float time_ms(Launch launch, uint32_t iters) {
    hipEvent_t a, b;
    check(hipEventCreate(&a), "event a"); check(hipEventCreate(&b), "event b");
    for (uint32_t i = 0; i < 8u; ++i) launch();
    check(hipDeviceSynchronize(), "warmup");
    check(hipEventRecord(a), "record a");
    for (uint32_t i = 0; i < iters; ++i) launch();
    check(hipEventRecord(b), "record b"); check(hipEventSynchronize(b), "wait b");
    float ms = 0.0f; check(hipEventElapsedTime(&ms, a, b), "elapsed");
    check(hipEventDestroy(a), "destroy a"); check(hipEventDestroy(b), "destroy b");
    return ms/(float)iters;
}

template <uint32_t HPG>
static bool test_pair(buffers &b, const std::vector<float> &ref, float ref_ms) {
    check(hipMemset(b.dcand, 0, b.q.size()*sizeof(float)), "candidate clear");
    launch_pair<HPG>(b); check(hipDeviceSynchronize(), "candidate run");
    std::vector<float> got(ref.size());
    check(hipMemcpy(got.data(), b.dcand, got.size()*sizeof(float), hipMemcpyDeviceToHost), "candidate read");
    uint64_t bit_diff = 0; float max_abs = 0.0f; double sum_sq = 0.0, ref_sq = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::memcmp(&ref[i], &got[i], sizeof(float)) != 0) ++bit_diff;
        const float d = std::fabs(ref[i] - got[i]);
        max_abs = std::max(max_abs, d); sum_sq += (double)d*d;
        ref_sq += (double)ref[i]*ref[i];
    }
    const float ms = time_ms([&] { launch_pair<HPG>(b); }, 200u);
    std::printf("pair_hpg=%u blocks=%u ms=%.6f speedup=%.3fx bit_diff=%llu/%zu max_abs=%.9g rel_rms=%.9g\n",
                HPG, kPairs*((kHeads+HPG-1u)/HPG), ms, ref_ms/ms,
                (unsigned long long)bit_diff, ref.size(), max_abs,
                ref_sq == 0.0 ? 0.0 : std::sqrt(sum_sq/ref_sq));
    return bit_diff == 0;
}

int main() {
    buffers b;
    std::printf("shape tokens=%u heads=%u dim=%u raw=%u comp=%u pair_counts=%u/%u/%u\n",
                kTokens, kHeads, kDim, kRaw, kComp,
                b.pair_counts[0], b.pair_counts[1], b.pair_counts[2]);
    launch_ref(b); check(hipDeviceSynchronize(), "reference run");
    std::vector<float> ref(b.q.size());
    check(hipMemcpy(ref.data(), b.dref, ref.size()*sizeof(float), hipMemcpyDeviceToHost), "reference read");
    const float ref_ms = time_ms([&] { launch_ref(b); }, 200u);
    std::printf("single_query blocks=%u ms=%.6f\n", kTokens*(kHeads/32u), ref_ms);
    bool exact = true;
    exact &= test_pair<8>(b, ref, ref_ms);
    exact &= test_pair<16>(b, ref, ref_ms);
    exact &= test_pair<24>(b, ref, ref_ms);
    exact &= test_pair<32>(b, ref, ref_ms);
    return exact ? 0 : 1;
}
