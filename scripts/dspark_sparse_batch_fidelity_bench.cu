// Production-shape DSpark sparse-attention fidelity oracle.
//
// Separates two differences between ordinary one-row ROCm decode and the
// five-row verifier path:
//   1. preserving the indexer's top-k order vs sorting indices by KV row;
//   2. decode's two-pass softmax vs the verifier's online recurrence.
//
// This intentionally does not modify the live inference path.  A candidate
// must first be bit-exact against `two_pass_original` here.
//
// Build:
//   /opt/rocm/bin/hipcc -O3 -ffast-math -fno-finite-math-only \
//     --offload-arch=gfx1151 scripts/dspark_sparse_batch_fidelity_bench.cu \
//     -o /tmp/dspark_sparse_batch_fidelity_bench

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

static constexpr uint32_t kTokens = 5;
static constexpr uint32_t kHeads = 32; // one TP rank
static constexpr uint32_t kDim = 512;
static constexpr uint32_t kRawCap = 256;
static constexpr uint32_t kRaw = 128;
static constexpr uint32_t kCompRows = 768;
static constexpr uint32_t kTopK = 512;
static constexpr uint32_t kPos0 = 2052;
static constexpr uint32_t kRatio = 4;
static constexpr uint32_t kRows = kRaw + kTopK;
static constexpr uint32_t kStage = 8;

static void check(hipError_t rc, const char *where) {
    if (rc != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, hipGetErrorString(rc));
        std::exit(1);
    }
}

static __device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (uint32_t d = 16; d != 0; d >>= 1u) v += __shfl_down(v, d, 32);
    return v;
}

static __device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (uint32_t d = 16; d != 0; d >>= 1u) v = fmaxf(v, __shfl_down(v, d, 32));
    return v;
}

static __device__ __forceinline__ float block_sum(float v) {
    __shared__ float sh[32];
    const uint32_t tid = threadIdx.x, lane = tid & 31u, wid = tid >> 5u;
    const uint32_t nw = (blockDim.x + 31u) >> 5u;
    v = warp_sum(v);
    if (lane == 0u) sh[wid] = v;
    __syncthreads();
    v = tid < nw ? sh[lane] : 0.0f;
    if (wid == 0u) v = warp_sum(v);
    if (tid == 0u) sh[0] = v;
    __syncthreads();
    return sh[0];
}

static __device__ __forceinline__ float block_max(float v) {
    __shared__ float sh[32];
    const uint32_t tid = threadIdx.x, lane = tid & 31u, wid = tid >> 5u;
    const uint32_t nw = (blockDim.x + 31u) >> 5u;
    v = warp_max(v);
    if (lane == 0u) sh[wid] = v;
    __syncthreads();
    v = tid < nw ? sh[lane] : -3.4e38f;
    if (wid == 0u) v = warp_max(v);
    if (tid == 0u) sh[0] = v;
    __syncthreads();
    return sh[0];
}

// This matches attention_dot_f32_vec4_oldhip: one thread computes the whole
// 512-wide dot using four independent accumulators.
static __device__ __forceinline__ float dot_vec4(const float *a, const float *b) {
    const float4 *a4 = reinterpret_cast<const float4 *>(a);
    const float4 *b4 = reinterpret_cast<const float4 *>(b);
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (uint32_t i = 0; i < kDim/4u; ++i) {
        const float4 av = a4[i], bv = b4[i];
        s0 += av.x*bv.x; s1 += av.y*bv.y;
        s2 += av.z*bv.z; s3 += av.w*bv.w;
    }
    return (s0 + s1) + (s2 + s3);
}

// Five rows in one grid, but each block is exactly the shipping one-row
// old-HIP arithmetic.  It is the fidelity reference, not an optimization.
__global__ static void attention_two_pass(
        float *out, const float *q, const float *raw_kv, const float *comp_kv,
        const int32_t *topk, const float *sinks) {
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    extern __shared__ float scores[];
    __shared__ uint32_t comp_rows[kTopK];
    const float *qh = q + ((uint64_t)t*kHeads + h)*kDim;
    const float scale = rsqrtf((float)kDim);

    for (uint32_t c = tid; c < kTopK; c += blockDim.x) {
        comp_rows[c] = (uint32_t)topk[(uint64_t)t*kTopK + c];
    }
    __syncthreads();
    float local_max = sinks[h];
    for (uint32_t r = tid; r < kRaw; r += blockDim.x) {
        const float s = dot_vec4(qh, raw_kv + (uint64_t)(t + r)*kDim)*scale;
        scores[r] = s;
        local_max = fmaxf(local_max, s);
    }
    for (uint32_t c = tid; c < kTopK; c += blockDim.x) {
        const float s = dot_vec4(qh, comp_kv + (uint64_t)comp_rows[c]*kDim)*scale;
        scores[kRaw + c] = s;
        local_max = fmaxf(local_max, s);
    }
    const float max_score = block_max(local_max);
    float local_sum = 0.0f;
    for (uint32_t r = tid; r < kRows; r += blockDim.x) {
        const float w = expf(scores[r] - max_score);
        scores[r] = w;
        local_sum += w;
    }
    if (tid == 0u) local_sum += expf(sinks[h] - max_score);
    const float inv = 1.0f/block_sum(local_sum);
    float *dst = out + ((uint64_t)t*kHeads + h)*kDim;
    for (uint32_t d = tid; d < kDim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < kRaw; ++r)
            acc += scores[r]*raw_kv[(uint64_t)(t + r)*kDim + d];
        for (uint32_t c = 0; c < kTopK; ++c)
            acc += scores[kRaw + c]*comp_kv[(uint64_t)comp_rows[c]*kDim + d];
        dst[d] = acc*inv;
    }
}

static __device__ __forceinline__ float dot4(float4 a, float4 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

// Shipping verifier arithmetic: one token x 32 heads per 1024-thread block,
// staged KV rows and online softmax.
template <uint32_t HeadsPerGroup>
__launch_bounds__(32u*HeadsPerGroup, 1)
__global__ static void attention_online(
        float *out, const float *q, const float *raw_kv, const float *comp_kv,
        const int32_t *topk, const float *sinks) {
    const uint32_t t = blockIdx.x;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t h = blockIdx.y*HeadsPerGroup + (threadIdx.x >> 5u);
    const bool valid_head = h < kHeads;
    __shared__ float4 tile[kStage*(kDim/4u)];
    const float4 *q4 = valid_head ? reinterpret_cast<const float4 *>(
        q + ((uint64_t)t*kHeads + h)*kDim) : nullptr;
    const float4 zero = make_float4(0,0,0,0);
    const float4 qv[4] = {
        valid_head ? q4[lane] : zero,
        valid_head ? q4[lane+32u] : zero,
        valid_head ? q4[lane+64u] : zero,
        valid_head ? q4[lane+96u] : zero};
    float m = valid_head ? sinks[h] : -INFINITY;
    float sum = valid_head ? 1.0f : 0.0f;
    float4 acc[4] = {};

    for (uint32_t row0 = 0; row0 < kRows; row0 += kStage) {
        for (uint32_t off = threadIdx.x; off < kStage*(kDim/4u); off += blockDim.x) {
            const uint32_t rr = off/(kDim/4u), c4 = off%(kDim/4u);
            const uint32_t sr = row0 + rr;
            const float4 *src;
            if (sr < kRaw) {
                src = reinterpret_cast<const float4 *>(raw_kv + (uint64_t)(t + sr)*kDim);
            } else {
                const uint32_t row = (uint32_t)topk[(uint64_t)t*kTopK + sr-kRaw];
                src = reinterpret_cast<const float4 *>(comp_kv + (uint64_t)row*kDim);
            }
            tile[off] = src[c4];
        }
        __syncthreads();
#pragma unroll
        for (uint32_t rr = 0; valid_head && rr < kStage; ++rr) {
            const float4 *v = tile + rr*(kDim/4u);
            const float4 vv[4] = {v[lane], v[lane+32u], v[lane+64u], v[lane+96u]};
            float score = dot4(qv[0],vv[0]) + dot4(qv[1],vv[1]) +
                          dot4(qv[2],vv[2]) + dot4(qv[3],vv[3]);
            score = __shfl(warp_sum(score)*rsqrtf((float)kDim), 0, 32);
            const float nm = fmaxf(m, score);
            const float os = expf(m-nm), ns = expf(score-nm);
            sum = sum*os + ns;
#pragma unroll
            for (uint32_t i = 0; i < 4u; ++i) {
                acc[i].x = acc[i].x*os + vv[i].x*ns;
                acc[i].y = acc[i].y*os + vv[i].y*ns;
                acc[i].z = acc[i].z*os + vv[i].z*ns;
                acc[i].w = acc[i].w*os + vv[i].w*ns;
            }
            m = nm;
        }
        __syncthreads();
    }
    if (!valid_head) return;
    float4 *dst = reinterpret_cast<float4 *>(out + ((uint64_t)t*kHeads+h)*kDim);
    const float inv = 1.0f/sum;
#pragma unroll
    for (uint32_t i = 0; i < 4u; ++i) {
        acc[i].x*=inv; acc[i].y*=inv; acc[i].z*=inv; acc[i].w*=inv;
        dst[lane+32u*i] = acc[i];
    }
}

struct buffers {
    std::vector<float> q, raw, comp, sinks;
    std::vector<int32_t> original, sorted;
    float *dq=nullptr, *draw=nullptr, *dcomp=nullptr, *dsinks=nullptr;
    float *dref=nullptr, *dout=nullptr;
    int32_t *doriginal=nullptr, *dsorted=nullptr;
    buffers() : q((uint64_t)kTokens*kHeads*kDim),
                raw((uint64_t)(kRaw+kTokens-1u)*kDim),
                comp((uint64_t)kCompRows*kDim), sinks(kHeads),
                original((uint64_t)kTokens*kTopK), sorted(original.size()) {
        std::mt19937 rng(1151);
        std::uniform_real_distribution<float> val(-0.125f,0.125f);
        for (float &v:q) v=val(rng); for(float &v:raw) v=val(rng);
        for (float &v:comp) v=val(rng); for(float &v:sinks) v=val(rng)-1.0f;
        std::vector<int32_t> base(kCompRows);
        std::iota(base.begin(),base.end(),0);
        for (uint32_t t=0;t<kTokens;++t) {
            std::shuffle(base.begin(),base.end(),rng);
            std::copy_n(base.begin(),kTopK,original.begin()+(uint64_t)t*kTopK);
            std::copy_n(original.begin()+(uint64_t)t*kTopK,kTopK,
                        sorted.begin()+(uint64_t)t*kTopK);
            std::sort(sorted.begin()+(uint64_t)t*kTopK,
                      sorted.begin()+(uint64_t)(t+1u)*kTopK);
        }
#define ALLOC_COPY(dev_, host_) do { \
    check(hipMalloc(&(dev_), (host_).size()*sizeof((host_)[0])), #dev_ " alloc"); \
    check(hipMemcpy((dev_), (host_).data(), (host_).size()*sizeof((host_)[0]), \
                    hipMemcpyHostToDevice), #dev_ " copy"); \
} while(0)
        ALLOC_COPY(dq,q); ALLOC_COPY(draw,raw); ALLOC_COPY(dcomp,comp);
        ALLOC_COPY(dsinks,sinks); ALLOC_COPY(doriginal,original); ALLOC_COPY(dsorted,sorted);
#undef ALLOC_COPY
        const size_t bytes=q.size()*sizeof(float);
        check(hipMalloc(&dref,bytes),"ref alloc"); check(hipMalloc(&dout,bytes),"out alloc");
    }
    ~buffers(){hipFree(dq);hipFree(draw);hipFree(dcomp);hipFree(dsinks);
        hipFree(doriginal);hipFree(dsorted);hipFree(dref);hipFree(dout);}
};

template<class F> static float time_ms(F f) {
    hipEvent_t a,b; check(hipEventCreate(&a),"event a"); check(hipEventCreate(&b),"event b");
    for(int i=0;i<8;++i)f(); check(hipDeviceSynchronize(),"warmup");
    check(hipEventRecord(a),"record a"); for(int i=0;i<100;++i)f();
    check(hipEventRecord(b),"record b"); check(hipEventSynchronize(b),"wait b");
    float ms=0; check(hipEventElapsedTime(&ms,a,b),"elapsed");
    hipEventDestroy(a);hipEventDestroy(b);return ms/100.0f;
}

static void compare(const char *name, buffers &b, const std::vector<float>& ref,
                    void (*launch)(buffers&)) {
    launch(b); check(hipDeviceSynchronize(),name);
    std::vector<float> got(ref.size());
    check(hipMemcpy(got.data(),b.dout,got.size()*sizeof(float),hipMemcpyDeviceToHost),"read");
    uint64_t bits=0; float max_abs=0; double dsq=0,rsq=0;
    for(size_t i=0;i<got.size();++i){if(std::memcmp(&got[i],&ref[i],4))++bits;
        const float d=got[i]-ref[i];max_abs=std::max(max_abs,std::fabs(d));
        dsq+=(double)d*d;rsq+=(double)ref[i]*ref[i];}
    const float ms=time_ms([&]{launch(b);});
    std::printf("%-24s ms=%.6f bit_diff=%llu/%zu max_abs=%.9g rel_rms=%.9g\n",
        name,ms,(unsigned long long)bits,got.size(),max_abs,std::sqrt(dsq/rsq));
}

static void two_original_ref(buffers&b){attention_two_pass<<<dim3(kTokens,kHeads),256,kRows*sizeof(float)>>>(b.dref,b.dq,b.draw,b.dcomp,b.doriginal,b.dsinks);}
static void two_original(buffers&b){attention_two_pass<<<dim3(kTokens,kHeads),256,kRows*sizeof(float)>>>(b.dout,b.dq,b.draw,b.dcomp,b.doriginal,b.dsinks);}
static void two_sorted(buffers&b){attention_two_pass<<<dim3(kTokens,kHeads),256,kRows*sizeof(float)>>>(b.dout,b.dq,b.draw,b.dcomp,b.dsorted,b.dsinks);}
template <uint32_t HPG> static void online_launch(float *out, buffers&b, const int32_t *topk) {
    attention_online<HPG><<<dim3(kTokens,(kHeads+HPG-1u)/HPG),32u*HPG>>>(
        out,b.dq,b.draw,b.dcomp,topk,b.dsinks);
}
static void online_original(buffers&b){online_launch<32>(b.dout,b,b.doriginal);}
static void online_sorted(buffers&b){online_launch<32>(b.dout,b,b.dsorted);}
static void online_sorted_ref(buffers&b){online_launch<32>(b.dref,b,b.dsorted);}
static void online_sorted_hpg4(buffers&b){online_launch<4>(b.dout,b,b.dsorted);}
static void online_sorted_hpg8(buffers&b){online_launch<8>(b.dout,b,b.dsorted);}
static void online_sorted_hpg16(buffers&b){online_launch<16>(b.dout,b,b.dsorted);}
static void online_sorted_hpg32(buffers&b){online_launch<32>(b.dout,b,b.dsorted);}

int main(){
    buffers b; two_original_ref(b);check(hipDeviceSynchronize(),"reference");
    std::vector<float> ref(b.q.size());
    check(hipMemcpy(ref.data(),b.dref,ref.size()*sizeof(float),hipMemcpyDeviceToHost),"ref read");
    std::printf("shape tokens=%u rank_heads=%u dim=%u raw=%u topk=%u\n",kTokens,kHeads,kDim,kRaw,kTopK);
    compare("two_pass_original",b,ref,two_original);
    compare("two_pass_sorted",b,ref,two_sorted);
    compare("online_original",b,ref,online_original);
    compare("online_sorted_shipping",b,ref,online_sorted);
    online_sorted_ref(b);check(hipDeviceSynchronize(),"online reference");
    std::vector<float> online_ref(b.q.size());
    check(hipMemcpy(online_ref.data(),b.dref,online_ref.size()*sizeof(float),
                    hipMemcpyDeviceToHost),"online ref read");
    std::printf("shipping-arithmetic occupancy sweep (HPG=32 reference)\n");
    compare("online_sorted_hpg4",b,online_ref,online_sorted_hpg4);
    compare("online_sorted_hpg8",b,online_ref,online_sorted_hpg8);
    compare("online_sorted_hpg16",b,online_ref,online_sorted_hpg16);
    compare("online_sorted_hpg32",b,online_ref,online_sorted_hpg32);
    return 0;
}
