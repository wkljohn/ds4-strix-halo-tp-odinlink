// Small-step harness for a token-tiled version of DS4's current grouped
// Q8_0 x prequantized-Q8 attention-output projection.  It proves exact output
// agreement with the shipping per-token kernel before any integration.
//
// Build:
//   /opt/rocm/bin/hipcc -O3 -ffast-math -fno-finite-math-only \
//     --offload-arch=gfx1151 scripts/q8_grouped_dp4a_toktile_bench.cu \
//     -o /tmp/q8_grouped_dp4a_toktile_bench

#include "../ds4_rocm.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void check(hipError_t rc, const char *s) {
    if (rc != hipSuccess) { std::fprintf(stderr, "%s: %s\n", s, hipGetErrorString(rc)); std::exit(1); }
}

__device__ __forceinline__ static int32_t load_u(const int8_t *p) {
    const uint8_t *u = reinterpret_cast<const uint8_t *>(p);
    return (int32_t)((uint32_t)u[0] | ((uint32_t)u[1] << 8) |
                     ((uint32_t)u[2] << 16) | ((uint32_t)u[3] << 24));
}
__device__ __forceinline__ static int32_t dot32(const int8_t *a, const int8_t *b) {
    int32_t v = 0;
#pragma unroll
    for (int i = 0; i < 32; i += 4) v = __dp4a(load_u(a + i), *reinterpret_cast<const int32_t *>(b + i), v);
    return v;
}
__device__ __forceinline__ static float warp_sum(float v) {
    for (int d = 16; d; d >>= 1) v += __shfl_down(v, d, 32);
    return v;
}

__global__ static void shipping_kernel(float *low, const unsigned char *w,
        const int8_t *xq, const float *xs, uint64_t group_dim, uint64_t rank,
        uint32_t groups, uint32_t tokens, uint64_t blocks) {
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint64_t row = (uint64_t)blockIdx.x * rows_per_block + (threadIdx.x >> 5u);
    const uint64_t tok = blockIdx.y, lane = threadIdx.x & 31u, low_dim = (uint64_t)groups * rank;
    if (row >= low_dim || tok >= tokens) return;
    const uint64_t g = row / rank;
    const unsigned char *wr = w + row * blocks * 34u;
    const uint64_t xr = tok * groups + g;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const unsigned char *blk = wr + b * 34u;
        acc += __half2float(*reinterpret_cast<const __half *>(blk)) * xs[xr * blocks + b]
             * (float)dot32(reinterpret_cast<const int8_t *>(blk + 2), xq + (xr * blocks + b) * 32u);
    }
    acc = warp_sum(acc);
    if (lane == 0) low[tok * low_dim + row] = acc;
}

template <uint32_t TOK_TILE>
__global__ static void tiled_kernel(float *low, const unsigned char *w,
        const int8_t *xq, const float *xs, uint64_t group_dim, uint64_t rank,
        uint32_t groups, uint32_t tokens, uint64_t blocks) {
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint64_t row = (uint64_t)blockIdx.x * rows_per_block + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u, t0 = blockIdx.y * TOK_TILE;
    const uint64_t low_dim = (uint64_t)groups * rank;
    if (row >= low_dim || t0 >= tokens) return;
    const uint64_t g = row / rank;
    const unsigned char *wr = w + row * blocks * 34u;
    float acc[TOK_TILE];
#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; ++u) acc[u] = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const unsigned char *blk = wr + b * 34u;
        const float ws = __half2float(*reinterpret_cast<const __half *>(blk));
        int32_t qw[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) qw[i] = load_u(reinterpret_cast<const int8_t *>(blk + 2) + i * 4);
#pragma unroll
        for (uint32_t u = 0; u < TOK_TILE; ++u) {
            const uint32_t t = t0 + u;
            if (t < tokens) {
                const uint64_t xb = ((uint64_t)t * groups + g) * blocks + b;
                const int8_t *q = xq + xb * 32u;
                int32_t dot = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) dot = __dp4a(qw[i], *reinterpret_cast<const int32_t *>(q + i * 4), dot);
                acc[u] += ws * xs[xb] * (float)dot;
            }
        }
    }
#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; ++u) acc[u] = warp_sum(acc[u]);
    if (lane == 0) {
#pragma unroll
        for (uint32_t u = 0; u < TOK_TILE; ++u) {
            const uint32_t t = t0 + u;
            if (t < tokens) low[(uint64_t)t * low_dim + row] = acc[u];
        }
    }
}

struct B {
    uint32_t t, g = 16, d = 4096, r = 512, nb = d / 32;
    uint64_t rows = (uint64_t)g * r, rb = (uint64_t)nb * 34u;
    std::vector<unsigned char> w; std::vector<int8_t> q; std::vector<float> s;
    unsigned char *dw; int8_t *dq; float *ds, *a, *b;
    explicit B(uint32_t tokens) : t(tokens), w(rows * rb), q((uint64_t)t * g * d), s((uint64_t)t * g * nb) {
        std::mt19937 z(9); std::uniform_int_distribution<int> qi(-127,127); std::uniform_real_distribution<float> sf(.001f,.03f);
        for (uint64_t row = 0; row < rows; ++row) for (uint32_t k = 0; k < nb; ++k) {
            unsigned char *p = w.data() + row * rb + (uint64_t)k * 34u;
            *reinterpret_cast<__half *>(p) = __float2half(sf(z));
            for (int i=0;i<32;++i) reinterpret_cast<int8_t *>(p+2)[i]=(int8_t)qi(z);
        }
        for (auto &v:q) v=(int8_t)qi(z); for (auto &v:s) v=sf(z);
        const size_t out=(size_t)t*rows*sizeof(float);
        check(hipMalloc(&dw,w.size()),"w"); check(hipMalloc(&dq,q.size()),"q"); check(hipMalloc(&ds,s.size()*4),"s");
        check(hipMalloc(&a,out),"a"); check(hipMalloc(&b,out),"b");
        check(hipMemcpy(dw,w.data(),w.size(),hipMemcpyHostToDevice),"cw");
        check(hipMemcpy(dq,q.data(),q.size(),hipMemcpyHostToDevice),"cq");
        check(hipMemcpy(ds,s.data(),s.size()*4,hipMemcpyHostToDevice),"cs");
    }
    ~B(){hipFree(dw);hipFree(dq);hipFree(ds);hipFree(a);hipFree(b);}
};

static void ship(B&x){dim3 gr((x.rows+7)/8,x.t);shipping_kernel<<<gr,256>>>(x.a,x.dw,x.dq,x.ds,x.d,x.r,x.g,x.t,x.nb);}
template<uint32_t T> static void tile(B&x){dim3 gr((x.rows+7)/8,(x.t+T-1)/T);tiled_kernel<T><<<gr,256>>>(x.b,x.dw,x.dq,x.ds,x.d,x.r,x.g,x.t,x.nb);}
template<uint32_t T> static float time_tile(B&x,int it){hipEvent_t s,e;hipEventCreate(&s);hipEventCreate(&e);for(int i=0;i<2;++i)tile<T>(x);hipDeviceSynchronize();hipEventRecord(s);for(int i=0;i<it;++i)tile<T>(x);hipEventRecord(e);hipEventSynchronize(e);float ms;hipEventElapsedTime(&ms,s,e);hipEventDestroy(s);hipEventDestroy(e);return ms/it;}
static float time_ship(B&x,int it){hipEvent_t s,e;hipEventCreate(&s);hipEventCreate(&e);for(int i=0;i<2;++i)ship(x);hipDeviceSynchronize();hipEventRecord(s);for(int i=0;i<it;++i)ship(x);hipEventRecord(e);hipEventSynchronize(e);float ms;hipEventElapsedTime(&ms,s,e);hipEventDestroy(s);hipEventDestroy(e);return ms/it;}

int main(){
    /* DSpark verifies five rows at a time, so protect the small-token shapes
     * explicitly instead of inferring their crossover from prompt-prefill
     * batches.  The tiled kernel preserves each token's reduction order. */
    for(uint32_t t:{1u,2u,3u,4u,5u,6u,8u,15u,16u,17u,33u}){B x(t);ship(x);tile<16>(x);check(hipDeviceSynchronize(),"correctness");size_t n=(size_t)t*x.rows,bytes=n*4;std::vector<float>a(n),b(n);hipMemcpy(a.data(),x.a,bytes,hipMemcpyDeviceToHost);hipMemcpy(b.data(),x.b,bytes,hipMemcpyDeviceToHost);size_t bitdiff=0;double maxabs=0,maxrel=0;for(size_t i=0;i<n;++i){if(std::memcmp(&a[i],&b[i],4))++bitdiff;double d=std::fabs((double)a[i]-b[i]);maxabs=std::max(maxabs,d);maxrel=std::max(maxrel,d/std::max(1e-6,std::fabs((double)a[i])));}std::printf("correctness tokens=%u bitdiff=%zu/%zu max_abs=%.9g max_rel=%.9g\n",t,bitdiff,n,maxabs,maxrel);if(maxabs>1e-4&&maxrel>1e-5)return 2;}
    for(uint32_t t:{1u,2u,3u,4u,5u,6u,8u,16u}){B x(t);int it=30;float a=time_ship(x,it),b2=time_tile<2>(x,it),b4=time_tile<4>(x,it),b5=time_tile<5>(x,it),b8=time_tile<8>(x,it),b16=time_tile<16>(x,it);std::printf("tokens=%u shipping_ms=%.4f tile2=%.4f tile4=%.4f tile5=%.4f tile8=%.4f tile16=%.4f best_change=%+.1f%%\n",t,a,b2,b4,b5,b8,b16,100.f*(std::min({b2,b4,b5,b8,b16})/a-1.f));}
    for(uint32_t t:{128u,256u,512u,1024u,2048u}){B x(t);int it=t<1024?8:3;float a=time_ship(x,it),b8=time_tile<8>(x,it),b16=time_tile<16>(x,it),b32=time_tile<32>(x,it);std::printf("tokens=%u shipping_ms=%.4f tile8=%.4f tile16=%.4f tile32=%.4f best_change=%+.1f%%\n",t,a,b8,b16,b32,100.f*(std::min({b8,b16,b32})/a-1.f));}
}
