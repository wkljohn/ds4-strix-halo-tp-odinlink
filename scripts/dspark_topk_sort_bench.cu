// Exact gfx1151 oracle for the DSpark verifier's 512-row top-K sort.
// The canonical ctx4096 path has <=1026 compressed rows, so a shared bitmap
// can replace the 45-barrier bitonic network while emitting identical
// ascending int32 indices.  No live path is changed by this benchmark.
//
// Build:
//   /opt/rocm/bin/hipcc -O3 --offload-arch=gfx1151 \
//     scripts/dspark_topk_sort_bench.cu -o /tmp/dspark_topk_sort_bench

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

static constexpr uint32_t kTokens = 5;
static constexpr uint32_t kTopK = 512;
static constexpr uint32_t kBitmapCap = 2048;
static constexpr uint32_t kWords = kBitmapCap/32u;

static void check(hipError_t rc, const char *where) {
    if (rc != hipSuccess) {
        std::fprintf(stderr,"%s: %s\n",where,hipGetErrorString(rc));
        std::exit(1);
    }
}

__global__ static void bitonic_sort(int32_t *dst,const int32_t *src,uint32_t nt) {
    const uint32_t t=blockIdx.x,tid=threadIdx.x;
    if(t>=nt||tid>=kTopK)return;
    __shared__ int32_t rows[kTopK];
    rows[tid]=src[(uint64_t)t*kTopK+tid];__syncthreads();
    for(uint32_t k=2;k<=kTopK;k<<=1u){
        for(uint32_t j=k>>1u;j>0;j>>=1u){
            const uint32_t other=tid^j;
            if(other>tid){const int32_t a=rows[tid],b=rows[other];
                const bool up=(tid&k)==0u;
                if((up&&a>b)||(!up&&a<b)){rows[tid]=b;rows[other]=a;}}
            __syncthreads();
        }
    }
    dst[(uint64_t)t*kTopK+tid]=rows[tid];
}

__global__ static void bitmap_sort(int32_t *dst,const int32_t *src,
                                   uint32_t nt,uint32_t n_comp) {
    const uint32_t t=blockIdx.x,tid=threadIdx.x;
    if(t>=nt||tid>=kTopK)return;
    __shared__ uint32_t bits[kWords];
    for(uint32_t w=tid;w<kWords;w+=blockDim.x)bits[w]=0u;
    __syncthreads();
    const int32_t ci=src[(uint64_t)t*kTopK+tid];
    if(ci>=0&&(uint32_t)ci<n_comp)
        atomicOr(&bits[(uint32_t)ci>>5u],1u<<((uint32_t)ci&31u));
    __syncthreads();

    for(uint32_t c=tid;c<n_comp;c+=blockDim.x){
        const uint32_t word=c>>5u,bit=c&31u;
        const uint32_t mask=1u<<bit;
        if((bits[word]&mask)==0u)continue;
        uint32_t rank=0u;
        for(uint32_t w=0;w<word;++w)rank+=(uint32_t)__popc(bits[w]);
        rank+=(uint32_t)__popc(bits[word]&(mask-1u));
        if(rank<kTopK)dst[(uint64_t)t*kTopK+rank]=(int32_t)c;
    }
}

template<class F> static float time_us(F f) {
    hipEvent_t a,b;check(hipEventCreate(&a),"event a");check(hipEventCreate(&b),"event b");
    for(int i=0;i<20;++i)f();check(hipDeviceSynchronize(),"warm");
    check(hipEventRecord(a),"record a");for(int i=0;i<2000;++i)f();
    check(hipEventRecord(b),"record b");check(hipEventSynchronize(b),"wait");
    float ms=0;check(hipEventElapsedTime(&ms,a,b),"elapsed");
    (void)hipEventDestroy(a);(void)hipEventDestroy(b);return ms*1000.0f/2000.0f;
}

static bool run(uint32_t n_comp) {
    std::mt19937 rng(1151+n_comp);
    std::vector<int32_t> src((uint64_t)kTokens*kTopK);
    std::vector<int32_t> pool(n_comp);std::iota(pool.begin(),pool.end(),0);
    for(uint32_t t=0;t<kTokens;++t){std::shuffle(pool.begin(),pool.end(),rng);
        std::copy_n(pool.begin(),kTopK,src.begin()+(uint64_t)t*kTopK);}
    const size_t bytes=src.size()*sizeof(int32_t);
    int32_t *dsrc=nullptr,*dref=nullptr,*dgot=nullptr;
    check(hipMalloc(&dsrc,bytes),"src alloc");check(hipMalloc(&dref,bytes),"ref alloc");
    check(hipMalloc(&dgot,bytes),"got alloc");
    check(hipMemcpy(dsrc,src.data(),bytes,hipMemcpyHostToDevice),"upload");
    bitonic_sort<<<kTokens,kTopK>>>(dref,dsrc,kTokens);
    bitmap_sort<<<kTokens,kTopK>>>(dgot,dsrc,kTokens,n_comp);
    check(hipDeviceSynchronize(),"run");
    std::vector<int32_t> ref(src.size()),got(src.size());
    check(hipMemcpy(ref.data(),dref,bytes,hipMemcpyDeviceToHost),"ref read");
    check(hipMemcpy(got.data(),dgot,bytes,hipMemcpyDeviceToHost),"got read");
    uint64_t diff=0;for(size_t i=0;i<ref.size();++i)diff+=ref[i]!=got[i];
    const float old_us=time_us([&]{bitonic_sort<<<kTokens,kTopK>>>(dref,dsrc,kTokens);});
    const float new_us=time_us([&]{bitmap_sort<<<kTokens,kTopK>>>(dgot,dsrc,kTokens,n_comp);});
    std::printf("n_comp=%u bitonic_us=%.4f bitmap_us=%.4f speedup=%.3fx diff=%llu/%zu\n",
        n_comp,old_us,new_us,old_us/new_us,(unsigned long long)diff,ref.size());
    (void)hipFree(dsrc);(void)hipFree(dref);(void)hipFree(dgot);
    return diff==0;
}

int main(){bool ok=true;ok&=run(513);ok&=run(768);ok&=run(1026);ok&=run(2048);return ok?0:1;}
