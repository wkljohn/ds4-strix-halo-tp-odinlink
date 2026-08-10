// Small oracle for DSpark's ROCm non-causal raw-ring attention kernel.
// Build: hipcc -O3 --offload-arch=gfx1151 -o /tmp/dspark-attn scripts/dspark_noncausal_attention_test.cu

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

__global__ static void baseline_kernel(float *out, const float *sinks, const float *q,
        const float *kv, unsigned nt, unsigned nr, unsigned cap,
        unsigned start, unsigned nh, unsigned hd) {
    unsigned t = blockIdx.x, h = blockIdx.y;
    extern __shared__ float score[];
    const float *qh = q + ((size_t)t * nh + h) * hd;
    for (unsigned r = threadIdx.x; r < nr; r += blockDim.x) {
        const float *kr = kv + (size_t)((start + r) % cap) * hd;
        float dot = 0;
        for (unsigned d = 0; d < hd; d++) dot += qh[d] * kr[d];
        score[r] = dot * rsqrtf((float)hd);
    }
    __syncthreads();
    __shared__ float part[256], mx, den;
    float lm = sinks[h];
    for (unsigned r = threadIdx.x; r < nr; r += blockDim.x) lm = fmaxf(lm, score[r]);
    part[threadIdx.x] = lm; __syncthreads();
    for (unsigned s = blockDim.x / 2; s; s /= 2) {
        if (threadIdx.x < s) part[threadIdx.x] = fmaxf(part[threadIdx.x], part[threadIdx.x + s]);
        __syncthreads();
    }
    if (!threadIdx.x) mx = part[0]; __syncthreads();
    float ld = 0;
    for (unsigned r = threadIdx.x; r < nr; r += blockDim.x) {
        score[r] = expf(score[r] - mx); ld += score[r];
    }
    part[threadIdx.x] = ld; __syncthreads();
    for (unsigned s = blockDim.x / 2; s; s /= 2) {
        if (threadIdx.x < s) part[threadIdx.x] += part[threadIdx.x + s];
        __syncthreads();
    }
    if (!threadIdx.x) den = part[0] + expf(sinks[h] - mx); __syncthreads();
    float *oh = out + ((size_t)t * nh + h) * hd;
    for (unsigned d = threadIdx.x; d < hd; d += blockDim.x) {
        float v = 0;
        for (unsigned r = 0; r < nr; r++) v += kv[(size_t)((start + r) % cap) * hd + d] * score[r];
        oh[d] = v / den;
    }
}

__device__ __forceinline__ float wave_sum(float v) {
    for (unsigned delta = warpSize / 2; delta; delta >>= 1) {
        v += __shfl_down(v, delta, warpSize);
    }
    return v;
}

__device__ __forceinline__ float wave_max(float v) {
    for (unsigned delta = warpSize / 2; delta; delta >>= 1) {
        v = fmaxf(v, __shfl_down(v, delta, warpSize));
    }
    return v;
}

/* Candidate copied from ATOM's architectural choice, not its implementation:
 * make the small non-causal DSpark attention a cooperative wave operation.
 * One wave computes one 192-wide Q.K score instead of one lane serializing all
 * 192 FMAs. The output reduction remains one lane per head dimension. */
__global__ static void wave_kernel(float *out, const float *sinks, const float *q,
        const float *kv, unsigned nt, unsigned nr, unsigned cap,
        unsigned start, unsigned nh, unsigned hd) {
    unsigned t = blockIdx.x, h = blockIdx.y;
    if (t >= nt || h >= nh) return;
    extern __shared__ float score[];
    __shared__ float wave_part[8], mx, den;
    const unsigned lane = threadIdx.x & (warpSize - 1);
    const unsigned wave = threadIdx.x / warpSize;
    const float *qh = q + ((size_t)t * nh + h) * hd;
    const float scale = rsqrtf((float)hd);

    for (unsigned r = wave; r < nr; r += 8) {
        const float *kr = kv + (size_t)((start + r) % cap) * hd;
        float dot = 0.0f;
        for (unsigned d = lane; d < hd; d += warpSize) dot += qh[d] * kr[d];
        dot = wave_sum(dot);
        if (lane == 0) score[r] = dot * scale;
    }
    __syncthreads();

    float local = sinks[h];
    for (unsigned r = threadIdx.x; r < nr; r += blockDim.x) {
        local = fmaxf(local, score[r]);
    }
    local = wave_max(local);
    if (lane == 0) wave_part[wave] = local;
    __syncthreads();
    if (wave == 0) {
        float v = lane < 8 ? wave_part[lane] : -INFINITY;
        v = wave_max(v);
        if (lane == 0) mx = v;
    }
    __syncthreads();

    float local_den = 0.0f;
    for (unsigned r = threadIdx.x; r < nr; r += blockDim.x) {
        score[r] = expf(score[r] - mx);
        local_den += score[r];
    }
    local_den = wave_sum(local_den);
    if (lane == 0) wave_part[wave] = local_den;
    __syncthreads();
    if (wave == 0) {
        float v = lane < 8 ? wave_part[lane] : 0.0f;
        v = wave_sum(v);
        if (lane == 0) den = v + expf(sinks[h] - mx);
    }
    __syncthreads();

    float *oh = out + ((size_t)t * nh + h) * hd;
    for (unsigned d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned r = 0; r < nr; r++) {
            acc += kv[(size_t)((start + r) % cap) * hd + d] * score[r];
        }
        oh[d] = acc / den;
    }
}

#define HIP(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s\n", hipGetErrorString(e)); return 1; } } while (0)

static float elapsed_ms(bool wave, float *out, const float *sinks,
        const float *q, const float *kv, unsigned nt, unsigned nr,
        unsigned cap, unsigned start, unsigned nh, unsigned hd, int iters) {
    hipEvent_t begin, end;
    hipEventCreate(&begin); hipEventCreate(&end);
    hipEventRecord(begin);
    for (int i = 0; i < iters; i++) {
        if (wave) wave_kernel<<<dim3(nt,nh),256,nr*4>>>(out,sinks,q,kv,nt,nr,cap,start,nh,hd);
        else baseline_kernel<<<dim3(nt,nh),256,nr*4>>>(out,sinks,q,kv,nt,nr,cap,start,nh,hd);
    }
    hipEventRecord(end); hipEventSynchronize(end);
    float ms = 0.0f; hipEventElapsedTime(&ms, begin, end);
    hipEventDestroy(end); hipEventDestroy(begin);
    return ms / iters;
}

int main() {
    constexpr unsigned nt=5, cap=128, start=113, nh=64, hd=192;
    std::vector<float> q((size_t)nt*nh*hd), kv((size_t)cap*hd), sink(nh), got(q.size()), ref(q.size()), base(q.size());
    unsigned state=7; auto rnd=[&](){ state=state*1664525u+1013904223u; return ((int)(state%2001)-1000)/1000.f; };
    for (float &v:q) v=rnd(); for (float &v:kv) v=rnd(); for (float &v:sink) v=rnd();
    float *dq,*dk,*ds,*dout; HIP(hipMalloc(&dq,q.size()*4)); HIP(hipMalloc(&dk,kv.size()*4)); HIP(hipMalloc(&ds,sink.size()*4)); HIP(hipMalloc(&dout,got.size()*4));
    HIP(hipMemcpy(dq,q.data(),q.size()*4,hipMemcpyHostToDevice)); HIP(hipMemcpy(dk,kv.data(),kv.size()*4,hipMemcpyHostToDevice)); HIP(hipMemcpy(ds,sink.data(),sink.size()*4,hipMemcpyHostToDevice));
    int failed = 0;
    for (unsigned nr : {6u, 32u, 64u, 128u}) {
        for (unsigned t=0;t<nt;t++) for(unsigned h=0;h<nh;h++) {
            std::vector<double> sc(nr); double mx=sink[h], den;
            for(unsigned r=0;r<nr;r++){ double d=0; for(unsigned x=0;x<hd;x++) d+=q[((size_t)t*nh+h)*hd+x]*kv[(size_t)((start+r)%cap)*hd+x]; sc[r]=d/std::sqrt((double)hd); mx=std::max(mx,sc[r]); }
            den=std::exp((double)sink[h]-mx); for(double &v:sc){v=std::exp(v-mx);den+=v;}
            for(unsigned d=0;d<hd;d++){double v=0;for(unsigned r=0;r<nr;r++)v+=kv[(size_t)((start+r)%cap)*hd+d]*sc[r];ref[((size_t)t*nh+h)*hd+d]=(float)(v/den);}
        }
        baseline_kernel<<<dim3(nt,nh),256,nr*4>>>(dout,ds,dq,dk,nt,nr,cap,start,nh,hd);
        HIP(hipMemcpy(base.data(),dout,base.size()*4,hipMemcpyDeviceToHost));
        wave_kernel<<<dim3(nt,nh),256,nr*4>>>(dout,ds,dq,dk,nt,nr,cap,start,nh,hd);
        HIP(hipGetLastError()); HIP(hipMemcpy(got.data(),dout,got.size()*4,hipMemcpyDeviceToHost));
        float ref_err=0, ab_err=0; for(size_t i=0;i<got.size();i++){ref_err=std::max(ref_err,std::fabs(got[i]-ref[i]));ab_err=std::max(ab_err,std::fabs(got[i]-base[i]));}
        const float base_ms=elapsed_ms(false,dout,ds,dq,dk,nt,nr,cap,start,nh,hd,200);
        const float wave_ms=elapsed_ms(true,dout,ds,dq,dk,nt,nr,cap,start,nh,hd,200);
        std::printf("rows=%3u ref_max=%g ab_max=%g baseline=%.4fms wave=%.4fms speedup=%.2fx\n",nr,ref_err,ab_err,base_ms,wave_ms,base_ms/wave_ms);
        if (ref_err > 3e-5f || ab_err > 3e-5f) failed = 1;
    }
    hipFree(dout); hipFree(ds); hipFree(dk); hipFree(dq); return failed ? 2 : 0;
}
