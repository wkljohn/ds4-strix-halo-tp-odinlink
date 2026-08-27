#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

#include "../rocm/ds4_rocm_glm5_conv_ref.cuh"

static void ck(hipError_t e, const char *s) {
    if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", s, hipGetErrorString(e)); std::exit(1); }
}

int main() {
    constexpr uint32_t T = 5, C = 17;
    std::vector<float> x(T*C), w(C*4), hist(C*3), ref_hist, ref(T*C), got(T*C);
    for (size_t i=0; i<x.size(); ++i) x[i] = 0.03f * float(int(i%9)-4);
    for (size_t i=0; i<w.size(); ++i) w[i] = 0.02f * float(int(i%7)-3);
    for (size_t i=0; i<hist.size(); ++i) hist[i] = 0.01f * float(int(i%5)-2);
    ref_hist = hist;
    for (uint32_t t=0; t<T; ++t) for (uint32_t c=0; c<C; ++c) {
        float y = ref_hist[c*3+0]*w[c*4+0] + ref_hist[c*3+1]*w[c*4+1] +
                  ref_hist[c*3+2]*w[c*4+2] + x[t*C+c]*w[c*4+3];
        ref[t*C+c] = y/(1.0f+std::exp(-y));
        ref_hist[c*3+0]=ref_hist[c*3+1]; ref_hist[c*3+1]=ref_hist[c*3+2]; ref_hist[c*3+2]=x[t*C+c];
    }
    float *dx,*dw,*dh,*dy;
    ck(hipMalloc(&dx,x.size()*4),"x alloc"); ck(hipMalloc(&dw,w.size()*4),"w alloc");
    ck(hipMalloc(&dh,hist.size()*4),"h alloc"); ck(hipMalloc(&dy,got.size()*4),"y alloc");
    ck(hipMemcpy(dx,x.data(),x.size()*4,hipMemcpyHostToDevice),"x copy");
    ck(hipMemcpy(dw,w.data(),w.size()*4,hipMemcpyHostToDevice),"w copy");
    ck(hipMemcpy(dh,hist.data(),hist.size()*4,hipMemcpyHostToDevice),"h copy");
    hipLaunchKernelGGL(ds4_glm5_causal_conv4_ref_kernel, dim3((C+63)/64), dim3(64), 0, 0, dx,dw,dh,dy,T,C);
    ck(hipGetLastError(),"conv launch"); ck(hipDeviceSynchronize(),"conv sync");
    ck(hipMemcpy(got.data(),dy,got.size()*4,hipMemcpyDeviceToHost),"y copy");
    ck(hipMemcpy(hist.data(),dh,hist.size()*4,hipMemcpyDeviceToHost),"h copy back");
    float e=0; for(size_t i=0;i<got.size();++i)e=std::fmax(e,std::fabs(got[i]-ref[i]));
    for(size_t i=0;i<hist.size();++i)e=std::fmax(e,std::fabs(hist[i]-ref_hist[i]));
    ck(hipFree(dx),"x free"); ck(hipFree(dw),"w free");
    ck(hipFree(dh),"h free"); ck(hipFree(dy),"y free");
    if(e>2e-6f){std::fprintf(stderr,"FAIL GLM5 conv max_err=%.9g\n",e);return 1;}
    std::printf("PASS GLM5 causal conv4 ROCm reference max_err=%.9g\n",e); return 0;
}
