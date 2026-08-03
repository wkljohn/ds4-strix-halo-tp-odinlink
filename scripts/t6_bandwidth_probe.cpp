/* T6: measure real achievable read bandwidth on this GPU.
 * The entire roofline is bandwidth / bytes-per-token, and the bandwidth figure
 * had been inherited from an earlier note rather than measured. Ground it. */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <ctime>
static double now_s(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec+1e-9*(double)t.tv_nsec;}

/* The sink MUST be data-dependent or the compiler deletes the whole loop.
 * The first version compared threadIdx.x against 0xffffffff, which is provably
 * false for a 256-thread block, so every load was dead code and the probe
 * reported 996 TB/s. `tripwire` comes from the host at runtime so no such proof
 * is available. */
__global__ void read_kernel(const float4 *__restrict__ src, size_t n4, float *sink, float tripwire){
    size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x*blockDim.x;
    float4 acc = make_float4(0,0,0,0);
    for (; i < n4; i += stride) {
        float4 v = src[i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    float s = acc.x+acc.y+acc.z+acc.w;
    if (s == tripwire) *sink = s;
}
int main(void){
    hipSetDevice(0);
    const size_t bytes = 8ull<<30;          /* 8 GiB - well past any cache */
    void *buf=NULL;
    if (hipMalloc(&buf, bytes) != hipSuccess) { printf("  alloc failed\n"); return 1; }
    hipMemset(buf, 1, bytes);
    float *sink=NULL; hipMalloc(&sink,4);
    const size_t n4 = bytes/sizeof(float4);
    volatile float host_tw = -1.0e30f;
    const float tripwire = host_tw;
    int blocks = 0, threads = 256;
    hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
    blocks = p.multiProcessorCount * 8;
    printf("  device: %s  CUs=%d\n", p.name, p.multiProcessorCount);
    /* warm */
    hipLaunchKernelGGL(read_kernel,dim3(blocks),dim3(threads),0,0,(const float4*)buf,n4,sink,tripwire);
    hipDeviceSynchronize();
    double best=0;
    for (int it=0; it<5; it++) {
        double t0=now_s();
        hipLaunchKernelGGL(read_kernel,dim3(blocks),dim3(threads),0,0,(const float4*)buf,n4,sink,tripwire);
        hipDeviceSynchronize();
        double el=now_s()-t0;
        double gibs = (double)bytes/el/1073741824.0;
        if (gibs>best) best=gibs;
        printf("    iter %d: %.3f s  -> %.1f GiB/s (%.1f GB/s)\n", it, el, gibs, gibs*1.073741824);
    }
    printf("  BEST: %.1f GiB/s  =  %.1f GB/s\n", best, best*1.073741824);
    printf("  (theoretical peak LPDDR5X-8000 x 256-bit = 256 GB/s = 238.4 GiB/s)\n");
    printf("  efficiency vs peak: %.0f%%\n", 100.0*best*1.073741824/256.0);
    return 0;
}
