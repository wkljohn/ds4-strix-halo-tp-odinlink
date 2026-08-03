/* T5: validate the FIX before touching ds4.
 *
 * T4 proved hipStreamWriteValue64 never lands on the NULL stream (stream 0) on
 * gfx1151/ROCm 7.2, but works instantly on a stream from hipStreamCreate().
 * ds4 puts the gate on stream 0 and launches every compute kernel there too.
 *
 * Proposed fix: put the gate write/wait on a dedicated CREATED stream. That is
 * only correct if ordering against the null-stream compute kernels still holds.
 * HIP's legacy null stream implicitly synchronises with "blocking" streams
 * (which hipStreamCreate produces), so it SHOULD hold - verify, don't assume.
 *
 * E1: kernel on null stream, then gate on created stream -> does arrival land?
 * E2: does a kernel launched on the null stream AFTER the gate's wait actually
 *     wait for the gate to be released? (if not, ordering is broken and the
 *     fix is unsafe - the combine kernel could run before the exchange)
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <ctime>

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec + 1e-9*(double)t.tv_nsec; }

__global__ void spin_kernel(unsigned long long iters, float *sink){
    float a=1.0f; for(unsigned long long i=0;i<iters;i++) a=a*1.0000001f+1e-7f;
    if(threadIdx.x==0xffffffffu) *sink=a;
}
__global__ void stamp_kernel(unsigned long long *out, unsigned long long v){
    if(threadIdx.x==0) *out = v;
}

int main(void){
    hipSetDevice(0);
    hipStream_t gate; hipStreamCreate(&gate);

    void *slab=NULL; hipMalloc(&slab,4096); hipMemset(slab,0,4096);
    volatile uint64_t *gpu_flag=(volatile uint64_t*)slab;
    void *sig=NULL; hipHostMalloc(&sig,64,hipHostMallocMapped);
    volatile uint64_t *cpu_flag=(volatile uint64_t*)sig;
    __atomic_store_n(cpu_flag,0,__ATOMIC_RELEASE);
    float *sink=NULL; hipMalloc(&sink,sizeof(float));
    unsigned long long *stamp=NULL; hipHostMalloc((void**)&stamp,8,hipHostMallocMapped);
    *stamp=0;

    /* --- E1: compute on NULL stream, gate on CREATED stream --- */
    hipLaunchKernelGGL(spin_kernel, dim3(256),dim3(256),0,0, 2000000ull, sink);
    hipStreamWriteValue64(gate,(void*)gpu_flag,(int64_t)1,0);
    hipStreamWaitValue64 (gate,(void*)cpu_flag,(int64_t)1,hipStreamWaitValueGte,~0ULL);

    double t0=now_s(); int seen=0;
    while(now_s()-t0<12.0){ if(__atomic_load_n(gpu_flag,__ATOMIC_ACQUIRE)>=1ull){seen=1;break;} }
    printf("  E1 gate on created stream (compute on null) : %s (%.2f s)\n",
           seen?"ARRIVAL SEEN":"*** NEVER SEEN ***", now_s()-t0);
    if(!seen){ printf("  fix does NOT work; stopping\n"); return 2; }

    /* --- E2: ordering. Queue a stamp kernel on the NULL stream now. It must
       NOT run until we release the gate. --- */
    hipLaunchKernelGGL(stamp_kernel, dim3(1),dim3(1),0,0, stamp, 42ull);
    struct timespec d={0,400000000}; nanosleep(&d,NULL);
    unsigned long long before=__atomic_load_n(stamp,__ATOMIC_ACQUIRE);
    printf("  E2 stamp BEFORE release (want 0)            : %llu %s\n",
           before, before==0?"OK - ordering holds":"*** RAN EARLY - ordering BROKEN ***");

    __atomic_store_n(cpu_flag,1ull,__ATOMIC_RELEASE);
    t0=now_s(); int ran=0;
    while(now_s()-t0<12.0){ if(__atomic_load_n(stamp,__ATOMIC_ACQUIRE)==42ull){ran=1;break;} }
    printf("  E2 stamp AFTER release  (want 42)           : %s (%.2f s)\n",
           ran?"42 - released correctly":"*** NEVER RAN ***", now_s()-t0);

    hipStreamSynchronize(gate); hipDeviceSynchronize();
    printf("  verdict: %s\n",
           (seen && before==0 && ran) ? "FIX IS VALID - dedicated stream works and preserves ordering"
                                      : "FIX IS NOT SAFE");
    return 0;
}
