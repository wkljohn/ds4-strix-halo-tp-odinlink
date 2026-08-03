// T2: when the GPU's arrival flag becomes host-visible, has the PAYLOAD landed?
//
// This is the failure Metal MEASURED and designed around (ds4_metal.m:8776-8783):
//   "a flag write carries no memory-visibility guarantee for the payload buffer
//    ... the service thread can observe the flag before the producing kernels'
//    stores reach CPU-visible memory (measured: stale rows in the first sub-kick)"
//
// Our ROCm gate runtime assumes hipStreamWriteValue64, being a stream-ordered
// barrier packet, implies the payload is visible. T0 tested ONE WORD. This tests
// a real payload, which is the actual question.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

#define PAYLOAD 4096          // floats, ~ one hidden-state row
#define ITERS   20000

__global__ void produce(float *p, uint32_t n, float v) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

int main() {
    float *payload = nullptr; uint64_t *flag = nullptr;
    if (hipMalloc(&payload, PAYLOAD * sizeof(float)) != hipSuccess) return 1;
    if (hipMalloc(&flag, 64) != hipSuccess) return 1;
    uint64_t zero = 0; hipMemcpy(flag, &zero, 8, hipMemcpyHostToDevice);
    hipDeviceSynchronize();

    volatile uint64_t *hflag = (volatile uint64_t *)flag;
    volatile float    *hpay  = (volatile float *)payload;

    long stale = 0, worst_scan = 0;
    for (int it = 1; it <= ITERS; it++) {
        const float want = (float)it;
        // producing kernel, then the arrival flag - exactly the gate ordering
        hipLaunchKernelGGL(produce, dim3((PAYLOAD+255)/256), dim3(256), 0, 0,
                           payload, (uint32_t)PAYLOAD, want);
        hipStreamWriteValue64(0, (void *)flag, (int64_t)it, 0);

        // service-thread role: spin on the flag, then read the payload
        long spins = 0;
        while (__atomic_load_n(hflag, __ATOMIC_ACQUIRE) < (uint64_t)it) {
            if (++spins > 200000000L) { printf("TIMEOUT at iter %d\n", it); return 2; }
        }
        // flag says arrived - is every element of the payload correct RIGHT NOW?
        int bad = 0;
        for (int i = 0; i < PAYLOAD; i++) if (hpay[i] != want) { bad = 1; break; }
        if (bad) {
            stale++;
            // how long until it settles? measures the size of the window
            long s2 = 0; int still;
            do { still = 0;
                 for (int i = 0; i < PAYLOAD; i++) if (hpay[i] != want) { still = 1; break; }
                 s2++;
            } while (still && s2 < 100000000L);
            if (s2 > worst_scan) worst_scan = s2;
        }
    }
    printf("iterations           : %d (payload %d floats)\n", ITERS, PAYLOAD);
    printf("STALE-ON-ARRIVAL     : %ld  (%.4f%%)\n", stale, 100.0*stale/ITERS);
    printf("worst settle scan    : %ld\n", worst_scan);
    printf("VERDICT: %s\n", stale == 0
        ? "flag arrival IMPLIES payload visible - gate design is sound"
        : "PAYLOAD CAN BE STALE - flag arrival is NOT sufficient (Metal was right)");
    hipFree(payload); hipFree(flag);
    return 0;
}
