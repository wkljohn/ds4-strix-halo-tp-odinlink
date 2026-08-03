/* T3: does the ds4 TP gate handshake actually work on gfx1151?
 *
 * The gate needs BOTH directions, and the earlier probes covered neither:
 *   A. GPU -> CPU : hipStreamWriteValue64 into the slab, CPU polls it
 *                   (T2 tested a KERNEL's write, not a stream write-value)
 *   B. CPU -> GPU : CPU stores to the release word, hipStreamWaitValue64 sees it
 *                   (T0 tested the CPU reading hipMalloc memory - opposite way)
 *
 * Tiny: 64 bytes device + 64 bytes host. No model.
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <ctime>
#include <atomic>

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
#define CK(e) do { hipError_t _e=(e); if(_e!=hipSuccess){ \
    printf("  FAIL %s -> %s\n", #e, hipGetErrorString(_e)); return 1; } } while(0)

int main(void) {
    int dev = 0; CK(hipSetDevice(dev));
    int attr = 0;
    hipDeviceGetAttribute(&attr, hipDeviceAttributeCanUseStreamWaitValue, dev);
    printf("  canUseStreamWaitValue = %d\n", attr);

    /* the two words, allocated exactly as ds4_rocm.cu does */
    void *slab = NULL; CK(hipMalloc(&slab, 4096));
    CK(hipMemset(slab, 0, 4096));
    volatile uint64_t *gpu_flag = (volatile uint64_t *)slab;

    void *sig = NULL;
    int sig_is_host = 0;
    if (hipExtMallocWithFlags(&sig, 64, hipMallocSignalMemory) != hipSuccess || !sig) {
        printf("  hipMallocSignalMemory UNAVAILABLE -> falling back to hipHostMalloc "
               "(this is what ds4 does)\n");
        CK(hipHostMalloc(&sig, 64, hipHostMallocMapped));
        sig_is_host = 1;
    } else {
        printf("  hipMallocSignalMemory available\n");
    }
    volatile uint64_t *cpu_flag = (volatile uint64_t *)sig;
    __atomic_store_n(cpu_flag, 0, __ATOMIC_RELEASE);

    /* ds4 passes the HOST pointer straight to hipStreamWaitValue64. For mapped
     * host memory the device address can differ - check. */
    void *devptr = NULL;
    if (sig_is_host) {
        if (hipHostGetDevicePointer(&devptr, sig, 0) == hipSuccess) {
            printf("  host ptr=%p  device ptr=%p  %s\n", sig, devptr,
                   devptr == sig ? "(identical)" : "*** DIFFERENT - ds4 passes the host one ***");
        } else {
            printf("  hipHostGetDevicePointer FAILED\n");
        }
    }

    hipStream_t s; CK(hipStreamCreate(&s));

    /* ---- direction A: stream write-value -> CPU poll ---- */
    CK(hipStreamWriteValue64(s, (void *)gpu_flag, (int64_t)12345, 0));
    double t0 = now_s(); int sawA = 0;
    while (now_s() - t0 < 3.0) {
        if (__atomic_load_n(gpu_flag, __ATOMIC_ACQUIRE) == 12345ull) { sawA = 1; break; }
    }
    printf("  A. GPU stream-write -> CPU poll : %s (%.3f s)\n",
           sawA ? "SEEN" : "*** NEVER SEEN ***", now_s() - t0);

    /* ---- direction B: CPU store -> stream wait-value releases ---- */
    hipError_t we = hipStreamWaitValue64(s, (void *)cpu_flag, (int64_t)7,
                                         hipStreamWaitValueGte, ~0ULL);
    printf("  B. hipStreamWaitValue64 enqueue : %s\n",
           we == hipSuccess ? "accepted" : hipGetErrorString(we));
    if (we == hipSuccess) {
        struct timespec d = {0, 200000000}; nanosleep(&d, NULL);
        __atomic_store_n(cpu_flag, 7ull, __ATOMIC_RELEASE);
        t0 = now_s(); int done = 0;
        while (now_s() - t0 < 5.0) {
            hipError_t q = hipStreamQuery(s);
            if (q == hipSuccess) { done = 1; break; }
            if (q != hipErrorNotReady) { printf("  query err %s\n", hipGetErrorString(q)); break; }
        }
        printf("  B. CPU store -> wait released  : %s (%.3f s)\n",
               done ? "RELEASED" : "*** NEVER RELEASED - this is the deadlock ***",
               now_s() - t0);
        if (!done) { printf("  -> leaving stream blocked; aborting probe\n"); return 2; }
    }
    CK(hipStreamSynchronize(s));
    printf("  probe complete\n");
    return 0;
}
