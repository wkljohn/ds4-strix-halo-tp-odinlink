/* T4: does the gate handshake work on the NULL STREAM, with a kernel queued
 * ahead of it - i.e. ds4's ACTUAL configuration?
 *
 * T3 validated hipStreamWriteValue64/WaitValue64 on a stream from
 * hipStreamCreate(). ds4 uses stream 0. HIP docs state the write executes only
 * "after all earlier commands on that stream have completed", and every ds4
 * compute kernel launches with no stream argument = the same null stream. So
 * T3 did NOT test the real configuration. This does.
 *
 * Case A: created stream, no prior kernel   (what T3 did - expect PASS)
 * Case B: NULL stream, no prior kernel
 * Case C: NULL stream, long kernel queued first  (ds4's real shape)
 * Case D: NULL stream, TWO write/wait pairs enqueued before any release
 *         (ds4 enqueues 44 before the service thread releases any)
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <ctime>

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec + 1e-9*(double)t.tv_nsec; }

__global__ void spin_kernel(unsigned long long iters, float *sink) {
    float a = 1.0f;
    for (unsigned long long i = 0; i < iters; i++) a = a * 1.0000001f + 1e-7f;
    if (threadIdx.x == 0xffffffffu) *sink = a;
}

/* returns 1 if the CPU saw the arrival flag within `budget` seconds */
static int run_case(const char *name, hipStream_t s, int queue_kernel, int npairs) {
    volatile uint64_t *gpu_flag = NULL;
    void *slab = NULL;
    if (hipMalloc(&slab, 4096) != hipSuccess) { printf("  %s: hipMalloc failed\n", name); return 0; }
    hipMemset(slab, 0, 4096);
    gpu_flag = (volatile uint64_t *)slab;

    void *sig = NULL;
    if (hipHostMalloc(&sig, 64, hipHostMallocMapped) != hipSuccess) { printf("  %s: host alloc failed\n", name); return 0; }
    volatile uint64_t *cpu_flag = (volatile uint64_t *)sig;
    __atomic_store_n(cpu_flag, 0, __ATOMIC_RELEASE);

    float *sink = NULL; hipMalloc(&sink, sizeof(float));
    if (queue_kernel) {
        /* ~1-2 s of GPU work ahead of the gate, like a real prefill layer */
        hipLaunchKernelGGL(spin_kernel, dim3(256), dim3(256), 0, s, 3000000ull, sink);
    }

    for (int i = 1; i <= npairs; i++) {
        hipError_t w = hipStreamWriteValue64(s, (void *)gpu_flag, (int64_t)i, 0);
        hipError_t v = hipStreamWaitValue64(s, (void *)cpu_flag, (int64_t)i,
                                            hipStreamWaitValueGte, ~0ULL);
        if (w != hipSuccess || v != hipSuccess) {
            printf("  %s: enqueue failed (write=%s wait=%s)\n", name,
                   hipGetErrorString(w), hipGetErrorString(v));
            return 0;
        }
    }

    /* CPU spins for the FIRST arrival, exactly like ds4_tp_pump */
    double t0 = now_s(); int seen = 0;
    while (now_s() - t0 < 12.0) {
        if (__atomic_load_n(gpu_flag, __ATOMIC_ACQUIRE) >= 1ull) { seen = 1; break; }
    }
    double el = now_s() - t0;
    printf("  %-58s : %s (%.2f s)\n", name,
           seen ? "ARRIVAL SEEN" : "*** NEVER SEEN - deadlock ***", el);

    if (seen) {
        /* release everything so the process can exit */
        for (int i = 1; i <= npairs; i++)
            __atomic_store_n(cpu_flag, (uint64_t)i, __ATOMIC_RELEASE);
    }
    return seen;
}

int main(void) {
    hipSetDevice(0);
    hipStream_t created; hipStreamCreate(&created);
    printf("  (each case has a 12 s budget)\n");
    run_case("A. created stream, no prior kernel, 1 pair", created, 0, 1);
    run_case("B. NULL stream,    no prior kernel, 1 pair", 0, 0, 1);
    run_case("C. NULL stream,    long kernel queued first, 1 pair", 0, 1, 1);
    run_case("D. NULL stream,    long kernel + 44 pairs (ds4's shape)", 0, 1, 44);
    printf("  done\n");
    return 0;
}
