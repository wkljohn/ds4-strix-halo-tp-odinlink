// T0: is hipMalloc memory host-accessible on gfx1151 (UMA APU)?
// The ROCm TP gate runtime assumed YES. Metal's equivalent design is legal only
// because MTLResourceStorageModeShared gives a real host pointer. If this probe
// faults or reads stale, the slab must be reallocated as host-visible.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

__global__ void writer(volatile uint64_t *p, uint64_t v) { *p = v; }

int main() {
    uint64_t *d = nullptr;
    if (hipMalloc(&d, 4096) != hipSuccess) { printf("hipMalloc FAILED\n"); return 1; }
    printf("hipMalloc ptr = %p\n", (void*)d);

    // 1. can the HOST write device memory directly?
    printf("host write to hipMalloc ptr ... "); fflush(stdout);
    volatile uint64_t *hp = (volatile uint64_t *)d;
    *hp = 0x1111;                       // faults here if not host-accessible
    printf("OK (wrote 0x1111)\n");

    // 2. can the HOST read back what it wrote?
    printf("host read back            ... %#lx\n", (unsigned long)*hp);

    // 3. GPU writes, host reads WITHOUT hipDeviceSynchronize - coherence test
    hipLaunchKernelGGL(writer, dim3(1), dim3(1), 0, 0, (volatile uint64_t*)d, 0xABCDULL);
    int spins = 0; uint64_t seen = 0;
    while (spins < 20000000) { seen = *hp; if (seen == 0xABCDULL) break; spins++; }
    printf("gpu write seen by host    ... %s (value %#lx after %d spins)\n",
           seen == 0xABCDULL ? "YES - coherent" : "NO - STALE", (unsigned long)seen, spins);
    hipDeviceSynchronize();
    printf("after explicit sync       ... %#lx\n", (unsigned long)*hp);

    // 4. for comparison: hipHostMalloc, what the release word uses
    uint64_t *h = nullptr;
    if (hipHostMalloc(&h, 4096, hipHostMallocMapped) == hipSuccess) {
        *h = 0x2222;
        hipLaunchKernelGGL(writer, dim3(1), dim3(1), 0, 0, (volatile uint64_t*)h, 0xBEEFULL);
        hipDeviceSynchronize();
        printf("hipHostMalloc GPU->host   ... %#lx (expect 0xbeef)\n", (unsigned long)*h);
        hipHostFree(h);
    }
    hipFree(d);
    printf("PROBE COMPLETED WITHOUT FAULT\n");
    return 0;
}
