#include <hip/hip_runtime.h>

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

constexpr uint64_t kPageBytes = 4096;

__device__ __forceinline__ uint8_t page_pattern(uint64_t page) {
    return static_cast<uint8_t>(0xa5u ^ page ^ (page >> 8) ^ (page >> 32));
}

__global__ void touch_pages(uint8_t *base, uint64_t pages) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                           threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t page = first; page < pages; page += stride) {
        base[page * kPageBytes] = page_pattern(page);
    }
}

__global__ void verify_pages(const uint8_t *base, uint64_t pages,
                             unsigned int *errors) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                           threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t page = first; page < pages; page += stride) {
        if (base[page * kPageBytes] != page_pattern(page)) {
            atomicAdd(errors, 1u);
        }
    }
}

bool hip_ok(hipError_t rc, const char *what) {
    if (rc == hipSuccess) return true;
    std::fprintf(stderr, "FAIL stage=%s hip=%s\n", what,
                 hipGetErrorString(rc));
    return false;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s BYTES\n", argv[0]);
        return 2;
    }
    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(argv[1], &end, 10);
    if (errno || !end || *end != '\0' || parsed < kPageBytes ||
        parsed > std::numeric_limits<size_t>::max()) {
        std::fprintf(stderr, "invalid byte count: %s\n", argv[1]);
        return 2;
    }
    const size_t bytes = static_cast<size_t>(parsed);
    const uint64_t pages = bytes / kPageBytes + (bytes % kPageBytes != 0);
    if (pages == 0) return 2;

    size_t free_before = 0, total = 0;
    if (!hip_ok(hipMemGetInfo(&free_before, &total), "meminfo-before")) return 1;
    std::printf("INFO requested_bytes=%zu requested_gib=%.3f pages=%" PRIu64
                " free_before=%zu total=%zu\n",
                bytes, static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0),
                pages, free_before, total);
    std::fflush(stdout);
    // On gfx1151, the pool total is the GPUVM aperture while free is the
    // currently backed capacity available to HIP.  A large fixed BIOS UMA
    // carve-out can therefore produce total >> free.  Never force a full-page
    // touch beyond this value: raised TTM limits otherwise permit destructive
    // host reclaim before hipMalloc reports ENOMEM.
    constexpr size_t kSafetyReserve = 3ull * 1024ull * 1024ull * 1024ull;
    if (free_before <= kSafetyReserve || bytes > free_before - kSafetyReserve) {
        std::fprintf(stderr,
                     "FAIL stage=capacity-preflight requested=%zu "
                     "free_before=%zu safety_reserve=%zu\n",
                     bytes, free_before, kSafetyReserve);
        return 1;
    }

    uint8_t *allocation = nullptr;
    unsigned int *errors = nullptr;
    if (!hip_ok(hipMalloc(&allocation, bytes), "hipMalloc-residency")) return 1;
    if (!hip_ok(hipMalloc(&errors, sizeof(*errors)), "hipMalloc-errors")) {
        (void)hipFree(allocation);
        return 1;
    }
    if (!hip_ok(hipMemset(errors, 0, sizeof(*errors)), "clear-errors")) return 1;

    constexpr int threads = 256;
    constexpr int blocks = 4096;
    hipLaunchKernelGGL(touch_pages, dim3(blocks), dim3(threads), 0, 0,
                       allocation, pages);
    if (!hip_ok(hipGetLastError(), "touch-launch") ||
        !hip_ok(hipDeviceSynchronize(), "touch-sync")) return 1;

    size_t free_after_touch = 0, total_after_touch = 0;
    if (!hip_ok(hipMemGetInfo(&free_after_touch, &total_after_touch),
                "meminfo-after-touch")) return 1;

    hipLaunchKernelGGL(verify_pages, dim3(blocks), dim3(threads), 0, 0,
                       allocation, pages, errors);
    if (!hip_ok(hipGetLastError(), "verify-launch") ||
        !hip_ok(hipDeviceSynchronize(), "verify-sync")) return 1;
    unsigned int host_errors = 0;
    if (!hip_ok(hipMemcpy(&host_errors, errors, sizeof(host_errors),
                          hipMemcpyDeviceToHost),
                "copy-errors")) return 1;

    const hipError_t free_errors_rc = hipFree(errors);
    const hipError_t free_allocation_rc = hipFree(allocation);
    if (!hip_ok(free_errors_rc, "free-errors") ||
        !hip_ok(free_allocation_rc, "free-residency")) return 1;
    if (host_errors != 0) {
        std::fprintf(stderr, "FAIL stage=verify page_errors=%u\n", host_errors);
        return 1;
    }
    std::printf("PASS requested_bytes=%zu pages_touched=%" PRIu64
                " page_errors=0 free_after_touch=%zu total=%zu\n",
                bytes, pages, free_after_touch, total_after_touch);
    return 0;
}
