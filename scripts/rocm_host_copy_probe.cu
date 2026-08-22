#include <hip/hip_runtime.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static const size_t kBytes = 8u * 1024u * 1024u;

static const char *memory_type_name(unsigned type) {
    switch (type) {
    case hipMemoryTypeHost: return "host";
    case hipMemoryTypeDevice: return "device";
    case hipMemoryTypeManaged: return "managed";
    case hipMemoryTypeUnregistered: return "unregistered";
    default: return "unknown";
    }
}

static void print_attributes(const char *label, const void *ptr) {
    hipPointerAttribute_t attr = {};
    hipError_t err = hipPointerGetAttributes(&attr, ptr);
    if (err != hipSuccess) {
        printf("allocation=%s attributes=%s\n", label, hipGetErrorString(err));
        (void)hipGetLastError();
        return;
    }
    printf("allocation=%s attributes=ok type=%s device=%d managed=%d ptr_mod_4k=%zu\n",
           label, memory_type_name((unsigned)attr.type), attr.device,
           attr.isManaged, (size_t)((uintptr_t)ptr & 4095u));
}

static int exercise(const char *label, void *src, void *dst, int register_src) {
    memset(src, 0x5a, kBytes);
    print_attributes(label, src);

    hipError_t reg = hipSuccess;
    if (register_src) {
        reg = hipHostRegister(src, kBytes,
                              hipHostRegisterMapped | hipHostRegisterReadOnly);
        printf("allocation=%s host_register=%s\n", label,
               hipGetErrorString(reg));
        if (reg != hipSuccess) (void)hipGetLastError();
    }

    hipError_t copy = hipMemcpy(dst, src, kBytes, hipMemcpyHostToDevice);
    printf("allocation=%s h2d_copy=%s\n", label, hipGetErrorString(copy));
    if (copy != hipSuccess) (void)hipGetLastError();

    if (register_src && reg == hipSuccess) {
        hipError_t unreg = hipHostUnregister(src);
        printf("allocation=%s host_unregister=%s\n", label,
               hipGetErrorString(unreg));
        if (unreg != hipSuccess) (void)hipGetLastError();
    }
    return copy == hipSuccess;
}

static int exercise_adjacent_ranges(void *base, void *dst) {
    const long page_l = sysconf(_SC_PAGESIZE);
    const size_t page = page_l > 0 ? (size_t)page_l : 4096u;
    const size_t half = kBytes;
    memset(base, 0x3c, 2u * half);

    void *registered[2] = {};
    size_t registered_bytes[2] = {};
    int ok = 1;
    for (unsigned range = 0; range < 2; range++) {
        char *src = (char *)base + (size_t)range * half;
        const uintptr_t start = (uintptr_t)src & ~(uintptr_t)(page - 1u);
        const size_t delta = (size_t)((uintptr_t)src - start);
        const size_t bytes = (delta + half + page - 1u) & ~(page - 1u);
        registered[range] = (void *)start;
        registered_bytes[range] = bytes;
        hipError_t reg = hipHostRegister((void *)start, bytes,
                                         hipHostRegisterMapped |
                                         hipHostRegisterReadOnly);
        printf("allocation=adjacent-malloc range=%u register_base_mod_4k=%zu "
               "register_bytes=%zu host_register=%s\n",
               range, (size_t)(start & 4095u), bytes,
               hipGetErrorString(reg));
        if (reg != hipSuccess) (void)hipGetLastError();

        hipError_t copy = hipMemcpy(dst, src, half, hipMemcpyHostToDevice);
        printf("allocation=adjacent-malloc range=%u h2d_copy=%s\n",
               range, hipGetErrorString(copy));
        if (copy != hipSuccess) {
            ok = 0;
            (void)hipGetLastError();
        }
    }
    for (unsigned range = 0; range < 2; range++) {
        if (!registered[range]) continue;
        hipError_t unreg = hipHostUnregister(registered[range]);
        printf("allocation=adjacent-malloc range=%u host_unregister=%s\n",
               range, hipGetErrorString(unreg));
        if (unreg != hipSuccess) (void)hipGetLastError();
    }
    return ok;
}

int main(int argc, char **argv) {
    const int expect_overlap_rejected =
        argc == 2 && strcmp(argv[1], "--expect-overlap-rejected") == 0;
    if (argc > 2 || (argc == 2 && !expect_overlap_rejected)) {
        fprintf(stderr, "usage: %s [--expect-overlap-rejected]\n", argv[0]);
        return 2;
    }
    void *dst = NULL;
    hipError_t err = hipMalloc(&dst, kBytes);
    if (err != hipSuccess) {
        fprintf(stderr, "hipMalloc failed: %s\n", hipGetErrorString(err));
        return 1;
    }

    int ordinary_passed = 1;
    void *plain = malloc(kBytes);
    if (!plain) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }
    ordinary_passed &= exercise("malloc-unregistered", plain, dst, 0);
    ordinary_passed &= exercise("malloc-register", plain, dst, 1);
    free(plain);

    void *adjacent = malloc(2u * kBytes);
    if (!adjacent) {
        fprintf(stderr, "adjacent malloc failed\n");
        return 1;
    }
    const int adjacent_passed = exercise_adjacent_ranges(adjacent, dst);
    free(adjacent);

    void *aligned = NULL;
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0 || posix_memalign(&aligned, (size_t)page, kBytes) != 0) {
        fprintf(stderr, "posix_memalign failed\n");
        return 1;
    }
    ordinary_passed &= exercise("page-aligned-register", aligned, dst, 1);
    free(aligned);

    void *mapped = mmap(NULL, kBytes, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "mmap failed: %s\n", strerror(errno));
        return 1;
    }
    ordinary_passed &= exercise("anonymous-mmap-register", mapped, dst, 1);
    (void)munmap(mapped, kBytes);

    void *pinned = NULL;
    err = hipHostMalloc(&pinned, kBytes, hipHostMallocMapped);
    if (err != hipSuccess) {
        fprintf(stderr, "hipHostMalloc failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    ordinary_passed &= exercise("hip-host-mapped", pinned, dst, 0);
    (void)hipHostFree(pinned);
    (void)hipFree(dst);

    const int expected = ordinary_passed &&
        (expect_overlap_rejected ? !adjacent_passed : adjacent_passed);
    printf("rocm_host_copy_probe=%s ordinary=%d adjacent=%d "
           "expect_overlap_rejected=%d\n",
           expected ? "PASS" : "FAIL", ordinary_passed, adjacent_passed,
           expect_overlap_rejected);
    return expected ? 0 : 1;
}
