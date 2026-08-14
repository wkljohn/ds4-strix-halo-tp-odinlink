#include <hip/hip_runtime.h>
#include <infiniband/verbs.h>

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/resource.h>

static uint64_t ds4_slab_bytes(uint32_t layers, uint32_t embd,
                               uint32_t big_rows) {
    const uint64_t vec = (uint64_t)embd * sizeof(float);
    const uint64_t slots = (uint64_t)layers * 2;
    return slots * vec * 2 + slots * 8 * 2 + 16 + slots * 4 +
           (uint64_t)layers * 8 * vec * 2 + (uint64_t)big_rows * vec * 2;
}

static void usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage: %s <mlx5-device> [layers=61] [embd=4096] "
                 "[big-rows=2048]\n",
                 argv0);
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 5) {
        usage(argv[0]);
        return 2;
    }
    const uint32_t layers = argc > 2 ? (uint32_t)std::strtoul(argv[2], nullptr, 10) : 61;
    const uint32_t embd = argc > 3 ? (uint32_t)std::strtoul(argv[3], nullptr, 10) : 4096;
    const uint32_t big_rows = argc > 4 ? (uint32_t)std::strtoul(argv[4], nullptr, 10) : 2048;
    if (!layers || !embd || !big_rows) {
        usage(argv[0]);
        return 2;
    }

    struct rlimit lim = {};
    if (getrlimit(RLIMIT_MEMLOCK, &lim) != 0) {
        std::perror("getrlimit(RLIMIT_MEMLOCK)");
        return 1;
    }

    const uint64_t bytes = ds4_slab_bytes(layers, embd, big_rows);
    void *device_slab = nullptr;
    void *host_slab = nullptr;
    hipError_t hip_rc = hipMalloc(&device_slab, (size_t)bytes);
    if (hip_rc != hipSuccess) {
        std::fprintf(stderr, "hipMalloc(%" PRIu64 "): %s\n", bytes,
                     hipGetErrorString(hip_rc));
        return 1;
    }
    hip_rc = hipMemset(device_slab, 0, (size_t)bytes);
    if (hip_rc != hipSuccess || hipDeviceSynchronize() != hipSuccess) {
        std::fprintf(stderr, "hipMemset/synchronize: %s\n",
                     hipGetErrorString(hip_rc));
        (void)hipFree(device_slab);
        return 1;
    }

    int count = 0;
    ibv_device **devices = ibv_get_device_list(&count);
    ibv_device *chosen = nullptr;
    for (int i = 0; devices && i < count; ++i) {
        if (std::strcmp(ibv_get_device_name(devices[i]), argv[1]) == 0) {
            chosen = devices[i];
            break;
        }
    }
    if (!chosen) {
        std::fprintf(stderr, "verbs device %s not found\n", argv[1]);
        if (devices) ibv_free_device_list(devices);
        (void)hipFree(device_slab);
        return 1;
    }

    ibv_context *ctx = ibv_open_device(chosen);
    ibv_pd *pd = ctx ? ibv_alloc_pd(ctx) : nullptr;
    errno = 0;
    ibv_mr *mr = pd ? ibv_reg_mr(pd, device_slab, (size_t)bytes,
                                 IBV_ACCESS_LOCAL_WRITE |
                                 IBV_ACCESS_REMOTE_READ |
                                 IBV_ACCESS_REMOTE_WRITE)
                    : nullptr;
    if (!mr && pd) {
        std::fprintf(stderr,
                     "INFO device=%s hipMalloc_bytes=%" PRIu64
                     " memlock_soft=%" PRIu64 " memlock_hard=%" PRIu64
                     " reg_mr=%s\n",
                     argv[1], bytes, (uint64_t)lim.rlim_cur,
                     (uint64_t)lim.rlim_max, std::strerror(errno));
        errno = 0;
        hip_rc = hipHostMalloc(&host_slab, (size_t)bytes, hipHostMallocMapped);
        const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_REMOTE_WRITE;
        const uint64_t big_bytes = (uint64_t)big_rows * embd * sizeof(float);
        const uint64_t core_bytes = bytes - 2 * big_bytes;
        ibv_mr *segments[3] = {};
        if (hip_rc == hipSuccess) {
            segments[0] = ibv_reg_mr(pd, host_slab, (size_t)core_bytes, access);
            segments[1] = ibv_reg_mr(pd, (char *)host_slab + core_bytes,
                                     (size_t)big_bytes, access);
            segments[2] = ibv_reg_mr(pd,
                                     (char *)host_slab + core_bytes + big_bytes,
                                     (size_t)big_bytes, access);
        }
        if (segments[0] && segments[1] && segments[2]) {
            std::printf("PASS device=%s allocator=hipHostMallocMapped "
                        "bytes=%" PRIu64 " mr_layout=3 core=%" PRIu64
                        " big=%" PRIu64 " memlock_soft=%" PRIu64
                        " memlock_hard=%" PRIu64 "\n",
                        argv[1], bytes, core_bytes, big_bytes,
                        (uint64_t)lim.rlim_cur, (uint64_t)lim.rlim_max);
            for (ibv_mr *segment : segments) ibv_dereg_mr(segment);
            mr = (ibv_mr *)(uintptr_t)1;
        } else {
            for (ibv_mr *segment : segments)
                if (segment) ibv_dereg_mr(segment);
        }
    }
    if (mr && mr != (ibv_mr *)(uintptr_t)1) {
        std::printf("PASS device=%s allocator=%s bytes=%" PRIu64
                    " lkey=0x%x rkey=0x%x memlock_soft=%" PRIu64
                    " memlock_hard=%" PRIu64 "\n",
                    argv[1], host_slab ? "hipHostMallocMapped" : "hipMalloc",
                    bytes, mr->lkey, mr->rkey,
                    (uint64_t)lim.rlim_cur, (uint64_t)lim.rlim_max);
    } else if (!mr) {
        std::fprintf(stderr,
                     "FAIL device=%s hipHostMallocMapped_bytes=%" PRIu64
                     " reg_mr=%s hip=%s\n",
                     argv[1], bytes, std::strerror(errno),
                     hipGetErrorString(hip_rc));
    }

    if (mr && mr != (ibv_mr *)(uintptr_t)1) ibv_dereg_mr(mr);
    if (pd) ibv_dealloc_pd(pd);
    if (ctx) ibv_close_device(ctx);
    ibv_free_device_list(devices);
    if (host_slab) (void)hipHostFree(host_slab);
    (void)hipFree(device_slab);
    return mr ? 0 : 1;
}
