// Sustained gfx1151 probe for an ordered host-callback TP gate.
//
// Build:
//   hipcc -O3 --offload-arch=gfx1151 \
//     scripts/hip_host_callback_gate_probe.cpp -o /tmp/hip_host_callback_gate_probe
// Run the decode-sized schedule (300 tokens * 86 row gates):
//   /tmp/hip_host_callback_gate_probe 25800 57

#include <hip/hip_runtime.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

struct gate_probe_context {
    volatile uint64_t *words;
    uint64_t seq;
    uint32_t delay_us;
    uint32_t *host_errors;
};

static uint64_t monotonic_ns() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

__global__ static void producer_kernel(uint64_t *words, uint64_t seq) {
    if (blockIdx.x == 0 && threadIdx.x == 0) words[0] = seq;
}

__global__ static void consumer_kernel(const uint64_t *words, uint64_t seq,
                                       uint32_t *errors) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && words[1] != seq) {
        atomicAdd(errors, 1u);
    }
}

static void gate_callback(void *opaque) {
    auto *ctx = static_cast<gate_probe_context *>(opaque);
    const uint64_t produced = __atomic_load_n(&ctx->words[0], __ATOMIC_ACQUIRE);
    if (produced != ctx->seq) {
        __atomic_fetch_add(ctx->host_errors, 1u, __ATOMIC_RELAXED);
    }
    if (ctx->delay_us != 0) {
        const uint64_t until = monotonic_ns() + (uint64_t)ctx->delay_us * 1000ull;
        while (monotonic_ns() < until) {
#if defined(__x86_64__)
            __builtin_ia32_pause();
#endif
        }
    }
    __atomic_store_n(&ctx->words[1], ctx->seq, __ATOMIC_RELEASE);
}

static int check(hipError_t err, const char *what) {
    if (err == hipSuccess) return 1;
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(err));
    return 0;
}

int main(int argc, char **argv) {
    const uint32_t gates = argc > 1 ? (uint32_t)std::strtoul(argv[1], nullptr, 10)
                                    : 25800u;
    const uint32_t delay_us = argc > 2 ? (uint32_t)std::strtoul(argv[2], nullptr, 10)
                                       : 57u;
    if (gates == 0) {
        std::fprintf(stderr, "gate count must be nonzero\n");
        return 2;
    }

    uint64_t *host_words = nullptr;
    uint64_t *device_words = nullptr;
    uint32_t *device_errors = nullptr;
    uint32_t host_errors = 0;
    if (!check(hipHostMalloc(&host_words, 2u * sizeof(uint64_t), hipHostMallocMapped),
               "hipHostMalloc") ||
        !check(hipHostGetDevicePointer((void **)&device_words, host_words, 0),
               "hipHostGetDevicePointer") ||
        !check(hipMalloc(&device_errors, sizeof(uint32_t)), "hipMalloc errors") ||
        !check(hipMemset(device_errors, 0, sizeof(uint32_t)), "hipMemset errors")) {
        return 1;
    }
    host_words[0] = 0;
    host_words[1] = 0;
    std::vector<gate_probe_context> contexts(gates);

    const uint64_t begin = monotonic_ns();
    for (uint32_t i = 0; i < gates; i++) {
        const uint64_t seq = (uint64_t)i + 1u;
        contexts[i] = {host_words, seq, delay_us, &host_errors};
        producer_kernel<<<1, 1>>>(device_words, seq);
        if (!check(hipGetLastError(), "producer launch") ||
            !check(hipLaunchHostFunc(nullptr, gate_callback, &contexts[i]),
                   "hipLaunchHostFunc")) {
            return 1;
        }
        consumer_kernel<<<1, 1>>>(device_words, seq, device_errors);
        if (!check(hipGetLastError(), "consumer launch")) return 1;
    }
    const uint64_t submitted = monotonic_ns();
    if (!check(hipDeviceSynchronize(), "hipDeviceSynchronize")) return 1;
    const uint64_t finished = monotonic_ns();

    uint32_t gpu_errors = 0;
    if (!check(hipMemcpy(&gpu_errors, device_errors, sizeof(gpu_errors),
                         hipMemcpyDeviceToHost), "read errors")) {
        return 1;
    }
    const double submit_ms = (double)(submitted - begin) / 1.0e6;
    const double total_ms = (double)(finished - begin) / 1.0e6;
    std::printf("gates=%u delay_us=%u submit_ms=%.3f total_ms=%.3f "
                "total_us_per_gate=%.3f host_errors=%u gpu_errors=%u\n",
                gates, delay_us, submit_ms, total_ms,
                total_ms * 1000.0 / (double)gates,
                host_errors, gpu_errors);

    (void)hipFree(device_errors);
    (void)hipHostFree(host_words);
    return host_errors == 0 && gpu_errors == 0 ? 0 : 1;
}
