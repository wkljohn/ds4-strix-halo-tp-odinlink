// Model-free feasibility probe for an exact fused DeepSeek V4 HC stage.
//
// Build:
//   /opt/rocm/bin/hipcc -O3 --offload-arch=gfx1151 \
//     scripts/hc_cooperative_grid_probe.cu -o /tmp/hc_cooperative_grid_probe

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace cg = cooperative_groups;

static constexpr uint32_t kBlocks = 24;
static constexpr uint32_t kThreads = 256;
static constexpr uint32_t kIterations = 10000;

static void check(hipError_t rc, const char *what) {
    if (rc != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(rc));
        std::exit(1);
    }
}

__global__ static void cooperative_order_probe(
        uint32_t *phase,
        uint32_t *errors,
        uint32_t iteration) {
    cg::grid_group grid = cg::this_grid();
    if (threadIdx.x == 0) {
        phase[blockIdx.x] = iteration ^ (0x9e3779b9u * (blockIdx.x + 1u));
    }
    grid.sync();

    for (uint32_t i = threadIdx.x; i < kBlocks; i += blockDim.x) {
        const uint32_t want = iteration ^ (0x9e3779b9u * (i + 1u));
        if (phase[i] != want) atomicAdd(errors, 1u);
    }
    grid.sync();

    if (blockIdx.x == 0 && threadIdx.x == 0) phase[kBlocks] = iteration;
    grid.sync();
    if (phase[kBlocks] != iteration) atomicAdd(errors, 1u);
}

int main() {
    int device = 0;
    int cooperative = 0;
    int multiprocessors = 0;
    int resident_per_cu = 0;
    check(hipGetDevice(&device), "hipGetDevice");
    check(hipDeviceGetAttribute(&cooperative,
                                hipDeviceAttributeCooperativeLaunch,
                                device),
          "cooperative attribute");
    check(hipDeviceGetAttribute(&multiprocessors,
                                hipDeviceAttributeMultiprocessorCount,
                                device),
          "multiprocessor count");
    check(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                  &resident_per_cu, cooperative_order_probe, kThreads, 0),
          "cooperative occupancy");

    const int resident_grid = resident_per_cu * multiprocessors;
    std::printf("cooperative=%d multiprocessors=%d resident_blocks_per_cu=%d "
                "resident_grid=%d requested_grid=%u\n",
                cooperative, multiprocessors, resident_per_cu, resident_grid,
                kBlocks);
    if (!cooperative || resident_grid < (int)kBlocks) return 2;

    uint32_t *phase = nullptr;
    uint32_t *errors = nullptr;
    check(hipMalloc(&phase, (kBlocks + 1u) * sizeof(uint32_t)), "phase alloc");
    check(hipMalloc(&errors, sizeof(uint32_t)), "error alloc");
    check(hipMemset(errors, 0, sizeof(uint32_t)), "error clear");

    hipEvent_t begin{}, end{};
    check(hipEventCreate(&begin), "begin event");
    check(hipEventCreate(&end), "end event");
    check(hipEventRecord(begin), "record begin");
    for (uint32_t i = 1; i <= kIterations; ++i) {
        void *args[] = {&phase, &errors, &i};
        check(hipLaunchCooperativeKernel(
                      reinterpret_cast<const void *>(cooperative_order_probe),
                      dim3(kBlocks), dim3(kThreads), args, 0, nullptr),
              "cooperative launch");
    }
    check(hipEventRecord(end), "record end");
    check(hipEventSynchronize(end), "wait end");

    float elapsed_ms = 0.0f;
    uint32_t host_errors = 0;
    check(hipEventElapsedTime(&elapsed_ms, begin, end), "elapsed time");
    check(hipMemcpy(&host_errors, errors, sizeof(host_errors),
                    hipMemcpyDeviceToHost),
          "read errors");
    std::printf("iterations=%u errors=%u average_us=%.3f\n",
                kIterations, host_errors,
                1000.0f * elapsed_ms / (float)kIterations);

    (void)hipEventDestroy(begin);
    (void)hipEventDestroy(end);
    (void)hipFree(phase);
    (void)hipFree(errors);
    return host_errors == 0 ? 0 : 3;
}
