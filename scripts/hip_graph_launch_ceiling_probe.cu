// Optimistic launch-gap ceiling for a localized DS4 partial HIP graph.
// Empty serialized kernels maximize the fraction graph replay can remove;
// production kernels can only expose less launch-only time.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

static void check(hipError_t rc, const char *what) {
    if (rc != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(rc));
        std::exit(1);
    }
}

__global__ static void ordered_tick(uint32_t *value) {
    if (threadIdx.x == 0u) *value += 1u;
}

static float elapsed_us(hipEvent_t begin, hipEvent_t end, uint32_t iters) {
    float ms = 0.0f;
    check(hipEventElapsedTime(&ms, begin, end), "hipEventElapsedTime");
    return ms * 1000.0f / (float)iters;
}

static int run_case(uint32_t nodes) {
    constexpr uint32_t kWarm = 64u;
    constexpr uint32_t kIters = 2000u;
    hipStream_t stream = nullptr;
    hipEvent_t begin = nullptr, end = nullptr;
    uint32_t *value = nullptr;
    check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
          "hipStreamCreateWithFlags");
    check(hipEventCreate(&begin), "hipEventCreate begin");
    check(hipEventCreate(&end), "hipEventCreate end");
    check(hipMalloc(&value, sizeof(*value)), "hipMalloc");

    check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal),
          "hipStreamBeginCapture");
    for (uint32_t n = 0; n < nodes; n++) {
        ordered_tick<<<1, 1, 0, stream>>>(value);
    }
    check(hipGetLastError(), "captured launch");
    hipGraph_t graph = nullptr;
    check(hipStreamEndCapture(stream, &graph), "hipStreamEndCapture");
    hipGraphExec_t exec = nullptr;
    check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
          "hipGraphInstantiate");

    for (uint32_t i = 0; i < kWarm; i++) check(hipGraphLaunch(exec, stream), "warm graph");
    check(hipStreamSynchronize(stream), "warm graph sync");
    check(hipMemsetAsync(value, 0, sizeof(*value), stream), "reset graph");
    check(hipEventRecord(begin, stream), "record graph begin");
    for (uint32_t i = 0; i < kIters; i++) check(hipGraphLaunch(exec, stream), "graph launch");
    check(hipEventRecord(end, stream), "record graph end");
    check(hipEventSynchronize(end), "graph end sync");
    const float graph_us = elapsed_us(begin, end, kIters);
    uint32_t graph_value = 0u;
    check(hipMemcpy(&graph_value, value, sizeof(graph_value), hipMemcpyDeviceToHost),
          "read graph value");

    for (uint32_t i = 0; i < kWarm; i++) {
        for (uint32_t n = 0; n < nodes; n++) ordered_tick<<<1, 1, 0, stream>>>(value);
    }
    check(hipStreamSynchronize(stream), "warm eager sync");
    check(hipMemsetAsync(value, 0, sizeof(*value), stream), "reset eager");
    check(hipEventRecord(begin, stream), "record eager begin");
    for (uint32_t i = 0; i < kIters; i++) {
        for (uint32_t n = 0; n < nodes; n++) ordered_tick<<<1, 1, 0, stream>>>(value);
    }
    check(hipEventRecord(end, stream), "record eager end");
    check(hipEventSynchronize(end), "eager end sync");
    const float eager_us = elapsed_us(begin, end, kIters);
    uint32_t eager_value = 0u;
    check(hipMemcpy(&eager_value, value, sizeof(eager_value), hipMemcpyDeviceToHost),
          "read eager value");

    const uint32_t expected = nodes * kIters;
    const int exact = graph_value == expected && eager_value == expected;
    std::printf("nodes=%u eager_us=%.3f graph_us=%.3f ceiling_saved_us=%.3f "
                "exact=%d graph_value=%u eager_value=%u\n",
                nodes, eager_us, graph_us, eager_us - graph_us, exact,
                graph_value, eager_value);

    (void)hipGraphExecDestroy(exec);
    (void)hipGraphDestroy(graph);
    (void)hipFree(value);
    (void)hipEventDestroy(begin);
    (void)hipEventDestroy(end);
    (void)hipStreamDestroy(stream);
    return exact ? 0 : 2;
}

int main() {
    const uint32_t cases[] = {4u, 8u, 12u, 16u, 24u, 32u};
    for (uint32_t nodes : cases) {
        if (run_case(nodes) != 0) return 2;
    }
    return 0;
}
