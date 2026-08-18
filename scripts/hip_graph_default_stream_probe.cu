// Minimal ROCm lifecycle probe for capturing DS4's legacy default stream.
// This deliberately uses stream 0 because current DS4 kernel wrappers launch
// there.  A failure means partial graph replay needs explicit-stream plumbing
// before any production-shaped oracle is meaningful.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

static void check(hipError_t rc, const char *what) {
    if (rc != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(rc));
        std::exit(1);
    }
}

__global__ static void add_one(uint32_t *value) {
    if (threadIdx.x == 0u) *value += 1u;
}

int main() {
    uint32_t *value = nullptr;
    check(hipMalloc(&value, sizeof(*value)), "hipMalloc");
    check(hipMemset(value, 0, sizeof(*value)), "hipMemset");

    hipError_t begin = hipStreamBeginCapture(nullptr, hipStreamCaptureModeGlobal);
    if (begin != hipSuccess) {
        std::fprintf(stderr, "default_stream_capture=unsupported error=%s\n",
                     hipGetErrorString(begin));
        (void)hipGetLastError();
        (void)hipFree(value);
        return 2;
    }
    add_one<<<1, 1>>>(value);
    check(hipGetLastError(), "captured kernel launch");

    hipGraph_t graph = nullptr;
    check(hipStreamEndCapture(nullptr, &graph), "hipStreamEndCapture");
    hipGraphExec_t exec = nullptr;
    check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
          "hipGraphInstantiate");
    for (uint32_t i = 0; i < 17u; i++) {
        check(hipGraphLaunch(exec, nullptr), "hipGraphLaunch");
    }
    check(hipDeviceSynchronize(), "hipDeviceSynchronize");
    uint32_t host = 0u;
    check(hipMemcpy(&host, value, sizeof(host), hipMemcpyDeviceToHost),
          "hipMemcpy");
    std::printf("default_stream_capture=supported replays=17 value=%u exact=%d\n",
                host, host == 17u);

    (void)hipGraphExecDestroy(exec);
    (void)hipGraphDestroy(graph);
    (void)hipFree(value);
    return host == 17u ? 0 : 3;
}
