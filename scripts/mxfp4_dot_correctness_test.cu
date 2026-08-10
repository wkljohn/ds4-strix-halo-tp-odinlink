// Standalone MXFP4 x F32 dot-product oracle for the DSpark routed-expert path.
// Build: hipcc -O3 --offload-arch=gfx1151 -o /tmp/mxfp4-dot scripts/mxfp4_dot_correctness_test.cu

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct mxfp4_block {
    uint8_t e;
    uint8_t qs[16];
};
static_assert(sizeof(mxfp4_block) == 17, "MXFP4 GGUF block must be 17 bytes");

static constexpr int8_t kvalues[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

static float e8m0_half_cpu(uint8_t e) {
    uint32_t bits = e < 2 ? (0x00200000u << e) : ((uint32_t)(e - 1u) << 23);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

__device__ static float e8m0_half_gpu(uint8_t e) {
    const uint32_t bits = e < 2 ? (0x00200000u << e) : ((uint32_t)(e - 1u) << 23);
    return __uint_as_float(bits);
}

__global__ static void mxfp4_dot_kernel(
        float *out, const mxfp4_block *w, const float *x, uint32_t n_blocks) {
    float sum = 0.0f;
    for (uint32_t b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        const mxfp4_block block = w[b];
        const float d = e8m0_half_gpu(block.e);
        float block_sum = 0.0f;
        for (uint32_t j = 0; j < 16; j++) {
            const uint8_t q = block.qs[j];
            block_sum += (float)kvalues[q & 15u] * x[b * 32u + j];
            block_sum += (float)kvalues[q >> 4] * x[b * 32u + j + 16u];
        }
        sum += d * block_sum;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = partial[0];
}

static float reference(const std::vector<mxfp4_block> &w, const std::vector<float> &x) {
    double sum = 0.0;
    for (size_t b = 0; b < w.size(); b++) {
        const double d = e8m0_half_cpu(w[b].e);
        for (uint32_t j = 0; j < 16; j++) {
            const uint8_t q = w[b].qs[j];
            sum += d * kvalues[q & 15u] * x[b * 32u + j];
            sum += d * kvalues[q >> 4] * x[b * 32u + j + 16u];
        }
    }
    return (float)sum;
}

#define HIP_OK(call) do { \
    hipError_t err_ = (call); \
    if (err_ != hipSuccess) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err_)); \
        return 1; \
    } \
} while (0)

int main() {
    constexpr uint32_t n_blocks = 127;
    std::vector<mxfp4_block> w(n_blocks);
    std::vector<float> x((size_t)n_blocks * 32u);
    uint32_t rng = 0x12345678u;
    auto next = [&]() { rng = rng * 1664525u + 1013904223u; return rng; };
    for (uint32_t b = 0; b < n_blocks; b++) {
        // Exercise denormal handling plus practical exponents around 1.0.
        static const uint8_t exponents[] = {0, 1, 120, 124, 127, 130};
        w[b].e = exponents[b % (sizeof(exponents) / sizeof(exponents[0]))];
        for (uint32_t j = 0; j < 16; j++) w[b].qs[j] = (uint8_t)next();
    }
    for (float &v : x) v = ((int32_t)(next() % 20001u) - 10000) / 4096.0f;

    mxfp4_block *dw = nullptr;
    float *dx = nullptr;
    float *dout = nullptr;
    HIP_OK(hipMalloc(&dw, w.size() * sizeof(w[0])));
    HIP_OK(hipMalloc(&dx, x.size() * sizeof(x[0])));
    HIP_OK(hipMalloc(&dout, sizeof(float)));
    HIP_OK(hipMemcpy(dw, w.data(), w.size() * sizeof(w[0]), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dx, x.data(), x.size() * sizeof(x[0]), hipMemcpyHostToDevice));
    mxfp4_dot_kernel<<<1, 256>>>(dout, dw, dx, n_blocks);
    HIP_OK(hipGetLastError());
    float got = 0.0f;
    HIP_OK(hipMemcpy(&got, dout, sizeof(got), hipMemcpyDeviceToHost));
    const float want = reference(w, x);
    const float abs_err = std::fabs(got - want);
    const float tolerance = 2e-5f * std::fmax(1.0f, std::fabs(want));
    std::printf("MXFP4 dot: gpu=%g cpu=%g abs_err=%g tolerance=%g\n",
                got, want, abs_err, tolerance);
    HIP_OK(hipFree(dout));
    HIP_OK(hipFree(dx));
    HIP_OK(hipFree(dw));
    return abs_err <= tolerance ? 0 : 2;
}
