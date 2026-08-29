#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../rocm/ds4_rocm_bf16_toktile.cuh"

namespace {

constexpr uint32_t kTokens = 33u;
constexpr uint32_t kThreads = kDs4Bf16ToktileThreads;

[[noreturn]] void fail(const char *what) {
    std::fprintf(stderr, "FAIL %s\n", what);
    std::exit(1);
}

void hip_ok(hipError_t status, const char *what) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "FAIL %s: %s\n", what,
                     hipGetErrorString(status));
        std::exit(1);
    }
}

void blas_ok(hipblasStatus_t status, const char *what) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL %s: hipBLAS status %d\n", what,
                     (int)status);
        std::exit(1);
    }
}

uint16_t bf16_rne(float value) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u)
        return (uint16_t)((bits >> 16u) | 0x0040u);
    const uint32_t tie_to_even = (bits >> 16u) & 1u;
    return (uint16_t)((bits + 0x00007fffu + tie_to_even) >> 16u);
}

__device__ uint16_t bf16_rne_device(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u)
        return (uint16_t)((bits >> 16u) | 0x0040u);
    const uint32_t tie_to_even = (bits >> 16u) & 1u;
    return (uint16_t)((bits + 0x00007fffu + tie_to_even) >> 16u);
}

__global__ void f32_to_bf16_kernel(uint16_t *out, const float *x,
                                    uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = bf16_rne_device(x[i]);
}

/* Diagnostic reference copied from the old production scalar kernel. The
 * candidate body itself is shared from the production header above; this arm
 * intentionally measures the pre-candidate arithmetic, not an external oracle. */
__global__ void current_bf16_f32_kernel(float *out, const uint16_t *weight,
                                         const float *x, uint32_t in_dim,
                                         uint32_t out_dim,
                                         uint32_t tokens) {
    const uint32_t row = blockIdx.x;
    const uint32_t token = blockIdx.y;
    if (row >= out_dim || token >= tokens) return;
    const uint16_t *wr = weight + (uint64_t)row * in_dim;
    const float *xr = x + (uint64_t)token * in_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < in_dim; i += blockDim.x)
        sum += __uint_as_float((uint32_t)wr[i] << 16u) * xr[i];
    __shared__ float partial[kThreads];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride)
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0u)
        out[(uint64_t)token * out_dim + row] = partial[0];
}

void launch_segmented_tiled(float *out, const uint16_t *weight,
                            const float *x, uint32_t in_dim,
                            uint32_t out_dim, uint32_t tokens) {
    uint32_t first = 0u;
    const uint32_t chunks32 = tokens / 32u;
    if (chunks32 > 0u) {
        matmul_bf16_f32_toktile_w32_kernel<32u><<<
            dim3(out_dim, chunks32), kThreads>>>(
            out, weight, x, in_dim, out_dim);
        hip_ok(hipGetLastError(), "segmented tile32 launch");
        first += chunks32 * 32u;
    }
#define LAUNCH_TAIL(T) do {                                                \
        if ((tokens - first) >= (T)) {                                     \
            matmul_bf16_f32_toktile_w32_kernel<(T)><<<out_dim, kThreads>>>(\
                out + (uint64_t)first * out_dim, weight,                   \
                x + (uint64_t)first * in_dim, in_dim, out_dim);           \
            hip_ok(hipGetLastError(), "segmented tail launch");           \
            first += (T);                                                  \
        }                                                                  \
    } while (0)
    LAUNCH_TAIL(16u);
    LAUNCH_TAIL(8u);
    LAUNCH_TAIL(4u);
    LAUNCH_TAIL(2u);
    LAUNCH_TAIL(1u);
#undef LAUNCH_TAIL
    if (first != tokens) fail("segmented tile decomposition");
}

void launch_lowrank128_tiled(float *out, const uint16_t *weight,
                             const float *x, uint32_t out_dim,
                             uint32_t tokens) {
    uint32_t first = 0u;
    const uint32_t chunks32 = tokens / 32u;
    if (chunks32 > 0u) {
        matmul_bf16_f32_lowrank128_toktile_w32_kernel<32u><<<
            dim3(out_dim, chunks32), 32u>>>(out, weight, x, out_dim);
        hip_ok(hipGetLastError(), "low-rank tile32 launch");
        first = chunks32 * 32u;
    }
#define LAUNCH_LOWRANK_TAIL(T) do {                                      \
        if ((tokens - first) >= (T)) {                                   \
            matmul_bf16_f32_lowrank128_toktile_w32_kernel<(T)><<<        \
                out_dim, 32u>>>(                                         \
                out + (uint64_t)first * out_dim, weight,                 \
                x + (uint64_t)first * 128u, out_dim);                    \
            hip_ok(hipGetLastError(), "low-rank tail launch");          \
            first += (T);                                                \
        }                                                                \
    } while (0)
    LAUNCH_LOWRANK_TAIL(16u);
    LAUNCH_LOWRANK_TAIL(8u);
    LAUNCH_LOWRANK_TAIL(4u);
    LAUNCH_LOWRANK_TAIL(2u);
    LAUNCH_LOWRANK_TAIL(1u);
#undef LAUNCH_LOWRANK_TAIL
    if (first != tokens) fail("low-rank tile decomposition");
}

struct Error {
    double nrmse;
    double cosine;
    double max_abs;
};

Error compare(const std::vector<float> &reference,
              const std::vector<float> &candidate) {
    if (reference.size() != candidate.size()) fail("output size mismatch");
    double square_error = 0.0, square_reference = 0.0;
    double dot = 0.0, square_candidate = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(candidate[i]))
            fail("non-finite output");
        const double error = (double)candidate[i] - reference[i];
        square_error += error * error;
        square_reference += (double)reference[i] * reference[i];
        square_candidate += (double)candidate[i] * candidate[i];
        dot += (double)reference[i] * candidate[i];
        max_abs = std::max(max_abs, std::abs(error));
    }
    if (!(square_reference > 0.0) || !(square_candidate > 0.0))
        fail("degenerate output norm");
    return {
        std::sqrt(square_error / square_reference),
        dot / std::sqrt(square_reference * square_candidate),
        max_abs,
    };
}

template <typename Launch>
float time_ms(Launch launch) {
    hipEvent_t begin = nullptr, end = nullptr;
    hip_ok(hipEventCreate(&begin), "create begin event");
    hip_ok(hipEventCreate(&end), "create end event");
    for (int i = 0; i < 2; ++i) launch();
    hip_ok(hipDeviceSynchronize(), "warm synchronize");
    hip_ok(hipEventRecord(begin), "record begin");
    for (int i = 0; i < 5; ++i) launch();
    hip_ok(hipEventRecord(end), "record end");
    hip_ok(hipEventSynchronize(end), "wait end");
    float elapsed = 0.0f;
    hip_ok(hipEventElapsedTime(&elapsed, begin, end), "elapsed time");
    hip_ok(hipEventDestroy(end), "destroy end event");
    hip_ok(hipEventDestroy(begin), "destroy begin event");
    return elapsed / 5.0f;
}

void run_shape(hipblasHandle_t handle, const uint16_t *weight,
               uint32_t in_dim, uint32_t out_dim) {
    const uint64_t x_count = (uint64_t)kTokens * in_dim;
    const uint64_t out_count = (uint64_t)kTokens * out_dim;
    std::vector<float> x(x_count);
    for (uint64_t i = 0; i < x_count; ++i)
        x[i] = 0.7f * std::cos((double)(i % 3571u) * 0.013) +
               0.03f * std::sin((double)i * 0.001);

    float *d_x = nullptr, *d_reference = nullptr;
    float *d_blas = nullptr, *d_exact33 = nullptr;
    float *d_segmented = nullptr, *d_lowrank = nullptr;
    uint16_t *d_x_bf16 = nullptr;
    hip_ok(hipMalloc(&d_x, x_count * sizeof(float)), "allocate F32 input");
    hip_ok(hipMalloc(&d_x_bf16, x_count * sizeof(uint16_t)),
           "allocate BF16 input");
    hip_ok(hipMalloc(&d_reference, out_count * sizeof(float)),
           "allocate reference output");
    hip_ok(hipMalloc(&d_blas, out_count * sizeof(float)),
           "allocate hipBLAS output");
    hip_ok(hipMalloc(&d_exact33, out_count * sizeof(float)),
           "allocate exact33 output");
    hip_ok(hipMalloc(&d_segmented, out_count * sizeof(float)),
           "allocate segmented output");
    if (in_dim == 128u)
        hip_ok(hipMalloc(&d_lowrank, out_count * sizeof(float)),
               "allocate low-rank output");
    hip_ok(hipMemcpy(d_x, x.data(), x_count * sizeof(float),
                     hipMemcpyHostToDevice), "copy input");

    const auto current = [&] {
        current_bf16_f32_kernel<<<dim3(out_dim, kTokens), kThreads>>>(
            d_reference, weight, d_x, in_dim, out_dim, kTokens);
        hip_ok(hipGetLastError(), "current BF16xF32 launch");
    };
    const auto exact33 = [&] {
        matmul_bf16_f32_toktile_w32_kernel<33u><<<out_dim, kThreads>>>(
            d_exact33, weight, d_x, in_dim, out_dim);
        hip_ok(hipGetLastError(), "exact33 BF16xF32 launch");
    };
    const auto segmented = [&] {
        launch_segmented_tiled(d_segmented, weight, d_x, in_dim,
                               out_dim, kTokens);
    };
    const auto lowrank = [&] {
        launch_lowrank128_tiled(d_lowrank, weight, d_x, out_dim, kTokens);
    };
    const auto blas = [&] {
        f32_to_bf16_kernel<<<(x_count + 255u) / 256u, 256u>>>(
            d_x_bf16, d_x, x_count);
        hip_ok(hipGetLastError(), "BF16 conversion launch");
        const float alpha = 1.0f;
        const float beta = 0.0f;
        blas_ok(hipblasGemmEx(
                    handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                    (int)out_dim, (int)kTokens, (int)in_dim,
                    &alpha, weight, HIP_R_16BF, (int)in_dim,
                    d_x_bf16, HIP_R_16BF, (int)in_dim,
                    &beta, d_blas, HIP_R_32F, (int)out_dim,
                    HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
                "BF16 hipBLAS GEMM");
    };

    const float current_ms = time_ms(current);
    const float exact33_ms = time_ms(exact33);
    const float segmented_ms = time_ms(segmented);
    const float lowrank_ms = in_dim == 128u ? time_ms(lowrank) : 0.0f;
    const float blas_ms = time_ms(blas);
    current();
    exact33();
    segmented();
    if (in_dim == 128u) lowrank();
    blas();
    hip_ok(hipDeviceSynchronize(), "output synchronize");
    std::vector<float> reference(out_count), exact33_output(out_count);
    std::vector<float> segmented_output(out_count);
    std::vector<float> lowrank_output;
    std::vector<float> blas_output(out_count);
    hip_ok(hipMemcpy(reference.data(), d_reference,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read reference");
    hip_ok(hipMemcpy(exact33_output.data(), d_exact33,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read exact33 output");
    hip_ok(hipMemcpy(segmented_output.data(), d_segmented,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read segmented output");
    if (in_dim == 128u) {
        lowrank_output.resize(out_count);
        hip_ok(hipMemcpy(lowrank_output.data(), d_lowrank,
                         out_count * sizeof(float), hipMemcpyDeviceToHost),
               "read low-rank output");
    }
    hip_ok(hipMemcpy(blas_output.data(), d_blas,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read hipBLAS output");
    const Error exact33_error = compare(reference, exact33_output);
    const Error segmented_error = compare(reference, segmented_output);
    if (in_dim == 128u) {
        const Error lowrank_error = compare(reference, lowrank_output);
        if (std::memcmp(reference.data(), lowrank_output.data(),
                        out_count * sizeof(float)) != 0)
            fail("low-rank token tile must bit-match scalar reduction");
        std::printf(
            "  lowrank128_ms=%.4f speedup=%.3fx nrmse=%.9g "
            "cosine=%.12g max_abs=%.9g bit_exact=1\n",
            lowrank_ms, current_ms / lowrank_ms, lowrank_error.nrmse,
            lowrank_error.cosine, lowrank_error.max_abs);
    }
    const Error blas_error = compare(reference, blas_output);
    std::printf(
        "BF16 batch GEMM shape=%ux%ux%u residency=host-registered "
        "current_ms=%.4f exact33_ms=%.4f segmented32_ms=%.4f "
        "bf16_blas_ms=%.4f exact33_speedup=%.3fx "
        "segmented_speedup=%.3fx blas_speedup=%.3fx\n",
        kTokens, out_dim, in_dim, current_ms, exact33_ms,
        segmented_ms, blas_ms, current_ms / exact33_ms,
        current_ms / segmented_ms, current_ms / blas_ms);
    std::printf(
        "  exact33 nrmse=%.9g cosine=%.12g max_abs=%.9g\n",
        exact33_error.nrmse, exact33_error.cosine,
        exact33_error.max_abs);
    std::printf(
        "  segmented32 nrmse=%.9g cosine=%.12g max_abs=%.9g\n",
        segmented_error.nrmse, segmented_error.cosine,
        segmented_error.max_abs);
    std::printf(
        "  bf16_blas nrmse=%.9g cosine=%.12g max_abs=%.9g\n",
        blas_error.nrmse, blas_error.cosine, blas_error.max_abs);

    if (d_lowrank) hip_ok(hipFree(d_lowrank), "free low-rank output");
    hip_ok(hipFree(d_segmented), "free segmented output");
    hip_ok(hipFree(d_exact33), "free exact33 output");
    hip_ok(hipFree(d_blas), "free hipBLAS output");
    hip_ok(hipFree(d_reference), "free reference output");
    hip_ok(hipFree(d_x_bf16), "free BF16 input");
    hip_ok(hipFree(d_x), "free F32 input");
}

void run_lowrank128_tail_coverage(const uint16_t *weight, uint32_t tokens) {
    constexpr uint32_t in_dim = 128u;
    constexpr uint32_t out_dim = 8192u;
    const uint64_t x_count = (uint64_t)tokens * in_dim;
    const uint64_t out_count = (uint64_t)tokens * out_dim;
    std::vector<float> x(x_count);
    for (uint64_t i = 0u; i < x_count; ++i)
        x[i] = 0.3f * std::cos((double)(i % 1877u) * 0.019) +
               0.04f * std::sin((double)i * 0.005);
    float *d_x = nullptr, *d_reference = nullptr, *d_candidate = nullptr;
    hip_ok(hipMalloc(&d_x, x_count * sizeof(float)),
           "allocate low-rank tail input");
    hip_ok(hipMalloc(&d_reference, out_count * sizeof(float)),
           "allocate low-rank tail reference");
    hip_ok(hipMalloc(&d_candidate, out_count * sizeof(float)),
           "allocate low-rank tail candidate");
    hip_ok(hipMemcpy(d_x, x.data(), x_count * sizeof(float),
                     hipMemcpyHostToDevice), "copy low-rank tail input");
    current_bf16_f32_kernel<<<dim3(out_dim, tokens), kThreads>>>(
        d_reference, weight, d_x, in_dim, out_dim, tokens);
    hip_ok(hipGetLastError(), "low-rank tail reference launch");
    launch_lowrank128_tiled(d_candidate, weight, d_x, out_dim, tokens);
    hip_ok(hipDeviceSynchronize(), "low-rank tail synchronize");
    std::vector<float> reference(out_count), candidate(out_count);
    hip_ok(hipMemcpy(reference.data(), d_reference,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read low-rank tail reference");
    hip_ok(hipMemcpy(candidate.data(), d_candidate,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read low-rank tail candidate");
    if (std::memcmp(reference.data(), candidate.data(),
                    out_count * sizeof(float)) != 0)
        fail("low-rank binary-tail bit identity");
    std::printf(
        "BF16 low-rank token-tile tail tokens=%u decomposition=32+16+8+4+2+1 "
        "bit_exact=1\n", tokens);
    hip_ok(hipFree(d_candidate), "free low-rank tail candidate");
    hip_ok(hipFree(d_reference), "free low-rank tail reference");
    hip_ok(hipFree(d_x), "free low-rank tail input");
}

void run_tail_coverage(const uint16_t *weight, uint32_t tokens) {
    constexpr uint32_t in_dim = 1024u;
    constexpr uint32_t out_dim = 1024u;
    const uint64_t x_count = (uint64_t)tokens * in_dim;
    const uint64_t out_count = (uint64_t)tokens * out_dim;
    std::vector<float> x(x_count);
    for (uint64_t i = 0; i < x_count; ++i)
        x[i] = 0.4f * std::cos((double)(i % 1291u) * 0.011) -
               0.02f * std::sin((double)i * 0.003);
    float *d_x = nullptr, *d_reference = nullptr, *d_segmented = nullptr;
    hip_ok(hipMalloc(&d_x, x_count * sizeof(float)),
           "allocate tail F32 input");
    hip_ok(hipMalloc(&d_reference, out_count * sizeof(float)),
           "allocate tail reference");
    hip_ok(hipMalloc(&d_segmented, out_count * sizeof(float)),
           "allocate tail segmented output");
    hip_ok(hipMemcpy(d_x, x.data(), x_count * sizeof(float),
                     hipMemcpyHostToDevice), "copy tail input");
    current_bf16_f32_kernel<<<dim3(out_dim, tokens), kThreads>>>(
        d_reference, weight, d_x, in_dim, out_dim, tokens);
    hip_ok(hipGetLastError(), "tail reference launch");
    launch_segmented_tiled(d_segmented, weight, d_x, in_dim, out_dim,
                           tokens);
    hip_ok(hipDeviceSynchronize(), "tail output synchronize");
    std::vector<float> reference(out_count), segmented(out_count);
    hip_ok(hipMemcpy(reference.data(), d_reference,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read tail reference");
    hip_ok(hipMemcpy(segmented.data(), d_segmented,
                     out_count * sizeof(float), hipMemcpyDeviceToHost),
           "read tail segmented output");
    const Error error = compare(reference, segmented);
    if (error.nrmse > 2.0e-7 || error.cosine < 0.999999999 ||
        error.max_abs > 1.0e-5)
        fail("segmented binary-tail accuracy");
    std::printf(
        "BF16 token-tile tail tokens=%u nrmse=%.9g cosine=%.12g "
        "max_abs=%.9g\n",
        tokens, error.nrmse, error.cosine, error.max_abs);
    hip_ok(hipFree(d_segmented), "free tail segmented output");
    hip_ok(hipFree(d_reference), "free tail reference");
    hip_ok(hipFree(d_x), "free tail input");
}

}  // namespace

int main() {
    constexpr uint64_t weight_count = (uint64_t)4096u * 8192u;
    constexpr uint64_t weight_bytes = weight_count * sizeof(uint16_t);
    void *host_allocation = nullptr;
    if (posix_memalign(&host_allocation, 4096u, weight_bytes) != 0 ||
        !host_allocation)
        fail("allocate page-aligned host weights");
    auto *host_weight = static_cast<uint16_t *>(host_allocation);
    for (uint64_t i = 0; i < weight_count; ++i) {
        const float value = 0.035f * std::sin((double)(i % 7919u) * 0.017);
        host_weight[i] = bf16_rne(value);
    }
    hip_ok(hipHostRegister(host_weight, weight_bytes,
                           hipHostRegisterMapped | hipHostRegisterReadOnly),
           "register mapped read-only weights");
    uint16_t *device_weight = nullptr;
    hip_ok(hipHostGetDevicePointer(
               reinterpret_cast<void **>(&device_weight), host_weight, 0u),
           "resolve mapped weight pointer");
    hipblasHandle_t handle = nullptr;
    blas_ok(hipblasCreate(&handle), "create hipBLAS handle");

    /* GLM-5.3 KDA f_b/g_b expand a 128-wide low-rank state into all 8192
     * channels. This shape was intentionally below the original production
     * token-tile gate, so keep it in the three-arm diagnostic before changing
     * dispatch. */
    run_shape(handle, device_weight, 128u, 8192u);
    run_shape(handle, device_weight, 4096u, 8192u);
    run_shape(handle, device_weight, 8192u, 4096u);
    run_lowrank128_tail_coverage(device_weight, 63u);
    for (const uint32_t tokens : {16u, 17u, 31u, 47u, 64u})
        run_tail_coverage(device_weight, tokens);

    blas_ok(hipblasDestroy(handle), "destroy hipBLAS handle");
    hip_ok(hipHostUnregister(host_weight), "unregister weights");
    std::free(host_weight);
    std::puts("PASS BF16 real-shape three-arm diagnostic");
    return 0;
}
