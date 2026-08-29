#include "tests/glm5_gguf_test.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(expr, message) do {                                         \
    if (!(expr)) {                                                        \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__); \
        return false;                                                     \
    }                                                                     \
} while (0)

namespace {

bool hip_ok(hipError_t status, const char *what) {
    if (status == hipSuccess) return true;
    std::fprintf(stderr, "FAIL %s: %s\n", what, hipGetErrorString(status));
    return false;
}

__device__ __forceinline__ float bf16_to_f32(uint16_t value) {
    return __uint_as_float((uint32_t)value << 16u);
}

__device__ __forceinline__ float wave_sum_f32(float value) {
    for (uint32_t delta = 16u; delta != 0u; delta >>= 1u)
        value += __shfl_down(value, delta, 32u);
    return value;
}

template <uint32_t RowsPerBlock>
__global__ void bf16_decode_reference(float *out, const uint16_t *weight,
                                      const float *x, uint32_t in_dim,
                                      uint32_t out_dim) {
    extern __shared__ float shared_x[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    for (uint32_t i = tid; i < in_dim; i += blockDim.x) shared_x[i] = x[i];
    __syncthreads();
    const uint32_t row = blockIdx.x * RowsPerBlock + wave;
    if (row >= out_dim) return;
    const uint16_t *wr = weight + (uint64_t)row * in_dim;
    float acc = 0.0f;
    for (uint32_t i = lane; i < in_dim; i += 32u)
        acc += bf16_to_f32(wr[i]) * shared_x[i];
    acc = wave_sum_f32(acc);
    if (lane == 0u) out[row] = acc;
}

template <uint32_t RowsPerBlock, uint32_t Unroll>
__global__ void bf16_decode_mlp(float *out, const uint16_t *weight,
                               const float *x, uint32_t in_dim,
                               uint32_t out_dim) {
    extern __shared__ float shared_x[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    for (uint32_t i = tid; i < in_dim; i += blockDim.x) shared_x[i] = x[i];
    __syncthreads();
    const uint32_t row = blockIdx.x * RowsPerBlock + wave;
    if (row >= out_dim) return;
    const uint16_t *wr = weight + (uint64_t)row * in_dim;
    float acc = 0.0f;
    uint32_t i = lane;
    for (; i + (Unroll - 1u) * 32u < in_dim; i += Unroll * 32u) {
        uint16_t packed_w[Unroll];
        float packed_x[Unroll];
#pragma unroll
        for (uint32_t u = 0u; u < Unroll; ++u) {
            const uint32_t index = i + u * 32u;
            packed_w[u] = wr[index];
            packed_x[u] = shared_x[index];
        }
#pragma unroll
        for (uint32_t u = 0u; u < Unroll; ++u)
            acc += bf16_to_f32(packed_w[u]) * packed_x[u];
    }
    for (; i < in_dim; i += 32u)
        acc += bf16_to_f32(wr[i]) * shared_x[i];
    acc = wave_sum_f32(acc);
    if (lane == 0u) out[row] = acc;
}

template <uint32_t RowsPerWave, uint32_t Unroll>
__global__ void bf16_decode_multirow_mlp(float *out, const uint16_t *weight,
                                        const float *x, uint32_t in_dim,
                                        uint32_t out_dim) {
    static_assert(32u % RowsPerWave == 0u, "whole rows per block");
    constexpr uint32_t waves_per_block = 32u / RowsPerWave;
    extern __shared__ float shared_x[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    for (uint32_t i = tid; i < in_dim; i += blockDim.x) shared_x[i] = x[i];
    __syncthreads();
    const uint32_t row0 = blockIdx.x * 32u + wave * RowsPerWave;
    float acc[RowsPerWave] = {};
    uint32_t i = lane;
    for (; i + (Unroll - 1u) * 32u < in_dim; i += Unroll * 32u) {
        float packed_x[Unroll];
        uint16_t packed_w[RowsPerWave][Unroll];
#pragma unroll
        for (uint32_t u = 0u; u < Unroll; ++u)
            packed_x[u] = shared_x[i + u * 32u];
#pragma unroll
        for (uint32_t r = 0u; r < RowsPerWave; ++r) {
            const uint32_t row = row0 + r;
            if (row < out_dim) {
                const uint16_t *wr = weight + (uint64_t)row * in_dim;
#pragma unroll
                for (uint32_t u = 0u; u < Unroll; ++u)
                    packed_w[r][u] = wr[i + u * 32u];
            }
        }
#pragma unroll
        for (uint32_t u = 0u; u < Unroll; ++u) {
#pragma unroll
            for (uint32_t r = 0u; r < RowsPerWave; ++r)
                if (row0 + r < out_dim)
                    acc[r] += bf16_to_f32(packed_w[r][u]) * packed_x[u];
        }
    }
    for (; i < in_dim; i += 32u) {
        const float xv = shared_x[i];
#pragma unroll
        for (uint32_t r = 0u; r < RowsPerWave; ++r) {
            const uint32_t row = row0 + r;
            if (row < out_dim)
                acc[r] += bf16_to_f32(weight[(uint64_t)row * in_dim + i]) * xv;
        }
    }
#pragma unroll
    for (uint32_t r = 0u; r < RowsPerWave; ++r) {
        const uint32_t row = row0 + r;
        const float sum = wave_sum_f32(acc[r]);
        if (lane == 0u && row < out_dim) out[row] = sum;
    }
    (void)waves_per_block;
}

template <uint32_t RowsPerBlock, uint32_t Unroll>
bool launch_candidate(float *out, const uint16_t *weight, const float *x,
                      uint32_t in_dim, uint32_t out_dim) {
    bf16_decode_mlp<RowsPerBlock, Unroll><<<
        (out_dim + RowsPerBlock - 1u) / RowsPerBlock,
        RowsPerBlock * 32u, (size_t)in_dim * sizeof(float)>>>(
            out, weight, x, in_dim, out_dim);
    return hip_ok(hipGetLastError(), "candidate BF16 launch");
}

template <uint32_t RowsPerWave, uint32_t Unroll>
bool launch_multirow(float *out, const uint16_t *weight, const float *x,
                     uint32_t in_dim, uint32_t out_dim) {
    constexpr uint32_t waves_per_block = 32u / RowsPerWave;
    bf16_decode_multirow_mlp<RowsPerWave, Unroll><<<
        (out_dim + 31u) / 32u, waves_per_block * 32u,
        (size_t)in_dim * sizeof(float)>>>(out, weight, x, in_dim, out_dim);
    return hip_ok(hipGetLastError(), "multirow BF16 launch");
}

bool launch_reference(float *out, const uint16_t *weight, const float *x,
                      uint32_t in_dim, uint32_t out_dim) {
    constexpr uint32_t rows = 32u;
    bf16_decode_reference<rows><<<
        (out_dim + rows - 1u) / rows, rows * 32u,
        (size_t)in_dim * sizeof(float)>>>(out, weight, x, in_dim, out_dim);
    return hip_ok(hipGetLastError(), "reference BF16 launch");
}

struct DeviceBuffer {
    void *ptr = nullptr;
    ~DeviceBuffer() { if (ptr) (void)hipFree(ptr); }
    bool allocate(uint64_t bytes) {
        return bytes <= SIZE_MAX &&
               hip_ok(hipMalloc(&ptr, (size_t)bytes), "device allocation");
    }
};

uint64_t fnv64(const void *data, uint64_t bytes) {
    const auto *p = static_cast<const uint8_t *>(data);
    uint64_t hash = UINT64_C(1469598103934665603);
    for (uint64_t i = 0u; i < bytes; ++i) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

bool tensor_offset(const Glm5TestGGUF &gguf, uint32_t layer,
                   const char *suffix, const std::vector<uint64_t> &dims,
                   uint64_t &offset) {
    char name[96];
    const int length = std::snprintf(name, sizeof(name), "blk.%u.%s",
                                     layer, suffix);
    return length > 0 && (size_t)length < sizeof(name) &&
           gguf.tensor(name, dims, 30u, offset);
}

struct Shape {
    const char *name;
    uint32_t in_dim;
    uint32_t out_dim;
    uint64_t offset;
};

using Launch = bool (*)(float *, const uint16_t *, const float *, uint32_t,
                        uint32_t);

double time_launch(Launch launch, float *out, const uint16_t *weight,
                   const float *x, uint32_t in_dim, uint32_t out_dim) {
    constexpr uint32_t warmup = 3u;
    constexpr uint32_t repeats = 12u;
    for (uint32_t i = 0u; i < warmup; ++i)
        if (!launch(out, weight, x, in_dim, out_dim)) return -1.0;
    hipEvent_t begin = nullptr, end = nullptr;
    if (!hip_ok(hipEventCreate(&begin), "create begin event") ||
        !hip_ok(hipEventCreate(&end), "create end event")) return -1.0;
    if (!hip_ok(hipEventRecord(begin), "record begin event")) return -1.0;
    for (uint32_t i = 0u; i < repeats; ++i)
        if (!launch(out, weight, x, in_dim, out_dim)) return -1.0;
    if (!hip_ok(hipEventRecord(end), "record end event") ||
        !hip_ok(hipEventSynchronize(end), "wait timing event")) return -1.0;
    float elapsed = 0.0f;
    if (!hip_ok(hipEventElapsedTime(&elapsed, begin, end), "read timing"))
        return -1.0;
    (void)hipEventDestroy(end);
    (void)hipEventDestroy(begin);
    return elapsed / repeats;
}

bool run_shape(const Glm5TestGGUF &gguf, const Shape &shape, bool benchmark,
               double *reference_ms_out = nullptr,
               double *best_ms_out = nullptr) {
    const uint64_t weight_bytes =
        (uint64_t)shape.in_dim * shape.out_dim * sizeof(uint16_t);
    const uint64_t out_bytes = (uint64_t)shape.out_dim * sizeof(float);
    CHECK(shape.offset <= gguf.size && weight_bytes <= gguf.size - shape.offset,
          "bounded real-GGUF BF16 tensor");
    std::vector<float> input(shape.in_dim);
    for (uint32_t i = 0u; i < shape.in_dim; ++i) {
        const int32_t centered =
            (int32_t)((i * 73u + shape.out_dim) % 509u) - 254;
        input[i] = (float)centered * (1.0f / 2048.0f);
    }
    DeviceBuffer weight, x, reference, candidate;
    CHECK(weight.allocate(weight_bytes) &&
          x.allocate((uint64_t)shape.in_dim * sizeof(float)) &&
          reference.allocate(out_bytes) && candidate.allocate(out_bytes),
          "allocate BF16 component buffers");
    CHECK(hip_ok(hipMemcpy(weight.ptr, gguf.map + shape.offset,
                           (size_t)weight_bytes, hipMemcpyHostToDevice),
                 "upload BF16 weights") &&
          hip_ok(hipMemcpy(x.ptr, input.data(), input.size() * sizeof(float),
                           hipMemcpyHostToDevice), "upload BF16 input") &&
          launch_reference((float *)reference.ptr,
                           (const uint16_t *)weight.ptr,
                           (const float *)x.ptr, shape.in_dim, shape.out_dim) &&
          hip_ok(hipDeviceSynchronize(), "reference synchronization"),
          "execute reference BF16 component");

    struct Variant { const char *name; Launch launch; };
    const Variant variants[] = {
        {"u4-r32", launch_candidate<32u, 4u>},
        {"u8-r32", launch_candidate<32u, 8u>},
        {"u16-r32", launch_candidate<32u, 16u>},
        {"u32-r32", launch_candidate<32u, 32u>},
        {"u64-r32", launch_candidate<32u, 64u>},
        {"u4-r16", launch_candidate<16u, 4u>},
        {"u8-r16", launch_candidate<16u, 8u>},
        {"rpw2-u4", launch_multirow<2u, 4u>},
        {"rpw2-u8", launch_multirow<2u, 8u>},
        {"rpw4-u4", launch_multirow<4u, 4u>},
        {"rpw4-u8", launch_multirow<4u, 8u>},
    };
    std::vector<float> expected(shape.out_dim), got(shape.out_dim);
    CHECK(hip_ok(hipMemcpy(expected.data(), reference.ptr, (size_t)out_bytes,
                           hipMemcpyDeviceToHost), "read BF16 reference"),
          "read reference output");
    double best_ms = 1.0e30;
    const char *best_name = nullptr;
    for (const Variant &variant : variants) {
        CHECK(variant.launch((float *)candidate.ptr,
                             (const uint16_t *)weight.ptr,
                             (const float *)x.ptr,
                             shape.in_dim, shape.out_dim) &&
              hip_ok(hipDeviceSynchronize(), "candidate synchronization") &&
              hip_ok(hipMemcpy(got.data(), candidate.ptr, (size_t)out_bytes,
                               hipMemcpyDeviceToHost), "read BF16 candidate"),
              "execute candidate BF16 component");
        CHECK(std::memcmp(expected.data(), got.data(), (size_t)out_bytes) == 0,
              "candidate BF16 output is byte-identical");
        if (!benchmark) continue;
        const double ms = time_launch(
            variant.launch, (float *)candidate.ptr,
            (const uint16_t *)weight.ptr, (const float *)x.ptr,
            shape.in_dim, shape.out_dim);
        CHECK(ms > 0.0, "measure candidate BF16 component");
        if (ms < best_ms) { best_ms = ms; best_name = variant.name; }
        std::fprintf(stderr,
                     "BF16 MLP shape=%s variant=%s ms=%.6f fnv=%016llx\n",
                     shape.name, variant.name, ms,
                     (unsigned long long)fnv64(got.data(), out_bytes));
    }
    if (benchmark) {
        const double reference_ms = time_launch(
            launch_reference, (float *)reference.ptr,
            (const uint16_t *)weight.ptr, (const float *)x.ptr,
            shape.in_dim, shape.out_dim);
        CHECK(reference_ms > 0.0 && best_name, "measure reference BF16 component");
        std::fprintf(stderr,
                     "PASS BF16 MLP shape=%s reference_ms=%.6f best=%s "
                     "best_ms=%.6f speedup=%.3fx\n",
                     shape.name, reference_ms, best_name, best_ms,
                     reference_ms / best_ms);
        if (reference_ms_out) *reference_ms_out = reference_ms;
        if (best_ms_out) *best_ms_out = best_ms;
    }
    return true;
}

bool run_test() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    CHECK(model && model[0], "DS4_GLM5_MODEL environment");
    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM5 GGUF");
    CHECK(hip_ok(hipSetDevice(0), "select gfx1151"), "select test device");

    uint32_t seen = 0u;
    for (uint32_t layer = 0u; layer < 45u; ++layer) {
        if ((layer & 3u) == 3u) continue;
        uint64_t q = 0u, output = 0u, f_b = 0u;
        CHECK(tensor_offset(gguf, layer, "kda_q.weight", {4096u, 8192u}, q) &&
              tensor_offset(gguf, layer, "kda_output.weight",
                            {8192u, 4096u}, output) &&
              tensor_offset(gguf, layer, "kda_f_b.weight",
                            {128u, 8192u}, f_b),
              "bind all real KDA BF16 component tensors");
        const Shape shapes[] = {
            {"q-rank0", 4096u, 4096u, q},
            {"output", 8192u, 4096u, output},
            {"f_b-rank0", 128u, 4096u, f_b},
        };
        for (const Shape &shape : shapes) {
            double reference_ms = 0.0, best_ms = 0.0;
            CHECK(run_shape(gguf, shape, layer == 0u,
                            &reference_ms, &best_ms),
                  "validate real KDA BF16 MLP shape");
            if (layer == 0u && std::strcmp(shape.name, "output") == 0)
                CHECK(reference_ms / best_ms >= 1.40,
                      "8192-to-4096 BF16 candidate reaches 1.4x");
        }
        ++seen;
    }
    CHECK(seen == 34u, "validate exactly 34 KDA layers");

    uint64_t vocab = 0u;
    CHECK(gguf.tensor("output.weight", {4096u, 154880u}, 30u, vocab),
          "bind real vocabulary BF16 tensor");
    CHECK(run_shape(gguf, {"vocab", 4096u, 154880u, vocab}, true),
          "validate vocabulary BF16 MLP shape");
    std::fprintf(stderr,
                 "PASS real-GGUF exact-order BF16 decode MLP gate layers=%u\n",
                 seen);
    return true;
}

}  // namespace

int main() { return run_test() ? 0 : 1; }
