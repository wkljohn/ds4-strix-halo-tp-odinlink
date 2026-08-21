// Exact-arithmetic, model-free oracle for the ordinary one-token HC pre-chain.
// It compares the shipped three-launch arithmetic with two cooperative twins:
// one retaining the F32 normalized scratch and one materializing the normalized
// value only in a register before the unchanged ordered dot product.
//
// Build:
//   /opt/rocm/bin/hipcc -O3 -ffast-math -fno-finite-math-only \
//     --offload-arch=gfx1151 scripts/hc_stage_exact_coop_bench.cu \
//     -o /tmp/hc_stage_exact_coop_bench

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace cg = cooperative_groups;

static constexpr uint32_t kEmbd = 4096;
static constexpr uint32_t kHc = 4;
static constexpr uint32_t kInput = kEmbd * kHc;
static constexpr uint32_t kMix = 24;
static constexpr uint32_t kThreads = 256;
static constexpr uint32_t kLayers = 43;
static constexpr uint32_t kSinkhorn = 20;
static constexpr uint32_t kIterations = kLayers * 80;
static constexpr float kRmsEps = 1.0e-6f;
static constexpr float kHcEps = 1.0e-6f;

static void check(hipError_t rc, const char *what) {
    if (rc != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(rc));
        std::exit(1);
    }
}

__device__ static void hc4_split_one(
        float *out, const float *mix, const float *scale, const float *base) {
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    for (int i = 0; i < 4; ++i) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + kHcEps;
    }
    for (int i = 0; i < 4; ++i) {
        const float z = mix[4 + i] * post_scale + base[4 + i];
        out[4 + i] = 2.0f / (1.0f + expf(-z));
    }
    float c[16];
    for (int r = 0; r < 4; ++r) {
        float m = -INFINITY;
        for (int col = 0; col < 4; ++col) {
            const float v = mix[8 + r * 4 + col] * comb_scale +
                            base[8 + r * 4 + col];
            c[r * 4 + col] = v;
            m = fmaxf(m, v);
        }
        float s = 0.0f;
        for (int col = 0; col < 4; ++col) {
            const float v = expf(c[r * 4 + col] - m);
            c[r * 4 + col] = v;
            s += v;
        }
        for (int col = 0; col < 4; ++col) {
            c[r * 4 + col] = c[r * 4 + col] / s + kHcEps;
        }
    }
    for (int col = 0; col < 4; ++col) {
        float s = kHcEps;
        for (int r = 0; r < 4; ++r) s += c[r * 4 + col];
        for (int r = 0; r < 4; ++r) c[r * 4 + col] /= s;
    }
    for (uint32_t it = 1; it < kSinkhorn; ++it) {
        for (int r = 0; r < 4; ++r) {
            float s = kHcEps;
            for (int col = 0; col < 4; ++col) s += c[r * 4 + col];
            for (int col = 0; col < 4; ++col) c[r * 4 + col] /= s;
        }
        for (int col = 0; col < 4; ++col) {
            float s = kHcEps;
            for (int r = 0; r < 4; ++r) s += c[r * 4 + col];
            for (int r = 0; r < 4; ++r) c[r * 4 + col] /= s;
        }
    }
    for (int i = 0; i < 16; ++i) out[8 + i] = c[i];
}

__global__ static void rms_plain(float *out, const float *x) {
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < kInput; i += blockDim.x) {
        const float v = x[i];
        sum += v * v;
    }
    __shared__ float partial[kThreads];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = kThreads / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float s = rsqrtf(partial[0] / (float)kInput + kRmsEps);
    for (uint32_t i = threadIdx.x; i < kInput; i += blockDim.x) out[i] = x[i] * s;
}

__global__ static void ordered_f16_projection(
        float *out, const __half *w, const float *x) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t chunk = (kInput + 31u) / 32u;
    const uint32_t k0 = tid * chunk;
    const uint32_t k1 = min(k0 + chunk, kInput);
    const __half *wr = w + (uint64_t)row * kInput;
    float sum = 0.0f;
    for (uint32_t i = k0; i < k1; ++i) sum += __half2float(wr[i]) * x[i];
    __shared__ float partial[32];
    partial[tid] = sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (uint32_t i = 0; i < 32; ++i) total += partial[i];
        out[row] = total;
    }
}

__device__ static void hc_tail(
        float *out, float *norm_out, float *split, const float *mix,
        const float *residual, const float *scale, const float *base,
        const float *norm_w) {
    const uint32_t d = threadIdx.x;
    if (d == 0) hc4_split_one(split, mix, scale, base);
    __syncthreads();
    float sum = 0.0f;
    for (uint32_t col = d; col < kEmbd; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < kHc; ++h) acc += residual[(uint64_t)h * kEmbd + col] * split[h];
        out[col] = acc;
        sum += acc * acc;
    }
    __shared__ float partial[kThreads];
    partial[d] = sum;
    __syncthreads();
    for (uint32_t stride = kThreads / 2; stride; stride >>= 1) {
        if (d < stride) partial[d] += partial[d + stride];
        __syncthreads();
    }
    const float ns = rsqrtf(partial[0] / (float)kEmbd + kRmsEps);
    for (uint32_t col = d; col < kEmbd; col += blockDim.x) {
        norm_out[col] = out[col] * ns * norm_w[col];
    }
}

__global__ static void hc_tail_kernel(
        float *out, float *norm_out, float *split, const float *mix,
        const float *residual, const float *scale, const float *base,
        const float *norm_w) {
    hc_tail(out, norm_out, split, mix, residual, scale, base, norm_w);
}

template <bool STORE_FLAT>
__global__ static void exact_hc_stage_cooperative(
        float *out, float *norm_out, float *flat, float *mix, float *split,
        float *rms_scale, const __half *w, const float *residual,
        const float *scale, const float *base, const float *norm_w) {
    cg::grid_group grid = cg::this_grid();
    const uint32_t tid = threadIdx.x;
    if (blockIdx.x == 0) {
        float sum = 0.0f;
        for (uint32_t i = tid; i < kInput; i += blockDim.x) {
            const float v = residual[i];
            sum += v * v;
        }
        __shared__ float rms_partial[kThreads];
        rms_partial[tid] = sum;
        __syncthreads();
        for (uint32_t stride = kThreads / 2; stride; stride >>= 1) {
            if (tid < stride) rms_partial[tid] += rms_partial[tid + stride];
            __syncthreads();
        }
        if (tid == 0) rms_scale[0] = rsqrtf(rms_partial[0] / (float)kInput + kRmsEps);
        __syncthreads();
        if (STORE_FLAT) {
            for (uint32_t i = tid; i < kInput; i += blockDim.x) {
                flat[i] = residual[i] * rms_scale[0];
            }
        }
    }
    grid.sync();

    const uint32_t row = blockIdx.x;
    __shared__ float dot_partial[32];
    if (tid < 32u) {
        const uint32_t chunk = (kInput + 31u) / 32u;
        const uint32_t k0 = tid * chunk;
        const uint32_t k1 = min(k0 + chunk, kInput);
        const __half *wr = w + (uint64_t)row * kInput;
        float sum = 0.0f;
        for (uint32_t i = k0; i < k1; ++i) {
            float xv;
            if (STORE_FLAT) {
                xv = flat[i];
            } else {
                xv = residual[i] * rms_scale[0];
                asm volatile("" : "+v"(xv));
            }
            sum += __half2float(wr[i]) * xv;
        }
        dot_partial[tid] = sum;
    }
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (uint32_t i = 0; i < 32u; ++i) total += dot_partial[i];
        mix[row] = total;
    }
    grid.sync();
    if (blockIdx.x == 0) hc_tail(out, norm_out, split, mix, residual, scale, base, norm_w);
}

struct buffers {
    __half *w{};
    float *residual{}, *scale{}, *base{}, *norm_w{};
    float *b_flat{}, *b_mix{}, *b_split{}, *b_out{}, *b_norm{};
    float *c_flat{}, *c_mix{}, *c_split{}, *c_out{}, *c_norm{}, *c_rms{};
    float *i_flat{}, *i_mix{}, *i_split{}, *i_out{}, *i_norm{}, *i_rms{};
};

static void launch_base(const buffers &b, uint32_t layer) {
    const __half *w = b.w + (uint64_t)(layer % kLayers) * kMix * kInput;
    rms_plain<<<1, kThreads>>>(b.b_flat, b.residual);
    ordered_f16_projection<<<kMix, 32>>>(b.b_mix, w, b.b_flat);
    hc_tail_kernel<<<1, kThreads>>>(b.b_out, b.b_norm, b.b_split, b.b_mix,
                                    b.residual, b.scale, b.base, b.norm_w);
}

template <bool STORE_FLAT>
static void launch_coop(const buffers &b, uint32_t layer) {
    const __half *w = b.w + (uint64_t)(layer % kLayers) * kMix * kInput;
    float *out = STORE_FLAT ? b.c_out : b.i_out;
    float *norm = STORE_FLAT ? b.c_norm : b.i_norm;
    float *flat = STORE_FLAT ? b.c_flat : b.i_flat;
    float *mix = STORE_FLAT ? b.c_mix : b.i_mix;
    float *split = STORE_FLAT ? b.c_split : b.i_split;
    float *rms = STORE_FLAT ? b.c_rms : b.i_rms;
    void *args[] = {&out, &norm, &flat, &mix, &split, &rms,
                    &w, const_cast<float **>(&b.residual),
                    const_cast<float **>(&b.scale), const_cast<float **>(&b.base),
                    const_cast<float **>(&b.norm_w)};
    check(hipLaunchCooperativeKernel(
                  reinterpret_cast<const void *>(exact_hc_stage_cooperative<STORE_FLAT>),
                  dim3(kMix), dim3(kThreads), args, 0, nullptr),
          STORE_FLAT ? "cooperative flat launch" : "cooperative inline launch");
}

template <typename F>
static float time_us(F fn) {
    hipEvent_t begin{}, end{};
    check(hipEventCreate(&begin), "event begin");
    check(hipEventCreate(&end), "event end");
    for (uint32_t i = 0; i < kLayers; ++i) fn(i);
    check(hipDeviceSynchronize(), "warm sync");
    check(hipEventRecord(begin), "record begin");
    for (uint32_t i = 0; i < kIterations; ++i) fn(i);
    check(hipEventRecord(end), "record end");
    check(hipEventSynchronize(end), "wait end");
    float ms = 0.0f;
    check(hipEventElapsedTime(&ms, begin, end), "elapsed");
    (void)hipEventDestroy(begin); (void)hipEventDestroy(end);
    return 1000.0f * ms / (float)kIterations;
}

static void alloc(float **p, size_t n, const char *what) {
    check(hipMalloc(p, n * sizeof(float)), what);
}

int main() {
    buffers b{};
    std::mt19937 gen(1701);
    std::uniform_real_distribution<float> dist(-0.08f, 0.08f);
    std::vector<__half> hw((uint64_t)kLayers * kMix * kInput);
    std::vector<float> hx(kInput), hs(3), hb(kMix), hn(kEmbd);
    for (auto &v : hw) v = __float2half(dist(gen));
    for (auto &v : hx) v = dist(gen);
    for (auto &v : hs) v = 0.5f + std::fabs(dist(gen));
    for (auto &v : hb) v = dist(gen);
    for (auto &v : hn) v = 0.9f + dist(gen);
    check(hipMalloc(&b.w, hw.size() * sizeof(__half)), "weights alloc");
    alloc(&b.residual, kInput, "residual alloc"); alloc(&b.scale, 3, "scale alloc");
    alloc(&b.base, kMix, "base alloc"); alloc(&b.norm_w, kEmbd, "norm alloc");
    check(hipMemcpy(b.w, hw.data(), hw.size() * sizeof(__half), hipMemcpyHostToDevice), "weights copy");
    check(hipMemcpy(b.residual, hx.data(), hx.size() * sizeof(float), hipMemcpyHostToDevice), "residual copy");
    check(hipMemcpy(b.scale, hs.data(), hs.size() * sizeof(float), hipMemcpyHostToDevice), "scale copy");
    check(hipMemcpy(b.base, hb.data(), hb.size() * sizeof(float), hipMemcpyHostToDevice), "base copy");
    check(hipMemcpy(b.norm_w, hn.data(), hn.size() * sizeof(float), hipMemcpyHostToDevice), "norm copy");
    float **all[] = {&b.b_flat,&b.b_mix,&b.b_split,&b.b_out,&b.b_norm,
                    &b.c_flat,&b.c_mix,&b.c_split,&b.c_out,&b.c_norm,&b.c_rms,
                    &b.i_flat,&b.i_mix,&b.i_split,&b.i_out,&b.i_norm,&b.i_rms};
    const size_t sizes[] = {kInput,kMix,kMix,kEmbd,kEmbd,kInput,kMix,kMix,kEmbd,kEmbd,1,
                            kInput,kMix,kMix,kEmbd,kEmbd,1};
    for (uint32_t i = 0; i < sizeof(all)/sizeof(all[0]); ++i) alloc(all[i], sizes[i], "scratch alloc");

    launch_base(b, 0); launch_coop<true>(b, 0); launch_coop<false>(b, 0);
    check(hipDeviceSynchronize(), "correctness sync");
    std::vector<float> ref(kEmbd), flat(kEmbd), inl(kEmbd);
    check(hipMemcpy(ref.data(), b.b_norm, ref.size()*sizeof(float), hipMemcpyDeviceToHost), "read ref");
    check(hipMemcpy(flat.data(), b.c_norm, flat.size()*sizeof(float), hipMemcpyDeviceToHost), "read flat");
    check(hipMemcpy(inl.data(), b.i_norm, inl.size()*sizeof(float), hipMemcpyDeviceToHost), "read inline");
    const bool flat_exact = std::memcmp(ref.data(), flat.data(), ref.size()*sizeof(float)) == 0;
    const bool inline_exact = std::memcmp(ref.data(), inl.data(), ref.size()*sizeof(float)) == 0;
    const float base_us = time_us([&](uint32_t i){ launch_base(b,i); });
    const float flat_us = time_us([&](uint32_t i){ launch_coop<true>(b,i); });
    const float inline_us = time_us([&](uint32_t i){ launch_coop<false>(b,i); });
    std::printf("shape=%ux%u layers=%u iterations=%u\n", kInput, kMix, kLayers, kIterations);
    std::printf("flat_exact=%d inline_exact=%d\n", flat_exact, inline_exact);
    std::printf("baseline_us=%.3f coop_flat_us=%.3f change=%+.1f%% "
                "coop_inline_us=%.3f change=%+.1f%%\n", base_us, flat_us,
                100.0f*(flat_us/base_us-1.0f), inline_us,
                100.0f*(inline_us/base_us-1.0f));
    return flat_exact && inline_exact ? 0 : 2;
}
