// Standalone correctness and timing harness for the routed Q4_K gate/up
// epilogue.  It compares the current pair-thread/serial-row layout with a
// row-thread/coalesced-pair layout before any production integration.
//
// Build:
//   /opt/rocm/bin/hipcc -O3 -ffast-math -fno-finite-math-only \
//     --offload-arch=gfx1151 scripts/moe_epilogue_layout_bench.cu \
//     -o /tmp/moe_epilogue_layout_bench

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void check(hipError_t rc, const char *where) {
    if (rc != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, hipGetErrorString(rc));
        std::exit(1);
    }
}

__device__ __forceinline__ static float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ static void serial_rows_kernel(
        float *gate, float *up, float *mid,
        const uint32_t *pairs, const uint32_t *offsets,
        const uint32_t *counts, const uint32_t *tile_total,
        const uint32_t *tile_experts, const uint32_t *tile_starts,
        const float *weights, uint32_t nrows, uint32_t write_gate_up,
        float clamp) {
    const uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    const uint32_t expert = tile_experts[tile];
    const uint32_t local = tile_starts[tile] + threadIdx.x;
    if (local >= counts[expert]) return;
    const uint32_t pair = pairs[offsets[expert] + local];
    for (uint32_t row = 0; row < nrows; ++row) {
        const uint64_t off = (uint64_t)pair * nrows + row;
        float g = gate[off], u = up[off];
        if (clamp > 1.0e-6f) {
            g = fminf(g, clamp);
            u = fmaxf(-clamp, fminf(u, clamp));
        }
        if (write_gate_up) { gate[off] = g; up[off] = u; }
        mid[off] = silu(g) * u * weights[pair];
    }
}

__global__ static void coalesced_rows_kernel(
        float *gate, float *up, float *mid,
        const uint32_t *pairs, const uint32_t *offsets,
        const uint32_t *counts, const uint32_t *tile_total,
        const uint32_t *tile_experts, const uint32_t *tile_starts,
        const float *weights, uint32_t nrows, uint32_t write_gate_up,
        float clamp) {
    const uint32_t tile = blockIdx.y;
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile >= *tile_total || row >= nrows) return;
    const uint32_t expert = tile_experts[tile];
    const uint32_t start = tile_starts[tile];
    const uint32_t count = counts[expert];
    const uint32_t np = min(16u, count > start ? count - start : 0u);
    for (uint32_t p = 0; p < np; ++p) {
        const uint32_t pair = pairs[offsets[expert] + start + p];
        const uint64_t off = (uint64_t)pair * nrows + row;
        float g = gate[off], u = up[off];
        if (clamp > 1.0e-6f) {
            g = fminf(g, clamp);
            u = fmaxf(-clamp, fminf(u, clamp));
        }
        if (write_gate_up) { gate[off] = g; up[off] = u; }
        mid[off] = silu(g) * u * weights[pair];
    }
}

struct Buffers {
    uint32_t npair, nrows, ntiles;
    std::vector<float> gate, up, weights;
    float *d_gate_a, *d_up_a, *d_mid_a, *d_gate_b, *d_up_b, *d_mid_b, *d_weights;
    uint32_t *d_pairs, *d_offsets, *d_counts, *d_total, *d_experts, *d_starts;

    Buffers(uint32_t pairs, uint32_t rows) : npair(pairs), nrows(rows), ntiles((pairs + 15) / 16),
        gate((uint64_t)pairs * rows), up((uint64_t)pairs * rows), weights(pairs) {
        std::mt19937 rng(11); std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (float &x : gate) x = dist(rng);
        for (float &x : up) x = dist(rng);
        for (float &x : weights) x = dist(rng);
        std::vector<uint32_t> p(npair), offsets{0u, npair}, counts{npair};
        std::vector<uint32_t> experts(ntiles, 0u), starts(ntiles);
        for (uint32_t i = 0; i < npair; ++i) p[i] = i;
        for (uint32_t i = 0; i < ntiles; ++i) starts[i] = i * 16u;
        const size_t elems = (size_t)npair * nrows, bytes = elems * sizeof(float);
        check(hipMalloc(&d_gate_a, bytes), "gate a"); check(hipMalloc(&d_up_a, bytes), "up a");
        check(hipMalloc(&d_mid_a, bytes), "mid a"); check(hipMalloc(&d_gate_b, bytes), "gate b");
        check(hipMalloc(&d_up_b, bytes), "up b"); check(hipMalloc(&d_mid_b, bytes), "mid b");
        check(hipMalloc(&d_weights, weights.size() * sizeof(float)), "weights");
        check(hipMalloc(&d_pairs, p.size() * 4), "pairs"); check(hipMalloc(&d_offsets, 8), "offsets");
        check(hipMalloc(&d_counts, 4), "counts"); check(hipMalloc(&d_total, 4), "total");
        check(hipMalloc(&d_experts, experts.size() * 4), "experts"); check(hipMalloc(&d_starts, starts.size() * 4), "starts");
        check(hipMemcpy(d_gate_a, gate.data(), bytes, hipMemcpyHostToDevice), "copy gate a");
        check(hipMemcpy(d_gate_b, gate.data(), bytes, hipMemcpyHostToDevice), "copy gate b");
        check(hipMemcpy(d_up_a, up.data(), bytes, hipMemcpyHostToDevice), "copy up a");
        check(hipMemcpy(d_up_b, up.data(), bytes, hipMemcpyHostToDevice), "copy up b");
        check(hipMemcpy(d_weights, weights.data(), weights.size() * 4, hipMemcpyHostToDevice), "copy weights");
        check(hipMemcpy(d_pairs, p.data(), p.size() * 4, hipMemcpyHostToDevice), "copy pairs");
        check(hipMemcpy(d_offsets, offsets.data(), 8, hipMemcpyHostToDevice), "copy offsets");
        check(hipMemcpy(d_counts, counts.data(), 4, hipMemcpyHostToDevice), "copy counts");
        check(hipMemcpy(d_total, &ntiles, 4, hipMemcpyHostToDevice), "copy total");
        check(hipMemcpy(d_experts, experts.data(), experts.size() * 4, hipMemcpyHostToDevice), "copy experts");
        check(hipMemcpy(d_starts, starts.data(), starts.size() * 4, hipMemcpyHostToDevice), "copy starts");
    }
    ~Buffers() {
        (void)hipFree(d_gate_a); (void)hipFree(d_up_a); (void)hipFree(d_mid_a);
        (void)hipFree(d_gate_b); (void)hipFree(d_up_b); (void)hipFree(d_mid_b);
        (void)hipFree(d_weights); (void)hipFree(d_pairs); (void)hipFree(d_offsets);
        (void)hipFree(d_counts); (void)hipFree(d_total); (void)hipFree(d_experts);
        (void)hipFree(d_starts);
    }
};

static void launch_old(Buffers &b, uint32_t write, float clamp) {
    serial_rows_kernel<<<dim3(1, b.ntiles), 16>>>(b.d_gate_a, b.d_up_a, b.d_mid_a,
        b.d_pairs, b.d_offsets, b.d_counts, b.d_total, b.d_experts, b.d_starts,
        b.d_weights, b.nrows, write, clamp);
}
static void launch_new(Buffers &b, uint32_t write, float clamp) {
    coalesced_rows_kernel<<<dim3((b.nrows + 255) / 256, b.ntiles), 256>>>(
        b.d_gate_b, b.d_up_b, b.d_mid_b, b.d_pairs, b.d_offsets, b.d_counts,
        b.d_total, b.d_experts, b.d_starts, b.d_weights, b.nrows, write, clamp);
}
static float elapsed(Buffers &b, bool newer, int iterations) {
    hipEvent_t start, stop;
    check(hipEventCreate(&start), "create start event");
    check(hipEventCreate(&stop), "create stop event");
    for (int i = 0; i < 2; ++i) newer ? launch_new(b, 0, 0.0f) : launch_old(b, 0, 0.0f);
    check(hipDeviceSynchronize(), "warmup");
    check(hipEventRecord(start), "record start event");
    for (int i = 0; i < iterations; ++i) newer ? launch_new(b, 0, 0.0f) : launch_old(b, 0, 0.0f);
    check(hipEventRecord(stop), "record stop event");
    check(hipEventSynchronize(stop), "synchronize stop event");
    float ms = 0;
    check(hipEventElapsedTime(&ms, start, stop), "elapsed time");
    check(hipEventDestroy(start), "destroy start event");
    check(hipEventDestroy(stop), "destroy stop event");
    return ms / iterations;
}

int main() {
    {
        Buffers b(35, 2051); launch_old(b, 1, 1.25f); launch_new(b, 1, 1.25f);
        check(hipDeviceSynchronize(), "correctness");
        const size_t n = (size_t)b.npair * b.nrows, bytes = n * sizeof(float);
        std::vector<float> a(n), c(n); size_t bad = 0;
        for (auto pair : {std::pair<float *, float *>{b.d_mid_a, b.d_mid_b},
                          {b.d_gate_a, b.d_gate_b}, {b.d_up_a, b.d_up_b}}) {
            check(hipMemcpy(a.data(), pair.first, bytes, hipMemcpyDeviceToHost), "copy ref");
            check(hipMemcpy(c.data(), pair.second, bytes, hipMemcpyDeviceToHost), "copy candidate");
            for (size_t i = 0; i < n; ++i) if (std::memcmp(&a[i], &c[i], 4)) ++bad;
        }
        std::printf("correctness bit_mismatches=%zu/%zu\n", bad, n * 3);
        if (bad) return 2;
    }
    Buffers b(2840u * 8u, 2048u);
    const float old_ms = elapsed(b, false, 5), new_ms = elapsed(b, true, 5);
    std::printf("shape pairs=%u rows=%u serial_ms=%.4f coalesced_ms=%.4f change=%+.1f%%\n",
                b.npair, b.nrows, old_ms, new_ms, 100.0f * (new_ms / old_ms - 1.0f));
    return 0;
}
