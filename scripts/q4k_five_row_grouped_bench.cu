// Exact-route oracle for a five-row Q4_K DSpark verifier MoE.
// Reuses the production-format dot helpers from the existing standalone Q4_K
// harness and compares the current pair-major row-128 kernel with a compact
// expert-major row-128 kernel on a captured 5 x top-k-6 routing block.
//
// Build:
//   /opt/rocm/bin/hipcc -O3 -ffast-math -fno-finite-math-only \
//     --offload-arch=gfx1151 scripts/q4k_five_row_grouped_bench.cu \
//     -o /tmp/q4k_five_row_grouped_bench

#define main q4k_dp4a_baseline_unused_main
#include "q4k_dp4a_baseline_bench.cu"
#undef main

static constexpr uint32_t kTokens = 5;
static constexpr uint32_t kUsed = 6;
static constexpr uint32_t kPairs = kTokens * kUsed;
static constexpr uint32_t kBlocks = 16;
static constexpr uint32_t kMid = 2048;
static constexpr uint32_t kMaxPerExpert = 5;

__device__ static float q4k_dot_one(
        const cuda_block_q4_K *w,
        const cuda_block_q8_K *x) {
    float acc[4] = {};
    dev_dot_q4_K_q8_K_block4(w, x, NULL, NULL, NULL, 1u, acc);
    return acc[0];
}

__global__ static void q4k_pair_major_reference(
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t expert_bytes,
        uint64_t row_bytes) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t pair = blockIdx.y;
    const uint32_t tok = pair / kUsed;
    const int32_t expert_i = selected[pair];
    if (expert_i < 0) return;
    const uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * kBlocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        const uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= kMid) continue;
        const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(
                gate_base + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes);
        const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(
                up_base + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < kBlocks; b += 8u) {
            gate += q4k_dot_one(gr + b, xqb + b);
            up += q4k_dot_one(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0u) {
            mid_out[(uint64_t)pair * kMid + row] =
                (gate / (1.0f + expf(-gate))) * up * weights[pair];
        }
    }
}

__global__ static void q4k_expert_major_grouped5(
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *slot_experts,
        const uint32_t *slot_counts,
        const uint32_t *slot_pairs,
        const float *weights,
        uint64_t expert_bytes,
        uint64_t row_bytes) {
    const uint32_t slot = blockIdx.y;
    const uint32_t count = slot_counts[slot];
    if (count == 0u || count > kMaxPerExpert) return;
    const uint32_t expert = slot_experts[slot];
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pairs[kMaxPerExpert] = {};
    const cuda_block_q8_K *acts[kMaxPerExpert] = {};
#pragma unroll
    for (uint32_t p = 0; p < kMaxPerExpert; p++) {
        if (p < count) {
            pairs[p] = slot_pairs[slot * kMaxPerExpert + p];
            acts[p] = xq + (uint64_t)(pairs[p] / kUsed) * kBlocks;
        }
    }
    for (uint32_t rr = 0; rr < 4u; rr++) {
        const uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= kMid) continue;
        const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(
                gate_base + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes);
        const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(
                up_base + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes);
        float gate[kMaxPerExpert] = {};
        float up[kMaxPerExpert] = {};
        const uint32_t first = count < 4u ? count : 4u;
        for (uint32_t b = lane; b < kBlocks; b += 8u) {
            dev_dot_q4_K_q8_K_block4(
                    gr + b,
                    acts[0] ? acts[0] + b : NULL,
                    acts[1] ? acts[1] + b : NULL,
                    acts[2] ? acts[2] + b : NULL,
                    acts[3] ? acts[3] + b : NULL,
                    first, gate);
            dev_dot_q4_K_q8_K_block4(
                    ur + b,
                    acts[0] ? acts[0] + b : NULL,
                    acts[1] ? acts[1] + b : NULL,
                    acts[2] ? acts[2] + b : NULL,
                    acts[3] ? acts[3] + b : NULL,
                    first, up);
            if (count == 5u) {
                gate[4] += q4k_dot_one(gr + b, acts[4] + b);
                up[4] += q4k_dot_one(ur + b, acts[4] + b);
            }
        }
#pragma unroll
        for (uint32_t p = 0; p < kMaxPerExpert; p++) {
            if (p >= count) continue;
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0u) {
                const uint32_t pair = pairs[p];
                mid_out[(uint64_t)pair * kMid + row] =
                    (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[pair];
            }
        }
    }
}

int main() {
    int devices = 0;
    hip_check(hipGetDeviceCount(&devices), "get device count");
    if (devices < 1) return 2;
    hipDeviceProp_t prop{};
    hip_check(hipGetDeviceProperties(&prop, 0), "get device properties");

    const int32_t captured[kPairs] = {
        250, 45, 187, 129, 219, 38,
        129, 45, 187, 39, 174, 36,
        181, 45, 144, 110, 73, 4,
        45, 91, 4, 11, 187, 9,
        75, 115, 45, 0, 60, 187,
    };
    std::vector<int32_t> remapped(kPairs);
    std::vector<uint32_t> original;
    std::vector<uint32_t> counts(kPairs, 0u);
    std::vector<uint32_t> pairs(kPairs * kMaxPerExpert, 0u);
    for (uint32_t pair = 0; pair < kPairs; pair++) {
        uint32_t slot = 0;
        while (slot < original.size() && original[slot] != (uint32_t)captured[pair]) slot++;
        if (slot == original.size()) original.push_back((uint32_t)captured[pair]);
        remapped[pair] = (int32_t)slot;
        pairs[slot * kMaxPerExpert + counts[slot]++] = pair;
    }
    const uint32_t unique = (uint32_t)original.size();
    std::vector<uint32_t> slot_experts(kPairs, 0u);
    for (uint32_t i = 0; i < unique; i++) slot_experts[i] = i;
    std::vector<int32_t> singleton_selected = remapped;
    std::vector<uint32_t> duplicate_experts;
    std::vector<uint32_t> duplicate_counts;
    std::vector<uint32_t> duplicate_pairs;
    for (uint32_t slot = 0; slot < unique; slot++) {
        if (counts[slot] <= 1u) continue;
        duplicate_experts.push_back(slot);
        duplicate_counts.push_back(counts[slot]);
        for (uint32_t p = 0; p < kMaxPerExpert; p++) {
            duplicate_pairs.push_back(pairs[slot * kMaxPerExpert + p]);
        }
    }
    for (uint32_t pair = 0; pair < kPairs; pair++) {
        if (counts[(uint32_t)remapped[pair]] > 1u) singleton_selected[pair] = -1;
    }

    std::mt19937 rng(0x514b5005u);
    const size_t blocks_per_matrix = (size_t)unique * kMid * kBlocks;
    std::vector<cuda_block_q4_K> gate(blocks_per_matrix), up(blocks_per_matrix);
    std::vector<cuda_block_q8_K> xq((size_t)kTokens * kBlocks);
    std::vector<float> weights(kPairs);
    for (auto &v : gate) fill_q4_K_block(&v, rng);
    for (auto &v : up) fill_q4_K_block(&v, rng);
    for (auto &v : xq) fill_q8_K_block(&v, rng);
    std::uniform_real_distribution<float> wdist(0.02f, 0.3f);
    for (auto &v : weights) v = wdist(rng);

    cuda_block_q4_K *d_gate = NULL, *d_up = NULL;
    cuda_block_q8_K *d_xq = NULL;
    int32_t *d_selected = NULL, *d_singletons = NULL;
    uint32_t *d_slot_experts = NULL, *d_counts = NULL, *d_pairs = NULL;
    uint32_t *d_duplicate_experts = NULL, *d_duplicate_counts = NULL;
    uint32_t *d_duplicate_pairs = NULL;
    float *d_weights = NULL, *d_ref = NULL, *d_grouped = NULL, *d_hybrid = NULL;
    const size_t weight_bytes = blocks_per_matrix * sizeof(cuda_block_q4_K);
    const size_t out_bytes = (size_t)kPairs * kMid * sizeof(float);
    hip_check(hipMalloc(&d_gate, weight_bytes), "malloc gate");
    hip_check(hipMalloc(&d_up, weight_bytes), "malloc up");
    hip_check(hipMalloc(&d_xq, xq.size() * sizeof(cuda_block_q8_K)), "malloc xq");
    hip_check(hipMalloc(&d_selected, remapped.size() * sizeof(int32_t)), "malloc selected");
    hip_check(hipMalloc(&d_singletons, singleton_selected.size() * sizeof(int32_t)), "malloc singletons");
    hip_check(hipMalloc(&d_slot_experts, slot_experts.size() * sizeof(uint32_t)), "malloc slots");
    hip_check(hipMalloc(&d_counts, counts.size() * sizeof(uint32_t)), "malloc counts");
    hip_check(hipMalloc(&d_pairs, pairs.size() * sizeof(uint32_t)), "malloc pairs");
    hip_check(hipMalloc(&d_duplicate_experts, duplicate_experts.size() * sizeof(uint32_t)), "malloc duplicate experts");
    hip_check(hipMalloc(&d_duplicate_counts, duplicate_counts.size() * sizeof(uint32_t)), "malloc duplicate counts");
    hip_check(hipMalloc(&d_duplicate_pairs, duplicate_pairs.size() * sizeof(uint32_t)), "malloc duplicate pairs");
    hip_check(hipMalloc(&d_weights, weights.size() * sizeof(float)), "malloc weights");
    hip_check(hipMalloc(&d_ref, out_bytes), "malloc ref");
    hip_check(hipMalloc(&d_grouped, out_bytes), "malloc grouped");
    hip_check(hipMalloc(&d_hybrid, out_bytes), "malloc hybrid");
    hip_check(hipMemcpy(d_gate, gate.data(), weight_bytes, hipMemcpyHostToDevice), "copy gate");
    hip_check(hipMemcpy(d_up, up.data(), weight_bytes, hipMemcpyHostToDevice), "copy up");
    hip_check(hipMemcpy(d_xq, xq.data(), xq.size() * sizeof(cuda_block_q8_K), hipMemcpyHostToDevice), "copy xq");
    hip_check(hipMemcpy(d_selected, remapped.data(), remapped.size() * sizeof(int32_t), hipMemcpyHostToDevice), "copy selected");
    hip_check(hipMemcpy(d_singletons, singleton_selected.data(), singleton_selected.size() * sizeof(int32_t), hipMemcpyHostToDevice), "copy singletons");
    hip_check(hipMemcpy(d_slot_experts, slot_experts.data(), slot_experts.size() * sizeof(uint32_t), hipMemcpyHostToDevice), "copy slots");
    hip_check(hipMemcpy(d_counts, counts.data(), counts.size() * sizeof(uint32_t), hipMemcpyHostToDevice), "copy counts");
    hip_check(hipMemcpy(d_pairs, pairs.data(), pairs.size() * sizeof(uint32_t), hipMemcpyHostToDevice), "copy pairs");
    hip_check(hipMemcpy(d_duplicate_experts, duplicate_experts.data(), duplicate_experts.size() * sizeof(uint32_t), hipMemcpyHostToDevice), "copy duplicate experts");
    hip_check(hipMemcpy(d_duplicate_counts, duplicate_counts.data(), duplicate_counts.size() * sizeof(uint32_t), hipMemcpyHostToDevice), "copy duplicate counts");
    hip_check(hipMemcpy(d_duplicate_pairs, duplicate_pairs.data(), duplicate_pairs.size() * sizeof(uint32_t), hipMemcpyHostToDevice), "copy duplicate pairs");
    hip_check(hipMemcpy(d_weights, weights.data(), weights.size() * sizeof(float), hipMemcpyHostToDevice), "copy weights");

    const uint64_t row_bytes = (uint64_t)kBlocks * sizeof(cuda_block_q4_K);
    const uint64_t expert_bytes = (uint64_t)kMid * row_bytes;
    const dim3 direct_grid((kMid + 127u) / 128u, kPairs, 1u);
    const dim3 grouped_grid((kMid + 127u) / 128u, kPairs, 1u);
    auto direct = [&]() {
        q4k_pair_major_reference<<<direct_grid, 256>>>(
                d_ref, (const char *)d_gate, (const char *)d_up, d_xq,
                d_selected, d_weights, expert_bytes, row_bytes);
    };
    auto grouped = [&]() {
        q4k_expert_major_grouped5<<<grouped_grid, 256>>>(
                d_grouped, (const char *)d_gate, (const char *)d_up, d_xq,
                d_slot_experts, d_counts, d_pairs, d_weights,
                expert_bytes, row_bytes);
    };
    const dim3 duplicate_grid((kMid + 127u) / 128u,
                              (uint32_t)duplicate_experts.size(), 1u);
    auto hybrid = [&]() {
        q4k_pair_major_reference<<<direct_grid, 256>>>(
                d_hybrid, (const char *)d_gate, (const char *)d_up, d_xq,
                d_singletons, d_weights, expert_bytes, row_bytes);
        q4k_expert_major_grouped5<<<duplicate_grid, 256>>>(
                d_hybrid, (const char *)d_gate, (const char *)d_up, d_xq,
                d_duplicate_experts, d_duplicate_counts, d_duplicate_pairs,
                d_weights, expert_bytes, row_bytes);
    };
    direct(); grouped(); hybrid();
    hip_check(hipDeviceSynchronize(), "correctness synchronize");
    std::vector<float> ref((size_t)kPairs * kMid), got(ref.size()), hybrid_got(ref.size());
    hip_check(hipMemcpy(ref.data(), d_ref, out_bytes, hipMemcpyDeviceToHost), "read ref");
    hip_check(hipMemcpy(got.data(), d_grouped, out_bytes, hipMemcpyDeviceToHost), "read grouped");
    hip_check(hipMemcpy(hybrid_got.data(), d_hybrid, out_bytes, hipMemcpyDeviceToHost), "read hybrid");
    double max_abs = 0.0, rms = 0.0, hybrid_max_abs = 0.0, hybrid_rms = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        const double d = fabs((double)ref[i] - got[i]);
        max_abs = std::max(max_abs, d);
        rms += d * d;
        const double hd = fabs((double)ref[i] - hybrid_got[i]);
        hybrid_max_abs = std::max(hybrid_max_abs, hd);
        hybrid_rms += hd * hd;
    }
    rms = sqrt(rms / ref.size());
    hybrid_rms = sqrt(hybrid_rms / ref.size());
    const BenchResult base = time_launch(direct);
    const BenchResult cand = time_launch(grouped);
    const BenchResult hybrid_time = time_launch(hybrid);
    printf("device=%s arch=%s tokens=%u pairs=%u unique=%u duplicate_reuse=%.1f%%\n",
           prop.name, prop.gcnArchName, kTokens, kPairs, unique,
           100.0 * (double)(kPairs - unique) / kPairs);
    printf("correctness max_abs=%.9g rms=%.9g\n", max_abs, rms);
    printf("pair_major_us=%.2f grouped_us=%.2f change=%+.1f%%\n",
           base.median_us, cand.median_us,
           100.0 * (cand.median_us / base.median_us - 1.0));
    printf("hybrid duplicate_experts=%zu max_abs=%.9g rms=%.9g us=%.2f change=%+.1f%%\n",
           duplicate_experts.size(), hybrid_max_abs, hybrid_rms,
           hybrid_time.median_us,
           100.0 * (hybrid_time.median_us / base.median_us - 1.0));

    (void)hipFree(d_gate); (void)hipFree(d_up); (void)hipFree(d_xq);
    (void)hipFree(d_selected); (void)hipFree(d_singletons); (void)hipFree(d_slot_experts);
    (void)hipFree(d_counts); (void)hipFree(d_pairs); (void)hipFree(d_weights);
    (void)hipFree(d_duplicate_experts); (void)hipFree(d_duplicate_counts);
    (void)hipFree(d_duplicate_pairs);
    (void)hipFree(d_ref); (void)hipFree(d_grouped); (void)hipFree(d_hybrid);
    return hybrid_max_abs > 1.0e-3 ? 3 : 0;
}
