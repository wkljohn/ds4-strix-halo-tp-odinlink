/* Original Q4_K gate/up projection, N128/J16. Eight waves consume one
 * shared activation panel. Stage a K128 weight half so wider N does not
 * double LDS. Keep this separate from the incumbent compiled control. */
__launch_bounds__(256)
__global__ static void moe_q4K_grouped_wide128_kernel(
        const char *weight_base, const ds4_q8_1_mmq_block *acts, float *out,
        const uint32_t *sorted_pairs, const uint32_t *offsets,
        const uint32_t *counts, const uint32_t *tile_total,
        const uint32_t *tile_experts, const uint32_t *tile_starts,
        uint32_t ntokens, uint32_t xq_blocks, uint32_t nrows,
        uint32_t n_expert, uint64_t weight_expert_bytes,
        uint64_t weight_row_bytes, uint32_t min_count) {
    constexpr int I = 128, J = 16, XS = 36, YS = 36;
    extern __shared__ int32_t smem[];
    int32_t *sy = smem, *sx = sy + J * YS;
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    const uint32_t expert = tile_experts[tile], start = tile_starts[tile];
    const uint32_t count = counts[expert];
    if (count < min_count) return;
    const uint32_t row0 = blockIdx.x * I;
    const char *expert_base = weight_base +
        uint64_t(moe_weight_expert<288u>(expert)) * weight_expert_bytes;
    float acc[J / 16][8] = {};
    for (uint32_t kb = 0; kb < xq_blocks; ++kb) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = wave * 16 + lane / 2;
            const uint32_t row = row0 + uint32_t(r);
            if (row < nrows) {
                const auto &w = reinterpret_cast<const cuda_block_q4_K *>(
                    expert_base + uint64_t(row) * weight_row_bytes)[kb];
                const int tx = (lane & 1) * 8 + h * 16;
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    const int32_t value = *reinterpret_cast<const int32_t *>(
                        w.qs + 4 * (tx + q));
                    const int col = 16 * (((tx + q) % 16) / 8) + (tx + q) % 8;
                    sx[r * XS + col] = value & 0x0f0f0f0f;
                    sx[r * XS + col + 8] = (value >> 4) & 0x0f0f0f0f;
                }
            }
            if (wave < 4) {
                const int scale_row = wave * 32 + lane;
                const uint32_t row = row0 + uint32_t(scale_row);
                if (row < nrows) {
                    const auto &w = reinterpret_cast<const cuda_block_q4_K *>(
                        expert_base + uint64_t(row) * weight_row_bytes)[kb];
                    const int32_t sc32 = ds4_q4k_unpack_scales(
                        reinterpret_cast<const int32_t *>(w.scales), h);
                    const int32_t mn32 = ds4_q4k_unpack_scales(
                        reinterpret_cast<const int32_t *>(w.scales), h + 2);
                    const auto *sc = reinterpret_cast<const uint8_t *>(&sc32);
                    const auto *mn = reinterpret_cast<const uint8_t *>(&mn32);
                    const float wd = __half2float(*reinterpret_cast<const half *>(&w.d));
                    const float wm = __half2float(*reinterpret_cast<const half *>(&w.dmin));
#pragma unroll
                    for (int l = 0; l < 4; ++l)
                        reinterpret_cast<half2 *>(sx + scale_row * XS + 32)[l] =
                            __floats2half2_rn(wd * float(sc[l]), -wm * float(mn[l]));
                }
            }
            for (int l = tid; l < J * YS; l += 256) {
                const int j = l / YS, e = l % YS;
                const uint32_t lp = start + uint32_t(j);
                if (lp < count) {
                    const uint32_t pair = sorted_pairs[offsets[expert] + lp];
                    const uint32_t token = pair / n_expert;
                    const int32_t *source = reinterpret_cast<const int32_t *>(
                        &acts[(uint64_t(kb) * 2u + uint32_t(h)) * ntokens + token]);
                    sy[l] = source[e];
                } else sy[l] = 0;
            }
            __syncthreads();
#pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                const ds4_q4k_wmma_ab_frag a =
                    ds4_q4k_load_rdna3_mirrored_16x8(
                        sx + wave * 16 * XS + kk * 8, XS);
                for (int j0 = 0; j0 < J; j0 += 16) {
                    const ds4_q4k_wmma_ab_frag b =
                        ds4_q4k_load_rdna3_mirrored_16x8(
                            sy + j0 * YS + 4 + kk * 8, YS);
                    ds4_q4k_i32x8 c = {};
                    c = ds4_q4k_wmma_i8_16x16x16(a, b, c);
#pragma unroll
                    for (int l = 0; l < 8; l++) {
                        const int i = 2 * l + lane / 16;
                        const int j = j0 + lane % 16;
                        const float2 bd = __half22float2(
                            reinterpret_cast<const half2 *>(sy + j * YS)[kk]);
                        const float2 ad = __half22float2(
                            reinterpret_cast<const half2 *>(sx + (wave * 16 + i) * XS + 32)[kk]);
                        acc[j0 / 16][l] +=
                            ad.x * bd.x * (float)c[l] + ad.y * bd.y;
                    }
                }
            }
            __syncthreads();
        }
    }
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const uint32_t row = row0 + uint32_t(wave * 16 + 2 * l + lane / 16);
        const uint32_t lp = start + uint32_t(lane % 16);
        if (row < nrows && lp < count) {
            const uint32_t pair = sorted_pairs[offsets[expert] + lp];
            out[uint64_t(pair) * nrows + row] = acc[0][l];
        }
    }
}
