static int routed_moe_u64_add_checked(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || a > UINT64_MAX - b) return 0;
    *out = a + b;
    return 1;
}

#ifndef DS4_TP_FEATURE_Q4K_WMMA
#define DS4_TP_FEATURE_Q4K_WMMA (UINT32_C(1) << 0)
#endif

static int routed_moe_align256_checked(uint64_t v, uint64_t *out) {
    if (!out || v > UINT64_MAX - 255ull) return 0;
    *out = (v + 255ull) & ~255ull;
    return 1;
}

/* Defined in ds4_rocm.cu after this textual include. */
extern "C" int ds4_gpu_tp_expert_shard_active(void);

typedef struct {
    int opt_in;
    int disabled;
    int gfx1151;
} ds4_q4k_wmma_dispatch_config;

static const ds4_q4k_wmma_dispatch_config *routed_moe_q4k_wmma_config(void) {
    static ds4_q4k_wmma_dispatch_config cfg = {-1, 0, 0};
    if (cfg.opt_in < 0) {
        const char *enable = getenv("DS4_ROCM_Q4K_WMMA");
        const char *disable = getenv("DS4_ROCM_DISABLE_Q4K_WMMA");
        /* The gfx1151 shape-gated implementation is the validated Strix Halo
         * production path. Keep an explicit 0 and the independent disable
         * variable as kill switches for diagnosis and future ROCm changes. */
        cfg.opt_in = enable == NULL || strcmp(enable, "0") != 0;
        cfg.disabled = disable != NULL && strcmp(disable, "1") == 0;
        int device = 0;
        cudaDeviceProp prop;
        memset(&prop, 0, sizeof(prop));
        if (cudaGetDevice(&device) == cudaSuccess &&
            cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
            cfg.gfx1151 = strncmp(prop.gcnArchName, "gfx1151", 7) == 0;
        } else {
            (void)cudaGetLastError();
        }
    }
    return &cfg;
}

static uint32_t g_q4k_wmma_tp_runtime_features;
static int g_q4k_wmma_tp_runtime_features_valid;

extern "C" uint32_t ds4_gpu_q4k_wmma_runtime_features(
        int q4k_weights,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        const void *gate_w,
        const void *up_w,
        int q4k_down_weights,
        uint32_t down_in_dim,
        uint32_t down_out_dim,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        const void *down_w) {
    const ds4_q4k_wmma_dispatch_config *cfg = routed_moe_q4k_wmma_config();
    const uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;
    const int shape_ok = q4k_weights &&
        expert_in_dim == 4096u && expert_mid_dim == 2048u &&
        xq_blocks == 16u &&
        gate_row_bytes == (uint64_t)xq_blocks * sizeof(cuda_block_q4_K) &&
        gate_expert_bytes == (uint64_t)expert_mid_dim * gate_row_bytes &&
        q4k_down_weights && down_in_dim == 2048u && down_out_dim == 4096u &&
        down_in_dim / CUDA_QK_K == 8u &&
        down_row_bytes == 8u * sizeof(cuda_block_q4_K) &&
        down_expert_bytes == (uint64_t)down_out_dim * down_row_bytes &&
        (((uintptr_t)gate_w | (uintptr_t)up_w | (uintptr_t)down_w) & 15u) == 0u;
    return cfg->gfx1151 && shape_ok && cfg->opt_in &&
           !cfg->disabled && !g_quality_mode ? DS4_TP_FEATURE_Q4K_WMMA : 0u;
}

extern "C" void ds4_gpu_set_tp_runtime_features(uint32_t rank,
                                                  uint32_t features) {
    const ds4_q4k_wmma_dispatch_config *cfg = routed_moe_q4k_wmma_config();
    g_q4k_wmma_tp_runtime_features = features;
    g_q4k_wmma_tp_runtime_features_valid = 1;
    const int q4k = (features & DS4_TP_FEATURE_Q4K_WMMA) != 0;
    fprintf(stderr, DS4_GPU_LOG_PREFIX
            "Q4_K WMMA startup rank=%u negotiated=0x%08x gate=%d up=%d "
            "down=%d gate_up_threshold=6 down_threshold=1 quality=%d "
            "kill_switch=%d\n",
            rank, features, q4k, q4k, q4k, g_quality_mode, cfg->disabled);
}

static int routed_moe_q4k_wmma_enabled(
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t xq_blocks,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        const void *gate_w,
        const void *up_w,
        const void *xq,
        uint32_t out_dim,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        const void *down_w) {
    const ds4_q4k_wmma_dispatch_config *cfg = routed_moe_q4k_wmma_config();
    const uint32_t local = ds4_gpu_q4k_wmma_runtime_features(
            1, expert_in_dim, expert_mid_dim, gate_expert_bytes,
            gate_row_bytes, gate_w, up_w, 1, expert_mid_dim, out_dim,
            down_expert_bytes, down_row_bytes, down_w);
    const int shape_ok = local != 0 &&
        (((uintptr_t)xq | (uintptr_t)down_w) & 15u) == 0u;
    const int tp_active = ds4_gpu_tp_expert_shard_active();
    const int tp_ok = !tp_active ||
        (g_q4k_wmma_tp_runtime_features_valid &&
         (g_q4k_wmma_tp_runtime_features & DS4_TP_FEATURE_Q4K_WMMA));
    (void)cfg;
    return shape_ok && tp_ok;
}

enum {
    DS4_ROCM_MOE_DECODE_PROFILE_GATE_RESIDENT_START = 0,
    DS4_ROCM_MOE_DECODE_PROFILE_GATE_RESIDENT_END,
    DS4_ROCM_MOE_DECODE_PROFILE_GATE_MISSING_START,
    DS4_ROCM_MOE_DECODE_PROFILE_GATE_MISSING_END,
    DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_START,
    DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_END,
    DS4_ROCM_MOE_DECODE_PROFILE_MID_QUANT_START,
    DS4_ROCM_MOE_DECODE_PROFILE_MID_QUANT_END,
    DS4_ROCM_MOE_DECODE_PROFILE_DOWN_START,
    DS4_ROCM_MOE_DECODE_PROFILE_DOWN_END,
    DS4_ROCM_MOE_DECODE_PROFILE_EVENT_COUNT
};

typedef struct {
    uint64_t calls;
    uint64_t split_calls;
    uint64_t q8_gateup_calls;
    uint64_t q8_down_calls;
    double finish_missing_ms;
    double gate_resident_ms;
    double gate_missing_ms;
    double gate_full_ms;
    double mid_quant_ms;
    double down_ms;
} ds4_rocm_moe_decode_profile_stats;

typedef struct {
    int gate_resident;
    int gate_missing;
    int gate_full;
    int mid_quant;
    int down;
} ds4_rocm_moe_decode_profile_record;

static ds4_rocm_moe_decode_profile_stats g_moe_decode_profile_stats;
static cudaEvent_t g_moe_decode_profile_events[DS4_ROCM_MOE_DECODE_PROFILE_EVENT_COUNT];
static int g_moe_decode_profile_enabled = -1;
static int g_moe_decode_profile_registered = 0;
static int g_moe_decode_profile_events_ready = 0;

static void routed_moe_decode_profile_print(void) {
    const ds4_rocm_moe_decode_profile_stats *p =
        &g_moe_decode_profile_stats;
    if (p->calls == 0u) return;
    const double calls = (double)p->calls;
    fprintf(stderr,
            DS4_GPU_LOG_PREFIX "Q2 decode MoE profile calls=%llu "
            "split=%llu q8_gateup=%llu q8_down=%llu "
            "finish_missing=%.3f ms gate_resident=%.3f ms "
            "gate_missing=%.3f ms gate_full=%.3f ms "
            "mid_quant=%.3f ms down=%.3f ms avg_sum=%.3f ms\n",
            (unsigned long long)p->calls,
            (unsigned long long)p->split_calls,
            (unsigned long long)p->q8_gateup_calls,
            (unsigned long long)p->q8_down_calls,
            p->finish_missing_ms,
            p->gate_resident_ms,
            p->gate_missing_ms,
            p->gate_full_ms,
            p->mid_quant_ms,
            p->down_ms,
            (p->finish_missing_ms + p->gate_resident_ms +
             p->gate_missing_ms + p->gate_full_ms + p->mid_quant_ms +
             p->down_ms) / calls);
}

static int routed_moe_decode_profile_enabled(void) {
    if (g_moe_decode_profile_enabled < 0) {
        const char *env = getenv("DS4_ROCM_MOE_DECODE_PROFILE");
        g_moe_decode_profile_enabled =
            (env != NULL && env[0] != '\0' && strcmp(env, "0") != 0) ? 1 : 0;
        if (g_moe_decode_profile_enabled && !g_moe_decode_profile_registered) {
            atexit(routed_moe_decode_profile_print);
            g_moe_decode_profile_registered = 1;
        }
    }
    return g_moe_decode_profile_enabled;
}

static int routed_moe_decode_profile_ensure_events(void) {
    if (g_moe_decode_profile_events_ready) return 1;
    for (uint32_t i = 0; i < DS4_ROCM_MOE_DECODE_PROFILE_EVENT_COUNT; i++) {
        cudaError_t err = cudaEventCreate(&g_moe_decode_profile_events[i]);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "Q2 decode MoE profile event create failed: %s\n",
                    cudaGetErrorString(err));
            for (uint32_t j = 0; j < i; j++) {
                (void)cudaEventDestroy(g_moe_decode_profile_events[j]);
                g_moe_decode_profile_events[j] = NULL;
            }
            return 0;
        }
    }
    g_moe_decode_profile_events_ready = 1;
    return 1;
}

static int routed_moe_decode_profile_record_event(uint32_t ev, const char *what) {
    if (ev >= DS4_ROCM_MOE_DECODE_PROFILE_EVENT_COUNT ||
        !g_moe_decode_profile_events_ready) {
        return 0;
    }
    return cuda_ok(cudaEventRecord(g_moe_decode_profile_events[ev], 0), what);
}

static int routed_moe_decode_profile_add_event_ms(uint32_t start_ev,
                                                  uint32_t end_ev,
                                                  double *accum,
                                                  const char *what) {
    if (!accum ||
        start_ev >= DS4_ROCM_MOE_DECODE_PROFILE_EVENT_COUNT ||
        end_ev >= DS4_ROCM_MOE_DECODE_PROFILE_EVENT_COUNT ||
        !g_moe_decode_profile_events_ready) {
        return 0;
    }
    if (!cuda_ok(cudaEventSynchronize(g_moe_decode_profile_events[end_ev]),
                 what)) {
        return 0;
    }
    float ms = 0.0f;
    if (!cuda_ok(cudaEventElapsedTime(&ms,
                                      g_moe_decode_profile_events[start_ev],
                                      g_moe_decode_profile_events[end_ev]),
                 what)) {
        return 0;
    }
    *accum += (double)ms;
    return 1;
}

static int routed_moe_decode_profile_collect(
        const ds4_rocm_moe_decode_profile_record *rec) {
    if (!rec) return 1;
    if (rec->gate_resident &&
        !routed_moe_decode_profile_add_event_ms(
                DS4_ROCM_MOE_DECODE_PROFILE_GATE_RESIDENT_START,
                DS4_ROCM_MOE_DECODE_PROFILE_GATE_RESIDENT_END,
                &g_moe_decode_profile_stats.gate_resident_ms,
                "Q2 decode MoE profile resident gate/up")) {
        return 0;
    }
    if (rec->gate_missing &&
        !routed_moe_decode_profile_add_event_ms(
                DS4_ROCM_MOE_DECODE_PROFILE_GATE_MISSING_START,
                DS4_ROCM_MOE_DECODE_PROFILE_GATE_MISSING_END,
                &g_moe_decode_profile_stats.gate_missing_ms,
                "Q2 decode MoE profile missing gate/up")) {
        return 0;
    }
    if (rec->gate_full &&
        !routed_moe_decode_profile_add_event_ms(
                DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_START,
                DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_END,
                &g_moe_decode_profile_stats.gate_full_ms,
                "Q2 decode MoE profile gate/up")) {
        return 0;
    }
    if (rec->mid_quant &&
        !routed_moe_decode_profile_add_event_ms(
                DS4_ROCM_MOE_DECODE_PROFILE_MID_QUANT_START,
                DS4_ROCM_MOE_DECODE_PROFILE_MID_QUANT_END,
                &g_moe_decode_profile_stats.mid_quant_ms,
                "Q2 decode MoE profile mid quant")) {
        return 0;
    }
    if (rec->down &&
        !routed_moe_decode_profile_add_event_ms(
                DS4_ROCM_MOE_DECODE_PROFILE_DOWN_START,
                DS4_ROCM_MOE_DECODE_PROFILE_DOWN_END,
                &g_moe_decode_profile_stats.down_ms,
                "Q2 decode MoE profile down")) {
        return 0;
    }
    return 1;
}

/* Mixed IQ2_XXS-gate/Q2_K-down models already compute routed mid activations
 * as float.  Reuse the newer Q2_K expert-batch/WMMA down kernels instead of
 * re-quantizing mid to Q8_K and taking the older qwarp down path.  This keeps
 * the CyberNeurova all-Q2 path untouched while giving the standard IQ2 mix the
 * same fast Q2 down projection used by q2k_path. */
static int routed_moe_q2_float_down_launch(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *down,
        const ds4_gpu_tensor *mid,
        const half *mid_h_hot,
        int hot_mid_f16,
        const char *down_w,
        const uint32_t *counts,
        const uint32_t *offsets,
        const uint32_t *sorted_pairs,
        uint32_t *hot_experts_dev,
        uint32_t n_tokens,
        uint32_t n_total_expert,
        uint32_t n_expert,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes) {
    if (!out || !down || !mid || !down_w || !counts || !offsets || !sorted_pairs ||
        n_tokens == 0u || n_total_expert == 0u || n_total_expert > DS4_ROCM_MAX_N_EXPERT ||
        n_expert == 0u || n_expert > DS4_ROCM_N_EXPERT_USED ||
        (expert_mid_dim % CUDA_QK_K) != 0u || expert_mid_dim == 0u || out_dim == 0u ||
        !cuda_tensor_has_elems3(mid, n_tokens, n_expert, expert_mid_dim, sizeof(float)) ||
        !cuda_tensor_has_elems3(down, n_tokens, n_expert, out_dim, sizeof(float)) ||
        !cuda_tensor_has_elems2(out, n_tokens, out_dim, sizeof(float))) {
        return 0;
    }

    uint32_t h_counts[DS4_ROCM_MAX_N_EXPERT] = {0};
    if (!cuda_ok(cudaMemcpy(h_counts, counts, n_total_expert * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                 "routed_moe iq2/q2 float-down counts copy")) {
        return 0;
    }

    const uint32_t down_tile = 4u;
    const uint32_t down_rpb = 16u;
    const uint32_t down_threads = down_rpb * 32u;
    const size_t down_shmem = (size_t)down_tile * 256u * sizeof(float);
    const int use_f16_down = (out_dim & 1u) == 0u;
    half *down_h = use_f16_down ? (half *)down->ptr : NULL;

    uint32_t hot_count = 0u;
    uint32_t hot_max = 0u;
    const uint32_t hot_threshold = 8u;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    const int use_wmma_hot = hot_experts_dev &&
        !g_quality_mode &&
        (expert_mid_dim % 16u) == 0u && (out_dim % 16u) == 0u;
#else
    const int use_wmma_hot = 0;
#endif
    uint32_t h_hot[DS4_ROCM_MAX_N_EXPERT] = {0};
    if (use_wmma_hot) {
        for (uint32_t e = 0; e < n_total_expert; e++) {
            const uint32_t c = h_counts[e];
            if (c >= hot_threshold) {
                h_hot[hot_count++] = e;
                if (c > hot_max) hot_max = c;
            }
        }
    }

    const uint32_t scalar_max = hot_count != 0u ? hot_threshold : 0u;
    const dim3 down_grid((out_dim + down_rpb - 1u) / down_rpb, n_total_expert, 1u);
    if (use_f16_down) {
        if (down_tile == 4u) {
            moe_down_q2K_expert_batch_sharedmid_kernel<4,false,true><<<down_grid, down_threads, down_shmem>>>(
                    NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes, n_expert);
        } else if (down_tile == 8u) {
            moe_down_q2K_expert_batch_sharedmid_kernel<8,false,true><<<down_grid, down_threads, down_shmem>>>(
                    NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes, n_expert);
        } else {
            moe_down_q2K_expert_batch_sharedmid_kernel<16,false,true><<<down_grid, down_threads, down_shmem>>>(
                    NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes, n_expert);
        }
    } else if (down_tile == 4u) {
        moe_down_q2K_expert_batch_sharedmid_kernel<4><<<down_grid, down_threads, down_shmem>>>(
                (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                down_expert_bytes, down_row_bytes, n_expert);
    } else if (down_tile == 8u) {
        moe_down_q2K_expert_batch_sharedmid_kernel<8><<<down_grid, down_threads, down_shmem>>>(
                (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                down_expert_bytes, down_row_bytes, n_expert);
    } else {
        moe_down_q2K_expert_batch_sharedmid_kernel<16><<<down_grid, down_threads, down_shmem>>>(
                (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                down_expert_bytes, down_row_bytes, n_expert);
    }
    if (!cuda_ok(cudaGetLastError(), "routed_moe iq2/q2 float-down scalar launch")) return 0;
    if (hot_count != 0u &&
        !cuda_ok(cudaMemcpy(hot_experts_dev, h_hot, hot_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                 "routed_moe iq2/q2 float-down hot copy")) {
        return 0;
    }

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    if (use_wmma_hot && hot_count != 0u) {
        constexpr uint32_t bm = 16u, bn = 16u, bk = 16u;
        const int no_n2 = 0;
        const uint32_t wmma_mtiles = 4u;
        if (!no_n2) {
            if (wmma_mtiles == 4u) {
                constexpr uint32_t mt = 4u;
                const dim3 block(32u * mt, 1u, 1u);
                const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                                (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
                const size_t shmem_n2 = (mt * bm * bk + 2u * bk * bn) * sizeof(half) +
                                        (2u * mt * bm * bn) * sizeof(float);
                if (use_f16_down && hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else if (use_f16_down) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else if (hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true,false><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else {
                    moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                }
            } else if (wmma_mtiles == 16u) {
                constexpr uint32_t mt = 16u;
                const dim3 block(32u * mt, 1u, 1u);
                const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                                (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
                const size_t shmem_n2 = (mt * bm * bk + 2u * bk * bn) * sizeof(half) +
                                        (2u * mt * bm * bn) * sizeof(float);
                if (use_f16_down && hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<16,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else if (use_f16_down) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<16,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else if (hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<16,16,16,16,true,false><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else {
                    moe_down_q2K_hotlist_wmma_n2_kernel<16,16,16,16><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                }
            } else {
                constexpr uint32_t mt = 8u;
                const dim3 block(32u * mt, 1u, 1u);
                const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                                (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
                const size_t shmem_n2 = (mt * bm * bk + 2u * bk * bn) * sizeof(half) +
                                        (2u * mt * bm * bn) * sizeof(float);
                if (use_f16_down && hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else if (use_f16_down) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else if (hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true,false><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else {
                    moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                }
            }
        } else if (wmma_mtiles == 16u) {
            constexpr uint32_t mt = 16u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid(out_dim / bn, (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
            const size_t shmem = (mt * bm * bk + bk * bn) * sizeof(half) +
                                 (mt * bm * bn) * sizeof(float);
            moe_down_q2K_hotlist_wmma_kernel<16,16,16,16><<<grid, block, shmem>>>(
                    (float *)down->ptr, down_w, (const float *)mid->ptr,
                    counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                    expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
        } else if (wmma_mtiles == 4u) {
            constexpr uint32_t mt = 4u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid(out_dim / bn, (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
            const size_t shmem = (mt * bm * bk + bk * bn) * sizeof(half) +
                                 (mt * bm * bn) * sizeof(float);
            moe_down_q2K_hotlist_wmma_kernel<4,16,16,16><<<grid, block, shmem>>>(
                    (float *)down->ptr, down_w, (const float *)mid->ptr,
                    counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                    expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
        } else {
            constexpr uint32_t mt = 8u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid(out_dim / bn, (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
            const size_t shmem = (mt * bm * bk + bk * bn) * sizeof(half) +
                                 (mt * bm * bn) * sizeof(float);
            moe_down_q2K_hotlist_wmma_kernel<8,16,16,16><<<grid, block, shmem>>>(
                    (float *)down->ptr, down_w, (const float *)mid->ptr,
                    counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                    expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
        }
        if (!cuda_ok(cudaGetLastError(), "routed_moe iq2/q2 float-down wmma launch")) return 0;
    }
#endif

    const uint64_t n = (uint64_t)n_tokens * out_dim;
    if (use_f16_down && (out_dim & 1u) == 0u) {
        const uint64_t n2 = (uint64_t)n_tokens * (out_dim >> 1u);
        moe_sum_f16x2_kernel<<<(n2 + 255u) / 256u, 256>>>(
                (float *)out->ptr, down_h, out_dim, n_expert, n_tokens);
    } else if (use_f16_down) {
        moe_sum_f16_kernel<<<(n + 255u) / 256u, 256>>>(
                (float *)out->ptr, down_h, out_dim, n_expert, n_tokens);
    } else {
        moe_sum_kernel<<<(n + 255u) / 256u, 256>>>(
                (float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
    }
    return cuda_ok(cudaGetLastError(), "routed_moe iq2/q2 float-down sum launch");
}

typedef struct {
    int q4k_path;
    int iq2_path;
    int iq2_iq2_path;
    int q2k_path;
    uint64_t gate_bytes;
    uint64_t down_bytes;
} routed_moe_launch_plan;

static int routed_moe_build_plan(
        const ds4_gpu_tensor *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        const ds4_gpu_tensor *mid,
        const ds4_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        const ds4_gpu_tensor *x,
        uint32_t n_tokens,
        routed_moe_launch_plan *plan) {
    if (!plan) return 0;
    memset(plan, 0, sizeof(*plan));
    if (!out || !gate || !up || !mid || !down || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_total_expert == 0u ||
        n_expert == 0u || n_expert > DS4_ROCM_N_EXPERT_USED ||
        expert_in_dim == 0u || expert_mid_dim == 0u || out_dim == 0u ||
        expert_in_dim % CUDA_QK_K != 0 || expert_mid_dim % CUDA_QK_K != 0 ||
        !cuda_tensor_has_elems2(x, n_tokens, expert_in_dim, sizeof(float)) ||
        !cuda_tensor_has_elems2(selected, n_tokens, n_expert, sizeof(int32_t)) ||
        !cuda_tensor_has_elems2(weights, n_tokens, n_expert, sizeof(float)) ||
        !cuda_tensor_has_elems3(gate, n_tokens, n_expert, expert_mid_dim, sizeof(float)) ||
        !cuda_tensor_has_elems3(up, n_tokens, n_expert, expert_mid_dim, sizeof(float)) ||
        !cuda_tensor_has_elems3(mid, n_tokens, n_expert, expert_mid_dim, sizeof(float)) ||
        !cuda_tensor_has_elems3(down, n_tokens, n_expert, out_dim, sizeof(float)) ||
        !cuda_tensor_has_elems2(out, n_tokens, out_dim, sizeof(float))) {
        return 0;
    }
    plan->q4k_path = (gate_type == 12u && down_type == 12u);
    plan->iq2_path = (gate_type == 16u && down_type == 10u);
    plan->iq2_iq2_path = (gate_type == 16u && down_type == 16u);
    plan->q2k_path = (gate_type == 10u && down_type == 10u);
    if (!plan->q4k_path && !plan->iq2_path &&
        !plan->iq2_iq2_path && !plan->q2k_path) return 0;
    if (!cuda_u64_mul_checked(n_total_expert, gate_expert_bytes, &plan->gate_bytes) ||
        !cuda_u64_mul_checked(n_total_expert, down_expert_bytes, &plan->down_bytes) ||
        !cuda_model_range_fits(model_size, gate_offset, plan->gate_bytes) ||
        !cuda_model_range_fits(model_size, up_offset, plan->gate_bytes) ||
        !cuda_model_range_fits(model_size, down_offset, plan->down_bytes)) {
        return 0;
    }
    return 1;
}

static int routed_moe_full_table_is_cached(
        const void *model_map,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_bytes,
        uint64_t down_bytes) {
    return cuda_model_range_is_cached(model_map, gate_offset, gate_bytes) &&
           cuda_model_range_is_cached(model_map, up_offset, gate_bytes) &&
           cuda_model_range_is_cached(model_map, down_offset, down_bytes);
}

static int routed_moe_launch(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        float clamp,
        const ds4_gpu_tensor *x,
        uint32_t layer_index,
        uint32_t n_tokens,
        bool force_resident) {
    routed_moe_launch_plan plan;
    if (!routed_moe_build_plan(out, gate, up, mid, down, model_map, model_size,
                               gate_offset, up_offset, down_offset, gate_type, down_type,
                               gate_expert_bytes, down_expert_bytes, expert_in_dim,
                               expert_mid_dim, out_dim, selected, weights, n_total_expert, n_expert, x,
                               n_tokens, &plan)) {
        return 0;
    }
    const int q4k_path = plan.q4k_path;
    const int iq2_path = plan.iq2_path;
    const int iq2_iq2_path = plan.iq2_iq2_path;
    const int iq2_gate_path = iq2_path || iq2_iq2_path;
    const int q2k_path = plan.q2k_path;
    const uint64_t gate_bytes = plan.gate_bytes;
    const uint64_t down_bytes = plan.down_bytes;
    uint64_t pair_count64 = 0;
    if (!cuda_u64_mul_checked(n_tokens, n_expert, &pair_count64) ||
        pair_count64 > UINT32_MAX) {
        return 0;
    }
    const uint32_t pair_count = (uint32_t)pair_count64;
    const ds4_gpu_tensor *selected_exec = selected;
    const char *gate_w = NULL;
    const char *up_w = NULL;
    const char *down_w = NULL;
    const char **gate_slot_ptrs = NULL;
    const char **up_slot_ptrs = NULL;
    const char **down_slot_ptrs = NULL;
    const char **resident_gate_slot_ptrs = NULL;
    const char **resident_up_slot_ptrs = NULL;
    const char **missing_gate_slot_ptrs = NULL;
    const char **missing_up_slot_ptrs = NULL;
    const uint8_t *stream_batch_pair_missing = NULL;
    uint32_t stream_resident_mask = 0;
    uint32_t stream_missing_mask = 0;
    uint32_t stream_batch_unique = 0;
    uint32_t stream_batch_resident_count = 0;
    uint32_t stream_batch_missing_count = 0;
    const int stream_full_layer =
        (n_tokens > 1u || force_resident) &&
        cuda_stream_layer_expert_cache_apply(model_map,
                                             layer_index,
                                             n_total_expert,
                                             gate_offset,
                                             up_offset,
                                             down_offset,
                                             gate_expert_bytes,
                                             down_expert_bytes,
                                             &gate_w,
                                             &up_w,
                                             &down_w);
    const int full_table_cached =
        !stream_full_layer &&
        routed_moe_full_table_is_cached(model_map,
                                        gate_offset,
                                        up_offset,
                                        down_offset,
                                        gate_bytes,
                                        down_bytes);
    const int batch_stream_split_selected =
        !stream_full_layer &&
        !full_table_cached &&
        n_tokens > 1u &&
        (iq2_gate_path || q2k_path) &&
        n_expert <= DS4_ROCM_N_EXPERT_USED &&
        cuda_stream_batch_selected_apply_split(model_map,
                                               layer_index,
                                               n_total_expert,
                                               n_expert,
                                               n_tokens,
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes,
                                               &selected_exec,
                                               &resident_gate_slot_ptrs,
                                               &resident_up_slot_ptrs,
                                               &missing_gate_slot_ptrs,
                                               &missing_up_slot_ptrs,
                                               &down_slot_ptrs,
                                               &stream_batch_pair_missing,
                                               &stream_batch_resident_count,
                                               &stream_batch_missing_count,
                                               &stream_batch_unique);
    const int batch_stream_selected =
        !stream_full_layer &&
        !full_table_cached &&
        !batch_stream_split_selected &&
        n_tokens > 1u &&
        (iq2_gate_path || q2k_path) &&
        n_expert <= DS4_ROCM_N_EXPERT_USED &&
        cuda_stream_batch_selected_prepare(model_map,
                                           model_size,
                                           layer_index,
                                           selected,
                                           n_tokens,
                                           n_total_expert,
                                           n_expert,
                                           gate_offset,
                                           up_offset,
                                           down_offset,
                                           gate_expert_bytes,
                                           down_expert_bytes,
                                           &selected_exec,
                                           &gate_slot_ptrs,
                                           &up_slot_ptrs,
                                           &down_slot_ptrs,
                                           &stream_batch_unique);
    const int split_selected =
        !stream_full_layer &&
        n_tokens == 1u &&
        getenv("DS4_ROCM_DISABLE_STREAMING_SPLIT_SELECTED") == NULL &&
        cuda_stream_selected_apply_split(model_map,
                                         layer_index,
                                         n_total_expert,
                                         n_expert,
                                         gate_expert_bytes,
                                         down_expert_bytes,
                                         &selected_exec,
                                         &gate_w,
                                         &up_w,
                                         &down_w,
                                         &gate_slot_ptrs,
                                         &up_slot_ptrs,
                                         &down_slot_ptrs,
                                         &stream_resident_mask,
                                         &stream_missing_mask);
    const int compact_selected =
        split_selected ||
        (!stream_full_layer &&
        n_tokens == 1u &&
        cuda_stream_selected_apply(model_map,
                                   layer_index,
                                   n_total_expert,
                                   n_expert,
                                   gate_expert_bytes,
                                   down_expert_bytes,
                                   &selected_exec,
                                   &gate_w,
                                   &up_w,
                                   &down_w));
    if (!compact_selected && !batch_stream_selected && !batch_stream_split_selected) {
        if (g_ssd_streaming_mode &&
            n_total_expert > n_expert &&
            !stream_full_layer &&
            !full_table_cached) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "SSD streaming routed MoE missing compact selected experts "
                    "(layer=%u tokens=%u total_experts=%u selected=%u); full expert table is not mapped\n",
                    layer_index,
                    n_tokens,
                    n_total_expert,
                    n_expert);
            return 0;
        }
        if (!stream_full_layer) {
            gate_w = cuda_model_range_ptr(model_map, gate_offset, gate_bytes, "moe_gate");
            up_w = cuda_model_range_ptr(model_map, up_offset, gate_bytes, "moe_up");
            down_w = cuda_model_range_ptr(model_map, down_offset, down_bytes, "moe_down");
        }
    }
    if (batch_stream_selected || batch_stream_split_selected) {
        if (!down_slot_ptrs ||
            stream_batch_unique == 0) {
            return 0;
        }
        if (batch_stream_selected && (!gate_slot_ptrs || !up_slot_ptrs)) return 0;
        if (batch_stream_split_selected &&
            (!resident_gate_slot_ptrs || !resident_up_slot_ptrs ||
             !missing_gate_slot_ptrs || !missing_up_slot_ptrs ||
             !stream_batch_pair_missing)) {
            return 0;
        }
        if (!cuda_stream_batch_selected_wait_upload_ready()) return 0;
    } else if (!gate_w || !up_w || !down_w) {
        return 0;
    }
    if (compact_selected && !cuda_stream_selected_wait_upload_ready()) return 0;

    int ok = 1;
    const uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;
    const uint32_t midq_blocks = expert_mid_dim / CUDA_QK_K;
    uint64_t xq_count = 0;
    uint64_t midq_count = 0;
    uint64_t xq_bytes = 0;
    uint64_t midq_bytes = 0;
    if (!cuda_u64_mul_checked(n_tokens, xq_blocks, &xq_count) ||
        !cuda_u64_mul_checked(pair_count64, midq_blocks, &midq_count) ||
        !cuda_u64_mul_checked(xq_count, sizeof(cuda_block_q8_K), &xq_bytes) ||
        !cuda_u64_mul_checked(midq_count, sizeof(cuda_block_q8_K), &midq_bytes)) {
        return 0;
    }
    if (!q2k_path && down->bytes >= xq_bytes && gate->bytes >= midq_bytes) {
        cuda_block_q8_K *xq = (cuda_block_q8_K *)down->ptr;
        cuda_block_q8_K *midq = (cuda_block_q8_K *)gate->ptr;
        /* Correctness rollback for the optimized resident IQ2 prefill path. */
        const uint32_t disable_resident_iq2_sorted =
            iq2_gate_path && getenv("DS4_ROCM_DISABLE_RESIDENT_IQ2_SORTED") != NULL;
        const uint32_t use_sorted_pairs =
            n_tokens > 1u &&
            (!q4k_path || n_tokens >= 32u) &&
            !disable_resident_iq2_sorted;
        const uint32_t use_expert_tiles = use_sorted_pairs;
        /* DS4-TP-gfx1151 (patch 22): expert tile width, env-selectable for A/B.
         *
         * tile8 stages the ENTIRE K dimension for 8 tokens in LDS
         * (8*16*sizeof(block_q8_K) = 37376 B), which caps occupancy at 37.5%
         * in WGP mode (128 KB LDS / 37376 = 3 workgroups = 24 of 64 wave32).
         * tile4 needs 18688 B -> 75% occupancy, at the cost of 2x weight
         * re-reads (largely absorbed by the 32 MB MALL). Both kernels exist;
         * the branch below is compile-time today so only one is instantiated.
         * Bit-identical either way - same math, different tiling. */
        static int tile_m_env = -1;
        if (tile_m_env < 0) {
            const char *e = getenv("DS4_ROCM_EXPERT_TILE_M");
            tile_m_env = (e && e[0]) ? atoi(e) : 8;
            if (tile_m_env != 4 && tile_m_env != 8) tile_m_env = 8;
        }
        const uint32_t expert_tile_m = (uint32_t)tile_m_env;
        const uint32_t use_q4k_wmma = q4k_path && use_sorted_pairs &&
            routed_moe_q4k_wmma_enabled(expert_in_dim, expert_mid_dim,
                                        xq_blocks, gate_expert_bytes,
                                        gate_row_bytes, gate_w, up_w, xq,
                                        out_dim, down_expert_bytes,
                                        down_row_bytes, down_w);
        const uint32_t routing_tile_m = use_q4k_wmma ? 16u : expert_tile_m;
        const uint32_t write_gate_up = 0u;
        const uint32_t use_p2_sorted = 0u;
        const uint32_t use_atomic_down = use_expert_tiles && n_tokens >= 128u;
        const uint32_t use_gate_row2048 = !q4k_path && use_expert_tiles && n_tokens >= 128u;
        const uint32_t use_down_tile16 = !q4k_path && use_atomic_down && n_tokens >= 128u;
        const uint32_t use_decode_lut_gate =
            n_tokens == 1u && xq_blocks <= 16u;
        const uint32_t gate_row_span = 1024u;
        const uint32_t down_row_span = 1024u;
        const uint32_t use_down_row2048 = !q4k_path && use_atomic_down && use_down_tile16;
        const uint32_t use_direct_down_sum6 =
            n_tokens == 1u && n_expert <= DS4_ROCM_N_EXPERT_USED;
        uint32_t *sorted_pairs = NULL;
        uint32_t *sorted_offsets = NULL;
        uint32_t *sorted_counts = NULL;
        uint32_t *tile_total = NULL;
        uint32_t *tile_experts = NULL;
        uint32_t *tile_starts = NULL;
        uint32_t *tile16_total = NULL;
        uint32_t *tile16_experts = NULL;
        uint32_t *tile16_starts = NULL;
        ds4_q8_1_mmq_block *q4k_q81 = NULL;
        ds4_q8_1_mmq_block *down_q81 = NULL;
        uint32_t *iq2_gate_hot_dev = NULL;
        uint32_t tile_capacity = 0;
        uint32_t tile16_capacity = 0;
        dim3 xq_grid(xq_blocks, n_tokens, 1);
        q8_K_quantize_kernel<<<xq_grid, 256>>>(xq, (const float *)x->ptr, expert_in_dim, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "routed_moe x quantize launch");
        if (ok && (batch_stream_selected || batch_stream_split_selected)) {
            dim3 qgrid((expert_mid_dim + 127u) / 128u, pair_count, 1);
            if (batch_stream_split_selected) {
                if (stream_batch_resident_count != 0u) {
                    moe_gate_up_mid_qwarp32_ptrs_split_kernel<<<qgrid, 256>>>(
                            (float *)gate->ptr,
                            (float *)up->ptr,
                            (float *)mid->ptr,
                            resident_gate_slot_ptrs,
                            resident_up_slot_ptrs,
                            stream_batch_pair_missing,
                            0u,
                            xq,
                            (const int32_t *)selected_exec->ptr,
                            (const float *)weights->ptr,
                            gate_row_bytes,
                            xq_blocks,
                            expert_mid_dim,
                            n_expert,
                            0xffffffffu,
                            clamp);
                    ok = cuda_ok(cudaGetLastError(),
                                 "routed_moe streaming batch resident gate/up launch");
                }
                if (!ok) {
                    (void)cuda_stream_batch_selected_finish_pending_missing();
                } else {
                    ok = cuda_stream_batch_selected_finish_pending_missing();
                }
                if (ok && stream_batch_missing_count != 0u) {
                    moe_gate_up_mid_qwarp32_ptrs_split_kernel<<<qgrid, 256>>>(
                            (float *)gate->ptr,
                            (float *)up->ptr,
                            (float *)mid->ptr,
                            missing_gate_slot_ptrs,
                            missing_up_slot_ptrs,
                            stream_batch_pair_missing,
                            1u,
                            xq,
                            (const int32_t *)selected_exec->ptr,
                            (const float *)weights->ptr,
                            gate_row_bytes,
                            xq_blocks,
                            expert_mid_dim,
                            n_expert,
                            0xffffffffu,
                            clamp);
                    ok = cuda_ok(cudaGetLastError(),
                                 "routed_moe streaming batch missing gate/up launch");
                }
            } else {
                moe_gate_up_mid_qwarp32_ptrs_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        0xffffffffu,
                        clamp);
                ok = cuda_ok(cudaGetLastError(),
                             "routed_moe streaming batch gate/up launch");
            }
            if (ok) {
                dim3 midq_grid(midq_blocks, pair_count, 1);
                q8_K_quantize_kernel<<<midq_grid, 256>>>(
                        midq,
                        (const float *)mid->ptr,
                        expert_mid_dim,
                        pair_count);
                ok = cuda_ok(cudaGetLastError(), "routed_moe streaming batch mid quantize launch");
            }
            if (ok) {
                dim3 dgrid((out_dim + 31u) / 32u, n_tokens, 1);
                if (iq2_iq2_path) {
                    moe_down_iq2_sum_qwarp32_ptrs_batch_kernel<<<dgrid, 256>>>(
                            (float *)out->ptr,
                            down_slot_ptrs,
                            midq,
                            (const int32_t *)selected_exec->ptr,
                            down_row_bytes,
                            midq_blocks,
                            out_dim,
                            n_expert,
                            n_tokens);
                    ok = cuda_ok(cudaGetLastError(),
                                 "routed_moe streaming batch iq2 down launch");
                } else {
                    moe_down_sum6_qwarp32_ptrs_batch_kernel<<<dgrid, 256>>>(
                            (float *)out->ptr,
                            down_slot_ptrs,
                            midq,
                            (const int32_t *)selected_exec->ptr,
                            down_row_bytes,
                            midq_blocks,
                            out_dim,
                            n_expert,
                            n_tokens);
                    ok = cuda_ok(cudaGetLastError(),
                                 "routed_moe streaming batch down launch");
                }
            }
            if (ok) ok = cuda_stream_batch_selected_mark_inflight();
            return ok;
        }
        if (ok && use_sorted_pairs) {
            const uint32_t bucket_count = n_total_expert;
            const uint64_t counts_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
            const uint64_t offsets_bytes = (uint64_t)(bucket_count + 1u) * sizeof(uint32_t);
            const uint64_t cursors_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
            const uint64_t sorted_bytes = (uint64_t)pair_count * sizeof(uint32_t);
            tile_capacity = (pair_count + routing_tile_m - 1u) / routing_tile_m + bucket_count;
            tile16_capacity = use_down_tile16 ? ((pair_count + 15u) / 16u + bucket_count) : 0u;
            const uint64_t tile_offsets_bytes = (uint64_t)(bucket_count + 1u) * sizeof(uint32_t);
            const uint64_t tile_total_bytes = sizeof(uint32_t);
            const uint64_t tile_experts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile_starts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile16_offsets_bytes = use_down_tile16 ? (uint64_t)(bucket_count + 1u) * sizeof(uint32_t) : 0u;
            const uint64_t tile16_total_bytes = use_down_tile16 ? sizeof(uint32_t) : 0u;
            const uint64_t tile16_experts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile16_starts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile_offsets_off = counts_bytes + offsets_bytes + cursors_bytes + sorted_bytes;
            const uint64_t tile_total_off = tile_offsets_off + tile_offsets_bytes;
            const uint64_t tile_experts_off = tile_total_off + tile_total_bytes;
            const uint64_t tile_starts_off = tile_experts_off + tile_experts_bytes;
            const uint64_t tile16_offsets_off = tile_starts_off + tile_starts_bytes;
            const uint64_t tile16_total_off = tile16_offsets_off + tile16_offsets_bytes;
            const uint64_t tile16_experts_off = tile16_total_off + tile16_total_bytes;
            const uint64_t tile16_starts_off = tile16_experts_off + tile16_experts_bytes;
            const uint64_t iq2_gate_hot_off = tile16_starts_off + tile16_starts_bytes;
            const uint64_t iq2_gate_hot_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
            uint64_t q81_off = 0;
            uint64_t q81_count = 0;
            uint64_t q81_bytes = 0;
            uint64_t down_q81_off = 0;
            uint64_t down_q81_count = 0;
            uint64_t down_q81_bytes = 0;
            uint64_t scratch_bytes = 0;
            if (!routed_moe_align256_checked(iq2_gate_hot_off + iq2_gate_hot_bytes,
                                             &q81_off) ||
                (use_q4k_wmma &&
                 (!cuda_u64_mul_checked(n_tokens, (uint64_t)xq_blocks * 2u,
                                        &q81_count) ||
                  !cuda_u64_mul_checked(q81_count, sizeof(ds4_q8_1_mmq_block),
                                        &q81_bytes))) ||
                !routed_moe_u64_add_checked(q81_off, q81_bytes, &down_q81_off) ||
                (use_q4k_wmma &&
                 (!cuda_u64_mul_checked(pair_count, (uint64_t)midq_blocks * 2u,
                                        &down_q81_count) ||
                  !cuda_u64_mul_checked(down_q81_count, sizeof(ds4_q8_1_mmq_block),
                                        &down_q81_bytes))) ||
                !routed_moe_u64_add_checked(down_q81_off, down_q81_bytes,
                                            &scratch_bytes)) {
                return 0;
            }
            uint8_t *scratch = (uint8_t *)cuda_tmp_alloc(scratch_bytes,
                                                         "routed_moe sorted pairs");
            if (!scratch) {
                ok = 0;
            } else {
                uint32_t *counts = (uint32_t *)scratch;
                uint32_t *offsets = (uint32_t *)(scratch + counts_bytes);
                uint32_t *cursors = (uint32_t *)(scratch + counts_bytes + offsets_bytes);
                sorted_pairs = (uint32_t *)(scratch + counts_bytes + offsets_bytes + cursors_bytes);
                sorted_offsets = offsets;
                sorted_counts = counts;
                uint32_t *tile_offsets = (uint32_t *)(scratch + tile_offsets_off);
                tile_total = (uint32_t *)(scratch + tile_total_off);
                tile_experts = (uint32_t *)(scratch + tile_experts_off);
                tile_starts = (uint32_t *)(scratch + tile_starts_off);
                uint32_t *tile16_offsets = use_down_tile16 ? (uint32_t *)(scratch + tile16_offsets_off) : NULL;
                tile16_total = use_down_tile16 ? (uint32_t *)(scratch + tile16_total_off) : NULL;
                tile16_experts = use_down_tile16 ? (uint32_t *)(scratch + tile16_experts_off) : NULL;
                tile16_starts = use_down_tile16 ? (uint32_t *)(scratch + tile16_starts_off) : NULL;
                iq2_gate_hot_dev = (uint32_t *)(scratch + iq2_gate_hot_off);
                q4k_q81 = use_q4k_wmma ?
                    (ds4_q8_1_mmq_block *)(scratch + q81_off) : NULL;
                down_q81 = use_q4k_wmma ?
                    (ds4_q8_1_mmq_block *)(scratch + down_q81_off) : NULL;
                ok = cuda_ok(cudaMemset(counts, 0, counts_bytes), "routed_moe sorted counts clear");
                if (ok) {
                    moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                        counts,
                        (const int32_t *)selected_exec->ptr,
                        pair_count,
                        bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted count launch");
                }
                if (ok) {
                    moe_prefix_sorted_pairs_kernel<<<1, 1>>>(offsets, cursors, counts, bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted prefix launch");
                }
                if (ok) {
                    moe_scatter_sorted_pairs_deterministic_kernel<<<bucket_count, 1u>>>(
                        sorted_pairs,
                        offsets,
                        (const int32_t *)selected_exec->ptr,
                        pair_count,
                        bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted scatter launch");
                }
                if (ok && use_expert_tiles) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile_offsets, tile_total, counts, routing_tile_m, bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile offsets launch");
                }
                if (ok && use_expert_tiles) {
                    moe_build_expert_tiles_kernel<<<(bucket_count + 255u) / 256u, 256>>>(
                            tile_experts, tile_starts, tile_offsets, counts, routing_tile_m, bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tiles launch");
                }
                if (ok && use_expert_tiles && use_down_tile16) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile16_offsets, tile16_total, counts, 16u, bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 offsets launch");
                }
                if (ok && use_expert_tiles && use_down_tile16) {
                    moe_build_expert_tiles_kernel<<<(bucket_count + 255u) / 256u, 256>>>(
                            tile16_experts, tile16_starts, tile16_offsets, counts, 16u, bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 launch");
                }
                if (ok && use_q4k_wmma) {
                    moe_q8_K_to_q8_1_mmq_kernel<<<dim3(n_tokens, xq_blocks), 32>>>(
                            xq, q4k_q81, n_tokens, xq_blocks);
                    ok = cuda_ok(cudaGetLastError(),
                                 "routed_moe Q4_K Q8_1 repack launch");
                }
            }
        }
        uint32_t iq2_gate_hot_count = 0u;
        uint32_t iq2_gate_hot_max = 0u;
        const uint32_t iq2_gate_hot_threshold = 8u;
        const uint32_t iq2_down_hot_threshold = 8u;
        uint32_t h_iq2_gate_hot[DS4_ROCM_MAX_N_EXPERT] = {0};
        const uint32_t use_iq2_gate_wmma =
            ok && iq2_gate_path && n_tokens > 1u && n_expert == 6u && !write_gate_up &&
            sorted_pairs && sorted_offsets && sorted_counts && tile_experts && iq2_gate_hot_dev && use_expert_tiles &&
            (expert_in_dim % 16u) == 0u && (expert_mid_dim % 16u) == 0u &&
            !g_quality_mode;
        if (use_iq2_gate_wmma) {
            uint32_t h_counts[DS4_ROCM_MAX_N_EXPERT] = {0};
            if (!cuda_ok(cudaMemcpy(h_counts, sorted_counts, n_total_expert * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                         "routed_moe iq2 gate wmma counts copy")) {
                ok = 0;
            } else {
                for (uint32_t e = 0; e < n_total_expert; e++) {
                    const uint32_t c = h_counts[e];
                    if (c >= iq2_gate_hot_threshold) {
                        h_iq2_gate_hot[iq2_gate_hot_count++] = e;
                        if (c > iq2_gate_hot_max) iq2_gate_hot_max = c;
                    }
                }
                if (iq2_gate_hot_count != 0u &&
                    !cuda_ok(cudaMemcpy(iq2_gate_hot_dev, h_iq2_gate_hot,
                                        iq2_gate_hot_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                             "routed_moe iq2 gate hot copy")) {
                    ok = 0;
                }
            }
        }
        const uint32_t iq2_gate_scalar_max = iq2_gate_hot_count != 0u ? iq2_gate_hot_threshold : 0u;
        const int use_iq2_hot_f16_mid = use_iq2_gate_wmma && iq2_gate_hot_count != 0u &&
            iq2_gate_hot_threshold == iq2_down_hot_threshold && (out_dim & 1u) == 0u &&
            !g_quality_mode;
        half *iq2_hot_mid_h = use_iq2_hot_f16_mid ? (half *)gate->ptr : NULL;
        const int use_iq2_x_f16 = use_iq2_gate_wmma && iq2_gate_hot_count != 0u &&
            up->bytes >= (uint64_t)n_tokens * expert_in_dim * sizeof(half);
        half *iq2_x_h = use_iq2_x_f16 ? (half *)up->ptr : NULL;
        if (ok && use_iq2_x_f16) {
            const uint64_t xh_count = (uint64_t)n_tokens * expert_in_dim;
            f32_to_f16_kernel<<<(xh_count + 255u) / 256u, 256>>>(iq2_x_h, (const float *)x->ptr, xh_count);
            ok = cuda_ok(cudaGetLastError(), "routed_moe iq2 gate x f16 launch");
        }
        int split_gateup_done = 0;
        if (ok && split_selected) {
            const int split_supported =
                iq2_gate_path &&
                n_tokens == 1u &&
                n_expert <= DS4_ROCM_N_EXPERT_USED &&
                !q4k_path &&
                !sorted_pairs &&
                stream_resident_mask != 0 &&
                stream_missing_mask != 0;
            if (split_supported) {
                dim3 qgrid((expert_mid_dim + 127u) / 128u, pair_count, 1);
                if (use_decode_lut_gate) {
                    moe_gate_up_mid_decode_lut_qwarp32_ptrs_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        stream_resident_mask,
                        clamp);
                } else {
                    moe_gate_up_mid_qwarp32_ptrs_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        stream_resident_mask,
                        clamp);
                }
                ok = cuda_ok(cudaGetLastError(), "routed_moe split resident gate/up launch");
                if (!ok) {
                    (void)cuda_stream_selected_finish_pending_missing(0);
                } else {
                    ok = cuda_stream_selected_finish_pending_missing(0);
                }
                if (ok && use_decode_lut_gate) {
                    moe_gate_up_mid_decode_lut_qwarp32_ptrs_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        stream_missing_mask,
                        clamp);
                } else if (ok) {
                    moe_gate_up_mid_qwarp32_ptrs_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        stream_missing_mask,
                        clamp);
                }
                if (ok) ok = cuda_ok(cudaGetLastError(), "routed_moe split missing gate/up launch");
                split_gateup_done = ok;
            } else {
                ok = cuda_stream_selected_finish_pending_missing(
                        stream_resident_mask | stream_missing_mask);
            }
        }
        if (ok && !split_gateup_done) {
            dim3 mgrid((expert_mid_dim + 31u) / 32u, pair_count, 1);
            if (ok && sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts && tile_total && tile_experts && tile_starts) {
                if (q4k_path) {
                    if (use_q4k_wmma) {
                        const uint32_t wmma_min_count = 6u;
                        const size_t wmma_smem =
                            (16u * 36u + 64u * 76u) * sizeof(int32_t);
                        dim3 wgrid((expert_mid_dim + 63u) / 64u,
                                   tile_capacity, 1);
                        moe_q4K_routed_wmma_kernel<16><<<wgrid, 256, wmma_smem>>>(
                            gate_w, q4k_q81,
                            (float *)gate->ptr, sorted_pairs, sorted_offsets,
                            sorted_counts, tile_total, tile_experts, tile_starts,
                            n_tokens, xq_blocks, expert_mid_dim, n_expert,
                            gate_expert_bytes, gate_row_bytes, wmma_min_count);
                        ok = cuda_ok(cudaGetLastError(),
                                     "routed_moe Q4_K gate WMMA launch");
                        if (ok) {
                            moe_q4K_routed_wmma_kernel<16><<<wgrid, 256, wmma_smem>>>(
                                up_w, q4k_q81,
                                (float *)up->ptr, sorted_pairs, sorted_offsets,
                                sorted_counts, tile_total, tile_experts, tile_starts,
                                n_tokens, xq_blocks, expert_mid_dim, n_expert,
                                gate_expert_bytes, gate_row_bytes, wmma_min_count);
                            ok = cuda_ok(cudaGetLastError(),
                                         "routed_moe Q4_K up WMMA launch");
                        }
                        if (ok) {
                            dim3 cold_grid((expert_mid_dim + 31u) / 32u,
                                           tile_capacity, 1);
                            moe_gate_up_q4K_cold_tile16_kernel<<<cold_grid, 256>>>(
                                (float *)gate->ptr, (float *)up->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets,
                                sorted_counts, tile_total, tile_experts, tile_starts,
                                gate_expert_bytes, gate_row_bytes, xq_blocks,
                                expert_mid_dim, n_expert, wmma_min_count);
                            ok = cuda_ok(cudaGetLastError(),
                                         "routed_moe Q4_K DP4A cold launch");
                        }
                        if (ok) {
                            dim3 egrid(1u, tile_capacity, 1u);
                            moe_gate_up_mid_q4K_routed_epilogue_kernel<<<egrid, 16>>>(
                                (float *)gate->ptr, (float *)up->ptr,
                                (float *)mid->ptr, sorted_pairs, sorted_offsets,
                                sorted_counts, tile_total, tile_experts, tile_starts,
                                (const float *)weights->ptr, expert_mid_dim,
                                n_expert, write_gate_up, clamp);
                            ok = cuda_ok(cudaGetLastError(),
                                         "routed_moe Q4_K WMMA epilogue launch");
                        }
                    } else {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    if (routing_tile_m == 8u) {
                        moe_gate_up_mid_q4K_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            0u, write_gate_up, clamp);
                    } else {
                        moe_gate_up_mid_q4K_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            0u, write_gate_up, clamp);
                    }
                    }
                } else if (use_gate_row2048) {
                    if (gate_row_span == 512u) {
                        dim3 tgrid((expert_mid_dim + 511u) / 512u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<512><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            iq2_gate_scalar_max, write_gate_up, clamp);
                    } else if (gate_row_span == 1024u) {
                        dim3 tgrid((expert_mid_dim + 1023u) / 1024u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<1024><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            iq2_gate_scalar_max, write_gate_up, clamp);
                    } else {
                        dim3 tgrid((expert_mid_dim + 2047u) / 2048u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_row2048_kernel<<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            iq2_gate_scalar_max, write_gate_up, clamp);
                    }
                } else if (routing_tile_m == 8u) {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    moe_gate_up_mid_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        iq2_gate_scalar_max, write_gate_up, clamp);
                } else {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    moe_gate_up_mid_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        iq2_gate_scalar_max, write_gate_up, clamp);
                }
            } else if (ok && sorted_pairs && use_p2_sorted) {
                dim3 p2_mgrid((expert_mid_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_gate_up_mid_sorted_p2_qwarp32_kernel<<<p2_mgrid, 256>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    xq,
                    sorted_pairs,
                    (const int32_t *)selected_exec->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    xq_blocks,
                    expert_mid_dim,
                    n_expert,
                    pair_count,
                    clamp);
            } else if (ok && sorted_pairs) {
                if (q4k_path) {
                    moe_gate_up_mid_q4K_sorted_qwarp32_kernel<<<mgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        sorted_pairs,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                } else {
                    moe_gate_up_mid_sorted_qwarp32_kernel<<<mgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        sorted_pairs,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                }
            } else if (ok) {
                dim3 qgrid((expert_mid_dim + 127u) / 128u, pair_count, 1);
                if (q4k_path) {
                    moe_gate_up_mid_decode_q4K_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        clamp);
                } else if (use_decode_lut_gate) {
                    moe_gate_up_mid_decode_lut_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        0xffffffffu,
                        clamp);
                } else {
                    moe_gate_up_mid_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        0xffffffffu,
                        clamp);
                }
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe gate/up launch");
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
            if (ok && use_iq2_gate_wmma && iq2_gate_hot_count != 0u) {
                constexpr uint32_t bm = 16u, bn = 16u, bk = 16u;
                const uint32_t wmma_mtiles = 4u;
                if (wmma_mtiles == 4u) {
                    constexpr uint32_t mt = 4u;
                    const dim3 block(32u * mt, 1u, 1u);
                    const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn),
                                    (iq2_gate_hot_max + mt * bm - 1u) / (mt * bm),
                                    iq2_gate_hot_count);
                    const size_t shmem_n2 = (mt * bm * bk + 4u * bk * bn) * sizeof(half) +
                                            (4u * mt * bm * bn) * sizeof(float);
                    if (use_iq2_hot_f16_mid && use_iq2_x_f16) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<4,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                                NULL, iq2_hot_mid_h, gate_w, up_w, (const float *)x->ptr, iq2_x_h,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else if (use_iq2_hot_f16_mid) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<4,16,16,16,true><<<grid, block, shmem_n2>>>(
                                NULL, iq2_hot_mid_h, gate_w, up_w, (const float *)x->ptr, NULL,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else if (use_iq2_x_f16) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<4,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                                (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, iq2_x_h,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<4,16,16,16><<<grid, block, shmem_n2>>>(
                                (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, NULL,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    }
                } else {
                    constexpr uint32_t mt = 8u;
                    const dim3 block(32u * mt, 1u, 1u);
                    const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn),
                                    (iq2_gate_hot_max + mt * bm - 1u) / (mt * bm),
                                    iq2_gate_hot_count);
                    const size_t shmem_n2 = (mt * bm * bk + 4u * bk * bn) * sizeof(half) +
                                            (4u * mt * bm * bn) * sizeof(float);
                    if (use_iq2_hot_f16_mid && use_iq2_x_f16) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<8,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                                NULL, iq2_hot_mid_h, gate_w, up_w, (const float *)x->ptr, iq2_x_h,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else if (use_iq2_hot_f16_mid) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<8,16,16,16,true><<<grid, block, shmem_n2>>>(
                                NULL, iq2_hot_mid_h, gate_w, up_w, (const float *)x->ptr, NULL,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else if (use_iq2_x_f16) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<8,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                                (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, iq2_x_h,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<8,16,16,16><<<grid, block, shmem_n2>>>(
                                (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, NULL,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    }
                }
                ok = cuda_ok(cudaGetLastError(), "routed_moe iq2 wmma hot gate/up launch");
            }
#endif
        }
        const uint32_t use_iq2_q2_float_down =
            ok && iq2_path && n_tokens > 1u &&
            n_expert <= DS4_ROCM_N_EXPERT_USED &&
            sorted_pairs && sorted_offsets && sorted_counts && tile_experts;
        if (ok && !use_iq2_q2_float_down) {
            dim3 midq_grid(midq_blocks, pair_count, 1);
            q8_K_quantize_kernel<<<midq_grid, 256>>>(midq, (const float *)mid->ptr, expert_mid_dim, pair_count);
            ok = cuda_ok(cudaGetLastError(), "routed_moe mid quantize launch");
            if (ok && use_q4k_wmma) {
                moe_q8_K_to_q8_1_mmq_kernel<<<dim3(pair_count, midq_blocks), 32>>>(
                        midq, down_q81, pair_count, midq_blocks);
                ok = cuda_ok(cudaGetLastError(),
                             "routed_moe Q4_K down Q8_1 repack launch");
            }
        }
        int direct_iq2_down_done = 0;
        if (ok && iq2_iq2_path) {
            dim3 dgrid((out_dim + 31u) / 32u, n_tokens, 1);
            if (split_gateup_done) {
                moe_down_iq2_sum_qwarp32_ptrs_batch_kernel<<<dgrid, 256>>>(
                        (float *)out->ptr,
                        down_slot_ptrs,
                        midq,
                        (const int32_t *)selected_exec->ptr,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert,
                        n_tokens);
            } else {
                moe_down_iq2_sum_qwarp32_batch_kernel<<<dgrid, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert,
                        n_tokens);
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe iq2 down launch");
            direct_iq2_down_done = ok;
        }
        int split_ptr_down_done = 0;
        if (ok && !direct_iq2_down_done && split_gateup_done) {
            moe_down_sum6_qwarp32_ptrs_kernel<<<(out_dim + 31u) / 32u, 256>>>(
                    (float *)out->ptr,
                    down_slot_ptrs,
                    midq,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert);
            ok = cuda_ok(cudaGetLastError(), "routed_moe split ptr down launch");
            split_ptr_down_done = ok;
        }
        if (ok) {
            if (direct_iq2_down_done) {
                /* The IQ2 direct-sum kernel writes final token rows. */
            } else if (split_ptr_down_done) {
                /* The split pointer-table path writes the final token row. */
            } else if (use_iq2_q2_float_down) {
                ok = routed_moe_q2_float_down_launch(
                        out, down, mid, iq2_hot_mid_h, use_iq2_hot_f16_mid, down_w,
                        sorted_counts, sorted_offsets, sorted_pairs, tile_experts,
                        n_tokens, n_total_expert, n_expert, expert_mid_dim, out_dim,
                        down_expert_bytes, down_row_bytes);
            } else {
            dim3 dgrid((out_dim + 31u) / 32u, pair_count, 1);
            uint32_t *down_tile_total = tile_total;
            uint32_t *down_tile_experts = tile_experts;
            uint32_t *down_tile_starts = tile_starts;
            uint32_t down_tile_capacity = tile_capacity;
            if (use_down_tile16 && tile16_total && tile16_experts && tile16_starts) {
                down_tile_total = tile16_total;
                down_tile_experts = tile16_experts;
                down_tile_starts = tile16_starts;
                down_tile_capacity = tile16_capacity;
            }
            if (use_direct_down_sum6) {
                dim3 sgrid((out_dim + 31u) / 32u, 1, 1);
                if (q4k_path) {
                    moe_down_q4K_sum6_qwarp32_kernel<<<sgrid, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                } else {
                    moe_down_sum6_qwarp32_kernel<<<sgrid, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                }
            } else if (use_atomic_down) {
                uint64_t n = (uint64_t)n_tokens * out_dim;
                zero_kernel<<<(n + 255u) / 256u, 256>>>((float *)out->ptr, n);
                ok = cuda_ok(cudaGetLastError(), "routed_moe atomic zero launch");
            }
            if (use_direct_down_sum6) {
                /* The direct decode kernel writes the final token row. */
            } else if (sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts &&
                down_tile_total && down_tile_experts && down_tile_starts) {
                if (q4k_path) {
                    if (use_q4k_wmma) {
                        if (routing_tile_m != 16u || down_tile_total != tile_total ||
                            down_tile_experts != tile_experts || down_tile_starts != tile_starts ||
                            !down_q81 || midq_blocks != 8u || out_dim != 4096u) {
                            fprintf(stderr, DS4_GPU_LOG_PREFIX
                                    "Q4_K WMMA down contract violation: routing_tile_m=%u "
                                    "midq_blocks=%u out_dim=%u descriptors_primary=%d q81=%d\n",
                                    routing_tile_m, midq_blocks, out_dim,
                                    down_tile_total == tile_total && down_tile_experts == tile_experts &&
                                    down_tile_starts == tile_starts, down_q81 != NULL);
                            return 0;
                        }
                        static int logged_down_launch = 0;
                        if (!logged_down_launch) {
                            logged_down_launch = 1;
                            fprintf(stderr, DS4_GPU_LOG_PREFIX
                                    "Q4_K WMMA dispatch gate=1 up=1 down_hot=1 down_cold=1 "
                                    "fallback=1 routing_tile_m=16 down_threshold=1 atomic=%u\n",
                                    use_atomic_down);
                        }
                        /* Measured Stage-2 crossover: threshold 1 retained all
                         * tested buckets (10.7201x geomean, no >5% regression). */
                        const uint32_t down_wmma_min_count = 1u;
                        const size_t down_wmma_smem =
                            (16u * 36u + 64u * 76u) * sizeof(int32_t);
                        dim3 wgrid((out_dim + 63u) / 64u, down_tile_capacity, 1);
                        if (use_atomic_down) {
                            moe_down_q4K_routed_wmma_kernel<16, true><<<wgrid, 256, down_wmma_smem>>>(
                                (float *)out->ptr, down_w, down_q81, sorted_pairs,
                                sorted_offsets, sorted_counts, down_tile_total,
                                down_tile_experts, down_tile_starts, pair_count,
                                midq_blocks, out_dim, n_expert, down_expert_bytes,
                                down_row_bytes, down_wmma_min_count);
                        } else {
                            moe_down_q4K_routed_wmma_kernel<16, false><<<wgrid, 256, down_wmma_smem>>>(
                                (float *)down->ptr, down_w, down_q81, sorted_pairs,
                                sorted_offsets, sorted_counts, down_tile_total,
                                down_tile_experts, down_tile_starts, pair_count,
                                midq_blocks, out_dim, n_expert, down_expert_bytes,
                                down_row_bytes, down_wmma_min_count);
                        }
                        ok = cuda_ok(cudaGetLastError(), "routed_moe Q4_K down WMMA launch");
                        if (ok) {
                            dim3 cgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                            moe_down_q4K_cold_tile16_kernel<<<cgrid, 256>>>(
                                use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                                down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                down_tile_total, down_tile_experts, down_tile_starts,
                                down_expert_bytes, down_row_bytes, midq_blocks, out_dim,
                                n_expert, down_wmma_min_count, use_atomic_down);
                            ok = cuda_ok(cudaGetLastError(), "routed_moe Q4_K down cold DP4A launch");
                        }
                    } else if (routing_tile_m == 8u) {
                        dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                        moe_down_q4K_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else if (routing_tile_m == 4u) {
                        dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                        moe_down_q4K_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else {
                        fprintf(stderr, DS4_GPU_LOG_PREFIX
                                "Q4_K legacy down contract violation: routing_tile_m=%u\n",
                                routing_tile_m);
                        return 0;
                    }
                } else if (use_down_row2048) {
                    if (down_row_span == 512u) {
                        dim3 tgrid((out_dim + 511u) / 512u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<512><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else if (down_row_span == 1024u) {
                        dim3 tgrid((out_dim + 1023u) / 1024u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<1024><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else {
                        dim3 tgrid((out_dim + 2047u) / 2048u, down_tile_capacity, 1);
                        moe_down_expert_tile16_row2048_kernel<<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    }
                } else if (use_down_tile16) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile16_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else if (routing_tile_m == 8u) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                }
            } else if (sorted_pairs && use_p2_sorted) {
                dim3 p2_dgrid((out_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_down_sorted_p2_qwarp32_kernel<<<p2_dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    sorted_pairs,
                    (const int32_t *)selected_exec->ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert,
                    pair_count);
            } else if (sorted_pairs) {
                if (q4k_path) {
                    moe_down_q4K_sorted_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        sorted_pairs,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                } else {
                    moe_down_sorted_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        sorted_pairs,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                }
            } else {
                if (q4k_path) {
                    moe_down_q4K_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                } else {
                    moe_down_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                }
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe down launch");
            }
        }
        if (ok && !direct_iq2_down_done && !use_atomic_down &&
            !use_direct_down_sum6 && !use_iq2_q2_float_down) {
            uint64_t n = (uint64_t)n_tokens * out_dim;
            moe_sum_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
            ok = cuda_ok(cudaGetLastError(), "routed_moe sum launch");
        }
        if (ok && compact_selected) ok = cuda_stream_selected_mark_inflight();
        return ok;
    }

    const ds4_rocm_runtime_config *cfg = cuda_runtime_config();
    if (q2k_path && (batch_stream_selected || batch_stream_split_selected)) {
        uint32_t gate_rows_per_block = cfg->moe_decode_gate_rpb;
        if (gate_rows_per_block == 0u) gate_rows_per_block = 1u;
        const uint32_t gate_threads = gate_rows_per_block * 32u;
        uint32_t down_rows_per_block = cfg->moe_decode_down_rpb;
        if (down_rows_per_block == 0u) down_rows_per_block = 1u;
        const uint32_t down_threads = down_rows_per_block * 32u;
        const int store_gate_up = (g_quality_mode || cfg->graph_dump) ? 1 : 0;
        dim3 gate_grid((expert_mid_dim + gate_rows_per_block - 1u) / gate_rows_per_block,
                       pair_count,
                       1);
        if (batch_stream_split_selected) {
            if (stream_batch_resident_count != 0u) {
                moe_gate_up_mid_q2K_rows_w32_ptrs_kernel<<<gate_grid, gate_threads>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        resident_gate_slot_ptrs,
                        resident_up_slot_ptrs,
                        stream_batch_pair_missing,
                        0u,
                        (const float *)x->ptr,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        expert_in_dim,
                        expert_mid_dim,
                        n_expert,
                        0xffffffffu,
                        clamp,
                        store_gate_up);
                ok = cuda_ok(cudaGetLastError(),
                             "routed_moe q2 streaming batch resident gate/up launch");
            }
            if (!ok) {
                (void)cuda_stream_batch_selected_finish_pending_missing();
            } else {
                ok = cuda_stream_batch_selected_finish_pending_missing();
            }
            if (ok && stream_batch_missing_count != 0u) {
                moe_gate_up_mid_q2K_rows_w32_ptrs_kernel<<<gate_grid, gate_threads>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        missing_gate_slot_ptrs,
                        missing_up_slot_ptrs,
                        stream_batch_pair_missing,
                        1u,
                        (const float *)x->ptr,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        expert_in_dim,
                        expert_mid_dim,
                        n_expert,
                        0xffffffffu,
                        clamp,
                        store_gate_up);
                ok = cuda_ok(cudaGetLastError(),
                             "routed_moe q2 streaming batch missing gate/up launch");
            }
        } else {
            moe_gate_up_mid_q2K_rows_w32_ptrs_kernel<<<gate_grid, gate_threads>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_slot_ptrs,
                    up_slot_ptrs,
                    NULL,
                    0u,
                    (const float *)x->ptr,
                    (const int32_t *)selected_exec->ptr,
                    (const float *)weights->ptr,
                    gate_row_bytes,
                    expert_in_dim,
                    expert_mid_dim,
                    n_expert,
                    0xffffffffu,
                    clamp,
                    store_gate_up);
            ok = cuda_ok(cudaGetLastError(),
                         "routed_moe q2 streaming batch gate/up launch");
        }
        if (ok) {
            dim3 down_grid((out_dim + down_rows_per_block - 1u) / down_rows_per_block,
                           n_tokens,
                           1);
            moe_down_q2K_sum_rows_w32_ptrs_batch_kernel<<<down_grid, down_threads>>>(
                    (float *)out->ptr,
                    down_slot_ptrs,
                    (const float *)mid->ptr,
                    (const int32_t *)selected_exec->ptr,
                    n_tokens,
                    expert_mid_dim,
                    out_dim,
                    down_row_bytes,
                    n_expert);
            ok = cuda_ok(cudaGetLastError(),
                         "routed_moe q2 streaming batch down launch");
        }
        if (ok) ok = cuda_stream_batch_selected_mark_inflight();
        return ok;
    }

    if (q2k_path && n_tokens >= 32u && !cfg->graph_dump) {
        const uint32_t bucket_count = n_total_expert;
        const uint64_t counts_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
        const uint64_t offsets_bytes = (uint64_t)(bucket_count + 1u) * sizeof(uint32_t);
        const uint64_t cursors_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
        uint64_t sorted_bytes = 0;
        const uint64_t hot_gate_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
        const uint64_t hot_down_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
        const uint64_t f16_low_gate_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
        const uint64_t f16_low_down_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        const int moe_wmma_hot = !g_quality_mode &&
                                 expert_in_dim % 16u == 0u &&
                                 expert_mid_dim % 16u == 0u &&
                                 out_dim % 16u == 0u;
#else
        const int moe_wmma_hot = 0;
#endif
        uint64_t f16_mid_bytes = 0;
        uint64_t f16_down_bytes = 0;
        uint64_t wmma_x_bytes = 0;
        if (!cuda_u64_mul_checked(pair_count64, sizeof(uint32_t), &sorted_bytes) ||
            (moe_wmma_hot && (
                !cuda_u64_mul3_checked(pair_count64, expert_mid_dim, sizeof(__half), &f16_mid_bytes) ||
                !cuda_u64_mul3_checked(pair_count64, out_dim, sizeof(__half), &f16_down_bytes) ||
                !cuda_u64_mul3_checked(n_tokens, expert_in_dim, sizeof(__half), &wmma_x_bytes)))) {
            return 0;
        }
        uint64_t wmma_list_base = 0;
        uint64_t base_scratch_end = 0;
        uint64_t tmp64 = 0;
        uint64_t f16_mid_off = 0;
        uint64_t f16_down_off = 0;
        uint64_t wmma_x_off = 0;
        uint64_t scratch_bytes = 0;
        if (!routed_moe_u64_add_checked(counts_bytes, offsets_bytes, &wmma_list_base) ||
            !routed_moe_u64_add_checked(wmma_list_base, cursors_bytes, &wmma_list_base) ||
            !routed_moe_u64_add_checked(wmma_list_base, sorted_bytes, &wmma_list_base) ||
            !routed_moe_u64_add_checked(wmma_list_base, hot_gate_bytes, &base_scratch_end) ||
            !routed_moe_u64_add_checked(base_scratch_end, hot_down_bytes, &base_scratch_end) ||
            !routed_moe_u64_add_checked(base_scratch_end, f16_low_gate_bytes, &base_scratch_end) ||
            !routed_moe_u64_add_checked(base_scratch_end, f16_low_down_bytes, &base_scratch_end) ||
            !routed_moe_align256_checked(base_scratch_end, &f16_mid_off) ||
            !routed_moe_u64_add_checked(f16_mid_off, f16_mid_bytes, &tmp64) ||
            !routed_moe_align256_checked(tmp64, &f16_down_off) ||
            !routed_moe_u64_add_checked(f16_down_off, f16_down_bytes, &tmp64) ||
            !routed_moe_align256_checked(tmp64, &wmma_x_off) ||
            !routed_moe_u64_add_checked(wmma_x_off, wmma_x_bytes, &tmp64) ||
            !routed_moe_align256_checked(tmp64, &scratch_bytes)) {
            return 0;
        }
        uint8_t *scratch = (uint8_t *)cuda_tmp_alloc(scratch_bytes, "routed_moe q2 expert batch buckets");
        if (!scratch) return 0;
        uint32_t *counts = (uint32_t *)scratch;
        uint32_t *offsets = (uint32_t *)(scratch + counts_bytes);
        uint32_t *cursors = (uint32_t *)(scratch + counts_bytes + offsets_bytes);
        uint32_t *sorted_pairs = (uint32_t *)(scratch + counts_bytes + offsets_bytes + cursors_bytes);
        uint32_t *wmma_gate_hot_dev = (uint32_t *)(scratch + wmma_list_base);
        uint32_t *wmma_down_hot_dev = (uint32_t *)(scratch + wmma_list_base + hot_gate_bytes);
        uint32_t *wmma_gate_f16_low_dev = (uint32_t *)(scratch + wmma_list_base + hot_gate_bytes + hot_down_bytes);
        uint32_t *wmma_down_f16_low_dev = (uint32_t *)(scratch + wmma_list_base + hot_gate_bytes + hot_down_bytes + f16_low_gate_bytes);
        __half *wmma_mid_h = moe_wmma_hot ? (__half *)(scratch + f16_mid_off) : NULL;
        __half *wmma_down_h = moe_wmma_hot ? (__half *)(scratch + f16_down_off) : NULL;
        __half *wmma_x_h = moe_wmma_hot ? (__half *)(scratch + wmma_x_off) : NULL;
        ok = cuda_ok(cudaMemset(counts, 0, counts_bytes), "routed_moe q2 expert counts clear");
        if (ok) {
            moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                    counts,
                    (const int32_t *)selected_exec->ptr,
                    pair_count,
                    bucket_count);
            ok = cuda_ok(cudaGetLastError(), "routed_moe q2 expert count launch");
        }
        if (ok) {
            moe_prefix_sorted_pairs_kernel<<<1, 1>>>(offsets, cursors, counts, bucket_count);
            ok = cuda_ok(cudaGetLastError(), "routed_moe q2 expert prefix launch");
        }
        if (ok) {
            moe_scatter_sorted_pairs_deterministic_kernel<<<bucket_count, 1u>>>(
                    sorted_pairs,
                    offsets,
                    (const int32_t *)selected_exec->ptr,
                    pair_count,
                    bucket_count);
            ok = cuda_ok(cudaGetLastError(), "routed_moe q2 expert scatter launch");
        }
        if (ok && moe_wmma_hot) {
            const uint64_t xh_count = (uint64_t)n_tokens * expert_in_dim;
            f32_to_f16_kernel<<<(xh_count + 255u) / 256u, 256>>>(wmma_x_h, (const float *)x->ptr, xh_count);
            ok = cuda_ok(cudaGetLastError(), "routed_moe q2 wmma x f16 launch");
        }
        if (!ok) return 0;

        uint32_t wmma_f16_hot_count = 0u, wmma_f16_hot_max = 0u;
        uint32_t wmma_f16_low_count = 0u, wmma_f16_low_max = 0u;
        uint32_t h_counts[DS4_ROCM_MAX_N_EXPERT] = {0};
        uint32_t h_f16_hot[DS4_ROCM_MAX_N_EXPERT] = {0};
        uint32_t h_f16_low[DS4_ROCM_MAX_N_EXPERT] = {0};
        const uint32_t wmma_hot_threshold = 8u;
        const uint32_t wmma_f16_low_threshold = 64u;
        if (moe_wmma_hot) {
            if (!cuda_ok(cudaMemcpy(h_counts, counts, bucket_count * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                         "routed_moe q2 wmma counts copy")) return 0;
            for (uint32_t e = 0; e < bucket_count; e++) {
                const uint32_t c = h_counts[e];
                if (c >= wmma_hot_threshold) {
                    if (c < wmma_f16_low_threshold) {
                        h_f16_low[wmma_f16_low_count++] = e;
                        if (c > wmma_f16_low_max) wmma_f16_low_max = c;
                    } else {
                        h_f16_hot[wmma_f16_hot_count++] = e;
                        if (c > wmma_f16_hot_max) wmma_f16_hot_max = c;
                    }
                }
            }
        }
        const uint32_t gate_rpb = 16u;
        const uint32_t down_rpb = 16u;
        const uint32_t gate_threads = gate_rpb * 32u;
        const uint32_t down_threads = down_rpb * 32u;
        const size_t gate_shmem = 4u * 256u * sizeof(float);
        const size_t down_shmem = 4u * 256u * sizeof(float);
        const uint32_t scalar_max = moe_wmma_hot && (wmma_f16_low_count != 0u || wmma_f16_hot_count != 0u)
            ? wmma_hot_threshold : 0u;
        dim3 gate_grid((expert_mid_dim + gate_rpb - 1u) / gate_rpb, bucket_count, 1);
        moe_gate_up_mid_q2K_expert_batch_sharedx_kernel<4><<<gate_grid, gate_threads, gate_shmem>>>(
                (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, (const float *)weights->ptr,
                counts, offsets, sorted_pairs, 1u, scalar_max, expert_in_dim, expert_mid_dim,
                gate_expert_bytes, gate_row_bytes, n_expert, clamp);
        if (!cuda_ok(cudaGetLastError(), "routed_moe q2 expert gate/up launch")) return 0;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        if (moe_wmma_hot && wmma_f16_low_count != 0u) {
            constexpr uint32_t mt4 = 4u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt4, 1u, 1u);
            const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn),
                            (wmma_f16_low_max + mt4 * bm - 1u) / (mt4 * bm),
                            wmma_f16_low_count);
            const size_t shmem_n2 = (mt4 * bm * bk + 4u * bk * bn) * sizeof(half) +
                                    (4u * mt4 * bm * bn) * sizeof(float);
            if (!cuda_ok(cudaMemcpy(wmma_gate_f16_low_dev, h_f16_low,
                                    wmma_f16_low_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma f16-low hot copy")) return 0;
            moe_gate_up_mid_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                    NULL, wmma_mid_h, gate_w, up_w, (const float *)x->ptr, wmma_x_h, (const float *)weights->ptr,
                    counts, offsets, sorted_pairs, wmma_gate_f16_low_dev, wmma_f16_low_count,
                    expert_in_dim, expert_mid_dim, gate_expert_bytes, gate_row_bytes, n_expert, clamp);
            if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma f16-low gate/up launch")) return 0;
        }
        if (moe_wmma_hot && wmma_f16_hot_count != 0u) {
            constexpr uint32_t mt = 8u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn),
                            (wmma_f16_hot_max + mt * bm - 1u) / (mt * bm),
                            wmma_f16_hot_count);
            const size_t shmem_n2 = (mt * bm * bk + 4u * bk * bn) * sizeof(half) +
                                    (4u * mt * bm * bn) * sizeof(float);
            if (!cuda_ok(cudaMemcpy(wmma_gate_hot_dev, h_f16_hot,
                                    wmma_f16_hot_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma f16-mid hot copy")) return 0;
            moe_gate_up_mid_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                    NULL, wmma_mid_h, gate_w, up_w, (const float *)x->ptr, wmma_x_h, (const float *)weights->ptr,
                    counts, offsets, sorted_pairs, wmma_gate_hot_dev, wmma_f16_hot_count,
                    expert_in_dim, expert_mid_dim, gate_expert_bytes, gate_row_bytes, n_expert, clamp);
            if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma f16-mid gate/up launch")) return 0;
        }
#endif
        dim3 down_grid((out_dim + down_rpb - 1u) / down_rpb, bucket_count, 1);
        if (moe_wmma_hot) {
            moe_down_q2K_expert_batch_sharedmid_kernel<4,false,true><<<down_grid, down_threads, down_shmem>>>(
                    NULL, wmma_down_h, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes, n_expert);
        } else {
            moe_down_q2K_expert_batch_sharedmid_kernel<4><<<down_grid, down_threads, down_shmem>>>(
                    (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes, n_expert);
        }
        if (!cuda_ok(cudaGetLastError(), "routed_moe q2 expert down launch")) return 0;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        if (moe_wmma_hot && wmma_f16_low_count != 0u) {
            constexpr uint32_t mt4 = 4u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt4, 1u, 1u);
            const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                            (wmma_f16_low_max + mt4 * bm - 1u) / (mt4 * bm),
                            wmma_f16_low_count);
            const size_t shmem_n2 = (mt4 * bm * bk + 2u * bk * bn) * sizeof(half) +
                                    (2u * mt4 * bm * bn) * sizeof(float);
            if (!cuda_ok(cudaMemcpy(wmma_down_f16_low_dev, h_f16_low,
                                    wmma_f16_low_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma f16-low down hot copy")) return 0;
            moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                    NULL, wmma_down_h, down_w, NULL, wmma_mid_h,
                    counts, offsets, sorted_pairs, wmma_down_f16_low_dev, wmma_f16_low_count,
                    expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
            if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma f16-low down launch")) return 0;
        }
        if (moe_wmma_hot && wmma_f16_hot_count != 0u) {
            constexpr uint32_t mt = 8u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                            (wmma_f16_hot_max + mt * bm - 1u) / (mt * bm),
                            wmma_f16_hot_count);
            const size_t shmem_n2 = (mt * bm * bk + 2u * bk * bn) * sizeof(half) +
                                    (2u * mt * bm * bn) * sizeof(float);
            if (!cuda_ok(cudaMemcpy(wmma_down_hot_dev, h_f16_hot,
                                    wmma_f16_hot_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma f16-mid down hot copy")) return 0;
            moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                    NULL, wmma_down_h, down_w, NULL, wmma_mid_h,
                    counts, offsets, sorted_pairs, wmma_down_hot_dev, wmma_f16_hot_count,
                    expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
            if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma f16-mid down launch")) return 0;
        }
#endif
        const uint64_t n = (uint64_t)n_tokens * out_dim;
        if (moe_wmma_hot) {
            if ((out_dim & 1u) == 0u) {
                const uint64_t n2 = n >> 1u;
                moe_sum_f16x2_kernel<<<(n2 + 255u) / 256u, 256>>>(
                        (float *)out->ptr, wmma_down_h, out_dim, n_expert, n_tokens);
            } else {
                moe_sum_f16_kernel<<<(n + 255u) / 256u, 256>>>(
                        (float *)out->ptr, wmma_down_h, out_dim, n_expert, n_tokens);
            }
        } else {
            moe_sum_kernel<<<(n + 255u) / 256u, 256>>>(
                    (float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
        }
        ok = cuda_ok(cudaGetLastError(), "routed_moe q2 expert sum launch");
        if (ok && compact_selected) ok = cuda_stream_selected_mark_inflight();
        return ok;
    }

    if (q2k_path) {
        uint32_t gate_rows_per_block = cfg->moe_decode_gate_rpb;
        if (gate_rows_per_block == 0u) gate_rows_per_block = 1u;
        const uint32_t gate_threads = gate_rows_per_block * 32u;
        uint32_t down_rows_per_block = cfg->moe_decode_down_rpb;
        if (down_rows_per_block == 0u) down_rows_per_block = 1u;
        const uint32_t down_threads = down_rows_per_block * 32u;
        const int store_gate_up = (g_quality_mode || cfg->graph_dump) ? 1 : 0;
        const int q8k_gateup = !g_quality_mode && n_tokens == 1u &&
            down->bytes >= xq_bytes;
        const int decode_profile =
            n_tokens == 1u && routed_moe_decode_profile_enabled();
        ds4_rocm_moe_decode_profile_record decode_profile_rec = {0};
        if (decode_profile) {
            if (!routed_moe_decode_profile_ensure_events()) return 0;
            g_moe_decode_profile_stats.calls++;
            if (split_selected) g_moe_decode_profile_stats.split_calls++;
        }
        int ok_gateup = 1;
        if (split_selected) {
            const uint32_t compact_mask = stream_resident_mask | stream_missing_mask;
            if (compact_mask == 0u) return 0;
            dim3 gate_grid((expert_mid_dim + gate_rows_per_block - 1u) / gate_rows_per_block,
                           pair_count,
                           1);
            if (stream_resident_mask != 0u) {
                if (decode_profile &&
                    !routed_moe_decode_profile_record_event(
                            DS4_ROCM_MOE_DECODE_PROFILE_GATE_RESIDENT_START,
                            "Q2 decode MoE profile resident gate/up start")) {
                    return 0;
                }
                moe_gate_up_mid_q2K_rows_w32_ptrs_kernel<<<gate_grid, gate_threads>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        NULL,
                        0u,
                        (const float *)x->ptr,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        expert_in_dim,
                        expert_mid_dim,
                        n_expert,
                        stream_resident_mask,
                        clamp,
                        store_gate_up);
                ok_gateup = cuda_ok(cudaGetLastError(),
                                     "routed_moe q2 split resident gate/up launch");
                if (decode_profile && ok_gateup) {
                    decode_profile_rec.gate_resident = 1;
                    if (!routed_moe_decode_profile_record_event(
                            DS4_ROCM_MOE_DECODE_PROFILE_GATE_RESIDENT_END,
                            "Q2 decode MoE profile resident gate/up end")) {
                        return 0;
                    }
                }
            }
            if (!ok_gateup) {
                (void)cuda_stream_selected_finish_pending_missing(0);
                return 0;
            }
            const double finish_missing_t0 =
                decode_profile ? cuda_wall_sec() : 0.0;
            ok_gateup = cuda_stream_selected_finish_pending_missing(0);
            if (decode_profile) {
                g_moe_decode_profile_stats.finish_missing_ms +=
                    (cuda_wall_sec() - finish_missing_t0) * 1000.0;
            }
            if (ok_gateup && stream_missing_mask != 0u) {
                if (decode_profile &&
                    !routed_moe_decode_profile_record_event(
                            DS4_ROCM_MOE_DECODE_PROFILE_GATE_MISSING_START,
                            "Q2 decode MoE profile missing gate/up start")) {
                    return 0;
                }
                moe_gate_up_mid_q2K_rows_w32_ptrs_kernel<<<gate_grid, gate_threads>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        NULL,
                        0u,
                        (const float *)x->ptr,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        expert_in_dim,
                        expert_mid_dim,
                        n_expert,
                        stream_missing_mask,
                        clamp,
                        store_gate_up);
                ok_gateup = cuda_ok(cudaGetLastError(),
                                     "routed_moe q2 split missing gate/up launch");
                if (decode_profile && ok_gateup) {
                    decode_profile_rec.gate_missing = 1;
                    if (!routed_moe_decode_profile_record_event(
                            DS4_ROCM_MOE_DECODE_PROFILE_GATE_MISSING_END,
                            "Q2 decode MoE profile missing gate/up end")) {
                        return 0;
                    }
                }
            }
        } else if (q8k_gateup) {
            if (decode_profile) g_moe_decode_profile_stats.q8_gateup_calls++;
            if (decode_profile &&
                !routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_START,
                        "Q2 decode MoE profile gate/up start")) {
                return 0;
            }
            cuda_block_q8_K *xq_gate = (cuda_block_q8_K *)down->ptr;
            dim3 xq_grid(xq_blocks, n_tokens, 1);
            q8_K_quantize_kernel<<<xq_grid, 256>>>(xq_gate, (const float *)x->ptr, expert_in_dim, n_tokens);
            ok_gateup = cuda_ok(cudaGetLastError(), "routed_moe q2 oldhip q8k gate input quantize launch");
            if (ok_gateup) {
                dim3 gate_grid((expert_mid_dim + 255u) / 256u, pair_count, 1);
                moe_gate_up_mid_q2K_decode_q8_qwarp32_kernel<<<gate_grid, 256u>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq_gate,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        (uint32_t)store_gate_up,
                        clamp);
                ok_gateup = cuda_ok(cudaGetLastError(), "routed_moe q2 oldhip q8k gate/up launch");
            }
            if (decode_profile && ok_gateup) {
                decode_profile_rec.gate_full = 1;
                if (!routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_END,
                        "Q2 decode MoE profile gate/up end")) {
                    return 0;
                }
            }
        } else if (gate_rows_per_block == 1u) {
            if (decode_profile &&
                !routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_START,
                        "Q2 decode MoE profile gate/up start")) {
                return 0;
            }
            dim3 gate_grid(expert_mid_dim, pair_count, 1);
            moe_gate_up_mid_q2K_rows_rpb1_w32_kernel<<<gate_grid, 32u>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    (const float *)x->ptr,
                    (const int32_t *)selected_exec->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    expert_in_dim,
                    expert_mid_dim,
                    n_expert,
                    clamp,
                    store_gate_up);
            ok_gateup = cuda_ok(cudaGetLastError(), "routed_moe q2 oldhip rows gate/up launch");
            if (decode_profile && ok_gateup) {
                decode_profile_rec.gate_full = 1;
                if (!routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_END,
                        "Q2 decode MoE profile gate/up end")) {
                    return 0;
                }
            }
        } else {
            if (decode_profile &&
                !routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_START,
                        "Q2 decode MoE profile gate/up start")) {
                return 0;
            }
            dim3 gate_grid((expert_mid_dim + gate_rows_per_block - 1u) / gate_rows_per_block, pair_count, 1);
            moe_gate_up_mid_q2K_rows_w32_kernel<<<gate_grid, gate_threads>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    (const float *)x->ptr,
                    (const int32_t *)selected_exec->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    expert_in_dim,
                    expert_mid_dim,
                    n_expert,
                    clamp,
                    store_gate_up);
            ok_gateup = cuda_ok(cudaGetLastError(), "routed_moe q2 oldhip rows gate/up launch");
            if (decode_profile && ok_gateup) {
                decode_profile_rec.gate_full = 1;
                if (!routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_END,
                        "Q2 decode MoE profile gate/up end")) {
                    return 0;
                }
            }
        }
        if (!ok_gateup) return 0;
        int ok_decode_moe = 1;
        const int q8k_down = !g_quality_mode && n_tokens == 1u &&
            down->bytes >= midq_bytes;
        if (decode_profile && q8k_down) {
            g_moe_decode_profile_stats.q8_down_calls++;
        }
        if (q8k_down) {
            cuda_block_q8_K *midq = (cuda_block_q8_K *)down->ptr;
            dim3 midq_grid(midq_blocks, pair_count, 1);
            if (decode_profile &&
                !routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_MID_QUANT_START,
                        "Q2 decode MoE profile mid quant start")) {
                return 0;
            }
            q8_K_quantize_kernel<<<midq_grid, 256>>>(midq, (const float *)mid->ptr, expert_mid_dim, pair_count);
            ok_decode_moe = cuda_ok(cudaGetLastError(), "routed_moe q2 oldhip q8k mid quantize launch");
            if (decode_profile && ok_decode_moe) {
                decode_profile_rec.mid_quant = 1;
                if (!routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_MID_QUANT_END,
                        "Q2 decode MoE profile mid quant end")) {
                    return 0;
                }
            }
            if (ok_decode_moe) {
                if (decode_profile &&
                    !routed_moe_decode_profile_record_event(
                            DS4_ROCM_MOE_DECODE_PROFILE_DOWN_START,
                            "Q2 decode MoE profile down start")) {
                    return 0;
                }
                if (split_selected) {
                    moe_down_sum6_qwarp32_ptrs_kernel<<<(out_dim + 31u) / 32u, 256>>>(
                            (float *)out->ptr,
                            down_slot_ptrs,
                            midq,
                            down_row_bytes,
                            midq_blocks,
                            out_dim,
                            n_expert);
                } else {
                    moe_down_sum6_qwarp32_kernel<<<(out_dim + 31u) / 32u, 256>>>(
                            (float *)out->ptr,
                            down_w,
                            midq,
                            (const int32_t *)selected_exec->ptr,
                            down_expert_bytes,
                            down_row_bytes,
                            midq_blocks,
                            out_dim,
                            n_expert);
                }
                ok_decode_moe = cuda_ok(cudaGetLastError(), "routed_moe q2 oldhip q8k down launch");
                if (decode_profile && ok_decode_moe) {
                    decode_profile_rec.down = 1;
                    if (!routed_moe_decode_profile_record_event(
                            DS4_ROCM_MOE_DECODE_PROFILE_DOWN_END,
                            "Q2 decode MoE profile down end")) {
                        return 0;
                    }
                }
            }
        } else {
            dim3 down_grid((out_dim + down_rows_per_block - 1u) / down_rows_per_block, n_tokens, 1);
            if (decode_profile &&
                !routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_DOWN_START,
                        "Q2 decode MoE profile down start")) {
                return 0;
            }
            if (split_selected) {
                moe_down_q2K_sum_rows_w32_ptrs_batch_kernel<<<down_grid, down_threads>>>(
                        (float *)out->ptr,
                        down_slot_ptrs,
                        (const float *)mid->ptr,
                        (const int32_t *)selected_exec->ptr,
                        n_tokens,
                        expert_mid_dim,
                        out_dim,
                        down_row_bytes,
                        n_expert);
            } else {
                moe_down_q2K_sum_rows_w32_kernel<<<down_grid, down_threads>>>(
                        (float *)out->ptr,
                        down_w,
                        (const float *)mid->ptr,
                        (const int32_t *)selected_exec->ptr,
                        n_tokens,
                        expert_mid_dim,
                        out_dim,
                        down_expert_bytes,
                        down_row_bytes,
                        n_expert);
            }
            ok_decode_moe = cuda_ok(cudaGetLastError(), "routed_moe q2 oldhip rows down launch");
            if (decode_profile && ok_decode_moe) {
                decode_profile_rec.down = 1;
                if (!routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_DOWN_END,
                        "Q2 decode MoE profile down end")) {
                    return 0;
                }
            }
        }
        if (ok_decode_moe && compact_selected) {
            ok_decode_moe = cuda_stream_selected_mark_inflight();
        }
        if (ok_decode_moe && decode_profile) {
            ok_decode_moe =
                routed_moe_decode_profile_collect(&decode_profile_rec);
        }
        return ok_decode_moe;
    }

    if (ok) {
        dim3 mgrid(expert_mid_dim, pair_count, 1);
        if (q2k_path) {
            moe_gate_up_mid_q2K_f32_kernel<<<mgrid, 256>>>(
                (float *)gate->ptr,
                (float *)up->ptr,
                (float *)mid->ptr,
                gate_w,
                up_w,
                (const float *)x->ptr,
                (const int32_t *)selected_exec->ptr,
                (const float *)weights->ptr,
                gate_expert_bytes,
                gate_row_bytes,
                expert_in_dim,
                expert_mid_dim,
                n_expert,
                clamp);
        } else {
            moe_gate_up_mid_f32_kernel<<<mgrid, 256>>>(
                (float *)gate->ptr,
                (float *)up->ptr,
                (float *)mid->ptr,
                gate_w,
                up_w,
                (const float *)x->ptr,
                (const int32_t *)selected_exec->ptr,
                (const float *)weights->ptr,
                gate_expert_bytes,
                gate_row_bytes,
                expert_in_dim,
                expert_mid_dim,
                n_expert,
                clamp);
        }
        ok = cuda_ok(cudaGetLastError(), "routed_moe gate/up launch");
    }
    if (ok) {
        dim3 dgrid(out_dim, pair_count, 1);
        moe_down_f32_kernel<<<dgrid, 256>>>(
            (float *)down->ptr,
            down_w,
            (const float *)mid->ptr,
            (const int32_t *)selected_exec->ptr,
            down_expert_bytes,
            down_row_bytes,
            expert_mid_dim,
            out_dim,
            n_expert);
        ok = cuda_ok(cudaGetLastError(), "routed_moe down launch");
    }
    if (ok) {
        uint64_t n = (uint64_t)n_tokens * out_dim;
        moe_sum_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "routed_moe sum launch");
    }
    if (ok && compact_selected) ok = cuda_stream_selected_mark_inflight();
    return ok;
}

/* DS4-TP-gfx1151 (patch 9): defined in the TP runtime, which ds4_rocm.cu
 * includes AFTER this header. */
extern "C" int ds4_gpu_tp_expert_shard_remap(
        const int32_t *selected, const float *weights, void *scratch,
        uint32_t n_pairs, uint32_t n_total_expert,
        const int32_t **out_selected, const float **out_weights,
        uint32_t *out_base, uint32_t *out_count);

/* Shared by both routed-MoE entry points. On success the caller must use the
 * returned views AND the shifted offsets/count: the launcher derives its span
 * as n_total_expert*expert_bytes, so shrinking the count is what stops it
 * requesting the whole layer and paging in the half this rank does not map. */
struct ds4_tp_shard_view {
    ds4_gpu_tensor sel;
    ds4_gpu_tensor w;
    uint64_t gate_off_delta;
    uint64_t down_off_delta;
    uint32_t n_total_expert;
    int      active;
};

static int ds4_tp_shard_prepare(const ds4_gpu_tensor *selected,
                                const ds4_gpu_tensor *weights,
                                uint32_t n_pairs,
                                uint32_t n_total_expert,
                                uint64_t gate_expert_bytes,
                                uint64_t down_expert_bytes,
                                ds4_tp_shard_view *v) {
    v->active = 0;
    if (!ds4_gpu_tp_expert_shard_active() || !selected || !weights || n_pairs == 0) return 1;
    /* DS4-TP-gfx1151 (patch 13): DEDICATED grow-only buffer, NOT cuda_tmp_alloc.
     *
     * cuda_tmp_alloc (ds4_rocm_runtime.cuh:567) is a ONE-SLOT global allocator:
     * it returns g_cuda_tmp when big enough, else cudaFree()s it and cudaMallocs
     * a new one. routed_moe_launch calls it AGAIN at :919 and :1771 while these
     * pointers are still live - the Q2_K/WMMA request at :1771 is far larger, so
     * the remap buffer is freed out from under use_selected->ptr; on the :919
     * path the sizes can alias exactly and the cudaMemset of `counts` then zeroes
     * the selection before the count kernel reads it. Either way counts[] ends up
     * garbage, and ds4_rocm_moe.cuh:3519 does `for (p0 = 0; p0 < count; ...)` -
     * an unbounded loop, i.e. 100% GPU forever with no fault.
     *
     * Splitting one cuda_tmp_alloc in two was necessary but NOT sufficient: the
     * callee re-enters the same allocator. Found by review, not by the hang. */
    static void    *g_tp_remap_buf = NULL;
    static uint64_t g_tp_remap_bytes = 0;
    const uint64_t need = (uint64_t)n_pairs * (sizeof(int32_t) + sizeof(float));
    if (need > g_tp_remap_bytes) {
        if (g_tp_remap_buf) (void)cudaFree(g_tp_remap_buf);
        g_tp_remap_buf = NULL; g_tp_remap_bytes = 0;
        if (cudaMalloc(&g_tp_remap_buf, (size_t)need) != cudaSuccess || !g_tp_remap_buf) {
            (void)cudaGetLastError();
            fprintf(stderr, DS4_GPU_LOG_PREFIX "tp expert shard remap alloc failed (%.2f MiB)\n",
                    (double)need / 1048576.0);
            return 0;
        }
        g_tp_remap_bytes = need;
    }
    void *scratch = g_tp_remap_buf;
    const int32_t *sel_p = NULL;
    const float *w_p = NULL;
    uint32_t base = 0, count = 0;
    if (!ds4_gpu_tp_expert_shard_remap((const int32_t *)selected->ptr,
                                       (const float *)weights->ptr,
                                       scratch, n_pairs, n_total_expert,
                                       &sel_p, &w_p, &base, &count)) {
        return 1; /* TP inactive or shard suspended: use inputs unchanged */
    }
    v->sel = *selected; v->sel.ptr = (void *)sel_p; v->sel.owner = 0;
    v->w   = *weights;  v->w.ptr   = (void *)w_p;   v->w.owner   = 0;
    v->gate_off_delta = (uint64_t)base * gate_expert_bytes;
    v->down_off_delta = (uint64_t)base * down_expert_bytes;
    v->n_total_expert = count;
    v->active = 1;
    return 1;
}
extern "C" int ds4_gpu_routed_moe_one_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, const ds4_gpu_tensor *add_in, uint32_t layer_index, bool force_resident) {
    /* DS4-TP-gfx1151 (patch 16): the addend must be folded AFTER the launch.
     *
     * Patch 15 pre-added it, on the stated premise that "routed_moe_launch
     * ACCUMULATES into out". That premise is FALSE on ROCm: every terminal
     * write is an assignment - ds4_rocm_moe.cuh:2091 and :2257
     * (`out[row] = total;`, and :2257 is the Q4_K path decode takes), and the
     * atomic path zeroes out first (:1470 here). There is no `out[...] +=`
     * anywhere in either file. So the pre-added addend was OVERWRITTEN and the
     * shared expert was silently dropped from every decode layer.
     *
     * Upstream refused this case outright ("routed MoE addend fold is
     * Metal-only") because Metal folds the addend INSIDE the down kernel
     * (ds4_metal.m:34779 -> metal/moe.metal:6091-6094). Patch 15 turned a
     * fail-closed refusal into a silent wrong answer, which is strictly worse.
     *
     * Post-add is mathematically equivalent to Metal's in-kernel fold here
     * because out is written, not accumulated: out = sum(experts), then
     * out += add_in. A future optimisation can thread a tp_addend pointer into
     * the sum6/moe_sum kernels to save the extra pass. */
    /* DS4-TP-gfx1151 (patch 12): rebase the selection onto this rank's expert
     * shard and shift the weight offsets to match, so the launcher addresses
     * ONLY resident memory. Replaces patch 9's weight masking, which left the
     * launcher requesting the whole layer and paged the unowned half in from
     * disk every layer until the device filled. */
    ds4_tp_shard_view v;
    const ds4_gpu_tensor *use_selected = selected;
    const ds4_gpu_tensor *use_weights = weights;
    if (!ds4_tp_shard_prepare(selected, weights, n_expert, n_total_expert,
                              gate_expert_bytes, down_expert_bytes, &v)) {
        return 0;
    }
    if (v.active) {
        use_selected = &v.sel;
        use_weights = &v.w;
        gate_offset += v.gate_off_delta;
        up_offset += v.gate_off_delta;
        down_offset += v.down_off_delta;
        n_total_expert = v.n_total_expert;
    }
    int rc = routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             use_selected, use_weights, n_total_expert, n_expert, clamp, x, layer_index, 1,
                             force_resident);
    /* out_dim, not out->bytes/sizeof(float): the latter is the ALLOCATION size.
     * For the tp_out slab view they coincide, but it is a latent overrun for any
     * caller with a larger buffer. */
    if (rc && add_in) {
        if (!ds4_gpu_add_tensor(out, out, add_in, (uint32_t)out_dim)) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "routed MoE addend fold failed\n");
            return 0;
        }
    }
    if (!rc) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "routed_moe_ONE failed: layer=%u shard=%d "
                "n_total=%u n_exp=%u gate_type=%u down_type=%u in=%u mid=%u out=%llu "
                "gate_off=%llu gate_eb=%llu\n",
                layer_index, v.active, n_total_expert, n_expert, gate_type, down_type,
                expert_in_dim, expert_mid_dim, (unsigned long long)out_dim,
                (unsigned long long)gate_offset, (unsigned long long)gate_expert_bytes);
    }
    return rc;
}
extern "C" int ds4_gpu_routed_moe_batch_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens, bool *mid_is_f16, bool force_resident) {
    if (mid_is_f16) *mid_is_f16 = false;
    /* DS4-TP-gfx1151 (patch 12): the BATCH path is the prefill path and had no
     * expert sharding at all - patch 9 only touched the one-token entry point.
     * Without this, prefill both double-counted experts across ranks AND paged
     * in the unowned half. n_pairs is n_tokens*n_expert here, not n_expert. */
    ds4_tp_shard_view v;
    const ds4_gpu_tensor *use_selected = selected;
    const ds4_gpu_tensor *use_weights = weights;
    uint64_t pairs64 = 0;
    if (!cuda_u64_mul_checked(n_tokens, n_expert, &pairs64) || pairs64 > UINT32_MAX) {
        return 0;
    }
    if (!ds4_tp_shard_prepare(selected, weights, (uint32_t)pairs64, n_total_expert,
                              gate_expert_bytes, down_expert_bytes, &v)) {
        return 0;
    }
    if (v.active) {
        use_selected = &v.sel;
        use_weights = &v.w;
        gate_offset += v.gate_off_delta;
        up_offset += v.gate_off_delta;
        down_offset += v.down_off_delta;
        n_total_expert = v.n_total_expert;
    }
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             use_selected, use_weights, n_total_expert, n_expert, clamp, x, layer_index, n_tokens,
                             force_resident);
}
