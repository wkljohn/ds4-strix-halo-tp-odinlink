static const void *g_model_host_base;
static const char *g_model_device_base;
static uint64_t g_model_registered_size;
static const void *g_support_host_base;
static uint64_t g_support_host_size;
static int g_support_fd = -1;
static int g_support_direct_fd = -1;
static uint64_t g_support_file_size;
static uint64_t g_support_direct_align = 1;
static uint64_t g_dspark_stage_offsets[3];
static char *g_dspark_selected_gate;
static char *g_dspark_selected_up;
static char *g_dspark_selected_down;
static uint64_t g_dspark_selected_gate_capacity;
static uint64_t g_dspark_selected_down_capacity;
static int32_t *g_dspark_selected_ids;
static uint64_t g_dspark_selected_ids_capacity;
static int g_model_device_owned;
static int g_model_range_mapping_supported = 1;
static int g_model_fd = -1;
static const void *g_model_fd_host_base;
static int g_model_direct_fd = -1;
static uint64_t g_model_direct_align = 1;
static uint64_t g_model_file_size;
static int g_model_cache_full;
static int g_ssd_streaming_mode;
static cudaStream_t g_model_upload_stream;
static cudaStream_t g_stream_selected_upload_stream;
static cudaStream_t g_selected_readback_stream;
static cudaEvent_t g_selected_readback_event;
static uint64_t g_selected_readback_event_value;
static cudaEvent_t g_token_span_start;
static cudaEvent_t g_token_span_head;
static cudaEvent_t g_token_span_head_stage[DS4_GPU_TOKEN_HEAD_STAGE_COUNT];
static cudaEvent_t g_token_span_stop;
static int g_token_span_active;
static int g_token_span_head_valid;
static uint32_t g_token_span_head_stage_mask;
static cudaEvent_t g_token_attn_span_start;
static cudaEvent_t g_token_attn_span_stop;
static int g_token_attn_span_active;
static int g_token_attn_span_valid;

static int token_span_head_stage_profile_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        enabled = getenv("DS4_TP_TOKEN_HEAD_STAGE_PROFILE") != NULL;
    }
    return enabled;
}
static cublasHandle_t g_cublas;
static int g_cublas_ready;
#ifdef __HIP_PLATFORM_AMD__
#include "ds4_rocm_hipblaslt.cuh"
#endif
static int g_quality_mode;

enum {
    DS4_ROCM_N_EXPERT = 256u,
    DS4_ROCM_MAX_N_EXPERT = 384u,
    DS4_ROCM_N_EXPERT_USED = 8u,
    DS4_ROCM_STREAM_READ_WORKERS = DS4_ROCM_N_EXPERT_USED * 3u,
    DS4_ROCM_STREAM_READ_DEFAULT_WORKERS = 16u,
    DS4_ROCM_STREAM_READ_MAX_JOBS = DS4_ROCM_MAX_N_EXPERT * 3u,
    DS4_ROCM_STREAM_CACHE_LAYER_STATS_MAX = 128u,
    DS4_ROCM_COMPRESSOR_MAX_RATIO = 128u
};
#define DS4_ROCM_EXPERT_WEIGHT_SCALE 1.5f
#define DS4_ROCM_EXPERT_WEIGHT_SCALE_TOL 1.0e-6f

struct cuda_model_range {
    const void *host_base;
    uint64_t offset;
    uint64_t bytes;
    char *device_ptr;
    void *registered_base;
    char *registered_device_base;
    uint64_t registered_bytes;
    int host_registered;
    int arena_allocated;
};

struct cuda_model_arena {
    char *device_ptr;
    uint64_t bytes;
    uint64_t used;
};

struct cuda_model_image {
    const void *host_base;
    uint64_t size;
    char *device_ptr;
    uint64_t device_offset;
    int owns_device_ptr = 1;
};

struct cuda_q4k_packed_slice {
    const void *host_base;
    uint64_t model_size;
    uint64_t tensor_offset;
    uint64_t source_tensor_bytes;
    uint64_t source_expert_bytes;
    uint64_t source_row_bytes;
    uint64_t packed_expert_bytes;
    uint64_t packed_bytes;
    uint64_t column_byte_base;
    uint64_t column_byte_count;
    uint32_t n_expert;
    uint32_t source_rows;
    uint32_t row_base;
    uint32_t row_count;
    ds4_gpu_q4k_packed_slice_kind kind;
    char *device_ptr;
    int owns_device_ptr;
    int loaded;
    int blocked_logged;
};

struct cuda_q4k_window_cache_entry {
    int32_t expert;
    uint64_t last_used;
    int valid;
};

struct ds4_gpu_q4k_window_cache {
    const void *model_map;
    uint64_t model_size;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    size_t gate_slice_index;
    size_t up_slice_index;
    size_t down_slice_index;
    char *base;
    char *gate;
    char *up;
    char *down;
    int32_t *slot_ids_device;
    uint64_t slot_ids_capacity;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    uint64_t capacity_bytes;
    uint64_t clock;
    uint64_t prepares;
    uint64_t hits;
    uint64_t misses;
    uint64_t fills;
    uint64_t evictions;
    uint64_t profile_upload_bytes;
    uint64_t profile_control_bytes;
    double profile_prepare_sec;
    uint32_t n_expert;
    uint32_t slots;
    cudaStream_t copy_stream;
    cudaEvent_t fill_event;
    int overlap_enabled;
    const ds4_gpu_tensor *prepared_ids;
    const ds4_gpu_tensor *prepared_weights;
    uint32_t prepared_count;
    int prepared_valid;
    std::vector<int32_t> expert_to_slot;
    std::vector<cuda_q4k_window_cache_entry> entries;
};

struct cuda_q8_f16_range {
    const void *host_base;
    uint64_t offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    __half *device_ptr;
};

struct cuda_q8_f16_transpose_range {
    const void *host_base;
    uint64_t offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    __half *device_ptr;
};

struct cuda_stream_selected_cache {
    int loaded;
    const void *model_map;
    uint32_t layer;
    uint32_t n_total_expert;
    uint32_t n_selected;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    int32_t selected_ids[DS4_ROCM_N_EXPERT_USED];
    char *gate;
    char *up;
    char *down;
    uint64_t gate_capacity;
    uint64_t down_capacity;
    int32_t *slot_ids;
    const char **gate_ptrs;
    const char **up_ptrs;
    const char **down_ptrs;
    const char **gate_ptrs_stage;
    const char **up_ptrs_stage;
    const char **down_ptrs_stage;
    ds4_gpu_tensor slot_tensor;
};

struct cuda_stream_resident_expert {
    const void *model_map;
    uint32_t layer;
    int32_t expert;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    char *base;
    char *gate;
    char *up;
    char *down;
    uint64_t bytes;
    uint64_t last_used;
    int pooled;
};

/*
 * Streamed experts share one size class per model (2*gate + down bytes), so
 * the resident cache carves fixed slots out of large slabs instead of doing a
 * cudaMalloc/cudaFree round trip per cache miss.  Allocator traffic was a
 * large fraction of decode time: hundreds of 11.8 MiB device allocations and
 * frees per generated token, each costing GTT page-table work.
 */
struct cuda_stream_expert_slab {
    char *base;
    uint64_t bytes;
};

struct cuda_stream_resident_key {
    const void *model_map;
    uint32_t layer;
    int32_t expert;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;

    bool operator==(const cuda_stream_resident_key &o) const {
        return model_map == o.model_map &&
               layer == o.layer &&
               expert == o.expert &&
               gate_offset == o.gate_offset &&
               up_offset == o.up_offset &&
               down_offset == o.down_offset &&
               gate_expert_bytes == o.gate_expert_bytes &&
               down_expert_bytes == o.down_expert_bytes;
    }
};

struct cuda_stream_resident_key_hash {
    size_t operator()(const cuda_stream_resident_key &k) const {
        uint64_t h = (uint64_t)(uintptr_t)k.model_map;
        h ^= (uint64_t)k.layer + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= (uint64_t)(uint32_t)k.expert + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= k.gate_offset + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= k.up_offset + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= k.down_offset + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= k.gate_expert_bytes + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= k.down_expert_bytes + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return (size_t)h;
    }
};

struct cuda_stream_batch_selected_cache {
    int loaded;
    const void *model_map;
    uint32_t layer;
    uint32_t n_total_expert;
    uint32_t n_selected;
    uint32_t n_tokens;
    uint32_t n_unique;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    int32_t *selected_ids;
    uint64_t selected_capacity;
    uint8_t *pair_missing;
    uint64_t pair_missing_capacity;
    const char **gate_ptrs;
    const char **up_ptrs;
    const char **down_ptrs;
    const char **resident_gate_ptrs;
    const char **resident_up_ptrs;
    const char **missing_gate_ptrs;
    const char **missing_up_ptrs;
    uint32_t ptr_capacity;
    int32_t *selected_stage;
    uint64_t selected_stage_capacity;
    uint8_t *pair_missing_stage;
    uint64_t pair_missing_stage_capacity;
    const char **gate_ptrs_stage;
    const char **up_ptrs_stage;
    const char **down_ptrs_stage;
    const char **resident_gate_ptrs_stage;
    const char **resident_up_ptrs_stage;
    const char **missing_gate_ptrs_stage;
    const char **missing_up_ptrs_stage;
    uint32_t ptr_stage_capacity;
    ds4_gpu_tensor selected_tensor;
};

struct cuda_stream_layer_expert_cache {
    int active;
    const void *model_map;
    uint32_t layer;
    uint32_t n_total_expert;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    uint64_t bytes;
    uint64_t capacity;
    char *base;
    char *gate;
    char *up;
    char *down;
};

struct cuda_stream_cache_stats {
    uint64_t selected_calls;
    uint64_t selected_slots;
    uint64_t selected_hits;
    uint64_t selected_misses;
    uint64_t batch_calls;
    uint64_t batch_unique;
    uint64_t batch_hits;
    uint64_t batch_misses;
    uint64_t seed_calls;
    uint64_t seed_unique;
    uint64_t layer_loads;
    uint64_t layer_load_bytes;
    uint64_t layer_resident_flushes;
    uint64_t allocs;
    uint64_t alloc_bytes;
    uint64_t evictions;
    uint64_t evict_bytes;
    uint64_t max_resident_count;
    uint64_t max_resident_bytes;
};

struct cuda_stream_cache_layer_stats {
    uint64_t selected_calls;
    uint64_t selected_slots;
    uint64_t selected_hits;
    uint64_t selected_misses;
    uint64_t batch_calls;
    uint64_t batch_unique;
    uint64_t batch_hits;
    uint64_t batch_misses;
};

static std::vector<cuda_model_range> g_model_ranges;
static std::vector<cuda_model_arena> g_model_arenas;
static std::vector<cuda_model_image> g_model_images;
static std::vector<cuda_model_image> g_q4k_kshard_pre_images;
static std::vector<cuda_model_image> g_q4k_kshard_borrowed_images;
static std::vector<cuda_q4k_packed_slice> g_q4k_packed_slices;
static std::unordered_multimap<uint64_t, size_t> g_q4k_packed_by_offset;
static std::vector<ds4_gpu_q4k_window_cache *> g_q4k_window_caches;
struct cuda_q4k_kshard_blocked_range {
    const void *host_base;
    uint64_t offset;
    uint64_t bytes;
    int logged;
};
static std::vector<cuda_q4k_kshard_blocked_range>
    g_q4k_kshard_blocked_ranges;
static std::unordered_map<uint64_t, size_t> g_model_range_by_offset;
static std::vector<cuda_q8_f16_range> g_q8_f16_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f16_by_offset;
static std::vector<cuda_q8_f16_transpose_range> g_q8_f16_transpose_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f16_transpose_by_offset;
static uint64_t g_model_range_bytes;
static uint64_t g_q4k_packed_slice_bytes;
static struct {
    int installed;
    int owns_shard_suspend;
    int snapshot_valid;
    int evacuated;
    const void *model_map;
    uint64_t model_size;
    uint64_t key_hash;
    ds4_gpu_q4k_kshard_windows windows;
    size_t pre_model_image_count;
    const void *pre_model_host_base;
    const char *pre_model_device_base;
    uint64_t pre_model_registered_size;
    int pre_model_device_owned;
    int pre_model_range_mapping_supported;
    int pre_model_cache_full;
    int pre_model_fd;
    const void *pre_model_fd_host_base;
    int pre_model_direct_fd;
    uint64_t pre_model_direct_align;
    uint64_t pre_model_file_size;
} g_q4k_kshard;

static int cuda_q4k_kshard_restore_borrowed(void);
static int g_q4k_kshard_cleanup_in_progress;

static void cuda_q4k_kshard_state_clear(void) {
    const int was_evacuated = g_q4k_kshard.evacuated;
    if (g_q4k_kshard.owns_shard_suspend) {
        ds4_gpu_tp_suspend_expert_sharding(0);
    }
    if (g_q4k_kshard.snapshot_valid) {
        if (g_q4k_kshard.evacuated) {
            for (const cuda_model_image &img : g_model_images) {
                if (img.device_ptr && img.owns_device_ptr) {
                    (void)cudaFree(img.device_ptr);
                }
            }
            g_model_images.clear();
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "Q4_K K-shard evacuated residency released; this "
                    "engine requires cleanup rather than in-place rollback\n");
        }
        const int had_borrowed = !g_q4k_kshard_borrowed_images.empty();
        const int restored = g_q4k_kshard.evacuated ? 0 :
            (g_q4k_kshard_cleanup_in_progress ||
             cuda_q4k_kshard_restore_borrowed());
        for (const cuda_model_image &img : g_model_images) {
            int existed = 0;
            for (const cuda_model_image &old : g_q4k_kshard_pre_images) {
                if (img.device_ptr == old.device_ptr) {
                    existed = 1;
                    break;
                }
            }
            if (!existed && img.device_ptr && img.owns_device_ptr) {
                (void)cudaFree(img.device_ptr);
            }
        }
        if (g_q4k_kshard.evacuated) {
            /* The snapshot's pointers named allocations freed before the
             * packed reload.  Never re-publish those stale addresses. */
        } else if (restored) {
            g_model_images = g_q4k_kshard_pre_images;
            if (had_borrowed && !g_q4k_kshard_cleanup_in_progress) {
                g_q4k_kshard_blocked_ranges.clear();
            }
        } else {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "Q4_K K-shard rollback reload failed; routed weights "
                    "remain fail-closed until model cleanup\n");
            for (const cuda_model_image &img :
                 g_q4k_kshard_borrowed_images) {
                if (img.device_ptr) (void)cudaFree(img.device_ptr);
            }
        }
        if (g_model_direct_fd >= 0) (void)close(g_model_direct_fd);
        g_model_host_base = g_q4k_kshard.pre_model_host_base;
        g_model_device_base = g_q4k_kshard.pre_model_device_base;
        g_model_registered_size =
            g_q4k_kshard.pre_model_registered_size;
        g_model_device_owned = g_q4k_kshard.pre_model_device_owned;
        g_model_range_mapping_supported =
            g_q4k_kshard.pre_model_range_mapping_supported;
        g_model_cache_full = g_q4k_kshard.pre_model_cache_full;
        g_model_fd = g_q4k_kshard.pre_model_fd;
        g_model_fd_host_base =
            g_q4k_kshard.pre_model_fd_host_base;
        g_model_direct_fd = g_q4k_kshard.pre_model_direct_fd;
        g_q4k_kshard.pre_model_direct_fd = -1;
        g_model_direct_align =
            g_q4k_kshard.pre_model_direct_align;
        g_model_file_size = g_q4k_kshard.pre_model_file_size;
        if (was_evacuated) {
            g_model_device_base = NULL;
            g_model_device_owned = 0;
            g_model_cache_full = 0;
        }
    }
    g_q4k_kshard_pre_images.clear();
    g_q4k_kshard_borrowed_images.clear();
    memset(&g_q4k_kshard, 0, sizeof(g_q4k_kshard));
    g_q4k_kshard.pre_model_direct_fd = -1;
}
static uint64_t g_q8_f16_bytes;
static int g_q8_f16_disabled_after_oom;
static int g_q8_f16_disabled_for_multi_model;
static int g_q8_f16_budget_notice_printed;
static uint64_t g_model_load_progress_next;
static double g_model_load_progress_last;
static int g_model_load_progress_started;
static int g_model_load_progress_tty;
static void *g_cuda_tmp;
static uint64_t g_cuda_tmp_bytes;
static void *g_attention_seq_scratch[DS4_MAX_GPUS];
static uint64_t g_attention_seq_scratch_bytes[DS4_MAX_GPUS];
static void *g_glm_causal_scratch[DS4_MAX_GPUS];
static uint64_t g_glm_causal_scratch_bytes[DS4_MAX_GPUS];
static void *g_model_stage_raw[4];
static void *g_model_stage[4];
static cudaEvent_t g_model_stage_event[4];
static uint64_t g_model_stage_bytes;
static uint32_t g_stream_expert_cache_budget;
static cuda_stream_selected_cache g_stream_selected_cache;
static cuda_stream_batch_selected_cache g_stream_batch_selected_cache;
static cuda_stream_layer_expert_cache g_stream_layer_expert_cache[2];
static std::vector<cuda_stream_resident_expert> g_stream_resident_experts;
static std::unordered_map<cuda_stream_resident_key,
                          size_t,
                          cuda_stream_resident_key_hash> g_stream_resident_index;
static uint64_t g_stream_resident_bytes;
static uint64_t g_stream_resident_clock;
static std::vector<cuda_stream_expert_slab> g_stream_expert_slabs;
static std::vector<char *> g_stream_expert_free_slots;
static uint64_t g_stream_expert_slot_bytes;
static uint32_t g_stream_expert_slot_count;
static cuda_stream_cache_stats g_stream_cache_stats;
static cuda_stream_cache_layer_stats
    g_stream_cache_layer_stats[DS4_ROCM_STREAM_CACHE_LAYER_STATS_MAX];
static int g_stream_cache_stats_enabled = -1;
static int g_stream_cache_layer_stats_enabled = -1;
static int g_stream_evict_past_layers_first_enabled = -1;
static int32_t g_routed_moe_selected_override[DS4_ROCM_N_EXPERT_USED];
static uint32_t g_routed_moe_selected_override_n;
static cudaEvent_t g_stream_selected_reuse_event;
static int g_stream_selected_reuse_event_pending;
static cudaEvent_t g_stream_selected_upload_ready_event;
static int g_stream_selected_upload_event_pending;
static cudaEvent_t g_stream_batch_selected_reuse_event;
static int g_stream_batch_selected_reuse_event_pending;
static cudaEvent_t g_stream_batch_selected_upload_ready_event;
static int g_stream_batch_selected_upload_event_pending;
static void *g_stream_read_stage_raw[DS4_ROCM_STREAM_READ_WORKERS];
static uint64_t g_stream_read_stage_bytes[DS4_ROCM_STREAM_READ_WORKERS];
static cudaStream_t g_stream_read_upload_streams[DS4_ROCM_STREAM_READ_WORKERS];
static pthread_t g_stream_read_threads[DS4_ROCM_STREAM_READ_WORKERS];
static uint32_t g_stream_read_thread_ids[DS4_ROCM_STREAM_READ_WORKERS];
static pthread_mutex_t g_stream_read_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_stream_read_work_cond = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_stream_read_done_cond = PTHREAD_COND_INITIALIZER;
static int g_stream_read_pool_started;
static uint32_t g_stream_read_pool_workers;
static int g_stream_read_pool_stop;
static struct cuda_stream_read_job *g_stream_read_active_jobs;
static uint32_t g_stream_read_active_count;
static uint32_t g_stream_read_active_next;
static uint32_t g_stream_read_active_done;
static int g_stream_read_active_ok;
static pthread_t g_stream_read_active_owner;
static int g_stream_read_active_owner_set;
static int g_stream_read_profile_enabled = -1;
static int g_stream_read_direct_disabled = -1;
static int g_stream_read_profile_registered;
static uint64_t g_stream_read_profile_jobs;
static uint64_t g_stream_read_profile_bytes;
static uint64_t g_stream_read_profile_read_us;
static uint64_t g_stream_read_profile_upload_us;
static uint64_t g_stream_read_profile_wait_calls;
static uint64_t g_stream_read_profile_wait_us;
static uint64_t g_stream_read_profile_start_calls;
static uint64_t g_stream_read_profile_exact_groups;
static uint64_t g_stream_read_profile_gap1m_groups;
static uint64_t g_stream_read_profile_gap4m_groups;
static uint64_t g_stream_read_profile_gap1m_extra;
static uint64_t g_stream_read_profile_gap4m_extra;

static int cuda_ok(cudaError_t err, const char *what);
static double cuda_wall_sec(void);
static uint64_t cuda_model_copy_chunk_bytes(void);
static void cuda_model_drop_file_pages(uint64_t offset, uint64_t bytes);
static uint64_t cuda_round_down(uint64_t v, uint64_t align);
static uint64_t cuda_round_up(uint64_t v, uint64_t align);
static int cuda_model_stage_pool_alloc(uint64_t bytes);
static int cuda_pread_full(int fd, void *buf, uint64_t bytes, uint64_t offset);
static int cuda_model_stage_read(void *stage, uint64_t stage_bytes,
                                 uint64_t offset, uint64_t bytes,
                                 const char **payload);
static int cuda_stream_selected_reuse_wait(const char *what);
static int cuda_stream_selected_upload_wait_host(const char *what);
static void cuda_stream_selected_stage_release(void);
static int cuda_stream_batch_selected_reuse_wait(const char *what);
static int cuda_stream_batch_selected_upload_wait_host(const char *what);
static void cuda_stream_batch_selected_stage_release(void);
static void cuda_stream_read_pool_shutdown(void);

static int cuda_u64_mul_checked(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out) return 0;
    if (a != 0u && b > UINT64_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int cuda_u64_add_checked(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || a > UINT64_MAX - b) return 0;
    *out = a + b;
    return 1;
}

static int cuda_u64_mul3_checked(uint64_t a, uint64_t b, uint64_t c, uint64_t *out) {
    uint64_t tmp = 0;
    return cuda_u64_mul_checked(a, b, &tmp) && cuda_u64_mul_checked(tmp, c, out);
}

static int cuda_stream_cache_layer_stats_on(void) {
    if (g_stream_cache_layer_stats_enabled < 0) {
        g_stream_cache_layer_stats_enabled =
            getenv("DS4_ROCM_STREAM_CACHE_LAYER_STATS") != NULL ? 1 : 0;
    }
    return g_stream_cache_layer_stats_enabled;
}

static int cuda_stream_cache_stats_on(void) {
    if (g_stream_cache_stats_enabled < 0) {
        g_stream_cache_stats_enabled =
            (getenv("DS4_ROCM_STREAM_CACHE_STATS") != NULL ||
             cuda_stream_cache_layer_stats_on()) ? 1 : 0;
    }
    return g_stream_cache_stats_enabled;
}

static int cuda_stream_evict_past_layers_first(void) {
    if (g_stream_evict_past_layers_first_enabled < 0) {
        const char *env = getenv("DS4_ROCM_STREAM_EVICT_PAST_LAYERS_FIRST");
        g_stream_evict_past_layers_first_enabled =
            (env != NULL && env[0] != '\0' && strcmp(env, "0") != 0) ? 1 : 0;
    }
    return g_stream_evict_past_layers_first_enabled;
}

static void cuda_stream_cache_stats_note_resident(void) {
    if (!cuda_stream_cache_stats_on()) return;
    const uint64_t count = (uint64_t)g_stream_resident_experts.size();
    if (count > g_stream_cache_stats.max_resident_count) {
        g_stream_cache_stats.max_resident_count = count;
    }
    if (g_stream_resident_bytes > g_stream_cache_stats.max_resident_bytes) {
        g_stream_cache_stats.max_resident_bytes = g_stream_resident_bytes;
    }
}

static void cuda_stream_cache_stats_print(const char *label) {
    if (!cuda_stream_cache_stats_on()) return;
    fprintf(stderr,
            DS4_GPU_LOG_PREFIX "stream cache stats %s: "
            "selected calls=%llu slots=%llu hits=%llu misses=%llu; "
            "batch calls=%llu unique=%llu hits=%llu misses=%llu; "
            "seed calls=%llu unique=%llu; "
            "full-layer loads=%llu bytes=%.2f GiB resident-flushes=%llu; "
            "allocs=%llu alloc=%.2f GiB evictions=%llu evicted=%.2f GiB; "
            "resident current=%zu/%.2f GiB max=%llu/%.2f GiB budget=%u\n",
            label ? label : "",
            (unsigned long long)g_stream_cache_stats.selected_calls,
            (unsigned long long)g_stream_cache_stats.selected_slots,
            (unsigned long long)g_stream_cache_stats.selected_hits,
            (unsigned long long)g_stream_cache_stats.selected_misses,
            (unsigned long long)g_stream_cache_stats.batch_calls,
            (unsigned long long)g_stream_cache_stats.batch_unique,
            (unsigned long long)g_stream_cache_stats.batch_hits,
            (unsigned long long)g_stream_cache_stats.batch_misses,
            (unsigned long long)g_stream_cache_stats.seed_calls,
            (unsigned long long)g_stream_cache_stats.seed_unique,
            (unsigned long long)g_stream_cache_stats.layer_loads,
            (double)g_stream_cache_stats.layer_load_bytes / 1073741824.0,
            (unsigned long long)g_stream_cache_stats.layer_resident_flushes,
            (unsigned long long)g_stream_cache_stats.allocs,
            (double)g_stream_cache_stats.alloc_bytes / 1073741824.0,
            (unsigned long long)g_stream_cache_stats.evictions,
            (double)g_stream_cache_stats.evict_bytes / 1073741824.0,
            g_stream_resident_experts.size(),
            (double)g_stream_resident_bytes / 1073741824.0,
            (unsigned long long)g_stream_cache_stats.max_resident_count,
            (double)g_stream_cache_stats.max_resident_bytes / 1073741824.0,
            g_stream_expert_cache_budget);
    if (!cuda_stream_cache_layer_stats_on()) return;
    for (uint32_t layer = 0;
         layer < DS4_ROCM_STREAM_CACHE_LAYER_STATS_MAX;
         layer++) {
        const cuda_stream_cache_layer_stats *s =
            &g_stream_cache_layer_stats[layer];
        if (s->selected_calls == 0 && s->batch_calls == 0) continue;
        const double selected_hit_pct =
            s->selected_slots ?
                100.0 * (double)s->selected_hits /
                    (double)s->selected_slots : 0.0;
        const double batch_hit_pct =
            s->batch_unique ?
                100.0 * (double)s->batch_hits /
                    (double)s->batch_unique : 0.0;
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "stream cache layer %u %s: "
                "selected calls=%llu slots=%llu hits=%llu misses=%llu hit=%.1f%%; "
                "batch calls=%llu unique=%llu hits=%llu misses=%llu hit=%.1f%%\n",
                layer,
                label ? label : "",
                (unsigned long long)s->selected_calls,
                (unsigned long long)s->selected_slots,
                (unsigned long long)s->selected_hits,
                (unsigned long long)s->selected_misses,
                selected_hit_pct,
                (unsigned long long)s->batch_calls,
                (unsigned long long)s->batch_unique,
                (unsigned long long)s->batch_hits,
                (unsigned long long)s->batch_misses,
                batch_hit_pct);
    }
}

static int cuda_model_range_fits(uint64_t model_size, uint64_t offset, uint64_t bytes) {
    return offset <= model_size && bytes <= model_size - offset;
}

static int cuda_tensor_has_bytes(const ds4_gpu_tensor *t, uint64_t bytes) {
    return t && t->ptr && t->bytes >= bytes;
}

static int cuda_tensor_has_elems(const ds4_gpu_tensor *t, uint64_t elems, uint64_t elem_size) {
    uint64_t bytes = 0;
    return cuda_u64_mul_checked(elems, elem_size, &bytes) && cuda_tensor_has_bytes(t, bytes);
}

static int cuda_tensor_has_elems2(const ds4_gpu_tensor *t, uint64_t a, uint64_t b, uint64_t elem_size) {
    uint64_t bytes = 0;
    return cuda_u64_mul3_checked(a, b, elem_size, &bytes) && cuda_tensor_has_bytes(t, bytes);
}

static int cuda_tensor_has_elems3(const ds4_gpu_tensor *t, uint64_t a, uint64_t b, uint64_t c, uint64_t elem_size) {
    uint64_t ab = 0, elems = 0, bytes = 0;
    return cuda_u64_mul_checked(a, b, &ab) &&
           cuda_u64_mul_checked(ab, c, &elems) &&
           cuda_u64_mul_checked(elems, elem_size, &bytes) &&
           cuda_tensor_has_bytes(t, bytes);
}

static int cuda_tensor_has_f32(const ds4_gpu_tensor *t, uint64_t elems) {
    return cuda_tensor_has_elems(t, elems, sizeof(float));
}

static int cuda_tensor_has_i32(const ds4_gpu_tensor *t, uint64_t elems) {
    return cuda_tensor_has_elems(t, elems, sizeof(int32_t));
}

static int cuda_tensor_has_f16(const ds4_gpu_tensor *t, uint64_t elems) {
    return cuda_tensor_has_elems(t, elems, sizeof(__half));
}

static int cuda_tensor_has_u16(const ds4_gpu_tensor *t, uint64_t elems) {
    return cuda_tensor_has_elems(t, elems, sizeof(uint16_t));
}

static const char *cuda_model_range_ptr_from_fd(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what);
static const char *cuda_model_range_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what);
static int cuda_model_range_is_cached(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes);
__global__ static void dequant_q8_0_to_f16_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);
__global__ static void dequant_q8_0_to_f32_kernel(
        float *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);
__global__ static void dequant_q8_0_to_f16_transpose_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);

__global__ static void cuda_copy_bytes_kernel(
        char *dst, const char *src, uint64_t bytes) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < bytes) dst[i] = src[i];
}

static void cuda_shared_gate_up_async_cleanup(void);

static void *cuda_tmp_alloc(uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g_cuda_tmp_bytes >= bytes) return g_cuda_tmp;
    if (g_cuda_tmp) {
        (void)cudaFree(g_cuda_tmp);
        g_cuda_tmp = NULL;
        g_cuda_tmp_bytes = 0;
    }
    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "temp alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "scratch", (double)bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    g_cuda_tmp = ptr;
    g_cuda_tmp_bytes = bytes;
    return g_cuda_tmp;
}

static void *cuda_attention_seq_scratch_alloc(uint64_t bytes) {
    if (bytes == 0) return NULL;
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess ||
        device < 0 || device >= DS4_MAX_GPUS) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "sequence-tiled attention cannot resolve current device\n");
        (void)cudaGetLastError();
        return NULL;
    }
    if (g_attention_seq_scratch_bytes[device] >= bytes) {
        return g_attention_seq_scratch[device];
    }
    if (g_attention_seq_scratch[device]) {
        (void)cudaFree(g_attention_seq_scratch[device]);
        g_attention_seq_scratch[device] = NULL;
        g_attention_seq_scratch_bytes[device] = 0;
    }
    void *ptr = NULL;
    const cudaError_t err = cudaMalloc(&ptr, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX
                "sequence-tiled attention scratch alloc failed (%.2f MiB): %s\n",
                (double)bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    g_attention_seq_scratch[device] = ptr;
    g_attention_seq_scratch_bytes[device] = bytes;
    return ptr;
}

static void *cuda_glm_causal_scratch_alloc(uint64_t bytes) {
    if (bytes == 0) return NULL;
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess ||
        device < 0 || device >= DS4_MAX_GPUS) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "GLM causal attention cannot resolve current device\n");
        (void)cudaGetLastError();
        return NULL;
    }
    if (g_glm_causal_scratch_bytes[device] >= bytes) {
        return g_glm_causal_scratch[device];
    }
    if (g_glm_causal_scratch[device]) {
        (void)cudaFree(g_glm_causal_scratch[device]);
        g_glm_causal_scratch[device] = NULL;
        g_glm_causal_scratch_bytes[device] = 0;
    }
    void *ptr = NULL;
    const cudaError_t err = cudaMalloc(&ptr, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX
                "GLM causal attention scratch alloc failed (%.2f MiB): %s\n",
                (double)bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    g_glm_causal_scratch[device] = ptr;
    g_glm_causal_scratch_bytes[device] = bytes;
    return ptr;
}

static int cuda_attention_score_buffer_fits(uint32_t n_comp) {
    return n_comp <= DS4_ROCM_ATTENTION_SCORE_CAP - DS4_ROCM_ATTENTION_RAW_SCORE_CAP;
}

static int cuda_model_image_find(const void *model_map) {
    if (!model_map) return -1;
    for (size_t i = 0; i < g_model_images.size(); i++) {
        if (g_model_images[i].host_base == model_map) return (int)i;
    }
    return -1;
}

static const char *cuda_model_image_ptr(const void *model_map, uint64_t offset) {
    for (const cuda_model_image &img : g_model_images) {
        if (img.host_base != model_map || offset < img.device_offset) continue;
        const uint64_t rel = offset - img.device_offset;
        if (rel >= img.size) continue;
        return img.device_ptr + rel;
    }
    return NULL;
}

static const char *cuda_model_image_range_ptr(
        const void *model_map,
        uint64_t    offset,
        uint64_t    bytes) {
    if (bytes == 0) return cuda_model_image_ptr(model_map, offset);
    for (const cuda_model_image &img : g_model_images) {
        if (img.host_base != model_map || offset < img.device_offset) continue;
        const uint64_t rel = offset - img.device_offset;
        if (rel <= img.size && bytes <= img.size - rel) {
            return img.device_ptr + rel;
        }
    }
    return NULL;
}

static int cuda_model_image_owned(const void *model_map) {
    return cuda_model_image_find(model_map) >= 0;
}

static uint64_t cuda_model_image_bytes(void) {
    uint64_t bytes = 0;
    for (const cuda_model_image &img : g_model_images) bytes += img.size;
    return bytes;
}

static cuda_q4k_packed_slice *cuda_q4k_packed_slice_find(
        const void *model_map, uint64_t tensor_offset,
        uint32_t row_base, uint32_t row_count,
        uint64_t column_byte_base, uint64_t column_byte_count) {
    const auto range = g_q4k_packed_by_offset.equal_range(tensor_offset);
    for (auto it = range.first; it != range.second; ++it) {
        cuda_q4k_packed_slice &p = g_q4k_packed_slices[it->second];
        if (p.host_base == model_map && p.row_base == row_base &&
            p.row_count == row_count &&
            p.column_byte_base == column_byte_base &&
            p.column_byte_count == column_byte_count) return &p;
    }
    return NULL;
}

static cuda_q4k_packed_slice *cuda_q4k_packed_slice_intersection(
        const void *model_map, uint64_t offset, uint64_t bytes) {
    if (!model_map || bytes == 0u || offset > UINT64_MAX - bytes) return NULL;
    const uint64_t end = offset + bytes;
    for (cuda_q4k_packed_slice &p : g_q4k_packed_slices) {
        if (p.host_base != model_map ||
            p.tensor_offset > UINT64_MAX - p.source_tensor_bytes) continue;
        const uint64_t p_end = p.tensor_offset + p.source_tensor_bytes;
        if (offset < p_end && p.tensor_offset < end) return &p;
    }
    return NULL;
}

static int cuda_u64_ranges_overlap(
        uint64_t a_offset, uint64_t a_bytes,
        uint64_t b_offset, uint64_t b_bytes) {
    if (a_bytes == 0u || b_bytes == 0u ||
        a_offset > UINT64_MAX - a_bytes ||
        b_offset > UINT64_MAX - b_bytes) return 0;
    return a_offset < b_offset + b_bytes && b_offset < a_offset + a_bytes;
}

static int cuda_q4k_linear_residency_intersection(
        const void *model_map, uint64_t offset, uint64_t bytes);

static int cuda_q4k_packed_slice_refuse_linear(
        const void *model_map, uint64_t offset, uint64_t bytes,
        const char *what) {
    if (model_map && bytes != 0u && offset <= UINT64_MAX - bytes) {
        for (cuda_q4k_kshard_blocked_range &r :
             g_q4k_kshard_blocked_ranges) {
            if (r.host_base != model_map ||
                !cuda_u64_ranges_overlap(offset, bytes,
                                         r.offset, r.bytes)) continue;
            if (!r.logged) {
                r.logged = 1;
                fprintf(stderr, DS4_GPU_LOG_PREFIX
                        "refusing linear model resolution for retired "
                        "Q4_K K-shard tensor at offset %.2f GiB (%s); "
                        "a fresh atomic install or model cleanup is required\n",
                        (double)r.offset / 1073741824.0,
                        what ? what : "weights");
            }
            return 1;
        }
    }
    cuda_q4k_packed_slice *p =
        cuda_q4k_packed_slice_intersection(model_map, offset, bytes);
    if (!p) return 0;
    if (!p->blocked_logged) {
        p->blocked_logged = 1;
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "refusing linear model resolution for packed Q4_K routed "
                "tensor at offset %.2f GiB (%s); packed descriptor required\n",
                (double)p->tensor_offset / 1073741824.0,
                what ? what : "weights");
    }
    return 1;
}

static int cuda_q4k_packed_slice_refuse_routed_tables(
        const void *model_map, uint32_t n_total_expert,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint64_t gate_expert_bytes, uint64_t down_expert_bytes,
        const char *what) {
    uint64_t gate_bytes = 0;
    uint64_t down_bytes = 0;
    if (!cuda_u64_mul_checked(n_total_expert, gate_expert_bytes,
                              &gate_bytes) ||
        !cuda_u64_mul_checked(n_total_expert, down_expert_bytes,
                              &down_bytes)) return 1;
    return cuda_q4k_packed_slice_refuse_linear(
               model_map, gate_offset, gate_bytes, what) ||
           cuda_q4k_packed_slice_refuse_linear(
               model_map, up_offset, gate_bytes, what) ||
           cuda_q4k_packed_slice_refuse_linear(
               model_map, down_offset, down_bytes, what);
}

static void cuda_q4k_packed_slice_release_all(void) {
    if (!g_q4k_window_caches.empty()) {
        (void)cudaDeviceSynchronize();
    }
    for (ds4_gpu_q4k_window_cache *cache : g_q4k_window_caches) {
        if (!cache) continue;
        if (getenv("DS4_ROCM_GLM5_WINDOW_PROFILE") &&
            cache->profile_prepare_sec > 0.0) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "GLM5 window prepare seconds=%.6f uploads=%.2f MiB "
                    "control=%.2f KiB effective=%.2f GiB/s fills=%llu\n",
                    cache->profile_prepare_sec,
                    (double)cache->profile_upload_bytes / 1048576.0,
                    (double)cache->profile_control_bytes / 1024.0,
                    (double)cache->profile_upload_bytes /
                        cache->profile_prepare_sec / 1073741824.0,
                    (unsigned long long)cache->fills);
        }
        if (cache->slot_ids_device) (void)cudaFree(cache->slot_ids_device);
        if (cache->base) (void)cudaFree(cache->base);
        delete cache;
    }
    g_q4k_window_caches.clear();
    for (cuda_q4k_packed_slice &p : g_q4k_packed_slices) {
        if (p.device_ptr && p.owns_device_ptr) (void)cudaFree(p.device_ptr);
        p.device_ptr = NULL;
    }
    g_q4k_packed_slices.clear();
    g_q4k_packed_by_offset.clear();
    g_q4k_packed_slice_bytes = 0;
    cuda_q4k_kshard_state_clear();
}

extern "C" void ds4_gpu_q4k_packed_slice_release_all(void) {
    /* Callers use this only after a device synchronization. */
    cuda_q4k_packed_slice_release_all();
}

static void cuda_model_image_release_all(void) {
    g_q4k_kshard_cleanup_in_progress = 1;
    cuda_q4k_packed_slice_release_all();
    g_q4k_kshard_cleanup_in_progress = 0;
    for (const cuda_model_image &img : g_model_images) {
        if (img.device_ptr && img.owns_device_ptr) {
            (void)cudaFree(img.device_ptr);
        }
    }
    g_model_images.clear();
    g_q4k_kshard_blocked_ranges.clear();
}

static int cuda_env_enabled_exact(const char *name) {
    const char *value = getenv(name);
    return value && value[0] && strcmp(value, "0") != 0;
}

/* Optional DSpark-good placement: retain the compact Q8 support GGUF as one
 * device image on the coordinator.  This is deliberately opt-in because the
 * V4 Q8 drafter consumes about 10.15 GiB of persistent VRAM.  Keeping the
 * original GGUF offsets lets every ordinary tensor lookup use the existing
 * cuda_model_image_* helpers, while routed experts can be gathered D2D below.
 *
 * Do not silently fall back to NVMe streaming when the requested placement
 * does not fit.  A dspark-good launch that accidentally streams is both much
 * slower and very difficult for an operator to distinguish from a working
 * resident run.
 */
static int cuda_dspark_make_support_resident(
        const void *model_map, uint64_t model_size) {
    if (!model_map || model_size == 0) return 0;
    if (cuda_model_image_range_ptr(model_map, 0, model_size)) return 1;

    uint64_t reserve_gib = 3;
    const char *reserve_env = getenv("DS4_DSPARK_RESIDENT_RESERVE_GB");
    if (reserve_env && reserve_env[0]) {
        char *end = NULL;
        unsigned long long parsed = strtoull(reserve_env, &end, 10);
        if (end && *end == '\0' && parsed >= 1 && parsed <= 32) {
            reserve_gib = (uint64_t)parsed;
        }
    }
    const uint64_t reserve_bytes = reserve_gib << 30;
    size_t free_bytes = 0, total_bytes = 0;
    if (!cuda_ok(cudaMemGetInfo(&free_bytes, &total_bytes),
                 "DSpark resident VRAM query")) return 0;
    if ((uint64_t)free_bytes < model_size ||
        (uint64_t)free_bytes - model_size < reserve_bytes) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX
                "DSpark resident Q8 refused: support=%.2f GiB free=%.2f GiB "
                "required-reserve=%llu GiB. This mode is intentionally not "
                "falling back to streamed weights.\n",
                (double)model_size / 1073741824.0,
                (double)free_bytes / 1073741824.0,
                (unsigned long long)reserve_gib);
        return 0;
    }

    fprintf(stderr,
            DS4_GPU_LOG_PREFIX
            "WARNING: DS4_DSPARK_RESIDENT_Q8=1 retains %.2f GiB of Q8 "
            "drafter weights in VRAM (opt-in; free before load %.2f/%.2f GiB, "
            "reserve %llu GiB)\n",
            (double)model_size / 1073741824.0,
            (double)free_bytes / 1073741824.0,
            (double)total_bytes / 1073741824.0,
            (unsigned long long)reserve_gib);

    char *device = NULL;
    cudaError_t err = cudaMalloc((void **)&device, (size_t)model_size);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "DSpark resident Q8 allocation failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    const double t0 = cuda_wall_sec();
    const uint64_t chunk = 64ull * 1024ull * 1024ull;
    for (uint64_t copied = 0; copied < model_size; copied += chunk) {
        const uint64_t bytes = model_size - copied < chunk
            ? model_size - copied : chunk;
        err = cudaMemcpy(device + copied,
                         (const char *)model_map + copied,
                         (size_t)bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "DSpark resident Q8 copy failed at %.2f/%.2f GiB: %s\n",
                    (double)copied / 1073741824.0,
                    (double)model_size / 1073741824.0,
                    cudaGetErrorString(err));
            (void)cudaFree(device);
            (void)cudaGetLastError();
            return 0;
        }
    }
    g_model_images.push_back({model_map, model_size, device, 0});
    fprintf(stderr, DS4_GPU_LOG_PREFIX
            "DSpark resident Q8 ready in %.3fs (%.2f GiB, compact GGUF "
            "representation)\n",
            cuda_wall_sec() - t0,
            (double)model_size / 1073741824.0);
    return 1;
}

static int cuda_stream_resident_reclaim_wait(const char *what) {
    if (!cuda_stream_selected_reuse_wait(what)) return 0;
    if (!cuda_stream_batch_selected_reuse_wait(what)) return 0;
    return 1;
}

static void cuda_stream_resident_cache_release(void) {
    if (!cuda_stream_resident_reclaim_wait(
                "streaming resident expert cache release")) {
        return;
    }
    for (cuda_stream_resident_expert &e : g_stream_resident_experts) {
        if (e.base && !e.pooled) (void)cudaFree(e.base);
    }
    g_stream_resident_experts.clear();
    g_stream_resident_index.clear();
    g_stream_resident_bytes = 0;
    g_stream_resident_clock = 0;
    for (cuda_stream_expert_slab &slab : g_stream_expert_slabs) {
        if (slab.base) (void)cudaFree(slab.base);
    }
    g_stream_expert_slabs.clear();
    g_stream_expert_free_slots.clear();
    g_stream_expert_slot_bytes = 0;
    g_stream_expert_slot_count = 0;
}

static void cuda_stream_layer_expert_cache_release(void) {
    bool any_active = false;
    for (uint32_t i = 0; i < 2u; i++) {
        if (g_stream_layer_expert_cache[i].base) {
            any_active = true;
            break;
        }
    }
    if (any_active) {
        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming full-layer expert cache "
                    "release sync failed: %s\n",
                    cudaGetErrorString(sync_err));
            (void)cudaGetLastError();
        }
    }
    for (uint32_t i = 0; i < 2u; i++) {
        cuda_stream_layer_expert_cache &c = g_stream_layer_expert_cache[i];
        if (c.base) (void)cudaFree(c.base);
        memset(&c, 0, sizeof(c));
    }
}

static void cuda_stream_read_stage_release(void) {
    cuda_stream_read_pool_shutdown();
    for (uint32_t i = 0; i < DS4_ROCM_STREAM_READ_WORKERS; i++) {
        if (g_stream_read_stage_raw[i]) {
            (void)cudaFreeHost(g_stream_read_stage_raw[i]);
            g_stream_read_stage_raw[i] = NULL;
            g_stream_read_stage_bytes[i] = 0;
        }
    }
    for (uint32_t i = 0; i < DS4_ROCM_STREAM_READ_WORKERS; i++) {
        if (g_stream_read_upload_streams[i]) {
            (void)cudaStreamDestroy(g_stream_read_upload_streams[i]);
            g_stream_read_upload_streams[i] = NULL;
        }
    }
}

static void cuda_stream_batch_selected_cache_release(void) {
    (void)cuda_stream_batch_selected_reuse_wait(
            "streaming batch selected cache release");
    (void)cuda_stream_batch_selected_upload_wait_host(
            "streaming batch selected cache release upload");
    if (g_stream_batch_selected_cache.selected_ids) {
        (void)cudaFree(g_stream_batch_selected_cache.selected_ids);
    }
    if (g_stream_batch_selected_cache.pair_missing) {
        (void)cudaFree(g_stream_batch_selected_cache.pair_missing);
    }
    if (g_stream_batch_selected_cache.gate_ptrs) {
        (void)cudaFree(g_stream_batch_selected_cache.gate_ptrs);
    }
    if (g_stream_batch_selected_cache.up_ptrs) {
        (void)cudaFree(g_stream_batch_selected_cache.up_ptrs);
    }
    if (g_stream_batch_selected_cache.down_ptrs) {
        (void)cudaFree(g_stream_batch_selected_cache.down_ptrs);
    }
    if (g_stream_batch_selected_cache.resident_gate_ptrs) {
        (void)cudaFree(g_stream_batch_selected_cache.resident_gate_ptrs);
    }
    if (g_stream_batch_selected_cache.resident_up_ptrs) {
        (void)cudaFree(g_stream_batch_selected_cache.resident_up_ptrs);
    }
    if (g_stream_batch_selected_cache.missing_gate_ptrs) {
        (void)cudaFree(g_stream_batch_selected_cache.missing_gate_ptrs);
    }
    if (g_stream_batch_selected_cache.missing_up_ptrs) {
        (void)cudaFree(g_stream_batch_selected_cache.missing_up_ptrs);
    }
    if (g_stream_batch_selected_reuse_event) {
        (void)cudaEventDestroy(g_stream_batch_selected_reuse_event);
        g_stream_batch_selected_reuse_event = NULL;
    }
    if (g_stream_batch_selected_upload_ready_event) {
        (void)cudaEventDestroy(g_stream_batch_selected_upload_ready_event);
        g_stream_batch_selected_upload_ready_event = NULL;
    }
    g_stream_batch_selected_reuse_event_pending = 0;
    g_stream_batch_selected_upload_event_pending = 0;
    cuda_stream_batch_selected_stage_release();
    memset(&g_stream_batch_selected_cache, 0, sizeof(g_stream_batch_selected_cache));
}

static void cuda_stream_selected_cache_release(void) {
    (void)cuda_stream_selected_reuse_wait("streaming selected cache release");
    (void)cuda_stream_selected_upload_wait_host(
            "streaming selected cache release upload");
    if (g_stream_selected_cache.gate) (void)cudaFree(g_stream_selected_cache.gate);
    if (g_stream_selected_cache.up) (void)cudaFree(g_stream_selected_cache.up);
    if (g_stream_selected_cache.down) (void)cudaFree(g_stream_selected_cache.down);
    if (g_stream_selected_cache.slot_ids) (void)cudaFree(g_stream_selected_cache.slot_ids);
    if (g_stream_selected_cache.gate_ptrs) (void)cudaFree(g_stream_selected_cache.gate_ptrs);
    if (g_stream_selected_cache.up_ptrs) (void)cudaFree(g_stream_selected_cache.up_ptrs);
    if (g_stream_selected_cache.down_ptrs) (void)cudaFree(g_stream_selected_cache.down_ptrs);
    if (g_stream_selected_reuse_event) {
        (void)cudaEventDestroy(g_stream_selected_reuse_event);
        g_stream_selected_reuse_event = NULL;
    }
    if (g_stream_selected_upload_ready_event) {
        (void)cudaEventDestroy(g_stream_selected_upload_ready_event);
        g_stream_selected_upload_ready_event = NULL;
    }
    g_stream_selected_reuse_event_pending = 0;
    g_stream_selected_upload_event_pending = 0;
    cuda_stream_selected_stage_release();
    memset(&g_stream_selected_cache, 0, sizeof(g_stream_selected_cache));
    cuda_stream_batch_selected_cache_release();
    cuda_stream_resident_cache_release();
    cuda_stream_layer_expert_cache_release();
    cuda_stream_read_stage_release();
    g_routed_moe_selected_override_n = 0;
}

static int cuda_stream_selected_ensure_stream(void) {
    if (g_stream_selected_upload_stream) return 1;
    cudaError_t err = cudaStreamCreateWithFlags(&g_stream_selected_upload_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming selected upload stream creation failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

static int cuda_stream_selected_reuse_ensure_event(void) {
    if (g_stream_selected_reuse_event) return 1;
    cudaError_t err =
        cudaEventCreateWithFlags(&g_stream_selected_reuse_event,
                                 cudaEventDisableTiming);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming selected reuse event creation failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

static int cuda_stream_selected_reuse_wait(const char *what) {
    if (!g_stream_selected_reuse_event_pending) return 1;
    cudaError_t err = cudaEventSynchronize(g_stream_selected_reuse_event);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "%s wait failed: %s\n",
                what ? what : "streaming selected cache reuse",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    g_stream_selected_reuse_event_pending = 0;
    return 1;
}

static int cuda_stream_selected_upload_ensure_event(void) {
    if (g_stream_selected_upload_ready_event) return 1;
    cudaError_t err =
        cudaEventCreateWithFlags(&g_stream_selected_upload_ready_event,
                                 cudaEventDisableTiming);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming selected upload event creation failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

static int cuda_stream_selected_upload_wait_host(const char *what) {
    if (!g_stream_selected_upload_event_pending) return 1;
    cudaError_t err = cudaEventSynchronize(g_stream_selected_upload_ready_event);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "%s wait failed: %s\n",
                what ? what : "streaming selected upload",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    g_stream_selected_upload_event_pending = 0;
    return 1;
}

static int cuda_stream_selected_upload_record_ready(void) {
    if (!cuda_stream_selected_upload_ensure_event()) return 0;
    cudaError_t err = cudaEventRecord(g_stream_selected_upload_ready_event,
                                      g_stream_selected_upload_stream);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming selected upload event record failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    g_stream_selected_upload_event_pending = 1;
    return 1;
}

static int cuda_stream_selected_wait_upload_ready(void) {
    if (!g_stream_selected_upload_event_pending) return 1;
#ifdef __HIP_PLATFORM_AMD__
    cudaError_t err =
        hipStreamWaitEvent(0, g_stream_selected_upload_ready_event, 0);
#else
    cudaError_t err =
        cudaStreamWaitEvent(0, g_stream_selected_upload_ready_event, 0);
#endif
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming selected upload stream wait failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

static int cuda_stream_selected_mark_inflight(void) {
    if (!g_ssd_streaming_mode) return 1;
    if (!cuda_stream_selected_reuse_ensure_event()) return 0;
    cudaError_t err = cudaEventRecord(g_stream_selected_reuse_event, 0);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming selected reuse event record failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    g_stream_selected_reuse_event_pending = 1;
    g_stream_selected_upload_event_pending = 0;
    return 1;
}

static int cuda_stream_batch_selected_reuse_ensure_event(void) {
    if (g_stream_batch_selected_reuse_event) return 1;
    cudaError_t err =
        cudaEventCreateWithFlags(&g_stream_batch_selected_reuse_event,
                                 cudaEventDisableTiming);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming batch selected reuse event creation failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

static int cuda_stream_batch_selected_upload_ensure_event(void) {
    if (g_stream_batch_selected_upload_ready_event) return 1;
    cudaError_t err =
        cudaEventCreateWithFlags(&g_stream_batch_selected_upload_ready_event,
                                 cudaEventDisableTiming);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming batch selected upload event creation failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

static int cuda_stream_batch_selected_upload_wait_host(const char *what) {
    if (!g_stream_batch_selected_upload_event_pending) return 1;
    cudaError_t err =
        cudaEventSynchronize(g_stream_batch_selected_upload_ready_event);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "%s wait failed: %s\n",
                what ? what : "streaming batch selected upload",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    g_stream_batch_selected_upload_event_pending = 0;
    return 1;
}

static int cuda_stream_batch_selected_upload_record_ready(void) {
    if (!cuda_stream_batch_selected_upload_ensure_event()) return 0;
    cudaError_t err = cudaEventRecord(g_stream_batch_selected_upload_ready_event,
                                      g_stream_selected_upload_stream);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming batch selected upload event record failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    g_stream_batch_selected_upload_event_pending = 1;
    return 1;
}

static int cuda_stream_batch_selected_wait_upload_ready(void) {
    if (!g_stream_batch_selected_upload_event_pending) return 1;
#ifdef __HIP_PLATFORM_AMD__
    cudaError_t err =
        hipStreamWaitEvent(0, g_stream_batch_selected_upload_ready_event, 0);
#else
    cudaError_t err =
        cudaStreamWaitEvent(0, g_stream_batch_selected_upload_ready_event, 0);
#endif
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming batch selected upload stream wait failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

static int cuda_stream_batch_selected_reuse_wait(const char *what) {
    if (!g_stream_batch_selected_reuse_event_pending) return 1;
    cudaError_t err = cudaEventSynchronize(g_stream_batch_selected_reuse_event);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "%s wait failed: %s\n",
                what ? what : "streaming batch selected cache reuse",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    g_stream_batch_selected_reuse_event_pending = 0;
    return 1;
}

static int cuda_stream_batch_selected_mark_inflight(void) {
    if (!g_ssd_streaming_mode) return 1;
    if (!cuda_stream_batch_selected_reuse_ensure_event()) return 0;
    cudaError_t err = cudaEventRecord(g_stream_batch_selected_reuse_event, 0);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming batch selected reuse event record failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    g_stream_batch_selected_reuse_event_pending = 1;
    g_stream_batch_selected_upload_event_pending = 0;
    return 1;
}

static void cuda_stream_selected_stage_release(void) {
    if (g_stream_selected_cache.gate_ptrs_stage) {
        (void)cudaFreeHost((void *)g_stream_selected_cache.gate_ptrs_stage);
    }
    if (g_stream_selected_cache.up_ptrs_stage) {
        (void)cudaFreeHost((void *)g_stream_selected_cache.up_ptrs_stage);
    }
    if (g_stream_selected_cache.down_ptrs_stage) {
        (void)cudaFreeHost((void *)g_stream_selected_cache.down_ptrs_stage);
    }
    g_stream_selected_cache.gate_ptrs_stage = NULL;
    g_stream_selected_cache.up_ptrs_stage = NULL;
    g_stream_selected_cache.down_ptrs_stage = NULL;
}

static int cuda_stream_selected_ensure_buffers(uint64_t gate_bytes, uint64_t down_bytes) {
    if (gate_bytes == 0 || down_bytes == 0) return 0;
    if (!cuda_stream_selected_upload_wait_host(
                "streaming selected cache upload")) {
        return 0;
    }
    cudaError_t err = cudaSuccess;
    if (g_stream_selected_cache.gate_capacity < gate_bytes) {
        if (g_stream_selected_cache.gate) (void)cudaFree(g_stream_selected_cache.gate);
        if (g_stream_selected_cache.up) (void)cudaFree(g_stream_selected_cache.up);
        g_stream_selected_cache.gate = NULL;
        g_stream_selected_cache.up = NULL;
        g_stream_selected_cache.gate_capacity = 0;
        err = cudaMalloc((void **)&g_stream_selected_cache.gate, (size_t)gate_bytes);
        if (err == cudaSuccess) {
            err = cudaMalloc((void **)&g_stream_selected_cache.up, (size_t)gate_bytes);
        }
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming selected gate/up alloc failed (%.2f MiB): %s\n",
                    (double)gate_bytes / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_stream_selected_cache.gate_capacity = gate_bytes;
    }
    if (g_stream_selected_cache.down_capacity < down_bytes) {
        if (g_stream_selected_cache.down) (void)cudaFree(g_stream_selected_cache.down);
        g_stream_selected_cache.down = NULL;
        g_stream_selected_cache.down_capacity = 0;
        err = cudaMalloc((void **)&g_stream_selected_cache.down, (size_t)down_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming selected down alloc failed (%.2f MiB): %s\n",
                    (double)down_bytes / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_stream_selected_cache.down_capacity = down_bytes;
    }
    if (!g_stream_selected_cache.slot_ids) {
        err = cudaMalloc((void **)&g_stream_selected_cache.slot_ids,
                         DS4_ROCM_N_EXPERT_USED * sizeof(int32_t));
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming selected slot-id alloc failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        int32_t slots[DS4_ROCM_N_EXPERT_USED];
        for (uint32_t i = 0; i < DS4_ROCM_N_EXPERT_USED; i++) slots[i] = (int32_t)i;
        err = cudaMemcpy(g_stream_selected_cache.slot_ids,
                         slots,
                         sizeof(slots),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming selected slot-id upload failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_stream_selected_cache.slot_tensor.ptr = g_stream_selected_cache.slot_ids;
        g_stream_selected_cache.slot_tensor.bytes =
            DS4_ROCM_N_EXPERT_USED * sizeof(int32_t);
        g_stream_selected_cache.slot_tensor.owner = 0;
    }
    if (!g_stream_selected_cache.gate_ptrs) {
        err = cudaMalloc((void **)&g_stream_selected_cache.gate_ptrs,
                         DS4_ROCM_N_EXPERT_USED * sizeof(char *));
        if (err == cudaSuccess) {
            err = cudaMalloc((void **)&g_stream_selected_cache.up_ptrs,
                             DS4_ROCM_N_EXPERT_USED * sizeof(char *));
        }
        if (err == cudaSuccess) {
            err = cudaMalloc((void **)&g_stream_selected_cache.down_ptrs,
                             DS4_ROCM_N_EXPERT_USED * sizeof(char *));
        }
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming selected pointer table alloc failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    if (!g_stream_selected_cache.gate_ptrs_stage) {
        void *stage = NULL;
        err = cudaMallocHost(&stage, DS4_ROCM_N_EXPERT_USED * sizeof(char *));
        g_stream_selected_cache.gate_ptrs_stage =
            err == cudaSuccess ? (const char **)stage : NULL;
        stage = NULL;
        if (err == cudaSuccess) {
            err = cudaMallocHost(&stage, DS4_ROCM_N_EXPERT_USED * sizeof(char *));
            g_stream_selected_cache.up_ptrs_stage =
                err == cudaSuccess ? (const char **)stage : NULL;
            stage = NULL;
        }
        if (err == cudaSuccess) {
            err = cudaMallocHost(&stage, DS4_ROCM_N_EXPERT_USED * sizeof(char *));
            g_stream_selected_cache.down_ptrs_stage =
                err == cudaSuccess ? (const char **)stage : NULL;
        }
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming selected pointer staging alloc failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            cuda_stream_selected_stage_release();
            return 0;
        }
    }
    return 1;
}

static void cuda_stream_batch_selected_stage_release(void) {
    if (g_stream_batch_selected_cache.selected_stage) {
        (void)cudaFreeHost(g_stream_batch_selected_cache.selected_stage);
    }
    if (g_stream_batch_selected_cache.pair_missing_stage) {
        (void)cudaFreeHost(g_stream_batch_selected_cache.pair_missing_stage);
    }
    if (g_stream_batch_selected_cache.gate_ptrs_stage) {
        (void)cudaFreeHost((void *)g_stream_batch_selected_cache.gate_ptrs_stage);
    }
    if (g_stream_batch_selected_cache.up_ptrs_stage) {
        (void)cudaFreeHost((void *)g_stream_batch_selected_cache.up_ptrs_stage);
    }
    if (g_stream_batch_selected_cache.down_ptrs_stage) {
        (void)cudaFreeHost((void *)g_stream_batch_selected_cache.down_ptrs_stage);
    }
    if (g_stream_batch_selected_cache.resident_gate_ptrs_stage) {
        (void)cudaFreeHost((void *)g_stream_batch_selected_cache.resident_gate_ptrs_stage);
    }
    if (g_stream_batch_selected_cache.resident_up_ptrs_stage) {
        (void)cudaFreeHost((void *)g_stream_batch_selected_cache.resident_up_ptrs_stage);
    }
    if (g_stream_batch_selected_cache.missing_gate_ptrs_stage) {
        (void)cudaFreeHost((void *)g_stream_batch_selected_cache.missing_gate_ptrs_stage);
    }
    if (g_stream_batch_selected_cache.missing_up_ptrs_stage) {
        (void)cudaFreeHost((void *)g_stream_batch_selected_cache.missing_up_ptrs_stage);
    }
    g_stream_batch_selected_cache.selected_stage = NULL;
    g_stream_batch_selected_cache.selected_stage_capacity = 0;
    g_stream_batch_selected_cache.pair_missing_stage = NULL;
    g_stream_batch_selected_cache.pair_missing_stage_capacity = 0;
    g_stream_batch_selected_cache.gate_ptrs_stage = NULL;
    g_stream_batch_selected_cache.up_ptrs_stage = NULL;
    g_stream_batch_selected_cache.down_ptrs_stage = NULL;
    g_stream_batch_selected_cache.resident_gate_ptrs_stage = NULL;
    g_stream_batch_selected_cache.resident_up_ptrs_stage = NULL;
    g_stream_batch_selected_cache.missing_gate_ptrs_stage = NULL;
    g_stream_batch_selected_cache.missing_up_ptrs_stage = NULL;
    g_stream_batch_selected_cache.ptr_stage_capacity = 0;
}

static int cuda_stream_batch_selected_ensure_buffers(
        uint64_t n_ids,
        uint32_t n_unique) {
    if (n_ids == 0 || n_unique == 0) return 0;
    if (!cuda_stream_batch_selected_reuse_wait(
                "streaming batch selected cache reuse")) {
        return 0;
    }
    if (!cuda_stream_batch_selected_upload_wait_host(
                "streaming batch selected cache upload")) {
        return 0;
    }
    cudaError_t err = cudaSuccess;
    const uint64_t selected_bytes = n_ids * sizeof(int32_t);
    if (g_stream_batch_selected_cache.selected_capacity < selected_bytes) {
        if (g_stream_batch_selected_cache.selected_ids) {
            (void)cudaFree(g_stream_batch_selected_cache.selected_ids);
            g_stream_batch_selected_cache.selected_ids = NULL;
            g_stream_batch_selected_cache.selected_capacity = 0;
        }
        err = cudaMalloc((void **)&g_stream_batch_selected_cache.selected_ids,
                         (size_t)selected_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming batch selected-id alloc failed "
                    "(%.2f MiB): %s\n",
                    (double)selected_bytes / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_stream_batch_selected_cache.selected_capacity = selected_bytes;
    }
    if (g_stream_batch_selected_cache.pair_missing_capacity < n_ids) {
        if (g_stream_batch_selected_cache.pair_missing) {
            (void)cudaFree(g_stream_batch_selected_cache.pair_missing);
            g_stream_batch_selected_cache.pair_missing = NULL;
            g_stream_batch_selected_cache.pair_missing_capacity = 0;
        }
        err = cudaMalloc((void **)&g_stream_batch_selected_cache.pair_missing,
                         (size_t)n_ids);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming batch selected split-flag alloc failed "
                    "(%.2f MiB): %s\n",
                    (double)n_ids / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_stream_batch_selected_cache.pair_missing_capacity = n_ids;
    }
    if (g_stream_batch_selected_cache.ptr_capacity < n_unique) {
        if (g_stream_batch_selected_cache.gate_ptrs) {
            (void)cudaFree(g_stream_batch_selected_cache.gate_ptrs);
            (void)cudaFree(g_stream_batch_selected_cache.up_ptrs);
            (void)cudaFree(g_stream_batch_selected_cache.down_ptrs);
            (void)cudaFree(g_stream_batch_selected_cache.resident_gate_ptrs);
            (void)cudaFree(g_stream_batch_selected_cache.resident_up_ptrs);
            (void)cudaFree(g_stream_batch_selected_cache.missing_gate_ptrs);
            (void)cudaFree(g_stream_batch_selected_cache.missing_up_ptrs);
            g_stream_batch_selected_cache.gate_ptrs = NULL;
            g_stream_batch_selected_cache.up_ptrs = NULL;
            g_stream_batch_selected_cache.down_ptrs = NULL;
            g_stream_batch_selected_cache.resident_gate_ptrs = NULL;
            g_stream_batch_selected_cache.resident_up_ptrs = NULL;
            g_stream_batch_selected_cache.missing_gate_ptrs = NULL;
            g_stream_batch_selected_cache.missing_up_ptrs = NULL;
            g_stream_batch_selected_cache.ptr_capacity = 0;
        }
        err = cudaMalloc((void **)&g_stream_batch_selected_cache.gate_ptrs,
                         (size_t)n_unique * sizeof(char *));
        if (err == cudaSuccess) {
            err = cudaMalloc((void **)&g_stream_batch_selected_cache.up_ptrs,
                             (size_t)n_unique * sizeof(char *));
        }
        if (err == cudaSuccess) {
            err = cudaMalloc((void **)&g_stream_batch_selected_cache.down_ptrs,
                             (size_t)n_unique * sizeof(char *));
        }
        if (err == cudaSuccess) {
            err = cudaMalloc((void **)&g_stream_batch_selected_cache.resident_gate_ptrs,
                             (size_t)n_unique * sizeof(char *));
        }
        if (err == cudaSuccess) {
            err = cudaMalloc((void **)&g_stream_batch_selected_cache.resident_up_ptrs,
                             (size_t)n_unique * sizeof(char *));
        }
        if (err == cudaSuccess) {
            err = cudaMalloc((void **)&g_stream_batch_selected_cache.missing_gate_ptrs,
                             (size_t)n_unique * sizeof(char *));
        }
        if (err == cudaSuccess) {
            err = cudaMalloc((void **)&g_stream_batch_selected_cache.missing_up_ptrs,
                             (size_t)n_unique * sizeof(char *));
        }
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming batch pointer-table alloc failed "
                    "(unique=%u): %s\n",
                    n_unique,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_stream_batch_selected_cache.ptr_capacity = n_unique;
    }
    if (g_stream_batch_selected_cache.selected_stage_capacity < selected_bytes ||
        g_stream_batch_selected_cache.pair_missing_stage_capacity < n_ids ||
        g_stream_batch_selected_cache.ptr_stage_capacity < n_unique) {
        cuda_stream_batch_selected_stage_release();
        err = cudaMallocHost((void **)&g_stream_batch_selected_cache.selected_stage,
                             (size_t)selected_bytes);
        if (err == cudaSuccess) {
            err = cudaMallocHost((void **)&g_stream_batch_selected_cache.pair_missing_stage,
                                 (size_t)n_ids);
        }
        void *stage = NULL;
        if (err == cudaSuccess) {
            err = cudaMallocHost(&stage, (size_t)n_unique * sizeof(char *));
            g_stream_batch_selected_cache.gate_ptrs_stage =
                err == cudaSuccess ? (const char **)stage : NULL;
            stage = NULL;
        }
        if (err == cudaSuccess) {
            err = cudaMallocHost(&stage, (size_t)n_unique * sizeof(char *));
            g_stream_batch_selected_cache.up_ptrs_stage =
                err == cudaSuccess ? (const char **)stage : NULL;
            stage = NULL;
        }
        if (err == cudaSuccess) {
            err = cudaMallocHost(&stage, (size_t)n_unique * sizeof(char *));
            g_stream_batch_selected_cache.down_ptrs_stage =
                err == cudaSuccess ? (const char **)stage : NULL;
            stage = NULL;
        }
        if (err == cudaSuccess) {
            err = cudaMallocHost(&stage, (size_t)n_unique * sizeof(char *));
            g_stream_batch_selected_cache.resident_gate_ptrs_stage =
                err == cudaSuccess ? (const char **)stage : NULL;
            stage = NULL;
        }
        if (err == cudaSuccess) {
            err = cudaMallocHost(&stage, (size_t)n_unique * sizeof(char *));
            g_stream_batch_selected_cache.resident_up_ptrs_stage =
                err == cudaSuccess ? (const char **)stage : NULL;
            stage = NULL;
        }
        if (err == cudaSuccess) {
            err = cudaMallocHost(&stage, (size_t)n_unique * sizeof(char *));
            g_stream_batch_selected_cache.missing_gate_ptrs_stage =
                err == cudaSuccess ? (const char **)stage : NULL;
            stage = NULL;
        }
        if (err == cudaSuccess) {
            err = cudaMallocHost(&stage, (size_t)n_unique * sizeof(char *));
            g_stream_batch_selected_cache.missing_up_ptrs_stage =
                err == cudaSuccess ? (const char **)stage : NULL;
        }
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming batch selected staging alloc failed "
                    "(ids=%.2f MiB unique=%u): %s\n",
                    (double)selected_bytes / 1048576.0,
                    n_unique,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            cuda_stream_batch_selected_stage_release();
            return 0;
        }
        g_stream_batch_selected_cache.selected_stage_capacity = selected_bytes;
        g_stream_batch_selected_cache.pair_missing_stage_capacity = n_ids;
        g_stream_batch_selected_cache.ptr_stage_capacity = n_unique;
    }
    g_stream_batch_selected_cache.selected_tensor.ptr =
        g_stream_batch_selected_cache.selected_ids;
    g_stream_batch_selected_cache.selected_tensor.bytes = selected_bytes;
    g_stream_batch_selected_cache.selected_tensor.owner = 0;
    return 1;
}

static int cuda_stream_selected_is_current(
        const cuda_stream_resident_expert &e,
        uint32_t layer,
        const int32_t *selected_ids,
        uint32_t n_selected) {
    if (!selected_ids || e.layer != layer) return 0;
    for (uint32_t i = 0; i < n_selected; i++) {
        if (e.expert == selected_ids[i]) return 1;
    }
    return 0;
}

static cuda_stream_resident_key cuda_stream_resident_make_key(
        const void *model_map,
        uint32_t layer,
        int32_t expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    cuda_stream_resident_key k;
    k.model_map = model_map;
    k.layer = layer;
    k.expert = expert;
    k.gate_offset = gate_offset;
    k.up_offset = up_offset;
    k.down_offset = down_offset;
    k.gate_expert_bytes = gate_expert_bytes;
    k.down_expert_bytes = down_expert_bytes;
    return k;
}

static cuda_stream_resident_key cuda_stream_resident_entry_key(
        const cuda_stream_resident_expert &e) {
    return cuda_stream_resident_make_key(e.model_map,
                                         e.layer,
                                         e.expert,
                                         e.gate_offset,
                                         e.up_offset,
                                         e.down_offset,
                                         e.gate_expert_bytes,
                                         e.down_expert_bytes);
}

static int cuda_stream_resident_find(
        const void *model_map,
        uint32_t layer,
        int32_t expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    const cuda_stream_resident_key key =
        cuda_stream_resident_make_key(model_map,
                                      layer,
                                      expert,
                                      gate_offset,
                                      up_offset,
                                      down_offset,
                                      gate_expert_bytes,
                                      down_expert_bytes);
    const auto it = g_stream_resident_index.find(key);
    if (it != g_stream_resident_index.end() &&
        it->second < g_stream_resident_experts.size()) {
        return (int)it->second;
    }
    return -1;
}

static int cuda_stream_resident_evict_at(size_t idx) {
    if (idx >= g_stream_resident_experts.size()) return 0;
    if (!cuda_stream_resident_reclaim_wait(
                "streaming resident expert cache eviction")) {
        return 0;
    }
    cuda_stream_resident_expert &e = g_stream_resident_experts[idx];
    const cuda_stream_resident_key evicted_key =
        cuda_stream_resident_entry_key(e);
    if (cuda_stream_cache_stats_on()) {
        g_stream_cache_stats.evictions++;
        g_stream_cache_stats.evict_bytes += e.bytes;
    }
    if (e.base) {
        if (e.pooled) g_stream_expert_free_slots.push_back(e.base);
        else (void)cudaFree(e.base);
    }
    if (g_stream_resident_bytes >= e.bytes) {
        g_stream_resident_bytes -= e.bytes;
    } else {
        g_stream_resident_bytes = 0;
    }
    g_stream_resident_index.erase(evicted_key);
    const size_t last = g_stream_resident_experts.size() - 1u;
    if (idx != last) {
        g_stream_resident_experts[idx] = g_stream_resident_experts[last];
        g_stream_resident_index[cuda_stream_resident_entry_key(
                g_stream_resident_experts[idx])] = idx;
    }
    g_stream_resident_experts.pop_back();
    return 1;
}

static int cuda_stream_resident_evict_one(
        uint32_t layer,
        const int32_t *selected_ids,
        uint32_t n_selected) {
    size_t victim = (size_t)-1;
    uint64_t oldest = UINT64_MAX;
    if (cuda_stream_evict_past_layers_first()) {
        for (size_t i = 0; i < g_stream_resident_experts.size(); i++) {
            const cuda_stream_resident_expert &e =
                g_stream_resident_experts[i];
            if (e.layer > layer ||
                cuda_stream_selected_is_current(e,
                                                layer,
                                                selected_ids,
                                                n_selected)) {
                continue;
            }
            if (e.last_used < oldest) {
                oldest = e.last_used;
                victim = i;
            }
        }
        if (victim != (size_t)-1) {
            return cuda_stream_resident_evict_at(victim);
        }
    }
    oldest = UINT64_MAX;
    for (size_t i = 0; i < g_stream_resident_experts.size(); i++) {
        const cuda_stream_resident_expert &e = g_stream_resident_experts[i];
        if (cuda_stream_selected_is_current(e, layer, selected_ids, n_selected)) {
            continue;
        }
        if (e.last_used < oldest) {
            oldest = e.last_used;
            victim = i;
        }
    }
    if (victim == (size_t)-1) return 0;
    return cuda_stream_resident_evict_at(victim);
}

static uint64_t cuda_stream_resident_free_reserve_bytes(void) {
    /*
     * Headroom kept free on the (unified-memory) device while growing the
     * expert cache.  It must cover decode scratch and transient graph buffers
     * but every reserved GiB is a GiB not spent caching experts, so keep it
     * tight.  Override with DS4_ROCM_STREAM_FREE_RESERVE_GB.
     */
    static int64_t cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_ROCM_STREAM_FREE_RESERVE_GB");
        uint64_t gib = 16;
        if (env && env[0]) {
            char *end = NULL;
            errno = 0;
            unsigned long v = strtoul(env, &end, 10);
            if (end != env && *end == '\0' && errno == 0 && v >= 2 && v <= 64) {
                gib = (uint64_t)v;
            }
        }
        cached = (int64_t)(gib * 1024ull * 1024ull * 1024ull);
    }
    return (uint64_t)cached;
}

static int cuda_stream_resident_make_room(
        uint64_t bytes,
        uint32_t layer,
        const int32_t *selected_ids,
        uint32_t n_selected) {
    while (g_stream_resident_experts.size() >= g_stream_expert_cache_budget) {
        if (!cuda_stream_resident_evict_one(layer, selected_ids, n_selected)) {
            break;
        }
    }

    size_t free_b = 0;
    size_t total_b = 0;
    const uint64_t reserve = cuda_stream_resident_free_reserve_bytes();
    while (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
        (void)total_b;
        if ((uint64_t)free_b >= reserve &&
            bytes <= (uint64_t)free_b - reserve) {
            return 1;
        }
        if (!cuda_stream_resident_evict_one(layer, selected_ids, n_selected)) {
            return 0;
        }
    }
    (void)cudaGetLastError();
    return 1;
}

static int cuda_stream_expert_slab_grow(uint64_t slot_bytes) {
    const uint64_t slab_target_bytes = 1024ull * 1024ull * 1024ull;
    if (g_stream_expert_slot_count >= g_stream_expert_cache_budget) return 0;
    uint32_t slab_slots =
        slot_bytes >= slab_target_bytes ?
            1u : (uint32_t)(slab_target_bytes / slot_bytes);
    const uint32_t want =
        g_stream_expert_cache_budget - g_stream_expert_slot_count;
    if (slab_slots > want) slab_slots = want;
    while (slab_slots != 0) {
        uint64_t slab_bytes = 0;
        if (!cuda_u64_mul_checked(slab_slots, slot_bytes, &slab_bytes)) {
            slab_slots >>= 1u;
            continue;
        }
        size_t free_b = 0;
        size_t total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
            (void)total_b;
            const uint64_t reserve = cuda_stream_resident_free_reserve_bytes();
            if ((uint64_t)free_b < reserve ||
                slab_bytes > (uint64_t)free_b - reserve) {
                slab_slots >>= 1u;
                continue;
            }
        } else {
            (void)cudaGetLastError();
        }
        void *base = NULL;
        cudaError_t err = cudaMalloc(&base, (size_t)slab_bytes);
        if (err != cudaSuccess) {
            (void)cudaGetLastError();
            slab_slots >>= 1u;
            continue;
        }
        g_stream_expert_slabs.push_back({(char *)base, slab_bytes});
        g_stream_expert_free_slots.reserve(
                g_stream_expert_free_slots.size() + slab_slots);
        for (uint32_t i = 0; i < slab_slots; i++) {
            g_stream_expert_free_slots.push_back(
                    (char *)base + (uint64_t)i * slot_bytes);
        }
        g_stream_expert_slot_count += slab_slots;
        return 1;
    }
    return 0;
}

static char *cuda_stream_expert_slot_acquire(
        uint64_t bytes,
        uint32_t layer,
        const int32_t *selected_ids,
        uint32_t n_selected) {
    if (g_stream_expert_slot_bytes == 0) g_stream_expert_slot_bytes = bytes;
    if (bytes != g_stream_expert_slot_bytes) return NULL;
    for (;;) {
        if (!g_stream_expert_free_slots.empty()) {
            char *slot = g_stream_expert_free_slots.back();
            g_stream_expert_free_slots.pop_back();
            return slot;
        }
        if (g_stream_expert_slot_count < g_stream_expert_cache_budget &&
            cuda_stream_expert_slab_grow(bytes)) {
            continue;
        }
        if (!cuda_stream_resident_evict_one(layer, selected_ids, n_selected)) {
            return NULL;
        }
    }
}

static int cuda_stream_resident_alloc(
        const void *model_map,
        uint32_t layer,
        int32_t expert,
        const int32_t *selected_ids,
        uint32_t n_selected,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (g_stream_expert_cache_budget == 0) return -1;
    if (!cuda_stream_resident_reclaim_wait(
                "streaming resident expert cache allocation")) {
        return -1;
    }
    uint64_t bytes = 0;
    uint64_t gate_pair = 0;
    if (!cuda_u64_mul_checked(2u, gate_expert_bytes, &gate_pair) ||
        gate_pair > UINT64_MAX - down_expert_bytes) {
        return -1;
    }
    bytes = gate_pair + down_expert_bytes;

    void *base = cuda_stream_expert_slot_acquire(bytes,
                                                 layer,
                                                 selected_ids,
                                                 n_selected);
    const int pooled = base != NULL;
    if (!pooled && bytes == g_stream_expert_slot_bytes) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming expert cache cannot reserve a "
                "%.2f MiB slot for layer=%u expert=%d\n",
                (double)bytes / 1048576.0,
                layer,
                expert);
        return -1;
    }
    if (!pooled) {
        /*
         * Mixed-precision layers whose experts do not match the slab size
         * class fall back to dedicated allocations.
         */
        if (!cuda_stream_resident_make_room(bytes, layer, selected_ids, n_selected)) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming expert cache cannot keep %.2f MiB "
                    "for layer=%u expert=%d while preserving %.2f GiB free\n",
                    (double)bytes / 1048576.0,
                    layer,
                    expert,
                    (double)cuda_stream_resident_free_reserve_bytes() / 1073741824.0);
            return -1;
        }

        cudaError_t err = cudaMalloc(&base, (size_t)bytes);
        while (err != cudaSuccess && cuda_stream_resident_evict_one(layer, selected_ids, n_selected)) {
            (void)cudaGetLastError();
            err = cudaMalloc(&base, (size_t)bytes);
        }
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming expert cache allocation failed "
                    "for layer=%u expert=%d (%.2f MiB): %s\n",
                    layer,
                    expert,
                    (double)bytes / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return -1;
        }
    }

    cuda_stream_resident_expert e;
    memset(&e, 0, sizeof(e));
    e.model_map = model_map;
    e.layer = layer;
    e.expert = expert;
    e.gate_expert_bytes = gate_expert_bytes;
    e.down_expert_bytes = down_expert_bytes;
    e.gate_offset = gate_offset;
    e.up_offset = up_offset;
    e.down_offset = down_offset;
    e.base = (char *)base;
    e.gate = e.base;
    e.up = e.base + gate_expert_bytes;
    e.down = e.base + 2u * gate_expert_bytes;
    e.bytes = bytes;
    e.last_used = ++g_stream_resident_clock;
    e.pooled = pooled;
    g_stream_resident_experts.push_back(e);
    g_stream_resident_index[cuda_stream_resident_entry_key(e)] =
        g_stream_resident_experts.size() - 1u;
    g_stream_resident_bytes += bytes;
    if (cuda_stream_cache_stats_on()) {
        g_stream_cache_stats.allocs++;
        g_stream_cache_stats.alloc_bytes += bytes;
    }
    cuda_stream_cache_stats_note_resident();
    return (int)g_stream_resident_experts.size() - 1;
}

typedef struct cuda_stream_read_job {
    char *dst;
    uint64_t offset;
    uint64_t bytes;
    void *host_raw;
    void *host_buf;
    int ok;
    int uploaded;
    int errnum;
    int direct;
    int use_own_fd;
    int fd;
    int direct_fd;
    uint64_t file_size;
    uint64_t direct_align;
} cuda_stream_read_job;

struct cuda_stream_batch_selected_pending {
    int active;
    const void *model_map;
    uint32_t layer;
    uint32_t n_total_expert;
    uint32_t n_selected;
    uint32_t n_tokens;
    uint32_t n_unique;
    uint32_t resident_count;
    uint32_t missing_count;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    cuda_stream_read_job read_jobs[DS4_ROCM_STREAM_READ_MAX_JOBS];
    uint32_t read_job_count;
};

static cuda_stream_batch_selected_pending g_stream_batch_selected_pending;

struct cuda_stream_selected_pending {
    int active;
    const void *model_map;
    uint32_t layer;
    uint32_t n_total_expert;
    uint32_t n_selected;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    uint32_t resident_mask;
    uint32_t missing_mask;
    int32_t selected_ids[DS4_ROCM_N_EXPERT_USED];
    cuda_stream_read_job read_jobs[DS4_ROCM_N_EXPERT_USED * 3u];
    uint32_t read_job_count;
};

static cuda_stream_selected_pending g_stream_selected_pending;

static int cuda_q4k_linear_residency_intersection(
        const void *model_map, uint64_t offset, uint64_t bytes) {
    if (!model_map || bytes == 0u || offset > UINT64_MAX - bytes) return 1;
    for (const cuda_model_image &img : g_model_images) {
        if (img.host_base == model_map &&
            cuda_u64_ranges_overlap(offset, bytes,
                                    img.device_offset, img.size)) {
            return 1;
        }
    }
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_base == model_map &&
            cuda_u64_ranges_overlap(offset, bytes, r.offset, r.bytes)) {
            return 1;
        }
    }
    /* Stream caches contain routed-table fragments outside g_model_ranges.
     * Descriptor declaration is a startup operation, so any live cache for
     * this model makes the transition ambiguous and must be rejected. */
    if ((g_stream_selected_cache.loaded &&
         g_stream_selected_cache.model_map == model_map) ||
        (g_stream_batch_selected_cache.loaded &&
         g_stream_batch_selected_cache.model_map == model_map) ||
        (g_stream_selected_pending.active &&
         g_stream_selected_pending.model_map == model_map) ||
        (g_stream_batch_selected_pending.active &&
         g_stream_batch_selected_pending.model_map == model_map)) return 1;
    for (const cuda_stream_layer_expert_cache &c :
         g_stream_layer_expert_cache) {
        if (c.active && c.model_map == model_map) return 1;
    }
    for (const cuda_stream_resident_expert &e :
         g_stream_resident_experts) {
        if (e.model_map == model_map) return 1;
    }
    return 0;
}

typedef struct cuda_stream_read_profile_span {
    uint64_t offset;
    uint64_t end;
} cuda_stream_read_profile_span;

static uint64_t cuda_stream_read_profile_us(double seconds) {
    return seconds <= 0.0 ? 0u : (uint64_t)(seconds * 1000000.0 + 0.5);
}

static void cuda_stream_read_profile_coalesce(
        cuda_stream_read_profile_span *spans,
        uint32_t count,
        uint64_t max_gap,
        uint64_t *groups_out,
        uint64_t *extra_out) {
    uint64_t groups = 0;
    uint64_t extra = 0;
    uint64_t cur_end = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (spans[i].end <= spans[i].offset) continue;
        if (groups == 0) {
            groups = 1;
            cur_end = spans[i].end;
            continue;
        }
        if (spans[i].offset <= cur_end) {
            if (spans[i].end > cur_end) cur_end = spans[i].end;
            continue;
        }
        const uint64_t gap = spans[i].offset - cur_end;
        if (gap <= max_gap) {
            extra += gap;
            if (spans[i].end > cur_end) cur_end = spans[i].end;
            continue;
        }
        groups++;
        cur_end = spans[i].end;
    }
    if (groups_out) *groups_out = groups;
    if (extra_out) *extra_out = extra;
}

static void cuda_stream_read_profile_note_jobs(
        const cuda_stream_read_job *jobs,
        uint32_t count) {
    if (g_stream_read_profile_enabled != 1 || !jobs || count == 0) return;
    if (count > DS4_ROCM_STREAM_READ_MAX_JOBS) return;

    cuda_stream_read_profile_span spans[DS4_ROCM_STREAM_READ_MAX_JOBS];
    uint32_t n = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (jobs[i].bytes == 0) continue;
        spans[n].offset = jobs[i].offset;
        spans[n].end = jobs[i].offset > UINT64_MAX - jobs[i].bytes ?
            UINT64_MAX : jobs[i].offset + jobs[i].bytes;
        n++;
    }
    for (uint32_t i = 1; i < n; i++) {
        const cuda_stream_read_profile_span x = spans[i];
        uint32_t j = i;
        while (j != 0 && spans[j - 1].offset > x.offset) {
            spans[j] = spans[j - 1];
            j--;
        }
        spans[j] = x;
    }

    uint64_t exact_groups = 0;
    uint64_t gap1m_groups = 0;
    uint64_t gap4m_groups = 0;
    uint64_t exact_extra = 0;
    uint64_t gap1m_extra = 0;
    uint64_t gap4m_extra = 0;
    cuda_stream_read_profile_coalesce(spans, n, 0,
                                      &exact_groups, &exact_extra);
    cuda_stream_read_profile_coalesce(spans, n, 1ull * 1048576ull,
                                      &gap1m_groups, &gap1m_extra);
    cuda_stream_read_profile_coalesce(spans, n, 4ull * 1048576ull,
                                      &gap4m_groups, &gap4m_extra);

    (void)exact_extra;
    g_stream_read_profile_start_calls++;
    g_stream_read_profile_exact_groups += exact_groups;
    g_stream_read_profile_gap1m_groups += gap1m_groups;
    g_stream_read_profile_gap4m_groups += gap4m_groups;
    g_stream_read_profile_gap1m_extra += gap1m_extra;
    g_stream_read_profile_gap4m_extra += gap4m_extra;
}

static void cuda_stream_read_profile_print(void) {
    if (g_stream_read_profile_jobs == 0u &&
        g_stream_read_profile_wait_calls == 0u) {
        return;
    }
    const double jobs = g_stream_read_profile_jobs ?
        (double)g_stream_read_profile_jobs : 1.0;
    const double waits = g_stream_read_profile_wait_calls ?
        (double)g_stream_read_profile_wait_calls : 1.0;
    fprintf(stderr,
            DS4_GPU_LOG_PREFIX "stream read profile jobs=%llu bytes=%.2f GiB "
            "read=%.3f ms upload=%.3f ms wait=%.3f ms "
            "avg_job=%.3f ms avg_wait=%.3f ms\n",
            (unsigned long long)g_stream_read_profile_jobs,
            (double)g_stream_read_profile_bytes / 1073741824.0,
            (double)g_stream_read_profile_read_us / 1000.0,
            (double)g_stream_read_profile_upload_us / 1000.0,
            (double)g_stream_read_profile_wait_us / 1000.0,
            (double)(g_stream_read_profile_read_us +
                     g_stream_read_profile_upload_us) / 1000.0 / jobs,
            (double)g_stream_read_profile_wait_us / 1000.0 / waits);
    fprintf(stderr,
            DS4_GPU_LOG_PREFIX "stream read locality batches=%llu "
            "exact_jobs=%llu gap1m_jobs=%llu gap1m_extra=%.2f GiB "
            "gap4m_jobs=%llu gap4m_extra=%.2f GiB\n",
            (unsigned long long)g_stream_read_profile_start_calls,
            (unsigned long long)g_stream_read_profile_exact_groups,
            (unsigned long long)g_stream_read_profile_gap1m_groups,
            (double)g_stream_read_profile_gap1m_extra / 1073741824.0,
            (unsigned long long)g_stream_read_profile_gap4m_groups,
            (double)g_stream_read_profile_gap4m_extra / 1073741824.0);
}

static int cuda_stream_read_profile_enabled(void) {
#if defined(DS4_ENABLE_PROFILING) && DS4_ENABLE_PROFILING
    if (g_stream_read_profile_enabled < 0) {
        const char *env = getenv("DS4_ROCM_STREAM_READ_PROFILE");
        g_stream_read_profile_enabled =
            (env != NULL && env[0] != '\0' && strcmp(env, "0") != 0) ? 1 : 0;
        if (g_stream_read_profile_enabled &&
            !g_stream_read_profile_registered) {
            atexit(cuda_stream_read_profile_print);
            g_stream_read_profile_registered = 1;
        }
    }
    return g_stream_read_profile_enabled;
#else
    return 0;
#endif
}

static int cuda_stream_read_direct_disabled(void) {
    if (g_stream_read_direct_disabled < 0) {
        const char *env = getenv("DS4_ROCM_STREAM_NO_DIRECT");
        g_stream_read_direct_disabled =
            (env != NULL && env[0] != '\0' && strcmp(env, "0") != 0) ? 1 : 0;
    }
    return g_stream_read_direct_disabled;
}

static void cuda_stream_read_job_run(cuda_stream_read_job *job,
                                     void *stage,
                                     uint64_t stage_bytes) {
    if (!job) return;
    job->ok = 0;
    job->uploaded = 0;
    job->errnum = 0;
    job->direct = 0;
    const int read_fd = job->use_own_fd ? job->fd : g_model_fd;
    const int direct_fd = job->use_own_fd ? job->direct_fd : g_model_direct_fd;
    const uint64_t file_size = job->use_own_fd ? job->file_size : g_model_file_size;
    const uint64_t direct_align = job->use_own_fd ? job->direct_align : g_model_direct_align;
    if (!stage || job->bytes == 0 || read_fd < 0) {
        job->errnum = EINVAL;
        return;
    }
    job->host_raw = stage;
    job->host_buf = stage;
#if defined(__linux__) && defined(O_DIRECT)
    /*
     * Direct reads skip the page cache: no page allocation, no extra copy and
     * no POSIX_FADV_DONTNEED churn per streamed expert.  Alignment slack is
     * pre-reserved in the staging buffers.  On any direct-read failure fall
     * back to the buffered fd for this job only; the shared direct fd is left
     * alone so concurrent workers are unaffected.
     */
    if (!cuda_stream_read_direct_disabled() &&
        direct_fd >= 0 &&
        direct_align > 1 &&
        file_size != 0) {
        const uint64_t aligned_off =
            cuda_round_down(job->offset, direct_align);
        const uint64_t delta = job->offset - aligned_off;
        const uint64_t read_size =
            cuda_round_up(delta + job->bytes, direct_align);
        if (read_size <= stage_bytes &&
            aligned_off <= file_size &&
            read_size <= file_size - aligned_off &&
            cuda_pread_full(direct_fd, stage, read_size, aligned_off)) {
            job->host_buf = (char *)stage + delta;
            job->direct = 1;
            job->ok = 1;
            return;
        }
    }
#endif
    if (cuda_pread_full(read_fd, job->host_buf, job->bytes, job->offset)) {
        job->ok = 1;
    } else {
        job->errnum = errno ? errno : EIO;
    }
}

static int cuda_stream_read_job_upload(
        cuda_stream_read_job *job,
        cudaStream_t stream) {
    if (!job || !job->ok || !job->dst || !job->host_buf || !stream) {
        if (job) job->errnum = EINVAL;
        return 0;
    }
    cudaError_t err = cudaMemcpyAsync(job->dst,
                                      job->host_buf,
                                      (size_t)job->bytes,
                                      cudaMemcpyHostToDevice,
                                      stream);
    if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming read-worker upload failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        job->ok = 0;
        job->errnum = EIO;
        return 0;
    }
    job->uploaded = 1;
    if (!job->direct) {
        if (job->use_own_fd) {
#if defined(POSIX_FADV_DONTNEED)
            (void)posix_fadvise(job->fd, (off_t)job->offset,
                                (off_t)job->bytes, POSIX_FADV_DONTNEED);
#endif
        } else {
            cuda_model_drop_file_pages(job->offset, job->bytes);
        }
    }
    return 1;
}

static void *cuda_stream_read_worker(void *arg) {
    const uint32_t worker_id = arg ? *(const uint32_t *)arg : 0u;
    (void)cudaSetDevice(0);
    for (;;) {
        pthread_mutex_lock(&g_stream_read_mutex);
        while (!g_stream_read_pool_stop &&
               (!g_stream_read_active_jobs ||
                g_stream_read_active_next >= g_stream_read_active_count)) {
            pthread_cond_wait(&g_stream_read_work_cond, &g_stream_read_mutex);
        }
        if (g_stream_read_pool_stop) {
            pthread_mutex_unlock(&g_stream_read_mutex);
            break;
        }
        const uint32_t idx = g_stream_read_active_next++;
        cuda_stream_read_job *job = &g_stream_read_active_jobs[idx];
        void *stage = NULL;
        uint64_t stage_bytes = 0;
        if (worker_id < DS4_ROCM_STREAM_READ_WORKERS) {
            stage = g_stream_read_stage_raw[worker_id];
            stage_bytes = g_stream_read_stage_bytes[worker_id];
        }
        pthread_mutex_unlock(&g_stream_read_mutex);

        const int profile = g_stream_read_profile_enabled == 1;
        const double read_t0 = profile ? cuda_wall_sec() : 0.0;
        cuda_stream_read_job_run(job, stage, stage_bytes);
        uint64_t read_us = 0;
        uint64_t upload_us = 0;
        if (profile) {
            read_us = cuda_stream_read_profile_us(cuda_wall_sec() - read_t0);
        }
        if (job->ok) {
            const double upload_t0 = profile ? cuda_wall_sec() : 0.0;
            (void)cuda_stream_read_job_upload(
                    job,
                    worker_id < DS4_ROCM_STREAM_READ_WORKERS ?
                        g_stream_read_upload_streams[worker_id] : NULL);
            if (profile) {
                upload_us =
                    cuda_stream_read_profile_us(cuda_wall_sec() - upload_t0);
            }
        }

        pthread_mutex_lock(&g_stream_read_mutex);
        if (profile) {
            g_stream_read_profile_jobs++;
            g_stream_read_profile_bytes += job->bytes;
            g_stream_read_profile_read_us += read_us;
            g_stream_read_profile_upload_us += upload_us;
        }
        if (!job->ok) g_stream_read_active_ok = 0;
        g_stream_read_active_done++;
        if (g_stream_read_active_done >= g_stream_read_active_count) {
            pthread_cond_signal(&g_stream_read_done_cond);
        }
        pthread_mutex_unlock(&g_stream_read_mutex);
    }
    return NULL;
}

static uint32_t cuda_stream_read_worker_count(void) {
    const char *env = getenv("DS4_ROCM_STREAM_READ_WORKERS");
    if (env && env[0]) {
        char *end = NULL;
        errno = 0;
        unsigned long v = strtoul(env, &end, 10);
        if (end != env && errno == 0) {
            if (v == 0) return 1u;
            if (v > DS4_ROCM_STREAM_READ_WORKERS) {
                return DS4_ROCM_STREAM_READ_WORKERS;
            }
            return (uint32_t)v;
        }
    }
    return DS4_ROCM_STREAM_READ_DEFAULT_WORKERS;
}

static void cuda_stream_read_upload_streams_destroy(void) {
    for (uint32_t i = 0; i < DS4_ROCM_STREAM_READ_WORKERS; i++) {
        if (g_stream_read_upload_streams[i]) {
            (void)cudaStreamDestroy(g_stream_read_upload_streams[i]);
            g_stream_read_upload_streams[i] = NULL;
        }
    }
}

static int cuda_stream_read_upload_streams_ensure(void) {
    const uint32_t workers = cuda_stream_read_worker_count();
    for (uint32_t i = 0; i < workers; i++) {
        if (g_stream_read_upload_streams[i]) continue;
        cudaError_t err = cudaStreamCreateWithFlags(
                &g_stream_read_upload_streams[i],
                cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming read upload stream creation failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            cuda_stream_read_upload_streams_destroy();
            return 0;
        }
    }
    return 1;
}

static int cuda_stream_read_pool_ensure(void) {
    if (g_stream_read_pool_started) return 1;
    pthread_mutex_lock(&g_stream_read_mutex);
    if (g_stream_read_pool_started) {
        pthread_mutex_unlock(&g_stream_read_mutex);
        return 1;
    }
    g_stream_read_pool_stop = 0;
    g_stream_read_active_jobs = NULL;
    g_stream_read_active_count = 0;
    g_stream_read_active_next = 0;
    g_stream_read_active_done = 0;
    g_stream_read_active_ok = 1;
    g_stream_read_active_owner_set = 0;
    g_stream_read_pool_workers = cuda_stream_read_worker_count();
    (void)cuda_stream_read_profile_enabled();
    if (!cuda_stream_read_upload_streams_ensure()) {
        pthread_mutex_unlock(&g_stream_read_mutex);
        return 0;
    }
    for (uint32_t i = 0; i < g_stream_read_pool_workers; i++) {
        g_stream_read_thread_ids[i] = i;
        const int rc = pthread_create(&g_stream_read_threads[i],
                                      NULL,
                                      cuda_stream_read_worker,
                                      &g_stream_read_thread_ids[i]);
        if (rc != 0) {
            g_stream_read_pool_stop = 1;
            pthread_cond_broadcast(&g_stream_read_work_cond);
            pthread_mutex_unlock(&g_stream_read_mutex);
            for (uint32_t j = 0; j < i; j++) {
                (void)pthread_join(g_stream_read_threads[j], NULL);
            }
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming read worker creation failed: %s\n",
                    strerror(rc));
            cuda_stream_read_upload_streams_destroy();
            return 0;
        }
    }
    g_stream_read_pool_started = 1;
    pthread_mutex_unlock(&g_stream_read_mutex);
    return 1;
}

static void cuda_stream_read_pool_shutdown(void) {
    if (!g_stream_read_pool_started) return;
    pthread_mutex_lock(&g_stream_read_mutex);
    g_stream_read_pool_stop = 1;
    pthread_cond_broadcast(&g_stream_read_work_cond);
    pthread_mutex_unlock(&g_stream_read_mutex);
    for (uint32_t i = 0; i < g_stream_read_pool_workers; i++) {
        (void)pthread_join(g_stream_read_threads[i], NULL);
    }
    pthread_mutex_lock(&g_stream_read_mutex);
    g_stream_read_pool_started = 0;
    g_stream_read_pool_workers = 0;
    g_stream_read_pool_stop = 0;
    g_stream_read_active_jobs = NULL;
    g_stream_read_active_count = 0;
    g_stream_read_active_next = 0;
    g_stream_read_active_done = 0;
    g_stream_read_active_ok = 1;
    g_stream_read_active_owner_set = 0;
    pthread_mutex_unlock(&g_stream_read_mutex);
}

static int cuda_stream_read_jobs_prepare(cuda_stream_read_job *jobs, uint32_t count) {
    if (!jobs || count == 0) return 1;
    if (count > DS4_ROCM_STREAM_READ_MAX_JOBS) return 0;

    uint64_t max_bytes = 0;
    for (uint32_t i = 0; i < count; i++) {
        jobs[i].ok = 0;
        jobs[i].uploaded = 0;
        jobs[i].errnum = 0;
        jobs[i].direct = 0;
        jobs[i].host_raw = NULL;
        jobs[i].host_buf = NULL;
        if (jobs[i].bytes > max_bytes) max_bytes = jobs[i].bytes;
    }
    uint64_t max_align = g_model_direct_align;
    for (uint32_t i = 0; i < count; i++) {
        if (jobs[i].use_own_fd && jobs[i].direct_align > max_align) {
            max_align = jobs[i].direct_align;
        }
    }
    /* Slack so direct reads can align their offset and length. */
    if (max_align > 1u && max_bytes <= UINT64_MAX - 2u * max_align) {
        max_bytes += 2u * max_align;
    }

    const uint32_t workers = g_stream_read_pool_started ?
        g_stream_read_pool_workers : cuda_stream_read_worker_count();
    for (uint32_t i = 0; i < workers; i++) {
        if (g_stream_read_stage_bytes[i] < max_bytes) {
            if (g_stream_read_stage_raw[i]) {
                (void)cudaFreeHost(g_stream_read_stage_raw[i]);
                g_stream_read_stage_raw[i] = NULL;
                g_stream_read_stage_bytes[i] = 0;
            }
            cudaError_t err = cudaMallocHost(&g_stream_read_stage_raw[i],
                                             (size_t)max_bytes);
            if (err != cudaSuccess) {
                fprintf(stderr,
                        DS4_GPU_LOG_PREFIX "streaming read pinned allocation failed "
                        "(%.2f MiB): %s\n",
                        (double)max_bytes / 1048576.0,
                        cudaGetErrorString(err));
                (void)cudaGetLastError();
                return 0;
            }
            g_stream_read_stage_bytes[i] = max_bytes;
        }
    }
    return 1;
}

static int cuda_stream_read_jobs_start(cuda_stream_read_job *jobs, uint32_t count) {
    if (!jobs || count == 0) return 1;
    if (!cuda_stream_read_pool_ensure()) return 0;
    const pthread_t self = pthread_self();
    pthread_mutex_lock(&g_stream_read_mutex);
    while (g_stream_read_active_jobs != NULL) {
        if (g_stream_read_active_owner_set &&
            pthread_equal(g_stream_read_active_owner, self)) {
            pthread_mutex_unlock(&g_stream_read_mutex);
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming read pool already has active work for this thread\n");
            return 0;
        }
        pthread_cond_wait(&g_stream_read_done_cond, &g_stream_read_mutex);
    }
    if (!cuda_stream_read_jobs_prepare(jobs, count)) {
        pthread_mutex_unlock(&g_stream_read_mutex);
        return 0;
    }
    cuda_stream_read_profile_note_jobs(jobs, count);
    g_stream_read_active_jobs = jobs;
    g_stream_read_active_count = count;
    g_stream_read_active_next = 0;
    g_stream_read_active_done = 0;
    g_stream_read_active_ok = 1;
    g_stream_read_active_owner = self;
    g_stream_read_active_owner_set = 1;
    pthread_cond_broadcast(&g_stream_read_work_cond);
    pthread_mutex_unlock(&g_stream_read_mutex);
    return 1;
}

static int cuda_stream_read_jobs_wait(cuda_stream_read_job *jobs, uint32_t count) {
    if (!jobs || count == 0) return 1;
    const int profile = g_stream_read_profile_enabled == 1;
    const double wait_t0 = profile ? cuda_wall_sec() : 0.0;
    pthread_mutex_lock(&g_stream_read_mutex);
    if (g_stream_read_active_jobs != jobs) {
        pthread_mutex_unlock(&g_stream_read_mutex);
        fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming read wait received inactive job set\n");
        return 0;
    }
    while (g_stream_read_active_done < g_stream_read_active_count) {
        pthread_cond_wait(&g_stream_read_done_cond, &g_stream_read_mutex);
    }
    const int pool_ok = g_stream_read_active_ok;
    g_stream_read_active_jobs = NULL;
    g_stream_read_active_count = 0;
    g_stream_read_active_next = 0;
    g_stream_read_active_done = 0;
    g_stream_read_active_ok = 1;
    g_stream_read_active_owner_set = 0;
    if (profile) {
        g_stream_read_profile_wait_calls++;
        g_stream_read_profile_wait_us +=
            cuda_stream_read_profile_us(cuda_wall_sec() - wait_t0);
    }
    pthread_cond_broadcast(&g_stream_read_done_cond);
    pthread_mutex_unlock(&g_stream_read_mutex);

    int ok = 1;
    if (!pool_ok) ok = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (!jobs[i].ok) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming read failed at offset %.2f GiB "
                    "size %.2f MiB: %s\n",
                    (double)jobs[i].offset / 1073741824.0,
                    (double)jobs[i].bytes / 1048576.0,
                    strerror(jobs[i].errnum ? jobs[i].errnum : EIO));
            ok = 0;
        }
    }
    return ok;
}

static int cuda_stream_read_jobs_parallel(cuda_stream_read_job *jobs, uint32_t count) {
    if (!jobs || count == 0) return 1;
    return cuda_stream_read_jobs_start(jobs, count) &&
           cuda_stream_read_jobs_wait(jobs, count);
}

static void cuda_stream_read_jobs_free(cuda_stream_read_job *jobs, uint32_t count) {
    if (!jobs) return;
    for (uint32_t i = 0; i < count; i++) {
        jobs[i].host_raw = NULL;
        jobs[i].host_buf = NULL;
        jobs[i].uploaded = 0;
        jobs[i].direct = 0;
    }
}

static int cuda_stream_selected_upload_read_jobs(
        cuda_stream_read_job *jobs,
        uint32_t count);

static int cuda_stream_batch_selected_pending_matches(
        const void *model_map,
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint32_t n_tokens,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    return g_stream_batch_selected_pending.active &&
           g_stream_batch_selected_pending.model_map == model_map &&
           g_stream_batch_selected_pending.layer == layer &&
           g_stream_batch_selected_pending.n_total_expert == n_total_expert &&
           g_stream_batch_selected_pending.n_selected == n_selected &&
           g_stream_batch_selected_pending.n_tokens == n_tokens &&
           g_stream_batch_selected_pending.gate_offset == gate_offset &&
           g_stream_batch_selected_pending.up_offset == up_offset &&
           g_stream_batch_selected_pending.down_offset == down_offset &&
           g_stream_batch_selected_pending.gate_expert_bytes == gate_expert_bytes &&
           g_stream_batch_selected_pending.down_expert_bytes == down_expert_bytes;
}

static int cuda_stream_batch_selected_apply_split(
        const void *model_map,
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint32_t n_tokens,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        const ds4_gpu_tensor **selected_exec,
        const char ***resident_gate_ptrs,
        const char ***resident_up_ptrs,
        const char ***missing_gate_ptrs,
        const char ***missing_up_ptrs,
        const char ***down_ptrs,
        const uint8_t **pair_missing,
        uint32_t *resident_count,
        uint32_t *missing_count,
        uint32_t *unique_out) {
    if (!selected_exec || !resident_gate_ptrs || !resident_up_ptrs ||
        !missing_gate_ptrs || !missing_up_ptrs || !down_ptrs ||
        !pair_missing ||
        !resident_count || !missing_count || !unique_out ||
        !cuda_stream_batch_selected_pending_matches(model_map,
                                                    layer,
                                                    n_total_expert,
                                                    n_selected,
                                                    n_tokens,
                                                    gate_offset,
                                                    up_offset,
                                                    down_offset,
                                                    gate_expert_bytes,
                                                    down_expert_bytes) ||
        !g_stream_batch_selected_cache.selected_ids ||
        !g_stream_batch_selected_cache.resident_gate_ptrs ||
        !g_stream_batch_selected_cache.resident_up_ptrs ||
        !g_stream_batch_selected_cache.missing_gate_ptrs ||
        !g_stream_batch_selected_cache.missing_up_ptrs ||
        !g_stream_batch_selected_cache.down_ptrs ||
        !g_stream_batch_selected_cache.pair_missing) {
        return 0;
    }
    *selected_exec = &g_stream_batch_selected_cache.selected_tensor;
    *resident_gate_ptrs = g_stream_batch_selected_cache.resident_gate_ptrs;
    *resident_up_ptrs = g_stream_batch_selected_cache.resident_up_ptrs;
    *missing_gate_ptrs = g_stream_batch_selected_cache.missing_gate_ptrs;
    *missing_up_ptrs = g_stream_batch_selected_cache.missing_up_ptrs;
    *down_ptrs = g_stream_batch_selected_cache.down_ptrs;
    *pair_missing = g_stream_batch_selected_cache.pair_missing;
    *resident_count = g_stream_batch_selected_pending.resident_count;
    *missing_count = g_stream_batch_selected_pending.missing_count;
    *unique_out = g_stream_batch_selected_pending.n_unique;
    return 1;
}

static int cuda_stream_batch_selected_finish_pending_missing(void) {
    if (!g_stream_batch_selected_pending.active) return 1;
    const uint32_t read_job_count =
        g_stream_batch_selected_pending.read_job_count;
    if (!cuda_stream_read_jobs_wait(g_stream_batch_selected_pending.read_jobs,
                                    read_job_count) ||
        !cuda_stream_selected_upload_read_jobs(
                g_stream_batch_selected_pending.read_jobs,
                read_job_count)) {
        cuda_stream_read_jobs_free(g_stream_batch_selected_pending.read_jobs,
                                   read_job_count);
        memset(&g_stream_batch_selected_pending, 0,
               sizeof(g_stream_batch_selected_pending));
        cuda_stream_resident_cache_release();
        return 0;
    }
    cuda_stream_read_jobs_free(g_stream_batch_selected_pending.read_jobs,
                               read_job_count);
    g_stream_batch_selected_cache.loaded = 0;
    memset(&g_stream_batch_selected_pending, 0,
           sizeof(g_stream_batch_selected_pending));
    return 1;
}

static void cuda_stream_batch_selected_abort_pending(void) {
    if (!g_stream_batch_selected_pending.active) return;
    const uint32_t read_job_count =
        g_stream_batch_selected_pending.read_job_count;
    (void)cuda_stream_read_jobs_wait(g_stream_batch_selected_pending.read_jobs,
                                     read_job_count);
    cuda_stream_read_jobs_free(g_stream_batch_selected_pending.read_jobs,
                               read_job_count);
    memset(&g_stream_batch_selected_pending, 0,
           sizeof(g_stream_batch_selected_pending));
}

static void cuda_stream_selected_abort_pending(void) {
    if (!g_stream_selected_pending.active) return;
    const uint32_t read_job_count = g_stream_selected_pending.read_job_count;
    if (read_job_count != 0) {
        (void)cuda_stream_read_jobs_wait(g_stream_selected_pending.read_jobs,
                                         read_job_count);
        cuda_stream_read_jobs_free(g_stream_selected_pending.read_jobs,
                                   read_job_count);
    }
    memset(&g_stream_selected_pending, 0, sizeof(g_stream_selected_pending));
}

static int cuda_stream_selected_upload_read_jobs(
        cuda_stream_read_job *jobs,
        uint32_t count) {
    if (!jobs || count == 0) return 1;
    int need_upload = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (!jobs[i].uploaded) {
            need_upload = 1;
            break;
        }
    }
    if (!need_upload) return 1;
    if (!cuda_stream_selected_ensure_stream()) return 0;
    for (uint32_t i = 0; i < count; i++) {
        if (jobs[i].uploaded) continue;
        cudaError_t err = cudaMemcpyAsync(jobs[i].dst,
                                          jobs[i].host_buf,
                                          (size_t)jobs[i].bytes,
                                          cudaMemcpyHostToDevice,
                                          g_stream_selected_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming selected cached upload failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        if (!jobs[i].direct) {
            cuda_model_drop_file_pages(jobs[i].offset, jobs[i].bytes);
        }
    }
    cudaError_t err = cudaStreamSynchronize(g_stream_selected_upload_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming selected upload sync failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

static int cuda_stream_flush_read_jobs(
        cuda_stream_read_job *jobs,
        uint32_t *count) {
    if (!count || *count == 0) return 1;
    if (!cuda_stream_read_jobs_parallel(jobs, *count) ||
        !cuda_stream_selected_upload_read_jobs(jobs, *count)) {
        cuda_stream_read_jobs_free(jobs, *count);
        *count = 0;
        return 0;
    }
    cuda_stream_read_jobs_free(jobs, *count);
    *count = 0;
    return 1;
}

static int cuda_stream_layer_expert_cache_apply(
        const void *model_map,
        uint32_t layer,
        uint32_t n_total_expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        const char **gate_w,
        const char **up_w,
        const char **down_w) {
    if (!g_ssd_streaming_mode || !gate_w || !up_w || !down_w) return 0;
    if (cuda_q4k_packed_slice_refuse_routed_tables(
            model_map, n_total_expert, gate_offset, up_offset, down_offset,
            gate_expert_bytes, down_expert_bytes,
            "streaming full-layer expert cache apply")) return 0;
    for (uint32_t i = 0; i < 2u; i++) {
        const cuda_stream_layer_expert_cache &c = g_stream_layer_expert_cache[i];
        if (c.active &&
            c.model_map == model_map &&
            c.layer == layer &&
            c.n_total_expert == n_total_expert &&
            c.gate_offset == gate_offset &&
            c.up_offset == up_offset &&
            c.down_offset == down_offset &&
            c.gate_expert_bytes == gate_expert_bytes &&
            c.down_expert_bytes == down_expert_bytes &&
            c.gate && c.up && c.down) {
            *gate_w = c.gate;
            *up_w = c.up;
            *down_w = c.down;
            return 1;
        }
    }
    return 0;
}

static int cuda_stream_layer_expert_cache_load(
        const void *model_map,
        uint64_t model_size,
        uint32_t layer,
        uint32_t n_total_expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!g_ssd_streaming_mode ||
        !model_map ||
        model_size == 0 ||
        n_total_expert == 0 ||
        n_total_expert > DS4_ROCM_STREAM_READ_MAX_JOBS / 3u ||
        gate_expert_bytes == 0 ||
        down_expert_bytes == 0 ||
        g_model_fd < 0 ||
        (g_model_fd_host_base != NULL && model_map != g_model_fd_host_base)) {
        return 0;
    }

    uint64_t gate_bytes = 0;
    uint64_t down_bytes = 0;
    uint64_t gate_pair_bytes = 0;
    uint64_t total_bytes = 0;
    if (!cuda_u64_mul_checked(n_total_expert, gate_expert_bytes, &gate_bytes) ||
        !cuda_u64_mul_checked(n_total_expert, down_expert_bytes, &down_bytes) ||
        !cuda_u64_mul_checked(2u, gate_bytes, &gate_pair_bytes) ||
        gate_pair_bytes > UINT64_MAX - down_bytes) {
        return 0;
    }
    total_bytes = gate_pair_bytes + down_bytes;
    if (cuda_q4k_packed_slice_refuse_routed_tables(
            model_map, n_total_expert, gate_offset, up_offset, down_offset,
            gate_expert_bytes, down_expert_bytes,
            "streaming full-layer expert cache load")) return 0;
    if (cuda_stream_cache_stats_on()) {
        g_stream_cache_stats.layer_loads++;
        g_stream_cache_stats.layer_load_bytes += total_bytes;
    }

    if (gate_offset > model_size ||
        up_offset > model_size ||
        down_offset > model_size ||
        gate_bytes > model_size - gate_offset ||
        gate_bytes > model_size - up_offset ||
        down_bytes > model_size - down_offset) {
        return 0;
    }

    cuda_stream_layer_expert_cache &slot =
        g_stream_layer_expert_cache[layer & 1u];
    slot.active = 0;
    if (slot.capacity < total_bytes) {
        if (slot.base) {
            (void)cudaFree(slot.base);
            memset(&slot, 0, sizeof(slot));
        }
        if (cuda_stream_cache_stats_on() &&
            !g_stream_resident_experts.empty()) {
            g_stream_cache_stats.layer_resident_flushes++;
        }
        cuda_stream_resident_cache_release();
        void *base = NULL;
        cudaError_t err = cudaMalloc(&base, (size_t)total_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming full-layer expert cache allocation "
                    "failed for layer=%u (%.2f GiB): %s\n",
                    layer,
                    (double)total_bytes / 1073741824.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        slot.base = (char *)base;
        slot.capacity = total_bytes;
    }

    slot.bytes = total_bytes;
    slot.gate = slot.base;
    slot.up = slot.base + gate_bytes;
    slot.down = slot.base + gate_pair_bytes;

    const uint64_t read_chunk = 32ull * 1048576ull;
    const uint64_t gate_chunks =
        (gate_bytes + read_chunk - 1u) / read_chunk;
    const uint64_t down_chunks =
        (down_bytes + read_chunk - 1u) / read_chunk;
    const uint64_t read_job_count64 = gate_chunks * 2u + down_chunks;
    if (read_job_count64 == 0 ||
        read_job_count64 > DS4_ROCM_STREAM_READ_MAX_JOBS ||
        read_job_count64 > UINT32_MAX) {
        return 0;
    }
    const uint32_t read_job_count = (uint32_t)read_job_count64;
    cuda_stream_read_job *jobs =
        (cuda_stream_read_job *)calloc((size_t)read_job_count, sizeof(jobs[0]));
    if (!jobs) return 0;

    int ok = 1;
    uint32_t j = 0;
    for (uint64_t off = 0; off < gate_bytes; off += read_chunk) {
        const uint64_t n = gate_bytes - off < read_chunk ? gate_bytes - off : read_chunk;
        jobs[j++] = {slot.gate + off, gate_offset + off, n, NULL, NULL, 0, 0};
    }
    for (uint64_t off = 0; off < gate_bytes; off += read_chunk) {
        const uint64_t n = gate_bytes - off < read_chunk ? gate_bytes - off : read_chunk;
        jobs[j++] = {slot.up + off, up_offset + off, n, NULL, NULL, 0, 0};
    }
    for (uint64_t off = 0; off < down_bytes; off += read_chunk) {
        const uint64_t n = down_bytes - off < read_chunk ? down_bytes - off : read_chunk;
        jobs[j++] = {slot.down + off, down_offset + off, n, NULL, NULL, 0, 0};
    }
    if (j != read_job_count ||
        !cuda_stream_read_jobs_parallel(jobs, read_job_count)) {
        ok = 0;
    }
    cuda_stream_read_jobs_free(jobs, read_job_count);
    free(jobs);
    if (!ok) return 0;

    slot.active = 1;
    slot.model_map = model_map;
    slot.layer = layer;
    slot.n_total_expert = n_total_expert;
    slot.gate_offset = gate_offset;
    slot.up_offset = up_offset;
    slot.down_offset = down_offset;
    slot.gate_expert_bytes = gate_expert_bytes;
    slot.down_expert_bytes = down_expert_bytes;
    return 1;
}

static int cuda_stream_resident_seed_experts(
        const void *model_map,
        uint64_t model_size,
        uint32_t layer,
        const int32_t *expert_ids,
        const uint32_t *expert_priorities,
        uint32_t n_experts,
        uint32_t n_total_expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!g_ssd_streaming_mode) return 1;
    if (!model_map || !expert_ids || n_experts == 0 ||
        n_total_expert == 0 ||
        n_total_expert > DS4_ROCM_MAX_N_EXPERT ||
        gate_expert_bytes == 0 ||
        down_expert_bytes == 0) {
        return 0;
    }
    if (g_stream_expert_cache_budget == 0) return 1;

    uint64_t gate_bytes = 0;
    uint64_t down_bytes = 0;
    if (!cuda_u64_mul_checked(n_total_expert, gate_expert_bytes, &gate_bytes) ||
        !cuda_u64_mul_checked(n_total_expert, down_expert_bytes, &down_bytes) ||
        gate_offset > model_size ||
        up_offset > model_size ||
        down_offset > model_size ||
        gate_bytes > model_size - gate_offset ||
        gate_bytes > model_size - up_offset ||
        down_bytes > model_size - down_offset) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming hotlist seed expert range outside model map\n");
        return 0;
    }
    if (cuda_q4k_packed_slice_refuse_routed_tables(
            model_map, n_total_expert, gate_offset, up_offset, down_offset,
            gate_expert_bytes, down_expert_bytes,
            "streaming resident expert seed")) return 0;

    uint32_t seed_cap = n_experts < g_stream_expert_cache_budget ?
        n_experts : g_stream_expert_cache_budget;
    if (seed_cap > DS4_ROCM_MAX_N_EXPERT) seed_cap = DS4_ROCM_MAX_N_EXPERT;
    if (seed_cap == 0) return 1;

    bool seen[DS4_ROCM_MAX_N_EXPERT] = {0};
    uint32_t best_index[DS4_ROCM_MAX_N_EXPERT];
    uint32_t best_priority[DS4_ROCM_MAX_N_EXPERT];
    for (uint32_t i = 0; i < n_experts; i++) {
        const int32_t expert_i = expert_ids[i];
        if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming hotlist seed expert id %d outside 0..%u "
                    "(layer=%u)\n",
                    expert_i,
                    n_total_expert,
                    layer);
            return 0;
        }
        const uint32_t expert = (uint32_t)expert_i;
        const uint32_t priority =
            expert_priorities ? expert_priorities[i] : (n_experts - i);
        if (!seen[expert] || priority > best_priority[expert]) {
            seen[expert] = true;
            best_index[expert] = i;
            best_priority[expert] = priority;
        }
    }

    uint32_t chosen_indices[DS4_ROCM_MAX_N_EXPERT];
    uint32_t chosen_priorities[DS4_ROCM_MAX_N_EXPERT];
    uint32_t chosen_count = 0;
    for (uint32_t expert = 0; expert < n_total_expert; expert++) {
        if (!seen[expert]) continue;
        const uint32_t priority = best_priority[expert];
        uint32_t pos = 0;
        while (pos < chosen_count && priority <= chosen_priorities[pos]) {
            pos++;
        }
        if (chosen_count < seed_cap) {
            for (uint32_t j = chosen_count; j > pos; j--) {
                chosen_indices[j] = chosen_indices[j - 1u];
                chosen_priorities[j] = chosen_priorities[j - 1u];
            }
            chosen_indices[pos] = best_index[expert];
            chosen_priorities[pos] = priority;
            chosen_count++;
        } else if (pos < chosen_count) {
            for (uint32_t j = chosen_count - 1u; j > pos; j--) {
                chosen_indices[j] = chosen_indices[j - 1u];
                chosen_priorities[j] = chosen_priorities[j - 1u];
            }
            chosen_indices[pos] = best_index[expert];
            chosen_priorities[pos] = priority;
        }
    }
    if (chosen_count == 0) return 1;

    int32_t protected_ids[DS4_ROCM_MAX_N_EXPERT];
    for (uint32_t i = 0; i < chosen_count; i++) {
        protected_ids[i] = expert_ids[chosen_indices[i]];
    }
    if (cuda_stream_cache_stats_on()) {
        g_stream_cache_stats.seed_calls++;
        g_stream_cache_stats.seed_unique += chosen_count;
    }

    const int use_fd =
        g_model_fd >= 0 &&
        (g_model_fd_host_base == NULL || model_map == g_model_fd_host_base);
    if (!use_fd && !cuda_stream_selected_ensure_stream()) return 1;

    cuda_stream_read_job read_jobs[DS4_ROCM_STREAM_READ_MAX_JOBS];
    memset(read_jobs, 0, sizeof(read_jobs));
    uint32_t read_job_count = 0;
    int ok = 1;
    uint32_t loaded = 0;

    for (uint32_t ri = 0; ok && ri < chosen_count; ri++) {
        const uint32_t chosen_i = chosen_indices[chosen_count - 1u - ri];
        const int32_t expert_i = expert_ids[chosen_i];
        int idx = cuda_stream_resident_find(model_map,
                                            layer,
                                            expert_i,
                                            gate_offset,
                                            up_offset,
                                            down_offset,
                                            gate_expert_bytes,
                                            down_expert_bytes);
        if (idx >= 0) {
            g_stream_resident_experts[(size_t)idx].last_used =
                ++g_stream_resident_clock;
            loaded++;
            continue;
        }

        idx = cuda_stream_resident_alloc(model_map,
                                         layer,
                                         expert_i,
                                         protected_ids,
                                         chosen_count,
                                         gate_offset,
                                         up_offset,
                                         down_offset,
                                         gate_expert_bytes,
                                         down_expert_bytes);
        if (idx < 0) {
            ok = 0;
            break;
        }

        const uint64_t expert = (uint64_t)(uint32_t)expert_i;
        uint64_t gate_rel = 0;
        uint64_t down_rel = 0;
        if (!cuda_u64_mul_checked(expert, gate_expert_bytes, &gate_rel) ||
            !cuda_u64_mul_checked(expert, down_expert_bytes, &down_rel)) {
            ok = 0;
            break;
        }

        cuda_stream_resident_expert &entry =
            g_stream_resident_experts[(size_t)idx];
        if (use_fd) {
            if (read_job_count + 3u > DS4_ROCM_STREAM_READ_MAX_JOBS) {
                ok = 0;
                break;
            }
            read_jobs[read_job_count++] =
                {entry.gate, gate_offset + gate_rel, gate_expert_bytes,
                 NULL, NULL, 0, 0};
            read_jobs[read_job_count++] =
                {entry.up, up_offset + gate_rel, gate_expert_bytes,
                 NULL, NULL, 0, 0};
            read_jobs[read_job_count++] =
                {entry.down, down_offset + down_rel, down_expert_bytes,
                 NULL, NULL, 0, 0};
        } else {
            cudaError_t err = cudaMemcpyAsync(entry.gate,
                                              (const char *)model_map + gate_offset + gate_rel,
                                              (size_t)gate_expert_bytes,
                                              cudaMemcpyHostToDevice,
                                              g_stream_selected_upload_stream);
            if (err == cudaSuccess) {
                err = cudaMemcpyAsync(entry.up,
                                      (const char *)model_map + up_offset + gate_rel,
                                      (size_t)gate_expert_bytes,
                                      cudaMemcpyHostToDevice,
                                      g_stream_selected_upload_stream);
            }
            if (err == cudaSuccess) {
                err = cudaMemcpyAsync(entry.down,
                                      (const char *)model_map + down_offset + down_rel,
                                      (size_t)down_expert_bytes,
                                      cudaMemcpyHostToDevice,
                                      g_stream_selected_upload_stream);
            }
            if (err != cudaSuccess) {
                fprintf(stderr,
                        DS4_GPU_LOG_PREFIX "streaming hotlist seed upload failed: %s\n",
                        cudaGetErrorString(err));
                (void)cudaGetLastError();
                ok = 0;
                break;
            }
        }
        loaded++;
    }

    if (ok && use_fd && read_job_count != 0) {
        ok = cuda_stream_read_jobs_parallel(read_jobs, read_job_count) &&
             cuda_stream_selected_upload_read_jobs(read_jobs, read_job_count);
        cuda_stream_read_jobs_free(read_jobs, read_job_count);
    } else if (ok && !use_fd) {
        cudaError_t err = cudaStreamSynchronize(g_stream_selected_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming hotlist seed upload sync failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            ok = 0;
        }
    }

    if (!ok) {
        cuda_stream_read_jobs_free(read_jobs, read_job_count);
        cuda_stream_resident_cache_release();
        if (getenv("DS4_ROCM_STREAMING_EXPERT_CACHE_VERBOSE") != NULL) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming hotlist seed skipped after partial load "
                    "(layer=%u requested=%u loaded=%u)\n",
                    layer,
                    n_experts,
                    loaded);
        }
        return 1;
    }

    if (getenv("DS4_ROCM_STREAMING_EXPERT_CACHE_VERBOSE") != NULL) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming hotlist seeded layer=%u requested=%u cached=%u budget=%u\n",
                layer,
                n_experts,
                chosen_count,
                g_stream_expert_cache_budget);
    }
    return 1;
}

static void cuda_stream_selected_cache_header(
        const void *model_map,
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t n_selected,
        const int32_t *selected_ids,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    g_stream_selected_cache.model_map = model_map;
    g_stream_selected_cache.layer = layer;
    g_stream_selected_cache.n_total_expert = n_total_expert;
    g_stream_selected_cache.n_selected = n_selected;
    g_stream_selected_cache.gate_expert_bytes = gate_expert_bytes;
    g_stream_selected_cache.down_expert_bytes = down_expert_bytes;
    for (uint32_t i = 0; i < n_selected; i++) {
        g_stream_selected_cache.selected_ids[i] = selected_ids[i];
    }
}

static int cuda_stream_selected_compact_mask(
        const void *model_map,
        uint32_t layer,
        const int32_t *selected_ids,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint32_t mask) {
    (void)n_total_expert;
    if (mask == 0) return 1;
    if (!selected_ids || !cuda_stream_selected_ensure_stream()) return 0;
    cudaError_t err = cudaSuccess;
    for (uint32_t i = 0; i < n_selected; i++) {
        if ((mask & (1u << i)) == 0) continue;
        int idx = cuda_stream_resident_find(model_map,
                                            layer,
                                            selected_ids[i],
                                            gate_offset,
                                            up_offset,
                                            down_offset,
                                            gate_expert_bytes,
                                            down_expert_bytes);
        if (idx < 0) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming selected resident expert missing during compact\n");
            return 0;
        }
        cuda_stream_resident_expert &entry =
            g_stream_resident_experts[(size_t)idx];
        entry.last_used = ++g_stream_resident_clock;
        err = cudaMemcpyAsync(g_stream_selected_cache.gate +
                                  (uint64_t)i * gate_expert_bytes,
                              entry.gate,
                              (size_t)gate_expert_bytes,
                              cudaMemcpyDeviceToDevice,
                              g_stream_selected_upload_stream);
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_selected_cache.up +
                                      (uint64_t)i * gate_expert_bytes,
                                  entry.up,
                                  (size_t)gate_expert_bytes,
                                  cudaMemcpyDeviceToDevice,
                                  g_stream_selected_upload_stream);
        }
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_selected_cache.down +
                                      (uint64_t)i * down_expert_bytes,
                                  entry.down,
                                  (size_t)down_expert_bytes,
                                  cudaMemcpyDeviceToDevice,
                                  g_stream_selected_upload_stream);
        }
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming selected compact copy failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    if (!cuda_stream_selected_upload_record_ready()) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming selected compact event record failed\n");
        return 0;
    }
    return 1;
}

static int cuda_stream_selected_prepare_ptrs(
        const void *model_map,
        uint32_t layer,
        const int32_t *selected_ids,
        uint32_t n_selected,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!selected_ids ||
        n_selected == 0 ||
        n_selected > DS4_ROCM_N_EXPERT_USED ||
        !g_stream_selected_cache.gate_ptrs ||
        !g_stream_selected_cache.up_ptrs ||
        !g_stream_selected_cache.down_ptrs ||
        !cuda_stream_selected_ensure_stream()) {
        return 0;
    }
    const char *gate_ptrs[DS4_ROCM_N_EXPERT_USED] = {0};
    const char *up_ptrs[DS4_ROCM_N_EXPERT_USED] = {0};
    const char *down_ptrs[DS4_ROCM_N_EXPERT_USED] = {0};
    for (uint32_t i = 0; i < n_selected; i++) {
        int idx = cuda_stream_resident_find(model_map,
                                            layer,
                                            selected_ids[i],
                                            gate_offset,
                                            up_offset,
                                            down_offset,
                                            gate_expert_bytes,
                                            down_expert_bytes);
        if (idx < 0) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming selected pointer expert missing\n");
            return 0;
        }
        cuda_stream_resident_expert &entry =
            g_stream_resident_experts[(size_t)idx];
        entry.last_used = ++g_stream_resident_clock;
        gate_ptrs[i] = entry.gate;
        up_ptrs[i] = entry.up;
        down_ptrs[i] = entry.down;
    }
    if (!g_stream_selected_cache.gate_ptrs_stage ||
        !g_stream_selected_cache.up_ptrs_stage ||
        !g_stream_selected_cache.down_ptrs_stage) {
        return 0;
    }
    const size_t ptr_bytes = n_selected * sizeof(gate_ptrs[0]);
    memcpy((void *)g_stream_selected_cache.gate_ptrs_stage,
           gate_ptrs,
           ptr_bytes);
    memcpy((void *)g_stream_selected_cache.up_ptrs_stage,
           up_ptrs,
           ptr_bytes);
    memcpy((void *)g_stream_selected_cache.down_ptrs_stage,
           down_ptrs,
           ptr_bytes);
    cudaError_t err = cudaMemcpyAsync(g_stream_selected_cache.gate_ptrs,
                                      g_stream_selected_cache.gate_ptrs_stage,
                                      ptr_bytes,
                                      cudaMemcpyHostToDevice,
                                      g_stream_selected_upload_stream);
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(g_stream_selected_cache.up_ptrs,
                              g_stream_selected_cache.up_ptrs_stage,
                              ptr_bytes,
                              cudaMemcpyHostToDevice,
                              g_stream_selected_upload_stream);
    }
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(g_stream_selected_cache.down_ptrs,
                              g_stream_selected_cache.down_ptrs_stage,
                              ptr_bytes,
                              cudaMemcpyHostToDevice,
                              g_stream_selected_upload_stream);
    }
    int upload_record_ok = 1;
    if (err == cudaSuccess) {
        upload_record_ok = cuda_stream_selected_upload_record_ready();
    }
    if (err != cudaSuccess || !upload_record_ok) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming selected pointer upload failed%s%s\n",
                err != cudaSuccess ? ": " : "",
                err != cudaSuccess ? cudaGetErrorString(err) : "");
        if (err != cudaSuccess) (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

static int cuda_stream_batch_selected_prepare_from_host(
        const void *model_map,
        uint64_t model_size,
        uint32_t layer,
        const int32_t *ids,
        uint32_t n_tokens,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        const ds4_gpu_tensor **selected_exec,
        const char ***gate_ptrs,
        const char ***up_ptrs,
        const char ***down_ptrs,
        uint32_t *unique_out,
        int begin_pending) {
    if (!g_ssd_streaming_mode ||
        !model_map ||
        !ids ||
        !selected_exec ||
        !gate_ptrs ||
        !up_ptrs ||
        !down_ptrs ||
        !unique_out ||
        n_tokens <= 1 ||
        n_total_expert == 0 ||
        n_total_expert > DS4_ROCM_MAX_N_EXPERT ||
        n_selected == 0 ||
        n_selected > DS4_ROCM_N_EXPERT_USED ||
        gate_expert_bytes == 0 ||
        down_expert_bytes == 0) {
        return 0;
    }
    if (g_stream_batch_selected_pending.active) {
        cuda_stream_batch_selected_abort_pending();
    }
    if (g_stream_selected_pending.active) {
        cuda_stream_selected_abort_pending();
    }
    g_stream_batch_selected_cache.loaded = 0;
    if (cuda_q4k_packed_slice_refuse_routed_tables(
            model_map, n_total_expert, gate_offset, up_offset, down_offset,
            gate_expert_bytes, down_expert_bytes,
            "streaming batch selected experts")) return 0;

    uint64_t n_ids64 = 0;
    if (!cuda_u64_mul_checked(n_tokens, n_selected, &n_ids64) ||
        n_ids64 > SIZE_MAX / sizeof(int32_t)) {
        return 0;
    }
    int32_t *compact_ids = (int32_t *)malloc((size_t)n_ids64 * sizeof(compact_ids[0]));
    if (!compact_ids) {
        free(compact_ids);
        return 0;
    }
    uint8_t *pair_missing = (uint8_t *)malloc((size_t)n_ids64);
    if (!pair_missing) {
        free(compact_ids);
        return 0;
    }

    int ok = 1;
    int32_t map[DS4_ROCM_MAX_N_EXPERT];
    int32_t unique_ids[DS4_ROCM_MAX_N_EXPERT];
    uint8_t unique_missing[DS4_ROCM_MAX_N_EXPERT] = {0};
    for (uint32_t i = 0; i < DS4_ROCM_MAX_N_EXPERT; i++) map[i] = -1;
    uint32_t unique_count = 0;
    if (ok) {
        for (uint64_t i = 0; i < n_ids64; i++) {
            const int32_t expert = ids[i];
            if (expert < 0 || (uint32_t)expert >= n_total_expert) {
                fprintf(stderr,
                        DS4_GPU_LOG_PREFIX "streaming batch selected expert id %d outside 0..%u "
                        "(layer=%u)\n",
                        expert,
                        n_total_expert,
                        layer);
                ok = 0;
                break;
            }
            int32_t slot = map[(uint32_t)expert];
            if (slot < 0) {
                if (unique_count >= DS4_ROCM_MAX_N_EXPERT) {
                    ok = 0;
                    break;
                }
                slot = (int32_t)unique_count;
                map[(uint32_t)expert] = slot;
                unique_ids[unique_count++] = expert;
            }
            compact_ids[i] = slot;
        }
    }
    if (ok && unique_count == 0) ok = 0;
    if (ok && !cuda_stream_batch_selected_ensure_buffers(n_ids64, unique_count)) {
        ok = 0;
    }
    if (ok && !cuda_stream_selected_ensure_stream()) ok = 0;
    const int stats_on = cuda_stream_cache_stats_on();
    const int layer_stats_on =
        cuda_stream_cache_layer_stats_on() &&
        layer < DS4_ROCM_STREAM_CACHE_LAYER_STATS_MAX;
    if (ok && stats_on) {
        g_stream_cache_stats.batch_calls++;
        g_stream_cache_stats.batch_unique += unique_count;
    }
    if (ok && layer_stats_on) {
        cuda_stream_cache_layer_stats *ls = &g_stream_cache_layer_stats[layer];
        ls->batch_calls++;
        ls->batch_unique += unique_count;
    }

    cuda_stream_read_job read_jobs[DS4_ROCM_STREAM_READ_MAX_JOBS];
    memset(read_jobs, 0, sizeof(read_jobs));
    uint32_t read_job_count = 0;
    const int use_fd =
        g_model_fd >= 0 &&
        (g_model_fd_host_base == NULL || model_map == g_model_fd_host_base);

    const char *gate_host[DS4_ROCM_MAX_N_EXPERT] = {0};
    const char *up_host[DS4_ROCM_MAX_N_EXPERT] = {0};
    const char *down_host[DS4_ROCM_MAX_N_EXPERT] = {0};
    const char *resident_gate_host[DS4_ROCM_MAX_N_EXPERT] = {0};
    const char *resident_up_host[DS4_ROCM_MAX_N_EXPERT] = {0};
    const char *missing_gate_host[DS4_ROCM_MAX_N_EXPERT] = {0};
    const char *missing_up_host[DS4_ROCM_MAX_N_EXPERT] = {0};
    uint32_t resident_count = 0;
    uint32_t missing_count = 0;

    for (uint32_t u = 0; ok && u < unique_count; u++) {
        const int32_t expert_i = unique_ids[u];
        const uint64_t expert = (uint64_t)(uint32_t)expert_i;
        uint64_t gate_rel = 0;
        uint64_t down_rel = 0;
        if (!cuda_u64_mul_checked(expert, gate_expert_bytes, &gate_rel) ||
            !cuda_u64_mul_checked(expert, down_expert_bytes, &down_rel) ||
            gate_rel > model_size ||
            down_rel > model_size ||
            gate_offset > model_size ||
            up_offset > model_size ||
            down_offset > model_size ||
            gate_rel > model_size - gate_offset ||
            gate_rel > model_size - up_offset ||
            down_rel > model_size - down_offset ||
            gate_expert_bytes > model_size - gate_offset - gate_rel ||
            gate_expert_bytes > model_size - up_offset - gate_rel ||
            down_expert_bytes > model_size - down_offset - down_rel) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming batch selected expert offset overflow\n");
            ok = 0;
            break;
        }

        int idx = cuda_stream_resident_find(model_map,
                                            layer,
                                            expert_i,
                                            gate_offset,
                                            up_offset,
                                            down_offset,
                                            gate_expert_bytes,
                                            down_expert_bytes);
        const int was_resident = idx >= 0;
        if (stats_on) {
            if (was_resident) {
                g_stream_cache_stats.batch_hits++;
            } else {
                g_stream_cache_stats.batch_misses++;
            }
        }
        if (layer_stats_on) {
            cuda_stream_cache_layer_stats *ls = &g_stream_cache_layer_stats[layer];
            if (was_resident) {
                ls->batch_hits++;
            } else {
                ls->batch_misses++;
            }
        }
        if (idx < 0) {
            idx = cuda_stream_resident_alloc(model_map,
                                             layer,
                                             expert_i,
                                             unique_ids,
                                             unique_count,
                                             gate_offset,
                                             up_offset,
                                             down_offset,
                                             gate_expert_bytes,
                                             down_expert_bytes);
            if (idx < 0) {
                ok = 0;
                break;
            }
            cuda_stream_resident_expert &entry =
                g_stream_resident_experts[(size_t)idx];
            if (use_fd) {
                if (read_job_count + 3u > DS4_ROCM_STREAM_READ_MAX_JOBS) {
                    if (!cuda_stream_flush_read_jobs(read_jobs, &read_job_count)) {
                        ok = 0;
                        break;
                    }
                }
                read_jobs[read_job_count++] =
                    {entry.gate, gate_offset + gate_rel, gate_expert_bytes,
                     NULL, NULL, 0, 0};
                read_jobs[read_job_count++] =
                    {entry.up, up_offset + gate_rel, gate_expert_bytes,
                     NULL, NULL, 0, 0};
                read_jobs[read_job_count++] =
                    {entry.down, down_offset + down_rel, down_expert_bytes,
                     NULL, NULL, 0, 0};
            } else {
                cudaError_t err = cudaMemcpyAsync(entry.gate,
                                                  (const char *)model_map + gate_offset + gate_rel,
                                                  (size_t)gate_expert_bytes,
                                                  cudaMemcpyHostToDevice,
                                                  g_stream_selected_upload_stream);
                if (err == cudaSuccess) {
                    err = cudaMemcpyAsync(entry.up,
                                          (const char *)model_map + up_offset + gate_rel,
                                          (size_t)gate_expert_bytes,
                                          cudaMemcpyHostToDevice,
                                          g_stream_selected_upload_stream);
                }
                if (err == cudaSuccess) {
                    err = cudaMemcpyAsync(entry.down,
                                          (const char *)model_map + down_offset + down_rel,
                                          (size_t)down_expert_bytes,
                                          cudaMemcpyHostToDevice,
                                          g_stream_selected_upload_stream);
                }
                if (err != cudaSuccess) {
                    fprintf(stderr,
                            DS4_GPU_LOG_PREFIX "streaming batch selected cached copy failed: %s\n",
                            cudaGetErrorString(err));
                    (void)cudaGetLastError();
                    ok = 0;
                    break;
                }
            }
        }
        if (idx >= 0) {
            cuda_stream_resident_expert &entry =
                g_stream_resident_experts[(size_t)idx];
            entry.last_used = ++g_stream_resident_clock;
            gate_host[u] = entry.gate;
            up_host[u] = entry.up;
            down_host[u] = entry.down;
            if (was_resident) {
                resident_gate_host[u] = entry.gate;
                resident_up_host[u] = entry.up;
                resident_count++;
            } else {
                unique_missing[u] = 1;
                missing_gate_host[u] = entry.gate;
                missing_up_host[u] = entry.up;
                missing_count++;
            }
        }
    }

    if (ok) {
        for (uint64_t i = 0; i < n_ids64; i++) {
            const int32_t slot = compact_ids[i];
            if (slot < 0 || (uint32_t)slot >= unique_count) {
                ok = 0;
                break;
            }
            pair_missing[i] = unique_missing[(uint32_t)slot];
        }
    }

    if (ok && begin_pending) {
        memset(&g_stream_batch_selected_pending, 0,
               sizeof(g_stream_batch_selected_pending));
        g_stream_batch_selected_pending.active = 1;
        g_stream_batch_selected_pending.model_map = model_map;
        g_stream_batch_selected_pending.layer = layer;
        g_stream_batch_selected_pending.n_total_expert = n_total_expert;
        g_stream_batch_selected_pending.n_selected = n_selected;
        g_stream_batch_selected_pending.n_tokens = n_tokens;
        g_stream_batch_selected_pending.n_unique = unique_count;
        g_stream_batch_selected_pending.resident_count = resident_count;
        g_stream_batch_selected_pending.missing_count = missing_count;
        g_stream_batch_selected_pending.gate_offset = gate_offset;
        g_stream_batch_selected_pending.up_offset = up_offset;
        g_stream_batch_selected_pending.down_offset = down_offset;
        g_stream_batch_selected_pending.gate_expert_bytes = gate_expert_bytes;
        g_stream_batch_selected_pending.down_expert_bytes = down_expert_bytes;
        g_stream_batch_selected_pending.read_job_count = read_job_count;
        memcpy(g_stream_batch_selected_pending.read_jobs,
               read_jobs,
               (size_t)read_job_count * sizeof(read_jobs[0]));
        if (use_fd && read_job_count != 0 &&
            !cuda_stream_read_jobs_start(
                    g_stream_batch_selected_pending.read_jobs,
                    read_job_count)) {
            memset(&g_stream_batch_selected_pending, 0,
                   sizeof(g_stream_batch_selected_pending));
            ok = 0;
        } else if (use_fd && read_job_count != 0) {
            read_job_count = 0;
        }
    }
    if (ok && !cuda_stream_flush_read_jobs(read_jobs, &read_job_count)) {
        ok = 0;
    }
    if (ok && !use_fd) {
        cudaError_t err = cudaStreamSynchronize(g_stream_selected_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming batch selected upload sync failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            ok = 0;
        }
    }
    if (ok) {
        if (!g_stream_batch_selected_cache.selected_stage ||
            !g_stream_batch_selected_cache.pair_missing_stage ||
            !g_stream_batch_selected_cache.gate_ptrs_stage ||
            !g_stream_batch_selected_cache.up_ptrs_stage ||
            !g_stream_batch_selected_cache.down_ptrs_stage ||
            !g_stream_batch_selected_cache.resident_gate_ptrs_stage ||
            !g_stream_batch_selected_cache.resident_up_ptrs_stage ||
            !g_stream_batch_selected_cache.missing_gate_ptrs_stage ||
            !g_stream_batch_selected_cache.missing_up_ptrs_stage) {
            ok = 0;
        }
    }
    if (ok) {
        const size_t selected_bytes =
            (size_t)n_ids64 * sizeof(compact_ids[0]);
        const size_t pair_missing_bytes = (size_t)n_ids64;
        const size_t ptr_bytes = (size_t)unique_count * sizeof(gate_host[0]);
        memcpy(g_stream_batch_selected_cache.selected_stage,
               compact_ids,
               selected_bytes);
        memcpy(g_stream_batch_selected_cache.pair_missing_stage,
               pair_missing,
               pair_missing_bytes);
        memcpy((void *)g_stream_batch_selected_cache.gate_ptrs_stage,
               gate_host,
               ptr_bytes);
        memcpy((void *)g_stream_batch_selected_cache.up_ptrs_stage,
               up_host,
               ptr_bytes);
        memcpy((void *)g_stream_batch_selected_cache.down_ptrs_stage,
               down_host,
               ptr_bytes);
        memcpy((void *)g_stream_batch_selected_cache.resident_gate_ptrs_stage,
               resident_gate_host,
               ptr_bytes);
        memcpy((void *)g_stream_batch_selected_cache.resident_up_ptrs_stage,
               resident_up_host,
               ptr_bytes);
        memcpy((void *)g_stream_batch_selected_cache.missing_gate_ptrs_stage,
               missing_gate_host,
               ptr_bytes);
        memcpy((void *)g_stream_batch_selected_cache.missing_up_ptrs_stage,
               missing_up_host,
               ptr_bytes);

        cudaError_t err = cudaMemcpyAsync(g_stream_batch_selected_cache.selected_ids,
                                          g_stream_batch_selected_cache.selected_stage,
                                          selected_bytes,
                                          cudaMemcpyHostToDevice,
                                          g_stream_selected_upload_stream);
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_batch_selected_cache.pair_missing,
                                  g_stream_batch_selected_cache.pair_missing_stage,
                                  pair_missing_bytes,
                                  cudaMemcpyHostToDevice,
                                  g_stream_selected_upload_stream);
        }
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_batch_selected_cache.gate_ptrs,
                                  g_stream_batch_selected_cache.gate_ptrs_stage,
                                  ptr_bytes,
                                  cudaMemcpyHostToDevice,
                                  g_stream_selected_upload_stream);
        }
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_batch_selected_cache.up_ptrs,
                                  g_stream_batch_selected_cache.up_ptrs_stage,
                                  ptr_bytes,
                                  cudaMemcpyHostToDevice,
                                  g_stream_selected_upload_stream);
        }
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_batch_selected_cache.down_ptrs,
                                  g_stream_batch_selected_cache.down_ptrs_stage,
                                  ptr_bytes,
                                  cudaMemcpyHostToDevice,
                                  g_stream_selected_upload_stream);
        }
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_batch_selected_cache.resident_gate_ptrs,
                                  g_stream_batch_selected_cache.resident_gate_ptrs_stage,
                                  ptr_bytes,
                                  cudaMemcpyHostToDevice,
                                  g_stream_selected_upload_stream);
        }
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_batch_selected_cache.resident_up_ptrs,
                                  g_stream_batch_selected_cache.resident_up_ptrs_stage,
                                  ptr_bytes,
                                  cudaMemcpyHostToDevice,
                                  g_stream_selected_upload_stream);
        }
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_batch_selected_cache.missing_gate_ptrs,
                                  g_stream_batch_selected_cache.missing_gate_ptrs_stage,
                                  ptr_bytes,
                                  cudaMemcpyHostToDevice,
                                  g_stream_selected_upload_stream);
        }
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_batch_selected_cache.missing_up_ptrs,
                                  g_stream_batch_selected_cache.missing_up_ptrs_stage,
                                  ptr_bytes,
                                  cudaMemcpyHostToDevice,
                                  g_stream_selected_upload_stream);
        }
        int upload_record_ok = 1;
        if (err == cudaSuccess) {
            upload_record_ok =
                cuda_stream_batch_selected_upload_record_ready();
        }
        if (err != cudaSuccess || !upload_record_ok) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming batch selected table upload failed%s%s\n",
                    err != cudaSuccess ? ": " : "",
                    err != cudaSuccess ? cudaGetErrorString(err) : "");
            if (err != cudaSuccess) (void)cudaGetLastError();
            if (g_stream_batch_selected_pending.active) {
                cuda_stream_batch_selected_abort_pending();
            }
            ok = 0;
        }
    }

    if (ok) {
        g_stream_batch_selected_cache.loaded = 0;
        g_stream_batch_selected_cache.model_map = model_map;
        g_stream_batch_selected_cache.layer = layer;
        g_stream_batch_selected_cache.n_total_expert = n_total_expert;
        g_stream_batch_selected_cache.n_selected = n_selected;
        g_stream_batch_selected_cache.n_tokens = n_tokens;
        g_stream_batch_selected_cache.n_unique = unique_count;
        g_stream_batch_selected_cache.gate_offset = gate_offset;
        g_stream_batch_selected_cache.up_offset = up_offset;
        g_stream_batch_selected_cache.down_offset = down_offset;
        g_stream_batch_selected_cache.gate_expert_bytes = gate_expert_bytes;
        g_stream_batch_selected_cache.down_expert_bytes = down_expert_bytes;
        *selected_exec = &g_stream_batch_selected_cache.selected_tensor;
        *gate_ptrs = g_stream_batch_selected_cache.gate_ptrs;
        *up_ptrs = g_stream_batch_selected_cache.up_ptrs;
        *down_ptrs = g_stream_batch_selected_cache.down_ptrs;
        *unique_out = unique_count;
    } else {
        g_stream_batch_selected_cache.loaded = 0;
        if (g_stream_batch_selected_pending.active) {
            cuda_stream_batch_selected_abort_pending();
        }
        if (read_job_count != 0) cuda_stream_read_jobs_free(read_jobs, read_job_count);
    }

    free(compact_ids);
    free(pair_missing);
    return ok;
}

static int cuda_stream_batch_selected_prepare(
        const void *model_map,
        uint64_t model_size,
        uint32_t layer,
        const ds4_gpu_tensor *selected,
        uint32_t n_tokens,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        const ds4_gpu_tensor **selected_exec,
        const char ***gate_ptrs,
        const char ***up_ptrs,
        const char ***down_ptrs,
        uint32_t *unique_out) {
    if (!selected ||
        !cuda_tensor_has_elems2(selected, n_tokens, n_selected, sizeof(int32_t))) {
        return 0;
    }
    uint64_t n_ids64 = 0;
    if (!cuda_u64_mul_checked(n_tokens, n_selected, &n_ids64) ||
        n_ids64 > SIZE_MAX / sizeof(int32_t)) {
        return 0;
    }
    int32_t *ids = (int32_t *)malloc((size_t)n_ids64 * sizeof(ids[0]));
    if (!ids) return 0;

    const int copy_ok = cuda_ok(cudaMemcpy(ids,
                                           selected->ptr,
                                           (size_t)n_ids64 * sizeof(ids[0]),
                                           cudaMemcpyDeviceToHost),
                                "streaming batch selected ids copy");
    const int ok = copy_ok &&
        cuda_stream_batch_selected_prepare_from_host(model_map,
                                                     model_size,
                                                     layer,
                                                     ids,
                                                     n_tokens,
                                                     n_total_expert,
                                                     n_selected,
                                                     gate_offset,
                                                     up_offset,
                                                     down_offset,
                                                     gate_expert_bytes,
                                                     down_expert_bytes,
                                                     selected_exec,
                                                     gate_ptrs,
                                                     up_ptrs,
                                                     down_ptrs,
                                                     unique_out,
                                                     0);
    free(ids);
    return ok;
}

static int cuda_stream_layer_expert_cache_prepare_batch(
        const void *model_map,
        uint32_t layer,
        const ds4_gpu_tensor *selected,
        uint32_t n_tokens,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        const ds4_gpu_tensor **selected_exec,
        const char ***gate_ptrs,
        const char ***up_ptrs,
        const char ***down_ptrs,
        uint32_t *unique_out) {
    if (!selected ||
        !selected_exec ||
        !gate_ptrs ||
        !up_ptrs ||
        !down_ptrs ||
        !unique_out ||
        !cuda_tensor_has_elems2(selected, n_tokens, n_selected, sizeof(int32_t)) ||
        n_tokens <= 1 ||
        n_total_expert == 0 ||
        n_total_expert > DS4_ROCM_MAX_N_EXPERT ||
        n_selected == 0 ||
        n_selected > DS4_ROCM_N_EXPERT_USED ||
        gate_expert_bytes == 0 ||
        down_expert_bytes == 0) {
        return 0;
    }
    const char *layer_gate = NULL;
    const char *layer_up = NULL;
    const char *layer_down = NULL;
    if (!cuda_stream_layer_expert_cache_apply(model_map,
                                              layer,
                                              n_total_expert,
                                              gate_offset,
                                              up_offset,
                                              down_offset,
                                              gate_expert_bytes,
                                              down_expert_bytes,
                                              &layer_gate,
                                              &layer_up,
                                              &layer_down)) {
        return 0;
    }

    uint64_t n_ids64 = 0;
    if (!cuda_u64_mul_checked(n_tokens, n_selected, &n_ids64) ||
        n_ids64 > SIZE_MAX / sizeof(int32_t)) {
        return 0;
    }
    int32_t *ids = (int32_t *)malloc((size_t)n_ids64 * sizeof(ids[0]));
    int32_t *compact_ids =
        (int32_t *)malloc((size_t)n_ids64 * sizeof(compact_ids[0]));
    if (!ids || !compact_ids) {
        free(ids);
        free(compact_ids);
        return 0;
    }

    int ok = cuda_ok(cudaMemcpy(ids,
                                selected->ptr,
                                (size_t)n_ids64 * sizeof(ids[0]),
                                cudaMemcpyDeviceToHost),
                     "streaming full-layer selected ids copy");

    int32_t map[DS4_ROCM_MAX_N_EXPERT];
    int32_t unique_ids[DS4_ROCM_MAX_N_EXPERT];
    for (uint32_t i = 0; i < DS4_ROCM_MAX_N_EXPERT; i++) map[i] = -1;
    uint32_t unique_count = 0;
    for (uint64_t i = 0; ok && i < n_ids64; i++) {
        const int32_t expert = ids[i];
        if (expert < 0 || (uint32_t)expert >= n_total_expert) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming full-layer selected expert id %d "
                    "outside 0..%u (layer=%u)\n",
                    expert,
                    n_total_expert,
                    layer);
            ok = 0;
            break;
        }
        int32_t slot = map[(uint32_t)expert];
        if (slot < 0) {
            if (unique_count >= DS4_ROCM_MAX_N_EXPERT) {
                ok = 0;
                break;
            }
            slot = (int32_t)unique_count;
            map[(uint32_t)expert] = slot;
            unique_ids[unique_count++] = expert;
        }
        compact_ids[i] = slot;
    }
    if (ok && unique_count == 0) ok = 0;
    if (ok && !cuda_stream_batch_selected_ensure_buffers(n_ids64, unique_count)) {
        ok = 0;
    }
    if (ok && !cuda_stream_selected_ensure_stream()) ok = 0;

    const char *gate_host[DS4_ROCM_MAX_N_EXPERT] = {0};
    const char *up_host[DS4_ROCM_MAX_N_EXPERT] = {0};
    const char *down_host[DS4_ROCM_MAX_N_EXPERT] = {0};
    for (uint32_t u = 0; ok && u < unique_count; u++) {
        const uint64_t expert = (uint64_t)(uint32_t)unique_ids[u];
        uint64_t gate_rel = 0;
        uint64_t down_rel = 0;
        if (!cuda_u64_mul_checked(expert, gate_expert_bytes, &gate_rel) ||
            !cuda_u64_mul_checked(expert, down_expert_bytes, &down_rel)) {
            ok = 0;
            break;
        }
        gate_host[u] = layer_gate + gate_rel;
        up_host[u] = layer_up + gate_rel;
        down_host[u] = layer_down + down_rel;
    }

    if (ok) {
        cudaError_t err = cudaMemcpyAsync(g_stream_batch_selected_cache.selected_ids,
                                          compact_ids,
                                          (size_t)n_ids64 * sizeof(compact_ids[0]),
                                          cudaMemcpyHostToDevice,
                                          g_stream_selected_upload_stream);
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_batch_selected_cache.gate_ptrs,
                                  gate_host,
                                  unique_count * sizeof(gate_host[0]),
                                  cudaMemcpyHostToDevice,
                                  g_stream_selected_upload_stream);
        }
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_batch_selected_cache.up_ptrs,
                                  up_host,
                                  unique_count * sizeof(up_host[0]),
                                  cudaMemcpyHostToDevice,
                                  g_stream_selected_upload_stream);
        }
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_batch_selected_cache.down_ptrs,
                                  down_host,
                                  unique_count * sizeof(down_host[0]),
                                  cudaMemcpyHostToDevice,
                                  g_stream_selected_upload_stream);
        }
        if (err == cudaSuccess) err = cudaStreamSynchronize(g_stream_selected_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming full-layer selected table upload failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            ok = 0;
        }
    }

    if (ok) {
        g_stream_batch_selected_cache.loaded = 0;
        g_stream_batch_selected_cache.model_map = model_map;
        g_stream_batch_selected_cache.layer = layer;
        g_stream_batch_selected_cache.n_total_expert = n_total_expert;
        g_stream_batch_selected_cache.n_selected = n_selected;
        g_stream_batch_selected_cache.n_tokens = n_tokens;
        g_stream_batch_selected_cache.n_unique = unique_count;
        g_stream_batch_selected_cache.gate_offset = gate_offset;
        g_stream_batch_selected_cache.up_offset = up_offset;
        g_stream_batch_selected_cache.down_offset = down_offset;
        g_stream_batch_selected_cache.gate_expert_bytes = gate_expert_bytes;
        g_stream_batch_selected_cache.down_expert_bytes = down_expert_bytes;
        *selected_exec = &g_stream_batch_selected_cache.selected_tensor;
        *gate_ptrs = g_stream_batch_selected_cache.gate_ptrs;
        *up_ptrs = g_stream_batch_selected_cache.up_ptrs;
        *down_ptrs = g_stream_batch_selected_cache.down_ptrs;
        *unique_out = unique_count;
    } else {
        g_stream_batch_selected_cache.loaded = 0;
    }

    free(ids);
    free(compact_ids);
    return ok;
}

static int cuda_stream_layer_expert_cache_seed_selected(
        const void *model_map,
        uint64_t model_size,
        uint32_t layer,
        const ds4_gpu_tensor *selected,
        uint32_t n_tokens,
        uint32_t n_seed_tokens,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!g_ssd_streaming_mode ||
        !model_map ||
        !selected ||
        n_tokens == 0 ||
        n_seed_tokens == 0 ||
        n_total_expert == 0 ||
        n_total_expert > DS4_ROCM_MAX_N_EXPERT ||
        n_selected == 0 ||
        n_selected > DS4_ROCM_N_EXPERT_USED ||
        gate_expert_bytes == 0 ||
        down_expert_bytes == 0 ||
        !cuda_tensor_has_elems2(selected, n_tokens, n_selected, sizeof(int32_t))) {
        return 0;
    }

    uint64_t gate_bytes = 0;
    uint64_t down_bytes = 0;
    if (!cuda_u64_mul_checked(n_total_expert, gate_expert_bytes, &gate_bytes) ||
        !cuda_u64_mul_checked(n_total_expert, down_expert_bytes, &down_bytes) ||
        gate_offset > model_size ||
        up_offset > model_size ||
        down_offset > model_size ||
        gate_bytes > model_size - gate_offset ||
        gate_bytes > model_size - up_offset ||
        down_bytes > model_size - down_offset) {
        return 0;
    }

    const char *layer_gate = NULL;
    const char *layer_up = NULL;
    const char *layer_down = NULL;
    if (!cuda_stream_layer_expert_cache_apply(model_map,
                                              layer,
                                              n_total_expert,
                                              gate_offset,
                                              up_offset,
                                              down_offset,
                                              gate_expert_bytes,
                                              down_expert_bytes,
                                              &layer_gate,
                                              &layer_up,
                                              &layer_down)) {
        if (!cuda_model_range_is_cached(model_map, gate_offset, gate_bytes) ||
            !cuda_model_range_is_cached(model_map, up_offset, gate_bytes) ||
            !cuda_model_range_is_cached(model_map, down_offset, down_bytes)) {
            return 0;
        }
        layer_gate = cuda_model_range_ptr(model_map,
                                          gate_offset,
                                          gate_bytes,
                                          "streaming full-layer seed gate");
        layer_up = cuda_model_range_ptr(model_map,
                                        up_offset,
                                        gate_bytes,
                                        "streaming full-layer seed up");
        layer_down = cuda_model_range_ptr(model_map,
                                          down_offset,
                                          down_bytes,
                                          "streaming full-layer seed down");
        if (!layer_gate || !layer_up || !layer_down) return 0;
    }

    if (n_seed_tokens > n_tokens) n_seed_tokens = n_tokens;
    const uint64_t n_ids64 = (uint64_t)n_seed_tokens * n_selected;
    if (n_ids64 == 0 || n_ids64 > SIZE_MAX / sizeof(int32_t)) return 0;
    int32_t ids_stack[DS4_ROCM_N_EXPERT_USED * 16u];
    int32_t *ids_heap = NULL;
    int32_t *ids = ids_stack;
    if (n_ids64 > sizeof(ids_stack) / sizeof(ids_stack[0])) {
        ids_heap = (int32_t *)malloc((size_t)n_ids64 * sizeof(ids_heap[0]));
        if (!ids_heap) return 0;
        ids = ids_heap;
    }

    const uint64_t src_off =
        (uint64_t)(n_tokens - n_seed_tokens) * n_selected * sizeof(int32_t);
    int ok = cuda_ok(cudaMemcpy(ids,
                                (const char *)selected->ptr + src_off,
                                (size_t)n_ids64 * sizeof(ids[0]),
                                cudaMemcpyDeviceToHost),
                     "streaming full-layer seed selected ids copy");

    int32_t unique_stack[DS4_ROCM_N_EXPERT_USED * 16u];
    int32_t *unique_heap = NULL;
    int32_t *unique = unique_stack;
    if (n_ids64 > sizeof(unique_stack) / sizeof(unique_stack[0])) {
        unique_heap = (int32_t *)malloc((size_t)n_ids64 * sizeof(unique_heap[0]));
        if (!unique_heap) ok = 0;
        unique = unique_heap;
    }
    uint32_t unique_count = 0;
    bool seen[DS4_ROCM_MAX_N_EXPERT] = {0};
    for (uint64_t i = 0; ok && i < n_ids64; i++) {
        const int32_t expert = ids[i];
        if (expert < 0 || (uint32_t)expert >= n_total_expert) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming full-layer seed expert id %d "
                    "outside 0..%u (layer=%u)\n",
                    expert,
                    n_total_expert,
                    layer);
            ok = 0;
            break;
        }
        if (seen[(uint32_t)expert]) continue;
        seen[(uint32_t)expert] = true;
        unique[unique_count++] = expert;
    }
    if (ok && cuda_stream_cache_stats_on()) {
        g_stream_cache_stats.seed_calls++;
        g_stream_cache_stats.seed_unique += unique_count;
    }

    if (ok && unique_count != 0 && !cuda_stream_selected_ensure_stream()) {
        ok = 0;
    }

    for (uint32_t u = 0; ok && u < unique_count; u++) {
        const int32_t expert_i32 = unique[u];
        int idx = cuda_stream_resident_find(model_map,
                                            layer,
                                            expert_i32,
                                            gate_offset,
                                            up_offset,
                                            down_offset,
                                            gate_expert_bytes,
                                            down_expert_bytes);
        if (idx >= 0) {
            g_stream_resident_experts[(size_t)idx].last_used =
                ++g_stream_resident_clock;
            continue;
        }

        idx = cuda_stream_resident_alloc(model_map,
                                         layer,
                                         expert_i32,
                                         unique,
                                         unique_count,
                                         gate_offset,
                                         up_offset,
                                         down_offset,
                                         gate_expert_bytes,
                                         down_expert_bytes);
        if (idx < 0) {
            ok = 0;
            break;
        }

        const uint64_t expert = (uint64_t)(uint32_t)expert_i32;
        uint64_t gate_rel = 0;
        uint64_t down_rel = 0;
        if (!cuda_u64_mul_checked(expert, gate_expert_bytes, &gate_rel) ||
            !cuda_u64_mul_checked(expert, down_expert_bytes, &down_rel)) {
            ok = 0;
            break;
        }

        cuda_stream_resident_expert &entry =
            g_stream_resident_experts[(size_t)idx];
        cudaError_t err = cudaMemcpyAsync(entry.gate,
                                          layer_gate + gate_rel,
                                          (size_t)gate_expert_bytes,
                                          cudaMemcpyDeviceToDevice,
                                          g_stream_selected_upload_stream);
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(entry.up,
                                  layer_up + gate_rel,
                                  (size_t)gate_expert_bytes,
                                  cudaMemcpyDeviceToDevice,
                                  g_stream_selected_upload_stream);
        }
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(entry.down,
                                  layer_down + down_rel,
                                  (size_t)down_expert_bytes,
                                  cudaMemcpyDeviceToDevice,
                                  g_stream_selected_upload_stream);
        }
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming full-layer seed D2D copy failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            ok = 0;
            break;
        }
    }

    if (ok && unique_count != 0) {
        cudaError_t err = cudaStreamSynchronize(g_stream_selected_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming full-layer seed sync failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            ok = 0;
        }
    }

    if (!ok) cuda_stream_resident_cache_release();
    free(ids_heap);
    free(unique_heap);
    return ok;
}

static int cuda_stream_selected_load(
        const void *model_map,
        uint64_t model_size,
        uint32_t layer,
        const int32_t *selected_ids,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    g_stream_selected_cache.loaded = 0;
    if (g_stream_selected_pending.active) {
        cuda_stream_selected_abort_pending();
    }
    if (g_stream_batch_selected_pending.active) {
        cuda_stream_batch_selected_abort_pending();
    }
    if (!g_ssd_streaming_mode) return 1;
    if (!model_map || !selected_ids ||
        n_total_expert == 0 ||
        n_total_expert > DS4_ROCM_MAX_N_EXPERT ||
        n_selected == 0 ||
        n_selected > DS4_ROCM_N_EXPERT_USED ||
        gate_expert_bytes == 0 ||
        down_expert_bytes == 0) {
        return 0;
    }
    if (cuda_q4k_packed_slice_refuse_routed_tables(
            model_map, n_total_expert, gate_offset, up_offset, down_offset,
            gate_expert_bytes, down_expert_bytes,
            "streaming selected experts")) return 0;
    uint64_t gate_bytes = 0;
    uint64_t down_bytes = 0;
    if (!cuda_u64_mul_checked(n_selected, gate_expert_bytes, &gate_bytes) ||
        !cuda_u64_mul_checked(n_selected, down_expert_bytes, &down_bytes)) {
        return 0;
    }
    if (!cuda_stream_selected_reuse_wait("streaming selected cache reuse")) {
        return 0;
    }
    if (!cuda_stream_selected_ensure_buffers(gate_bytes, down_bytes)) return 0;
    if (!cuda_stream_selected_ensure_stream()) return 0;
    cuda_stream_selected_cache_header(model_map,
                                      layer,
                                      n_total_expert,
                                      n_selected,
                                      selected_ids,
                                      gate_expert_bytes,
                                      down_expert_bytes);
    const int stats_on = cuda_stream_cache_stats_on();
    const int layer_stats_on =
        cuda_stream_cache_layer_stats_on() &&
        layer < DS4_ROCM_STREAM_CACHE_LAYER_STATS_MAX;
    if (stats_on) {
        g_stream_cache_stats.selected_calls++;
        g_stream_cache_stats.selected_slots += n_selected;
    }
    if (layer_stats_on) {
        cuda_stream_cache_layer_stats *ls = &g_stream_cache_layer_stats[layer];
        ls->selected_calls++;
        ls->selected_slots += n_selected;
    }

    cuda_stream_read_job read_jobs[DS4_ROCM_N_EXPERT_USED * 3u];
    memset(read_jobs, 0, sizeof(read_jobs));
    uint32_t read_job_count = 0;
    uint32_t resident_mask = 0;
    uint32_t missing_mask = 0;
    const int use_fd =
        g_model_fd >= 0 &&
        (g_model_fd_host_base == NULL || model_map == g_model_fd_host_base);

    for (uint32_t i = 0; i < n_selected; i++) {
        if (selected_ids[i] < 0 || (uint32_t)selected_ids[i] >= n_total_expert) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming selected expert id %d outside 0..%u "
                    "(layer=%u slot=%u selected=[%d,%d,%d,%d,%d,%d,%d,%d])\n",
                    selected_ids[i],
                    n_total_expert,
                    layer,
                    i,
                    n_selected > 0 ? selected_ids[0] : -1,
                    n_selected > 1 ? selected_ids[1] : -1,
                    n_selected > 2 ? selected_ids[2] : -1,
                    n_selected > 3 ? selected_ids[3] : -1,
                    n_selected > 4 ? selected_ids[4] : -1,
                    n_selected > 5 ? selected_ids[5] : -1,
                    n_selected > 6 ? selected_ids[6] : -1,
                    n_selected > 7 ? selected_ids[7] : -1);
            return 0;
        }
        const uint64_t expert = (uint64_t)(uint32_t)selected_ids[i];
        uint64_t gate_rel = 0;
        uint64_t down_rel = 0;
        if (!cuda_u64_mul_checked(expert, gate_expert_bytes, &gate_rel) ||
            !cuda_u64_mul_checked(expert, down_expert_bytes, &down_rel) ||
            gate_rel > model_size ||
            down_rel > model_size ||
            gate_offset > model_size ||
            up_offset > model_size ||
            down_offset > model_size ||
            gate_rel > model_size - gate_offset ||
            gate_rel > model_size - up_offset ||
            down_rel > model_size - down_offset ||
            gate_expert_bytes > model_size - gate_offset - gate_rel ||
            gate_expert_bytes > model_size - up_offset - gate_rel ||
            down_expert_bytes > model_size - down_offset - down_rel) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming selected expert offset overflow\n");
            return 0;
        }

        int idx = cuda_stream_resident_find(model_map,
                                            layer,
                                            selected_ids[i],
                                            gate_offset,
                                            up_offset,
                                            down_offset,
                                            gate_expert_bytes,
                                            down_expert_bytes);
        if (stats_on) {
            if (idx >= 0) {
                g_stream_cache_stats.selected_hits++;
            } else {
                g_stream_cache_stats.selected_misses++;
            }
        }
        if (layer_stats_on) {
            cuda_stream_cache_layer_stats *ls = &g_stream_cache_layer_stats[layer];
            if (idx >= 0) {
                ls->selected_hits++;
            } else {
                ls->selected_misses++;
            }
        }
        if (idx >= 0) {
            g_stream_resident_experts[(size_t)idx].last_used =
                ++g_stream_resident_clock;
            resident_mask |= 1u << i;
            continue;
        }

        idx = cuda_stream_resident_alloc(model_map,
                                         layer,
                                         selected_ids[i],
                                         selected_ids,
                                         n_selected,
                                         gate_offset,
                                         up_offset,
                                         down_offset,
                                         gate_expert_bytes,
                                         down_expert_bytes);
        if (idx < 0) return 0;
        missing_mask |= 1u << i;
        cuda_stream_resident_expert &entry =
            g_stream_resident_experts[(size_t)idx];

        if (use_fd) {
            if (read_job_count + 3u > DS4_ROCM_N_EXPERT_USED * 3u) return 0;
            read_jobs[read_job_count++] =
                {entry.gate, gate_offset + gate_rel, gate_expert_bytes,
                 NULL, NULL, 0, 0};
            read_jobs[read_job_count++] =
                {entry.up, up_offset + gate_rel, gate_expert_bytes,
                 NULL, NULL, 0, 0};
            read_jobs[read_job_count++] =
                {entry.down, down_offset + down_rel, down_expert_bytes,
                 NULL, NULL, 0, 0};
        } else {
            cudaError_t err = cudaMemcpyAsync(entry.gate,
                                              (const char *)model_map + gate_offset + gate_rel,
                                              (size_t)gate_expert_bytes,
                                              cudaMemcpyHostToDevice,
                                              g_stream_selected_upload_stream);
            if (err == cudaSuccess) {
                err = cudaMemcpyAsync(entry.up,
                                      (const char *)model_map + up_offset + gate_rel,
                                      (size_t)gate_expert_bytes,
                                      cudaMemcpyHostToDevice,
                                      g_stream_selected_upload_stream);
            }
            if (err == cudaSuccess) {
                err = cudaMemcpyAsync(entry.down,
                                      (const char *)model_map + down_offset + down_rel,
                                      (size_t)down_expert_bytes,
                                      cudaMemcpyHostToDevice,
                                      g_stream_selected_upload_stream);
            }
            if (err != cudaSuccess) {
                fprintf(stderr,
                        DS4_GPU_LOG_PREFIX "streaming selected cached copy failed: %s\n",
                        cudaGetErrorString(err));
                (void)cudaGetLastError();
                cuda_stream_resident_cache_release();
                return 0;
            }
        }
    }

    if (resident_mask != 0 && missing_mask == 0) {
        g_stream_selected_pending.active = 1;
        g_stream_selected_pending.model_map = model_map;
        g_stream_selected_pending.layer = layer;
        g_stream_selected_pending.n_total_expert = n_total_expert;
        g_stream_selected_pending.n_selected = n_selected;
        g_stream_selected_pending.gate_offset = gate_offset;
        g_stream_selected_pending.up_offset = up_offset;
        g_stream_selected_pending.down_offset = down_offset;
        g_stream_selected_pending.gate_expert_bytes = gate_expert_bytes;
        g_stream_selected_pending.down_expert_bytes = down_expert_bytes;
        g_stream_selected_pending.resident_mask = resident_mask;
        g_stream_selected_pending.missing_mask = 0;
        g_stream_selected_pending.read_job_count = 0;
        for (uint32_t i = 0; i < n_selected; i++) {
            g_stream_selected_pending.selected_ids[i] = selected_ids[i];
        }
        if (!cuda_stream_selected_prepare_ptrs(model_map,
                                               layer,
                                               selected_ids,
                                               n_selected,
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes)) {
            memset(&g_stream_selected_pending, 0,
                   sizeof(g_stream_selected_pending));
            cuda_stream_resident_cache_release();
            return 0;
        }
        return 1;
    }

    if (use_fd && read_job_count != 0) {
        g_stream_selected_pending.active = 1;
        g_stream_selected_pending.model_map = model_map;
        g_stream_selected_pending.layer = layer;
        g_stream_selected_pending.n_total_expert = n_total_expert;
        g_stream_selected_pending.n_selected = n_selected;
        g_stream_selected_pending.gate_offset = gate_offset;
        g_stream_selected_pending.up_offset = up_offset;
        g_stream_selected_pending.down_offset = down_offset;
        g_stream_selected_pending.gate_expert_bytes = gate_expert_bytes;
        g_stream_selected_pending.down_expert_bytes = down_expert_bytes;
        g_stream_selected_pending.resident_mask = resident_mask;
        g_stream_selected_pending.missing_mask = missing_mask;
        g_stream_selected_pending.read_job_count = read_job_count;
        for (uint32_t i = 0; i < n_selected; i++) {
            g_stream_selected_pending.selected_ids[i] = selected_ids[i];
        }
        memcpy(g_stream_selected_pending.read_jobs,
               read_jobs,
               (size_t)read_job_count * sizeof(read_jobs[0]));
        if (!cuda_stream_read_jobs_start(g_stream_selected_pending.read_jobs,
                                         read_job_count)) {
            memset(&g_stream_selected_pending, 0, sizeof(g_stream_selected_pending));
            cuda_stream_read_jobs_free(read_jobs, read_job_count);
            cuda_stream_resident_cache_release();
            return 0;
        }
        if (!cuda_stream_selected_prepare_ptrs(model_map,
                                               layer,
                                               selected_ids,
                                               n_selected,
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes)) {
            (void)cuda_stream_read_jobs_wait(g_stream_selected_pending.read_jobs,
                                             read_job_count);
            cuda_stream_read_jobs_free(g_stream_selected_pending.read_jobs,
                                       read_job_count);
            memset(&g_stream_selected_pending, 0, sizeof(g_stream_selected_pending));
            cuda_stream_read_jobs_free(read_jobs, read_job_count);
            cuda_stream_resident_cache_release();
            return 0;
        }
        return 1;
    }

    if (resident_mask != 0 &&
        !cuda_stream_selected_compact_mask(model_map,
                                           layer,
                                           selected_ids,
                                           n_total_expert,
                                           n_selected,
                                           gate_offset,
                                           up_offset,
                                           down_offset,
                                           gate_expert_bytes,
                                           down_expert_bytes,
                                           resident_mask)) {
        cuda_stream_read_jobs_free(read_jobs, read_job_count);
        cuda_stream_resident_cache_release();
        return 0;
    }

    if (read_job_count != 0) {
        if (!cuda_stream_read_jobs_parallel(read_jobs, read_job_count) ||
            !cuda_stream_selected_upload_read_jobs(read_jobs, read_job_count)) {
            cuda_stream_read_jobs_free(read_jobs, read_job_count);
            cuda_stream_resident_cache_release();
            return 0;
        }
    } else {
        cudaError_t err = cudaStreamSynchronize(g_stream_selected_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming selected upload sync failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            cuda_stream_read_jobs_free(read_jobs, read_job_count);
            cuda_stream_resident_cache_release();
            return 0;
        }
    }
    cuda_stream_read_jobs_free(read_jobs, read_job_count);

    {
        const uint32_t all_mask =
            n_selected >= 32u ? 0xffffffffu : ((1u << n_selected) - 1u);
        const uint32_t compact_mask = resident_mask != 0 ? missing_mask : all_mask;
        if (!cuda_stream_selected_compact_mask(model_map,
                                               layer,
                                               selected_ids,
                                               n_total_expert,
                                               n_selected,
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes,
                                               compact_mask)) {
            cuda_stream_resident_cache_release();
            return 0;
        }
    }

    g_stream_selected_cache.loaded = 1;
    return 1;
}

static int cuda_stream_selected_pending_matches(
        const void *model_map,
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!g_stream_selected_pending.active ||
        g_routed_moe_selected_override_n != n_selected ||
        g_stream_selected_pending.model_map != model_map ||
        g_stream_selected_pending.layer != layer ||
        g_stream_selected_pending.n_total_expert != n_total_expert ||
        g_stream_selected_pending.n_selected != n_selected ||
        g_stream_selected_pending.gate_expert_bytes != gate_expert_bytes ||
        g_stream_selected_pending.down_expert_bytes != down_expert_bytes) {
        return 0;
    }
    for (uint32_t i = 0; i < n_selected; i++) {
        if (g_stream_selected_pending.selected_ids[i] !=
            g_routed_moe_selected_override[i]) {
            return 0;
        }
    }
    return 1;
}

static int cuda_stream_selected_finish_pending_missing(uint32_t compact_mask);

static int cuda_stream_selected_apply_split(
        const void *model_map,
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        const ds4_gpu_tensor **selected_exec,
        const char **gate_w,
        const char **up_w,
        const char **down_w,
        const char ***gate_ptrs,
        const char ***up_ptrs,
        const char ***down_ptrs,
        uint32_t *resident_mask,
        uint32_t *missing_mask) {
    if (!g_ssd_streaming_mode ||
        !selected_exec ||
        !gate_w ||
        !up_w ||
        !down_w ||
        !gate_ptrs ||
        !up_ptrs ||
        !down_ptrs ||
        !resident_mask ||
        !missing_mask ||
        !cuda_stream_selected_pending_matches(model_map,
                                              layer,
                                              n_total_expert,
                                              n_selected,
                                              gate_expert_bytes,
                                              down_expert_bytes)) {
        return 0;
    }
    if ((g_stream_selected_pending.resident_mask |
         g_stream_selected_pending.missing_mask) == 0 ||
        !g_stream_selected_cache.gate_ptrs ||
        !g_stream_selected_cache.up_ptrs ||
        !g_stream_selected_cache.down_ptrs) {
        return 0;
    }
    *selected_exec = &g_stream_selected_cache.slot_tensor;
    *gate_w = g_stream_selected_cache.gate;
    *up_w = g_stream_selected_cache.up;
    *down_w = g_stream_selected_cache.down;
    *gate_ptrs = g_stream_selected_cache.gate_ptrs;
    *up_ptrs = g_stream_selected_cache.up_ptrs;
    *down_ptrs = g_stream_selected_cache.down_ptrs;
    *resident_mask = g_stream_selected_pending.resident_mask;
    *missing_mask = g_stream_selected_pending.missing_mask;
    g_routed_moe_selected_override_n = 0;
    return 1;
}

static int cuda_stream_selected_finish_pending_missing(uint32_t compact_mask) {
    if (!g_stream_selected_pending.active) return 1;
    const uint32_t read_job_count = g_stream_selected_pending.read_job_count;
    if (!cuda_stream_read_jobs_wait(g_stream_selected_pending.read_jobs,
                                    read_job_count) ||
        !cuda_stream_selected_upload_read_jobs(g_stream_selected_pending.read_jobs,
                                              read_job_count)) {
        cuda_stream_read_jobs_free(g_stream_selected_pending.read_jobs,
                                   read_job_count);
        memset(&g_stream_selected_pending, 0,
               sizeof(g_stream_selected_pending));
        cuda_stream_resident_cache_release();
        return 0;
    }
    cuda_stream_read_jobs_free(g_stream_selected_pending.read_jobs,
                               read_job_count);
    if (compact_mask != 0 &&
        !cuda_stream_selected_compact_mask(
                g_stream_selected_pending.model_map,
                g_stream_selected_pending.layer,
                g_stream_selected_pending.selected_ids,
                g_stream_selected_pending.n_total_expert,
                g_stream_selected_pending.n_selected,
                g_stream_selected_pending.gate_offset,
                g_stream_selected_pending.up_offset,
                g_stream_selected_pending.down_offset,
                g_stream_selected_pending.gate_expert_bytes,
                g_stream_selected_pending.down_expert_bytes,
                compact_mask)) {
        memset(&g_stream_selected_pending, 0, sizeof(g_stream_selected_pending));
        cuda_stream_resident_cache_release();
        return 0;
    }
    g_stream_selected_cache.loaded = compact_mask != 0 ? 1 : 0;
    memset(&g_stream_selected_pending, 0, sizeof(g_stream_selected_pending));
    return 1;
}

static int cuda_stream_selected_apply(
        const void *model_map,
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        const ds4_gpu_tensor **selected_exec,
        const char **gate_w,
        const char **up_w,
        const char **down_w) {
    if (g_ssd_streaming_mode &&
        !g_stream_selected_cache.loaded &&
        getenv("DS4_ROCM_DISABLE_STREAMING_SPLIT_SELECTED") != NULL &&
        cuda_stream_selected_pending_matches(model_map,
                                             layer,
                                             n_total_expert,
                                             n_selected,
                                             gate_expert_bytes,
                                             down_expert_bytes)) {
        const uint32_t compact_mask =
            g_stream_selected_pending.resident_mask |
            g_stream_selected_pending.missing_mask;
        if (!cuda_stream_selected_finish_pending_missing(compact_mask)) {
            return 0;
        }
    }
    if (!g_ssd_streaming_mode ||
        !g_stream_selected_cache.loaded ||
        !selected_exec ||
        !gate_w ||
        !up_w ||
        !down_w ||
        g_routed_moe_selected_override_n != n_selected ||
        g_stream_selected_cache.model_map != model_map ||
        g_stream_selected_cache.layer != layer ||
        g_stream_selected_cache.n_total_expert != n_total_expert ||
        g_stream_selected_cache.n_selected != n_selected ||
        g_stream_selected_cache.gate_expert_bytes != gate_expert_bytes ||
        g_stream_selected_cache.down_expert_bytes != down_expert_bytes) {
        return 0;
    }
    for (uint32_t i = 0; i < n_selected; i++) {
        if (g_stream_selected_cache.selected_ids[i] !=
            g_routed_moe_selected_override[i]) {
            return 0;
        }
    }
    *selected_exec = &g_stream_selected_cache.slot_tensor;
    *gate_w = g_stream_selected_cache.gate;
    *up_w = g_stream_selected_cache.up;
    *down_w = g_stream_selected_cache.down;
    g_routed_moe_selected_override_n = 0;
    return 1;
}

static const char *cuda_model_ptr(const void *model_map, uint64_t offset) {
    const char *owned = cuda_model_image_ptr(model_map, offset);
    if (owned) return owned;
    if (g_q4k_kshard.snapshot_valid && model_map == g_model_host_base) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "Q4_K K-shard refusing uncovered start-only model pointer "
                "at offset %.6f GiB\n",
                (double)offset / 1073741824.0);
        return NULL;
    }
    if (model_map == g_model_host_base && g_model_device_base) return g_model_device_base + offset;
    return (const char *)model_map + offset;
}

static const char *cuda_model_range_copy_uncached(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what) {
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "model range alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "weights", (double)bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    const char *src = (const char *)model_map + offset;
    err = cudaMemcpy(dev, src, (size_t)bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "model range copy failed for %s: %s\n",
                what ? what : "weights", cudaGetErrorString(err));
        (void)cudaFree(dev);
        (void)cudaGetLastError();
        return NULL;
    }
    g_model_ranges.push_back({model_map, offset, bytes, (char *)dev, NULL, NULL, 0, 0, 0});
    g_model_range_bytes += bytes;
    return (const char *)dev;
}

/* DSpark's three MXFP4 expert stages total about 9.56 GiB.  A TP rank has
 * room for one stage, but not all three beside its 80.76 GiB base shard.
 * Drop the previous stage's three independently allocated tables before
 * resolving the next one; small dense support tensors remain cached. */
static int cuda_dspark_select_expert_stage(
        const void *model_map,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset) {
    if (!model_map || model_map != g_support_host_base) return 1;
    if (g_dspark_stage_offsets[0] == gate_offset &&
        g_dspark_stage_offsets[1] == up_offset &&
        g_dspark_stage_offsets[2] == down_offset) return 1;
    if (g_dspark_stage_offsets[0] || g_dspark_stage_offsets[1] ||
        g_dspark_stage_offsets[2]) {
        if (!cuda_ok(cudaDeviceSynchronize(), "DSpark expert-stage switch")) {
            return 0;
        }
        for (cuda_model_range &r : g_model_ranges) {
            if (r.host_base != model_map || r.arena_allocated ||
                !r.device_ptr) continue;
            bool old_stage = false;
            for (uint32_t i = 0; i < 3; i++) {
                if (r.offset == g_dspark_stage_offsets[i]) old_stage = true;
            }
            if (!old_stage) continue;
            (void)cudaFree(r.device_ptr);
            if (g_model_range_bytes >= r.bytes) g_model_range_bytes -= r.bytes;
            r.host_base = NULL;
            r.offset = 0;
            r.bytes = 0;
            r.device_ptr = NULL;
        }
    }
    g_dspark_stage_offsets[0] = gate_offset;
    g_dspark_stage_offsets[1] = up_offset;
    g_dspark_stage_offsets[2] = down_offset;
    return 1;
}

static int cuda_dspark_prepare_selected_experts(
        const void *model_map,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint32_t n_total_expert,
        const ds4_gpu_tensor *selected,
        uint64_t pair_count,
        ds4_gpu_tensor *selected_compact,
        const char **gate_out,
        const char **up_out,
        const char **down_out) {
    if (!model_map || model_map != g_support_host_base || !selected ||
        !selected_compact || !gate_out || !up_out || !down_out ||
        pair_count == 0 || pair_count > SIZE_MAX / sizeof(int32_t)) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "DSpark selected expert staging rejected arguments: map=%p "
                "support=%p selected=%p compact=%p pairs=%llu\n",
                model_map, g_support_host_base, (const void *)selected,
                (void *)selected_compact, (unsigned long long)pair_count);
        return 0;
    }
    const char *profile_env = getenv("DS4_DSPARK_STAGE_PROFILE");
    const bool profile = profile_env && profile_env[0] &&
                         strcmp(profile_env, "0") != 0;
    const double profile_t0 = profile ? cuda_wall_sec() : 0.0;
    if (!cuda_ok(cudaDeviceSynchronize(), "DSpark selected expert staging")) return 0;
    const double profile_t_sync = profile ? cuda_wall_sec() : 0.0;
    std::vector<int32_t> ids((size_t)pair_count);
    if (!cuda_ok(cudaMemcpy(ids.data(), selected->ptr,
                            (size_t)pair_count * sizeof(int32_t),
                            cudaMemcpyDeviceToHost),
                 "DSpark selected id readback")) return 0;
    std::vector<int32_t> unique;
    unique.reserve((size_t)pair_count);
    for (uint64_t i = 0; i < pair_count; i++) {
        int32_t id = ids[(size_t)i];
        /* Router padding uses -1 with a zero weight. Preserve that sentinel:
         * the MXFP4 kernels map it to compact slot zero for safe reads, while
         * the zero pair weight makes its gate/up/mid/down contribution zero. */
        if (id < 0) {
            ids[(size_t)i] = -1;
            continue;
        }
        if ((uint32_t)id >= n_total_expert) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "DSpark selected expert id %d at pair %llu is outside "
                    "0..%u\n", id, (unsigned long long)i, n_total_expert);
            return 0;
        }
        auto it = std::find(unique.begin(), unique.end(), id);
        if (it == unique.end()) {
            unique.push_back(id);
            ids[(size_t)i] = (int32_t)unique.size() - 1;
        } else {
            ids[(size_t)i] = (int32_t)(it - unique.begin());
        }
    }
    if (unique.empty()) {
        /* All router slots are padding (-1, zero weight). The caller can
         * materialize the mathematically correct zero routed contribution
         * without staging or reading any expert weights. */
        return 2;
    }
    const double profile_t_route = profile ? cuda_wall_sec() : 0.0;
    uint64_t gate_need = 0, down_need = 0, ids_need = 0;
    if (!cuda_u64_mul_checked(unique.size(), gate_expert_bytes, &gate_need) ||
        !cuda_u64_mul_checked(unique.size(), down_expert_bytes, &down_need) ||
        !cuda_u64_mul_checked(pair_count, sizeof(int32_t), &ids_need)) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "DSpark selected expert staging size overflow: unique=%llu "
                "pairs=%llu gate_eb=%llu down_eb=%llu\n",
                (unsigned long long)unique.size(),
                (unsigned long long)pair_count,
                (unsigned long long)gate_expert_bytes,
                (unsigned long long)down_expert_bytes);
        return 0;
    }
    if (g_dspark_selected_gate_capacity < gate_need) {
        if (g_dspark_selected_gate) (void)cudaFree(g_dspark_selected_gate);
        if (g_dspark_selected_up) (void)cudaFree(g_dspark_selected_up);
        g_dspark_selected_gate = NULL;
        g_dspark_selected_up = NULL;
        if (!cuda_ok(cudaMalloc((void **)&g_dspark_selected_gate, (size_t)gate_need),
                     "DSpark selected gate alloc") ||
            !cuda_ok(cudaMalloc((void **)&g_dspark_selected_up, (size_t)gate_need),
                     "DSpark selected up alloc")) return 0;
        g_dspark_selected_gate_capacity = gate_need;
    }
    if (g_dspark_selected_down_capacity < down_need) {
        if (g_dspark_selected_down) (void)cudaFree(g_dspark_selected_down);
        g_dspark_selected_down = NULL;
        if (!cuda_ok(cudaMalloc((void **)&g_dspark_selected_down, (size_t)down_need),
                     "DSpark selected down alloc")) return 0;
        g_dspark_selected_down_capacity = down_need;
    }
    if (g_dspark_selected_ids_capacity < ids_need) {
        if (g_dspark_selected_ids) (void)cudaFree(g_dspark_selected_ids);
        g_dspark_selected_ids = NULL;
        if (!cuda_ok(cudaMalloc((void **)&g_dspark_selected_ids, (size_t)ids_need),
                     "DSpark selected ids alloc")) return 0;
        g_dspark_selected_ids_capacity = ids_need;
    }
    const char *base = (const char *)model_map;
    /* Upload the tiny compact routing map before scheduling asynchronous
     * weight copies.  Keeping its source in this stack-owned vector avoids a
     * pinned host allocation or a lifetime wait. */
    if (!cuda_ok(cudaMemcpy(g_dspark_selected_ids, ids.data(), (size_t)ids_need,
                            cudaMemcpyHostToDevice),
                 "DSpark selected id upload")) return 0;
    const double profile_t_setup = profile ? cuda_wall_sec() : 0.0;
    bool transfer_done = false;
    bool used_parallel_read = false;
    bool used_resident_d2d = false;
    const char *resident_base = cuda_model_image_range_ptr(
            model_map, 0, g_support_host_size);
    if (resident_base) {
        for (size_t slot = 0; slot < unique.size(); slot++) {
            const uint64_t expert = (uint64_t)(uint32_t)unique[slot];
            if (!cuda_ok(cudaMemcpy(
                             g_dspark_selected_gate + slot * gate_expert_bytes,
                             resident_base + gate_offset +
                                 expert * gate_expert_bytes,
                             (size_t)gate_expert_bytes,
                             cudaMemcpyDeviceToDevice),
                         "DSpark resident selected gate gather") ||
                !cuda_ok(cudaMemcpy(
                             g_dspark_selected_up + slot * gate_expert_bytes,
                             resident_base + up_offset +
                                 expert * gate_expert_bytes,
                             (size_t)gate_expert_bytes,
                             cudaMemcpyDeviceToDevice),
                         "DSpark resident selected up gather") ||
                !cuda_ok(cudaMemcpy(
                             g_dspark_selected_down + slot * down_expert_bytes,
                             resident_base + down_offset +
                                 expert * down_expert_bytes,
                             (size_t)down_expert_bytes,
                             cudaMemcpyDeviceToDevice),
                         "DSpark resident selected down gather")) return 0;
        }
        transfer_done = true;
        used_resident_d2d = true;
    }
#if defined(__HIP_PLATFORM_AMD__) && defined(HIP_VERSION_MAJOR) && HIP_VERSION_MAJOR >= 7
    static int gfx1151 = -1;
    if (gfx1151 < 0) {
        cudaDeviceProp prop;
        int device = 0;
        gfx1151 = cudaGetDevice(&device) == cudaSuccess &&
                  cudaGetDeviceProperties(&prop, device) == cudaSuccess &&
                  strncmp(prop.gcnArchName, "gfx1151", 7) == 0;
    }
#else
    const int gfx1151 = 0;
#endif
    const char *parallel_read_env = getenv("DS4_DSPARK_PARALLEL_READ");
    const bool use_parallel_read = parallel_read_env
            ? parallel_read_env[0] && strcmp(parallel_read_env, "0") != 0
            : gfx1151 != 0;
    if (!transfer_done && use_parallel_read && g_support_fd >= 0) {
        const char *direct_read_env = getenv("DS4_DSPARK_DIRECT_READ");
        const bool use_direct_read = direct_read_env &&
                direct_read_env[0] && strcmp(direct_read_env, "0") != 0;
        static int reader_notice = 0;
        if (!reader_notice) {
            const uint64_t max_expert_bytes =
                gate_expert_bytes > down_expert_bytes ?
                    gate_expert_bytes : down_expert_bytes;
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "DSpark parallel expert reader enabled "
                    "(%s, bounded pinned staging <= %.2f MiB, "
                    "no weight cache)\n",
                    use_direct_read ? "direct-I/O override" : "buffered+evicted",
                    (double)(max_expert_bytes *
                             cuda_stream_read_worker_count()) / 1048576.0);
            reader_notice = 1;
        }
        const size_t job_count = unique.size() * 3u;
        std::vector<cuda_stream_read_job> jobs(job_count);
        size_t job = 0;
        for (size_t slot = 0; slot < unique.size(); slot++) {
            const uint64_t expert = (uint64_t)(uint32_t)unique[slot];
            cuda_stream_read_job *j = &jobs[job++];
            j->dst = g_dspark_selected_gate + slot * gate_expert_bytes;
            j->offset = gate_offset + expert * gate_expert_bytes;
            j->bytes = gate_expert_bytes;
            j = &jobs[job++];
            j->dst = g_dspark_selected_up + slot * gate_expert_bytes;
            j->offset = up_offset + expert * gate_expert_bytes;
            j->bytes = gate_expert_bytes;
            j = &jobs[job++];
            j->dst = g_dspark_selected_down + slot * down_expert_bytes;
            j->offset = down_offset + expert * down_expert_bytes;
            j->bytes = down_expert_bytes;
        }
        for (size_t i = 0; i < job_count; i++) {
            jobs[i].use_own_fd = 1;
            jobs[i].fd = g_support_fd;
            jobs[i].direct_fd = use_direct_read ? g_support_direct_fd : -1;
            jobs[i].file_size = g_support_file_size;
            jobs[i].direct_align = use_direct_read ? g_support_direct_align : 1;
        }
        /* The three expert tables are far apart in the GGUF.  Interleaving
         * gate/up/down per expert turns a bounded read into repeated
         * multi-gigabyte seeks.  Queue by file offset so the worker pool
         * presents monotonic, locally adjacent requests to NVMe. */
        std::sort(jobs.begin(), jobs.end(),
                  [](const cuda_stream_read_job &a,
                     const cuda_stream_read_job &b) {
                      return a.offset < b.offset;
                  });
        if (cuda_stream_read_jobs_parallel(jobs.data(), (uint32_t)job_count) &&
            cuda_stream_selected_upload_read_jobs(jobs.data(),
                                                   (uint32_t)job_count)) {
            transfer_done = true;
            used_parallel_read = true;
        }
        cuda_stream_read_jobs_free(jobs.data(), (uint32_t)job_count);
    }
#if defined(__HIP_PLATFORM_AMD__) && defined(HIP_VERSION_MAJOR) && HIP_VERSION_MAJOR >= 7
    const char *batch_copy_env = getenv("DS4_DSPARK_BATCH_COPY");
    const bool use_batch_copy = batch_copy_env
            ? atoi(batch_copy_env) != 0
            : gfx1151 != 0;
    if (!transfer_done && use_batch_copy) {
        /* ROCm's batch API avoids serializing each mmap-backed gate/up/down
         * range through its own synchronous pageable-memory copy.  It cut the
         * measured DSpark support chain substantially on gfx1151 and adds no
         * persistent weight storage.  The source map outlives the copies; the
         * descriptor vectors are consumed by the submission call itself. */
        if (!cuda_stream_selected_ensure_stream()) return 0;
        const size_t copy_count = unique.size() * 3;
        std::vector<void *> dsts(copy_count);
        std::vector<void *> srcs(copy_count);
        std::vector<size_t> sizes(copy_count);
        size_t copy = 0;
        for (size_t slot = 0; slot < unique.size(); slot++) {
            const uint64_t expert = (uint64_t)(uint32_t)unique[slot];
            dsts[copy] = g_dspark_selected_gate + slot * gate_expert_bytes;
            srcs[copy] = (void *)(base + gate_offset + expert * gate_expert_bytes);
            sizes[copy++] = (size_t)gate_expert_bytes;
            dsts[copy] = g_dspark_selected_up + slot * gate_expert_bytes;
            srcs[copy] = (void *)(base + up_offset + expert * gate_expert_bytes);
            sizes[copy++] = (size_t)gate_expert_bytes;
            dsts[copy] = g_dspark_selected_down + slot * down_expert_bytes;
            srcs[copy] = (void *)(base + down_offset + expert * down_expert_bytes);
            sizes[copy++] = (size_t)down_expert_bytes;
        }
        size_t fail_index = SIZE_MAX;
        hipError_t batch_err = hipMemcpyBatchAsync(
                dsts.data(), srcs.data(), sizes.data(), copy_count,
                NULL, NULL, 0, &fail_index, g_stream_selected_upload_stream);
        if (batch_err != hipSuccess) {
            static int warned = 0;
            if (!warned) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX
                        "DSpark selected expert batch copy unavailable at %llu: "
                        "%s; using synchronous copies\n",
                        (unsigned long long)fail_index,
                        hipGetErrorString(batch_err));
                warned = 1;
            }
            (void)hipGetLastError();
        } else {
            if (!cuda_stream_selected_upload_record_ready()) return 0;
            if (profile) {
                if (!cuda_stream_selected_upload_wait_host(
                            "DSpark selected expert profiled batch copy")) return 0;
            } else if (!cuda_stream_selected_wait_upload_ready()) {
                return 0;
            }
            transfer_done = true;
        }
    }
#endif
    if (!transfer_done) {
        for (size_t slot = 0; slot < unique.size(); slot++) {
            const uint64_t expert = (uint64_t)(uint32_t)unique[slot];
            if (!cuda_ok(cudaMemcpy(g_dspark_selected_gate + slot * gate_expert_bytes,
                                    base + gate_offset + expert * gate_expert_bytes,
                                    (size_t)gate_expert_bytes, cudaMemcpyHostToDevice),
                         "DSpark selected gate copy") ||
                !cuda_ok(cudaMemcpy(g_dspark_selected_up + slot * gate_expert_bytes,
                                    base + up_offset + expert * gate_expert_bytes,
                                    (size_t)gate_expert_bytes, cudaMemcpyHostToDevice),
                         "DSpark selected up copy") ||
                !cuda_ok(cudaMemcpy(g_dspark_selected_down + slot * down_expert_bytes,
                                    base + down_offset + expert * down_expert_bytes,
                                    (size_t)down_expert_bytes, cudaMemcpyHostToDevice),
                         "DSpark selected down copy")) return 0;
        }
    }
    if (profile) {
        const double profile_done = cuda_wall_sec();
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "DSpark expert stage profile unique=%llu pairs=%llu bytes=%.2f MiB "
                "mode=%s sync=%.3f ms route=%.3f ms setup=%.3f ms copy=%.3f ms "
                "total=%.3f ms\n",
                (unsigned long long)unique.size(),
                (unsigned long long)pair_count,
                (double)(gate_need * 2u + down_need) / 1048576.0,
                used_resident_d2d ? "resident-d2d" :
                    (used_parallel_read ? "parallel-read" :
                     (transfer_done ? "batch" : "sync")),
                (profile_t_sync - profile_t0) * 1000.0,
                (profile_t_route - profile_t_sync) * 1000.0,
                (profile_t_setup - profile_t_route) * 1000.0,
                (profile_done - profile_t_setup) * 1000.0,
                (profile_done - profile_t0) * 1000.0);
    }
    selected_compact->ptr = g_dspark_selected_ids;
    selected_compact->bytes = ids_need;
    selected_compact->owner = 0;
    *gate_out = g_dspark_selected_gate;
    *up_out = g_dspark_selected_up;
    *down_out = g_dspark_selected_down;
    return 1;
}

static const char *cuda_model_range_ptr(const void *model_map, uint64_t offset, uint64_t bytes, const char *what) {
    if (cuda_q4k_packed_slice_refuse_linear(model_map, offset, bytes, what)) {
        return NULL;
    }
    if (bytes == 0) return cuda_model_ptr(model_map, offset);
    const char *image_ptr =
        cuda_model_image_range_ptr(model_map, offset, bytes);
    if (image_ptr) return image_ptr;

    const uint64_t end = offset + bytes;
    auto exact = g_model_range_by_offset.find(offset);
    if (exact != g_model_range_by_offset.end()) {
        const cuda_model_range &r = g_model_ranges[exact->second];
        if (r.host_base == model_map && end >= offset && bytes <= r.bytes) return r.device_ptr;
    }
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_base == model_map && offset >= r.offset && end >= offset && end <= r.offset + r.bytes) {
            return r.device_ptr + (offset - r.offset);
        }
        if (r.host_base == model_map && r.host_registered && r.registered_base && r.registered_device_base) {
            const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
            const uintptr_t h1 = h0 + bytes;
            const uintptr_t r0 = (uintptr_t)r.registered_base;
            const uintptr_t r1 = r0 + r.registered_bytes;
            if (h1 >= h0 && h0 >= r0 && h1 <= r1) return r.registered_device_base + (h0 - r0);
        }
    }

    if (g_q4k_kshard.snapshot_valid && model_map == g_model_host_base) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "Q4_K K-shard refusing uncovered model range %s at "
                "offset %.6f GiB bytes=%llu\n",
                what ? what : "weights",
                (double)offset / 1073741824.0,
                (unsigned long long)bytes);
        return NULL;
    }
    const char *fd_ptr = cuda_model_range_ptr_from_fd(model_map, offset, bytes, what);
    if (fd_ptr) return fd_ptr;
    if (g_ssd_streaming_mode && model_map == g_model_host_base) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming model range %s at offset %.2f GiB "
                "is not device-cached; refusing host-pointer fallback\n",
                what ? what : "weights",
                (double)offset / 1073741824.0);
        return NULL;
    }

    if (model_map != g_model_host_base) {
        return cuda_model_range_copy_uncached(model_map, offset, bytes, what);
    }

    cudaError_t err = cudaSuccess;
    int overlapping_host_registration = 0;
    if (g_model_range_mapping_supported && model_map == g_model_host_base) {
        const long page_sz_l = sysconf(_SC_PAGESIZE);
        const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
        const uintptr_t host_addr = (uintptr_t)((const char *)model_map + offset);
        const uintptr_t reg_addr = host_addr & ~(uintptr_t)(page_sz - 1u);
        const uint64_t reg_delta = (uint64_t)(host_addr - reg_addr);
        const uint64_t reg_bytes = (reg_delta + bytes + page_sz - 1u) & ~(page_sz - 1u);
        void *reg_dev = NULL;
        err = cudaHostRegister((void *)reg_addr,
                               (size_t)reg_bytes,
                               cudaHostRegisterMapped | cudaHostRegisterReadOnly);
        if (err == cudaSuccess) {
            err = cudaHostGetDevicePointer(&reg_dev, (void *)reg_addr, 0);
            if (err == cudaSuccess && reg_dev) {
                char *dev_ptr = (char *)reg_dev + reg_delta;
                g_model_ranges.push_back({model_map, offset, bytes, dev_ptr, (void *)reg_addr, (char *)reg_dev, reg_bytes, 1, 0});
                g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
                return dev_ptr;
            }
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model range map pointer failed for %s: %s\n",
                    what ? what : "weights", cudaGetErrorString(err));
            (void)cudaHostUnregister((void *)reg_addr);
            (void)cudaGetLastError();
        } else {
#if defined(__HIP_PLATFORM_AMD__)
            overlapping_host_registration =
                err == hipErrorHostMemoryAlreadyRegistered ||
                err == hipErrorAlreadyMapped;
#else
            overlapping_host_registration =
                err == cudaErrorHostMemoryAlreadyRegistered ||
                err == cudaErrorAlreadyMapped;
#endif
            if (err == cudaErrorNotSupported || err == cudaErrorInvalidValue) g_model_range_mapping_supported = 0;
            (void)cudaGetLastError();
        }
    }

    void *dev = NULL;
    err = cudaMalloc(&dev, (size_t)bytes);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, DS4_GPU_LOG_PREFIX "model range alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "weights", (double)bytes / 1048576.0, cudaGetErrorString(err));
        return NULL;
    }

    const char *src = (const char *)model_map + offset;
    const uint64_t chunk = 64ull * 1024ull * 1024ull;
    void *pinned_stage = NULL;
    uint64_t pinned_stage_bytes = 0;
    if (overlapping_host_registration) {
        pinned_stage_bytes = bytes < chunk ? bytes : chunk;
        err = cudaMallocHost(&pinned_stage, (size_t)pinned_stage_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "overlapping model range staging allocation failed for %s "
                    "(%.2f MiB): %s\n",
                    what ? what : "weights",
                    (double)pinned_stage_bytes / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return NULL;
        }
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "overlapping host-registered model range for %s; "
                "using transient pinned staging\n",
                what ? what : "weights");
    }
    for (uint64_t done = 0; done < bytes; done += chunk) {
        uint64_t n = bytes - done < chunk ? bytes - done : chunk;
        const void *copy_src = src + done;
        if (pinned_stage) {
            memcpy(pinned_stage, copy_src, (size_t)n);
            copy_src = pinned_stage;
        }
        err = cudaMemcpy((char *)dev + done, copy_src, (size_t)n, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model range copy failed for %s at %.2f/%.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)done / 1048576.0,
                    (double)bytes / 1048576.0,
                    cudaGetErrorString(err));
            if (pinned_stage) (void)cudaFreeHost(pinned_stage);
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return NULL;
        }
    }
    if (pinned_stage) (void)cudaFreeHost(pinned_stage);
    g_model_ranges.push_back({model_map, offset, bytes, (char *)dev, NULL, NULL, 0, 0, 0});
    g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
    g_model_range_bytes += bytes;
    return (const char *)dev;
}

static int cuda_model_range_is_cached(const void *model_map, uint64_t offset, uint64_t bytes) {
    if (bytes == 0) return 1;
    if (cuda_model_image_range_ptr(model_map, offset, bytes)) return 1;

    const uint64_t end = offset + bytes;
    if (end < offset) return 0;
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_base == model_map &&
            offset >= r.offset &&
            end <= r.offset + r.bytes) {
            return 1;
        }
        if (r.host_base == model_map &&
            r.host_registered &&
            r.registered_base &&
            r.registered_device_base) {
            const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
            const uintptr_t h1 = h0 + bytes;
            const uintptr_t r0 = (uintptr_t)r.registered_base;
            const uintptr_t r1 = r0 + r.registered_bytes;
            if (h1 >= h0 && h0 >= r0 && h1 <= r1) return 1;
        }
    }
    return 0;
}

static void cuda_q8_f16_cache_release_all(void) {
    for (const cuda_q8_f16_transpose_range &r : g_q8_f16_transpose_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    for (const cuda_q8_f16_range &r : g_q8_f16_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f16_transpose_ranges.clear();
    g_q8_f16_transpose_by_offset.clear();
    g_q8_f16_ranges.clear();
    g_q8_f16_by_offset.clear();
    g_q8_f16_bytes = 0;
}

static int cuda_env_present(const char *env) {
    if (env != NULL) return env[0] != '\0' && strcmp(env, "0") != 0;
    return 0;
}

static uint32_t cuda_rows_per_block_or_default(uint32_t v, uint32_t def) {
    return (v == 1u || v == 2u || v == 4u || v == 8u || v == 16u || v == 32u) ? v : def;
}

static uint32_t cuda_rows_per_block_env_or_default(const char *name, uint32_t def) {
    const char *env = name ? getenv(name) : NULL;
    if (!env || !env[0]) return def;
    char *end = NULL;
    errno = 0;
    unsigned long v = strtoul(env, &end, 10);
    if (end == env || errno != 0) return def;
    return cuda_rows_per_block_or_default((uint32_t)v, def);
}

struct ds4_rocm_runtime_config {
    int initialized;
    int disable_splitk_attn_out_low;
    int disable_shared_gate_up_fused_w32;
    int attention_output_cublas_all;
    int attention_output_q8_a_preq_toktile;
    int moe_gate_up_epilogue_coalesced;
    int shared_down_cublas;
    int glm_grouped_value_project;
    int glm_grouped_qk_low;
    int q8_decode_sharedx_64k;
    int graph_dump;
    uint32_t moe_decode_rpb;
    uint32_t moe_decode_gate_rpb;
    uint32_t moe_decode_down_rpb;
    int oldhip_attention_decode;
};

static ds4_rocm_runtime_config g_rocm_cfg;

static const ds4_rocm_runtime_config *cuda_runtime_config(void) {
    if (!g_rocm_cfg.initialized) {
        g_rocm_cfg.disable_splitk_attn_out_low = !g_quality_mode;
        g_rocm_cfg.disable_shared_gate_up_fused_w32 = !g_quality_mode;
        g_rocm_cfg.attention_output_cublas_all = !g_quality_mode;
        const char *attention_output_q8_a_preq_toktile_env =
            getenv("DS4_ROCM_ATTN_OUT_Q8_A_PREQ_TOKTILE");
        /* Token reuse is the validated low-memory prefill path.  Preserve an
         * explicit =0 rollback for architecture and correctness bisects. */
        g_rocm_cfg.attention_output_q8_a_preq_toktile =
            attention_output_q8_a_preq_toktile_env == NULL ||
            cuda_env_present(attention_output_q8_a_preq_toktile_env);
        const char *moe_gate_up_epilogue_coalesced_env =
            getenv("DS4_ROCM_MOE_GATE_UP_EPILOGUE_COALESCED");
        g_rocm_cfg.moe_gate_up_epilogue_coalesced =
            moe_gate_up_epilogue_coalesced_env == NULL ||
            cuda_env_present(moe_gate_up_epilogue_coalesced_env);
        g_rocm_cfg.shared_down_cublas = !g_quality_mode;
        const char *glm_grouped_value_project_env =
            getenv("DS4_ROCM_GLM_GROUPED_VALUE_PROJECT");
        /*
         * The grouped batch kernel is the validated production path.  Keep
         * the environment variable as an explicit =0 correctness fallback.
         */
        g_rocm_cfg.glm_grouped_value_project =
            glm_grouped_value_project_env == NULL ||
            cuda_env_present(glm_grouped_value_project_env);
        const char *glm_grouped_qk_low_env =
            getenv("DS4_ROCM_GLM_GROUPED_QK_LOW");
        g_rocm_cfg.glm_grouped_qk_low =
            glm_grouped_qk_low_env == NULL ||
            cuda_env_present(glm_grouped_qk_low_env);
        const char *q8_decode_sharedx_64k_env =
            getenv("DS4_ROCM_Q8_DECODE_SHAREDX_64K");
        /*
         * The 64 KiB shared-input path is bit-exact on gfx1151 and falls back
         * automatically when a device cannot launch that much dynamic LDS.
         * Keep an explicit =0 diagnostic rollback.
         */
        g_rocm_cfg.q8_decode_sharedx_64k =
            q8_decode_sharedx_64k_env == NULL ||
            cuda_env_present(q8_decode_sharedx_64k_env);
        const int graph_dump_requested =
            cuda_env_present(getenv("DS4_ROCM_GRAPH_DUMP_PREFIX")) ||
            cuda_env_present(getenv("DS4_METAL_GRAPH_DUMP_PREFIX"));
        /* Graph dumps traditionally select conservative kernels so their
         * intermediate tensors are easier to inspect.  Correctness bisects
         * sometimes need the opposite: observe the exact production kernel
         * path without the diagnostic changing it. */
        const int graph_dump_noninvasive =
            cuda_env_present(getenv("DS4_ROCM_GRAPH_DUMP_NONINVASIVE"));
        g_rocm_cfg.graph_dump =
            graph_dump_requested && !graph_dump_noninvasive;
        const char *moe_decode_rpb_env = getenv("DS4_ROCM_MOE_DECODE_RPB");
        const int moe_decode_rpb_env_present =
            moe_decode_rpb_env != NULL && moe_decode_rpb_env[0] != '\0';
        g_rocm_cfg.moe_decode_rpb =
            cuda_rows_per_block_env_or_default("DS4_ROCM_MOE_DECODE_RPB",
                                               g_quality_mode ? 8u :
                                               (g_ssd_streaming_mode ? 2u : 1u));
        g_rocm_cfg.moe_decode_gate_rpb =
            cuda_rows_per_block_env_or_default("DS4_ROCM_MOE_DECODE_GATE_RPB",
                                               (!g_quality_mode &&
                                                g_ssd_streaming_mode &&
                                                !moe_decode_rpb_env_present) ?
                                                    1u : g_rocm_cfg.moe_decode_rpb);
        g_rocm_cfg.moe_decode_down_rpb =
            cuda_rows_per_block_env_or_default("DS4_ROCM_MOE_DECODE_DOWN_RPB",
                                               (!g_quality_mode &&
                                                g_ssd_streaming_mode &&
                                                !moe_decode_rpb_env_present) ?
                                                    2u : g_rocm_cfg.moe_decode_rpb);
        g_rocm_cfg.oldhip_attention_decode = !g_quality_mode;
        g_rocm_cfg.initialized = 1;
    }
    return &g_rocm_cfg;
}

static uint64_t cuda_q8_f16_cache_limit_bytes(void) {
    if (!g_ssd_streaming_mode) return UINT64_MAX;
    const char *env = getenv("DS4_ROCM_STREAM_Q8_F16_CACHE_GB");
    if (env && env[0]) {
        char *end = NULL;
        errno = 0;
        unsigned long long gib = strtoull(env, &end, 10);
        if (end != env && *end == '\0' && errno == 0 &&
            gib <= UINT64_MAX / 1073741824ull) {
            return (uint64_t)gib * 1073741824ull;
        }
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "invalid DS4_ROCM_STREAM_Q8_F16_CACHE_GB=%s; "
                "using automatic q8 fp16 streaming cache limit\n",
                env);
    }

    size_t free_b = 0;
    size_t total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || total_b == 0) {
        (void)cudaGetLastError();
        return 8ull * 1073741824ull;
    }
    (void)free_b;
    uint64_t limit = (uint64_t)total_b / 8ull;
    const uint64_t min_limit = 2ull * 1073741824ull;
    const uint64_t max_limit = 16ull * 1073741824ull;
    if (limit < min_limit) limit = min_limit;
    if (limit > max_limit) limit = max_limit;
    return limit;
}

static uint64_t cuda_q8_f16_cache_reserve_bytes(uint64_t total_bytes) {
    if (g_ssd_streaming_mode) {
        return cuda_stream_resident_free_reserve_bytes();
    }
    if (total_bytes >= 112ull * 1024ull * 1024ull * 1024ull) {
        return 512ull * 1048576ull;
    }

    /* The expanded Q8->F16 cache is only an acceleration path.  Keep enough
     * device memory free for cuBLAS workspaces, transient graph buffers, and
     * driver bookkeeping instead of letting optional cached weights consume the
     * last few GiB on 96 GiB cards. */
    const uint64_t min_reserve = 4096ull * 1048576ull;
    const uint64_t pct_reserve = total_bytes / 20u; /* 5% */
    return pct_reserve > min_reserve ? pct_reserve : min_reserve;
}

static void cuda_q8_f16_cache_budget_notice(
        const char *reason,
        uint64_t request_bytes,
        uint64_t free_bytes,
        uint64_t total_bytes,
        uint64_t reserve_bytes,
        uint64_t limit_bytes) {
    if (g_q8_f16_budget_notice_printed) return;
    g_q8_f16_budget_notice_printed = 1;
    if (limit_bytes != UINT64_MAX && free_bytes == 0 && total_bytes == 0 && reserve_bytes == 0) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB limit=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)limit_bytes / 1073741824.0);
    } else if (limit_bytes == UINT64_MAX) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB free=%.2f GiB reserve=%.2f GiB total=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)free_bytes / 1073741824.0,
                (double)reserve_bytes / 1073741824.0,
                (double)total_bytes / 1073741824.0);
    } else {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB limit=%.2f GiB free=%.2f GiB reserve=%.2f GiB total=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)limit_bytes / 1073741824.0,
                (double)free_bytes / 1073741824.0,
                (double)reserve_bytes / 1073741824.0,
                (double)total_bytes / 1073741824.0);
    }
}

static int cuda_q8_f16_cache_has_budget(uint64_t request_bytes, const char *label) {
    (void)label;
    const uint64_t limit = cuda_q8_f16_cache_limit_bytes();
    if (limit == 0) return 0;
    if (g_q8_f16_bytes > limit || request_bytes > limit - g_q8_f16_bytes) {
        cuda_q8_f16_cache_budget_notice("limit reached", request_bytes, 0, 0, 0, limit);
        return 0;
    }

    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "q8 fp16 cache memory query failed: %s; using q8 kernels\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    const uint64_t free_bytes = (uint64_t)free_b;
    const uint64_t total_bytes = (uint64_t)total_b;
    const uint64_t reserve_bytes = cuda_q8_f16_cache_reserve_bytes(total_bytes);
    if (request_bytes > free_bytes ||
        free_bytes - request_bytes < reserve_bytes) {
        cuda_q8_f16_cache_budget_notice("budget exhausted", request_bytes,
                                        free_bytes, total_bytes,
                                        reserve_bytes, limit);
        return 0;
    }
    return 1;
}

static void cuda_q8_f16_cache_disable_after_failure(const char *what, uint64_t request_bytes) {
    if (!g_q8_f16_disabled_after_oom) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "q8 fp16 cache disabled after %s "
                "(request=%.2f MiB cached=%.2f GiB); using q8 kernels\n",
                what ? what : "allocation failure",
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0);
    }
    g_q8_f16_disabled_after_oom = 1;
    if (!g_q8_f16_ranges.empty() || !g_q8_f16_transpose_ranges.empty()) {
        (void)cudaDeviceSynchronize();
        cuda_q8_f16_cache_release_all();
    }
    (void)cudaGetLastError();
}

static int cuda_q8_f16_cache_opted_in(void) {
    static int enabled = -1;
    if (enabled >= 0) return enabled;
    const char *s = getenv("DS4_ROCM_ENABLE_Q8_F16_CACHE");
    enabled = s && strcmp(s, "1") == 0;
    if (enabled) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX
                "WARNING: DS4_ROCM_ENABLE_Q8_F16_CACHE=1 enables an optional "
                "Q8->F16 weight cache that may consume about 10 GiB per rank "
                "and drive reported VRAM usage near 99%%\n");
    } else if (s && s[0] != '\0') {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX
                "ignoring DS4_ROCM_ENABLE_Q8_F16_CACHE=%s; only exact value 1 "
                "enables the memory-heavy optional cache\n",
                s);
    }
    return enabled;
}

static int cuda_q8_f16_cache_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (g_quality_mode) return 0;
    if (g_q8_f16_disabled_after_oom) return 0;
    if (g_q8_f16_disabled_for_multi_model) return 0;
    if (getenv("DS4_CUDA_NO_Q8_F16_CACHE") != NULL) return 0;
    if (!cuda_q8_f16_cache_opted_in()) return 0;
    if (!label) return 0;
    if (strstr(label, "attn_output_a") != NULL ||
        strstr(label, "attn_output_b") != NULL ||
        strstr(label, "attention_output_a") != NULL ||
        strstr(label, "attention_output_b") != NULL) {
        return 1;
    }
    if (strstr(label, "attn_q_b") != NULL) {
        return 1;
    }
    if (strstr(label, "ffn_gate_shexp") != NULL ||
        strstr(label, "ffn_up_shexp") != NULL ||
        strstr(label, "ffn_down_shexp") != NULL) {
        return 1;
    }
    return (in_dim == 4096u && out_dim == 2048u) ||
           (in_dim == 2048u && out_dim == 4096u) ||
           (in_dim == 4096u && out_dim == 1024u) ||
           (in_dim == 4096u && out_dim == 512u) ||
           (in_dim == 1024u && out_dim == 32768u);
}

static int cuda_q8_label_is_attention_output(const char *label) {
    return label &&
           (strstr(label, "attn_output_a") != NULL ||
            strstr(label, "attn_output_b") != NULL ||
            strstr(label, "attention_output_a") != NULL ||
            strstr(label, "attention_output_b") != NULL);
}

static int cuda_q8_f16_preload_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (cuda_q8_label_is_attention_output(label) &&
        !cuda_runtime_config()->attention_output_cublas_all) {
        return 0;
    }
    return cuda_q8_f16_cache_allowed(label, in_dim, out_dim);
}

static const __half *cuda_q8_f16_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        const char *label) {
    auto exact = g_q8_f16_by_offset.find(offset);
    if (exact != g_q8_f16_by_offset.end()) {
        const cuda_q8_f16_range &r = g_q8_f16_ranges[exact->second];
        if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    for (const cuda_q8_f16_range &r : g_q8_f16_ranges) {
        if (r.host_base == model_map && r.offset == offset &&
            r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    if (!cuda_q8_f16_cache_allowed(label, in_dim, out_dim)) return NULL;

    uint64_t out_bytes = 0;
    if (in_dim == 0u || out_dim == 0u ||
        !cuda_u64_mul3_checked(in_dim, out_dim, sizeof(__half), &out_bytes)) return NULL;
    if (!cuda_q8_f16_cache_has_budget(out_bytes, label)) return NULL;

    const char *q8 = cuda_model_range_ptr(model_map, offset, weight_bytes, "q8_0");
    if (!q8) return NULL;

    __half *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)out_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "q8 fp16 cache alloc failed (%.2f MiB): %s\n",
                (double)out_bytes / 1048576.0, cudaGetErrorString(err));
        cuda_q8_f16_cache_disable_after_failure("allocation failure", out_bytes);
        return NULL;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    const uint64_t n = in_dim * out_dim;
    dequant_q8_0_to_f16_kernel<<<(n + 255) / 256, 256>>>(dev,
                                                          (const unsigned char *)q8,
                                                          in_dim,
                                                          out_dim,
                                                          blocks);
    if (!cuda_ok(cudaGetLastError(), "q8 fp16 dequant launch")) {
        (void)cudaFree(dev);
        cuda_q8_f16_cache_disable_after_failure("dequant launch failure", out_bytes);
        return NULL;
    }
    g_q8_f16_ranges.push_back({model_map, offset, weight_bytes, in_dim, out_dim, dev});
    g_q8_f16_by_offset[offset] = g_q8_f16_ranges.size() - 1u;
    g_q8_f16_bytes += out_bytes;
    return dev;
}

static const __half *cuda_q8_f16_transpose_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        const char *label) {
    auto exact = g_q8_f16_transpose_by_offset.find(offset);
    if (exact != g_q8_f16_transpose_by_offset.end()) {
        const cuda_q8_f16_transpose_range &r = g_q8_f16_transpose_ranges[exact->second];
        if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    for (const cuda_q8_f16_transpose_range &r : g_q8_f16_transpose_ranges) {
        if (r.host_base == model_map && r.offset == offset &&
            r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    if (!cuda_q8_f16_cache_allowed(label, in_dim, out_dim)) return NULL;
    uint64_t out_bytes = 0;
    if (in_dim == 0u || out_dim == 0u ||
        !cuda_u64_mul3_checked(in_dim, out_dim, sizeof(__half), &out_bytes)) return NULL;
    if (!cuda_q8_f16_cache_has_budget(out_bytes, label)) return NULL;
    const char *q8 = cuda_model_range_ptr(model_map, offset, weight_bytes, "q8_0");
    if (!q8) return NULL;
    __half *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)out_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "q8 fp16 transpose cache alloc failed (%.2f MiB): %s\n",
                (double)out_bytes / 1048576.0, cudaGetErrorString(err));
        cuda_q8_f16_cache_disable_after_failure("transpose allocation failure", out_bytes);
        return NULL;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t n = in_dim * out_dim;
    dequant_q8_0_to_f16_transpose_kernel<<<(n + 255u) / 256u, 256>>>(dev,
                                                                     (const unsigned char *)q8,
                                                                     in_dim,
                                                                     out_dim,
                                                                     blocks);
    if (!cuda_ok(cudaGetLastError(), "q8 fp16 transpose dequant launch")) {
        (void)cudaFree(dev);
        cuda_q8_f16_cache_disable_after_failure("transpose launch failure", out_bytes);
        return NULL;
    }
    g_q8_f16_transpose_ranges.push_back({model_map, offset, weight_bytes, in_dim, out_dim, dev});
    g_q8_f16_transpose_by_offset[offset] = g_q8_f16_transpose_ranges.size() - 1u;
    g_q8_f16_bytes += out_bytes;
    return dev;
}

static uint32_t cuda_prefill_warmup_tokens(void) {
    uint32_t n_tok = 2048u;
    const char *chunk_env = getenv("DS4_METAL_PREFILL_CHUNK");
    if (chunk_env && chunk_env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(chunk_env, &end, 10);
        if (end != chunk_env && *end == '\0' && v > 0 && v <= 4096u) n_tok = (uint32_t)v;
    }
    return n_tok;
}

static void cuda_q8_f16_warmup_attention_output_a_gemm(const __half *out_a_f16,
                                                       uint64_t group_dim,
                                                       uint64_t rank,
                                                       uint32_t n_groups) {
    static int warmed = 0;
    if (warmed || !g_cublas_ready || !out_a_f16 || group_dim == 0 || rank == 0 || n_groups == 0) return;
    const ds4_rocm_runtime_config *cfg = cuda_runtime_config();
    if (!cfg->attention_output_cublas_all) return;
    warmed = 1;
    const uint32_t n_tok = cuda_prefill_warmup_tokens();
    const uint64_t heads_h_count = (uint64_t)n_groups * n_tok * group_dim;
    const uint64_t low_h_count = (uint64_t)n_groups * n_tok * rank;
    const uint64_t heads_h_bytes = heads_h_count * sizeof(__half);
    const uint64_t low_h_off = (heads_h_bytes + 255ull) & ~255ull;
    if (low_h_count > (UINT64_MAX - low_h_off) / sizeof(__half)) return;
    void *tmp = cuda_tmp_alloc(low_h_off + low_h_count * sizeof(__half), "attention output a warmup");
    if (!tmp) return;
    __half *heads_h = (__half *)tmp;
    __half *low_h = (__half *)((char *)tmp + low_h_off);
    if (cudaMemset(heads_h, 0, (size_t)heads_h_bytes) != cudaSuccess) return;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmStridedBatchedEx(g_cublas,
                                                   CUBLAS_OP_T,
                                                   CUBLAS_OP_N,
                                                   (int)rank,
                                                   (int)n_tok,
                                                   (int)group_dim,
                                                   &alpha,
                                                   out_a_f16,
                                                   CUDA_R_16F,
                                                   (int)group_dim,
                                                   (long long)rank * (long long)group_dim,
                                                   heads_h,
                                                   CUDA_R_16F,
                                                   (int)group_dim,
                                                   (long long)n_tok * (long long)group_dim,
                                                   &beta,
                                                   low_h,
                                                   CUDA_R_16F,
                                                   (int)(n_groups * rank),
                                                   (long long)rank,
                                                   (int)n_groups,
                                                   CUBLAS_COMPUTE_32F,
                                                   CUBLAS_GEMM_DEFAULT);
    if (st == CUBLAS_STATUS_SUCCESS) (void)cudaDeviceSynchronize();
}

static void cuda_q8_f16_warmup_attention_output_b_gemm(const __half *out_b_f16_t,
                                                       uint64_t low_dim,
                                                       uint64_t out_dim) {
    static int warmed = 0;
    if (warmed || !g_cublas_ready || !out_b_f16_t || low_dim == 0 || out_dim == 0) return;
    if (!cuda_runtime_config()->attention_output_cublas_all) return;
    warmed = 1;
    const uint32_t n_tok = cuda_prefill_warmup_tokens();
    const uint64_t low_h_count = (uint64_t)n_tok * low_dim;
    const uint64_t out_count = (uint64_t)n_tok * out_dim;
    const uint64_t low_h_bytes = low_h_count * sizeof(__half);
    const uint64_t out_off = (low_h_bytes + 255ull) & ~255ull;
    if (out_count > (UINT64_MAX - out_off) / sizeof(float)) return;
    void *tmp = cuda_tmp_alloc(out_off + out_count * sizeof(float), "attention output b warmup");
    if (!tmp) return;
    __half *low_h = (__half *)tmp;
    float *out = (float *)((char *)tmp + out_off);
    if (cudaMemset(low_h, 0, (size_t)low_h_bytes) != cudaSuccess) return;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(g_cublas,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     (int)out_dim,
                                     (int)n_tok,
                                     (int)low_dim,
                                     &alpha,
                                     out_b_f16_t,
                                     CUDA_R_16F,
                                     (int)out_dim,
                                     low_h,
                                     CUDA_R_16F,
                                     (int)low_dim,
                                     &beta,
                                     out,
                                     CUDA_R_32F,
                                     (int)out_dim,
                                     CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT);
    if (st == CUBLAS_STATUS_SUCCESS) (void)cudaDeviceSynchronize();
}

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "%s failed: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static double cuda_wall_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static int cuda_model_load_progress_enabled(void) {
    return 1;
}

static void cuda_model_load_progress_reset(void) {
    g_model_load_progress_next = 0;
    g_model_load_progress_last = 0.0;
    g_model_load_progress_started = 0;
    g_model_load_progress_tty = 0;
}

static void cuda_model_load_progress_note(uint64_t cached_bytes) {
    if (!cuda_model_load_progress_enabled()) return;

    const double now = cuda_wall_sec();
    if (!g_model_load_progress_started) {
        g_model_load_progress_started = 1;
        g_model_load_progress_tty = isatty(STDERR_FILENO) != 0;
        g_model_load_progress_next = (g_model_load_progress_tty ? 2ull : 16ull) *
                                     1024ull * 1024ull * 1024ull;
        g_model_load_progress_last = now;
        if (g_model_load_progress_tty) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "loading model tensors into device cache: 0.00 GiB");
        } else {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "loading model tensors into device cache\n");
        }
    }

    if (cached_bytes < g_model_load_progress_next &&
        now - g_model_load_progress_last < (g_model_load_progress_tty ? 2.0 : 10.0)) {
        return;
    }

    if (g_model_load_progress_tty) {
        fprintf(stderr, "\r" DS4_GPU_LOG_PREFIX "loading model tensors into device cache: %.2f GiB",
                (double)cached_bytes / 1073741824.0);
    } else {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "loading model tensors %.2f GiB cached\n",
                (double)cached_bytes / 1073741824.0);
    }
    fflush(stderr);
    g_model_load_progress_last = now;
    const uint64_t step = (g_model_load_progress_tty ? 2ull : 16ull) *
                          1024ull * 1024ull * 1024ull;
    while (g_model_load_progress_next <= cached_bytes) {
        g_model_load_progress_next += step;
    }
}

static uint64_t cuda_model_copy_chunk_bytes(void) {
    return 64ull * 1048576ull;
}

static void cuda_model_discard_source_pages(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes) {
#if defined(POSIX_MADV_DONTNEED)
    if (!model_map || bytes == 0 || offset > model_size) return;
    if (bytes > model_size - offset) bytes = model_size - offset;
    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
    const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
    const uintptr_t h1 = h0 + bytes;
    const uintptr_t p0 = h0 & ~(uintptr_t)(page_sz - 1u);
    const uintptr_t p1 = (h1 + page_sz - 1u) & ~(uintptr_t)(page_sz - 1u);
    if (p1 > p0) (void)posix_madvise((void *)p0, (size_t)(p1 - p0), POSIX_MADV_DONTNEED);
#else
    (void)model_map;
    (void)model_size;
    (void)offset;
    (void)bytes;
#endif
}

static void cuda_model_drop_file_pages(uint64_t offset, uint64_t bytes) {
#if defined(POSIX_FADV_DONTNEED)
    if (g_model_fd < 0 || bytes == 0) return;
    (void)posix_fadvise(g_model_fd, (off_t)offset, (off_t)bytes, POSIX_FADV_DONTNEED);
#else
    (void)offset;
    (void)bytes;
#endif
}

static uint64_t cuda_round_down(uint64_t v, uint64_t align) {
    if (align <= 1) return v;
    return (v / align) * align;
}

static uint64_t cuda_round_up(uint64_t v, uint64_t align) {
    if (align <= 1) return v;
    const uint64_t rem = v % align;
    return rem == 0 ? v : v + (align - rem);
}

/* Discard only complete pages wholly contained by a selected expert window.
 * Rounding outward would evict an adjacent expert's boundary pages and, for
 * ROW_RANGE, the peer row half.  A K_RANGE window necessarily spans complete
 * source rows, including the locally unused peer column half. */
static void cuda_model_discard_source_full_pages(
        const void *model_map, uint64_t model_size,
        uint64_t offset, uint64_t bytes) {
    if (!model_map || bytes == 0u || offset > model_size) return;
    if (bytes > model_size - offset) bytes = model_size - offset;
    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
    const uint64_t first = cuda_round_up(offset, page_sz);
    const uint64_t last = cuda_round_down(offset + bytes, page_sz);
    if (last <= first) return;
#if defined(POSIX_FADV_DONTNEED)
    if (g_model_fd >= 0) {
        (void)posix_fadvise(g_model_fd, (off_t)first,
                            (off_t)(last - first), POSIX_FADV_DONTNEED);
    }
#endif
#if defined(POSIX_MADV_DONTNEED)
    (void)posix_madvise((char *)model_map + first,
                        (size_t)(last - first), POSIX_MADV_DONTNEED);
#endif
}

static void *cuda_align_ptr(void *ptr, uint64_t align) {
    if (align <= 1) return ptr;
    uintptr_t p = (uintptr_t)ptr;
    uintptr_t a = (uintptr_t)align;
    return (void *)(((p + a - 1u) / a) * a);
}

static int cuda_model_stage_pool_alloc(uint64_t bytes) {
    if (g_model_stage_bytes >= bytes) return 1;
    for (size_t i = 0; i < 4; i++) {
        if (g_model_stage_event[i]) {
            (void)cudaEventDestroy(g_model_stage_event[i]);
            g_model_stage_event[i] = NULL;
        }
        if (g_model_stage_raw[i]) {
            (void)cudaFreeHost(g_model_stage_raw[i]);
            g_model_stage_raw[i] = NULL;
            g_model_stage[i] = NULL;
        }
    }
    g_model_stage_bytes = 0;
    if (!g_model_upload_stream) {
        cudaError_t err = cudaStreamCreateWithFlags(&g_model_upload_stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model upload stream creation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    uint64_t alloc_bytes = bytes;
    if (g_model_direct_align > 1u) {
        const uint64_t pad = g_model_direct_align - 1u;
        if (alloc_bytes > UINT64_MAX - pad) return 0;
        alloc_bytes += pad;
    }
    if (alloc_bytes > (uint64_t)SIZE_MAX) return 0;
    for (size_t i = 0; i < 4; i++) {
        cudaError_t err = cudaMallocHost(&g_model_stage_raw[i], (size_t)alloc_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "pinned model staging allocation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_model_stage[i] = cuda_align_ptr(g_model_stage_raw[i], g_model_direct_align);
        err = cudaEventCreateWithFlags(&g_model_stage_event[i], cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model staging event creation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    g_model_stage_bytes = bytes;
    return 1;
}

static int cuda_pread_full(int fd, void *buf, uint64_t bytes, uint64_t offset) {
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n_req = (bytes - done > (uint64_t)SSIZE_MAX) ? (size_t)SSIZE_MAX : (size_t)(bytes - done);
        ssize_t n = pread(fd, (char *)buf + done, n_req, (off_t)(offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (n == 0) return 0;
        done += (uint64_t)n;
    }
    return 1;
}

static int cuda_model_stage_read(void *stage, uint64_t stage_bytes,
                                 uint64_t offset, uint64_t bytes,
                                 const char **payload) {
    *payload = (const char *)stage;
#if defined(__linux__) && defined(O_DIRECT)
    if (g_model_direct_fd >= 0 && g_model_direct_align > 1 && g_model_file_size != 0) {
        const uint64_t aligned_off = cuda_round_down(offset, g_model_direct_align);
        const uint64_t delta = offset - aligned_off;
        uint64_t read_size = cuda_round_up(delta + bytes, g_model_direct_align);
        if (aligned_off <= g_model_file_size &&
            read_size <= stage_bytes &&
            read_size <= g_model_file_size - aligned_off) {
            const int saved_errno = errno;
            errno = 0;
            if (cuda_pread_full(g_model_direct_fd, stage, read_size, aligned_off)) {
                *payload = (const char *)stage + delta;
                errno = saved_errno;
                return 1;
            }
            const int direct_errno = errno;
            if (direct_errno == EINVAL || direct_errno == EFAULT || direct_errno == ENOTSUP || direct_errno == EOPNOTSUPP) {
                (void)close(g_model_direct_fd);
                g_model_direct_fd = -1;
                g_model_direct_align = 1;
            }
            errno = direct_errno;
        }
    }
#else
    (void)stage_bytes;
#endif
    return cuda_pread_full(g_model_fd, stage, bytes, offset);
}

static int cuda_q4k_kshard_restore_borrowed(void) {
    if (g_q4k_kshard_borrowed_images.empty()) return 1;
    if (g_model_fd < 0 ||
        g_model_fd_host_base != g_q4k_kshard.model_map) return 0;
    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    uint64_t stage_bytes = chunk;
    if (!cuda_u64_add_checked(
            stage_bytes,
            g_model_direct_align > 1u ? g_model_direct_align : 1u,
            &stage_bytes) || !cuda_model_stage_pool_alloc(stage_bytes)) {
        return 0;
    }
    uint64_t chunk_index = 0;
    for (const cuda_model_image &img : g_q4k_kshard_borrowed_images) {
        uint64_t copied = 0;
        while (copied < img.size) {
            const uint64_t n = std::min(chunk, img.size - copied);
            const uint32_t bi = (uint32_t)(chunk_index % 4u);
            cudaError_t err = cudaSuccess;
            if (chunk_index >= 4u &&
                (err = cudaEventSynchronize(g_model_stage_event[bi])) !=
                    cudaSuccess) return 0;
            const char *payload = NULL;
            if (!cuda_model_stage_read(g_model_stage[bi],
                                       g_model_stage_bytes,
                                       img.device_offset + copied, n,
                                       &payload)) return 0;
            err = cudaMemcpyAsync(img.device_ptr + copied, payload, (size_t)n,
                                  cudaMemcpyHostToDevice,
                                  g_model_upload_stream);
            if (err != cudaSuccess ||
                cudaEventRecord(g_model_stage_event[bi],
                                g_model_upload_stream) != cudaSuccess) {
                return 0;
            }
            copied += n;
            ++chunk_index;
        }
    }
    return cudaStreamSynchronize(g_model_upload_stream) == cudaSuccess;
}

extern "C" int ds4_gpu_q4k_packed_slice_declare(
        const void *model_map, uint64_t model_size, uint64_t tensor_offset,
        uint32_t n_expert, uint32_t source_rows,
        uint64_t source_row_bytes, uint32_t row_base, uint32_t row_count,
        uint64_t column_byte_base, uint64_t column_byte_count,
        ds4_gpu_q4k_packed_slice_kind kind) {
    if (!model_map || model_size == 0u || n_expert == 0u ||
        n_expert > DS4_ROCM_MAX_N_EXPERT || source_rows == 0u ||
        source_row_bytes == 0u ||
        (source_row_bytes % sizeof(cuda_block_q4_K)) != 0u ||
        row_count == 0u || row_base > source_rows ||
        row_count > source_rows - row_base ||
        column_byte_count == 0u ||
        (column_byte_base % sizeof(cuda_block_q4_K)) != 0u ||
        (column_byte_count % sizeof(cuda_block_q4_K)) != 0u ||
        column_byte_base > source_row_bytes ||
        column_byte_count > source_row_bytes - column_byte_base ||
        (kind != DS4_GPU_Q4K_PACKED_ROW_RANGE &&
         kind != DS4_GPU_Q4K_PACKED_K_RANGE)) return 0;
    if (kind == DS4_GPU_Q4K_PACKED_ROW_RANGE &&
        (column_byte_base != 0u ||
         column_byte_count != source_row_bytes ||
         (row_base % CUDA_QK_K) != 0u ||
         (row_count % CUDA_QK_K) != 0u)) return 0;
    if (kind == DS4_GPU_Q4K_PACKED_K_RANGE &&
        (row_base != 0u || row_count != source_rows)) return 0;

    uint64_t source_expert_bytes = 0, source_tensor_bytes = 0;
    uint64_t packed_expert_bytes = 0, packed_bytes = 0;
    if (!cuda_u64_mul_checked(source_rows, source_row_bytes,
                              &source_expert_bytes) ||
        !cuda_u64_mul_checked(n_expert, source_expert_bytes,
                              &source_tensor_bytes) ||
        !cuda_u64_mul_checked(row_count, column_byte_count,
                              &packed_expert_bytes) ||
        !cuda_u64_mul_checked(n_expert, packed_expert_bytes,
                              &packed_bytes) ||
        packed_bytes > (uint64_t)SIZE_MAX ||
        tensor_offset > model_size ||
        source_tensor_bytes > model_size - tensor_offset) return 0;

    if (cuda_q4k_linear_residency_intersection(
            model_map, tensor_offset, source_tensor_bytes)) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "packed Q4_K declaration refused because the source tensor "
                "already has linear device residency\n");
        return 0;
    }

    cuda_q4k_packed_slice *existing = cuda_q4k_packed_slice_find(
        model_map, tensor_offset, row_base, row_count,
        column_byte_base, column_byte_count);
    if (existing) {
        return existing->model_size == model_size &&
               existing->n_expert == n_expert &&
               existing->source_rows == source_rows &&
               existing->source_row_bytes == source_row_bytes &&
               existing->kind == kind;
    }
    const size_t index = g_q4k_packed_slices.size();
    g_q4k_packed_slices.push_back({
        model_map, model_size, tensor_offset, source_tensor_bytes,
        source_expert_bytes, source_row_bytes, packed_expert_bytes,
        packed_bytes, column_byte_base, column_byte_count, n_expert,
        source_rows, row_base, row_count, kind, NULL, 0, 0, 0});
    g_q4k_packed_by_offset.emplace(tensor_offset, index);
    return 1;
}

extern "C" int ds4_gpu_q4k_packed_slice_load(
        const void *model_map, uint64_t tensor_offset,
        uint32_t row_base, uint32_t row_count,
        uint64_t column_byte_base, uint64_t column_byte_count) {
    cuda_q4k_packed_slice *p = cuda_q4k_packed_slice_find(
        model_map, tensor_offset, row_base, row_count,
        column_byte_base, column_byte_count);
    if (!p) return 0;
    if (p->loaded) return 1;
    if (g_model_fd >= 0 && g_model_fd_host_base != model_map) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "packed Q4_K loader model-fd/map mismatch\n");
        return 0;
    }

    void *dev = p->device_ptr;
    cudaError_t err = cudaSuccess;
    if (!dev) err = cudaMalloc(&dev, (size_t)p->packed_bytes);
    if (err != cudaSuccess || !dev) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "packed Q4_K slice allocation failed (%.2f MiB): %s\n",
                (double)p->packed_bytes / 1048576.0,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    if (!p->device_ptr) {
        p->device_ptr = (char *)dev;
        p->owns_device_ptr = 1;
    }

    const int use_fd_staging = g_model_fd >= 0;
    const int strided = p->column_byte_base != 0u ||
                        p->column_byte_count != p->source_row_bytes;
    uint64_t source_window_bytes = 0;
    if (!cuda_u64_mul_checked(p->row_count, p->source_row_bytes,
                              &source_window_bytes)) {
        if (p->owns_device_ptr) (void)cudaFree(dev);
        p->device_ptr = NULL;
        p->owns_device_ptr = 0;
        return 0;
    }
    void *packed_raw[4] = {};
    char *packed_stage[4] = {};
    void *borrowed_device_stage = NULL;
    std::vector<char> host_pack;
    int ok = 1;

    if (!p->owns_device_ptr) {
        err = cudaMalloc(&borrowed_device_stage,
                         (size_t)p->packed_expert_bytes);
        if (err != cudaSuccess || !borrowed_device_stage) ok = 0;
    }

    if (use_fd_staging) {
        uint64_t stage_need = strided ? source_window_bytes :
                                        p->packed_expert_bytes;
        if (!cuda_u64_add_checked(
                stage_need,
                g_model_direct_align > 1u ? g_model_direct_align : 1u,
                &stage_need) ||
            !cuda_model_stage_pool_alloc(stage_need)) ok = 0;
        if (ok && strided) {
            uint64_t alloc_bytes = p->packed_expert_bytes;
            const uint64_t pad = g_model_direct_align > 1u ?
                                 g_model_direct_align - 1u : 0u;
            if (!cuda_u64_add_checked(alloc_bytes, pad, &alloc_bytes) ||
                alloc_bytes > (uint64_t)SIZE_MAX) ok = 0;
            for (uint32_t i = 0; ok && i < 4u; i++) {
                err = cudaMallocHost(&packed_raw[i], (size_t)alloc_bytes);
                if (err != cudaSuccess) {
                    (void)cudaGetLastError();
                    ok = 0;
                } else {
                    packed_stage[i] = (char *)cuda_align_ptr(
                        packed_raw[i], g_model_direct_align);
                }
            }
        }
    } else if (strided) {
        try {
            host_pack.resize((size_t)p->packed_expert_bytes);
        } catch (...) {
            ok = 0;
        }
    }

    for (uint32_t expert = 0; ok && expert < p->n_expert; expert++) {
        const uint64_t src_offset = p->tensor_offset +
            (uint64_t)expert * p->source_expert_bytes +
            (uint64_t)p->row_base * p->source_row_bytes;
        char *dst = (char *)dev +
                    (uint64_t)expert * p->packed_expert_bytes;
        if (use_fd_staging) {
            const uint32_t bi = expert % 4u;
            if (p->owns_device_ptr && expert >= 4u &&
                (err = cudaEventSynchronize(g_model_stage_event[bi])) !=
                    cudaSuccess) {
                ok = 0;
                break;
            }
            const char *payload = NULL;
            const uint64_t read_bytes = strided ? source_window_bytes :
                                                  p->packed_expert_bytes;
            if (!cuda_model_stage_read(g_model_stage[bi],
                                       g_model_stage_bytes, src_offset,
                                       read_bytes, &payload)) {
                ok = 0;
                break;
            }
            const char *upload = payload;
            if (strided) {
                for (uint32_t row = 0; row < p->row_count; row++) {
                    memcpy(packed_stage[bi] +
                               (uint64_t)row * p->column_byte_count,
                           payload + (uint64_t)row * p->source_row_bytes +
                               p->column_byte_base,
                           (size_t)p->column_byte_count);
                }
                upload = packed_stage[bi];
            }
            if (p->owns_device_ptr) {
                err = cudaMemcpyAsync(dst, upload,
                                      (size_t)p->packed_expert_bytes,
                                      cudaMemcpyHostToDevice,
                                      g_model_upload_stream);
                if (err == cudaSuccess) {
                    err = cudaEventRecord(g_model_stage_event[bi],
                                          g_model_upload_stream);
                }
            } else {
                err = cudaMemcpy(borrowed_device_stage, upload,
                                 (size_t)p->packed_expert_bytes,
                                 cudaMemcpyHostToDevice);
                if (err == cudaSuccess) {
                    const uint32_t threads = 256u;
                    const uint32_t blocks = (uint32_t)(
                        (p->packed_expert_bytes + threads - 1u) / threads);
                    hipLaunchKernelGGL(cuda_copy_bytes_kernel,
                        dim3(blocks), dim3(threads), 0,
                        g_model_upload_stream, dst,
                        (const char *)borrowed_device_stage,
                        p->packed_expert_bytes);
                    err = cudaGetLastError();
                    if (err == cudaSuccess) {
                        err = cudaStreamSynchronize(g_model_upload_stream);
                    }
                }
            }
            if (err != cudaSuccess) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX
                        "packed Q4_K upload failed expert=%u dst=%p "
                        "bytes=%llu: %s\n", expert, (void *)dst,
                        (unsigned long long)p->packed_expert_bytes,
                        cudaGetErrorString(err));
                ok = 0;
                break;
            }
            cuda_model_drop_file_pages(src_offset, read_bytes);
            cuda_model_discard_source_pages(
                model_map, p->model_size, src_offset, read_bytes);
        } else if (!strided) {
            err = cudaMemcpy(dst, (const char *)model_map + src_offset,
                             (size_t)p->packed_expert_bytes,
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess) ok = 0;
        } else {
            const char *src = (const char *)model_map + src_offset;
            for (uint32_t row = 0; row < p->row_count; row++) {
                memcpy(host_pack.data() +
                           (uint64_t)row * p->column_byte_count,
                       src + (uint64_t)row * p->source_row_bytes +
                           p->column_byte_base,
                       (size_t)p->column_byte_count);
            }
            err = cudaMemcpy(dst, host_pack.data(),
                             (size_t)p->packed_expert_bytes,
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess) ok = 0;
        }
    }
    if (use_fd_staging) {
        const cudaError_t sync_err = cudaStreamSynchronize(g_model_upload_stream);
        if (sync_err != cudaSuccess) {
            err = sync_err;
            ok = 0;
        }
    }
    for (uint32_t i = 0; i < 4u; i++) {
        if (packed_raw[i]) (void)cudaFreeHost(packed_raw[i]);
    }
    if (borrowed_device_stage) (void)cudaFree(borrowed_device_stage);
    if (!ok) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "packed Q4_K slice load failed at offset %.2f GiB: %s\n",
                (double)p->tensor_offset / 1073741824.0,
                err == cudaSuccess ? strerror(errno) : cudaGetErrorString(err));
        if (p->owns_device_ptr) (void)cudaFree(dev);
        p->device_ptr = NULL;
        p->owns_device_ptr = 0;
        (void)cudaGetLastError();
        return 0;
    }

    p->loaded = 1;
    g_q4k_packed_slice_bytes += p->packed_bytes;
    cuda_model_load_progress_note(g_model_range_bytes +
                                  g_q4k_packed_slice_bytes);
    fprintf(stderr, DS4_GPU_LOG_PREFIX
            "loaded packed Q4_K routed slice offset=%.2f GiB "
            "rows=%u:%u columns=%llu:%llu experts=%u bytes=%.2f MiB "
            "kind=%s\n",
            (double)p->tensor_offset / 1073741824.0,
            p->row_base, p->row_base + p->row_count,
            (unsigned long long)p->column_byte_base,
            (unsigned long long)(p->column_byte_base +
                                 p->column_byte_count),
            p->n_expert, (double)p->packed_bytes / 1048576.0,
            p->kind == DS4_GPU_Q4K_PACKED_ROW_RANGE ? "row" : "K");
    return 1;
}

static int cuda_q4k_packed_slice_load_expert_impl(
        const void *model_map, uint64_t tensor_offset,
        uint32_t row_base, uint32_t row_count,
        uint64_t column_byte_base, uint64_t column_byte_count,
        uint32_t expert, ds4_gpu_tensor *dst, int synchronize_before,
        cudaStream_t transfer_stream) {
    cuda_q4k_packed_slice *p = cuda_q4k_packed_slice_find(
        model_map, tensor_offset, row_base, row_count,
        column_byte_base, column_byte_count);
    if (!p || !dst || !dst->ptr || expert >= p->n_expert ||
        dst->bytes < p->packed_expert_bytes ||
        (g_model_fd >= 0 && g_model_fd_host_base != model_map)) {
        return 0;
    }

    uint64_t source_window_bytes = 0;
    if (!cuda_u64_mul_checked(p->row_count, p->source_row_bytes,
                              &source_window_bytes)) {
        return 0;
    }
    const uint64_t source_offset = p->tensor_offset +
        (uint64_t)expert * p->source_expert_bytes +
        (uint64_t)p->row_base * p->source_row_bytes;
    if (source_offset > p->model_size ||
        source_window_bytes > p->model_size - source_offset) {
        return 0;
    }

    const char *source = (const char *)model_map + source_offset;
    const int contiguous = p->column_byte_base == 0u &&
                           p->column_byte_count == p->source_row_bytes;
    cudaError_t err = cudaSuccess;
    if (synchronize_before) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "packed Q4_K selected expert reuse wait failed expert=%u: "
                "%s\n", expert, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    const char *async_window = getenv("DS4_ROCM_GLM5_WINDOW_ASYNC");
    const int use_async = async_window && strcmp(async_window, "1") == 0;
    const cudaStream_t stream = transfer_stream ? transfer_stream :
                                (cudaStream_t)0;
    if (contiguous) {
        err = use_async ? cudaMemcpyAsync(
            dst->ptr, source, (size_t)p->packed_expert_bytes,
            cudaMemcpyHostToDevice, stream) :
            cudaMemcpy(dst->ptr, source, (size_t)p->packed_expert_bytes,
                       cudaMemcpyHostToDevice);
    } else {
        /* Preserve the strided Q4_K source layout in the transfer engine.
         * This avoids allocating and filling a multi-MiB CPU temporary for
         * every selected down expert. */
        err = use_async ? hipMemcpy2DAsync(
            dst->ptr, (size_t)p->column_byte_count,
            source + p->column_byte_base, (size_t)p->source_row_bytes,
            (size_t)p->column_byte_count, p->row_count,
            hipMemcpyHostToDevice, stream) :
            hipMemcpy2D(dst->ptr, (size_t)p->column_byte_count,
                        source + p->column_byte_base,
                        (size_t)p->source_row_bytes,
                        (size_t)p->column_byte_count, p->row_count,
                        hipMemcpyHostToDevice);
    }
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "packed Q4_K selected expert upload failed expert=%u "
                "bytes=%llu: %s\n",
                expert, (unsigned long long)p->packed_expert_bytes,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    /* The default keeps the previous bounded-page eviction policy.  For the
     * GLM selected-window diagnosis, allowing the host pages to remain cached
     * separates source fault-in cost from copy/kernel cost without allocating
     * any additional device memory.  Require an explicit value of 1 so a
     * merely-present environment variable cannot silently alter production. */
    const char *keep_source_pages =
        getenv("DS4_ROCM_Q4K_KEEP_SOURCE_PAGES");
    /* An async transfer may still reference the mmap source after this
     * function returns, so eviction is unsafe until the caller's epoch wait.
     * Treat async mode as implicitly keep-pages for correctness; it remains
     * opt-in and does not alter the default path. */
    if ((!keep_source_pages || strcmp(keep_source_pages, "1") != 0) &&
        !use_async) {
        cuda_model_discard_source_full_pages(
            model_map, p->model_size, source_offset, source_window_bytes);
    }
    return 1;
}

extern "C" int ds4_gpu_q4k_packed_slice_load_expert(
        const void *model_map, uint64_t tensor_offset,
        uint32_t row_base, uint32_t row_count,
        uint64_t column_byte_base, uint64_t column_byte_count,
        uint32_t expert, ds4_gpu_tensor *dst) {
    return cuda_q4k_packed_slice_load_expert_impl(
        model_map, tensor_offset, row_base, row_count,
        column_byte_base, column_byte_count, expert, dst, 1, 0);
}

extern "C" ds4_gpu_q4k_window_cache *ds4_gpu_q4k_window_cache_create(
        const ds4_gpu_q4k_window_cache_config *config) {
    /* The cache is layer-local and callers may size it to the complete expert
     * union of one prompt tile.  n_expert remains the absolute allocation
     * bound; no cache survives release_all(). */
    const uint32_t max_slots = config ? config->n_expert : 0u;
    if (!config || !config->model_map || config->n_expert == 0u ||
        config->slots < DS4_ROCM_N_EXPERT_USED ||
        config->slots > max_slots ||
        config->slots > config->n_expert) return NULL;

    cuda_q4k_packed_slice *gate = cuda_q4k_packed_slice_find(
        config->model_map, config->gate_offset,
        config->gate_row_base, config->gate_row_count,
        config->gate_column_byte_base,
        config->gate_column_byte_count);
    cuda_q4k_packed_slice *up = cuda_q4k_packed_slice_find(
        config->model_map, config->up_offset,
        config->gate_row_base, config->gate_row_count,
        config->gate_column_byte_base,
        config->gate_column_byte_count);
    cuda_q4k_packed_slice *down = cuda_q4k_packed_slice_find(
        config->model_map, config->down_offset,
        config->down_row_base, config->down_row_count,
        config->down_column_byte_base,
        config->down_column_byte_count);
    const uint64_t block_bytes = sizeof(cuda_block_q4_K);
    uint64_t expected_down_base = 0, expected_down_count = 0;
    if (!gate || !up || !down || gate == up || gate == down || up == down ||
        gate->kind != DS4_GPU_Q4K_PACKED_ROW_RANGE ||
        up->kind != DS4_GPU_Q4K_PACKED_ROW_RANGE ||
        down->kind != DS4_GPU_Q4K_PACKED_K_RANGE ||
        gate->loaded || up->loaded || down->loaded ||
        gate->n_expert != config->n_expert ||
        up->n_expert != config->n_expert ||
        down->n_expert != config->n_expert ||
        gate->model_size != up->model_size ||
        gate->model_size != down->model_size ||
        gate->source_rows != up->source_rows ||
        gate->source_row_bytes != up->source_row_bytes ||
        gate->row_base != up->row_base ||
        gate->row_count != up->row_count ||
        gate->column_byte_base != 0u ||
        up->column_byte_base != 0u ||
        gate->column_byte_count != gate->source_row_bytes ||
        up->column_byte_count != up->source_row_bytes ||
        gate->packed_expert_bytes != up->packed_expert_bytes ||
        gate->row_count * 2u != gate->source_rows ||
        (gate->row_base != 0u && gate->row_base != gate->row_count) ||
        (gate->row_base % CUDA_QK_K) != 0u ||
        (gate->row_count % CUDA_QK_K) != 0u ||
        down->row_base != 0u ||
        down->row_count != down->source_rows ||
        down->column_byte_count * 2u != down->source_row_bytes ||
        (down->column_byte_base != 0u &&
         down->column_byte_base != down->column_byte_count) ||
        !cuda_u64_mul_checked(gate->row_base / CUDA_QK_K,
                              block_bytes, &expected_down_base) ||
        !cuda_u64_mul_checked(gate->row_count / CUDA_QK_K,
                              block_bytes, &expected_down_count) ||
        down->column_byte_base != expected_down_base ||
        down->column_byte_count != expected_down_count) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "packed Q4_K window cache descriptor coupling refused\n");
        return NULL;
    }

    uint64_t gate_slab = 0, down_slab = 0, gate_pair = 0, total = 0;
    if (!cuda_u64_mul_checked(config->slots, gate->packed_expert_bytes,
                              &gate_slab) ||
        !cuda_u64_mul_checked(config->slots, down->packed_expert_bytes,
                              &down_slab) ||
        !cuda_u64_mul_checked(2u, gate_slab, &gate_pair) ||
        !cuda_u64_add_checked(gate_pair, down_slab, &total) ||
        total > (uint64_t)SIZE_MAX) return NULL;

    ds4_gpu_q4k_window_cache *cache = NULL;
    try {
        cache = new ds4_gpu_q4k_window_cache();
        cache->expert_to_slot.assign(config->n_expert, -1);
        cache->entries.resize(config->slots);
    } catch (...) {
        delete cache;
        return NULL;
    }
    cache->model_map = config->model_map;
    cache->model_size = gate->model_size;
    cache->gate_offset = config->gate_offset;
    cache->up_offset = config->up_offset;
    cache->down_offset = config->down_offset;
    cache->gate_slice_index = (size_t)(gate - g_q4k_packed_slices.data());
    cache->up_slice_index = (size_t)(up - g_q4k_packed_slices.data());
    cache->down_slice_index = (size_t)(down - g_q4k_packed_slices.data());
    cache->gate_expert_bytes = gate->packed_expert_bytes;
    cache->down_expert_bytes = down->packed_expert_bytes;
    cache->capacity_bytes = total;
    cache->n_expert = config->n_expert;
    cache->slots = config->slots;
    cache->copy_stream = (cudaStream_t)0;
    cache->fill_event = (cudaEvent_t)0;
    const char *overlap_env = getenv("DS4_ROCM_GLM5_WINDOW_OVERLAP");
    cache->overlap_enabled = overlap_env && strcmp(overlap_env, "1") == 0;
    cache->prepared_ids = NULL;
    cache->prepared_weights = NULL;
    cache->prepared_count = 0u;
    cache->prepared_valid = 0;
    cache->base = NULL;
    cache->slot_ids_device = NULL;
    cache->slot_ids_capacity = 0u;
    cudaError_t err = cudaMalloc((void **)&cache->base, (size_t)total);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "packed Q4_K window cache allocation failed slots=%u "
                "bytes=%.2f MiB: %s\n", config->slots,
                (double)total / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        delete cache;
        return NULL;
    }
    cache->gate = cache->base;
    cache->up = cache->gate + gate_slab;
    cache->down = cache->up + gate_slab;
    if (cudaMemset(cache->base, 0, (size_t)total) != cudaSuccess) {
        (void)cudaGetLastError();
        (void)cudaFree(cache->base);
        delete cache;
        return NULL;
    }
    if (cache->overlap_enabled &&
        (cudaStreamCreateWithFlags(&cache->copy_stream,
                                   cudaStreamNonBlocking) != cudaSuccess ||
         cudaEventCreateWithFlags(&cache->fill_event,
                                  cudaEventDisableTiming) != cudaSuccess)) {
        if (cache->fill_event) (void)cudaEventDestroy(cache->fill_event);
        if (cache->copy_stream) (void)cudaStreamDestroy(cache->copy_stream);
        (void)cudaFree(cache->base);
        delete cache;
        return NULL;
    }
    try {
        g_q4k_window_caches.push_back(cache);
    } catch (...) {
        if (cache->fill_event) (void)cudaEventDestroy(cache->fill_event);
        if (cache->copy_stream) (void)cudaStreamDestroy(cache->copy_stream);
        (void)cudaFree(cache->base);
        delete cache;
        return NULL;
    }
    return cache;
}

extern "C" int ds4_gpu_q4k_window_cache_rebind(
        ds4_gpu_q4k_window_cache *cache,
        const ds4_gpu_q4k_window_cache_config *config) {
    if (!cache || !config ||
        std::find(g_q4k_window_caches.begin(), g_q4k_window_caches.end(),
                  cache) == g_q4k_window_caches.end() ||
        config->model_map != cache->model_map ||
        config->n_expert != cache->n_expert ||
        config->slots != cache->slots) return 0;
    cuda_q4k_packed_slice *gate = cuda_q4k_packed_slice_find(
        config->model_map, config->gate_offset,
        config->gate_row_base, config->gate_row_count,
        config->gate_column_byte_base, config->gate_column_byte_count);
    cuda_q4k_packed_slice *up = cuda_q4k_packed_slice_find(
        config->model_map, config->up_offset,
        config->gate_row_base, config->gate_row_count,
        config->gate_column_byte_base, config->gate_column_byte_count);
    cuda_q4k_packed_slice *down = cuda_q4k_packed_slice_find(
        config->model_map, config->down_offset,
        config->down_row_base, config->down_row_count,
        config->down_column_byte_base, config->down_column_byte_count);
    if (!gate || !up || !down || gate == up || gate == down || up == down ||
        gate->kind != DS4_GPU_Q4K_PACKED_ROW_RANGE ||
        up->kind != DS4_GPU_Q4K_PACKED_ROW_RANGE ||
        down->kind != DS4_GPU_Q4K_PACKED_K_RANGE ||
        gate->loaded || up->loaded || down->loaded ||
        gate->n_expert != config->n_expert ||
        up->n_expert != config->n_expert ||
        down->n_expert != config->n_expert ||
        gate->model_size != up->model_size ||
        gate->model_size != down->model_size ||
        gate->source_rows != up->source_rows ||
        gate->source_row_bytes != up->source_row_bytes ||
        gate->row_base != up->row_base || gate->row_count != up->row_count ||
        gate->column_byte_base != 0u || up->column_byte_base != 0u ||
        gate->column_byte_count != gate->source_row_bytes ||
        up->column_byte_count != up->source_row_bytes ||
        gate->packed_expert_bytes != up->packed_expert_bytes ||
        gate->packed_expert_bytes != cache->gate_expert_bytes ||
        down->packed_expert_bytes != cache->down_expert_bytes ||
        gate->row_count * 2u != gate->source_rows ||
        (gate->row_base != 0u && gate->row_base != gate->row_count) ||
        (gate->row_base % CUDA_QK_K) != 0u ||
        (gate->row_count % CUDA_QK_K) != 0u || down->row_base != 0u ||
        down->row_count != down->source_rows ||
        down->column_byte_count * 2u != down->source_row_bytes ||
        (down->column_byte_base != 0u &&
         down->column_byte_base != down->column_byte_count) ||
        gate->tensor_offset != config->gate_offset ||
        up->tensor_offset != config->up_offset ||
        down->tensor_offset != config->down_offset) return 0;
    const uint64_t block_bytes = sizeof(cuda_block_q4_K);
    uint64_t expected_down_base = 0u, expected_down_count = 0u;
    if (!cuda_u64_mul_checked(gate->row_base / CUDA_QK_K, block_bytes,
                              &expected_down_base) ||
        !cuda_u64_mul_checked(gate->row_count / CUDA_QK_K, block_bytes,
                              &expected_down_count) ||
        down->column_byte_base != expected_down_base ||
        down->column_byte_count != expected_down_count) return 0;
    /* The scalar scratch consumer performs a terminal synchronize at the end
     * of every routed layer before rebinding this same slab.  In that
     * explicitly opt-in async mode the prior-layer fence is the required
     * lifetime proof, so avoid paying a duplicate device-wide wait here.
     * Retain the conservative wait for every other caller. */
    const char *scratch_async = getenv("DS4_ROCM_GLM5_WINDOW_SCRATCH");
    const char *window_async = getenv("DS4_ROCM_GLM5_WINDOW_ASYNC");
    const bool prior_layer_fenced = scratch_async &&
        strcmp(scratch_async, "1") == 0 && window_async &&
        strcmp(window_async, "1") == 0;
    if (!prior_layer_fenced && cudaDeviceSynchronize() != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    cache->model_size = gate->model_size;
    cache->gate_offset = config->gate_offset;
    cache->up_offset = config->up_offset;
    cache->down_offset = config->down_offset;
    cache->gate_slice_index = (size_t)(gate - g_q4k_packed_slices.data());
    cache->up_slice_index = (size_t)(up - g_q4k_packed_slices.data());
    cache->down_slice_index = (size_t)(down - g_q4k_packed_slices.data());
    std::fill(cache->expert_to_slot.begin(), cache->expert_to_slot.end(), -1);
    for (cuda_q4k_window_cache_entry &entry : cache->entries) {
        entry.expert = -1;
        entry.last_used = 0u;
        entry.valid = 0;
    }
    cache->clock = 0u;
    cache->prepares = cache->hits = cache->misses = 0u;
    cache->fills = cache->evictions = 0u;
    cache->profile_upload_bytes = cache->profile_control_bytes = 0u;
    cache->profile_prepare_sec = 0.0;
    cache->prepared_valid = 0;
    cache->prepared_ids = NULL;
    cache->prepared_weights = NULL;
    cache->prepared_count = 0u;
    return 1;
}

extern "C" void ds4_gpu_q4k_window_cache_destroy(
        ds4_gpu_q4k_window_cache *cache) {
    if (!cache) return;
    const auto found = std::find(g_q4k_window_caches.begin(),
                                 g_q4k_window_caches.end(), cache);
    if (found == g_q4k_window_caches.end()) return;
    (void)cudaDeviceSynchronize();
    if (getenv("DS4_ROCM_GLM5_WINDOW_PROFILE") &&
        cache->profile_prepare_sec > 0.0) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "GLM5 window prepare seconds=%.6f uploads=%.2f MiB "
                "control=%.2f KiB effective=%.2f GiB/s fills=%llu\n",
                cache->profile_prepare_sec,
                (double)cache->profile_upload_bytes / 1048576.0,
                (double)cache->profile_control_bytes / 1024.0,
                (double)cache->profile_upload_bytes /
                    cache->profile_prepare_sec / 1073741824.0,
                (unsigned long long)cache->fills);
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "GLM5 window cache stats prepares=%llu hits=%llu misses=%llu "
                "evictions=%llu slots=%u capacity=%.2f MiB\n",
                (unsigned long long)cache->prepares,
                (unsigned long long)cache->hits,
                (unsigned long long)cache->misses,
                (unsigned long long)cache->evictions,
                cache->slots, (double)cache->capacity_bytes / 1048576.0);
    }
    if (cache->slot_ids_device) (void)cudaFree(cache->slot_ids_device);
    if (cache->fill_event) (void)cudaEventDestroy(cache->fill_event);
    if (cache->copy_stream) (void)cudaStreamDestroy(cache->copy_stream);
    if (cache->base) (void)cudaFree(cache->base);
    g_q4k_window_caches.erase(found);
    delete cache;
}

extern "C" int ds4_gpu_q4k_window_cache_prepare(
        ds4_gpu_q4k_window_cache *cache, const int32_t *expert_ids,
        uint32_t count, int32_t *slot_ids) {
    if (!cache ||
        std::find(g_q4k_window_caches.begin(), g_q4k_window_caches.end(),
                  cache) == g_q4k_window_caches.end() ||
        cache->gate_slice_index >= g_q4k_packed_slices.size() ||
        cache->up_slice_index >= g_q4k_packed_slices.size() ||
        cache->down_slice_index >= g_q4k_packed_slices.size() ||
        !expert_ids || !slot_ids || count == 0u || count > 1048576u) return 0;
    const cuda_q4k_packed_slice &gate_slice =
        g_q4k_packed_slices[cache->gate_slice_index];
    const cuda_q4k_packed_slice &up_slice =
        g_q4k_packed_slices[cache->up_slice_index];
    const cuda_q4k_packed_slice &down_slice =
        g_q4k_packed_slices[cache->down_slice_index];
    if (gate_slice.host_base != cache->model_map ||
        up_slice.host_base != cache->model_map ||
        down_slice.host_base != cache->model_map ||
        gate_slice.packed_expert_bytes != cache->gate_expert_bytes ||
        up_slice.packed_expert_bytes != cache->gate_expert_bytes ||
        down_slice.packed_expert_bytes != cache->down_expert_bytes) return 0;
    std::vector<uint8_t> requested;
    std::vector<int32_t> unique;
    try {
        requested.assign(cache->n_expert, 0u);
        unique.reserve(cache->slots);
    } catch (...) {
        return 0;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const int32_t expert = expert_ids[i];
        if (expert == -1) continue;
        if (expert < -1 || (uint32_t)expert >= cache->n_expert) return 0;
        if (!requested[(uint32_t)expert]) {
            requested[(uint32_t)expert] = 1u;
            unique.push_back(expert);
        }
    }
    if (unique.size() > cache->slots) return 0;
    cache->prepares++;

    bool needs_fill = false;
    for (int32_t expert : unique) {
        if (cache->expert_to_slot[(uint32_t)expert] < 0) {
            needs_fill = true;
            break;
        }
    }
    /* All three tables share the null stream. One epoch-level wait is enough
     * before overwriting cache slots; individual blocking copies remain
     * ordered and published only after gate/up/down all succeed. */
    const char *overlap_wait_env =
        getenv("DS4_ROCM_GLM5_WINDOW_OVERLAP");
    const char *scratch_wait_env =
        getenv("DS4_ROCM_GLM5_WINDOW_SCRATCH");
    const bool prior_layer_fenced = overlap_wait_env &&
        strcmp(overlap_wait_env, "1") == 0 && scratch_wait_env &&
        strcmp(scratch_wait_env, "1") == 0 && cache->overlap_enabled;
    if (needs_fill && !prior_layer_fenced &&
        cudaDeviceSynchronize() != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }

    for (int32_t expert : unique) {
        int32_t slot = cache->expert_to_slot[(uint32_t)expert];
        if (slot >= 0) {
            if ((uint32_t)slot >= cache->slots ||
                !cache->entries[(uint32_t)slot].valid ||
                cache->entries[(uint32_t)slot].expert != expert) return 0;
            cache->entries[(uint32_t)slot].last_used = ++cache->clock;
            cache->hits++;
            continue;
        }
        cache->misses++;
        uint32_t victim = cache->slots;
        uint64_t oldest = UINT64_MAX;
        for (uint32_t s = 0; s < cache->slots; ++s) {
            const cuda_q4k_window_cache_entry &entry = cache->entries[s];
            if (!entry.valid) {
                victim = s;
                break;
            }
            if (entry.expert >= 0 &&
                requested[(uint32_t)entry.expert]) continue;
            if (entry.last_used < oldest) {
                oldest = entry.last_used;
                victim = s;
            }
        }
        if (victim == cache->slots) return 0;
        cuda_q4k_window_cache_entry &entry = cache->entries[victim];
        if (entry.valid) {
            if (entry.expert < 0 ||
                (uint32_t)entry.expert >= cache->n_expert ||
                cache->expert_to_slot[(uint32_t)entry.expert] !=
                    (int32_t)victim) return 0;
            cache->expert_to_slot[(uint32_t)entry.expert] = -1;
            entry.valid = 0;
            cache->evictions++;
        }

        ds4_gpu_tensor gate_dst = {};
        ds4_gpu_tensor up_dst = {};
        ds4_gpu_tensor down_dst = {};
        gate_dst.ptr = cache->gate +
            (uint64_t)victim * cache->gate_expert_bytes;
        gate_dst.bytes = cache->gate_expert_bytes;
        gate_dst.device_id = 0;
        up_dst.ptr = cache->up +
            (uint64_t)victim * cache->gate_expert_bytes;
        up_dst.bytes = cache->gate_expert_bytes;
        up_dst.device_id = 0;
        down_dst.ptr = cache->down +
            (uint64_t)victim * cache->down_expert_bytes;
        down_dst.bytes = cache->down_expert_bytes;
        down_dst.device_id = 0;
        if (!cuda_q4k_packed_slice_load_expert_impl(
                cache->model_map, gate_slice.tensor_offset,
                gate_slice.row_base, gate_slice.row_count,
                gate_slice.column_byte_base,
                gate_slice.column_byte_count,
                (uint32_t)expert, &gate_dst, 0, cache->copy_stream) ||
            !cuda_q4k_packed_slice_load_expert_impl(
                cache->model_map, up_slice.tensor_offset,
                up_slice.row_base, up_slice.row_count,
                up_slice.column_byte_base,
                up_slice.column_byte_count,
                (uint32_t)expert, &up_dst, 0, cache->copy_stream) ||
            !cuda_q4k_packed_slice_load_expert_impl(
                cache->model_map, down_slice.tensor_offset,
                down_slice.row_base, down_slice.row_count,
                down_slice.column_byte_base,
                down_slice.column_byte_count,
                (uint32_t)expert, &down_dst, 0, cache->copy_stream)) {
            entry.expert = -1;
            entry.valid = 0;
            return 0;
        }
        entry.expert = expert;
        entry.last_used = ++cache->clock;
        cache->fills++;
        entry.valid = 1;
        cache->expert_to_slot[(uint32_t)expert] = (int32_t)victim;
    }
    if (cache->fills != 0u) {
        const char *async_window = getenv("DS4_ROCM_GLM5_WINDOW_ASYNC");
        const char *scratch_window =
            getenv("DS4_ROCM_GLM5_WINDOW_SCRATCH");
        const bool same_stream_fenced = async_window &&
            strcmp(async_window, "1") == 0 && scratch_window &&
            strcmp(scratch_window, "1") == 0;
        /* Scratch decode queues the routed consumer on the same stream and
         * the layer epilogue synchronizes before slab rebind.  Avoid a host
         * wait here; retain the old conservative wait for every other arm. */
        if (cache->overlap_enabled && cache->fill_event &&
            cudaEventRecord(cache->fill_event, cache->copy_stream) != cudaSuccess) {
            (void)cudaGetLastError();
            return 0;
        }
        if (!cache->overlap_enabled && !same_stream_fenced && async_window &&
            strcmp(async_window, "1") == 0 &&
            cudaDeviceSynchronize() != cudaSuccess) {
            (void)cudaGetLastError();
            return 0;
        }
    }
    for (uint32_t i = 0; i < count; ++i) {
        if (expert_ids[i] == -1) {
            slot_ids[i] = -1;
            continue;
        }
        const int32_t slot =
            cache->expert_to_slot[(uint32_t)expert_ids[i]];
        if (slot < 0 || (uint32_t)slot >= cache->slots ||
            !cache->entries[(uint32_t)slot].valid) return 0;
        slot_ids[i] = slot;
    }
    return 1;
}

extern "C" int ds4_gpu_q4k_window_cache_prepare_device(
        ds4_gpu_q4k_window_cache *cache,
        const ds4_gpu_tensor *expert_ids,
        const ds4_gpu_tensor *weights,
        uint32_t pair_count,
        ds4_gpu_tensor *slot_ids) {
    if (slot_ids) memset(slot_ids, 0, sizeof(*slot_ids));
    if (!cache ||
        std::find(g_q4k_window_caches.begin(), g_q4k_window_caches.end(),
                  cache) == g_q4k_window_caches.end() ||
        !expert_ids || !expert_ids->ptr || !weights || !weights->ptr ||
        !slot_ids || pair_count == 0u || pair_count > 1048576u) return 0;
    if (cache->overlap_enabled && cache->prepared_valid &&
        cache->prepared_ids == expert_ids &&
        cache->prepared_weights == weights &&
        cache->prepared_count == pair_count && cache->slot_ids_device) {
        slot_ids->ptr = cache->slot_ids_device;
        slot_ids->bytes = (uint64_t)pair_count * sizeof(int32_t);
        slot_ids->owner = 0;
        slot_ids->device_id = expert_ids->device_id;
        slot_ids->host_ptr = NULL;
        return 1;
    }
    uint64_t id_bytes = 0, weight_bytes = 0;
    if (!cuda_u64_mul_checked(pair_count, sizeof(int32_t), &id_bytes) ||
        !cuda_u64_mul_checked(pair_count, sizeof(float), &weight_bytes) ||
        expert_ids->bytes < id_bytes || weights->bytes < weight_bytes ||
        id_bytes > (uint64_t)SIZE_MAX || weight_bytes > (uint64_t)SIZE_MAX) {
        return 0;
    }
    const int profile = getenv("DS4_ROCM_GLM5_WINDOW_PROFILE") != NULL;
    const double profile_started = profile ? cuda_wall_sec() : 0.0;
    const uint64_t fills_before = cache->fills;
    std::vector<int32_t> host_ids;
    std::vector<int32_t> host_slots;
    std::vector<float> host_weights;
    try {
        host_ids.resize(pair_count);
        host_slots.resize(pair_count);
        host_weights.resize(pair_count);
    } catch (...) {
        return 0;
    }
    /* Blocking null-stream copies are load-bearing here: the router and the
     * current Q4_K MoE consumer also use the null stream, so this D2H observes
     * the producer and the later H2D cannot overtake the previous consumer. */
    if (cudaMemcpy(host_ids.data(), expert_ids->ptr, (size_t)id_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(host_weights.data(), weights->ptr, (size_t)weight_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    for (uint32_t i = 0; i < pair_count; ++i) {
        uint32_t weight_bits = 0;
        memcpy(&weight_bits, &host_weights[i], sizeof(weight_bits));
        const int nonfinite =
            (weight_bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000);
        if (nonfinite ||
            (host_ids[i] == -1 && host_weights[i] != 0.0f)) return 0;
    }
    if (!ds4_gpu_q4k_window_cache_prepare(
            cache, host_ids.data(), pair_count, host_slots.data())) return 0;
    if (id_bytes > cache->slot_ids_capacity) {
        if (cudaDeviceSynchronize() != cudaSuccess) {
            (void)cudaGetLastError();
            return 0;
        }
        int32_t *replacement = NULL;
        if (cudaMalloc((void **)&replacement, (size_t)id_bytes) !=
                cudaSuccess || !replacement) {
            (void)cudaGetLastError();
            return 0;
        }
        if (cache->slot_ids_device) (void)cudaFree(cache->slot_ids_device);
        cache->slot_ids_device = replacement;
        cache->slot_ids_capacity = id_bytes;
    }
    /* This blocking H2D is ordered after every prior null-stream MoE read of
     * the reused ID buffer.  See the D2H comment above before introducing a
     * non-default compute stream. */
    if (cudaMemcpy(cache->slot_ids_device, host_slots.data(),
                   (size_t)id_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    slot_ids->ptr = cache->slot_ids_device;
    slot_ids->bytes = id_bytes;
    slot_ids->owner = 0;
    slot_ids->device_id = expert_ids->device_id;
    slot_ids->host_ptr = NULL;
    if (cache->overlap_enabled) {
        cache->prepared_ids = expert_ids;
        cache->prepared_weights = weights;
        cache->prepared_count = pair_count;
        cache->prepared_valid = 1;
    }
    if (profile) {
        const uint64_t filled = cache->fills - fills_before;
        cache->profile_prepare_sec += cuda_wall_sec() - profile_started;
        cache->profile_upload_bytes += filled *
            (2u * cache->gate_expert_bytes + cache->down_expert_bytes);
        cache->profile_control_bytes += 2u * id_bytes + weight_bytes;
    }
    return 1;
}

extern "C" int ds4_gpu_q4k_window_cache_wait(
        const ds4_gpu_q4k_window_cache *cache) {
    if (!cache || !cache->overlap_enabled || !cache->fill_event ||
        std::find(g_q4k_window_caches.begin(), g_q4k_window_caches.end(),
                  cache) == g_q4k_window_caches.end()) return 1;
    return hipStreamWaitEvent((hipStream_t)0, cache->fill_event, 0) ==
           cudaSuccess;
}

extern "C" int ds4_gpu_q4k_window_cache_prefetch(
        ds4_gpu_q4k_window_cache *cache,
        const ds4_gpu_tensor *expert_ids,
        const ds4_gpu_tensor *weights,
        uint32_t pair_count) {
    ds4_gpu_tensor slot_ids = {};
    return ds4_gpu_q4k_window_cache_prepare_device(
        cache, expert_ids, weights, pair_count, &slot_ids);
}

extern "C" int ds4_gpu_q4k_window_cache_device_view(
        const ds4_gpu_q4k_window_cache *cache, const void **gate,
        const void **up, const void **down, uint64_t *gate_expert_bytes,
        uint64_t *down_expert_bytes) {
    if (gate) *gate = NULL;
    if (up) *up = NULL;
    if (down) *down = NULL;
    if (gate_expert_bytes) *gate_expert_bytes = 0u;
    if (down_expert_bytes) *down_expert_bytes = 0u;
    if (!cache ||
        std::find(g_q4k_window_caches.begin(), g_q4k_window_caches.end(),
                  cache) == g_q4k_window_caches.end() ||
        !gate || !up || !down || !gate_expert_bytes ||
        !down_expert_bytes || !cache->base) return 0;
    *gate = cache->gate;
    *up = cache->up;
    *down = cache->down;
    *gate_expert_bytes = cache->gate_expert_bytes;
    *down_expert_bytes = cache->down_expert_bytes;
    return 1;
}

extern "C" int ds4_gpu_q4k_window_cache_read_slot(
        const ds4_gpu_q4k_window_cache *cache, uint32_t slot,
        void *gate, uint64_t gate_bytes, void *up, uint64_t up_bytes,
        void *down, uint64_t down_bytes) {
    if (!cache ||
        std::find(g_q4k_window_caches.begin(), g_q4k_window_caches.end(),
                  cache) == g_q4k_window_caches.end() ||
        slot >= cache->slots || !cache->entries[slot].valid ||
        !gate || gate_bytes != cache->gate_expert_bytes ||
        !up || up_bytes != cache->gate_expert_bytes ||
        !down || down_bytes != cache->down_expert_bytes) return 0;
    cudaError_t err = cudaMemcpy(
        gate, cache->gate + (uint64_t)slot * cache->gate_expert_bytes,
        (size_t)cache->gate_expert_bytes, cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) {
        err = cudaMemcpy(
            up, cache->up + (uint64_t)slot * cache->gate_expert_bytes,
            (size_t)cache->gate_expert_bytes, cudaMemcpyDeviceToHost);
    }
    if (err == cudaSuccess) {
        err = cudaMemcpy(
            down, cache->down + (uint64_t)slot * cache->down_expert_bytes,
            (size_t)cache->down_expert_bytes, cudaMemcpyDeviceToHost);
    }
    if (err == cudaSuccess) return 1;
    (void)cudaGetLastError();
    return 0;
}

extern "C" int ds4_gpu_q4k_window_cache_get_stats(
        const ds4_gpu_q4k_window_cache *cache,
        ds4_gpu_q4k_window_cache_stats *stats) {
    if (!cache ||
        std::find(g_q4k_window_caches.begin(), g_q4k_window_caches.end(),
                  cache) == g_q4k_window_caches.end() ||
        !stats) return 0;
    memset(stats, 0, sizeof(*stats));
    stats->prepares = cache->prepares;
    stats->hits = cache->hits;
    stats->misses = cache->misses;
    stats->fills = cache->fills;
    stats->evictions = cache->evictions;
    stats->capacity_bytes = cache->capacity_bytes;
    stats->slot_count = cache->slots;
    for (const cuda_q4k_window_cache_entry &entry : cache->entries) {
        if (entry.valid) stats->resident_count++;
    }
    return 1;
}

extern "C" int ds4_gpu_q4k_window_cache_get_view(
        const ds4_gpu_q4k_window_cache *cache,
        ds4_gpu_q4k_window_cache_view *view) {
    if (view) memset(view, 0, sizeof(*view));
    if (!cache ||
        std::find(g_q4k_window_caches.begin(), g_q4k_window_caches.end(),
                  cache) == g_q4k_window_caches.end() ||
        !view || cache->gate_slice_index >= g_q4k_packed_slices.size() ||
        cache->up_slice_index >= g_q4k_packed_slices.size() ||
        cache->down_slice_index >= g_q4k_packed_slices.size()) return 0;
    const cuda_q4k_packed_slice &gate =
        g_q4k_packed_slices[cache->gate_slice_index];
    const cuda_q4k_packed_slice &up =
        g_q4k_packed_slices[cache->up_slice_index];
    const cuda_q4k_packed_slice &down =
        g_q4k_packed_slices[cache->down_slice_index];
    uint64_t expected_down_base = 0, expected_down_count = 0;
    if (gate.host_base != cache->model_map ||
        up.host_base != cache->model_map ||
        down.host_base != cache->model_map ||
        gate.model_size != up.model_size || gate.model_size != down.model_size ||
        gate.kind != DS4_GPU_Q4K_PACKED_ROW_RANGE ||
        up.kind != DS4_GPU_Q4K_PACKED_ROW_RANGE ||
        down.kind != DS4_GPU_Q4K_PACKED_K_RANGE ||
        gate.source_rows != up.source_rows ||
        gate.source_row_bytes != up.source_row_bytes ||
        gate.row_base != up.row_base || gate.row_count != up.row_count ||
        gate.column_byte_base != 0u || up.column_byte_base != 0u ||
        gate.column_byte_count != gate.source_row_bytes ||
        up.column_byte_count != up.source_row_bytes ||
        gate.row_count * 2u != gate.source_rows ||
        (gate.row_base != 0u && gate.row_base != gate.row_count) ||
        down.row_base != 0u || down.row_count != down.source_rows ||
        !cuda_u64_mul_checked(gate.row_base / CUDA_QK_K,
                              sizeof(cuda_block_q4_K),
                              &expected_down_base) ||
        !cuda_u64_mul_checked(gate.row_count / CUDA_QK_K,
                              sizeof(cuda_block_q4_K),
                              &expected_down_count) ||
        down.column_byte_base != expected_down_base ||
        down.column_byte_count != expected_down_count ||
        gate.packed_expert_bytes != cache->gate_expert_bytes ||
        up.packed_expert_bytes != cache->gate_expert_bytes ||
        down.packed_expert_bytes != cache->down_expert_bytes ||
        !cache->base) return 0;
    view->model_map = cache->model_map;
    view->model_size = gate.model_size;
    view->gate_offset = gate.tensor_offset;
    view->up_offset = up.tensor_offset;
    view->down_offset = down.tensor_offset;
    view->gate = cache->gate;
    view->up = cache->up;
    view->down = cache->down;
    view->gate_expert_bytes = cache->gate_expert_bytes;
    view->down_expert_bytes = cache->down_expert_bytes;
    view->gate_row_bytes = gate.column_byte_count;
    view->down_row_bytes = down.column_byte_count;
    view->row_base = gate.row_base;
    view->row_count = gate.row_count;
    view->slot_count = cache->slots;
    return 1;
}

extern "C" int ds4_gpu_q4k_packed_slice_readback(
        const void *model_map, uint64_t tensor_offset,
        uint32_t row_base, uint32_t row_count,
        uint64_t column_byte_base, uint64_t column_byte_count,
        void *dst, uint64_t bytes) {
    cuda_q4k_packed_slice *p = cuda_q4k_packed_slice_find(
        model_map, tensor_offset, row_base, row_count,
        column_byte_base, column_byte_count);
    if (!p || !p->device_ptr || !dst || bytes != p->packed_bytes) return 0;
    return cuda_ok(cudaMemcpy(dst, p->device_ptr, (size_t)bytes,
                              cudaMemcpyDeviceToHost),
                   "packed Q4_K slice readback");
}

extern "C" int ds4_gpu_q4k_packed_slice_resolve(
        const void *model_map, uint64_t tensor_offset,
        uint32_t n_expert, uint32_t source_rows,
        uint64_t source_row_bytes, uint32_t row_base, uint32_t row_count,
        uint64_t column_byte_base, uint64_t column_byte_count,
        ds4_gpu_q4k_packed_slice_kind kind,
        const void **device_ptr, uint64_t *packed_bytes,
        uint64_t *packed_expert_bytes, uint64_t *packed_row_bytes) {
    if (device_ptr) *device_ptr = NULL;
    if (packed_bytes) *packed_bytes = 0u;
    if (packed_expert_bytes) *packed_expert_bytes = 0u;
    if (packed_row_bytes) *packed_row_bytes = 0u;
    if (!device_ptr || !packed_bytes || !packed_expert_bytes ||
        !packed_row_bytes) return 0;
    cuda_q4k_packed_slice *p = cuda_q4k_packed_slice_find(
        model_map, tensor_offset, row_base, row_count,
        column_byte_base, column_byte_count);
    if (!p || !p->device_ptr || !p->loaded || p->n_expert != n_expert ||
        p->source_rows != source_rows ||
        p->source_row_bytes != source_row_bytes || p->kind != kind) return 0;
    *device_ptr = p->device_ptr;
    *packed_bytes = p->packed_bytes;
    *packed_expert_bytes = p->packed_expert_bytes;
    *packed_row_bytes = p->column_byte_count;
    return 1;
}

extern "C" uint64_t ds4_gpu_q4k_packed_slice_bytes(void) {
    return g_q4k_packed_slice_bytes;
}

static uint64_t cuda_model_cache_limit_bytes(void) {
    if (!g_ssd_streaming_mode) return UINT64_MAX;
    const char *env = getenv("DS4_ROCM_STREAM_MODEL_CACHE_GB");
    if (env && env[0]) {
        char *end = NULL;
        errno = 0;
        unsigned long long gib = strtoull(env, &end, 10);
        if (end != env && *end == '\0' && errno == 0 && gib != 0 &&
            gib <= UINT64_MAX / 1073741824ull) {
            return (uint64_t)gib * 1073741824ull;
        }
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "invalid DS4_ROCM_STREAM_MODEL_CACHE_GB=%s; "
                "using automatic streaming model span cache limit\n",
                env);
    }

    size_t free_b = 0;
    size_t total_b = 0;
    const uint64_t fallback = 32ull * 1073741824ull;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || total_b == 0) {
        (void)cudaGetLastError();
        return fallback;
    }
    (void)free_b;
    uint64_t limit = (uint64_t)total_b / 3ull;
    const uint64_t min_limit = 8ull * 1073741824ull;
    const uint64_t max_limit = 48ull * 1073741824ull;
    if (limit < min_limit) limit = min_limit;
    if (limit > max_limit) limit = max_limit;
    return limit;
}

static int cuda_stream_model_cache_prepare_memory(
        uint64_t request_bytes,
        const char *what) {
    if (!g_ssd_streaming_mode || request_bytes == 0 || g_q8_f16_bytes == 0) {
        return 1;
    }

    size_t free_b = 0;
    size_t total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        (void)cudaGetLastError();
        return 1;
    }
    (void)total_b;
    const uint64_t reserve = cuda_stream_resident_free_reserve_bytes();
    const uint64_t free_bytes = (uint64_t)free_b;
    if (free_bytes >= reserve && request_bytes <= free_bytes - reserve) {
        return 1;
    }

    if (!cuda_ok(cudaDeviceSynchronize(),
                 "streaming model cache q8 fp16 release sync")) {
        return 0;
    }
    fprintf(stderr,
            DS4_GPU_LOG_PREFIX "releasing %.2f GiB q8 fp16 cache for %s "
            "(request=%.2f GiB free=%.2f GiB reserve=%.2f GiB)\n",
            (double)g_q8_f16_bytes / 1073741824.0,
            what ? what : "streaming model spans",
            (double)request_bytes / 1073741824.0,
            (double)free_bytes / 1073741824.0,
            (double)reserve / 1073741824.0);
    cuda_q8_f16_cache_release_all();
    g_q8_f16_budget_notice_printed = 0;
    return 1;
}

static uint64_t cuda_model_arena_chunk_bytes(uint64_t need) {
    uint64_t bytes = 1792ull * 1048576ull;
    if (bytes < need) {
        const uint64_t align = 256ull * 1048576ull;
        bytes = (need + align - 1u) & ~(align - 1u);
    }
    return bytes;
}

static char *cuda_model_arena_alloc(uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g_model_cache_full) return NULL;
    const uint64_t align = 256u;
    const uint64_t aligned = (bytes + align - 1u) & ~(align - 1u);

    for (cuda_model_arena &a : g_model_arenas) {
        const uint64_t used = (a.used + align - 1u) & ~(align - 1u);
        if (used <= a.bytes && aligned <= a.bytes - used) {
            char *ptr = a.device_ptr + used;
            a.used = used + aligned;
            return ptr;
        }
    }

    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || aligned > limit - g_model_range_bytes) return NULL;

    const uint64_t chunk = cuda_model_arena_chunk_bytes(aligned);
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)chunk);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        uint64_t fallback = chunk / 2u;
        while (fallback >= aligned) {
            err = cudaMalloc(&dev, (size_t)fallback);
            if (err == cudaSuccess) break;
            (void)cudaGetLastError();
            fallback /= 2u;
        }
        if (err != cudaSuccess) {
            err = cudaMalloc(&dev, (size_t)aligned);
            if (err != cudaSuccess) {
                fprintf(stderr,
                        DS4_GPU_LOG_PREFIX "model arena alloc failed for %s "
                        "(%.2f MiB request): %s\n",
                        what ? what : "weights",
                        (double)aligned / 1048576.0,
                        cudaGetErrorString(err));
                (void)cudaGetLastError();
                g_model_cache_full = 1;
                return NULL;
            }
            fallback = aligned;
        }
        g_model_arenas.push_back({(char *)dev, fallback, aligned});
        return (char *)dev;
    }
    g_model_arenas.push_back({(char *)dev, chunk, aligned});
    return (char *)dev;
}

static const char *cuda_model_range_ptr_from_fd(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what) {
    if (g_model_fd < 0 || bytes == 0) return NULL;
    if (cuda_q4k_packed_slice_refuse_linear(model_map, offset, bytes, what)) {
        return NULL;
    }
    if (g_model_fd_host_base != NULL && model_map != g_model_fd_host_base) return NULL;
    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || bytes > limit - g_model_range_bytes) {
        if (g_ssd_streaming_mode) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming model cache limit prevents "
                    "loading %s range %.2f MiB; refusing host-pointer fallback\n",
                    what ? what : "weights",
                    (double)bytes / 1048576.0);
            return NULL;
        }
        return cuda_model_ptr(model_map, offset);
    }

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    const uint64_t stage_bytes =
        chunk + (g_model_direct_align > 1 ? g_model_direct_align : 1);
    if (!cuda_model_stage_pool_alloc(stage_bytes)) return NULL;

    char *dev = cuda_model_arena_alloc(bytes, what);
    if (!dev) {
        if (g_ssd_streaming_mode) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming model cache allocation failed "
                    "for %s range %.2f MiB; refusing host-pointer fallback\n",
                    what ? what : "weights",
                    (double)bytes / 1048576.0);
            return NULL;
        }
        return cuda_model_ptr(model_map, offset);
    }
    cudaError_t err = cudaSuccess;

    uint64_t copied = 0;
    uint64_t chunk_idx = 0;
    while (copied < bytes) {
        const uint64_t n = (bytes - copied < chunk) ? (bytes - copied) : chunk;
        const uint64_t bi = chunk_idx % 4u;
        if (chunk_idx >= 4u) {
            err = cudaEventSynchronize(g_model_stage_event[bi]);
            if (err != cudaSuccess) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX "model staging wait failed for %s: %s\n",
                        what ? what : "weights", cudaGetErrorString(err));
                (void)cudaGetLastError();
                return NULL;
            }
        }
        const char *payload = NULL;
        if (!cuda_model_stage_read(g_model_stage[bi], g_model_stage_bytes,
                                   offset + copied, n, &payload)) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model range read failed for %s at %.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    strerror(errno));
            return NULL;
        }
        err = cudaMemcpyAsync(dev + copied, payload, (size_t)n,
                              cudaMemcpyHostToDevice, g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model range copy failed for %s at %.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return NULL;
        }
        err = cudaEventRecord(g_model_stage_event[bi], g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model staging record failed for %s: %s\n",
                    what ? what : "weights", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return NULL;
        }
        cuda_model_drop_file_pages(offset + copied, n);
        cuda_model_discard_source_pages(model_map, g_model_registered_size, offset + copied, n);
        copied += n;
        cuda_model_load_progress_note(g_model_range_bytes + copied);
        chunk_idx++;
    }
    err = cudaStreamSynchronize(g_model_upload_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "model range upload sync failed for %s: %s\n",
                what ? what : "weights", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }

    g_model_ranges.push_back({model_map, offset, bytes, dev, NULL, NULL, 0, 0, 1});
    g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
    g_model_range_bytes += bytes;
    cuda_model_load_progress_note(g_model_range_bytes);
    return (const char *)dev;
}

static int cuda_model_copy_chunked(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    if (!model_map || model_size == 0 || map_offset > model_size || map_size > model_size - map_offset) return 0;
    if (map_size == 0) return 0;
    if (cuda_q4k_packed_slice_refuse_linear(
            model_map, map_offset, map_size, "chunked model image")) return 0;
    if (cuda_model_image_range_ptr(model_map, map_offset, map_size)) return 1;

    void *dev = NULL;
    const double t0 = cuda_wall_sec();
    cudaError_t err = cudaMalloc(&dev, (size_t)map_size);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "model allocation skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    fprintf(stderr, DS4_GPU_LOG_PREFIX "chunk-copying %.2f GiB model image\n",
            (double)map_size / 1073741824.0);

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    const uint64_t stage_bytes = chunk + (g_model_direct_align > 1 ? g_model_direct_align : 1);
    if (!cuda_model_stage_pool_alloc(stage_bytes)) {
        (void)cudaFree(dev);
        return 0;
    }

    uint64_t copied = 0;
    uint64_t chunk_idx = 0;
    while (copied < map_size) {
        const uint64_t n = (map_size - copied < chunk) ? (map_size - copied) : chunk;
        const uint64_t bi = chunk_idx % 4u;
        if (chunk_idx >= 4u) {
            err = cudaEventSynchronize(g_model_stage_event[bi]);
            if (err != cudaSuccess) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX "model staging wait failed: %s\n", cudaGetErrorString(err));
                (void)cudaFree(dev);
                (void)cudaGetLastError();
                return 0;
            }
        }
        const char *payload = NULL;
        if (!cuda_model_stage_read(g_model_stage[bi], g_model_stage_bytes,
                                   map_offset + copied, n, &payload)) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model staged read failed at %.2f GiB: %s\n",
                    (double)copied / 1073741824.0, strerror(errno));
            (void)cudaFree(dev);
            return 0;
        }
        err = cudaMemcpyAsync((char *)dev + copied, payload, (size_t)n,
                              cudaMemcpyHostToDevice, g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model chunk copy failed at %.2f GiB: %s\n",
                    (double)copied / 1073741824.0, cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return 0;
        }
        err = cudaEventRecord(g_model_stage_event[bi], g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model staging record failed: %s\n", cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return 0;
        }
        cuda_model_drop_file_pages(map_offset + copied, n);
        cuda_model_discard_source_pages(model_map, model_size,
                                        map_offset + copied, n);
        copied += n;
        chunk_idx++;
        cuda_model_load_progress_note(copied);
    }
    err = cudaStreamSynchronize(g_model_upload_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "model upload sync failed: %s\n", cudaGetErrorString(err));
        (void)cudaFree(dev);
        (void)cudaGetLastError();
        return 0;
    }
    g_model_images.push_back({model_map, map_size, (char *)dev, map_offset});
    g_model_host_base = model_map;
    /* Sparse layer slices may create several disjoint device images, so no
     * single base pointer can represent this model. Tensor lookups scan the
     * image table and subtract each image's file offset. */
    g_model_device_base = NULL;
    g_model_registered_size = model_size;
    g_model_device_owned = 1;
    const double t1 = cuda_wall_sec();
    fprintf(stderr,
            DS4_GPU_LOG_PREFIX "model chunk copy complete in %.3fs (%.2f GiB tensors)\n",
            t1 - t0,
            (double)map_size / 1073741824.0);
    return 1;
}

static void cuda_model_range_release_ranges_only(void) {
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_registered && r.registered_base) {
            (void)cudaHostUnregister(r.registered_base);
        } else if (r.device_ptr && !r.arena_allocated) {
            (void)cudaFree(r.device_ptr);
        }
    }
    for (const cuda_model_arena &a : g_model_arenas) {
        if (a.device_ptr) (void)cudaFree(a.device_ptr);
    }
    g_model_arenas.clear();
    g_model_ranges.clear();
    g_model_range_by_offset.clear();
    g_model_range_bytes = 0;
    g_model_cache_full = 0;
}

static void cuda_model_range_release_all(void) {
    cuda_model_range_release_ranges_only();
    g_stream_selected_cache.loaded = 0;
    cuda_stream_resident_cache_release();
    cuda_model_load_progress_reset();
}

static int cublas_ok(cublasStatus_t st, const char *what) {
    if (st == CUBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "ds4: " DS4_GPU_BLAS_NAME " %s failed: status %d\n", what, (int)st);
    return 0;
}


extern "C" int ds4_gpu_init(void) {
    int dev = 0;
    if (!cuda_ok(cudaSetDevice(dev), "set device")) return 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "backend initialized on %s (sm_%d%d)\n",
                prop.name, prop.major, prop.minor);
    }
    if (!g_cublas_ready) {
        if (!cublas_ok(cublasCreate(&g_cublas), "create handle")) return 0;
        const cublasMath_t math_mode = g_quality_mode ? CUBLAS_DEFAULT_MATH : CUBLAS_TF32_TENSOR_OP_MATH;
        (void)cublasSetMathMode(g_cublas, math_mode);
        g_cublas_ready = 1;
    }
#ifdef __HIP_PLATFORM_AMD__
    if (!g_hipblaslt_ready) {
        if (hipblaslt_ok(hipblasLtCreate(&g_hipblaslt), "create handle")) {
            g_hipblaslt_ready = 1;
        }
    }
#endif
    return 1;
}

extern "C" void ds4_gpu_cleanup(void) {
    (void)cudaDeviceSynchronize();
    if (g_token_span_start) {
        (void)cudaEventDestroy(g_token_span_start);
        g_token_span_start = NULL;
    }
    if (g_token_span_head) {
        (void)cudaEventDestroy(g_token_span_head);
        g_token_span_head = NULL;
    }
    for (uint32_t i = 0; i < DS4_GPU_TOKEN_HEAD_STAGE_COUNT; i++) {
        if (g_token_span_head_stage[i]) {
            (void)cudaEventDestroy(g_token_span_head_stage[i]);
            g_token_span_head_stage[i] = NULL;
        }
    }
    if (g_token_span_stop) {
        (void)cudaEventDestroy(g_token_span_stop);
        g_token_span_stop = NULL;
    }
    if (g_token_attn_span_start) {
        (void)cudaEventDestroy(g_token_attn_span_start);
        g_token_attn_span_start = NULL;
    }
    if (g_token_attn_span_stop) {
        (void)cudaEventDestroy(g_token_attn_span_stop);
        g_token_attn_span_stop = NULL;
    }
    cuda_stream_cache_stats_print("cleanup");
    cuda_shared_gate_up_async_cleanup();
#ifdef __HIP_PLATFORM_AMD__
    hipblaslt_gemm_plan_clear();
#endif
    if (g_cublas_ready) {
        (void)cublasDestroy(g_cublas);
        g_cublas_ready = 0;
        g_cublas = NULL;
    }
#ifdef __HIP_PLATFORM_AMD__
    if (g_hipblaslt_ready) {
        (void)hipblasLtDestroy(g_hipblaslt);
        g_hipblaslt_ready = 0;
        g_hipblaslt = NULL;
    }
#endif
    cuda_model_range_release_all();
    cuda_q8_f16_cache_release_all();
    cuda_stream_selected_cache_release();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_disabled_for_multi_model = 0;
    g_q8_f16_budget_notice_printed = 0;
    if (g_cuda_tmp) {
        (void)cudaFree(g_cuda_tmp);
        g_cuda_tmp = NULL;
        g_cuda_tmp_bytes = 0;
    }
    int attention_seq_saved_device = -1;
    (void)cudaGetDevice(&attention_seq_saved_device);
    for (int device = 0; device < DS4_MAX_GPUS; device++) {
        if (!g_attention_seq_scratch[device]) continue;
        if (cudaSetDevice(device) == cudaSuccess) {
            (void)cudaFree(g_attention_seq_scratch[device]);
        }
        g_attention_seq_scratch[device] = NULL;
        g_attention_seq_scratch_bytes[device] = 0;
    }
    for (int device = 0; device < DS4_MAX_GPUS; device++) {
        if (!g_glm_causal_scratch[device]) continue;
        if (cudaSetDevice(device) == cudaSuccess) {
            (void)cudaFree(g_glm_causal_scratch[device]);
        }
        g_glm_causal_scratch[device] = NULL;
        g_glm_causal_scratch_bytes[device] = 0;
    }
    if (attention_seq_saved_device >= 0) {
        (void)cudaSetDevice(attention_seq_saved_device);
    }
    for (size_t i = 0; i < 4; i++) {
        if (g_model_stage_event[i]) {
            (void)cudaEventDestroy(g_model_stage_event[i]);
            g_model_stage_event[i] = NULL;
        }
        if (g_model_stage_raw[i]) {
            (void)cudaFreeHost(g_model_stage_raw[i]);
            g_model_stage_raw[i] = NULL;
            g_model_stage[i] = NULL;
        }
    }
    g_model_stage_bytes = 0;
    if (g_model_upload_stream) {
        (void)cudaStreamDestroy(g_model_upload_stream);
        g_model_upload_stream = NULL;
    }
    if (g_stream_selected_upload_stream) {
        (void)cudaStreamDestroy(g_stream_selected_upload_stream);
        g_stream_selected_upload_stream = NULL;
    }
    if (g_selected_readback_stream) {
        (void)cudaStreamDestroy(g_selected_readback_stream);
        g_selected_readback_stream = NULL;
    }
    if (g_selected_readback_event) {
        (void)cudaEventDestroy(g_selected_readback_event);
        g_selected_readback_event = NULL;
    }
    g_selected_readback_event_value = 0;
    cuda_model_image_release_all();
    g_model_host_base = NULL;
    g_model_device_base = NULL;
    g_model_registered_size = 0;
    g_support_host_base = NULL;
    g_support_host_size = 0;
    g_support_fd = -1;
    if (g_support_direct_fd >= 0) {
        (void)close(g_support_direct_fd);
        g_support_direct_fd = -1;
    }
    g_support_file_size = 0;
    g_support_direct_align = 1;
    g_dspark_stage_offsets[0] = 0;
    g_dspark_stage_offsets[1] = 0;
    g_dspark_stage_offsets[2] = 0;
    if (g_dspark_selected_gate) (void)cudaFree(g_dspark_selected_gate);
    if (g_dspark_selected_up) (void)cudaFree(g_dspark_selected_up);
    if (g_dspark_selected_down) (void)cudaFree(g_dspark_selected_down);
    if (g_dspark_selected_ids) (void)cudaFree(g_dspark_selected_ids);
    g_dspark_selected_gate = NULL;
    g_dspark_selected_up = NULL;
    g_dspark_selected_down = NULL;
    g_dspark_selected_ids = NULL;
    g_dspark_selected_gate_capacity = 0;
    g_dspark_selected_down_capacity = 0;
    g_dspark_selected_ids_capacity = 0;
    g_model_device_owned = 0;
    g_model_range_mapping_supported = 1;
    g_model_fd = -1;
    if (g_model_direct_fd >= 0) {
        (void)close(g_model_direct_fd);
        g_model_direct_fd = -1;
    }
    g_model_direct_align = 1;
    g_model_file_size = 0;
    g_model_cache_full = 0;
}

__global__ static void fill_f32_kernel(float *x, uint64_t n, float v);

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (!cuda_ok(cudaMalloc(&t->ptr, (size_t)bytes), "tensor alloc")) {
        free(t);
        return NULL;
    }
    t->bytes = bytes;
    t->owner = 1;
    t->device_id = 0;
    return t;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (!cuda_ok(cudaMallocManaged(&t->ptr, (size_t)bytes), "managed tensor alloc")) {
        free(t);
        return NULL;
    }
    t->bytes = bytes;
    t->owner = 1;
    t->device_id = 0;
    return t;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_rdma_host(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    void *host = NULL;
    void *device = NULL;
    unsigned int host_flags = hipHostMallocMapped;
    const char *coherent_env = getenv("DS4_ROCM_RDMA_HOST_COHERENT");
    const int force_coherent = coherent_env && coherent_env[0] &&
                               strcmp(coherent_env, "0") != 0;
    if (force_coherent) host_flags |= hipHostMallocCoherent;
    cudaError_t err = hipHostMalloc(&host, (size_t)bytes, host_flags);
    if (err == hipSuccess)
        err = hipHostGetDevicePointer(&device, host, 0);
    if (err != hipSuccess || !host || !device) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "RDMA host slab allocation failed: %s\n",
                hipGetErrorString(err));
        if (host) (void)hipHostFree(host);
        free(t);
        return NULL;
    }
    t->ptr = device;
    t->host_ptr = host;
    t->bytes = bytes;
    t->owner = 2;
    t->device_id = 0;
    fprintf(stderr, DS4_GPU_LOG_PREFIX
            "using mapped host-pinned TP slab for generic RDMA (%.2f MiB, "
            "coherent=%d)\n",
            (double)bytes / 1048576.0, force_coherent);
    return t;
}

static uint64_t cuda_managed_kv_reserve_bytes(uint64_t total_bytes) {
    const uint64_t min_reserve = 8ull * 1073741824ull;
    const uint64_t max_reserve = 40ull * 1073741824ull;
    uint64_t reserve = total_bytes / 4u;
    if (reserve < min_reserve) reserve = min_reserve;
    if (reserve > max_reserve) reserve = max_reserve;
    return reserve;
}

extern "C" int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes) {
    if (kv_cache_bytes == 0) return 0;

    /* Very large KV caches are where device-only cudaMalloc() can make a
     * unified-memory machine unresponsive.  Managed memory restores the old
     * demand-paged behavior for this one long-lived allocation class only. */
    const uint64_t huge_kv = 8ull * 1073741824ull;
    if (kv_cache_bytes >= huge_kv) return 1;

    const uint64_t large_context = 8ull * 1073741824ull;
    if (context_bytes < large_context) return 0;

    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }

    const uint64_t free_bytes = (uint64_t)free_b;
    const uint64_t total_bytes = (uint64_t)total_b;
    const uint64_t reserve_bytes = cuda_managed_kv_reserve_bytes(total_bytes);
    if (context_bytes > free_bytes) return 1;
    return free_bytes - context_bytes < reserve_bytes;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base || offset > base->bytes || bytes > base->bytes - offset) return NULL;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->ptr = (char *)base->ptr + offset;
    if (base->host_ptr)
        t->host_ptr = (char *)base->host_ptr + offset;
    t->bytes = bytes;
    t->owner = 0;
    t->device_id = base->device_id;
    return t;
}

extern "C" void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    if (tensor->owner == 2 && tensor->host_ptr)
        (void)hipHostFree(tensor->host_ptr);
    else if (tensor->owner == 1 && tensor->ptr)
        (void)cudaFree(tensor->ptr);
    free(tensor);
}

extern "C" uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

extern "C" void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor) {
    if (!tensor) return NULL;
    (void)cudaDeviceSynchronize();
    return tensor->host_ptr ? tensor->host_ptr : tensor->ptr;
}

extern "C" int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count) {
    if (!tensor || count > tensor->bytes / sizeof(float)) return 0;
    if (count == 0) return 1;
    /* DS4-TP-gfx1151 (diag): cudaGetLastError() returns the last error from
     * ANYWHERE, so a sticky error from an earlier op gets misattributed to this
     * launch. Clear it first, then report only what this launch produced. */
    const cudaError_t stale = cudaGetLastError();
    const uint64_t grid = (count + 255u) / 256u;
    fill_f32_kernel<<<(unsigned)grid, 256>>>((float *)tensor->ptr, count, value);
    const cudaError_t mine = cudaGetLastError();
    if (mine != cudaSuccess || stale != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "fill_f32 diag: stale=%d(%s) mine=%d(%s) "
                "ptr=%p bytes=%llu count=%llu grid=%llu\n",
                (int)stale, cudaGetErrorString(stale),
                (int)mine, cudaGetErrorString(mine),
                tensor->ptr,
                (unsigned long long)tensor->bytes,
                (unsigned long long)count,
                (unsigned long long)grid);
    }
    return cuda_ok(mine, "tensor fill f32 launch");
}

extern "C" int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    return cuda_ok(cudaMemcpy((char *)tensor->ptr + offset, data, (size_t)bytes, cudaMemcpyHostToDevice), "tensor write");
}

extern "C" int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    return cuda_ok(cudaMemcpy(data, (const char *)tensor->ptr + offset, (size_t)bytes, cudaMemcpyDeviceToHost), "tensor read");
}

__global__ static void rdma_system_release_kernel(void) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        __threadfence_system();
    }
#endif
}

__global__ static void rdma_system_acquire_kernel(void) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        asm volatile("buffer_gl1_inv\n\tbuffer_gl0_inv" ::: "memory");
    }
#endif
}

extern "C" int ds4_rocm_rdma_cache_release(void) {
    rdma_system_release_kernel<<<1, 64>>>();
    return cuda_ok(cudaGetLastError(), "RDMA system release launch") &&
           cuda_ok(cudaDeviceSynchronize(), "RDMA system release sync");
}

extern "C" int ds4_rocm_rdma_cache_acquire(void) {
    rdma_system_acquire_kernel<<<1, 64>>>();
    return cuda_ok(cudaGetLastError(), "RDMA system acquire launch");
}

extern "C" int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                     const ds4_gpu_tensor *src, uint64_t src_offset,
                                     uint64_t bytes) {
    if (!dst || !src || dst_offset > dst->bytes || src_offset > src->bytes ||
        bytes > dst->bytes - dst_offset || bytes > src->bytes - src_offset) {
        return 0;
    }
    if (bytes == 0) return 1;
    return cuda_ok(cudaMemcpyAsync((char *)dst->ptr + dst_offset,
                                   (const char *)src->ptr + src_offset,
                                   (size_t)bytes,
                                   cudaMemcpyDeviceToDevice,
                                   0),
                   "tensor copy enqueue");
}

extern "C" int ds4_gpu_begin_commands(void) { return 1; }
extern "C" int ds4_gpu_token_span_begin(void) {
    if (!g_token_span_start &&
        !cuda_ok(cudaEventCreate(&g_token_span_start),
                 "token span start event create")) return 0;
    if (!g_token_span_stop &&
        !cuda_ok(cudaEventCreate(&g_token_span_stop),
                 "token span stop event create")) return 0;
    if (!g_token_span_head &&
        !cuda_ok(cudaEventCreate(&g_token_span_head),
                 "token span head event create")) return 0;
    if (token_span_head_stage_profile_enabled()) {
        for (uint32_t i = 0; i < DS4_GPU_TOKEN_HEAD_STAGE_COUNT; i++) {
            if (!g_token_span_head_stage[i] &&
                !cuda_ok(cudaEventCreate(&g_token_span_head_stage[i]),
                         "token span head stage event create")) return 0;
        }
    }
    g_token_span_active = 1;
    g_token_span_head_valid = 0;
    g_token_span_head_stage_mask = 0u;
    return cuda_ok(cudaEventRecord(g_token_span_start, 0),
                   "token span start event record");
}

extern "C" int ds4_gpu_token_span_head_stage(uint32_t stage) {
    if (!g_token_span_active) return 1;
    if (!token_span_head_stage_profile_enabled()) return 1;
    if (!g_token_span_head_valid || stage >= DS4_GPU_TOKEN_HEAD_STAGE_COUNT) {
        return 0;
    }
    if (!cuda_ok(cudaEventRecord(g_token_span_head_stage[stage], 0),
                 "token span head stage event record")) return 0;
    g_token_span_head_stage_mask |= 1u << stage;
    return 1;
}

extern "C" int ds4_gpu_token_span_head(void) {
    if (!g_token_span_active) return 1;
    if (!cuda_ok(cudaEventRecord(g_token_span_head, 0),
                 "token span head event record")) return 0;
    g_token_span_head_valid = 1;
    return 1;
}

extern "C" int ds4_gpu_token_span_end(void) {
    const int ok = g_token_span_start && g_token_span_stop &&
        cuda_ok(cudaEventRecord(g_token_span_stop, 0),
                "token span stop event record");
    g_token_span_active = 0;
    return ok;
}

extern "C" int ds4_gpu_token_span_sections_ms(float *pre_head_ms,
                                                 float *head_ms) {
    return pre_head_ms && head_ms && g_token_span_start &&
           g_token_span_head && g_token_span_stop &&
           g_token_span_head_valid &&
           cuda_ok(cudaEventElapsedTime(pre_head_ms,
                                        g_token_span_start,
                                        g_token_span_head),
                   "token span pre-head elapsed") &&
           cuda_ok(cudaEventElapsedTime(head_ms,
                                        g_token_span_head,
                                        g_token_span_stop),
                   "token span head elapsed");
}

extern "C" int ds4_gpu_token_span_head_stages_ms(
        float elapsed_ms[DS4_GPU_TOKEN_HEAD_STAGE_COUNT]) {
    if (!token_span_head_stage_profile_enabled() || !elapsed_ms ||
        !g_token_span_head_valid ||
        g_token_span_head_stage_mask !=
            ((1u << DS4_GPU_TOKEN_HEAD_STAGE_COUNT) - 1u)) return 0;
    cudaEvent_t previous = g_token_span_head;
    for (uint32_t i = 0; i < DS4_GPU_TOKEN_HEAD_STAGE_COUNT; i++) {
        if (!cuda_ok(cudaEventElapsedTime(&elapsed_ms[i], previous,
                                          g_token_span_head_stage[i]),
                     "token span head stage elapsed")) return 0;
        previous = g_token_span_head_stage[i];
    }
    return 1;
}

extern "C" int ds4_gpu_token_span_elapsed_ms(float *elapsed_ms) {
    return elapsed_ms && g_token_span_start && g_token_span_stop &&
           cuda_ok(cudaEventElapsedTime(elapsed_ms,
                                        g_token_span_start,
                                        g_token_span_stop),
                   "token span event elapsed");
}

extern "C" int ds4_gpu_token_attn_span_begin(void) {
    if (!g_token_attn_span_start &&
        !cuda_ok(cudaEventCreate(&g_token_attn_span_start),
                 "token attention span start event create")) return 0;
    if (!g_token_attn_span_stop &&
        !cuda_ok(cudaEventCreate(&g_token_attn_span_stop),
                 "token attention span stop event create")) return 0;
    g_token_attn_span_active = 1;
    g_token_attn_span_valid = 0;
    return cuda_ok(cudaEventRecord(g_token_attn_span_start, 0),
                   "token attention span start event record");
}

extern "C" int ds4_gpu_token_attn_span_end(void) {
    if (!g_token_attn_span_active || !g_token_attn_span_start ||
        !g_token_attn_span_stop) return 0;
    const int ok = cuda_ok(cudaEventRecord(g_token_attn_span_stop, 0),
                           "token attention span stop event record");
    g_token_attn_span_active = 0;
    g_token_attn_span_valid = ok;
    return ok;
}

extern "C" int ds4_gpu_token_attn_span_elapsed_ms(float *elapsed_ms) {
    return elapsed_ms && g_token_attn_span_valid &&
           cuda_ok(cudaEventElapsedTime(elapsed_ms,
                                        g_token_attn_span_start,
                                        g_token_attn_span_stop),
                   "token attention span elapsed");
}
extern "C" int ds4_gpu_flush_commands(void) { return cuda_ok(cudaDeviceSynchronize(), "flush"); }
extern "C" int ds4_gpu_flush_encoder(void) { return ds4_gpu_flush_commands(); }
extern "C" int ds4_gpu_commands_active(void) { return 0; }
extern "C" int ds4_gpu_signal_selected_readback_ready(uint64_t *event_value) {
    if (!event_value) return 0;
    *event_value = 0;
    if (!g_selected_readback_event) {
        cudaError_t err =
            cudaEventCreateWithFlags(&g_selected_readback_event,
                                     cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "selected readback event creation failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    cudaError_t err = cudaEventRecord(g_selected_readback_event, 0);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "selected readback event record failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    *event_value = ++g_selected_readback_event_value;
    return 1;
}
extern "C" int ds4_gpu_commit_and_wait_selected_readback(uint64_t event_value, const char *label) {
    if (event_value == 0 || !g_selected_readback_event) return 0;
    return cuda_ok(cudaEventSynchronize(g_selected_readback_event),
                   label ? label : "selected readback");
}
extern "C" int ds4_gpu_wait_selected_readback_ready(uint64_t event_value, const char *label) {
    if (event_value == 0 || !g_selected_readback_event) return 0;
    return cuda_ok(cudaEventSynchronize(g_selected_readback_event),
                   label ? label : "selected readback wait");
}

#if defined(DS4_ENABLE_PROFILING) && DS4_ENABLE_PROFILING
enum {
    DS4_ROCM_DECODE_ATTN_EVENT_POOL_SIZE = 64,
    DS4_ROCM_DECODE_ATTN_EVENT_REPORT_SAMPLES = 500,
    DS4_ROCM_DECODE_ATTN_EVENT_SAMPLE_MAX = 16384
};

typedef struct {
    uint64_t count;
    double sum_ms;
    float min_ms;
    float max_ms;
} ds4_rocm_decode_attn_event_stat;

typedef struct {
    hipEvent_t events[DS4_GPU_DECODE_ATTN_EVENT_COUNT];
    uint32_t valid_mask;
    uint32_t rank;
    uint32_t layer;
    uint32_t pos;
    uint32_t compress_ratio;
    uint32_t compress_emit;
    int complete;
} ds4_rocm_decode_attn_event_slot;

typedef struct {
    float elapsed_ms[DS4_GPU_DECODE_ATTN_EVENT_COUNT];
    uint32_t valid_mask;
    uint32_t rank;
    uint32_t layer;
    uint32_t pos;
    uint32_t compress_ratio;
    uint32_t compress_emit;
} ds4_rocm_decode_attn_event_sample;

static ds4_rocm_decode_attn_event_slot
    g_decode_attn_event_slots[DS4_ROCM_DECODE_ATTN_EVENT_POOL_SIZE];
static ds4_rocm_decode_attn_event_stat
    g_decode_attn_event_stats[DS4_GPU_DECODE_ATTN_EVENT_COUNT];
static int g_decode_attn_event_enabled = -1;
static int g_decode_attn_event_events_ready;
static int g_decode_attn_event_atexit_registered;
static int g_decode_attn_event_active_slot = -1;
static uint32_t g_decode_attn_event_next_slot;
static uint32_t g_decode_attn_event_rank;
static uint64_t g_decode_attn_event_samples;
static uint64_t g_decode_attn_event_dropped;
static int g_decode_attn_event_samples_enabled;
static ds4_rocm_decode_attn_event_sample
    g_decode_attn_event_sample_archive[DS4_ROCM_DECODE_ATTN_EVENT_SAMPLE_MAX];
static uint32_t g_decode_attn_event_sample_count;
static uint64_t g_decode_attn_event_sample_dropped;

static const char *const g_decode_attn_event_names[DS4_GPU_DECODE_ATTN_EVENT_COUNT] = {
    "start", "qkv_proj", "qkv_norm_rope", "q_b_proj", "q_norm_rope",
    "kv_path", "compressor_proj", "compressor_update",
    "compressor_quantize", "compressor_commit",
    "indexer_compressor_proj", "indexer_compressor_update",
    "indexer_compressor_qat", "indexer_query_proj", "indexer_score",
    "indexer_topk", "compressor_indexer",
    "attn_inv_rope", "attn_output_low", "attn_output_expand", "attn_gate",
    "attn_output", "attn_hc_post", "ffn_hc_pre", "ffn_norm", "router",
    "shared_gate_up", "shared_down", "routed_moe", "ffn_gate", "ffn_hc_post"
};

static void ds4_rocm_decode_attn_event_print_summary(void) {
    if (g_decode_attn_event_samples == 0u) return;
    fprintf(stderr, DS4_GPU_LOG_PREFIX
            "decode attention event profile rank=%u samples=%llu dropped=%llu",
            g_decode_attn_event_rank,
            (unsigned long long)g_decode_attn_event_samples,
            (unsigned long long)g_decode_attn_event_dropped);
    for (uint32_t i = 1; i < DS4_GPU_DECODE_ATTN_EVENT_COUNT; i++) {
        const ds4_rocm_decode_attn_event_stat *s = &g_decode_attn_event_stats[i];
        if (s->count == 0u) continue;
        fprintf(stderr, " %s[n=%llu mean=%.3f min=%.3f max=%.3f]",
                g_decode_attn_event_names[i],
                (unsigned long long)s->count,
                s->sum_ms / (double)s->count,
                (double)s->min_ms, (double)s->max_ms);
    }
    fputc('\n', stderr);
}

static void ds4_rocm_decode_attn_event_print(void) {
    ds4_rocm_decode_attn_event_print_summary();
    if (!g_decode_attn_event_samples_enabled) return;
    for (uint32_t sample_index = 0;
         sample_index < g_decode_attn_event_sample_count;
         sample_index++) {
        const ds4_rocm_decode_attn_event_sample *sample =
            &g_decode_attn_event_sample_archive[sample_index];
        fprintf(stderr,
                "{\"ds4_decode_stage_sample\":true,\"rank\":%u,"
                "\"layer\":%u,\"pos\":%u,\"ratio\":%u,\"emit\":%u,"
                "\"stages_ms\":{",
                sample->rank, sample->layer, sample->pos,
                sample->compress_ratio, sample->compress_emit);
        int separator = 0;
        for (uint32_t stage = 1;
             stage < DS4_GPU_DECODE_ATTN_EVENT_COUNT;
             stage++) {
            if ((sample->valid_mask & (1u << stage)) == 0u) continue;
            fprintf(stderr, "%s\"%s\":%.6f",
                    separator ? "," : "", g_decode_attn_event_names[stage],
                    (double)sample->elapsed_ms[stage]);
            separator = 1;
        }
        fputs("}}\n", stderr);
    }
    if (g_decode_attn_event_sample_dropped != 0u) {
        fprintf(stderr,
                "{\"ds4_decode_stage_samples_dropped\":true,"
                "\"rank\":%u,\"count\":%llu}\n",
                g_decode_attn_event_rank,
                (unsigned long long)g_decode_attn_event_sample_dropped);
    }
}

static int ds4_rocm_decode_attn_event_ensure_events(void) {
    if (g_decode_attn_event_events_ready) return 1;
    for (uint32_t slot = 0; slot < DS4_ROCM_DECODE_ATTN_EVENT_POOL_SIZE; slot++) {
        for (uint32_t stage = 0; stage < DS4_GPU_DECODE_ATTN_EVENT_COUNT; stage++) {
            hipError_t err = hipEventCreate(&g_decode_attn_event_slots[slot].events[stage]);
            if (err != hipSuccess) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX
                        "decode attention event profile create failed: %s\n",
                        hipGetErrorString(err));
                g_decode_attn_event_enabled = 0;
                return 0;
            }
        }
    }
    g_decode_attn_event_events_ready = 1;
    return 1;
}

extern "C" int ds4_gpu_decode_attn_event_profile_enabled(void) {
    if (g_decode_attn_event_enabled < 0) {
        const char *env = getenv("DS4_ROCM_DECODE_ATTN_EVENT_PROFILE");
        const char *samples_env =
            getenv("DS4_ROCM_DECODE_ATTN_EVENT_SAMPLES");
        g_decode_attn_event_samples_enabled =
            samples_env != NULL && samples_env[0] != '\0' &&
            strcmp(samples_env, "0") != 0;
        g_decode_attn_event_enabled =
            (env != NULL && env[0] != '\0' && strcmp(env, "0") != 0) ||
            g_decode_attn_event_samples_enabled;
        if (g_decode_attn_event_enabled && !g_decode_attn_event_atexit_registered) {
            atexit(ds4_rocm_decode_attn_event_print);
            g_decode_attn_event_atexit_registered = 1;
        }
    }
    return g_decode_attn_event_enabled;
}

static int ds4_rocm_decode_attn_event_harvest(
        ds4_rocm_decode_attn_event_slot *slot) {
    if (!slot->complete) return 1;
    /* Querying the terminal event is the only readiness test.  In particular,
     * do not replace this with hipEventSynchronize: a busy slot drops the new
     * sample below so profiling cannot stall the decode/TP streams. */
    hipError_t query = hipEventQuery(
            slot->events[DS4_GPU_DECODE_ATTN_EVENT_FFN_HC_POST]);
    if (query == hipErrorNotReady) {
        /* Not an error, but hipGetLastError() returns the last error from
         * ANYWHERE in this thread - if left latched, the next unrelated
         * cuda_ok(cudaGetLastError(), ...) check elsewhere in the decode
         * path (hundreds of call sites, including the exact
         * compressor/indexer/attention kernels this profiler instruments)
         * would spuriously see it and fail that kernel launch. This project
         * already shipped that exact bug once (hipMallocSignalMemory
         * unavailable leaving hipErrorInvalidValue latched) - clear it here
         * defensively regardless of ROCm's precise hipEventQuery semantics. */
        (void)hipGetLastError();
        return 0;
    }
    if (query != hipSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "decode attention event profile query failed: %s\n",
                hipGetErrorString(query));
        g_decode_attn_event_enabled = 0;
        return 0;
    }
    uint32_t previous = DS4_GPU_DECODE_ATTN_EVENT_START;
    ds4_rocm_decode_attn_event_sample sample = {};
    sample.valid_mask = slot->valid_mask;
    sample.rank = slot->rank;
    sample.layer = slot->layer;
    sample.pos = slot->pos;
    sample.compress_ratio = slot->compress_ratio;
    sample.compress_emit = slot->compress_emit;
    for (uint32_t stage = 1; stage < DS4_GPU_DECODE_ATTN_EVENT_COUNT; stage++) {
        const uint32_t bit = 1u << stage;
        if ((slot->valid_mask & bit) == 0u) continue;
        float elapsed_ms = 0.0f;
        hipError_t err = hipEventElapsedTime(&elapsed_ms,
                                              slot->events[previous],
                                              slot->events[stage]);
        if (err != hipSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "decode attention event profile elapsed failed: %s\n",
                    hipGetErrorString(err));
            g_decode_attn_event_enabled = 0;
            return 0;
        }
        ds4_rocm_decode_attn_event_stat *stat = &g_decode_attn_event_stats[stage];
        if (stat->count == 0u || elapsed_ms < stat->min_ms) stat->min_ms = elapsed_ms;
        if (stat->count == 0u || elapsed_ms > stat->max_ms) stat->max_ms = elapsed_ms;
        stat->count++;
        stat->sum_ms += elapsed_ms;
        sample.elapsed_ms[stage] = elapsed_ms;
        previous = stage;
    }
    if (g_decode_attn_event_samples_enabled) {
        if (g_decode_attn_event_sample_count <
            DS4_ROCM_DECODE_ATTN_EVENT_SAMPLE_MAX) {
            g_decode_attn_event_sample_archive[g_decode_attn_event_sample_count++] =
                sample;
        } else {
            g_decode_attn_event_sample_dropped++;
        }
    }
    slot->complete = 0;
    slot->valid_mask = 0u;
    g_decode_attn_event_samples++;
    if (g_decode_attn_event_samples % DS4_ROCM_DECODE_ATTN_EVENT_REPORT_SAMPLES == 0u) {
        ds4_rocm_decode_attn_event_print_summary();
    }
    return 1;
}

extern "C" void ds4_gpu_decode_attn_event_profile_record(
        ds4_gpu_decode_attn_event_stage stage, uint32_t rank) {
    if (!g_decode_attn_event_enabled ||
        stage < DS4_GPU_DECODE_ATTN_EVENT_START ||
        stage >= DS4_GPU_DECODE_ATTN_EVENT_COUNT ||
        !ds4_rocm_decode_attn_event_ensure_events()) {
        return;
    }
    if (stage == DS4_GPU_DECODE_ATTN_EVENT_START) {
        ds4_rocm_decode_attn_event_slot *slot =
            &g_decode_attn_event_slots[g_decode_attn_event_next_slot];
        /* Sixteen layers of deferral normally make this event ready.  If it is
         * not, preserve the in-flight events and skip this layer wholesale. */
        if (slot->complete && !ds4_rocm_decode_attn_event_harvest(slot)) {
            g_decode_attn_event_dropped++;
            g_decode_attn_event_active_slot = -1;
            return;
        }
        if (!g_decode_attn_event_enabled) return;
        slot->valid_mask = 0u;
        slot->complete = 0;
        g_decode_attn_event_active_slot = (int)g_decode_attn_event_next_slot;
        g_decode_attn_event_next_slot =
            (g_decode_attn_event_next_slot + 1u) % DS4_ROCM_DECODE_ATTN_EVENT_POOL_SIZE;
        g_decode_attn_event_rank = rank;
    }
    if (g_decode_attn_event_active_slot < 0) return;
    ds4_rocm_decode_attn_event_slot *slot =
        &g_decode_attn_event_slots[g_decode_attn_event_active_slot];
    hipError_t err = hipEventRecord(slot->events[stage], 0);
    if (err != hipSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "decode attention event profile record %s failed: %s\n",
                g_decode_attn_event_names[stage], hipGetErrorString(err));
        g_decode_attn_event_enabled = 0;
        g_decode_attn_event_active_slot = -1;
        return;
    }
    slot->valid_mask |= 1u << stage;
    if (stage == DS4_GPU_DECODE_ATTN_EVENT_FFN_HC_POST) {
        slot->complete = 1;
        g_decode_attn_event_active_slot = -1;
    }
}

extern "C" void ds4_gpu_decode_attn_event_profile_begin(
        uint32_t rank, uint32_t layer, uint32_t pos,
        uint32_t compress_ratio, int compress_emit) {
    ds4_gpu_decode_attn_event_profile_record(
            DS4_GPU_DECODE_ATTN_EVENT_START, rank);
    if (g_decode_attn_event_active_slot < 0) return;
    ds4_rocm_decode_attn_event_slot *slot =
        &g_decode_attn_event_slots[g_decode_attn_event_active_slot];
    slot->rank = rank;
    slot->layer = layer;
    slot->pos = pos;
    slot->compress_ratio = compress_ratio;
    slot->compress_emit = compress_emit != 0;
}
#endif

/* Verifier stage timing deliberately uses enough slots for one complete
 * DS4-Flash verifier graph.  The graph queues every layer before its normal
 * completion/readback, so a 16-slot pool would drop most of a 43-layer
 * sample even when the GPU is behaving normally.  Reuse is still guarded by
 * hipEventQuery and drops rather than waits. */
#if defined(DS4_ENABLE_PROFILING) && DS4_ENABLE_PROFILING
enum { DS4_ROCM_VERIFY_STAGE_EVENT_POOL_SIZE = 64 };

typedef struct {
    hipEvent_t events[DS4_GPU_VERIFY_STAGE_EVENT_COUNT];
    uint32_t valid_mask;
    int complete;
} ds4_rocm_verify_stage_event_slot;

typedef struct {
    uint64_t count;
    double sum_ms;
    float min_ms;
    float max_ms;
} ds4_rocm_verify_stage_event_stat;

enum {
    DS4_ROCM_VERIFY_STAT_LAYER = 0,
    DS4_ROCM_VERIFY_STAT_ATTN,
    DS4_ROCM_VERIFY_STAT_ATTN_FRONT,
    DS4_ROCM_VERIFY_STAT_ATTN_QKV_TO_CORE,
    DS4_ROCM_VERIFY_STAT_ATTN_PRE_INDEXER,
    DS4_ROCM_VERIFY_STAT_ATTN_COMPRESSOR,
    DS4_ROCM_VERIFY_STAT_ATTN_INDEXER_PROJ,
    DS4_ROCM_VERIFY_STAT_ATTN_INDEXER_KV_GATE,
    DS4_ROCM_VERIFY_STAT_ATTN_INDEXER_Q,
    DS4_ROCM_VERIFY_STAT_ATTN_INDEXER_Q_POST,
    DS4_ROCM_VERIFY_STAT_ATTN_INDEXER_WEIGHT,
    DS4_ROCM_VERIFY_STAT_ATTN_INDEXER_COMPRESSOR,
    DS4_ROCM_VERIFY_STAT_ATTN_CACHE_STORE,
    DS4_ROCM_VERIFY_STAT_ATTN_INDEXER,
    DS4_ROCM_VERIFY_STAT_ATTN_INDEXED_CORE,
    DS4_ROCM_VERIFY_STAT_ATTN_OUTPUT,
    DS4_ROCM_VERIFY_STAT_ATTN_POST,
    DS4_ROCM_VERIFY_STAT_DENSE_Q8,
    DS4_ROCM_VERIFY_STAT_ROUTED_MOE,
    DS4_ROCM_VERIFY_STAT_RESIDUAL,
    DS4_ROCM_VERIFY_STAT_COUNT
};

static ds4_rocm_verify_stage_event_slot
    g_verify_stage_event_slots[DS4_ROCM_VERIFY_STAGE_EVENT_POOL_SIZE];
static ds4_rocm_verify_stage_event_stat
    g_verify_stage_event_stats[DS4_ROCM_VERIFY_STAT_COUNT];
static int g_verify_stage_events_enabled = -1;
static int g_verify_stage_events_ready;
static int g_verify_stage_events_atexit_registered;
static int g_verify_stage_events_active_slot = -1;
static uint32_t g_verify_stage_events_next_slot;
static uint32_t g_verify_stage_events_rank;
static uint64_t g_verify_stage_events_samples;
static uint64_t g_verify_stage_events_dropped;

static void ds4_rocm_verify_stage_stat_add(uint32_t stat_index, float ms) {
    ds4_rocm_verify_stage_event_stat *s = &g_verify_stage_event_stats[stat_index];
    if (s->count == 0u || ms < s->min_ms) s->min_ms = ms;
    if (s->count == 0u || ms > s->max_ms) s->max_ms = ms;
    s->count++;
    s->sum_ms += ms;
}

extern "C" void ds4_gpu_verify_stage_events_print(void) {
    if (!g_verify_stage_events_enabled || g_verify_stage_events_samples == 0u) return;
    static const char *const names[DS4_ROCM_VERIFY_STAT_COUNT] = {
        "layer", "attention", "attn_front", "attn_qkv_to_core",
        "attn_pre_indexer", "attn_compressor", "attn_indexer_proj",
        "attn_indexer_kv_gate", "attn_indexer_q",
        "attn_indexer_q_post", "attn_indexer_weight",
        "attn_indexer_compressor", "attn_cache_store",
        "attn_indexer", "attn_indexed_core",
        "attn_output", "attn_post", "dense_q8", "routed_moe", "residual"
    };
    fprintf(stderr, DS4_GPU_LOG_PREFIX
            "DSpark verify stage events rank=%u layers=%llu dropped=%llu",
            g_verify_stage_events_rank,
            (unsigned long long)g_verify_stage_events_samples,
            (unsigned long long)g_verify_stage_events_dropped);
    for (uint32_t i = 0; i < DS4_ROCM_VERIFY_STAT_COUNT; i++) {
        const ds4_rocm_verify_stage_event_stat *s = &g_verify_stage_event_stats[i];
        if (!s->count) continue;
        fprintf(stderr, " %s[n=%llu mean=%.3f min=%.3f max=%.3f]",
                names[i], (unsigned long long)s->count,
                s->sum_ms / (double)s->count,
                (double)s->min_ms, (double)s->max_ms);
    }
    fputc('\n', stderr);
}

extern "C" int ds4_gpu_verify_stage_events_enabled(void) {
    if (g_verify_stage_events_enabled < 0) {
        const char *env = getenv("DS4_DSPARK_VERIFY_STAGE_EVENTS");
        g_verify_stage_events_enabled =
            env != NULL && env[0] != '\0' && strcmp(env, "0") != 0;
        if (g_verify_stage_events_enabled &&
            !g_verify_stage_events_atexit_registered) {
            atexit(ds4_gpu_verify_stage_events_print);
            g_verify_stage_events_atexit_registered = 1;
        }
    }
    return g_verify_stage_events_enabled;
}

extern "C" int ds4_gpu_verify_stage_events_prepare(void) {
    if (!ds4_gpu_verify_stage_events_enabled()) return 0;
    if (g_verify_stage_events_ready) return 1;
    for (uint32_t slot = 0; slot < DS4_ROCM_VERIFY_STAGE_EVENT_POOL_SIZE; slot++) {
        for (uint32_t stage = 0; stage < DS4_GPU_VERIFY_STAGE_EVENT_COUNT; stage++) {
            hipError_t err = hipEventCreate(&g_verify_stage_event_slots[slot].events[stage]);
            if (err != hipSuccess) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX
                        "verify stage event creation failed: %s\n",
                        hipGetErrorString(err));
                g_verify_stage_events_enabled = 0;
                return 0;
            }
        }
    }
    g_verify_stage_events_ready = 1;
    return 1;
}

static int ds4_rocm_verify_stage_elapsed(
        ds4_rocm_verify_stage_event_slot *slot,
        ds4_gpu_verify_stage_event_stage start,
        ds4_gpu_verify_stage_event_stage end,
        float *ms) {
    const uint32_t bits = (1u << start) | (1u << end);
    if ((slot->valid_mask & bits) != bits) return 0;
    hipError_t err = hipEventElapsedTime(ms, slot->events[start], slot->events[end]);
    if (err != hipSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "verify stage event elapsed failed: %s\n",
                hipGetErrorString(err));
        g_verify_stage_events_enabled = 0;
        return -1;
    }
    return 1;
}

static int ds4_rocm_verify_stage_harvest_slot(
        ds4_rocm_verify_stage_event_slot *slot) {
    if (!slot->complete) return 1;
    hipError_t query = hipEventQuery(slot->events[DS4_GPU_VERIFY_STAGE_EVENT_LAYER_END]);
    if (query == hipErrorNotReady) {
        /* hipEventQuery is the only readiness test.  Clear NotReady from the
         * per-thread last-error slot so a later launch check cannot consume it. */
        (void)hipGetLastError();
        return 0;
    }
    if (query != hipSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "verify stage event query failed: %s\n", hipGetErrorString(query));
        g_verify_stage_events_enabled = 0;
        return 0;
    }

    float layer_ms = 0.0f, attn_ms = 0.0f, gate_up_ms = 0.0f;
    float down_ms = 0.0f, routed_ms = 0.0f;
    float attn_front_ms = 0.0f, attn_qkv_to_core_ms = 0.0f;
    float attn_pre_indexer_ms = 0.0f, attn_indexer_ms = 0.0f;
    float attn_compressor_ms = 0.0f, attn_indexer_proj_ms = 0.0f;
    float attn_indexer_kv_gate_ms = 0.0f, attn_indexer_q_ms = 0.0f;
    float attn_indexer_q_post_ms = 0.0f, attn_indexer_weight_ms = 0.0f;
    float attn_indexer_compressor_ms = 0.0f, attn_cache_store_ms = 0.0f;
    float attn_indexed_core_ms = 0.0f, attn_output_ms = 0.0f;
    float attn_post_ms = 0.0f;
    const int have_layer = ds4_rocm_verify_stage_elapsed(
        slot, DS4_GPU_VERIFY_STAGE_EVENT_LAYER_START,
        DS4_GPU_VERIFY_STAGE_EVENT_LAYER_END, &layer_ms);
    const int have_attn = ds4_rocm_verify_stage_elapsed(
        slot, DS4_GPU_VERIFY_STAGE_EVENT_LAYER_START,
        DS4_GPU_VERIFY_STAGE_EVENT_ATTN_END, &attn_ms);
    const int have_attn_front = ds4_rocm_verify_stage_elapsed(
        slot, DS4_GPU_VERIFY_STAGE_EVENT_LAYER_START,
        DS4_GPU_VERIFY_STAGE_EVENT_ATTN_QKV_END, &attn_front_ms);
    const int have_attn_qkv_to_core = ds4_rocm_verify_stage_elapsed(
        slot, DS4_GPU_VERIFY_STAGE_EVENT_ATTN_QKV_END,
        DS4_GPU_VERIFY_STAGE_EVENT_ATTN_CORE_END, &attn_qkv_to_core_ms);
    const int have_attn_pre_indexer = ds4_rocm_verify_stage_elapsed(
        slot, DS4_GPU_VERIFY_STAGE_EVENT_ATTN_QKV_END,
        DS4_GPU_VERIFY_STAGE_EVENT_ATTN_INDEXER_START, &attn_pre_indexer_ms);
    const int have_attn_compressor = ds4_rocm_verify_stage_elapsed(
        slot, DS4_GPU_VERIFY_STAGE_EVENT_ATTN_QKV_END,
        DS4_GPU_VERIFY_STAGE_EVENT_ATTN_COMPRESSOR_END,
        &attn_compressor_ms);
    const int have_attn_indexer_proj = ds4_rocm_verify_stage_elapsed(
        slot, DS4_GPU_VERIFY_STAGE_EVENT_ATTN_COMPRESSOR_END,
        DS4_GPU_VERIFY_STAGE_EVENT_ATTN_INDEXER_PROJ_END,
        &attn_indexer_proj_ms);
    const int have_attn_indexer_kv_gate = ds4_rocm_verify_stage_elapsed(
        slot, DS4_GPU_VERIFY_STAGE_EVENT_ATTN_COMPRESSOR_END,
        DS4_GPU_VERIFY_STAGE_EVENT_ATTN_INDEXER_KV_GATE_END,
        &attn_indexer_kv_gate_ms);
    const int have_attn_indexer_q = ds4_rocm_verify_stage_elapsed(
        slot, DS4_GPU_VERIFY_STAGE_EVENT_ATTN_INDEXER_KV_GATE_END,
        DS4_GPU_VERIFY_STAGE_EVENT_ATTN_INDEXER_Q_END,
        &attn_indexer_q_ms);
    const int have_attn_indexer_q_post = ds4_rocm_verify_stage_elapsed(
        slot, DS4_GPU_VERIFY_STAGE_EVENT_ATTN_INDEXER_Q_END,
        DS4_GPU_VERIFY_STAGE_EVENT_ATTN_INDEXER_Q_POST_END,
        &attn_indexer_q_post_ms);
    const int have_attn_indexer_weight = ds4_rocm_verify_stage_elapsed(
        slot, DS4_GPU_VERIFY_STAGE_EVENT_ATTN_INDEXER_Q_POST_END,
        DS4_GPU_VERIFY_STAGE_EVENT_ATTN_INDEXER_PROJ_END,
        &attn_indexer_weight_ms);
    const int have_attn_indexer_compressor = ds4_rocm_verify_stage_elapsed(
        slot, DS4_GPU_VERIFY_STAGE_EVENT_ATTN_INDEXER_PROJ_END,
        DS4_GPU_VERIFY_STAGE_EVENT_ATTN_INDEXER_COMPRESSOR_END,
        &attn_indexer_compressor_ms);
    const int have_attn_cache_store = ds4_rocm_verify_stage_elapsed(
        slot, DS4_GPU_VERIFY_STAGE_EVENT_ATTN_INDEXER_COMPRESSOR_END,
        DS4_GPU_VERIFY_STAGE_EVENT_ATTN_CACHE_STORE_END,
        &attn_cache_store_ms);
    const int have_attn_indexer = ds4_rocm_verify_stage_elapsed(
        slot, DS4_GPU_VERIFY_STAGE_EVENT_ATTN_INDEXER_START,
        DS4_GPU_VERIFY_STAGE_EVENT_ATTN_INDEXER_END, &attn_indexer_ms);
    const int have_attn_indexed_core = ds4_rocm_verify_stage_elapsed(
        slot, DS4_GPU_VERIFY_STAGE_EVENT_ATTN_INDEXER_END,
        DS4_GPU_VERIFY_STAGE_EVENT_ATTN_CORE_END, &attn_indexed_core_ms);
    const int have_attn_output = ds4_rocm_verify_stage_elapsed(
        slot, DS4_GPU_VERIFY_STAGE_EVENT_ATTN_CORE_END,
        DS4_GPU_VERIFY_STAGE_EVENT_ATTN_OUTPUT_END, &attn_output_ms);
    const int have_attn_post = ds4_rocm_verify_stage_elapsed(
        slot, DS4_GPU_VERIFY_STAGE_EVENT_ATTN_OUTPUT_END,
        DS4_GPU_VERIFY_STAGE_EVENT_ATTN_END, &attn_post_ms);
    const int have_gate_up = ds4_rocm_verify_stage_elapsed(
        slot, DS4_GPU_VERIFY_STAGE_EVENT_DENSE_GATE_UP_START,
        DS4_GPU_VERIFY_STAGE_EVENT_DENSE_GATE_UP_END, &gate_up_ms);
    const int have_down = ds4_rocm_verify_stage_elapsed(
        slot, DS4_GPU_VERIFY_STAGE_EVENT_DENSE_DOWN_START,
        DS4_GPU_VERIFY_STAGE_EVENT_DENSE_DOWN_END, &down_ms);
    const int have_routed = ds4_rocm_verify_stage_elapsed(
        slot, DS4_GPU_VERIFY_STAGE_EVENT_ROUTED_MOE_START,
        DS4_GPU_VERIFY_STAGE_EVENT_ROUTED_MOE_END, &routed_ms);
    if (!g_verify_stage_events_enabled) return 0;
    if (have_layer > 0) ds4_rocm_verify_stage_stat_add(DS4_ROCM_VERIFY_STAT_LAYER, layer_ms);
    if (have_attn > 0) ds4_rocm_verify_stage_stat_add(DS4_ROCM_VERIFY_STAT_ATTN, attn_ms);
    if (have_attn_front > 0) {
        ds4_rocm_verify_stage_stat_add(DS4_ROCM_VERIFY_STAT_ATTN_FRONT,
                                      attn_front_ms);
    }
    if (have_attn_qkv_to_core > 0) {
        ds4_rocm_verify_stage_stat_add(DS4_ROCM_VERIFY_STAT_ATTN_QKV_TO_CORE,
                                      attn_qkv_to_core_ms);
    }
    if (have_attn_pre_indexer > 0) {
        ds4_rocm_verify_stage_stat_add(DS4_ROCM_VERIFY_STAT_ATTN_PRE_INDEXER,
                                      attn_pre_indexer_ms);
    }
    if (have_attn_compressor > 0) {
        ds4_rocm_verify_stage_stat_add(DS4_ROCM_VERIFY_STAT_ATTN_COMPRESSOR,
                                      attn_compressor_ms);
    }
    if (have_attn_indexer_proj > 0) {
        ds4_rocm_verify_stage_stat_add(DS4_ROCM_VERIFY_STAT_ATTN_INDEXER_PROJ,
                                      attn_indexer_proj_ms);
    }
    if (have_attn_indexer_kv_gate > 0) {
        ds4_rocm_verify_stage_stat_add(
            DS4_ROCM_VERIFY_STAT_ATTN_INDEXER_KV_GATE,
            attn_indexer_kv_gate_ms);
    }
    if (have_attn_indexer_q > 0) {
        ds4_rocm_verify_stage_stat_add(DS4_ROCM_VERIFY_STAT_ATTN_INDEXER_Q,
                                      attn_indexer_q_ms);
    }
    if (have_attn_indexer_q_post > 0) {
        ds4_rocm_verify_stage_stat_add(
            DS4_ROCM_VERIFY_STAT_ATTN_INDEXER_Q_POST,
            attn_indexer_q_post_ms);
    }
    if (have_attn_indexer_weight > 0) {
        ds4_rocm_verify_stage_stat_add(
            DS4_ROCM_VERIFY_STAT_ATTN_INDEXER_WEIGHT,
            attn_indexer_weight_ms);
    }
    if (have_attn_indexer_compressor > 0) {
        ds4_rocm_verify_stage_stat_add(
            DS4_ROCM_VERIFY_STAT_ATTN_INDEXER_COMPRESSOR,
            attn_indexer_compressor_ms);
    }
    if (have_attn_cache_store > 0) {
        ds4_rocm_verify_stage_stat_add(DS4_ROCM_VERIFY_STAT_ATTN_CACHE_STORE,
                                      attn_cache_store_ms);
    }
    if (have_attn_indexer > 0) {
        ds4_rocm_verify_stage_stat_add(DS4_ROCM_VERIFY_STAT_ATTN_INDEXER,
                                      attn_indexer_ms);
    }
    if (have_attn_indexed_core > 0) {
        ds4_rocm_verify_stage_stat_add(DS4_ROCM_VERIFY_STAT_ATTN_INDEXED_CORE,
                                      attn_indexed_core_ms);
    }
    if (have_attn_output > 0) {
        ds4_rocm_verify_stage_stat_add(DS4_ROCM_VERIFY_STAT_ATTN_OUTPUT,
                                      attn_output_ms);
    }
    if (have_attn_post > 0) {
        ds4_rocm_verify_stage_stat_add(DS4_ROCM_VERIFY_STAT_ATTN_POST,
                                      attn_post_ms);
    }
    if (have_gate_up > 0 || have_down > 0) {
        ds4_rocm_verify_stage_stat_add(DS4_ROCM_VERIFY_STAT_DENSE_Q8,
                                      gate_up_ms + down_ms);
    }
    if (have_routed > 0) {
        ds4_rocm_verify_stage_stat_add(DS4_ROCM_VERIFY_STAT_ROUTED_MOE, routed_ms);
    }
    if (have_layer > 0 && have_attn > 0 && have_routed > 0) {
        float residual = layer_ms - attn_ms - routed_ms - gate_up_ms - down_ms;
        if (residual < 0.0f) residual = 0.0f;
        ds4_rocm_verify_stage_stat_add(DS4_ROCM_VERIFY_STAT_RESIDUAL, residual);
    }
    slot->complete = 0;
    slot->valid_mask = 0u;
    g_verify_stage_events_samples++;
    return 1;
}

extern "C" void ds4_gpu_verify_stage_events_harvest(void) {
    if (!g_verify_stage_events_enabled || !g_verify_stage_events_ready) return;
    for (uint32_t i = 0; i < DS4_ROCM_VERIFY_STAGE_EVENT_POOL_SIZE; i++) {
        if (g_verify_stage_event_slots[i].complete) {
            (void)ds4_rocm_verify_stage_harvest_slot(&g_verify_stage_event_slots[i]);
        }
    }
}

extern "C" void ds4_gpu_verify_stage_events_record(
        ds4_gpu_verify_stage_event_stage stage, uint32_t rank) {
    if (!g_verify_stage_events_enabled || !g_verify_stage_events_ready ||
        stage < DS4_GPU_VERIFY_STAGE_EVENT_LAYER_START ||
        stage >= DS4_GPU_VERIFY_STAGE_EVENT_COUNT) return;
    if (stage == DS4_GPU_VERIFY_STAGE_EVENT_LAYER_START) {
        ds4_rocm_verify_stage_event_slot *slot =
            &g_verify_stage_event_slots[g_verify_stage_events_next_slot];
        if (slot->complete && !ds4_rocm_verify_stage_harvest_slot(slot)) {
            g_verify_stage_events_dropped++;
            g_verify_stage_events_active_slot = -1;
            return;
        }
        if (!g_verify_stage_events_enabled) return;
        slot->valid_mask = 0u;
        slot->complete = 0;
        g_verify_stage_events_active_slot = (int)g_verify_stage_events_next_slot;
        g_verify_stage_events_next_slot =
            (g_verify_stage_events_next_slot + 1u) %
            DS4_ROCM_VERIFY_STAGE_EVENT_POOL_SIZE;
        g_verify_stage_events_rank = rank;
    }
    if (g_verify_stage_events_active_slot < 0) return;
    ds4_rocm_verify_stage_event_slot *slot =
        &g_verify_stage_event_slots[g_verify_stage_events_active_slot];
    hipError_t err = hipEventRecord(slot->events[stage], 0);
    if (err != hipSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "verify stage event record failed: %s\n", hipGetErrorString(err));
        g_verify_stage_events_enabled = 0;
        g_verify_stage_events_active_slot = -1;
        return;
    }
    slot->valid_mask |= 1u << stage;
    if (stage == DS4_GPU_VERIFY_STAGE_EVENT_LAYER_END) {
        slot->complete = 1;
        g_verify_stage_events_active_slot = -1;
    }
}
#else
extern "C" int ds4_gpu_verify_stage_events_enabled(void) { return 0; }
extern "C" int ds4_gpu_verify_stage_events_prepare(void) { return 0; }
extern "C" void ds4_gpu_verify_stage_events_harvest(void) {}
extern "C" void ds4_gpu_verify_stage_events_record(
        ds4_gpu_verify_stage_event_stage stage, uint32_t rank) {
    (void)stage;
    (void)rank;
}
extern "C" void ds4_gpu_verify_stage_events_print(void) {}
#endif

extern "C" int ds4_gpu_end_commands(void) {
    return cuda_ok(cudaDeviceSynchronize(), "end commands");
}
extern "C" int ds4_gpu_synchronize(void) { return cuda_ok(cudaDeviceSynchronize(), "synchronize"); }

extern "C" int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    if (!model_map || model_size == 0) return 0;
    if (g_model_host_base == model_map && g_model_registered_size == model_size) return 1;
    const int multi_model =
        g_model_host_base != NULL &&
        (g_model_host_base != model_map || g_model_registered_size != model_size);
    cuda_model_range_release_all();
    cuda_q8_f16_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    if (multi_model) {
        /*
         * MTP loads a second GGUF mapping.  Its weights are small, but on UMA
         * ROCm systems the optional expanded Q8->F16 cache can consume the
         * memory margin needed for session/context tensors once both model
         * mappings are resident.  The cache is only a speed path; the normal
         * Q8 kernels remain available and keep MTP startup reliable.
         */
        g_q8_f16_disabled_for_multi_model = 1;
    }
    g_model_host_base = model_map;
    g_model_device_base = cuda_model_image_owned(model_map) ?
                          cuda_model_image_ptr(model_map, 0) :
                          (const char *)model_map;
    g_model_registered_size = model_size;
    g_model_device_owned = cuda_model_image_owned(model_map);
    g_model_range_mapping_supported = 1;
    g_model_cache_full = 0;
    if (g_model_fd >= 0 && g_model_fd_host_base == NULL) {
        g_model_fd_host_base = model_map;
    }

    /* Strix Halo uses the staged full-copy path in ds4_gpu_set_model_map_range().
     * Avoid host-registering the mmap here: that would make the staged copier
     * believe the model is already device-resident. */
    return 1;
}

extern "C" int ds4_gpu_register_support_map(
        const void *map, uint64_t size, uint64_t bias, int fd) {
    (void)bias;
    if (!map || size == 0) return 0;
    g_support_host_base = map;
    g_support_host_size = size;
    g_support_fd = fd;
    if (g_support_direct_fd >= 0) {
        (void)close(g_support_direct_fd);
        g_support_direct_fd = -1;
    }
    g_support_file_size = 0;
    g_support_direct_align = 1;
    if (fd >= 0) {
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size > 0) {
            g_support_file_size = (uint64_t)st.st_size;
            if (st.st_blksize > 1) {
                g_support_direct_align = (uint64_t)st.st_blksize;
            }
        }
#if defined(__linux__) && defined(O_DIRECT)
        char proc_path[64];
        snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", fd);
        g_support_direct_fd = open(proc_path, O_RDONLY | O_DIRECT);
        if (g_support_direct_fd >= 0 && g_support_direct_align < 512) {
            g_support_direct_align = 512;
        }
#endif
    }
    g_dspark_stage_offsets[0] = 0;
    g_dspark_stage_offsets[1] = 0;
    g_dspark_stage_offsets[2] = 0;
    if (cuda_env_enabled_exact("DS4_DSPARK_RESIDENT_Q8") &&
        !cuda_dspark_make_support_resident(map, size)) {
        return 0;
    }
    fprintf(stderr,
            DS4_GPU_LOG_PREFIX "registered %.2f GiB compact DSpark support "
            "(%s)\n",
            (double)size / 1073741824.0,
            cuda_model_image_range_ptr(map, 0, size)
                ? "resident Q8 device image" : "bounded per-stage loading");
    return 1;
}

extern "C" int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!model_map || model_size == 0 ||
        map_offset > model_size ||
        map_size > model_size - map_offset) {
        return 0;
    }
    if (cuda_q4k_packed_slice_intersection(model_map, map_offset, map_size)) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "model range intersects a declared packed Q4_K routed "
                "tensor; refusing linear residency\n");
        return 0;
    }
    if (!ds4_gpu_set_model_map(model_map, model_size)) return 0;
    if (g_ssd_streaming_mode) {
        const uint64_t limit = cuda_model_cache_limit_bytes();
        if (map_size > limit) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming model range %.2f GiB exceeds "
                    "cache limit %.2f GiB; increase DS4_ROCM_STREAM_MODEL_CACHE_GB\n",
                    (double)map_size / 1073741824.0,
                    (double)limit / 1073741824.0);
            return 0;
        }
        if (g_model_range_bytes > limit ||
            map_size > limit - g_model_range_bytes) {
            if (!cuda_ok(cudaDeviceSynchronize(),
                         "streaming model range cache eviction sync")) {
                return 0;
            }
            cuda_model_range_release_ranges_only();
        }
        if (!cuda_stream_model_cache_prepare_memory(map_size,
                                                    "streaming model range")) {
            return 0;
        }
        if (!cuda_model_range_ptr(model_map, map_offset, map_size, "stream_range")) return 0;
        return cuda_model_range_is_cached(model_map, map_offset, map_size);
    }
    /*
     * Do not eagerly copy a contiguous model image here.  On Strix Halo the
     * caller immediately follows with accelerator_cache_model_tensors(), which
     * prepares the exact tensor spans selected by --layers.  Copying here would
     * either allocate the whole GGUF image or, for sparse span sets, an oversized
     * envelope before the precise tensor-span cache gets a chance to run.
     */
    return 1;
}

extern "C" int ds4_gpu_set_model_map_spans(
        const void *model_map,
        uint64_t model_size,
        const uint64_t *offsets,
        const uint64_t *sizes,
        uint32_t count,
        uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!model_map || model_size == 0 || !offsets || !sizes || count == 0) return 0;
    for (uint32_t i = 0; i < count; i++) {
        if (offsets[i] > model_size ||
            sizes[i] == 0 ||
            sizes[i] > model_size - offsets[i]) {
            return 0;
        }
        if (cuda_q4k_packed_slice_intersection(
                model_map, offsets[i], sizes[i])) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "model span %u intersects a declared packed Q4_K "
                    "routed tensor; refusing linear residency\n", i);
            return 0;
        }
    }
    if (!ds4_gpu_set_model_map(model_map, model_size)) return 0;
    if (g_ssd_streaming_mode) {
        uint64_t request_bytes = 0;
        for (uint32_t i = 0; i < count; i++) {
            if (!cuda_u64_add_checked(request_bytes, sizes[i], &request_bytes)) {
                fprintf(stderr,
                        DS4_GPU_LOG_PREFIX "streaming model span byte count overflow\n");
                return 0;
            }
        }
        const uint64_t limit = cuda_model_cache_limit_bytes();
        if (request_bytes > limit) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming model spans %.2f GiB exceed "
                    "cache limit %.2f GiB; increase DS4_ROCM_STREAM_MODEL_CACHE_GB\n",
                    (double)request_bytes / 1073741824.0,
                    (double)limit / 1073741824.0);
            return 0;
        }
        if (g_model_range_bytes > limit ||
            request_bytes > limit - g_model_range_bytes) {
            if (!cuda_ok(cudaDeviceSynchronize(),
                         "streaming model span cache eviction sync")) {
                return 0;
            }
            cuda_model_range_release_ranges_only();
        }
        if (!cuda_stream_model_cache_prepare_memory(request_bytes,
                                                    "streaming model spans")) {
            return 0;
        }
        for (uint32_t i = 0; i < count; i++) {
            if (!cuda_model_range_ptr(model_map, offsets[i], sizes[i], "stream_span")) return 0;
            if (!cuda_model_range_is_cached(model_map, offsets[i], sizes[i])) return 0;
        }
        return 1;
    }

    uint64_t span_bytes = 0;
    uint64_t min_offset = offsets[0];
    uint64_t max_end = offsets[0] + sizes[0];
    for (uint32_t i = 0; i < count; i++) {
        if (span_bytes > UINT64_MAX - sizes[i]) return 0;
        span_bytes += sizes[i];
        if (offsets[i] < min_offset) min_offset = offsets[i];
        const uint64_t end = offsets[i] + sizes[i];
        if (end > max_end) max_end = end;
    }

    /* Preserve the working rocm-multi-node residency policy: copy tight
     * slices once, but split sparse coordinator layer+head selections into
     * separate device images instead of allocating their huge file envelope. */
    const uint64_t bbox = max_end - min_offset;
    /* GLM5 TP intentionally omits large routed-expert tensors from the
     * resident span set.  Its remaining tensors can still have a compact
     * bounding box numerically, but copying that box would bridge the
     * omission and make packed Q4_K declaration fail later. */
    const bool glm5_span_policy =
        getenv("DS4_GLM5_NEXT_ENABLE_ORDINARY") != NULL;
    if (!glm5_span_policy &&
        bbox <= span_bytes + span_bytes / 10u) {
        return cuda_model_copy_chunked(model_map, model_size,
                                       min_offset, bbox);
    }

    std::vector<std::pair<uint64_t, uint64_t>> sorted(count);
    for (uint32_t i = 0; i < count; i++) {
        sorted[i] = {offsets[i], offsets[i] + sizes[i]};
    }
    std::sort(sorted.begin(), sorted.end());
    uint64_t group_offset = sorted[0].first;
    uint64_t group_end = sorted[0].second;
    const uint64_t merge_gap = 64ull * 1024ull;
    for (uint32_t i = 1; i <= count; i++) {
        if (i < count &&
            (sorted[i].first <= group_end ||
             sorted[i].first - group_end <= merge_gap)) {
            if (sorted[i].second > group_end) group_end = sorted[i].second;
            continue;
        }
        if (!cuda_model_copy_chunked(model_map, model_size,
                                     group_offset, group_end - group_offset)) {
            return 0;
        }
        if (i < count) {
            group_offset = sorted[i].first;
            group_end = sorted[i].second;
        }
    }
    return 1;
}

static uint64_t cuda_q4k_kshard_hash_bytes(
        uint64_t h, const void *data, size_t bytes) {
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static int cuda_q4k_kshard_layer_geometry_ok(
        const ds4_gpu_q4k_kshard_layer *layer) {
    const uint64_t block = sizeof(cuda_block_q4_K);
    if (!layer || layer->n_expert < 2u ||
        layer->n_expert > DS4_ROCM_MAX_N_EXPERT ||
        layer->expert_in_dim != 4096u ||
        layer->expert_mid_dim != 2048u || layer->out_dim != 4096u ||
        layer->gate_row_bytes != 16u * block ||
        layer->up_row_bytes != layer->gate_row_bytes ||
        layer->down_row_bytes != 8u * block) return 0;
    return 1;
}

static int cuda_q4k_kshard_dense_spans_exclude_routed(
        const uint64_t *offsets, const uint64_t *sizes, uint32_t count,
        const ds4_gpu_q4k_kshard_layer *layers, uint32_t n_layers) {
    for (uint32_t i = 0; i < count; ++i) {
        for (uint32_t il = 0; il < n_layers; ++il) {
            const ds4_gpu_q4k_kshard_layer &l = layers[il];
            const uint64_t gate_expert =
                (uint64_t)l.expert_mid_dim * l.gate_row_bytes;
            const uint64_t down_expert =
                (uint64_t)l.out_dim * l.down_row_bytes;
            const uint64_t gate_bytes = (uint64_t)l.n_expert * gate_expert;
            const uint64_t down_bytes = (uint64_t)l.n_expert * down_expert;
            if (cuda_u64_ranges_overlap(offsets[i], sizes[i],
                                        l.gate_offset, gate_bytes) ||
                cuda_u64_ranges_overlap(offsets[i], sizes[i],
                                        l.up_offset, gate_bytes) ||
                cuda_u64_ranges_overlap(offsets[i], sizes[i],
                                        l.down_offset, down_bytes)) return 0;
        }
    }
    return 1;
}

static uint64_t cuda_q4k_kshard_hash_layer(
        uint64_t h, const ds4_gpu_q4k_kshard_layer &layer) {
#define DS4_Q4K_KSHARD_HASH_FIELD(field) \
    h = cuda_q4k_kshard_hash_bytes(      \
        h, &layer.field, sizeof(layer.field))
    DS4_Q4K_KSHARD_HASH_FIELD(gate_offset);
    DS4_Q4K_KSHARD_HASH_FIELD(up_offset);
    DS4_Q4K_KSHARD_HASH_FIELD(down_offset);
    DS4_Q4K_KSHARD_HASH_FIELD(n_expert);
    DS4_Q4K_KSHARD_HASH_FIELD(expert_in_dim);
    DS4_Q4K_KSHARD_HASH_FIELD(expert_mid_dim);
    DS4_Q4K_KSHARD_HASH_FIELD(out_dim);
    DS4_Q4K_KSHARD_HASH_FIELD(gate_row_bytes);
    DS4_Q4K_KSHARD_HASH_FIELD(up_row_bytes);
    DS4_Q4K_KSHARD_HASH_FIELD(down_row_bytes);
#undef DS4_Q4K_KSHARD_HASH_FIELD
    return h;
}

static int cuda_q4k_kshard_block_ranges_prepare(
        const void *model_map,
        const ds4_gpu_q4k_kshard_layer *layers, uint32_t n_layers) {
    std::vector<cuda_q4k_kshard_blocked_range> next;
    try {
        next.reserve((size_t)n_layers * 3u);
        for (uint32_t il = 0; il < n_layers; ++il) {
            const ds4_gpu_q4k_kshard_layer &l = layers[il];
            const uint64_t gate_expert =
                (uint64_t)l.expert_mid_dim * l.gate_row_bytes;
            const uint64_t down_expert =
                (uint64_t)l.out_dim * l.down_row_bytes;
            const uint64_t gate_bytes = (uint64_t)l.n_expert * gate_expert;
            const uint64_t down_bytes = (uint64_t)l.n_expert * down_expert;
            next.push_back({model_map, l.gate_offset, gate_bytes, 0});
            next.push_back({model_map, l.up_offset, gate_bytes, 0});
            next.push_back({model_map, l.down_offset, down_bytes, 0});
        }
    } catch (...) {
        return 0;
    }
    g_q4k_kshard_blocked_ranges.swap(next);
    return 1;
}

static int cuda_q4k_kshard_evict_images_prepare(
        const void *model_map, uint32_t rank,
        const ds4_gpu_q4k_kshard_layer *layers, uint32_t n_layers) {
    std::vector<size_t> indices;
    size_t found_owned = 0;
    try {
        indices.reserve((size_t)n_layers * 3u);
    } catch (...) {
        return 0;
    }
    for (uint32_t il = 0; il < n_layers; ++il) {
        const ds4_gpu_q4k_kshard_layer &l = layers[il];
        if ((l.n_expert & 1u) != 0u) return 0;
        const uint64_t expert_bytes[3] = {
            (uint64_t)l.expert_mid_dim * l.gate_row_bytes,
            (uint64_t)l.expert_mid_dim * l.up_row_bytes,
            (uint64_t)l.out_dim * l.down_row_bytes,
        };
        const uint64_t tensor_offsets[3] = {
            l.gate_offset, l.up_offset, l.down_offset,
        };
        for (uint32_t which = 0; which < 3u; ++which) {
            const uint64_t owned_bytes =
                (uint64_t)(l.n_expert / 2u) * expert_bytes[which];
            const uint64_t owned_offset = tensor_offsets[which] +
                (uint64_t)rank * owned_bytes;
            const uint64_t tensor_bytes =
                (uint64_t)l.n_expert * expert_bytes[which];
            size_t containing = SIZE_MAX;
            for (size_t i = 0; i < g_model_images.size(); ++i) {
                const cuda_model_image &img = g_model_images[i];
                if (img.host_base != model_map) continue;
                const uint64_t img_end = img.device_offset + img.size;
                const uint64_t owned_end = owned_offset + owned_bytes;
                if (img.device_offset > owned_offset || img_end < owned_end) {
                    continue;
                }
                if (containing != SIZE_MAX) {
                    fprintf(stderr, DS4_GPU_LOG_PREFIX
                            "Q4_K K-shard found duplicate residency for "
                            "owned routed range: tensor_off=%llu "
                            "owned_off=%llu owned_bytes=%llu\n",
                            (unsigned long long)tensor_offsets[which],
                            (unsigned long long)owned_offset,
                            (unsigned long long)owned_bytes);
                    return 0;
                }
                containing = i;
            }
            if (containing != SIZE_MAX) {
                ++found_owned;
            }
        }
    }
    const size_t expected = (size_t)n_layers * 3u;
    if (found_owned != 0u && found_owned != expected) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "Q4_K K-shard refuses partial routed-image carving "
                "(%zu/%zu)\n", found_owned, expected);
        return 0;
    }
    if (found_owned == 0u) return 1;
    for (size_t i = 0; i < g_model_images.size(); ++i) {
        const cuda_model_image &img = g_model_images[i];
        if (img.host_base != model_map) continue;
        for (uint32_t il = 0; il < n_layers; ++il) {
            const ds4_gpu_q4k_kshard_layer &l = layers[il];
            const uint64_t expert_bytes[3] = {
                (uint64_t)l.expert_mid_dim * l.gate_row_bytes,
                (uint64_t)l.expert_mid_dim * l.up_row_bytes,
                (uint64_t)l.out_dim * l.down_row_bytes,
            };
            const uint64_t tensor_offsets[3] = {
                l.gate_offset, l.up_offset, l.down_offset,
            };
            for (uint32_t which = 0; which < 3u; ++which) {
                const uint64_t tensor_end = tensor_offsets[which] +
                    (uint64_t)l.n_expert * expert_bytes[which];
                const uint64_t img_end = img.device_offset + img.size;
                const uint64_t cut_begin =
                    std::max(img.device_offset, tensor_offsets[which]);
                const uint64_t cut_end = std::min(img_end, tensor_end);
                if (cut_begin < cut_end) {
                    indices.push_back(i);
                }
            }
        }
    }
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    uint64_t evict_bytes = 0;
    for (size_t i : indices) {
        const cuda_model_image &img = g_model_images[i];
        if (!img.device_ptr || !img.owns_device_ptr ||
            evict_bytes > UINT64_MAX - img.size) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "Q4_K K-shard refuses non-owning or invalid routed "
                    "device image during evacuation\n");
            return 0;
        }
        evict_bytes += img.size;
    }

    if (!cuda_ok(cudaDeviceSynchronize(),
                 "Q4_K K-shard pre-evacuation sync")) return 0;

    /* ROCm 7.14 on gfx1151 faults when the packed kernels read interior
     * pointers carved out of the old grouped model allocations.  Evacuate
     * each complete allocation that intersects routed weights, then reload
     * dense spans and packed routed slices into fresh allocations.  This is
     * a representation replacement, not a second persistent weight cache. */
    g_q4k_kshard.evacuated = 1;
    for (auto it = indices.rbegin(); it != indices.rend(); ++it) {
        cuda_model_image &img = g_model_images[*it];
        const cudaError_t err = cudaFree(img.device_ptr);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "Q4_K K-shard routed-image evacuation failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_model_images.erase(g_model_images.begin() + (ptrdiff_t)*it);
    }
    fprintf(stderr, DS4_GPU_LOG_PREFIX
            "Q4_K K-shard evacuated %zu routed-containing device images "
            "(%.2f GiB) for zero-net-memory packed reload\n",
            indices.size(), (double)evict_bytes / 1073741824.0);
    return 1;
}

static int cuda_q4k_kshard_attach_borrowed(
        cuda_q4k_packed_slice *p, uint64_t owned_offset,
        uint64_t owned_bytes) {
    if (!p) return 0;
    for (const cuda_model_image &img : g_q4k_kshard_borrowed_images) {
        if (img.host_base == p->host_base &&
            owned_offset >= img.device_offset &&
            owned_bytes <= img.size - (owned_offset - img.device_offset)) {
            if (p->packed_bytes != owned_bytes) return 0;
            p->device_ptr = img.device_ptr +
                            (owned_offset - img.device_offset);
            p->owns_device_ptr = 0;
            p->loaded = 0;
            return 1;
        }
    }
    return g_q4k_kshard_borrowed_images.empty();
}

static int cuda_q4k_kshard_dense_coverage_ok(
        const void *model_map, const uint64_t *offsets,
        const uint64_t *sizes, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        if (!cuda_model_image_range_ptr(model_map, offsets[i], sizes[i])) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "Q4_K K-shard dense coverage audit failed: span=%u "
                    "offset=%.6f GiB bytes=%llu\n", i,
                    (double)offsets[i] / 1073741824.0,
                    (unsigned long long)sizes[i]);
            return 0;
        }
    }
    return 1;
}

static int cuda_q4k_kshard_overlay_packed_window(
        const cuda_q4k_packed_slice &p, uint64_t packed_offset,
        char *dst, uint64_t bytes) {
    while (bytes != 0u) {
        const uint64_t expert = packed_offset / p.packed_expert_bytes;
        const uint64_t expert_offset =
            packed_offset % p.packed_expert_bytes;
        if (expert >= p.n_expert) return 0;
        uint64_t source_offset = 0;
        uint64_t take = 0;
        if (p.column_byte_base == 0u &&
            p.column_byte_count == p.source_row_bytes) {
            source_offset = p.tensor_offset +
                expert * p.source_expert_bytes +
                (uint64_t)p.row_base * p.source_row_bytes + expert_offset;
            take = std::min(bytes,
                            p.packed_expert_bytes - expert_offset);
        } else {
            const uint64_t row = expert_offset / p.column_byte_count;
            const uint64_t column = expert_offset % p.column_byte_count;
            if (row >= p.row_count) return 0;
            source_offset = p.tensor_offset +
                expert * p.source_expert_bytes +
                ((uint64_t)p.row_base + row) * p.source_row_bytes +
                p.column_byte_base + column;
            take = std::min(bytes, p.column_byte_count - column);
        }
        if (!cuda_pread_full(g_model_fd, dst, take, source_offset)) return 0;
        packed_offset += take;
        dst += take;
        bytes -= take;
    }
    return 1;
}

static int cuda_q4k_kshard_load_borrowed_roots(void) {
    if (g_q4k_kshard_borrowed_images.empty()) return 1;
    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    uint64_t stage_bytes = chunk;
    if (!cuda_u64_add_checked(
            stage_bytes,
            g_model_direct_align > 1u ? g_model_direct_align : 1u,
            &stage_bytes) || !cuda_model_stage_pool_alloc(stage_bytes)) {
        return 0;
    }
    for (const cuda_model_image &root : g_q4k_kshard_borrowed_images) {
        uint64_t root_offset = 0;
        while (root_offset < root.size) {
            const uint64_t n = std::min(chunk, root.size - root_offset);
            const char *payload = NULL;
            if (!cuda_model_stage_read(g_model_stage[0],
                                       g_model_stage_bytes,
                                       root.device_offset + root_offset, n,
                                       &payload)) return 0;
            if (payload != (const char *)g_model_stage[0]) {
                memmove(g_model_stage[0], payload, (size_t)n);
            }
            const uint64_t chunk_end = root_offset + n;
            for (const cuda_q4k_packed_slice &p : g_q4k_packed_slices) {
                if (!p.device_ptr || p.owns_device_ptr) continue;
                if (p.device_ptr < root.device_ptr) continue;
                const uint64_t packed_begin =
                    (uint64_t)(p.device_ptr - root.device_ptr);
                if (packed_begin > root.size ||
                    p.packed_bytes > root.size - packed_begin) continue;
                const uint64_t packed_end = packed_begin + p.packed_bytes;
                const uint64_t overlap_begin =
                    std::max(root_offset, packed_begin);
                const uint64_t overlap_end =
                    std::min(chunk_end, packed_end);
                if (overlap_begin >= overlap_end) continue;
                if (!cuda_q4k_kshard_overlay_packed_window(
                        p, overlap_begin - packed_begin,
                        (char *)g_model_stage[0] +
                            (overlap_begin - root_offset),
                        overlap_end - overlap_begin)) return 0;
            }
            cudaError_t err = cudaMemcpyAsync(
                root.device_ptr + root_offset, g_model_stage[0], (size_t)n,
                cudaMemcpyHostToDevice, g_model_upload_stream);
            if (err != cudaSuccess ||
                (err = cudaStreamSynchronize(g_model_upload_stream)) !=
                    cudaSuccess) return 0;
            root_offset += n;
        }
    }
    for (cuda_q4k_packed_slice &p : g_q4k_packed_slices) {
        if (!p.device_ptr || p.owns_device_ptr || p.loaded) continue;
        p.loaded = 1;
        g_q4k_packed_slice_bytes += p.packed_bytes;
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "loaded in-place packed Q4_K routed slice offset=%.2f GiB "
                "bytes=%.2f MiB\n",
                (double)p.tensor_offset / 1073741824.0,
                (double)p.packed_bytes / 1048576.0);
    }
    return 1;
}

static int cuda_q4k_kshard_snapshot_begin(
        const void *model_map, uint64_t model_size) {
    if (g_q4k_kshard.snapshot_valid || g_q4k_kshard.installed ||
        g_ssd_streaming_mode ||
        (g_model_host_base &&
         (g_model_host_base != model_map ||
          g_model_registered_size != model_size)) ||
        (!g_model_host_base &&
         (!g_model_ranges.empty() || !g_model_arenas.empty() ||
          !g_q8_f16_ranges.empty() ||
          !g_q8_f16_transpose_ranges.empty()))) return 0;

    const int saved_direct_fd =
        g_model_direct_fd >= 0 ? dup(g_model_direct_fd) : -1;
    if (g_model_direct_fd >= 0 && saved_direct_fd < 0) return 0;
    try {
        g_q4k_kshard_pre_images = g_model_images;
        g_q4k_kshard_borrowed_images.clear();
    } catch (...) {
        if (saved_direct_fd >= 0) (void)close(saved_direct_fd);
        return 0;
    }
    g_q4k_kshard.snapshot_valid = 1;
    g_q4k_kshard.model_map = model_map;
    g_q4k_kshard.model_size = model_size;
    g_q4k_kshard.pre_model_image_count = g_model_images.size();
    g_q4k_kshard.pre_model_host_base = g_model_host_base;
    g_q4k_kshard.pre_model_device_base = g_model_device_base;
    g_q4k_kshard.pre_model_registered_size = g_model_registered_size;
    g_q4k_kshard.pre_model_device_owned = g_model_device_owned;
    g_q4k_kshard.pre_model_range_mapping_supported =
        g_model_range_mapping_supported;
    g_q4k_kshard.pre_model_cache_full = g_model_cache_full;
    g_q4k_kshard.pre_model_fd = g_model_fd;
    g_q4k_kshard.pre_model_fd_host_base = g_model_fd_host_base;
    g_q4k_kshard.pre_model_direct_fd = saved_direct_fd;
    g_q4k_kshard.pre_model_direct_align = g_model_direct_align;
    g_q4k_kshard.pre_model_file_size = g_model_file_size;
    return 1;
}

static int cuda_q4k_kshard_enabled(void) {
    const char *legacy = getenv("DS4_ROCM_Q4K_KSHARD_RESEARCH");
    const char *disable = getenv("DS4_ROCM_DISABLE_Q4K_KSHARD");
    if (disable && disable[0] == '1' && disable[1] == '\0') return 0;
    return !legacy || (legacy[0] == '1' && legacy[1] == '\0');
}

extern "C" int ds4_gpu_q4k_kshard_install(
        const void *model_map, uint64_t model_size, int model_fd,
        uint32_t rank, const uint64_t *dense_offsets,
        const uint64_t *dense_sizes, uint32_t dense_count,
        uint64_t dense_max_tensor_bytes,
        const ds4_gpu_q4k_kshard_layer *layers, uint32_t n_layers) {
    if (!cuda_q4k_kshard_enabled() ||
        !model_map || model_size == 0u || rank > 1u ||
        !layers || n_layers == 0u ||
        dense_count == 0u || !dense_offsets || !dense_sizes) return 0;

    uint64_t key = UINT64_C(1469598103934665603);
    key = cuda_q4k_kshard_hash_bytes(key, &model_map, sizeof(model_map));
    key = cuda_q4k_kshard_hash_bytes(key, &model_size, sizeof(model_size));
    key = cuda_q4k_kshard_hash_bytes(key, &model_fd, sizeof(model_fd));
    key = cuda_q4k_kshard_hash_bytes(key, &rank, sizeof(rank));
    key = cuda_q4k_kshard_hash_bytes(key, &n_layers, sizeof(n_layers));
    for (uint32_t il = 0; il < n_layers; ++il) {
        key = cuda_q4k_kshard_hash_layer(key, layers[il]);
    }
    key = cuda_q4k_kshard_hash_bytes(key, &dense_count, sizeof(dense_count));
    key = cuda_q4k_kshard_hash_bytes(
        key, &dense_max_tensor_bytes, sizeof(dense_max_tensor_bytes));
    if (dense_count != 0u) {
        key = cuda_q4k_kshard_hash_bytes(
            key, dense_offsets, (size_t)dense_count * sizeof(dense_offsets[0]));
        key = cuda_q4k_kshard_hash_bytes(
            key, dense_sizes, (size_t)dense_count * sizeof(dense_sizes[0]));
    }
    if (g_q4k_kshard.installed) return g_q4k_kshard.key_hash == key;
    if (!g_q4k_packed_slices.empty()) return 0;
    if (!g_q4k_kshard_blocked_ranges.empty() &&
        g_q4k_kshard_blocked_ranges[0].host_base != model_map) return 0;

    const uint32_t n_expert = layers[0].n_expert;
    for (uint32_t il = 0; il < n_layers; ++il) {
        if (!cuda_q4k_kshard_layer_geometry_ok(&layers[il]) ||
            layers[il].n_expert != n_expert) return 0;
    }
    for (uint32_t i = 0; i < dense_count; ++i) {
        if (dense_offsets[i] > model_size || dense_sizes[i] == 0u ||
            dense_sizes[i] > model_size - dense_offsets[i]) return 0;
    }
    if (!cuda_q4k_kshard_dense_spans_exclude_routed(
            dense_offsets, dense_sizes, dense_count, layers, n_layers)) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "Q4_K K-shard dense spans intersect routed tensors\n");
        return 0;
    }

    if (!cuda_q4k_kshard_snapshot_begin(model_map, model_size)) return 0;
    if (!cuda_q4k_kshard_block_ranges_prepare(
            model_map, layers, n_layers)) {
        cuda_q4k_kshard_state_clear();
        return 0;
    }
    if (!ds4_gpu_set_model_fd_for_map(model_fd, model_map) ||
        !ds4_gpu_set_model_map_spans(
            model_map, model_size, dense_offsets, dense_sizes, dense_count,
            dense_max_tensor_bytes)) {
        cuda_q4k_kshard_state_clear();
        return 0;
    }
    if (!cuda_q4k_kshard_evict_images_prepare(
            model_map, rank, layers, n_layers)) {
        cuda_q4k_kshard_state_clear();
        return 0;
    }
    if (g_q4k_kshard.evacuated &&
        !ds4_gpu_set_model_map_spans(
            model_map, model_size, dense_offsets, dense_sizes, dense_count,
            dense_max_tensor_bytes)) {
        cuda_q4k_kshard_state_clear();
        return 0;
    }
    if (!cuda_q4k_kshard_dense_coverage_ok(
            model_map, dense_offsets, dense_sizes, dense_count)) {
        cuda_q4k_kshard_state_clear();
        return 0;
    }

    const uint64_t block = sizeof(cuda_block_q4_K);
    const uint32_t row_base = rank == 0u ? 0u : 1024u;
    const uint64_t down_base = rank == 0u ? 0u : 4u * block;
    int ok = 1;
    for (uint32_t il = 0; ok && il < n_layers; ++il) {
        const ds4_gpu_q4k_kshard_layer &l = layers[il];
        ok = ds4_gpu_q4k_packed_slice_declare(
                 model_map, model_size, l.gate_offset, n_expert, 2048u,
                 l.gate_row_bytes, row_base, 1024u, 0u,
                 l.gate_row_bytes, DS4_GPU_Q4K_PACKED_ROW_RANGE) &&
             ds4_gpu_q4k_packed_slice_declare(
                 model_map, model_size, l.up_offset, n_expert, 2048u,
                 l.up_row_bytes, row_base, 1024u, 0u,
                 l.up_row_bytes, DS4_GPU_Q4K_PACKED_ROW_RANGE) &&
             ds4_gpu_q4k_packed_slice_declare(
                 model_map, model_size, l.down_offset, n_expert, 4096u,
                 l.down_row_bytes, 0u, 4096u, down_base, 4u * block,
                 DS4_GPU_Q4K_PACKED_K_RANGE);
    }
    for (uint32_t il = 0; ok && il < n_layers; ++il) {
        const ds4_gpu_q4k_kshard_layer &l = layers[il];
        ok = ds4_gpu_q4k_packed_slice_load(
                 model_map, l.gate_offset, row_base, 1024u, 0u,
                 l.gate_row_bytes) &&
             ds4_gpu_q4k_packed_slice_load(
                 model_map, l.up_offset, row_base, 1024u, 0u,
                 l.up_row_bytes) &&
             ds4_gpu_q4k_packed_slice_load(
                 model_map, l.down_offset, 0u, 4096u, down_base,
                 4u * block);
    }
    if (!ok) {
        cuda_q4k_packed_slice_release_all();
        return 0;
    }

    ds4_gpu_q4k_kshard_windows windows = {};
    windows.rank = rank;
    windows.n_layers = n_layers;
    windows.n_expert = n_expert;
    windows.expert_in_dim = 4096u;
    windows.expert_mid_dim = 1024u;
    windows.out_dim = 4096u;
    windows.row_base = row_base;
    windows.row_count = 1024u;
    windows.source_gate_row_bytes = 16u * block;
    windows.source_down_row_bytes = 8u * block;
    windows.down_column_byte_base = down_base;
    windows.down_column_byte_count = 4u * block;
    windows.packed_gate_expert_bytes = 1024u * 16u * block;
    windows.packed_down_expert_bytes = 4096u * 4u * block;
    ds4_gpu_tp_suspend_expert_sharding(1);
    g_q4k_kshard.installed = 1;
    g_q4k_kshard.owns_shard_suspend = 1;
    g_q4k_kshard.model_map = model_map;
    g_q4k_kshard.model_size = model_size;
    g_q4k_kshard.key_hash = key;
    g_q4k_kshard.windows = windows;
    return 1;
}

extern "C" int ds4_gpu_q4k_kshard_windows_get(
        ds4_gpu_q4k_kshard_windows *windows) {
    if (windows) memset(windows, 0, sizeof(*windows));
    if (!windows || !g_q4k_kshard.installed) return 0;
    *windows = g_q4k_kshard.windows;
    return 1;
}

extern "C" void ds4_gpu_q4k_kshard_release(void) {
    cuda_q4k_packed_slice_release_all();
}

extern "C" int ds4_gpu_set_model_fd(int fd) {
    g_model_fd = fd;
    g_model_fd_host_base = g_model_host_base;
    g_model_file_size = 0;
    if (g_model_direct_fd >= 0) {
        (void)close(g_model_direct_fd);
        g_model_direct_fd = -1;
    }
    g_model_direct_align = 1;
    if (fd >= 0) {
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size > 0) {
            g_model_file_size = (uint64_t)st.st_size;
            if (st.st_blksize > 1) g_model_direct_align = (uint64_t)st.st_blksize;
        }
#if defined(__linux__) && defined(O_DIRECT)
        {
            char proc_path[64];
            snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", fd);
            int direct_fd = open(proc_path, O_RDONLY | O_DIRECT);
            if (direct_fd >= 0) {
                g_model_direct_fd = direct_fd;
                if (g_model_direct_align < 512) g_model_direct_align = 512;
            }
        }
#endif
    }
    return 1;
}

extern "C" int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label) {
    if (!model_map || bytes == 0) return 1;
    if (offset > model_size || bytes > model_size - offset) return 0;
    if (!cuda_model_range_ptr(model_map, offset, bytes, label ? label : "model_tensor")) return 0;
    return cuda_model_range_is_cached(model_map, offset, bytes);
}

extern "C" int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label) {
    if (!model_map || bytes == 0) return 1;
    if (offset > model_size || bytes > model_size - offset) return 0;
    static int optional_q8_preload_disabled = 0;
    if (optional_q8_preload_disabled) return 1;
    const char *cache_label = label ? label : "q8_0";
    if (!cuda_q8_f16_preload_allowed(cache_label, in_dim, out_dim)) return 1;
    const int preload_transposed_b = !g_quality_mode &&
                                     strstr(cache_label, "attn_output_b") != NULL;
    if (preload_transposed_b) {
        const __half *f16_t = cuda_q8_f16_transpose_ptr(model_map, offset, bytes, in_dim, out_dim, cache_label);
        if (f16_t) {
            if (strstr(cache_label, "attn_output_b") != NULL && in_dim == 8192u && out_dim == 4096u) {
                cuda_q8_f16_warmup_attention_output_b_gemm(f16_t, in_dim, out_dim);
            }
            return 1;
        }
    } else {
        const __half *f16 = cuda_q8_f16_ptr(model_map, offset, bytes, in_dim, out_dim, cache_label);
        if (f16) {
            if (strstr(cache_label, "attn_output_a") != NULL && in_dim == 4096u && out_dim == 8192u) {
                cuda_q8_f16_warmup_attention_output_a_gemm(f16, in_dim, 1024u, 8u);
            }
            return 1;
        }
    }
    optional_q8_preload_disabled = 1;
    return 1;
}

extern "C" void ds4_gpu_release_q8_f16_cache(void) {
    cuda_q8_f16_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
}

extern "C" void ds4_gpu_print_memory_report(const char *label) {
    size_t free_b = 0, total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "memory %s: query failed: %s\n",
                label ? label : "", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return;
    }
    const uint64_t used_b = (uint64_t)total_b - (uint64_t)free_b;
    const char *placement = cuda_model_image_bytes() ? "device_copy" : "mapped/range_cache";
    fprintf(stderr,
            DS4_GPU_LOG_PREFIX "memory %s: used=%.2f GiB free=%.2f GiB total=%.2f GiB "
            "placement=%s model_image=%.2f GiB range_cache=%.2f GiB "
            "q8_f16_cache=%.2f GiB scratch=%.2f GiB",
            label ? label : "",
            (double)used_b / 1073741824.0,
            (double)free_b / 1073741824.0,
            (double)total_b / 1073741824.0,
            placement,
            (double)cuda_model_image_bytes() / 1073741824.0,
            (double)g_model_range_bytes / 1073741824.0,
            (double)g_q8_f16_bytes / 1073741824.0,
            (double)g_cuda_tmp_bytes / 1073741824.0);
    fprintf(stderr, "\n");
}

extern "C" int ds4_gpu_memory_info(uint64_t *free_bytes,
                                     uint64_t *total_bytes) {
    if (free_bytes) *free_bytes = 0u;
    if (total_bytes) *total_bytes = 0u;
    if (!free_bytes || !total_bytes) return 0;
    size_t free_b = 0u, total_b = 0u;
    const cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    *free_bytes = (uint64_t)free_b;
    *total_bytes = (uint64_t)total_b;
    return 1;
}

extern "C" void ds4_gpu_set_quality(bool quality) {
    const int new_quality_mode = quality ? 1 : 0;
    if (g_quality_mode != new_quality_mode) {
        g_rocm_cfg.initialized = 0;
    }
    g_quality_mode = new_quality_mode;
    if (g_cublas_ready) {
        const cublasMath_t math_mode = g_quality_mode ? CUBLAS_DEFAULT_MATH : CUBLAS_TF32_TENSOR_OP_MATH;
        (void)cublasSetMathMode(g_cublas, math_mode);
    }
}
