#include <hip/hip_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_gpu_mgpu.h"
#include "ds4_gpu.h"
#include "ds4_gpu_args.h"

#include "rocm/ds4_rocm_glm5_kda.cuh"

ds4_gpu_ctx g_gpu[DS4_MAX_GPUS] = {};
int g_n_gpus = 1;
int g_gpu_peer_ok[DS4_MAX_GPUS][DS4_MAX_GPUS] = {{1}};

static int rocm_tier_valid(int tier) {
    return tier == 0 && g_n_gpus == 1;
}

static int rocm_ranges_overlap(const ds4_gpu_tensor *a,
                               const ds4_gpu_tensor *b) {
    if (!a || !b || !a->ptr || !b->ptr) return 0;
    const uintptr_t a0 = (uintptr_t)a->ptr;
    const uintptr_t b0 = (uintptr_t)b->ptr;
    if (a->bytes > UINTPTR_MAX - a0 || b->bytes > UINTPTR_MAX - b0) return 1;
    return a0 < b0 + b->bytes && b0 < a0 + a->bytes;
}

extern "C" int ds4_gpu_glm5_causal_conv4_tensor(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *history,
        const ds4_gpu_tensor *input,
        const ds4_gpu_tensor *weight,
        uint32_t n_tokens,
        uint32_t channels) {
    constexpr uint32_t expected_channels = 8192u;
    if (!out || !history || !input || !weight || !out->ptr ||
        !history->ptr || !input->ptr || !weight->ptr || n_tokens == 0u ||
        channels != expected_channels) {
        return 0;
    }
    const uint64_t row_bytes = (uint64_t)channels * sizeof(float);
    const uint64_t io_bytes = (uint64_t)n_tokens * row_bytes;
    const uint64_t history_bytes =
        (uint64_t)channels * 3u * sizeof(float);
    const uint64_t weight_bytes =
        (uint64_t)channels * 4u * sizeof(float);
    if (out->bytes < io_bytes || input->bytes < io_bytes ||
        history->bytes < history_bytes || weight->bytes < weight_bytes ||
        out->device_id != history->device_id ||
        out->device_id != input->device_id ||
        out->device_id != weight->device_id ||
        rocm_ranges_overlap(out, history)) {
        return 0;
    }
    constexpr uint32_t threads = 256u;
    hipLaunchKernelGGL(ds4_glm5_causal_conv4_kernel,
                       dim3((channels + threads - 1u) / threads),
                       dim3(threads), 0, 0,
                       (float *)out->ptr, (float *)history->ptr,
                       (const float *)input->ptr,
                       (const float *)weight->ptr, n_tokens, channels);
    return hipGetLastError() == hipSuccess;
}

static int rocm_glm5_wave32_available(void) {
#if defined(DS4_GFX1151_WAVE32) && DS4_GFX1151_WAVE32
    static int cached = -1;
    if (cached >= 0) return cached;
    int device = 0;
    hipDeviceProp_t properties = {};
    if (hipGetDevice(&device) != hipSuccess ||
        hipGetDeviceProperties(&properties, device) != hipSuccess) {
        cached = 0;
    } else {
        cached = properties.warpSize == 32 ? 1 : 0;
    }
    return cached;
#else
    return 0;
#endif
}

extern "C" int ds4_gpu_glm5_kda_recurrent_tensor(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *state,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *beta,
        uint32_t n_tokens,
        uint32_t n_heads,
        uint32_t head_dim) {
#if !defined(DS4_GFX1151_WAVE32) || !DS4_GFX1151_WAVE32
    (void)out;
    (void)state;
    (void)q;
    (void)k;
    (void)v;
    (void)gate;
    (void)beta;
    (void)n_tokens;
    (void)n_heads;
    (void)head_dim;
    return 0;
#else
    constexpr uint32_t expected_heads = 64u;
    constexpr uint32_t expected_dim = 128u;
    if (!out || !state || !q || !k || !v || !gate || !beta ||
        !out->ptr || !state->ptr || !q->ptr || !k->ptr || !v->ptr ||
        !gate->ptr || !beta->ptr || n_tokens == 0u ||
        n_heads != expected_heads || head_dim != expected_dim ||
        !rocm_glm5_wave32_available()) {
        return 0;
    }
    const uint64_t row_values = (uint64_t)n_heads * head_dim;
    const uint64_t vector_bytes =
        (uint64_t)n_tokens * row_values * sizeof(float);
    const uint64_t beta_bytes =
        (uint64_t)n_tokens * n_heads * sizeof(float);
    const uint64_t state_bytes =
        (uint64_t)n_heads * head_dim * head_dim * sizeof(float);
    if (out->bytes < vector_bytes || q->bytes < vector_bytes ||
        k->bytes < vector_bytes || v->bytes < vector_bytes ||
        gate->bytes < vector_bytes || beta->bytes < beta_bytes ||
        state->bytes < state_bytes ||
        out->device_id != state->device_id ||
        out->device_id != q->device_id || out->device_id != k->device_id ||
        out->device_id != v->device_id || out->device_id != gate->device_id ||
        out->device_id != beta->device_id ||
        rocm_ranges_overlap(out, state) || rocm_ranges_overlap(out, q) ||
        rocm_ranges_overlap(out, k) || rocm_ranges_overlap(out, v) ||
        rocm_ranges_overlap(out, gate) || rocm_ranges_overlap(out, beta) ||
        rocm_ranges_overlap(state, q) || rocm_ranges_overlap(state, k) ||
        rocm_ranges_overlap(state, v) || rocm_ranges_overlap(state, gate) ||
        rocm_ranges_overlap(state, beta)) {
        return 0;
    }
    hipLaunchKernelGGL(ds4_glm5_kda_wave32_kernel,
                       dim3(1u, n_heads, head_dim / 4u),
                       dim3(32u, 4u), 0, 0,
                       (float *)out->ptr, (float *)state->ptr,
                       (const float *)q->ptr, (const float *)k->ptr,
                       (const float *)v->ptr, (const float *)gate->ptr,
                       (const float *)beta->ptr, n_tokens);
    return hipGetLastError() == hipSuccess;
#endif
}

extern "C" int ds4_gpu_init_multi(const ds4_gpu_config *cfg) {
    if (!cfg || cfg->n_gpus != 1) {
        fprintf(stderr, "ds4: ROCm supports one GPU per process\n");
        return 0;
    }
    g_gpu[0].device_id = cfg->device_indices[0];
    if (hipSetDevice(g_gpu[0].device_id) != hipSuccess) return 0;
    return ds4_gpu_init();
}

extern "C" int ds4_gpu_set_current_device(int tier) {
    if (!rocm_tier_valid(tier)) return 1;
    return hipSetDevice(g_gpu[0].device_id) == hipSuccess ? 0 : 1;
}

extern "C" int ds4_gpu_set_current_device_fenced(int tier) {
    return ds4_gpu_set_current_device(tier);
}

extern "C" int ds4_gpu_tensor_alloc_on(ds4_gpu_tensor *t, int tier,
                                         uint64_t bytes) {
    if (!t) return 1;
    if (!rocm_tier_valid(tier)) return 2;
    if (bytes == 0) bytes = 1;
    if (hipMalloc(&t->ptr, (size_t)bytes) != hipSuccess) return 3;
    t->bytes = bytes;
    t->owner = 1;
    t->device_id = 0;
    return 0;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_ptr_on(int tier,
                                                         uint64_t bytes) {
    if (!rocm_tier_valid(tier)) return NULL;
    return ds4_gpu_tensor_alloc(bytes);
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed_on(int tier,
                                                             uint64_t bytes) {
    if (!rocm_tier_valid(tier)) return NULL;
    return ds4_gpu_tensor_alloc_managed(bytes);
}

extern "C" void ds4_gpu_tensor_free_in_place(ds4_gpu_tensor *t) {
    if (!t) return;
    if (t->owner && t->ptr) (void)hipFree(t->ptr);
    memset(t, 0, sizeof(*t));
}

extern "C" int ds4_gpu_tensor_device(const ds4_gpu_tensor *t) {
    return t ? 0 : -1;
}

extern "C" int ds4_gpu_tensor_copy_async(ds4_gpu_tensor *dst,
                                           const ds4_gpu_tensor *src,
                                           uint64_t bytes) {
    if (!dst || !src || bytes > dst->bytes || bytes > src->bytes) return 0;
    if (bytes == 0) return 1;
    return hipMemcpyAsync(dst->ptr, src->ptr, (size_t)bytes,
                          hipMemcpyDeviceToDevice, 0) == hipSuccess;
}

extern "C" int ds4_gpu_tensor_copy_xdev(ds4_gpu_tensor *dst,
                                          const ds4_gpu_tensor *src,
                                          uint64_t bytes) {
    return ds4_gpu_tensor_copy(dst, 0, src, 0, bytes);
}

extern "C" int ds4_gpu_tensor_copy_xdev_default(ds4_gpu_tensor *dst,
                                                  const ds4_gpu_tensor *src,
                                                  uint64_t bytes) {
    return ds4_gpu_tensor_copy_xdev(dst, src, bytes);
}

extern "C" int ds4_gpu_tensor_copy_xdev_ordered(ds4_gpu_tensor *dst,
                                                  const ds4_gpu_tensor *src,
                                                  uint64_t bytes) {
    return ds4_gpu_tensor_copy_xdev(dst, src, bytes);
}

extern "C" int ds4_gpu_tensor_copy_xdev3(
        ds4_gpu_tensor *dst0, const ds4_gpu_tensor *src0, uint64_t bytes0,
        ds4_gpu_tensor *dst1, const ds4_gpu_tensor *src1, uint64_t bytes1,
        ds4_gpu_tensor *dst2, const ds4_gpu_tensor *src2, uint64_t bytes2) {
    return (bytes0 == 0 || ds4_gpu_tensor_copy_xdev(dst0, src0, bytes0)) &&
           (bytes1 == 0 || ds4_gpu_tensor_copy_xdev(dst1, src1, bytes1)) &&
           (bytes2 == 0 || ds4_gpu_tensor_copy_xdev(dst2, src2, bytes2));
}

extern "C" int ds4_gpu_tensor_copy_xdev3_default_dst(
        ds4_gpu_tensor *dst0, const ds4_gpu_tensor *src0, uint64_t bytes0,
        ds4_gpu_tensor *dst1, const ds4_gpu_tensor *src1, uint64_t bytes1,
        ds4_gpu_tensor *dst2, const ds4_gpu_tensor *src2, uint64_t bytes2) {
    return ds4_gpu_tensor_copy_xdev3(dst0, src0, bytes0, dst1, src1, bytes1,
                                     dst2, src2, bytes2);
}

extern "C" int ds4_gpu_tensor_wait_xdev(const ds4_gpu_tensor *src,
                                          int dst_tier) {
    return src && rocm_tier_valid(dst_tier);
}

extern "C" int ds4_gpu_tensor_wait_xdev_default(const ds4_gpu_tensor *src,
                                                  int dst_tier) {
    return ds4_gpu_tensor_wait_xdev(src, dst_tier);
}

extern "C" uint64_t ds4_gpu_tier_free_vram(int tier) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (!rocm_tier_valid(tier) ||
        hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
        return 0;
    }
    return (uint64_t)free_bytes;
}

extern "C" int ds4_gpu_args_probe_auto_cuda(
        const int *device_filter, int filter_len, ds4_gpu_config *out,
        size_t safety_margin_bytes, char *errbuf, size_t errbuflen) {
    if (!out) {
        if (errbuf && errbuflen) snprintf(errbuf, errbuflen, "internal: NULL out");
        return 1;
    }
    int visible = 0;
    hipError_t rc = hipGetDeviceCount(&visible);
    if (rc != hipSuccess || visible <= 0) {
        if (errbuf && errbuflen) {
            snprintf(errbuf, errbuflen, "hipGetDeviceCount failed: %s",
                     rc == hipSuccess ? "no devices" : hipGetErrorString(rc));
        }
        return 1;
    }
    if (filter_len > 1 || (!device_filter && visible > 1)) {
        if (errbuf && errbuflen) {
            snprintf(errbuf, errbuflen,
                     "ROCm supports one GPU per process; select one device");
        }
        return 1;
    }
    const int device = device_filter && filter_len == 1 ? device_filter[0] : 0;
    if (device < 0 || device >= visible || hipSetDevice(device) != hipSuccess) {
        if (errbuf && errbuflen) {
            snprintf(errbuf, errbuflen, "invalid ROCm device %d", device);
        }
        return 1;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    rc = hipMemGetInfo(&free_bytes, &total_bytes);
    if (rc != hipSuccess) {
        if (errbuf && errbuflen) {
            snprintf(errbuf, errbuflen, "hipMemGetInfo failed: %s",
                     hipGetErrorString(rc));
        }
        return 1;
    }
    const size_t reserve_floor = (size_t)2ull * 1024ull * 1024ull * 1024ull;
    const size_t reserve_pct = free_bytes / 20u;
    const size_t reserve = reserve_floor > reserve_pct ? reserve_floor : reserve_pct;
    memset(out, 0, sizeof(*out));
    out->device_indices[0] = device;
    out->vram_bytes[0] = free_bytes > reserve ? free_bytes - reserve : 0;
    out->n_gpus = 1;
    out->safety_margin_bytes = safety_margin_bytes;
    return 0;
}

extern "C" void ds4_gpu_enable_q8_dequant_gemm(void) {
}

static int g_rocm_q8_cache_suppressed = 0;

extern "C" int ds4_gpu_q8_cache_suppressed(void) {
    return g_rocm_q8_cache_suppressed;
}

extern "C" void ds4_gpu_set_q8_cache_suppressed(int suppressed) {
    g_rocm_q8_cache_suppressed = suppressed != 0;
}

extern "C" int ds4_gpu_set_decode_fast_attention(int enabled) {
    (void)enabled;
    return 0;
}

extern "C" int ds4_gpu_set_decode_score_vec4(int enabled) {
    (void)enabled;
    return 0;
}

extern "C" int ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint32_t n_rows) {
    return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                      weight_offset, in_dim, out_dim, x,
                                      n_rows);
}

extern "C" int ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor(
        ds4_gpu_tensor *out0, ds4_gpu_tensor *out1, const void *model_map,
        uint64_t model_size, uint64_t weight0_offset,
        uint64_t weight1_offset, uint64_t in_dim, uint64_t out0_dim,
        uint64_t out1_dim, const ds4_gpu_tensor *x, uint32_t n_rows) {
    return ds4_gpu_matmul_q8_0_pair_tensor(
            out0, out1, model_map, model_size, weight0_offset, weight1_offset,
            in_dim, out0_dim, out1_dim, x, n_rows);
}

extern "C" int ds4_gpu_matmul_f16_router_rows_exact_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, const ds4_gpu_tensor *x, uint32_t n_rows) {
    if (!out || !x || n_rows == 0u ||
        x->bytes < (uint64_t)n_rows * 4096u * sizeof(float) ||
        out->bytes < (uint64_t)n_rows * 256u * sizeof(float)) {
        return 0;
    }
    for (uint32_t row = 0; row < n_rows; row++) {
        ds4_gpu_tensor in_row = *x;
        ds4_gpu_tensor out_row = *out;
        in_row.ptr = (char *)x->ptr +
            (uint64_t)row * 4096u * sizeof(float);
        in_row.bytes = 4096u * sizeof(float);
        in_row.owner = 0;
        out_row.ptr = (char *)out->ptr +
            (uint64_t)row * 256u * sizeof(float);
        out_row.bytes = 256u * sizeof(float);
        out_row.owner = 0;
        if (!ds4_gpu_matmul_f16_tensor(
                    &out_row, model_map, model_size, weight_offset,
                    4096u, 256u, &in_row, 1u)) {
            return 0;
        }
    }
    return 1;
}

extern "C" int ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
        ds4_gpu_tensor *q_out, const ds4_gpu_tensor *q,
        const void *model_map, uint64_t model_size,
        uint64_t q_weight_offset, uint32_t q_n,
        ds4_gpu_tensor *kv_out, const ds4_gpu_tensor *kv,
        uint64_t kv_weight_offset, uint32_t kv_n, uint32_t rows,
        uint32_t kv_n_head, uint32_t kv_head_dim, uint32_t n_rot,
        uint32_t pos0, uint32_t n_ctx_orig, bool inverse,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow, float eps) {
    return ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
                   q_out, q, model_map, model_size, q_weight_offset, q_n,
                   kv_out, kv, kv_weight_offset, kv_n, rows, eps) != 0 &&
           ds4_gpu_rope_tail_tensor(
                   kv_out, rows, kv_n_head, kv_head_dim, n_rot, pos0,
                   n_ctx_orig, inverse, freq_base, freq_scale, ext_factor,
                   attn_factor, beta_fast, beta_slow) != 0;
}

extern "C" int ds4_gpu_embed_token_quant_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t weight_type, uint32_t n_vocab,
        uint32_t token, uint32_t n_embd) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_embed_token_q8_0_tensor(out, model_map, model_size,
                                           weight_offset, n_vocab, token,
                                           n_embd);
}

extern "C" int ds4_gpu_embed_tokens_quant_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *tokens,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t weight_type, uint32_t n_vocab, uint32_t n_tokens,
        uint32_t n_embd) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_embed_tokens_q8_0_tensor(out, tokens, model_map, model_size,
                                            weight_offset, n_vocab, n_tokens,
                                            n_embd);
}

extern "C" int ds4_gpu_matmul_quant_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t weight_type, uint64_t in_dim,
        uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (weight_type == 8u) {
        return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                          weight_offset, in_dim, out_dim, x,
                                          n_tok);
    }
    if (weight_type == 1u) {
        return ds4_gpu_matmul_f16_tensor(out, model_map, model_size,
                                         weight_offset, in_dim, out_dim, x,
                                         n_tok);
    }
    return 0;
}

extern "C" int ds4_gpu_matmul_quant_decode_mpp_model_view_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t weight_type, uint64_t in_dim,
        uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (weight_type == 8u) {
        return ds4_gpu_matmul_q8_0_decode_mpp_model_view_tensor(
                out, model_map, model_size, weight_offset, in_dim, out_dim,
                x, n_tok);
    }
    return ds4_gpu_matmul_quant_tensor(out, model_map, model_size,
                                       weight_offset, weight_type, in_dim,
                                       out_dim, x, n_tok);
}

extern "C" int ds4_gpu_glm_k_b_project_typed_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *kv_norm,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t weight_type, uint32_t n_tokens, uint32_t kv_lora_dim,
        uint32_t qk_nope, uint32_t n_head) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_glm_k_b_project_tensor(out, kv_norm, model_map, model_size,
                                           weight_offset, n_tokens,
                                           kv_lora_dim, qk_nope, n_head);
}

extern "C" int ds4_gpu_glm_qk_lowrank_typed_tensor(
        ds4_gpu_tensor *qk_low, const ds4_gpu_tensor *q,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t weight_type, uint32_t n_head, uint32_t kv_lora_dim,
        uint32_t qk_nope, uint32_t qk_dim) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_glm_qk_lowrank_q8_0_tensor(
            qk_low, q, model_map, model_size, weight_offset, n_head,
            kv_lora_dim, qk_nope, qk_dim);
}

extern "C" int ds4_gpu_glm_qk_lowrank_typed_batch_tensor(
        ds4_gpu_tensor *qk_low, const ds4_gpu_tensor *q,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t weight_type, uint32_t n_tokens, uint32_t n_head,
        uint32_t kv_lora_dim, uint32_t qk_nope, uint32_t qk_dim) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_glm_qk_lowrank_q8_0_batch_tensor(
            qk_low, q, model_map, model_size, weight_offset, n_tokens, n_head,
            kv_lora_dim, qk_nope, qk_dim);
}

extern "C" int ds4_gpu_glm_value_project_typed_batch_heads_tensor(
        ds4_gpu_tensor *heads, const ds4_gpu_tensor *lora,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t weight_type, uint32_t n_tokens, uint32_t n_head,
        uint32_t kv_lora_dim, uint32_t value_dim) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_glm_value_project_q8_0_batch_heads_tensor(
            heads, lora, model_map, model_size, weight_offset, n_tokens,
            n_head, kv_lora_dim, value_dim);
}

extern "C" int ds4_gpu_glm_attention_indexed_decode_typed_tensor(
        ds4_gpu_tensor *heads, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low, const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache, const void *model_map,
        uint64_t model_size, uint64_t value_weight_offset,
        uint32_t value_weight_type, const ds4_gpu_tensor *selected,
        uint32_t n_selected, uint32_t cache_cap, bool cache_f16,
        uint32_t n_head, uint32_t kv_lora_dim, uint32_t qk_nope,
        uint32_t qk_rope, uint32_t value_dim, uint32_t n_ctx_orig,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow) {
    if (value_weight_type != 8u) return 0;
    return ds4_gpu_glm_attention_indexed_decode_tensor(
            heads, q, qk_low, kv_lora_cache, k_rope_cache, model_map,
            model_size, value_weight_offset, selected, n_selected, cache_cap,
            cache_f16, n_head, kv_lora_dim, qk_nope, qk_rope, value_dim,
            n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow);
}

extern "C" int ds4_gpu_glm_attention_indexed_batch_typed_tensor(
        ds4_gpu_tensor *heads, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low, const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache, const void *model_map,
        uint64_t model_size, uint64_t value_weight_offset,
        uint32_t value_weight_type, const ds4_gpu_tensor *selected,
        uint32_t n_tokens, uint32_t n_selected, uint32_t cache_cap,
        bool cache_f16, uint32_t n_head, uint32_t kv_lora_dim,
        uint32_t qk_nope, uint32_t qk_rope, uint32_t value_dim,
        uint32_t n_ctx_orig, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast,
        float beta_slow) {
    if (value_weight_type != 8u) return 0;
    return ds4_gpu_glm_attention_indexed_batch_tensor(
            heads, q, qk_low, kv_lora_cache, k_rope_cache, model_map,
            model_size, value_weight_offset, selected, n_tokens, n_selected,
            cache_cap, cache_f16, n_head, kv_lora_dim, qk_nope, qk_rope,
            value_dim, n_ctx_orig, freq_base, freq_scale, ext_factor,
            attn_factor, beta_fast, beta_slow);
}
