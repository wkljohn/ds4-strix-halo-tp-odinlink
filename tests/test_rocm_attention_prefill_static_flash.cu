/* Focused regression test for the opt-in single-pass static mixed-attention
 * prefill kernel.  The legacy two-pass kernel is the numerical oracle here;
 * the flash form may reorder FP32 arithmetic but must remain close, finite,
 * and exactly deterministic across identical launches. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define CHECK(cond, msg) do {                                                \
    if (!(cond)) {                                                           \
        fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);          \
        return 1;                                                            \
    }                                                                        \
} while (0)

typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

static double rel_rms(const std::vector<float> &a,
                      const std::vector<float> &b) {
    double diff2 = 0.0, ref2 = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = (double)a[i] - b[i];
        diff2 += d * d;
        ref2 += (double)b[i] * b[i];
    }
    return sqrt(diff2 / fmax(ref2, 1.0e-30));
}

static uint64_t align64(uint64_t v) {
    return (v + 63u) & ~UINT64_C(63);
}

static void pack_q8_rows(unsigned char *dst, uint32_t in_dim,
                         uint32_t out_dim, uint32_t salt) {
    const uint32_t blocks = in_dim / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    for (uint32_t row = 0; row < out_dim; row++) {
        for (uint32_t b = 0; b < blocks; b++) {
            unsigned char *blk = dst + (uint64_t)row * row_bytes +
                                 (uint64_t)b * 34u;
            const __half scale = __float2half(
                0.0015f + 0.000125f * (float)((row + 3u * b + salt) % 23u));
            memcpy(blk, &scale, sizeof(scale));
            int8_t *q = (int8_t *)(blk + 2u);
            for (uint32_t k = 0; k < 32u; k++) {
                q[k] = (int8_t)((int)((row * 13u + b * 7u + k * 5u + salt) %
                                      255u) - 127);
            }
        }
    }
}

static int test_compressor_wavefront(
        const std::vector<unsigned char> &model,
        uint64_t ape_offset,
        uint64_t norm_offset,
        uint32_t head_dim,
        bool quantize_fp8,
        bool indexer_qat) {
    constexpr uint32_t n_tokens = 2048;
    constexpr uint32_t n_chunks = 4;
    constexpr uint32_t chunk_rows = n_tokens / n_chunks;
    constexpr uint32_t ratio = 4;
    constexpr uint32_t n_rot = 64;
    const uint32_t width = 2u * head_dim;
    const uint32_t n_comp = n_tokens / ratio;
    const uint32_t chunk_comp = chunk_rows / ratio;
    const uint64_t input_elems = (uint64_t)n_tokens * width;
    const uint64_t comp_elems = (uint64_t)n_comp * head_dim;
    const uint64_t state_elems = (uint64_t)8u * width;
    std::vector<float> kv((size_t)input_elems);
    std::vector<float> score((size_t)input_elems);
    for (uint64_t i = 0; i < input_elems; i++) {
        kv[(size_t)i] = 0.13f * sinf((float)(i * 31u + 3u) * 0.00091f);
        score[(size_t)i] =
            0.07f * cosf((float)(i * 17u + 9u) * 0.00113f);
    }

    ds4_gpu_tensor kv_dev = {}, score_dev = {};
    ds4_gpu_tensor full_comp = {}, chunked_comp = {};
    ds4_gpu_tensor full_state_kv = {}, full_state_score = {};
    ds4_gpu_tensor chunked_state_kv = {}, chunked_state_score = {};
    CHECK(ds4_gpu_tensor_alloc_on(&kv_dev, 0,
                                  input_elems * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&score_dev, 0,
                                  input_elems * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&full_comp, 0,
                                  comp_elems * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&chunked_comp, 0,
                                  comp_elems * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&full_state_kv, 0,
                                  state_elems * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&full_state_score, 0,
                                  state_elems * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&chunked_state_kv, 0,
                                  state_elems * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&chunked_state_score, 0,
                                  state_elems * sizeof(float)) == 0,
          "allocate compressor exactness tensors");
    CHECK(ds4_gpu_tensor_write(&kv_dev, 0, kv.data(),
                               input_elems * sizeof(float)) &&
          ds4_gpu_tensor_write(&score_dev, 0, score.data(),
                               input_elems * sizeof(float)),
          "upload compressor inputs");

    CHECK(ds4_gpu_compressor_prefill_tensor(
              &full_comp, &full_state_kv, &full_state_score,
              &kv_dev, &score_dev, model.data(), model.size(),
              ape_offset, 1u, norm_offset, 0u, head_dim, ratio,
              0u, n_tokens, n_rot, 0u, quantize_fp8,
              160000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f),
          "run full compressor prefill");
    if (indexer_qat) {
        CHECK(ds4_gpu_dsv4_indexer_qat_tensor(
                  &full_comp, n_comp, head_dim),
              "run full indexer QAT");
    }

    for (uint32_t chunk = 0; chunk < n_chunks; chunk++) {
        const uint32_t row0 = chunk * chunk_rows;
        ds4_gpu_tensor *kv_view = ds4_gpu_tensor_view(
            &kv_dev, (uint64_t)row0 * width * sizeof(float),
            (uint64_t)chunk_rows * width * sizeof(float));
        ds4_gpu_tensor *score_view = ds4_gpu_tensor_view(
            &score_dev, (uint64_t)row0 * width * sizeof(float),
            (uint64_t)chunk_rows * width * sizeof(float));
        ds4_gpu_tensor *comp_view = ds4_gpu_tensor_view(
            &chunked_comp, (uint64_t)chunk * chunk_comp * head_dim *
                               sizeof(float),
            (uint64_t)chunk_comp * head_dim * sizeof(float));
        CHECK(kv_view && score_view && comp_view,
              "create chunked compressor views");
        const int chunk_ok = chunk == 0u
            ? ds4_gpu_compressor_prefill_tensor(
                  comp_view, &chunked_state_kv, &chunked_state_score,
                  kv_view, score_view, model.data(), model.size(),
                  ape_offset, 1u, norm_offset, 0u, head_dim, ratio,
                  row0, chunk_rows, n_rot, 0u, quantize_fp8,
                  160000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f)
            : ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                  comp_view, &chunked_state_kv, &chunked_state_score,
                  kv_view, score_view, model.data(), model.size(),
                  ape_offset, 1u, norm_offset, 0u, head_dim,
                  row0, chunk_rows, n_rot, 0u, quantize_fp8,
                  160000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f);
        if (chunk_ok && indexer_qat) {
            CHECK(ds4_gpu_dsv4_indexer_qat_tensor(
                      comp_view, chunk_comp, head_dim),
                  "run chunked indexer QAT");
        }
        ds4_gpu_tensor_free(comp_view);
        ds4_gpu_tensor_free(score_view);
        ds4_gpu_tensor_free(kv_view);
        CHECK(chunk_ok, "run chunked compressor prefill/replay");
    }

    std::vector<float> full_comp_host((size_t)comp_elems);
    std::vector<float> chunked_comp_host((size_t)comp_elems);
    std::vector<float> full_state_kv_host((size_t)state_elems);
    std::vector<float> chunked_state_kv_host((size_t)state_elems);
    std::vector<float> full_state_score_host((size_t)state_elems);
    std::vector<float> chunked_state_score_host((size_t)state_elems);
    CHECK(ds4_gpu_tensor_read(&full_comp, 0, full_comp_host.data(),
                              comp_elems * sizeof(float)) &&
          ds4_gpu_tensor_read(&chunked_comp, 0, chunked_comp_host.data(),
                              comp_elems * sizeof(float)) &&
          ds4_gpu_tensor_read(&full_state_kv, 0, full_state_kv_host.data(),
                              state_elems * sizeof(float)) &&
          ds4_gpu_tensor_read(&chunked_state_kv, 0,
                              chunked_state_kv_host.data(),
                              state_elems * sizeof(float)) &&
          ds4_gpu_tensor_read(&full_state_score, 0,
                              full_state_score_host.data(),
                              state_elems * sizeof(float)) &&
          ds4_gpu_tensor_read(&chunked_state_score, 0,
                              chunked_state_score_host.data(),
                              state_elems * sizeof(float)),
          "read compressor exactness tensors");
    CHECK(memcmp(full_comp_host.data(), chunked_comp_host.data(),
                 comp_elems * sizeof(float)) == 0,
          "chunked compressor cache must match full prefill bitwise");
    CHECK(memcmp(full_state_kv_host.data(), chunked_state_kv_host.data(),
                 state_elems * sizeof(float)) == 0,
          "chunked compressor KV state must match full prefill bitwise");
    CHECK(memcmp(full_state_score_host.data(),
                 chunked_state_score_host.data(),
                 state_elems * sizeof(float)) == 0,
          "chunked compressor score state must match full prefill bitwise");

    ds4_gpu_tensor_free_in_place(&kv_dev);
    ds4_gpu_tensor_free_in_place(&score_dev);
    ds4_gpu_tensor_free_in_place(&full_comp);
    ds4_gpu_tensor_free_in_place(&chunked_comp);
    ds4_gpu_tensor_free_in_place(&full_state_kv);
    ds4_gpu_tensor_free_in_place(&full_state_score);
    ds4_gpu_tensor_free_in_place(&chunked_state_kv);
    ds4_gpu_tensor_free_in_place(&chunked_state_score);
    fprintf(stderr,
            "test_rocm_attention_prefill_static_flash: compressor head=%u "
            "fp8=%d qat=%d 2048/512 exact\n",
            head_dim, quantize_fp8 ? 1 : 0, indexer_qat ? 1 : 0);
    return 0;
}

int main(void) {
    /* Match the production 2,048-row / ratio-4 static-mixed geometry.  Sixteen
     * heads are sufficient to cover both 8-head workgroup slots while keeping
     * this focused regression's allocation bounded. */
    constexpr uint32_t n_tokens = 2048, n_comp = 512, window = 256;
    constexpr uint32_t ratio = 4, n_head = 16, head_dim = 512;
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");

    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm backend");

    constexpr uint32_t out_rank = 64;
    constexpr uint32_t out_dim = 4096;
    constexpr uint32_t low_dim = n_head * out_rank;
    constexpr uint32_t hc_dim = 16384;
    constexpr uint32_t hc_mix_dim = 24;
    constexpr uint64_t a_row_bytes = (uint64_t)(head_dim / 32u) * 34u;
    constexpr uint64_t b_row_bytes = (uint64_t)(low_dim / 32u) * 34u;
    const uint64_t sinks_offset = 0;
    const uint64_t out_a_offset = align64((uint64_t)n_head * sizeof(float));
    const uint64_t out_a_bytes = (uint64_t)n_head * out_rank * a_row_bytes;
    const uint64_t out_b_offset = align64(out_a_offset + out_a_bytes);
    const uint64_t out_b_bytes = (uint64_t)out_dim * b_row_bytes;
    const uint64_t hc_weight_offset = align64(out_b_offset + out_b_bytes);
    const uint64_t hc_weight_bytes =
        (uint64_t)hc_dim * hc_mix_dim * sizeof(__half);
    const uint64_t attn_ape_offset =
        align64(hc_weight_offset + hc_weight_bytes);
    const uint64_t attn_ape_bytes =
        (uint64_t)(2u * head_dim) * ratio * sizeof(__half);
    const uint64_t attn_norm_offset = align64(attn_ape_offset + attn_ape_bytes);
    const uint64_t attn_norm_bytes = (uint64_t)head_dim * sizeof(float);
    constexpr uint32_t index_head_dim = 128;
    const uint64_t index_ape_offset =
        align64(attn_norm_offset + attn_norm_bytes);
    const uint64_t index_ape_bytes =
        (uint64_t)(2u * index_head_dim) * ratio * sizeof(__half);
    const uint64_t index_norm_offset = align64(index_ape_offset + index_ape_bytes);
    const uint64_t index_norm_bytes =
        (uint64_t)index_head_dim * sizeof(float);
    std::vector<unsigned char> model(
        (size_t)(index_norm_offset + index_norm_bytes));
    std::vector<float> sinks(n_head);
    std::vector<float> q((size_t)n_tokens * n_head * head_dim);
    std::vector<float> raw((size_t)n_tokens * head_dim);
    std::vector<float> comp((size_t)n_comp * head_dim);
    for (uint32_t h = 0; h < n_head; h++) sinks[h] = -0.4f + 0.03f * h;
    for (size_t i = 0; i < q.size(); i++)
        q[i] = 0.09f * sinf((float)(i * 13u + 7u) * 0.0017f);
    for (size_t i = 0; i < raw.size(); i++)
        raw[i] = 0.23f * cosf((float)(i * 11u + 5u) * 0.0023f);
    for (size_t i = 0; i < comp.size(); i++)
        comp[i] = 0.19f * sinf((float)(i * 17u + 3u) * 0.0031f);
    memcpy(model.data() + sinks_offset, sinks.data(),
           sinks.size() * sizeof(float));
    pack_q8_rows(model.data() + out_a_offset, head_dim,
                 n_head * out_rank, 17u);
    pack_q8_rows(model.data() + out_b_offset, low_dim, out_dim, 31u);
    __half *hc_weight = reinterpret_cast<__half *>(
        model.data() + hc_weight_offset);
    for (uint64_t i = 0; i < (uint64_t)hc_dim * hc_mix_dim; i++) {
        hc_weight[i] = __float2half(
            0.002f * (float)((int)((i * 29u + 11u) % 127u) - 63));
    }
    __half *attn_ape = reinterpret_cast<__half *>(
        model.data() + attn_ape_offset);
    for (uint64_t i = 0; i < attn_ape_bytes / sizeof(__half); i++) {
        attn_ape[i] = __float2half(
            0.01f * (float)((int)((i * 13u + 7u) % 31u) - 15));
    }
    float *attn_norm = reinterpret_cast<float *>(
        model.data() + attn_norm_offset);
    for (uint32_t i = 0; i < head_dim; i++) {
        attn_norm[i] = 0.8f + 0.001f * (float)(i % 97u);
    }
    __half *index_ape = reinterpret_cast<__half *>(
        model.data() + index_ape_offset);
    for (uint64_t i = 0; i < index_ape_bytes / sizeof(__half); i++) {
        index_ape[i] = __float2half(
            0.012f * (float)((int)((i * 19u + 5u) % 29u) - 14));
    }
    float *index_norm = reinterpret_cast<float *>(
        model.data() + index_norm_offset);
    for (uint32_t i = 0; i < index_head_dim; i++) {
        index_norm[i] = 0.9f + 0.0015f * (float)(i % 61u);
    }

    CHECK(ds4_gpu_set_model_map(model.data(), model.size()),
          "install attention sinks");
    ds4_gpu_tensor q_dev = {}, raw_dev = {}, comp_dev = {}, heads_dev = {};
    ds4_gpu_tensor chunked_heads_dev = {};
    CHECK(ds4_gpu_tensor_alloc_on(&q_dev, 0, q.size() * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&raw_dev, 0, raw.size() * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&comp_dev, 0, comp.size() * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&heads_dev, 0, q.size() * sizeof(float)) == 0,
          "allocate tensors");
    CHECK(ds4_gpu_tensor_alloc_on(&chunked_heads_dev, 0,
                                  q.size() * sizeof(float)) == 0,
          "allocate chunked attention output");
    CHECK(ds4_gpu_tensor_write(&q_dev, 0, q.data(), q.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(&raw_dev, 0, raw.data(), raw.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(&comp_dev, 0, comp.data(), comp.size() * sizeof(float)),
          "upload tensors");

    std::vector<float> legacy(q.size()), flash1(q.size()), flash2(q.size());
    std::vector<float> chunked(q.size());
    setenv("DS4_ROCM_ATTENTION_PREFILL_STATIC_FLASH", "0", 1);
    CHECK(ds4_gpu_attention_prefill_static_mixed_heads_tensor(
              &heads_dev, model.data(), model.size(), sinks_offset,
              &q_dev, &raw_dev, &comp_dev, 0, n_tokens, n_comp, window,
              ratio, n_head, head_dim) &&
          ds4_gpu_tensor_read(&heads_dev, 0, legacy.data(), legacy.size() * sizeof(float)),
          "run legacy two-pass attention");

    setenv("DS4_ROCM_ATTENTION_PREFILL_STATIC_FLASH", "1", 1);
    CHECK(ds4_gpu_attention_prefill_static_mixed_heads_tensor(
              &heads_dev, model.data(), model.size(), sinks_offset,
              &q_dev, &raw_dev, &comp_dev, 0, n_tokens, n_comp, window,
              ratio, n_head, head_dim) &&
          ds4_gpu_tensor_read(&heads_dev, 0, flash1.data(), flash1.size() * sizeof(float)),
          "run flash attention first time");
    CHECK(ds4_gpu_attention_prefill_static_mixed_heads_tensor(
              &heads_dev, model.data(), model.size(), sinks_offset,
              &q_dev, &raw_dev, &comp_dev, 0, n_tokens, n_comp, window,
              ratio, n_head, head_dim) &&
          ds4_gpu_tensor_read(&heads_dev, 0, flash2.data(), flash2.size() * sizeof(float)),
          "run flash attention second time");

    /* A cross-layer token wavefront can only preserve the established
     * fingerprint if query-row chunks execute the exact same arithmetic as
     * the square launch.  Keep the full KV/comp tensors and absolute query
     * positions unchanged; only expose four disjoint q/output row views to
     * the production rectangular entry point. */
    constexpr uint32_t n_chunks = 4;
    static_assert(n_tokens % n_chunks == 0,
                  "wavefront regression requires equal row chunks");
    constexpr uint32_t chunk_rows = n_tokens / n_chunks;
    const uint64_t row_bytes =
        (uint64_t)n_head * head_dim * sizeof(float);
    for (uint32_t chunk = 0; chunk < n_chunks; chunk++) {
        const uint32_t row0 = chunk * chunk_rows;
        ds4_gpu_tensor *q_view = ds4_gpu_tensor_view(
            &q_dev, (uint64_t)row0 * row_bytes,
            (uint64_t)chunk_rows * row_bytes);
        ds4_gpu_tensor *out_view = ds4_gpu_tensor_view(
            &chunked_heads_dev, (uint64_t)row0 * row_bytes,
            (uint64_t)chunk_rows * row_bytes);
        CHECK(q_view && out_view, "create chunked attention row views");
        const int chunk_ok =
            ds4_gpu_attention_prefill_static_mixed_heads_range_tensor(
                out_view, model.data(), model.size(), sinks_offset,
                q_view, &raw_dev, &comp_dev, 0, row0, chunk_rows,
                n_tokens, n_comp, window, ratio, n_head, head_dim);
        ds4_gpu_tensor_free(out_view);
        ds4_gpu_tensor_free(q_view);
        CHECK(chunk_ok, "run chunked flash attention range");
    }
    CHECK(ds4_gpu_tensor_read(&chunked_heads_dev, 0, chunked.data(),
                              chunked.size() * sizeof(float)),
          "read chunked flash attention");

    CHECK(memcmp(flash1.data(), flash2.data(), flash1.size() * sizeof(float)) == 0,
          "flash attention must be bit-deterministic");
    CHECK(memcmp(flash1.data(), chunked.data(),
                 flash1.size() * sizeof(float)) == 0,
          "four query-row ranges must match the square flash launch bitwise");

    ds4_gpu_tensor full_low = {}, chunked_low = {};
    ds4_gpu_tensor full_out = {}, chunked_out = {};
    CHECK(ds4_gpu_tensor_alloc_on(&full_low, 0,
                                  (uint64_t)n_tokens * low_dim * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&chunked_low, 0,
                                  (uint64_t)n_tokens * low_dim * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&full_out, 0,
                                  (uint64_t)n_tokens * out_dim * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&chunked_out, 0,
                                  (uint64_t)n_tokens * out_dim * sizeof(float)) == 0,
          "allocate attention output projection tensors");
    CHECK(ds4_gpu_attention_output_q8_batch_tensor(
              &full_out, &full_low, NULL, NULL, model.data(), model.size(),
              out_a_offset, out_b_offset, head_dim, out_rank, n_head, out_dim,
              &heads_dev, n_tokens),
          "run full attention Q8 A/B projection");
    for (uint32_t chunk = 0; chunk < n_chunks; chunk++) {
        const uint32_t row0 = chunk * chunk_rows;
        ds4_gpu_tensor *heads_view = ds4_gpu_tensor_view(
            &chunked_heads_dev, (uint64_t)row0 * row_bytes,
            (uint64_t)chunk_rows * row_bytes);
        ds4_gpu_tensor *low_view = ds4_gpu_tensor_view(
            &chunked_low, (uint64_t)row0 * low_dim * sizeof(float),
            (uint64_t)chunk_rows * low_dim * sizeof(float));
        ds4_gpu_tensor *out_view = ds4_gpu_tensor_view(
            &chunked_out, (uint64_t)row0 * out_dim * sizeof(float),
            (uint64_t)chunk_rows * out_dim * sizeof(float));
        CHECK(heads_view && low_view && out_view,
              "create chunked attention-output views");
        const int chunk_ok = ds4_gpu_attention_output_q8_batch_tensor(
            out_view, low_view, NULL, NULL, model.data(), model.size(),
            out_a_offset, out_b_offset, head_dim, out_rank, n_head, out_dim,
            heads_view, chunk_rows);
        ds4_gpu_tensor_free(out_view);
        ds4_gpu_tensor_free(low_view);
        ds4_gpu_tensor_free(heads_view);
        CHECK(chunk_ok, "run chunked attention Q8 A/B projection");
    }
    std::vector<float> full_low_host((size_t)n_tokens * low_dim);
    std::vector<float> chunked_low_host(full_low_host.size());
    std::vector<float> full_out_host((size_t)n_tokens * out_dim);
    std::vector<float> chunked_out_host(full_out_host.size());
    CHECK(ds4_gpu_tensor_read(&full_low, 0, full_low_host.data(),
                              full_low_host.size() * sizeof(float)) &&
          ds4_gpu_tensor_read(&chunked_low, 0, chunked_low_host.data(),
                              chunked_low_host.size() * sizeof(float)) &&
          ds4_gpu_tensor_read(&full_out, 0, full_out_host.data(),
                              full_out_host.size() * sizeof(float)) &&
          ds4_gpu_tensor_read(&chunked_out, 0, chunked_out_host.data(),
                              chunked_out_host.size() * sizeof(float)),
          "read attention output projection tensors");
    CHECK(memcmp(full_low_host.data(), chunked_low_host.data(),
                 full_low_host.size() * sizeof(float)) == 0,
          "chunked grouped-Q8 A output must match full launch bitwise");
    CHECK(memcmp(full_out_host.data(), chunked_out_host.data(),
                 full_out_host.size() * sizeof(float)) == 0,
          "chunked Q8 A/B output must match full launch bitwise");

    /* ROCm implements the HC projection as an exact rowwise RMS kernel
     * followed by hipBLAS F16xF16->F32 GEMM.  A wavefront changes GEMM M from
     * 2048 to 512 and can therefore select a different hipBLAS solution.  Use
     * the production HC dimensions to prove that the selected solutions keep
     * the established arithmetic bit-for-bit before a scheduler may split it. */
    const uint64_t hc_elems = (uint64_t)n_tokens * hc_dim;
    const uint64_t hc_out_elems = (uint64_t)n_tokens * hc_mix_dim;
    std::vector<float> hc_input((size_t)hc_elems);
    for (uint64_t i = 0; i < hc_elems; i++) {
        hc_input[(size_t)i] =
            0.17f * sinf((float)(i * 19u + 23u) * 0.00031f) +
            0.03f * cosf((float)(i * 7u + 5u) * 0.00017f);
    }
    ds4_gpu_tensor hc_input_dev = {}, hc_full_norm = {}, hc_chunked_norm = {};
    ds4_gpu_tensor hc_full_out = {}, hc_chunked_out = {};
    CHECK(ds4_gpu_tensor_alloc_on(&hc_input_dev, 0,
                                  hc_elems * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&hc_full_norm, 0,
                                  hc_elems * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&hc_chunked_norm, 0,
                                  hc_elems * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&hc_full_out, 0,
                                  hc_out_elems * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&hc_chunked_out, 0,
                                  hc_out_elems * sizeof(float)) == 0,
          "allocate HC projection exactness tensors");
    CHECK(ds4_gpu_tensor_write(&hc_input_dev, 0, hc_input.data(),
                               hc_elems * sizeof(float)),
          "upload HC projection input");
    CHECK(ds4_gpu_rms_norm_plain_rows_tensor(
              &hc_full_norm, &hc_input_dev, hc_dim, n_tokens, 1.0e-6f) &&
          ds4_gpu_matmul_f16_tensor(
              &hc_full_out, model.data(), model.size(), hc_weight_offset,
              hc_dim, hc_mix_dim, &hc_full_norm, n_tokens),
          "run full HC RMS projection");
    for (uint32_t chunk = 0; chunk < n_chunks; chunk++) {
        const uint32_t row0 = chunk * chunk_rows;
        ds4_gpu_tensor *input_view = ds4_gpu_tensor_view(
            &hc_input_dev, (uint64_t)row0 * hc_dim * sizeof(float),
            (uint64_t)chunk_rows * hc_dim * sizeof(float));
        ds4_gpu_tensor *norm_view = ds4_gpu_tensor_view(
            &hc_chunked_norm, (uint64_t)row0 * hc_dim * sizeof(float),
            (uint64_t)chunk_rows * hc_dim * sizeof(float));
        ds4_gpu_tensor *out_view = ds4_gpu_tensor_view(
            &hc_chunked_out, (uint64_t)row0 * hc_mix_dim * sizeof(float),
            (uint64_t)chunk_rows * hc_mix_dim * sizeof(float));
        CHECK(input_view && norm_view && out_view,
              "create chunked HC projection views");
        const int chunk_ok =
            ds4_gpu_rms_norm_plain_rows_tensor(
                norm_view, input_view, hc_dim, chunk_rows, 1.0e-6f) &&
            ds4_gpu_matmul_f16_tensor(
                out_view, model.data(), model.size(), hc_weight_offset,
                hc_dim, hc_mix_dim, norm_view, chunk_rows);
        ds4_gpu_tensor_free(out_view);
        ds4_gpu_tensor_free(norm_view);
        ds4_gpu_tensor_free(input_view);
        CHECK(chunk_ok, "run chunked HC RMS projection");
    }
    std::vector<float> hc_full_norm_host((size_t)hc_elems);
    std::vector<float> hc_chunked_norm_host((size_t)hc_elems);
    std::vector<float> hc_full_out_host((size_t)hc_out_elems);
    std::vector<float> hc_chunked_out_host((size_t)hc_out_elems);
    CHECK(ds4_gpu_tensor_read(&hc_full_norm, 0, hc_full_norm_host.data(),
                              hc_elems * sizeof(float)) &&
          ds4_gpu_tensor_read(&hc_chunked_norm, 0,
                              hc_chunked_norm_host.data(),
                              hc_elems * sizeof(float)) &&
          ds4_gpu_tensor_read(&hc_full_out, 0, hc_full_out_host.data(),
                              hc_out_elems * sizeof(float)) &&
          ds4_gpu_tensor_read(&hc_chunked_out, 0,
                              hc_chunked_out_host.data(),
                              hc_out_elems * sizeof(float)),
          "read HC projection exactness tensors");
    CHECK(memcmp(hc_full_norm_host.data(), hc_chunked_norm_host.data(),
                 hc_elems * sizeof(float)) == 0,
          "chunked HC RMS output must match full launch bitwise");
    CHECK(memcmp(hc_full_out_host.data(), hc_chunked_out_host.data(),
                 hc_out_elems * sizeof(float)) == 0,
          "chunked hipBLAS HC projection must match full launch bitwise");
    fprintf(stderr,
            "test_rocm_attention_prefill_static_flash: HC 2048/512 exact\n");
    CHECK(test_compressor_wavefront(model, attn_ape_offset, attn_norm_offset,
                                    head_dim, true, false) == 0,
          "attention compressor wavefront exactness");
    CHECK(test_compressor_wavefront(model, index_ape_offset,
                                    index_norm_offset, index_head_dim,
                                    false, true) == 0,
          "indexer compressor wavefront exactness");
    for (float v : flash1) CHECK(isfinite(v), "flash output must be finite");
    const double error = rel_rms(flash1, legacy);
    fprintf(stderr, "test_rocm_attention_prefill_static_flash: rel_rms=%g\n", error);
    CHECK(error <= 2.0e-5, "flash attention must remain close to two-pass oracle");

    ds4_gpu_tensor_free_in_place(&q_dev);
    ds4_gpu_tensor_free_in_place(&raw_dev);
    ds4_gpu_tensor_free_in_place(&comp_dev);
    ds4_gpu_tensor_free_in_place(&heads_dev);
    ds4_gpu_tensor_free_in_place(&chunked_heads_dev);
    ds4_gpu_tensor_free_in_place(&full_low);
    ds4_gpu_tensor_free_in_place(&chunked_low);
    ds4_gpu_tensor_free_in_place(&full_out);
    ds4_gpu_tensor_free_in_place(&chunked_out);
    ds4_gpu_tensor_free_in_place(&hc_input_dev);
    ds4_gpu_tensor_free_in_place(&hc_full_norm);
    ds4_gpu_tensor_free_in_place(&hc_chunked_norm);
    ds4_gpu_tensor_free_in_place(&hc_full_out);
    ds4_gpu_tensor_free_in_place(&hc_chunked_out);
    ds4_gpu_cleanup();
    fprintf(stderr, "test_rocm_attention_prefill_static_flash: PASS\n");
    return 0;
}
