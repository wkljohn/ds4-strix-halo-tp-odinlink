/* Model-free compressor/indexer output-row oracle (Codex gpt-5.6-sol gate 4).
 *
 * Drives shipped F16 pair projection, compressor update, indexer scores, and
 * top-k. Full width vs two concatenated output-row halves. No GGUF, no
 * dispatcher, no extra TP gate.
 */
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg) do {                                                \
    if (!(cond)) {                                                           \
        fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);           \
        return 1;                                                            \
    }                                                                        \
} while (0)

typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

enum {
    DS4_TENSOR_F32 = 0,
    DS4_TENSOR_F16 = 1,
    IN_DIM = 4096,
    OUT_DIM = 1024,
    HALF = 512,
    RATIO = 4,
    HEAD_DIM = 128,
    N_ROT = 64,
    N_HEAD = 8,
    TOP_K = 4,
    N_COMP = 8,
};

static int alloc_tensor(ds4_gpu_tensor *t, uint64_t bytes) {
    memset(t, 0, sizeof(*t));
    return ds4_gpu_tensor_alloc_on(t, 0, bytes) == 0;
}

static int upload(ds4_gpu_tensor *t, const void *src, uint64_t bytes) {
    return ds4_gpu_tensor_write(t, 0, src, bytes) != 0;
}

static void fill_f16(uint16_t *dst, uint64_t n, uint32_t salt) {
    for (uint64_t i = 0; i < n; i++) {
        const float v = (float)((int)((salt + 13u * (uint32_t)i) % 21u) - 10) * 0.015625f;
        dst[i] = __half_as_ushort(__float2half(v));
    }
}

int main(void) {
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm");

    const uint64_t w_elems = (uint64_t)IN_DIM * OUT_DIM;
    const uint64_t half_elems = (uint64_t)IN_DIM * HALF;
    const uint64_t ape_elems = (uint64_t)OUT_DIM * RATIO;
    const uint64_t model_bytes =
        2u * w_elems * 2u + ape_elems * 2u + HEAD_DIM * sizeof(float);
    unsigned char *model = (unsigned char *)malloc((size_t)model_bytes);
    CHECK(model, "model");
    uint64_t off_kv = 0;
    uint64_t off_sc = w_elems * 2u;
    uint64_t off_ape = off_sc + w_elems * 2u;
    uint64_t off_norm = off_ape + ape_elems * 2u;
    fill_f16((uint16_t *)(model + off_kv), w_elems, 3);
    fill_f16((uint16_t *)(model + off_sc), w_elems, 11);
    fill_f16((uint16_t *)(model + off_ape), ape_elems, 19);
    float *hnorm = (float *)(model + off_norm);
    for (uint32_t i = 0; i < HEAD_DIM; i++) hnorm[i] = 1.0f;
    CHECK(ds4_gpu_set_model_map(model, model_bytes), "map");

    float hx[IN_DIM];
    for (uint32_t i = 0; i < IN_DIM; i++)
        hx[i] = (float)((int)(i % 29u) - 14) * 0.03125f;

    ds4_gpu_tensor x = {}, kv = {}, sc = {}, kv0 = {}, sc0 = {}, kv1 = {}, sc1 = {};
    CHECK(alloc_tensor(&x, IN_DIM * sizeof(float)), "x");
    CHECK(alloc_tensor(&kv, OUT_DIM * sizeof(float)), "kv");
    CHECK(alloc_tensor(&sc, OUT_DIM * sizeof(float)), "sc");
    CHECK(alloc_tensor(&kv0, HALF * sizeof(float)), "kv0");
    CHECK(alloc_tensor(&sc0, HALF * sizeof(float)), "sc0");
    CHECK(alloc_tensor(&kv1, HALF * sizeof(float)), "kv1");
    CHECK(alloc_tensor(&sc1, HALF * sizeof(float)), "sc1");
    CHECK(upload(&x, hx, sizeof(hx)), "upload x");

    CHECK(ds4_gpu_matmul_f16_pair_tensor(&kv, &sc, model, model_bytes,
                                         off_kv, off_sc, IN_DIM, OUT_DIM,
                                         &x, 1) != 0,
          "full pair proj");
    CHECK(ds4_gpu_matmul_f16_pair_tensor(&kv0, &sc0, model, model_bytes,
                                         off_kv, off_sc, IN_DIM, HALF,
                                         &x, 1) != 0,
          "rank0 pair proj");
    const uint64_t half_off = HALF * IN_DIM * 2u;
    CHECK(ds4_gpu_matmul_f16_pair_tensor(&kv1, &sc1, model, model_bytes,
                                         off_kv + half_off, off_sc + half_off,
                                         IN_DIM, HALF, &x, 1) != 0,
          "rank1 pair proj");

    float hfull_kv[OUT_DIM], hfull_sc[OUT_DIM], h0[HALF], h1[HALF], s0[HALF], s1[HALF];
    CHECK(ds4_gpu_tensor_read(&kv, 0, hfull_kv, sizeof(hfull_kv)), "read kv");
    CHECK(ds4_gpu_tensor_read(&sc, 0, hfull_sc, sizeof(hfull_sc)), "read sc");
    CHECK(ds4_gpu_tensor_read(&kv0, 0, h0, sizeof(h0)), "read kv0");
    CHECK(ds4_gpu_tensor_read(&kv1, 0, h1, sizeof(h1)), "read kv1");
    CHECK(ds4_gpu_tensor_read(&sc0, 0, s0, sizeof(s0)), "read sc0");
    CHECK(ds4_gpu_tensor_read(&sc1, 0, s1, sizeof(s1)), "read sc1");
    CHECK(memcmp(hfull_kv, h0, sizeof(h0)) == 0, "kv rank0 rows");
    CHECK(memcmp(hfull_kv + HALF, h1, sizeof(h1)) == 0, "kv rank1 rows");
    CHECK(memcmp(hfull_sc, s0, sizeof(s0)) == 0, "score rank0 rows");
    CHECK(memcmp(hfull_sc + HALF, s1, sizeof(s1)) == 0, "score rank1 rows");

    /* Concatenate rank halves, then the shipped update/score/top-k see the
     * same full activation the unsplit path produced. Splitting the update
     * itself is not head-offset-safe (RoPE/APE). */
    ds4_gpu_tensor kv_cat = {}, sc_cat = {};
    CHECK(alloc_tensor(&kv_cat, OUT_DIM * sizeof(float)), "kv_cat");
    CHECK(alloc_tensor(&sc_cat, OUT_DIM * sizeof(float)), "sc_cat");
    CHECK(ds4_gpu_tensor_copy(&kv_cat, 0, &kv0, 0, HALF * sizeof(float)) != 0, "cat kv0");
    CHECK(ds4_gpu_tensor_copy(&kv_cat, HALF * sizeof(float), &kv1, 0,
                              HALF * sizeof(float)) != 0, "cat kv1");
    CHECK(ds4_gpu_tensor_copy(&sc_cat, 0, &sc0, 0, HALF * sizeof(float)) != 0, "cat sc0");
    CHECK(ds4_gpu_tensor_copy(&sc_cat, HALF * sizeof(float), &sc1, 0,
                              HALF * sizeof(float)) != 0, "cat sc1");
    float hcat_kv[OUT_DIM], hcat_sc[OUT_DIM];
    CHECK(ds4_gpu_tensor_read(&kv_cat, 0, hcat_kv, sizeof(hcat_kv)), "read kv_cat");
    CHECK(ds4_gpu_tensor_read(&sc_cat, 0, hcat_sc, sizeof(hcat_sc)), "read sc_cat");
    CHECK(memcmp(hfull_kv, hcat_kv, sizeof(hfull_kv)) == 0, "concat kv == full");
    CHECK(memcmp(hfull_sc, hcat_sc, sizeof(hfull_sc)) == 0, "concat score == full");

    ds4_gpu_tensor st_kv = {}, st_sc = {}, cache = {};
    ds4_gpu_tensor st_kv_c = {}, st_sc_c = {}, cache_c = {};
    CHECK(alloc_tensor(&st_kv, (uint64_t)RATIO * OUT_DIM * sizeof(float)), "st_kv");
    CHECK(alloc_tensor(&st_sc, (uint64_t)RATIO * OUT_DIM * sizeof(float)), "st_sc");
    CHECK(alloc_tensor(&cache, (uint64_t)N_COMP * OUT_DIM * sizeof(float)), "cache");
    CHECK(alloc_tensor(&st_kv_c, (uint64_t)RATIO * OUT_DIM * sizeof(float)), "st_kv_c");
    CHECK(alloc_tensor(&st_sc_c, (uint64_t)RATIO * OUT_DIM * sizeof(float)), "st_sc_c");
    CHECK(alloc_tensor(&cache_c, (uint64_t)N_COMP * OUT_DIM * sizeof(float)), "cache_c");
    CHECK(ds4_gpu_tensor_fill_f32(&st_kv, 0.0f, (uint64_t)RATIO * OUT_DIM) != 0, "zero st");
    CHECK(ds4_gpu_tensor_fill_f32(&st_sc, 0.0f, (uint64_t)RATIO * OUT_DIM) != 0, "zero stsc");
    CHECK(ds4_gpu_tensor_fill_f32(&cache, 0.0f, (uint64_t)N_COMP * OUT_DIM) != 0, "zero cache");
    CHECK(ds4_gpu_tensor_fill_f32(&st_kv_c, 0.0f, (uint64_t)RATIO * OUT_DIM) != 0, "zero stc");
    CHECK(ds4_gpu_tensor_fill_f32(&st_sc_c, 0.0f, (uint64_t)RATIO * OUT_DIM) != 0, "zero stscc");
    CHECK(ds4_gpu_tensor_fill_f32(&cache_c, 0.0f, (uint64_t)N_COMP * OUT_DIM) != 0, "zero cc");

    const uint32_t pos = RATIO - 1; /* emit */
    CHECK(ds4_gpu_compressor_update_tensor(
              &kv, &sc, &st_kv, &st_sc, &cache, model, model_bytes,
              off_ape, DS4_TENSOR_F16, off_norm, DS4_TENSOR_F32,
              HEAD_DIM, RATIO, pos, 0, N_ROT, 0,
              10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1e-6f, false) != 0,
          "full compressor update");
    CHECK(ds4_gpu_compressor_update_tensor(
              &kv_cat, &sc_cat, &st_kv_c, &st_sc_c, &cache_c, model, model_bytes,
              off_ape, DS4_TENSOR_F16, off_norm, DS4_TENSOR_F32,
              HEAD_DIM, RATIO, pos, 0, N_ROT, 0,
              10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1e-6f, false) != 0,
          "concat compressor update");

    float *full_cache = (float *)malloc((size_t)N_COMP * OUT_DIM * sizeof(float));
    float *cat_cache = (float *)malloc((size_t)N_COMP * OUT_DIM * sizeof(float));
    float *full_st = (float *)malloc((size_t)RATIO * OUT_DIM * sizeof(float));
    float *cat_st = (float *)malloc((size_t)RATIO * OUT_DIM * sizeof(float));
    CHECK(full_cache && cat_cache && full_st && cat_st, "host cache");
    CHECK(ds4_gpu_tensor_read(&cache, 0, full_cache,
                              (uint64_t)N_COMP * OUT_DIM * sizeof(float)),
          "read full cache");
    CHECK(ds4_gpu_tensor_read(&cache_c, 0, cat_cache,
                              (uint64_t)N_COMP * OUT_DIM * sizeof(float)),
          "read cat cache");
    CHECK(ds4_gpu_tensor_read(&st_kv, 0, full_st,
                              (uint64_t)RATIO * OUT_DIM * sizeof(float)),
          "read full state");
    CHECK(ds4_gpu_tensor_read(&st_kv_c, 0, cat_st,
                              (uint64_t)RATIO * OUT_DIM * sizeof(float)),
          "read cat state");
    CHECK(memcmp(full_cache, cat_cache,
                 (size_t)N_COMP * OUT_DIM * sizeof(float)) == 0,
          "state cache bitwise after concat");
    CHECK(memcmp(full_st, cat_st,
                 (size_t)RATIO * OUT_DIM * sizeof(float)) == 0,
          "recurrent state bitwise after concat");

    ds4_gpu_tensor q = {}, w = {}, scores = {}, scores_cat = {}, selected = {}, selected_cat = {};
    CHECK(alloc_tensor(&q, (uint64_t)N_HEAD * HEAD_DIM * sizeof(float)), "q");
    CHECK(alloc_tensor(&w, (uint64_t)N_HEAD * sizeof(float)), "w");
    CHECK(alloc_tensor(&scores, (uint64_t)N_COMP * sizeof(float)), "scores");
    CHECK(alloc_tensor(&scores_cat, (uint64_t)N_COMP * sizeof(float)), "scores_cat");
    CHECK(alloc_tensor(&selected, TOP_K * sizeof(int32_t)), "sel");
    CHECK(alloc_tensor(&selected_cat, TOP_K * sizeof(int32_t)), "sel_cat");
    float hq[(uint32_t)N_HEAD * HEAD_DIM], hw[N_HEAD];
    for (uint32_t i = 0; i < (uint32_t)N_HEAD * HEAD_DIM; i++)
        hq[i] = (float)((int)(i % 11u) - 5) * 0.125f;
    for (uint32_t i = 0; i < N_HEAD; i++) hw[i] = 1.0f;
    CHECK(upload(&q, hq, sizeof(hq)), "upload q");
    CHECK(upload(&w, hw, sizeof(hw)), "upload w");
    CHECK(ds4_gpu_indexer_score_one_tensor(&scores, &q, &w, &cache,
                                           N_COMP, N_HEAD, HEAD_DIM, 1.0f) != 0,
          "full scores");
    CHECK(ds4_gpu_indexer_score_one_tensor(&scores_cat, &q, &w, &cache_c,
                                           N_COMP, N_HEAD, HEAD_DIM, 1.0f) != 0,
          "concat scores");
    float hs[N_COMP], hs2[N_COMP];
    CHECK(ds4_gpu_tensor_read(&scores, 0, hs, sizeof(hs)), "read scores");
    CHECK(ds4_gpu_tensor_read(&scores_cat, 0, hs2, sizeof(hs2)), "read scores2");
    CHECK(memcmp(hs, hs2, sizeof(hs)) == 0, "scores bitwise on concatenated cache");
    CHECK(ds4_gpu_indexer_topk_tensor(&selected, &scores, N_COMP, 1, TOP_K) != 0,
          "topk full");
    CHECK(ds4_gpu_indexer_topk_tensor(&selected_cat, &scores_cat, N_COMP, 1, TOP_K) != 0,
          "topk concat");
    int32_t top[TOP_K], top2[TOP_K];
    CHECK(ds4_gpu_tensor_read(&selected, 0, top, sizeof(top)), "read top");
    CHECK(ds4_gpu_tensor_read(&selected_cat, 0, top2, sizeof(top2)), "read top2");
    CHECK(memcmp(top, top2, sizeof(top)) == 0, "topk bitwise");

    printf("test_rocm_compressor_row_shard_oracle: output_row_halves bitwise "
           "proj+state_cache+scores+topk ok out_dim=%u half=%u\n",
           OUT_DIM, HALF);

    ds4_gpu_tensor_free_in_place(&selected_cat);
    ds4_gpu_tensor_free_in_place(&selected);
    ds4_gpu_tensor_free_in_place(&scores_cat);
    ds4_gpu_tensor_free_in_place(&scores);
    ds4_gpu_tensor_free_in_place(&w);
    ds4_gpu_tensor_free_in_place(&q);
    ds4_gpu_tensor_free_in_place(&cache_c);
    ds4_gpu_tensor_free_in_place(&st_sc_c);
    ds4_gpu_tensor_free_in_place(&st_kv_c);
    ds4_gpu_tensor_free_in_place(&cache);
    ds4_gpu_tensor_free_in_place(&st_sc);
    ds4_gpu_tensor_free_in_place(&st_kv);
    ds4_gpu_tensor_free_in_place(&sc_cat);
    ds4_gpu_tensor_free_in_place(&kv_cat);
    ds4_gpu_tensor_free_in_place(&sc1);
    ds4_gpu_tensor_free_in_place(&kv1);
    ds4_gpu_tensor_free_in_place(&sc0);
    ds4_gpu_tensor_free_in_place(&kv0);
    ds4_gpu_tensor_free_in_place(&sc);
    ds4_gpu_tensor_free_in_place(&kv);
    ds4_gpu_tensor_free_in_place(&x);
    free(full_cache); free(cat_cache); free(full_st); free(cat_st); free(model);
    ds4_gpu_cleanup();
    return 0;
}
