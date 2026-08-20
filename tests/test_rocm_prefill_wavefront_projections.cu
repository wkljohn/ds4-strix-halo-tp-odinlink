/* Production-shape full-vs-wave exactness oracle for the row-local
 * projections that precede DeepSeek V4 Flash attention on ROCm. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(cond, msg) do {                                                \
    if (!(cond)) {                                                           \
        std::fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);      \
        return 1;                                                            \
    }                                                                        \
} while (0)

typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

static uint64_t align64(uint64_t value) {
    return (value + 63u) & ~UINT64_C(63);
}
static uint64_t q8_bytes(uint32_t in_dim, uint32_t out_dim) {
    return (uint64_t)out_dim * (in_dim / 32u) * 34u;
}

static void pack_q8(unsigned char *dst, uint32_t in_dim,
                    uint32_t out_dim, uint32_t salt) {
    const uint32_t blocks = in_dim / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    for (uint32_t row = 0; row < out_dim; row++) {
        for (uint32_t block = 0; block < blocks; block++) {
            unsigned char *p = dst + (uint64_t)row * row_bytes +
                               (uint64_t)block * 34u;
            const __half scale = __float2half(
                0.002f + 0.0001f * (float)((row + block + salt) % 19u));
            std::memcpy(p, &scale, sizeof(scale));
            for (uint32_t lane = 0; lane < 32u; lane++) {
                p[2u + lane] = (unsigned char)(int8_t)(
                    (int)((row * 17u + block * 11u + lane * 7u + salt) %
                          255u) - 127);
            }
        }
    }
}

static void pack_f16(unsigned char *dst, uint64_t elements, uint32_t salt) {
    __half *w = reinterpret_cast<__half *>(dst);
    for (uint64_t i = 0; i < elements; i++) {
        w[i] = __float2half(
            0.0025f * (float)((int)((i * 23u + salt) % 127u) - 63));
    }
}

static int tensors_bit_equal(const ds4_gpu_tensor *a,
                             const ds4_gpu_tensor *b,
                             uint64_t elements,
                             const char *label) {
    std::vector<float> ah((size_t)elements), bh((size_t)elements);
    if (!ds4_gpu_tensor_read(a, 0, ah.data(), elements * sizeof(float)) ||
        !ds4_gpu_tensor_read(b, 0, bh.data(), elements * sizeof(float))) {
        std::fprintf(stderr, "FAIL: read %s\n", label);
        return 0;
    }
    uint64_t mismatches = 0;
    for (uint64_t i = 0; i < elements; i++) {
        if (std::memcmp(&ah[(size_t)i], &bh[(size_t)i], sizeof(float)) != 0)
            mismatches++;
    }
    std::fprintf(stderr, "%s mismatches=%llu/%llu\n", label,
                 (unsigned long long)mismatches,
                 (unsigned long long)elements);
    return mismatches == 0u;
}

int main(void) {
    constexpr uint32_t tokens = 2048u;
    constexpr uint32_t waves = 4u;
    constexpr uint32_t wave_rows = tokens / waves;
    constexpr uint32_t embd = 4096u;
    constexpr uint32_t q_rank = 1024u;
    constexpr uint32_t kv_dim = 512u;
    constexpr uint32_t q_dim = 32768u;
    constexpr uint32_t rank_rows = tokens / 2u;
    constexpr uint32_t rank_wave_rows = rank_rows / waves;
    constexpr uint32_t comp_width = 1024u;
    constexpr uint32_t index_width = 256u;
    constexpr uint32_t index_q_dim = 8192u;

    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm backend");

    uint64_t cursor = 0u;
    const uint64_t q_a_offset = cursor;
    cursor = align64(cursor + q8_bytes(embd, q_rank));
    const uint64_t kv_offset = cursor;
    cursor = align64(cursor + q8_bytes(embd, kv_dim));
    const uint64_t q_b_offset = cursor;
    cursor = align64(cursor + q8_bytes(q_rank, q_dim));
    const uint64_t comp_kv_offset = cursor;
    cursor = align64(cursor + (uint64_t)embd * comp_width * sizeof(__half));
    const uint64_t comp_sc_offset = cursor;
    cursor = align64(cursor + (uint64_t)embd * comp_width * sizeof(__half));
    const uint64_t index_kv_offset = cursor;
    cursor = align64(cursor + (uint64_t)embd * index_width * sizeof(__half));
    const uint64_t index_sc_offset = cursor;
    cursor = align64(cursor + (uint64_t)embd * index_width * sizeof(__half));
    const uint64_t index_q_offset = cursor;
    cursor = align64(cursor + (uint64_t)q_rank * index_q_dim * sizeof(__half));
    std::vector<unsigned char> model((size_t)cursor);
    pack_q8(model.data() + q_a_offset, embd, q_rank, 11u);
    pack_q8(model.data() + kv_offset, embd, kv_dim, 23u);
    pack_q8(model.data() + q_b_offset, q_rank, q_dim, 37u);
    pack_f16(model.data() + comp_kv_offset,
             (uint64_t)embd * comp_width, 41u);
    pack_f16(model.data() + comp_sc_offset,
             (uint64_t)embd * comp_width, 43u);
    pack_f16(model.data() + index_kv_offset,
             (uint64_t)embd * index_width, 47u);
    pack_f16(model.data() + index_sc_offset,
             (uint64_t)embd * index_width, 53u);
    pack_f16(model.data() + index_q_offset,
             (uint64_t)q_rank * index_q_dim, 59u);
    CHECK(ds4_gpu_set_model_map(model.data(), model.size()),
          "install synthetic projection model");

    const uint64_t x_elems = (uint64_t)tokens * embd;
    std::vector<float> x_host((size_t)x_elems);
    for (uint64_t i = 0; i < x_elems; i++) {
        x_host[(size_t)i] =
            0.11f * std::sin((float)(i * 13u + 5u) * 0.00037f) +
            0.04f * std::cos((float)(i * 7u + 3u) * 0.00019f);
    }
    ds4_gpu_tensor x = {};
    CHECK(ds4_gpu_tensor_alloc_on(&x, 0, x_elems * sizeof(float)) == 0 &&
          ds4_gpu_tensor_write(&x, 0, x_host.data(),
                               x_elems * sizeof(float)),
          "allocate/upload projection input");

    ds4_gpu_tensor full_qr = {}, wave_qr = {}, full_kv = {}, wave_kv = {};
    CHECK(ds4_gpu_tensor_alloc_on(&full_qr, 0,
                                  (uint64_t)tokens * q_rank * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&wave_qr, 0,
                                  (uint64_t)tokens * q_rank * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&full_kv, 0,
                                  (uint64_t)tokens * kv_dim * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&wave_kv, 0,
                                  (uint64_t)tokens * kv_dim * sizeof(float)) == 0,
          "allocate Q/KV pair outputs");
    CHECK(ds4_gpu_matmul_q8_0_pair_tensor(
              &full_qr, &full_kv, model.data(), model.size(),
              q_a_offset, kv_offset, embd, q_rank, kv_dim, &x, tokens),
          "run full Q/KV pair");
    for (uint32_t wave = 0; wave < waves; wave++) {
        const uint32_t row0 = wave * wave_rows;
        ds4_gpu_tensor *xv = ds4_gpu_tensor_view(
            &x, (uint64_t)row0 * embd * sizeof(float),
            (uint64_t)wave_rows * embd * sizeof(float));
        ds4_gpu_tensor *qv = ds4_gpu_tensor_view(
            &wave_qr, (uint64_t)row0 * q_rank * sizeof(float),
            (uint64_t)wave_rows * q_rank * sizeof(float));
        ds4_gpu_tensor *kvv = ds4_gpu_tensor_view(
            &wave_kv, (uint64_t)row0 * kv_dim * sizeof(float),
            (uint64_t)wave_rows * kv_dim * sizeof(float));
        CHECK(xv && qv && kvv && ds4_gpu_matmul_q8_0_pair_tensor(
                  qv, kvv, model.data(), model.size(), q_a_offset, kv_offset,
                  embd, q_rank, kv_dim, xv, wave_rows),
              "run wave Q/KV pair");
        ds4_gpu_tensor_free(kvv);
        ds4_gpu_tensor_free(qv);
        ds4_gpu_tensor_free(xv);
    }
    CHECK(tensors_bit_equal(&full_qr, &wave_qr,
                            (uint64_t)tokens * q_rank,
                            "Q8 Q-A 2048/512"),
          "Q-A wave exactness");
    CHECK(tensors_bit_equal(&full_kv, &wave_kv,
                            (uint64_t)tokens * kv_dim,
                            "Q8 KV 2048/512"),
          "KV wave exactness");

    ds4_gpu_tensor full_q = {}, wave_q = {};
    CHECK(ds4_gpu_tensor_alloc_on(&full_q, 0,
                                  (uint64_t)rank_rows * q_dim * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&wave_q, 0,
                                  (uint64_t)rank_rows * q_dim * sizeof(float)) == 0,
          "allocate Q-B rank outputs");
    ds4_gpu_tensor *rank_qr = ds4_gpu_tensor_view(
        &full_qr, 0u, (uint64_t)rank_rows * q_rank * sizeof(float));
    CHECK(rank_qr && ds4_gpu_matmul_q8_0_tensor(
              &full_q, model.data(), model.size(), q_b_offset,
              q_rank, q_dim, rank_qr, rank_rows),
          "run full rank Q-B projection");
    for (uint32_t wave = 0; wave < waves; wave++) {
        const uint32_t row0 = wave * rank_wave_rows;
        ds4_gpu_tensor *xv = ds4_gpu_tensor_view(
            rank_qr, (uint64_t)row0 * q_rank * sizeof(float),
            (uint64_t)rank_wave_rows * q_rank * sizeof(float));
        ds4_gpu_tensor *ov = ds4_gpu_tensor_view(
            &wave_q, (uint64_t)row0 * q_dim * sizeof(float),
            (uint64_t)rank_wave_rows * q_dim * sizeof(float));
        CHECK(xv && ov && ds4_gpu_matmul_q8_0_tensor(
                  ov, model.data(), model.size(), q_b_offset,
                  q_rank, q_dim, xv, rank_wave_rows),
              "run wave Q-B projection");
        ds4_gpu_tensor_free(ov);
        ds4_gpu_tensor_free(xv);
    }
    CHECK(tensors_bit_equal(&full_q, &wave_q,
                            (uint64_t)rank_rows * q_dim,
                            "Q8 Q-B 1024/256"),
          "Q-B wave exactness");

    ds4_gpu_tensor full_comp_kv = {}, full_comp_sc = {};
    ds4_gpu_tensor wave_comp_kv = {}, wave_comp_sc = {};
    CHECK(ds4_gpu_tensor_alloc_on(&full_comp_kv, 0,
                                  (uint64_t)tokens * comp_width * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&full_comp_sc, 0,
                                  (uint64_t)tokens * comp_width * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&wave_comp_kv, 0,
                                  (uint64_t)tokens * comp_width * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&wave_comp_sc, 0,
                                  (uint64_t)tokens * comp_width * sizeof(float)) == 0,
          "allocate compressor pair outputs");
    CHECK(ds4_gpu_matmul_f16_pair_tensor(
              &full_comp_kv, &full_comp_sc, model.data(), model.size(),
              comp_kv_offset, comp_sc_offset, embd, comp_width, &x, tokens),
          "run full compressor F16 pair");
    for (uint32_t wave = 0; wave < waves; wave++) {
        const uint32_t row0 = wave * wave_rows;
        ds4_gpu_tensor *xv = ds4_gpu_tensor_view(
            &x, (uint64_t)row0 * embd * sizeof(float),
            (uint64_t)wave_rows * embd * sizeof(float));
        ds4_gpu_tensor *kvv = ds4_gpu_tensor_view(
            &wave_comp_kv, (uint64_t)row0 * comp_width * sizeof(float),
            (uint64_t)wave_rows * comp_width * sizeof(float));
        ds4_gpu_tensor *scv = ds4_gpu_tensor_view(
            &wave_comp_sc, (uint64_t)row0 * comp_width * sizeof(float),
            (uint64_t)wave_rows * comp_width * sizeof(float));
        CHECK(xv && kvv && scv && ds4_gpu_matmul_f16_pair_tensor(
                  kvv, scv, model.data(), model.size(),
                  comp_kv_offset, comp_sc_offset, embd, comp_width,
                  xv, wave_rows),
              "run wave compressor F16 pair");
        ds4_gpu_tensor_free(scv);
        ds4_gpu_tensor_free(kvv);
        ds4_gpu_tensor_free(xv);
    }
    CHECK(tensors_bit_equal(&full_comp_kv, &wave_comp_kv,
                            (uint64_t)tokens * comp_width,
                            "F16 compressor KV 2048/512") &&
          tensors_bit_equal(&full_comp_sc, &wave_comp_sc,
                            (uint64_t)tokens * comp_width,
                            "F16 compressor gate 2048/512"),
          "compressor pair wave exactness");

    ds4_gpu_tensor full_index_kv = {}, full_index_sc = {};
    ds4_gpu_tensor wave_index_kv = {}, wave_index_sc = {};
    CHECK(ds4_gpu_tensor_alloc_on(&full_index_kv, 0,
                                  (uint64_t)tokens * index_width * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&full_index_sc, 0,
                                  (uint64_t)tokens * index_width * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&wave_index_kv, 0,
                                  (uint64_t)tokens * index_width * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&wave_index_sc, 0,
                                  (uint64_t)tokens * index_width * sizeof(float)) == 0,
          "allocate index compressor pair outputs");
    CHECK(ds4_gpu_matmul_f16_pair_tensor(
              &full_index_kv, &full_index_sc, model.data(), model.size(),
              index_kv_offset, index_sc_offset, embd, index_width, &x, tokens),
          "run full index compressor F16 pair");
    for (uint32_t wave = 0; wave < waves; wave++) {
        const uint32_t row0 = wave * wave_rows;
        ds4_gpu_tensor *xv = ds4_gpu_tensor_view(
            &x, (uint64_t)row0 * embd * sizeof(float),
            (uint64_t)wave_rows * embd * sizeof(float));
        ds4_gpu_tensor *kvv = ds4_gpu_tensor_view(
            &wave_index_kv, (uint64_t)row0 * index_width * sizeof(float),
            (uint64_t)wave_rows * index_width * sizeof(float));
        ds4_gpu_tensor *scv = ds4_gpu_tensor_view(
            &wave_index_sc, (uint64_t)row0 * index_width * sizeof(float),
            (uint64_t)wave_rows * index_width * sizeof(float));
        CHECK(xv && kvv && scv && ds4_gpu_matmul_f16_pair_tensor(
                  kvv, scv, model.data(), model.size(),
                  index_kv_offset, index_sc_offset, embd, index_width,
                  xv, wave_rows),
              "run wave index compressor F16 pair");
        ds4_gpu_tensor_free(scv);
        ds4_gpu_tensor_free(kvv);
        ds4_gpu_tensor_free(xv);
    }
    CHECK(tensors_bit_equal(&full_index_kv, &wave_index_kv,
                            (uint64_t)tokens * index_width,
                            "F16 index KV 2048/512") &&
          tensors_bit_equal(&full_index_sc, &wave_index_sc,
                            (uint64_t)tokens * index_width,
                            "F16 index gate 2048/512"),
          "index compressor pair wave exactness");

    ds4_gpu_tensor full_index_q = {}, wave_index_q = {};
    CHECK(ds4_gpu_tensor_alloc_on(&full_index_q, 0,
                                  (uint64_t)tokens * index_q_dim * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&wave_index_q, 0,
                                  (uint64_t)tokens * index_q_dim * sizeof(float)) == 0,
          "allocate indexer Q outputs");
    CHECK(ds4_gpu_matmul_f16_tensor(
              &full_index_q, model.data(), model.size(), index_q_offset,
              q_rank, index_q_dim, &full_qr, tokens),
          "run full indexer Q projection");
    for (uint32_t wave = 0; wave < waves; wave++) {
        const uint32_t row0 = wave * wave_rows;
        ds4_gpu_tensor *xv = ds4_gpu_tensor_view(
            &full_qr, (uint64_t)row0 * q_rank * sizeof(float),
            (uint64_t)wave_rows * q_rank * sizeof(float));
        ds4_gpu_tensor *ov = ds4_gpu_tensor_view(
            &wave_index_q, (uint64_t)row0 * index_q_dim * sizeof(float),
            (uint64_t)wave_rows * index_q_dim * sizeof(float));
        CHECK(xv && ov && ds4_gpu_matmul_f16_tensor(
                  ov, model.data(), model.size(), index_q_offset,
                  q_rank, index_q_dim, xv, wave_rows),
              "run wave indexer Q projection");
        ds4_gpu_tensor_free(ov);
        ds4_gpu_tensor_free(xv);
    }
    CHECK(tensors_bit_equal(&full_index_q, &wave_index_q,
                            (uint64_t)tokens * index_q_dim,
                            "F16 indexer Q 2048/512"),
          "indexer Q wave exactness");

    ds4_gpu_tensor_free(rank_qr);
    ds4_gpu_tensor_free_in_place(&full_index_q);
    ds4_gpu_tensor_free_in_place(&wave_index_q);
    ds4_gpu_tensor_free_in_place(&full_index_kv);
    ds4_gpu_tensor_free_in_place(&full_index_sc);
    ds4_gpu_tensor_free_in_place(&wave_index_kv);
    ds4_gpu_tensor_free_in_place(&wave_index_sc);
    ds4_gpu_tensor_free_in_place(&full_comp_kv);
    ds4_gpu_tensor_free_in_place(&full_comp_sc);
    ds4_gpu_tensor_free_in_place(&wave_comp_kv);
    ds4_gpu_tensor_free_in_place(&wave_comp_sc);
    ds4_gpu_tensor_free_in_place(&full_q);
    ds4_gpu_tensor_free_in_place(&wave_q);
    ds4_gpu_tensor_free_in_place(&full_qr);
    ds4_gpu_tensor_free_in_place(&wave_qr);
    ds4_gpu_tensor_free_in_place(&full_kv);
    ds4_gpu_tensor_free_in_place(&wave_kv);
    ds4_gpu_tensor_free_in_place(&x);
    ds4_gpu_cleanup();
    std::fprintf(stderr,
                 "test_rocm_prefill_wavefront_projections: PASS\n");
    return 0;
}
