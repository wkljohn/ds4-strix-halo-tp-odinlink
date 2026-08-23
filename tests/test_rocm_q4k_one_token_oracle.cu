/* Isolated one-token Q4_K oracle (Codex gpt-5.6-sol 2026-08-20 option A).
 *
 * Drives the shipped ds4_gpu_routed_moe_one_tensor for bitwise self-check and
 * device-resident timing. Test-local kernels compare a one-wave Q4_K×Q8_K
 * DP4A candidate (bitwise vs CPU scalar of the same association), an MMVDQ
 * F32-dequant control that is allowed to lose, and an optimistic unique-byte
 * load floor. No production dispatcher is changed.
 */
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>
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
    Q4_K_TYPE = 12,
    QK_K = 256,
    Q4_K_BLOCK_BYTES = 144,
    N_TOTAL_EXPERT = 8,
    N_USED = 6,
    IN_DIM = 4096,
    MID_DIM = 2048,
    OUT_DIM = 4096,
    COLD_COPIES = 4,
    MODEL_COPIES = COLD_COPIES + 1,
    WARMUP = 4,
    ITERS = 32,
    STREAM_ITERS = 8,
};

static constexpr uint64_t STREAM_BYTES = UINT64_C(256) * 1024u * 1024u;

struct block_q4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct block_q8_K {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};

static void pack_q4k_block(unsigned char *dst, uint32_t seed) {
    dst[0] = 0x00;
    dst[1] = 0x28;
    dst[2] = 0x00;
    dst[3] = 0x00;
    for (uint32_t i = 0; i < 4; i++) dst[4 + i] = 1;
    for (uint32_t i = 4; i < 8; i++) dst[4 + i] = 0;
    for (uint32_t i = 8; i < 12; i++) dst[4 + i] = 1;
    for (uint32_t i = 0; i < 128; i++) {
        const uint8_t lo = (uint8_t)((seed + 3u * i + 1u) & 15u);
        const uint8_t hi = (uint8_t)((seed + 5u * i + 7u) & 15u);
        dst[16 + i] = (uint8_t)(lo | (hi << 4));
    }
}

static void pack_q4k_table(unsigned char *dst, uint32_t experts,
                           uint32_t rows, uint32_t blocks_per_row,
                           uint32_t salt) {
    for (uint32_t e = 0; e < experts; e++) {
        for (uint32_t row = 0; row < rows; row++) {
            for (uint32_t b = 0; b < blocks_per_row; b++) {
                const uint64_t i = ((uint64_t)e * rows + row) *
                                   blocks_per_row + b;
                pack_q4k_block(dst + i * Q4_K_BLOCK_BYTES,
                               salt + 17u * e + 13u * row + 7u * b);
            }
        }
    }
}

static int alloc_tensor(ds4_gpu_tensor *t, uint64_t bytes) {
    memset(t, 0, sizeof(*t));
    return ds4_gpu_tensor_alloc_on(t, 0, bytes) == 0;
}

static int upload(ds4_gpu_tensor *t, const void *src, uint64_t bytes) {
    return ds4_gpu_tensor_write(t, 0, src, bytes) != 0;
}

static uint64_t fnv1a64(const void *data, size_t bytes) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < bytes; i++) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static void q4k_scale_min(uint32_t j, const uint8_t *scales,
                          uint8_t *d_out, uint8_t *m_out) {
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (uint8_t)((scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u));
        *m_out = (uint8_t)((scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u));
    }
}

static void quant_q8_K(const float *x, block_q8_K *y, uint32_t n) {
    const uint32_t nb = n / QK_K;
    for (uint32_t ib = 0; ib < nb; ib++) {
        const float *src = x + ib * QK_K;
        float amax = 0.0f;
        for (uint32_t i = 0; i < QK_K; i++) {
            const float a = fabsf(src[i]);
            if (a > amax) amax = a;
        }
        const float d = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        y[ib].d = d;
        for (uint32_t j = 0; j < 16; j++) {
            int sum = 0;
            for (uint32_t i = 0; i < 16; i++) {
                int q = (int)lrintf(src[j * 16 + i] * id);
                if (q < -127) q = -127;
                if (q > 127) q = 127;
                y[ib].qs[j * 16 + i] = (int8_t)q;
                sum += q;
            }
            y[ib].bsums[j] = (int16_t)sum;
        }
    }
}

static float cpu_dot_q4k_q8k(const block_q4_K *x, const block_q8_K *y) {
    const float xd = __half2float(*reinterpret_cast<const __half *>(&x->d));
    const float xmin = __half2float(*reinterpret_cast<const __half *>(&x->dmin));
    int isum = 0;
    int summs = 0;
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        q4k_scale_min(j, x->scales, &sc, &m);
        summs += (int)m * (int)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        int part = 0;
        for (uint32_t i = 0; i < 32u; i++) {
            const int v = (x->qs[byte_off + i] >> shift) & 0x0f;
            part += v * (int)y->qs[j * 32u + i];
        }
        isum += (int)sc * part;
    }
    return y->d * xd * (float)isum - y->d * xmin * (float)summs;
}

static float cpu_dot_q4k_f32(const block_q4_K *x, const float *y) {
    const float xd = __half2float(*reinterpret_cast<const __half *>(&x->d));
    const float xmin = __half2float(*reinterpret_cast<const __half *>(&x->dmin));
    float acc = 0.0f;
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        q4k_scale_min(j, x->scales, &sc, &m);
        const float d = xd * (float)sc;
        const float dm = xmin * (float)m;
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        for (uint32_t i = 0; i < 32u; i++) {
            const int v = (x->qs[byte_off + i] >> shift) & 0x0f;
            acc += (d * (float)v - dm) * y[j * 32u + i];
        }
    }
    return acc;
}

__device__ static void dev_q4k_scale_min(uint32_t j, const uint8_t *scales,
                                         uint8_t *d_out, uint8_t *m_out) {
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (uint8_t)((scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u));
        *m_out = (uint8_t)((scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u));
    }
}

__device__ static int32_t dev_dot_q4_32(const uint8_t *qs, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i++) {
        const int v = (qs[i] >> shift) & 0x0f;
        sum += v * (int)q8[i];
    }
    return sum;
}

/* One-wave packed Q4_K×Q8_K down: 8 lanes cover 8 superblock groups. */
__global__ void k_q4k_q8k_down_wave32(const block_q4_K *weights,
                                      const block_q8_K *xq,
                                      float *out,
                                      uint32_t n_rows,
                                      uint32_t n_blocks) {
    const uint32_t row = blockIdx.x;
    if (row >= n_rows) return;
    const uint32_t lane = threadIdx.x;
    float acc = 0.0f;
    const block_q4_K *wr = weights + (uint64_t)row * n_blocks;
    for (uint32_t b = lane; b < n_blocks; b += 32u) {
        const block_q4_K x = wr[b];
        const block_q8_K y = xq[b];
        const float xd = __half2float(*reinterpret_cast<const __half *>(&x.d));
        const float xmin = __half2float(*reinterpret_cast<const __half *>(&x.dmin));
        int isum = 0;
        int summs = 0;
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            uint8_t sc, m;
            dev_q4k_scale_min(j, x.scales, &sc, &m);
            summs += (int)m * (int)(y.bsums[2u * j] + y.bsums[2u * j + 1u]);
            const uint32_t byte_off = (j >> 1u) * 32u;
            const int shift = (j & 1u) ? 4 : 0;
            isum += (int)sc * dev_dot_q4_32(x.qs + byte_off, y.qs + j * 32u, shift);
        }
        acc += y.d * xd * (float)isum - y.d * xmin * (float)summs;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down(acc, off, 32);
    if (lane == 0) out[row] = acc;
}

__global__ void k_q4k_mmvdq_down_wave32(const block_q4_K *weights,
                                        const float *x,
                                        float *out,
                                        uint32_t n_rows,
                                        uint32_t n_blocks) {
    const uint32_t row = blockIdx.x;
    if (row >= n_rows) return;
    const uint32_t lane = threadIdx.x;
    float acc = 0.0f;
    const block_q4_K *wr = weights + (uint64_t)row * n_blocks;
    for (uint32_t b = lane; b < n_blocks; b += 32u) {
        const block_q4_K blk = wr[b];
        const float *y = x + b * QK_K;
        const float xd = __half2float(*reinterpret_cast<const __half *>(&blk.d));
        const float xmin = __half2float(*reinterpret_cast<const __half *>(&blk.dmin));
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            uint8_t sc, m;
            dev_q4k_scale_min(j, blk.scales, &sc, &m);
            const float d = xd * (float)sc;
            const float dm = xmin * (float)m;
            const uint32_t byte_off = (j >> 1u) * 32u;
            const int shift = (j & 1u) ? 4 : 0;
            const float *yy = y + j * 32u;
            #pragma unroll
            for (uint32_t i = 0; i < 32u; i += 4u) {
                float4 v4 = *reinterpret_cast<const float4 *>(yy + i);
                const float a[4] = {v4.x, v4.y, v4.z, v4.w};
                #pragma unroll
                for (uint32_t k = 0; k < 4u; k++) {
                    const int q = (blk.qs[byte_off + i + k] >> shift) & 0x0f;
                    acc += (d * (float)q - dm) * a[k];
                }
            }
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down(acc, off, 32);
    if (lane == 0) out[row] = acc;
}

__device__ __forceinline__ static uint4 load_q4k_block_144(
        const block_q4_K *block, uint4 acc) {
    const uint4 *words = reinterpret_cast<const uint4 *>(block);
    #pragma unroll
    for (uint32_t i = 0; i < 9u; ++i) {
        const uint4 v = words[i];
        /* Four independent chains keep every loaded word observable without
         * turning the optimistic bandwidth floor into a serial ALU test. */
        acc.x ^= v.x;
        acc.y ^= v.y;
        acc.z ^= v.z;
        acc.w ^= v.w;
    }
    return acc;
}

/* Optimistic unique-byte floor for the production split gate/up geometry.
 * The grid keeps all six route slots so zero-owned slots retain launch/return
 * overhead, while active threads consume every byte of both Q4_K tables once.
 */
__global__ void k_q4k_gate_up_load_only(
        const block_q4_K *gate,
        const block_q4_K *up,
        uint32_t owned,
        uint32_t rows,
        uint32_t blocks_per_row,
        uint32_t *sink) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t slot = blockIdx.y;
    const uint32_t seed = 0x9e3779b9u ^ threadIdx.x ^ (slot << 16u);
    uint4 acc = make_uint4(seed, seed ^ 0x85ebca6bu,
                           seed ^ 0xc2b2ae35u, seed ^ 0x27d4eb2fu);
    if (slot < owned) {
        for (uint32_t rr = 0; rr < 8u; ++rr) {
            const uint32_t row = blockIdx.x * 128u + row_lane + rr * 16u;
            if (row >= rows) continue;
            const uint64_t row_base =
                ((uint64_t)slot * rows + row) * blocks_per_row;
            for (uint32_t b = lane; b < blocks_per_row; b += 8u) {
                acc = load_q4k_block_144(gate + row_base + b, acc);
                acc = load_q4k_block_144(up + row_base + b, acc);
            }
        }
    }
    const uint64_t out =
        ((uint64_t)blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x +
        threadIdx.x;
    sink[out] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}

/* One-projection form matching each of the two production gate/up launches. */
__global__ void k_q4k_projection_load_only(
        const block_q4_K *weight,
        uint32_t owned,
        uint32_t rows,
        uint32_t blocks_per_row,
        uint32_t *sink) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t slot = blockIdx.y;
    const uint32_t seed = 0x165667b1u ^ threadIdx.x ^ (slot << 16u);
    uint4 acc = make_uint4(seed, seed ^ 0x9e3779b9u,
                           seed ^ 0x85ebca6bu, seed ^ 0xc2b2ae35u);
    if (slot < owned) {
        for (uint32_t rr = 0; rr < 8u; ++rr) {
            const uint32_t row = blockIdx.x * 128u + row_lane + rr * 16u;
            if (row >= rows) continue;
            const uint64_t row_base =
                ((uint64_t)slot * rows + row) * blocks_per_row;
            for (uint32_t b = lane; b < blocks_per_row; b += 8u)
                acc = load_q4k_block_144(weight + row_base + b, acc);
        }
    }
    const uint64_t out =
        ((uint64_t)blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x +
        threadIdx.x;
    sink[out] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}

/* Optimistic unique-byte floor for the production direct sum6 down geometry.
 * Each lane consumes its one 144-byte block from every active expert/row.
 */
__global__ void k_q4k_down_load_only(
        const block_q4_K *down,
        uint32_t owned,
        uint32_t rows,
        uint32_t blocks_per_row,
        uint32_t *sink) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t seed = 0x85ebca6bu ^ threadIdx.x;
    uint4 acc = make_uint4(seed, seed ^ 0x9e3779b9u,
                           seed ^ 0xc2b2ae35u, seed ^ 0x27d4eb2fu);
    if (row < rows) {
        for (uint32_t slot = 0; slot < owned; ++slot) {
            const uint64_t block =
                ((uint64_t)slot * rows + row) * blocks_per_row + lane;
            if (lane < blocks_per_row)
                acc = load_q4k_block_144(down + block, acc);
        }
    }
    sink[(uint64_t)blockIdx.x * blockDim.x + threadIdx.x] =
        acc.x ^ acc.y ^ acc.z ^ acc.w;
}

/* Read-only device-memory calibration with the same observable XOR shape as
 * the Q4_K floor.  The 256 MiB source is larger than gfx1151's last-level
 * cache, so repeated iterations remain a DRAM-bandwidth reference. */
__global__ void k_stream_read_uint4(
        const uint4 *src, uint64_t words, uint4 *sink) {
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    const uint32_t seed = 0x27d4eb2fu ^ (uint32_t)tid;
    uint4 acc = make_uint4(seed, seed ^ 0x9e3779b9u,
                           seed ^ 0x85ebca6bu, seed ^ 0xc2b2ae35u);
    for (uint64_t i = tid; i < words; i += stride) {
        const uint4 v = src[i];
        acc.x ^= v.x;
        acc.y ^= v.y;
        acc.z ^= v.z;
        acc.w ^= v.w;
    }
    sink[tid] = acc;
}

static int run_shipped(ds4_gpu_tensor *out, ds4_gpu_tensor *gate,
                       ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
                       ds4_gpu_tensor *down, ds4_gpu_tensor *selected,
                       ds4_gpu_tensor *weights, ds4_gpu_tensor *x,
                       ds4_gpu_tensor *add_in, const void *model,
                       uint64_t model_bytes, uint64_t gate_off,
                       uint64_t up_off, uint64_t down_off,
                       uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
                       uint64_t down_expert_bytes, uint64_t down_row_bytes) {
    return ds4_gpu_routed_moe_one_tensor(
        out, gate, up, mid, down, model, model_bytes,
        gate_off, up_off, down_off, Q4_K_TYPE, Q4_K_TYPE,
        gate_expert_bytes, gate_row_bytes,
        down_expert_bytes, down_row_bytes,
        IN_DIM, MID_DIM, OUT_DIM, selected, weights,
        N_TOTAL_EXPERT, N_USED, 0.0f, x, add_in, 0, true);
}

int main(int argc, char **argv) {
    uint32_t owned = 6u;
    if (argc > 2) {
        fprintf(stderr, "usage: %s [owned-routes:1|3|6]\n", argv[0]);
        return 2;
    }
    if (argc == 2) owned = (uint32_t)strtoul(argv[1], NULL, 10);
    if (owned != 1u && owned != 3u && owned != 6u) {
        fprintf(stderr, "usage: %s [owned-routes:1|3|6]\n", argv[0]);
        return 2;
    }
    CHECK(setenv("DS4_ROCM_Q4K_DECODE_SPLIT_GATE_UP", "1", 1) == 0,
          "select benchmark split gate/up");
    CHECK(setenv("DS4_ROCM_Q4K_WMMA_PAIR_GATE_UP", "1", 1) == 0,
          "select benchmark Q4 WMMA policy");
    CHECK(setenv("DS4_ROCM_Q4K_WMMA_FUSE_MID", "1", 1) == 0,
          "select benchmark fused-mid policy");
    CHECK(setenv("DS4_ROCM_TP_SKIP_UNOWNED", "1", 1) == 0,
          "select benchmark zero-route skip");
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm backend");

    const uint64_t in_blocks = IN_DIM / QK_K;
    const uint64_t mid_blocks = MID_DIM / QK_K;
    const uint64_t gate_row_bytes = in_blocks * Q4_K_BLOCK_BYTES;
    const uint64_t down_row_bytes = mid_blocks * Q4_K_BLOCK_BYTES;
    const uint64_t gate_expert_bytes = MID_DIM * gate_row_bytes;
    const uint64_t down_expert_bytes = OUT_DIM * down_row_bytes;
    const uint64_t gate_off = 0;
    const uint64_t up_off = N_TOTAL_EXPERT * gate_expert_bytes;
    const uint64_t down_off = up_off + N_TOTAL_EXPERT * gate_expert_bytes;
    const uint64_t model_bytes = down_off + N_TOTAL_EXPERT * down_expert_bytes;

    unsigned char *copies[MODEL_COPIES];
    for (int c = 0; c < MODEL_COPIES; c++) {
        copies[c] = (unsigned char *)malloc((size_t)model_bytes);
        CHECK(copies[c], "allocate cold model copy");
        pack_q4k_table(copies[c] + gate_off, N_TOTAL_EXPERT, MID_DIM,
                       (uint32_t)in_blocks, 11);
        pack_q4k_table(copies[c] + up_off, N_TOTAL_EXPERT, MID_DIM,
                       (uint32_t)in_blocks, 37);
        pack_q4k_table(copies[c] + down_off, N_TOTAL_EXPERT, OUT_DIM,
                       (uint32_t)mid_blocks, 73);
    }
    CHECK(ds4_gpu_set_model_map(copies[0], model_bytes), "install model 0");

    const uint64_t pair_values = (uint64_t)N_USED * MID_DIM;
    ds4_gpu_tensor out = {}, gate = {}, up = {}, mid = {}, down = {};
    ds4_gpu_tensor selected = {}, weights = {}, x = {}, add_in = {};
    CHECK(alloc_tensor(&out, OUT_DIM * sizeof(float)), "allocate out");
    CHECK(alloc_tensor(&gate, pair_values * sizeof(float)), "allocate gate");
    CHECK(alloc_tensor(&up, pair_values * sizeof(float)), "allocate up");
    CHECK(alloc_tensor(&mid, pair_values * sizeof(float)), "allocate mid");
    CHECK(alloc_tensor(&down, (uint64_t)N_USED * OUT_DIM * sizeof(float)),
          "allocate down scratch");
    CHECK(alloc_tensor(&selected, N_USED * sizeof(int32_t)), "allocate selection");
    CHECK(alloc_tensor(&weights, N_USED * sizeof(float)), "allocate weights");
    CHECK(alloc_tensor(&x, IN_DIM * sizeof(float)), "allocate input");
    CHECK(alloc_tensor(&add_in, OUT_DIM * sizeof(float)), "allocate addend");

    int32_t route[N_USED] = {};
    float route_weights[N_USED] = {};
    const float base_weights[N_USED] =
        {0.31f, 0.23f, 0.17f, 0.13f, 0.09f, 0.07f};
    for (uint32_t i = 0; i < N_USED; ++i) {
        route[i] = i < owned ? (int32_t)i : 0;
        route_weights[i] = i < owned ? base_weights[i] : 0.0f;
    }
    float hx[IN_DIM], hadd[OUT_DIM];
    for (uint32_t i = 0; i < IN_DIM; i++)
        hx[i] = (float)((int)(i % 31u) - 15) * 0.03125f;
    for (uint32_t i = 0; i < OUT_DIM; i++)
        hadd[i] = (float)((int)(i % 19u) - 9) * 0.00390625f;
    CHECK(upload(&selected, route, sizeof(route)), "upload route");
    CHECK(upload(&weights, route_weights, sizeof(route_weights)), "upload weights");
    CHECK(upload(&x, hx, sizeof(hx)), "upload input");
    CHECK(upload(&add_in, hadd, sizeof(hadd)), "upload addend");

    for (uint32_t i = 0; i < WARMUP; i++) {
        CHECK(run_shipped(&out, &gate, &up, &mid, &down, &selected, &weights,
                          &x, &add_in, copies[0], model_bytes, gate_off, up_off,
                          down_off, gate_expert_bytes, gate_row_bytes,
                          down_expert_bytes, down_row_bytes), "warmup");
    }
    CHECK(hipDeviceSynchronize() == hipSuccess, "sync warmup");

    float hout_a[OUT_DIM], hout_b[OUT_DIM];
    CHECK(run_shipped(&out, &gate, &up, &mid, &down, &selected, &weights,
                      &x, &add_in, copies[0], model_bytes, gate_off, up_off,
                      down_off, gate_expert_bytes, gate_row_bytes,
                      down_expert_bytes, down_row_bytes), "gold A");
    CHECK(hipDeviceSynchronize() == hipSuccess, "sync A");
    CHECK(ds4_gpu_tensor_read(&out, 0, hout_a, sizeof(hout_a)), "read A");
    CHECK(run_shipped(&out, &gate, &up, &mid, &down, &selected, &weights,
                      &x, &add_in, copies[1], model_bytes, gate_off, up_off,
                      down_off, gate_expert_bytes, gate_row_bytes,
                      down_expert_bytes, down_row_bytes), "gold B");
    CHECK(hipDeviceSynchronize() == hipSuccess, "sync B");
    CHECK(ds4_gpu_tensor_read(&out, 0, hout_b, sizeof(hout_b)), "read B");
    CHECK(memcmp(hout_a, hout_b, sizeof(hout_a)) == 0,
          "shipped kernel bitwise across cold copies");

    hipEvent_t start = nullptr, stop = nullptr;
    CHECK(hipEventCreate(&start) == hipSuccess &&
          hipEventCreate(&stop) == hipSuccess, "events");
    CHECK(ds4_gpu_set_model_map(copies[0], model_bytes), "restore warm map");
    for (uint32_t i = 0; i < WARMUP; ++i) {
        CHECK(run_shipped(&out, &gate, &up, &mid, &down, &selected, &weights,
                          &x, &add_in, copies[0], model_bytes, gate_off, up_off,
                          down_off, gate_expert_bytes, gate_row_bytes,
                          down_expert_bytes, down_row_bytes), "warm shipped");
    }
    CHECK(hipDeviceSynchronize() == hipSuccess, "sync warm shipped");
    CHECK(hipEventRecord(start) == hipSuccess, "warm start");
    for (uint32_t i = 0; i < ITERS; ++i) {
        CHECK(run_shipped(&out, &gate, &up, &mid, &down, &selected, &weights,
                          &x, &add_in, copies[0], model_bytes, gate_off, up_off,
                          down_off, gate_expert_bytes, gate_row_bytes,
                          down_expert_bytes, down_row_bytes), "timed warm shipped");
    }
    CHECK(hipEventRecord(stop) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess, "warm stop");
    float shipped_warm_ms = 0.0f;
    CHECK(hipEventElapsedTime(&shipped_warm_ms, start, stop) == hipSuccess,
          "warm elapsed");

    /* Unique-byte floors use device-resident copies and the production grids.
     * Copy zero is warm-only. Four other rotating copies exceed gfx1151's
     * cache capacity and never reuse the just-warmed allocation. */
    block_q4_K *load_gate[MODEL_COPIES] = {};
    block_q4_K *load_up[MODEL_COPIES] = {};
    block_q4_K *load_down[MODEL_COPIES] = {};
    const uint64_t load_gate_bytes = (uint64_t)N_USED * gate_expert_bytes;
    const uint64_t load_down_bytes = (uint64_t)N_USED * down_expert_bytes;
    for (uint32_t c = 0; c < MODEL_COPIES; ++c) {
        CHECK(hipMalloc(&load_gate[c], (size_t)load_gate_bytes) == hipSuccess,
              "allocate load-only gate");
        CHECK(hipMalloc(&load_up[c], (size_t)load_gate_bytes) == hipSuccess,
              "allocate load-only up");
        CHECK(hipMalloc(&load_down[c], (size_t)load_down_bytes) == hipSuccess,
              "allocate load-only down");
        CHECK(hipMemcpy(load_gate[c], copies[c] + gate_off,
                        (size_t)load_gate_bytes, hipMemcpyHostToDevice) ==
                  hipSuccess,
              "copy load-only gate");
        CHECK(hipMemcpy(load_up[c], copies[c] + up_off,
                        (size_t)load_gate_bytes, hipMemcpyHostToDevice) ==
                  hipSuccess,
              "copy load-only up");
        CHECK(hipMemcpy(load_down[c], copies[c] + down_off,
                        (size_t)load_down_bytes, hipMemcpyHostToDevice) ==
                  hipSuccess,
              "copy load-only down");
    }
    uint32_t *load_sink = nullptr;
    const uint64_t sink_words =
        (uint64_t)((OUT_DIM + 31u) / 32u) * 256u;
    CHECK(hipMalloc(&load_sink, sink_words * sizeof(uint32_t)) == hipSuccess,
          "allocate load-only sink");
    const dim3 gate_load_grid((MID_DIM + 127u) / 128u, N_USED, 1u);
    const dim3 down_load_grid((OUT_DIM + 31u) / 32u, 1u, 1u);
    for (uint32_t i = 0; i < WARMUP; ++i) {
        k_q4k_gate_up_load_only<<<gate_load_grid, 128>>>(
            load_gate[0], load_up[0], owned, MID_DIM, (uint32_t)in_blocks,
            load_sink);
        k_q4k_projection_load_only<<<gate_load_grid, 128>>>(
            load_gate[0], owned, MID_DIM, (uint32_t)in_blocks, load_sink);
        k_q4k_projection_load_only<<<gate_load_grid, 128>>>(
            load_up[0], owned, MID_DIM, (uint32_t)in_blocks, load_sink);
        k_q4k_down_load_only<<<down_load_grid, 256>>>(
            load_down[0], owned, OUT_DIM, (uint32_t)mid_blocks, load_sink);
    }
    CHECK(hipGetLastError() == hipSuccess, "load-only warmup launch");
    CHECK(hipDeviceSynchronize() == hipSuccess, "load-only warmup");

    CHECK(hipEventRecord(start) == hipSuccess, "warm fused gate-load start");
    for (uint32_t i = 0; i < ITERS; ++i)
        k_q4k_gate_up_load_only<<<gate_load_grid, 128>>>(
            load_gate[0], load_up[0], owned, MID_DIM, (uint32_t)in_blocks,
            load_sink);
    CHECK(hipGetLastError() == hipSuccess, "warm fused gate-load launch");
    CHECK(hipEventRecord(stop) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess,
          "warm fused gate-load stop");
    float fused_gate_load_warm_ms = 0.0f;
    CHECK(hipEventElapsedTime(&fused_gate_load_warm_ms, start, stop) ==
              hipSuccess,
          "warm fused gate-load elapsed");

    CHECK(hipEventRecord(start) == hipSuccess, "warm split gate-load start");
    for (uint32_t i = 0; i < ITERS; ++i) {
        k_q4k_projection_load_only<<<gate_load_grid, 128>>>(
            load_gate[0], owned, MID_DIM, (uint32_t)in_blocks, load_sink);
        k_q4k_projection_load_only<<<gate_load_grid, 128>>>(
            load_up[0], owned, MID_DIM, (uint32_t)in_blocks, load_sink);
    }
    CHECK(hipGetLastError() == hipSuccess, "warm split gate-load launch");
    CHECK(hipEventRecord(stop) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess,
          "warm split gate-load stop");
    float split_gate_load_warm_ms = 0.0f;
    CHECK(hipEventElapsedTime(&split_gate_load_warm_ms, start, stop) ==
              hipSuccess,
          "warm split gate-load elapsed");

    CHECK(hipEventRecord(start) == hipSuccess, "warm down-load start");
    for (uint32_t i = 0; i < ITERS; ++i)
        k_q4k_down_load_only<<<down_load_grid, 256>>>(
            load_down[0], owned, OUT_DIM, (uint32_t)mid_blocks, load_sink);
    CHECK(hipGetLastError() == hipSuccess, "warm down-load launch");
    CHECK(hipEventRecord(stop) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess, "warm down-load stop");
    float down_load_warm_ms = 0.0f;
    CHECK(hipEventElapsedTime(&down_load_warm_ms, start, stop) == hipSuccess,
          "warm down-load elapsed");

    CHECK(hipEventRecord(start) == hipSuccess, "cold fused gate-load start");
    for (uint32_t i = 0; i < ITERS; ++i) {
        const uint32_t c = 1u + i % COLD_COPIES;
        k_q4k_gate_up_load_only<<<gate_load_grid, 128>>>(
            load_gate[c], load_up[c], owned, MID_DIM, (uint32_t)in_blocks,
            load_sink);
    }
    CHECK(hipGetLastError() == hipSuccess, "cold fused gate-load launch");
    CHECK(hipEventRecord(stop) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess,
          "cold fused gate-load stop");
    float fused_gate_load_cold_ms = 0.0f;
    CHECK(hipEventElapsedTime(&fused_gate_load_cold_ms, start, stop) ==
              hipSuccess,
          "cold fused gate-load elapsed");

    CHECK(hipEventRecord(start) == hipSuccess, "cold split gate-load start");
    for (uint32_t i = 0; i < ITERS; ++i) {
        const uint32_t c = 1u + i % COLD_COPIES;
        k_q4k_projection_load_only<<<gate_load_grid, 128>>>(
            load_gate[c], owned, MID_DIM, (uint32_t)in_blocks, load_sink);
        k_q4k_projection_load_only<<<gate_load_grid, 128>>>(
            load_up[c], owned, MID_DIM, (uint32_t)in_blocks, load_sink);
    }
    CHECK(hipGetLastError() == hipSuccess, "cold split gate-load launch");
    CHECK(hipEventRecord(stop) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess,
          "cold split gate-load stop");
    float split_gate_load_cold_ms = 0.0f;
    CHECK(hipEventElapsedTime(&split_gate_load_cold_ms, start, stop) ==
              hipSuccess,
          "cold split gate-load elapsed");

    CHECK(hipEventRecord(start) == hipSuccess, "cold down-load start");
    for (uint32_t i = 0; i < ITERS; ++i) {
        const uint32_t c = 1u + i % COLD_COPIES;
        k_q4k_down_load_only<<<down_load_grid, 256>>>(
            load_down[c], owned, OUT_DIM, (uint32_t)mid_blocks, load_sink);
    }
    CHECK(hipGetLastError() == hipSuccess, "cold down-load launch");
    CHECK(hipEventRecord(stop) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess, "cold down-load stop");
    float down_load_cold_ms = 0.0f;
    CHECK(hipEventElapsedTime(&down_load_cold_ms, start, stop) == hipSuccess,
          "cold down-load elapsed");

    fused_gate_load_warm_ms /= (float)ITERS;
    split_gate_load_warm_ms /= (float)ITERS;
    down_load_warm_ms /= (float)ITERS;
    fused_gate_load_cold_ms /= (float)ITERS;
    split_gate_load_cold_ms /= (float)ITERS;
    down_load_cold_ms /= (float)ITERS;

    uint32_t observed_load_sink = 0;
    CHECK(hipMemcpy(&observed_load_sink, load_sink, sizeof(observed_load_sink),
                    hipMemcpyDeviceToHost) == hipSuccess,
          "observe load-only sink");

    uint4 *stream_src = nullptr, *stream_sink = nullptr;
    const uint32_t stream_blocks = 1024u;
    const uint32_t stream_threads = 256u;
    const uint64_t stream_sink_words =
        (uint64_t)stream_blocks * stream_threads;
    CHECK(hipMalloc(&stream_src, (size_t)STREAM_BYTES) == hipSuccess,
          "allocate STREAM source");
    CHECK(hipMalloc(&stream_sink,
                    (size_t)stream_sink_words * sizeof(uint4)) == hipSuccess,
          "allocate STREAM sink");
    CHECK(hipMemset(stream_src, 0x5a, (size_t)STREAM_BYTES) == hipSuccess,
          "initialize STREAM source");
    k_stream_read_uint4<<<stream_blocks, stream_threads>>>(
        stream_src, STREAM_BYTES / sizeof(uint4), stream_sink);
    CHECK(hipGetLastError() == hipSuccess, "STREAM warmup launch");
    CHECK(hipDeviceSynchronize() == hipSuccess, "STREAM warmup");
    CHECK(hipEventRecord(start) == hipSuccess, "STREAM start");
    for (uint32_t i = 0; i < STREAM_ITERS; ++i)
        k_stream_read_uint4<<<stream_blocks, stream_threads>>>(
            stream_src, STREAM_BYTES / sizeof(uint4), stream_sink);
    CHECK(hipGetLastError() == hipSuccess, "STREAM timed launch");
    CHECK(hipEventRecord(stop) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess, "STREAM stop");
    float stream_ms = 0.0f;
    CHECK(hipEventElapsedTime(&stream_ms, start, stop) == hipSuccess,
          "STREAM elapsed");
    stream_ms /= (float)STREAM_ITERS;
    uint4 observed_stream_sink = {};
    CHECK(hipMemcpy(&observed_stream_sink, stream_sink,
                    sizeof(observed_stream_sink), hipMemcpyDeviceToHost) ==
              hipSuccess,
          "observe STREAM sink");
    const double stream_gbps =
        (double)STREAM_BYTES / ((double)stream_ms * 1.0e6);

    const uint64_t unique_gate_up_bytes =
        2u * (uint64_t)owned * MID_DIM * in_blocks * Q4_K_BLOCK_BYTES;
    const uint64_t unique_down_bytes =
        (uint64_t)owned * OUT_DIM * mid_blocks * Q4_K_BLOCK_BYTES;
    const uint64_t q4_blocks =
        2u * (uint64_t)owned * MID_DIM * in_blocks +
        (uint64_t)owned * OUT_DIM * mid_blocks;
    const uint64_t per_block_line_ceiling_bytes = q4_blocks * 192u;
    const double fused_warm_gbps =
        (double)(unique_gate_up_bytes + unique_down_bytes) /
        ((double)(fused_gate_load_warm_ms + down_load_warm_ms) * 1.0e6);
    const double split_warm_gbps =
        (double)(unique_gate_up_bytes + unique_down_bytes) /
        ((double)(split_gate_load_warm_ms + down_load_warm_ms) * 1.0e6);
    const double fused_cold_gbps =
        (double)(unique_gate_up_bytes + unique_down_bytes) /
        ((double)(fused_gate_load_cold_ms + down_load_cold_ms) * 1.0e6);
    const double split_cold_gbps =
        (double)(unique_gate_up_bytes + unique_down_bytes) /
        ((double)(split_gate_load_cold_ms + down_load_cold_ms) * 1.0e6);
    printf("q4k_stream_read bytes=%llu avg_ms=%.6f gbps=%.2f "
           "sink=%08x\n",
           (unsigned long long)STREAM_BYTES, stream_ms, stream_gbps,
           observed_stream_sink.x ^ observed_stream_sink.y ^
               observed_stream_sink.z ^ observed_stream_sink.w);
    printf("q4k_load_floor owned=%u unique_mib=%.3f "
           "per_block_line_ceiling_mib=%.3f "
           "warm_fused_gate_up_ms=%.6f warm_split_gate_up_ms=%.6f "
           "warm_down_ms=%.6f warm_fused_gbps=%.2f warm_split_gbps=%.2f "
           "cold_fused_gate_up_ms=%.6f cold_split_gate_up_ms=%.6f "
           "cold_down_ms=%.6f cold_fused_gbps=%.2f cold_split_gbps=%.2f "
           "sink=%08x\n",
           owned,
           (double)(unique_gate_up_bytes + unique_down_bytes) / 1048576.0,
           (double)per_block_line_ceiling_bytes / 1048576.0,
           fused_gate_load_warm_ms, split_gate_load_warm_ms,
           down_load_warm_ms, fused_warm_gbps, split_warm_gbps,
           fused_gate_load_cold_ms, split_gate_load_cold_ms,
           down_load_cold_ms, fused_cold_gbps, split_cold_gbps,
           observed_load_sink);

    /* Independent one-wave down oracle vs CPU of the same Q4_K×Q8_K formula. */
    const uint32_t n_blocks = (uint32_t)mid_blocks;
    const uint32_t n_rows = 1024; /* one TP half-row slice, still production K */
    block_q8_K hq8[MID_DIM / QK_K];
    float hmid[MID_DIM];
    for (uint32_t i = 0; i < MID_DIM; i++)
        hmid[i] = (float)((int)(i % 17u) - 8) * 0.0625f;
    quant_q8_K(hmid, hq8, MID_DIM);

    block_q4_K *h_w = (block_q4_K *)(copies[0] + down_off);
    float *cpu_pack = (float *)malloc(n_rows * sizeof(float));
    float *cpu_mmvdq = (float *)malloc(n_rows * sizeof(float));
    CHECK(cpu_pack && cpu_mmvdq, "cpu refs");
    for (uint32_t row = 0; row < n_rows; row++) {
        float acc = 0.0f, accf = 0.0f;
        for (uint32_t b = 0; b < n_blocks; b++) {
            acc += cpu_dot_q4k_q8k(h_w + (uint64_t)row * n_blocks + b, &hq8[b]);
            accf += cpu_dot_q4k_f32(h_w + (uint64_t)row * n_blocks + b,
                                    hmid + b * QK_K);
        }
        cpu_pack[row] = acc;
        cpu_mmvdq[row] = accf;
    }

    block_q4_K *d_w = nullptr;
    block_q8_K *d_q8 = nullptr;
    float *d_x = nullptr, *d_out = nullptr;
    CHECK(hipMalloc(&d_w, (size_t)n_rows * n_blocks * sizeof(block_q4_K)) == hipSuccess,
          "d_w");
    CHECK(hipMalloc(&d_q8, sizeof(hq8)) == hipSuccess, "d_q8");
    CHECK(hipMalloc(&d_x, MID_DIM * sizeof(float)) == hipSuccess, "d_x");
    CHECK(hipMalloc(&d_out, n_rows * sizeof(float)) == hipSuccess, "d_out");
    CHECK(hipMemcpy(d_w, h_w, (size_t)n_rows * n_blocks * sizeof(block_q4_K),
                    hipMemcpyHostToDevice) == hipSuccess, "copy w");
    CHECK(hipMemcpy(d_q8, hq8, sizeof(hq8), hipMemcpyHostToDevice) == hipSuccess,
          "copy q8");
    CHECK(hipMemcpy(d_x, hmid, sizeof(hmid), hipMemcpyHostToDevice) == hipSuccess,
          "copy x");

    k_q4k_q8k_down_wave32<<<n_rows, 32>>>(d_w, d_q8, d_out, n_rows, n_blocks);
    CHECK(hipDeviceSynchronize() == hipSuccess, "packed launch");
    float *gpu_pack = (float *)malloc(n_rows * sizeof(float));
    CHECK(gpu_pack, "gpu_pack");
    CHECK(hipMemcpy(gpu_pack, d_out, n_rows * sizeof(float),
                    hipMemcpyDeviceToHost) == hipSuccess, "read packed");
    float pack_max_abs = 0.0f, pack_max_rel = 0.0f;
    for (uint32_t i = 0; i < n_rows; i++) {
        const float e = fabsf(gpu_pack[i] - cpu_pack[i]);
        const float d = fmaxf(1.0f, fabsf(cpu_pack[i]));
        if (e > pack_max_abs) pack_max_abs = e;
        if (e / d > pack_max_rel) pack_max_rel = e / d;
    }
    /* Warp F32 reduction reassociates the 8 superblock sums vs CPU order. */
    CHECK(pack_max_rel < 1e-5f, "packed DP4A documented-ULP vs CPU Q4_K×Q8_K");
    printf("packed_vs_cpu max_abs=%.6e max_rel=%.6e\n", pack_max_abs, pack_max_rel);

    k_q4k_mmvdq_down_wave32<<<n_rows, 32>>>(d_w, d_x, d_out, n_rows, n_blocks);
    CHECK(hipDeviceSynchronize() == hipSuccess, "mmvdq launch");
    float *gpu_mmvdq = (float *)malloc(n_rows * sizeof(float));
    CHECK(gpu_mmvdq, "gpu_mmvdq");
    CHECK(hipMemcpy(gpu_mmvdq, d_out, n_rows * sizeof(float),
                    hipMemcpyDeviceToHost) == hipSuccess, "read mmvdq");
    double sse = 0.0, nrm = 0.0;
    for (uint32_t i = 0; i < n_rows; i++) {
        const double e = (double)gpu_mmvdq[i] - (double)cpu_mmvdq[i];
        sse += e * e;
        nrm += (double)cpu_mmvdq[i] * (double)cpu_mmvdq[i];
    }
    const double nrmse = nrm > 0.0 ? sqrt(sse / nrm) : 0.0;
    CHECK(nrmse < 1e-5, "MMVDQ NRMSE vs F32 reference");

    for (int i = 0; i < 8; i++)
        k_q4k_q8k_down_wave32<<<n_rows, 32>>>(d_w, d_q8, d_out, n_rows, n_blocks);
    CHECK(hipDeviceSynchronize() == hipSuccess, "packed warmup");
    CHECK(hipEventRecord(start) == hipSuccess, "packed start");
    for (int i = 0; i < ITERS; i++)
        k_q4k_q8k_down_wave32<<<n_rows, 32>>>(d_w, d_q8, d_out, n_rows, n_blocks);
    CHECK(hipEventRecord(stop) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess, "packed stop");
    float packed_ms = 0.0f;
    CHECK(hipEventElapsedTime(&packed_ms, start, stop) == hipSuccess, "packed elapsed");

    CHECK(hipEventRecord(start) == hipSuccess, "mmvdq start");
    for (int i = 0; i < ITERS; i++)
        k_q4k_mmvdq_down_wave32<<<n_rows, 32>>>(d_w, d_x, d_out, n_rows, n_blocks);
    CHECK(hipEventRecord(stop) == hipSuccess &&
          hipEventSynchronize(stop) == hipSuccess, "mmvdq stop");
    float mmvdq_ms = 0.0f;
    CHECK(hipEventElapsedTime(&mmvdq_ms, start, stop) == hipSuccess, "mmvdq elapsed");

    const float shipped_warm_avg = shipped_warm_ms / (float)ITERS;
    const float packed_avg = packed_ms / (float)ITERS;
    const float mmvdq_avg = mmvdq_ms / (float)ITERS;
    printf("test_rocm_q4k_one_token_oracle: owned=%u "
           "shipped_device_resident_avg_ms=%.6f "
           "packed_down_1024_avg_ms=%.6f mmvdq_down_1024_avg_ms=%.6f "
           "mmvdq_nrmse=%.3e shipped_fnv64=%016llx waves=1 "
           "kill_note=10pct_isolated_not_25tps\n",
           owned, shipped_warm_avg, packed_avg, mmvdq_avg, nrmse,
           (unsigned long long)fnv1a64(hout_a, sizeof(hout_a)));

    (void)hipFree(d_w); (void)hipFree(d_q8); (void)hipFree(d_x); (void)hipFree(d_out);
    (void)hipFree(stream_sink);
    (void)hipFree(stream_src);
    (void)hipFree(load_sink);
    for (uint32_t c = 0; c < MODEL_COPIES; ++c) {
        (void)hipFree(load_down[c]);
        (void)hipFree(load_up[c]);
        (void)hipFree(load_gate[c]);
    }
    (void)hipEventDestroy(start); (void)hipEventDestroy(stop);
    ds4_gpu_tensor_free_in_place(&add_in);
    ds4_gpu_tensor_free_in_place(&x);
    ds4_gpu_tensor_free_in_place(&weights);
    ds4_gpu_tensor_free_in_place(&selected);
    ds4_gpu_tensor_free_in_place(&down);
    ds4_gpu_tensor_free_in_place(&mid);
    ds4_gpu_tensor_free_in_place(&up);
    ds4_gpu_tensor_free_in_place(&gate);
    ds4_gpu_tensor_free_in_place(&out);
    for (int c = 0; c < MODEL_COPIES; c++) free(copies[c]);
    free(cpu_pack); free(cpu_mmvdq); free(gpu_pack); free(gpu_mmvdq);
    ds4_gpu_cleanup();
    return 0;
}
