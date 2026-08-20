/* Model-free correctness oracle for the ROCm split big-gate primitive.
 * It records real producer events on the default stream, queues several
 * exchanges before waiting, verifies payload visibility, then proves that an
 * ordinary blocking big gate may safely follow the drained split queue. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <hip/hip_runtime.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <vector>

#define CHECK(cond, msg) do {                                                \
    if (!(cond)) {                                                           \
        std::fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);      \
        return 1;                                                            \
    }                                                                        \
} while (0)

typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

static constexpr uint32_t kChunks = 8u;
static constexpr uint32_t kValuesPerChunk = 4096u;
static constexpr uint64_t kChunkBytes =
    (uint64_t)kValuesPerChunk * sizeof(uint32_t);
static constexpr uint64_t kFlagsOffset = 0u;
static constexpr uint64_t kOutOffset = 64u;
static constexpr uint64_t kInOffset = kOutOffset + kChunks * kChunkBytes;
static constexpr uint64_t kSlabBytes = kInOffset + kChunks * kChunkBytes;

struct fake_transport {
    unsigned char *host_base;
    unsigned char *device_base;
    uint64_t bytes;
    std::atomic<uint32_t> calls;
};

__global__ static void produce_chunk(uint32_t *dst, uint32_t chunk) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kValuesPerChunk) {
        dst[i] = 0x51000000u ^ (chunk * 0x00010001u) ^ i;
    }
}

static int row_exchange(void *, uint32_t, uint32_t, uint64_t) {
    return 1;
}

static int big_exchange(void *opaque, uint32_t, uint64_t,
                        const void *out, void *in, uint64_t bytes) {
    fake_transport *t = static_cast<fake_transport *>(opaque);
    const uintptr_t base = reinterpret_cast<uintptr_t>(t->device_base);
    const uintptr_t outp = reinterpret_cast<uintptr_t>(out);
    const uintptr_t inp = reinterpret_cast<uintptr_t>(in);
    if (outp < base || inp < base || bytes > t->bytes ||
        outp - base > t->bytes - bytes || inp - base > t->bytes - bytes) {
        return 0;
    }
    /* Make multiple requests observably in flight instead of merely testing
     * a fast callback that happens to finish before the next kick. */
    usleep(1000);
    std::memcpy(t->host_base + (inp - base),
                t->host_base + (outp - base), (size_t)bytes);
    t->calls.fetch_add(1u, std::memory_order_release);
    return 1;
}

int main(void) {
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm backend");

    ds4_gpu_tensor *slab = ds4_gpu_tensor_alloc_rdma_host(kSlabBytes);
    void *slab_host = slab ? ds4_gpu_tensor_contents(slab) : nullptr;
    CHECK(slab && slab->ptr && slab_host, "allocate mapped gate slab");
    std::memset(slab_host, 0, (size_t)kSlabBytes);
    fake_transport transport = {
        static_cast<unsigned char *>(slab_host),
        static_cast<unsigned char *>(slab->ptr),
        kSlabBytes,
        0u,
    };
    CHECK(ds4_gpu_tp_init(0u, slab, kFlagsOffset,
                          row_exchange, &transport),
          "initialize TP gate service");
    ds4_gpu_tp_set_big_exchange(big_exchange);

    std::vector<ds4_gpu_tensor *> out(kChunks), in(kChunks);
    std::vector<uint64_t> seq(kChunks);
    for (uint32_t chunk = 0; chunk < kChunks; chunk++) {
        out[chunk] = ds4_gpu_tensor_view(
            slab, kOutOffset + (uint64_t)chunk * kChunkBytes, kChunkBytes);
        in[chunk] = ds4_gpu_tensor_view(
            slab, kInOffset + (uint64_t)chunk * kChunkBytes, kChunkBytes);
        CHECK(out[chunk] && in[chunk], "create split gate views");
        produce_chunk<<<(kValuesPerChunk + 255u) / 256u, 256u>>>(
            static_cast<uint32_t *>(out[chunk]->ptr), chunk);
        CHECK(hipGetLastError() == hipSuccess, "launch gate producer");
        seq[chunk] = ds4_gpu_tp_big_gate_kick(
            7u, kValuesPerChunk, out[chunk], in[chunk], kChunkBytes);
        CHECK(seq[chunk] != 0u, "queue split big gate");
    }
    for (uint32_t chunk = 0; chunk < kChunks; chunk++) {
        CHECK(ds4_gpu_tp_big_gate_wait(seq[chunk]),
              "wait split big gate chunk");
        const uint32_t *got = reinterpret_cast<const uint32_t *>(
            transport.host_base + kInOffset +
            (uint64_t)chunk * kChunkBytes);
        for (uint32_t i = 0; i < kValuesPerChunk; i++) {
            const uint32_t expected =
                0x51000000u ^ (chunk * 0x00010001u) ^ i;
            CHECK(got[i] == expected,
                  "split gate must expose complete producer payload");
        }
    }
    CHECK(transport.calls.load(std::memory_order_acquire) == kChunks,
          "all split callbacks must run exactly once");

    /* The queue count and release word must become consistent atomically.
     * This arm catches the old pop-before-release race. */
    produce_chunk<<<(kValuesPerChunk + 255u) / 256u, 256u>>>(
        static_cast<uint32_t *>(out[0]->ptr), 99u);
    CHECK(hipGetLastError() == hipSuccess, "launch blocking-gate producer");
    CHECK(ds4_gpu_tp_big_gate_encode(
              8u, kValuesPerChunk, out[0], in[0], kChunkBytes),
          "ordinary big gate after split queue");
    CHECK(transport.calls.load(std::memory_order_acquire) == kChunks + 1u,
          "blocking callback must follow split callbacks");

    ds4_gpu_tp_shutdown();
    for (uint32_t chunk = 0; chunk < kChunks; chunk++) {
        ds4_gpu_tensor_free(out[chunk]);
        ds4_gpu_tensor_free(in[chunk]);
    }
    ds4_gpu_tensor_free(slab);
    ds4_gpu_cleanup();
    std::fprintf(stderr,
                 "test_rocm_tp_split_gate: PASS chunks=%u calls=%u\n",
                 kChunks, kChunks + 1u);
    return 0;
}
