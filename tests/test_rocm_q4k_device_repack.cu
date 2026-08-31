/* Small oracle for the resident-source Q4_K strided repack primitive.
 * It intentionally copies bytes only; Q4_K block decoding is not involved. */
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(x, msg) do { if (!(x)) { std::fprintf(stderr, "FAIL %s\n", msg); return 1; } } while (0)

__global__ static void repack_rows(const uint8_t *src, uint8_t *dst,
                                   uint32_t experts, uint32_t src_rows,
                                   uint32_t src_row_bytes, uint32_t row_base,
                                   uint32_t row_count, uint32_t col_base,
                                   uint32_t col_bytes) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)experts * row_count * col_bytes;
    if (i >= total) return;
    const uint64_t e = i / ((uint64_t)row_count * col_bytes);
    const uint64_t r = (i / col_bytes) % row_count;
    const uint64_t c = i % col_bytes;
    const uint64_t so = e * (uint64_t)src_rows * src_row_bytes +
                        (row_base + r) * (uint64_t)src_row_bytes + col_base + c;
    dst[i] = src[so];
}

int main() {
    constexpr uint32_t experts = 6, src_rows = 64, src_row_bytes = 256;
    constexpr uint32_t row_base = 16, row_count = 32, col_base = 64, col_bytes = 96;
    const uint64_t src_n = (uint64_t)experts * src_rows * src_row_bytes;
    const uint64_t dst_n = (uint64_t)experts * row_count * col_bytes;
    std::vector<uint8_t> hsrc(src_n), href(dst_n), hdst(dst_n);
    for (uint64_t i = 0; i < src_n; ++i) hsrc[i] = (uint8_t)((i * 37u + 19u) & 255u);
    for (uint32_t e = 0; e < experts; ++e)
        for (uint32_t r = 0; r < row_count; ++r)
            std::memcpy(href.data() + ((uint64_t)e * row_count + r) * col_bytes,
                        hsrc.data() + (uint64_t)e * src_rows * src_row_bytes +
                            (row_base + r) * src_row_bytes + col_base, col_bytes);
    uint8_t *src = nullptr, *dst = nullptr;
    CHECK(hipMalloc(&src, src_n) == hipSuccess && hipMalloc(&dst, dst_n) == hipSuccess,
          "device allocation");
    CHECK(hipMemcpy(src, hsrc.data(), src_n, hipMemcpyHostToDevice) == hipSuccess,
          "source upload");
    const uint64_t blocks = (dst_n + 255u) / 256u;
    hipLaunchKernelGGL(repack_rows, dim3(blocks), dim3(256), 0, 0, src, dst,
                       experts, src_rows, src_row_bytes, row_base, row_count,
                       col_base, col_bytes);
    CHECK(hipGetLastError() == hipSuccess && hipDeviceSynchronize() == hipSuccess,
          "repack launch");
    CHECK(hipMemcpy(hdst.data(), dst, dst_n, hipMemcpyDeviceToHost) == hipSuccess,
          "result download");
    CHECK(std::memcmp(href.data(), hdst.data(), dst_n) == 0,
          "device repack differs from byte reference");
    std::printf("PASS resident Q4_K device repack oracle: experts=%u rows=%u cols=%u bytes=%llu\n",
                experts, row_count, col_bytes, (unsigned long long)dst_n);
    hipFree(dst); hipFree(src);
    return 0;
}
