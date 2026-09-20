#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ds4_glm5_mla_capture.h"

struct ds4_gpu_tensor { uint64_t bytes; };
static unsigned reads;
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *t) { return t->bytes; }
int ds4_gpu_tensor_read(const ds4_gpu_tensor *t, uint64_t off,
                       void *data, uint64_t bytes) {
    assert(off <= t->bytes && bytes <= t->bytes - off);
    ++reads;
    memset(data, 0, (size_t)bytes);
    return 1;
}

int main(int argc, char **argv) {
    assert(argc == 2); /* Unique directory under DS4_RESEARCH_ROOT. */
    char prefix[768];
    assert(snprintf(prefix, sizeof(prefix), "%s/capture", argv[1]) > 0);
    ds4_gpu_tensor x = {1024ull * 16384u * 4u}, y = {1024ull * 4096u * 4u};
    assert(glm5_mla_output_capture(0, 3, 3072, 1024, 123, 16384, &x, &y));
    assert(reads == 0);
    assert(setenv("DS4_GLM5_MLA_OUTPUT_CAPTURE_PREFIX", prefix, 1) == 0);
    assert(!glm5_mla_output_capture(0, 3, 3072, 1024, 123, 16384, &x, &y));
    assert(setenv("DS4_BENCH_RUN_ID", "test-capture", 1) == 0);
    assert(setenv("DS4_ROCM_GLM5_MLA_OUTPUT_WMMA", "1", 1) == 0);
    assert(setenv("DS4_GLM5_MLA_OUTPUT_CAPTURE_POS", "3072", 1) == 0);
    assert(!glm5_mla_output_capture(0, 3, 3072, 1024, 123, 16384, &x, &y));
    assert(setenv("DS4_ROCM_GLM5_MLA_OUTPUT_WMMA", "0", 1) == 0);
    assert(setenv("DS4_GLM5_MLA_OUTPUT_CAPTURE_POS", "3072junk", 1) == 0);
    assert(!glm5_mla_output_capture(0, 3, 3072, 1024, 123, 16384, &x, &y));
    assert(setenv("DS4_GLM5_MLA_OUTPUT_CAPTURE_POS", "3072", 1) == 0);
    assert(glm5_mla_output_capture(0, 7, 3072, 1024, 123, 16384, &x, &y));
    assert(glm5_mla_output_capture(0, 3, 2048, 1024, 123, 16384, &x, &y));
    assert(!glm5_mla_output_capture(0, 3, 3072, 256, 123, 16384, &x, &y));
    assert(!glm5_mla_output_capture(0, 3, 3072, 1024, 123, 8192, &x, &y));
    assert(!glm5_mla_output_capture(2, 3, 3072, 1024, 123, 16384, &x, &y));
    ds4_gpu_tensor short_x = {x.bytes - 1u};
    assert(!glm5_mla_output_capture(0, 3, 3072, 1024, 123, 16384, &short_x, &y));
    assert(reads == 0);
    assert(glm5_mla_output_capture(0, 3, 3072, 1024, 123, 16384, &x, &y));
    assert(reads == 2);
    assert(!glm5_mla_output_capture(0, 3, 3072, 1024, 123, 16384, &x, &y));
    puts("PASS capture off/selection/refusal/exclusive files");
    return 0;
}
