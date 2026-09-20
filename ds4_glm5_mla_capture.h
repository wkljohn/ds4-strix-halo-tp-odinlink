#ifndef DS4_GLM5_MLA_CAPTURE_H
#define DS4_GLM5_MLA_CAPTURE_H

/* Narrow research capture: no graph/quality mode, GPU allocation or arithmetic
 * selection changes. Reads synchronize, so enabled runs are not timing proof.
 * Capture full physical rows at three layers; never overwrite prior evidence. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ds4_gpu.h"

static int glm5_mla_capture_file(const char *path,
        const ds4_gpu_tensor *tensor, uint64_t bytes) {
    if (!tensor || ds4_gpu_tensor_bytes(tensor) < bytes) return 0;
    void *host = malloc((size_t)bytes);
    if (!host) return 0;
    int ok = ds4_gpu_tensor_read(tensor, 0u, host, bytes);
    FILE *fp = ok ? fopen(path, "wbx") : NULL;
    if (fp) {
        ok = fwrite(host, 1u, (size_t)bytes, fp) == (size_t)bytes;
        if (fclose(fp) != 0) ok = 0;
    } else ok = 0;
    free(host);
    return ok;
}

static int glm5_mla_output_capture(uint32_t rank, uint32_t layer,
        uint32_t pos, uint32_t rows, uint64_t weight_offset,
        uint64_t stride, const ds4_gpu_tensor *heads,
        const ds4_gpu_tensor *output) {
    const char *prefix = getenv("DS4_GLM5_MLA_OUTPUT_CAPTURE_PREFIX");
    if (!prefix) return 1;
    const char *position = getenv("DS4_GLM5_MLA_OUTPUT_CAPTURE_POS");
    const char *run_id = getenv("DS4_BENCH_RUN_ID");
    const char *wmma = getenv("DS4_ROCM_GLM5_MLA_OUTPUT_WMMA");
    if (prefix[0] != '/' || !position || !run_id || !run_id[0] ||
        !wmma || strcmp(wmma, "0") != 0 ||
        strspn(run_id, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                       "0123456789-_.") != strlen(run_id) ||
        (strcmp(position, "3072") && strcmp(position, "7168"))) return 0;
    const uint32_t selected = strcmp(position, "3072") == 0 ? 3072u : 7168u;
    if (pos != selected || (layer != 3u && layer != 23u && layer != 43u))
        return 1;
    /* This first diagnostic deliberately covers the production full-head
     * M1024 layout only. An owned-half or smaller batch needs a separate plan. */
    if (rank > 1u || rows != 1024u || stride != 16384u) return 0;
    char path[1024], stem[960];
    int n = snprintf(stem, sizeof(stem), "%s.r%u.l%u.p%u.m%u",
                     prefix, rank, layer, pos, rows);
    if (n <= 0 || (size_t)n >= sizeof(stem)) return 0;
    n = snprintf(path, sizeof(path), "%s.heads.f32", stem);
    if (n <= 0 || (size_t)n >= sizeof(path) ||
        !glm5_mla_capture_file(path, heads,
            (uint64_t)rows * stride * sizeof(float))) return 0;
    n = snprintf(path, sizeof(path), "%s.output.f32", stem);
    if (n <= 0 || (size_t)n >= sizeof(path) ||
        !glm5_mla_capture_file(path, output,
            (uint64_t)rows * 4096u * sizeof(float))) return 0;
    n = snprintf(path, sizeof(path), "%s.meta", stem);
    if (n <= 0 || (size_t)n >= sizeof(path)) return 0;
    FILE *fp = fopen(path, "wx");
    if (!fp) return 0;
    int ok = fprintf(fp,
        "schema=1\nwmma=0\nrun_id=%s\nrank=%u\nlayer=%u\npos=%u\nrows=%u\n"
        "stride=%llu\nlocal_k=8192\nout_dim=4096\nweight_offset=%llu\n"
        "heads_bytes=%llu\noutput_bytes=%llu\n",
        run_id, rank, layer, pos, rows, (unsigned long long)stride,
        (unsigned long long)weight_offset,
        (unsigned long long)rows * stride * sizeof(float),
        (unsigned long long)rows * 4096u * sizeof(float)) > 0;
    if (fclose(fp) != 0) ok = 0;
    if (ok) fprintf(stderr,
        "ds4: MLA output capture completed rank=%u layer=%u pos=%u rows=%u "
        "stem=%s timing_eligible=0\n", rank, layer, pos, rows, stem);
    return ok;
}
#endif
