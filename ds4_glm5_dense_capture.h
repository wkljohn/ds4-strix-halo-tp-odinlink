#ifndef DS4_GLM5_DENSE_CAPTURE_H
#define DS4_GLM5_DENSE_CAPTURE_H

/* Diagnostic-only original dense inputs/results. This hook changes no
 * arithmetic selection or device allocations; reads synchronize. */
#include "ds4_glm5_mla_capture.h"

static int glm5_dense_prefill_capture(uint32_t rank, uint32_t layer,
        uint64_t pos, uint32_t rows, uint64_t gate_offset,
        uint64_t up_offset, uint64_t down_offset,
        const ds4_gpu_tensor *input, const ds4_gpu_tensor *mid,
        const ds4_gpu_tensor *down) {
    const char *prefix = getenv("DS4_GLM5_DENSE_CAPTURE_PREFIX");
    if (!prefix) return 1;
    const char *position = getenv("DS4_GLM5_DENSE_CAPTURE_POS");
    const char *run_id = getenv("DS4_BENCH_RUN_ID");
    if (prefix[0] != '/' || !position || !run_id || !run_id[0] ||
        strspn(run_id, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                       "0123456789-_.") != strlen(run_id) ||
        (strcmp(position, "3072") && strcmp(position, "7168"))) return 0;
    const uint32_t selected = strcmp(position, "3072") == 0 ? 3072u : 7168u;
    if (pos != selected || layer > 2u) return 1;
    const uint64_t input_bytes = 1024ull * 4096u * sizeof(float);
    const uint64_t mid_bytes = 1024ull * 12288u * sizeof(float);
    if (rank > 1u || rows != 1024u || !input || !mid || !down ||
        ds4_gpu_tensor_bytes(input) < input_bytes ||
        ds4_gpu_tensor_bytes(mid) < mid_bytes ||
        ds4_gpu_tensor_bytes(down) < input_bytes ||
        !gate_offset || !up_offset || !down_offset ||
        gate_offset == up_offset || gate_offset == down_offset ||
        up_offset == down_offset) return 0;
    char stem[960], path[1024];
    int n = snprintf(stem, sizeof(stem), "%s.r%u.l%u.p%u.m%u",
                     prefix, rank, layer, selected, rows);
    if (n <= 0 || (size_t)n >= sizeof(stem)) return 0;
    const char *suffix[] = {"input", "mid", "down"};
    const ds4_gpu_tensor *tensors[] = {input, mid, down};
    const uint64_t sizes[] = {input_bytes, mid_bytes, input_bytes};
    for (unsigned i = 0; i < 3; ++i) {
        n = snprintf(path, sizeof(path), "%s.%s.f32", stem, suffix[i]);
        if (n <= 0 || (size_t)n >= sizeof(path) ||
            !glm5_mla_capture_file(path, tensors[i], sizes[i])) return 0;
    }
    n = snprintf(path, sizeof(path), "%s.meta", stem);
    if (n <= 0 || (size_t)n >= sizeof(path)) return 0;
    FILE *fp = fopen(path, "wx");
    if (!fp) return 0;
    int ok = fprintf(fp,
        "schema=1\nimplementation=f32-token-tile\nrun_id=%s\nrank=%u\n"
        "layer=%u\npos=%u\nrows=1024\nin_dim=4096\nmid_dim=12288\n"
        "clamp=10\ngate_offset=%llu\nup_offset=%llu\ndown_offset=%llu\n"
        "input_bytes=%llu\nmid_bytes=%llu\ndown_bytes=%llu\n",
        run_id, rank, layer, selected, (unsigned long long)gate_offset,
        (unsigned long long)up_offset, (unsigned long long)down_offset,
        (unsigned long long)input_bytes, (unsigned long long)mid_bytes,
        (unsigned long long)input_bytes) > 0;
    if (fclose(fp) != 0) ok = 0;
    if (ok) fprintf(stderr,
        "ds4: dense prefill capture completed rank=%u layer=%u pos=%u "
        "rows=%u stem=%s timing_eligible=0\n", rank, layer, selected, rows, stem);
    return ok;
}
#endif
