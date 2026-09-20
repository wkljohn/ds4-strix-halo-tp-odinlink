// Compare the real packed-half production entry with independent <=M256
// calls, including the final partial group. Rare routes change hot/cold
// occupancy if an outer batch is accidentally treated as one group.
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
extern "C" {
#include "ds4_tp.h"
}
#include "glm5_gguf_test.hpp"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}
#define REQUIRE(x) do { if (!(x)) { \
    std::fprintf(stderr, "FAIL line=%d: %s\n", __LINE__, #x); \
    std::exit(1); } } while (0)

int main(int argc, char **argv) {
    constexpr uint32_t width = 4096, mid_width = 1024;
    uint32_t rows = 1024;
    REQUIRE(argc <= 3);
    const bool wide = argc == 3 && std::strcmp(argv[2], "--prefill-wide128") == 0;
    const unsigned schedule = wide ? 1u : argc != 3 ? 0u :
        std::strcmp(argv[2], "--prefill-schedule1") == 0 ? 1u :
        std::strcmp(argv[2], "--prefill-schedule2") == 0 ? 2u : 0u;
    const bool cold_i8 = schedule || (argc == 3 && std::strcmp(argv[2], "--grouped-cold-i8") == 0);
    const bool cold_coalesce = cold_i8 || (argc == 3 && std::strcmp(argv[2], "--grouped-cold") == 0);
    const bool grouped = cold_coalesce || (argc == 3 && std::strcmp(argv[2], "--grouped") == 0);
    const bool dot_unroll = argc == 3 && std::strcmp(argv[2], "--decode-dot2") == 0;
    const bool dot_lanes = argc == 3 && std::strcmp(argv[2], "--decode-dot4") == 0;
    const unsigned decode_rows = argc != 3 ? 0u :
        (dot_unroll || dot_lanes) ? 128u :
        std::strcmp(argv[2], "--decode32") == 0 ? 32u :
        std::strcmp(argv[2], "--decode64") == 0 ? 64u : 0u;
    const bool cold_lds5 = argc == 3 && !grouped && !decode_rows;
    const bool cold_padding = cold_lds5 &&
        std::strcmp(argv[2], "--cold-lds5-pad") == 0;
    if (cold_lds5) REQUIRE(cold_padding || std::strcmp(argv[2], "--cold-lds5") == 0);
    const char *timing_selector = wide ? "DS4_ROCM_GLM5_Q4K_PREFILL_WIDE128" :
        schedule ? "DS4_ROCM_GLM5_Q4K_PREFILL_SCHEDULE" :
        cold_i8 ? "DS4_ROCM_GLM5_Q4K_GROUPED_COLD_I8" :
        cold_coalesce ? "DS4_ROCM_GLM5_Q4K_GROUPED_COLD_COALESCE" :
        dot_lanes ? "DS4_ROCM_GLM5_Q4K_DECODE_DOT_LANES" :
        dot_unroll ? "DS4_ROCM_GLM5_Q4K_DECODE_DOT_UNROLL" :
        decode_rows ? "DS4_ROCM_GLM5_Q4K_DECODE_GATE_ROWS" :
        grouped ? "DS4_ROCM_GLM5_Q4K_PREFILL_GROUPED" : cold_padding ?
        "DS4_ROCM_GLM5_Q4K_COLD_LDS5_PAD" : "DS4_ROCM_GLM5_Q4K_COLD_LDS5";
    const char *candidate_mode = schedule == 2u ? "2" : dot_lanes ? "4" : dot_unroll ? "2" :
        decode_rows == 32u ? "32" : decode_rows == 64u ? "64" : "1";
    const char *input_case = std::getenv("DS4_TEST_MOE_INPUT_CASE");
    unsigned fixture = 0;
    if (input_case) {
        REQUIRE(std::strcmp(input_case,"0") == 0 || std::strcmp(input_case,"1") == 0 ||
                std::strcmp(input_case,"2") == 0 ||
                (schedule && (std::strcmp(input_case,"3") == 0 || std::strcmp(input_case,"4") == 0 || std::strcmp(input_case,"5") == 0)));
        fixture = unsigned(input_case[0]-'0');
    }
    if (argc >= 2) {
        char *end = nullptr;
        const unsigned long count = std::strtoul(argv[1], &end, 10);
        REQUIRE(end != argv[1] && *end == '\0' && count > 0 && count <= 1024);
        rows = uint32_t(count);
    }
    if (decode_rows) REQUIRE(rows == 1u);
    constexpr uint32_t experts = 288, used = 8, tile = 256;
    constexpr uint64_t gate_row = 16u * 144u, down_row = 8u * 144u;
    const char *model = std::getenv("DS4_GLM5_MODEL");
    REQUIRE(model);
    Glm5TestGGUF gguf;
    REQUIRE(gguf.open_file(model));
    uint64_t go, uo, down_offset;
    REQUIRE(gguf.tensor("blk.3.ffn_gate_exps.weight", {4096,2048,288},12,go));
    REQUIRE(gguf.tensor("blk.3.ffn_up_exps.weight", {4096,2048,288},12,uo));
    REQUIRE(gguf.tensor("blk.3.ffn_down_exps.weight", {2048,4096,288},12,down_offset));
    REQUIRE(setenv("DS4_ROCM_Q4K_KSHARD_RESEARCH", "1", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_Q4K_WMMA", "1", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_DISABLE_Q4K_WMMA", "0", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_Q4K_WMMA_MIN_COUNT", "6", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_Q4K_WMMA_PAIR_GATE_UP", "0", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_Q4K_WMMA_FUSE_MID", "0", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_Q4K_COLD_TILE4", "0", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_COLD_LDS5", "0", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_COLD_LDS5_PAD", "0", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_PREFILL_GROUPED", "0", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_PREFILL_SCHEDULE", "0", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_PREFILL_WIDE128", "0", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_GROUPED_COLD_COALESCE", "0", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_GROUPED_COLD_I8", "0", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_DECODE_GATE_ROWS", "0", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_DECODE_DOT_UNROLL", "0", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_DECODE_DOT_LANES", "0", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_Q4K_DECODE_STAGE_XQ", "0", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_Q4K_DECODE_SPLIT_GATE_UP", "0", 1) == 0);
    /* Test-only opt-in for the repaired split geometry.  Production launchers
     * still keep both selectors explicit and default-off; this lets the real
     * GGUF oracle compare rows32/64 in the staged split arm rather than the
     * independent fused kernel. */
    if (decode_rows && std::getenv("DS4_TEST_Q4K_DECODE_SPLIT") &&
        std::strcmp(std::getenv("DS4_TEST_Q4K_DECODE_SPLIT"), "1") == 0) {
        REQUIRE(setenv("DS4_ROCM_Q4K_DECODE_STAGE_XQ", "1", 1) == 0);
        REQUIRE(setenv("DS4_ROCM_Q4K_DECODE_SPLIT_GATE_UP", "1", 1) == 0);
    }
    REQUIRE(setenv("DS4_ROCM_TP_SKIP_UNOWNED", "1", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_TP_PREFILL_SKIP_UNOWNED", "1", 1) == 0);
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    REQUIRE(ds4_gpu_init_multi(&config));
    REQUIRE(ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map));
    REQUIRE(ds4_gpu_set_model_map(gguf.map, gguf.size));
    ds4_gpu_set_tp_runtime_features(0, DS4_TP_FEATURE_Q4K_WMMA |
                                      DS4_TP_FEATURE_Q4K_KSHARD);

    // out, gate, up, mid, down, selected, routing weights, input.
    const uint64_t strides[] = {width*4u, used*mid_width*4u,
        used*mid_width*4u, used*mid_width*4u, used*width*4u,
        used*4u, used*4u, width*4u};
    ds4_gpu_tensor *t[8];
    for (unsigned i=0; i<8; ++i) {
        t[i] = ds4_gpu_tensor_alloc(rows*strides[i]);
        REQUIRE(t[i]);
    }
    std::vector<float> x(rows*width), weights(rows*used);
    std::vector<int32_t> selected(rows*used);
    for (size_t i=0; i<x.size(); ++i)
        x[i] = float(std::sin(double(i)*0.013+fixture)*0.17 +
                     std::cos(double(i)*0.031+fixture*2)*0.11);
    for (unsigned r=0; r<rows; ++r) for (unsigned s=0; s<used; ++s) {
        selected[r*used+s] = s<7 ? int(s) : 7+int(r%128);
        if ((cold_lds5 || grouped) && s == 7u) {
            // Each M256 domain includes experts used 1,2,...,7 times,
            // exercising every cold count and both sides of threshold6.
            unsigned within = (r % tile) % 28u, group = 0u;
            while (within >= group + 1u) within -= ++group;
            selected[r*used+s] = int(7u + ((r % tile) / 28u) * 7u + group);
            if (cold_coalesce) {
                // Shared cold counts1..5 span domains (including totals>8
                // and>16). Expert7 is cold elsewhere but hot in domain2;
                // hot routing and its original WMMA row positions must stay.
                if (r / tile == 2u && group == 6u)
                    selected[r*used+s] = 7;
            } else if (grouped) {
                // Disjoint second domain, a shared expert hot only in the
                // third, and a fourth domain with every expert cold once.
                const unsigned domain = r / tile;
                if (domain == 1u) selected[r*used+s] += 128;
                else if (domain == 2u) selected[r*used+s] = 7;
                else if (domain == 3u) selected[r*used+s] = int(7u + r % tile);
            }
        }
        weights[r*used+s] = float(1+(r+s)%7)/16.0f;
        if (grouped && s == 6u && r % 17u == 0u) {
            // Exercise omitted TP routes beside the ordinary live slots.
            selected[r*used+s] = -1;
            weights[r*used+s] = 0.0f;
        }
        if (decode_rows) {
            selected[r*used+s] = int((s*37u + fixture*41u) % experts);
            if (fixture == 1u && s == 6u) {
                selected[r*used+s] = -1;
                weights[r*used+s] = 0.0f;
            }
            if (fixture == 2u && s == 7u) weights[r*used+s] = 0.0f;
        }
        if (schedule && fixture == 3u) {
            selected[r*used+s] = -1;
            weights[r*used+s] = 0.0f;
        }
        if (schedule && fixture == 4u)
            selected[r*used+s] = int((r*used+s+(r/tile)*7u) % experts);
        if (schedule && fixture == 5u && s == 7u) {
            const unsigned within = r % tile;
            selected[r*used+s] = within < 17u ? 7 : within < 33u ? 8 :
                                 int(9u + (within-33u) / 2u);
        }
    }
    if (schedule) {
        unsigned nonempty=0, hot=0, tiles=0, hot_tiles=0;
        for (unsigned first=0; first<rows; first+=tile) {
            unsigned counts[experts] = {};
            for (unsigned r=first; r<rows && r<first+tile; ++r)
                for (unsigned s=0; s<used; ++s)
                    if (selected[r*used+s] >= 0)
                        ++counts[unsigned(selected[r*used+s])];
            for (unsigned count:counts) {
                nonempty += count > 0u;
                hot += count >= 6u;
                tiles += (count+15u)/16u;
                if (count >= 6u) hot_tiles += (count+15u)/16u;
            }
        }
        std::printf("synthetic_routes fixture=%u rows=%u nonempty=%u hot=%u tiles=%u hot_tiles=%u capacity=%u\n",
                    fixture,rows,nonempty,hot,tiles,hot_tiles,
                    (rows*used+15u)/16u+experts*((rows+tile-1u)/tile));
    }
    REQUIRE(ds4_gpu_tensor_write(t[7],0,x.data(),rows*strides[7]));
    REQUIRE(ds4_gpu_tensor_write(t[5],0,selected.data(),rows*strides[5]));
    REQUIRE(ds4_gpu_tensor_write(t[6],0,weights.data(),rows*strides[6]));
    std::vector<float> reference(rows*width), actual(rows*width);
    // The production path suppresses gate/up stores; mid is the exposed
    // pre-quantization state, so compare it before down-projection can hide
    // a difference through activation quantization.
    std::vector<float> mid_reference(decode_rows ? used*mid_width : 0u);
    std::vector<float> mid_actual(mid_reference.size());
    // gate aliases packed Q8_K mid storage by the end of the call. Its
    // bytes are no longer FP32 gate values; up and pre-quantization mid
    // remain inspectable. Poison gate too to expose omitted gate writes.
    std::vector<float> stage_reference[2], stage_actual;
    if (cold_coalesce) {
        for (auto &v : stage_reference) v.resize(rows*used*mid_width);
        stage_actual.resize(rows*used*mid_width);
    }
    for (unsigned rank=0; rank<2; ++rank) {
        for (uint64_t offset : {go, uo}) {
            REQUIRE(ds4_gpu_q4k_packed_slice_declare(gguf.map,gguf.size,
                offset,experts,2048,gate_row,rank*mid_width,mid_width,
                0,gate_row,DS4_GPU_Q4K_PACKED_ROW_RANGE));
            REQUIRE(ds4_gpu_q4k_packed_slice_load(gguf.map,offset,
                rank*mid_width,mid_width,0,gate_row));
        }
        REQUIRE(ds4_gpu_q4k_packed_slice_declare(gguf.map,gguf.size,
            down_offset,experts,width,down_row,0,width,rank*down_row/2,
            down_row/2,DS4_GPU_Q4K_PACKED_K_RANGE));
        REQUIRE(ds4_gpu_q4k_packed_slice_load(gguf.map,down_offset,
            0,width,rank*down_row/2,down_row/2));
        auto call = [&](ds4_gpu_tensor **v, unsigned count) {
            if (decode_rows) {
                REQUIRE(count == 1u);
                return ds4_gpu_routed_moe_one_packed_q4k_tensor(
                    v[0],v[1],v[2],v[3],v[4],gguf.map,gguf.size,go,uo,
                    down_offset,experts,gate_row,down_row,rank*mid_width,
                    mid_width,rank*down_row/2,down_row/2,v[5],v[6],used,
                    fixture == 2u ? 0.0f : 10.0f,v[7],nullptr,3) != 0;
            }
            bool half_mid = true;
            const int ok = ds4_gpu_routed_moe_batch_packed_q4k_tensor(
                v[0],v[1],v[2],v[3],v[4],gguf.map,gguf.size,go,uo,
                down_offset,experts,gate_row,down_row,rank*mid_width,
                mid_width,rank*down_row/2,down_row/2,v[5],v[6],used,
                cold_coalesce && fixture == 2u ? 0.0f : 10.0f,v[7],3,count,&half_mid);
            return ok && !half_mid;
        };
        REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_COLD_LDS5", cold_padding ? "1" : "0", 1) == 0);
        REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_COLD_LDS5_PAD", "0", 1) == 0);
        REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_PREFILL_GROUPED", "0", 1) == 0);
        REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_PREFILL_SCHEDULE", "0", 1) == 0);
        REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_PREFILL_WIDE128", "0", 1) == 0);
        REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_GROUPED_COLD_COALESCE", "0", 1) == 0);
        REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_GROUPED_COLD_I8", "0", 1) == 0);
        REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_DECODE_GATE_ROWS", "0", 1) == 0);
        REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_DECODE_DOT_UNROLL", "0", 1) == 0);
        REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_DECODE_DOT_LANES", "0", 1) == 0);
        REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_PREFILL_PARTITION", "0", 1) == 0);
        for (unsigned first=0; first<rows; first+=tile) {
            const unsigned count = rows-first < tile ? rows-first : tile;
            ds4_gpu_tensor *v[8];
            for (unsigned i=0; i<8; ++i) {
                v[i] = ds4_gpu_tensor_view(t[i],first*strides[i],count*strides[i]);
                REQUIRE(v[i]);
            }
            REQUIRE(call(v,count));
            for (auto *view:v) ds4_gpu_tensor_free(view);
        }
        REQUIRE(ds4_gpu_tensor_read(t[0],0,reference.data(),rows*strides[0]));
        if (cold_coalesce) {
            REQUIRE(ds4_gpu_tensor_fill_f32(t[1],NAN,rows*used*mid_width));
            for (unsigned stage=0; stage<2u; ++stage) {
                REQUIRE(ds4_gpu_tensor_read(t[stage+2],0,stage_reference[stage].data(),rows*strides[stage+2]));
                REQUIRE(ds4_gpu_tensor_fill_f32(t[stage+2],NAN,rows*used*mid_width));
            }
        }
        if (decode_rows) {
            REQUIRE(ds4_gpu_tensor_read(t[3],0,mid_reference.data(),strides[3]));
            REQUIRE(ds4_gpu_tensor_fill_f32(t[3],NAN,used*mid_width));
        }
        REQUIRE(ds4_gpu_tensor_fill_f32(t[0],NAN,rows*width));
        REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_PREFILL_PARTITION", "256", 1) == 0);
        REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_COLD_LDS5", cold_lds5 ? "1" : "0", 1) == 0);
        REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_COLD_LDS5_PAD", cold_padding ? "1" : "0", 1) == 0);
        REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_PREFILL_GROUPED", grouped ? "1" : "0", 1) == 0);
        REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_GROUPED_COLD_COALESCE", cold_coalesce ? "1" : "0", 1) == 0);
        REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_GROUPED_COLD_I8", cold_i8 ? "1" : "0", 1) == 0);
        if (decode_rows) REQUIRE(setenv(timing_selector,candidate_mode,1) == 0);
        // Establish both the grouped control and the scheduling candidate
        // against the independent M256 oracle, including exposed up/mid.
        for (unsigned schedule_arm=0; schedule_arm<(schedule ? 2u : 1u); ++schedule_arm) {
        if (schedule) {
            REQUIRE(setenv(timing_selector,schedule_arm ? candidate_mode : "0",1) == 0);
            REQUIRE(ds4_gpu_tensor_fill_f32(t[0],NAN,rows*width));
            for (unsigned stage=1; stage<=3; ++stage)
                REQUIRE(ds4_gpu_tensor_fill_f32(t[stage],NAN,rows*used*mid_width));
            std::printf("schedule oracle selector=%s mode=%s workers=%s fixture=%u\n",timing_selector,
                schedule_arm ? candidate_mode : "0",
                std::getenv("DS4_ROCM_GLM5_Q4K_PREFILL_WORKERS") ?
                std::getenv("DS4_ROCM_GLM5_Q4K_PREFILL_WORKERS") : "120",fixture);
        }
        REQUIRE(call(t,rows));
        REQUIRE(ds4_gpu_tensor_read(t[0],0,actual.data(),rows*strides[0]));
        size_t different=0;
        double max_abs=0;
        for (size_t i=0; i<actual.size(); ++i) {
            REQUIRE(std::isfinite(reference[i]) && std::isfinite(actual[i]));
            different += std::memcmp(&reference[i],&actual[i],sizeof(float)) != 0;
            max_abs = std::fmax(max_abs,std::fabs(double(reference[i])-actual[i]));
        }
        std::printf("rank=%u rows=%u values=%zu different=%zu max_abs=%.9g\n",
                    rank,rows,actual.size(),different,max_abs);
        std::fflush(stdout);
        REQUIRE(different == 0);
        if (cold_coalesce) {
            for (unsigned stage=0; stage<2u; ++stage) {
                REQUIRE(ds4_gpu_tensor_read(t[stage+2],0,stage_actual.data(),rows*strides[stage+2]));
                size_t live_values=0;
                for (size_t i=0; i<stage_actual.size(); ++i) {
                    const size_t pair = i/mid_width;
                    if (selected[pair] < 0 || weights[pair] == 0.0f) continue;
                    REQUIRE(std::isfinite(stage_reference[stage][i]) && std::isfinite(stage_actual[i]));
                    REQUIRE(std::memcmp(&stage_reference[stage][i],&stage_actual[i],sizeof(float)) == 0);
                    ++live_values;
                }
                std::printf("rank=%u stage=%u live_values=%zu bit_exact=1\n",rank,stage,live_values);
            }
        }
        if (decode_rows) {
            REQUIRE(ds4_gpu_tensor_read(t[3],0,mid_actual.data(),strides[3]));
            for (size_t i=0; i<mid_actual.size(); ++i) {
                REQUIRE(std::isfinite(mid_reference[i]) && std::isfinite(mid_actual[i]));
                REQUIRE(std::memcmp(&mid_reference[i],&mid_actual[i],sizeof(float)) == 0);
            }
            std::printf("rank=%u pre-quantization mid values=%zu bit_exact=1\n",
                        rank,mid_actual.size());
        }
        }
        if (cold_lds5 || grouped || decode_rows) {
            // Warm both full-MoE arms before timing. These are GPU-event
            // microbenchmarks with synthetic routes, not model throughput.
            hipEvent_t begin, end;
            REQUIRE(hipEventCreate(&begin) == hipSuccess);
            REQUIRE(hipEventCreate(&end) == hipSuccess);
            for (const char *mode : {"0", candidate_mode}) {
                REQUIRE(setenv(timing_selector, mode, 1) == 0);
                REQUIRE(call(t,rows));
            }
            REQUIRE(ds4_gpu_synchronize());
            for (unsigned pair = 0; pair < 3u; ++pair) {
                for (unsigned arm = 0; arm < 2u; ++arm) {
                    const unsigned mode = arm ^ (pair & 1u);
                    REQUIRE(setenv(timing_selector, mode ? candidate_mode : "0", 1) == 0);
                    REQUIRE(hipEventRecord(begin, nullptr) == hipSuccess);
                    for (unsigned repeat = 0; repeat < 5u; ++repeat)
                        REQUIRE(call(t,rows));
                    REQUIRE(hipEventRecord(end, nullptr) == hipSuccess);
                    REQUIRE(hipEventSynchronize(end) == hipSuccess);
                    float ms = 0.0f;
                    REQUIRE(hipEventElapsedTime(&ms, begin, end) == hipSuccess);
                    REQUIRE(std::isfinite(ms) && ms > 0.0f);
                    std::printf("microbench rank=%u rows=%u pair=%u %s=%u moe_ms=%.6f\n",
                                rank, rows, pair, wide ? "wide128" : schedule ? "schedule_candidate" : decode_rows ? "decode_rows" : cold_i8 ? "cold_i8" : cold_coalesce ? "cold_coalesce" : grouped ? "grouped" : cold_padding ? "pad" : "lds5", mode, ms / 5.0f);
                }
            }
            REQUIRE(hipEventDestroy(begin) == hipSuccess);
            REQUIRE(hipEventDestroy(end) == hipSuccess);
        }
        // Invalid outer capacity must be rejected before writing earlier tiles.
        REQUIRE(ds4_gpu_tensor_fill_f32(t[0],NAN,rows*width));
        ds4_gpu_tensor *short_out = ds4_gpu_tensor_view(
            t[0],0,rows*strides[0]-sizeof(float));
        REQUIRE(short_out);
        ds4_gpu_tensor *invalid[8];
        std::memcpy(invalid,t,sizeof(t));
        invalid[0] = short_out;
        REQUIRE(!call(invalid,rows));
        ds4_gpu_tensor_free(short_out);
        REQUIRE(ds4_gpu_tensor_read(t[0],0,actual.data(),rows*strides[0]));
        for (float value:actual) REQUIRE(std::isnan(value));
        if (decode_rows) {
            for (const char *invalid_mode : {"invalid", "16", "33"}) {
                REQUIRE(setenv(timing_selector,invalid_mode,1) == 0);
                REQUIRE(!call(t,rows));
                REQUIRE(ds4_gpu_tensor_read(t[0],0,actual.data(),rows*strides[0]));
                for (float value:actual) REQUIRE(std::isnan(value));
            }
            REQUIRE(setenv(timing_selector,"0",1) == 0);
            if (dot_unroll || dot_lanes) {
                REQUIRE(setenv(timing_selector,candidate_mode,1) == 0);
                REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_DECODE_GATE_ROWS","32",1) == 0);
                REQUIRE(!call(t,rows));
                REQUIRE(ds4_gpu_tensor_read(t[0],0,actual.data(),rows*strides[0]));
                for (float value:actual) REQUIRE(std::isnan(value));
                REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_DECODE_GATE_ROWS","0",1) == 0);
                if (dot_lanes) {
                    REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_DECODE_DOT_UNROLL","2",1) == 0);
                    REQUIRE(!call(t,rows));
                    REQUIRE(ds4_gpu_tensor_read(t[0],0,actual.data(),rows*strides[0]));
                    for (float value:actual) REQUIRE(std::isnan(value));
                    REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_DECODE_DOT_UNROLL","0",1) == 0);
                }
                REQUIRE(setenv(timing_selector,"0",1) == 0);
            }
        }
        if (cold_lds5) {
            REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_COLD_LDS5_PAD", "invalid", 1) == 0);
            REQUIRE(!call(t,rows));
            REQUIRE(ds4_gpu_tensor_read(t[0],0,actual.data(),rows*strides[0]));
            for (float value:actual) REQUIRE(std::isnan(value));
            REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_COLD_LDS5", "0", 1) == 0);
            REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_COLD_LDS5_PAD", "1", 1) == 0);
            REQUIRE(!call(t,rows));
            REQUIRE(ds4_gpu_tensor_read(t[0],0,actual.data(),rows*strides[0]));
            for (float value:actual) REQUIRE(std::isnan(value));
            REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_COLD_LDS5_PAD", "0", 1) == 0);
            REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_COLD_LDS5", "invalid", 1) == 0);
            REQUIRE(!call(t,rows));
            REQUIRE(ds4_gpu_tensor_read(t[0],0,actual.data(),rows*strides[0]));
            for (float value:actual) REQUIRE(std::isnan(value));
            REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_COLD_LDS5", "0", 1) == 0);
        }
        if (schedule) {
            for (const char *invalid_mode : {"invalid", "3", ""}) {
                REQUIRE(setenv(timing_selector,invalid_mode,1) == 0);
                REQUIRE(!call(t,rows));
                REQUIRE(ds4_gpu_tensor_read(t[0],0,actual.data(),rows*strides[0]));
                for (float value:actual) REQUIRE(std::isnan(value));
            }
            REQUIRE(setenv(timing_selector,candidate_mode,1) == 0);
            const char *worker_env = std::getenv("DS4_ROCM_GLM5_Q4K_PREFILL_WORKERS");
            const unsigned saved_workers = worker_env ? unsigned(std::atoi(worker_env)) : 120u;
            if (!wide) {
            for (const char *invalid_workers : {"0", "-1", "121", "invalid", ""}) {
                REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_PREFILL_WORKERS",invalid_workers,1) == 0);
                REQUIRE(!call(t,rows));
                REQUIRE(ds4_gpu_tensor_read(t[0],0,actual.data(),rows*strides[0]));
                for (float value:actual) REQUIRE(std::isnan(value));
            }
            REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_PREFILL_WORKERS",saved_workers == 240u ? "240" : "120",1) == 0);
            } else {
                REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_PREFILL_SCHEDULE","1",1) == 0);
                REQUIRE(!call(t,rows));
                REQUIRE(ds4_gpu_tensor_read(t[0],0,actual.data(),rows*strides[0]));
                for (float value:actual) REQUIRE(std::isnan(value));
                REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_PREFILL_SCHEDULE","0",1) == 0);
                REQUIRE(setenv(timing_selector,"2",1) == 0);
                REQUIRE(!call(t,rows));
                REQUIRE(setenv(timing_selector,"1",1) == 0);
            }
            REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_PREFILL_GROUPED","0",1) == 0);
            REQUIRE(!call(t,rows));
            REQUIRE(ds4_gpu_tensor_read(t[0],0,actual.data(),rows*strides[0]));
            for (float value:actual) REQUIRE(std::isnan(value));
            REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_PREFILL_GROUPED","1",1) == 0);
            REQUIRE(setenv(timing_selector,"0",1) == 0);
        }
        if (cold_coalesce && !schedule) {
            if (cold_i8) {
                REQUIRE(setenv(timing_selector,"1",1) == 0);
                REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_GROUPED_COLD_COALESCE","0",1) == 0);
                REQUIRE(!call(t,rows));
                REQUIRE(ds4_gpu_tensor_read(t[0],0,actual.data(),rows*strides[0]));
                for (float value:actual) REQUIRE(std::isnan(value));
                REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_GROUPED_COLD_COALESCE","1",1) == 0);
            }
            if (rows > tile && rows % tile == 0u) {
                // Input Q8_K fits, but no tail remains for cold metadata.
                // Refuse before even the input quantization writes down.
                REQUIRE(setenv(timing_selector,"1",1) == 0);
                REQUIRE(ds4_gpu_tensor_fill_f32(t[4],NAN,16u));
                ds4_gpu_tensor *short_down = ds4_gpu_tensor_view(
                    t[4],0,uint64_t(rows)*16u*292u);
                REQUIRE(short_down);
                std::memcpy(invalid,t,sizeof(t));
                invalid[4] = short_down;
                REQUIRE(!call(invalid,rows));
                ds4_gpu_tensor_free(short_down);
                float untouched[16];
                REQUIRE(ds4_gpu_tensor_read(t[4],0,untouched,sizeof(untouched)));
                for (float value:untouched) REQUIRE(std::isnan(value));
            }
            for (const char *invalid_mode : {"invalid", "2", ""}) {
                REQUIRE(setenv(timing_selector,invalid_mode,1) == 0);
                REQUIRE(!call(t,rows));
                REQUIRE(ds4_gpu_tensor_read(t[0],0,actual.data(),rows*strides[0]));
                for (float value:actual) REQUIRE(std::isnan(value));
            }
            REQUIRE(setenv(timing_selector,"1",1) == 0);
            REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_PREFILL_GROUPED","0",1) == 0);
            REQUIRE(!call(t,rows));
            REQUIRE(ds4_gpu_tensor_read(t[0],0,actual.data(),rows*strides[0]));
            for (float value:actual) REQUIRE(std::isnan(value));
            REQUIRE(setenv(timing_selector,"0",1) == 0);
            REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_PREFILL_GROUPED","1",1) == 0);
        }
        if (grouped) {
            REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_PREFILL_GROUPED", "invalid", 1) == 0);
            REQUIRE(!call(t,rows));
            REQUIRE(ds4_gpu_tensor_read(t[0],0,actual.data(),rows*strides[0]));
            for (float value:actual) REQUIRE(std::isnan(value));
            REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_PREFILL_GROUPED", "1", 1) == 0);
            if (rows > tile && rows % tile == 0u) {
                REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_COLD_LDS5", "1", 1) == 0);
                REQUIRE(!call(t,rows));
                REQUIRE(ds4_gpu_tensor_read(t[0],0,actual.data(),rows*strides[0]));
                for (float value:actual) REQUIRE(std::isnan(value));
                REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_COLD_LDS5", "0", 1) == 0);
                REQUIRE(setenv("DS4_ROCM_TP_PREFILL_SKIP_UNOWNED", "0", 1) == 0);
                REQUIRE(!call(t,rows));
                REQUIRE(ds4_gpu_tensor_read(t[0],0,actual.data(),rows*strides[0]));
                for (float value:actual) REQUIRE(std::isnan(value));
                REQUIRE(setenv("DS4_ROCM_TP_PREFILL_SKIP_UNOWNED", "1", 1) == 0);
            }
            REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_PREFILL_PARTITION", "0", 1) == 0);
            REQUIRE(!call(t,rows));
            REQUIRE(ds4_gpu_tensor_read(t[0],0,actual.data(),rows*strides[0]));
            for (float value:actual) REQUIRE(std::isnan(value));
            REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_PREFILL_GROUPED", "0", 1) == 0);
        }
        REQUIRE(setenv("DS4_ROCM_GLM5_Q4K_PREFILL_PARTITION", "invalid", 1) == 0);
        if (!decode_rows) REQUIRE(!call(t,rows));
        REQUIRE(ds4_gpu_synchronize());
        ds4_gpu_q4k_packed_slice_release_all();
    }
    for (auto *tensor:t) ds4_gpu_tensor_free(tensor);
    ds4_gpu_cleanup();
    std::puts("PASS packed Q4_K batch equals independent <=M256 groups on both halves");
}
