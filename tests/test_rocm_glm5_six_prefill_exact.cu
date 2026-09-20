// Compare six-pointer prefill with the projections used by layer_begin:
// hi/lo WMMA QKV, but ordinary F32-activation f_a/g_a/beta reductions.
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

static const char *weight_selector =
    "DS4_ROCM_GLM5_BF16_WMMA_COALESCED_WEIGHT";

template <typename Launch>
static void time_weight_modes(Launch launch, const char *role,
                              unsigned rank, unsigned rows) {
    hipEvent_t begin, end;
    REQUIRE(hipEventCreate(&begin) == hipSuccess);
    REQUIRE(hipEventCreate(&end) == hipSuccess);
    for (const char *mode : {"0", "1"}) {
        REQUIRE(setenv(weight_selector,mode,1) == 0);
        REQUIRE(launch() == 1);
    }
    REQUIRE(ds4_gpu_synchronize());
    for (unsigned pair=0; pair<3; ++pair) for (unsigned arm=0; arm<2; ++arm) {
        const unsigned mode = arm ^ (pair & 1u);
        REQUIRE(setenv(weight_selector,mode ? "1" : "0",1) == 0);
        REQUIRE(hipEventRecord(begin,nullptr) == hipSuccess);
        for (unsigned repeat=0; repeat<3; ++repeat) REQUIRE(launch() == 1);
        REQUIRE(hipEventRecord(end,nullptr) == hipSuccess);
        REQUIRE(hipEventSynchronize(end) == hipSuccess);
        float ms = 0;
        REQUIRE(hipEventElapsedTime(&ms,begin,end) == hipSuccess);
        REQUIRE(std::isfinite(ms) && ms > 0);
        std::printf("weight_load role=%s rank=%u rows=%u pair=%u mode=%u ms=%.6f\n",
                    role,rank,rows,pair,mode,ms/3.0f);
    }
    REQUIRE(hipEventDestroy(begin) == hipSuccess);
    REQUIRE(hipEventDestroy(end) == hipSuccess);
}

int main(int argc, char **argv) {
    REQUIRE(argc <= 3);
    const bool coalesced = argc == 3 && std::strcmp(argv[2],"--coalesced") == 0;
    const bool m96 = argc == 3 && std::strcmp(argv[2],"--exact-m96") == 0;
    const bool shared_a = argc == 3 && std::strcmp(argv[2],"--shared-a") == 0;
    const bool fused_shared_a = argc == 3 &&
        std::strcmp(argv[2],"--fused-shared-a") == 0;
    const bool native_qkv = argc == 3 &&
        std::strcmp(argv[2],"--fused-native-qkv") == 0;
    const bool live_native = argc == 3 &&
        std::strcmp(argv[2],"--live-native-qkv") == 0;
    REQUIRE(argc != 3 || coalesced || m96 || shared_a || fused_shared_a ||
            native_qkv || live_native);
    const bool compare_modes = coalesced || m96 || live_native;
    if (m96) weight_selector = "DS4_ROCM_GLM5_BF16_KDA_SIX_EXACT_M96";
    if (live_native) weight_selector = "DS4_ROCM_GLM5_BF16_KDA_SIX_LIVE_NATIVE_QKV";
    const char *skinny = argc >= 2 ? argv[1] : "0";
    REQUIRE(std::strcmp(skinny,"0") == 0 || std::strcmp(skinny,"1") == 0);
    const char *model = std::getenv("DS4_GLM5_MODEL");
    REQUIRE(model);
    Glm5TestGGUF gguf;
    REQUIRE(gguf.open_file(model));
    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_KDA_SIX_PREFILL", "1", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_KDA_SIX_SHARED_A",
                   shared_a ? "1" : "0", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_KDA_SIX_FUSED_SHARED_A",
                   "0", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_KDA_SIX_NATIVE_QKV",
                   "0", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_KDA_SIX_LIVE_NATIVE_QKV",
                   "0", 1) == 0);
    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_WMMA_NATIVE", "0", 1) == 0);
    if (m96) for (const char *name : {"DS4_ROCM_GLM5_BF16_WMMA_COALESCED_WEIGHT",
            "DS4_ROCM_GLM5_BF16_WMMA_WIDE_TILE", "DS4_ROCM_GLM5_BF16_LT_HILO",
            "DS4_ROCM_GLM5_BF16_WMMA_EXACT_M96"})
        REQUIRE(setenv(name,"0",1) == 0);
    REQUIRE(setenv(weight_selector,"0",1) == 0);
    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_SKINNY_EXACT_TOKTILE", skinny, 1) == 0);
    REQUIRE(unsetenv("DS4_ROCM_DISABLE_BF16_BATCH_TOKTILE") == 0);
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    REQUIRE(ds4_gpu_init_multi(&config));
    REQUIRE(ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map));
    REQUIRE(ds4_gpu_set_model_map(gguf.map, gguf.size));
    const char *names[] = {"q", "k", "v", "f_a", "g_a", "beta"};
    const uint32_t full_widths[] = {8192,8192,8192,128,128,64};
    size_t total = 0;
    for (unsigned layer : {0u, 1u, 44u}) {
        uint64_t offsets[6];
        for (unsigned i=0; i<6; ++i) {
            char name[80];
            std::snprintf(name,sizeof(name),"blk.%u.kda_%s.weight",layer,names[i]);
            REQUIRE(gguf.tensor(name,{4096,full_widths[i]},30,offsets[i]));
        }
        // Layouts 0/1 are TP halves; layout 2 is the supported full-head API.
        for (unsigned rank=0; rank<3; ++rank) for (unsigned rows : {256u,1024u,2048u}) {
            if (rank == 2 && rows != 256 && !m96) continue;
            const uint32_t q_width = rank < 2 ? 4096u : 8192u;
            const uint32_t widths[] = {q_width,q_width,q_width,128,128,
                                      rank < 2 ? 32u : 64u};
            uint64_t local[6];
            ds4_gpu_tensor *reference[6], *candidate[6];
            for (unsigned i=0; i<6; ++i) {
                local[i] = offsets[i] + (rank < 2 && (i<3 || i==5) ?
                    uint64_t(rank)*widths[i]*4096u*2u : 0u);
                reference[i] = ds4_gpu_tensor_alloc(uint64_t(rows)*widths[i]*4u);
                candidate[i] = ds4_gpu_tensor_alloc(uint64_t(rows)*widths[i]*4u);
                REQUIRE(reference[i] && candidate[i]);
                REQUIRE(ds4_gpu_tensor_fill_f32(candidate[i],NAN,rows*widths[i]));
            }
            std::vector<float> x(size_t(rows)*4096u);
            for (size_t i=0; i<x.size(); ++i)
                x[i] = float(std::sin(double(i)*0.013+layer)*0.17 +
                             std::cos(double(i)*0.031+rank)*0.11);
            ds4_gpu_tensor *input = ds4_gpu_tensor_alloc(x.size()*4u);
            REQUIRE(input && ds4_gpu_tensor_write(input,0,x.data(),x.size()*4u));
            REQUIRE(setenv(weight_selector,"0",1) == 0);
            if (rank < 2) {
                REQUIRE(ds4_gpu_matmul_bf16_wmma_hilo_qkv_tensor(
                    reference[0],reference[1],reference[2],gguf.map,gguf.size,
                    local[0],local[1],local[2],4096,q_width,input,rows) == 1);
            } else {
                // The incumbent fused-QKV selector supports TP halves only;
                // full heads use the three ordinary hi/lo dispatches.
                for (unsigned i=0; i<3; ++i)
                    REQUIRE(ds4_gpu_matmul_bf16_wmma_hilo_tensor(reference[i],
                        gguf.map,gguf.size,local[i],4096,q_width,input,rows) == 1);
            }
            for (unsigned i=3; i<6; ++i)
                REQUIRE(ds4_gpu_matmul_bf16_tensor(reference[i],gguf.map,
                    gguf.size,local[i],4096,widths[i],input,rows));
            if (live_native) {
                // Independent host rounding by choosing adjacent BF16 values;
                // keep the original-input reference for all skinny gates.
                std::vector<float> rounded(x.size());
                for (size_t i=0; i<x.size(); ++i) {
                    uint32_t bits;
                    std::memcpy(&bits,&x[i],4);
                    uint32_t lo_bits=bits&0xffff0000u, hi_bits=lo_bits+0x10000u;
                    float a,b;
                    std::memcpy(&a,&lo_bits,4);
                    std::memcpy(&b,&hi_bits,4);
                    const double da=std::abs(double(x[i])-a), db=std::abs(double(x[i])-b);
                    rounded[i]=da<db ? a : db<da ? b : ((bits>>16u)&1u) ? b : a;
                }
                auto *rounded_input=ds4_gpu_tensor_alloc(rounded.size()*4u);
                REQUIRE(rounded_input && ds4_gpu_tensor_write(rounded_input,0,
                    rounded.data(),rounded.size()*4u));
                for (unsigned i=0; i<3; ++i)
                    REQUIRE(ds4_gpu_matmul_bf16_wmma_hilo_tensor(reference[i],
                        gguf.map,gguf.size,local[i],4096,q_width,rounded_input,rows)==1);
                REQUIRE(ds4_gpu_synchronize());
                ds4_gpu_tensor_free(rounded_input);
            }
            auto launch = [&]() { return ds4_gpu_matmul_bf16_kda_six_multiptr_tensor(
                candidate[0],candidate[1],candidate[2],candidate[3],
                candidate[4],candidate[5],gguf.map,gguf.size,
                local[0],local[1],local[2],local[3],local[4],local[5],
                4096,q_width,128,widths[5],input,rows); };
            const bool fused_eligible = (fused_shared_a || native_qkv) &&
                rank < 2u &&
                rows >= 1024u;
            REQUIRE(setenv("DS4_ROCM_GLM5_BF16_KDA_SIX_FUSED_SHARED_A",
                           fused_eligible ? "1" : "0", 1) == 0);
            REQUIRE(setenv("DS4_ROCM_GLM5_BF16_KDA_SIX_NATIVE_QKV",
                           native_qkv && fused_eligible ? "1" : "0", 1) == 0);
            REQUIRE(setenv(weight_selector,compare_modes ? "1" : "0",1) == 0);
            REQUIRE(launch() == 1);
            if (fused_eligible) {
                REQUIRE(setenv("DS4_ROCM_GLM5_BF16_KDA_SIX_FUSED_SHARED_A",
                               "invalid", 1) == 0);
                REQUIRE(launch() == 0);
                REQUIRE(setenv("DS4_ROCM_GLM5_BF16_KDA_SIX_FUSED_SHARED_A",
                               "1", 1) == 0);
            }
            // No partial tile may reach the exact kernel or alter its output.
            REQUIRE(ds4_gpu_matmul_bf16_kda_six_multiptr_tensor(
                candidate[0],candidate[1],candidate[2],candidate[3],
                candidate[4],candidate[5],gguf.map,gguf.size,
                local[0],local[1],local[2],local[3],local[4],local[5],
                4096,q_width,128,widths[5],input,rows-1u) == -1);
            if (compare_modes) {
                // Invalid selectors must leave the recorded outputs intact.
                REQUIRE(setenv(weight_selector,"invalid",1) == 0);
                REQUIRE(launch() == 0);
                REQUIRE(setenv(weight_selector,"1",1) == 0);
            }
            if (live_native) {
                for (const char *invalid : {"", "2", "true"}) {
                    REQUIRE(setenv(weight_selector,invalid,1)==0);
                    REQUIRE(launch()==0);
                }
                REQUIRE(setenv(weight_selector,"1",1)==0);
                for (const char *name : {"DS4_ROCM_GLM5_BF16_KDA_SIX_SHARED_A",
                        "DS4_ROCM_GLM5_BF16_KDA_SIX_FUSED_SHARED_A",
                        "DS4_ROCM_GLM5_BF16_KDA_SIX_NATIVE_QKV",
                        "DS4_ROCM_GLM5_BF16_KDA_SIX_EXACT_M96",
                        "DS4_ROCM_GLM5_BF16_WMMA_COALESCED_WEIGHT",
                        "DS4_ROCM_GLM5_BF16_WMMA_WIDE_TILE",
                        "DS4_ROCM_GLM5_BF16_LT_HILO"}) {
                    REQUIRE(setenv(name,"1",1)==0);
                    REQUIRE(launch()==0);
                    REQUIRE(setenv(name,"0",1)==0);
                }
                REQUIRE(setenv("DS4_ROCM_GLM5_BF16_KDA_SIX_PREFILL","0",1)==0);
                REQUIRE(launch()==0);
                REQUIRE(setenv("DS4_ROCM_GLM5_BF16_KDA_SIX_PREFILL","1",1)==0);
                auto *saved=candidate[1];
                candidate[1]=candidate[0];
                REQUIRE(launch()==0);
                candidate[1]=saved;
                auto *short_out=ds4_gpu_tensor_view(candidate[0],0,uint64_t(rows)*q_width*4u-4u);
                REQUIRE(short_out);
                saved=candidate[0]; candidate[0]=short_out;
                REQUIRE(launch()==0);
                candidate[0]=saved;
                ds4_gpu_tensor_free(short_out);
            }
            if (m96) {
                for (const char *invalid : {"", "2"}) {
                    REQUIRE(setenv(weight_selector,invalid,1) == 0);
                    REQUIRE(launch() == 0);
                }
                REQUIRE(setenv(weight_selector,"1",1) == 0);
                for (const char *name : {"DS4_ROCM_GLM5_BF16_WMMA_NATIVE",
                        "DS4_ROCM_GLM5_BF16_WMMA_COALESCED_WEIGHT",
                        "DS4_ROCM_GLM5_BF16_WMMA_WIDE_TILE", "DS4_ROCM_GLM5_BF16_LT_HILO"}) {
                    REQUIRE(setenv(name,"1",1) == 0);
                    REQUIRE(launch() == 0);
                    REQUIRE(setenv(name,"0",1) == 0);
                }
                if (rows == 1024u) for (unsigned i=0; i<3; ++i) {
                    local[i] += 2u;
                    REQUIRE(launch() == 0);
                    local[i] -= 2u;
                }
                auto *saved = candidate[0];
                auto *short_out = ds4_gpu_tensor_view(saved,0,uint64_t(rows)*q_width*4u-4u);
                REQUIRE(short_out);
                candidate[0] = short_out;
                REQUIRE(launch() == 0);
                candidate[0] = saved;
                ds4_gpu_tensor_free(short_out);
                saved = candidate[1];
                candidate[1] = candidate[0];
                REQUIRE(launch() == 0);
                candidate[1] = saved;
                saved = input;
                auto *short_in = ds4_gpu_tensor_view(input,0,x.size()*4u-4u);
                REQUIRE(short_in);
                input = short_in;
                REQUIRE(launch() == 0);
                input = saved;
                ds4_gpu_tensor_free(short_in);
            }
            bool exact = true;
            for (unsigned i=0; i<6; ++i) {
                const size_t count = size_t(rows)*widths[i];
                std::vector<float> a(count), b(count);
                REQUIRE(ds4_gpu_tensor_read(reference[i],0,a.data(),count*4u));
                REQUIRE(ds4_gpu_tensor_read(candidate[i],0,b.data(),count*4u));
                size_t different = 0;
                double max_abs = 0;
                for (size_t j=0; j<count; ++j) {
                    REQUIRE(std::isfinite(a[j]) && std::isfinite(b[j]));
                    different += std::memcmp(&a[j],&b[j],sizeof(float)) != 0;
                    max_abs = std::fmax(max_abs,std::fabs(double(a[j])-b[j]));
                }
                double error2 = 0.0, norm2 = 0.0;
                for (size_t j=0; j<count; ++j) {
                    const double error = double(a[j]) - double(b[j]);
                    error2 += error * error;
                    norm2 += double(a[j]) * double(a[j]);
                }
                const double nrmse = std::sqrt(error2 /
                    std::fmax(norm2, 1e-30));
                std::printf("layer=%u rank=%u rows=%u projection=%s values=%zu different=%zu max_abs=%.9g nrmse=%.9g\n",
                    layer,rank,rows,names[i],count,different,max_abs,nrmse);
                if (native_qkv && fused_eligible && i < 3u) {
                    REQUIRE(std::isfinite(nrmse) && nrmse < 0.01 &&
                            max_abs < 0.1);
                } else {
                    exact = exact && different == 0;
                }
                total += count;
            }
            REQUIRE(exact || (native_qkv && fused_eligible));
            if (shared_a && layer == 0u && rank < 2u) {
                hipEvent_t begin, end;
                REQUIRE(hipEventCreate(&begin) == hipSuccess);
                REQUIRE(hipEventCreate(&end) == hipSuccess);
                float elapsed[2] = {};
                for (unsigned arm = 0u; arm < 2u; ++arm) {
                    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_KDA_SIX_SHARED_A",
                                   arm ? "1" : "0", 1) == 0);
                    REQUIRE(ds4_gpu_synchronize());
                    REQUIRE(hipEventRecord(begin, nullptr) == hipSuccess);
                    for (unsigned repeat = 0u; repeat < 5u; ++repeat)
                        REQUIRE(launch() == 1);
                    REQUIRE(hipEventRecord(end, nullptr) == hipSuccess);
                    REQUIRE(hipEventSynchronize(end) == hipSuccess);
                    REQUIRE(hipEventElapsedTime(&elapsed[arm], begin, end) ==
                            hipSuccess);
                    elapsed[arm] /= 5.0f;
                }
                REQUIRE(setenv("DS4_ROCM_GLM5_BF16_KDA_SIX_SHARED_A", "1", 1) == 0);
                std::printf("timing shared_a layer=%u rank=%u rows=%u baseline_ms=%.6f shared_ms=%.6f speedup=%.4f\n",
                            layer, rank, rows, elapsed[0], elapsed[1],
                            elapsed[0] / elapsed[1]);
                REQUIRE(hipEventDestroy(begin) == hipSuccess);
                REQUIRE(hipEventDestroy(end) == hipSuccess);
            }
            if (fused_eligible && layer == 0u && rank < 2u) {
                hipEvent_t begin, end;
                REQUIRE(hipEventCreate(&begin) == hipSuccess);
                REQUIRE(hipEventCreate(&end) == hipSuccess);
                float elapsed[2] = {};
                for (unsigned arm = 0u; arm < 2u; ++arm) {
                    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_KDA_SIX_FUSED_SHARED_A",
                                   arm ? "1" : "0", 1) == 0);
                    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_KDA_SIX_NATIVE_QKV",
                                   native_qkv && arm ? "1" : "0", 1) == 0);
                    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_KDA_SIX_SHARED_A",
                                   "0", 1) == 0);
                    REQUIRE(ds4_gpu_synchronize());
                    REQUIRE(hipEventRecord(begin, nullptr) == hipSuccess);
                    for (unsigned repeat = 0u; repeat < 5u; ++repeat)
                        REQUIRE(launch() == 1);
                    REQUIRE(hipEventRecord(end, nullptr) == hipSuccess);
                    REQUIRE(hipEventSynchronize(end) == hipSuccess);
                    REQUIRE(hipEventElapsedTime(&elapsed[arm], begin, end) ==
                            hipSuccess);
                    elapsed[arm] /= 5.0f;
                }
                REQUIRE(setenv("DS4_ROCM_GLM5_BF16_KDA_SIX_FUSED_SHARED_A",
                               "1", 1) == 0);
                REQUIRE(setenv("DS4_ROCM_GLM5_BF16_KDA_SIX_NATIVE_QKV",
                               native_qkv ? "1" : "0", 1) == 0);
                std::printf("timing fused_shared_a layer=%u rank=%u rows=%u "
                            "baseline_ms=%.6f fused_ms=%.6f speedup=%.4f\n",
                            layer, rank, rows, elapsed[0], elapsed[1],
                            elapsed[0] / elapsed[1]);
                REQUIRE(hipEventDestroy(begin) == hipSuccess);
                REQUIRE(hipEventDestroy(end) == hipSuccess);
            }
            if (coalesced) {
                REQUIRE(ds4_gpu_matmul_bf16_wmma_hilo_tensor(candidate[0],
                    gguf.map,gguf.size,local[0],4096,q_width,input,rows) == 1);
                const size_t count = size_t(rows)*q_width;
                std::vector<float> a(count), b(count);
                REQUIRE(ds4_gpu_tensor_read(reference[0],0,a.data(),count*4u));
                REQUIRE(ds4_gpu_tensor_read(candidate[0],0,b.data(),count*4u));
                for (size_t i=0; i<count; ++i) {
                    REQUIRE(std::isfinite(a[i]) && std::isfinite(b[i]));
                    REQUIRE(std::memcmp(&a[i],&b[i],sizeof(float)) == 0);
                }
                total += count;
                std::printf("generic_weight_load layer=%u rank=%u rows=%u K=4096 N=%u values=%zu different=0\n",
                            layer,rank,rows,q_width,count);
            }
            if (m96) {
                // Reusing the private activation scratch must not alter the
                // incumbent path after the selector is removed.
                REQUIRE(unsetenv(weight_selector) == 0);
                REQUIRE(launch() == 1);
                for (unsigned i=0; i<6; ++i) {
                    const size_t count = size_t(rows)*widths[i];
                    std::vector<float> a(count), b(count);
                    REQUIRE(ds4_gpu_tensor_read(reference[i],0,a.data(),count*4u));
                    REQUIRE(ds4_gpu_tensor_read(candidate[i],0,b.data(),count*4u));
                    REQUIRE(std::memcmp(a.data(),b.data(),count*4u) == 0);
                }
            }
            if (compare_modes && layer == 0 && rank < 2)
                time_weight_modes(launch,"six",rank,rows);
            for (unsigned i=0; i<6; ++i) {
                ds4_gpu_tensor_free(reference[i]);
                ds4_gpu_tensor_free(candidate[i]);
            }
            ds4_gpu_tensor_free(input);
            std::fflush(stdout);
            REQUIRE(exact || (native_qkv && fused_eligible));
        }
    }
    if (coalesced) {
        // Generic prefill hi/lo admits the full K8192/N4096 output shape.
        // Its load schedule is separate from the six-pointer kernel above.
        for (unsigned layer : {0u,44u}) for (unsigned rows : {256u,1024u}) {
            constexpr unsigned rank = 2u; // Full projection, as in the QKV oracle.
            char name[80];
            std::snprintf(name,sizeof(name),"blk.%u.kda_output.weight",layer);
            uint64_t offset;
            REQUIRE(gguf.tensor(name,{8192,4096},30,offset));
            std::vector<float> x(size_t(rows)*8192u), a(size_t(rows)*4096u), b(a.size());
            for (size_t i=0; i<x.size(); ++i)
                x[i] = float(std::sin(double(i)*0.017+layer)*0.13 +
                             std::cos(double(i)*0.037+rank)*0.19);
            auto *input = ds4_gpu_tensor_alloc(x.size()*4u);
            auto *reference = ds4_gpu_tensor_alloc(a.size()*4u);
            auto *candidate = ds4_gpu_tensor_alloc(a.size()*4u);
            REQUIRE(input && reference && candidate);
            REQUIRE(ds4_gpu_tensor_write(input,0,x.data(),x.size()*4u));
            auto call = [&](ds4_gpu_tensor *out) {
                return ds4_gpu_matmul_bf16_wmma_hilo_tensor(out,gguf.map,
                    gguf.size,offset,8192,4096,input,rows);
            };
            REQUIRE(setenv(weight_selector,"0",1) == 0);
            REQUIRE(call(reference) == 1);
            REQUIRE(setenv(weight_selector,"1",1) == 0);
            REQUIRE(call(candidate) == 1);
            REQUIRE(setenv(weight_selector,"invalid",1) == 0);
            REQUIRE(call(candidate) == 0);
            REQUIRE(setenv(weight_selector,"1",1) == 0);
            REQUIRE(setenv("DS4_ROCM_GLM5_BF16_WMMA_NATIVE","1",1) == 0);
            REQUIRE(call(candidate) == 0);
            REQUIRE(setenv("DS4_ROCM_GLM5_BF16_WMMA_NATIVE","0",1) == 0);
            REQUIRE(ds4_gpu_tensor_read(reference,0,a.data(),a.size()*4u));
            REQUIRE(ds4_gpu_tensor_read(candidate,0,b.data(),b.size()*4u));
            size_t different = 0;
            for (size_t i=0; i<a.size(); ++i) {
                REQUIRE(std::isfinite(a[i]) && std::isfinite(b[i]));
                different += std::memcmp(&a[i],&b[i],sizeof(float)) != 0;
            }
            std::printf("output_weight_load layer=%u rank=%u rows=%u values=%zu different=%zu\n",
                        layer,rank,rows,a.size(),different);
            REQUIRE(different == 0);
            total += a.size();
            if (layer == 0) time_weight_modes([&]() { return call(candidate); },"output",rank,rows);
            ds4_gpu_tensor_free(input);
            ds4_gpu_tensor_free(reference);
            ds4_gpu_tensor_free(candidate);
        }
    }
    REQUIRE(ds4_gpu_synchronize());
    ds4_gpu_cleanup();
    std::printf("PASS six-prefill matches production projections skinny=%s values=%zu\n",
                skinny,total);
}
