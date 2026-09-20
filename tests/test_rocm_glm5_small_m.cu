#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
extern "C" {
#include "ds4_tp.h"
}
#include "glm5_gguf_test.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}
#define REQUIRE(x) do { if (!(x)) { \
    std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); std::exit(1); \
} } while (0)

static void stream_probe(const Glm5TestGGUF &gguf) {
    struct Projection { uint64_t offset; uint32_t k,n; };
    std::vector<Projection> projections;
    uint64_t weight_bytes=0;
    for (unsigned layer=0;layer<45u;++layer) {
        if (layer%4u==3u) continue;
        for (const char *role : {"q","k","v","output"}) {
            const bool output=std::strcmp(role,"output")==0;
            const uint32_t k=output?8192u:4096u, n=output?2048u:4096u;
            char name[80];
            std::snprintf(name,sizeof(name),"blk.%u.kda_%s.weight",layer,role);
            uint64_t offset;
            REQUIRE(gguf.tensor(name,{k,2u*n},30u,offset));
            projections.push_back({offset,k,n});
            weight_bytes+=(uint64_t)k*n*2u;
        }
    }
    REQUIRE(projections.size()==136u);
    std::printf("STREAM_SCOPE projections=%zu weight_bytes_per_rank=%llu\n",
        projections.size(),(unsigned long long)weight_bytes);
    for (unsigned rank=0;rank<2u;++rank) for (unsigned m : glm5_test_verifier_widths()) {
        std::vector<float> host((size_t)m*8192u);
        for (size_t i=0;i<host.size();++i)
            host[i]=(float)((int)((i*193u+(i>>5u)*761u)%997u)-498)/(1001.3f+(float)(i%7u));
        ds4_gpu_tensor *x=ds4_gpu_tensor_alloc(host.size()*4u);
        ds4_gpu_tensor *y=ds4_gpu_tensor_alloc((uint64_t)m*4096u*4u);
        REQUIRE(x && y && ds4_gpu_tensor_write(x,0u,host.data(),host.size()*4u));
        ds4_gpu_tensor *xs[2][8]={}, *ys[2][8]={};
        for (unsigned role=0;role<2u;++role) for (unsigned t=0;t<m;++t) {
            const uint32_t k=role?8192u:4096u,n=role?2048u:4096u;
            xs[role][t]=ds4_gpu_tensor_view(x,(uint64_t)t*k*4u,(uint64_t)k*4u);
            ys[role][t]=ds4_gpu_tensor_view(y,(uint64_t)t*n*4u,(uint64_t)n*4u);
            REQUIRE(xs[role][t] && ys[role][t]);
        }
        auto launch=[&](unsigned arm) {
            if (arm==0u) {
                // Ordinary serial tokens traverse the entire weight stream
                // before returning to a projection for the next token.
                for (unsigned t=0;t<m;++t) for (const auto &p : projections) {
                    const uint64_t offset=p.offset+(uint64_t)rank*p.k*p.n*2u;
                    const unsigned role=p.k==8192u;
                        if (!ds4_gpu_matmul_bf16_tensor(ys[role][t],gguf.map,
                            gguf.size,offset,p.k,p.n,xs[role][t],1u)) return 0;
                }
            } else for (const auto &p : projections) {
                const uint64_t offset=p.offset+(uint64_t)rank*p.k*p.n*2u;
                if (!ds4_gpu_matmul_bf16_tensor(y,gguf.map,gguf.size,
                           offset,p.k,p.n,x,m)) return 0;
            }
            return 1;
        };
        hipEvent_t begin,end;
        REQUIRE(hipEventCreate(&begin)==hipSuccess && hipEventCreate(&end)==hipSuccess);
        std::vector<double> times[3];
        for (unsigned round=0;round<12u;++round) for (unsigned j=0;j<3u;++j) {
            const unsigned arm=(round+j)%3u;
            REQUIRE(setenv("DS4_ROCM_GLM5_BF16_SMALL_M_EXACT",arm==2u?"1":"0",1)==0);
            REQUIRE(hipEventRecord(begin,nullptr)==hipSuccess);
            REQUIRE(launch(arm));
            REQUIRE(hipEventRecord(end,nullptr)==hipSuccess && hipEventSynchronize(end)==hipSuccess);
            float ms=0;
            REQUIRE(hipEventElapsedTime(&ms,begin,end)==hipSuccess && std::isfinite(ms) && ms>0);
            if (round>=3u) {
                times[arm].push_back(ms);
                std::printf("SMALL_M_STREAM_SAMPLE rank=%u m=%u arm=%u round=%u ms=%.6f\n",rank,m,arm,round-3u,ms);
            }
        }
        for (unsigned arm=0;arm<3u;++arm) {
            std::sort(times[arm].begin(),times[arm].end());
            std::printf("SMALL_M_STREAM_MEDIAN rank=%u m=%u arm=%u ms=%.6f\n",rank,m,arm,times[arm][4]);
        }
        REQUIRE(hipEventDestroy(begin)==hipSuccess && hipEventDestroy(end)==hipSuccess);
        for (unsigned role=0;role<2u;++role) for (unsigned t=0;t<m;++t) {
            ds4_gpu_tensor_free(xs[role][t]); ds4_gpu_tensor_free(ys[role][t]);
        }
        ds4_gpu_tensor_free(y); ds4_gpu_tensor_free(x);
    }
}

int main(int argc, char **argv) {
    const bool baseline = argc == 2 && std::strcmp(argv[1],"--baseline") == 0;
    const bool stream = argc == 2 && std::strcmp(argv[1],"--stream") == 0;
    REQUIRE(argc == 1 || baseline || stream);
    const char *model = std::getenv("DS4_GLM5_MODEL");
    REQUIRE(model);
    Glm5TestGGUF gguf;
    REQUIRE(gguf.open_file(model));
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    REQUIRE(ds4_gpu_init_multi(&config));
    REQUIRE(ds4_gpu_set_model_fd_for_map(gguf.fd,gguf.map));
    REQUIRE(ds4_gpu_set_model_map(gguf.map,gguf.size));
    uint64_t exact_values = 0;
    for (unsigned layer : {0u,1u,44u}) for (const char *role : {
            "q","k","v","output","f_a","g_a","beta","f_b","g_b","head"}) {
        if (baseline && (layer != 0u || (std::strcmp(role,"q") && std::strcmp(role,"output")))) continue;
        const bool head = std::strcmp(role,"head") == 0;
        if (head && layer != 0u) continue;
        const bool output = std::strcmp(role,"output") == 0;
        const bool lowrank = std::strcmp(role,"f_b")==0 || std::strcmp(role,"g_b")==0;
        const bool shared = std::strcmp(role,"f_a")==0 || std::strcmp(role,"g_a")==0;
        const bool beta = std::strcmp(role,"beta")==0;
        const uint32_t k = output ? 8192u : lowrank ? 128u : 4096u;
        const uint32_t n = head ? 154880u : output ? 2048u : shared ? 128u : beta ? 32u : 4096u;
        char name[80];
        if (head) std::snprintf(name,sizeof(name),"output.weight");
        else std::snprintf(name,sizeof(name),"blk.%u.kda_%s.weight",layer,role);
        uint64_t weight;
        REQUIRE(gguf.tensor(name,{k,shared || head ? n : 2u*n},30u,weight));
        for (unsigned rank=0;rank<2u;++rank) for (unsigned m : glm5_test_verifier_widths()) {
            std::vector<float> host_x((size_t)m*k), ref((size_t)m*n), got(ref.size()+16u);
            for (size_t i=0;i<host_x.size();++i)
                host_x[i] = (float)((int)((i*193u+(i/k)*761u+layer*47u)%997u)-498) /
                    (1001.3f+(float)(i%7u));
            ds4_gpu_tensor *x=ds4_gpu_tensor_alloc(host_x.size()*sizeof(float));
            ds4_gpu_tensor *storage=ds4_gpu_tensor_alloc(got.size()*sizeof(float));
            ds4_gpu_tensor *y=ds4_gpu_tensor_view(storage,0u,ref.size()*sizeof(float));
            REQUIRE(x && storage && y);
            REQUIRE(ds4_gpu_tensor_write(x,0u,host_x.data(),host_x.size()*sizeof(float)));
            std::vector<ds4_gpu_tensor *> xs(m),ys(m);
            for (unsigned t=0;t<m;++t) {
                xs[t]=ds4_gpu_tensor_view(x,(uint64_t)t*k*4u,(uint64_t)k*4u);
                ys[t]=ds4_gpu_tensor_view(y,(uint64_t)t*n*4u,(uint64_t)n*4u);
                REQUIRE(xs[t] && ys[t]);
            }
            const uint64_t offset=weight+(shared || head ? 0u : (uint64_t)rank*k*n*2u);
            auto launch=[&](unsigned arm) {
                if (arm == 0u) {
                    for (unsigned t=0;t<m;++t)
                        if (!ds4_gpu_matmul_bf16_tensor(ys[t],gguf.map,gguf.size,
                                offset,k,n,xs[t],1u)) return 0;
                    return 1;
                }
                return ds4_gpu_matmul_bf16_tensor(y,gguf.map,gguf.size,offset,k,n,x,m);
            };
            const unsigned arms=baseline ? 2u : 3u;
            for (unsigned arm=0;arm<arms;++arm) {
                REQUIRE(setenv("DS4_ROCM_GLM5_BF16_SMALL_M_EXACT",arm==2u?"1":"0",1)==0);
                REQUIRE(ds4_gpu_tensor_fill_f32(storage,12345.0f,got.size()));
                REQUIRE(launch(arm) && ds4_gpu_synchronize());
                REQUIRE(ds4_gpu_tensor_read(storage,0u,got.data(),got.size()*sizeof(float)));
                uint64_t mismatches=0;
                double max_abs=0;
                for (size_t i=0;i<ref.size();++i) {
                    REQUIRE(std::isfinite(got[i]));
                    if (arm==0u) ref[i]=got[i];
                    else {
                        mismatches += std::memcmp(&ref[i],&got[i],sizeof(float)) != 0;
                        max_abs=std::max(max_abs,std::fabs((double)ref[i]-got[i]));
                    }
                }
                for (size_t i=ref.size();i<got.size();++i) REQUIRE(got[i]==12345.0f);
                if (arm) std::printf("SMALL_M_NUMERIC layer=%u role=%s rank=%u m=%u arm=%u different=%llu max_abs=%.9g\n",
                    layer,role,rank,m,arm,(unsigned long long)mismatches,max_abs);
                if (arm==2u) { REQUIRE(mismatches==0u); exact_values+=ref.size(); }
            }
            if (!baseline && layer==0u && !output && rank==0u && m==glm5_test_verifier_widths().front()) {
                REQUIRE(setenv("DS4_ROCM_GLM5_BF16_SMALL_M_EXACT","invalid",1)==0);
                REQUIRE(!launch(2u));
                REQUIRE(setenv("DS4_ROCM_GLM5_BF16_SMALL_M_EXACT","1",1)==0);
                const char *prefetch_env=std::getenv("DS4_ROCM_GLM5_BF16_SMALL_M_PREFETCH");
                const std::string saved_prefetch=prefetch_env?prefetch_env:"0";
                for (const char *bad : {"", "1", "08", "-8", "invalid"}) {
                    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_SMALL_M_PREFETCH",bad,1)==0);
                    REQUIRE(ds4_gpu_tensor_fill_f32(storage,12345.0f,got.size()));
                    REQUIRE(!launch(2u));
                    REQUIRE(ds4_gpu_tensor_read(storage,0u,got.data(),got.size()*sizeof(float)));
                    for (float value:got) REQUIRE(value==12345.0f);
                }
                REQUIRE(setenv("DS4_ROCM_GLM5_BF16_SMALL_M_PREFETCH",saved_prefetch.c_str(),1)==0);
                const char *panel_env=std::getenv("DS4_ROCM_GLM5_BF16_SMALL_M_PANEL");
                const std::string saved_panel=panel_env?panel_env:"1024";
                for (const char *bad : {"", "0", "256", "0512", "invalid"}) {
                    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_SMALL_M_PANEL",bad,1)==0);
                    REQUIRE(ds4_gpu_tensor_fill_f32(storage,12345.0f,got.size()));
                    REQUIRE(!launch(2u));
                    REQUIRE(ds4_gpu_tensor_read(storage,0u,got.data(),got.size()*sizeof(float)));
                    for (float value:got) REQUIRE(value==12345.0f);
                }
                REQUIRE(setenv("DS4_ROCM_GLM5_BF16_SMALL_M_PANEL","512",1)==0);
                REQUIRE(setenv("DS4_ROCM_GLM5_BF16_SMALL_M_PREFETCH","8",1)==0);
                REQUIRE(!launch(2u));
                REQUIRE(setenv("DS4_ROCM_GLM5_BF16_SMALL_M_PANEL",saved_panel.c_str(),1)==0);
                REQUIRE(setenv("DS4_ROCM_GLM5_BF16_SMALL_M_PREFETCH",saved_prefetch.c_str(),1)==0);
                REQUIRE(setenv("DS4_ROCM_DISABLE_BF16_SHAREDX","1",1)==0);
                REQUIRE(!launch(2u));
                REQUIRE(unsetenv("DS4_ROCM_DISABLE_BF16_SHAREDX")==0);
                REQUIRE(setenv("DS4_ROCM_BF16_FULL_SPLIT_ORDER","1",1)==0);
                REQUIRE(!launch(2u));
                REQUIRE(unsetenv("DS4_ROCM_BF16_FULL_SPLIT_ORDER")==0);
                ds4_gpu_tensor *short_y=ds4_gpu_tensor_view(y,0u,ref.size()*4u-4u);
                REQUIRE(short_y && !ds4_gpu_matmul_bf16_tensor(short_y,gguf.map,
                    gguf.size,offset,k,n,x,m));
                ds4_gpu_tensor_free(short_y);
                REQUIRE(!ds4_gpu_matmul_bf16_tensor(y,gguf.map,gguf.size,
                    gguf.size-2u,k,n,x,m));
                REQUIRE(launch(0u) && ds4_gpu_synchronize());
                REQUIRE(ds4_gpu_tensor_read(y,0u,got.data(),ref.size()*4u));
                REQUIRE(std::memcmp(ref.data(),got.data(),ref.size()*4u)==0);
                std::puts("PASS small-M guards and unchanged M1 under opt-in");
            }
            if (!baseline && layer==0u && !output && rank==0u && m==glm5_test_verifier_widths().back()) {
                for (unsigned tail : {3u,5u,7u}) {
                    if (tail>m) continue;
                    std::vector<float> control((size_t)tail*n), candidate(control.size());
                    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_SMALL_M_EXACT","0",1)==0);
                    REQUIRE(ds4_gpu_matmul_bf16_tensor(y,gguf.map,gguf.size,offset,k,n,x,tail) && ds4_gpu_synchronize());
                    REQUIRE(ds4_gpu_tensor_read(y,0u,control.data(),control.size()*4u));
                    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_SMALL_M_EXACT","1",1)==0);
                    REQUIRE(ds4_gpu_matmul_bf16_tensor(y,gguf.map,gguf.size,offset,k,n,x,tail) && ds4_gpu_synchronize());
                    REQUIRE(ds4_gpu_tensor_read(y,0u,candidate.data(),candidate.size()*4u));
                    REQUIRE(std::memcmp(control.data(),candidate.data(),control.size()*4u)==0);
                }
                std::puts("PASS fitting unsupported M3/M5/M7 keep incumbent dispatch");
            }
            if (layer==0u && (head || output || std::strcmp(role,"q")==0)) {
                hipEvent_t begin,end;
                REQUIRE(hipEventCreate(&begin)==hipSuccess && hipEventCreate(&end)==hipSuccess);
                std::vector<double> times[3];
                for (unsigned round=0;round<12u;++round) for (unsigned j=0;j<arms;++j) {
                    const unsigned arm=(round+j)%arms;
                    REQUIRE(setenv("DS4_ROCM_GLM5_BF16_SMALL_M_EXACT",arm==2u?"1":"0",1)==0);
                    REQUIRE(hipEventRecord(begin,nullptr)==hipSuccess);
                    for (unsigned repeat=0;repeat<3u;++repeat) REQUIRE(launch(arm));
                    REQUIRE(hipEventRecord(end,nullptr)==hipSuccess && hipEventSynchronize(end)==hipSuccess);
                    float ms=0;
                    REQUIRE(hipEventElapsedTime(&ms,begin,end)==hipSuccess && std::isfinite(ms) && ms>0);
                    if (round>=3u) {
                        times[arm].push_back(ms/3.0);
                        std::printf("SMALL_M_SAMPLE role=%s rank=%u m=%u arm=%u round=%u ms=%.6f\n",role,rank,m,arm,round-3u,ms/3.0);
                    }
                }
                for (unsigned arm=0;arm<arms;++arm) {
                    std::sort(times[arm].begin(),times[arm].end());
                    std::printf("SMALL_M_MEDIAN role=%s rank=%u m=%u arm=%u ms=%.6f\n",role,rank,m,arm,times[arm][4]);
                }
                REQUIRE(hipEventDestroy(begin)==hipSuccess && hipEventDestroy(end)==hipSuccess);
            }
            for (unsigned t=0;t<m;++t) { ds4_gpu_tensor_free(xs[t]); ds4_gpu_tensor_free(ys[t]); }
            ds4_gpu_tensor_free(y); ds4_gpu_tensor_free(storage); ds4_gpu_tensor_free(x);
        }
    }
    std::printf("PASS small-M projection probe baseline=%d candidate_exact_values=%llu\n",baseline,(unsigned long long)exact_values);
    if (stream) stream_probe(gguf);
    ds4_gpu_cleanup();
    return 0;
}
