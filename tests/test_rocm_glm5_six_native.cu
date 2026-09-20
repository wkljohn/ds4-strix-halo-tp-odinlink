#include "glm5_gguf_test.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>

extern "C" hipError_t glm5_six_native(float *const *,const uint16_t *const *,
    const float *,unsigned,unsigned,unsigned,unsigned);
extern "C" hipError_t glm5_six_rounding(uint32_t *,const float *,uint64_t);
// Compile the unmodified parent geometry TU with this external symbol name.
extern "C" hipError_t glm5_six_incumbent(float *const *,const uint16_t *const *,
    const float *,unsigned,unsigned,unsigned,unsigned,uint32_t *);
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr,"FAIL line=%d: %s\n",__LINE__,#x); std::exit(1); } } while (0)

static float bf16_value(uint16_t bits) {
    const uint32_t word = uint32_t(bits)<<16u;
    float value;
    std::memcpy(&value,&word,4);
    return value;
}
// Independent host implementation: choose between the adjacent BF16 values
// in double precision. Do not duplicate the device bit-bias implementation.
static float round_bf16(float value) {
    REQUIRE(std::isfinite(value));
    uint32_t bits;
    std::memcpy(&bits,&value,4);
    const uint16_t trunc = uint16_t(bits>>16u);
    if (!(bits&0xffffu)) return value;
    const float a = bf16_value(trunc), b = bf16_value(uint16_t(trunc+1u));
    const double da = std::abs(double(value)-a), db = std::abs(double(value)-b);
    return da<db ? a : db<da ? b : (trunc&1u) ? b : a;
}

int main(int argc,char **argv) {
    REQUIRE(argc==1 || (argc==2 && !std::strcmp(argv[1],"--stream")));
    const bool stream = argc==2;
    const char *model = std::getenv("DS4_GLM5_MODEL");
    REQUIRE(model);
    Glm5TestGGUF gguf;
    REQUIRE(gguf.open_file(model));
    const char *names[] = {"q","k","v","f_a","g_a","beta"};
    constexpr unsigned full_rows[] = {8192,8192,8192,128,128,64};
    constexpr unsigned k = 4096u;
    constexpr size_t guard = 32;
    constexpr float sentinel = -1234567.0f;
    size_t total = 0, cases = 0;
    std::vector<unsigned> layers = {0u,44u};
    if (stream) {
        layers.clear();
        for (unsigned layer=0; layer<45; ++layer)
            if (layer%4u != 3u) layers.push_back(layer);
        REQUIRE(layers.size()==34u);
    }
    // Stream mode walks all matrices, not a persistent matrix cache. Each
    // layer is copied once and reused only for its own correctness/timing.
    for (unsigned layer : layers) for (unsigned layout=0; layout<(stream?2u:3u); ++layout) {
        const unsigned n = layout<2u ? 4096u : 8192u;
        const unsigned beta = layout<2u ? 32u : 64u;
        const unsigned rows[] = {n,n,n,128,128,beta};
        uint16_t *weight[6];
        const uint16_t *w[6], *host_w[6];
        for (unsigned i=0; i<6; ++i) {
            char name[80];
            std::snprintf(name,sizeof(name),"blk.%u.kda_%s.weight",layer,names[i]);
            uint64_t offset;
            REQUIRE(gguf.tensor(name,{k,full_rows[i]},30,offset));
            if (layout==1u && (i<3u || i==5u)) offset += uint64_t(rows[i])*k*2u;
            const size_t bytes = size_t(rows[i])*k*2u;
            REQUIRE(offset<=gguf.size && bytes<=gguf.size-offset);
            host_w[i] = reinterpret_cast<const uint16_t *>(gguf.map+offset);
            REQUIRE(hipMalloc(&weight[i],bytes)==hipSuccess);
            REQUIRE(hipMemcpy(weight[i],host_w[i],bytes,hipMemcpyHostToDevice)==hipSuccess);
            w[i] = weight[i];
        }
        const std::vector<unsigned> batches = stream ? std::vector<unsigned>{1024u} :
                                                       std::vector<unsigned>{256u,1024u};
        for (unsigned m : batches) {
            std::vector<float> x(size_t(m)*k), rounded(x.size());
            for (size_t i=0; i<x.size(); ++i) {
                x[i] = float(std::sin(double(i)*0.017+layer)*0.13+
                             std::cos(double(i)*0.037+layout)*0.19);
                rounded[i] = round_bf16(x[i]);
            }
            float *dx, *dr, *allocation[4][6], *out[4][6];
            REQUIRE(hipMalloc(&dx,x.size()*4u)==hipSuccess);
            REQUIRE(hipMalloc(&dr,x.size()*4u)==hipSuccess);
            REQUIRE(hipMemcpy(dx,x.data(),x.size()*4u,hipMemcpyHostToDevice)==hipSuccess);
            REQUIRE(hipMemcpy(dr,rounded.data(),x.size()*4u,hipMemcpyHostToDevice)==hipSuccess);
            uint32_t *pairs;
            REQUIRE(hipMalloc(&pairs,x.size()*4u)==hipSuccess);
            REQUIRE(glm5_six_rounding(pairs,dx,x.size())==hipSuccess);
            std::vector<uint32_t> host_pairs(x.size());
            REQUIRE(hipMemcpy(host_pairs.data(),pairs,x.size()*4u,hipMemcpyDeviceToHost)==hipSuccess);
            for (size_t i=0; i<x.size(); ++i) {
                const float device_round = bf16_value(uint16_t(host_pairs[i]));
                REQUIRE(std::memcmp(&device_round,&rounded[i],4)==0);
            }
            REQUIRE(hipFree(pairs)==hipSuccess);
            for (unsigned arm=0; arm<4; ++arm) for (unsigned i=0; i<6; ++i) {
                const size_t count = size_t(m)*rows[i];
                std::vector<float> init(count+2*guard,sentinel);
                std::fill(init.begin()+guard,init.end()-guard,NAN);
                REQUIRE(hipMalloc(&allocation[arm][i],init.size()*4u)==hipSuccess);
                out[arm][i] = allocation[arm][i]+guard;
                REQUIRE(hipMemcpy(allocation[arm][i],init.data(),init.size()*4u,hipMemcpyHostToDevice)==hipSuccess);
            }
            REQUIRE(glm5_six_incumbent(out[0],w,dx,n,beta,m,0,nullptr)==hipSuccess);
            REQUIRE(glm5_six_incumbent(out[3],w,dr,n,beta,m,0,nullptr)==hipSuccess);
            REQUIRE(glm5_six_native(out[1],w,dx,n,beta,m,0)==hipSuccess);
            REQUIRE(glm5_six_native(out[2],w,dx,n,beta,m,1)==hipSuccess);
            for (unsigned i=0; i<6; ++i) {
                const size_t count = size_t(m)*rows[i];
                std::array<std::vector<float>,4> result;
                for (unsigned arm=0; arm<4; ++arm) {
                    result[arm].resize(count+2*guard);
                    REQUIRE(hipMemcpy(result[arm].data(),allocation[arm][i],result[arm].size()*4u,hipMemcpyDeviceToHost)==hipSuccess);
                    for (size_t j=0; j<guard; ++j)
                        REQUIRE(result[arm][j]==sentinel && result[arm][count+guard+j]==sentinel);
                    for (size_t j=guard; j<count+guard; ++j) REQUIRE(std::isfinite(result[arm][j]));
                }
                size_t off_diff=0, intended_diff=0, drift=0;
                double squared=0, base_squared=0, max_abs=0;
                for (size_t j=guard; j<count+guard; ++j) {
                    off_diff += std::memcmp(&result[0][j],&result[1][j],4)!=0;
                    intended_diff += std::memcmp(&result[i<3u?3:0][j],&result[2][j],4)!=0;
                    drift += std::memcmp(&result[0][j],&result[2][j],4)!=0;
                    const double delta = double(result[2][j])-result[0][j];
                    squared += delta*delta;
                    base_squared += double(result[0][j])*result[0][j];
                    max_abs = std::max(max_abs,std::abs(delta));
                }
                std::printf("six_native layer=%u layout=%u M=%u role=%u values=%zu off_diff=%zu intended_diff=%zu drift=%zu rel_l2=%.9g max_abs=%.9g\n",
                    layer,layout,m,i,count,off_diff,intended_diff,drift,
                    std::sqrt(squared/std::max(base_squared,1e-300)),max_abs);
                std::fflush(stdout);
                REQUIRE(off_diff==0 && intended_diff==0);
                double max_original_error=0, max_intended_error=0, max_rounding=0;
                double max_parent_error=0, max_error_over_sumabs=0;
                for (unsigned sample=0; sample<32u; ++sample) {
                    const unsigned t = sample<2 ? (sample?m-1u:0u) : (sample*137u+layer)%m;
                    const unsigned row = sample<2 ? (sample?rows[i]-1u:0u) : (sample*257u+layout)%rows[i];
                    double original=0, intended=0, magnitude=0;
                    for (unsigned col=0; col<k; ++col) {
                        const double value = bf16_value(host_w[i][size_t(row)*k+col]);
                        const size_t index = size_t(t)*k+col;
                        original += value*x[index];
                        const double product = value*(i<3u?rounded[index]:x[index]);
                        intended += product;
                        magnitude += std::abs(product);
                    }
                    const double observed = result[2][guard+size_t(t)*rows[i]+row];
                    const double error = std::abs(observed-intended);
                    // Conservative FP32 dot-product forward error; a numeric
                    // implementation check, not a model-quality tolerance.
                    const double ku = 2.0*k*std::numeric_limits<float>::epsilon();
                    REQUIRE(error <= ku/(1.0-ku)*magnitude+std::numeric_limits<float>::min());
                    max_original_error = std::max(max_original_error,std::abs(observed-original));
                    max_intended_error = std::max(max_intended_error,error);
                    max_rounding = std::max(max_rounding,std::abs(intended-original));
                    max_parent_error = std::max(max_parent_error,
                        std::abs(double(result[0][guard+size_t(t)*rows[i]+row])-original));
                    max_error_over_sumabs = std::max(max_error_over_sumabs,error/std::max(magnitude,1e-300));
                }
                std::printf("six_fp64 layer=%u layout=%u M=%u role=%u samples=32 original_error=%.9g intended_error=%.9g rounding=%.9g parent_error=%.9g intended_error_over_sumabs=%.9g\n",
                    layer,layout,m,i,max_original_error,max_intended_error,max_rounding,
                    max_parent_error,max_error_over_sumabs);
                total += count;
            }
            REQUIRE(glm5_six_native(out[2],w,dx,n,beta,m-1u,1)==hipErrorInvalidValue);
            REQUIRE(glm5_six_native(out[2],w,dx,n,beta,m,2)==hipErrorInvalidValue);
            REQUIRE(glm5_six_native(out[2],w,dx,n,beta+1u,m,1)==hipErrorInvalidValue);
            REQUIRE(glm5_six_native(out[2],w,nullptr,n,beta,m,1)==hipErrorInvalidValue);
            float *saved = out[2][5];
            out[2][5] = nullptr;
            REQUIRE(glm5_six_native(out[2],w,dx,n,beta,m,1)==hipErrorInvalidValue);
            out[2][5] = saved;
            hipEvent_t begin,end;
            REQUIRE(hipEventCreate(&begin)==hipSuccess);
            REQUIRE(hipEventCreate(&end)==hipSuccess);
            auto launch = [&](unsigned mode) {
                REQUIRE(glm5_six_native(out[mode+1],w,dx,n,beta,m,mode)==hipSuccess);
            };
            launch(0); launch(1);
            REQUIRE(hipDeviceSynchronize()==hipSuccess);
            for (unsigned pair=0; pair<3; ++pair) for (unsigned arm=0; arm<2; ++arm) {
                const unsigned mode = arm^(pair&1u);
                const auto start = std::chrono::steady_clock::now();
                REQUIRE(hipEventRecord(begin,nullptr)==hipSuccess);
                for (unsigned repeat=0; repeat<3; ++repeat) launch(mode);
                REQUIRE(hipEventRecord(end,nullptr)==hipSuccess);
                REQUIRE(hipEventSynchronize(end)==hipSuccess);
                const double wall = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/3;
                float ms=0;
                REQUIRE(hipEventElapsedTime(&ms,begin,end)==hipSuccess);
                REQUIRE(std::isfinite(ms) && ms>0);
                std::printf("six_native_time layer=%u layout=%u M=%u pair=%u mode=%u device_ms=%.6f wall_ms=%.6f\n",
                    layer,layout,m,pair,mode,ms/3,wall);
            }
            REQUIRE(hipEventDestroy(begin)==hipSuccess);
            REQUIRE(hipEventDestroy(end)==hipSuccess);
            for (unsigned arm=0; arm<4; ++arm) for (unsigned i=0; i<6; ++i)
                REQUIRE(hipFree(allocation[arm][i])==hipSuccess);
            REQUIRE(hipFree(dr)==hipSuccess);
            REQUIRE(hipFree(dx)==hipSuccess);
            ++cases;
            std::fflush(stdout);
        }
        for (unsigned i=0; i<6; ++i) REQUIRE(hipFree(weight[i])==hipSuccess);
    }
    std::printf("PASS six native cases=%zu values=%zu stream=%d; arithmetic diagnostic only, no quality or model-speed claim\n",cases,total,stream);
}
