// Original GGUF dense prefill: production F32, separate WMMA, paired WMMA.
// Host references compile precisely; device kernels keep production flags.
#include "glm5_dense_prefill_probe.h"
#include "ds4_gpu_mgpu.h"
#include "glm5_gguf_test.hpp"
extern "C" {
#include "ds4_tp.h"
}
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>

extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr,"FAIL line=%d: %s\n",__LINE__,#x); std::exit(1); } } while(0)
constexpr unsigned K=4096, N=12288;
constexpr float guard=12345.0f;
static uint32_t bits(float v) { uint32_t b; std::memcpy(&b,&v,4); return b; }
static uint64_t digest(const std::vector<float>&v) {
    uint64_t h=UINT64_C(14695981039346656037);
    for(float x:v) for(unsigned shift=0;shift<32;shift+=8) {
        h^=(bits(x)>>shift)&255u; h*=UINT64_C(1099511628211);
    }
    return h;
}
struct Buffer {
    ds4_gpu_tensor *storage, *view;
    size_t count;
    uint32_t stamp=0;
    explicit Buffer(size_t n): storage(ds4_gpu_tensor_alloc((n+32)*4)),
        view(storage?ds4_gpu_tensor_view(storage,64,n*4):nullptr), count(n) { REQUIRE(view); }
    ~Buffer(){ ds4_gpu_tensor_free(view); ds4_gpu_tensor_free(storage); }
    void fill(){
        static uint32_t next=1;
        stamp=0x7fc00000u | (next++ & 0x003fffffu);
        float value; std::memcpy(&value,&stamp,4);
        REQUIRE(ds4_gpu_tensor_fill_f32(storage,guard,count+32));
        REQUIRE(ds4_gpu_tensor_fill_f32(view,value,count));
    }
    std::vector<float> read(bool written=true) {
        std::vector<float> all(count+32);
        REQUIRE(ds4_gpu_tensor_read(storage,0,all.data(),all.size()*4));
        for(unsigned i=0;i<16;++i) REQUIRE(all[i]==guard && all[count+16+i]==guard);
        for(size_t i=16;i<count+16;++i)
            REQUIRE(written ? std::isfinite(all[i]) : bits(all[i])==stamp);
        return {all.begin()+16,all.end()-16};
    }
};
static bool exact(const std::vector<float>&a,const std::vector<float>&b) {
    return a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*4);
}
static void errors(const char *kind,unsigned layer,unsigned m,unsigned seed,unsigned arm,
                   const std::vector<float>&a,const std::vector<float>&b) {
    REQUIRE(a.size()==b.size());
    double sq=0,base=0,max_abs=0; size_t neq=0,worst=0;
    for(size_t i=0;i<a.size();++i) {
        const double d=double(b[i])-a[i]; sq+=d*d; base+=double(a[i])*a[i];
        neq+=std::memcmp(&a[i],&b[i],4)!=0;
        if(std::abs(d)>max_abs){ max_abs=std::abs(d); worst=i; }
    }
    std::printf("DENSE_ERROR layer=%u m=%u seed=%u arm=%u stage=%s neq=%zu max_abs=%.9g nmse=%.9g worst=%zu\n",
        layer,m,seed,arm,kind,neq,max_abs,base?sq/base:sq,worst);
}
static float scale_value(const unsigned char *p) {
    _Float16 h; std::memcpy(&h,p,2); return float(h);
}
static void dot_reference(const unsigned char *w,const std::vector<float>&x,
                          const std::vector<float>&a,const std::vector<float>&b,
                          unsigned layer,unsigned m,unsigned seed,unsigned projection) {
    // Cover deterministic dispersed positions, edges, and measured error argmax.
    size_t worst=0; double max_error=-1;
    for(size_t i=0;i<a.size();++i) if(std::abs(double(a[i])-b[i])>max_error) {
        max_error=std::abs(double(a[i])-b[i]); worst=i;
    }
    std::vector<size_t> selected={0,a.size()-1,worst,size_t(m-1)*N,N-1};
    for(size_t j=0;j<64;++j)
        selected.push_back((j% m)*N+(j*193+layer*997+projection*31)%N);
    double max_a=0,max_b=0,max_rounded=0;
    for(size_t pos:selected) {
        const unsigned row=pos%N, tok=pos/N;
        const auto *wr=w+size_t(row)*(K/32*34);
        long double original=0,rounded=0,l1=0,rounded_l1=0;
        for(unsigned k=0;k<K;++k) {
            const auto *block=wr+(k/32)*34;
            const float weight=scale_value(block)*float((int8_t)block[2+k%32]);
            const float xv=x[size_t(tok)*K+k];
            const float rw=float((_Float16)weight), rx=float((_Float16)xv);
            REQUIRE(std::isfinite(rw) && std::isfinite(rx));
            original+=(long double)xv*weight; l1+=std::abs((long double)xv*weight);
            rounded+=(long double)rx*rw; rounded_l1+=std::abs((long double)rx*rw);
        }
        const double ea=std::abs((long double)a[pos]-original);
        const double eb=std::abs((long double)b[pos]-original);
        const double er=std::abs((long double)b[pos]-rounded);
        max_a=std::max(max_a,ea); max_b=std::max(max_b,eb); max_rounded=std::max(max_rounded,er);
        // Conservative F32 sequential-accumulation bound, not a quality envelope.
        constexpr long double eps=0x1p-24L, gamma=(K*eps)/(1-K*eps);
        REQUIRE(ea<=gamma*l1+1e-12L && er<=gamma*rounded_l1+1e-12L);
    }
    std::printf("DENSE_DOT layer=%u m=%u seed=%u projection=%u samples=%zu incumbent_abs=%.9g wmma_original_abs=%.9g wmma_rounded_abs=%.9g\n",
        layer,m,seed,projection,selected.size(),max_a,max_b,max_rounded);
}

static unsigned onehot_column(unsigned token,unsigned layer) {
    const unsigned edges[]={0,31,32,127,128,4095};
    return token<6?edges[token]:(token*37+layer*19)%K;
}
static void onehot_reference(const unsigned char *w,const std::vector<float>&v,
                            unsigned m,unsigned layer,bool rounded) {
    size_t changed=0; double max_abs=0;
    constexpr long double eps=0x1p-24L, gamma=(K*eps)/(1-K*eps);
    for(unsigned tok=0;tok<m;++tok) {
        const unsigned col=onehot_column(tok,layer);
        for(unsigned row=0;row<N;++row) {
            const auto *block=w+size_t(row)*(K/32*34)+(col/32)*34;
            float ref=scale_value(block)*float((int8_t)block[2+col%32]);
            if(rounded) ref=float((_Float16)ref);
            const float got=v[size_t(tok)*N+row];
            const double error=std::abs(double(got)-ref);
            changed+=got!=ref; max_abs=std::max(max_abs,error);
            // On gfx1151 even a single WMMA can shift a one-hot result by
            // one F32 ULP when its other products are negative zero. Retain
            // exact F32 control; for WMMA use the already-declared K-term
            // dot sanity bound, not a new model-quality tolerance.
            const bool valid=rounded?error<=gamma*std::abs((long double)ref)+1e-12L:got==ref;
            if(!valid) {
                std::fprintf(stderr,"ONEHOT_FAIL tok=%u row=%u col=%u rounded=%u got=%.9g ref=%.9g\n",
                    tok,row,col,unsigned(rounded),got,ref);
                REQUIRE(false);
            }
        }
    }
    std::printf("DENSE_ONEHOT layer=%u m=%u rounded=%u changed=%zu max_abs=%.9g quality_admission=0\n",
        layer,m,unsigned(rounded),changed,max_abs);
}

static std::vector<float> read_capture(const std::string &path,size_t count) {
    std::vector<float> data(count);
    FILE *fp=std::fopen(path.c_str(),"rb"); REQUIRE(fp);
    REQUIRE(std::fread(data.data(),4,count,fp)==count);
    REQUIRE(std::fgetc(fp)==EOF && !std::ferror(fp) && std::fclose(fp)==0);
    for(float value:data) REQUIRE(std::isfinite(value));
    return data;
}

static std::string capture_identity(const std::string &stem,unsigned rank,
        unsigned layer,unsigned pos,const uint64_t *offsets) {
    FILE *fp=std::fopen((stem+".meta").c_str(),"rb"); REQUIRE(fp);
    std::unordered_map<std::string,std::string> values;
    char line[1024];
    while(std::fgets(line,sizeof(line),fp)) {
        std::string s(line); REQUIRE(!s.empty() && s.back()=='\n'); s.pop_back();
        auto split=s.find('='); REQUIRE(split!=std::string::npos);
        REQUIRE(values.emplace(s.substr(0,split),s.substr(split+1)).second);
    }
    REQUIRE(!std::ferror(fp) && std::fclose(fp)==0 && values.size()==16);
    const std::pair<const char*,std::string> expected[]={
        {"schema","1"},{"implementation","f32-token-tile"},
        {"rank",std::to_string(rank)},{"layer",std::to_string(layer)},
        {"pos",std::to_string(pos)},{"rows","1024"},{"in_dim","4096"},
        {"mid_dim","12288"},{"clamp","10"},
        {"gate_offset",std::to_string(offsets[0])},{"up_offset",std::to_string(offsets[1])},
        {"down_offset",std::to_string(offsets[2])},{"input_bytes","16777216"},
        {"mid_bytes","50331648"},{"down_bytes","16777216"}};
    for(const auto &item:expected) REQUIRE(values.at(item.first)==item.second);
    REQUIRE(!values.at("run_id").empty());
    return values.at("run_id");
}

int main(int argc,char **argv) {
    const char *capture=nullptr; unsigned position=0,m=0;
    if(argc==4 && !std::strcmp(argv[1],"--replay")) {
        REQUIRE(!std::strcmp(argv[3],"3072") || !std::strcmp(argv[3],"7168"));
        capture=argv[2]; position=(unsigned)std::strtoul(argv[3],nullptr,10); m=1024;
    } else {
        REQUIRE(argc==3 && !std::strcmp(argv[1],"--rows"));
        REQUIRE(!std::strcmp(argv[2],"256") || !std::strcmp(argv[2],"1024"));
        m=(unsigned)std::strtoul(argv[2],nullptr,10);
    }
    const char *path=std::getenv("DS4_GLM5_MODEL"); REQUIRE(path);
    const std::pair<const char*,const char*> settings[]={
        {"DS4_GLM5_NEXT_ENABLE_ORDINARY","1"}, {"DS4_ROCM_ENABLE_Q8_F16_CACHE","0"},
        {"DS4_ROCM_SHARED_GU_WMMA_BATCH","0"}, {"DS4_ROCM_Q8_BATCH_WMMA_M128","1"},
        {"DS4_ROCM_Q8_BATCH_WMMA_K128_PADDED","1"}, {"DS4_ROCM_Q8_BATCH_WMMA_M256_K128","0"}};
    for(auto item:settings) REQUIRE(setenv(item.first,item.second,1)==0);
    REQUIRE(unsetenv("DS4_ROCM_DISABLE_Q8_BATCH_WMMA_M128")==0);
    REQUIRE(unsetenv("DS4_ROCM_DISABLE_Q8_BATCH_WMMA_K128_PADDED")==0);
    Glm5TestGGUF g; REQUIRE(g.open_file(path));
    uint64_t offsets[9],sizes[9];
    for(unsigned layer=0;layer<3;++layer) for(unsigned p=0;p<3;++p) {
        char name[80]; std::snprintf(name,sizeof(name),"blk.%u.ffn_%s.weight",layer,p==0?"gate":p==1?"up":"down");
        REQUIRE(g.tensor(name,p==2?std::vector<uint64_t>{N,K}:std::vector<uint64_t>{K,N},8,offsets[3*layer+p]));
        sizes[3*layer+p]=uint64_t(K)*N/32*34;
    }
    REQUIRE(ds4_gpu_init()); ds4_gpu_set_glm_model(true); ds4_gpu_set_q8_cache_suppressed(1);
    hipDeviceProp_t device; REQUIRE(hipGetDeviceProperties(&device,0)==hipSuccess);
    std::printf("DENSE_DEVICE arch=%s cache_suppressed=1 disable_m128=absent disable_k128=absent\n",device.gcnArchName);
    REQUIRE(!std::strncmp(device.gcnArchName,"gfx1151",7));
    REQUIRE(ds4_gpu_set_model_fd_for_map(g.fd,g.map));
    REQUIRE(ds4_gpu_set_model_map_spans(g.map,g.size,offsets,sizes,9,sizes[0]));
    {
        Buffer input(size_t(m)*K), gate(size_t(m)*N), up(gate.count), mid(gate.count), down(input.count);
        Buffer *out[]={&gate,&up,&mid,&down};
        auto run=[&](unsigned layer,unsigned arm,bool store,bool complete) {
            int ok=0;
            if(arm==0 && !store)
                ok=ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(gate.view,up.view,mid.view,g.map,g.size,
                    offsets[layer*3],offsets[layer*3+1],K,N,input.view,m,10.0f);
#ifndef DENSE_PREFILL_CONTROL_ONLY
            else
                ok=ds4_gpu_test_glm5_dense_prefill(gate.view,up.view,mid.view,g.map,g.size,
                    offsets[layer*3],offsets[layer*3+1],input.view,m,arm==1?2:arm==2?1:0,store?1:0);
#endif
            return ok && (!complete || ds4_gpu_matmul_q8_0_tensor(down.view,g.map,g.size,
                offsets[layer*3+2],N,K,mid.view,m));
        };
        std::string captured_run;
        for(unsigned layer=0;layer<3;++layer) for(unsigned seed=0;seed<(capture?2u:4u);++seed) {
            std::vector<float> x(input.count);
            for(size_t i=0;i<x.size();++i)
                x[i]=seed==0||seed==3?0.0f:seed==1?float(std::sin(double(i)*0.017+layer)*0.31+
                    std::cos(double(i)*0.037)*0.19):float(std::sin(double(i)*0.019+layer))*
                    (i%3?0.00013f:10.0003f);
            if(seed==3) for(unsigned tok=0;tok<m;++tok) x[size_t(tok)*K+onehot_column(tok,layer)]=1.0f;
            std::vector<float> captured_mid,captured_down;
            if(capture) {
                const std::string stem=std::string(capture)+".r"+std::to_string(seed)+
                    ".l"+std::to_string(layer)+".p"+std::to_string(position)+".m1024";
                const auto run_id=capture_identity(stem,seed,layer,position,offsets+3*layer);
                if(captured_run.empty()) captured_run=run_id;
                REQUIRE(captured_run==run_id);
                x=read_capture(stem+".input.f32",input.count);
                captured_mid=read_capture(stem+".mid.f32",mid.count);
                captured_down=read_capture(stem+".down.f32",down.count);
            }
            size_t overflows=0;
            double input_max=0; size_t subnormal=0,rounded_zero=0;
            for(float v:x) {
                overflows+=std::abs(v)>65504.0f || !std::isfinite(v);
                input_max=std::max(input_max,std::abs(double(v)));
                const float half=float((_Float16)v);
                subnormal+=half!=0 && std::abs(half)<0x1p-14f;
                rounded_zero+=v!=0 && half==0;
            }
            REQUIRE(overflows==0);
            input.fill(); REQUIRE(ds4_gpu_tensor_write(input.view,0,x.data(),x.size()*4));
            for(auto *o:out) o->fill();
            REQUIRE(run(layer,0,false,true) && ds4_gpu_synchronize());
            auto production=mid.read(), production_down=down.read();
            if(capture) {
                REQUIRE(exact(production,captured_mid) && exact(production_down,captured_down));
                std::printf("DENSE_REPLAY layer=%u rank=%u pos=%u run_id=%s exact=1 input_max=%.9g fp16_subnormal=%zu rounded_zero=%zu\n",
                    layer,seed,position,captured_run.c_str(),input_max,subnormal,rounded_zero);
            }
            gate.read(false); up.read(false);
            std::printf("DENSE_CONTROL layer=%u m=%u seed=%u mid=%016llx down=%016llx fp16_overflows=%zu\n",
                layer,m,seed,(unsigned long long)digest(production),(unsigned long long)digest(production_down),overflows);
#ifndef DENSE_PREFILL_CONTROL_ONLY
            std::array<std::vector<float>,4> result[3];
            for(unsigned arm=0;arm<3;++arm) {
                for(auto *o:out) o->fill();
                REQUIRE(run(layer,arm,true,true) && ds4_gpu_synchronize());
                for(unsigned p=0;p<4;++p) result[arm][p]=out[p]->read();
                REQUIRE(input.read()==x);
            }
            REQUIRE(exact(production,result[0][2]) && exact(production_down,result[0][3]));
            // Independent separate-kernel operands/reduction check for fusion.
            REQUIRE(exact(result[1][0],result[2][0]) && exact(result[1][1],result[2][1]));
            if(!capture && seed==1) REQUIRE(!exact(result[0][0],result[1][0]));
            // The ordinary generic route must agree with the explicit B kernel.
            for(auto *o:out) o->fill();
            REQUIRE(ds4_gpu_matmul_q8_0_tensor(gate.view,g.map,g.size,offsets[layer*3],K,N,input.view,m));
            REQUIRE(ds4_gpu_matmul_q8_0_tensor(up.view,g.map,g.size,offsets[layer*3+1],K,N,input.view,m));
            REQUIRE(ds4_gpu_swiglu_tensor(mid.view,gate.view,up.view,m*N,10.0f,1.0f));
            REQUIRE(ds4_gpu_synchronize());
            REQUIRE(exact(gate.read(),result[1][0]) && exact(up.read(),result[1][1]) && exact(mid.read(),result[1][2]));
            for(auto *o:out) o->fill();
            REQUIRE(run(layer,2,false,true) && ds4_gpu_synchronize());
            REQUIRE(exact(mid.read(),result[2][2]) && exact(down.read(),result[2][3]));
            gate.read(false); up.read(false);
            // Hold projections fixed to identify a separate epilogue effect.
            REQUIRE(ds4_gpu_tensor_write(gate.view,0,result[2][0].data(),gate.count*4));
            REQUIRE(ds4_gpu_tensor_write(up.view,0,result[2][1].data(),up.count*4));
            REQUIRE(ds4_gpu_swiglu_tensor(mid.view,gate.view,up.view,m*N,10.0f,1.0f) && ds4_gpu_synchronize());
            errors("paired-epilogue-only",layer,m,seed,2,result[2][2],mid.read());
            const char *names[]={"gate","up","mid","down"};
            for(unsigned arm=1;arm<3;++arm) for(unsigned p=0;p<4;++p)
                errors(names[p],layer,m,seed,arm,result[0][p],result[arm][p]);
            for(unsigned p=2;p<4;++p) errors(p==2?"paired-vs-separate-mid":"paired-vs-separate-down",
                layer,m,seed,2,result[1][p],result[2][p]);
            size_t crossings=0;
            for(size_t i=0;i<gate.count;++i) {
                crossings+=(result[0][0][i]>10)!=(result[1][0][i]>10);
                crossings+=(result[0][1][i]>10)!=(result[1][1][i]>10);
                crossings+=(result[0][1][i]<-10)!=(result[1][1][i]<-10);
            }
            std::printf("DENSE_CLAMP layer=%u m=%u seed=%u crossings=%zu\n",layer,m,seed,crossings);
            for(unsigned p=0;p<2;++p) dot_reference(g.map+offsets[3*layer+p],x,result[0][p],result[1][p],layer,m,seed,p);
            if(!capture && seed==3) for(unsigned arm=0;arm<3;++arm) for(unsigned p=0;p<2;++p)
                onehot_reference(g.map+offsets[3*layer+p],result[arm][p],m,layer,arm!=0);
#endif
            std::fflush(stdout);
        }
#ifndef DENSE_PREFILL_CONTROL_ONLY
        if(!capture) {
        // Bad admissions cannot modify outputs. All selected calls must succeed.
        for(auto *o:out) o->fill();
        for(unsigned bad:{0u,1u,255u,257u,512u,1023u,1025u})
            REQUIRE(!ds4_gpu_test_glm5_dense_prefill(gate.view,up.view,mid.view,g.map,g.size,offsets[0],offsets[1],input.view,bad,1,0));
        REQUIRE(!ds4_gpu_test_glm5_dense_prefill(gate.view,up.view,mid.view,g.map,g.size,offsets[0],offsets[1],input.view,m,3,0));
        REQUIRE(!ds4_gpu_test_glm5_dense_prefill(gate.view,up.view,mid.view,g.map,g.size,offsets[0],offsets[1],input.view,m,1,2));
        REQUIRE(!ds4_gpu_test_glm5_dense_prefill(gate.view,gate.view,mid.view,g.map,g.size,offsets[0],offsets[1],input.view,m,1,0));
        REQUIRE(!ds4_gpu_test_glm5_dense_prefill(gate.view,up.view,mid.view,g.map,g.size,g.size-2,offsets[1],input.view,m,1,0));
        REQUIRE(!ds4_gpu_test_glm5_dense_prefill(gate.view,up.view,mid.view,g.map,g.size,offsets[0],offsets[0],input.view,m,1,0));
        auto *short_out=ds4_gpu_tensor_view(gate.view,0,gate.count*4-4); REQUIRE(short_out);
        REQUIRE(!ds4_gpu_test_glm5_dense_prefill(short_out,up.view,mid.view,g.map,g.size,offsets[0],offsets[1],input.view,m,1,0));
        ds4_gpu_tensor_free(short_out);
        for(auto *o:out) o->read(false);
        // Precise host epilogue reference around both clamp boundaries.
        std::vector<float> gv(gate.count),uv(gate.count);
        const float values[]={-20.0f,-10.00001f,-10.0f,-9.99999f,-0.0f,0.0f,9.99999f,10.0f,10.00001f,20.0f};
        for(size_t i=0;i<gv.size();++i){ gv[i]=values[i%10]; uv[i]=values[(i/10)%10]; }
        REQUIRE(ds4_gpu_tensor_write(gate.view,0,gv.data(),gv.size()*4));
        REQUIRE(ds4_gpu_tensor_write(up.view,0,uv.data(),uv.size()*4));
        REQUIRE(ds4_gpu_swiglu_tensor(mid.view,gate.view,up.view,m*N,10.0f,1.0f));
        auto mv=mid.read();
        for(size_t i=0;i<mv.size();++i) {
            const double gc=std::min(double(gv[i]),10.0),uc=std::clamp(double(uv[i]),-10.0,10.0);
            const double ref=gc/(1+std::exp(-gc))*uc;
            REQUIRE(std::abs(double(mv[i])-ref)<=2e-6*std::max(1.0,std::abs(ref)));
        }
        // Nonfinite epilogue behaviour is diagnostic: clamping can hide it.
        const float special[]={std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),1.0f};
        for(size_t i=0;i<gv.size();++i){ gv[i]=special[i%4]; uv[i]=special[(i/4)%4]; }
        REQUIRE(ds4_gpu_tensor_write(gate.view,0,gv.data(),gv.size()*4));
        REQUIRE(ds4_gpu_tensor_write(up.view,0,uv.data(),uv.size()*4));
        REQUIRE(ds4_gpu_swiglu_tensor(mid.view,gate.view,up.view,m*N,10.0f,1.0f) && ds4_gpu_synchronize());
        float diag[16]; REQUIRE(ds4_gpu_tensor_read(mid.view,0,diag,sizeof(diag)));
        for(unsigned i=0;i<16;++i) std::printf("DENSE_EPILOGUE_NONFINITE case=%u gate_bits=%08x up_bits=%08x mid_bits=%08x\n",
            i,bits(gv[i]),bits(uv[i]),bits(diag[i]));
        // Timings stream original layers on a fixed nonzero input, no readbacks inside spans.
        std::vector<float> timing_x(input.count);
        for(size_t i=0;i<timing_x.size();++i) timing_x[i]=float(std::sin(double(i)*0.031)*0.3);
        REQUIRE(ds4_gpu_tensor_write(input.view,0,timing_x.data(),timing_x.size()*4));
        uint64_t timed_mid[3],timed_down[3];
        for(unsigned arm=0;arm<3;++arm) {
            for(auto *o:out) o->fill();
            REQUIRE(run(2,arm,false,true) && ds4_gpu_synchronize());
            timed_mid[arm]=digest(mid.read()); timed_down[arm]=digest(down.read());
        }
        hipEvent_t begin[3],middle[3],end[3];
        for(unsigned layer=0;layer<3;++layer)
            REQUIRE(hipEventCreate(&begin[layer])==hipSuccess && hipEventCreate(&middle[layer])==hipSuccess && hipEventCreate(&end[layer])==hipSuccess);
        double complete_samples[3][6];
        for(unsigned sample=0;sample<10;++sample)
            for(unsigned turn=0;turn<3;++turn) {
                const unsigned arm=(sample&1)?2-turn:turn;
                for(auto *o:out) o->fill();
                REQUIRE(ds4_gpu_synchronize());
                const auto start=std::chrono::steady_clock::now();
                for(unsigned layer=0;layer<3;++layer) {
                    REQUIRE(hipEventRecord(begin[layer])==hipSuccess);
                    REQUIRE(run(layer,arm,false,false));
                    REQUIRE(hipEventRecord(middle[layer])==hipSuccess);
                    REQUIRE(ds4_gpu_matmul_q8_0_tensor(down.view,g.map,g.size,offsets[layer*3+2],N,K,mid.view,m));
                    REQUIRE(hipEventRecord(end[layer])==hipSuccess);
                }
                REQUIRE(hipEventSynchronize(end[2])==hipSuccess);
                const double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                double total=0,gate_ms=0;
                for(unsigned layer=0;layer<3;++layer) {
                    float full=0,part=0;
                    REQUIRE(hipEventElapsedTime(&full,begin[layer],end[layer])==hipSuccess);
                    REQUIRE(hipEventElapsedTime(&part,begin[layer],middle[layer])==hipSuccess);
                    total+=full; gate_ms+=part;
                }
                if(sample>=4) {
                    complete_samples[arm][sample-4]=total;
                    std::printf("DENSE_TIME m=%u sample=%u order=%u arm=%u gate_ms=%.6f full_ms=%.6f wall_ms=%.6f\n",
                        m,sample-4,turn,arm,gate_ms,total,wall);
                }
                if(sample==9) {
                    REQUIRE(digest(mid.read())==timed_mid[arm] && digest(down.read())==timed_down[arm]);
                    if(arm!=1){ gate.read(false); up.read(false); }
                    else { gate.read(); up.read(); }
                }
            }
        for(unsigned layer=0;layer<3;++layer)
            REQUIRE(hipEventDestroy(begin[layer])==hipSuccess && hipEventDestroy(middle[layer])==hipSuccess && hipEventDestroy(end[layer])==hipSuccess);
        for(unsigned arm=0;arm<3;++arm) {
            std::sort(complete_samples[arm],complete_samples[arm]+6);
            std::printf("DENSE_MEDIAN m=%u arm=%u layers=3 complete_ms=%.6f\n",m,arm,
                (complete_samples[arm][2]+complete_samples[arm][3])*0.5);
        }
        }
#endif
    }
    ds4_gpu_cleanup();
    if(capture) std::printf("PASS dense-replay pos=%u m=%u cases=6 weights=original quality_admission=0\n",position,m);
    else std::printf("PASS dense-prefill m=%u layers=3 weights=original expanded_weight_cache_bytes=0 quality_admission=0\n",m);
}
