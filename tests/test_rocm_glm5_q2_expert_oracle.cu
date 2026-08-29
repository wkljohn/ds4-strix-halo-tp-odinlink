/* S1 component gate: one real IQ2_XXS/Q2_K expert, no cache or TP.
 * The reference uses independently built llama.cpp GGML dequantizers loaded
 * from libggml-base.so; it intentionally does not reuse DS4's CPU decoder. */
#include "ds4_gpu.h"
#include "tests/glm5_gguf_test.hpp"
#define GGML_COMMON_DECL_CPP
#include "/home/wkljohn/Desktop/cc/research-results/external-src/llama.cpp/ggml/src/ggml-common.h"
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

using deq_iq2_fn = void (*)(const block_iq2_xxs *, float *, int64_t);
using deq_q2_fn = void (*)(const block_q2_K *, float *, int64_t);

static float f16(uint16_t x) { uint32_t s=(x>>15)&1,e=(x>>10)&31,m=x&1023;
  uint32_t v; if(!e) v=m?((127-14-10+__builtin_clz(m))<<23)|((m<<(14-__builtin_clz(m)))&0x7fffff):s<<31;
  else if(e==31) v=(s<<31)|0x7f800000|(m<<13); else v=(s<<31)|((e+112)<<23)|(m<<13);
  float z; std::memcpy(&z,&v,4); return z; }
static void q8(const float *x, std::vector<int8_t>& q, std::vector<float>& d) {
  q.resize(256); d.resize(1); float mx=0, val=0; for(int i=0;i<256;i++) if(std::fabs(x[i])>mx){mx=std::fabs(x[i]);val=x[i];}
  if(mx==0){d[0]=0; std::fill(q.begin(),q.end(),0); return;} float is=-127.0f/val; d[0]=1.0f/is;
  for(int i=0;i<256;i++){int v=(int)std::lrint(is*x[i]); q[i]=(int8_t)std::max(-128,std::min(127,v));}
}
static float qdot(const float *w,const float *x,uint32_t n){float s=0; for(uint32_t i=0;i<n;i++) s+=w[i]*x[i]; return s;}
int main(){
  const char *mp=std::getenv("DS4_GLM5_MODEL"), *lp=std::getenv("DS4_LLAMA_GGML_BASE");
  if(!mp||!lp){std::fprintf(stderr,"FAIL DS4_GLM5_MODEL and DS4_LLAMA_GGML_BASE are required\n");return 2;}
  Glm5TestGGUF g; if(!g.open_file(mp)) return 1; uint64_t go,uo,doff;
  if(!g.tensor("blk.3.ffn_gate_exps.weight",{4096,2048,288},16,go)||!g.tensor("blk.3.ffn_up_exps.weight",{4096,2048,288},16,uo)||!g.tensor("blk.3.ffn_down_exps.weight",{2048,4096,288},10,doff)){std::fprintf(stderr,"FAIL Q2 tensors\n");return 1;}
  std::fprintf(stderr,"offsets gate=%llu up=%llu down=%llu size=%llu\n",(unsigned long long)go,(unsigned long long)uo,(unsigned long long)doff,(unsigned long long)g.size);
  void *h=dlopen(lp,RTLD_NOW); if(!h){std::fprintf(stderr,"FAIL dlopen %s\n",dlerror());return 1;}
  auto di=(deq_iq2_fn)dlsym(h,"dequantize_row_iq2_xxs"); auto dq=(deq_q2_fn)dlsym(h,"dequantize_row_q2_K"); if(!di||!dq){std::fprintf(stderr,"FAIL llama dequant symbols\n");return 1;}
  constexpr uint32_t E=17, N=4096, M=2048; std::vector<float>x(N); for(uint32_t i=0;i<N;i++) x[i]=std::sin(i*.017f)+.25f*std::cos(i*.0031f);
  std::vector<float> gate(M),up(M),mid(M),down(N), refg((uint64_t)M*N),refu((uint64_t)M*N),refd((uint64_t)N*M),out(N);
  for(uint32_t r=0;r<M;r++){di((const block_iq2_xxs*)(g.map+go+(uint64_t)E*M*16*66+r*16*66),refg.data()+(uint64_t)r*N,N);di((const block_iq2_xxs*)(g.map+uo+(uint64_t)E*M*16*66+r*16*66),refu.data()+(uint64_t)r*N,N);}
  for(uint32_t r=0;r<M;r++){gate[r]=qdot(refg.data()+(uint64_t)r*N,x.data(),N); up[r]=qdot(refu.data()+(uint64_t)r*N,x.data(),N); if(gate[r]>10)gate[r]=10; if(up[r]>10)up[r]=10; if(up[r]<-10)up[r]=-10; mid[r]=gate[r]/(1+std::exp(-gate[r]))*up[r];}
  for(uint32_t r=0;r<N;r++){dq((const block_q2_K*)(g.map+doff+(uint64_t)E*N*8*84+r*8*84),refd.data()+(uint64_t)r*M,M); down[r]=qdot(refd.data()+(uint64_t)r*M,mid.data(),M);}
  for(uint32_t i=0;i<M;i++) if(!std::isfinite(mid[i])) { std::fprintf(stderr,"bad mid[%u]=%g gate=%g up=%g\n",i,mid[i],gate[i],up[i]); return 1; }
  if(!ds4_gpu_init()||!ds4_gpu_set_model_map(g.map,g.size)||!ds4_gpu_set_model_fd_for_map(g.fd,g.map)) return 1;
  const uint64_t ge = go + (uint64_t)E*M*16*66, ue = uo + (uint64_t)E*M*16*66,
                 de = doff + (uint64_t)E*N*8*84;
  if(!ds4_gpu_cache_model_range(g.map,g.size,ge,(uint64_t)M*16*66,"q2 gate") ||
     !ds4_gpu_cache_model_range(g.map,g.size,ue,(uint64_t)M*16*66,"q2 up") ||
     !ds4_gpu_cache_model_range(g.map,g.size,de,(uint64_t)N*8*84,"q2 down")) return 1;
  ds4_gpu_tensor *tx=ds4_gpu_tensor_alloc(N*4),*to=ds4_gpu_tensor_alloc(N*4),*tg=ds4_gpu_tensor_alloc(M*4),*tu=ds4_gpu_tensor_alloc(M*4),*tm=ds4_gpu_tensor_alloc(M*4),*te=ds4_gpu_tensor_alloc(N*4),*td=ds4_gpu_tensor_alloc(N*4),*ts=ds4_gpu_tensor_alloc(4),*tw=ds4_gpu_tensor_alloc(4); int32_t eid=0; float weight=1;
  if(!tx||!to||!tg||!tu||!tm||!td||!ts||!tw||!ds4_gpu_tensor_write(tx,0,x.data(),N*4)||!ds4_gpu_tensor_write(ts,0,&eid,4)||!ds4_gpu_tensor_write(tw,0,&weight,4)) return 1;
  const int ok=ds4_gpu_routed_moe_one_tensor(to,tg,tu,tm,te,g.map,g.size,ge,ue,de,16,10,((uint64_t)M*16*66),16*66,((uint64_t)N*8*84),8*84,N,M,N,ts,tw,1,1,10.0f,tx,nullptr,3,false);
  if(!ok||!ds4_gpu_synchronize()||!ds4_gpu_tensor_read(to,0,out.data(),N*4)) return 1;
  double a=0,b=0,e=0; uint32_t nf=0; for(uint32_t i=0;i<N;i++){if(!std::isfinite(out[i])||!std::isfinite(down[i])) nf++; a+=down[i]*down[i];b+=out[i]*out[i];e+=(out[i]-down[i])*(out[i]-down[i]);} double n=std::sqrt(e/std::max(a,1e-30)); double c=0; for(uint32_t i=0;i<N;i++)c+=out[i]*down[i]; c/=std::sqrt(std::max(a*b,1e-30)); std::printf("GLM5 Q2 one-expert oracle nrmse=%.9g cosine=%.12g nonfinite=%u ref0=%g out0=%g\n",n,c,nf,down[0],out[0]); return (nf==0&&n<0.08&&c>0.995)?0:1;
}
