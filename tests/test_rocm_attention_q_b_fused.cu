/* Exact-shape oracle for TP Q-B projection plus head norm and RoPE. */
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define CHECK(c,m) do { if (!(c)) { fprintf(stderr,"FAIL: %s (line %d)\n",m,__LINE__); return 1; } } while(0)
typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

static void pack(unsigned char *w, uint32_t rows) {
    for (uint32_t r=0;r<rows;r++) for(uint32_t b=0;b<32u;b++) {
        unsigned char *p=w+((uint64_t)r*32u+b)*34u;
        p[0]=0;p[1]=0x34;
        for(uint32_t i=0;i<32u;i++) p[2u+i]=(unsigned char)(int8_t)
            ((int)((r*13u+b*7u+i*3u)%31u)-15);
    }
}

int main(void) {
    constexpr uint32_t in_dim=1024u,n_head=32u,head_dim=512u,n_rot=64u;
    constexpr uint32_t out_dim=n_head*head_dim,pos0=4095u,nctx=32768u;
    constexpr uint64_t weight_bytes=(uint64_t)out_dim*32u*34u;
    constexpr float freq_base=10000.0f,freq_scale=0.25f,ext=1.0f;
    constexpr float beta_fast=32.0f,beta_slow=1.0f,eps=1.0e-6f;
    const float attn=1.0f/(1.0f+0.1f*logf(1.0f/freq_scale));
    int ndev=0; CHECK(hipGetDeviceCount(&ndev)==hipSuccess&&ndev>0,"ROCm device");
    ds4_gpu_config cfg={};cfg.n_gpus=1;cfg.device_indices[0]=0;
    CHECK(ds4_gpu_init_multi(&cfg),"init");ds4_gpu_set_quality(false);
    unsigned char *model=(unsigned char*)malloc(weight_bytes);
    std::vector<float>x(in_dim),ref(out_dim),got(out_dim);
    CHECK(model,"model allocation");pack(model,out_dim);
    for(uint32_t i=0;i<in_dim;i++)x[i]=0.03125f*(float)((int)(i%47u)-23);
    CHECK(ds4_gpu_set_model_map(model,weight_bytes),"model map");
    ds4_gpu_tensor xd={},rd={},gd={};
    CHECK(ds4_gpu_tensor_alloc_on(&xd,0,in_dim*4u)==0&&
          ds4_gpu_tensor_alloc_on(&rd,0,(uint64_t)out_dim*4u)==0&&
          ds4_gpu_tensor_alloc_on(&gd,0,(uint64_t)out_dim*4u)==0,"alloc");
    CHECK(ds4_gpu_tensor_write(&xd,0,x.data(),in_dim*4u),"write x");
    CHECK(ds4_gpu_matmul_q8_0_tensor(&rd,model,weight_bytes,0,in_dim,out_dim,&xd,1)&&
          ds4_gpu_head_rms_norm_rope_tail_tensor(&rd,1,n_head,head_dim,n_rot,pos0,nctx,
              false,freq_base,freq_scale,ext,attn,beta_fast,beta_slow,eps),"reference");
    CHECK(setenv("DS4_ROCM_ATTENTION_Q_B_QNORM_ROPE_FUSE","1",1)==0,"enable");
    CHECK(ds4_gpu_attention_q_b_qnorm_rope_q8_0_tensor(&gd,model,weight_bytes,0,
          in_dim,n_head,head_dim,&xd,n_rot,pos0,nctx,freq_base,freq_scale,ext,
          attn,beta_fast,beta_slow,eps),"candidate");
    CHECK(ds4_gpu_tensor_read(&rd,0,ref.data(),(uint64_t)out_dim*4u)&&
          ds4_gpu_tensor_read(&gd,0,got.data(),(uint64_t)out_dim*4u),"read");
    uint64_t diff=0,nonrope=0,first=0;double max_abs=0;
    for(uint64_t i=0;i<out_dim;i++)if(memcmp(&ref[i],&got[i],4u)){
        if(!diff)first=i;diff++;if(i%head_dim<head_dim-n_rot)nonrope++;
        max_abs=fmax(max_abs,fabs((double)ref[i]-got[i]));
    }
    if(diff)fprintf(stderr,"diff=%llu nonrope=%llu max_abs=%.9g first=%llu ref=%a got=%a\n",
        (unsigned long long)diff,(unsigned long long)nonrope,max_abs,
        (unsigned long long)first,ref[first],got[first]);
    CHECK(nonrope==0,"projection and normalized non-RoPE values exact");
    CHECK(max_abs<=1.0e-6,"RoPE tail error bound");
    hipEvent_t s=nullptr,e=nullptr;CHECK(hipEventCreate(&s)==hipSuccess&&hipEventCreate(&e)==hipSuccess,"events");
    constexpr uint32_t it=200u;float mr=0,mf=0;
    CHECK(hipEventRecord(s)==hipSuccess,"rs");for(uint32_t i=0;i<it;i++){
        CHECK(ds4_gpu_matmul_q8_0_tensor(&rd,model,weight_bytes,0,in_dim,out_dim,&xd,1)&&
              ds4_gpu_head_rms_norm_rope_tail_tensor(&rd,1,n_head,head_dim,n_rot,pos0,nctx,
              false,freq_base,freq_scale,ext,attn,beta_fast,beta_slow,eps),"rtime");
    }CHECK(hipEventRecord(e)==hipSuccess&&hipEventSynchronize(e)==hipSuccess&&
      hipEventElapsedTime(&mr,s,e)==hipSuccess,"rend");
    CHECK(hipEventRecord(s)==hipSuccess,"fs");for(uint32_t i=0;i<it;i++)
        CHECK(ds4_gpu_attention_q_b_qnorm_rope_q8_0_tensor(&gd,model,weight_bytes,0,
          in_dim,n_head,head_dim,&xd,n_rot,pos0,nctx,freq_base,freq_scale,ext,
          attn,beta_fast,beta_slow,eps),"ftime");
    CHECK(hipEventRecord(e)==hipSuccess&&hipEventSynchronize(e)==hipSuccess&&
      hipEventElapsedTime(&mf,s,e)==hipSuccess,"fend");
    fprintf(stderr,"test_rocm_attention_q_b_fused: PASS diff=%llu max_abs=%.9g reference=%.3fus fused=%.3fus change=%+.1f%%\n",
      (unsigned long long)diff,max_abs,mr*1000.0/it,mf*1000.0/it,100.0*((double)mf/mr-1.0));
    CHECK(hipEventDestroy(s)==hipSuccess&&hipEventDestroy(e)==hipSuccess,"destroy events");
    ds4_gpu_tensor_free_in_place(&xd);ds4_gpu_tensor_free_in_place(&rd);ds4_gpu_tensor_free_in_place(&gd);
    free(model);ds4_gpu_cleanup();return 0;
}
