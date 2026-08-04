// Stage 2 routed Q4_K integer-WMMA correctness harness.
// Build exactly like q4k_dp4a_baseline_bench.cu (see that file's header).
// The include reuses the already validated Q4_K/Q8_K layouts, WMMA helpers,
// generators, and the verbatim shipping tile8 consumer.  It does not alter
// or link ds4-upstream.
#define main q4k_stage0_unused_main
#include "q4k_dp4a_baseline_bench.cu"
#undef main

// The following builders are verbatim from current
// ds4-upstream/rocm/ds4_rocm_moe.cuh:988-1082.  Production's current launch
// sequence is ds4_rocm_moe_launch.cuh:960-1002 and uses the deterministic
// scatter at :1030 (stable order), rather than the older atomic scatter.
// Verbatim body: ds4-upstream/rocm/ds4_rocm_moe.cuh:988-999.
__global__ static void moe_count_sorted_pairs_kernel(uint32_t *counts,const int32_t *selected,uint32_t pair_count,uint32_t n_total_expert) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    if ((uint32_t)expert_i >= n_total_expert) return;
    atomicAdd(counts + (uint32_t)expert_i, 1u);
}
// Verbatim body: ds4-upstream/rocm/ds4_rocm_moe.cuh:1001-1015.
__global__ static void moe_prefix_sorted_pairs_kernel(uint32_t *offsets,uint32_t *cursors,const uint32_t *counts,uint32_t n_total_expert) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < n_total_expert; e++) { offsets[e] = sum; cursors[e] = sum; sum += counts[e]; }
        offsets[n_total_expert] = sum;
    }
}
// Verbatim body: ds4-upstream/rocm/ds4_rocm_moe.cuh:1017-1029 (defined in
// production, retained here, but production currently launches the next one).
__global__ [[maybe_unused]] static void moe_scatter_sorted_pairs_kernel(uint32_t *sorted_pairs,uint32_t *cursors,const int32_t *selected,uint32_t pair_count,uint32_t n_total_expert) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    if ((uint32_t)expert_i >= n_total_expert) return;
    uint32_t pos = atomicAdd(cursors + (uint32_t)expert_i, 1u);
    sorted_pairs[pos] = pair;
}
// Verbatim body: ds4-upstream/rocm/ds4_rocm_moe.cuh:1034-1048.
__global__ static void moe_scatter_sorted_pairs_deterministic_kernel(uint32_t *sorted_pairs,const uint32_t *offsets,const int32_t *selected,uint32_t pair_count,uint32_t n_total_expert) {
    const uint32_t expert = (uint32_t)blockIdx.x;
    if (expert >= n_total_expert || threadIdx.x != 0u) return;
    uint32_t pos = offsets[expert];
    for (uint32_t pair = 0; pair < pair_count; pair++) {
        int32_t expert_i = selected[pair];
        if (expert_i < 0) expert_i = 0;
        if ((uint32_t)expert_i == expert) sorted_pairs[pos++] = pair;
    }
}
// Verbatim body: ds4-upstream/rocm/ds4_rocm_moe.cuh:1050-1065.
__global__ static void moe_build_expert_tile_offsets_kernel(uint32_t *tile_offsets,uint32_t *tile_total,const uint32_t *counts,uint32_t block_m,uint32_t n_total_expert) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < n_total_expert; e++) { tile_offsets[e] = sum; sum += (counts[e] + block_m - 1u) / block_m; }
        tile_offsets[n_total_expert] = sum; *tile_total = sum;
    }
}
// Verbatim body: ds4-upstream/rocm/ds4_rocm_moe.cuh:1067-1082.
__global__ static void moe_build_expert_tiles_kernel(uint32_t *tile_experts,uint32_t *tile_starts,const uint32_t *tile_offsets,const uint32_t *counts,uint32_t block_m,uint32_t n_total_expert) {
    uint32_t e = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= n_total_expert) return;
    uint32_t ntiles = (counts[e] + block_m - 1u) / block_m;
    uint32_t off = tile_offsets[e];
    for (uint32_t t = 0; t < ntiles; t++) { tile_experts[off + t] = e; tile_starts[off + t] = t * block_m; }
}

// One J=16 tile is one production CSR tile.  Grid.y spans the entire device
// tile stream; expert selection and pair/token decoding are device-side.
template<int J>
__global__ __launch_bounds__(256) static void q4k_wmma_routed_kernel(
 const cuda_block_q4_K *weights,const q8_1_mmq_block *acts,float *out,
 const uint32_t *sorted_pairs,const uint32_t *offsets,const uint32_t *counts,
 const uint32_t *tile_total,const uint32_t *tile_experts,const uint32_t *tile_starts,
 uint32_t ntokens,uint32_t xq_blocks,uint32_t nrows,uint32_t n_expert,
 uint32_t wmma_min_count,uint32_t *dispatch_counters,uint32_t *write_markers) {
    constexpr int I=64, XS=76, YS=36;
    extern __shared__ int32_t smem[]; int32_t *sy=smem,*sx=sy+J*YS;
    int tid=threadIdx.x,wave=tid>>5,lane=tid&31; uint32_t tile=blockIdx.y;
    if(tile>=*tile_total)return; uint32_t expert=tile_experts[tile],start=tile_starts[tile],count=counts[expert];
    if(count<wmma_min_count){if(dispatch_counters&&blockIdx.x==0&&tid==0)atomicAdd(dispatch_counters+1,1u);return;}
    if(dispatch_counters&&blockIdx.x==0&&tid==0)atomicAdd(dispatch_counters+0,1u);
    uint32_t row0=blockIdx.x*I; float acc[J/16][8]={};
    for(uint32_t kb=0;kb<xq_blocks;++kb){
      if(wave<8){int r=wave*8+lane/4,txi=(lane%4)*8;if(row0+r<nrows){const auto&w=weights[((uint64_t)expert*nrows+row0+r)*xq_blocks+kb];
        #pragma unroll
        for(int q=0;q<8;++q){int v=*reinterpret_cast<const int32_t*>(w.qs+4*(txi+q));sx[r*XS+16*((txi+q)/8)+(txi+q)%8]=(v>>0)&0x0f0f0f0f;sx[r*XS+16*((txi+q)/8)+(txi+q)%8+8]=(v>>4)&0x0f0f0f0f;}}}
      if(wave<4){int r=wave*16+lane/2;if(row0+r<nrows){const auto&w=weights[((uint64_t)expert*nrows+row0+r)*xq_blocks+kb];int ksc=lane&1,sc32=unpack_q4k_scales(reinterpret_cast<const int*>(w.scales),ksc),m32=unpack_q4k_scales(reinterpret_cast<const int*>(w.scales),ksc+2);const uint8_t*sc=(const uint8_t*)&sc32,*mn=(const uint8_t*)&m32;float wd=__half2float(*reinterpret_cast<const half*>(&w.d)),wm=__half2float(*reinterpret_cast<const half*>(&w.dmin));
        #pragma unroll
        for(int l=0;l<4;++l)reinterpret_cast<half2*>(sx+r*XS+64)[4*ksc+l]=__floats2half2_rn(wd*sc[l],-wm*mn[l]);}}
      __syncthreads();
      #pragma unroll
      for(int half=0;half<2;++half){for(int l=tid;l<J*YS;l+=256){int j=l/YS,e=l%YS;uint32_t lp=start+j;if(lp<count){uint32_t pair=sorted_pairs[offsets[expert]+lp],tok=pair/n_expert;const int32_t*src=(const int32_t*)&acts[((uint64_t)kb*2+half)*ntokens+tok];sy[l]=src[e];}else sy[l]=0;}__syncthreads();
        if(wave<4){for(int kk=0;kk<4;++kk){auto A=load_rdna3_mirrored_16x8(sx+wave*16*XS+half*32+kk*8,XS);for(int j0=0;j0<J;j0+=16){auto B=load_rdna3_mirrored_16x8(sy+j0*YS+4+kk*8,YS);i32x8 c={};c=wmma_i8_16x16x16(A,B,c);for(int l=0;l<8;++l){int i=2*l+lane/16,j=j0+lane%16;float2 bd=__half22float2(reinterpret_cast<const half2*>(sy+j*YS)[kk]);float2 ad=__half22float2(reinterpret_cast<const half2*>(sx+(wave*16+i)*XS+64)[half*4+kk]);acc[j0/16][l]+=ad.x*bd.x*(float)c[l]+ad.y*bd.y;}}}}
        __syncthreads();}}
    if(wave<4)for(int j0=0;j0<J;j0+=16)for(int l=0;l<8;++l){uint32_t row=row0+wave*16+2*l+lane/16,lp=start+j0+lane%16;if(row<nrows&&lp<count){uint32_t pair=sorted_pairs[offsets[expert]+lp];uint64_t off=(uint64_t)pair*nrows+row;out[off]=acc[j0/16][l];if(write_markers)atomicAdd(write_markers+off,1u);}}
}

// Harness-only DP4A fallback.  This is intentionally separate from the
// verbatim shipping tile8 kernel included above.  The shared crossover stream
// has 16-pair tiles, so each workgroup consumes it as two tile8 halves.
__global__ static void q4k_dp4a_cold_tile16_kernel(
 float *gate_out,float *up_out,const char *gate_base,const char *up_base,
 const cuda_block_q8_K *xq,const uint32_t *sorted_pairs,const uint32_t *offsets,
 const uint32_t *counts,const uint32_t *tile_total,const uint32_t *tile_experts,
 const uint32_t *tile_starts,uint64_t gate_expert_bytes,uint64_t gate_row_bytes,
 uint32_t xq_blocks,uint32_t nrows,uint32_t n_expert,uint32_t wmma_min_count,
 uint32_t *dispatch_counters,uint32_t *gate_markers,uint32_t *up_markers) {
 uint32_t tile=blockIdx.y;if(tile>=*tile_total)return;uint32_t expert=tile_experts[tile],count=counts[expert];
 if(count>=wmma_min_count){if(dispatch_counters&&blockIdx.x==0&&threadIdx.x==0)atomicAdd(dispatch_counters+3,1u);return;}
 if(dispatch_counters&&blockIdx.x==0&&threadIdx.x==0)atomicAdd(dispatch_counters+2,1u);
 uint32_t lane=threadIdx.x&7u,row=blockIdx.x*32u+(threadIdx.x>>3u),tile_start=tile_starts[tile];
 __shared__ cuda_block_q8_K sxq[8][16];
 for(uint32_t half=0;half<2u;++half){
  uint32_t pair[8]={},tok[8]={};const cuda_block_q8_K *xqb[8]={};uint32_t np=0,local_start=tile_start+half*8u;
  for(;np<8u;++np){uint32_t lp=local_start+np;if(lp>=count)break;pair[np]=sorted_pairs[offsets[expert]+lp];tok[np]=pair[np]/n_expert;xqb[np]=xq+(uint64_t)tok[np]*xq_blocks;}
  if(xq_blocks<=16u){for(uint32_t i=threadIdx.x;i<np*xq_blocks;i+=blockDim.x){uint32_t p=i/xq_blocks,b=i-p*xq_blocks;sxq[p][b]=xqb[p][b];}__syncthreads();for(uint32_t p=0;p<np;++p)xqb[p]=sxq[p];}
  if(row<nrows){
   const cuda_block_q4_K *gr=(const cuda_block_q4_K *)(gate_base+(uint64_t)expert*gate_expert_bytes+(uint64_t)row*gate_row_bytes);
   const cuda_block_q4_K *ur=(const cuda_block_q4_K *)(up_base+(uint64_t)expert*gate_expert_bytes+(uint64_t)row*gate_row_bytes);
   float gate[8]={},up[8]={};
   for(uint32_t b=lane;b<xq_blocks;b+=8u){dev_dot_q4_K_q8_K_block8(gr+b,xqb[0]?xqb[0]+b:NULL,xqb[1]?xqb[1]+b:NULL,xqb[2]?xqb[2]+b:NULL,xqb[3]?xqb[3]+b:NULL,xqb[4]?xqb[4]+b:NULL,xqb[5]?xqb[5]+b:NULL,xqb[6]?xqb[6]+b:NULL,xqb[7]?xqb[7]+b:NULL,np,gate);dev_dot_q4_K_q8_K_block8(ur+b,xqb[0]?xqb[0]+b:NULL,xqb[1]?xqb[1]+b:NULL,xqb[2]?xqb[2]+b:NULL,xqb[3]?xqb[3]+b:NULL,xqb[4]?xqb[4]+b:NULL,xqb[5]?xqb[5]+b:NULL,xqb[6]?xqb[6]+b:NULL,xqb[7]?xqb[7]+b:NULL,np,up);}
   for(uint32_t p=0;p<np;++p){gate[p]=quarter_warp_sum_f32(gate[p],lane);up[p]=quarter_warp_sum_f32(up[p],lane);if(lane==0){uint64_t off=(uint64_t)pair[p]*nrows+row;gate_out[off]=gate[p];up_out[off]=up[p];if(gate_markers)atomicAdd(gate_markers+off,2u);if(up_markers)atomicAdd(up_markers+off,2u);}}
  }
  __syncthreads();
 }
}

__global__ static void routed_epilogue(const float *gate,const float *up,float *mid,const uint32_t *sorted_pairs,const uint32_t *offsets,const uint32_t *counts,const uint32_t *tile_total,const uint32_t *tile_experts,const uint32_t *tile_starts,const float *weights,uint32_t nrows,uint32_t n_expert,float clamp){
 uint32_t tile=blockIdx.y;if(tile>=*tile_total)return;uint32_t e=tile_experts[tile],lp=tile_starts[tile]+blockIdx.x*blockDim.x+threadIdx.x;if(lp>=counts[e])return;uint32_t pair=sorted_pairs[offsets[e]+lp],tok=pair/n_expert,slot=pair-tok*n_expert;
 for(uint32_t row=0;row<nrows;++row){uint64_t off=(uint64_t)pair*nrows+row;float g=gate[off],u=up[off];if(clamp>1e-6f){if(g>clamp)g=clamp;if(u>clamp)u=clamp;if(u<-clamp)u=-clamp;}mid[off]=(g/(1.0f+expf(-g)))*u*weights[(uint64_t)tok*n_expert+slot];}
}

static bool close_vec(const char *tag,const std::vector<float>&a,const std::vector<float>&b,double at,double rt){uint64_t bad=0;double ma=0;for(size_t i=0;i<a.size();++i){double d=fabs((double)a[i]-b[i]);ma=std::max(ma,d);if(!std::isfinite(a[i])||d>at+rt*fabs((double)b[i]))++bad;}printf("  %-5s bad=%llu/%zu max_abs=%.6g\n",tag,(unsigned long long)bad,a.size(),ma);return bad==0;}

static float host_f16(uint16_t bits){half h;memcpy(&h,&bits,sizeof(h));return __half2float(h);}
static void host_scale_min(uint32_t j,const uint8_t*s,uint8_t&sc,uint8_t&mn){if(j<4){sc=s[j]&63u;mn=s[j+4]&63u;}else{sc=(s[j+4]&15u)|((s[j-4]>>6u)<<4u);mn=(s[j+4]>>4u)|((s[j]>>6u)<<4u);}}
static float host_q4k_q8k(const cuda_block_q4_K*w,const cuda_block_q8_K*x,uint32_t xb){float acc=0;for(uint32_t b=0;b<xb;++b){int isum=0,summs=0;for(uint32_t j=0;j<8;++j){uint8_t sc,mn;host_scale_min(j,w[b].scales,sc,mn);int dot=0,qsum=0;uint32_t byte_off=(j>>1u)*32u,shift=(j&1u)?4u:0u;for(uint32_t k=0;k<32;++k){int q=(w[b].qs[byte_off+k]>>shift)&15;int a=x[b].qs[j*32u+k];dot+=q*a;qsum+=a;}isum+=(int)sc*dot;summs+=(int)mn*qsum;}acc+=x[b].d*(host_f16(w[b].d)*(float)isum-host_f16(w[b].dmin)*(float)summs);}return acc;}
static void host_routed_reference(const std::vector<cuda_block_q4_K>&gate_w,const std::vector<cuda_block_q4_K>&up_w,const std::vector<cuda_block_q8_K>&x,const std::vector<int32_t>&selected,const std::vector<float>&route_w,uint32_t ne,uint32_t used,uint32_t xb,uint32_t nr,std::vector<float>&gate,std::vector<float>&up,std::vector<float>&mid){gate.resize((size_t)selected.size()*nr);up.resize(gate.size());mid.resize(gate.size());for(uint32_t p=0;p<selected.size();++p){uint32_t e=(uint32_t)std::max(selected[p],0),tok=p/used;for(uint32_t row=0;row<nr;++row){size_t out=(size_t)p*nr+row,wi=((size_t)e*nr+row)*xb;float g=host_q4k_q8k(gate_w.data()+wi,x.data()+(size_t)tok*xb,xb),u=host_q4k_q8k(up_w.data()+wi,x.data()+(size_t)tok*xb,xb);gate[out]=g;up[out]=u;float gc=std::min(g,3.0f),uc=std::max(-3.0f,std::min(u,3.0f));mid[out]=(gc/(1.0f+std::exp(-gc)))*uc*route_w[p];}}(void)ne;}

struct mid_tolerance_terms { double floor_and_up,gate,relative,total; };

// For F(g,u)=w*SiLU(g)*u, split the input error exactly as
// w*[SiLU(gw)*(uw-ur) + ur*(SiLU(gw)-SiLU(gr))].  The second term is
// w*ur*SiLU'(xi)*(gw-gr) by the mean-value theorem, but using the finite
// difference is exact and also handles the +clamp kink.  Keep the original
// absolute floor: small/negative SiLU values must not erase the established
// Q8_K-to-Q8_1 conversion-noise budget.
static mid_tolerance_terms mid_tolerance(float wmma_gate_raw,float ref_gate_clamped,float ref_up_clamped,float weight,float ref_mid,double at,double rt,float clamp){double gw=wmma_gate_raw;if(clamp>1e-6f&&gw>clamp)gw=clamp;double gr=ref_gate_clamped,sw=gw/(1.0+exp(-gw)),sr=gr/(1.0+exp(-gr)),mul=fabs((double)weight*sw);mid_tolerance_terms t;t.floor_and_up=std::max(at,mul*at);t.gate=fabs((double)weight*(double)ref_up_clamped*(sw-sr));t.relative=rt*fabs((double)ref_mid);t.total=t.floor_and_up+t.gate+t.relative;return t;}

static bool close_mid(const std::vector<float>&a,const std::vector<float>&b,const std::vector<float>&wmma_gate,const std::vector<float>&ref_gate,const std::vector<float>&ref_up,const std::vector<float>&weights,uint32_t nrows,double at,double rt,float clamp){uint64_t bad=0;double ma=0,min_margin=INFINITY;for(size_t i=0;i<a.size();++i){double d=fabs((double)a[i]-b[i]);mid_tolerance_terms t=mid_tolerance(wmma_gate[i],ref_gate[i],ref_up[i],weights[i/nrows],b[i],at,rt,clamp);ma=std::max(ma,d);min_margin=std::min(min_margin,t.total-d);if(!std::isfinite(a[i])||d>t.total)++bad;}printf("  %-5s bad=%llu/%zu max_abs=%.6g min_margin=%.6g\n","mid",(unsigned long long)bad,a.size(),ma,min_margin);return bad==0;}

// Print enough information to distinguish expected Q8_K -> Q8_1 conversion
// noise at a clamp boundary from a routed tile/indexing error.  Keep this in
// the validation harness: it is silent on passing cases and has no device-side
// effect on either implementation being compared.
static void diagnose_mid_mismatches(
 const char *name,const std::vector<float>&wmma_mid,const std::vector<float>&ref_mid,
 const std::vector<float>&wmma_gate,const std::vector<float>&wmma_up,
 const std::vector<float>&ref_gate_raw,const std::vector<float>&ref_up_raw,
 const std::vector<float>&ref_gate_clamped,const std::vector<float>&ref_up_clamped,
 const std::vector<float>&route_weights,const std::vector<int32_t>&selected,
 const std::vector<uint32_t>&counts,const std::vector<uint32_t>&offsets,
 const std::vector<uint32_t>&sorted_pairs,const std::vector<uint32_t>&tile_experts,
 const std::vector<uint32_t>&tile_starts,uint32_t nrows,uint32_t n_expert,
 double atol,double rtol,float clamp) {
 uint32_t shown=0;
 for(size_t i=0;i<wmma_mid.size();++i){
  double diff=fabs((double)wmma_mid[i]-ref_mid[i]);uint32_t pair=(uint32_t)(i/nrows);mid_tolerance_terms mt=mid_tolerance(wmma_gate[i],ref_gate_clamped[i],ref_up_clamped[i],route_weights[pair],ref_mid[i],atol,rtol,clamp);double tol=mt.total;
  if(std::isfinite(wmma_mid[i])&&diff<=tol)continue;
  uint32_t row=(uint32_t)(i%nrows),tok=pair/n_expert,slot=pair-tok*n_expert;
  int32_t selected_expert=selected[pair];uint32_t expert=selected_expert<0?0u:(uint32_t)selected_expert;
  uint32_t local_pair=UINT32_MAX;
  if(expert<counts.size())for(uint32_t lp=0;lp<counts[expert];++lp)if(sorted_pairs[offsets[expert]+lp]==pair){local_pair=lp;break;}
  uint32_t tile16=UINT32_MAX;
  if(local_pair!=UINT32_MAX)for(uint32_t t=0;t<tile_experts.size();++t)if(tile_experts[t]==expert&&tile_starts[t]==(local_pair/16u)*16u){tile16=t;break;}
  uint32_t tile8=0;for(uint32_t e=0;e<expert&&e<counts.size();++e)tile8+=(counts[e]+7u)/8u;
  if(local_pair!=UINT32_MAX)tile8+=local_pair/8u;else tile8=UINT32_MAX;
  float wg=wmma_gate[i],wu=wmma_up[i],rg=ref_gate_raw[i],ru=ref_up_raw[i];
  float wgc=wg>clamp?clamp:wg,wuc=wu>clamp?clamp:(wu<-clamp?-clamp:wu);
  float route_weight=route_weights[pair];
  float wmid_recomputed=(wgc/(1.0f+expf(-wgc)))*wuc*route_weight;
  float rmid_recomputed=(ref_gate_clamped[i]/(1.0f+expf(-ref_gate_clamped[i])))*ref_up_clamped[i]*route_weight;
  printf("\n  MID-MISMATCH case=%s index=%zu pair=%u row=%u token=%u slot=%u expert=%u selected=%d\n",name,i,pair,row,tok,slot,expert,selected_expert);
  printf("    route: expert_local_pair=%u count=%u sorted_offset=%u sorted_pair=%u route_weight=%.9g\n",local_pair,expert<counts.size()?counts[expert]:0u,expert<offsets.size()?offsets[expert]:UINT32_MAX,(local_pair!=UINT32_MAX&&expert<offsets.size())?sorted_pairs[offsets[expert]+local_pair]:UINT32_MAX,route_weight);
  printf("    coords: wmma_gemm(block_x=%u tile_y=%u warp=%u lane=%u) epilogue(tile_y=%u thread_x=%u) dp4a(block_x=%u tile8_y=%u writer_thread=%u)\n",row/64u,tile16,(row%64u)/16u,(local_pair%16u)+16u*((row%16u)&1u),tile16,local_pair%16u,row/32u,tile8,(row%32u)*8u);
  printf("    gate: wmma_raw=%.9g ref_raw=%.9g delta=%+.9g wmma_clamped=%.9g ref_clamped=%.9g dist_to_+clamp(wmma=%+.9g ref=%+.9g)\n",wg,rg,wg-rg,wgc,ref_gate_clamped[i],clamp-wg,clamp-rg);
  printf("    up:   wmma_raw=%.9g ref_raw=%.9g delta=%+.9g wmma_clamped=%.9g ref_clamped=%.9g dist_to_clamp(wmma+=%+.9g wmma-=%+.9g ref+=%+.9g ref-=%+.9g)\n",wu,ru,wu-ru,wuc,ref_up_clamped[i],clamp-wu,wu+clamp,clamp-ru,ru+clamp);
  printf("    mid:  wmma=%.9g ref=%.9g delta=%+.9g abs_diff=%.9g tolerance=%.9g (floor+up=%.9g gate=%.9g relative=%.9g) excess=%.9g recomputed(wmma=%.9g ref=%.9g)\n",wmma_mid[i],ref_mid[i],wmma_mid[i]-ref_mid[i],diff,tol,mt.floor_and_up,mt.gate,mt.relative,diff-tol,wmid_recomputed,rmid_recomputed);
  if(++shown==16u){printf("    (additional mismatches suppressed)\n");break;}
 }
}

struct pipeline_timing { uint32_t bucket; double crossover_us,dp4a_us; };
static std::vector<pipeline_timing> pipeline_timings;
static constexpr uint32_t kWmmaMinCount=6;

static bool run_case(const char *name,uint32_t nt,uint32_t ne,const std::vector<int32_t>&sel,uint32_t timing_bucket=0){
 const uint32_t used=6,xb=2,nr=64,pc=nt*used,wmma_min_count=kWmmaMinCount;bool mixed=!strcmp(name,"mixed");std::mt19937 rng(0x2200u+nt+ne);
 if(sel.size()!=pc){fprintf(stderr,"CASE %s invalid selection size %zu (expected %u)\n",name,sel.size(),pc);return false;}
 std::vector<cuda_block_q8_K> hx((size_t)nt*xb);for(auto&b:hx)fill_q8_K_block(&b,rng);
 std::vector<cuda_block_q4_K> hg((size_t)ne*nr*xb),hu(hg.size());for(auto&b:hg)fill_q4_K_block(&b,rng);for(auto&b:hu)fill_q4_K_block(&b,rng);
 std::vector<float> hw(pc);std::uniform_real_distribution<float>wd(.5f,1.5f);for(auto&w:hw)w=wd(rng);
 int32_t *ds;uint32_t *dc,*doo,*dcur,*dsp,*dto,*dtt,*dte,*dts,*rto,*rtt,*rte,*rts,*dispatch_counts,*gate_marks,*up_marks;cuda_block_q8_K*dx;q8_1_mmq_block*dq;cuda_block_q4_K*dg,*du;float *dwgt,*rg,*ru,*rm,*wg,*wu,*wm,*cg,*cu,*cm;size_t os=(size_t)pc*nr*sizeof(float),ms=(size_t)pc*nr*sizeof(uint32_t);
 #define HM(p,n,msg) hip_check(hipMalloc(&(p),(n)),msg)
 HM(ds,pc*sizeof(*ds),"selected");HM(dc,ne*sizeof(*dc),"counts");HM(doo,(ne+1)*sizeof(*doo),"offsets");HM(dcur,ne*sizeof(*dcur),"cursors");HM(dsp,pc*sizeof(*dsp),"pairs");HM(dto,(ne+1)*sizeof(*dto),"tile16 offsets");HM(dtt,sizeof(*dtt),"tile16 total");HM(dte,pc*sizeof(*dte),"tile16 experts");HM(dts,pc*sizeof(*dts),"tile16 starts");HM(rto,(ne+1)*sizeof(*rto),"tile8 offsets");HM(rtt,sizeof(*rtt),"tile8 total");HM(rte,pc*sizeof(*rte),"tile8 experts");HM(rts,pc*sizeof(*rts),"tile8 starts");HM(dx,hx.size()*sizeof(*dx),"xq");HM(dq,(size_t)nt*xb*2*sizeof(*dq),"q81");HM(dg,hg.size()*sizeof(*dg),"gate weights");HM(du,hu.size()*sizeof(*du),"up weights");HM(dwgt,pc*sizeof(*dwgt),"route weights");HM(rg,os,"ref gate");HM(ru,os,"ref up");HM(rm,os,"ref mid");HM(wg,os+sizeof(float),"wmma gate guard");HM(wu,os+sizeof(float),"wmma up guard");HM(wm,os+sizeof(float),"wmma mid guard");HM(cg,os+sizeof(float),"crossover gate guard");HM(cu,os+sizeof(float),"crossover up guard");HM(cm,os+sizeof(float),"crossover mid guard");HM(dispatch_counts,4*sizeof(*dispatch_counts),"dispatch counters");HM(gate_marks,ms+sizeof(uint32_t),"gate marker guard");HM(up_marks,ms+sizeof(uint32_t),"up marker guard");
 hip_check(hipMemcpy(ds,sel.data(),pc*sizeof(*ds),hipMemcpyHostToDevice),"copy selected");hip_check(hipMemcpy(dx,hx.data(),hx.size()*sizeof(*dx),hipMemcpyHostToDevice),"copy xq");hip_check(hipMemcpy(dg,hg.data(),hg.size()*sizeof(*dg),hipMemcpyHostToDevice),"copy gate");hip_check(hipMemcpy(du,hu.data(),hu.size()*sizeof(*du),hipMemcpyHostToDevice),"copy up");hip_check(hipMemcpy(dwgt,hw.data(),pc*sizeof(*dwgt),hipMemcpyHostToDevice),"copy weights");hip_check(hipMemset(dc,0,ne*sizeof(*dc)),"clear counts");
 moe_count_sorted_pairs_kernel<<<(pc+255)/256,256>>>(dc,ds,pc,ne);moe_prefix_sorted_pairs_kernel<<<1,1>>>(doo,dcur,dc,ne);moe_scatter_sorted_pairs_deterministic_kernel<<<ne,1>>>(dsp,doo,ds,pc,ne);moe_build_expert_tile_offsets_kernel<<<1,1>>>(dto,dtt,dc,16,ne);moe_build_expert_tiles_kernel<<<(ne+255)/256,256>>>(dte,dts,dto,dc,16,ne);moe_build_expert_tile_offsets_kernel<<<1,1>>>(rto,rtt,dc,8,ne);moe_build_expert_tiles_kernel<<<(ne+255)/256,256>>>(rte,rts,rto,dc,8,ne);hip_check(hipDeviceSynchronize(),"routing builders");
 std::vector<uint32_t> hc(ne),ho(ne+1),hp(pc),hte(pc),hts(pc);uint32_t tiles;hip_check(hipMemcpy(hc.data(),dc,ne*4,hipMemcpyDeviceToHost),"counts back");hip_check(hipMemcpy(ho.data(),doo,(ne+1)*4,hipMemcpyDeviceToHost),"offsets back");hip_check(hipMemcpy(hp.data(),dsp,pc*4,hipMemcpyDeviceToHost),"pairs back");hip_check(hipMemcpy(&tiles,dtt,4,hipMemcpyDeviceToHost),"tiles back");hip_check(hipMemcpy(hte.data(),dte,tiles*4,hipMemcpyDeviceToHost),"experts back");hip_check(hipMemcpy(hts.data(),dts,tiles*4,hipMemcpyDeviceToHost),"starts back");
 bool route=ho[ne]==pc;uint32_t expected_tiles=0;for(uint32_t e=0;e<ne;++e){uint32_t k=0;for(uint32_t p=0;p<pc;++p)if((uint32_t)std::max(sel[p],0)==e){route&=hp[ho[e]+k]==p;++k;}route&=k==hc[e];for(uint32_t start=0;start<hc[e];start+=16u){route&=expected_tiles<tiles&&hte[expected_tiles]==e&&hts[expected_tiles]==start;++expected_tiles;}}route&=expected_tiles==tiles;
 uint32_t rtiles;hip_check(hipMemcpy(&rtiles,rtt,4,hipMemcpyDeviceToHost),"tile8 total back");hip_check(hipMemset(rg,0xff,os),"poison ref gate");hip_check(hipMemset(ru,0xff,os),"poison ref up");hip_check(hipMemset(rm,0xff,os),"poison ref mid");dim3 dpgrid((nr+31)/32,rtiles);moe_gate_up_mid_q4K_expert_tile8_row32_kernel<<<dpgrid,256>>>(rg,ru,rm,(const char*)dg,(const char*)du,dx,dsp,doo,dc,rtt,rte,rts,dwgt,(uint64_t)nr*xb*sizeof(cuda_block_q4_K),(uint64_t)xb*sizeof(cuda_block_q4_K),xb,nr,used,0,1,0.0f);
 q8_K_to_q8_1_mmq_kernel<<<dim3(nt,xb),32>>>(dx,dq,nt,xb);hip_check(hipMemset(wg,0xff,os+4),"poison wg");hip_check(hipMemset(wu,0xff,os+4),"poison wu");dim3 wgrid((nr+63)/64,tiles);size_t sh=(16*36+64*76)*sizeof(int32_t);q4k_wmma_routed_kernel<16><<<wgrid,256,sh>>>(dg,dq,wg,dsp,doo,dc,dtt,dte,dts,nt,xb,nr,used,0,NULL,NULL);q4k_wmma_routed_kernel<16><<<wgrid,256,sh>>>(du,dq,wu,dsp,doo,dc,dtt,dte,dts,nt,xb,nr,used,0,NULL,NULL);hip_check(hipDeviceSynchronize(),"GEMM-only phase");
 std::vector<float> hrg(pc*nr),hru(pc*nr),hrm(pc*nr),hwg(pc*nr),hwu(pc*nr),hwm(pc*nr),hrgc(pc*nr),hruc(pc*nr),cpu_g,cpu_u,cpu_m;host_routed_reference(hg,hu,hx,sel,hw,ne,used,xb,nr,cpu_g,cpu_u,cpu_m);hip_check(hipMemcpy(hrg.data(),rg,os,hipMemcpyDeviceToHost),"ref gate back");hip_check(hipMemcpy(hru.data(),ru,os,hipMemcpyDeviceToHost),"ref up back");hip_check(hipMemcpy(hwg.data(),wg,os,hipMemcpyDeviceToHost),"wmma gate back");hip_check(hipMemcpy(hwu.data(),wu,os,hipMemcpyDeviceToHost),"wmma up back");uint32_t gg=0,ug=0;hip_check(hipMemcpy(&gg,(const char*)wg+os,4,hipMemcpyDeviceToHost),"gate guard back");hip_check(hipMemcpy(&ug,(const char*)wu+os,4,hipMemcpyDeviceToHost),"up guard back");route&=(gg==0xffffffffu&&ug==0xffffffffu);bool shipping=close_vec("dgate",hrg,cpu_g,.05,.01)&close_vec("dup",hru,cpu_u,.05,.01);bool gemm=shipping&&close_vec("gate",hwg,cpu_g,.05,.01)&close_vec("up",hwu,cpu_u,.05,.01);printf("CASE %-10s routing+guards=%s shipping-vs-CPU=%s GEMM-only=%s",name,route?"PASS":"FAIL",shipping?"PASS":"FAIL",gemm?"PASS":"FAIL");
 bool epi=false;if(gemm&&route){moe_gate_up_mid_q4K_expert_tile8_row32_kernel<<<dpgrid,256>>>(rg,ru,rm,(const char*)dg,(const char*)du,dx,dsp,doo,dc,rtt,rte,rts,dwgt,(uint64_t)nr*xb*sizeof(cuda_block_q4_K),(uint64_t)xb*sizeof(cuda_block_q4_K),xb,nr,used,0,1,3.0f);hip_check(hipMemset(wm,0xff,os+4),"poison wm");routed_epilogue<<<dim3(1,tiles),16>>>(wg,wu,wm,dsp,doo,dc,dtt,dte,dts,dwgt,nr,used,3.0f);hip_check(hipDeviceSynchronize(),"epilogue phase");hip_check(hipMemcpy(hrgc.data(),rg,os,hipMemcpyDeviceToHost),"clamped ref gate back");hip_check(hipMemcpy(hruc.data(),ru,os,hipMemcpyDeviceToHost),"clamped ref up back");hip_check(hipMemcpy(hrm.data(),rm,os,hipMemcpyDeviceToHost),"ref mid back");hip_check(hipMemcpy(hwm.data(),wm,os,hipMemcpyDeviceToHost),"wmma mid back");uint32_t mg=0;hip_check(hipMemcpy(&mg,(const char*)wm+os,4,hipMemcpyDeviceToHost),"mid guard back");bool shipping_mid=close_vec("dmid",hrm,cpu_m,.05,.01);std::vector<float> cpu_gc=cpu_g,cpu_uc=cpu_u;for(auto&v:cpu_gc)v=std::min(v,3.0f);for(auto&v:cpu_uc)v=std::max(-3.0f,std::min(v,3.0f));bool mid_ok=shipping_mid&&close_mid(hwm,cpu_m,hwg,cpu_gc,cpu_uc,hw,nr,.05,.01,3.0f);if(!mid_ok)diagnose_mid_mismatches(name,hwm,cpu_m,hwg,hwu,cpu_g,cpu_u,cpu_gc,cpu_uc,hw,sel,hc,ho,hp,hte,hts,nr,used,.05,.01,3.0f);epi=(mg==0xffffffffu)&&mid_ok;}printf(" epilogue=%s\n",epi?"PASS":(gemm&&route?"FAIL":"SKIP"));

 // Device-resident crossover: all three kernels launch over the same tile16
 // stream.  Additive owner markers must be exactly 1 (WMMA) or 2 (DP4A), so
 // missing, duplicate, and cross-path writes are all observable.
 hip_check(hipMemset(cg,0xff,os+4),"poison crossover gate");hip_check(hipMemset(cu,0xff,os+4),"poison crossover up");hip_check(hipMemset(cm,0xff,os+4),"poison crossover mid");hip_check(hipMemset(dispatch_counts,0,4*sizeof(*dispatch_counts)),"clear dispatch counters");hip_check(hipMemset(gate_marks,0,ms+4),"clear gate markers");hip_check(hipMemset(up_marks,0,ms+4),"clear up markers");
 q4k_wmma_routed_kernel<16><<<wgrid,256,sh>>>(dg,dq,cg,dsp,doo,dc,dtt,dte,dts,nt,xb,nr,used,wmma_min_count,dispatch_counts,gate_marks);
 q4k_wmma_routed_kernel<16><<<wgrid,256,sh>>>(du,dq,cu,dsp,doo,dc,dtt,dte,dts,nt,xb,nr,used,wmma_min_count,NULL,up_marks);
 q4k_dp4a_cold_tile16_kernel<<<dim3((nr+31)/32,tiles),256>>>(cg,cu,(const char*)dg,(const char*)du,dx,dsp,doo,dc,dtt,dte,dts,(uint64_t)nr*xb*sizeof(cuda_block_q4_K),(uint64_t)xb*sizeof(cuda_block_q4_K),xb,nr,used,wmma_min_count,dispatch_counts,gate_marks,up_marks);
 routed_epilogue<<<dim3(1,tiles),16>>>(cg,cu,cm,dsp,doo,dc,dtt,dte,dts,dwgt,nr,used,3.0f);hip_check(hipDeviceSynchronize(),"crossover phase");
 std::vector<float> hcg(pc*nr),hcu(pc*nr),hcm(pc*nr);std::vector<uint32_t> hgm(pc*nr),hum(pc*nr);uint32_t hdispatch[4];hip_check(hipMemcpy(hcg.data(),cg,os,hipMemcpyDeviceToHost),"crossover gate back");hip_check(hipMemcpy(hcu.data(),cu,os,hipMemcpyDeviceToHost),"crossover up back");hip_check(hipMemcpy(hcm.data(),cm,os,hipMemcpyDeviceToHost),"crossover mid back");hip_check(hipMemcpy(hgm.data(),gate_marks,ms,hipMemcpyDeviceToHost),"gate markers back");hip_check(hipMemcpy(hum.data(),up_marks,ms,hipMemcpyDeviceToHost),"up markers back");hip_check(hipMemcpy(hdispatch,dispatch_counts,sizeof(hdispatch),hipMemcpyDeviceToHost),"dispatch counters back");
 uint32_t cg_guard=0,cu_guard=0,cm_guard=0,gm_guard=1,um_guard=1;hip_check(hipMemcpy(&cg_guard,(const char*)cg+os,4,hipMemcpyDeviceToHost),"crossover gate guard back");hip_check(hipMemcpy(&cu_guard,(const char*)cu+os,4,hipMemcpyDeviceToHost),"crossover up guard back");hip_check(hipMemcpy(&cm_guard,(const char*)cm+os,4,hipMemcpyDeviceToHost),"crossover mid guard back");hip_check(hipMemcpy(&gm_guard,(const char*)gate_marks+ms,4,hipMemcpyDeviceToHost),"gate marker guard back");hip_check(hipMemcpy(&um_guard,(const char*)up_marks+ms,4,hipMemcpyDeviceToHost),"up marker guard back");
 uint32_t hot_tiles=0,cold_tiles=0;for(uint32_t t=0;t<tiles;++t)(hc[hte[t]]>=wmma_min_count?hot_tiles:cold_tiles)++;bool markers=cg_guard==0xffffffffu&&cu_guard==0xffffffffu&&cm_guard==0xffffffffu&&gm_guard==0u&&um_guard==0u;for(uint32_t p=0;p<pc;++p){uint32_t expected=hc[(uint32_t)std::max(sel[p],0)]>=wmma_min_count?1u:2u;for(uint32_t row=0;row<nr;++row){size_t i=(size_t)p*nr+row;markers&=hgm[i]==expected&&hum[i]==expected;}}
 bool counters=hdispatch[0]==hot_tiles&&hdispatch[1]==cold_tiles&&hdispatch[2]==cold_tiles&&hdispatch[3]==hot_tiles;std::vector<float> cpu_gc=cpu_g,cpu_uc=cpu_u;for(auto&v:cpu_gc)v=std::min(v,3.0f);for(auto&v:cpu_uc)v=std::max(-3.0f,std::min(v,3.0f));bool cross_gate=close_vec("xgate",hcg,cpu_g,.05,.01),cross_up=close_vec("xup",hcu,cpu_u,.05,.01),cross_mid=close_mid(hcm,cpu_m,hcg,cpu_gc,cpu_uc,hw,nr,.05,.01,3.0f);bool crossover=markers&&counters&&cross_gate&&cross_up&&cross_mid;printf("  crossover threshold=%u hot/cold=%u/%u counters=%u,%u,%u,%u markers=%s result=%s\n",wmma_min_count,hot_tiles,cold_tiles,hdispatch[0],hdispatch[1],hdispatch[2],hdispatch[3],markers?"PASS":"FAIL",crossover?"PASS":"FAIL");

 // Full Stage-2 pipeline metric.  Routing is already built, just as it is at
 // dispatch time.  The crossover timed unit converts each unique token once,
 // runs both routed WMMA GEMMs, the complementary DP4A fallback, and the
 // epilogue.  The baseline is the unmodified shipping tile8 fused path (it
 // consumes Q8_K directly, so it neither needs nor receives Q8_1 conversion).
 if(timing_bucket){
  BenchResult cross_time=time_launch([&](){q8_K_to_q8_1_mmq_kernel<<<dim3(nt,xb),32>>>(dx,dq,nt,xb);q4k_wmma_routed_kernel<16><<<wgrid,256,sh>>>(dg,dq,cg,dsp,doo,dc,dtt,dte,dts,nt,xb,nr,used,wmma_min_count,NULL,NULL);q4k_wmma_routed_kernel<16><<<wgrid,256,sh>>>(du,dq,cu,dsp,doo,dc,dtt,dte,dts,nt,xb,nr,used,wmma_min_count,NULL,NULL);q4k_dp4a_cold_tile16_kernel<<<dim3((nr+31)/32,tiles),256>>>(cg,cu,(const char*)dg,(const char*)du,dx,dsp,doo,dc,dtt,dte,dts,(uint64_t)nr*xb*sizeof(cuda_block_q4_K),(uint64_t)xb*sizeof(cuda_block_q4_K),xb,nr,used,wmma_min_count,NULL,NULL,NULL);routed_epilogue<<<dim3(1,tiles),16>>>(cg,cu,cm,dsp,doo,dc,dtt,dte,dts,dwgt,nr,used,3.0f);});
  BenchResult dp4a_time=time_launch([&](){moe_gate_up_mid_q4K_expert_tile8_row32_kernel<<<dpgrid,256>>>(rg,ru,rm,(const char*)dg,(const char*)du,dx,dsp,doo,dc,rtt,rte,rts,dwgt,(uint64_t)nr*xb*sizeof(cuda_block_q4_K),(uint64_t)xb*sizeof(cuda_block_q4_K),xb,nr,used,0,1,3.0f);});
  pipeline_timings.push_back({timing_bucket,cross_time.median_us,dp4a_time.median_us});
  printf("  full-pipeline bucket=%u crossover_us=%.3f shipping_dp4a_us=%.3f speedup=%.4fx retained=%s\n",timing_bucket,cross_time.median_us,dp4a_time.median_us,dp4a_time.median_us/cross_time.median_us,timing_bucket>=wmma_min_count?"yes":"no");
 }

 // On the mixed {1,3,4,8,16,40} bucket case, compare the real unconditional
 // full-stream dispatch with an idealized pre-compacted stream.  Compaction is
 // done on the host only to establish a zero-idle-workgroup timing baseline;
 // it is outside the timed region and is not part of the crossover mechanism.
 if(mixed){for(uint32_t threshold:std::vector<uint32_t>{2,kWmmaMinCount,8,16}){std::vector<uint32_t> he,hs,ce,cs;for(uint32_t t=0;t<tiles;++t){if(hc[hte[t]]>=threshold){he.push_back(hte[t]);hs.push_back(hts[t]);}else{ce.push_back(hte[t]);cs.push_back(hts[t]);}}uint32_t *dhe,*dhs,*dce,*dcs,*dhn,*dcn;HM(dhe,he.size()*4,"hot experts");HM(dhs,hs.size()*4,"hot starts");HM(dce,ce.size()*4,"cold experts");HM(dcs,cs.size()*4,"cold starts");HM(dhn,4,"hot total");HM(dcn,4,"cold total");uint32_t hn=(uint32_t)he.size(),cn=(uint32_t)ce.size();hip_check(hipMemcpy(dhe,he.data(),hn*4,hipMemcpyHostToDevice),"hot experts copy");hip_check(hipMemcpy(dhs,hs.data(),hn*4,hipMemcpyHostToDevice),"hot starts copy");hip_check(hipMemcpy(dce,ce.data(),cn*4,hipMemcpyHostToDevice),"cold experts copy");hip_check(hipMemcpy(dcs,cs.data(),cn*4,hipMemcpyHostToDevice),"cold starts copy");hip_check(hipMemcpy(dhn,&hn,4,hipMemcpyHostToDevice),"hot total copy");hip_check(hipMemcpy(dcn,&cn,4,hipMemcpyHostToDevice),"cold total copy");BenchResult actual=time_launch([&](){q4k_wmma_routed_kernel<16><<<wgrid,256,sh>>>(dg,dq,cg,dsp,doo,dc,dtt,dte,dts,nt,xb,nr,used,threshold,NULL,NULL);q4k_wmma_routed_kernel<16><<<wgrid,256,sh>>>(du,dq,cu,dsp,doo,dc,dtt,dte,dts,nt,xb,nr,used,threshold,NULL,NULL);q4k_dp4a_cold_tile16_kernel<<<dim3((nr+31)/32,tiles),256>>>(cg,cu,(const char*)dg,(const char*)du,dx,dsp,doo,dc,dtt,dte,dts,(uint64_t)nr*xb*sizeof(cuda_block_q4_K),(uint64_t)xb*sizeof(cuda_block_q4_K),xb,nr,used,threshold,NULL,NULL,NULL);});dim3 hwgrid((nr+63)/64,hn),hcgrid((nr+31)/32,cn);BenchResult ideal=time_launch([&](){q4k_wmma_routed_kernel<16><<<hwgrid,256,sh>>>(dg,dq,cg,dsp,doo,dc,dhn,dhe,dhs,nt,xb,nr,used,0,NULL,NULL);q4k_wmma_routed_kernel<16><<<hwgrid,256,sh>>>(du,dq,cu,dsp,doo,dc,dhn,dhe,dhs,nt,xb,nr,used,0,NULL,NULL);q4k_dp4a_cold_tile16_kernel<<<hcgrid,256>>>(cg,cu,(const char*)dg,(const char*)du,dx,dsp,doo,dc,dcn,dce,dcs,(uint64_t)nr*xb*sizeof(cuda_block_q4_K),(uint64_t)xb*sizeof(cuda_block_q4_K),xb,nr,used,UINT32_MAX,NULL,NULL,NULL);});printf("  crossover-overhead threshold=%u hot/cold=%u/%u dual_us=%.3f ideal_compact_us=%.3f overhead_us=%+.3f overhead_pct=%+.2f%%\n",threshold,hn,cn,actual.median_us,ideal.median_us,actual.median_us-ideal.median_us,100.0*(actual.median_us/ideal.median_us-1.0));(void)hipFree(dhe);(void)hipFree(dhs);(void)hipFree(dce);(void)hipFree(dcs);(void)hipFree(dhn);(void)hipFree(dcn);}}
 #define HF(p) (void)hipFree(p)
 HF(ds);HF(dc);HF(doo);HF(dcur);HF(dsp);HF(dto);HF(dtt);HF(dte);HF(dts);HF(rto);HF(rtt);HF(rte);HF(rts);HF(dx);HF(dq);HF(dg);HF(du);HF(dwgt);HF(rg);HF(ru);HF(rm);HF(wg);HF(wu);HF(wm);HF(cg);HF(cu);HF(cm);HF(dispatch_counts);HF(gate_marks);HF(up_marks);return route&&gemm&&epi&&crossover;
}

static std::vector<int32_t> selections_from_counts(const std::vector<uint32_t>&counts){
 std::vector<int32_t> out;for(uint32_t e=0;e<counts.size();++e)for(uint32_t i=0;i<counts[e];++i)out.push_back((int32_t)e);return out;
}

static std::vector<int32_t> formula_selections(uint32_t nt,uint32_t ne,bool skew,bool mixed){
 const uint32_t used=6,pc=nt*used;std::vector<int32_t> sel(pc);
 for(uint32_t t=0;t<nt;++t)for(uint32_t s=0;s<used;++s){uint32_t p=t*used+s;sel[p]=mixed?(int32_t)(p<1?0:p<4?1:p<8?2:p<16?3:p<32?4:5):(skew?(int32_t)(s?(t+s)%std::min(ne,7u):0):(int32_t)(p%ne));}
 return sel;
}

static std::vector<int32_t> zipf_selections(uint32_t pair_count,uint32_t ne,double exponent){
 // Best-available synthetic proxy only: this is not production telemetry.
 // std::discrete_distribution samples the exact finite Zipf law p(rank=i)
 // proportional to 1/(i+1)^s; there is no ad-hoc rank bucketing formula.
 std::vector<double> weights(ne);for(uint32_t i=0;i<ne;++i)weights[i]=1.0/std::pow((double)i+1.0,exponent);
 std::mt19937 rng(0x5a495046u);std::discrete_distribution<uint32_t> zipf(weights.begin(),weights.end());std::vector<int32_t> out(pair_count);for(auto&e:out)e=(int32_t)zipf(rng);return out;
}

static bool report_pipeline_gate(){
 constexpr uint32_t threshold=kWmmaMinCount;double log_sum=0.0;uint32_t retained=0;bool data_ok=true,no_regression=true;
 printf("\nFULL-PIPELINE GATE (threshold=%u)\n",threshold);
 for(const auto&r:pipeline_timings){bool valid=std::isfinite(r.crossover_us)&&std::isfinite(r.dp4a_us)&&r.crossover_us>0.0&&r.dp4a_us>0.0;double speedup=valid?r.dp4a_us/r.crossover_us:NAN;bool keep=r.bucket>=threshold;bool regress=keep&&valid&&r.crossover_us>1.05*r.dp4a_us;printf("  bucket=%3u retained=%s crossover_us=%9.3f dp4a_us=%9.3f speedup=%7.4fx regression_gt_5pct=%s\n",r.bucket,keep?"yes":"no",r.crossover_us,r.dp4a_us,speedup,regress?"YES":"no");if(keep){data_ok&=valid;if(valid){log_sum+=std::log(speedup);++retained;no_regression&=!regress;}}}
 if(!data_ok||retained==0){printf("  gate=UNDECIDED (missing or invalid retained-bucket measurements)\n");return false;}
 double geomean=std::exp(log_sum/retained);bool pass=geomean>=1.20&&no_regression;printf("  retained_geomean=%.4fx required>=1.20x per_bucket_regression=%s gate=%s\n",geomean,no_regression?"PASS":"FAIL",pass?"PASS":"FAIL");return pass;
}

int main(){bool ok=true;
 // Preserve the five already-validated cases, now expressed through the same
 // explicit selection-vector entry point used by the extended sweep.
 ok&=run_case("balanced",17,256,formula_selections(17,256,false,false));ok&=run_case("skewed",23,256,formula_selections(23,256,true,false));ok&=run_case("tiny-tail",3,256,formula_selections(3,256,false,false));ok&=run_case("single",16,1,formula_selections(16,1,true,false));ok&=run_case("mixed",12,256,formula_selections(12,256,false,true));
 // Exact WMMA-16 and DP4A-8 boundary counts.  The final count makes the total
 // divisible by six routing slots without changing any required edge bucket.
 ok&=run_case("tile-edges",14,16,selections_from_counts({1,7,8,9,15,16,17,11}));
 // Empty experts alternate with populated experts; populated count 6 is hot,
 // so this cannot pass merely because every nonempty bucket fell back.
 ok&=run_case("interleaved-empty",8,16,selections_from_counts({6,0,6,0,6,0,6,0,6,0,6,0,6,0,6,0}));
 ok&=run_case("all-256-active",256,256,selections_from_counts(std::vector<uint32_t>(256,6)));
 ok&=run_case("zipf-s1.2-proxy",256,256,zipf_selections(1536,256,1.2));
 // Pair indices 0..16 all select expert 0.  Consequently tokens 0 and 1 each
 // duplicate that expert in all six slots and token 2 does so in five slots;
 // duplicate selections straddle local positions 15/16, the WMMA tile edge.
 // Each pair/slot remains a distinct CPU-reference output, so the reference
 // neither coalesces duplicates nor itself double-counts an accumulated value.
 std::vector<int32_t> repeated(24,1);for(uint32_t p=0;p<17;++p)repeated[p]=0;ok&=run_case("repeated-boundary",4,8,repeated);
 // Uniform six-expert histograms provide one independently timed point per
 // bucket.  They also run the complete correctness suite before timing.
 for(uint32_t m:std::vector<uint32_t>{1,4,5,6,7,8,9,15,16,17,32,64}){char name[48];snprintf(name,sizeof(name),"timing-bucket-%u",m);ok&=run_case(name,m,16,selections_from_counts(std::vector<uint32_t>(6,m)),m);}
 bool timing_gate=report_pipeline_gate();printf("routed Stage-2 validation: %s; timing gate: %s\n",ok?"PASS":"FAIL",timing_gate?"PASS":"FAIL");return ok&&timing_gate?0:1;}
