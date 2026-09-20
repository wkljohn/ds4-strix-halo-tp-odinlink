// Production arithmetic flags belong to this TU; the host oracle is precise.
#include <hip/hip_runtime.h>
#include <cstdint>
#include "../rocm/ds4_rocm_bf16_toktile.cuh"

extern "C" hipError_t glm5_six_rounding(uint32_t *pairs, const float *x,
        uint64_t count) {
    if (!pairs || !x || count==0u) return hipErrorInvalidValue;
    ds4_bf16_hilo_prepare_kernel<<<(count+255u)/256u,256>>>(pairs,x,count);
    return hipGetLastError();
}

template <bool NativeQkv, unsigned NTilesN = 2u>
static hipError_t launch_six(float *const *out, const uint16_t *const *w,
        const float *x, unsigned n, unsigned beta, unsigned m) {
    constexpr unsigned width = NTilesN * 16u;
    matmul_bf16_f32_wmma_hilo_kda_six_multiptr_kernel<false,false,NativeQkv,NTilesN><<<
        dim3(3u*((n+width-1u)/width)+128u+beta/2u,m/256u),512>>>(
        out[0],out[1],out[2],out[3],out[4],out[5],
        w[0],w[1],w[2],w[3],w[4],w[5],x,4096u,n,128u,beta,m);
    return hipGetLastError();
}

extern "C" hipError_t glm5_six_native(float *const *out,
        const uint16_t *const *w, const float *x, unsigned n, unsigned beta,
        unsigned m, unsigned mode) {
    if (!out || !w || !x || (n != 4096u && n != 8192u) ||
        beta != (n == 4096u ? 32u : 64u) ||
        (m != 256u && m != 1024u) || mode > 1u)
        return hipErrorInvalidValue;
    for (unsigned i=0; i<6; ++i)
        if (!out[i] || !w[i]) return hipErrorInvalidValue;
    return mode ? launch_six<true>(out,w,x,n,beta,m) :
                  launch_six<false>(out,w,x,n,beta,m);
}

// Separate component API: preserve the previously recorded native mode0/1.
// 0=native N32, 1=native N64, 2=skinny only, 3=existing fused-shared-A native.
extern "C" hipError_t glm5_six_native_geometry(float *const *out,
        const uint16_t *const *w, const float *x, unsigned n, unsigned beta,
        unsigned m, unsigned mode) {
    if (!out || !w || !x || n != 4096u || beta != 32u || m != 1024u || mode > 3u)
        return hipErrorInvalidValue;
    for (unsigned i=0; i<6; ++i)
        if (!out[i] || !w[i]) return hipErrorInvalidValue;
    if (mode==0u) return launch_six<true>(out,w,x,n,beta,m);
    if (mode==1u) return launch_six<true,4u>(out,w,x,n,beta,m);
    if (mode==2u) {
        matmul_bf16_f32_wmma_hilo_kda_six_multiptr_kernel<false,true,false><<<
            dim3(128u+beta/2u,m/256u),512>>>(
            out[0],out[1],out[2],out[3],out[4],out[5],
            w[0],w[1],w[2],w[3],w[4],w[5],x,4096u,n,128u,beta,m);
    } else {
        matmul_bf16_f32_wmma_hilo_kda_six_fused_shared_a_m256_kernel<true><<<
            dim3(n/32u,m/256u),512>>>(
            out[0],out[1],out[2],out[3],out[4],out[5],
            w[0],w[1],w[2],w[3],w[4],w[5],x,4096u,n,128u,beta,m);
    }
    return hipGetLastError();
}
