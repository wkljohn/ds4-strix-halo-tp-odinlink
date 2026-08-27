#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

#include "../rocm/ds4_rocm_glm5_kda_ref.cuh"

static void check(hipError_t e, const char *what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    constexpr uint32_t T = 3, H = 2, C = 8;
    const size_t tokens = (size_t) T * H * C;
    const size_t state_n = (size_t) H * C * C;
    std::vector<float> q(tokens), k(tokens), v(tokens), g(tokens), beta(T * H);
    std::vector<float> state(state_n), ref_state, ref_out(tokens), got(tokens);
    for (size_t i = 0; i < tokens; ++i) {
        q[i] = 0.01f * float(int(i % 13) - 6);
        k[i] = 0.02f * float(int(i % 11) - 5);
        v[i] = 0.03f * float(int(i % 7) - 3);
        g[i] = -0.02f * float(i % 5 + 1);
    }
    for (size_t i = 0; i < state_n; ++i) state[i] = 0.001f * float(i % 17);
    for (size_t i = 0; i < beta.size(); ++i) beta[i] = 0.4f + 0.01f * float(i);
    ref_state = state;
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t h = 0; h < H; ++h) {
            const size_t base = ((size_t)t * H + h) * C;
            for (uint32_t j = 0; j < C; ++j) {
                float pred = 0.0f;
                for (uint32_t i = 0; i < C; ++i)
                    pred += ref_state[((size_t)h * C + i) * C + j] * k[base + i];
                for (uint32_t i = 0; i < C; ++i)
                    ref_state[((size_t)h * C + i) * C + j] =
                        std::exp(g[base + i]) * ref_state[((size_t)h * C + i) * C + j] +
                        k[base + i] * beta[(size_t)t * H + h] * (v[base + j] - pred);
                for (uint32_t i = 0; i < C; ++i)
                    ref_out[base + j] += ref_state[((size_t)h * C + i) * C + j] * q[base + i];
            }
        }
    }
    float *dq, *dk, *dv, *dg, *db, *ds, *do_; 
    check(hipMalloc(&dq, q.size()*sizeof(float)), "q alloc");
    check(hipMalloc(&dk, k.size()*sizeof(float)), "k alloc");
    check(hipMalloc(&dv, v.size()*sizeof(float)), "v alloc");
    check(hipMalloc(&dg, g.size()*sizeof(float)), "g alloc");
    check(hipMalloc(&db, beta.size()*sizeof(float)), "beta alloc");
    check(hipMalloc(&ds, state.size()*sizeof(float)), "state alloc");
    check(hipMalloc(&do_, q.size()*sizeof(float)), "out alloc");
    check(hipMemcpy(dq,q.data(),q.size()*sizeof(float),hipMemcpyHostToDevice), "q copy");
    check(hipMemcpy(dk,k.data(),k.size()*sizeof(float),hipMemcpyHostToDevice), "k copy");
    check(hipMemcpy(dv,v.data(),v.size()*sizeof(float),hipMemcpyHostToDevice), "v copy");
    check(hipMemcpy(dg,g.data(),g.size()*sizeof(float),hipMemcpyHostToDevice), "g copy");
    check(hipMemcpy(db,beta.data(),beta.size()*sizeof(float),hipMemcpyHostToDevice), "beta copy");
    check(hipMemcpy(ds,state.data(),state.size()*sizeof(float),hipMemcpyHostToDevice), "state copy");
    check(hipMemset(do_, 0, q.size()*sizeof(float)), "out clear");
    hipLaunchKernelGGL(ds4_glm5_kda_ref_kernel, dim3(H), dim3(C), 0, 0,
                       dq,dk,dv,dg,db,ds,do_,T,H,C);
    check(hipGetLastError(), "KDA launch");
    check(hipDeviceSynchronize(), "KDA synchronize");
    check(hipMemcpy(got.data(),do_,got.size()*sizeof(float),hipMemcpyDeviceToHost), "out copy");
    check(hipMemcpy(state.data(),ds,state.size()*sizeof(float),hipMemcpyDeviceToHost), "state copy back");
    float max_err = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) max_err = std::fmax(max_err, std::fabs(got[i]-ref_out[i]));
    for (size_t i = 0; i < state.size(); ++i) max_err = std::fmax(max_err, std::fabs(state[i]-ref_state[i]));
    hipFree(dq); hipFree(dk); hipFree(dv); hipFree(dg); hipFree(db); hipFree(ds); hipFree(do_);
    if (max_err > 2e-6f) {
        std::fprintf(stderr, "FAIL GLM5 KDA reference max_err=%.9g\n", max_err);
        return 1;
    }
    std::printf("PASS GLM5 KDA ROCm reference max_err=%.9g\n", max_err);
    return 0;
}
