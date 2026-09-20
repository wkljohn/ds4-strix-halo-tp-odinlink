#include <assert.h>
#include "ds4_glm5_dense_capture.h"
struct ds4_gpu_tensor { uint64_t bytes; };
static unsigned reads;
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *t) { return t->bytes; }
int ds4_gpu_tensor_read(const ds4_gpu_tensor *t, uint64_t off,
                       void *data, uint64_t bytes) {
    assert(off <= t->bytes && bytes <= t->bytes-off);
    ++reads; memset(data, 0, (size_t)bytes); return 1;
}
int main(int argc, char **argv) {
    (void)glm5_mla_output_capture; /* Shared capture-file helper's other caller. */
    assert(argc == 2);
    char prefix[768];
    assert(snprintf(prefix,sizeof(prefix),"%s/dense",argv[1]) > 0);
    ds4_gpu_tensor x={1024ull*4096*4}, m={1024ull*12288*4};
#define CAP(rank,layer,pos,rows,mid) glm5_dense_prefill_capture(rank,layer,pos,rows,1,2,3,&x,mid,&x)
    assert(CAP(0,0,3072,1024,&m) && reads==0);
    assert(setenv("DS4_GLM5_DENSE_CAPTURE_PREFIX",prefix,1)==0);
    assert(!CAP(0,0,3072,1024,&m));
    assert(setenv("DS4_GLM5_DENSE_CAPTURE_POS","3072",1)==0);
    assert(!CAP(0,0,3072,1024,&m));
    assert(setenv("DS4_BENCH_RUN_ID","dense-capture-test",1)==0);
    assert(CAP(0,3,3072,1024,&m) && CAP(0,0,2048,1024,&m) && reads==0);
    assert(!CAP(2,0,3072,1024,&m) && !CAP(0,0,3072,256,&m));
    ds4_gpu_tensor short_mid={m.bytes-1};
    assert(!CAP(0,0,3072,1024,&short_mid) && reads==0);
    assert(CAP(0,0,3072,1024,&m) && reads==3);
    assert(!CAP(0,0,3072,1024,&m));
    assert(setenv("DS4_GLM5_DENSE_CAPTURE_POS","3072junk",1)==0);
    assert(!CAP(0,1,3072,1024,&m));
    assert(setenv("DS4_GLM5_DENSE_CAPTURE_POS","7168",1)==0);
    assert(CAP(1,2,7168,1024,&m));
    puts("PASS dense capture selection/refusal/exclusive files");
    return 0;
}
