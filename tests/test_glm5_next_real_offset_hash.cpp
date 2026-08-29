#include "tests/glm5_next_real_offsets.hpp"

#include <cinttypes>
#include <cstdio>
#include <cstring>

static uint64_t fnv1a(const ds4_glm5_next_model_offsets &model) {
    const uint8_t *p = reinterpret_cast<const uint8_t *>(&model);
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < sizeof(model); ++i) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s GLM5_MODEL.gguf\n", argv[0]);
        return 2;
    }
    Glm5TestGGUF gguf;
    ds4_glm5_next_model_offsets offsets{};
    if (!gguf.open_file(argv[1]) || !glm5_next_bind_real_offsets(gguf, offsets)) {
        std::fprintf(stderr, "FAIL independent GLM5 offset bind\n");
        return 1;
    }
    std::printf("%016" PRIx64 "\n", fnv1a(offsets));
    return 0;
}
