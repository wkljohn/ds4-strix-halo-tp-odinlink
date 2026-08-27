#ifndef DS4_TESTS_GLM5_GGUF_TEST_HPP
#define DS4_TESTS_GLM5_GGUF_TEST_HPP

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct Glm5TestCursor {
    const uint8_t *base = nullptr;
    uint64_t size = 0;
    uint64_t pos = 0;

    bool take(void *out, uint64_t bytes) {
        if (pos > size || bytes > size - pos) return false;
        std::memcpy(out, base + pos, (size_t)bytes);
        pos += bytes;
        return true;
    }
    bool u32(uint32_t &value) { return take(&value, sizeof(value)); }
    bool u64(uint64_t &value) { return take(&value, sizeof(value)); }
    bool string(std::string &value) {
        uint64_t length = 0;
        if (!u64(length) || length > size - pos || length > SIZE_MAX)
            return false;
        value.assign((const char *)base + pos, (size_t)length);
        pos += length;
        return true;
    }
    bool skip(uint64_t bytes) {
        if (bytes > size - pos) return false;
        pos += bytes;
        return true;
    }
};

static inline bool glm5_test_skip_metadata(Glm5TestCursor &cursor,
                                            uint32_t type) {
    static const uint8_t scalar_bytes[] = {1, 1, 2, 2, 4, 4, 4, 1};
    if (type < sizeof(scalar_bytes)) return cursor.skip(scalar_bytes[type]);
    if (type == 8u) {
        std::string value;
        return cursor.string(value);
    }
    if (type == 9u) {
        uint32_t element_type = 0;
        uint64_t count = 0;
        if (!cursor.u32(element_type) || !cursor.u64(count)) return false;
        for (uint64_t i = 0; i < count; ++i)
            if (!glm5_test_skip_metadata(cursor, element_type)) return false;
        return true;
    }
    if (type == 10u || type == 11u || type == 12u) return cursor.skip(8u);
    return false;
}

struct Glm5TestTensorInfo {
    std::vector<uint64_t> dims;
    uint32_t type = 0;
    uint64_t relative_offset = 0;
};

struct Glm5TestGGUF {
    int fd = -1;
    uint8_t *map = nullptr;
    uint64_t size = 0;
    uint64_t data_start = 0;
    std::unordered_map<std::string, Glm5TestTensorInfo> tensors;

    ~Glm5TestGGUF() { close_all(); }
    void close_all() {
        if (map && map != MAP_FAILED && size) munmap(map, (size_t)size);
        if (fd >= 0) close(fd);
        fd = -1;
        map = nullptr;
        size = 0;
        data_start = 0;
        tensors.clear();
    }

    bool open_file(const char *path) {
        close_all();
        fd = open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) return false;
        struct stat st = {};
        if (fstat(fd, &st) != 0 || st.st_size <= 0) return false;
        size = (uint64_t)st.st_size;
        map = (uint8_t *)mmap(nullptr, (size_t)size, PROT_READ, MAP_PRIVATE,
                              fd, 0);
        if (map == MAP_FAILED) {
            map = nullptr;
            return false;
        }
        Glm5TestCursor cursor = {map, size, 0};
        uint32_t magic = 0, version = 0;
        uint64_t tensor_count = 0, metadata_count = 0;
        if (!cursor.u32(magic) || !cursor.u32(version) ||
            !cursor.u64(tensor_count) || !cursor.u64(metadata_count) ||
            magic != UINT32_C(0x46554747) || version != 3u) return false;
        uint32_t alignment = 32u;
        for (uint64_t i = 0; i < metadata_count; ++i) {
            std::string key;
            uint32_t type = 0;
            if (!cursor.string(key) || !cursor.u32(type)) return false;
            if (key == "general.alignment" && type == 4u) {
                if (!cursor.u32(alignment) || alignment == 0u) return false;
            } else if (!glm5_test_skip_metadata(cursor, type)) {
                return false;
            }
        }
        for (uint64_t i = 0; i < tensor_count; ++i) {
            std::string name;
            uint32_t dimensions = 0;
            if (!cursor.string(name) || !cursor.u32(dimensions) ||
                dimensions == 0u || dimensions > 4u) return false;
            Glm5TestTensorInfo info;
            info.dims.resize(dimensions);
            for (uint32_t d = 0; d < dimensions; ++d)
                if (!cursor.u64(info.dims[d])) return false;
            if (!cursor.u32(info.type) || !cursor.u64(info.relative_offset))
                return false;
            tensors.emplace(std::move(name), std::move(info));
        }
        if (cursor.pos > UINT64_MAX - (alignment - 1u)) return false;
        data_start = (cursor.pos + alignment - 1u) / alignment * alignment;
        return data_start <= size;
    }

    bool tensor(const std::string &name,
                const std::vector<uint64_t> &dims,
                uint32_t type,
                uint64_t &offset) const {
        const auto found = tensors.find(name);
        if (found == tensors.end() || found->second.type != type ||
            found->second.dims != dims ||
            found->second.relative_offset > UINT64_MAX - data_start)
            return false;
        offset = data_start + found->second.relative_offset;
        return offset < size;
    }
};

#endif
