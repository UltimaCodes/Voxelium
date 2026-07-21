#pragma once

#include <ankerl/unordered_dense.h>

#include <cstdint>

namespace vx {

struct ChunkPos {
    std::int32_t x, y, z;
    bool operator==(const ChunkPos&) const = default;
};

// wyhash over the raw 12 bytes; is_avalanching tells unordered_dense the bits are
// already well mixed so it can skip its own post-mix.
struct ChunkPosHash {
    using is_avalanching = void;
    std::uint64_t operator()(const ChunkPos& p) const {
        return ankerl::unordered_dense::detail::wyhash::hash(&p, sizeof(p));
    }
};

} // namespace vx
