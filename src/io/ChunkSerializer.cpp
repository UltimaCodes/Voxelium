#include "io/ChunkSerializer.hpp"

namespace vx::ChunkSerializer {

namespace {

void putVarint(std::vector<std::uint8_t>& out, std::uint32_t v) {
    while (v >= 0x80) {
        out.push_back(std::uint8_t(v) | 0x80);
        v >>= 7;
    }
    out.push_back(std::uint8_t(v));
}

// Returns false if the varint runs past the end.
bool getVarint(std::span<const std::uint8_t> bytes, std::size_t& pos, std::uint32_t& out) {
    std::uint32_t result = 0;
    int shift = 0;
    while (pos < bytes.size()) {
        const std::uint8_t b = bytes[pos++];
        result |= std::uint32_t(b & 0x7F) << shift;
        if ((b & 0x80) == 0) {
            out = result;
            return true;
        }
        shift += 7;
        if (shift >= 35)
            return false; // malformed: more than 5 bytes
    }
    return false;
}

} // namespace

std::size_t serialize(const Chunk& chunk, std::vector<std::uint8_t>& out) {
    const std::size_t start = out.size();

    // Emit runs over the whole grid in index order. Reserve a slot for the run count
    // and back-patch it — one pass, no scratch array.
    const std::size_t countPos = out.size();
    out.push_back(0);
    out.push_back(0);
    out.push_back(0);
    out.push_back(0);

    std::uint32_t runs = 0;
    BlockId runId = chunk.get(0, 0, 0);
    std::uint32_t runLen = 0;
    for (int y = 0; y < kChunkSize; ++y)
        for (int z = 0; z < kChunkSize; ++z)
            for (int x = 0; x < kChunkSize; ++x) {
                const BlockId id = chunk.get(x, y, z);
                if (id == runId) {
                    ++runLen;
                } else {
                    putVarint(out, runLen);
                    putVarint(out, runId);
                    ++runs;
                    runId = id;
                    runLen = 1;
                }
            }
    putVarint(out, runLen);
    putVarint(out, runId);
    ++runs;

    // fixed 4-byte run count so the reader knows when to stop without a length prefix
    out[countPos + 0] = std::uint8_t(runs);
    out[countPos + 1] = std::uint8_t(runs >> 8);
    out[countPos + 2] = std::uint8_t(runs >> 16);
    out[countPos + 3] = std::uint8_t(runs >> 24);
    return out.size() - start;
}

bool deserialize(std::span<const std::uint8_t> bytes, Chunk& chunk) {
    if (bytes.size() < 4)
        return false;
    std::size_t pos = 0;
    const std::uint32_t runs = std::uint32_t(bytes[0]) | (std::uint32_t(bytes[1]) << 8) |
                               (std::uint32_t(bytes[2]) << 16) |
                               (std::uint32_t(bytes[3]) << 24);
    pos = 4;

    // A single run covering the grid is a uniform chunk, and it has to come back AS one.
    // Rebuilding it through set() grows the palette and leaves m_bits > 0, so the chunk is
    // byte-identical when read but no longer reports uniform() — which silently costs it
    // the whole-chunk fast paths (the light seeder's fillLight, the descent's chunk skip,
    // the mesher's all-air early-out) plus a 32 KB index array it does not need. Most of
    // the world is uniform air or uniform stone, so this is the common case, not an edge.
    if (runs == 1) {
        std::uint32_t len = 0;
        std::uint32_t id = 0;
        if (!getVarint(bytes, pos, len) || !getVarint(bytes, pos, id))
            return false;
        if (len != kChunkSize3)
            return false;
        chunk.makeUniform(BlockId(id));
        return true;
    }

    chunk.makeUniform(blocks::Air);
    std::int64_t voxel = 0;
    for (std::uint32_t r = 0; r < runs; ++r) {
        std::uint32_t len = 0;
        std::uint32_t id = 0;
        if (!getVarint(bytes, pos, len) || !getVarint(bytes, pos, id))
            return false;
        if (voxel + std::int64_t(len) > kChunkSize3)
            return false; // corrupt: runs overflow the grid
        for (std::uint32_t i = 0; i < len; ++i, ++voxel) {
            const int x = int(voxel) & 31;
            const int z = (int(voxel) >> 5) & 31;
            const int y = int(voxel) >> 10;
            chunk.set(x, y, z, BlockId(id));
        }
    }
    return voxel == kChunkSize3;
}

} // namespace vx::ChunkSerializer
