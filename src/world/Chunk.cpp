#include "world/Chunk.hpp"

#include <cmath>

namespace vx {

void Chunk::set(int x, int y, int z, BlockId id) {
    if (m_bits == 0) {
        if (id == m_palette[0])
            return;
        // leave uniform: 1-bit indices, all zero, all pointing at the old uniform id
        m_indices.assign(std::size_t(kChunkSize3) / 64, 0);
        m_bits = 1;
    }
    writeIndex(index(x, y, z), paletteIndexOf(id));
}

void Chunk::writeIndex(int voxel, std::uint32_t value) {
    const std::size_t bitPos = std::size_t(voxel) * std::size_t(m_bits);
    const std::size_t word = bitPos >> 6;
    const unsigned shift = unsigned(bitPos & 63);
    const std::uint64_t mask = (std::uint64_t(1) << m_bits) - 1;

    m_indices[word] = (m_indices[word] & ~(mask << shift)) | (std::uint64_t(value) << shift);
    if (shift + unsigned(m_bits) > 64) {
        const unsigned done = 64u - shift; // bits that landed in the first word
        m_indices[word + 1] =
            (m_indices[word + 1] & ~(mask >> done)) | (std::uint64_t(value) >> done);
    }
}

std::uint32_t Chunk::paletteIndexOf(BlockId id) {
    // Linear search: palettes are a handful of entries, a hash map would cost more.
    for (std::uint32_t i = 0; i < m_palette.size(); ++i)
        if (m_palette[i] == id)
            return i;

    const std::uint32_t idx = std::uint32_t(m_palette.size());
    if (idx == (1u << m_bits))
        repackTo(m_bits + 1);
    m_palette.push_back(id);
    return idx;
}

void Chunk::repackTo(int newBits) {
    std::vector<std::uint64_t> next((std::size_t(kChunkSize3) * std::size_t(newBits) + 63) / 64,
                                    0);
    for (int i = 0; i < kChunkSize3; ++i) {
        const std::uint32_t v = readIndex(i);
        const std::size_t bitPos = std::size_t(i) * std::size_t(newBits);
        const std::size_t word = bitPos >> 6;
        const unsigned shift = unsigned(bitPos & 63);
        next[word] |= std::uint64_t(v) << shift;
        if (shift + unsigned(newBits) > 64)
            next[word + 1] |= std::uint64_t(v) >> (64u - shift);
    }
    m_indices = std::move(next);
    m_bits = newBits;
}

void fillTestTerrain(Chunk& chunk) {
    chunk.makeUniform(blocks::Air);
    for (int z = 0; z < kChunkSize; ++z) {
        for (int x = 0; x < kChunkSize; ++x) {
            const int h = 18 + int(5.0f * std::sin(float(x) * 0.35f) +
                                   4.0f * std::cos(float(z) * 0.30f));
            for (int y = 0; y <= h; ++y) {
                BlockId id = blocks::Stone;
                if (y == h)
                    id = blocks::Grass;
                else if (y >= h - 3)
                    id = blocks::Dirt;
                chunk.set(x, y, z, id);
            }
        }
    }

    // air pocket under the surface, its walls must be meshed, the solid faces around
    // it must not; keeps hidden-face culling honest in the oracle numbers
    for (int z = 0; z < kChunkSize; ++z) {
        for (int y = 0; y < kChunkSize; ++y) {
            for (int x = 0; x < kChunkSize; ++x) {
                const float dx = float(x - 16);
                const float dy = float(y - 10);
                const float dz = float(z - 16);
                if (dx * dx + dy * dy + dz * dz < 6.0f * 6.0f)
                    chunk.set(x, y, z, blocks::Air);
            }
        }
    }
}

} // namespace vx
