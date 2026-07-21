#pragma once

#include <cstdint>
#include <vector>

namespace vx {

using BlockId = std::uint16_t;

inline constexpr int kChunkSize = 32;
inline constexpr int kChunkSize3 = kChunkSize * kChunkSize * kChunkSize;

namespace blocks {
inline constexpr BlockId Air = 0;
inline constexpr BlockId Stone = 1;
inline constexpr BlockId Dirt = 2;
inline constexpr BlockId Grass = 3;
} // namespace blocks

// Palette-compressed voxel storage. A uniform chunk (bits == 0) stores one id and no
// index array — most of a world is air or deep stone and must cost nothing. Otherwise
// indices into the palette are bit-packed at ceil(log2(palette size)) bits per voxel,
// growing one bit at a time (repack) as the palette grows.
//
// Packing is straddled: index i lives at bit i*bits and may span two words. The read
// is a multiply and two shifts — no division, no per-width magic numbers. A value's
// second word is only touched when it actually straddles, which the branch guards.
//
// Palette entries are never garbage collected when blocks are overwritten; stale ids
// cost palette bits until a future repack-on-save (M12) compacts them. Fine for now.
class Chunk {
public:
    Chunk() { makeUniform(blocks::Air); }

    // x/y/z in [0,32). Power-of-two size: indexing is shifts, never mul/div.
    static int index(int x, int y, int z) { return x | (z << 5) | (y << 10); }

    void makeUniform(BlockId id) {
        m_palette.clear();
        m_palette.push_back(id);
        m_indices.clear();
        m_bits = 0;
    }

    bool uniform() const { return m_bits == 0; }
    BlockId uniformId() const { return m_palette[0]; }

    BlockId get(int x, int y, int z) const {
        if (m_bits == 0)
            return m_palette[0];
        return m_palette[readIndex(index(x, y, z))];
    }

    void set(int x, int y, int z, BlockId id);

    int paletteBits() const { return m_bits; }
    std::size_t paletteSize() const { return m_palette.size(); }
    std::size_t storageBytes() const {
        return m_indices.size() * sizeof(std::uint64_t) + m_palette.size() * sizeof(BlockId);
    }

private:
    std::uint32_t readIndex(int voxel) const {
        const std::size_t bitPos = std::size_t(voxel) * std::size_t(m_bits);
        const std::size_t word = bitPos >> 6;
        const unsigned shift = unsigned(bitPos & 63);
        std::uint64_t v = m_indices[word] >> shift;
        if (shift + unsigned(m_bits) > 64)
            v |= m_indices[word + 1] << (64u - shift);
        return std::uint32_t(v) & ((1u << m_bits) - 1u);
    }

    void writeIndex(int voxel, std::uint32_t value);
    std::uint32_t paletteIndexOf(BlockId id); // adds the id, growing bits if needed
    void repackTo(int newBits);

    std::vector<BlockId> m_palette; // [0] doubles as the uniform id when bits == 0
    std::vector<std::uint64_t> m_indices;
    int m_bits = 0;
};

// Deterministic single-chunk test fixture for the mesher oracle (see docs/M3, M4).
void fillTestTerrain(Chunk& chunk);

} // namespace vx
