#pragma once

#include <array>
#include <cstdint>

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

// M3 storage: flat array, 64 KB per chunk. Palette compression lands in M5 with the
// chunk manager — meshing code only goes through get(), so storage can change under it.
class Chunk {
public:
    // x/y/z in [0,32). Power-of-two size: indexing is shifts, never mul/div.
    // Layout is y-major so a horizontal slice is one contiguous 32x32 run.
    static int index(int x, int y, int z) { return x | (z << 5) | (y << 10); }

    BlockId get(int x, int y, int z) const { return m_blocks[std::size_t(index(x, y, z))]; }
    void set(int x, int y, int z, BlockId id) { m_blocks[std::size_t(index(x, y, z))] = id; }

private:
    std::array<BlockId, kChunkSize3> m_blocks{};
};

// Deterministic test terrain: sine heightfield (grass/dirt/stone) plus a carved air
// pocket. Same input for the naive mesher now and the greedy mesher in M4, so the
// before/after numbers compare the meshers, not the data.
void fillTestTerrain(Chunk& chunk);

} // namespace vx
