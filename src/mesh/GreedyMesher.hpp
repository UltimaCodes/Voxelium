#pragma once

#include "world/Chunk.hpp"

#include <cstdint>
#include <vector>

namespace vx {

struct GreedyStats {
    int quads = 0;
    int triangles = 0;
    // Summed w*h per direction. Every merged quad covers exactly the unit faces the
    // naive mesher would emit, so these must equal its per-direction face counts —
    // that equality is the mesher's correctness oracle.
    std::int64_t areaPerDir[6] = {};
};

// Binary greedy mesher. Three passes:
//   1. solidity bitmasks: one uint64 column per (axis, row), voxel i at bit i+1 so
//      bits 0/33 hold the neighbor pad (air until M5 wires real neighbors)
//   2. face masks via col & ~(col >> 1) / col & ~(col << 1), scattered into
//      per-(dir, slice, block) 32x32 bit planes — block is the merge key, AO and
//      light join it in M9
//   3. greedy merge each plane: countr_zero/countr_one find a run, rows extend it
// Scratch lives in the instance and is reused across calls; one instance per worker
// thread when M6 lands.
class GreedyMesher {
public:
    // neighbors: 6 pointers in dir order (+X,-X,+Y,-Y,+Z,-Z) feeding the pad bits, or
    // nullptr entries (or a null array) meaning air — border faces then get emitted,
    // which is right for a lone chunk and for the edge of the loaded world.
    GreedyStats mesh(const Chunk& chunk, const Chunk* const* neighbors,
                     std::vector<std::uint64_t>& outQuads);

private:
    struct Plane {
        std::uint32_t rows[kChunkSize];
        BlockId block;
        std::uint8_t light; // sky<<4 | block, part of the merge key like the spec says
        std::uint8_t dir;
        std::uint8_t slice;
        std::int16_t next; // chain of planes sharing (dir, slice)
    };

    void addFace(int dir, int slice, int u, int v, BlockId block, std::uint8_t light);

    std::uint64_t m_alongX[kChunkSize][kChunkSize]; // [y][z], solidity bits along x
    std::uint64_t m_alongY[kChunkSize][kChunkSize]; // [z][x], along y
    std::uint64_t m_alongZ[kChunkSize][kChunkSize]; // [y][x], along z
    std::int16_t m_heads[6][kChunkSize];            // (dir, slice) -> first plane index
    std::vector<Plane> m_planes;
};

} // namespace vx
