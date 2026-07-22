#pragma once

#include "world/Chunk.hpp"

#include <cstdint>
#include <vector>

namespace vx {

struct GreedyStats {
    int quads = 0;      // opaque
    int waterQuads = 0; // transparent
    int triangles = 0;
    // Summed w*h per direction for the OPAQUE layer. Every merged quad covers exactly
    // the unit faces the naive mesher would emit, so these must equal its per-direction
    // face counts — that equality is the mesher's correctness oracle.
    std::int64_t areaPerDir[6] = {};
};

// Binary greedy mesher, two layers. Opaque geometry (stone/dirt/grass/wood/leaves/glow)
// and transparent geometry (water) are meshed separately into two quad lists:
//   - an opaque face exists where an opaque voxel is next to a non-opaque one (air OR
//     water — so you see the seabed through the water)
//   - a water face exists where a water voxel is next to air only (the surface + shore
//     edges); water-water and water-opaque faces never render
// Each layer runs the same three passes (solidity bitmasks -> shifted-AND face cull ->
// countr-based greedy merge). Scratch lives in the instance, reused across calls; one
// instance per worker thread.
class GreedyMesher {
public:
    GreedyStats mesh(const Chunk& chunk, const Chunk* const* neighbors,
                     std::vector<std::uint64_t>& outOpaque,
                     std::vector<std::uint64_t>& outWater);

private:
    struct Plane {
        std::uint32_t rows[kChunkSize];
        BlockId block;
        std::uint8_t light; // sky<<4 | block, part of the merge key
        std::uint8_t dir;
        std::uint8_t slice;
        std::int16_t next; // chain of planes sharing (dir, slice)
    };

    // Builds the opaque + water solidity columns (incl. neighbor pads) once.
    void buildColumns(const Chunk& chunk, const Chunk* const* neighbors);
    // Meshes one layer; `water` selects which faces to emit and how blocking works.
    void meshLayer(const Chunk& chunk, const Chunk* const* neighbors, bool water,
                   std::vector<std::uint64_t>& out, GreedyStats& stats);
    void addFace(int dir, int slice, int u, int v, BlockId block, std::uint8_t light);

    std::uint64_t m_opaqueX[kChunkSize][kChunkSize]; // [y][z]
    std::uint64_t m_opaqueY[kChunkSize][kChunkSize]; // [z][x]
    std::uint64_t m_opaqueZ[kChunkSize][kChunkSize]; // [y][x]
    std::uint64_t m_waterX[kChunkSize][kChunkSize];
    std::uint64_t m_waterY[kChunkSize][kChunkSize];
    std::uint64_t m_waterZ[kChunkSize][kChunkSize];
    std::int16_t m_heads[6][kChunkSize];
    std::vector<Plane> m_planes;
    bool m_hasWater = false;
};

} // namespace vx
