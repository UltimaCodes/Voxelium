#pragma once

#include "world/Chunk.hpp"

#include <cstdint>
#include <vector>

namespace vx {

struct GreedyStats {
    int quads = 0;      // opaque
    int waterQuads = 0; // transparent
    int triangles = 0;
    // Summed w*h per direction for the OPAQUE layer at LOD 0 — must equal the naive
    // mesher's per-direction face counts (the correctness oracle).
    std::int64_t areaPerDir[6] = {};
};

// Binary greedy mesher over a padded raw grid (34^3 = 32 voxels + 1 pad each side). The
// chunk + its neighbor borders are copied into that grid up front (fillGrid), so the
// mesher itself has no per-voxel neighbor branching.
//
// LOD: for a level > 0 the grid is filled with the chunk DOWNSAMPLED 2^lod (dominant
// block per cube, replicated back to full size). The greedy merge then collapses the
// coarsened surface into far fewer, bigger quads — distant chunks cost a fraction of the
// triangles for both the main pass and the shadow cascades, with no format/shader change.
//
// Two output layers: opaque and transparent water (see the .cpp for the culling rules).
class GreedyMesher {
public:
    GreedyStats mesh(const Chunk& chunk, const Chunk* const* neighbors,
                     std::vector<std::uint64_t>& outOpaque,
                     std::vector<std::uint64_t>& outWater, int lod = 0);

private:
    struct Plane {
        std::uint32_t rows[kChunkSize];
        BlockId block;
        std::uint8_t light; // sky<<4 | block, part of the merge key
        std::uint8_t dir;
        std::uint8_t slice;
        std::int16_t next;
    };

    static constexpr int kPad = kChunkSize + 2; // 34
    static int gi(int x, int y, int z) {
        return (x + 1) + (y + 1) * kPad + (z + 1) * kPad * kPad;
    }

    void fillGrid(const Chunk& chunk, const Chunk* const* neighbors, int lod);
    void buildColumns();
    void meshLayer(bool water, std::vector<std::uint64_t>& out, GreedyStats& stats);
    void addFace(int dir, int slice, int u, int v, BlockId block, std::uint8_t light);

    BlockId m_grid[kPad * kPad * kPad];
    std::uint8_t m_gridLight[kPad * kPad * kPad];

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
