#pragma once

#include "mesh/QuadFormat.hpp"
#include "world/Chunk.hpp"

#include <cstdint>
#include <vector>

namespace vx {

struct GreedyStats {
    int quads = 0;      // opaque
    int waterQuads = 0; // transparent
    int triangles = 0;
    // Any opaque face here reached by sky light. A chunk with none is sealed from the sky,
    // which means it can never be the surface nearest the sun — whatever encloses it
    // already is — so it contributes nothing to a shadow map and can be skipped entirely.
    bool sunlit = false;
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
    // coarseNeighbors: bit per face dir, set when that neighbour is meshed at a COARSER
    // LOD than this chunk. Those borders don't get culled against the neighbour's real
    // voxels (it isn't drawing them — it's drawing a downsampled surface), so the border
    // faces are emitted as a skirt that plugs the step between the two resolutions.
    GreedyStats mesh(const Chunk& chunk, const Chunk* const* neighbors,
                     std::vector<PackedQuad>& outOpaque,
                     std::vector<PackedQuad>& outWater, int lod = 0,
                     std::uint8_t coarseNeighbors = 0);

private:
    struct Plane {
        std::uint32_t rows[kChunkSize];
        BlockId block;
        std::uint32_t light; // packed r,g,b,sky (see namespace light), part of the merge key
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
    // two diagonal quads per cross-plane foliage voxel, into the opaque list
    void emitCross(std::vector<PackedQuad>& out, GreedyStats& stats);
    void meshLayer(bool water, std::vector<PackedQuad>& out, GreedyStats& stats);
    void addFace(int dir, int slice, int u, int v, BlockId block, std::uint32_t light);
    // Per-corner water surface level for the face of water voxel (x,y,z) in direction dir,
    // packed 4 corners x 3 bits. Averages the depth of the columns meeting at each corner,
    // so a sheet that thins out ramps instead of stepping once per cell. Water quads never
    // read the r/g/b bytes of their light word, so this rides there for free.
    std::uint32_t waterCorners(int x, int y, int z, int dir) const;
    // Per-corner vertex brightness (4 corners x 3 bits, 7 = full) for a face whose outer
    // (air) voxel is (ax,ay,az): ambient occlusion times the corner's own light relative
    // to the brightest corner. Each corner averages the light of the four voxels touching
    // it, skipping opaque ones, which is what turns a hard quad-to-quad step into a ramp.
    //
    // outLight receives the per-channel brightest corner light — the value the quad
    // stores, so every corner's 0..7 only ever scales it down. kQuadShadeFull means the
    // face has no corner variation to lose and can still greedy-merge.
    std::uint32_t cornerShade(int ax, int ay, int az, int dir, std::uint32_t& outLight) const;

    BlockId m_grid[kPad * kPad * kPad];
    std::uint32_t m_gridLight[kPad * kPad * kPad];

    // Two masks, because a voxel HAVING geometry and a voxel HIDING its neighbour stopped
    // being the same question once leaves became a cutout. m_solid* is presence (leaves
    // included, so they still emit faces); m_opaque* is occlusion (leaves excluded, so the
    // faces behind them survive and you can see through the gaps).
    std::uint64_t m_solidX[kChunkSize][kChunkSize];
    std::uint64_t m_solidY[kChunkSize][kChunkSize];
    std::uint64_t m_solidZ[kChunkSize][kChunkSize];
    std::uint64_t m_opaqueX[kChunkSize][kChunkSize]; // [y][z]
    std::uint64_t m_opaqueY[kChunkSize][kChunkSize]; // [z][x]
    std::uint64_t m_opaqueZ[kChunkSize][kChunkSize]; // [y][x]
    std::uint64_t m_waterX[kChunkSize][kChunkSize];
    std::uint64_t m_waterY[kChunkSize][kChunkSize];
    std::uint64_t m_waterZ[kChunkSize][kChunkSize];
    std::int16_t m_heads[6][kChunkSize];
    std::vector<Plane> m_planes;
    bool m_hasWater = false;
    bool m_hasCross = false;
    bool m_smoothAO = false; // LOD 0 only: distant chunks don't bother (pad is air anyway)
    std::uint8_t m_coarseNeighbors = 0; // faces meshed as a cross-LOD skirt
};

} // namespace vx
