#include "mesh/GreedyMesher.hpp"

#include "mesh/QuadFormat.hpp"

#include <algorithm>
#include <bit>
#include <cstring>

namespace vx {

namespace {
constexpr int kStep[6][3] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                             {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};

// in-plane U/V world axes per face dir, and the 4 corners' (cx,cy) in kCorner order — must
// match chunk.vert so the packed AO lines up with the emitted vertices.
constexpr int kUAxis[6][3] = {{0, 0, 1}, {0, 0, 1}, {1, 0, 0},
                              {1, 0, 0}, {1, 0, 0}, {1, 0, 0}};
constexpr int kVAxis[6][3] = {{0, 1, 0}, {0, 1, 0}, {0, 0, 1},
                              {0, 0, 1}, {0, 1, 0}, {0, 1, 0}};
constexpr int kCornerCX[6][4] = {{0, 0, 1, 1}, {1, 1, 0, 0}, {0, 0, 1, 1},
                                 {0, 1, 1, 0}, {0, 1, 1, 0}, {1, 0, 0, 1}};
constexpr int kCornerCY[6][4] = {{0, 1, 1, 0}, {0, 1, 1, 0}, {0, 1, 1, 0},
                                 {0, 0, 1, 1}, {0, 0, 1, 1}, {0, 0, 1, 1}};
} // namespace

std::uint32_t GreedyMesher::faceAO(int ax, int ay, int az, int dir) const {
    // skirt faces sit against a coarser neighbour — mostly buried in it, and where they
    // do show they're plugging a hole. shading them per-corner would only cost quads
    // (an occluded corner blocks merging), so report them fully lit and let them merge.
    if ((m_coarseNeighbors >> dir) & 1) {
        const int outer = (dir >> 1) == 0 ? ax : ((dir >> 1) == 1 ? ay : az);
        if (outer == ((dir & 1) ? -1 : kChunkSize))
            return 0xFFu;
    }
    const auto opaque = [&](int gx, int gy, int gz) {
        return blocks::opaque(m_grid[gi(gx, gy, gz)]) ? 1 : 0;
    };
    std::uint32_t packed = 0;
    for (int i = 0; i < 4; ++i) {
        const int su = kCornerCX[dir][i] ? 1 : -1;
        const int sv = kCornerCY[dir][i] ? 1 : -1;
        const int s1 = opaque(ax + su * kUAxis[dir][0], ay + su * kUAxis[dir][1],
                              az + su * kUAxis[dir][2]);
        const int s2 = opaque(ax + sv * kVAxis[dir][0], ay + sv * kVAxis[dir][1],
                              az + sv * kVAxis[dir][2]);
        const int cn = opaque(ax + su * kUAxis[dir][0] + sv * kVAxis[dir][0],
                              ay + su * kUAxis[dir][1] + sv * kVAxis[dir][1],
                              az + su * kUAxis[dir][2] + sv * kVAxis[dir][2]);
        // two flanking solids fully occlude the corner; otherwise darken per solid neighbour
        const int ao = (s1 && s2) ? 0 : (3 - s1 - s2 - cn);
        packed |= std::uint32_t(ao) << (i * 2);
    }
    return packed;
}

void GreedyMesher::addFace(int dir, int slice, int u, int v, BlockId block,
                           std::uint16_t light) {
    std::int16_t idx = m_heads[dir][slice];
    while (idx >= 0) {
        Plane& p = m_planes[std::size_t(idx)];
        if (p.block == block && p.light == light) {
            p.rows[v] |= 1u << u;
            return;
        }
        idx = p.next;
    }
    Plane p{};
    p.block = block;
    p.light = light;
    p.dir = std::uint8_t(dir);
    p.slice = std::uint8_t(slice);
    p.next = m_heads[dir][slice];
    p.rows[v] = 1u << u;
    m_heads[dir][slice] = std::int16_t(m_planes.size());
    m_planes.push_back(p);
}

void GreedyMesher::fillGrid(const Chunk& chunk, const Chunk* const* neighbors, int lod) {
    // default: air voxels lit as open sky (border faces at the world edge look right)
    std::fill(std::begin(m_grid), std::end(m_grid), blocks::Air);
    std::fill(std::begin(m_gridLight), std::end(m_gridLight), light::kFullSky);
    m_hasWater = false;

    if (lod == 0) {
        for (int z = 0; z < kChunkSize; ++z)
            for (int y = 0; y < kChunkSize; ++y)
                for (int x = 0; x < kChunkSize; ++x) {
                    const BlockId b = chunk.get(x, y, z);
                    m_grid[gi(x, y, z)] = b;
                    m_gridLight[gi(x, y, z)] = chunk.lightAt(x, y, z);
                    m_hasWater = m_hasWater || blocks::isWater(b);
                }
        // copy the six neighbour border planes into the pad so seam faces cull correctly
        const auto pad = [&](int face, int px, int py, int pz, int nx, int ny, int nz) {
            // coarser neighbour: leave the pad air. culling against voxels it isn't
            // actually drawing is what opens the cross-LOD cracks.
            if ((m_coarseNeighbors >> face) & 1)
                return;
            const Chunk* n = neighbors ? neighbors[face] : nullptr;
            if (!n)
                return;
            m_grid[gi(px, py, pz)] = n->get(nx, ny, nz);
            m_gridLight[gi(px, py, pz)] = n->lightAt(nx, ny, nz);
        };
        for (int b = 0; b < kChunkSize; ++b)
            for (int a = 0; a < kChunkSize; ++a) {
                pad(0, kChunkSize, a, b, 0, a, b);              // +X
                pad(1, -1, a, b, kChunkSize - 1, a, b);          // -X
                pad(2, a, kChunkSize, b, a, 0, b);               // +Y
                pad(3, a, -1, b, a, kChunkSize - 1, b);          // -Y
                pad(4, a, b, kChunkSize, a, b, 0);               // +Z
                pad(5, a, b, -1, a, b, kChunkSize - 1);          // -Z
            }
        // no -Y neighbour = the void below the world: unlit, so the bottom face darkens
        if (!(neighbors && neighbors[3]))
            for (int z = 0; z < kChunkSize; ++z)
                for (int x = 0; x < kChunkSize; ++x)
                    m_gridLight[gi(x, -1, z)] = 0;
        return;
    }

    // LOD: dominant block per 2^lod cube, replicated back to full resolution. cubes are
    // chunk-local but chunk size is a power of two, so they align globally -> adjacent
    // same-LOD chunks share border surfaces (coincident faces, no cracks). pads stay air
    // (isolated); cross-LOD boundaries can seam, which is acceptable at distance.
    const int f = 1 << lod;
    const int n = kChunkSize / f;
    const int cube = f * f * f;
    for (int sz = 0; sz < n; ++sz)
        for (int sy = 0; sy < n; ++sy)
            for (int sx = 0; sx < n; ++sx) {
                int counts[blocks::kBlockCount] = {0};
                int maxSky = 0, maxBlk = 0;
                for (int dz = 0; dz < f; ++dz)
                    for (int dy = 0; dy < f; ++dy)
                        for (int dx = 0; dx < f; ++dx) {
                            const BlockId b = chunk.get(sx * f + dx, sy * f + dy, sz * f + dz);
                            if (b < blocks::kBlockCount)
                                ++counts[b];
                            const std::uint16_t l = chunk.lightAt(sx * f + dx, sy * f + dy,
                                                                  sz * f + dz);
                            maxSky = std::max(maxSky, light::sky(l));
                            maxBlk = std::max(maxBlk, light::block(l));
                        }
                const int solid = cube - counts[blocks::Air];
                BlockId dom = blocks::Air;
                if (solid * 2 >= cube) { // majority solid -> keep it solid (no thin holes)
                    int best = 0;
                    for (int b = 1; b < blocks::kBlockCount; ++b)
                        if (counts[b] > best) {
                            best = counts[b];
                            dom = BlockId(b);
                        }
                }
                if (blocks::isWater(dom))
                    m_hasWater = true;
                const std::uint16_t lv = light::pack(maxBlk, maxBlk, maxBlk, maxSky);
                for (int dz = 0; dz < f; ++dz)
                    for (int dy = 0; dy < f; ++dy)
                        for (int dx = 0; dx < f; ++dx) {
                            m_grid[gi(sx * f + dx, sy * f + dy, sz * f + dz)] = dom;
                            m_gridLight[gi(sx * f + dx, sy * f + dy, sz * f + dz)] = lv;
                        }
            }
}

void GreedyMesher::buildColumns() {
    std::memset(m_opaqueX, 0, sizeof(m_opaqueX));
    std::memset(m_opaqueY, 0, sizeof(m_opaqueY));
    std::memset(m_opaqueZ, 0, sizeof(m_opaqueZ));
    std::memset(m_waterX, 0, sizeof(m_waterX));
    std::memset(m_waterY, 0, sizeof(m_waterY));
    std::memset(m_waterZ, 0, sizeof(m_waterZ));

    // bit (i+1) = voxel i; bits 0 and 33 are the neighbour pad. read the padded range.
    for (int a = 0; a < kChunkSize; ++a)
        for (int i = -1; i <= kChunkSize; ++i) {
            const std::uint64_t bit = 1ull << (i + 1);
            for (int b = 0; b < kChunkSize; ++b) {
                const BlockId bx = m_grid[gi(i, a, b)]; // column along x, (y=a,z=b)
                if (blocks::opaque(bx)) m_opaqueX[a][b] |= bit;
                else if (blocks::isWater(bx)) m_waterX[a][b] |= bit;
            }
            for (int b = 0; b < kChunkSize; ++b) {
                const BlockId by = m_grid[gi(b, i, a)]; // along y, (z=a,x=b)
                if (blocks::opaque(by)) m_opaqueY[a][b] |= bit;
                else if (blocks::isWater(by)) m_waterY[a][b] |= bit;
            }
            for (int b = 0; b < kChunkSize; ++b) {
                const BlockId bz = m_grid[gi(b, a, i)]; // along z, (y=a,x=b)
                if (blocks::opaque(bz)) m_opaqueZ[a][b] |= bit;
                else if (blocks::isWater(bz)) m_waterZ[a][b] |= bit;
            }
        }
}

void GreedyMesher::meshLayer(bool water, std::vector<std::uint64_t>& out, GreedyStats& stats) {
    m_planes.clear();
    std::memset(m_heads, 0xFF, sizeof(m_heads));

    auto (&sx)[kChunkSize][kChunkSize] = water ? m_waterX : m_opaqueX;
    auto (&sy)[kChunkSize][kChunkSize] = water ? m_waterY : m_opaqueY;
    auto (&sz)[kChunkSize][kChunkSize] = water ? m_waterZ : m_opaqueZ;

    for (int a = 0; a < kChunkSize; ++a) {
        for (int b = 0; b < kChunkSize; ++b) {
            const std::uint64_t colX = sx[a][b];
            const std::uint64_t colY = sy[a][b];
            const std::uint64_t colZ = sz[a][b];
            // opaque faces are blocked only by opaque; water faces by opaque OR water
            const std::uint64_t blkX = water ? (colX | m_opaqueX[a][b]) : colX;
            const std::uint64_t blkY = water ? (colY | m_opaqueY[a][b]) : colY;
            const std::uint64_t blkZ = water ? (colZ | m_opaqueZ[a][b]) : colZ;

            const auto scatter = [&](int dir, std::uint64_t faceBits) {
                std::uint32_t m = std::uint32_t(faceBits >> 1);
                while (m != 0) {
                    const int s = std::countr_zero(m);
                    m &= m - 1;
                    int x, y, z;
                    switch (dir >> 1) {
                        case 0: x = s, y = a, z = b; break;
                        case 1: x = b, y = s, z = a; break;
                        default: x = b, y = a, z = s; break;
                    }
                    const int ax = x + kStep[dir][0];
                    const int ay = y + kStep[dir][1];
                    const int az = z + kStep[dir][2];
                    // Water never gets a skirt. The skirt works by leaving the border pad
                    // air, which stops the WATER layer culling at that seam too — so every
                    // chunk next to a coarser one grew a wall of vertical water faces
                    // standing up out of the surface, visible as hard chunk-shaped seams
                    // across every lake. Only the opaque layer needs the crack plugged.
                    if (water && ((m_coarseNeighbors >> dir) & 1)) {
                        const int outer = (dir >> 1) == 0 ? ax : ((dir >> 1) == 1 ? ay : az);
                        if (outer == ((dir & 1) ? -1 : kChunkSize))
                            continue;
                    }
                    const BlockId block = m_grid[gi(x, y, z)];
                    const std::uint16_t lv = m_gridLight[gi(ax, ay, az)];
                    if (!water && light::sky(lv) != 0)
                        stats.sunlit = true; // some sky reaches this face
                    // a face with any corner occlusion carries per-corner AO and can't be
                    // merged (its AO would smear); emit it 1x1. fully-lit faces merge.
                    if (m_smoothAO) {
                        const std::uint32_t ao = faceAO(ax, ay, az, dir);
                        if (ao != 0xFFu) {
                            out.push_back(packQuad(x, y, z, 1, 1, dir, block, ao, lv));
                            if (water)
                                ++stats.waterQuads;
                            else {
                                ++stats.quads;
                                stats.areaPerDir[dir] += 1;
                            }
                            continue;
                        }
                    }
                    addFace(dir, s, b, a, block, lv);
                }
            };
            scatter(0, colX & ~(blkX >> 1));
            scatter(1, colX & ~(blkX << 1));
            scatter(2, colY & ~(blkY >> 1));
            scatter(3, colY & ~(blkY << 1));
            scatter(4, colZ & ~(blkZ >> 1));
            scatter(5, colZ & ~(blkZ << 1));
        }
    }

    for (Plane& p : m_planes) {
        for (int v = 0; v < kChunkSize; ++v) {
            while (p.rows[v] != 0) {
                const int u0 = std::countr_zero(p.rows[v]);
                const int w = std::countr_one(p.rows[v] >> u0);
                const std::uint32_t mask = w == 32 ? 0xFFFFFFFFu : ((1u << w) - 1u) << u0;

                int h = 1;
                while (v + h < kChunkSize && (p.rows[v + h] & mask) == mask) {
                    p.rows[v + h] &= ~mask;
                    ++h;
                }
                p.rows[v] &= ~mask;

                int x, y, z;
                switch (p.dir >> 1) {
                    case 0: x = p.slice, y = v, z = u0; break;
                    case 1: x = u0, y = p.slice, z = v; break;
                    default: x = u0, y = v, z = p.slice; break;
                }
                out.push_back(packQuad(x, y, z, w, h, p.dir, p.block, 0xFFu, p.light));
                if (water) {
                    ++stats.waterQuads;
                } else {
                    ++stats.quads;
                    stats.areaPerDir[p.dir] += std::int64_t(w) * h;
                }
            }
        }
    }
}

GreedyStats GreedyMesher::mesh(const Chunk& chunk, const Chunk* const* neighbors,
                               std::vector<std::uint64_t>& outOpaque,
                               std::vector<std::uint64_t>& outWater, int lod,
                               std::uint8_t coarseNeighbors) {
    outOpaque.clear();
    outWater.clear();
    GreedyStats stats;
    if (chunk.uniform() && chunk.uniformId() == blocks::Air)
        return stats;

    // only the LOD 0 path pads from neighbours at all, so a skirt is only ever needed there
    m_coarseNeighbors = lod == 0 ? coarseNeighbors : 0;
    m_smoothAO = (lod == 0); // near chunks only; far LOD pads with air so AO would be wrong
    fillGrid(chunk, neighbors, lod);
    buildColumns();
    meshLayer(/*water=*/false, outOpaque, stats);
    if (m_hasWater)
        meshLayer(/*water=*/true, outWater, stats);
    stats.triangles = (stats.quads + stats.waterQuads) * 2;
    return stats;
}

} // namespace vx
