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
// match chunk.vert so the packed shade lines up with the emitted vertices.
constexpr int kUAxis[6][3] = {{0, 0, 1}, {0, 0, 1}, {1, 0, 0},
                              {1, 0, 0}, {1, 0, 0}, {1, 0, 0}};
constexpr int kVAxis[6][3] = {{0, 1, 0}, {0, 1, 0}, {0, 0, 1},
                              {0, 0, 1}, {0, 1, 0}, {0, 1, 0}};
constexpr int kCornerCX[6][4] = {{0, 0, 1, 1}, {1, 1, 0, 0}, {0, 0, 1, 1},
                                 {0, 1, 1, 0}, {0, 1, 1, 0}, {1, 0, 0, 1}};
constexpr int kCornerCY[6][4] = {{0, 1, 1, 0}, {0, 1, 1, 0}, {0, 1, 1, 0},
                                 {0, 0, 1, 1}, {0, 0, 1, 1}, {0, 0, 1, 1}};
} // namespace

std::uint32_t GreedyMesher::cornerShade(int ax, int ay, int az, int dir,
                                        std::uint16_t& outLight) const {
    outLight = m_gridLight[gi(ax, ay, az)];
    // skirt faces sit against a coarser neighbour — mostly buried in it, and where they
    // do show they're plugging a hole. shading them per-corner would only cost quads
    // (a varying corner blocks merging), so report them fully lit and let them merge.
    // The pad behind them is left air with no light, so a real average would read them
    // as pitch black anyway.
    if ((m_coarseNeighbors >> dir) & 1) {
        const int outer = (dir >> 1) == 0 ? ax : ((dir >> 1) == 1 ? ay : az);
        if (outer == ((dir & 1) ? -1 : kChunkSize))
            return kQuadShadeFull;
    }
    const auto opaque = [&](int gx, int gy, int gz) {
        return blocks::opaque(m_grid[gi(gx, gy, gz)]) ? 1 : 0;
    };

    int ao[4];
    int cr[4], cg[4], cb[4], cs[4];
    int maxR = 0, maxG = 0, maxB = 0, maxS = 0;
    for (int i = 0; i < 4; ++i) {
        const int su = kCornerCX[dir][i] ? 1 : -1;
        const int sv = kCornerCY[dir][i] ? 1 : -1;
        const int ux = ax + su * kUAxis[dir][0], uy = ay + su * kUAxis[dir][1],
                  uz = az + su * kUAxis[dir][2];
        const int vx = ax + sv * kVAxis[dir][0], vy = ay + sv * kVAxis[dir][1],
                  vz = az + sv * kVAxis[dir][2];
        const int dx = ux + sv * kVAxis[dir][0], dy = uy + sv * kVAxis[dir][1],
                  dz = uz + sv * kVAxis[dir][2];
        const int s1 = opaque(ux, uy, uz);
        const int s2 = opaque(vx, vy, vz);
        const int cn = opaque(dx, dy, dz);
        // two flanking solids fully occlude the corner; otherwise darken per solid neighbour
        ao[i] = (s1 && s2) ? 0 : (3 - s1 - s2 - cn);

        // average the light of the four voxels touching this corner, skipping the opaque
        // ones — they carry no light of their own and would drag every lit corner toward
        // black. The face's own outer voxel is transparent by construction, so the count
        // is never zero. A corner boxed in by two solids can't see the diagonal at all.
        int n = 1, r = light::red(outLight), g = light::green(outLight),
            b = light::blue(outLight), s = light::sky(outLight);
        const auto take = [&](int gx, int gy, int gz, int solid) {
            if (solid)
                return;
            const std::uint16_t l = m_gridLight[gi(gx, gy, gz)];
            r += light::red(l);
            g += light::green(l);
            b += light::blue(l);
            s += light::sky(l);
            ++n;
        };
        take(ux, uy, uz, s1);
        take(vx, vy, vz, s2);
        take(dx, dy, dz, (s1 && s2) ? 1 : cn);
        cr[i] = r / n;
        cg[i] = g / n;
        cb[i] = b / n;
        cs[i] = s / n;
        maxR = std::max(maxR, cr[i]);
        maxG = std::max(maxG, cg[i]);
        maxB = std::max(maxB, cb[i]);
        maxS = std::max(maxS, cs[i]);
    }

    outLight = light::pack(maxR, maxG, maxB, maxS);
    const int total = maxR + maxG + maxB + maxS;
    std::uint32_t packed = 0;
    for (int i = 0; i < 4; ++i) {
        // Ratio against the stored light, not against the brightest corner, so a corner
        // renders at the brightness it actually computed rather than slightly over it.
        // A face with no light at all stays uniform and merges instead of striping.
        const float ratio = total > 0 ? float(cr[i] + cg[i] + cb[i] + cs[i]) / float(total)
                                      : 1.0f;
        // AO keeps the weight it had when chunk.frag applied it as 0.68 + 0.32 * ao,
        // so folding the two terms together doesn't quietly deepen every crevice
        const float shade = ratio * (0.68f + 0.32f * float(ao[i]) / 3.0f);
        const int level = std::clamp(int(shade * float(kQuadShadeMax) + 0.5f), 0,
                                     int(kQuadShadeMax));
        packed |= std::uint32_t(level) << (i * kQuadShadeBits);
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
            const Chunk* n = neighbors ? neighbors[face] : nullptr;
            if (!n)
                return;
            const BlockId nb = n->get(nx, ny, nz);
            // Coarser neighbour: the OPAQUE layer wants this pad left as air, because
            // culling against voxels the neighbour isn't actually drawing is what opens
            // the cross-LOD cracks the skirt exists to plug.
            //
            // Water is different. It is not opaque, so letting it through the pad does not
            // suppress a single skirt face — but it does let the water layer cull against
            // the neighbour's real surface. Without it, water at a cross-LOD border sees
            // air and either walls up or (with the old blanket skip) loses its border
            // faces entirely, which is the seam that opens along every lake once LOD
            // boundaries appear. Land in the pad still reads as air here, so a genuine
            // water-meets-land edge is still emitted.
            if ((m_coarseNeighbors >> face) & 1) {
                if (blocks::isWater(nb))
                    m_grid[gi(px, py, pz)] = nb;
                return;
            }
            m_grid[gi(px, py, pz)] = nb;
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
                    const BlockId block = m_grid[gi(x, y, z)];
                    std::uint16_t lv = m_gridLight[gi(ax, ay, az)];
                    if (!water && light::sky(lv) != 0)
                        stats.sunlit = true; // some sky reaches this face
                    // A face whose corners differ — in occlusion or in light — carries
                    // per-corner shade and can't be merged (interpolating across a merged
                    // span would smear it), so emit it 1x1. Uniform faces stay mergeable,
                    // which is why open terrain and solid rock cost the same as before.
                    if (m_smoothAO) {
                        const std::uint32_t shade = cornerShade(ax, ay, az, dir, lv);
                        if (shade != kQuadShadeFull) {
                            out.push_back(packQuad(x, y, z, 1, 1, dir, block, shade, lv));
                            if (water)
                                ++stats.waterQuads;
                            else {
                                ++stats.quads;
                                stats.areaPerDir[dir] += 1;
                            }
                            continue;
                        }
                    }
                    // merge on the light the quad will actually store
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
                out.push_back(
                    packQuad(x, y, z, w, h, p.dir, p.block, kQuadShadeFull, p.light));
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
