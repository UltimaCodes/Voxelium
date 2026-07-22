#include "mesh/GreedyMesher.hpp"

#include "mesh/QuadFormat.hpp"

#include <bit>
#include <cstring>

namespace vx {

namespace {
constexpr int kStep[6][3] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                             {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
}

void GreedyMesher::addFace(int dir, int slice, int u, int v, BlockId block,
                           std::uint8_t light) {
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

void GreedyMesher::buildColumns(const Chunk& chunk, const Chunk* const* neighbors) {
    std::memset(m_opaqueX, 0, sizeof(m_opaqueX));
    std::memset(m_opaqueY, 0, sizeof(m_opaqueY));
    std::memset(m_opaqueZ, 0, sizeof(m_opaqueZ));
    std::memset(m_waterX, 0, sizeof(m_waterX));
    std::memset(m_waterY, 0, sizeof(m_waterY));
    std::memset(m_waterZ, 0, sizeof(m_waterZ));
    m_hasWater = false;

    if (chunk.uniform()) {
        const BlockId id = chunk.uniformId();
        if (blocks::opaque(id)) {
            constexpr std::uint64_t kAll = 0x1FFFFFFFEull; // bits 1..32
            for (int a = 0; a < kChunkSize; ++a)
                for (int b = 0; b < kChunkSize; ++b)
                    m_opaqueX[a][b] = m_opaqueY[a][b] = m_opaqueZ[a][b] = kAll;
        } else if (blocks::isWater(id)) {
            constexpr std::uint64_t kAll = 0x1FFFFFFFEull;
            for (int a = 0; a < kChunkSize; ++a)
                for (int b = 0; b < kChunkSize; ++b)
                    m_waterX[a][b] = m_waterY[a][b] = m_waterZ[a][b] = kAll;
            m_hasWater = true;
        }
    } else {
        for (int y = 0; y < kChunkSize; ++y)
            for (int z = 0; z < kChunkSize; ++z)
                for (int x = 0; x < kChunkSize; ++x) {
                    const BlockId id = chunk.get(x, y, z);
                    if (blocks::opaque(id)) {
                        m_opaqueX[y][z] |= 1ull << (x + 1);
                        m_opaqueY[z][x] |= 1ull << (y + 1);
                        m_opaqueZ[y][x] |= 1ull << (z + 1);
                    } else if (blocks::isWater(id)) {
                        m_waterX[y][z] |= 1ull << (x + 1);
                        m_waterY[z][x] |= 1ull << (y + 1);
                        m_waterZ[y][x] |= 1ull << (z + 1);
                        m_hasWater = true;
                    }
                }
    }

    if (!neighbors)
        return;

    // neighbor pads at bit 0 (voxel -1) and bit 33 (voxel 32), for both masks
    const auto pad = [&](int face, int padBit) {
        const Chunk* n = neighbors[face];
        if (!n)
            return;
        const int fixed = padBit == 0 ? kChunkSize - 1 : 0; // sample slice of the neighbor
        for (int a = 0; a < kChunkSize; ++a)
            for (int b = 0; b < kChunkSize; ++b) {
                int x, y, z;
                switch (face >> 1) {
                    case 0: x = fixed, y = a, z = b; break; // +/-X: a=y,b=z
                    case 1: x = b, y = fixed, z = a; break; // +/-Y: a=z,b=x
                    default: x = b, y = a, z = fixed; break; // +/-Z: a=y,b=x
                }
                const BlockId id = n->get(x, y, z);
                const int axis = face >> 1;
                std::uint64_t(&opq)[kChunkSize][kChunkSize] =
                    axis == 0 ? m_opaqueX : (axis == 1 ? m_opaqueY : m_opaqueZ);
                std::uint64_t(&wat)[kChunkSize][kChunkSize] =
                    axis == 0 ? m_waterX : (axis == 1 ? m_waterY : m_waterZ);
                if (blocks::opaque(id))
                    opq[a][b] |= 1ull << padBit;
                else if (blocks::isWater(id))
                    wat[a][b] |= 1ull << padBit;
            }
    };
    pad(0, 33); pad(1, 0); pad(2, 33); pad(3, 0); pad(4, 33); pad(5, 0);
}

void GreedyMesher::meshLayer(const Chunk& chunk, const Chunk* const* neighbors, bool water,
                             std::vector<std::uint64_t>& out, GreedyStats& stats) {
    m_planes.clear();
    std::memset(m_heads, 0xFF, sizeof(m_heads));

    auto (&sx)[kChunkSize][kChunkSize] = water ? m_waterX : m_opaqueX;
    auto (&sy)[kChunkSize][kChunkSize] = water ? m_waterY : m_opaqueY;
    auto (&sz)[kChunkSize][kChunkSize] = water ? m_waterZ : m_opaqueZ;

    for (int a = 0; a < kChunkSize; ++a) {
        for (int b = 0; b < kChunkSize; ++b) {
            // self column, and the blocker column that suppresses a face. opaque faces
            // are blocked only by opaque; water faces are blocked by opaque OR water.
            const std::uint64_t colX = sx[a][b];
            const std::uint64_t colY = sy[a][b];
            const std::uint64_t colZ = sz[a][b];
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
                    std::uint8_t light;
                    if (ax >= 0 && ax < kChunkSize && ay >= 0 && ay < kChunkSize &&
                        az >= 0 && az < kChunkSize) {
                        light = chunk.lightAt(ax, ay, az);
                    } else {
                        const Chunk* n = neighbors ? neighbors[dir] : nullptr;
                        light = n ? n->lightAt(ax & 31, ay & 31, az & 31)
                                  : (dir == 3 ? std::uint8_t(0x00) : std::uint8_t(0xF0));
                    }
                    addFace(dir, s, b, a, chunk.get(x, y, z), light);
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
                out.push_back(packQuad(x, y, z, w, h, p.dir, p.block, 0, p.light));
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
                               std::vector<std::uint64_t>& outWater) {
    outOpaque.clear();
    outWater.clear();
    GreedyStats stats;
    if (chunk.uniform() && chunk.uniformId() == blocks::Air)
        return stats; // nothing to mesh

    buildColumns(chunk, neighbors);
    meshLayer(chunk, neighbors, /*water=*/false, outOpaque, stats);
    if (m_hasWater)
        meshLayer(chunk, neighbors, /*water=*/true, outWater, stats);
    stats.triangles = (stats.quads + stats.waterQuads) * 2;
    return stats;
}

} // namespace vx
