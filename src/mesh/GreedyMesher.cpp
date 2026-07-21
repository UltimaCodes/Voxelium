#include "mesh/GreedyMesher.hpp"

#include "mesh/QuadFormat.hpp"

#include <bit>
#include <cstring>

namespace vx {

void GreedyMesher::addFace(int dir, int slice, int u, int v, BlockId block) {
    std::int16_t idx = m_heads[dir][slice];
    while (idx >= 0) {
        Plane& p = m_planes[std::size_t(idx)];
        if (p.block == block) {
            p.rows[v] |= 1u << u;
            return;
        }
        idx = p.next;
    }
    Plane p{};
    p.block = block;
    p.dir = std::uint8_t(dir);
    p.slice = std::uint8_t(slice);
    p.next = m_heads[dir][slice];
    p.rows[v] = 1u << u;
    m_heads[dir][slice] = std::int16_t(m_planes.size());
    m_planes.push_back(p);
}

GreedyStats GreedyMesher::mesh(const Chunk& chunk, std::vector<std::uint64_t>& outQuads) {
    outQuads.clear();
    m_planes.clear();
    std::memset(m_alongX, 0, sizeof(m_alongX));
    std::memset(m_alongY, 0, sizeof(m_alongY));
    std::memset(m_alongZ, 0, sizeof(m_alongZ));
    std::memset(m_heads, 0xFF, sizeof(m_heads)); // all -1

    // pass 1: solidity columns
    for (int y = 0; y < kChunkSize; ++y)
        for (int z = 0; z < kChunkSize; ++z)
            for (int x = 0; x < kChunkSize; ++x)
                if (chunk.get(x, y, z) != blocks::Air) {
                    m_alongX[y][z] |= 1ull << (x + 1);
                    m_alongY[z][x] |= 1ull << (y + 1);
                    m_alongZ[y][x] |= 1ull << (z + 1);
                }

    // pass 2: face masks per column, scattered into planes. A face exists where a solid
    // bit has an air bit next to it along the axis; the shifted-AND does all 32 voxels
    // of a column at once, pad bits make the chunk border come out right.
    for (int a = 0; a < kChunkSize; ++a) {
        for (int b = 0; b < kChunkSize; ++b) {
            const std::uint64_t cx = m_alongX[a][b]; // a=y, b=z
            const std::uint64_t cy = m_alongY[a][b]; // a=z, b=x
            const std::uint64_t cz = m_alongZ[a][b]; // a=y, b=x

            const auto scatter = [&](int dir, std::uint64_t faceBits) {
                std::uint32_t m = std::uint32_t(faceBits >> 1); // drop pad: bit i = voxel i
                while (m != 0) {
                    const int s = std::countr_zero(m);
                    m &= m - 1;
                    int x, y, z;
                    switch (dir >> 1) {
                        case 0: x = s, y = a, z = b; break;
                        case 1: x = b, y = s, z = a; break;
                        default: x = b, y = a, z = s; break;
                    }
                    addFace(dir, s, b, a, chunk.get(x, y, z));
                }
            };
            scatter(0, cx & ~(cx >> 1));
            scatter(1, cx & ~(cx << 1));
            scatter(2, cy & ~(cy >> 1));
            scatter(3, cy & ~(cy << 1));
            scatter(4, cz & ~(cz >> 1));
            scatter(5, cz & ~(cz << 1));
        }
    }

    // pass 3: greedy merge. Grab the lowest run in the row, widen it down through
    // following rows while they contain the same span, clear consumed bits, emit.
    GreedyStats stats;
    for (Plane& p : m_planes) {
        for (int v = 0; v < kChunkSize; ++v) {
            while (p.rows[v] != 0) {
                const int u0 = std::countr_zero(p.rows[v]);
                const int w = std::countr_one(p.rows[v] >> u0);
                const std::uint32_t mask =
                    w == 32 ? 0xFFFFFFFFu : ((1u << w) - 1u) << u0;

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
                outQuads.push_back(packQuad(x, y, z, w, h, p.dir, p.block));
                ++stats.quads;
                stats.areaPerDir[p.dir] += std::int64_t(w) * h;
            }
        }
    }
    stats.triangles = stats.quads * 2;
    return stats;
}

} // namespace vx
