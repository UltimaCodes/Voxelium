#include "mesh/Connectivity.hpp"

#include <vector>

namespace vx {

namespace {

int faceTouchMask(int x, int y, int z) {
    int mask = 0;
    if (x == kChunkSize - 1) mask |= 1 << 0;
    if (x == 0)              mask |= 1 << 1;
    if (y == kChunkSize - 1) mask |= 1 << 2;
    if (y == 0)              mask |= 1 << 3;
    if (z == kChunkSize - 1) mask |= 1 << 4;
    if (z == 0)              mask |= 1 << 5;
    return mask;
}

} // namespace

std::uint16_t computeConnectivity(const Chunk& chunk) {
    if (chunk.uniform())
        return chunk.uniformId() == blocks::Air ? kAllFacesConnected : 0;

    // explicit-queue BFS, seeded only from border air voxels — interior air bubbles
    // can't connect faces, so they are never visited at all
    thread_local std::vector<std::uint16_t> queue;
    thread_local std::vector<std::uint64_t> visited; // 32768 bits
    visited.assign(kChunkSize3 / 64, 0);

    const auto isVisited = [](int idx) {
        return (visited[std::size_t(idx) >> 6] >> (idx & 63)) & 1u;
    };
    const auto markVisited = [](int idx) {
        visited[std::size_t(idx) >> 6] |= 1ull << (idx & 63);
    };

    std::uint16_t result = 0;

    const auto floodFrom = [&](int sx, int sy, int sz) {
        const int seed = Chunk::index(sx, sy, sz);
        if (isVisited(seed) || chunk.get(sx, sy, sz) != blocks::Air)
            return;
        markVisited(seed);
        queue.clear();
        queue.push_back(std::uint16_t(seed));
        int touched = 0;
        for (std::size_t head = 0; head < queue.size(); ++head) {
            const int idx = queue[head];
            const int x = idx & 31;
            const int z = (idx >> 5) & 31;
            const int y = idx >> 10;
            touched |= faceTouchMask(x, y, z);

            static constexpr int kStep[6][3] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                                {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
            for (const auto& s : kStep) {
                const int nx = x + s[0];
                const int ny = y + s[1];
                const int nz = z + s[2];
                if (nx < 0 || nx >= kChunkSize || ny < 0 || ny >= kChunkSize || nz < 0 ||
                    nz >= kChunkSize)
                    continue;
                const int nidx = Chunk::index(nx, ny, nz);
                if (isVisited(nidx) || chunk.get(nx, ny, nz) != blocks::Air)
                    continue;
                markVisited(nidx);
                queue.push_back(std::uint16_t(nidx));
            }
        }
        for (int a = 0; a < 6; ++a)
            for (int b = a + 1; b < 6; ++b)
                if ((touched & (1 << a)) && (touched & (1 << b)))
                    result |= std::uint16_t(1u << facePairIndex(a, b));
    };

    for (int v = 0; v < kChunkSize; ++v) {
        for (int u = 0; u < kChunkSize; ++u) {
            floodFrom(kChunkSize - 1, v, u); // +X plane
            floodFrom(0, v, u);              // -X
            floodFrom(u, kChunkSize - 1, v); // +Y
            floodFrom(u, 0, v);              // -Y
            floodFrom(u, v, kChunkSize - 1); // +Z
            floodFrom(u, v, 0);              // -Z
        }
    }
    return result;
}

} // namespace vx
