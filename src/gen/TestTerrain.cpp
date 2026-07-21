#include "gen/TestTerrain.hpp"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>

namespace vx {

static int surfaceHeight(float wx, float wz) {
    return int(72.0f + 12.0f * std::sin(wx * 0.040f) + 10.0f * std::cos(wz * 0.035f) +
               5.0f * std::sin(wx * 0.013f + wz * 0.021f));
}

namespace {

// deterministic glow-lit test room, carved after the terrain fill. entirely enclosed:
// sky light inside must be exactly 0, the glow block is the only source. M10's real
// caves replace this as the lighting showcase.
constexpr glm::ivec3 kRoomCenter{16, 48, 16};
constexpr int kRoomRadius = 5;
constexpr glm::ivec3 kGlowPos{16, 45, 16};

void carveTestRoom(Chunk& chunk, ChunkPos pos) {
    const glm::ivec3 base{pos.x * kChunkSize, pos.y * kChunkSize, pos.z * kChunkSize};
    const glm::ivec3 lo = kRoomCenter - kRoomRadius - base;
    const glm::ivec3 hi = kRoomCenter + kRoomRadius - base;
    if (hi.x < 0 || hi.y < 0 || hi.z < 0 || lo.x >= kChunkSize || lo.y >= kChunkSize ||
        lo.z >= kChunkSize)
        return;
    for (int z = std::max(lo.z, 0); z <= std::min(hi.z, kChunkSize - 1); ++z)
        for (int y = std::max(lo.y, 0); y <= std::min(hi.y, kChunkSize - 1); ++y)
            for (int x = std::max(lo.x, 0); x <= std::min(hi.x, kChunkSize - 1); ++x) {
                const glm::ivec3 d = base + glm::ivec3(x, y, z) - kRoomCenter;
                if (d.x * d.x + d.y * d.y + d.z * d.z <= kRoomRadius * kRoomRadius)
                    chunk.set(x, y, z, blocks::Air);
            }
    const glm::ivec3 g = kGlowPos - base;
    if (g.x >= 0 && g.x < kChunkSize && g.y >= 0 && g.y < kChunkSize && g.z >= 0 &&
        g.z < kChunkSize)
        chunk.set(g.x, g.y, g.z, blocks::Glow);
}

} // namespace

void generateTerrain(Chunk& chunk, ChunkPos pos) {
    chunk.fillLight(0); // pooled chunks carry the previous tenant's light
    const int baseY = pos.y * kChunkSize;

    int heights[kChunkSize][kChunkSize];
    int hMin = 1 << 30;
    int hMax = -(1 << 30);
    for (int z = 0; z < kChunkSize; ++z) {
        for (int x = 0; x < kChunkSize; ++x) {
            const int h = surfaceHeight(float(pos.x * kChunkSize + x),
                                        float(pos.z * kChunkSize + z));
            heights[z][x] = h;
            hMin = std::min(hMin, h);
            hMax = std::max(hMax, h);
        }
    }

    // uniform fast paths — this is where palette compression earns its keep
    if (baseY > hMax) {
        chunk.makeUniform(blocks::Air);
        return;
    }
    if (baseY + kChunkSize - 1 <= hMin - 4) {
        chunk.makeUniform(blocks::Stone);
        carveTestRoom(chunk, pos);
        return;
    }

    chunk.makeUniform(blocks::Air);
    for (int z = 0; z < kChunkSize; ++z) {
        for (int x = 0; x < kChunkSize; ++x) {
            const int h = heights[z][x];
            const int top = std::min(h - baseY, kChunkSize - 1);
            for (int y = 0; y <= top; ++y) {
                const int wy = baseY + y;
                BlockId id = blocks::Stone;
                if (wy == h)
                    id = blocks::Grass;
                else if (wy >= h - 3)
                    id = blocks::Dirt;
                chunk.set(x, y, z, id);
            }
        }
    }
    carveTestRoom(chunk, pos);
}

} // namespace vx
