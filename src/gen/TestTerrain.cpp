#include "gen/TestTerrain.hpp"

#include <algorithm>
#include <cmath>

namespace vx {

static int surfaceHeight(float wx, float wz) {
    return int(72.0f + 12.0f * std::sin(wx * 0.040f) + 10.0f * std::cos(wz * 0.035f) +
               5.0f * std::sin(wx * 0.013f + wz * 0.021f));
}

void generateTerrain(Chunk& chunk, ChunkPos pos) {
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
}

} // namespace vx
