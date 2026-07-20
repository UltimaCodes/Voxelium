#include "world/Chunk.hpp"

#include <cmath>

namespace vx {

void fillTestTerrain(Chunk& chunk) {
    for (int z = 0; z < kChunkSize; ++z) {
        for (int x = 0; x < kChunkSize; ++x) {
            const int h = 18 + int(5.0f * std::sin(float(x) * 0.35f) +
                                   4.0f * std::cos(float(z) * 0.30f));
            for (int y = 0; y <= h; ++y) {
                BlockId id = blocks::Stone;
                if (y == h)
                    id = blocks::Grass;
                else if (y >= h - 3)
                    id = blocks::Dirt;
                chunk.set(x, y, z, id);
            }
        }
    }

    // Air pocket under the surface — its walls must be meshed, while every solid-solid
    // face around it must not. Makes hidden-face culling correctness visible in numbers.
    for (int z = 0; z < kChunkSize; ++z) {
        for (int y = 0; y < kChunkSize; ++y) {
            for (int x = 0; x < kChunkSize; ++x) {
                const float dx = float(x - 16);
                const float dy = float(y - 10);
                const float dz = float(z - 16);
                if (dx * dx + dy * dy + dz * dz < 6.0f * 6.0f)
                    chunk.set(x, y, z, blocks::Air);
            }
        }
    }
}

} // namespace vx
