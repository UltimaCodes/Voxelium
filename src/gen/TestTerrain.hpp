#pragma once

#include "world/Chunk.hpp"
#include "world/ChunkPos.hpp"

namespace vx {

// Placeholder world gen until FastNoise2 lands in M10. Deterministic and
// chunk-independent: a column's surface height depends only on world x/z, never on
// neighboring chunks existing.
void generateTerrain(Chunk& chunk, ChunkPos pos);

} // namespace vx
