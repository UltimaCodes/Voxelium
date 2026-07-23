#pragma once

#include "world/Chunk.hpp"
#include "world/ChunkPos.hpp"

#include <FastNoise/FastNoise.h>

namespace vx {

// Real world gen: warped fractal simplex heights, biome amplitude from a low-frequency
// continentalness noise through a spline, cheese + noodle caves, glow deposits on cave
// floors. Fully deterministic from the seed and chunk-independent — every value is a
// pure function of world coordinates. Node trees are immutable after construction, so
// one instance serves all worker threads.
class TerrainGenerator {
public:
    explicit TerrainGenerator(int seed);

    // detailed=false skips caves, glow deposits and trees — used for far chunks that will
    // be meshed at a downsampled LOD where those details are averaged away anyway. the
    // surface + water still generate so the coarse chunk has the right silhouette. when
    // such a chunk is approached, ChunkManager re-generates it detailed.
    void generate(Chunk& chunk, ChunkPos pos, bool detailed = true) const;
    int surfaceHeight(int wx, int wz) const; // for spawn placement

private:
    void placeTrees(Chunk& chunk, ChunkPos pos) const;

    int m_seed;
    FastNoise::SmartNode<> m_height;    // domain-warped fbm, [-1,1]
    FastNoise::SmartNode<> m_continent; // very low frequency, drives the spline
    FastNoise::SmartNode<> m_cheese;    // 3d fbm cave blobs
    FastNoise::SmartNode<> m_noodle;    // ridged thin winding tunnels
};

} // namespace vx
