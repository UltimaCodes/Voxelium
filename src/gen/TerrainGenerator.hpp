#pragma once

#include "world/Chunk.hpp"
#include "world/ChunkPos.hpp"

#include <FastNoise/FastNoise.h>

namespace vx {

// Sea level, shared so spawn placement can keep the player above water rather than
// dropping them onto a seabed.
inline constexpr int kSeaLevelForSpawn = 62;

// Real world gen: warped fractal simplex heights, biome amplitude from a low-frequency
// continentalness noise through a spline, cheese + noodle caves, glow deposits on cave
// floors. Fully deterministic from the seed and chunk-independent — every value is a
// pure function of world coordinates. Node trees are immutable after construction, so
// one instance serves all worker threads.
class TerrainGenerator {
public:
    explicit TerrainGenerator(int seed);

    // detailed=false skips caves + glow deposits (which live underground and vanish under
    // the LOD downsample anyway) — the bulk of gen cost. surface, water and trees still
    // generate, so a coarse chunk looks right from above and nothing pops in when the
    // camera approaches and ChunkManager re-generates it detailed (only caves fill in,
    // and those are underground).
    void generate(Chunk& chunk, ChunkPos pos, bool detailed = true) const;
    int surfaceHeight(int wx, int wz) const; // for spawn placement

private:
    void placeTrees(Chunk& chunk, ChunkPos pos) const;
    void placeFoliage(Chunk& chunk, ChunkPos pos) const;

    int m_seed;
    FastNoise::SmartNode<> m_height;    // domain-warped fbm, [-1,1]
    FastNoise::SmartNode<> m_continent; // very low frequency, drives the spline
    FastNoise::SmartNode<> m_cheese;    // 3d fbm cave blobs
    // Worm tunnels are the intersection of these two: the places where BOTH are near zero
    // form a 1D curve through the volume, and carving a radius around it gives a tunnel.
    // One ridged field on its own only ever gives sheets, which is what the old noodle was.
    FastNoise::SmartNode<> m_wormA;
    FastNoise::SmartNode<> m_wormB;
};

} // namespace vx
