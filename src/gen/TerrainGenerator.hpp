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
    int seed() const { return m_seed; }

    // What the ground at a column is, as 0..1 weights: open sea (deeper is higher), the sandy
    // band around it, and desert. Pure function of the seed, same as everything here, so the
    // ambience director can ask where the player is standing without touching the world.
    struct BiomeSample {
        float ocean = 0.0f;
        float beach = 0.0f;
        float desert = 0.0f;
    };
    BiomeSample biomeAt(int wx, int wz) const;

    // Everything the far-terrain horizon needs about a column, in one noise evaluation: its
    // height, how desert it is, and how wooded (0 open grass .. 1 dense forest).
    struct FarSample {
        int h;
        float desert;
        float forest;
    };
    FarSample farSample(int wx, int wz) const;

private:
    // Final height of a column and how desert it is. Every place that needs to agree about the
    // ground (chunks, trees, foliage, spawn, light) goes through this one shaping function.
    struct Column {
        int h;
        float desert;
    };
    Column column(int wx, int wz) const;
    static Column shapeColumn(float heightNoise, float continent, float arid, float dune,
                              int wx, int wz, int seed);

    void placeTrees(Chunk& chunk, ChunkPos pos) const;
    void placeFoliage(Chunk& chunk, ChunkPos pos, const int (*heights)[kChunkSize],
                      const float (*desert)[kChunkSize]) const;
    void decorateSea(Chunk& chunk, int baseX, int baseY, int baseZ, int x, int z, int h) const;
    void placePalm(Chunk& chunk, ChunkPos pos, int tx, int tz, int surf,
                   std::uint32_t rv) const;

    int m_seed;
    FastNoise::SmartNode<> m_height;    // domain-warped fbm, [-1,1]
    FastNoise::SmartNode<> m_continent; // very low frequency, drives the spline
    FastNoise::SmartNode<> m_arid;      // very low frequency, where it is hot and dry
    FastNoise::SmartNode<> m_dune;      // crest lines for the desert's dunes
    FastNoise::SmartNode<> m_cheese;    // 3d fbm cave blobs
    // Worm tunnels are the intersection of these two: the places where BOTH are near zero
    // form a 1D curve through the volume, and carving a radius around it gives a tunnel.
    // One ridged field on its own only ever gives sheets, which is what the old noodle was.
    FastNoise::SmartNode<> m_wormA;
    FastNoise::SmartNode<> m_wormB;
};

} // namespace vx
