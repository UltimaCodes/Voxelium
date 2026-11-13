#pragma once

#include <FastNoise/FastNoise.h>

#include <array>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace vx {

// The land of generator version 3, as heights and climate. Every value is a pure function of the
// seed and the world position, so chunks stay seamless and any thread can ask.
//
// Mountains come from plates: each cell of a warped grid moves its own way, and where they push
// together the ground rises in a long belt. The sea keeps version 2's coast and sea floor profile,
// so nothing climbs right out of a beach. Moisture blows in from the sea along the wind and is
// lost crossing high ground, so deserts lie behind ranges and deep inside continents.
class Terrain3 {
public:
    explicit Terrain3(int seed);

    struct Sample {
        float h = 0.0f;        // height of the ground
        float broad = 0.0f;    // the same without the rolling hills: the lie of the land, for climate
        float coast = 0.0f;    // roughly how many blocks inland, negative out at sea
        float desert = 0.0f;   // 0..1
        float moisture = 0.5f; // 0 parched .. 1 soaked
        float temp = 0.5f;     // 0 frozen .. 1 hot, at sea level
        float snowLine = 0.0f; // the height snow starts at here
        float water = -1.0f;   // the surface of a river or lake over this column; below h for none
        float volcanic = 0.0f; // 0..1 up a volcano's cone: its dark rock and ash
        float oasis = 0.0f;    // 0..1 in the green ring round a desert pool
    };
    // n x n points from (x0, z0), step blocks apart, row-major. step 1 carries all the detail;
    // coarser steps leave out what would alias, and are for maps and the far horizon.
    void grid(int x0, int z0, int n, int step, Sample* out) const;

private:
    // smooth leaves out everything small, for the drainage network to read the land's broad shape from
    void gridImpl(int x0, int z0, int n, int step, Sample* out, bool smooth) const;

    // Rivers. A node in every kRiverCell cell, jittered, drains to its lowest neighbour; how much land
    // drains through it sets its size. Every value is a function of the node's neighbourhood alone.
    static constexpr int kRiverCell = 384;
    static constexpr int kRiverTile = 16;   // nodes a side, per cached tile
    static constexpr int kRiverPad = 9;     // read around a tile, more than kRiverReach
    static constexpr int kRiverReach = 8;   // how far upstream a node counts what drains into it
    struct RiverNode {
        float x = 0.0f, z = 0.0f; // where it sits
        float h = 0.0f;           // the land there, broadly
        float level = 0.0f;       // its water surface
        float lake = -1.0f;       // a basin with no way out: the level it fills to
        float desert = 0.0f;      // how desert the land there is, broadly
        int acc = 0;              // nodes draining through it
        int down = -1;            // 0..7 the neighbour it drains to, -1 none
    };
    struct RiverTile {
        std::array<RiverNode, kRiverTile * kRiverTile> nodes;
    };
    std::shared_ptr<const RiverTile> riverTile(int tx, int tz) const;
    struct Reach {
        float ax, az, bx, bz;      // a stretch of river from node to node
        float levelA, levelB;      // water surface at each end, never rising downstream
        float half, depth, valley; // channel half width, depth, and how far its valley reaches
    };
    struct Lake {
        float x, z, radius, level;
        bool oasis = false; // sunk into the sand rather than filling a basin
    };
    // Every reach and lake that can touch the box, padded by how far a valley and a meander reach.
    void collectRivers(float x0, float z0, float x1, float z1, std::vector<Reach>& reaches, std::vector<Lake>& lakes) const;

    // Coarse fields, cached in tiles 1024 blocks wide and read through a cubic.
    static constexpr int kTile = 1024;
    static constexpr int kPlateStep = 32;
    static constexpr int kPlateN = kTile / kPlateStep + 4;
    static constexpr int kClimateStep = 64;
    static constexpr int kClimateN = kTile / kClimateStep + 4;
    struct PlateTile {
        std::array<float, kPlateN * kPlateN> narrow, wide, rift;
    };
    struct ClimateTile {
        std::array<float, kClimateN * kClimateN> temp, moisture;
    };
    template <class T>
    struct TileMap {
        struct Slot {
            std::int64_t key = INT64_MIN;
            std::shared_ptr<const T> tile;
        };
        mutable std::mutex lock;
        // Tiles being built right now. A thread asking for one of them waits for it rather than
        // building it again; different tiles build side by side.
        mutable std::condition_variable built;
        mutable std::vector<std::int64_t> inFlight;
        mutable std::vector<Slot> slots;
    };

    struct Plates {
        float narrow, wide, rift;
    };
    Plates platesAt(float x, float z) const;
    std::shared_ptr<const PlateTile> plateTile(int tx, int tz) const;
    std::shared_ptr<const ClimateTile> climateTile(int tx, int tz) const;
    float continentAt(float x, float z) const;

    int m_seed;
    FastNoise::SmartNode<> m_cont;
    FastNoise::SmartNode<> m_warpX, m_warpZ; // bends the plate boundaries
    FastNoise::SmartNode<> m_hills;
    FastNoise::SmartNode<> m_hilly;
    FastNoise::SmartNode<> m_plateau;
    FastNoise::SmartNode<> m_gap;    // passes and gaps along a range
    FastNoise::SmartNode<> m_ridge;  // ridgelines
    FastNoise::SmartNode<> m_duneA, m_duneB, m_swell;
    FastNoise::SmartNode<> m_tempBand;
    FastNoise::SmartNode<> m_wetNoise;

    TileMap<PlateTile> m_plateTiles;
    TileMap<ClimateTile> m_climateTiles;
    TileMap<RiverTile> m_riverTiles;
};

} // namespace vx
