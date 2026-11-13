#include "gen/TerrainGenerator.hpp"
#include "jobs/JobSystem.hpp"

#include <glm/glm.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <initializer_list>
#include <thread>
#include <climits>
#include <cmath>
#include <cstring>
#include <mutex>

namespace vx {

namespace {

// continentalness -> base surface height, piecewise linear ("splines" per the spec,
// linear segments are all the control this terrain needs)
// Exponential relief, applied to the AMPLITUDE rather than to the noise.
//
// The first attempt raised 2^(detail noise), which was wrong in a way that is obvious in
// a screenshot: the detail octaves are high frequency, so exponentiating them turned every
// small wiggle into a cliff and the whole world became chaotic ridging with no plains left
// anywhere. The shape of the terrain has to stay the shape the noise describes.
//
// What should scale exponentially is how much relief a REGION gets. The continental field
// is the low-frequency "how mountainous is it here" signal, so it drives the amplitude the
// detail noise is multiplied by. Lowlands get a gentler swing than the old flat 26 and read
// as actual plains; high continent runs away into real mountains. Same mountain shapes as
// before, an order of magnitude apart in scale.
float reliefAmplitude(float c) {
    constexpr float kBase = 14.0f;     // blocks of swing in the flattest regions
    constexpr float kOctaves = 2.4f;   // ~5x more in the most mountainous
    const float t = std::clamp(c * 0.5f + 0.5f, 0.0f, 1.0f);
    return kBase * std::exp2(t * kOctaves);
}

float heightSpline(float c) {
    // 384-block world: valleys near sea level up to real mountains around 240, which with
    // the exponential amplitude on top reaches ~314, comfortably under the 376 ceiling so
    // a peak is shaped by the noise rather than sheared off by the clamp.
    //
    // The low end used to bottom out at 50, i.e. an ocean 12 blocks deep at its deepest, which
    // is a puddle. The land side (c >= -0.5) is untouched; only what lies past the coast
    // drops away, so there is a shelf, a slope and a deep floor to build a reef on.
    static constexpr float kX[] = {-1.0f, -0.75f, -0.5f, 0.0f, 0.45f, 1.0f};
    static constexpr float kY[] = {16.0f, 38.0f, 64.0f, 82.0f, 130.0f, 240.0f};
    if (c <= kX[0])
        return kY[0];
    for (int i = 1; i < 6; ++i)
        if (c <= kX[i]) {
            const float t = (c - kX[i - 1]) / (kX[i] - kX[i - 1]);
            return kY[i - 1] + t * (kY[i] - kY[i - 1]);
        }
    return kY[5];
}

// deterministic per-voxel hash for glow deposit placement
std::uint32_t voxelHash(int x, int y, int z) {
    std::uint32_t h = std::uint32_t(x) * 0x8DA6B343u ^ std::uint32_t(y) * 0xD8163841u ^
                      std::uint32_t(z) * 0xCB1AB31Fu;
    h ^= h >> 13;
    h *= 0x7FEB352Du;
    h ^= h >> 15;
    return h;
}

// Seeds run up to 2^31-1 and are offset per noise layer, multiplied into hashes and added to
// world coordinates. Done in int that overflows (undefined behaviour) for most seeds; these do
// the same arithmetic in unsigned, which wraps to exactly the bits the int version produced on
// every platform this ships on, so existing worlds are unchanged.
int seedOff(int seed, int k) { return int(std::uint32_t(seed) + std::uint32_t(k)); }
std::uint32_t wrapMul(int a, std::uint32_t k) { return std::uint32_t(a) * k; }

std::uint32_t cellHash(std::uint32_t x, std::uint32_t z) {
    std::uint32_t h = x * 0x27D4EB2Fu ^ z * 0x165667B1u;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    h *= 0x297A2D39u;
    h ^= h >> 15;
    return h;
}

std::int32_t floorDivI(std::int32_t a, std::int32_t b) {
    const std::int32_t q = a / b;
    return (a % b != 0 && (a < 0) != (b < 0)) ? q - 1 : q;
}

float smooth01(float a, float b, float x) {
    const float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// 0..1 lattice noise for the decoration masks (reef patches, kelp forests). These only decide
// WHERE something grows, at scales of tens of blocks, so a hashed lattice is plenty and a
// fraction of the cost of a FastNoise node per column.
float valueNoise(float x, float z, int salt) {
    const int ix = int(std::floor(x)), iz = int(std::floor(z));
    const float fx = x - float(ix), fz = z - float(iz);
    const float sx = fx * fx * (3.0f - 2.0f * fx), sz = fz * fz * (3.0f - 2.0f * fz);
    const auto at = [&](int cx, int cz) {
        return float(cellHash(std::uint32_t(cx) + wrapMul(salt, 7919u),
                              std::uint32_t(cz) - wrapMul(salt, 104729u)) &
                     0xFFFFu) /
               65535.0f;
    };
    const float a = at(ix, iz), b = at(ix + 1, iz), c = at(ix, iz + 1), d = at(ix + 1, iz + 1);
    return (a + (b - a) * sx) + ((c + (d - c) * sx) - (a + (b - a) * sx)) * sz;
}

// valueNoise cut at a threshold makes round blobs whose edges run straight along the lattice.
// Bent, and with two finer octaves turned off the axes, the same cut wanders like a treeline.
// x and z are in the base octave's lattice cells; the spread matches a single valueNoise.
float regionNoise(float x, float z, int salt) {
    const float bx = x + (valueNoise(x * 0.4f + 5.3f, z * 0.4f, salt + 1) - 0.5f) * 1.4f;
    const float bz = z + (valueNoise(x * 0.4f, z * 0.4f + 8.1f, salt + 2) - 0.5f) * 1.4f;
    const float v = valueNoise(bx, bz, salt) * 0.66f +
                    valueNoise(bx * 1.62f + bz * 1.18f + 3.7f, bz * 1.62f - bx * 1.18f, salt + 3) * 0.24f +
                    valueNoise(bx * 1.9f - bz * 3.5f, bz * 1.9f + bx * 3.5f + 9.2f, salt + 4) * 0.10f;
    return std::clamp(0.5f + (v - 0.5f) * 1.38f, 0.0f, 1.0f);
}

// The small patches a biome's ground is broken into, with a ragged rim rather than a clean cut.
float patchNoise(float wx, float wz, int salt) {
    const float v = valueNoise(wx * 0.09f + wz * 0.05f, wz * 0.09f - wx * 0.05f, salt) * 0.72f +
                    valueNoise(wx * 0.13f - wz * 0.21f, wz * 0.13f + wx * 0.21f, salt + 1) * 0.28f;
    const float dither = float(cellHash(std::uint32_t(int(wx)) * 7u + std::uint32_t(salt), std::uint32_t(int(wz))) & 255u) / 255.0f;
    return std::clamp(0.5f + (v - 0.5f) * 1.45f + (dither - 0.5f) * 0.12f, 0.0f, 1.0f);
}

// Flattens the band of heights around sea level into a wide, gentle shelf. Raw terrain crosses
// the waterline at whatever slope the noise happens to have there, usually a couple of blocks
// of sand and then a wall, so beaches were a fringe you could not stand on. Compressing
// [56, 70] into [57, 65.2] (slope ~0.58) leaves room to walk, and the catch-up above it
// (slope ~1.6 to 78) is the dune ridge behind the beach. Monotonic, so it can never fold the
// terrain over on itself, and the identity outside the band means mountains are untouched.
float beachRemap(float h) {
    static constexpr float kX[] = {40.0f, 56.0f, 62.0f, 70.0f, 78.0f};
    static constexpr float kY[] = {40.0f, 57.0f, 61.8f, 65.2f, 78.0f};
    if (h <= kX[0] || h >= kX[4])
        return h;
    for (int i = 1; i < 5; ++i)
        if (h <= kX[i]) {
            const float t = (h - kX[i - 1]) / (kX[i] - kX[i - 1]);
            return kY[i - 1] + t * (kY[i] - kY[i - 1]);
        }
    return h;
}

constexpr int kSeaLevel = 62;   // air at or below this over the seafloor becomes water
// Version 2 measures the continent field's slope across this many blocks either side.
constexpr float kSlopeSpan = 16.0f;
// Version 2's coast: where the continent field crosses this.
constexpr float kCoast2 = -0.06f;
// Version 2 also looks for the coast further off (see coastLook): the continent field about 480
// blocks out in eight directions, sampled on a kLookStep lattice and interpolated between, so a
// chunk and a single column read exactly the same values. Whole-block offsets keep every sample
// position exact in a float.
constexpr int kLookStep = 8;
constexpr int kLookDirs = 8;
constexpr int kLookX[kLookDirs] = {480, -480, 0, 0, 340, 340, -340, -340};
constexpr int kLookZ[kLookDirs] = {0, 0, 480, -480, 340, -340, 340, -340};

// Version 2: how far the coast is, from the field at a column (c, past kCoast2) and the field in
// each look direction (corners[k * 4 ..], the four lattice samples around the column, x then z).
// In each direction the field's drop over the distance is its average slope that way, and c over
// that is how far until it crosses the coast: a secant hundreds of blocks long, which sees the
// field curve where the local slope cannot. The nearest of the eight.
float coastLook(float c, const float* corners, float tx, float tz) {
    float best = 2000.0f;
    for (int k = 0; k < kLookDirs; ++k) {
        const float* v = corners + k * 4;
        const float a = v[0] + (v[1] - v[0]) * tx;
        const float b = v[2] + (v[3] - v[2]) * tx;
        const float drop = c - (a + (b - a) * tz - kCoast2);
        if (drop > 1e-4f)
            best = std::min(best, std::sqrt(float(kLookX[k] * kLookX[k] + kLookZ[k] * kLookZ[k])) * c / drop);
    }
    return best;
}
// Version 2's caves reach down to here; below it is the unbroken floor of the world.
constexpr int kCaveFloor2 = TerrainGenerator::kCaveFloor2;
// Version 3's go much further: they were a band from here to the surface, a layer or two of
// tunnels however far you dug.
constexpr int kCaveFloor3 = TerrainGenerator::kCaveFloor3;
int caveFloor(int version) { return version >= 3 ? kCaveFloor3 : kCaveFloor2; }
constexpr int kCaveDepth = 56;  // caves only within this many blocks below the surface
// Squared radius of the worm tunnels around the curve where both worm fields vanish.
// This is the single knob for tunnel width, and it is sharp: the carved volume goes with
// the square of it, so small changes here move the cave-air count a long way.
constexpr float kWormR2 = 0.26f;
constexpr int kTreeCell = 7; // one tree candidate per 7x7 column cell
enum TreeKind { kTreeSpruce, kTreeOak, kTreeBirch, kTreeBigOak, kTreeTallSpruce, kTreeSmallSpruce, kTreeJungle,
                kTreeBigJungle, kTreeAcacia, kTreeSwampOak, kTreeAspen, kTreeCherry, kTreeRedwood, kTreeDarkOak,
                kTreeMangrove, kTreeBamboo, kTreeMushroomRed, kTreeMushroomBrown };
// Climate thresholds between biomes, temperature at the column's height and moisture.
constexpr float kBiomeFrozen = 0.04f;
constexpr float kBiomeCold = 0.13f;
constexpr float kBiomeHot = 0.60f;
constexpr float kBiomeWarm = 0.42f;
constexpr float kBiomeJungleWet = 0.55f;
constexpr float kBiomeSavannaDry = 0.35f;
constexpr float kBiomeSwampWet = 0.68f;
constexpr float kBiomeForestWet = 0.30f;
// Scan overlap. MUST be >= the widest horizontal reach of any species or trees get sliced
// off at chunk borders, a tree is only placed by chunks that bother to consider its cell.
// Worst case is an oak: branch length 3 plus a radius-2 leaf blob on the end. A palm is
// bounded to the same figure: 2 blocks of lean plus a 4-block frond.
constexpr int kTreeCanopyR = 6;
constexpr int kSnowLine = 205;  // peaks at/above this get snow (jittered per column)
// The sandy band. Beach sand runs from just under the waterline to a few blocks above it;
// everything past that is grass.
constexpr int kBeachTop = kSeaLevel + 3;

// Version 1 worlds have a small glow-lit room under (16, 16), from when it anchored the lighting
// tests; it stays in them so their terrain never changes, and new worlds do not have it. A
// 1-block stone shell is re-solidified around it so no cave can crack it open.
constexpr glm::ivec3 kRoomCenter{16, 48, 16};
constexpr int kRoomRadius = 5;
constexpr glm::ivec3 kGlowPos{16, 45, 16};

void carveTestRoom(Chunk& chunk, ChunkPos pos) {
    const glm::ivec3 base{pos.x * kChunkSize, pos.y * kChunkSize, pos.z * kChunkSize};
    const int shell = kRoomRadius + 1;
    const glm::ivec3 lo = kRoomCenter - shell - base;
    const glm::ivec3 hi = kRoomCenter + shell - base;
    if (hi.x < 0 || hi.y < 0 || hi.z < 0 || lo.x >= kChunkSize || lo.y >= kChunkSize ||
        lo.z >= kChunkSize)
        return;
    for (int z = std::max(lo.z, 0); z <= std::min(hi.z, kChunkSize - 1); ++z)
        for (int y = std::max(lo.y, 0); y <= std::min(hi.y, kChunkSize - 1); ++y)
            for (int x = std::max(lo.x, 0); x <= std::min(hi.x, kChunkSize - 1); ++x) {
                const glm::ivec3 d = base + glm::ivec3(x, y, z) - kRoomCenter;
                const int d2 = d.x * d.x + d.y * d.y + d.z * d.z;
                if (d2 <= kRoomRadius * kRoomRadius)
                    chunk.set(x, y, z, blocks::Air);
                else if (d2 <= shell * shell)
                    chunk.set(x, y, z, blocks::Stone);
            }
    const glm::ivec3 g = kGlowPos - base;
    if (g.x >= 0 && g.x < kChunkSize && g.y >= 0 && g.y < kChunkSize && g.z >= 0 &&
        g.z < kChunkSize)
        chunk.set(g.x, g.y, g.z, blocks::Glow);
}

} // namespace

// Direct mapped and bounded: a collision only costs working the tile out again. About 60 MB when
// full, which is the whole resident ring at the longest render distance.
struct TerrainGenerator::ColumnCache {
    static constexpr std::size_t kSlots = 5003;
    static constexpr std::size_t kLocks = 64;
    struct Slot {
        int cx = INT32_MIN, cz = INT32_MIN;
        std::shared_ptr<const ColumnTile> tile;
    };
    std::vector<Slot> slots{kSlots};
    std::mutex locks[kLocks];
};

std::shared_ptr<const TerrainGenerator::ColumnTile> TerrainGenerator::columnTile(int cx, int cz) const {
    const std::size_t index = (std::size_t(std::uint32_t(cx)) * 2654435761u + std::uint32_t(cz) * 40503u) % ColumnCache::kSlots;
    ColumnCache& cache = *m_columnCache;
    std::mutex& lock = cache.locks[index % ColumnCache::kLocks];
    {
        std::lock_guard<std::mutex> g(lock);
        const auto& slot = cache.slots[index];
        if (slot.cx == cx && slot.cz == cz && slot.tile)
            return slot.tile;
    }
    constexpr int kP = kSidePad, kN = kChunkSize + 2 * kSidePad;
    // never destroyed on purpose: see computeConnectivity
    static thread_local std::vector<Terrain3::Sample>& samples = *new std::vector<Terrain3::Sample>();
    samples.resize(std::size_t(kN) * kN);
    const int x0 = cx * kChunkSize - kP, z0 = cz * kChunkSize - kP;
    m_land3->grid(x0, z0, kN, 1, samples.data());
    int ph[kN][kN];
    for (int z = 0; z < kN; ++z)
        for (int x = 0; x < kN; ++x)
            ph[z][x] = std::clamp(int(std::floor(samples[std::size_t(z) * kN + x].h)), 8, kWorldHeight - 8);
    auto tile = std::make_shared<ColumnTile>();
    for (int z = 0; z < kChunkSize; ++z)
        for (int x = 0; x < kChunkSize; ++x) {
            const Terrain3::Sample& s = samples[std::size_t(z + kP) * kN + x + kP];
            const int h = ph[z + kP][x + kP];
            int lo = h;
            for (int dz = -kP; dz <= kP; ++dz)
                for (int dx = -kP; dx <= kP; ++dx)
                    lo = std::min(lo, ph[z + kP + dz][x + kP + dx]);
            const float wx = float(x0 + kP + x), wz = float(z0 + kP + z);
            const int jitter = int(cellHash(std::uint32_t(x0 + kP + x), std::uint32_t(z0 + kP + z)) & 7u) - 3;
            const int snow = int(s.snowLine) + jitter;
            // Bare rock on steep faces, more readily the higher it is, in patches so it is outcrops
            // and not speckle. Grass gives out well below the snow, so the two never touch.
            const int rise = std::max(std::abs(ph[z + kP][x + kP + 1] - ph[z + kP][x + kP - 1]),
                                      std::abs(ph[z + kP + 1][x + kP] - ph[z + kP - 1][x + kP]));
            const int need = 5 - (h > 130) - (h > 160) - (h > 190) +
                             (valueNoise(wx * 0.07f, wz * 0.07f, seedOff(m_seed, 231)) > 0.5f ? 1 : 0);
            std::uint8_t rock = rise >= need + 2 ? 2 : rise >= need ? 1 : 0;
            const int grassTop = snow - 14 - int(valueNoise(wx * 0.05f, wz * 0.05f, seedOff(m_seed, 233)) * 18.0f);
            if (h >= grassTop && rock == 0)
                rock = 1;
            tile->h[z][x] = std::int16_t(h);
            tile->coast[z][x] = std::int16_t(std::clamp(s.coast, -2000.0f, 2000.0f));
            tile->ground[z][x] = std::int16_t(lo);
            tile->snow[z][x] = std::int16_t(snow);
            tile->water[z][x] = std::int16_t(s.water > float(h) ? int(s.water) : -1);
            if (tile->water[z][x] >= 0)
                rock = 0; // a river bed is its own thing, see generate()
            tile->desert[z][x] = std::uint8_t(std::clamp(s.desert, 0.0f, 1.0f) * 255.0f + 0.5f);
            tile->rock[z][x] = rock;
            tile->moisture[z][x] = std::uint8_t(std::clamp(s.moisture, 0.0f, 1.0f) * 255.0f + 0.5f);
            tile->temp[z][x] = std::uint8_t(std::clamp(s.temp, 0.0f, 1.0f) * 255.0f + 0.5f);
            tile->volcanic[z][x] = std::uint8_t(std::clamp(s.volcanic, 0.0f, 1.0f) * 255.0f + 0.5f);
            tile->biome[z][x] = classify(x0 + kP + x, z0 + kP + z, h, s.broad, s.desert, s.coast, s.moisture, s.temp, snow, s.oasis);
        }
    std::lock_guard<std::mutex> g(lock);
    auto& slot = cache.slots[index];
    slot.cx = cx;
    slot.cz = cz;
    slot.tile = tile;
    return tile;
}

TerrainGenerator::TerrainGenerator(int seed, int version) : m_seed(seed), m_version(version) {
    if (m_version >= 3) {
        m_land3 = std::make_shared<Terrain3>(seed);
        m_columnCache = std::make_shared<ColumnCache>();
    }
    // scales are feature sizes in blocks (SetScale == 1/frequency in this API)
    auto heightBase = FastNoise::New<FastNoise::Simplex>();
    heightBase->SetScale(215.0f);
    auto fbm = FastNoise::New<FastNoise::FractalFBm>();
    fbm->SetSource(heightBase);
    fbm->SetOctaveCount(4);
    fbm->SetLacunarity(2.0f);
    // Gain sets how much the fine octaves contribute, and with the amplitude now reaching
    // ~74 blocks in mountains the old 0.5 was the reason every slope was a 1-block
    // staircase: the fourth octave has a ~20 block wavelength and carried 12.5% of that
    // amplitude, i.e. ~9 blocks of height change across 20 blocks of ground. A gradient
    // near 1 everywhere means terrain that only ever goes up-in-up-in, so a summit is a
    // spike with no area on it and the snow line reads as a thin fringe instead of a
    // capped peak. 0.38 drops that octave to ~4 blocks over 20, so the large scales carry
    // the relief and the small ones only texture it.
    fbm->SetGain(0.38f);
    auto warp = FastNoise::New<FastNoise::DomainWarpGradient>();
    warp->SetSource(fbm);
    warp->SetScale(125.0f);
    warp->SetWarpAmplitude(28.0f);
    m_height = warp;

    auto contBase = FastNoise::New<FastNoise::Simplex>();
    contBase->SetScale(830.0f);
    auto contFbm = FastNoise::New<FastNoise::FractalFBm>();
    contFbm->SetSource(contBase);
    contFbm->SetOctaveCount(2);
    m_continent = contFbm;

    // Deserts are regions hundreds of blocks across, unrelated to continentalness, so a dry
    // coast and a dry plateau are both possible.
    auto aridBase = FastNoise::New<FastNoise::Simplex>();
    aridBase->SetScale(560.0f);
    auto aridFbm = FastNoise::New<FastNoise::FractalFBm>();
    aridFbm->SetSource(aridBase);
    aridFbm->SetOctaveCount(2);
    m_arid = aridFbm;

    auto duneBase = FastNoise::New<FastNoise::Simplex>();
    duneBase->SetScale(46.0f);
    auto duneFbm = FastNoise::New<FastNoise::FractalFBm>();
    duneFbm->SetSource(duneBase);
    duneFbm->SetOctaveCount(2);
    duneFbm->SetGain(0.4f);
    m_dune = duneFbm;

    auto cheeseBase = FastNoise::New<FastNoise::Simplex>();
    cheeseBase->SetScale(45.0f);
    auto cheese = FastNoise::New<FastNoise::FractalFBm>();
    cheese->SetSource(cheeseBase);
    cheese->SetOctaveCount(3);
    m_cheese = cheese;

    // Two independent low-frequency fields. Low because the tunnel a player walks is the
    // zero-set of these, so the scale here is the length of a corridor between bends, not
    // the width of it, width comes from the carve radius below.
    auto wormABase = FastNoise::New<FastNoise::Simplex>();
    wormABase->SetScale(96.0f);
    auto wormA = FastNoise::New<FastNoise::FractalFBm>();
    wormA->SetSource(wormABase);
    wormA->SetOctaveCount(2);
    m_wormA = wormA;

    auto wormBBase = FastNoise::New<FastNoise::Simplex>();
    wormBBase->SetScale(88.0f); // deliberately not 96: equal scales correlate the two
    auto wormB = FastNoise::New<FastNoise::FractalFBm>();
    wormB->SetSource(wormBBase);
    wormB->SetOctaveCount(2);
    m_wormB = wormB;

    // ---- version 2 ----------------------------------------------------------------------
    // Continents: features thousands of blocks across, where version 1's continent field was
    // ~800 and only reached sea level in its rare dips, which is why the old world was hills
    // with lakes in them. Warped so a coast wanders into bays and headlands instead of
    // following the round contours of the noise.
    const auto fbm2 = [](float scale, int octaves, float gain) {
        auto base = FastNoise::New<FastNoise::Simplex>();
        base->SetScale(scale);
        auto f = FastNoise::New<FastNoise::FractalFBm>();
        f->SetSource(base);
        f->SetOctaveCount(octaves);
        f->SetGain(gain);
        return f;
    };
    {
        auto warpC = FastNoise::New<FastNoise::DomainWarpGradient>();
        warpC->SetSource(fbm2(2400.0f, 4, 0.5f));
        warpC->SetScale(700.0f);
        warpC->SetWarpAmplitude(260.0f);
        m_cont2 = warpC;
    }
    // Mountain ranges are long belts: where this field crosses zero is a line running for
    // thousands of blocks, winding but continuous (warped so it does not follow round contours),
    // and a range runs along it. The first version masked a ridged field with a patchy one,
    // which gave isolated spires standing straight up out of the plains.
    {
        auto warpB = FastNoise::New<FastNoise::DomainWarpGradient>();
        warpB->SetSource(fbm2(2800.0f, 2, 0.5f));
        // Gentle enough to never fold the field over itself: along a fold its slope drops to
        // nothing, the distance read from it shoots up, and that cut straight trenches hundreds
        // of blocks long through the foothills.
        warpB->SetScale(1200.0f);
        warpB->SetWarpAmplitude(160.0f);
        m_belt = warpB;
    }
    m_beltOn = fbm2(3200.0f, 1, 0.5f); // which stretches of those lines actually rise into a range
    {
        // The range's own relief: ridged, so its crest is a ridgeline with passes and the
        // valleys between spurs, not a field of round bumps.
        auto base = FastNoise::New<FastNoise::Simplex>();
        base->SetScale(400.0f);
        auto ridged = FastNoise::New<FastNoise::FractalRidged>();
        ridged->SetSource(base);
        ridged->SetOctaveCount(3);
        ridged->SetGain(0.5f);
        m_ridge = ridged;
    }
    m_hilly = fbm2(900.0f, 2, 0.5f);
    // Deserts: one smooth field, warped, so a desert is one big coherent region with a soft edge.
    // Two climate fields multiplied together made patchwork: small sand pits wherever both
    // happened to cross their thresholds at once.
    {
        auto warpA = FastNoise::New<FastNoise::DomainWarpGradient>();
        warpA->SetSource(fbm2(2600.0f, 1, 0.5f));
        warpA->SetScale(900.0f);
        warpA->SetWarpAmplitude(120.0f); // no folds, as for the belts
        m_aridity = warpA;
    }

    m_cavern = fbm2(70.0f, 3, 0.5f);
    m_chamber = fbm2(120.0f, 2, 0.5f);
    m_tunnelA = fbm2(150.0f, 2, 0.5f);
    m_tunnelB = fbm2(137.0f, 2, 0.5f); // not 150: equal scales correlate the pair
    m_tunnelW = fbm2(90.0f, 1, 0.5f);
    m_ravine = fbm2(260.0f, 2, 0.45f);
    m_ravineMask = fbm2(1000.0f, 2, 0.5f);
}

// Version 2: real seas, wide low coasts, relief that builds inland, mountains in ranges and
// deserts that are a climate. All from the inputs at one column, so it is the same function for
// a whole chunk (columnGrid) and a single column (column).
TerrainGenerator::Column TerrainGenerator::shapeColumn2(const Noise2& n, int wx, int wz, int seed) {
    // The coast is where the continent field crosses kCoast2; c is how far past it, in units of
    // the field.
    const float c = n.cont - kCoast2;
    // ...and d is roughly how far past it in BLOCKS: c over the field's local slope. Bands of c
    // alone made the land climb at whatever rate the field happened to have at that coast,
    // which near a steep one was tens of blocks up within a few tens of blocks in. Measuring in
    // blocks is what lets a beach be the first dozen blocks and the lowland the next hundred,
    // everywhere.
    //
    // The slope is taken across kSlopeSpan either side, not one block: the field's finest detail
    // made a one-block slope swing, the distance flip, and the ground step into ridges and swirls
    // wherever it did. Floored so far from a coast the distance grows smoothly instead of
    // blowing up where the field flattens out.
    const float grad = std::max(std::sqrt(n.contGx * n.contGx + n.contGz * n.contGz), 1.0f / 2500.0f);
    float d = std::clamp(c / grad, -2000.0f, 2000.0f);
    // The local slope only knows about the coast close by. Further in it guesses as if the field
    // ran straight on, and deep in a continent it does not: around the top of the field the slope
    // goes from nothing to something while c hardly changes, so this d swung between a few hundred
    // and 2000 over a few dozen blocks, both ways, and the ground with it into trenches and ridges
    // 40 high. Past the first stretch the look around (lookD, see coastLook) takes over,
    // which sees the field actually curve; the beach and the low country keep the local answer.
    if (c > 0.0f)
        d = d > n.lookD ? n.lookD : glm::mix(d, n.lookD, smooth01(60.0f, 200.0f, d));

    // A monotone cubic through the table (Fritsch-Carlson): smooth through every point and never
    // overshooting between them. Straight segments met at creases that a low sun picks out as
    // lines running along every coast.
    const auto curve = [](const float* xs, const float* ys, int count, float x) {
        if (x <= xs[0])
            return ys[0];
        if (x >= xs[count - 1])
            return ys[count - 1];
        int i = 1;
        while (x > xs[i])
            ++i;
        const auto slope = [&](int k) { return (ys[k + 1] - ys[k]) / (xs[k + 1] - xs[k]); };
        const auto tangent = [&](int k) {
            if (k == 0)
                return slope(0);
            if (k == count - 1)
                return slope(count - 2);
            const float a = slope(k - 1), b = slope(k);
            return a * b <= 0.0f ? 0.0f : 2.0f / (1.0f / a + 1.0f / b); // harmonic mean
        };
        const float hgt = xs[i] - xs[i - 1];
        const float t = (x - xs[i - 1]) / hgt;
        const float t2 = t * t, t3 = t2 * t;
        return (2 * t3 - 3 * t2 + 1) * ys[i - 1] + (t3 - 2 * t2 + t) * hgt * tangent(i - 1) +
               (-2 * t3 + 3 * t2) * ys[i] + (t3 - t2) * hgt * tangent(i);
    };
    // Near the coast, by blocks from the waterline: the shallows, a wide shelf, the slope down to
    // the ocean floor; a flat beach band, low coastal country, then rising into the interior.
    static constexpr float kDX[] = {-700.0f, -420.0f, -260.0f, -30.0f, 0.0f, 14.0f, 120.0f, 400.0f};
    static constexpr float kDY[] = {24.0f, 30.0f, 48.0f, 58.0f, 62.0f, 64.0f, 71.0f, 85.0f};
    const float nearCoast = curve(kDX, kDY, 8, d);
    // Far from any coast, by the field itself: the deep ocean floor, or the high interior.
    static constexpr float kCX[] = {-0.40f, -0.20f, 0.0f, 0.20f, 0.45f, 0.80f};
    static constexpr float kCY[] = {20.0f, 24.0f, 72.0f, 85.0f, 106.0f, 122.0f};
    const float farCoast = curve(kCX, kCY, 6, c);
    const float handOver = smooth01(350.0f, 650.0f, std::abs(d));
    // Inland never below the coast's profile, at sea never above it, so the two meet without a
    // step; a smooth max (min) rather than a hard one, which left a crease where they cross.
    const auto smax = [](float a, float b, float k) {
        const float hk = std::max(k - std::abs(a - b), 0.0f) / k;
        return std::max(a, b) + hk * hk * k * 0.25f;
    };
    const float far = glm::mix(nearCoast, farCoast, handOver);
    const float base = c >= 0.0f ? smax(nearCoast, far, 6.0f) : -smax(-nearCoast, -far, 6.0f);

    // How far inland, in blocks: hills, ranges and deserts all wait on it, which is what keeps
    // the slope from starting at the waterline.
    const float inland = smooth01(12.0f, 220.0f, d);
    const float hilly = smooth01(-0.25f, 0.35f, n.hilly);
    // Desert: one smooth field over a threshold, so a desert is a region thousands of blocks
    // across, not a scatter of patches; and kept back from the sea a little, so there is a strip
    // of ordinary country (or a beach) along its coast. Its edge is measured in blocks the same
    // way the coast is, so it is about a hundred wide everywhere, and the mountains below can
    // start giving way to it from a long way off.
    const float aridGrad = std::max(std::sqrt(n.aridGx * n.aridGx + n.aridGz * n.aridGz), 1.0f / 4000.0f);
    const float dry = std::clamp((n.aridity - 0.50f) / aridGrad, -2000.0f, 2000.0f); // blocks into the desert
    const float desert = smooth01(-50.0f, 70.0f, dry) * smooth01(40.0f, 160.0f, d);

    float amp = 1.2f + inland * (4.0f + 14.0f * hilly);
    amp *= 1.0f - 0.65f * desert;
    // Lifted a little so the low country is not pitted with ponds wherever the noise dips.
    float h = base + (n.hills + 0.15f * inland) * amp;
    if (d < 0.0f) // the sea floor has a little relief of its own, more of it the deeper it goes
        h += n.hills * 4.0f * smooth01(-40.0f, -300.0f, d);

    // Mountain ranges run along the belt field's zero lines, laid out by distance in blocks from
    // that crest line (the field over its slope, like the coast). Bands of the field itself were
    // a few dozen blocks across wherever the warp squeezed it, and the range went up them like a
    // wall. Going in toward the crest: foothills climbing ~25 over a couple of hundred blocks, the
    // massif ~75 more, and on the crest the ridged relief puts ridgelines, peaks and passes up to
    // ~70 higher still. Most of the height is the massif: with more in the ridges, a range was a
    // row of spires. The ridges are not squared: squared, every crest was a needle. Only
    // inland, fading out well before a desert, and only along the stretches the presence field
    // turns on, so ranges have ends.
    const float beltGrad = std::max(std::sqrt(n.beltGx * n.beltGx + n.beltGz * n.beltGz), 1.0f / 4000.0f);
    const float axis = std::abs(n.belt) / beltGrad;
    const float on = smooth01(-0.35f, 0.05f, n.beltOn) * smooth01(50.0f, 360.0f, d) * (1.0f - smooth01(-200.0f, -30.0f, dry));
    if (on > 0.0f && axis < 480.0f) {
        const float foot = smooth01(480.0f, 220.0f, axis);
        const float core = smooth01(300.0f, 40.0f, axis);
        const float r = std::clamp(n.ridge * 0.5f + 0.5f, 0.0f, 1.0f);
        const float rise = on * (24.0f * foot + core * (75.0f + 70.0f * r * core));
        // ...and never climbing off the coast faster than about one up for every two in: where a
        // range's crest ran close along a shore it went straight up out of the beach country.
        // (A soft min, so the cap does not leave a crease where it takes over.)
        const float cap = 0.55f * std::max(d - 40.0f, 0.0f);
        h += std::max(0.5f * (rise + cap - std::sqrt((rise - cap) * (rise - cap) + 64.0f)), 0.0f);
    }
    if (desert > 0.0f) {
        // the dune field's crest lines, squared so the crests are narrow and the troughs wide
        const float ridge = 1.0f - std::abs(std::clamp(n.dune, -1.0f, 1.0f));
        h += ridge * ridge * 6.0f * desert;
    }
    // A coast that is too smooth quantises into long straight terraces: a little noise near the
    // waterline wanders it into small bays, spits and sandbars.
    const float shore = 1.0f - smooth01(1.5f, 6.0f, std::abs(h - float(kSeaLevel)));
    if (shore > 0.0f)
        h += ((valueNoise(float(wx) * 0.04f, float(wz) * 0.04f, seedOff(seed, 211)) - 0.5f) * 2.2f +
              (valueNoise(float(wx) * 0.17f, float(wz) * 0.17f, seedOff(seed, 213)) - 0.5f) * 0.8f) * shore;
    return {std::clamp(int(std::floor(h)), 8, kWorldHeight - 8), desert, d};
}

void TerrainGenerator::columnGrid2(int x0, int z0, int n, int* heights, float* desert, float* coast) const {
    constexpr int kS = int(kSlopeSpan);
    // the fields read as distances get a ring of kSlopeSpan more, for their slopes
    const int np = n + 2 * kS;
    const std::size_t pp = std::size_t(np) * std::size_t(np);
    // never destroyed on purpose: see computeConnectivity (a thread_local with a destructor
    // corrupts the heap when a worker exits, on this toolchain)
    static thread_local std::vector<float>& padded = *new std::vector<float>();
    static thread_local std::vector<float>& buf = *new std::vector<float>();
    padded.resize(pp * 3);
    float* cont = padded.data();
    float* belt = cont + pp;
    float* aridity = belt + pp;
    const std::size_t nn = std::size_t(n) * std::size_t(n);
    buf.resize(nn * 5);
    float* beltOn = buf.data();
    float* ridge = beltOn + nn;
    float* hilly = ridge + nn;
    float* hills = hilly + nn;
    float* dune = hills + nn;
    const float fx = float(x0), fz = float(z0);
    m_cont2->GenUniformGrid2D(cont, fx - kSlopeSpan, fz - kSlopeSpan, np, np, 1.0f, 1.0f, seedOff(m_seed, 201));
    m_belt->GenUniformGrid2D(belt, fx - kSlopeSpan, fz - kSlopeSpan, np, np, 1.0f, 1.0f, seedOff(m_seed, 202));
    m_aridity->GenUniformGrid2D(aridity, fx - kSlopeSpan, fz - kSlopeSpan, np, np, 1.0f, 1.0f, seedOff(m_seed, 205));
    m_beltOn->GenUniformGrid2D(beltOn, fx, fz, n, n, 1.0f, 1.0f, seedOff(m_seed, 207));
    m_ridge->GenUniformGrid2D(ridge, fx, fz, n, n, 1.0f, 1.0f, seedOff(m_seed, 203));
    m_hilly->GenUniformGrid2D(hilly, fx, fz, n, n, 1.0f, 1.0f, seedOff(m_seed, 204));
    m_height->GenUniformGrid2D(hills, fx, fz, n, n, 1.0f, 1.0f, m_seed);
    m_dune->GenUniformGrid2D(dune, fx, fz, n, n, 1.0f, 1.0f, seedOff(m_seed, 6));
    // the look around for the coast: each direction's samples on the lattice cells this block covers
    static thread_local std::vector<float>& look = *new std::vector<float>();
    const int lx0 = floorDivI(x0, kLookStep), lz0 = floorDivI(z0, kLookStep);
    const int ln = floorDivI(x0 + n - 1, kLookStep) - lx0 + 2, lm = floorDivI(z0 + n - 1, kLookStep) - lz0 + 2;
    const std::size_t ll = std::size_t(ln) * std::size_t(lm);
    look.resize(ll * kLookDirs);
    for (int k = 0; k < kLookDirs; ++k)
        m_cont2->GenUniformGrid2D(look.data() + ll * std::size_t(k), float(lx0 * kLookStep + kLookX[k]),
                                  float(lz0 * kLookStep + kLookZ[k]), ln, lm, float(kLookStep), float(kLookStep),
                                  seedOff(m_seed, 201));
    const std::size_t row = std::size_t(kS) * std::size_t(np);
    const auto gx = [&](const float* f, std::size_t p) { return (f[p + kS] - f[p - kS]) / (2.0f * kSlopeSpan); };
    const auto gz = [&](const float* f, std::size_t p) { return (f[p + row] - f[p - row]) / (2.0f * kSlopeSpan); };
    float corners[kLookDirs * 4];
    for (int z = 0; z < n; ++z)
        for (int x = 0; x < n; ++x) {
            const std::size_t i = std::size_t(x) + std::size_t(z) * std::size_t(n);
            const std::size_t p = std::size_t(x + kS) + std::size_t(z + kS) * std::size_t(np);
            const int lx = floorDivI(x0 + x, kLookStep), lz = floorDivI(z0 + z, kLookStep);
            const std::size_t l = std::size_t(lx - lx0) + std::size_t(lz - lz0) * std::size_t(ln);
            for (int k = 0; k < kLookDirs; ++k) {
                const float* v = look.data() + ll * std::size_t(k) + l;
                corners[k * 4 + 0] = v[0];
                corners[k * 4 + 1] = v[1];
                corners[k * 4 + 2] = v[ln];
                corners[k * 4 + 3] = v[ln + 1];
            }
            const float lookD = coastLook(cont[p] - kCoast2, corners, float(x0 + x - lx * kLookStep) / float(kLookStep),
                                          float(z0 + z - lz * kLookStep) / float(kLookStep));
            const Column col = shapeColumn2({cont[p], belt[p], beltOn[i], ridge[i], hilly[i], aridity[p], hills[i], dune[i],
                                             gx(cont, p), gz(cont, p), gx(belt, p), gz(belt, p), gx(aridity, p), gz(aridity, p),
                                             lookD},
                                            x0 + x, z0 + z, m_seed);
            heights[i] = col.h;
            desert[i] = col.desert;
            coast[i] = col.coast;
        }
}

bool TerrainGenerator::isBeach(int h, float coast) const {
    if (h < kSeaLevel - 1 || h > kBeachTop)
        return false;
    return m_version < 2 || coast < 45.0f;
}

void TerrainGenerator::columnGrid(int baseX, int baseZ, int (*heights)[kChunkSize],
                                  float (*desert)[kChunkSize]) const {
    constexpr int kN = kChunkSize * kChunkSize;
    if (m_version >= 2) {
        int h[kN];
        float de[kN], co[kN];
        columnGrid2(baseX, baseZ, kChunkSize, h, de, co);
        for (int z = 0; z < kChunkSize; ++z)
            for (int x = 0; x < kChunkSize; ++x) {
                heights[z][x] = h[x + z * kChunkSize];
                desert[z][x] = de[x + z * kChunkSize];
            }
        return;
    }
    float heightNoise[kN];
    float contNoise[kN];
    float aridNoise[kN];
    float duneNoise[kN];
    m_height->GenUniformGrid2D(heightNoise, float(baseX), float(baseZ), kChunkSize,
                               kChunkSize, 1.0f, 1.0f, m_seed);
    m_continent->GenUniformGrid2D(contNoise, float(baseX), float(baseZ), kChunkSize,
                                  kChunkSize, 1.0f, 1.0f, seedOff(m_seed, 1));
    m_arid->GenUniformGrid2D(aridNoise, float(baseX), float(baseZ), kChunkSize, kChunkSize,
                             1.0f, 1.0f, seedOff(m_seed, 5));
    float aridMax = -1.0f;
    for (const float a : aridNoise)
        aridMax = std::max(aridMax, a);
    if (aridMax > 0.14f)
        m_dune->GenUniformGrid2D(duneNoise, float(baseX), float(baseZ), kChunkSize, kChunkSize,
                                 1.0f, 1.0f, seedOff(m_seed, 6));
    else
        std::fill(std::begin(duneNoise), std::end(duneNoise), 0.0f);
    for (int z = 0; z < kChunkSize; ++z)
        for (int x = 0; x < kChunkSize; ++x) {
            // grid order: x + z * kChunkSize (x start axis first)
            const int i = x + z * kChunkSize;
            const Column col = shapeColumn(heightNoise[i], contNoise[i], aridNoise[i], duneNoise[i],
                                           baseX + x, baseZ + z, m_seed);
            heights[z][x] = col.h;
            desert[z][x] = col.desert;
        }
}

TerrainGenerator::Column TerrainGenerator::shapeColumn(float heightNoise, float continent,
                                                        float arid, float dune, int wx, int wz,
                                                        int seed) {
    const float base = heightSpline(continent);
    float raw = base + heightNoise * reliefAmplitude(continent);

    // A desert is a climate, not a patch of sand: dry, INLAND, and not up in the mountains.
    // The inland ramp is what keeps a belt of ordinary country (and the beach) between a desert
    // and the sea; the old mask keyed on height alone put dunes straight onto the shoreline.
    const float dry = smooth01(0.14f, 0.32f, arid);
    const float inland = smooth01(-0.30f, -0.08f, continent) *
                         (1.0f - smooth01(0.02f, 0.22f, continent));
    // ...and not down where a depression has let the sea (or a lake) in: sand meeting open water
    // directly is a beach, not a desert. Judged on the height before the desert reshapes it.
    const float high = smooth01(float(kSeaLevel + 4), float(kSeaLevel + 12), raw);
    const float desert = dry * inland * high;
    if (desert > 0.0f) {
        // Desert country is low and rolling, not mountainous: most of the relief is taken out,
        // leaving a broad basin, and the dunes ride on top of that. Blended by the desert weight
        // so it eases into the neighbouring land instead of stepping.
        const float calm = base + heightNoise * reliefAmplitude(continent) * 0.28f;
        raw += (calm - raw) * desert;
        // ridged: the crest lines of the noise become dune crests, squared so the crests are
        // narrow and the troughs between them wide and flat
        const float ridge = 1.0f - std::abs(std::clamp(dune, -1.0f, 1.0f));
        raw += ridge * ridge * 5.5f * desert;
    }
    raw = beachRemap(raw);

    // Roughen the shoreline. The beach shelf is nearly flat, and a flat surface quantised to whole
    // blocks turns into a few huge terraces with smooth polygonal edges (foam-outlined patches in
    // the shallows). A little low-frequency noise near the waterline wanders those contours into
    // bays, spits and sandbars, and a finer one breaks them into ragged steps.
    const float shore = 1.0f - smooth01(2.0f, 9.0f, std::abs(raw - float(kSeaLevel)));
    if (shore > 0.0f)
        raw += ((valueNoise(float(wx) * 0.045f, float(wz) * 0.045f, seedOff(seed, 61)) - 0.5f) * 4.0f +
                (valueNoise(float(wx) * 0.19f, float(wz) * 0.19f, seedOff(seed, 67)) - 0.5f) * 1.6f) * shore;
    return {std::clamp(int(raw), 8, kWorldHeight - 8), desert};
}

TerrainGenerator::Column TerrainGenerator::column(int wx, int wz) const {
    if (m_version >= 3) {
        const auto t = columnTile(floorDivI(wx, kChunkSize), floorDivI(wz, kChunkSize));
        const int x = wx & (kChunkSize - 1), z = wz & (kChunkSize - 1);
        Column c{int(t->h[z][x]), float(t->desert[z][x]) / 255.0f, float(t->coast[z][x])};
        c.rock = t->rock[z][x] != 0;
        c.snowLine = t->snow[z][x];
        c.moisture = float(t->moisture[z][x]) / 255.0f;
        c.temp = float(t->temp[z][x]) / 255.0f;
        c.water = t->water[z][x];
        c.biome = t->biome[z][x];
        c.volcanic = t->volcanic[z][x] > 100;
        return c;
    }
    if (m_version >= 2) {
        Noise2 n{};
        const float fx = float(wx), fz = float(wz);
        // the column and 16 blocks either side, the same samples columnGrid takes its slopes from
        float s9[9];
        const auto sampled = [&](const FastNoise::SmartNode<>& node, int seedOffset, float& v, float& gx, float& gz) {
            node->GenUniformGrid2D(s9, fx - kSlopeSpan, fz - kSlopeSpan, 3, 3, kSlopeSpan, kSlopeSpan,
                                   seedOff(m_seed, seedOffset));
            v = s9[4];
            gx = (s9[5] - s9[3]) / (2.0f * kSlopeSpan);
            gz = (s9[7] - s9[1]) / (2.0f * kSlopeSpan);
        };
        sampled(m_cont2, 201, n.cont, n.contGx, n.contGz);
        sampled(m_belt, 202, n.belt, n.beltGx, n.beltGz);
        sampled(m_aridity, 205, n.aridity, n.aridGx, n.aridGz);
        m_beltOn->GenUniformGrid2D(&n.beltOn, fx, fz, 1, 1, 1.0f, 1.0f, seedOff(m_seed, 207));
        m_ridge->GenUniformGrid2D(&n.ridge, fx, fz, 1, 1, 1.0f, 1.0f, seedOff(m_seed, 203));
        m_hilly->GenUniformGrid2D(&n.hilly, fx, fz, 1, 1, 1.0f, 1.0f, seedOff(m_seed, 204));
        m_height->GenUniformGrid2D(&n.hills, fx, fz, 1, 1, 1.0f, 1.0f, m_seed);
        m_dune->GenUniformGrid2D(&n.dune, fx, fz, 1, 1, 1.0f, 1.0f, seedOff(m_seed, 6));
        // the look around, from the same lattice samples columnGrid2 reads
        const int lx = floorDivI(wx, kLookStep), lz = floorDivI(wz, kLookStep);
        float corners[kLookDirs * 4];
        for (int k = 0; k < kLookDirs; ++k)
            m_cont2->GenUniformGrid2D(corners + k * 4, float(lx * kLookStep + kLookX[k]), float(lz * kLookStep + kLookZ[k]),
                                      2, 2, float(kLookStep), float(kLookStep), seedOff(m_seed, 201));
        n.lookD = coastLook(n.cont - kCoast2, corners, float(wx - lx * kLookStep) / float(kLookStep),
                            float(wz - lz * kLookStep) / float(kLookStep));
        return shapeColumn2(n, wx, wz, m_seed);
    }
    float hn = 0.0f, c = 0.0f, a = 0.0f, d = 0.0f;
    m_height->GenUniformGrid2D(&hn, float(wx), float(wz), 1, 1, 1.0f, 1.0f, m_seed);
    m_continent->GenUniformGrid2D(&c, float(wx), float(wz), 1, 1, 1.0f, 1.0f, seedOff(m_seed, 1));
    m_arid->GenUniformGrid2D(&a, float(wx), float(wz), 1, 1, 1.0f, 1.0f, seedOff(m_seed, 5));
    if (a > 0.14f) // the dune field only matters where it can show
        m_dune->GenUniformGrid2D(&d, float(wx), float(wz), 1, 1, 1.0f, 1.0f, seedOff(m_seed, 6));
    return shapeColumn(hn, c, a, d, wx, wz, m_seed);
}

int TerrainGenerator::surfaceHeight(int wx, int wz) const { return column(wx, wz).h; }

glm::ivec2 TerrainGenerator::spawnColumn() const {
    if (m_version < 2)
        return {16, 16};
    // Outward from (16, 16) in rings 16 blocks apart: the first column that is dry, low and not
    // desert, so a new world opens on ordinary country with the sea not far off.
    for (int ring = 0; ring <= 250; ++ring)
        for (int dz = -ring; dz <= ring; ++dz)
            for (int dx = -ring; dx <= ring; ++dx) {
                if (std::max(std::abs(dx), std::abs(dz)) != ring)
                    continue;
                const int x = 16 + dx * 16, z = 16 + dz * 16;
                const Column c = column(x, z);
                if (c.h >= kSeaLevel + 3 && c.h <= kSeaLevel + 40 && c.desert < 0.3f && c.water <= c.h)
                    return {x, z};
            }
    return {16, 16};
}

int TerrainGenerator::snowLineAt(int wx, int wz) const {
    if (m_version >= 3)
        return column(wx, wz).snowLine;
    // One height, +-3 a column, ended every snowcap on a near-perfect contour, and from far
    // off a range read as if someone had ruled a line across it (SPEC #126). The line wanders
    // now, on scales smaller than one mountain face or it still reads straight across each:
    // tongues of snow some 80 blocks apart, a closer wobble over about 20, and the per-column
    // jitter that keeps the edge ragged up close.
    const float broad = valueNoise(float(wx) * 0.0125f, float(wz) * 0.0125f, seedOff(m_seed, 97)) - 0.5f;
    const float close = valueNoise(float(wx) * 0.05f, float(wz) * 0.05f, seedOff(m_seed, 101)) - 0.5f;
    const int jitter = int(cellHash(std::uint32_t(wx), std::uint32_t(wz)) & 7u) - 3;
    return kSnowLine + int(std::lround(broad * 56.0f + close * 16.0f)) + jitter;
}

const char* biomeName(Biome b) {
    constexpr const char* kNames[] = {"ocean", "beach", "desert", "red desert", "badlands", "savanna",
                                      "jungle", "swamp", "plains", "meadow", "forest", "birch forest",
                                      "taiga", "snowy taiga", "tundra", "alpine", "snowcap", "cherry grove",
                                      "autumn forest", "spring forest", "flower field", "aspen grove", "redwoods",
                                      "dark forest", "bamboo", "mangroves", "oasis"};
    return b < Biome::Count ? kNames[int(b)] : "";
}

Biome TerrainGenerator::classify(int wx, int wz, int h, float broad, float desert, float coast, float moisture,
                                 float temp, int snowLine, float oasis) const {
    if (h < kSeaLevel - 1)
        return Biome::Ocean;
    const float fx = float(wx), fz = float(wz);
    const float dither = float(cellHash(std::uint32_t(wx), std::uint32_t(wz)) & 255u) / 255.0f;
    if (oasis > 0.40f + 0.2f * dither)
        return Biome::Oasis;
    // the same ragged edge the sand has
    const bool dry = desert > 0.42f + 0.16f * dither;
    // on a hot, wet coast the mangroves come right down to the water instead of a beach
    if (!dry && isBeach(h, coast))
        return temp > 0.50f && moisture > 0.62f ? Biome::Mangrove : Biome::Beach;
    if (dry) {
        // in some deserts the high ground weathers into banded mesas
        if (regionNoise(fx * 0.0018f, fz * 0.0018f, seedOff(m_seed, 309)) > 0.6f &&
            h > kSeaLevel + 18 + int(valueNoise(fx * 0.01f, fz * 0.01f, seedOff(m_seed, 301)) * 18.0f))
            return Biome::Badlands;
        return regionNoise(fx * 0.0025f, fz * 0.0025f, seedOff(m_seed, 303)) > 0.62f ? Biome::RedDesert : Biome::Desert;
    }
    if (h >= snowLine)
        return Biome::Snowcap;
    if (h >= snowLine - 30)
        return Biome::Alpine;
    // A wobble, so borders are not contour lines of the climate, and a finer one with a little dither
    // on top, so where two biomes meet their ground interleaves for a few blocks instead of a clean cut.
    const float j = (valueNoise(fx * 0.02f, fz * 0.02f, seedOff(m_seed, 305)) - 0.5f) * 0.05f +
                    (valueNoise(fx * 0.11f, fz * 0.11f, seedOff(m_seed, 315)) - 0.5f) * 0.022f + (dither - 0.5f) * 0.008f;
    const float t = temp - std::max(0.0f, broad - float(kSeaLevel)) * 0.0022f + j;
    const float m = moisture + j;
    if (t < kBiomeFrozen)
        return m > 0.30f ? Biome::SnowyTaiga : Biome::Tundra;
    // Slow fields that pick, region by region, the rarer woods and fields a climate can have.
    const float edge = (dither - 0.5f) * 0.03f;
    const float va = regionNoise(fx * 0.0023f, fz * 0.0023f, seedOff(m_seed, 341)) + edge;
    const float vb = regionNoise(fx * 0.0019f, fz * 0.0019f, seedOff(m_seed, 343)) + edge;
    if (t < kBiomeCold) {
        if (m > 0.46f && va > 0.62f)
            return Biome::Redwood; // the old giants, where it is wet enough for them
        if (t > kBiomeCold - 0.05f && vb > 0.62f)
            return Biome::AspenGrove;
        return Biome::Taiga;
    }
    if (t > kBiomeHot) {
        if (m > kBiomeSwampWet && broad < float(kSeaLevel + 7) && coast < 260.0f)
            return Biome::Mangrove;
        if (m > kBiomeJungleWet)
            return va > 0.63f ? Biome::Bamboo : Biome::Jungle;
        if (m < kBiomeSavannaDry || desert > 0.2f)
            return Biome::Savanna;
    } else if (t > kBiomeWarm && (m < 0.12f || desert > 0.25f)) {
        return Biome::Savanna;
    }
    if (m > kBiomeSwampWet && broad < float(kSeaLevel + 9))
        return t > kBiomeWarm + 0.04f && coast < 400.0f ? Biome::Mangrove : Biome::Swamp;
    const float woods = regionNoise(fx * 0.0045f, fz * 0.0045f, seedOff(m_seed, 71)) + edge;
    const float region = regionNoise(fx * 0.004f, fz * 0.004f, seedOff(m_seed, 307)) + edge;
    // cherry trees on the mild hills, more readily the higher the hills are
    const float vc = regionNoise(fx * 0.0021f, fz * 0.0021f, seedOff(m_seed, 345)) + edge;
    if (t > 0.20f && t < kBiomeWarm && m > 0.22f && vc + 0.16f * smooth01(float(kSeaLevel + 12), float(kSeaLevel + 48), broad) > 0.75f)
        return Biome::CherryGrove;
    if (m > kBiomeForestWet && woods > 0.36f) {
        if (region > 0.64f)
            return t < 0.24f && vb > 0.5f ? Biome::AspenGrove : Biome::BirchForest;
        if (m > 0.52f && va > 0.62f)
            return Biome::DarkForest;
        if (vb > 0.60f)
            return Biome::AutumnForest;
        if (vb < 0.32f)
            return Biome::SpringForest;
        return Biome::Forest;
    }
    if (vb > 0.67f || (region < 0.30f && vb > 0.57f))
        return Biome::FlowerField;
    return region < 0.30f ? Biome::Meadow : Biome::Plains;
}

float TerrainGenerator::treeDensity(Biome b) {
    switch (b) {
        case Biome::Forest: return 0.85f;
        case Biome::BirchForest: return 0.8f;
        case Biome::Taiga: return 0.78f;
        case Biome::SnowyTaiga: return 0.45f;
        case Biome::Tundra: return 0.03f;
        case Biome::Jungle: return 0.8f;
        case Biome::Savanna: return 0.11f;
        case Biome::Swamp: return 0.45f;
        case Biome::Plains: return 0.10f;
        case Biome::Meadow: return 0.04f;
        case Biome::Alpine: return 0.08f;
        case Biome::CherryGrove: return 0.5f;
        case Biome::AutumnForest: return 0.85f;
        case Biome::SpringForest: return 0.75f;
        case Biome::FlowerField: return 0.03f;
        case Biome::AspenGrove: return 0.8f;
        case Biome::Redwood: return 0.75f;
        case Biome::DarkForest: return 0.95f;
        case Biome::Bamboo: return 0.6f;
        case Biome::Mangrove: return 0.55f;
        case Biome::Oasis: return 0.2f;
        default: return 0.0f;
    }
}

Biome TerrainGenerator::biomeOf(int wx, int wz) const {
    return m_version >= 3 ? column(wx, wz).biome : Biome::Plains;
}

TerrainGenerator::BiomeSample TerrainGenerator::biomeAt(int wx, int wz) const {
    const Column col = column(wx, wz);
    BiomeSample b;
    b.desert = col.desert;
    b.ocean = smooth01(float(kSeaLevel), float(kSeaLevel - 8), float(col.h));
    // the sandy band, only where it is beach rather than a dune or a lake shore inland
    b.beach = (1.0f - smooth01(0.0f, 6.0f, std::abs(float(col.h) - float(kSeaLevel + 1)))) *
              (1.0f - col.desert) * (m_version >= 2 ? 1.0f - smooth01(45.0f, 90.0f, col.coast) : 1.0f);
    return b;
}

TerrainGenerator::FarSample TerrainGenerator::farSample(int wx, int wz) const {
    const Column col = column(wx, wz);
    FarSample f;
    f.h = col.h;
    f.desert = col.desert;
    f.forest = smooth01(0.30f, 0.70f, valueNoise(float(wx) * 0.0045f, float(wz) * 0.0045f, seedOff(m_seed, 71)));
    f.snowLine = m_version >= 3 ? col.snowLine : snowLineAt(wx, wz);
    f.moisture = col.moisture;
    f.temp = col.temp;
    f.coast = col.coast;
    f.water = col.water;
    if (m_version >= 3) {
        f.biome = col.biome;
        f.forest = std::min(treeDensity(col.biome) * (0.55f + 0.9f * smooth01(0.30f, 0.70f, regionNoise(float(wx) * 0.0045f, float(wz) * 0.0045f, seedOff(m_seed, 71)))), 1.0f);
    }
    return f;
}

namespace {
// A big grid in 64 x 64 blocks, claimed one at a time by the caller and any worker free to help.
// Heap-held and counted: a helper that only gets to run once all the work is gone still reads it.
struct GridSplit {
    static constexpr int kB = 64;
    const TerrainGenerator* gen;
    int x0, z0, n, step, blocks;
    TerrainGenerator::FarSample* out;
    std::atomic<int> next{0}, active{0}, refs{1};

    void work() {
        active.fetch_add(1);
        std::vector<TerrainGenerator::FarSample> block(std::size_t(kB) * kB);
        const int nb = (n + kB - 1) / kB;
        for (int b = next.fetch_add(1); b < blocks; b = next.fetch_add(1)) {
            const int bx = b % nb, bz = b / nb;
            gen->farGrid(x0 + bx * kB * step, z0 + bz * kB * step, kB, step, block.data());
            const int w = std::min(kB, n - bx * kB), h = std::min(kB, n - bz * kB);
            for (int z = 0; z < h; ++z)
                std::copy_n(block.data() + std::size_t(z) * kB, w, out + std::size_t(bz * kB + z) * n + std::size_t(bx * kB));
        }
        active.fetch_sub(1);
    }
    void release() {
        if (refs.fetch_sub(1) == 1)
            delete this;
    }
};
} // namespace

void TerrainGenerator::farGrid(int x0, int z0, int n, int step, FarSample* out, JobSystem* jobs) const {
    if (m_version < 3) {
        for (int z = 0; z < n; ++z)
            for (int x = 0; x < n; ++x)
                out[std::size_t(z) * n + x] = farSample(x0 + x * step, z0 + z * step);
        return;
    }
    // In blocks: each gathers only the rivers near it, and the land tiles a wide grid needs are
    // built side by side. One pass over a 16 km grid took four seconds on one thread.
    if (jobs != nullptr && n > GridSplit::kB) {
        const int nb = (n + GridSplit::kB - 1) / GridSplit::kB;
        auto* split = new GridSplit{this, x0, z0, n, step, nb * nb, out};
        const int helpers = std::min(nb * nb - 1, std::max(jobs->workerCount() - 1, 0));
        split->refs.fetch_add(helpers);
        for (int i = 0; i < helpers; ++i)
            jobs->submit([split] {
                split->work();
                split->release();
            });
        split->work();
        // every block is claimed; wait out the helpers still writing theirs
        while (split->active.load() != 0)
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        split->release();
        return;
    }
    // never destroyed on purpose: see computeConnectivity
    static thread_local std::vector<Terrain3::Sample>& samples = *new std::vector<Terrain3::Sample>();
    samples.resize(std::size_t(n) * n);
    m_land3->grid(x0, z0, n, step, samples.data());
    for (int z = 0; z < n; ++z)
        for (int x = 0; x < n; ++x) {
            const Terrain3::Sample& s = samples[std::size_t(z) * n + x];
            FarSample& f = out[std::size_t(z) * n + x];
            f.h = std::clamp(int(std::floor(s.h)), 8, kWorldHeight - 8);
            f.desert = s.desert;
            f.snowLine = int(s.snowLine);
            f.moisture = s.moisture;
            f.temp = s.temp;
            f.coast = s.coast;
            f.water = s.water > s.h ? int(s.water) : -1;
            f.biome = classify(x0 + x * step, z0 + z * step, f.h, s.broad, s.desert, s.coast, s.moisture, s.temp, f.snowLine, s.oasis);
            // the canopy the near trees will grow here: the same biome and the same thinning
            f.forest = std::min(treeDensity(f.biome) * (0.55f + 0.9f * smooth01(0.30f, 0.70f, regionNoise(float(x0 + x * step) * 0.0045f, float(z0 + z * step) * 0.0045f, seedOff(m_seed, 71)))), 1.0f);
        }
}

void TerrainGenerator::generate(Chunk& chunk, ChunkPos pos, bool detailed) const {
    chunk.fillLight(0); // pooled chunks carry the previous tenant's light
    const int baseY = pos.y * kChunkSize;
    const int baseX = pos.x * kChunkSize;
    const int baseZ = pos.z * kChunkSize;

    int heights[kChunkSize][kChunkSize];
    float desertW[kChunkSize][kChunkSize];
    float coast[kChunkSize][kChunkSize] = {};
    // Version 2: where the ground is too steep to hold soil, as placeFoliage takes it.
    std::uint8_t rock[kChunkSize][kChunkSize] = {};
    // The lowest ground within kSidePad of each column: a cave has to stay under that, or it
    // breaks out of the side of a slope. (Version 1 measured straight down only.)
    int groundMin[kChunkSize][kChunkSize];
    std::shared_ptr<const ColumnTile> tile3;
    if (m_version >= 3) {
        tile3 = columnTile(pos.x, pos.z);
        for (int z = 0; z < kChunkSize; ++z)
            for (int x = 0; x < kChunkSize; ++x) {
                heights[z][x] = tile3->h[z][x];
                desertW[z][x] = float(tile3->desert[z][x]) / 255.0f;
                coast[z][x] = float(tile3->coast[z][x]);
                rock[z][x] = tile3->rock[z][x];
                groundMin[z][x] = tile3->ground[z][x];
            }
    } else if (m_version >= 2) {
        constexpr int kP = kSidePad, kN = kChunkSize + 2 * kSidePad;
        // never destroyed on purpose: see computeConnectivity
        static thread_local std::vector<int>& ph = *new std::vector<int>();
        static thread_local std::vector<float>& pd = *new std::vector<float>();
        static thread_local std::vector<float>& pc = *new std::vector<float>();
        ph.resize(std::size_t(kN) * kN);
        pd.resize(std::size_t(kN) * kN);
        pc.resize(std::size_t(kN) * kN);
        columnGrid2(baseX - kP, baseZ - kP, kN, ph.data(), pd.data(), pc.data());
        for (int z = 0; z < kChunkSize; ++z)
            for (int x = 0; x < kChunkSize; ++x) {
                const std::size_t i = std::size_t(x + kP) + std::size_t(z + kP) * kN;
                heights[z][x] = ph[i];
                desertW[z][x] = pd[i];
                coast[z][x] = pc[i];
                int lo = ph[i];
                for (int dz = -kP; dz <= kP; ++dz)
                    for (int dx = -kP; dx <= kP; ++dx)
                        lo = std::min(lo, ph[std::size_t(x + kP + dx) + std::size_t(z + kP + dz) * kN]);
                groundMin[z][x] = lo;
                // Bare rock on cliffs (the rise over the two columns either side), and on gentler
                // faces the higher it is: a mountainside of grass stairs is the first thing that
                // reads as voxel terrain rather than a mountain. Low hills keep their grass unless
                // they are sheer. Snow still lies on rock unless it is a cliff.
                const int rise = std::max(std::abs(ph[i + 1] - ph[i - 1]), std::abs(ph[i + kN] - ph[i - kN]));
                // (the +1 in patches a dozen blocks across, so the rock is outcrops, not speckle)
                const float wx = float(baseX + x), wz = float(baseZ + z);
                const int need = 5 - (ph[i] > 130) - (ph[i] > 160) - (ph[i] > 190) +
                                 (valueNoise(wx * 0.07f, wz * 0.07f, seedOff(m_seed, 231)) > 0.5f ? 1 : 0);
                rock[z][x] = rise >= need + 2 ? 2 : rise >= need ? 1 : 0;
            }
    } else {
        columnGrid(baseX, baseZ, heights, desertW);
        std::memcpy(groundMin, heights, sizeof(heights));
    }
    int hMin = 1 << 30;
    int hMax = -(1 << 30);
    int wMax = -(1 << 30); // the highest lake or river surface, and what floats on it
    for (int z = 0; z < kChunkSize; ++z)
        for (int x = 0; x < kChunkSize; ++x) {
            hMin = std::min(hMin, heights[z][x]);
            hMax = std::max(hMax, heights[z][x]);
            if (tile3)
                wMax = std::max(wMax, tile3->water[z][x] + 1);
        }

    // wholly above both terrain and sea: air, except trees can still reach up into it
    if (baseY > std::max(hMax, wMax) && baseY > kSeaLevel) {
        chunk.makeUniform(blocks::Air);
        placeTrees(chunk, pos);
        // foliage sits ON the surface, so only a chunk that holds a surface cell has any
        if (baseY <= hMax + 1)
            placeFoliage(chunk, pos, heights, desertW, m_version >= 2 ? coast : nullptr, rock);
        placeStructures(chunk, pos); // a lighthouse rises well above the ground
        if (m_version < 2)
            carveTestRoom(chunk, pos); // no-op unless the room overlaps
        return;
    }

    // deep bedrock: caves only reach ~56 blocks below the surface, so a chunk entirely
    // below that band is solid stone, makeUniform costs nothing (no palette/index array,
    // no quads). this is what keeps a 256-block world as cheap as the old 128.
    // Version 2's caves go all the way down (caveFloor), but only in detailed chunks: a coarse
    // one never has caves at all, so it keeps the shortcut.
    // Version 3 has a floor: bedrock all the way down from the bottom of its caves.
    if (m_version >= 3 && baseY + kChunkSize - 1 < kCaveFloor3) {
        chunk.makeUniform(blocks::Bedrock);
        return;
    }
    const bool deepCaves = m_version >= 2 && detailed;
    if (baseY + kChunkSize - 1 < (deepCaves ? caveFloor(m_version) : hMin - kCaveDepth)) {
        chunk.makeUniform(blocks::Stone);
        if (m_version < 2)
            carveTestRoom(chunk, pos);
        return;
    }

    chunk.makeUniform(blocks::Air);
    for (int z = 0; z < kChunkSize; ++z)
        for (int x = 0; x < kChunkSize; ++x) {
            const int h = heights[z][x];
            // The desert edge is feathered per column so it is a ragged border, not a contour.
            const bool desert = desertW[z][x] > 0.42f + 0.16f * float(cellHash(baseX + x, baseZ + z) & 255u) / 255.0f &&
                                !(tile3 && tile3->biome[z][x] == Biome::Oasis);
            const bool beach = !desert && isBeach(h, coast[z][x]) && !(tile3 && tile3->biome[z][x] == Biome::Mangrove);
            const Biome bio = tile3 ? tile3->biome[z][x] : Biome::Plains;
            const float wxf = float(baseX + x), wzf = float(baseZ + z);
            // patches of a biome's other ground, a few blocks across
            const float patch = tile3 ? patchNoise(wxf, wzf, seedOff(m_seed, 311)) : 0.5f;
            // what bare rock is at the surface, in swathes across a mountainside
            const float rockKind = tile3 ? regionNoise(wxf * 0.012f, wzf * 0.012f, seedOff(m_seed, 313)) : 0.5f;
            const BlockId bareRock = rockKind < 0.22f ? blocks::Andesite : (rockKind > 0.82f ? blocks::Granite : blocks::Stone);
            const BlockId surfaceTop = [&] {
                switch (bio) {
                    case Biome::Savanna: return patch > 0.74f ? blocks::DryDirt : blocks::SavannaGrass;
                    case Biome::Jungle: return patch > 0.6f ? blocks::JungleFloor : blocks::Grass;
                    case Biome::Swamp: return patch > 0.66f ? blocks::Mud : (patch < 0.2f ? blocks::Moss : blocks::Grass);
                    case Biome::Forest: return patch > 0.8f ? blocks::Moss : blocks::Grass;
                    // mostly needles under spruce, the grass and moss in the gaps: an even split read as camouflage
                    case Biome::Taiga: return patch > 0.34f ? blocks::ForestFloor : (patch < 0.07f ? blocks::Moss : blocks::Grass);
                    case Biome::SnowyTaiga: return patch > 0.78f ? blocks::ForestFloor : blocks::SnowyGrass;
                    case Biome::Tundra: return patch > 0.64f ? blocks::Permafrost : (patch < 0.14f ? blocks::Gravel : blocks::SnowyGrass);
                    case Biome::DarkForest: return patch > 0.62f ? blocks::Moss : (patch < 0.18f ? blocks::ForestFloor : blocks::Grass);
                    case Biome::Redwood: return patch > 0.30f ? blocks::ForestFloor : (patch < 0.10f ? blocks::Moss : blocks::Grass);
                    case Biome::AspenGrove: return patch > 0.76f ? blocks::ForestFloor : blocks::Grass;
                    case Biome::AutumnForest: return patch > 0.84f ? blocks::ForestFloor : blocks::Grass;
                    case Biome::Bamboo: return patch > 0.62f ? blocks::JungleFloor : blocks::Grass;
                    case Biome::Mangrove: return patch > 0.28f ? blocks::Mud : (patch < 0.1f ? blocks::Moss : blocks::Grass);
                    default: return blocks::Grass;
                }
            }();
            const BlockId subsoil = bio == Biome::Savanna    ? blocks::DryDirt
                                    : bio == Biome::Tundra   ? blocks::Permafrost
                                    : bio == Biome::Mangrove ? blocks::Mud
                                                             : blocks::Dirt;
            // mesas: the desert's high ground in bands of coloured clay, the bands wavering a little
            const bool mesa = bio == Biome::Badlands;
            const int bandWarp = mesa ? int(valueNoise(wxf * 0.02f, wzf * 0.02f, seedOff(m_seed, 317)) * 5.0f) : 0;
            const bool seabed = h < kSeaLevel - 1;
            const bool bare = !desert && !beach && !seabed && rock[z][x] != 0;
            const int snowAt = tile3 ? int(tile3->snow[z][x]) : snowLineAt(baseX + x, baseZ + z);
            const bool snowy = !desert && !beach && !seabed && rock[z][x] < 2 && h >= snowAt;
            // a volcano's cone: dark rock, and ash lying in its hollows
            const bool volcanic = tile3 && !seabed && tile3->volcanic[z][x] > 100;
            // Ice where the snow lies deep and level well above the line: glaciers in the high basins.
            const bool glacier = tile3 && snowy && rock[z][x] == 0 && h >= snowAt + 14 && patch > 0.3f;
            // Scree: the loose stone under the crags, on slopes too steep for grass and not sheer.
            const bool scree = tile3 && rock[z][x] == 1 && !snowy && !desert && h > kSeaLevel + 20 &&
                               valueNoise(float(baseX + x) * 0.09f, float(baseZ + z) * 0.09f, seedOff(m_seed, 319)) > 0.55f;
            // patches of gravel on the sea floor break up an otherwise flat sheet of sand
            const bool gravelBed = seabed && h < kSeaLevel - 5 &&
                                   valueNoise(float(baseX + x) * 0.05f, float(baseZ + z) * 0.05f,
                                              seedOff(m_seed, 53)) > 0.66f;
            const int top = std::min(h - baseY, kChunkSize - 1);
            for (int y = 0; y <= top; ++y) {
                const int wy = baseY + y;
                BlockId id = blocks::Stone;
                // Ore veins. Depth bands rather than a flat chance, so digging down has a
                // shape to it: coal is shallow and common, iron sits deeper and rarer.
                // The hash is quantised to a 2x2x2 lattice so hits clump into small veins
                // instead of scattering one isolated block at a time.
                if (wy < h - 4) {
                    const int depth = h - wy;
                    const std::uint32_t vein = voxelHash(x >> 1, wy >> 1, z >> 1) ^
                                               voxelHash(baseX >> 5, 0, baseZ >> 5);
                    const std::uint32_t roll = vein % 1000u;
                    if (depth > 8 && roll < 26u)
                        id = blocks::CoalOre;
                    else if (depth > 28 && roll >= 26u && roll < 38u)
                        id = blocks::IronOre;
                    // Gravel rides the same lattice but in a wide shallow band, so a pocket
                    // opened by a cave collapses instead of holding a clean roof, which is
                    // the point of it falling at all.
                    else if (depth > 5 && depth < 34 && roll >= 38u && roll < 62u)
                        id = blocks::Gravel;
                }
                if (id == blocks::Stone && tile3) {
                    // slate deep down, and pockets of granite and andesite through the stone
                    if (wy < 16 + int(voxelHash(baseX + x, 5, baseZ + z) & 3u))
                        id = blocks::Slate;
                    else {
                        const int cx = (baseX + x) >> 4, cy = wy >> 4, cz = (baseZ + z) >> 4;
                        const std::uint32_t hc = voxelHash(cx, cy, cz) ^ std::uint32_t(m_seed);
                        if (hc % 100u < 22u) {
                            const int dx = ((baseX + x) & 15) - 5 - int((hc >> 8) % 6u);
                            const int dy = (wy & 15) - 5 - int((hc >> 12) % 6u);
                            const int dz = ((baseZ + z) & 15) - 5 - int((hc >> 16) % 6u);
                            const int r = 3 + int((hc >> 20) % 3u);
                            if (dx * dx + dy * dy * 2 + dz * dz < r * r)
                                id = (hc >> 24) & 1u ? blocks::Granite : blocks::Andesite;
                        }
                    }
                }
                if (mesa && wy >= std::max(h - 70, kSeaLevel - 4)) {
                    constexpr BlockId kBands[8] = {blocks::Terracotta, blocks::TerracottaRed, blocks::Terracotta, blocks::TerracottaYellow,
                                                   blocks::TerracottaBrown, blocks::TerracottaWhite, blocks::Terracotta, blocks::TerracottaRed};
                    const int band = wy + bandWarp;
                    id = kBands[voxelHash(band >> 1, 11, m_seed) & 7u];
                    if (rock[z][x] == 0 && wy >= h - 1)
                        id = blocks::RedSand; // flat tops keep a skin of sand
                } else if (wy >= h - (tile3 && (beach || desert) ? 6 : 3)) {
                    if (tile3 && tile3->water[z][x] > h) {
                        const bool clay = patch > 0.72f;
                        id = bio == Biome::Swamp || bio == Biome::Mangrove ? (clay ? blocks::Clay : blocks::Mud)
                             : clay              ? blocks::Clay
                             : (cellHash(std::uint32_t(baseX + x) * 3u, std::uint32_t(baseZ + z)) & 3u) == 0u ? blocks::Gravel
                                                                                                              : blocks::Sand;
                    } else if (beach)
                        id = blocks::BeachSand;
                    else if (seabed)
                        id = gravelBed ? blocks::Gravel : blocks::Sand;
                    else if (desert)
                        id = bio == Biome::RedDesert ? blocks::RedSand : blocks::Sand;
                    else if (glacier && wy >= h - 4)
                        id = wy == h && patch > 0.85f ? blocks::Snow : blocks::Ice;
                    else if (snowy && wy >= h - 1)
                        id = blocks::Snow; // a couple blocks of cap
                    else if (volcanic)
                        id = wy == h && patch > 0.45f ? blocks::Gravel : blocks::Slate;
                    else if (scree && wy >= h - 1)
                        id = blocks::Gravel;
                    else if (bare)
                        id = tile3 ? bareRock : blocks::Stone;
                    else if (wy == h)
                        id = surfaceTop;
                    else
                        id = subsoil;
                } else if (desert && wy >= h - 8) {
                    id = bio == Biome::RedDesert ? blocks::Terracotta : blocks::Sandstone; // the strata under the dunes
                }
                chunk.set(x, y, z, id);
            }
        }

    // caves: cheese blobs + ridged noodles, only meaningfully below the surface. skipped
    // for coarse (far) chunks, the LOD downsample collapses caves to solid anyway, and
    // the 3d noise + carve is by far the most expensive part of gen.
    if (m_version >= 2) {
        if (detailed && baseY <= hMax + 1) {
            carveCaves2(chunk, baseX, baseY, baseZ, heights, groundMin);
            placeMineshafts(chunk, pos, heights);
        }
    } else if (detailed && baseY <= hMax - 2) {
        // never destroyed on purpose: see computeConnectivity (a thread_local with a destructor
        // corrupts the heap when a worker exits, on this toolchain)
        static thread_local std::vector<float>& cheeseBuf = *new std::vector<float>();
        static thread_local std::vector<float>& wormABuf = *new std::vector<float>();
        static thread_local std::vector<float>& wormBBuf = *new std::vector<float>();
        cheeseBuf.resize(std::size_t(kChunkSize3));
        wormABuf.resize(std::size_t(kChunkSize3));
        wormBBuf.resize(std::size_t(kChunkSize3));
        // grid order: x + y * size + z * size * size
        m_cheese->GenUniformGrid3D(cheeseBuf.data(), float(baseX), float(baseY), float(baseZ),
                                   kChunkSize, kChunkSize, kChunkSize, 1.0f, 1.0f, 1.0f,
                                   seedOff(m_seed, 2));
        m_wormA->GenUniformGrid3D(wormABuf.data(), float(baseX), float(baseY), float(baseZ),
                                  kChunkSize, kChunkSize, kChunkSize, 1.0f, 1.0f, 1.0f,
                                  seedOff(m_seed, 3));
        m_wormB->GenUniformGrid3D(wormBBuf.data(), float(baseX), float(baseY), float(baseZ),
                                  kChunkSize, kChunkSize, kChunkSize, 1.0f, 1.0f, 1.0f,
                                  seedOff(m_seed, 4));
        for (int z = 0; z < kChunkSize; ++z)
            for (int y = 0; y < kChunkSize; ++y)
                for (int x = 0; x < kChunkSize; ++x) {
                    const int wy = baseY + y;
                    if (wy < 4 || wy > heights[z][x] - 4 || wy < heights[z][x] - kCaveDepth)
                        continue; // skip the skin, the bedrock floor, and deep stone
                    const std::size_t i =
                        std::size_t(x) + std::size_t(y) * kChunkSize +
                        std::size_t(z) * kChunkSize * kChunkSize;
                    // Cheese opens up with depth. Near the surface a big blob just punches
                    // a crater in the ground you fall into; the rooms worth finding belong
                    // further down, and this also keeps the crust from reading as sponge.
                    const int depth = heights[z][x] - wy;
                    const float cheeseT = depth > 40 ? 0.50f : (depth > 20 ? 0.56f : 0.62f);
                    // distance from the curve where both fields vanish -> a round tunnel
                    const float a = wormABuf[i], b = wormBBuf[i];
                    if (cheeseBuf[i] > cheeseT || a * a + b * b < kWormR2)
                        chunk.set(x, y, z, blocks::Air);
                }

        // glow deposits on cave floors, ~1 per few hundred floor cells
        for (int z = 0; z < kChunkSize; ++z)
            for (int y = 1; y < kChunkSize; ++y)
                for (int x = 0; x < kChunkSize; ++x)
                    if (chunk.get(x, y, z) == blocks::Air &&
                        chunk.get(x, y - 1, z) == blocks::Stone &&
                        (voxelHash(baseX + x, baseY + y, baseZ + z) & 511u) == 0u) {
                        // Hue comes from where it formed: warm near the surface, cold deep
                        // down, with the odd stray so a depth band is not one flat colour.
                        //
                        // Derived from depth and world position ONLY. It is tempting to
                        // look at the surrounding stone for ore, but the neighbours of an
                        // edge voxel live in a chunk that may not exist yet, reading them
                        // would make the crystal's colour depend on which chunk generated
                        // it first, which is the one thing a deferred structure cannot do.
                        const int depth = heights[z][x] - (baseY + y);
                        const std::uint32_t j = voxelHash(baseX + x, 3, baseZ + z) >> 7;
                        // warm (hue 0) shallow through to cold (hue 8) deep, +-1 of jitter
                        int hue = depth > 44 ? 8 : (depth > 34 ? 7 : (depth > 26 ? 6
                                                 : (depth > 18 ? 4 : (depth > 10 ? 2 : 0))));
                        hue += int(j % 3u) - 1;
                        if ((j & 15u) == 0u) // the odd stray so a band is not one flat colour
                            hue = int(j >> 4) % blocks::kLampHues;
                        hue = (hue + blocks::kLampHues) % blocks::kLampHues;
                        chunk.set(x, y, z, BlockId(blocks::LampFirst + hue));
                    }
    }

    // flood air that sits over the seafloor and at/below sea level (oceans + lakes).
    // caves stay dry because they're below the surface height, never above it.
    for (int z = 0; z < kChunkSize; ++z)
        for (int x = 0; x < kChunkSize; ++x) {
            const int h = heights[z][x];
            if (h >= kSeaLevel)
                continue;
            const int yLo = std::max(h + 1 - baseY, 0);
            const int yHi = std::min(kSeaLevel - baseY, kChunkSize - 1);
            for (int y = yLo; y <= yHi; ++y)
                if (chunk.get(x, y, z) == blocks::Air)
                    chunk.set(x, y, z, blocks::Water);
            // life on the sea floor, now that there is water to stand it in
            if (h < kSeaLevel - 1 && baseY <= h + 16 && baseY + kChunkSize - 1 >= h + 1)
                decorateSea(chunk, baseX, baseY, baseZ, x, z, h);
        }

    // rivers and lakes above the sea: still water, which the fluid only moves once something is dug
    if (tile3)
        for (int z = 0; z < kChunkSize; ++z)
            for (int x = 0; x < kChunkSize; ++x) {
                const int top = tile3->water[z][x];
                if (top <= heights[z][x])
                    continue;
                // cold country's lakes and rivers freeze over
                const bool frozen = tile3->biome[z][x] == Biome::Tundra || tile3->biome[z][x] == Biome::SnowyTaiga;
                for (int wy = std::max(heights[z][x] + 1, baseY); wy <= std::min(top, baseY + kChunkSize - 1); ++wy)
                    if (chunk.get(x, wy - baseY, z) == blocks::Air)
                        chunk.set(x, wy - baseY, z, frozen && wy == top ? blocks::Ice : blocks::Water);
            }

    placeTrees(chunk, pos); // trees even in coarse gen, cheap, and stops them popping in
    placeFoliage(chunk, pos, heights, desertW, m_version >= 2 ? coast : nullptr, rock);
    placeStructures(chunk, pos);
    if (m_version < 2)
        carveTestRoom(chunk, pos);
}

// Version 2's caves. Version 1 carved a band some 56 blocks under the surface and never broke it
// (a 4-block skin was left everywhere), so caves were shallow, sealed, and mostly not there at
// all. These run all the way down to the cave floor (kCaveFloor2, version 3's far deeper kCaveFloor3,
// where they also thin out and thicken from region to region):
//   - caverns, the big rooms: a 3d field squashed vertically so a chamber is wider than it is
//     tall, opening up the deeper it goes;
//   - two sets of tunnels, the worm trick from version 1 (where two fields both cross zero is a
//     line through the volume; a radius round it is a tunnel) plus a longer, wider-ranging pair
//     whose width swells and narrows along its length;
//   - ravines, rare: a 2d path cut straight down from the surface, narrowing as it goes;
//   - openings: tunnels and ravines may break out of the ground on land, which is how a cave is
//     found from outside. Never under water, a skin of rock is kept under every sea and lake
//     bed, or the water would sit over open air;
//   - water lying in the lowest caves, below a level that wanders;
//   - stalactites and stalagmites, and the glow deposits version 1 had.
// Everything is a function of world position, so it runs per chunk with no neighbour lookups.
void TerrainGenerator::carveCaves2(Chunk& chunk, int baseX, int baseY, int baseZ,
                                   const int (*heights)[kChunkSize], const int (*groundMin)[kChunkSize]) const {
    const int floorY = caveFloor(m_version);
    if (baseY + kChunkSize - 1 < floorY)
        return;
    const bool v3 = m_version >= 3;
    // never destroyed on purpose: see computeConnectivity (a thread_local with a destructor
    // corrupts the heap when a worker exits, on this toolchain)
    static thread_local std::vector<float>& cav = *new std::vector<float>();
    static thread_local std::vector<float>& wa = *new std::vector<float>();
    static thread_local std::vector<float>& wb = *new std::vector<float>();
    static thread_local std::vector<float>& ta = *new std::vector<float>();
    static thread_local std::vector<float>& tb = *new std::vector<float>();
    static thread_local std::vector<float>& tw = *new std::vector<float>();
    for (std::vector<float>* v : {&cav, &wa, &wb, &ta, &tb, &tw})
        v->resize(std::size_t(kChunkSize3));
    const float fx = float(baseX), fy = float(baseY), fz = float(baseZ);
    // grid order: x + y * size + z * size * size. The cavern field steps 1.6 a block in y, which
    // squashes its blobs to about two thirds as tall as they are wide. Version 3 has its own, on a
    // much larger scale: rooms you can lose a light in, not wide places in a tunnel.
    const float squash = v3 ? 1.45f : 1.6f;
    (v3 ? m_chamber : m_cavern)
        ->GenUniformGrid3D(cav.data(), fx, fy * squash, fz, kChunkSize, kChunkSize, kChunkSize, 1.0f, squash, 1.0f,
                           seedOff(m_seed, 221));
    m_wormA->GenUniformGrid3D(wa.data(), fx, fy, fz, kChunkSize, kChunkSize, kChunkSize, 1.0f, 1.0f, 1.0f,
                              seedOff(m_seed, 3));
    m_wormB->GenUniformGrid3D(wb.data(), fx, fy, fz, kChunkSize, kChunkSize, kChunkSize, 1.0f, 1.0f, 1.0f,
                              seedOff(m_seed, 4));
    m_tunnelA->GenUniformGrid3D(ta.data(), fx, fy, fz, kChunkSize, kChunkSize, kChunkSize, 1.0f, 1.0f, 1.0f,
                                seedOff(m_seed, 222));
    m_tunnelB->GenUniformGrid3D(tb.data(), fx, fy, fz, kChunkSize, kChunkSize, kChunkSize, 1.0f, 1.0f, 1.0f,
                                seedOff(m_seed, 223));
    m_tunnelW->GenUniformGrid3D(tw.data(), fx, fy, fz, kChunkSize, kChunkSize, kChunkSize, 1.0f, 1.0f, 1.0f,
                                seedOff(m_seed, 224));
    float rav[kChunkSize * kChunkSize], ravMask[kChunkSize * kChunkSize];
    m_ravine->GenUniformGrid2D(rav, fx, fz, kChunkSize, kChunkSize, 1.0f, 1.0f, seedOff(m_seed, 225));
    m_ravineMask->GenUniformGrid2D(ravMask, fx, fz, kChunkSize, kChunkSize, 1.0f, 1.0f, seedOff(m_seed, 226));

    for (int z = 0; z < kChunkSize; ++z)
        for (int x = 0; x < kChunkSize; ++x) {
            const int h = heights[z][x];
            const int wx = baseX + x, wz = baseZ + z;
            // How much rock there is over a cell is measured two ways: straight down from this
            // column's surface (depth), and from the lowest ground within kSidePad (cover), which
            // is what tells a cell it is just inside a hillside. Measuring straight down only, a
            // tunnel running along a slope a block or two in broke out of its side for its whole
            // length, and a cavern under one opened the hill up: the gaps and chasms in the hills.
            const int side = groundMin[z][x];
            const bool wet = h < kSeaLevel + 2; // under the sea or a lake: never open, or it drains
            // A few scattered spots where a tunnel IS let out into the open: a cave mouth.
            const bool mouth = !wet && valueNoise(float(wx) * 0.02f, float(wz) * 0.02f, seedOff(m_seed, 229)) > 0.93f;
            // the lowest caves stand in water up to here, a level that wanders across the map
            const int aquifer = (v3 ? floorY + 6 : 12) +
                                int(valueNoise(float(wx) * 0.008f, float(wz) * 0.008f, seedOff(m_seed, 227)) * 10.0f);
            // Version 3: how riddled the rock is here, over a few hundred blocks. Some country is
            // honeycombed with tunnels and great rooms, some is near solid with the odd passage,
            // instead of the same density of the same shapes under every hill.
            float richCol = 1.0f;
            // Version 3: columns of rock left standing through the great rooms, and in some country a
            // lake lying in the caves, at a level of its own.
            bool pillar = false;
            int lakeLevel = INT32_MIN, damBelow = INT32_MIN;
            if (v3) {
                const float n = 0.75f * valueNoise(float(wx) * 0.0016f, float(wz) * 0.0016f, seedOff(m_seed, 230)) +
                                0.25f * valueNoise(float(wx) * 0.006f, float(wz) * 0.006f, seedOff(m_seed, 231));
                richCol = smooth01(0.34f, 0.86f, n); // most country is solid rock with the odd passage
                pillar = valueNoise(float(wx) * 0.055f, float(wz) * 0.055f, seedOff(m_seed, 235)) > 0.9f;
                constexpr int kLakeCell = 192;
                const auto levelOf = [&](int rx, int rz) {
                    const std::uint32_t lc = cellHash(std::uint32_t(rx) + std::uint32_t(seedOff(m_seed, 236)), std::uint32_t(rz));
                    return lc % 100u < 40u ? -70 + int((lc >> 8) % 70u) : INT32_MIN;
                };
                const int rx = floorDivI(wx, kLakeCell), rz = floorDivI(wz, kLakeCell);
                lakeLevel = levelOf(rx, rz);
                // Where two of these meet at different levels the water would stand as a wall: a dam
                // of rock along the line instead, up to the higher of the two.
                const int ox = wx - rx * kLakeCell, oz = wz - rz * kLakeCell;
                for (const auto& [nx, nz] : {std::pair<int, int>{ox < 2 ? rx - 1 : (ox >= kLakeCell - 2 ? rx + 1 : rx), rz},
                                             {rx, oz < 2 ? rz - 1 : (oz >= kLakeCell - 2 ? rz + 1 : rz)}})
                    if (nx != rx || nz != rz) {
                        const int other = levelOf(nx, nz);
                        if (other != lakeLevel)
                            damBelow = std::max(damBelow, std::max(other, lakeLevel));
                    }
            }
            // A ravine: where the path field is near zero inside the rare mask, cut down from the
            // surface by up to ~45, narrower the deeper it goes so its walls lean in. Rare: the
            // first version's mask covered a fifth of the land. Version 3 has none: a winding
            // trench that starts and stops in open country reads as a scratch across the map.
            const int i2 = x + z * kChunkSize;
            const float rm = v3 ? 0.0f : smooth01(0.48f, 0.58f, ravMask[i2]);
            const int ravDepth = rm > 0.0f ? int(18.0f + 27.0f * rm) : 0;
            for (int y = 0; y < kChunkSize; ++y) {
                const int wy = baseY + y;
                if (wy < floorY || wy > h)
                    continue;
                const int depth = h - wy;
                const int cover = side - wy;
                if (wet && std::min(depth, cover) < 6)
                    continue;
                const std::size_t i = std::size_t(x) + std::size_t(y) * kChunkSize +
                                      std::size_t(z) * kChunkSize * kChunkSize;
                // and in layers going down: a riddled level can sit over solid rock and another
                // riddled level under that, so digging down is not the same few metres over and over
                float rich = richCol;
                if (v3) {
                    const float band = valueNoise(float(wy) * 0.03f, float(wx + wz) * 0.0011f, seedOff(m_seed, 232));
                    rich = std::clamp(richCol * (0.25f + 1.0f * band), 0.0f, 1.0f);
                }
                // version 3's tunnels are the passages between its rooms, not the whole of its caves
                const float wormR2 = v3 ? kWormR2 * (0.02f + 0.18f * rich) : kWormR2;
                const float tunnelScale = v3 ? 0.03f + 0.25f * rich : 1.0f;
                bool open = false;
                // caverns, never within 12 of the ground in any direction: there a big room is
                // just a crater, or a hole in a hillside
                if (cover >= 12) {
                    const int deep = std::min(depth, cover);
                    const float t = v3 ? (deep > 48 ? 0.43f : (deep > 24 ? 0.49f : 0.58f)) + 0.32f * (1.0f - rich)
                                       : (deep > 48 ? 0.42f : (deep > 24 ? 0.50f : 0.60f));
                    open = cav[i] > t && !pillar;
                }
                // tunnels: the short pair, and the long one whose radius breathes; 4 of rock over
                // them on every side, except at a mouth
                if (!open && (cover >= 4 || mouth)) {
                    open = wa[i] * wa[i] + wb[i] * wb[i] < wormR2;
                    if (!open) {
                        const float r2 = (0.05f + 0.16f * (tw[i] * 0.5f + 0.5f)) * tunnelScale;
                        open = ta[i] * ta[i] + tb[i] * tb[i] < r2;
                    }
                }
                if (!open && depth < ravDepth && !wet) {
                    // a few blocks across at the top (0.045 cut gashes 20-30 wide), pinching in below
                    const float width = 0.016f * (1.0f - 0.7f * float(depth) / float(ravDepth));
                    open = std::abs(rav[i2]) < width;
                }
                if (open && wy <= damBelow)
                    open = false;
                if (open)
                    chunk.set(x, y, z, wy <= std::max(aquifer, lakeLevel) ? blocks::Water : blocks::Air);
            }
            // Version 3's bedrock: solid at the cave floor, then ragged for three blocks up, so the
            // lowest caves stand on it and nothing ever opens under them.
            if (v3)
                for (int wy = std::max(floorY, baseY); wy <= std::min(floorY + 3, baseY + kChunkSize - 1); ++wy)
                    if ((voxelHash(wx, wy, wz) & 3u) >= std::uint32_t(wy - floorY))
                        chunk.set(x, wy - baseY, z, blocks::Bedrock);
        }

    if (v3) {
        placeGeodes(chunk, baseX, baseY, baseZ);
        decorateCaves3(chunk, baseX, baseY, baseZ, heights);
        return;
    }
    // Stalactites from cave ceilings and stalagmites from their floors, a few blocks long, and
    // the glow deposits on the floors. Only inside this chunk: a spike that would cross into the
    // next one is cut, which is invisible underground.
    for (int z = 0; z < kChunkSize; ++z)
        for (int x = 0; x < kChunkSize; ++x) {
            const int h = heights[z][x];
            for (int y = 1; y < kChunkSize - 1; ++y) {
                const int wy = baseY + y;
                if (wy > h - 5 || chunk.get(x, y, z) != blocks::Air)
                    continue;
                const std::uint32_t r = voxelHash(baseX + x, wy, baseZ + z);
                const BlockId above = chunk.get(x, y + 1, z), below = chunk.get(x, y - 1, z);
                if (above == blocks::Stone && (r & 127u) < 3u) {
                    const int len = 1 + int((r >> 8) % 3u);
                    for (int k = 0; k < len && y - k >= 0 && chunk.get(x, y - k, z) == blocks::Air; ++k)
                        chunk.set(x, y - k, z, blocks::Stone);
                } else if (below == blocks::Stone && (r & 127u) >= 3u && (r & 127u) < 5u) {
                    const int len = 1 + int((r >> 10) % 2u);
                    for (int k = 0; k < len && y + k < kChunkSize && chunk.get(x, y + k, z) == blocks::Air; ++k)
                        chunk.set(x, y + k, z, blocks::Stone);
                } else if (below == blocks::Stone && (r & 511u) == 511u) {
                    // the coloured glow deposits, by the same warm-shallow / cold-deep hue rule
                    // version 1 uses
                    const int depth = h - wy;
                    const std::uint32_t j = voxelHash(baseX + x, 3, baseZ + z) >> 7;
                    int hue = depth > 44 ? 8 : (depth > 34 ? 7 : (depth > 26 ? 6 : (depth > 18 ? 4 : (depth > 10 ? 2 : 0))));
                    hue += int(j % 3u) - 1;
                    if ((j & 15u) == 0u)
                        hue = int(j >> 4) % blocks::kLampHues;
                    hue = (hue + blocks::kLampHues) % blocks::kLampHues;
                    chunk.set(x, y, z, BlockId(blocks::LampFirst + hue));
                }
            }
        }
}

// Crystal geodes: a hollow lined with glowing crystal, inside a shell of pale rock and then dark,
// in a few of the 48-block cells below y 10. A cave that runs into one runs into its shell.
void TerrainGenerator::placeGeodes(Chunk& chunk, int baseX, int baseY, int baseZ) const {
    constexpr int kCell = 48, kReach = 9;
    const int cx0 = floorDivI(baseX - kReach, kCell), cx1 = floorDivI(baseX + kChunkSize - 1 + kReach, kCell);
    const int cy0 = floorDivI(baseY - kReach, kCell), cy1 = floorDivI(baseY + kChunkSize - 1 + kReach, kCell);
    const int cz0 = floorDivI(baseZ - kReach, kCell), cz1 = floorDivI(baseZ + kChunkSize - 1 + kReach, kCell);
    for (int cy = cy0; cy <= cy1; ++cy)
        for (int cz = cz0; cz <= cz1; ++cz)
            for (int cx = cx0; cx <= cx1; ++cx) {
                const std::uint32_t hs = voxelHash(cx, cy, cz) ^ std::uint32_t(seedOff(m_seed, 250));
                if (hs % 100u >= 14u)
                    continue;
                const int gx = cx * kCell + 9 + int((hs >> 7) % std::uint32_t(kCell - 18));
                const int gy = cy * kCell + 9 + int((hs >> 13) % std::uint32_t(kCell - 18));
                const int gz = cz * kCell + 9 + int((hs >> 19) % std::uint32_t(kCell - 18));
                if (gy > 10 || gy < kCaveFloor3 + 14)
                    continue;
                const float r = 4.0f + float((hs >> 25) % 4u);
                const int ri = int(r) + 2;
                for (int wy = std::max(gy - ri, baseY); wy <= std::min(gy + ri, baseY + kChunkSize - 1); ++wy)
                    for (int wz = std::max(gz - ri, baseZ); wz <= std::min(gz + ri, baseZ + kChunkSize - 1); ++wz)
                        for (int wx = std::max(gx - ri, baseX); wx <= std::min(gx + ri, baseX + kChunkSize - 1); ++wx) {
                            const float dx = float(wx - gx), dy = float(wy - gy), dz = float(wz - gz);
                            const float d = std::sqrt(dx * dx + dy * dy + dz * dz);
                            BlockId id;
                            if (d < r - 1.6f)
                                id = blocks::Air;
                            else if (d < r - 0.6f)
                                id = blocks::Crystal;
                            else if (d < r + 0.4f)
                                id = blocks::Andesite;
                            else if (d < r + 1.4f)
                                id = blocks::Slate;
                            else
                                continue;
                            chunk.set(wx - baseX, wy - baseY, wz - baseZ, id);
                        }
            }
}

// What the caves are lined with, by what lies above them. Lush under wet country: moss on the
// floors and walls, plants, vines down from the roof with glow berries on them. Dripstone under
// dry country and here and there anywhere: its floors and roofs and the long points hanging and
// standing. Frozen under the cold: ice and snow, icicles. Deep down, now and then, the fungal
// caves: mud, mushrooms and the glowing ones. Everywhere else stone spikes and the old coloured
// glow deposits. Inside this chunk only, like the rest of the carving: a point that would cross
// into the next one is cut.
void TerrainGenerator::decorateCaves3(Chunk& chunk, int baseX, int baseY, int baseZ,
                                      const int (*heights)[kChunkSize]) const {
    enum Kind : std::uint8_t { Plain, Lush, Drip, Frozen, Fungal };
    const auto tile = columnTile(floorDivI(baseX, kChunkSize), floorDivI(baseZ, kChunkSize));
    const auto rock = [](BlockId b) {
        return b == blocks::Stone || b == blocks::Slate || b == blocks::Granite || b == blocks::Andesite ||
               b == blocks::CoalOre || b == blocks::IronOre;
    };
    const auto inChunk = [](int x, int y, int z) {
        return x >= 0 && x < kChunkSize && y >= 0 && y < kChunkSize && z >= 0 && z < kChunkSize;
    };
    for (int z = 0; z < kChunkSize; ++z)
        for (int x = 0; x < kChunkSize; ++x) {
            const int h = heights[z][x];
            const int wx = baseX + x, wz = baseZ + z;
            const Biome sb = tile->biome[z][x];
            const bool cold = sb == Biome::Tundra || sb == Biome::SnowyTaiga || sb == Biome::Snowcap || sb == Biome::Alpine;
            const bool wetLand = sb == Biome::Jungle || sb == Biome::Swamp || sb == Biome::Forest ||
                                 sb == Biome::BirchForest || tile->moisture[z][x] > 160;
            const bool dry = sb == Biome::Desert || sb == Biome::RedDesert || sb == Biome::Badlands || sb == Biome::Savanna;
            const float cn = valueNoise(float(wx) * 0.0045f, float(wz) * 0.0045f, seedOff(m_seed, 240));
            const float cn2 = valueNoise(float(wx) * 0.011f, float(wz) * 0.011f, seedOff(m_seed, 241));
            const Kind shallow = cold                  ? Frozen
                                 : wetLand && cn > 0.25f ? Lush
                                 : dry && cn > 0.10f     ? Drip
                                 : cn > 0.80f            ? Lush
                                 : cn < 0.16f            ? Drip
                                                         : Plain;
            const Kind deepKind = cn2 > 0.72f ? Fungal : (cn < 0.22f ? Drip : (wetLand && cn > 0.55f ? Lush : Plain));
            for (int y = 0; y < kChunkSize; ++y) {
                const int wy = baseY + y;
                if (wy > h - 4 || chunk.get(x, y, z) != blocks::Air)
                    continue;
                // what lies above sets the caves down to around y 10; below that it is the deep's own
                const Kind kind = wy > 6 + int(cn2 * 12.0f) ? shallow : deepKind;
                const std::uint32_t r = voxelHash(wx, wy, wz);
                // the walls take the lining as well as the floor and the roof: moss creeping up them,
                // dripstone flowstone down them, ice
                if (kind == Lush || kind == Drip || kind == Frozen) {
                    const BlockId lining = kind == Lush ? blocks::Moss : (kind == Drip ? blocks::Dripstone : blocks::Ice);
                    const std::uint32_t share = kind == Lush ? 55u : (kind == Drip ? 70u : 40u);
                    for (const auto& d : {std::pair<int, int>{1, 0}, {-1, 0}, {0, 1}, {0, -1}})
                        if (inChunk(x + d.first, y, z + d.second) && rock(chunk.get(x + d.first, y, z + d.second)) &&
                            voxelHash(wx + d.first, wy, wz + d.second) % 100u < share)
                            chunk.set(x + d.first, y, z + d.second, lining);
                }
                const BlockId below = y > 0 ? chunk.get(x, y - 1, z) : blocks::Air;
                const BlockId above = y + 1 < kChunkSize ? chunk.get(x, y + 1, z) : blocks::Air;
                const bool floor = rock(below), roof = rock(above);
                const auto hang = [&](int len, BlockId body, BlockId tip) {
                    for (int k = 0; k < len && inChunk(x, y - k, z) && chunk.get(x, y - k, z) == blocks::Air; ++k)
                        chunk.set(x, y - k, z, k + 1 == len ? tip : body);
                };
                const auto stand = [&](int len, BlockId body, BlockId tip) {
                    for (int k = 0; k < len && inChunk(x, y + k, z) && chunk.get(x, y + k, z) == blocks::Air; ++k)
                        chunk.set(x, y + k, z, k + 1 == len ? tip : body);
                };
                switch (kind) {
                    case Lush:
                        if (floor) {
                            chunk.set(x, y - 1, z, (r % 10u) < 8u ? blocks::Moss : blocks::Clay);
                            const std::uint32_t p = (r >> 4) % 100u;
                            if (p < 30u)
                                chunk.set(x, y, z, p < 12u ? blocks::Fern : (p < 22u ? blocks::GrassTuft : blocks::TallGrass));
                        }
                        if (roof) {
                            if ((r >> 8) % 10u < 7u)
                                chunk.set(x, y + 1, z, blocks::Moss);
                            if ((r >> 12) % 100u < 22u) {
                                const int len = 1 + int((r >> 16) % 5u);
                                for (int k = 0; k < len && inChunk(x, y - k, z) && chunk.get(x, y - k, z) == blocks::Air; ++k)
                                    chunk.set(x, y - k, z, voxelHash(wx, wy - k, wz + 7) % 6u == 0u ? blocks::GlowBerries
                                                                                                  : blocks::CaveVines);
                            }
                        }
                        break;
                    case Drip:
                        if (floor) {
                            if (r % 4u != 0u)
                                chunk.set(x, y - 1, z, blocks::Dripstone);
                            if ((r >> 4) % 100u < 7u)
                                stand(1 + int((r >> 10) % 3u), blocks::Dripstone, blocks::DripstoneUp);
                        } else if (roof) {
                            if ((r >> 6) % 4u != 0u)
                                chunk.set(x, y + 1, z, blocks::Dripstone);
                            if ((r >> 8) % 100u < 16u)
                                hang(1 + int((r >> 12) % 4u), blocks::Dripstone, blocks::DripstoneDown);
                        }
                        break;
                    case Frozen:
                        if (floor)
                            chunk.set(x, y - 1, z, r % 3u == 0u ? blocks::Snow : blocks::Ice);
                        else if (roof) {
                            if ((r >> 6) % 2u == 0u)
                                chunk.set(x, y + 1, z, blocks::Ice);
                            if ((r >> 8) % 100u < 6u)
                                hang(1 + int((r >> 12) % 3u), blocks::Ice, blocks::Ice);
                        }
                        break;
                    case Fungal:
                        if (floor) {
                            chunk.set(x, y - 1, z, (r % 5u) < 4u ? blocks::Mud : blocks::Moss);
                            const std::uint32_t p = (r >> 4) % 100u;
                            if (p < 24u)
                                chunk.set(x, y, z, p < 7u ? blocks::GlowShroom
                                                          : (p < 16u ? blocks::MushroomBrown : blocks::MushroomRed));
                        }
                        break;
                    case Plain:
                        if (roof && (r & 127u) < 3u)
                            hang(1 + int((r >> 8) % 3u), blocks::Stone, blocks::Stone);
                        else if (floor && (r & 127u) >= 3u && (r & 127u) < 5u)
                            stand(1 + int((r >> 10) % 2u), blocks::Stone, blocks::Stone);
                        else if (floor && (r & 511u) == 511u) {
                            // the coloured glow deposits: warm near the surface, cold deep down
                            const int depth = h - wy;
                            const std::uint32_t j = voxelHash(wx, 3, wz) >> 7;
                            int hue = depth > 44 ? 8 : (depth > 34 ? 7 : (depth > 26 ? 6 : (depth > 18 ? 4 : (depth > 10 ? 2 : 0))));
                            hue += int(j % 3u) - 1;
                            if ((j & 15u) == 0u)
                                hue = int(j >> 4) % blocks::kLampHues;
                            hue = (hue + blocks::kLampHues) % blocks::kLampHues;
                            chunk.set(x, y, z, BlockId(blocks::LampFirst + hue));
                        }
                        break;
                }
            }
        }
}

int TerrainGenerator::shaftNetwork(int cx, int cz, int& y, ShaftSeg* out) const {
    std::uint32_t s = cellHash(wrapMul(cx, 2654435761u) + std::uint32_t(seedOff(m_seed, 231)),
                               wrapMul(cz, 40503u) + std::uint32_t(seedOff(m_seed, 233)));
    const auto next = [&s] {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        return s;
    };
    if (next() % 100u >= 35u)
        return 0; // about a third of cells have one
    const int ax = cx * kShaftCell + 32 + int(next() % std::uint32_t(kShaftCell - 64));
    const int az = cz * kShaftCell + 32 + int(next() % std::uint32_t(kShaftCell - 64));
    const int h = column(ax, az).h;
    if (h < kSeaLevel + 4)
        return 0; // under the sea or a lake: nobody dug here
    y = std::clamp(h - 18 - int(next() % 14u), 14, 52);

    // A level random walk: each corridor runs 8..24 blocks, then the next carries on straight or
    // turns, and now and then a junction sends a second one off the other way. Kept within
    // kReachShaft of the anchor so a chunk knows which cells can reach it.
    constexpr int kReachShaft = 60;
    int n = 0;
    struct Open {
        int x, z, dir;
    };
    Open stack[kShaftMaxSegs];
    int top = 0;
    stack[top++] = {ax, az, int(next() % 4u)};
    while (top > 0 && n < kShaftMaxSegs) {
        const Open o = stack[--top];
        const int len = 8 + int(next() % 17u);
        static constexpr int kDx[4] = {1, -1, 0, 0}, kDz[4] = {0, 0, 1, -1};
        const int ex = o.x + kDx[o.dir] * len, ez = o.z + kDz[o.dir] * len;
        if (std::abs(ex - ax) > kReachShaft || std::abs(ez - az) > kReachShaft)
            continue;
        out[n++] = {o.x, o.z, o.dir, len};
        const std::uint32_t r = next() % 100u;
        // turn: the two directions square to this one
        const int left = o.dir < 2 ? 2 + int(next() & 1u) : int(next() & 1u);
        const int right = o.dir < 2 ? (left == 2 ? 3 : 2) : (left == 0 ? 1 : 0);
        if (r < 45u && top < kShaftMaxSegs)
            stack[top++] = {ex, ez, o.dir}; // straight on
        else if (r < 75u && top < kShaftMaxSegs)
            stack[top++] = {ex, ez, left};
        if (r >= 45u && r < 90u && top < kShaftMaxSegs)
            stack[top++] = {ex, ez, right}; // a junction, or the turn the other way
    }
    return n;
}

// Mineshafts, carved after the caves so a corridor runs straight through a cavern on a bridge.
// Corridors are 3 wide and 3 high. Every 4 blocks a frame: two log posts and a plank beam over
// them, and every third frame a lamp hung from the beam. The floor is patched with planks, and
// where there is no floor at all (a cave under it) planks on log stilts down to the rock. A
// corridor keeps 4 blocks of rock over it: where the ground comes lower it simply stops.
void TerrainGenerator::placeMineshafts(Chunk& chunk, ChunkPos pos, const int (*heights)[kChunkSize]) const {
    const int baseX = pos.x * kChunkSize, baseY = pos.y * kChunkSize, baseZ = pos.z * kChunkSize;
    constexpr int kReach = 60 + 24 + 2; // the network's own reach, a corridor's length, its width
    const int cx0 = floorDivI(baseX - kReach, kShaftCell), cx1 = floorDivI(baseX + kChunkSize - 1 + kReach, kShaftCell);
    const int cz0 = floorDivI(baseZ - kReach, kShaftCell), cz1 = floorDivI(baseZ + kChunkSize - 1 + kReach, kShaftCell);
    const auto inside = [&](int wx, int wy, int wz) {
        return wx >= baseX && wx < baseX + kChunkSize && wy >= baseY && wy < baseY + kChunkSize && wz >= baseZ &&
               wz < baseZ + kChunkSize;
    };
    const auto get = [&](int wx, int wy, int wz) { return chunk.get(wx - baseX, wy - baseY, wz - baseZ); };
    const auto put = [&](int wx, int wy, int wz, BlockId id) {
        if (inside(wx, wy, wz))
            chunk.set(wx - baseX, wy - baseY, wz - baseZ, id);
    };
    // ground height at a column of this chunk (corridors outside it are not built here)
    const auto roofOk = [&](int wx, int wz, int y) {
        return heights[wz - baseZ][wx - baseX] - 4 >= y + 2;
    };
    const BlockId lamp = BlockId(blocks::LampFirst + blocks::kLampHues + 1); // pale amber
    ShaftSeg segs[kShaftMaxSegs];
    for (int cz = cz0; cz <= cz1; ++cz)
        for (int cx = cx0; cx <= cx1; ++cx) {
            int y = 0;
            const int n = shaftNetwork(cx, cz, y, segs);
            if (n == 0 || y + 3 < baseY || y - 13 > baseY + kChunkSize - 1)
                continue;
            for (int si = 0; si < n; ++si) {
                const ShaftSeg& sg = segs[si];
                static constexpr int kDx[4] = {1, -1, 0, 0}, kDz[4] = {0, 0, 1, -1};
                const int sx = kDz[sg.dir] != 0 ? 1 : 0, sz = kDx[sg.dir] != 0 ? 1 : 0; // sideways
                for (int t = 0; t <= sg.len; ++t) {
                    const int wx = sg.x + kDx[sg.dir] * t, wz = sg.z + kDz[sg.dir] * t;
                    // the corridor, wherever its centre column is in this chunk and still under rock
                    for (int w = -1; w <= 1; ++w) {
                        const int px = wx + sx * w, pz = wz + sz * w;
                        if (px < baseX || px >= baseX + kChunkSize || pz < baseZ || pz >= baseZ + kChunkSize)
                            continue;
                        if (!roofOk(px, pz, y))
                            continue;
                        for (int k = 0; k < 3; ++k)
                            if (inside(px, y + k, pz) && !blocks::isWater(get(px, y + k, pz)))
                                put(px, y + k, pz, blocks::Air);
                        if (inside(px, y - 1, pz)) {
                            const BlockId under = get(px, y - 1, pz);
                            const std::uint32_t r = voxelHash(px, y, pz);
                            if (under == blocks::Air || blocks::isWater(under) || (r & 3u) != 0u)
                                put(px, y - 1, pz, blocks::Planks);
                        }
                        // the odd seam of ore showing in the walls
                        const int ox = px + sx * w, oz = pz + sz * w; // one further out
                        if (w != 0 && inside(ox, y + 1, oz) && get(ox, y + 1, oz) == blocks::Stone) {
                            const std::uint32_t r = voxelHash(ox, y + 1, oz) % 100u;
                            if (r < 4u)
                                put(ox, y + 1, oz, r < 3u ? blocks::CoalOre : blocks::IronOre);
                        }
                    }
                    if (t % 4 != 0 || !roofOk(std::clamp(wx, baseX, baseX + kChunkSize - 1),
                                               std::clamp(wz, baseZ, baseZ + kChunkSize - 1), y))
                        continue;
                    // a frame: posts each side, a beam across the top, and stilts under a bridge
                    for (const int w : {-1, 1}) {
                        const int px = wx + sx * w, pz = wz + sz * w;
                        put(px, y, pz, blocks::Wood);
                        put(px, y + 1, pz, blocks::Wood);
                        for (int k = 2; k <= 13 && inside(px, y - k, pz) && get(px, y - k, pz) == blocks::Air; ++k)
                            put(px, y - k, pz, blocks::Wood);
                    }
                    // the beam, with a lamp set into its middle every third frame: up in the
                    // beam, not hanging in the corridor where it would be in the way
                    for (int w = -1; w <= 1; ++w)
                        put(wx + sx * w, y + 2, wz + sz * w, w == 0 && t % 12 == 0 ? lamp : blocks::Planks);
                }
            }
        }
}

TerrainGenerator::StructureSite TerrainGenerator::pyramidSite(int cx, int cz, int& halfWidth) const {
    StructureSite none;
    if (m_version < 2)
        return none;
    const std::uint32_t hsh = cellHash(wrapMul(cx, 7907u) + std::uint32_t(seedOff(m_seed, 241)),
                                       wrapMul(cz, 104723u) + std::uint32_t(seedOff(m_seed, 243)));
    if ((hsh % 100u) >= 55u)
        return none;
    const int ax = cx * kPyramidCell + 30 + int((hsh >> 8) % std::uint32_t(kPyramidCell - 60));
    const int az = cz * kPyramidCell + 30 + int((hsh >> 16) % std::uint32_t(kPyramidCell - 60));
    halfWidth = 11 + int((hsh >> 24) % 5u);
    // Deep in the desert all the way across the footprint, and near enough level that the base
    // course is not a cliff on one side. It stands at the middle of the ground under it: it used to
    // take the lowest of five samples, and between dunes that is a trough, so every pyramid stood
    // half buried in the sand around it.
    int hs[25];
    int n = 0;
    for (int j = -2; j <= 2; ++j)
        for (int i = -2; i <= 2; ++i) {
            const Column c = column(ax + i * halfWidth / 2, az + j * halfWidth / 2);
            if (c.desert < 0.6f || c.h < kSeaLevel + 4)
                return none;
            hs[n++] = c.h;
        }
    std::sort(hs, hs + n);
    if (hs[n - 1] - hs[0] > 12)
        return none;
    StructureSite s;
    s.kind = 4;
    s.x = ax;
    s.z = az;
    s.h = hs[n / 2];
    return s;
}

// Pyramids: a stepped sandstone pyramid, each course one block in from the one under it. A
// corridor runs in from one side at ground level to a chamber at the heart lit by four lamps set
// in its walls; from the chamber a stepped passage goes down to a hidden room with a glow block on
// its floor. Sand is drifted against two of its sides, the dunes are cleared off its steps, and a
// warm lamp caps it so it catches the eye from a long way off.
void TerrainGenerator::placePyramids(Chunk& chunk, ChunkPos pos) const {
    if (m_version < 2)
        return;
    const int baseX = pos.x * kChunkSize, baseY = pos.y * kChunkSize, baseZ = pos.z * kChunkSize;
    constexpr int kReach = 15 + 4;
    const int cx0 = floorDivI(baseX - kReach, kPyramidCell), cx1 = floorDivI(baseX + kChunkSize - 1 + kReach, kPyramidCell);
    const int cz0 = floorDivI(baseZ - kReach, kPyramidCell), cz1 = floorDivI(baseZ + kChunkSize - 1 + kReach, kPyramidCell);
    const auto put = [&](int wx, int wy, int wz, BlockId id) {
        const int lx = wx - baseX, ly = wy - baseY, lz = wz - baseZ;
        if (lx >= 0 && lx < kChunkSize && ly >= 0 && ly < kChunkSize && lz >= 0 && lz < kChunkSize)
            chunk.set(lx, ly, lz, id);
    };
    const auto get = [&](int wx, int wy, int wz) {
        const int lx = wx - baseX, ly = wy - baseY, lz = wz - baseZ;
        if (lx >= 0 && lx < kChunkSize && ly >= 0 && ly < kChunkSize && lz >= 0 && lz < kChunkSize)
            return chunk.get(lx, ly, lz);
        return blocks::Stone;
    };
    for (int cz = cz0; cz <= cz1; ++cz)
        for (int cx = cx0; cx <= cx1; ++cx) {
            int R = 0;
            const StructureSite st = pyramidSite(cx, cz, R);
            if (st.kind == 0)
                continue;
            const int ax = st.x, az = st.z, y0 = st.h;
            if (baseY > y0 + R + 4 || baseY + kChunkSize - 1 < y0 - 16) // the foundation reaches 12 + 3 down
                continue;
            const std::uint32_t rv = voxelHash(ax, 97, az);
            // which side the door is on, and the two sides the sand has drifted against
            static constexpr int kDx[4] = {1, -1, 0, 0}, kDz[4] = {0, 0, 1, -1};
            const int door = int(rv & 3u);
            const int driftA = int((rv >> 2) & 3u), driftB = int((rv >> 4) & 3u);

            // the body: courses up to the top, a foundation under it, the dunes cleared off it
            for (int dz = -R - 3; dz <= R + 3; ++dz)
                for (int dx = -R - 3; dx <= R + 3; ++dx) {
                    const int m = std::max(std::abs(dx), std::abs(dz));
                    const int wx = ax + dx, wz = az + dz;
                    if (m <= R) {
                        // down to the ground under it wherever that dips below the base course
                        const int top = y0 + (R - m);
                        const int foot = std::min(y0, column(wx, wz).h) - 3;
                        for (int wy = foot; wy <= top; ++wy)
                            put(wx, wy, wz, blocks::Sandstone);
                        for (int wy = top + 1; wy <= y0 + R + 3; ++wy)
                            put(wx, wy, wz, blocks::Air);
                        continue;
                    }
                    // drifts: sand piled against the base on two sides, thinning outward
                    const int out = m - R; // 1..3
                    const bool onA = (kDx[driftA] != 0 ? dx * kDx[driftA] > 0 && std::abs(dx) > R
                                                       : dz * kDz[driftA] > 0 && std::abs(dz) > R);
                    const bool onB = (kDx[driftB] != 0 ? dx * kDx[driftB] > 0 && std::abs(dx) > R
                                                       : dz * kDz[driftB] > 0 && std::abs(dz) > R);
                    if (!onA && !onB)
                        continue;
                    const int pile = 3 - out + int(voxelHash(wx, 98, wz) % 2u);
                    for (int k = 0; k < pile; ++k)
                        if (get(wx, y0 + 1 + k, wz) == blocks::Air)
                            put(wx, y0 + 1 + k, wz, blocks::Sand);
                }

            // the corridor in from the door to the chamber, 3 wide and 3 high
            const int sx = kDz[door] != 0 ? 1 : 0, sz = kDx[door] != 0 ? 1 : 0;
            for (int t = 3; t <= R + 1; ++t)
                for (int w = -1; w <= 1; ++w)
                    for (int k = 1; k <= 3; ++k)
                        put(ax + kDx[door] * t + sx * w, y0 + k, az + kDz[door] * t + sz * w, blocks::Air);
            // the chamber, 7 x 7 and 5 high, lamps set into the middle of each wall
            for (int dz = -3; dz <= 3; ++dz)
                for (int dx = -3; dx <= 3; ++dx)
                    for (int k = 1; k <= 5; ++k)
                        put(ax + dx, y0 + k, az + dz, blocks::Air);
            // pale amber: the saturated one flooded the whole chamber orange
            const BlockId warm = BlockId(blocks::LampFirst + blocks::kLampHues + 1);
            for (int d = 0; d < 4; ++d)
                put(ax + kDx[d] * 4, y0 + 3, az + kDz[d] * 4, warm);
            // down from the chamber's back corner: steps, one block down for each one along
            const int back = door ^ 1; // the opposite side
            for (int s = 0; s < 7; ++s) {
                const int wx = ax + kDx[back] * (3 - s) - sx * 3, wz = az + kDz[back] * (3 - s) - sz * 3;
                for (int k = 0; k <= 2; ++k)
                    put(wx, y0 - s + k, wz, blocks::Air);
            }
            // the hidden room the steps come down into, under the front half of the chamber and
            // over to the stairs' side, a glow block set in its floor
            const int rx = ax + kDx[back] * -4 - sx * 2, rz = az + kDz[back] * -4 - sz * 2;
            for (int dz = -2; dz <= 2; ++dz)
                for (int dx = -2; dx <= 2; ++dx)
                    for (int k = 0; k < 3; ++k)
                        put(rx + dx, y0 - 7 + k, rz + dz, blocks::Air);
            put(rx, y0 - 8, rz, blocks::Glow);
            // the cap
            put(ax, y0 + R, az, warm);
        }
}

// Coral reefs, kelp forests and seagrass meadows. Everything here is a pure function of the
// column's world position, so a reef that straddles a chunk border is built identically by
// both halves with no neighbour lookups, the same rule trees follow.
//
// Three habitats, by depth and by a low-frequency mask that makes them patches rather than a
// uniform carpet: a reef on the shelf (bright, dense, the thing you swim toward), kelp in the
// deeper water beyond it, and seagrass on the sand between.
void TerrainGenerator::decorateSea(Chunk& chunk, int baseX, int baseY, int baseZ, int x, int z,
                                   int h) const {
    const int wx = baseX + x, wz = baseZ + z;
    const int depth = kSeaLevel - h;
    const std::uint32_t hh = cellHash(wrapMul(wx, 3u) + std::uint32_t(m_seed), wrapMul(wz, 5u) - std::uint32_t(m_seed));
    const float reefM = valueNoise(float(wx) * 0.022f, float(wz) * 0.022f, seedOff(m_seed, 31));
    const float kelpM = valueNoise(float(wx) * 0.017f, float(wz) * 0.017f, seedOff(m_seed, 37));
    const float grassM = valueNoise(float(wx) * 0.031f, float(wz) * 0.031f, seedOff(m_seed, 41));

    // only ever replaces water, so a plant can never overwrite terrain or another plant
    const auto put = [&](int wy, BlockId id) {
        const int ly = wy - baseY;
        if (ly >= 0 && ly < kChunkSize && chunk.get(x, ly, z) == blocks::Water)
            chunk.set(x, ly, z, id);
    };

    // Version 3 reefs start a few blocks out from the beach, leave a third of their colonies as
    // sand runs and stay two blocks under the surface: before, they carpeted the shallows up to
    // the sand and to the waves, lit like land and a maze to swim through.
    const bool v3 = m_version >= 3;
    if (depth >= (v3 ? 5 : 3) && depth <= 20 && reefM > 0.58f) {
        const float strength = std::min((reefM - 0.58f) / 0.30f, 1.0f); // 0 at the rim, 1 at the heart
        // colonies are 4-block cells of one species, so a reef is patchwork, not confetti
        const std::uint32_t cell = cellHash(std::uint32_t(wx / 4) + std::uint32_t(m_seed), std::uint32_t(wz / 4) + 17u);
        const int colony = int(cell % 5u);
        const bool sandRun = v3 && (cell >> 8) % 3u == 0u;
        const float cover = v3 ? 30.0f + 80.0f * strength : 40.0f + 110.0f * strength;
        if (!sandRun && float(hh & 255u) < cover) {
            // never up through the surface: a coral head poking out of the sea reads as a
            // stray coloured tile floating on the waves
            const int height = std::min(1 + int((hh >> 8) % (1u + std::uint32_t(strength * 2.0f + 1.0f))),
                                        depth - (v3 ? 4 : 2));
            const BlockId coral = BlockId(blocks::BrainCoral + colony);
            for (int k = 1; k <= height; ++k)
                put(h + k, coral);
            // the odd glowing head, deep enough that it reads against the water
            if (depth > 6 && ((hh >> 16) & 31u) == 0u)
                put(h + height, blocks::GlowCoral);
            // fans sprout off the top of a colony
            if (((hh >> 20) & 3u) == 0u)
                put(h + height + 1, BlockId(blocks::CoralFanPink + colony % 3));
        } else if (((hh >> 12) & 3u) != 0u) {
            put(h + 1, blocks::Seagrass);
        }
    } else if (depth >= 6 && depth <= 32 && kelpM > 0.55f && (hh & 255u) < 46u) {
        const int maxH = std::min(depth - 2, 15);
        const int height = 3 + int((hh >> 8) % std::uint32_t(std::max(maxH - 2, 1)));
        for (int k = 1; k <= std::min(height, maxH); ++k)
            put(h + k, blocks::Kelp);
    } else if (depth <= 14 && grassM > 0.40f && (hh & 255u) < 128u) {
        put(h + 1, blocks::Seagrass);
    }
}

// Deferred structure placement: scans every tree candidate cell that could reach into
// this chunk (including neighbors, hence kTreeCanopyR of overlap) and places the parts
// that fall inside it. Fully deterministic from the seed, the same tree is computed
// identically by every chunk it touches, so trunks/canopies cross borders seamlessly
// without any neighbor lookups.
// Grass tufts and flowers on top of grass. Per-column and purely positional, so like the
// trees it needs no neighbour lookups, but unlike them it never reaches past its own
// column, so there is no overlap scan and no canopy-radius bound to get wrong.
//
// Takes the chunk's own height grid instead of asking the noise again per column: it used
// to call surfaceHeight for all 1024 columns of every chunk it ran on, which was most of the
// cost of an all-air chunk.
// Version 3: trees, fallen logs, stumps and boulders by biome, with clearings in the woods.
void TerrainGenerator::placeTrees3(Chunk& chunk, ChunkPos pos) const {
    const int baseX = pos.x * kChunkSize;
    const int baseY = pos.y * kChunkSize;
    const int baseZ = pos.z * kChunkSize;
    // force: a boulder's buried half takes the ground's place
    const auto put = [&](int wx, int wy, int wz, BlockId id, bool force) {
        const int lx = wx - baseX, ly = wy - baseY, lz = wz - baseZ;
        if (lx < 0 || lx >= kChunkSize || ly < 0 || ly >= kChunkSize || lz < 0 || lz >= kChunkSize)
            return;
        const BlockId cur = chunk.get(lx, ly, lz);
        if (cur == blocks::Air || (blocks::isCross(cur) && !blocks::isWaterlogged(cur)) ||
            (force && !blocks::isWater(cur) && !blocks::isLogBlock(cur) && !blocks::isLeafBlock(cur)))
            chunk.set(lx, ly, lz, id);
    };
    const int cx0 = floorDivI(baseX - kTreeCanopyR, kTreeCell);
    const int cx1 = floorDivI(baseX + kChunkSize - 1 + kTreeCanopyR, kTreeCell);
    const int cz0 = floorDivI(baseZ - kTreeCanopyR, kTreeCell);
    const int cz1 = floorDivI(baseZ + kChunkSize - 1 + kTreeCanopyR, kTreeCell);

    for (int cz = cz0; cz <= cz1; ++cz)
        for (int cx = cx0; cx <= cx1; ++cx) {
            const std::uint32_t hsh = cellHash(wrapMul(cx, 1013u) + std::uint32_t(m_seed), wrapMul(cz, 79u) + std::uint32_t(m_seed));
            const int tx = cx * kTreeCell + int((hsh >> 8) % kTreeCell);
            const int tz = cz * kTreeCell + int((hsh >> 13) % kTreeCell);
            const Column col = column(tx, tz);
            const int surf = col.h;
            // mangroves stand in the shallows too
            const bool mangrove = col.biome == Biome::Mangrove && col.water - col.h <= 3 && surf >= kSeaLevel - 1;
            if ((col.water > col.h || surf < kSeaLevel + 1) && !mangrove)
                continue;
            const std::uint32_t rv = voxelHash(tx, 7, tz);
            if (col.biome == Biome::Oasis) {
                if ((hsh & 255u) < 150u)
                    placePalm(chunk, pos, tx, tz, surf, rv);
                continue;
            }
            if (surf <= kBeachTop + 1 && col.coast < 45.0f && col.biome == Biome::Beach) {
                // palms only on warm shores, not beside the spruce woods
                if (col.temp > 0.45f && ((hsh >> 18) & 255u) <= 96u && (hsh & 255u) < 120u)
                    placePalm(chunk, pos, tx, tz, surf, rv);
                continue;
            }
            const Biome b = col.biome;
            const std::uint32_t extra = voxelHash(tx, 13, tz) % 1000u;
            // boulders sit on rock and soil alike
            int boulderRate = 0;
            switch (b) {
                case Biome::Alpine: boulderRate = 45; break;
                case Biome::Tundra: boulderRate = 28; break;
                case Biome::Taiga:
                case Biome::SnowyTaiga: boulderRate = 18; break;
                case Biome::Savanna: boulderRate = 10; break;
                case Biome::Forest:
                case Biome::Plains: boulderRate = 7; break;
                case Biome::Meadow:
                case Biome::BirchForest: boulderRate = 5; break;
                case Biome::Jungle:
                case Biome::Swamp: boulderRate = 2; break;
                case Biome::Redwood: boulderRate = 22; break;
                case Biome::AspenGrove: boulderRate = 9; break;
                case Biome::AutumnForest:
                case Biome::DarkForest: boulderRate = 7; break;
                case Biome::SpringForest:
                case Biome::CherryGrove: boulderRate = 5; break;
                case Biome::FlowerField:
                case Biome::Bamboo: boulderRate = 2; break;
                default: break;
            }
            const auto boulder = [&] {
                const float rad = 1.2f + float(rv % 100u) * 0.014f;
                const bool mossy = b == Biome::Taiga || b == Biome::Forest || b == Biome::Swamp || b == Biome::Jungle ||
                                   b == Biome::BirchForest || b == Biome::Redwood || b == Biome::DarkForest ||
                                   b == Biome::AutumnForest || b == Biome::SpringForest || b == Biome::AspenGrove;
                const int R = int(std::ceil(rad));
                for (int dy = -R; dy <= R; ++dy)
                    for (int dz = -R; dz <= R; ++dz)
                        for (int dx = -R; dx <= R; ++dx) {
                            const std::uint32_t vh = voxelHash(tx + dx, surf + dy, tz + dz);
                            const float d2 = float(dx * dx + dz * dz) + float(dy * dy) * 1.6f;
                            if (d2 > rad * rad + float(vh & 7u) * 0.12f - 0.4f)
                                continue;
                            BlockId id = (vh >> 4) % 5u == 0u ? blocks::Cobblestone
                                                              : (b == Biome::Alpine || b == Biome::Tundra) && (vh >> 7) % 3u == 0u
                                                                    ? blocks::Andesite
                                                                    : blocks::Stone;
                            if (mossy && (dy >= 0 || (vh >> 9) % 3u == 0u) && (vh >> 11) % 5u < 3u)
                                id = blocks::MossyCobble;
                            put(tx + dx, surf + dy, tz + dz, id, dy <= 0);
                        }
            };
            if (col.rock) {
                if (int(extra) < boulderRate)
                    boulder();
                continue;
            }
            if (col.volcanic || col.h > col.snowLine - 34)
                continue; // nothing roots in fresh ash, or up in the snow

            if (col.h > col.snowLine - 74 && int((hsh >> 21) % 40u) < col.h - (col.snowLine - 74))
                continue;

            float density = 0.0f;
            int kindA = kTreeOak, kindB = kTreeOak;
            float shareB = 0.0f;
            bool woods = false;
            switch (b) {
                case Biome::Forest: density = 0.85f, kindA = kTreeOak, kindB = kTreeBirch, shareB = 0.22f, woods = true; break;
                case Biome::BirchForest: density = 0.8f, kindA = kTreeBirch, kindB = kTreeOak, shareB = 0.15f, woods = true; break;
                case Biome::Taiga: density = 0.78f, kindA = kTreeSpruce, kindB = kTreeTallSpruce, shareB = 0.3f, woods = true; break;
                case Biome::SnowyTaiga: density = 0.45f, kindA = kTreeSpruce, kindB = kTreeSmallSpruce, shareB = 0.4f, woods = true; break;
                case Biome::Tundra: density = 0.03f, kindA = kTreeSmallSpruce; break;
                case Biome::Jungle: density = 0.8f, kindA = kTreeJungle, kindB = kTreeBigJungle, shareB = 0.16f, woods = true; break;
                case Biome::Savanna: density = 0.11f, kindA = kTreeAcacia; break;
                case Biome::Swamp: density = 0.45f, kindA = kTreeSwampOak, woods = true; break;
                case Biome::Plains: density = 0.10f, kindA = kTreeOak, kindB = kTreeBigOak, shareB = 0.25f; break;
                case Biome::Meadow: density = 0.04f, kindA = kTreeOak, kindB = kTreeBirch, shareB = 0.5f; break;
                case Biome::Alpine: density = 0.08f, kindA = kTreeSmallSpruce, kindB = kTreeSpruce, shareB = 0.3f; break;
                case Biome::CherryGrove: density = 0.5f, kindA = kTreeCherry; break;
                case Biome::AutumnForest: density = 0.85f, kindA = kTreeOak, kindB = kTreeBirch, shareB = 0.25f, woods = true; break;
                case Biome::SpringForest: density = 0.75f, kindA = kTreeOak, kindB = kTreeBirch, shareB = 0.3f, woods = true; break;
                case Biome::FlowerField: density = 0.03f, kindA = kTreeOak, kindB = kTreeBirch, shareB = 0.4f; break;
                case Biome::AspenGrove: density = 0.8f, kindA = kTreeAspen, kindB = kTreeSpruce, shareB = 0.12f, woods = true; break;
                case Biome::Redwood: density = 0.75f, kindA = kTreeRedwood, kindB = kTreeTallSpruce, shareB = 0.4f, woods = true; break;
                case Biome::DarkForest: density = 0.95f, kindA = kTreeDarkOak, kindB = kTreeBigOak, shareB = 0.2f, woods = true; break;
                case Biome::Bamboo: density = 0.8f, kindA = kTreeBamboo, kindB = kTreeJungle, shareB = 0.2f, woods = true; break;
                case Biome::Mangrove: density = 0.55f, kindA = kTreeMangrove, woods = true; break;
                default: break;
            }
            if (kindA == kTreeOak && (b == Biome::Forest || b == Biome::AutumnForest) && (rv >> 20) % 100u < 16u)
                kindA = kTreeBigOak; // the odd old giant among them
            if (density > 0.0f) {
                // where two biomes meet the woods thin out over a few dozen blocks, not at a line
                static constexpr int kAround[4][2] = {{11, 4}, {-4, 11}, {-11, -4}, {4, -11}};
                float sum = density * 2.0f;
                for (const auto& o : kAround)
                    sum += treeDensity(column(tx + o[0], tz + o[1]).biome);
                density = sum / 6.0f;
            }
            const float forestM = regionNoise(float(cx * kTreeCell) * 0.0045f, float(cz * kTreeCell) * 0.0045f, seedOff(m_seed, 71));
            density *= 0.55f + 0.9f * smooth01(0.30f, 0.70f, forestM);
            // clearings in the woods, a few dozen blocks across
            if (woods && valueNoise(float(tx) * 0.018f, float(tz) * 0.018f, seedOff(m_seed, 321)) > 0.76f)
                density *= 0.06f;
            const float roll = float(hsh & 1023u) / 1024.0f;
            if (roll < density * 0.62f) {
                const int kind = float((rv >> 24) & 255u) / 256.0f < shareB ? kindB : kindA;
                // the colourful woods: the same trees in other leaves
                BlockId leaves = blocks::Air;
                const std::uint32_t pick = (rv >> 27) % 10u;
                if (b == Biome::AutumnForest)
                    leaves = kind == kTreeBirch ? blocks::AutumnYellow
                             : pick < 4u        ? blocks::AutumnOrange
                             : pick < 7u        ? blocks::AutumnRed
                             : pick < 9u        ? blocks::AutumnYellow
                                                : blocks::Air;
                else if (b == Biome::SpringForest && kind != kTreeBirch && pick < 6u)
                    leaves = blocks::BlossomLeaves;
                else if (b == Biome::AspenGrove && kind == kTreeAspen && pick < 9u)
                    leaves = blocks::AutumnYellow;
                growTree(chunk, pos, kind, tx, tz, surf, rv, leaves);
                continue;
            }
            // giant mushrooms in the dark woods, where the trees leave room
            if (b == Biome::DarkForest && extra < 150u) {
                growTree(chunk, pos, (rv >> 4) & 1u ? kTreeMushroomRed : kTreeMushroomBrown, tx, tz, surf, rv);
                continue;
            }
            if (mangrove)
                continue;
            if (woods && b != Biome::Swamp && b != Biome::Mangrove && b != Biome::Bamboo && extra < 32u) {
                // a fallen trunk, where the ground is level enough to lie on
                const bool alongX = (rv >> 3) & 1u;
                const int len = 3 + int((rv >> 5) % 4u);
                int lo = surf, hi = surf;
                bool ok = true;
                for (int i = 1; i < len && ok; ++i) {
                    const Column c = column(tx + (alongX ? i : 0), tz + (alongX ? 0 : i));
                    lo = std::min(lo, c.h), hi = std::max(hi, c.h);
                    ok = c.water <= c.h && !c.rock;
                }
                if (!ok || hi - lo > 1)
                    continue;
                const int sp = b == Biome::Taiga || b == Biome::SnowyTaiga || b == Biome::Redwood ? 1
                               : b == Biome::BirchForest || b == Biome::AspenGrove                 ? 2
                                                                                                   : 0;
                const BlockId log = BlockId(blocks::WoodX + sp * 2 + (alongX ? 0 : 1));
                for (int i = 0; i < len; ++i)
                    put(tx + (alongX ? i : 0), hi + 1, tz + (alongX ? 0 : i), log, false);
                if ((rv >> 9) % 3u == 0u)
                    put(tx + (alongX ? len / 2 : 0), hi + 2, tz + (alongX ? 0 : len / 2),
                        (rv >> 11) & 1u ? blocks::MushroomBrown : blocks::Moss, false);
                continue;
            }
            if (woods && extra >= 32u && extra < 44u) {
                const BlockId log = b == Biome::Taiga || b == Biome::SnowyTaiga  ? blocks::SpruceWood
                                    : b == Biome::Redwood                        ? blocks::RedwoodWood
                                    : b == Biome::BirchForest || b == Biome::AspenGrove ? blocks::BirchWood
                                    : b == Biome::Jungle || b == Biome::Bamboo   ? blocks::JungleWood
                                                                                 : blocks::Wood;
                put(tx, surf + 1, tz, log, false);
                if ((rv >> 13) & 1u)
                    put(tx, surf + 2, tz, log, false);
                continue;
            }
            if (int(extra) >= 44 && int(extra) < 44 + boulderRate)
                boulder();
        }
}

// Version 3's ground cover, by biome: grasses, ferns, flowers in drifts of one colour, mushrooms,
// leaf litter, pebbles and bushes, reeds along the banks and lily pads out on still water.
void TerrainGenerator::placeFoliage3(Chunk& chunk, ChunkPos pos) const {
    const auto tile = columnTile(pos.x, pos.z);
    const int baseX = pos.x * kChunkSize;
    const int baseY = pos.y * kChunkSize;
    const int baseZ = pos.z * kChunkSize;
    const auto at = [&](int x, int y, int z) {
        return y >= 0 && y < kChunkSize ? chunk.get(x, y, z) : blocks::Stone;
    };
    const auto set = [&](int x, int y, int z, BlockId id) {
        if (y >= 0 && y < kChunkSize && chunk.get(x, y, z) == blocks::Air)
            chunk.set(x, y, z, id);
    };
    for (int z = 0; z < kChunkSize; ++z)
        for (int x = 0; x < kChunkSize; ++x) {
            const int wx = baseX + x, wz = baseZ + z;
            const int surf = tile->h[z][x];
            const int water = tile->water[z][x];
            const Biome bio = tile->biome[z][x];
            const std::uint32_t h = cellHash(wrapMul(wx, 7u) + std::uint32_t(m_seed), wrapMul(wz, 13u));
            const int roll = int(h % 1000u);
            const int roll2 = int((h >> 10) % 1000u);

            if (water > surf) {
                // lily pads on still water a few blocks deep
                if (bio == Biome::Tundra || bio == Biome::SnowyTaiga || water - surf > 6)
                    continue;
                const int rate = bio == Biome::Swamp || bio == Biome::Mangrove ? 150
                                 : bio == Biome::Oasis                         ? 60
                                 : bio == Biome::Taiga || bio == Biome::Savanna ? 8
                                                                                : 26;
                if (roll < rate)
                    set(x, water + 1 - baseY, z, blocks::LilyPad);
                continue;
            }
            if (surf >= tile->snow[z][x] || surf < kSeaLevel)
                continue;
            const int y = surf + 1 - baseY;
            if (y < 0 || y >= kChunkSize || chunk.get(x, y, z) != blocks::Air)
                continue;
            const BlockId ground = y > 0 ? chunk.get(x, y - 1, z) : blocks::Grass;
            if (y > 0 && (ground == blocks::Air || blocks::isWater(ground) || ground == blocks::Snow || ground == blocks::Ice))
                continue;
            if (tile->volcanic[z][x] > 100)
                continue; // nothing grows on fresh ash
            if (tile->rock[z][x] != 0) {
                if ((bio == Biome::Alpine || bio == Biome::Tundra || bio == Biome::Taiga) && roll < 30)
                    set(x, y, z, blocks::Pebbles);
                continue;
            }
            if (bio == Biome::Beach) {
                if (surf >= kSeaLevel && roll < 30)
                    set(x, y, z, blocks::Shell);
                else if (surf >= kSeaLevel + 2 && roll >= 860)
                    set(x, y, z, blocks::DuneGrass);
                continue;
            }

            // a bank: river or lake water right beside, at about this height
            bool bank = false;
            for (const auto& d : {std::pair<int, int>{1, 0}, {-1, 0}, {0, 1}, {0, -1}}) {
                const int nx = x + d.first, nz = z + d.second;
                if (nx < 0 || nx >= kChunkSize || nz < 0 || nz >= kChunkSize)
                    continue;
                const int w = tile->water[nz][nx];
                bank = bank || (w > tile->h[nz][nx] && w >= surf - 1 && w <= surf + 1);
            }
            const bool cold = bio == Biome::Tundra || bio == Biome::SnowyTaiga || bio == Biome::Alpine;
            if (bank && !cold) {
                const int reeds = bio == Biome::Swamp || bio == Biome::Mangrove ? 380
                                  : bio == Biome::Jungle || bio == Biome::Oasis ? 260
                                  : bio == Biome::Taiga                         ? 70
                                                                                : 200;
                if (roll < reeds) {
                    const int tall = 1 + int((h >> 20) % 3u);
                    for (int k = 0; k < tall; ++k)
                        set(x, y + k, z, blocks::Reed);
                    continue;
                }
                if (roll < reeds + 45 && bio != Biome::Savanna && bio != Biome::Taiga) {
                    set(x, y, z, blocks::FireflyBush);
                    continue;
                }
            }

            const float fwx = float(wx), fwz = float(wz);
            // flowers grow in drifts of one colour
            const float drift = valueNoise(fwx * 0.07f, fwz * 0.07f, seedOff(m_seed, 331));
            const float hue = valueNoise(fwx * 0.03f, fwz * 0.03f, seedOff(m_seed, 333));
            const auto flowerFrom = [&](std::initializer_list<BlockId> set) {
                const int n = int(set.size());
                return *(set.begin() + std::min(n - 1, int(hue * float(n))));
            };
            const auto flowerRoll = [&](int perMille) { return roll2 < (drift > 0.62f ? perMille * 5 : perMille * 2 / 5); };
            const auto grass = [&](int tallShare) {
                set(x, y, z, int((h >> 16) % 100u) < tallShare ? blocks::TallGrass : blocks::GrassTuft);
            };
            const auto bush = [&](BlockId leaf) {
                set(x, y, z, leaf);
                if ((h >> 22) & 1u)
                    set(x, y + 1, z, leaf);
            };
            switch (bio) {
                case Biome::Plains:
                    if (flowerRoll(18))
                        set(x, y, z, flowerFrom({blocks::Flower, blocks::FlowerWhite, blocks::FlowerRed, blocks::FlowerOrange}));
                    else if (roll < 330)
                        grass(30);
                    else if (roll < 333)
                        set(x, y, z, blocks::Pebbles);
                    else if (roll < 335)
                        bush(blocks::Leaves);
                    break;
                case Biome::Meadow:
                    if (flowerRoll(55))
                        set(x, y, z, flowerFrom({blocks::FlowerPurple, blocks::Flower, blocks::FlowerWhite, blocks::FlowerBlue,
                                                 blocks::FlowerRed}));
                    else if (roll < 450)
                        grass(50);
                    break;
                case Biome::Forest:
                case Biome::BirchForest: {
                    const bool birch = bio == Biome::BirchForest;
                    if (flowerRoll(birch ? 14 : 5))
                        set(x, y, z, flowerFrom({blocks::FlowerWhite, blocks::FlowerBlue, blocks::Flower}));
                    else if (roll < (birch ? 240 : 170))
                        grass(25);
                    else if (roll < (birch ? 270 : 230))
                        set(x, y, z, blocks::Fern);
                    else if (roll < (birch ? 340 : 370))
                        set(x, y, z, blocks::LeafLitter);
                    else if (roll < (birch ? 344 : 378))
                        set(x, y, z, (h >> 18) % 4u == 0u ? blocks::MushroomRed : blocks::MushroomBrown);
                    else if (roll < (birch ? 350 : 390))
                        bush(birch ? blocks::BirchLeaves : blocks::Leaves);
                    break;
                }
                case Biome::Taiga:
                    if (roll < 160)
                        set(x, y, z, blocks::Fern);
                    else if (roll < 240)
                        grass(20);
                    else if (roll < 252)
                        set(x, y, z, (h >> 18) % 3u == 0u ? blocks::MushroomRed : blocks::MushroomBrown);
                    else if (roll < 262)
                        bush(blocks::BerryBush);
                    else if (roll < 270)
                        set(x, y, z, blocks::Pebbles);
                    else if (roll < 300)
                        set(x, y, z, blocks::LeafLitter);
                    break;
                case Biome::SnowyTaiga:
                    if (roll < 30)
                        set(x, y, z, blocks::Fern);
                    else if (roll < 34)
                        bush(blocks::BerryBush);
                    else if (roll < 40)
                        set(x, y, z, blocks::Pebbles);
                    break;
                case Biome::Tundra:
                    if (roll < 40)
                        set(x, y, z, blocks::DryGrass);
                    else if (roll < 66)
                        set(x, y, z, blocks::Pebbles);
                    else if (roll < 71)
                        set(x, y, z, blocks::Fern);
                    break;
                case Biome::Jungle:
                    if (flowerRoll(10))
                        set(x, y, z, flowerFrom({blocks::FlowerRed, blocks::FlowerOrange}));
                    else if (roll < 100)
                        set(x, y, z, blocks::JungleGrass);
                    else if (roll < 260)
                        set(x, y, z, blocks::Fern);
                    else if (roll < 420)
                        grass(60);
                    else if (roll < 450)
                        bush(blocks::JungleLeaves);
                    else if (roll < 453)
                        set(x, y, z, blocks::MushroomBrown);
                    break;
                case Biome::Savanna:
                    if (flowerRoll(3))
                        set(x, y, z, blocks::FlowerOrange);
                    else if (roll < 220)
                        set(x, y, z, blocks::DryGrass);
                    else if (roll < 320)
                        grass(50);
                    else if (roll < 326)
                        set(x, y, z, blocks::DeadBush);
                    else if (roll < 330)
                        set(x, y, z, blocks::Pebbles);
                    break;
                case Biome::Swamp:
                    if (ground == blocks::Mud) {
                        if (roll < 60)
                            set(x, y, z, blocks::TallGrass);
                        break;
                    }
                    if (roll < 180)
                        set(x, y, z, blocks::TallGrass);
                    else if (roll < 280)
                        set(x, y, z, blocks::GrassTuft);
                    else if (roll < 340)
                        set(x, y, z, blocks::Fern);
                    else if (roll < 360)
                        set(x, y, z, (h >> 18) % 3u == 0u ? blocks::MushroomRed : blocks::MushroomBrown);
                    else if (roll < 375)
                        set(x, y, z, blocks::FireflyBush);
                    break;
                case Biome::Alpine:
                    if (flowerRoll(32))
                        set(x, y, z, flowerFrom({blocks::FlowerBlue, blocks::FlowerPurple, blocks::FlowerWhite}));
                    else if (roll < 250)
                        set(x, y, z, blocks::GrassTuft);
                    else if (roll < 290)
                        set(x, y, z, blocks::Pebbles);
                    break;
                case Biome::Desert:
                case Biome::RedDesert: {
                    // cacti stand in loose groups
                    const float group = valueNoise(fwx * 0.05f, fwz * 0.05f, seedOff(m_seed, 335));
                    const int cactus = group > 0.62f ? 30 : 5;
                    if (roll < cactus) {
                        const int tall = 1 + int((h >> 10) % 3u);
                        int k = 0;
                        for (; k < tall && y + k < kChunkSize && at(x, y + k, z) == blocks::Air; ++k)
                            chunk.set(x, y + k, z, blocks::Cactus);
                        if (k == tall && (h >> 14) % 10u < 3u)
                            set(x, y + k, z, blocks::CactusFlower);
                    } else if (roll < cactus + 28) {
                        set(x, y, z, blocks::DeadBush);
                    } else if (roll < cactus + 34) {
                        set(x, y, z, blocks::DryGrass);
                    } else if (roll < cactus + 37) {
                        set(x, y, z, blocks::Pebbles);
                    }
                    break;
                }
                case Biome::Badlands:
                    if (roll < 22)
                        set(x, y, z, blocks::DeadBush);
                    else if (roll < 32)
                        set(x, y, z, blocks::DryGrass);
                    break;
                case Biome::CherryGrove:
                    // fallen petals lie thickest in drifts under the trees
                    if (roll < (drift > 0.45f ? 330 : 90))
                        set(x, y, z, blocks::PinkPetals);
                    else if (flowerRoll(16))
                        set(x, y, z, flowerFrom({blocks::FlowerPink, blocks::FlowerWhite, blocks::FlowerPurple}));
                    else if (roll < 520)
                        grass(25);
                    break;
                case Biome::AutumnForest:
                    if (roll < 270)
                        set(x, y, z, blocks::LeafLitter);
                    else if (roll < 390)
                        grass(20);
                    else if (roll < 430)
                        set(x, y, z, blocks::Fern);
                    else if (roll < 440)
                        set(x, y, z, (h >> 18) % 3u == 0u ? blocks::MushroomRed : blocks::MushroomBrown);
                    else if (roll < 452)
                        bush((h >> 20) % 3u == 0u ? blocks::AutumnRed : blocks::AutumnOrange);
                    break;
                case Biome::SpringForest:
                    if (flowerRoll(36))
                        set(x, y, z, flowerFrom({blocks::FlowerWhite, blocks::FlowerPink, blocks::Flower, blocks::FlowerBlue}));
                    else if (roll < 300)
                        grass(30);
                    else if (roll < 335)
                        set(x, y, z, blocks::Fern);
                    else if (roll < 350)
                        bush(blocks::BlossomLeaves);
                    break;
                case Biome::FlowerField: {
                    // whole slopes of one flower, lavender or poppies, and the mixed meadow between them
                    const float field = valueNoise(fwx * 0.011f, fwz * 0.011f, seedOff(m_seed, 337));
                    if (field < 0.36f) {
                        if (roll < 680)
                            set(x, y, z, blocks::Lavender);
                        else if (roll < 780)
                            grass(30);
                    } else if (field > 0.64f) {
                        if (roll < 540)
                            set(x, y, z, blocks::FlowerRed);
                        else if (roll < 575)
                            set(x, y, z, blocks::FlowerWhite);
                        else if (roll < 720)
                            grass(40);
                    } else if (roll < 540) {
                        constexpr BlockId kMix[8] = {blocks::Flower,     blocks::FlowerWhite, blocks::FlowerRed,  blocks::FlowerOrange,
                                                     blocks::FlowerBlue, blocks::FlowerPurple, blocks::Lavender, blocks::FlowerPink};
                        set(x, y, z, kMix[(h >> 12) % 8u]);
                    } else if (roll < 740) {
                        grass(45);
                    }
                    break;
                }
                case Biome::AspenGrove:
                    if (roll < 220)
                        grass(35);
                    else if (roll < 300)
                        set(x, y, z, blocks::Fern);
                    else if (roll < 370)
                        set(x, y, z, blocks::LeafLitter);
                    else if (flowerRoll(8))
                        set(x, y, z, flowerFrom({blocks::FlowerWhite, blocks::FlowerPurple}));
                    else if (roll < 382)
                        bush(blocks::BerryBush);
                    break;
                case Biome::Redwood:
                    if (roll < 270)
                        set(x, y, z, blocks::Fern);
                    else if (roll < 340)
                        grass(40);
                    else if (roll < 356)
                        set(x, y, z, (h >> 18) % 3u == 0u ? blocks::MushroomRed : blocks::MushroomBrown);
                    else if (roll < 370)
                        bush(blocks::BerryBush);
                    else if (roll < 410)
                        set(x, y, z, blocks::LeafLitter);
                    break;
                case Biome::DarkForest:
                    if (roll < 90)
                        set(x, y, z, (h >> 18) % 3u == 0u ? blocks::MushroomRed : blocks::MushroomBrown);
                    else if (roll < 210)
                        set(x, y, z, blocks::Fern);
                    else if (roll < 320)
                        set(x, y, z, blocks::LeafLitter);
                    else if (roll < 400)
                        grass(15);
                    else if (roll < 410)
                        set(x, y, z, blocks::FireflyBush);
                    break;
                case Biome::Bamboo:
                    if (roll < 150)
                        set(x, y, z, blocks::JungleGrass);
                    else if (roll < 330)
                        grass(55);
                    else if (roll < 420)
                        set(x, y, z, blocks::Fern);
                    else if (roll < 426)
                        bush(blocks::JungleLeaves);
                    break;
                case Biome::Mangrove:
                    if (ground == blocks::Mud) {
                        if (roll < 50)
                            set(x, y, z, blocks::TallGrass);
                        else if (roll < 70)
                            set(x, y, z, blocks::Fern);
                        break;
                    }
                    if (roll < 200)
                        set(x, y, z, blocks::TallGrass);
                    else if (roll < 300)
                        set(x, y, z, blocks::Fern);
                    else if (roll < 320)
                        set(x, y, z, blocks::FireflyBush);
                    break;
                case Biome::Oasis:
                    if (roll < 260)
                        grass(50);
                    else if (roll < 330)
                        set(x, y, z, blocks::Fern);
                    else if (roll < 370)
                        set(x, y, z, blocks::JungleGrass);
                    else if (flowerRoll(12))
                        set(x, y, z, flowerFrom({blocks::FlowerOrange, blocks::FlowerRed}));
                    break;
                default:
                    break;
            }
        }

    // One feature a chunk at most, kept inside it: a fallen log in the woods with mushrooms on it,
    // or a spike of ice standing out on the tundra.
    const std::uint32_t fh = cellHash(wrapMul(pos.x, 31u) + std::uint32_t(seedOff(m_seed, 261)), wrapMul(pos.z, 57u));
    const int fx = 4 + int((fh >> 4) % 24u), fz = 4 + int((fh >> 9) % 24u);
    const Biome fb = tile->biome[fz][fx];
    const int fsurf = tile->h[fz][fx];
    if (tile->rock[fz][fx] != 0 || tile->water[fz][fx] > fsurf || fsurf < kSeaLevel || fsurf >= tile->snow[fz][fx])
        return;
    const bool woods = fb == Biome::Forest || fb == Biome::BirchForest || fb == Biome::Taiga || fb == Biome::Jungle ||
                       fb == Biome::AutumnForest || fb == Biome::SpringForest || fb == Biome::DarkForest ||
                       fb == Biome::AspenGrove || fb == Biome::Redwood;
    if (woods && fh % 100u < 30u) {
        const bool alongX = ((fh >> 14) & 1u) != 0u;
        const int len = 3 + int((fh >> 15) % 3u);
        const BlockId log = fb == Biome::BirchForest || fb == Biome::AspenGrove ? (alongX ? blocks::BirchWoodX : blocks::BirchWoodZ)
                            : fb == Biome::Taiga || fb == Biome::Redwood        ? (alongX ? blocks::SpruceWoodX : blocks::SpruceWoodZ)
                                                                                : (alongX ? blocks::WoodX : blocks::WoodZ);
        for (int k = 0; k < len; ++k) {
            const int lx = alongX ? fx + k - len / 2 : fx, lz = alongX ? fz : fz + k - len / 2;
            const int gy = tile->h[lz][lx];
            if (std::abs(gy - fsurf) > 1)
                break;
            const int ly = gy + 1 - baseY;
            if (ly < 0 || ly >= kChunkSize)
                continue;
            // the plants on the ground give way to it, a tree does not
            const BlockId cur = chunk.get(lx, ly, lz);
            if (cur != blocks::Air && !blocks::isCross(cur))
                break;
            chunk.set(lx, ly, lz, log);
            if (ly + 1 < kChunkSize && chunk.get(lx, ly + 1, lz) == blocks::Air && voxelHash(baseX + lx, gy, baseZ + lz) % 4u == 0u)
                chunk.set(lx, ly + 1, lz, blocks::MushroomBrown);
        }
    } else if (fb == Biome::Tundra && fh % 100u < 9u) {
        const int tall = 7 + int((fh >> 15) % 14u);
        for (int k = 0; k < tall; ++k) {
            const int r = k < tall / 3 ? 2 : (k < tall * 2 / 3 ? 1 : 0);
            const int ly = fsurf + 1 + k - baseY;
            if (ly < 0 || ly >= kChunkSize)
                continue;
            for (int dz = -r; dz <= r; ++dz)
                for (int dx = -r; dx <= r; ++dx) {
                    if (r == 2 && std::abs(dx) == 2 && std::abs(dz) == 2)
                        continue;
                    const BlockId cur = chunk.get(fx + dx, ly, fz + dz);
                    if (cur == blocks::Air || blocks::isCross(cur) || cur == blocks::Snow)
                        chunk.set(fx + dx, ly, fz + dz, blocks::Ice);
                }
        }
    }
}

void TerrainGenerator::placeFoliage(Chunk& chunk, ChunkPos pos, const int (*heights)[kChunkSize],
                                    const float (*desert)[kChunkSize], const float (*coast)[kChunkSize],
                                    const std::uint8_t (*rock)[kChunkSize]) const {
    if (m_version >= 3) {
        placeFoliage3(chunk, pos);
        return;
    }
    const int baseX = pos.x * kChunkSize;
    const int baseY = pos.y * kChunkSize;
    const int baseZ = pos.z * kChunkSize;
    for (int z = 0; z < kChunkSize; ++z)
        for (int x = 0; x < kChunkSize; ++x) {
            const int wx = baseX + x, wz = baseZ + z;
            const int surf = heights[z][x];
            if (surf >= snowLineAt(wx, wz) || rock[z][x] != 0)
                continue; // up in the snow, or bare rock
            const int y = surf + 1 - baseY;
            if (y < 0 || y >= kChunkSize)
                continue;
            if (chunk.get(x, y, z) != blocks::Air)
                continue; // a tree trunk or a cave mouth already owns this cell
            if (y > 0 && chunk.get(x, y - 1, z) == blocks::Air)
                continue; // a cave broke out of the ground here: nothing to stand on
            const std::uint32_t h = cellHash(wrapMul(wx, 7u) + std::uint32_t(m_seed), wrapMul(wz, 13u));
            const std::uint32_t roll = h % 100u;
            const bool isDesert = desert[z][x] > 0.42f + 0.16f * float(cellHash(wx, wz) & 255u) / 255.0f;

            if (isDesert) {
                // dry country: the odd cactus and dead bush, nothing green
                const std::uint32_t r = h % 1000u;
                if (r < 13u) {
                    const int height = 1 + int((h >> 10) % 3u);
                    for (int k = 0; k < height; ++k)
                        if (y + k < kChunkSize && chunk.get(x, y + k, z) == blocks::Air)
                            chunk.set(x, y + k, z, blocks::Cactus);
                } else if (r < 44u) {
                    chunk.set(x, y, z, blocks::DeadBush);
                }
                continue;
            }
            if (isBeach(surf, coast ? coast[z][x] : 0.0f)) {
                // beach: shells on the sand, and dune grass creeping down from the back of it
                if (surf >= kSeaLevel && roll < 3u)
                    chunk.set(x, y, z, blocks::Shell);
                else if (surf >= kSeaLevel + 2 && roll >= 90u)
                    chunk.set(x, y, z, blocks::GrassTuft);
                continue;
            }
            if (surf < kBeachTop + 1)
                continue; // underwater
            if (roll < 26u)
                chunk.set(x, y, z, blocks::GrassTuft);
            else if (roll < 30u)
                chunk.set(x, y, z, blocks::Flower);
        }
}

// A palm: a trunk that leans away from the sea wind and arcs, a crown of fronds that arch up
// and droop, and a few coconuts hanging under it. Like the other species it is a pure function
// of its world position, so every chunk it overlaps builds the same tree.
void TerrainGenerator::placePalm(Chunk& chunk, ChunkPos pos, int tx, int tz, int surf,
                                 std::uint32_t rv) const {
    const int baseX = pos.x * kChunkSize;
    const int baseY = pos.y * kChunkSize;
    const int baseZ = pos.z * kChunkSize;
    const auto put = [&](int wx, int wy, int wz, BlockId id, bool overLeaves) {
        const int lx = wx - baseX, ly = wy - baseY, lz = wz - baseZ;
        if (lx < 0 || lx >= kChunkSize || ly < 0 || ly >= kChunkSize || lz < 0 || lz >= kChunkSize)
            return;
        const BlockId cur = chunk.get(lx, ly, lz);
        if (cur == blocks::Air || (overLeaves && cur == blocks::PalmLeaves))
            chunk.set(lx, ly, lz, id);
    };

    const int height = 6 + int(rv % 5u);                  // 6..10
    const int lean = int((rv >> 4) % 3u);                 // 0..2 blocks over the whole trunk
    const float lang = float((rv >> 8) & 63u) * 0.09817f; // direction it leans
    const int lx = int(std::lround(std::cos(lang) * float(lean)));
    const int lz = int(std::lround(std::sin(lang) * float(lean)));

    int px = tx, pz = tz;
    for (int t = 1; t <= height; ++t) {
        // a smooth arc: nearly straight at the foot, bending most toward the top
        const float u = float(t) / float(height);
        const int ox = int(std::lround(float(lx) * u * u));
        const int oz = int(std::lround(float(lz) * u * u));
        const int nx = tx + ox, nz = tz + oz;
        if (nx != px || nz != pz)
            put(nx, surf + t - 1, nz, blocks::PalmWood, false); // keep the trunk face-connected
        put(nx, surf + t, nz, blocks::PalmWood, false);
        px = nx;
        pz = nz;
    }
    const int cy = surf + height + 1; // the crown sits on top of the trunk
    put(px, cy, pz, blocks::PalmLeaves, false);
    put(px, cy + 1, pz, blocks::PalmLeaves, false);

    // fronds: 7 of them, each an arch (rise, level, dip, droop) with a leaflet either side
    static constexpr int kArch[4] = {0, 1, 0, -1};
    const int fronds = 7;
    for (int f = 0; f < fronds; ++f) {
        const std::uint32_t fh = voxelHash(tx, 60 + f, tz);
        const float ang = float(f) * 6.2831853f / float(fronds) + float(rv & 15u) * 0.04f +
                          (float(fh & 15u) - 7.5f) * 0.03f;
        const float ex = std::cos(ang), ez = std::sin(ang);
        const float sx = -ez, sz = ex; // sideways, for the leaflets
        for (int i = 1; i <= 4; ++i) {
            const int bx = px + int(std::lround(ex * float(i)));
            const int bz = pz + int(std::lround(ez * float(i)));
            const int by = cy + kArch[i - 1] - (i == 4 && (fh & 32u) ? 1 : 0);
            put(bx, by, bz, blocks::PalmLeaves, false);
            if (i <= 3) { // a fuller frond near the crown
                put(bx + int(std::lround(sx)), by, bz + int(std::lround(sz)),
                    blocks::PalmLeaves, false);
                put(bx - int(std::lround(sx)), by - (i == 3 ? 1 : 0), bz - int(std::lround(sz)),
                    blocks::PalmLeaves, false);
            }
        }
    }
    // coconuts, tucked under the crown where the fronds leave the trunk
    for (int k = 0; k < 4; ++k)
        if (((rv >> (12 + k)) & 1u) != 0u) {
            static constexpr int kDx[4] = {1, -1, 0, 0}, kDz[4] = {0, 0, 1, -1};
            put(px + kDx[k], cy - 1, pz + kDz[k], blocks::Coconut, true);
        }
}

TerrainGenerator::StructureSite TerrainGenerator::structureSite(int cx, int cz) const {
    constexpr int kCell = 96;
    StructureSite none;
    const std::uint32_t hsh = cellHash(wrapMul(cx, 7919u) + std::uint32_t(seedOff(m_seed, 83)),
                                       wrapMul(cz, 104729u) + std::uint32_t(seedOff(m_seed, 89)));
    if ((hsh & 255u) > 104u)
        return none; // about 40% of cells have a candidate
    const int ax = cx * kCell + 20 + int((hsh >> 8) % std::uint32_t(kCell - 40));
    const int az = cz * kCell + 20 + int((hsh >> 16) % std::uint32_t(kCell - 40));
    const Column col = column(ax, az);
    const int g = col.h;
    const std::uint32_t rv = voxelHash(ax, 91, az);
    StructureSite site;
    site.x = ax;
    site.z = az;
    site.h = g;
    // shipwreck: shallow seabed, not by a desert, on a third of the candidates
    if (col.desert < 0.3f && g <= kSeaLevel - 3 && g >= kSeaLevel - 10) {
        site.kind = rv % 3u == 0u ? 1 : 0;
        return site;
    }
    // desert ruins, on a fifth of the candidates
    if (col.desert > 0.55f && g >= kSeaLevel + 6) {
        site.kind = (rv >> 2) % 5u == 0u ? 2 : 0;
        return site;
    }
    // lighthouse: a bit of low coast with open water not far off, on a quarter of the candidates
    if (col.desert < 0.3f && g >= kSeaLevel + 1 && g <= kSeaLevel + 8 && (rv >> 4) % 4u == 0u) {
        bool coast = false;
        for (const auto& d : {std::pair<int, int>{12, 0}, {-12, 0}, {0, 12}, {0, -12}})
            coast = coast || column(ax + d.first, az + d.second).h < kSeaLevel;
        site.kind = coast ? 3 : 0;
        return site;
    }
    // Inland, version 3: what people built out in the country and left.
    if (m_version >= 3 && g >= kSeaLevel + 3 && !col.rock && col.water <= g && col.desert < 0.3f && !col.volcanic) {
        int lo = g, hi = g;
        for (const auto& d : {std::pair<int, int>{6, 6}, {-6, 6}, {6, -6}, {-6, -6}}) {
            const int hh = column(ax + d.first, az + d.second).h;
            lo = std::min(lo, hh);
            hi = std::max(hi, hh);
        }
        const bool level = hi - lo <= 3;
        const Biome b = col.biome;
        const bool woods = b == Biome::Taiga || b == Biome::SnowyTaiga || b == Biome::Forest || b == Biome::BirchForest;
        const bool open = b == Biome::Plains || b == Biome::Meadow || b == Biome::Savanna || b == Biome::Tundra;
        // a cabin in the woods
        if (woods && level && (rv >> 6) % 4u == 0u) {
            site.kind = 5;
            return site;
        }
        // a ring of standing stones out in the open
        if (open && level && (rv >> 6) % 6u == 0u) {
            site.kind = 6;
            return site;
        }
        // a watchtower on the brow of a hill: higher than the ground all round it
        if ((open || woods || b == Biome::Alpine) && (rv >> 9) % 3u == 0u) {
            int lower = 0;
            for (const auto& d : {std::pair<int, int>{14, 0}, {-14, 0}, {0, 14}, {0, -14}, {10, 10}, {-10, -10}})
                lower += column(ax + d.first, az + d.second).h <= g - 2 ? 1 : 0;
            if (lower >= 5) {
                site.kind = 7;
                return site;
            }
        }
    }
    return none;
}

void TerrainGenerator::placeStructures(Chunk& chunk, ChunkPos pos) const {
    constexpr int kCell = 96;
    constexpr int kReach = 14; // the widest any of them gets from its anchor (a wreck, 6 either way)
    const int baseX = pos.x * kChunkSize;
    const int baseY = pos.y * kChunkSize;
    const int baseZ = pos.z * kChunkSize;
    const int cx0 = floorDivI(baseX - kReach, kCell), cx1 = floorDivI(baseX + kChunkSize - 1 + kReach, kCell);
    const int cz0 = floorDivI(baseZ - kReach, kCell), cz1 = floorDivI(baseZ + kChunkSize - 1 + kReach, kCell);

    const auto put = [&](int wx, int wy, int wz, BlockId id) {
        const int lx = wx - baseX, ly = wy - baseY, lz = wz - baseZ;
        if (lx >= 0 && lx < kChunkSize && ly >= 0 && ly < kChunkSize && lz >= 0 && lz < kChunkSize)
            chunk.set(lx, ly, lz, id);
    };
    // Trees cleared off the ground round a cabin or a tower, someone having cut them back: logs and
    // leaves within r of it, from the ground up.
    const auto clearTrees = [&](int ax, int az, int r, int g) {
        for (int wz = az - r; wz <= az + r; ++wz)
            for (int wx = ax - r; wx <= ax + r; ++wx) {
                if ((wx - ax) * (wx - ax) + (wz - az) * (wz - az) > r * r)
                    continue;
                for (int wy = g - 12; wy <= g + 26; ++wy) { // a tower stands on a brow, its trees lower down
                    const int lx = wx - baseX, ly = wy - baseY, lz = wz - baseZ;
                    if (lx < 0 || lx >= kChunkSize || ly < 0 || ly >= kChunkSize || lz < 0 || lz >= kChunkSize)
                        continue;
                    const BlockId b = chunk.get(lx, ly, lz);
                    if (blocks::isLogBlock(b) || blocks::isLeafBlock(b))
                        chunk.set(lx, ly, lz, blocks::Air);
                }
            }
    };
    // a block that only replaces water or air: the hull of a wreck does not carve the seabed
    const auto putSoft = [&](int wx, int wy, int wz, BlockId id) {
        const int lx = wx - baseX, ly = wy - baseY, lz = wz - baseZ;
        if (lx >= 0 && lx < kChunkSize && ly >= 0 && ly < kChunkSize && lz >= 0 && lz < kChunkSize) {
            const BlockId cur = chunk.get(lx, ly, lz);
            if (cur == blocks::Air || blocks::isWater(cur))
                chunk.set(lx, ly, lz, id);
        }
    };

    for (int cz = cz0; cz <= cz1; ++cz)
        for (int cx = cx0; cx <= cx1; ++cx) {
            const StructureSite st = structureSite(cx, cz);
            if (st.kind == 0)
                continue;
            const int ax = st.x, az = st.z, g = st.h;
            // most of the structure lies within a chunk or two of its anchor's height
            if (baseY > g + 28 || baseY + kChunkSize - 1 < g - 10)
                continue;
            const std::uint32_t rv = voxelHash(ax, 91, az);

            // ---- shipwreck: shallow seabed, not by a desert --------------------------------------
            if (st.kind == 1) {
                const bool alongX = (rv & 4u) != 0u;
                const int y0 = g + 1; // the keel sits a block into the seabed
                for (int u = -6; u <= 6; ++u) {
                    // the hull narrows toward both ends
                    const int half = std::abs(u) > 4 ? 0 : (std::abs(u) > 2 ? 1 : 2);
                    for (int v = -half; v <= half; ++v) {
                        const int wx = ax + (alongX ? u : v), wz = az + (alongX ? v : u);
                        const std::uint32_t h2 = voxelHash(wx, y0, wz);
                        // the keel and ribs are logs, the rest planks, with holes where it has rotted
                        putSoft(wx, y0, wz, v == 0 ? blocks::Wood : blocks::Planks);
                        // pilings under it, so it lies on the bottom instead of hanging in the water
                        for (int k = 1; k <= 5; ++k)
                            putSoft(wx, y0 - k, wz, blocks::Planks);
                        if (std::abs(v) == half && (h2 % 100u) > 22u) { // the sides
                            putSoft(wx, y0 + 1, wz, blocks::Planks);
                            if ((h2 >> 8) % 100u > 45u)
                                putSoft(wx, y0 + 2, wz, blocks::Planks);
                        }
                        if (std::abs(v) < half && (h2 >> 16) % 100u > 55u) // a patchy deck
                            putSoft(wx, y0 + 1, wz, blocks::Planks);
                    }
                }
                // a snapped mast, and the stump of a second
                for (int k = 1; k <= 6; ++k)
                    putSoft(ax, y0 + k, az, blocks::Wood);
                for (int k = 1; k <= 2; ++k)
                    putSoft(alongX ? ax + 3 : ax, y0 + k, alongX ? az : az + 3, blocks::Wood);
                continue;
            }

            // ---- desert ruins ------------------------------------------------------------------
            if (st.kind == 2) {
                // a paved square with broken pillars at the corners and a low wall along two sides
                constexpr int kR = 6;
                for (int dz = -kR; dz <= kR; ++dz)
                    for (int dx = -kR; dx <= kR; ++dx) {
                        put(ax + dx, g, az + dz, blocks::Sandstone);
                        // clear the dunes off it
                        for (int k = 1; k <= 4; ++k)
                            put(ax + dx, g + k, az + dz, blocks::Air);
                    }
                // eight pillars round the edge, each broken off at its own height
                for (int i = 0; i < 8; ++i) {
                    static constexpr int kOff[8][2] = {{-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1}};
                    const int px = ax + kOff[i][0] * kR, pz = az + kOff[i][1] * kR;
                    const int ph = 2 + int((voxelHash(px, 92, pz) >> 5) % 6u);
                    for (int k = 1; k <= ph; ++k)
                        put(px, g + k, pz, blocks::Sandstone);
                }
                // a low wall along two sides with gaps where it has fallen
                for (int d = -kR + 1; d <= kR - 1; ++d) {
                    if (voxelHash(ax + d, 93, az) % 100u > 35u)
                        put(ax + d, g + 1, az - kR, blocks::Sandstone);
                    if (voxelHash(ax, 94, az + d) % 100u > 35u)
                        put(ax - kR, g + 1, az + d, blocks::Sandstone);
                }
                // an altar in the middle
                put(ax, g + 1, az, blocks::Sandstone);
                put(ax + 1, g + 1, az, blocks::Sandstone);
                put(ax, g + 2, az, blocks::Sandstone);
                continue;
            }

            // ---- lighthouse: a bit of coast with open water not far off --------------------------
            if (st.kind == 3) {
                // foundation: a stone footing down to whatever is under it
                for (int dz = -3; dz <= 3; ++dz)
                    for (int dx = -3; dx <= 3; ++dx)
                        if (dx * dx + dz * dz <= 10)
                            for (int k = -6; k <= 0; ++k)
                                put(ax + dx, g + k, az + dz, blocks::Stone);
                constexpr int kTower = 16;
                for (int k = 1; k <= kTower; ++k) {
                    const int R = 3 - (k - 1) / 8; // 3 low down, 2 near the top
                    const BlockId wall = ((k - 1) / 3) % 2 == 0 ? blocks::Bricks : blocks::Snow; // red and white
                    for (int dz = -R; dz <= R; ++dz)
                        for (int dx = -R; dx <= R; ++dx) {
                            const int d2 = dx * dx + dz * dz;
                            if (d2 > R * R + 1)
                                continue;
                            const bool hollow = d2 <= (R - 1) * (R - 1) + (R > 2 ? 0 : 0);
                            BlockId id = hollow ? blocks::Air : wall;
                            // a doorway on the +z side, and a slit window every fourth level on the others
                            if (!hollow && dx == 0 && dz == R && k <= 2)
                                id = blocks::Air;
                            if (!hollow && k % 4 == 2 && k > 2 && (dx == 0 || dz == 0) && !(dx == 0 && dz == R))
                                id = blocks::Air;
                            put(ax + dx, g + k, az + dz, id);
                        }
                }
                // the gallery, then the lantern room with a light in it
                const int top = g + kTower + 1;
                for (int dz = -4; dz <= 4; ++dz)
                    for (int dx = -4; dx <= 4; ++dx)
                        if (dx * dx + dz * dz <= 17)
                            put(ax + dx, top, az + dz, blocks::Stone);
                for (int dz = -2; dz <= 2; ++dz)
                    for (int dx = -2; dx <= 2; ++dx) {
                        const int d2 = dx * dx + dz * dz;
                        if (d2 > 5)
                            continue;
                        for (int k = 1; k <= 3; ++k) {
                            // stone posts at the corners of the ring, open between them
                            const bool post = std::abs(dx) == 2 || std::abs(dz) == 2 ? (std::abs(dx) + std::abs(dz)) >= 3 : false;
                            put(ax + dx, top + k, az + dz, d2 >= 4 ? (post ? blocks::Stone : blocks::Air) : blocks::Air);
                        }
                        put(ax + dx, top + 4, az + dz, blocks::Stone); // the roof
                    }
                put(ax, top + 2, az, blocks::Glow);
                put(ax, top + 5, az, blocks::Stone);
                continue;
            }

            // ---- a cabin in the woods: logs at the corners, plank walls, a gable roof ------------
            if (st.kind == 5) {
                const bool alongX = (rv & 4u) != 0u;
                constexpr int kLong = 4, kShort = 3; // 9 x 7
                const bool spruce = column(ax, az).biome == Biome::Taiga || column(ax, az).biome == Biome::SnowyTaiga;
                const BlockId log = spruce ? blocks::SpruceWood : blocks::Wood;
                const BlockId planks = spruce ? blocks::SprucePlanks : blocks::Planks;
                const auto at = [&](int l, int s, int k, BlockId id) { put(alongX ? ax + l : ax + s, g + k, alongX ? az + s : az + l, id); };
                clearTrees(ax, az, 9, g);
                for (int l = -kLong - 1; l <= kLong + 1; ++l)
                    for (int s = -kShort - 1; s <= kShort + 1; ++s) {
                        const bool inside = std::abs(l) <= kLong && std::abs(s) <= kShort;
                        if (inside) {
                            for (int k = -3; k <= -1; ++k)
                                at(l, s, k, blocks::Cobblestone);
                            at(l, s, 0, planks);
                            const bool corner = std::abs(l) == kLong && std::abs(s) == kShort;
                            const bool wall = std::abs(l) == kLong || std::abs(s) == kShort;
                            for (int k = 1; k <= 4; ++k) {
                                BlockId id = corner ? log : (wall ? planks : blocks::Air);
                                if (wall && !corner && k == 2 && (l == 0 || s == 0 || std::abs(l) == 2))
                                    id = blocks::Glass;
                                if (s == -kShort && l == 0 && k <= 2)
                                    id = blocks::Air; // the door
                                at(l, s, k, id);
                            }
                            // the gable ends fill up to the roof
                            if (std::abs(l) == kLong)
                                for (int k = 5; k < 5 + kShort - std::abs(s); ++k)
                                    at(l, s, k, planks);
                            else
                                for (int k = 5; k < 5 + kShort - std::abs(s); ++k)
                                    at(l, s, k, blocks::Air);
                        }
                        // the roof, one plank thick, a block of overhang all round
                        at(l, s, 5 + kShort - std::abs(s), planks);
                        for (int k = 1; k <= 3; ++k)
                            if (!inside)
                                at(l, s, 5 + kShort - std::abs(s) + k, blocks::Air);
                    }
                // a stone chimney up the far end, and a lamp lit inside
                for (int k = 1; k <= 5 + kShort + 3; ++k)
                    at(kLong - 1, kShort - 1, k, blocks::Cobblestone);
                at(-kLong + 1, kShort - 1, 1, BlockId(blocks::LampFirst + blocks::kLampHues + 1));
                continue;
            }

            // ---- standing stones: a ring of them, some fallen, a flat stone in the middle --------
            if (st.kind == 6) {
                const int count = 8 + int((rv >> 12) % 4u);
                const float radius = 6.0f + float((rv >> 16) % 3u);
                for (int i = 0; i < count; ++i) {
                    const std::uint32_t sh = voxelHash(ax + i, 96, az);
                    const float ang = (float(i) + 0.3f * (float(sh % 100u) / 100.0f)) * 6.2831853f / float(count);
                    const int px = ax + int(std::lround(std::cos(ang) * radius)), pz = az + int(std::lround(std::sin(ang) * radius));
                    const int gy = column(px, pz).h;
                    const BlockId stone = (sh >> 8) % 3u == 0u ? blocks::Andesite : blocks::Stone;
                    if ((sh >> 12) % 5u == 0u) { // fallen, lying outward
                        put(px, gy + 1, pz, stone);
                        put(px + (std::cos(ang) > 0.0f ? 1 : -1), gy + 1, pz, blocks::MossyCobble);
                        continue;
                    }
                    const int tall = 2 + int((sh >> 16) % 3u);
                    for (int k = 0; k <= tall; ++k)
                        put(px, gy + k, pz, k == tall && (sh & 1u) ? blocks::MossyCobble : stone);
                }
                put(ax, g + 1, az, blocks::Andesite);
                put(ax + 1, g + 1, az, blocks::Andesite);
                continue;
            }

            // ---- a watchtower gone to ruin on the brow of a hill ---------------------------------
            if (st.kind == 7) {
                constexpr int kHalf = 2; // 5 x 5
                const int tall = 9 + int((rv >> 12) % 6u);
                clearTrees(ax, az, 7, g);
                for (int dz = -kHalf; dz <= kHalf; ++dz)
                    for (int dx = -kHalf; dx <= kHalf; ++dx) {
                        const int wx = ax + dx, wz = az + dz;
                        for (int k = -4; k <= 0; ++k)
                            put(wx, g + k, wz, blocks::StoneBricks);
                        const bool wall = std::abs(dx) == kHalf || std::abs(dz) == kHalf;
                        const int topK = tall - int(voxelHash(wx, 95, wz) % 4u); // the top broken unevenly
                        for (int k = 1; k <= tall + 1; ++k) {
                            const std::uint32_t bh = voxelHash(wx, g + k, wz);
                            BlockId id = blocks::Air;
                            if (wall && k <= topK)
                                id = bh % 5u == 0u ? blocks::MossyCobble : (bh % 3u == 0u ? blocks::Cobblestone : blocks::StoneBricks);
                            if (wall && dz == kHalf && dx == 0 && k <= 2)
                                id = blocks::Air; // the doorway
                            if (wall && k % 4 == 3 && (dx == 0 || dz == 0) && !(dz == kHalf && dx == 0))
                                id = blocks::Air; // arrow slits
                            put(wx, g + k, wz, id);
                        }
                    }
                // the stones that fell, lying round its foot
                for (int i = 0; i < 12; ++i) {
                    const std::uint32_t fh = voxelHash(ax + i * 7, 97, az - i * 3);
                    const int ox = int(fh % 13u) - 6, oz = int((fh >> 8) % 13u) - 6;
                    if (std::abs(ox) <= kHalf + 1 && std::abs(oz) <= kHalf + 1)
                        continue;
                    put(ax + ox, column(ax + ox, az + oz).h + 1, az + oz, (fh >> 16) & 1u ? blocks::MossyCobble : blocks::Cobblestone);
                }
                continue;
            }
        }
    placePyramids(chunk, pos);
}

// One tree, every part of it a function of its world position, so each chunk it overlaps builds
// the same one. kind is a TreeKind.
void TerrainGenerator::growTree(Chunk& chunk, ChunkPos pos, int kind, int tx, int tz, int surf,
                                std::uint32_t rv, BlockId leaves) const {
    const int baseX = pos.x * kChunkSize;
    const int baseY = pos.y * kChunkSize;
    const int baseZ = pos.z * kChunkSize;
    const auto put = [&](int wx, int wy, int wz, BlockId id) {
        const int lx = wx - baseX, ly = wy - baseY, lz = wz - baseZ;
        if (lx < 0 || lx >= kChunkSize || ly < 0 || ly >= kChunkSize || lz < 0 || lz >= kChunkSize)
            return;
        const BlockId cur = chunk.get(lx, ly, lz);
        // wood overwrites air, plants and leaves; leaves only fill air and plants (so trunks stay solid);
        // a mangrove grows up out of the water
        if (cur == blocks::Air || (blocks::isCross(cur) && !blocks::isWaterlogged(cur)) ||
            (blocks::isLogBlock(id) && blocks::isLeafBlock(cur)) || (kind == kTreeMangrove && cur == blocks::Water))
            chunk.set(lx, ly, lz, id);
    };
    const int species = kind == kTreeSpruce || kind == kTreeTallSpruce || kind == kTreeSmallSpruce || kind == kTreeRedwood ? 0
                        : kind == kTreeBirch || kind == kTreeAspen                                                         ? 2
                                                                                                                           : 1;
    const BlockId logId = kind == kTreeRedwood    ? blocks::RedwoodWood
                          : kind == kTreeCherry   ? blocks::CherryWood
                          : kind == kTreeMangrove ? blocks::MangroveWood
                          : species == 0          ? blocks::SpruceWood
                          : species == 2          ? blocks::BirchWood
                          : (kind == kTreeJungle || kind == kTreeBigJungle) ? blocks::JungleWood
                          : kind == kTreeAcacia ? blocks::AcaciaWood
                                                : blocks::Wood;
    const BlockId leafId = leaves != blocks::Air ? leaves
                           : kind == kTreeCherry ? blocks::CherryLeaves
                           : species == 0 ? blocks::SpruceLeaves
                           : species == 2 ? blocks::BirchLeaves
                           : (kind == kTreeJungle || kind == kTreeBigJungle) ? blocks::JungleLeaves
                           : kind == kTreeAcacia ? blocks::AcaciaLeaves
                                                 : blocks::Leaves;
        const auto leaf = [&](int wx, int wy, int wz, int holePerMille) {
            if (int(voxelHash(wx, wy, wz) % 1000) >= holePerMille)
                put(wx, wy, wz, leafId);
        };
        // A clump of leaves centred anywhere, the piece that lets a canopy be built
        // out of parts hanging off branch ends instead of one ellipsoid sitting on the
        // trunk. Squashed vertically because a sphere of leaves reads as a ball.
        const auto blob = [&](int bx, int by, int bz, int r, int holePerMille) {
            for (int dy = -r; dy <= r; ++dy)
                for (int dz = -r; dz <= r; ++dz)
                    for (int dx = -r; dx <= r; ++dx) {
                        const int rr = dx * dx + dz * dz + dy * dy * 2;
                        if (rr <= r * r + 1)
                            leaf(bx + dx, by + dy, bz + dz,
                                 rr > (r - 1) * (r - 1) ? holePerMille : holePerMille / 4);
                    }
        };
        // A limb from the trunk at fromY, out along ang, rising as it goes. Returns
        // where it ended so the caller can hang a clump there.
        const auto limb = [&](int fromY, float ang, int len, int rise) {
            int ex = tx, ey = fromY, ez = tz;
            for (int t = 1; t <= len; ++t) {
                ex = tx + int(std::lround(std::cos(ang) * float(t)));
                ez = tz + int(std::lround(std::sin(ang) * float(t)));
                ey = fromY + (t * rise + len - 1) / len;
                put(ex, ey, ez, logId);
            }
            return glm::ivec3{ex, ey, ez};
        };

    // a flat layer of leaves, for acacia crowns and swamp oaks
    const auto disc = [&](int cx, int cy, int cz, int r, int holePerMille) {
        for (int dz = -r; dz <= r; ++dz)
            for (int dx = -r; dx <= r; ++dx)
                if (dx * dx + dz * dz <= r * r + 1)
                    leaf(cx + dx, cy, cz + dz, dx * dx + dz * dz > (r - 1) * (r - 1) ? holePerMille : holePerMille / 4);
    };
    // leaves hanging off a crown's edge, one to three blocks
    const auto drape = [&](int cx, int cy, int cz, int r, int count, std::uint32_t seed) {
        for (int k = 0; k < count; ++k) {
            const std::uint32_t dh = voxelHash(cx + k, int(seed & 1023u), cz - k);
            const float ang = float(dh & 255u) * 0.02464f;
            const int ex = cx + int(std::lround(std::cos(ang) * float(r)));
            const int ez = cz + int(std::lround(std::sin(ang) * float(r)));
            const int len = 1 + int((dh >> 8) % 3u);
            for (int d = 0; d < len; ++d)
                put(ex, cy - d, ez, leafId);
        }
    };

    if (kind == kTreeSpruce || kind == kTreeBirch || kind == kTreeOak || kind == kTreeAspen) {
        if (species == 0) {
                // SPRUCE: tall bare lower trunk, then narrow tiers with GAPS between them.
                // The gaps are the whole silhouette, a continuous cone just reads as a
                // green triangle, which is what the old spruce was.
                const int trunk = 9 + int(rv % 6u); // 9..14
                for (int i = 1; i <= trunk; ++i)
                    put(tx, surf + i, tz, logId);
                put(tx, surf + trunk + 1, tz, leafId); // tip
                const int bare = trunk / 3 + 1;                // no branches down here
                for (int wy = surf + trunk; wy > surf + bare; --wy) {
                    const int fromTop = surf + trunk - wy;
                    if (fromTop % 3 == 2)
                        continue; // the gap between tiers
                    int r = 1 + fromTop / 4;
                    if (r > 3)
                        r = 3;
                    if (fromTop % 3 == 1 && r > 1)
                        --r; // tiers taper within themselves
                    for (int dz = -r; dz <= r; ++dz)
                        for (int dx = -r; dx <= r; ++dx) {
                            if (dx == 0 && dz == 0)
                                continue; // keep the trunk clear
                            if (dx * dx + dz * dz <= r * r + 1)
                                leaf(tx + dx, wy, tz + dz, 200);
                        }
                }
            } else if (species == 1) {
                // OAK: real limbs. 3-4 of them fan off the upper trunk and each carries its
                // own clump, so the canopy is the union of off-centre parts, lumpy and
                // asymmetric by construction rather than by punching holes in a ball.
                const int trunk = 5 + int(rv % 3u); // 5..7
                for (int i = 1; i <= trunk; ++i)
                    put(tx, surf + i, tz, logId);
                const int n = 3 + int((rv >> 5) & 1u);              // 3..4 limbs
                const float base = float(rv >> 9 & 63u) * 0.0981f;  // random fan rotation
                for (int b = 0; b < n; ++b) {
                    const std::uint32_t bh = voxelHash(tx, 20 + b, tz);
                    const float ang = base + float(b) * 6.2832f / float(n) +
                                      (float(bh & 15u) - 7.5f) * 0.035f;
                    const int len = 2 + int((bh >> 6) & 1u);        // 2..3
                    const int from = surf + trunk - int((bh >> 8) % 3u); // stagger heights
                    const glm::ivec3 end = limb(from, ang, len, 1 + int((bh >> 11) & 1u));
                    blob(end.x, end.y + 1, end.z, 2, 300);
                }
                // r=3 rather than 2: at r=2 the vertical squash leaves only one row above
                // and below centre, which reads as a disc balanced on the trunk instead of
                // a crown. 3 keeps the reach inside kTreeCanopyR while giving it volume.
                blob(tx, surf + trunk + 1, tz, 3, 220);
            } else {
                // BIRCH: slender and tall with a small high canopy, so a stand of them
                // reads as vertical lines. Deliberately the sparsest of the three. An aspen is the
                // same tree a few blocks taller.
                const int trunk = (kind == kTreeAspen ? 10 : 7) + int(rv % 4u); // 7..10, 10..13
                for (int i = 1; i <= trunk; ++i)
                    put(tx, surf + i, tz, logId);
                const int topY = surf + trunk;
                for (int b = 0; b < 2; ++b) {
                    const std::uint32_t bh = voxelHash(tx, 40 + b, tz);
                    const float ang = float(bh & 63u) * 0.0981f + float(b) * 3.1416f;
                    const glm::ivec3 end = limb(topY - 1 - int((bh >> 7) & 1u), ang, 2, 1);
                    blob(end.x, end.y + 1, end.z, 2, 260);
                }
                blob(tx, topY + 1, tz, 2, 240);
            }
        return;
    }
    if (kind == kTreeTallSpruce || kind == kTreeSmallSpruce) {
        // the old forest's giants and the treeline's stunted ones: the same tiers at another size
        const bool tall = kind == kTreeTallSpruce;
        const int trunk = tall ? 15 + int(rv % 6u) : 4 + int(rv % 3u);
        for (int i = 1; i <= trunk; ++i)
            put(tx, surf + i, tz, logId);
        put(tx, surf + trunk + 1, tz, leafId);
        put(tx, surf + trunk + 2, tz, leafId);
        const int bare = tall ? trunk / 2 : 1;
        const int maxR = tall ? 3 : 2;
        for (int wy = surf + trunk; wy > surf + bare; --wy) {
            const int fromTop = surf + trunk - wy;
            if (fromTop % 3 == 2)
                continue;
            int r = std::min(1 + fromTop / (tall ? 5 : 3), maxR);
            if (fromTop % 3 == 1 && r > 1)
                --r;
            for (int dz = -r; dz <= r; ++dz)
                for (int dx = -r; dx <= r; ++dx)
                    if ((dx != 0 || dz != 0) && dx * dx + dz * dz <= r * r + 1)
                        leaf(tx + dx, wy, tz + dz, 220);
        }
        return;
    }
    if (kind == kTreeBigOak || kind == kTreeSwampOak) {
        const bool swamp = kind == kTreeSwampOak;
        const int trunk = swamp ? 4 + int(rv % 3u) : 7 + int(rv % 3u);
        for (int i = 1; i <= trunk; ++i)
            put(tx, surf + i, tz, logId);
        if (swamp) {
            // low and wide, hung with leaves
            disc(tx, surf + trunk - 1, tz, 3, 260);
            disc(tx, surf + trunk, tz, 4, 300);
            disc(tx, surf + trunk + 1, tz, 2, 200);
            drape(tx, surf + trunk - 1, tz, 4, 6, rv);
            return;
        }
        const int n = 4 + int((rv >> 5) & 1u);
        const float base = float(rv >> 9 & 63u) * 0.0981f;
        for (int b = 0; b < n; ++b) {
            const std::uint32_t bh = voxelHash(tx, 60 + b, tz);
            const float ang = base + float(b) * 6.2832f / float(n) + (float(bh & 15u) - 7.5f) * 0.035f;
            const int len = 3 + int((bh >> 6) & 1u);
            const int from = surf + trunk - 1 - int((bh >> 8) % 4u);
            const glm::ivec3 e = limb(from, ang, len, 1 + int((bh >> 11) & 1u));
            blob(e.x, e.y + 1, e.z, 2, 280);
        }
        blob(tx, surf + trunk + 1, tz, 3, 240);
        return;
    }
    if (kind == kTreeJungle || kind == kTreeBigJungle) {
        const bool big = kind == kTreeBigJungle;
        const int trunk = big ? 16 + int(rv % 7u) : 9 + int(rv % 6u);
        for (int i = 1; i <= trunk; ++i) {
            put(tx, surf + i, tz, logId);
            if (big) {
                put(tx + 1, surf + i, tz, logId);
                put(tx, surf + i, tz + 1, logId);
                put(tx + 1, surf + i, tz + 1, logId);
            }
        }
        const int topY = surf + trunk;
        for (int b = 0; b < (big ? 3 : 2); ++b) {
            const std::uint32_t bh = voxelHash(tx, 80 + b, tz);
            const float ang = float(bh & 63u) * 0.0981f + float(b) * 2.1f;
            const glm::ivec3 e = limb(topY - 3 - int((bh >> 7) % 4u), ang, big ? 3 : 2, 1);
            blob(e.x, e.y + 1, e.z, 2, 250);
        }
        blob(tx, topY + 1, tz, big ? 4 : 3, 220);
        drape(tx, topY, tz, big ? 4 : 3, big ? 8 : 5, rv);
        return;
    }
    if (kind == kTreeAcacia) {
        // a short trunk that leans and forks, each fork carrying a flat crown
        const int stem = 2 + int(rv % 2u);
        for (int i = 1; i <= stem; ++i)
            put(tx, surf + i, tz, logId);
        const float ang = float(rv >> 4 & 63u) * 0.0997f;
        const auto lean = [&](float a, int steps, int y0) {
            int x = tx, z = tz, y = y0;
            for (int i = 1; i <= steps; ++i) {
                x = tx + int(std::lround(std::cos(a) * float(i)));
                z = tz + int(std::lround(std::sin(a) * float(i)));
                ++y;
                put(x, y, z, logId);
            }
            disc(x, y + 1, z, 3, 160);
            disc(x, y + 2, z, 2, 160);
        };
        lean(ang, 3, surf + stem);
        if ((rv >> 12) % 3u != 0u)
            lean(ang + 2.6f + float(rv >> 14 & 7u) * 0.1f, 2, surf + stem);
        return;
    }
    if (kind == kTreeCherry) {
        // a short trunk opening into two or three long limbs, each under a wide pink cloud, with
        // blossom hanging off the edges
        const int trunk = 4 + int(rv % 3u);
        for (int i = 1; i <= trunk; ++i)
            put(tx, surf + i, tz, logId);
        const int n = 2 + int((rv >> 5) % 2u);
        const float base = float(rv >> 9 & 63u) * 0.0981f;
        for (int b = 0; b < n; ++b) {
            const std::uint32_t bh = voxelHash(tx, 120 + b, tz);
            const float ang = base + float(b) * 6.2832f / float(n) + (float(bh & 15u) - 7.5f) * 0.05f;
            const glm::ivec3 e = limb(surf + trunk - int((bh >> 8) % 2u), ang, 2 + int((bh >> 6) & 1u), 2);
            disc(e.x, e.y, e.z, 2, 260);
            disc(e.x, e.y + 1, e.z, 3, 300);
            disc(e.x, e.y + 2, e.z, 2, 340);
            drape(e.x, e.y, e.z, 3, 3, bh);
        }
        disc(tx, surf + trunk + 1, tz, 3, 280);
        disc(tx, surf + trunk + 2, tz, 2, 320);
        return;
    }
    if (kind == kTreeRedwood) {
        // a giant: a 2x2 red trunk bare for half its height, then short tiers of needles up to a point
        const int trunk = 20 + int(rv % 8u); // 20..27
        for (int i = 1; i <= trunk; ++i)
            for (int k = 0; k < 4; ++k)
                put(tx + (k & 1), surf + i, tz + (k >> 1), logId);
        const int top = surf + trunk;
        for (int k = 0; k < 4; ++k)
            put(tx + (k & 1), top + 1, tz + (k >> 1), leafId);
        put(tx, top + 2, tz, leafId);
        put(tx + 1, top + 2, tz + 1, leafId);
        put(tx, top + 3, tz, leafId);
        for (int wy = top; wy > surf + trunk / 2; --wy) {
            const int fromTop = top - wy;
            if (fromTop % 3 == 2)
                continue;
            int r = std::min(2 + fromTop / 4, 4);
            if (fromTop % 3 == 1)
                --r;
            for (int dz = -r; dz <= r + 1; ++dz)
                for (int dx = -r; dx <= r + 1; ++dx) {
                    if ((dx == 0 || dx == 1) && (dz == 0 || dz == 1))
                        continue;
                    const float ex = float(dx) - 0.5f, ez = float(dz) - 0.5f;
                    if (ex * ex + ez * ez <= float(r * r) + 1.0f)
                        leaf(tx + dx, wy, tz + dz, 200);
                }
        }
        return;
    }
    if (kind == kTreeDarkOak) {
        // a squat 2x2 trunk under a broad, thick crown that keeps the floor in shade
        const int trunk = 6 + int(rv % 3u);
        for (int i = 1; i <= trunk; ++i)
            for (int k = 0; k < 4; ++k)
                put(tx + (k & 1), surf + i, tz + (k >> 1), logId);
        const int n = 4;
        const float base = float(rv >> 9 & 63u) * 0.0981f;
        for (int b = 0; b < n; ++b) {
            const std::uint32_t bh = voxelHash(tx, 140 + b, tz);
            const float ang = base + float(b) * 1.5708f + (float(bh & 15u) - 7.5f) * 0.04f;
            const glm::ivec3 e = limb(surf + trunk - 1 - int((bh >> 8) % 2u), ang, 3, 1);
            blob(e.x, e.y + 1, e.z, 2, 160);
        }
        blob(tx, surf + trunk + 1, tz, 3, 140);
        blob(tx + 1, surf + trunk + 1, tz + 1, 3, 140);
        return;
    }
    if (kind == kTreeMangrove) {
        // up on arching roots out of the mud and the shallows, a slim trunk, a broad crown
        const int lift = 3 + int(rv % 2u);
        const int trunk = 5 + int((rv >> 3) % 4u);
        for (int i = lift + 1; i <= lift + trunk; ++i)
            put(tx, surf + i, tz, logId);
        for (int i = 1; i <= lift; ++i)
            put(tx, surf + i, tz, blocks::MangroveRoots);
        const int n = 4 + int((rv >> 6) % 3u);
        const float base = float(rv >> 9 & 63u) * 0.0981f;
        for (int b = 0; b < n; ++b) {
            const std::uint32_t bh = voxelHash(tx, 160 + b, tz);
            const float ang = base + float(b) * 6.2832f / float(n) + (float(bh & 15u) - 7.5f) * 0.05f;
            const int len = 2 + int((bh >> 5) % 2u);
            for (int t = 1; t <= len; ++t) {
                const int rx = tx + int(std::lround(std::cos(ang) * float(t))), rz = tz + int(std::lround(std::sin(ang) * float(t)));
                const int ry = surf + lift + 1 - (t * (lift + 1) + len - 1) / len;
                put(rx, ry + 1, rz, blocks::MangroveRoots);
                if (t == len) // down into the bed
                    for (int d = ry; d >= ry - 3; --d)
                        put(rx, d, rz, blocks::MangroveRoots);
            }
        }
        const int top = surf + lift + trunk;
        for (int b = 0; b < 3; ++b) {
            const std::uint32_t bh = voxelHash(tx, 170 + b, tz);
            const float ang = float(bh & 63u) * 0.0981f + float(b) * 2.1f;
            const glm::ivec3 e = limb(top - 1 - int((bh >> 7) % 2u), ang, 2, 1);
            blob(e.x, e.y + 1, e.z, 2, 240);
        }
        blob(tx, top + 1, tz, 3, 220);
        drape(tx, top, tz, 3, 5, rv);
        return;
    }
    if (kind == kTreeBamboo) {
        // a clump of stalks, each standing on its own bit of ground
        const int n = 14 + int(rv % 10u);
        for (int k = 0; k < n; ++k) {
            const std::uint32_t kh = voxelHash(tx, 180 + k, tz);
            const int sx = tx + int(kh % 7u) - 3, sz = tz + int((kh >> 3) % 7u) - 3;
            const Column c = column(sx, sz);
            if (c.water > c.h || c.rock || c.h < kSeaLevel + 1)
                continue;
            const int tall = 7 + int((kh >> 6) % 11u);
            for (int i = 1; i <= tall; ++i)
                put(sx, c.h + i, sz, blocks::Bamboo);
        }
        return;
    }
    if (kind == kTreeMushroomRed || kind == kTreeMushroomBrown) {
        const bool red = kind == kTreeMushroomRed;
        const int stem = (red ? 5 : 4) + int(rv % 3u);
        for (int i = 1; i <= stem; ++i)
            put(tx, surf + i, tz, blocks::MushroomStem);
        const int top = surf + stem;
        if (red) {
            // a dome: a cap on top and its sides hanging three blocks down
            for (int dz = -1; dz <= 1; ++dz)
                for (int dx = -1; dx <= 1; ++dx)
                    put(tx + dx, top + 1, tz + dz, blocks::MushroomCapRed);
            for (int dy = 0; dy >= -2; --dy)
                for (int dz = -2; dz <= 2; ++dz)
                    for (int dx = -2; dx <= 2; ++dx)
                        if ((std::abs(dx) == 2 || std::abs(dz) == 2) && !(std::abs(dx) == 2 && std::abs(dz) == 2))
                            put(tx + dx, top + dy, tz + dz, blocks::MushroomCapRed);
        } else {
            // a flat plate, wide and one block thick
            for (int dz = -3; dz <= 3; ++dz)
                for (int dx = -3; dx <= 3; ++dx)
                    if (!(std::abs(dx) == 3 && std::abs(dz) == 3))
                        put(tx + dx, top + 1, tz + dz, blocks::MushroomCapBrown);
        }
        return;
    }
}

void TerrainGenerator::placeTrees(Chunk& chunk, ChunkPos pos) const {
    if (m_version >= 3) {
        placeTrees3(chunk, pos);
        return;
    }
    const int baseX = pos.x * kChunkSize;
    const int baseZ = pos.z * kChunkSize;
    const int cx0 = floorDivI(baseX - kTreeCanopyR, kTreeCell);
    const int cx1 = floorDivI(baseX + kChunkSize - 1 + kTreeCanopyR, kTreeCell);
    const int cz0 = floorDivI(baseZ - kTreeCanopyR, kTreeCell);
    const int cz1 = floorDivI(baseZ + kChunkSize - 1 + kTreeCanopyR, kTreeCell);

    for (int cz = cz0; cz <= cz1; ++cz)
        for (int cx = cx0; cx <= cx1; ++cx) {
            const std::uint32_t hsh = cellHash(wrapMul(cx, 1013u) + std::uint32_t(m_seed), wrapMul(cz, 79u) + std::uint32_t(m_seed));
            // Forests and meadows: a broad density field decides how many cells grow a tree, from
            // open grassland (about 1 in 8) to dense woodland (about 1 in 2). It used to be a flat
            // 43% everywhere, which reads as one endless orchard.
            const float forestM = valueNoise(float(cx * kTreeCell) * 0.0045f,
                                             float(cz * kTreeCell) * 0.0045f, seedOff(m_seed, 71));
            const int treeLimit = 30 + int(smooth01(0.30f, 0.70f, forestM) * 105.0f);
            if (int(hsh & 255u) > treeLimit)
                continue;
            const int tx = cx * kTreeCell + int((hsh >> 8) % kTreeCell);
            const int tz = cz * kTreeCell + int((hsh >> 13) % kTreeCell);
            const Column col = column(tx, tz);
            if (m_version >= 3) {
                // Roots need soil, and dry country carries few trees. The woods thin out and stop
                // a good way under the snow.
                if (col.rock || col.water > col.h || col.h > col.snowLine - 34)
                    continue;
                if (col.h > col.snowLine - 74 && int((hsh >> 21) % 40u) < col.h - (col.snowLine - 74))
                    continue;
                const float wet = 0.12f + 0.88f * smooth01(0.16f, 0.5f, col.moisture);
                if (float(hsh & 255u) > float(treeLimit) * wet)
                    continue;
            }
            const int surf = col.h;
            if (surf < kSeaLevel + 1)
                continue; // nothing grows underwater
            if (col.desert > 0.35f)
                continue; // and nothing tall in the desert
            // Version 2's mountains have a treeline: the woods thin out above 130 and stop by
            // about 175, leaving the upper slopes rock and snow.
            if (m_version == 2 && surf > 130 && int((hsh >> 21) % 48u) < surf - 130)
                continue;

            // per-tree variation: an independent hash drives size and shape. everything
            // below is a pure function of the tree's world position, which is what lets
            // every chunk it overlaps compute the identical tree with no neighbour lookups.
            const std::uint32_t rv = voxelHash(tx, 7, tz);

            // The beach belongs to palms, and they are sparser than a forest. (Low ground by an
            // inland lake is not a beach in version 2, so it keeps ordinary trees.)
            if (surf <= kBeachTop + 1 && (m_version < 2 || col.coast < 45.0f)) {
                if (((hsh >> 18) & 255u) > 96u)
                    continue;
                placePalm(chunk, pos, tx, tz, surf, rv);
                continue;
            }

            // Species: cold high ground grows spruce, the lowlands oak and birch, and a coarse region
            // hash keeps stands together (a spruce forest, an oak grove) instead of salt and pepper.
            const std::uint32_t region = cellHash(wrapMul(cx >> 3, 31u) + std::uint32_t(m_seed), wrapMul(cz >> 3, 17u));
            const float regionR = float(region & 1023u) / 1024.0f;
            const float cold = m_version >= 3 ? smooth01(0.40f, 0.26f, col.temp - float(surf - kSeaLevel) * 0.0022f)
                                              : smooth01(100.0f, 160.0f, float(surf));
            int species; // 0 spruce, 1 oak, 2 birch
            if (regionR < cold * 0.9f)
                species = 0;
            else
                species = (region >> 10) % 100u < 62u ? 1 : 2;
            if ((rv >> 17) % 6u == 0u) // a few strays keep a stand from looking planted
                species = int((rv >> 19) % 3u);
            growTree(chunk, pos, species, tx, tz, surf, rv);
        }
}

} // namespace vx
