#include "gen/TerrainGenerator.hpp"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>

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
constexpr int kCaveFloor2 = 6;
constexpr int kCaveDepth = 56;  // caves only within this many blocks below the surface
// Squared radius of the worm tunnels around the curve where both worm fields vanish.
// This is the single knob for tunnel width, and it is sharp: the carved volume goes with
// the square of it, so small changes here move the cave-air count a long way.
constexpr float kWormR2 = 0.26f;
constexpr int kTreeCell = 7; // one tree candidate per 7x7 column cell
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

TerrainGenerator::TerrainGenerator(int seed, int version) : m_seed(seed), m_version(version) {
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
                if (c.h >= kSeaLevel + 3 && c.h <= kSeaLevel + 40 && c.desert < 0.3f)
                    return {x, z};
            }
    return {16, 16};
}

int TerrainGenerator::snowLineAt(int wx, int wz) const {
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
    return {col.h, col.desert,
            smooth01(0.30f, 0.70f, valueNoise(float(wx) * 0.0045f, float(wz) * 0.0045f, seedOff(m_seed, 71)))};
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
    if (m_version >= 2) {
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
    for (int z = 0; z < kChunkSize; ++z)
        for (int x = 0; x < kChunkSize; ++x) {
            hMin = std::min(hMin, heights[z][x]);
            hMax = std::max(hMax, heights[z][x]);
        }

    // wholly above both terrain and sea: air, except trees can still reach up into it
    if (baseY > hMax && baseY > kSeaLevel) {
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
    // Version 2's caves go all the way down (kCaveFloor2), but only in detailed chunks: a coarse
    // one never has caves at all, so it keeps the shortcut.
    const bool deepCaves = m_version >= 2 && detailed;
    if (baseY + kChunkSize - 1 < (deepCaves ? kCaveFloor2 : hMin - kCaveDepth)) {
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
            const bool desert = desertW[z][x] > 0.42f + 0.16f * float(cellHash(baseX + x, baseZ + z) & 255u) / 255.0f;
            const bool beach = !desert && isBeach(h, coast[z][x]);
            const bool seabed = h < kSeaLevel - 1;
            const bool bare = !desert && !beach && !seabed && rock[z][x] != 0;
            const bool snowy = !desert && !beach && !seabed && rock[z][x] < 2 && h >= snowLineAt(baseX + x, baseZ + z);
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
                if (wy >= h - 3) {
                    if (beach)
                        id = blocks::BeachSand;
                    else if (seabed)
                        id = gravelBed ? blocks::Gravel : blocks::Sand;
                    else if (desert)
                        id = blocks::Sand;
                    else if (snowy && wy >= h - 1)
                        id = blocks::Snow; // a couple blocks of cap
                    else if (bare)
                        id = blocks::Stone;
                    else if (wy == h)
                        id = blocks::Grass;
                    else
                        id = blocks::Dirt;
                } else if (desert && wy >= h - 8) {
                    id = blocks::Sandstone; // the strata under the dunes
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

    placeTrees(chunk, pos); // trees even in coarse gen, cheap, and stops them popping in
    placeFoliage(chunk, pos, heights, desertW, m_version >= 2 ? coast : nullptr, rock);
    placeStructures(chunk, pos);
    if (m_version < 2)
        carveTestRoom(chunk, pos);
}

// Version 2's caves. Version 1 carved a band some 56 blocks under the surface and never broke it
// (a 4-block skin was left everywhere), so caves were shallow, sealed, and mostly not there at
// all. These run all the way down to kCaveFloor2:
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
    if (baseY + kChunkSize - 1 < kCaveFloor2)
        return;
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
    // squashes its blobs to about two thirds as tall as they are wide.
    m_cavern->GenUniformGrid3D(cav.data(), fx, fy * 1.6f, fz, kChunkSize, kChunkSize, kChunkSize, 1.0f, 1.6f,
                               1.0f, seedOff(m_seed, 221));
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
            const int aquifer = 12 + int(valueNoise(float(wx) * 0.008f, float(wz) * 0.008f, seedOff(m_seed, 227)) * 10.0f);
            // A ravine: where the path field is near zero inside the rare mask, cut down from the
            // surface by up to ~45, narrower the deeper it goes so its walls lean in. Rare: the
            // first version's mask covered a fifth of the land.
            const int i2 = x + z * kChunkSize;
            const float rm = smooth01(0.48f, 0.58f, ravMask[i2]);
            const int ravDepth = rm > 0.0f ? int(18.0f + 27.0f * rm) : 0;
            for (int y = 0; y < kChunkSize; ++y) {
                const int wy = baseY + y;
                if (wy < kCaveFloor2 || wy > h)
                    continue;
                const int depth = h - wy;
                const int cover = side - wy;
                if (wet && std::min(depth, cover) < 6)
                    continue;
                const std::size_t i = std::size_t(x) + std::size_t(y) * kChunkSize +
                                      std::size_t(z) * kChunkSize * kChunkSize;
                bool open = false;
                // caverns, never within 12 of the ground in any direction: there a big room is
                // just a crater, or a hole in a hillside
                if (cover >= 12) {
                    const int deep = std::min(depth, cover);
                    const float t = deep > 48 ? 0.42f : (deep > 24 ? 0.50f : 0.60f);
                    open = cav[i] > t;
                }
                // tunnels: the short pair, and the long one whose radius breathes; 4 of rock over
                // them on every side, except at a mouth
                if (!open && (cover >= 4 || mouth)) {
                    open = wa[i] * wa[i] + wb[i] * wb[i] < kWormR2;
                    if (!open) {
                        const float r2 = 0.05f + 0.16f * (tw[i] * 0.5f + 0.5f);
                        open = ta[i] * ta[i] + tb[i] * tb[i] < r2;
                    }
                }
                if (!open && depth < ravDepth && !wet) {
                    // a few blocks across at the top (0.045 cut gashes 20-30 wide), pinching in below
                    const float width = 0.016f * (1.0f - 0.7f * float(depth) / float(ravDepth));
                    open = std::abs(rav[i2]) < width;
                }
                if (open)
                    chunk.set(x, y, z, wy <= aquifer ? blocks::Water : blocks::Air);
            }
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
                    // the same warm-shallow / cold-deep hue rule version 1 uses
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
    if ((hsh % 100u) >= 45u)
        return none;
    const int ax = cx * kPyramidCell + 30 + int((hsh >> 8) % std::uint32_t(kPyramidCell - 60));
    const int az = cz * kPyramidCell + 30 + int((hsh >> 16) % std::uint32_t(kPyramidCell - 60));
    halfWidth = 11 + int((hsh >> 24) % 5u);
    // deep in the desert, all the way across the footprint, and near enough level that the base
    // course is not a cliff on one side
    int lo = 1 << 30, hi = -(1 << 30);
    for (const auto& o : {std::pair<int, int>{0, 0}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}}) {
        const Column c = column(ax + o.first * halfWidth, az + o.second * halfWidth);
        if (c.desert < 0.6f || c.h < kSeaLevel + 4)
            return none;
        lo = std::min(lo, c.h);
        hi = std::max(hi, c.h);
    }
    if (hi - lo > 9)
        return none;
    StructureSite s;
    s.kind = 4;
    s.x = ax;
    s.z = az;
    s.h = lo - 1; // sunk a little, so the dunes lap its lower courses
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
            if (baseY > y0 + R + 4 || baseY + kChunkSize - 1 < y0 - 10)
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
                        const int top = y0 + (R - m);
                        for (int wy = y0 - 3; wy <= top; ++wy)
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

    if (depth >= 3 && depth <= 20 && reefM > 0.58f) {
        const float strength = std::min((reefM - 0.58f) / 0.30f, 1.0f); // 0 at the rim, 1 at the heart
        // colonies are 4-block cells of one species, so a reef is patchwork, not confetti
        const int colony = int(cellHash(std::uint32_t(wx / 4) + std::uint32_t(m_seed), std::uint32_t(wz / 4) + 17u) % 5u);
        if (float(hh & 255u) < 40.0f + 110.0f * strength) {
            // never up through the surface: a coral head poking out of the sea reads as a
            // stray coloured tile floating on the waves
            const int height = std::min(1 + int((hh >> 8) % (1u + std::uint32_t(strength * 2.0f + 1.0f))),
                                        depth - 2);
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
void TerrainGenerator::placeFoliage(Chunk& chunk, ChunkPos pos, const int (*heights)[kChunkSize],
                                    const float (*desert)[kChunkSize], const float (*coast)[kChunkSize],
                                    const std::uint8_t (*rock)[kChunkSize]) const {
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
    // shipwreck: shallow seabed, not by a desert; a quarter of the candidates are left empty
    if (col.desert < 0.3f && g <= kSeaLevel - 3 && g >= kSeaLevel - 10) {
        site.kind = (rv & 3u) == 0u ? 0 : 1;
        return site;
    }
    // desert ruins, on half the candidates
    if (col.desert > 0.55f && g >= kSeaLevel + 6) {
        site.kind = (rv & 1u) == 0u ? 0 : 2;
        return site;
    }
    // lighthouse: a bit of low coast with open water not far off, on a third of the candidates
    if (col.desert < 0.3f && g >= kSeaLevel + 1 && g <= kSeaLevel + 8 && (rv >> 4) % 3u == 0u) {
        bool coast = false;
        for (const auto& d : {std::pair<int, int>{12, 0}, {-12, 0}, {0, 12}, {0, -12}})
            coast = coast || column(ax + d.first, az + d.second).h < kSeaLevel;
        site.kind = coast ? 3 : 0;
        return site;
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
        }
    placePyramids(chunk, pos);
}

void TerrainGenerator::placeTrees(Chunk& chunk, ChunkPos pos) const {
    const int baseX = pos.x * kChunkSize;
    const int baseY = pos.y * kChunkSize;
    const int baseZ = pos.z * kChunkSize;

    const auto put = [&](int wx, int wy, int wz, BlockId id) {
        const int lx = wx - baseX, ly = wy - baseY, lz = wz - baseZ;
        if (lx < 0 || lx >= kChunkSize || ly < 0 || ly >= kChunkSize || lz < 0 ||
            lz >= kChunkSize)
            return;
        const BlockId cur = chunk.get(lx, ly, lz);
        // wood overwrites air/leaves; leaves only fill air (so trunks stay solid)
        if (cur == blocks::Air || (blocks::isLogBlock(id) && blocks::isLeafBlock(cur)))
            chunk.set(lx, ly, lz, id);
    };

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
            const int surf = col.h;
            if (surf < kSeaLevel + 1)
                continue; // nothing grows underwater
            if (col.desert > 0.35f)
                continue; // and nothing tall in the desert
            // Version 2's mountains have a treeline: the woods thin out above 130 and stop by
            // about 175, leaving the upper slopes rock and snow.
            if (m_version >= 2 && surf > 130 && int((hsh >> 21) % 48u) < surf - 130)
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
            const float cold = smooth01(100.0f, 160.0f, float(surf));
            int species; // 0 spruce, 1 oak, 2 birch
            if (regionR < cold * 0.9f)
                species = 0;
            else
                species = (region >> 10) % 100u < 62u ? 1 : 2;
            if ((rv >> 17) % 6u == 0u) // a few strays keep a stand from looking planted
                species = int((rv >> 19) % 3u);
            const BlockId logId = species == 0 ? blocks::SpruceWood
                                  : (species == 2 ? blocks::BirchWood : blocks::Wood);
            const BlockId leafId = species == 0 ? blocks::SpruceLeaves
                                   : (species == 2 ? blocks::BirchLeaves : blocks::Leaves);
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
                // reads as vertical lines. Deliberately the sparsest of the three.
                const int trunk = 7 + int(rv % 4u); // 7..10
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
        }
}

} // namespace vx
