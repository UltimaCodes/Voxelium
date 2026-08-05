#include "gen/TerrainGenerator.hpp"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>

namespace vx {

namespace {

// continentalness -> base surface height, piecewise linear ("splines" per the spec —
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
    // the exponential amplitude on top reaches ~314 — comfortably under the 376 ceiling so
    // a peak is shaped by the noise rather than sheared off by the clamp.
    static constexpr float kX[] = {-1.0f, -0.5f, 0.0f, 0.45f, 1.0f};
    static constexpr float kY[] = {50.0f, 64.0f, 82.0f, 130.0f, 240.0f};
    if (c <= kX[0])
        return kY[0];
    for (int i = 1; i < 5; ++i)
        if (c <= kX[i]) {
            const float t = (c - kX[i - 1]) / (kX[i] - kX[i - 1]);
            return kY[i - 1] + t * (kY[i] - kY[i - 1]);
        }
    return kY[4];
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

std::uint32_t cellHash(int x, int z) {
    std::uint32_t h = std::uint32_t(x) * 0x27D4EB2Fu ^ std::uint32_t(z) * 0x165667B1u;
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

constexpr int kSeaLevel = 62;   // air at or below this over the seafloor becomes water
constexpr int kCaveDepth = 56;  // caves only within this many blocks below the surface
// Squared radius of the worm tunnels around the curve where both worm fields vanish.
// This is the single knob for tunnel width, and it is sharp: the carved volume goes with
// the square of it, so small changes here move the cave-air count a long way.
constexpr float kWormR2 = 0.26f;
constexpr int kTreeCell = 7; // one tree candidate per 7x7 column cell
// Scan overlap. MUST be >= the widest horizontal reach of any species or trees get sliced
// off at chunk borders — a tree is only placed by chunks that bother to consider its cell.
// Worst case is an oak: branch length 3 plus a radius-2 leaf blob on the end.
constexpr int kTreeCanopyR = 6;
constexpr int kSnowLine = 205;  // peaks at/above this get snow (jittered per column)

// glow-lit test room kept from M9 as the lighting regression anchor; a 1-block stone
// shell is re-solidified around it so no cave can crack it open
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

TerrainGenerator::TerrainGenerator(int seed) : m_seed(seed) {
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

    auto cheeseBase = FastNoise::New<FastNoise::Simplex>();
    cheeseBase->SetScale(45.0f);
    auto cheese = FastNoise::New<FastNoise::FractalFBm>();
    cheese->SetSource(cheeseBase);
    cheese->SetOctaveCount(3);
    m_cheese = cheese;

    // Two independent low-frequency fields. Low because the tunnel a player walks is the
    // zero-set of these, so the scale here is the length of a corridor between bends, not
    // the width of it — width comes from the carve radius below.
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
}

int TerrainGenerator::surfaceHeight(int wx, int wz) const {
    float h = 0.0f;
    float c = 0.0f;
    m_height->GenUniformGrid2D(&h, float(wx), float(wz), 1, 1, 1.0f, 1.0f, m_seed);
    m_continent->GenUniformGrid2D(&c, float(wx), float(wz), 1, 1, 1.0f, 1.0f, m_seed + 1);
    return std::clamp(int(heightSpline(c) + h * reliefAmplitude(c)), 8, kWorldHeight - 8);
}

void TerrainGenerator::generate(Chunk& chunk, ChunkPos pos, bool detailed) const {
    chunk.fillLight(0); // pooled chunks carry the previous tenant's light
    const int baseY = pos.y * kChunkSize;
    const int baseX = pos.x * kChunkSize;
    const int baseZ = pos.z * kChunkSize;

    float heightNoise[kChunkSize * kChunkSize];
    float contNoise[kChunkSize * kChunkSize];
    m_height->GenUniformGrid2D(heightNoise, float(baseX), float(baseZ), kChunkSize,
                               kChunkSize, 1.0f, 1.0f, m_seed);
    m_continent->GenUniformGrid2D(contNoise, float(baseX), float(baseZ), kChunkSize,
                                  kChunkSize, 1.0f, 1.0f, m_seed + 1);

    int heights[kChunkSize][kChunkSize];
    int hMin = 1 << 30;
    int hMax = -(1 << 30);
    for (int z = 0; z < kChunkSize; ++z)
        for (int x = 0; x < kChunkSize; ++x) {
            // grid order: x + z * kChunkSize (x start axis first)
            const float hn = heightNoise[x + z * kChunkSize];
            const float c = contNoise[x + z * kChunkSize];
            const int h = std::clamp(int(heightSpline(c) + hn * reliefAmplitude(c)), 8,
                                     kWorldHeight - 8);
            heights[z][x] = h;
            hMin = std::min(hMin, h);
            hMax = std::max(hMax, h);
        }

    // wholly above both terrain and sea: air, except trees can still reach up into it
    if (baseY > hMax && baseY > kSeaLevel) {
        chunk.makeUniform(blocks::Air);
        placeTrees(chunk, pos);
        carveTestRoom(chunk, pos); // no-op unless the room overlaps
        return;
    }

    // deep bedrock: caves only reach ~56 blocks below the surface, so a chunk entirely
    // below that band is solid stone — makeUniform costs nothing (no palette/index array,
    // no quads). this is what keeps a 256-block world as cheap as the old 128.
    if (baseY + kChunkSize - 1 < hMin - kCaveDepth) {
        chunk.makeUniform(blocks::Stone);
        carveTestRoom(chunk, pos);
        return;
    }

    chunk.makeUniform(blocks::Air);
    for (int z = 0; z < kChunkSize; ++z)
        for (int x = 0; x < kChunkSize; ++x) {
            const int h = heights[z][x];
            const bool coastal = h <= kSeaLevel + 1; // beaches and seabed are sand
            // ragged snow line so peaks cap in snow without a razor-straight edge
            const int snowLine = kSnowLine + int(cellHash(baseX + x, baseZ + z) & 7u) - 3;
            const bool snowy = !coastal && h >= snowLine;
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
                    // opened by a cave collapses instead of holding a clean roof — which is
                    // the point of it falling at all.
                    else if (depth > 5 && depth < 34 && roll >= 38u && roll < 62u)
                        id = blocks::Gravel;
                }
                if (wy >= h - 3) {
                    if (coastal)
                        id = blocks::Sand;
                    else if (snowy && wy >= h - 1)
                        id = blocks::Snow; // a couple blocks of cap
                    else if (wy == h)
                        id = blocks::Grass;
                    else
                        id = blocks::Dirt;
                }
                chunk.set(x, y, z, id);
            }
        }

    // caves: cheese blobs + ridged noodles, only meaningfully below the surface. skipped
    // for coarse (far) chunks — the LOD downsample collapses caves to solid anyway, and
    // the 3d noise + carve is by far the most expensive part of gen.
    if (detailed && baseY <= hMax - 2) {
        static thread_local std::vector<float> cheeseBuf;
        static thread_local std::vector<float> wormABuf;
        static thread_local std::vector<float> wormBBuf;
        cheeseBuf.resize(std::size_t(kChunkSize3));
        wormABuf.resize(std::size_t(kChunkSize3));
        wormBBuf.resize(std::size_t(kChunkSize3));
        // grid order: x + y * size + z * size * size
        m_cheese->GenUniformGrid3D(cheeseBuf.data(), float(baseX), float(baseY), float(baseZ),
                                   kChunkSize, kChunkSize, kChunkSize, 1.0f, 1.0f, 1.0f,
                                   m_seed + 2);
        m_wormA->GenUniformGrid3D(wormABuf.data(), float(baseX), float(baseY), float(baseZ),
                                  kChunkSize, kChunkSize, kChunkSize, 1.0f, 1.0f, 1.0f,
                                  m_seed + 3);
        m_wormB->GenUniformGrid3D(wormBBuf.data(), float(baseX), float(baseY), float(baseZ),
                                  kChunkSize, kChunkSize, kChunkSize, 1.0f, 1.0f, 1.0f,
                                  m_seed + 4);
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
                        // edge voxel live in a chunk that may not exist yet — reading them
                        // would make the crystal's colour depend on which chunk generated
                        // it first, which is the one thing a deferred structure cannot do.
                        const int depth = heights[z][x] - (baseY + y);
                        const std::uint32_t j = voxelHash(baseX + x, 3, baseZ + z) >> 7;
                        int hue = depth > 44 ? 4 : (depth > 30 ? 3 : (depth > 18 ? 2 : 1));
                        if ((j & 7u) == 0u)
                            hue = int(j >> 3) % 6;
                        chunk.set(x, y, z, BlockId(blocks::LampRed + hue));
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
        }

    placeTrees(chunk, pos); // trees even in coarse gen — cheap, and stops them popping in
    carveTestRoom(chunk, pos);
}

// Deferred structure placement: scans every tree candidate cell that could reach into
// this chunk (including neighbors, hence kTreeCanopyR of overlap) and places the parts
// that fall inside it. Fully deterministic from the seed — the same tree is computed
// identically by every chunk it touches, so trunks/canopies cross borders seamlessly
// without any neighbor lookups.
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
        if (cur == blocks::Air || (id == blocks::Wood && cur == blocks::Leaves))
            chunk.set(lx, ly, lz, id);
    };

    const int cx0 = floorDivI(baseX - kTreeCanopyR, kTreeCell);
    const int cx1 = floorDivI(baseX + kChunkSize - 1 + kTreeCanopyR, kTreeCell);
    const int cz0 = floorDivI(baseZ - kTreeCanopyR, kTreeCell);
    const int cz1 = floorDivI(baseZ + kChunkSize - 1 + kTreeCanopyR, kTreeCell);

    for (int cz = cz0; cz <= cz1; ++cz)
        for (int cx = cx0; cx <= cx1; ++cx) {
            const std::uint32_t hsh = cellHash(cx * 1013 + m_seed, cz * 79 + m_seed);
            if ((hsh & 255u) > 110u)
                continue; // ~43% of cells grow a tree
            const int tx = cx * kTreeCell + int((hsh >> 8) % kTreeCell);
            const int tz = cz * kTreeCell + int((hsh >> 13) % kTreeCell);
            const int surf = surfaceHeight(tx, tz);
            if (surf < kSeaLevel + 2)
                continue; // no trees on beaches or underwater

            // per-tree variation: an independent hash drives size and shape. everything
            // below is a pure function of the tree's world position, which is what lets
            // every chunk it overlaps compute the identical tree with no neighbour lookups.
            const std::uint32_t rv = voxelHash(tx, 7, tz);
            const auto leaf = [&](int wx, int wy, int wz, int holePerMille) {
                if (int(voxelHash(wx, wy, wz) % 1000) >= holePerMille)
                    put(wx, wy, wz, blocks::Leaves);
            };
            // A clump of leaves centred anywhere — the piece that lets a canopy be built
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
                    put(ex, ey, ez, blocks::Wood);
                }
                return glm::ivec3{ex, ey, ez};
            };

            // Species comes from a coarse region hash, not per tree, so the world grows
            // pine stands and oak groves instead of salt-and-pepper mixing every cell.
            const std::uint32_t region = cellHash((cx >> 3) * 31 + m_seed, (cz >> 3) * 17);
            int species = int(region % 3u);
            if ((rv >> 17) % 5u == 0u) // a few strays keep a stand from looking planted
                species = int((rv >> 19) % 3u);

            if (species == 0) {
                // PINE: tall bare lower trunk, then narrow tiers with GAPS between them.
                // The gaps are the whole silhouette — a continuous cone just reads as a
                // green triangle, which is what the old spruce was.
                const int trunk = 9 + int(rv % 6u); // 9..14
                for (int i = 1; i <= trunk; ++i)
                    put(tx, surf + i, tz, blocks::Wood);
                put(tx, surf + trunk + 1, tz, blocks::Leaves); // tip
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
                // own clump, so the canopy is the union of off-centre parts — lumpy and
                // asymmetric by construction rather than by punching holes in a ball.
                const int trunk = 5 + int(rv % 3u); // 5..7
                for (int i = 1; i <= trunk; ++i)
                    put(tx, surf + i, tz, blocks::Wood);
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
                    put(tx, surf + i, tz, blocks::Wood);
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
