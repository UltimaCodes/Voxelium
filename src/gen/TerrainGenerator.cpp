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
constexpr int kTreeCell = 7;    // one tree candidate per 7x7 column cell
constexpr int kTreeCanopyR = 4; // scan overlap: >= the widest canopy so no tree is missed
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

    auto noodleBase = FastNoise::New<FastNoise::Simplex>();
    noodleBase->SetScale(71.0f);
    auto ridged = FastNoise::New<FastNoise::FractalRidged>();
    ridged->SetSource(noodleBase);
    ridged->SetOctaveCount(2);
    m_noodle = ridged;
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
        static thread_local std::vector<float> noodleBuf;
        cheeseBuf.resize(std::size_t(kChunkSize3));
        noodleBuf.resize(std::size_t(kChunkSize3));
        // grid order: x + y * size + z * size * size
        m_cheese->GenUniformGrid3D(cheeseBuf.data(), float(baseX), float(baseY), float(baseZ),
                                   kChunkSize, kChunkSize, kChunkSize, 1.0f, 1.0f, 1.0f,
                                   m_seed + 2);
        m_noodle->GenUniformGrid3D(noodleBuf.data(), float(baseX), float(baseY), float(baseZ),
                                   kChunkSize, kChunkSize, kChunkSize, 1.0f, 1.0f, 1.0f,
                                   m_seed + 3);
        for (int z = 0; z < kChunkSize; ++z)
            for (int y = 0; y < kChunkSize; ++y)
                for (int x = 0; x < kChunkSize; ++x) {
                    const int wy = baseY + y;
                    if (wy < 4 || wy > heights[z][x] - 4 || wy < heights[z][x] - kCaveDepth)
                        continue; // skip the skin, the bedrock floor, and deep stone
                    const std::size_t i =
                        std::size_t(x) + std::size_t(y) * kChunkSize +
                        std::size_t(z) * kChunkSize * kChunkSize;
                    if (cheeseBuf[i] > 0.56f || noodleBuf[i] > 0.90f)
                        chunk.set(x, y, z, blocks::Air);
                }

        // glow deposits on cave floors, ~1 per few hundred floor cells
        for (int z = 0; z < kChunkSize; ++z)
            for (int y = 1; y < kChunkSize; ++y)
                for (int x = 0; x < kChunkSize; ++x)
                    if (chunk.get(x, y, z) == blocks::Air &&
                        chunk.get(x, y - 1, z) == blocks::Stone &&
                        (voxelHash(baseX + x, baseY + y, baseZ + z) & 511u) == 0u)
                        chunk.set(x, y, z, blocks::Glow);
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

            // per-tree variation: an independent hash drives size/shape/raggedness, and a
            // per-voxel hash punches holes in the canopy so no two trees are the same and
            // none is a solid blob. all deterministic -> still seamless across chunk borders.
            const std::uint32_t rv = voxelHash(tx, 7, tz);
            const auto leaf = [&](int wx, int wy, int wz, int holePerMille) {
                if (int(voxelHash(wx, wy, wz) % 1000) >= holePerMille)
                    put(wx, wy, wz, blocks::Leaves);
            };

            if (hsh & (1u << 22)) {
                // SPRUCE: tall, tiered conical needles; height + width jittered, gappy
                const int trunk = 7 + int(rv % 6u);       // 7..12
                const int maxR = 2 + int((rv >> 4) & 1u); // 2..3
                for (int i = 1; i <= trunk; ++i)
                    put(tx, surf + i, tz, blocks::Wood);
                put(tx, surf + trunk + 1, tz, blocks::Leaves); // tip
                const int leafTop = surf + trunk;
                const int leafBot = surf + 2 + int((rv >> 8) & 1u);
                for (int wy = leafTop; wy >= leafBot; --wy) {
                    int r = (leafTop - wy) / 2; // widening cone, a tier every two rows
                    if (r > maxR)
                        r = maxR;
                    for (int dz = -r; dz <= r; ++dz)
                        for (int dx = -r; dx <= r; ++dx) {
                            if (dx == 0 && dz == 0)
                                continue; // keep the trunk clear
                            if (dx * dx + dz * dz <= r * r + 1)
                                leaf(tx + dx, wy, tz + dz, 250);
                        }
                }
            } else {
                // OAK: bushy rounded canopy; radius, height + edge raggedness jittered
                const int trunk = 4 + int(rv % 4u);      // 4..7
                const int cr = 2 + int((rv >> 3) & 1u);  // canopy radius 2..3
                const int ch = 1 + int((rv >> 6) & 1u);  // vertical half-extent 1..2
                const int topY = surf + trunk;
                for (int i = 1; i <= trunk; ++i)
                    put(tx, surf + i, tz, blocks::Wood);
                const float edge = float(cr) + 0.4f;
                const float inner = float(cr - 1) * float(cr - 1);
                for (int dy = -ch; dy <= ch + 1; ++dy)
                    for (int dz = -cr; dz <= cr; ++dz)
                        for (int dx = -cr; dx <= cr; ++dx) {
                            const float rr = float(dx * dx + dz * dz) + float(dy * dy) * 1.5f;
                            if (rr <= edge * edge)
                                leaf(tx + dx, topY + dy, tz + dz, rr > inner ? 430 : 80);
                        }
            }
        }
}

} // namespace vx
