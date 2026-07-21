#include "gen/TerrainGenerator.hpp"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>

namespace vx {

namespace {

// continentalness -> base surface height, piecewise linear ("splines" per the spec —
// linear segments are all the control this terrain needs)
float heightSpline(float c) {
    static constexpr float kX[] = {-1.0f, -0.5f, 0.0f, 0.45f, 1.0f};
    static constexpr float kY[] = {52.0f, 62.0f, 72.0f, 88.0f, 108.0f};
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
    heightBase->SetScale(165.0f);
    auto fbm = FastNoise::New<FastNoise::FractalFBm>();
    fbm->SetSource(heightBase);
    fbm->SetOctaveCount(4);
    fbm->SetLacunarity(2.0f);
    fbm->SetGain(0.5f);
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
    return std::clamp(int(heightSpline(c) + h * 18.0f), 8, kWorldHeight - 8);
}

void TerrainGenerator::generate(Chunk& chunk, ChunkPos pos) const {
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
            const int h = std::clamp(int(heightSpline(c) + hn * 18.0f), 8,
                                     kWorldHeight - 8);
            heights[z][x] = h;
            hMin = std::min(hMin, h);
            hMax = std::max(hMax, h);
        }

    if (baseY > hMax) {
        chunk.makeUniform(blocks::Air);
        carveTestRoom(chunk, pos); // no-op unless the room overlaps
        return;
    }

    chunk.makeUniform(blocks::Air);
    for (int z = 0; z < kChunkSize; ++z)
        for (int x = 0; x < kChunkSize; ++x) {
            const int h = heights[z][x];
            const int top = std::min(h - baseY, kChunkSize - 1);
            for (int y = 0; y <= top; ++y) {
                const int wy = baseY + y;
                BlockId id = blocks::Stone;
                if (wy == h)
                    id = blocks::Grass;
                else if (wy >= h - 3)
                    id = blocks::Dirt;
                chunk.set(x, y, z, id);
            }
        }

    // caves: cheese blobs + ridged noodles, only meaningfully below the surface
    if (baseY <= hMax - 2) {
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
                    if (wy < 4 || wy > heights[z][x] - 4)
                        continue; // keep bedrock floor and the surface skin intact
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

    carveTestRoom(chunk, pos);
}

} // namespace vx
