#include "render/FarTerrain.hpp"

#include "core/Log.hpp"
#include "gen/TerrainGenerator.hpp"
#include "gfx/Shader.hpp"
#include "jobs/JobSystem.hpp"

#include <glad/gl.h>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>

namespace vx {

namespace {
// Rebuild once the camera is this far from the centre of what was built. Well inside what either
// grid covers, so the drift never shows and rebuilding is rare.
constexpr double kRebuildDistance[FarTerrain::kRings] = {512.0, 1536.0};

float sstep(float a, float b, float x) {
    const float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
std::uint32_t hash2(int x, int z) {
    std::uint32_t h = std::uint32_t(x) * 0x27D4EB2Fu ^ std::uint32_t(z) * 0x165667B1u;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    return h;
}
} // namespace

bool FarTerrain::init(const std::filesystem::path& shaderDir) {
    m_prog = loadProgram(shaderDir / "farterrain.vert", shaderDir / "farterrain.frag");
    if (!m_prog)
        return false;
    glCreateVertexArrays(1, &m_vao);
    for (int r = 0; r < kRings; ++r) {
        const int n = kCells[r] + 1;
        Ring& ring = m_rings[r];
        glCreateBuffers(1, &ring.ssbo);
        glNamedBufferStorage(ring.ssbo, GLsizeiptr(n) * n * GLsizeiptr(sizeof(Vertex)), nullptr, GL_DYNAMIC_STORAGE_BIT);
        glCreateTextures(GL_TEXTURE_2D, 1, &ring.heightTex);
        glTextureStorage2D(ring.heightTex, 1, GL_R32F, n, n);
        glTextureParameteri(ring.heightTex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTextureParameteri(ring.heightTex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTextureParameteri(ring.heightTex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(ring.heightTex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        if (r == 0) {
            glCreateTextures(GL_TEXTURE_2D, 1, &ring.coastTex);
            glTextureStorage2D(ring.coastTex, 1, GL_R16F, n, n);
            glTextureParameteri(ring.coastTex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTextureParameteri(ring.coastTex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTextureParameteri(ring.coastTex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTextureParameteri(ring.coastTex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
    }
    glCreateTextures(GL_TEXTURE_2D, 1, &m_climateTex);
    glTextureStorage2D(m_climateTex, 1, GL_RG8, kCells[1] + 1, kCells[1] + 1);
    glTextureParameteri(m_climateTex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(m_climateTex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(m_climateTex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(m_climateTex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return true;
}

// A build reads the terrain generator and writes into this object, so neither may change or go
// away underneath it. m_building is not the right thing to wait on: it stays set after the job
// has finished until update() consumes the result, so waiting on it hangs if nobody ever does
// (which is exactly the situation at shutdown).
void FarTerrain::waitIdle() {
    const auto start = std::chrono::steady_clock::now();
    for (Ring& ring : m_rings)
        while (ring.jobRunning.load(std::memory_order_acquire)) {
            if (std::chrono::steady_clock::now() - start > std::chrono::seconds(10)) {
                VX_ERROR("far terrain: build did not finish in 10 s, not waiting any longer");
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
}

void FarTerrain::shutdown() {
    waitIdle();
    if (m_prog)
        glDeleteProgram(m_prog);
    for (Ring& ring : m_rings) {
        glDeleteBuffers(1, &ring.ssbo);
        glDeleteTextures(1, &ring.heightTex);
        glDeleteTextures(1, &ring.coastTex);
        ring.ssbo = ring.heightTex = ring.coastTex = 0;
        ring.haveMesh = false;
    }
    glDeleteTextures(1, &m_climateTex);
    if (m_vao)
        glDeleteVertexArrays(1, &m_vao);
    m_prog = m_vao = m_climateTex = 0;
}

void FarTerrain::reset() {
    // The caller is about to replace the generator (a new world), which a running build must not
    // be reading while it happens.
    waitIdle();
    for (Ring& ring : m_rings) {
        ring.building.store(false, std::memory_order_release);
        ring.haveMesh = false;
        ring.centre = glm::dvec2(1e18);
        ring.stagingReady.store(false, std::memory_order_release);
    }
    m_climateXf = glm::vec3(0.0f);
    ++m_seedEpoch;
}

// One vertex per grid point: its height, the colour it would have from above, a normal from the
// slope and how shut in it is. Colour is decided here so the shader is only lighting.
void FarTerrain::build(const TerrainGenerator& gen, int r, std::int64_t originX, std::int64_t originZ, JobSystem* jobs) {
    constexpr int kSea = kSeaLevelForSpawn;
    const int n = kCells[r] + 1, spacing = kSpacing[r];
    const std::size_t nn = std::size_t(n) * n;
    Ring& ring = m_rings[r];
    ring.staging.assign(nn, Vertex{});
    ring.stagingHeights.resize(nn);
    std::vector<TerrainGenerator::FarSample> sample(nn);
    gen.farGrid(int(originX), int(originZ), n, spacing, sample.data(), jobs);
    std::vector<float>& height = ring.stagingHeights;
    // the ground as the shadow march sees it: trees stand on it, the sea is flat
    for (std::size_t i = 0; i < nn; ++i)
        height[i] = float(std::max(sample[i].h, kSea));
    const auto at = [&](int i, int j) {
        return height[std::size_t(std::clamp(j, 0, n - 1)) * n + std::size_t(std::clamp(i, 0, n - 1))];
    };
    const auto q8 = [](float v) { return std::uint8_t(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f); };
    // The same display colours as the tiles the near land is built from.
    const glm::vec3 kGrass(0.38f, 0.54f, 0.27f), kCanopy(0.20f, 0.33f, 0.13f), kSand(0.79f, 0.73f, 0.55f);
    const glm::vec3 kRock(0.50f, 0.50f, 0.51f), kSnow(0.92f, 0.94f, 0.96f);
    // each biome's own ground and trees, so the horizon is the same country the chunks fill in
    const glm::vec3 kNeedles(0.42f, 0.33f, 0.24f), kSavanna(0.62f, 0.58f, 0.33f), kJungleFloor(0.36f, 0.30f, 0.18f);
    const glm::vec3 kSwamp(0.33f, 0.38f, 0.24f), kRedSand(0.74f, 0.45f, 0.27f), kClay(0.66f, 0.40f, 0.28f);
    const glm::vec3 kSpruce(0.15f, 0.26f, 0.17f), kAcacia(0.30f, 0.38f, 0.16f), kJungleLeaf(0.18f, 0.37f, 0.10f);
    const glm::vec3 kBirchLeaf(0.30f, 0.44f, 0.18f);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const auto& s = sample[std::size_t(j) * n + i];
            Vertex& v = ring.staging[std::size_t(j) * n + i];
            const int wx = int(originX + std::int64_t(i) * spacing);
            const int wz = int(originZ + std::int64_t(j) * spacing);
            const float h = float(s.h);
            const float jit = float(hash2(wx, wz) & 255u) / 255.0f - 0.5f;
            const float gx = (at(i + 1, j) - at(i - 1, j)) / float(2 * spacing);
            const float gz = (at(i, j + 1) - at(i, j - 1)) / float(2 * spacing);
            const float slope = std::sqrt(gx * gx + gz * gz);
            glm::vec3 col;
            const bool inland = s.water > s.h;
            const bool water = s.h < kSea || inland;
            if (water) {
                // the sea floor tints the water: how deep it is, as the near water works it out
                const float depth = inland ? float(s.water - s.h) + 2.0f : float(kSea) - h;
                col = glm::vec3(std::min(depth / 40.0f, 1.0f), 0.0f, 0.0f);
            } else {
                // what a dry or wet climate does to green, the far twin of climateTint in lighting.glsl
                const float dry = sstep(0.45f, 0.2f, s.moisture), wet = sstep(0.5f, 0.85f, s.moisture);
                const float cold = sstep(0.42f, 0.22f, s.temp);
                glm::vec3 tint = glm::mix(glm::mix(glm::vec3(1.0f), glm::vec3(0.78f, 0.92f, 0.67f), wet),
                                          glm::vec3(1.40f, 1.10f, 1.07f), dry);
                tint = glm::mix(tint, glm::vec3(0.87f, 0.88f, 1.22f), cold * 0.8f);
                glm::vec3 ground = kGrass * tint, canopy = kCanopy * tint, sand = kSand;
                switch (s.biome) {
                    case Biome::Taiga: ground = glm::mix(ground, kNeedles, 0.65f), canopy = kSpruce; break;
                    case Biome::SnowyTaiga: ground = glm::mix(kNeedles, kSnow, 0.8f), canopy = kSpruce; break;
                    case Biome::Tundra: ground = glm::mix(kNeedles * 0.8f, kSnow, 0.7f); break;
                    case Biome::Savanna: ground = kSavanna, canopy = kAcacia; break;
                    case Biome::Jungle: ground = glm::mix(ground, kJungleFloor, 0.4f), canopy = kJungleLeaf; break;
                    case Biome::Swamp: ground = kSwamp, canopy = kCanopy * 0.85f; break;
                    case Biome::BirchForest: canopy = kBirchLeaf * tint; break;
                    case Biome::RedDesert: sand = kRedSand; break;
                    case Biome::Badlands: sand = kClay; break;
                    default: break;
                }
                col = glm::mix(ground, canopy, s.forest * 0.9f);
                col *= 1.0f + 0.06f * jit;
                const float highRock = sstep(float(s.snowLine) - 56.0f, float(s.snowLine) - 26.0f, h);
                col = glm::mix(col, kRock * (1.0f + 0.10f * jit), std::max(sstep(0.70f, 1.15f, slope), highRock));
                col = glm::mix(col, kSnow, sstep(float(s.snowLine) - 4.0f, float(s.snowLine) + 10.0f, h) * (1.0f - sstep(1.5f, 2.2f, slope)));
                col = glm::mix(col, sand * (1.0f + 0.05f * jit), sstep(0.30f, 0.55f, s.desert));
                if (h <= float(kSea + 3) && s.coast < 45.0f)
                    col = kSand;
            }
            // A valley sits under what is around it. Two reaches, so a gully and a whole valley both show.
            float around = 0.0f, aroundFar = 0.0f;
            for (int k = 0; k < 8; ++k) {
                static constexpr int kDX[8] = {1, -1, 0, 0, 1, 1, -1, -1};
                static constexpr int kDZ[8] = {0, 0, 1, -1, 1, -1, 1, -1};
                around += at(i + kDX[k] * 2, j + kDZ[k] * 2);
                aroundFar += at(i + kDX[k] * 8, j + kDZ[k] * 8);
            }
            const float sunk = std::max(around / 8.0f - h, 0.0f) / float(2 * spacing);
            const float sunkFar = std::max(aroundFar / 8.0f - h, 0.0f) / float(8 * spacing);
            const float ao = water ? 1.0f : 1.0f / (1.0f + sunk * 2.2f + sunkFar * 2.6f);

            // canopy stands a little off the ground, and sits under the chunks' real trees at the seam
            v.h = water ? float(inland ? s.water : kSea) + 0.6f : h + 6.0f * s.forest;
            v.r = q8(col.r);
            v.g = q8(col.g);
            v.b = q8(col.b);
            v.water = water ? 1 : 0;
            v.ao = q8(ao);
            glm::vec3 nrm(0.0f, 1.0f, 0.0f);
            if (!water)
                nrm = glm::normalize(glm::vec3(-gx, 1.0f, -gz));
            v.nx = std::int8_t(std::lround(nrm.x * 127.0f));
            v.ny = std::int8_t(std::lround(nrm.y * 127.0f));
            v.nz = std::int8_t(std::lround(nrm.z * 127.0f));
        }
    if (r == 0) {
        ring.stagingCoast.resize(nn);
        for (std::size_t i = 0; i < nn; ++i)
            ring.stagingCoast[i] = sample[i].coast;
    }
    if (r == kRings - 1) {
        ring.stagingClimate.resize(nn * 2);
        for (std::size_t i = 0; i < nn; ++i) {
            ring.stagingClimate[i * 2] = q8(sample[i].moisture);
            ring.stagingClimate[i * 2 + 1] = q8(sample[i].temp);
        }
    }
}

void FarTerrain::update(const TerrainGenerator& gen, JobSystem& jobs, const glm::dvec3& cam) {
    for (int r = 0; r < kRings; ++r) {
        Ring& ring = m_rings[r];
        const int n = kCells[r] + 1;
        const double half = double(kCells[r] * kSpacing[r]) * 0.5;
        if (ring.stagingReady.load(std::memory_order_acquire)) {
            // a result from before a reset() belongs to another world: drop it
            if (ring.jobEpoch.load(std::memory_order_relaxed) == m_seedEpoch) {
                glNamedBufferSubData(ring.ssbo, 0, GLsizeiptr(ring.staging.size() * sizeof(Vertex)), ring.staging.data());
                glTextureSubImage2D(ring.heightTex, 0, 0, 0, n, n, GL_RED, GL_FLOAT, ring.stagingHeights.data());
                if (ring.coastTex)
                    glTextureSubImage2D(ring.coastTex, 0, 0, 0, n, n, GL_RED, GL_FLOAT, ring.stagingCoast.data());
                ring.origin = ring.stagingOrigin;
                ring.centre = ring.stagingOrigin + glm::dvec2(half);
                ring.haveMesh = true;
                if (r == kRings - 1) {
                    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                    glTextureSubImage2D(m_climateTex, 0, 0, 0, n, n, GL_RG, GL_UNSIGNED_BYTE, ring.stagingClimate.data());
                    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
                    m_climateXf = {float(ring.origin.x), float(ring.origin.y), 1.0f / float(kCells[r] * kSpacing[r])};
                }
            }
            ring.stagingReady.store(false, std::memory_order_release);
            ring.building.store(false, std::memory_order_release);
        }
        if (ring.building.load(std::memory_order_acquire))
            continue;
        const glm::dvec2 cxz(cam.x, cam.z);
        if (ring.haveMesh && glm::length(cxz - ring.centre) < kRebuildDistance[r])
            continue;

        // grid corner snapped to the spacing, so the same vertices are sampled however the camera
        // wandered in and the terrain never shimmers between rebuilds
        const std::int64_t ox = std::int64_t(std::floor(cam.x / kSpacing[r])) * kSpacing[r] - std::int64_t(half);
        const std::int64_t oz = std::int64_t(std::floor(cam.z / kSpacing[r])) * kSpacing[r] - std::int64_t(half);
        ring.building.store(true, std::memory_order_release);
        ring.jobRunning.store(true, std::memory_order_release);
        ring.jobEpoch.store(m_seedEpoch, std::memory_order_relaxed);
        const TerrainGenerator* g = &gen;
        JobSystem* js = &jobs;
        jobs.submit([this, g, js, r, ox, oz] {
            build(*g, r, ox, oz, js);
            Ring& done = m_rings[r];
            done.stagingOrigin = glm::dvec2(double(ox), double(oz));
            done.stagingReady.store(true, std::memory_order_release);
            done.jobRunning.store(false, std::memory_order_release); // last: nothing here after this
        });
    }
}

void FarTerrain::draw(const glm::mat4& viewProj, const glm::dvec3& cam, const glm::vec3& sunL,
                      const glm::vec3& sunColor, const glm::vec3& skyAmbient, const glm::vec3& landSun,
                      const glm::vec3& landAmbient, const glm::vec3& waterLevel, float cloudShadow,
                      float cloudTime, float cloudCover, float innerRadius) {
    if (m_prog == 0)
        return;
    glUseProgram(m_prog);
    glProgramUniformMatrix4fv(m_prog, 0, 1, GL_FALSE, glm::value_ptr(viewProj));
    glProgramUniform1f(m_prog, 5, float(cam.y));
    glProgramUniform3fv(m_prog, 7, 1, &sunL.x);
    glProgramUniform3fv(m_prog, 8, 1, &sunColor.x);
    glProgramUniform3fv(m_prog, 9, 1, &skyAmbient.x);
    glProgramUniform3fv(m_prog, 13, 1, &landSun.x);
    glProgramUniform3fv(m_prog, 14, 1, &landAmbient.x);
    const glm::vec3 camF(cam);
    glProgramUniform3fv(m_prog, 15, 1, &camF.x);
    glProgramUniform1f(m_prog, 16, cloudShadow);
    glProgramUniform3fv(m_prog, 17, 1, &waterLevel.x);
    glProgramUniform1f(m_prog, 31, cloudTime);
    glProgramUniform1f(m_prog, 33, cloudCover);
    glBindVertexArray(m_vao);
    glDisable(GL_CULL_FACE); // one sheet seen from either side; winding is not worth a bug
    for (int r = 0; r < kRings; ++r) {
        const Ring& ring = m_rings[r];
        if (!ring.haveMesh)
            continue;
        const float span = float(kCells[r] * kSpacing[r]);
        const glm::vec3 originRel(float(ring.origin.x - cam.x), 0.0f, float(ring.origin.y - cam.z));
        glProgramUniform3fv(m_prog, 4, 1, &originRel.x);
        glProgramUniform1f(m_prog, 6, innerRadius);
        glProgramUniform1f(m_prog, 10, float(kSpacing[r]));
        glProgramUniform2f(m_prog, 12, float(ring.origin.x), float(ring.origin.y));
        glProgramUniform1i(m_prog, 11, kCells[r]);
        // the outer ring leaves a hole where the inner one is drawn
        const Ring& in = m_rings[0];
        const bool hole = r > 0 && in.haveMesh;
        const float inHalf = float(kCells[0] * kSpacing[0]) * 0.5f;
        glProgramUniform3f(m_prog, 18, hole ? float(in.centre.x) : 0.0f, hole ? float(in.centre.y) : 0.0f,
                           hole ? inHalf - 96.0f : 0.0f);
        glProgramUniform3f(m_prog, 19, float(ring.origin.x), float(ring.origin.y), 1.0f / span);
        glBindTextureUnit(8, ring.heightTex);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ring.ssbo);
        glDrawArrays(GL_TRIANGLES, 0, kCells[r] * kCells[r] * 6);
    }
    glEnable(GL_CULL_FACE);
}

} // namespace vx
