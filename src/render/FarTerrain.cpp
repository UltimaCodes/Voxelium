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
constexpr int kVerts1 = FarTerrain::kCells + 1;
// Rebuild once the camera is this far from the centre of what was built. The grid reaches
// kHalfExtent (5120) each way and fog closes in well before that, so a few hundred blocks of
// drift never shows and rebuilding is rare.
constexpr double kRebuildDistance = 640.0;

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
    glCreateBuffers(1, &m_ssbo);
    glNamedBufferStorage(m_ssbo, GLsizeiptr(kVerts1) * kVerts1 * GLsizeiptr(sizeof(Vertex)),
                         nullptr, GL_DYNAMIC_STORAGE_BIT);
    return true;
}

// A build reads the terrain generator and writes into this object, so neither may change or go
// away underneath it. m_building is not the right thing to wait on: it stays set after the job
// has finished until update() consumes the result, so waiting on it hangs if nobody ever does
// (which is exactly the situation at shutdown).
void FarTerrain::waitIdle() {
    const auto start = std::chrono::steady_clock::now();
    while (m_jobRunning.load(std::memory_order_acquire)) {
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
    if (m_ssbo)
        glDeleteBuffers(1, &m_ssbo);
    if (m_vao)
        glDeleteVertexArrays(1, &m_vao);
    m_prog = m_ssbo = m_vao = 0;
    m_haveMesh = false;
}

void FarTerrain::reset() {
    // The caller is about to replace the generator (a new world), which a running build must not
    // be reading while it happens.
    waitIdle();
    m_building.store(false, std::memory_order_release);
    m_haveMesh = false;
    m_centre = glm::dvec2(1e18);
    m_stagingReady.store(false, std::memory_order_release);
    ++m_seedEpoch;
}

// One vertex per grid point: its height, the colour it would have from above, and a normal from
// the slope. Colour is decided here so the shader is only lighting.
void FarTerrain::build(const TerrainGenerator& gen, std::int64_t originX, std::int64_t originZ,
                       std::vector<Vertex>& out) const {
    constexpr int kSea = kSeaLevelForSpawn;
    out.assign(std::size_t(kVerts1) * kVerts1, Vertex{});
    std::vector<float> height(std::size_t(kVerts1) * kVerts1);
    std::vector<TerrainGenerator::FarSample> sample(std::size_t(kVerts1) * kVerts1);
    for (int j = 0; j < kVerts1; ++j)
        for (int i = 0; i < kVerts1; ++i) {
            const int wx = int(originX + std::int64_t(i) * kSpacing);
            const int wz = int(originZ + std::int64_t(j) * kSpacing);
            const auto s = gen.farSample(wx, wz);
            sample[std::size_t(j) * kVerts1 + i] = s;
            height[std::size_t(j) * kVerts1 + i] = float(s.h);
        }
    const auto at = [&](int i, int j) {
        return height[std::size_t(std::clamp(j, 0, kVerts1 - 1)) * kVerts1 +
                      std::size_t(std::clamp(i, 0, kVerts1 - 1))];
    };
    const auto q8 = [](float v) { return std::uint8_t(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f); };
    for (int j = 0; j < kVerts1; ++j)
        for (int i = 0; i < kVerts1; ++i) {
            const auto& s = sample[std::size_t(j) * kVerts1 + i];
            Vertex& v = out[std::size_t(j) * kVerts1 + i];
            const int wx = int(originX + std::int64_t(i) * kSpacing);
            const int wz = int(originZ + std::int64_t(j) * kSpacing);
            const float h = float(s.h);
            const float jit = float(hash2(wx, wz) & 255u) / 255.0f - 0.5f;
            glm::vec3 col;
            bool water = false;
            if (s.h < kSea) {
                // the sea floor tints the water: turquoise where it is shallow, navy where deep
                water = true;
                col = glm::mix(glm::vec3(0.16f, 0.46f, 0.55f), glm::vec3(0.04f, 0.13f, 0.30f),
                               sstep(0.0f, 26.0f, float(kSea) - h));
            } else if (s.desert > 0.42f) {
                col = glm::vec3(0.82f, 0.68f, 0.42f) * (1.0f + 0.10f * jit);
            } else if (h <= float(kSea + 3)) {
                col = glm::vec3(0.78f, 0.66f, 0.44f); // the beach
            } else if (s.h >= gen.snowLineAt(wx, wz)) {
                col = glm::vec3(0.90f, 0.92f, 0.96f);
            } else if (h >= 150.0f) {
                col = glm::mix(glm::vec3(0.36f, 0.42f, 0.22f), glm::vec3(0.46f, 0.45f, 0.47f),
                               sstep(150.0f, 185.0f, h));
            } else {
                col = glm::mix(glm::vec3(0.36f, 0.52f, 0.22f), glm::vec3(0.15f, 0.34f, 0.14f),
                               s.forest);
                col *= 1.0f + 0.12f * jit;
            }
            v.h = water ? float(kSea) - 0.2f : h;
            v.r = q8(col.r);
            v.g = q8(col.g);
            v.b = q8(col.b);
            v.water = water ? 1 : 0;
            glm::vec3 n(0.0f, 1.0f, 0.0f);
            if (!water) {
                n = glm::normalize(glm::vec3(at(i - 1, j) - at(i + 1, j), 2.0f * kSpacing,
                                             at(i, j - 1) - at(i, j + 1)));
            }
            v.nx = std::int8_t(std::lround(n.x * 127.0f));
            v.ny = std::int8_t(std::lround(n.y * 127.0f));
            v.nz = std::int8_t(std::lround(n.z * 127.0f));
        }
}

void FarTerrain::update(const TerrainGenerator& gen, JobSystem& jobs, const glm::dvec3& cam) {
    if (m_stagingReady.load(std::memory_order_acquire)) {
        // a result from before a reset() belongs to another world: drop it
        if (m_jobEpoch.load(std::memory_order_relaxed) == m_seedEpoch) {
            glNamedBufferSubData(m_ssbo, 0, GLsizeiptr(m_staging.size() * sizeof(Vertex)),
                                 m_staging.data());
            m_origin = m_stagingOrigin;
            m_centre = m_stagingOrigin + glm::dvec2(kHalfExtent);
            m_haveMesh = true;
        }
        m_stagingReady.store(false, std::memory_order_release);
        m_building.store(false, std::memory_order_release);
    }
    if (m_building.load(std::memory_order_acquire))
        return;
    const glm::dvec2 cxz(cam.x, cam.z);
    if (m_haveMesh && glm::length(cxz - m_centre) < kRebuildDistance)
        return;

    // grid corner snapped to the spacing, so the same vertices are sampled however the camera
    // wandered in and the terrain never shimmers between rebuilds
    const std::int64_t ox =
        std::int64_t(std::floor(cam.x / kSpacing)) * kSpacing - std::int64_t(kHalfExtent);
    const std::int64_t oz =
        std::int64_t(std::floor(cam.z / kSpacing)) * kSpacing - std::int64_t(kHalfExtent);
    m_building.store(true, std::memory_order_release);
    m_jobRunning.store(true, std::memory_order_release);
    m_jobEpoch.store(m_seedEpoch, std::memory_order_relaxed);
    const TerrainGenerator* g = &gen;
    jobs.submit([this, g, ox, oz] {
        build(*g, ox, oz, m_staging);
        m_stagingOrigin = glm::dvec2(double(ox), double(oz));
        m_stagingReady.store(true, std::memory_order_release);
        m_jobRunning.store(false, std::memory_order_release); // last: nothing here after this
    });
}

void FarTerrain::draw(const glm::mat4& viewProj, const glm::dvec3& cam, const glm::vec3& sunL,
                      const glm::vec3& sunColor, const glm::vec3& skyAmbient, float innerRadius) {
    if (!m_haveMesh || m_prog == 0)
        return;
    const glm::vec3 originRel(float(m_origin.x - cam.x), 0.0f, float(m_origin.y - cam.z));
    glUseProgram(m_prog);
    glProgramUniformMatrix4fv(m_prog, 0, 1, GL_FALSE, glm::value_ptr(viewProj));
    glProgramUniform3fv(m_prog, 4, 1, &originRel.x);
    glProgramUniform1f(m_prog, 5, float(cam.y));
    glProgramUniform1f(m_prog, 6, innerRadius);
    glProgramUniform3fv(m_prog, 7, 1, &sunL.x);
    glProgramUniform3fv(m_prog, 8, 1, &sunColor.x);
    glProgramUniform3fv(m_prog, 9, 1, &skyAmbient.x);
    glProgramUniform1f(m_prog, 10, float(kSpacing));
    glProgramUniform2f(m_prog, 12, float(m_origin.x), float(m_origin.y));
    glProgramUniform1i(m_prog, 11, kCells);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssbo);
    glBindVertexArray(m_vao);
    glDisable(GL_CULL_FACE); // one sheet seen from either side; winding is not worth a bug
    glDrawArrays(GL_TRIANGLES, 0, kCells * kCells * 6);
    glEnable(GL_CULL_FACE);
}

} // namespace vx
