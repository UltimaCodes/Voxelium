#include "render/LightVolume.hpp"

#include "gfx/Shader.hpp"
#include "world/Chunk.hpp"
#include "world/ChunkManager.hpp"

#include <glad/gl.h>

#include <algorithm>
#include <cmath>

namespace vx {
namespace {

int wrap(int v, int n) {
    const int m = v % n;
    return m < 0 ? m + n : m;
}

} // namespace

LightVolume::LightVolume() {
    glCreateTextures(GL_TEXTURE_3D, 1, &m_tex);
    glTextureStorage3D(m_tex, 1, GL_RGBA8, kSize, kSize, kSize);
    glTextureParameteri(m_tex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(m_tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(m_tex, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTextureParameteri(m_tex, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTextureParameteri(m_tex, GL_TEXTURE_WRAP_R, GL_REPEAT);
    glClearTexImage(m_tex, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    m_staging.resize(std::size_t(kChunkSize3));
}

LightVolume::~LightVolume() {
    glDeleteTextures(1, &m_tex);
    glDeleteTextures(2, m_bounce);
    glDeleteTextures(1, &m_seed);
    glDeleteProgram(m_seedProg);
    glDeleteProgram(m_spreadProg);
}

bool LightVolume::initBounce(const std::filesystem::path& shaderDir) {
    m_seedProg = loadComputeProgram(shaderDir / "bounce_seed.comp");
    m_spreadProg = loadComputeProgram(shaderDir / "bounce_spread.comp");
    constexpr int n = kSize / 2;
    glCreateTextures(GL_TEXTURE_3D, 1, &m_seed);
    glTextureStorage3D(m_seed, 1, GL_RG16F, n, n, n);
    glCreateTextures(GL_TEXTURE_3D, 2, m_bounce);
    for (const unsigned int t : m_bounce) {
        glTextureStorage3D(t, 1, GL_R16F, n, n, n);
        glTextureParameteri(t, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTextureParameteri(t, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTextureParameteri(t, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTextureParameteri(t, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTextureParameteri(t, GL_TEXTURE_WRAP_R, GL_REPEAT);
        glClearTexImage(t, 0, GL_RED, GL_FLOAT, nullptr);
    }
    return haveBounce();
}

void LightVolume::computeBounce(bool shadows, unsigned int shadowTex, const glm::mat4* cascadeVp,
                                const glm::vec3& splits, const glm::vec3& cascadeTexel, const glm::vec3& sunL) {
    if (!haveBounce())
        return;
    constexpr int groups = kSize / 2 / 4;
    const auto wrapF = [](float v) { return v - std::floor(v / float(kSize)) * float(kSize); };
    const glm::vec3 base{wrapF(m_camTexel.x + m_minRel.x), wrapF(m_camTexel.y + m_minRel.z),
                         wrapF(m_camTexel.z + m_minRel.y)};
    glUseProgram(m_seedProg);
    glProgramUniform3fv(m_seedProg, 0, 1, &base.x);
    glProgramUniform3fv(m_seedProg, 1, 1, &m_minRel.x);
    glProgramUniformMatrix4fv(m_seedProg, 2, 3, GL_FALSE, &cascadeVp[0][0][0]);
    glProgramUniform3fv(m_seedProg, 5, 1, &splits.x);
    glProgramUniform3fv(m_seedProg, 6, 1, &sunL.x);
    glProgramUniform3fv(m_seedProg, 7, 1, &cascadeTexel.x);
    glProgramUniform1f(m_seedProg, 8, shadows ? 1.0f : 0.0f);
    glBindTextureUnit(0, m_tex);
    glBindTextureUnit(1, shadowTex);
    glBindImageTexture(0, m_seed, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RG16F);
    glDispatchCompute(groups, groups, groups);
    // it spreads a cell a pass: seed -> 0 -> 1 -> 0 ... ending in 1
    glUseProgram(m_spreadProg);
    glBindTextureUnit(0, m_seed);
    constexpr int kSteps = 6;
    for (int i = 0; i < kSteps; ++i) {
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        glBindTextureUnit(1, i == 0 ? m_seed : m_bounce[(i - 1) % 2]);
        glBindImageTexture(0, m_bounce[i % 2], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R16F);
        glDispatchCompute(groups, groups, groups);
    }
    static_assert(kSteps % 2 == 0, "the result has to end up in m_bounce[1]");
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
}

void LightVolume::upload(const ChunkManager& world, ChunkPos pos) {
    Slot& slot = m_slots[(wrap(pos.y, kChunks) * kChunks + wrap(pos.z, kChunks)) * kChunks + wrap(pos.x, kChunks)];
    slot.pos = pos;
    slot.lit = false;
    if (const Chunk* c = world.chunkData(pos)) {
        c->copyLight(m_staging.data());
        std::uint32_t any = 0;
        for (const std::uint32_t v : m_staging)
            any |= v;
        slot.lit = (any & 0xFFFFFFu) != 0;
    } else {
        // not there (or still generating): no light in the air
        std::fill(m_staging.begin(), m_staging.end(), 0u);
    }
    // Chunk::index is x | z << 5 | y << 10, so the texture's axes are (x, z, y)
    glTextureSubImage3D(m_tex, 0, wrap(pos.x, kChunks) * kChunkSize, wrap(pos.z, kChunks) * kChunkSize,
                        wrap(pos.y, kChunks) * kChunkSize, kChunkSize, kChunkSize, kChunkSize, GL_RGBA,
                        GL_UNSIGNED_BYTE, m_staging.data());
}

void LightVolume::update(const ChunkManager& world, const glm::dvec3& camera, int refresh) {
    // Low corner placed so the camera has 48 to 80 blocks of the volume on every side.
    const ChunkPos lo{int(std::floor((camera.x - 48.0) / kChunkSize)), int(std::floor((camera.y - 48.0) / kChunkSize)),
                      int(std::floor((camera.z - 48.0) / kChunkSize))};
    m_minRel = glm::vec3(glm::dvec3(lo.x, lo.y, lo.z) * double(kChunkSize) - camera);
    const auto wrapD = [](double v) { return float(v - std::floor(v / double(kSize)) * double(kSize)); };
    m_camTexel = {wrapD(camera.x), wrapD(camera.z), wrapD(camera.y)};

    // Chunks that just came into range first, all of them: a slot showing the chunk it held
    // before the camera moved would put that chunk's light in the wrong place.
    for (int y = 0; y < kChunks; ++y)
        for (int z = 0; z < kChunks; ++z)
            for (int x = 0; x < kChunks; ++x) {
                const ChunkPos want{lo.x + x, lo.y + y, lo.z + z};
                const Slot& slot =
                    m_slots[(wrap(want.y, kChunks) * kChunks + wrap(want.z, kChunks)) * kChunks + wrap(want.x, kChunks)];
                if (!(slot.pos == want))
                    upload(world, want);
            }
    // Then a few more re-read whatever they hold, so light that changed in place turns up.
    constexpr int kSlots = kChunks * kChunks * kChunks;
    for (int i = 0; i < refresh; ++i) {
        const ChunkPos pos = m_slots[m_cursor].pos;
        m_cursor = (m_cursor + 1) % kSlots;
        upload(world, pos);
    }
}

void LightVolume::invalidate() {
    for (Slot& s : m_slots)
        s = Slot{};
}

bool LightVolume::anyLight() const {
    for (const Slot& s : m_slots)
        if (s.lit)
            return true;
    return false;
}

} // namespace vx
