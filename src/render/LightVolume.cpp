#include "render/LightVolume.hpp"

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
