#pragma once

#include "world/ChunkPos.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

namespace vx {

class ChunkManager;

// The voxel light around the camera as a 3D texture, so the composite pass can march it and
// put a lamp's light in the air around it, not only on the faces it reaches. Underground that
// is most of what reads as light: a cave lit only on its walls looks painted.
//
// 4x4x4 chunks, 128^3 voxels of RGBA8, which is byte for byte how a chunk already stores its
// light, so a chunk goes up with one copy and no conversion. Each chunk sits at its world
// position modulo the texture (the sampler wraps), so crossing a chunk re-uploads the slab that
// came into range instead of the whole volume. Chunks are re-read a few a frame, round robin,
// so a lamp placed or broken shows in the air within a quarter second or so.
class LightVolume {
public:
    static constexpr int kChunks = 4;
    static constexpr int kSize = kChunks * 32;

    LightVolume();
    ~LightVolume();
    LightVolume(const LightVolume&) = delete;
    LightVolume& operator=(const LightVolume&) = delete;

    // `refresh` is how many unchanged chunks to re-read this frame on top of any that just came
    // into range.
    void update(const ChunkManager& world, const glm::dvec3& camera, int refresh = 4);

    unsigned int texture() const { return m_tex; }
    // The camera in texels, already in the texture's axis order (x, z, y) and wrapped.
    glm::vec3 cameraTexel() const { return m_camTexel; }
    // The volume's low corner relative to the camera, in blocks, world axis order. A sample
    // outside [min, min + kSize) is a wrapped-around neighbour and has to be ignored.
    glm::vec3 regionMinRel() const { return m_minRel; }
    // False when no chunk in range holds any block light: nothing to march.
    bool anyLight() const;

private:
    struct Slot {
        ChunkPos pos{INT32_MIN, INT32_MIN, INT32_MIN};
        bool lit = false; // holds some block light
    };
    void upload(const ChunkManager& world, ChunkPos pos);

    unsigned int m_tex = 0;
    Slot m_slots[kChunks * kChunks * kChunks];
    int m_cursor = 0;
    std::vector<std::uint32_t> m_staging;
    glm::vec3 m_camTexel{0.0f};
    glm::vec3 m_minRel{0.0f};
};

} // namespace vx
