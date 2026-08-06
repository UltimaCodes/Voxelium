#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <filesystem>

namespace vx {

// Short-lived world particles: block-break shards, water splashes, footstep puffs.
//
// Fixed capacity, simulated on the CPU, drawn in one vertex-pulled instanced call. The
// capacity is fixed because the engine holds heap allocs/frame at 0 in steady state and a
// growing pool would break that; when it is full, new spawns replace the oldest, which is
// the right answer for something nobody is looking at any more anyway.
//
// Positions are kept in doubles and made camera-relative at upload time, for the same
// reason the world is: float world coordinates rot with distance from the origin.
class Particles {
public:
    static constexpr int kCapacity = 4096;

    bool init(const std::filesystem::path& shaderDir);
    void shutdown();

    // A block broke: throw shards tinted by that block's texture.
    void spawnBlockBreak(const glm::dvec3& blockMin, int texLayer);
    // Something entered water at this surface point.
    void spawnSplash(const glm::dvec3& at);
    // A foot landed on `texLayer` ground.
    void spawnFootstep(const glm::dvec3& at, int texLayer);

    void update(float dt);
    // Call inside the HDR scene pass, after the world: depth-tested, no depth write.
    void draw(const glm::mat4& viewProj, const glm::dvec3& cameraPos, const glm::vec3& right,
              const glm::vec3& up, unsigned int blockTexArray);

    int liveCount() const { return m_live; }

private:
    struct Particle {
        glm::dvec3 pos;
        glm::vec3 vel;
        float life;    // seconds remaining
        float invLife; // 1 / initial life, so fade needs no divide per frame
        float size;
        int layer; // block texture array layer, -1 = flat colour (splash)
    };
    // What the shader reads: camera-relative centre + size, then tint + alpha.
    struct GpuParticle {
        float x, y, z, size;
        float r, g, b, a;
        float layer, pad0, pad1, pad2;
    };

    Particle& acquire();

    unsigned int m_prog = 0;
    unsigned int m_vao = 0;
    unsigned int m_ssbo = 0;
    GpuParticle* m_map = nullptr;
    Particle m_pool[kCapacity];
    int m_live = 0;
    std::uint32_t m_rng = 0x9E3779B9u;
    float m_rand01();
};

} // namespace vx
