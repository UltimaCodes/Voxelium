#pragma once

#include "gfx/MappedRing.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <filesystem>

namespace vx {

class ChunkManager;

// Short-lived world particles: block-break shards, dust and sand puffs, wind-blown grains, water
// splashes, rain splashes.
//
// Fixed capacity, simulated on the CPU, drawn in one vertex-pulled instanced call. The
// capacity is fixed because the engine holds heap allocs/frame at 0 in steady state and a
// growing pool would break that; when it is full, new spawns replace the oldest, which is
// the right answer for something nobody is looking at any more anyway.
//
// Positions are kept in doubles and made camera-relative at upload time, for the same
// reason the world is: float world coordinates rot with distance from the origin.
//
// Most of them have real physics against the world: they fall, hit the ground, bounce and slide
// to a stop, stall in water, and the light ones (dust, sand) are pushed about by the wind. Give
// update() the world to switch that on.
class Particles {
public:
    static constexpr int kCapacity = 4096;

    // Ambient conditions the emitters below react to. Assembled by the caller.
    struct Ambient {
        float desert = 0.0f; // 0..1, how sandy the surroundings are
        float coast = 0.0f;  // 0..1, beach and shoreline
        float precip = 0.0f; // 0..1
        float snow = 0.0f;   // 0 rain .. 1 snow; rain splashes only
        float time = 0.0f;   // seconds, drives the wind
        bool underwater = false;
    };

    bool init(const std::filesystem::path& shaderDir);
    void shutdown();

    // A block broke: throw shards tinted by that block's texture. Loose ground (sand, gravel,
    // snow, dirt) also kicks up a cloud of dust.
    void spawnBlockBreak(const glm::dvec3& blockMin, int texLayer, int blockId = 0);
    // Something entered water at this surface point.
    void spawnSplash(const glm::dvec3& at);
    // A foot landed on `texLayer` ground. Sand and snow put up a proper puff.
    void spawnFootstep(const glm::dvec3& at, int texLayer, int blockId = 0);
    // A heavier version for landing from a jump or fall.
    void spawnLanding(const glm::dvec3& at, int texLayer, int blockId = 0);
    // A lightning bolt from `top` down to `bottom`: a jagged line of very bright particles with a
    // few forks, gone in about a fifth of a second. `thickness` is the width in blocks (the
    // caller scales it with distance so a bolt stays visible at range).
    void spawnBolt(const glm::dvec3& top, const glm::dvec3& bottom, std::uint32_t seed,
                   float thickness);

    // Wind-blown sand over deserts and beaches, and rain splashing off whatever it lands on.
    // Cheap enough to call every frame: it spawns a handful of particles and does a few block
    // lookups, and does nothing at all when the conditions are not there.
    void emitAmbient(float dt, const ChunkManager& world, const glm::dvec3& cam, const Ambient& a);

    void update(float dt, const ChunkManager* world = nullptr);
    // Call inside the HDR scene pass, after the world: depth-tested, no depth write.
    void draw(const glm::mat4& viewProj, const glm::dvec3& cameraPos, const glm::vec3& right,
              const glm::vec3& up, unsigned int blockTexArray);

    int liveCount() const { return m_live; }
    // Harness only: live particles drawing texture layer `layer`.
    int countLayer(int layer) const;

private:
    struct Particle {
        glm::dvec3 pos;
        glm::vec3 vel;
        float life;    // seconds remaining
        float invLife; // 1 / initial life, so fade needs no divide per frame
        float size;
        int layer;     // block texture array layer, -1 = flat colour (splash)
        float gravity = 1.0f; // multiple of kGravity: dust barely falls, shards fall hard
        float drag = 0.0f;    // 1/s of velocity lost to the air
        float bounce = 0.0f;  // fraction of speed kept off a surface
        float growth = 0.0f;  // size gained per second (a puff spreads as it thins)
        float maxAlpha = 1.0f;
        glm::vec3 tint{0.62f, 0.80f, 0.95f}; // flat colour for layer < 0, HDR for a bolt
        float wind = 0.0f;    // how hard the wind pushes it
        bool collide = false;
        bool resting = false;
        bool solid = false; // writes depth: the composite pass paints sky over anything that does not
    };
    // What the shader reads: camera-relative centre + size, then tint + alpha.
    struct GpuParticle {
        float x, y, z, size;
        float r, g, b, a;
        float layer, pad0, pad1, pad2;
    };

    Particle& acquire();
    Particle& fresh(); // acquire() with every field back at its default
    void puff(const glm::dvec3& at, int texLayer, int count, float spread, float lift);
    bool groundBelow(const ChunkManager& w, int x, int z, int fromY, int toY, int& outY,
                     int& outBlock) const;

    unsigned int m_prog = 0;
    unsigned int m_vao = 0;
    MappedRing m_ring;
    Particle m_pool[kCapacity];
    int m_live = 0;
    int m_victim = 0; // next slot overwritten while the pool is full
    std::uint32_t m_rng = 0x9E3779B9u;
    float m_time = 0.0f;
    float m_sandCarry = 0.0f;  // fractional spawns carried between frames
    float m_rainCarry = 0.0f;
    float m_rand01();
};

} // namespace vx
