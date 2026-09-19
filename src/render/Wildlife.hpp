#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <filesystem>

namespace vx {

class ChunkManager;

// Things that live in the world rather than being made of it: butterflies over the meadows,
// fireflies after dusk, gulls circling a coast, fish schooling over a reef, leaves coming off
// the canopy, motes drifting in the light, bubbles rising off the coral.
//
// None of it is saved, none of it collides with anything but the terrain, and none of it is
// simulated further than the player can see: creatures spawn in a ring around the camera when
// the surroundings call for them (the Env below), wander under simple steering, and are
// recycled the moment they leave the ring. So the cost is a few hundred sprites and a handful
// of block lookups per frame, at any render distance.
//
// Rendering is one vertex-pulled draw, procedural shapes in the fragment shader (see
// wildlife.frag). It runs BEFORE the water pass so a fish under a lake is seen through the
// water, tinted and refracted like everything else beneath it.
class Wildlife {
public:
    static constexpr int kCapacity = 384;

    // What the surroundings are like right now. Assembled by the caller from the same state
    // the ambience director hears, so what you see and what you hear agree about where you are.
    struct Env {
        float daylight = 1.0f; // 1 full day, 0 after dusk
        float precip = 0.0f;
        float skyAccess = 1.0f;
        float coast = 0.0f;
        float forest = 0.0f;
        float desert = 0.0f;
        float reef = 0.0f;
        float nearWater = 0.0f;
        bool underwater = false;
    };

    bool init(const std::filesystem::path& shaderDir);
    void shutdown();
    void clear() { m_live = 0; }

    void update(float dt, const ChunkManager& world, const glm::dvec3& cam, const Env& env);
    // Call inside the HDR scene pass, after the opaque world and BEFORE water. Depth tested,
    // no depth write.
    void draw(const glm::mat4& viewProj, const glm::dvec3& cam, const glm::vec3& right,
              const glm::vec3& up, bool underwater);

    int liveCount() const { return m_live; }

private:
    enum Kind : std::uint8_t { Butterfly = 1, Firefly, Gull, Fish, Leaf, Mote, Bubble, Bird };
    static bool writesDepth(Kind k) {
        return k == Butterfly || k == Gull || k == Leaf || k == Bird;
    }

    struct Critter {
        glm::dvec3 pos{0.0};
        glm::vec3 vel{0.0f};
        glm::vec2 anchor{0.0f}; // gulls: the centre of the circle they fly
        glm::vec3 color{1.0f};
        float size = 0.1f;
        float phase = 0.0f; // animation
        float age = 0.0f;
        float life = 1e9f;  // seconds until it should leave, whether or not it wandered off
        float aux = 0.0f;   // kind-specific: heading, orbit angle, ...
        float aux2 = 0.0f;  // kind-specific: ground height, orbit radius, ...
        float aux3 = 0.0f;
        int group = -1;     // fish: which school
        Kind kind = Mote;
    };
    struct School {
        glm::dvec3 center{0.0};
        glm::vec3 vel{0.0f};
        float surface = 0.0f;
        float floor = 0.0f;
        bool alive = false;
    };
    // What the shader reads.
    struct GpuSprite {
        float x, y, z, size;
        float r, g, b, a;
        float kind, dirX, dirY, phase;
    };

    Critter* acquire();
    int count(Kind k) const;
    float rnd();
    void spawnButterfly(const ChunkManager& w, const glm::dvec3& cam);
    void spawnFirefly(const ChunkManager& w, const glm::dvec3& cam);
    void spawnGull(const ChunkManager& w, const glm::dvec3& cam);
    void spawnSchool(const ChunkManager& w, const glm::dvec3& cam, const Env& env);
    void spawnLeaf(const ChunkManager& w, const glm::dvec3& cam);
    void spawnMote(const glm::dvec3& cam, const Env& env);
    void spawnBubble(const ChunkManager& w, const glm::dvec3& cam);
    void spawnFlock(const glm::dvec3& cam);

    unsigned int m_prog = 0;
    unsigned int m_vao = 0;
    unsigned int m_ssbo = 0;
    GpuSprite* m_map = nullptr;

    Critter m_pool[kCapacity];
    int m_live = 0;
    static constexpr int kSchools = 12;
    School m_schools[kSchools];
    std::uint32_t m_rng = 0x2545F491u;
    float m_spawnClock = 0.0f;
    float m_leafClock = 0.0f;
    float m_bubbleClock = 0.0f;
    float m_flockClock = 30.0f;
    float m_time = 0.0f;
    float m_light = 1.0f; // how bright the unlit sprites are drawn, from the time of day
};

} // namespace vx
