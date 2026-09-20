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
// Rendering is vertex-pulled from one buffer in three batches. The glows (fireflies, motes,
// bubbles) are camera-facing sprites, since a round soft dot looks the same from every side.
// Everything with a body - butterflies, birds, fish, leaves - is real geometry built from small
// boxes (see wildlife3d.vert), because a flat sprite turns edge-on and vanishes when seen from
// the side. It runs BEFORE the water pass so a fish under a lake is seen through the water,
// tinted and refracted like everything else beneath it.
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

    // Harness only (--tour, VX_TOUR_WILD=zoo): one of each animal in a row in front of the camera,
    // close enough to judge the models from any side. The game never calls this.
    // set 1 is just the crab, turtle and jellyfish, larger and closer together.
    void placeDemo(const glm::dvec3& cam, const glm::vec3& forward, int set = 0);

private:
    enum Kind : std::uint8_t { Butterfly = 1, Firefly, Gull, Fish, Leaf, Mote, Bubble, Bird, Crab, Turtle, Jelly };
    // Solid, airborne things write depth (composite paints sky over any pixel still at the far
    // plane, so they would be erased otherwise). Fish are solid too but stay depth-free: one that
    // wrote depth would read to the water pass as shallow water and grow foam around itself.
    static bool writesDepth(Kind k) {
        return k == Butterfly || k == Gull || k == Leaf || k == Bird;
    }
    static bool isSolid(Kind k) { return k != Firefly && k != Mote && k != Bubble; }

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
    void spawnCrab(const ChunkManager& w, const glm::dvec3& cam);
    // turtles and jellyfish both live in open water; this finds a column of it near the camera
    bool findWater(const ChunkManager& w, const glm::dvec3& cam, float rMin, float rMax, int minDepth,
                   int& x, int& z, int& surface, int& floor);
    void spawnTurtle(const ChunkManager& w, const glm::dvec3& cam);
    void spawnJelly(const ChunkManager& w, const glm::dvec3& cam);

    unsigned int m_prog = 0;   // the glows, as camera-facing sprites
    unsigned int m_prog3d = 0; // the animals, as boxes
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
