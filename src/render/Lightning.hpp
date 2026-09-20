#pragma once

#include <glm/glm.hpp>

#include <cstdint>

namespace vx {

class Particles;
class TerrainGenerator;

// Thunderstorms: bolts out of the cloud deck onto the terrain, the flash they throw over the
// whole sky and ground, and the thunder that arrives after them.
//
// A bolt is a jagged line of very bright particles (midpoint displacement, so it forks and
// kinks the way one does), drawn by the ordinary particle pass and picked up by bloom because
// it is HDR. The flash is a number that jumps to 1 on the strike, flickers once, and decays; the
// renderer lifts the sky and the ambient light by it. Thunder is scheduled at the strike's
// distance over a slowed speed of sound, so the further off a bolt is, the longer you wait and
// the quieter it is, which is most of what makes a storm feel like it has a size.
class Lightning {
public:
    struct Cue {
        bool thunder = false; // fire a thunder clip now
        float volume = 0.0f;  // 0..1, from how far off the strike was
        float pitch = 1.0f;
    };

    // storm: 0 nothing .. 1 the worst of it. `forward` is where the camera faces (xz), so the
    // strikes land where they can be seen.
    Cue update(float dt, float storm, const glm::dvec3& cam, const glm::vec2& forward,
               const TerrainGenerator& gen, Particles& particles);

    float flash() const { return m_flash; }
    void reset();

private:
    float rnd();
    void strike(const glm::dvec3& cam, const glm::vec2& forward, const TerrainGenerator& gen,
                Particles& particles);

    static constexpr int kMaxPending = 6;
    struct Pending {
        float delay = 0.0f;
        float volume = 0.0f;
        float pitch = 1.0f;
    };
    Pending m_pending[kMaxPending];
    int m_pendingCount = 0;

    float m_flash = 0.0f;
    float m_next = 12.0f;     // seconds to the next strike
    float m_reflash = -1.0f;  // seconds to the flicker after a strike, or < 0 for none
    glm::dvec3 m_top{0.0}, m_bottom{0.0};
    float m_thickness = 1.0f;
    std::uint32_t m_rng = 0x51ED270Bu;
};

} // namespace vx
