#include "render/Lightning.hpp"

#include "gen/TerrainGenerator.hpp"
#include "render/Particles.hpp"

#include <algorithm>
#include <cmath>

namespace vx {

namespace {
// Gameplay speed of sound, in blocks a second. Real thunder is far too slow to be fun (a bolt
// 900 blocks off would take three seconds to be heard at 340; this makes it 6), and far too
// fast to be felt at a block-sized scale; this is the compromise that still lets a counted
// second mean something.
constexpr float kSoundSpeed = 150.0f;
constexpr float kCloudBase = 440.0f;
} // namespace

float Lightning::rnd() {
    m_rng = m_rng * 1664525u + 1013904223u;
    return float(m_rng >> 8) / float(1 << 24);
}

void Lightning::reset() {
    m_pendingCount = 0;
    m_flash = 0.0f;
    m_next = 12.0f;
    m_reflash = -1.0f;
}

void Lightning::strike(const glm::dvec3& cam, const glm::vec2& forward, const TerrainGenerator& gen,
                       Particles& particles) {
    // Somewhere in the half of the sky the player is facing, so a strike is something seen, not
    // something that happened behind them.
    const float base = std::atan2(forward.x, -forward.y);
    const float bearing = base + (rnd() - 0.5f) * 2.4f;
    const float dist = 220.0f + 700.0f * rnd();
    const double x = cam.x + double(std::sin(bearing) * dist);
    const double z = cam.z - double(std::cos(bearing) * dist);
    const int ground = std::max(gen.surfaceHeight(int(std::floor(x)), int(std::floor(z))), kSeaLevelForSpawn);
    m_top = {x + double((rnd() - 0.5f) * 60.0f), double(kCloudBase), z + double((rnd() - 0.5f) * 60.0f)};
    m_bottom = {x, double(ground) + 1.0, z};
    // a bolt has to stay visible at range, so it thickens with distance: about a pixel wide
    m_thickness = 0.35f + dist * 0.0022f;
    particles.spawnBolt(m_top, m_bottom, m_rng, m_thickness);

    m_flash = 0.85f;
    m_reflash = 0.06f + 0.07f * rnd();

    if (m_pendingCount < kMaxPending) {
        Pending& p = m_pending[m_pendingCount++];
        p.delay = dist / kSoundSpeed;
        p.volume = std::clamp(1.15f - dist / 1300.0f, 0.3f, 1.0f);
        p.pitch = 0.85f + 0.3f * rnd() + 0.15f * (dist / 900.0f); // far thunder rolls lower
        p.pitch = std::clamp(p.pitch, 0.8f, 1.1f);
    }
}

Lightning::Cue Lightning::update(float dt, float storm, const glm::dvec3& cam,
                                 const glm::vec2& forward, const TerrainGenerator& gen,
                                 Particles& particles) {
    Cue cue;
    m_flash = std::max(0.0f, m_flash - dt * 7.5f);

    if (m_reflash >= 0.0f) {
        m_reflash -= dt;
        if (m_reflash < 0.0f) {
            // the flicker: same path, a fresh set of forks
            m_flash = std::max(m_flash, 0.8f);
            particles.spawnBolt(m_top, m_bottom, m_rng + 977u, m_thickness);
            m_reflash = -1.0f;
        }
    }

    for (int i = 0; i < m_pendingCount;) {
        m_pending[i].delay -= dt;
        if (m_pending[i].delay <= 0.0f) {
            // the loudest of several arriving in one frame wins; the rest are dropped
            if (!cue.thunder || m_pending[i].volume > cue.volume) {
                cue.thunder = true;
                cue.volume = m_pending[i].volume;
                cue.pitch = m_pending[i].pitch;
            }
            m_pending[i] = m_pending[--m_pendingCount];
        } else {
            ++i;
        }
    }

    if (storm > 0.05f) {
        m_next -= dt * (0.4f + 1.6f * storm);
        if (m_next <= 0.0f) {
            strike(cam, forward, gen, particles);
            // a bad storm strikes every quarter minute or so, a marginal one about every half
            m_next = 6.0f + 30.0f * rnd() / (0.35f + storm);
        }
    } else if (m_next < 8.0f) {
        m_next = 8.0f; // leaving a storm resets the wait, so re-entering one is not instant
    }
    return cue;
}

} // namespace vx
