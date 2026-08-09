#include "audio/PlayerSounds.hpp"

#include "audio/AudioEngine.hpp"
#include "world/ChunkManager.hpp"
#include "world/Player.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace vx {

namespace {

// Tries the block's own id, then its material family, then the generic name. The id tier is
// what lets someone give one specific block its own sound; the material tier is what stops
// them having to do it twenty-five times for the lamps.
bool playMaterial(AudioEngine& audio, const char* kind, int blockId, const glm::dvec3& at,
                  float volume, float pitch) {
    const std::string k = kind;
    if (audio.playIfPresent(k + "_" + std::to_string(blockId), at, volume, pitch))
        return true;
    if (audio.playIfPresent(k + "_" + blocks::soundMaterial(BlockId(blockId)), at, volume,
                            pitch))
        return true;
    return audio.playIfPresent(k, at, volume, pitch);
}

} // namespace

float PlayerSounds::rand01() {
    m_rng = m_rng * 1664525u + 1013904223u;
    return float(m_rng >> 8) / float(1 << 24);
}

void PlayerSounds::reset(const Player& player) {
    // Every one of these describes a fall or a crossing that is no longer happening. The
    // dangerous one is m_fallFrom: it only ever climbs while airborne, so a player dropped
    // into walk mode from a hundred blocks up in fly mode lands with a hundred-block thud
    // that nobody fell. Re-anchoring it to where the player actually IS is the fix.
    m_wasOnGround = true;
    m_fallFrom = player.position.y;
    m_stepAccum = 0.0;
    m_swimAccum = 0.0;
    m_wasInWater = player.inWater;
}

PlayerSounds::Event PlayerSounds::tick(AudioEngine& audio, const ChunkManager& world,
                                       const Player& player, const glm::dvec3& before) {
    Event ev;
    const glm::dvec3 d = player.position - before;
    const double moved = std::sqrt(d.x * d.x + d.z * d.z);

    // What is underfoot. One block down from the feet, which is the block being struck
    // rather than the one being stood in.
    const glm::ivec3 under(std::floor(player.position.x), std::floor(player.position.y) - 1,
                           std::floor(player.position.z));
    const int id = int(world.blockAtWorld(under));
    ev.at = player.position;
    ev.blockId = id;

    // Entering and leaving the water. The body edge, not the head's: you hear the splash
    // when you break the surface, which happens well before your eyes go under.
    if (player.inWater != m_wasInWater) {
        // Scaled by how hard you hit it. Sliding in off a beach should not sound like a
        // cannonball, and falling forty blocks into a lake should.
        const double speed = std::abs(player.velocity.y) + moved * 60.0;
        const float v = float(std::clamp(0.25 + speed * 0.045, 0.25, 1.0));
        audio.playAt(player.inWater ? "splash_in" : "splash_out", player.position, v);
        m_wasInWater = player.inWater;
        // Whatever fall was in progress ended in water, so it must not also thud.
        m_fallFrom = player.position.y;
        m_swimAccum = 0.0;
    }

    if (player.inWater) {
        // Swimming paces off distance the same way walking does, just longer strokes.
        m_swimAccum += moved + std::abs(d.y) * 0.5;
        if (m_swimAccum > kSwimStrokeBlocks) {
            m_swimAccum = 0.0;
            audio.playAt("swim", player.position, 0.35f + 0.1f * rand01(),
                         0.95f + 0.1f * rand01());
        }
        m_wasOnGround = player.onGround;
        m_fallFrom = std::max(m_fallFrom, player.position.y);
        return ev;
    }

    if (player.onGround) {
        if (!m_wasOnGround) {
            // Landed. The drop is measured from the highest point of the fall, not from
            // where the jump started — otherwise walking off a cliff registers as nothing.
            const double drop = m_fallFrom - player.position.y;
            if (drop > kLandMin) {
                const double t = std::clamp((drop - kLandMin) / 6.0, 0.0, 1.0);
                playMaterial(audio, "step", id, player.position, float(0.45 + 0.4 * t),
                             float(0.95 - 0.1 * t));
                if (drop > kLandHard) {
                    // A second, heavier layer ON TOP of the material's own sound rather than
                    // instead of it — landing hard on sand and on stone are still different.
                    const float heavy = float(std::clamp((drop - kLandHard) / 12.0, 0.15, 1.0));
                    audio.playAt("land_hard", player.position, 0.35f + 0.5f * heavy,
                                 1.05f - 0.15f * heavy);
                }
                ev.stepped = true;
                ev.landed = true;
            }
            m_stepAccum = 0.0;
        }
        m_fallFrom = player.position.y;
        if (!ev.stepped) {
            m_stepAccum += moved;
            if (m_stepAccum > kStrideBlocks) {
                m_stepAccum = 0.0;
                playMaterial(audio, "step", id, player.position, 0.5f,
                             0.94f + 0.12f * rand01());
                ev.stepped = true;
            }
        }
    } else {
        // Airborne. A jump is the frame the ground is lost while still moving up; walking
        // off a ledge is the same transition with velocity pointing the other way and is
        // deliberately silent.
        if (m_wasOnGround && player.velocity.y > 0.5) {
            audio.playAt("jump", player.position, 0.30f, 0.96f + 0.1f * rand01());
            m_fallFrom = player.position.y;
        }
        m_stepAccum = 0.0;
        m_fallFrom = std::max(m_fallFrom, player.position.y);
    }
    m_wasOnGround = player.onGround;
    return ev;
}

void PlayerSounds::frame(AudioEngine& audio, const glm::dvec3& eye, bool eyeUnderwater,
                         float dt) {
    if (eyeUnderwater != m_wasEyeUnder) {
        // The muffle crossfade is what sells this; the clip is only the cue that it started.
        audio.playAt(eyeUnderwater ? "splash_in" : "splash_out", eye, 0.45f,
                     eyeUnderwater ? 0.85f : 1.15f);
        m_wasEyeUnder = eyeUnderwater;
        m_bubbleIn = 0.8f;
    }
    if (!eyeUnderwater)
        return;
    // Irregular on purpose. A bubble on a fixed timer stops reading as breathing within
    // about three of them.
    m_bubbleIn -= dt;
    if (m_bubbleIn <= 0.0f) {
        m_bubbleIn = 2.2f + 3.0f * rand01();
        audio.playAt("bubble", eye, 0.30f + 0.15f * rand01(), 0.9f + 0.25f * rand01());
    }
}

} // namespace vx
