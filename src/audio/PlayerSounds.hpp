#pragma once

#include <glm/glm.hpp>

namespace vx {

class AudioEngine;
class ChunkManager;
class Player;

// Turns player state into sound events: footsteps, jumps, landings, entering and leaving
// water, swimming, bubbles.
//
// It exists as a class rather than as code at the call site because every one of those is an
// EDGE, not a state. "Landed" is the frame onGround goes true, and the volume depends on how
// far the fall was, which nothing knows unless something remembered where it started. That
// memory is the whole reason this is not a lambda in main.
//
// Unlike AudioEngine this DOES take a ChunkManager, deliberately: the engine stays free of
// the world so it can be tested headless, and the glue that needs both lives here instead.
class PlayerSounds {
public:
    // What the physics tick produced, so the caller can spawn the matching particles without
    // this having to know about them.
    struct Event {
        bool stepped = false; // a footstep fired this tick
        bool landed = false;  // and it was a landing rather than a stride
        glm::dvec3 at{0.0};   // feet, where the sound was placed
        int blockId = 0;      // what was underfoot
    };

    // Call from the fixed 60 Hz physics tick, once per step(), with the position from before
    // it. Distance-paced rather than time-paced, so sprinting does not machine-gun and
    // walking into a wall does not tick.
    Event tick(AudioEngine& audio, const ChunkManager& world, const Player& player,
               const glm::dvec3& before);

    // Call once a frame with the camera eye. Handles the submerged head, which is a camera
    // fact, not a player fact, and so is true in fly mode too.
    void frame(AudioEngine& audio, const glm::dvec3& eye, bool eyeUnderwater, float dt);

    // Call whenever the player is MOVED rather than having walked: entering walk mode from
    // fly, loading a world, changing seed. Without it the next landing is measured from
    // wherever the fall counter was left, and dropping into walk mode from high up in fly
    // mode arrives with a thud nobody fell.
    void reset(const Player& player);

private:
    // Below this a landing is just the next footstep. A block is 1.0, so stepping off one is
    // deliberately under the bar, that is a stride, not a drop.
    static constexpr double kLandMin = 0.55;
    // Where a landing stops being a step and starts being an impact. Minecraft's fall damage
    // threshold, borrowed because it is well-tuned rather than because it means anything
    // here: this engine has no health.
    static constexpr double kLandHard = 3.0;
    static constexpr double kStrideBlocks = 2.2;
    static constexpr double kSwimStrokeBlocks = 1.5;

    bool m_wasOnGround = true;
    bool m_wasInWater = false;
    bool m_wasEyeUnder = false;
    double m_fallFrom = 0.0; // highest y since leaving the ground
    double m_stepAccum = 0.0;
    double m_swimAccum = 0.0;
    float m_bubbleIn = 0.0f; // seconds until the next underwater bubble
    unsigned m_rng = 0x2545F491u;

    float rand01();
};

} // namespace vx
