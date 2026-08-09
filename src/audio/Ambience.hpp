#pragma once

namespace vx {

class AudioEngine;

// What the world sounds like from where the player is standing.
//
// Split from AudioEngine because the interesting part is the decision, not the playback: a
// pure state -> gains function can be asserted directly in --smoke with no device, no clips
// and no timing, which is the only way to test "a cave sounds like a cave" at all.
struct AmbienceState {
    float daylight = 1.0f;   // 1 full day, 0 after dusk — FrameRenderer's SkyLight.daylight
    float skyAccess = 1.0f;  // 0 sealed in, 1 open sky, from the voxel sky light at the ear
    float depth = 0.0f;      // blocks below the local surface; negative above it
    float precip = 0.0f;     // weather intensity, 0..1
    float nearWater = 0.0f;  // 0..1, how much water surrounds the ear
    bool underwater = false; // camera eye submerged
};

// Gain per bed, 0..1. Named rather than an array because the names ARE the clip names in
// sounds/ and a mismatch should not compile away into silence.
struct BedMix {
    float day = 0.0f;
    float night = 0.0f;
    float wind = 0.0f;
    float cave = 0.0f;
    float deep = 0.0f;
    float underwater = 0.0f;
    float rain = 0.0f;
    float water = 0.0f;
};

// Deterministic, no clock, no engine. Every gain is a function of the state passed in.
BedMix mixAmbience(const AmbienceState& s);

// Hands the mix to the engine under one master scale, then ends anything not in it. Separate
// from mixAmbience so the decision stays testable without an AudioEngine at all.
void applyAmbience(AudioEngine& audio, const BedMix& mix, float master);

// Thunder. Not part of the mix because it is an event on a clock rather than a gain, and a
// storm that rumbles on a fixed interval stops being weather after the second one.
class StormSounds {
public:
    // Returns true on the frames it actually fired, which is what --smoke counts.
    bool update(AudioEngine& audio, float precip, float skyAccess, float dt);

private:
    // Nothing until it is genuinely coming down. Light drizzle with thunder in it reads as
    // a bug rather than as atmosphere.
    static constexpr float kMinPrecip = 0.35f;
    float m_next = 9.0f;
    unsigned m_rng = 0x6C078965u;
};

} // namespace vx
