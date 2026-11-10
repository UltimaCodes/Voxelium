#pragma once

#include <glm/glm.hpp>

namespace vx {

class AudioEngine;

// What the world sounds like from where the player is.
//
// Split from AudioEngine because the interesting part is the decision, not the playback: a
// pure state -> gains function can be asserted directly in --smoke with no device, no clips
// and no timing, which is the only way to test "a cave sounds like a cave" at all.
struct AmbienceState {
    float daylight = 1.0f;   // 1 full day, 0 after dusk, FrameRenderer's SkyLight.daylight
    float skyAccess = 1.0f;  // 0 sealed in, 1 open sky, from the voxel sky light at the ear
    float depth = 0.0f;      // blocks below the local surface; negative above it
    float precip = 0.0f;     // weather intensity, 0..1
    float nearWater = 0.0f;  // 0..1, how much water surrounds the ear
    bool underwater = false; // camera eye submerged
    // The biomes around the ear as 0..1 weights, from the generator, so the ground's sounds
    // carry on while flying over it.
    float coast = 0.0f;   // open sea and the sand around it
    float desert = 0.0f;
    float forest = 0.0f;  // broadleaf woods
    float open = 0.0f;    // plains, meadows, savanna
    float conifer = 0.0f; // taiga
    float jungle = 0.0f;
    float swamp = 0.0f;
    float snow = 0.0f;    // tundra, snowy taiga, snowcaps
    float alpine = 0.0f;
    float canopy = 0.0f;  // leaves physically around the ear
    float reef = 0.0f;    // coral, kelp and seagrass around the eye, only heard submerged
    float speed = 0.0f;   // how fast the ear moves, blocks a second
};

// Gain per bed, 0..1. Named rather than an array because the names ARE the clip names in
// sounds/ and a mismatch should not compile away into silence.
struct BedMix {
    float forest = 0.0f;  // birdsong in broadleaf woods
    float open = 0.0f;    // fields: distant birds, insects, a breeze
    float taiga = 0.0f;
    float jungle = 0.0f;
    float swamp = 0.0f;   // frogs and wetland birds
    float night = 0.0f;   // crickets
    float desert = 0.0f;
    float snow = 0.0f;    // cold wind
    float wind = 0.0f;    // open air, height and speed
    float leaves = 0.0f;  // wind in a canopy
    float surf = 0.0f;    // waves on a beach or open water
    float water = 0.0f;   // a river or lake beside you
    float rain = 0.0f;
    float cave = 0.0f;
    float drips = 0.0f;
    float underwater = 0.0f;
};

// Deterministic, no clock, no engine. Every gain is a function of the state passed in.
BedMix mixAmbience(const AmbienceState& s);

// Hands the mix to the engine under one master scale, then ends anything not in it. Separate
// from mixAmbience so the decision stays testable without an AudioEngine at all.
void applyAmbience(AudioEngine& audio, const BedMix& mix, float master);

// Single calls from around the listener on top of the beds: birds in the trees, a woodpecker,
// crows, an owl after dark, gulls on the coast, frogs by the water, an insect going past. Each
// kind keeps its own clock, so they never fall into a rhythm.
class AmbientCalls {
public:
    // Returns how many calls fired this frame.
    int update(AudioEngine& audio, const AmbienceState& s, const glm::dvec3& ear, float dt);

private:
    static constexpr int kKinds = 8;
    float m_next[kKinds] = {3.0f, 20.0f, 15.0f, 10.0f, 30.0f, 4.0f, 2.0f, 12.0f};
    unsigned m_rng = 0x2545F491u;
    float rand01();
};

// Music the way it should be in a world like this: now and then, not all the time. A track every
// few minutes, chosen by time of day and whether you are underground, with long quiet between.
class MusicDirector {
public:
    // Returns true on the frame a track starts.
    bool update(AudioEngine& audio, float dt, float daylight, bool underground, bool underwater);

private:
    float m_wait = 45.0f;
    unsigned m_rng = 0x6C078965u;
};

// Thunder. Not part of the mix because it is an event on a clock rather than a gain, and a
// storm that rumbles on a fixed interval stops being weather after the second one.
class StormSounds {
public:
    // Returns true on the frames it actually fired, which is what --smoke counts.
    bool update(AudioEngine& audio, float precip, float skyAccess, float dt);

    // Off when something else decides when thunder happens (the game's lightning schedules it
    // from the strikes, so it arrives after the flash). On by default: --smoke counts on it.
    bool selfSchedule = true;

private:
    // Nothing until it is genuinely coming down. Light drizzle with thunder in it reads as
    // a bug rather than as atmosphere.
    static constexpr float kMinPrecip = 0.35f;
    float m_next = 9.0f;
    unsigned m_rng = 0x6C078965u;
};

} // namespace vx
