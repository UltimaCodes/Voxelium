#include "audio/Ambience.hpp"

#include "audio/AudioEngine.hpp"

#include <algorithm>

namespace vx {

namespace {

float smoothstep(float a, float b, float x) {
    const float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

} // namespace

BedMix mixAmbience(const AmbienceState& s) {
    BedMix m;

    // Underwater is not a layer on top of the surface world, it replaces it. Anything else
    // and you hear crickets through six feet of lake.
    if (s.underwater) {
        m.underwater = 1.0f;
        // The deep rumble stays, quietly — it is the only thing that still reads as "there
        // is a world outside this" once the rest is gone.
        m.deep = 0.25f;
        m.reef = std::clamp(s.reef, 0.0f, 1.0f);
        return m;
    }

    // How enclosed the ear is. Sky access is the voxel sky light, which already answers this
    // exactly and for free: it is what #29's rain uses to decide whether you are under cover.
    // Depth is the second half, because a shallow overhang is not a cave.
    const float enclosed = 1.0f - smoothstep(0.15f, 0.75f, s.skyAccess);
    const float below = smoothstep(3.0f, 16.0f, s.depth);
    const float cave = enclosed * below;

    m.cave = cave;
    m.drips = cave * 0.85f;
    // A separate sub-bass layer that keeps coming up as you descend, so a deep shaft is not
    // just the same cave loop louder.
    m.deep = cave * smoothstep(20.0f, 70.0f, s.depth);

    // Everything above ground fades out as you go in, rather than being switched off — the
    // cave mouth should be a place where you hear both.
    const float outside = 1.0f - cave;
    m.day = outside * smoothstep(0.12f, 0.45f, s.daylight);
    m.night = outside * (1.0f - smoothstep(0.02f, 0.30f, s.daylight));

    // Wind needs open sky and answers to height. It is the one bed that survives a little
    // way into a cave mouth, which is what an opening sounds like from inside.
    //
    // The band is height ABOVE the local surface, not world Y — on a terrain that runs from
    // sea level to 384 an absolute threshold means "windy in the mountains and nowhere else".
    // 8 to 80 blocks up is the range a player actually builds and climbs through; the first
    // version used 70 to 200 and so never engaged at all.
    m.wind = std::clamp(s.skyAccess, 0.0f, 1.0f) *
             (0.35f + 0.65f * smoothstep(8.0f, 80.0f, -s.depth));

    // Rain is heard through an opening but not through rock.
    m.rain = s.precip * std::clamp(s.skyAccess, 0.0f, 1.0f);
    // Rain washes out the quiet beds rather than stacking on top of them.
    const float duck = 1.0f - 0.7f * m.rain;
    m.day *= duck;
    m.night *= duck;

    m.water = std::clamp(s.nearWater, 0.0f, 1.0f) * (0.35f + 0.65f * outside);

    // The living world. All of it is outdoors, so it goes with `outside` the way day and night
    // do, and rain washes out the quiet ones. Each answers to a different part of the day:
    // gulls, birds and cicadas are daytime, frogs are the night shift.
    const float open = std::clamp(s.skyAccess, 0.0f, 1.0f);
    const float dayPart = smoothstep(0.2f, 0.55f, s.daylight);
    const float nightPart = 1.0f - smoothstep(0.02f, 0.30f, s.daylight);
    const float dry = 1.0f - 0.8f * m.rain;
    const float coast = std::clamp(s.coast, 0.0f, 1.0f);
    const float forest = std::clamp(s.forest, 0.0f, 1.0f);
    const float desert = std::clamp(s.desert, 0.0f, 1.0f);
    m.surf = coast * outside * (0.45f + 0.55f * open);
    m.gulls = coast * outside * dayPart * dry;
    m.birds = forest * outside * dayPart * dry;
    m.leaves = forest * open * (0.4f + 0.6f * smoothstep(8.0f, 80.0f, -s.depth));
    m.desertWind = desert * outside;
    m.cicadas = std::clamp(desert * 0.9f + forest * 0.4f, 0.0f, 1.0f) * outside * dayPart * dry;
    m.frogs = std::clamp(s.nearWater, 0.0f, 1.0f) * (0.3f + 0.7f * forest) * outside *
              nightPart * dry;
    return m;
}

bool StormSounds::update(AudioEngine& audio, float precip, float skyAccess, float dt) {
    if (!selfSchedule)
        return false;
    if (precip < kMinPrecip) {
        m_next = 9.0f; // leaving a storm resets the wait, so re-entering one is not instant
        return false;
    }
    m_next -= dt * (0.6f + precip); // harder rain, more of it
    if (m_next > 0.0f)
        return false;
    m_rng = m_rng * 1664525u + 1013904223u;
    const float r = float(m_rng >> 8) / float(1 << 24);
    m_next = 11.0f + 26.0f * r;
    // Thunder is 2D: it is the whole sky, so panning it to a point is wrong. Rolled off by
    // how much sky the ear can see, because underground you should barely hear it. On the
    // AMBIENT bus, not the interface one — it is weather, so it rides the ambience slider
    // and ducks under the submerged filter with everything else outdoors.
    audio.play("thunder", 0.25f + 0.55f * std::clamp(skyAccess, 0.0f, 1.0f) * precip,
               0.9f + 0.2f * r, AudioEngine::Bus::Ambient);
    return true;
}

void applyAmbience(AudioEngine& audio, const BedMix& mix, float master) {
    const float k = std::clamp(master, 0.0f, 1.0f);
    audio.setBed("amb_day", mix.day * k);
    audio.setBed("amb_night", mix.night * k);
    audio.setBed("amb_wind", mix.wind * k);
    audio.setBed("amb_cave", mix.cave * k);
    audio.setBed("amb_deep", mix.deep * k);
    audio.setBed("amb_underwater", mix.underwater * k);
    audio.setBed("amb_rain", mix.rain * k);
    audio.setBed("amb_water", mix.water * k);
    audio.setBed("amb_surf", mix.surf * k);
    audio.setBed("amb_gulls", mix.gulls * k);
    audio.setBed("amb_birds", mix.birds * k);
    audio.setBed("amb_leaves", mix.leaves * k);
    audio.setBed("amb_desert", mix.desertWind * k);
    audio.setBed("amb_cicadas", mix.cicadas * k);
    audio.setBed("amb_frogs", mix.frogs * k);
    audio.setBed("amb_drips", mix.drips * k);
    audio.setBed("amb_reef", mix.reef * k);
    audio.endUnsetBeds();
}

} // namespace vx
