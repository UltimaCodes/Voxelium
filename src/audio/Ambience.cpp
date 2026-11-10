#include "audio/Ambience.hpp"

#include "audio/AudioEngine.hpp"

#include <algorithm>
#include <cmath>

namespace vx {

namespace {

float smoothstep(float a, float b, float x) {
    const float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

float clamp01(float x) { return std::clamp(x, 0.0f, 1.0f); }

} // namespace

BedMix mixAmbience(const AmbienceState& s) {
    BedMix m;

    // Underwater is not a layer on top of the surface world, it replaces it. Anything else
    // and you hear crickets through six feet of lake.
    if (s.underwater) {
        m.underwater = 1.0f;
        return m;
    }

    // How enclosed the ear is: the voxel sky light answers it exactly and for free. Depth is the
    // second half, because a shallow overhang is not a cave.
    const float enclosed = 1.0f - smoothstep(0.15f, 0.75f, s.skyAccess);
    const float below = smoothstep(3.0f, 16.0f, s.depth);
    const float cave = enclosed * below;
    m.cave = cave;
    m.drips = cave * 0.8f;

    // Everything outdoors fades as you go in, so a cave mouth is a place where you hear both.
    const float outside = 1.0f - cave;
    const float height = std::max(0.0f, -s.depth);
    // Up in the air the ground's sounds thin out but never go: the world is still down there.
    const float ground = outside * (1.0f - 0.7f * smoothstep(12.0f, 150.0f, height));
    const float sky = clamp01(s.skyAccess);
    const float dayPart = smoothstep(0.2f, 0.55f, s.daylight);
    const float nightPart = 1.0f - smoothstep(0.02f, 0.30f, s.daylight);

    m.rain = s.precip * sky;
    // rain washes out the quiet beds rather than stacking on top of them
    const float dry = 1.0f - 0.65f * m.rain;
    const float snow = clamp01(s.snow), desert = clamp01(s.desert);

    m.forest = ground * dayPart * dry * clamp01(s.forest);
    m.open = ground * dayPart * dry * clamp01(s.open);
    m.taiga = ground * dayPart * dry * clamp01(s.conifer);
    m.jungle = ground * dry * clamp01(s.jungle) * (0.55f + 0.45f * dayPart);
    m.swamp = ground * dry * clamp01(s.swamp + 0.5f * s.nearWater * (1.0f - snow)) * (0.4f + 0.6f * nightPart);
    m.night = ground * nightPart * dry * (1.0f - snow) * (1.0f - 0.6f * desert) * (1.0f - 0.5f * clamp01(s.jungle));
    m.desert = ground * desert;
    m.snow = outside * snow * (0.5f + 0.5f * sky);

    // A breeze anywhere under open sky, stronger the higher you are and the faster you go.
    m.wind = sky * clamp01(0.10f + 0.55f * smoothstep(10.0f, 160.0f, height) + 0.45f * smoothstep(10.0f, 45.0f, s.speed) +
                           0.25f * clamp01(s.alpine));
    m.leaves = clamp01(s.canopy) * sky * outside;
    m.surf = clamp01(s.coast) * outside * (0.45f + 0.55f * sky) * (1.0f - 0.5f * smoothstep(20.0f, 150.0f, height));
    m.water = clamp01(s.nearWater) * (0.35f + 0.65f * outside);
    return m;
}

void applyAmbience(AudioEngine& audio, const BedMix& mix, float master) {
    const float k = std::clamp(master, 0.0f, 1.0f);
    audio.setBed("amb_forest", mix.forest * k);
    audio.setBed("amb_open", mix.open * k);
    audio.setBed("amb_taiga", mix.taiga * k);
    audio.setBed("amb_jungle", mix.jungle * k);
    audio.setBed("amb_swamp", mix.swamp * k);
    audio.setBed("amb_night", mix.night * k);
    audio.setBed("amb_desert", mix.desert * k);
    audio.setBed("amb_snow", mix.snow * k);
    audio.setBed("amb_wind", mix.wind * k);
    audio.setBed("amb_leaves", mix.leaves * k);
    audio.setBed("amb_surf", mix.surf * k);
    audio.setBed("amb_water", mix.water * k);
    audio.setBed("amb_rain", mix.rain * k);
    audio.setBed("amb_cave", mix.cave * k);
    audio.setBed("amb_drips", mix.drips * k);
    audio.setBed("amb_underwater", mix.underwater * k);
    audio.endUnsetBeds();
}

float AmbientCalls::rand01() {
    m_rng = m_rng * 1664525u + 1013904223u;
    return float(m_rng >> 8) / float(1 << 24);
}

int AmbientCalls::update(AudioEngine& audio, const AmbienceState& s, const glm::dvec3& ear, float dt) {
    if (s.underwater)
        return 0;
    const float outside = 1.0f - (1.0f - smoothstep(0.15f, 0.75f, s.skyAccess)) * smoothstep(3.0f, 16.0f, s.depth);
    if (outside < 0.3f)
        return 0;
    const float height = std::max(0.0f, -s.depth);
    const float ground = outside * (1.0f - 0.8f * smoothstep(10.0f, 90.0f, height));
    const float day = smoothstep(0.2f, 0.55f, s.daylight);
    const float night = 1.0f - smoothstep(0.02f, 0.30f, s.daylight);
    const float dry = 1.0f - 0.8f * clamp01(s.precip);
    struct Kind {
        const char* clip;
        float weight;   // how much of this place it belongs to, 0..1
        float every;    // mean seconds between calls at full weight
        float nearest, farthest, low, high, volume;
    };
    const Kind kinds[kKinds] = {
        {"bird", ground * day * dry * clamp01(s.forest + 0.6f * s.open + 0.7f * s.conifer + s.jungle + 0.4f * s.swamp), 6.0f, 8.0f, 30.0f, 3.0f, 12.0f, 0.8f},
        {"woodpecker", ground * day * dry * clamp01(s.forest + s.conifer), 55.0f, 14.0f, 36.0f, 2.0f, 8.0f, 0.75f},
        {"crow", ground * day * dry * clamp01(0.6f * s.open + 0.4f * s.forest + 0.4f * s.conifer + 0.3f * s.snow), 45.0f, 18.0f, 40.0f, 6.0f, 18.0f, 0.7f},
        {"owl", ground * night * dry * clamp01(s.forest + s.conifer + 0.5f * s.swamp), 28.0f, 12.0f, 34.0f, 5.0f, 14.0f, 0.8f},
        {"hawk", outside * day * dry * clamp01(s.alpine + 0.4f * s.open + 0.3f * s.desert + smoothstep(30.0f, 120.0f, height)), 70.0f, 28.0f, 46.0f, 18.0f, 34.0f, 0.65f},
        {"gull", outside * day * clamp01(s.coast), 9.0f, 12.0f, 40.0f, 4.0f, 20.0f, 0.7f},
        {"frog", ground * dry * clamp01(s.swamp + 0.6f * s.nearWater) * (0.3f + 0.7f * night) * (1.0f - clamp01(s.snow)), 3.5f, 5.0f, 18.0f, -2.0f, 0.5f, 0.6f},
        {"insect", ground * day * dry * clamp01(s.open + s.swamp + s.jungle) * (1.0f - clamp01(s.snow)), 32.0f, 1.5f, 4.0f, 0.3f, 1.8f, 0.45f},
    };
    int fired = 0;
    for (int k = 0; k < kKinds; ++k) {
        const Kind& c = kinds[k];
        if (c.weight < 0.05f) {
            m_next[k] = std::max(m_next[k], c.every * 0.3f); // not here: start fresh when it is
            continue;
        }
        m_next[k] -= dt * c.weight;
        if (m_next[k] > 0.0f)
            continue;
        // exponential-ish spacing, never shorter than a third of the mean
        m_next[k] = c.every * (0.35f - std::log(std::max(rand01(), 0.02f)) * 0.65f);
        const float a = rand01() * 6.2831853f;
        const float d = c.nearest + (c.farthest - c.nearest) * rand01();
        const float up = c.low + (c.high - c.low) * rand01();
        const glm::dvec3 at = ear + glm::dvec3(std::cos(a) * d, up, std::sin(a) * d);
        if (audio.playCall(c.clip, at, c.volume * (0.75f + 0.25f * rand01()), 0.93f + 0.14f * rand01()))
            ++fired;
    }
    return fired;
}

bool MusicDirector::update(AudioEngine& audio, float dt, float daylight, bool underground, bool underwater) {
    if (audio.musicPlaying() || underwater)
        return false;
    m_wait -= dt;
    if (m_wait > 0.0f)
        return false;
    m_rng = m_rng * 1664525u + 1013904223u;
    const float r = float(m_rng >> 8) / float(1 << 24);
    // the quiet between tracks, counted once one has finished
    m_wait = 240.0f + 300.0f * r;
    const char* pool = underground && audio.streams("music_cave") ? "music_cave"
                       : daylight < 0.3f && audio.streams("music_night") ? "music_night"
                                                                         : "music_day";
    return audio.playMusic(pool, 0.85f, 6.0f);
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
    // AMBIENT bus, not the interface one, it is weather, so it rides the ambience slider
    // and ducks under the submerged filter with everything else outdoors.
    audio.play("thunder", 0.25f + 0.55f * std::clamp(skyAccess, 0.0f, 1.0f) * precip,
               0.9f + 0.2f * r, AudioEngine::Bus::Ambient);
    return true;
}

} // namespace vx
