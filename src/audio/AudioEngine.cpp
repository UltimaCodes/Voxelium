#include "audio/AudioEngine.hpp"

#include "core/Log.hpp"

#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_ENCODING   // playback only
#define MA_NO_GENERATION // no synthesised waveforms
#include <miniaudio.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace vx {

namespace {

ma_engine* eng(void* p) { return static_cast<ma_engine*>(p); }
ma_sound* sounds(void* p) { return static_cast<ma_sound*>(p); }
ma_audio_buffer_ref* refs(void* p) { return static_cast<ma_audio_buffer_ref*>(p); }
ma_sound_group* groups(void* p) { return static_cast<ma_sound_group*>(p); }

constexpr int kBusCount = 3;

// Attenuation. miniaudio's defaults are inverse/rolloff 1/min-distance 1, which puts a
// source ten blocks away at about a tenth volume — everything more than a few blocks off
// was inaudible and nothing in the engine ever set them. These are tuned by ear against
// footsteps and block breaks at walking distance: silent by ~48 blocks, no falloff at all
// inside 2 so standing on top of a source does not blow up.
constexpr float kMinDistance = 2.0f;
constexpr float kMaxDistance = 48.0f;
constexpr float kRolloff = 1.15f;

// Underwater muffle. 900 Hz is low enough to read as "submerged" without sounding broken.
constexpr double kWetCutoff = 900.0;
constexpr float kWetLerpPerSec = 6.5f; // ~150 ms to cross over

} // namespace

bool AudioEngine::init(const std::filesystem::path& soundsDir, JobSystem* jobs,
                       bool offline) {
    // Bank first, and unconditionally. The old code returned before ever scanning the
    // directory when there was no output device, which meant the decode path could not be
    // exercised headless — so --smoke could never assert anything about it.
    m_bank.load(soundsDir, jobs);

    m_engine = new ma_engine{};
    ma_engine_config cfg = ma_engine_config_init();
    if (offline) {
        // No device: nothing is opened, and the caller drives mixing through readFrames().
        cfg.noDevice = MA_TRUE;
        cfg.channels = 2;
        cfg.sampleRate = SoundBank::kSampleRate;
        m_offline = true;
    }
    if (ma_engine_init(&cfg, eng(m_engine)) != MA_SUCCESS) {
        // no audio device (headless, no drivers) is not fatal — the engine plays silently
        delete eng(m_engine);
        m_engine = nullptr;
        VX_INFO("audio: no output device, running silent (%d clip(s) still loaded)",
                m_bank.clipCount());
        return false;
    }

    m_groups = new ma_sound_group[kBusCount]{};
    for (int i = 0; i < kBusCount; ++i) {
        if (ma_sound_group_init(eng(m_engine), 0, nullptr, &groups(m_groups)[i]) !=
            MA_SUCCESS) {
            VX_ERROR("audio: could not create bus %d", i);
            return false;
        }
    }
    // Only the world bus is spatialised; ui and ambient are 2D by definition and would
    // otherwise pan with the listener's heading.
    ma_sound_group_set_spatialization_enabled(&groups(m_groups)[int(Bus::Ui)], MA_FALSE);
    ma_sound_group_set_spatialization_enabled(&groups(m_groups)[int(Bus::Ambient)], MA_FALSE);

    m_sounds = new ma_sound[kVoices]{};
    m_refs = new ma_audio_buffer_ref[kVoices]{};
    m_ready = true;
    return true;
}

void AudioEngine::shutdown() {
    if (!m_ready)
        return;
    for (int i = 0; i < kVoices; ++i) {
        if (m_voice[i].live) {
            ma_sound_uninit(&sounds(m_sounds)[i]);
            ma_audio_buffer_ref_uninit(&refs(m_refs)[i]);
            m_voice[i].live = false;
        }
    }
    for (int i = 0; i < kBusCount; ++i)
        ma_sound_group_uninit(&groups(m_groups)[i]);
    ma_engine_uninit(eng(m_engine));
    delete[] sounds(m_sounds);
    delete[] refs(m_refs);
    delete[] groups(m_groups);
    delete eng(m_engine);
    m_sounds = nullptr;
    m_refs = nullptr;
    m_groups = nullptr;
    m_engine = nullptr;
    m_ready = false;
}

void AudioEngine::setListener(const glm::dvec3& pos, const glm::vec3& forward) {
    m_listener = pos;
    if (!m_ready)
        return;
    // the listener stays at the origin and sounds are placed relative to it, so world
    // coordinates in the thousands never lose precision on the way into a float
    ma_engine_listener_set_position(eng(m_engine), 0, 0.0f, 0.0f, 0.0f);
    ma_engine_listener_set_direction(eng(m_engine), 0, forward.x, forward.y, forward.z);
}

void AudioEngine::setMasterVolume(float v) {
    if (m_ready)
        ma_engine_set_volume(eng(m_engine), std::clamp(v, 0.0f, 1.0f));
}

void AudioEngine::setBusVolume(Bus bus, float v) {
    if (m_ready)
        ma_sound_group_set_volume(&groups(m_groups)[int(bus)], std::clamp(v, 0.0f, 1.0f));
}

void AudioEngine::setUnderwater(bool on) { m_underwater = on; }

int AudioEngine::readFrames(float* out, int frames) {
    if (!m_ready || !m_offline)
        return 0;
    ma_uint64 got = 0;
    if (ma_engine_read_pcm_frames(eng(m_engine), out, ma_uint64(frames), &got) != MA_SUCCESS)
        return 0;
    return int(got);
}

float AudioEngine::renderPeak(int frames) {
    if (!m_ready || !m_offline)
        return 0.0f;
    std::vector<float> buf(std::size_t(frames) * 2, 0.0f);
    const int got = readFrames(buf.data(), frames);
    float peak = 0.0f;
    for (int i = 0; i < got * 2; ++i)
        peak = std::max(peak, std::abs(buf[std::size_t(i)]));
    return peak;
}

int AudioEngine::liveVoices() const {
    int n = 0;
    for (int i = 0; i < kVoices; ++i)
        if (m_voice[i].live && !m_voice[i].retiring)
            ++n;
    return n;
}

int AudioEngine::retiringVoices() const {
    int n = 0;
    for (int i = 0; i < kVoices; ++i)
        if (m_voice[i].live && m_voice[i].retiring)
            ++n;
    return n;
}

void AudioEngine::update(float dt) {
    m_clock += double(dt);
    if (!m_ready)
        return;

    // Crossfade the submerged filter rather than switching it, so breaking the surface is a
    // transition and not a click.
    const float target = m_underwater ? 1.0f : 0.0f;
    m_wet += (target - m_wet) * std::clamp(kWetLerpPerSec * dt, 0.0f, 1.0f);

    for (int i = 0; i < kVoices; ++i) {
        Voice& v = m_voice[i];
        if (!v.live)
            continue;
        ma_sound* s = &sounds(m_sounds)[i];

        if (v.retiring) {
            v.retireLeft -= dt;
            if (v.retireLeft <= 0.0f) {
                ma_sound_uninit(s);
                ma_audio_buffer_ref_uninit(&refs(m_refs)[i]);
                v.live = false;
            }
            continue;
        }
        if (ma_sound_at_end(s)) {
            ma_sound_uninit(s);
            ma_audio_buffer_ref_uninit(&refs(m_refs)[i]);
            v.live = false;
            continue;
        }
        // Re-place against the CURRENT listener. Without this a sound keeps the offset it
        // had when it started, and since the listener is pinned at the origin that means it
        // travels with the player instead of staying where it happened.
        if (v.spatial) {
            const glm::vec3 rel(v.worldPos - m_listener);
            ma_sound_set_position(s, rel.x, rel.y, rel.z);
            // Muffle submerged sources along with the bus, so a lapping shore heard from
            // under the surface is filtered too.
            if (m_wet > 0.01f)
                ma_sound_set_directional_attenuation_factor(s, 1.0f - 0.35f * m_wet);
        }
    }
}

int AudioEngine::acquireVoice() {
    // A voice that has finished or is already retiring past its fade is free.
    for (int i = 0; i < kVoices; ++i) {
        const int idx = (m_nextVoice + i) % kVoices;
        Voice& v = m_voice[idx];
        if (!v.live) {
            m_nextVoice = (idx + 1) % kVoices;
            return idx;
        }
        if (!v.retiring && ma_sound_at_end(&sounds(m_sounds)[idx])) {
            ma_sound_uninit(&sounds(m_sounds)[idx]);
            ma_audio_buffer_ref_uninit(&refs(m_refs)[idx]);
            v.live = false;
            m_nextVoice = (idx + 1) % kVoices;
            return idx;
        }
    }
    // Everything is still playing. Steal the oldest, but FADE it rather than calling uninit
    // on a sound mid-waveform — that is an instant amplitude step, which is a click. The
    // stolen voice keeps its slot until the fade expires, so find a genuinely free one; if
    // there is none, take the oldest non-retiring slot and accept the fade being cut short.
    for (int i = 0; i < kVoices; ++i) {
        const int idx = (m_nextVoice + i) % kVoices;
        Voice& v = m_voice[idx];
        if (v.retiring)
            continue;
        ma_sound_set_fade_in_milliseconds(&sounds(m_sounds)[idx], -1.0f, 0.0f,
                                          ma_uint64(kStealFadeMs));
        v.retiring = true;
        v.retireLeft = kStealFadeMs * 0.001f;
        m_nextVoice = (idx + 1) % kVoices;
        break;
    }
    // Now grab whichever slot is actually free (a retiring one that has expired, or none).
    for (int i = 0; i < kVoices; ++i) {
        const int idx = (m_nextVoice + i) % kVoices;
        if (!m_voice[idx].live) {
            m_nextVoice = (idx + 1) % kVoices;
            return idx;
        }
    }
    return -1; // every voice is mid-fade; dropping this play beats cutting one short
}

const SoundBank::Sample* AudioEngine::pick(const std::string& name) {
    const SoundBank::Clip* clip = m_bank.find(name);
    if (clip == nullptr || clip->variants.empty())
        return nullptr; // the user just hasn't supplied this one
    m_rng = m_rng * 1664525u + 1013904223u;
    return &clip->variants[m_rng % clip->variants.size()];
}

bool AudioEngine::throttled(const std::string& name) {
    for (Recent& r : m_recent) {
        if (r.name != name)
            continue;
        if (m_clock - r.when < kMinRepeatSec) {
            ++m_throttled;
            return true;
        }
        r.when = m_clock;
        return false;
    }
    m_recent.push_back({name, m_clock});
    return false;
}

void AudioEngine::startVoice(int idx, const SoundBank::Sample& s, float volume, float pitch,
                             bool spatialised, float lowpass) {
    ma_audio_buffer_ref* ref = &refs(m_refs)[idx];
    // Per-voice cursor over the bank's shared PCM. This is the whole fix: no file is opened
    // and nothing is decoded, so a play is a pointer and a couple of stores.
    if (ma_audio_buffer_ref_init(ma_format_f32, s.channels, s.pcm.data(), s.frames, ref) !=
        MA_SUCCESS)
        return;

    ma_sound* snd = &sounds(m_sounds)[idx];
    ma_sound_group* bus = &groups(m_groups)[int(spatialised ? Bus::World : Bus::Ui)];
    const ma_uint32 flags = spatialised ? 0u : ma_uint32(MA_SOUND_FLAG_NO_SPATIALIZATION);
    if (ma_sound_init_from_data_source(eng(m_engine), ref, flags, bus, snd) != MA_SUCCESS) {
        ma_audio_buffer_ref_uninit(ref);
        return;
    }

    ma_sound_set_volume(snd, std::clamp(volume, 0.0f, 1.0f) * (1.0f - 0.45f * lowpass));
    ma_sound_set_pitch(snd, std::clamp(pitch, 0.25f, 4.0f));
    if (spatialised) {
        ma_sound_set_attenuation_model(snd, ma_attenuation_model_inverse);
        ma_sound_set_min_distance(snd, kMinDistance);
        ma_sound_set_max_distance(snd, kMaxDistance);
        ma_sound_set_rolloff(snd, kRolloff);
    }
    ma_sound_start(snd);

    Voice& v = m_voice[idx];
    v.live = true;
    v.retiring = false;
    v.retireLeft = 0.0f;
    v.spatial = spatialised;
}

void AudioEngine::playAt(const std::string& name, const glm::dvec3& worldPos, float volume,
                         float pitch) {
    playOccluded(name, worldPos, 0.0f, volume, pitch);
}

void AudioEngine::playOccluded(const std::string& name, const glm::dvec3& worldPos,
                               float occlusion, float volume, float pitch) {
    if (!m_ready || throttled(name))
        return;
    const SoundBank::Sample* s = pick(name);
    if (s == nullptr)
        return;
    const int idx = acquireVoice();
    if (idx < 0)
        return;
    m_voice[idx].worldPos = worldPos;
    // Submerged listener muffles everything in the world, on top of any geometry occlusion.
    const float muffle = std::clamp(std::max(occlusion, m_wet), 0.0f, 1.0f);
    startVoice(idx, *s, volume, pitch, /*spatialised=*/true, muffle);
    const glm::vec3 rel(worldPos - m_listener);
    ma_sound_set_position(&sounds(m_sounds)[idx], rel.x, rel.y, rel.z);
}

bool AudioEngine::playIfPresent(const std::string& name, const glm::dvec3& worldPos,
                                float volume, float pitch) {
    if (!m_ready || !m_bank.has(name))
        return false;
    playAt(name, worldPos, volume, pitch);
    return true;
}

void AudioEngine::play(const std::string& name, float volume, float pitch) {
    if (!m_ready || throttled(name))
        return;
    const SoundBank::Sample* s = pick(name);
    if (s == nullptr)
        return;
    const int idx = acquireVoice();
    if (idx < 0)
        return;
    startVoice(idx, *s, volume, pitch, /*spatialised=*/false, 0.0f);
}

} // namespace vx
