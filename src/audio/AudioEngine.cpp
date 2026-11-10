#include "audio/AudioEngine.hpp"

#include "core/Log.hpp"

// Vorbis through stb_vorbis, the way miniaudio documents: its header before miniaudio's body, its
// body after.
#define STB_VORBIS_HEADER_ONLY
#include <stb_vorbis.c>
#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_ENCODING   // playback only
#define MA_NO_GENERATION // no synthesised waveforms
#include <miniaudio.h>
#undef STB_VORBIS_HEADER_ONLY
#include <stb_vorbis.c>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

namespace vx {

namespace {

ma_engine* eng(void* p) { return static_cast<ma_engine*>(p); }
ma_sound* sounds(void* p) { return static_cast<ma_sound*>(p); }
ma_audio_buffer_ref* refs(void* p) { return static_cast<ma_audio_buffer_ref*>(p); }
ma_sound_group* groups(void* p) { return static_cast<ma_sound_group*>(p); }

constexpr int kBusCount = 4;

// Attenuation. miniaudio's defaults are inverse/rolloff 1/min-distance 1, which puts a
// source ten blocks away at about a tenth volume, everything more than a few blocks off
// was inaudible and nothing in the engine ever set them. These are tuned by ear against
// footsteps and block breaks at walking distance: silent by ~48 blocks, no falloff at all
// inside 2 so standing on top of a source does not blow up.
constexpr float kMinDistance = 2.0f;
constexpr float kMaxDistance = 48.0f;
constexpr float kRolloff = 1.15f;

ma_lpf_node* lpf(void* p) { return static_cast<ma_lpf_node*>(p); }

// Underwater muffle. A real low-pass on the world and ambient buses, because ducking the
// volume is not what water does, it takes the top off, and the difference between those two
// is the entire effect. 700 Hz reads as submerged without sounding broken; 18 kHz is the dry
// end, high enough to be inaudible and far enough from Nyquist to stay well-behaved.
constexpr double kWetCutoff = 700.0;
constexpr double kDryCutoff = 18000.0;
constexpr float kWetLerpPerSec = 6.5f; // ~150 ms to cross over
// Coefficients are only recomputed when the cutoff moves a step, rather than every frame.
// Sixteen steps across the crossfade is smooth to the ear and costs sixteen biquad updates
// instead of one per frame forever.
constexpr int kWetSteps = 16;

// Streamed beds hand over to the next file over this long, and every recording is brought to
// about this loudness (sounds/levels.txt has each file's measured RMS). Beds sit well under the
// music and the world's own sounds: they are the room, not the thing in it.
constexpr float kBedCrossfadeSec = 6.0f;
constexpr float kBedTargetDb = -29.0f;
constexpr float kMusicTargetDb = -21.0f;

bool streamedName(const std::string& stem) {
    return stem.rfind("amb_", 0) == 0 || stem.rfind("music_", 0) == 0 || stem.rfind("menu_", 0) == 0;
}

} // namespace

bool AudioEngine::init(const std::filesystem::path& soundsDir, JobSystem* jobs,
                       bool offline) {
    // Bank first, and unconditionally. The old code returned before ever scanning the
    // directory when there was no output device, which meant the decode path could not be
    // exercised headless, so --smoke could never assert anything about it.
    m_bank.load(soundsDir, jobs);

    // the streamed files and how loud each one is
    {
        std::map<std::string, float> rms;
        std::ifstream lv(soundsDir / "levels.txt");
        for (std::string line; std::getline(lv, line);) {
            std::istringstream ls(line);
            std::string file;
            float db = 0.0f;
            if (ls >> file >> db)
                rms[file] = db;
        }
        std::error_code ec;
        std::vector<std::filesystem::path> found;
        for (const auto& e : std::filesystem::directory_iterator(soundsDir, ec)) {
            const std::string ext = e.path().extension().string();
            if (e.is_regular_file() && (ext == ".mp3" || ext == ".ogg" || ext == ".flac" || ext == ".wav") &&
                streamedName(e.path().stem().string()))
                found.push_back(e.path());
        }
        std::sort(found.begin(), found.end());
        for (const auto& f : found) {
            const std::string name = SoundBank::baseName(f.stem().string());
            auto it = std::find_if(m_streams.begin(), m_streams.end(), [&](const Stream& s) { return s.name == name; });
            if (it == m_streams.end()) {
                m_streams.push_back({name, {}, {}});
                it = m_streams.end() - 1;
            }
            it->files.push_back(f);
            const auto r = rms.find(f.filename().string());
            const float target = name.rfind("amb_", 0) == 0 ? kBedTargetDb : kMusicTargetDb;
            it->trim.push_back(r == rms.end() ? 1.0f : std::clamp(std::pow(10.0f, (target - r->second) / 20.0f), 0.1f, 16.0f));
        }
        if (!m_streams.empty())
            VX_INFO("audio: %d streamed sound(s) from %d file(s)", int(m_streams.size()), int(found.size()));
    }

    m_engine = new ma_engine{};
    ma_engine_config cfg = ma_engine_config_init();
    // PIN THE RATE, both online and off. A ma_audio_buffer_ref reports a sample rate of 0
    // (miniaudio hardcodes it, see ma_audio_buffer_ref_init), and an engine node reading 0
    // assumes the PCM is already at the engine's own rate. Left to itself the engine takes
    // the device's native rate, so on a machine whose output is not 48 kHz every clip in the
    // bank plays at the wrong speed. Measured against the real device: a 0.373 s footstep
    // takes 0.408 s at 44.1 kHz and 0.183 s at 96 kHz, an octave up, which is a click
    // rather than a footstep. Requesting the rate costs one resample in the device layer and
    // makes the buffer-ref assumption true.
    cfg.sampleRate = SoundBank::kSampleRate;
    if (offline) {
        // No device: nothing is opened, and the caller drives mixing through readFrames().
        cfg.noDevice = MA_TRUE;
        cfg.channels = 2;
        m_offline = true;
    }
    if (ma_engine_init(&cfg, eng(m_engine)) != MA_SUCCESS) {
        // no audio device (headless, no drivers) is not fatal, the engine plays silently
        delete eng(m_engine);
        m_engine = nullptr;
        VX_INFO("audio: no output device, running silent (%d clip(s) still loaded)",
                m_bank.clipCount());
        return false;
    }
    if (!offline) {
        // The owner reported total silence that no offline test could reproduce, and an
        // offline engine bypasses device selection entirely. This line is the first thing to
        // look at if it happens again: it names the backend, the device and the negotiated
        // format, none of which the harness can see.
        const ma_device* dev = ma_engine_get_device(eng(m_engine));
        VX_INFO("audio: %s | %s | %u ch @ %u Hz (engine %u Hz)",
                ma_get_backend_name(dev->pContext->backend), dev->playback.name,
                dev->playback.channels, dev->sampleRate,
                ma_engine_get_sample_rate(eng(m_engine)));
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
    ma_sound_group_set_spatialization_enabled(&groups(m_groups)[int(Bus::Music)], MA_FALSE);

    // The submerged filter, spliced between those two buses and the endpoint. UI deliberately
    // stays dry: a menu click is not in the water with you.
    m_lowpass = new ma_lpf_node{};
    ma_lpf_node_config lpfCfg = ma_lpf_node_config_init(
        ma_engine_get_channels(eng(m_engine)), ma_engine_get_sample_rate(eng(m_engine)),
        kDryCutoff, 2);
    if (ma_lpf_node_init(ma_engine_get_node_graph(eng(m_engine)), &lpfCfg, nullptr,
                         lpf(m_lowpass)) != MA_SUCCESS) {
        // Not fatal. Without it the world simply does not muffle underwater, which is worth
        // far less than refusing to make any sound at all.
        VX_ERROR("audio: no underwater filter (buses run dry)");
        delete lpf(m_lowpass);
        m_lowpass = nullptr;
    } else {
        ma_node_attach_output_bus(lpf(m_lowpass), 0,
                                  ma_engine_get_endpoint(eng(m_engine)), 0);
        ma_node_attach_output_bus(&groups(m_groups)[int(Bus::World)], 0, lpf(m_lowpass), 0);
        ma_node_attach_output_bus(&groups(m_groups)[int(Bus::Ambient)], 0, lpf(m_lowpass), 0);
    }

    m_sounds = new ma_sound[kVoices]{};
    m_refs = new ma_audio_buffer_ref[kVoices]{};
    m_bedSounds = new ma_sound[kBeds * 2]{};
    m_bedRefs = new ma_audio_buffer_ref[kBeds]{};
    m_musicSounds = new ma_sound[2]{};
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
    for (int i = 0; i < kBeds; ++i) {
        Bed& b = m_bed[i];
        if (b.name.empty())
            continue;
        if (b.streamed) {
            ma_sound_uninit(&sounds(m_bedSounds)[2 * i + b.live]);
            if (b.handing)
                ma_sound_uninit(&sounds(m_bedSounds)[2 * i + (b.live ^ 1)]);
        } else {
            ma_sound_uninit(&sounds(m_bedSounds)[2 * i]);
            ma_audio_buffer_ref_uninit(&refs(m_bedRefs)[i]);
        }
        b = {};
    }
    for (int k = 0; k < 2; ++k)
        if (m_track[k].live) {
            ma_sound_uninit(&sounds(m_musicSounds)[k]);
            m_track[k] = {};
        }
    for (int i = 0; i < kBusCount; ++i)
        ma_sound_group_uninit(&groups(m_groups)[i]);
    if (m_lowpass != nullptr) {
        ma_lpf_node_uninit(lpf(m_lowpass), nullptr);
        delete lpf(m_lowpass);
        m_lowpass = nullptr;
    }
    ma_engine_uninit(eng(m_engine));
    delete[] sounds(m_sounds);
    delete[] refs(m_refs);
    delete[] sounds(m_bedSounds);
    delete[] refs(m_bedRefs);
    delete[] sounds(m_musicSounds);
    m_musicSounds = nullptr;
    delete[] groups(m_groups);
    delete eng(m_engine);
    m_sounds = nullptr;
    m_refs = nullptr;
    m_bedSounds = nullptr;
    m_bedRefs = nullptr;
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

void AudioEngine::setBed(const std::string& name, float gain, Bus bus) {
    if (!m_ready)
        return;
    gain = std::clamp(gain, 0.0f, 1.0f);
    for (int i = 0; i < kBeds; ++i) {
        if (m_bed[i].name != name)
            continue;
        m_bed[i].target = gain;
        m_bed[i].named = true;
        return;
    }
    // Not running. A bed asked for at zero is not worth a voice, that is the common case
    // every frame for every bed the player is nowhere near.
    if (gain <= 0.0f)
        return;
    if (const Stream* st = findStream(name)) {
        for (int i = 0; i < kBeds; ++i) {
            if (!m_bed[i].name.empty())
                continue;
            const int v = pickVariant(*st, -1);
            if (!openStream(&sounds(m_bedSounds)[2 * i], *st, v, bus))
                return;
            Bed& b = m_bed[i];
            b = {};
            b.name = name;
            b.target = gain;
            b.named = true;
            b.streamed = true;
            b.bus = bus;
            b.variant[0] = v;
            b.trim[0] = st->trim[std::size_t(v)];
            return;
        }
        return;
    }
    const SoundBank::Sample* s = pick(name);
    if (s == nullptr)
        return; // the user has not supplied this bed; silence is the documented behaviour
    for (int i = 0; i < kBeds; ++i) {
        if (!m_bed[i].name.empty())
            continue;
        ma_audio_buffer_ref* ref = &refs(m_bedRefs)[i];
        if (ma_audio_buffer_ref_init(ma_format_f32, s->channels, s->pcm.data(), s->frames,
                                     ref) != MA_SUCCESS)
            return;
        ma_sound* snd = &sounds(m_bedSounds)[2 * i];
        if (ma_sound_init_from_data_source(eng(m_engine), ref,
                                           ma_uint32(MA_SOUND_FLAG_NO_SPATIALIZATION),
                                           &groups(m_groups)[int(bus)],
                                           snd) != MA_SUCCESS) {
            ma_audio_buffer_ref_uninit(ref);
            return;
        }
        ma_sound_set_looping(snd, MA_TRUE);
        // Starts silent and fades up. Punching a bed in at full gain is audible as a click
        // even when the clip itself starts on a zero crossing.
        ma_sound_set_volume(snd, 0.0f);
        ma_sound_start(snd);
        m_bed[i].name = name;
        m_bed[i].gain = 0.0f;
        m_bed[i].target = gain;
        m_bed[i].named = true;
        return;
    }
    // Every slot is busy. Dropping the request beats evicting a bed that is currently
    // audible; the director restates its gains every frame, so it lands as soon as one frees.
}

void AudioEngine::endUnsetBeds() {
    for (int i = 0; i < kBeds; ++i) {
        if (!m_bed[i].name.empty() && !m_bed[i].named)
            m_bed[i].target = 0.0f;
        m_bed[i].named = false;
    }
}

int AudioEngine::liveBeds() const {
    int n = 0;
    for (int i = 0; i < kBeds; ++i)
        if (!m_bed[i].name.empty())
            ++n;
    return n;
}

float AudioEngine::bedGain(const std::string& name) const {
    for (int i = 0; i < kBeds; ++i)
        if (m_bed[i].name == name)
            return m_bed[i].gain;
    return 0.0f;
}

const AudioEngine::Stream* AudioEngine::findStream(const std::string& name) const {
    for (const Stream& s : m_streams)
        if (s.name == name)
            return &s;
    return nullptr;
}

int AudioEngine::pickVariant(const Stream& st, int avoid) {
    const int n = int(st.files.size());
    if (n <= 1)
        return 0;
    m_rng = m_rng * 1664525u + 1013904223u;
    int v = int((m_rng >> 8) % std::uint32_t(n - 1));
    if (avoid >= 0 && v >= avoid)
        ++v;
    return std::min(v, n - 1);
}

bool AudioEngine::openStream(void* sound, const Stream& st, int variant, Bus bus) {
    if (variant < 0 || variant >= int(st.files.size()))
        return false;
    ma_sound* snd = static_cast<ma_sound*>(sound);
    // streamed from disk and opened on miniaudio's own thread, so starting one never stalls a frame
    const ma_uint32 flags = MA_SOUND_FLAG_STREAM | MA_SOUND_FLAG_NO_SPATIALIZATION | MA_SOUND_FLAG_ASYNC;
#ifdef _WIN32
    const ma_result r = ma_sound_init_from_file_w(eng(m_engine), st.files[std::size_t(variant)].c_str(), flags,
                                                  &groups(m_groups)[int(bus)], nullptr, snd);
#else
    const ma_result r = ma_sound_init_from_file(eng(m_engine), st.files[std::size_t(variant)].c_str(), flags,
                                                &groups(m_groups)[int(bus)], nullptr, snd);
#endif
    if (r != MA_SUCCESS) {
        VX_ERROR("audio: could not stream %s", st.files[std::size_t(variant)].string().c_str());
        return false;
    }
    ma_sound_set_volume(snd, 0.0f);
    ma_sound_start(snd);
    return true;
}

bool AudioEngine::playMusic(const std::string& name, float volume, float fadeSec) {
    if (!m_ready)
        return false;
    const Stream* st = findStream(name);
    if (st == nullptr)
        return false;
    stopMusic(fadeSec);
    const int k = m_track[0].live ? 1 : 0;
    if (m_track[k].live) {
        ma_sound_uninit(&sounds(m_musicSounds)[k]);
        m_track[k] = {};
    }
    const int v = pickVariant(*st, st->files.size() > 1 ? m_lastTrack : -1);
    if (!openStream(&sounds(m_musicSounds)[k], *st, v, Bus::Music))
        return false;
    m_lastTrack = v;
    m_track[k] = {true, 0.0f, std::clamp(volume, 0.0f, 1.0f), 1.0f / std::max(fadeSec, 0.05f),
                  st->trim[std::size_t(v)]};
    return true;
}

void AudioEngine::stopMusic(float fadeSec) {
    for (Track& t : m_track)
        if (t.live) {
            t.target = 0.0f;
            t.rate = 1.0f / std::max(fadeSec, 0.05f);
        }
}

bool AudioEngine::musicPlaying() const {
    for (int k = 0; k < 2; ++k)
        if (m_track[k].live && m_track[k].target > 0.0f)
            return true;
    return false;
}

void AudioEngine::updateMusic(float dt) {
    for (int k = 0; k < 2; ++k) {
        Track& t = m_track[k];
        if (!t.live)
            continue;
        ma_sound* s = &sounds(m_musicSounds)[k];
        const float d = t.rate * dt;
        t.gain = t.gain < t.target ? std::min(t.target, t.gain + d) : std::max(t.target, t.gain - d);
        if ((t.target <= 0.0f && t.gain <= 0.0f) || ma_sound_at_end(s)) {
            ma_sound_uninit(s);
            t = {};
            continue;
        }
        ma_sound_set_volume(s, t.gain * t.trim * (1.0f - 0.55f * m_wet));
    }
}

void AudioEngine::updateBeds(float dt) {
    const float step = std::clamp(kBedFadePerSec * dt, 0.0f, 1.0f);
    for (int i = 0; i < kBeds; ++i) {
        Bed& b = m_bed[i];
        if (b.name.empty())
            continue;
        b.gain += (b.target - b.gain) * step;
        // A bed the player has left keeps its slot until it is genuinely inaudible, so
        // stepping back through a cave mouth resumes the fade rather than restarting the loop.
        if (b.target <= 0.0f && b.gain < 0.002f) {
            if (b.streamed) {
                ma_sound_uninit(&sounds(m_bedSounds)[2 * i + b.live]);
                if (b.handing)
                    ma_sound_uninit(&sounds(m_bedSounds)[2 * i + (b.live ^ 1)]);
            } else {
                ma_sound_uninit(&sounds(m_bedSounds)[2 * i]);
                ma_audio_buffer_ref_uninit(&refs(m_bedRefs)[i]);
            }
            b = {};
            continue;
        }
        // Submerging muffles the beds too, an above-water bird bed heard from under the
        // surface at full brightness is the thing that gives away a fake underwater.
        const float wet = 1.0f - 0.55f * m_wet;
        if (!b.streamed) {
            ma_sound_set_volume(&sounds(m_bedSounds)[2 * i], b.gain * wet);
            continue;
        }
        ma_sound* cur = &sounds(m_bedSounds)[2 * i + b.live];
        if (b.lengthSec[b.live] <= 0.0f) {
            float len = 0.0f;
            if (ma_sound_get_length_in_seconds(cur, &len) == MA_SUCCESS && len > 0.0f)
                b.lengthSec[b.live] = len;
        }
        if (!b.handing) {
            const float len = b.lengthSec[b.live];
            const float xf = std::min(kBedCrossfadeSec, len * 0.3f);
            float at = 0.0f;
            ma_sound_get_cursor_in_seconds(cur, &at);
            if ((len > 0.0f && at >= len - xf) || ma_sound_at_end(cur)) {
                const Stream* st = findStream(b.name);
                const int next = b.live ^ 1;
                const int v = st ? pickVariant(*st, b.variant[b.live]) : -1;
                if (st && openStream(&sounds(m_bedSounds)[2 * i + next], *st, v, b.bus)) {
                    b.handing = true;
                    b.hand = 0.0f;
                    b.handSec = std::max(xf, 0.25f);
                    b.variant[next] = v;
                    b.trim[next] = st->trim[std::size_t(v)];
                    b.lengthSec[next] = 0.0f;
                }
            }
        } else {
            b.hand += dt / b.handSec;
            if (b.hand >= 1.0f) {
                ma_sound_uninit(cur);
                b.live ^= 1;
                b.handing = false;
            }
        }
        // equal power, so the handover neither dips nor swells
        const float out = b.handing ? std::cos(b.hand * 1.5707963f) : 1.0f;
        ma_sound_set_volume(&sounds(m_bedSounds)[2 * i + b.live], b.gain * wet * out * b.trim[b.live]);
        if (b.handing)
            ma_sound_set_volume(&sounds(m_bedSounds)[2 * i + (b.live ^ 1)],
                                b.gain * wet * std::sin(b.hand * 1.5707963f) * b.trim[b.live ^ 1]);
    }
}

void AudioEngine::update(float dt) {
    m_clock += double(dt);
    if (!m_ready)
        return;

    // Crossfade the submerged filter rather than switching it, so breaking the surface is a
    // transition and not a click.
    const float target = m_underwater ? 1.0f : 0.0f;
    m_wet += (target - m_wet) * std::clamp(kWetLerpPerSec * dt, 0.0f, 1.0f);
    if (m_lowpass != nullptr) {
        const int step = int(m_wet * kWetSteps + 0.5f);
        if (step != m_wetStep) {
            m_wetStep = step;
            // Logarithmic, because pitch is: a linear sweep spends most of its travel in a
            // region the ear reads as already closed.
            const double t = double(step) / kWetSteps;
            const double cutoff = kDryCutoff * std::pow(kWetCutoff / kDryCutoff, t);
            ma_lpf_config c = ma_lpf_config_init(ma_format_f32,
                                                 ma_engine_get_channels(eng(m_engine)),
                                                 ma_engine_get_sample_rate(eng(m_engine)),
                                                 cutoff, 2);
            ma_lpf_node_reinit(&c, lpf(m_lowpass));
        }
    }
    updateBeds(dt);
    updateMusic(dt);

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
    // on a sound mid-waveform, that is an instant amplitude step, which is a click. The
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
                             Bus bus, bool spatialised, float lowpass) {
    ma_audio_buffer_ref* ref = &refs(m_refs)[idx];
    // Per-voice cursor over the bank's shared PCM. This is the whole fix: no file is opened
    // and nothing is decoded, so a play is a pointer and a couple of stores.
    if (ma_audio_buffer_ref_init(ma_format_f32, s.channels, s.pcm.data(), s.frames, ref) !=
        MA_SUCCESS)
        return;

    ma_sound* snd = &sounds(m_sounds)[idx];
    ma_sound_group* group = &groups(m_groups)[int(bus)];
    const ma_uint32 flags = spatialised ? 0u : ma_uint32(MA_SOUND_FLAG_NO_SPATIALIZATION);
    if (ma_sound_init_from_data_source(eng(m_engine), ref, flags, group, snd) != MA_SUCCESS) {
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
    startVoice(idx, *s, volume, pitch, Bus::World, /*spatialised=*/true, muffle);
    const glm::vec3 rel(worldPos - m_listener);
    ma_sound_set_position(&sounds(m_sounds)[idx], rel.x, rel.y, rel.z);
}

bool AudioEngine::playCall(const std::string& name, const glm::dvec3& worldPos, float volume, float pitch) {
    if (!m_ready || !m_bank.has(name) || throttled(name))
        return false;
    const SoundBank::Sample* s = pick(name);
    const int idx = s ? acquireVoice() : -1;
    if (idx < 0)
        return false;
    m_voice[idx].worldPos = worldPos;
    startVoice(idx, *s, volume, pitch, Bus::Ambient, /*spatialised=*/true, m_wet);
    ma_sound* snd = &sounds(m_sounds)[idx];
    // a call carries: a bird forty blocks off is still clearly a bird
    ma_sound_set_min_distance(snd, 10.0f);
    ma_sound_set_max_distance(snd, 110.0f);
    ma_sound_set_rolloff(snd, 0.8f);
    const glm::vec3 rel(worldPos - m_listener);
    ma_sound_set_position(snd, rel.x, rel.y, rel.z);
    return true;
}

bool AudioEngine::playIfPresent(const std::string& name, const glm::dvec3& worldPos,
                                float volume, float pitch) {
    if (!m_ready || !m_bank.has(name))
        return false;
    playAt(name, worldPos, volume, pitch);
    return true;
}

void AudioEngine::play(const std::string& name, float volume, float pitch, Bus bus) {
    if (!m_ready || throttled(name))
        return;
    const SoundBank::Sample* s = pick(name);
    if (s == nullptr)
        return;
    const int idx = acquireVoice();
    if (idx < 0)
        return;
    startVoice(idx, *s, volume, pitch, bus, /*spatialised=*/false, 0.0f);
}

} // namespace vx
