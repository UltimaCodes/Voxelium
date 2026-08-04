#include "audio/AudioEngine.hpp"

#include "core/Log.hpp"

#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_ENCODING   // playback only
#define MA_NO_GENERATION // no synthesised waveforms
#include <miniaudio.h>

#include <algorithm>
#include <cctype>

namespace vx {

namespace {

ma_engine* eng(void* p) { return static_cast<ma_engine*>(p); }
ma_sound* voices(void* p) { return static_cast<ma_sound*>(p); }

// miniaudio decodes these itself. ogg would need stb_vorbis wired in, so it is
// deliberately not accepted rather than silently failing at load.
bool decodable(const std::filesystem::path& ext) {
    return ext == ".wav" || ext == ".WAV" || ext == ".mp3" || ext == ".MP3" ||
           ext == ".flac" || ext == ".FLAC";
}

// "step_grass3" -> "step_grass". variants of one sound differ only by a trailing number.
std::string baseName(std::string stem) {
    while (!stem.empty() && std::isdigit(static_cast<unsigned char>(stem.back())))
        stem.pop_back();
    return stem;
}

} // namespace

bool AudioEngine::init(const std::filesystem::path& soundsDir) {
    m_engine = new ma_engine{};
    if (ma_engine_init(nullptr, eng(m_engine)) != MA_SUCCESS) {
        // no audio device (headless, no drivers) is not fatal — the engine plays silently
        delete eng(m_engine);
        m_engine = nullptr;
        VX_INFO("audio: no output device, running silent");
        return false;
    }
    m_voices = new ma_sound[kVoices]{};
    m_ready = true;

    std::error_code ec;
    if (!std::filesystem::is_directory(soundsDir, ec)) {
        VX_INFO("audio: no %s directory — drop your own clips there to hear anything",
                soundsDir.string().c_str());
        return true;
    }

    int skipped = 0;
    for (const auto& entry : std::filesystem::directory_iterator(soundsDir, ec)) {
        if (!entry.is_regular_file(ec))
            continue;
        if (!decodable(entry.path().extension())) {
            // don't count the README as a rejected clip — only things that look like audio
            const auto ext = entry.path().extension();
            if (ext != ".md" && ext != ".txt")
                ++skipped;
            continue;
        }
        const std::string base = baseName(entry.path().stem().string());
        if (base.empty())
            continue;
        const auto it = std::find_if(m_clips.begin(), m_clips.end(),
                                     [&](const Clip& c) { return c.name == base; });
        if (it == m_clips.end())
            m_clips.push_back({base, {entry.path()}});
        else
            it->variants.push_back(entry.path());
    }
    std::sort(m_clips.begin(), m_clips.end(),
              [](const Clip& a, const Clip& b) { return a.name < b.name; });

    VX_INFO("audio: %d clip(s), %d variant file(s) from %s%s", clipCount(), variantCount(),
            soundsDir.string().c_str(),
            skipped > 0 ? " (some files skipped: wav/mp3/flac only)" : "");
    return true;
}

void AudioEngine::shutdown() {
    if (!m_ready)
        return;
    for (int i = 0; i < kVoices; ++i)
        if (m_voiceLive[i])
            ma_sound_uninit(&voices(m_voices)[i]);
    ma_engine_uninit(eng(m_engine));
    delete[] voices(m_voices);
    delete eng(m_engine);
    m_voices = nullptr;
    m_engine = nullptr;
    m_ready = false;
}

int AudioEngine::variantCount() const {
    int n = 0;
    for (const Clip& c : m_clips)
        n += int(c.variants.size());
    return n;
}

const AudioEngine::Clip* AudioEngine::find(const std::string& name) const {
    const auto it = std::find_if(m_clips.begin(), m_clips.end(),
                                 [&](const Clip& c) { return c.name == name; });
    return it == m_clips.end() ? nullptr : &*it;
}

void AudioEngine::setListener(const glm::dvec3& pos, const glm::vec3& forward) {
    if (!m_ready)
        return;
    m_listener = pos;
    // the listener stays at the origin and sounds are placed relative to it, so world
    // coordinates in the thousands never lose precision on the way into a float
    ma_engine_listener_set_position(eng(m_engine), 0, 0.0f, 0.0f, 0.0f);
    ma_engine_listener_set_direction(eng(m_engine), 0, forward.x, forward.y, forward.z);
}

void AudioEngine::setMasterVolume(float v) {
    if (m_ready)
        ma_engine_set_volume(eng(m_engine), std::clamp(v, 0.0f, 1.0f));
}

int AudioEngine::acquireVoice() {
    ma_sound* v = voices(m_voices);
    for (int i = 0; i < kVoices; ++i) {
        const int idx = (m_nextVoice + i) % kVoices;
        if (!m_voiceLive[idx] || ma_sound_at_end(&v[idx])) {
            if (m_voiceLive[idx]) {
                ma_sound_uninit(&v[idx]);
                m_voiceLive[idx] = false;
            }
            m_nextVoice = (idx + 1) % kVoices;
            return idx;
        }
    }
    // everything is still playing: steal the oldest rather than dropping the sound
    const int idx = m_nextVoice;
    ma_sound_uninit(&v[idx]);
    m_voiceLive[idx] = false;
    m_nextVoice = (idx + 1) % kVoices;
    return idx;
}

void AudioEngine::playAt(const std::string& name, const glm::dvec3& worldPos, float volume,
                         float pitch) {
    if (!m_ready)
        return;
    const Clip* clip = find(name);
    if (!clip) // the user just hasn't supplied this one
        return;

    m_rng = m_rng * 1664525u + 1013904223u;
    const auto& path = clip->variants[m_rng % clip->variants.size()];

    const int idx = acquireVoice();
    ma_sound* s = &voices(m_voices)[idx];
    const std::string p = path.string();
    if (ma_sound_init_from_file(eng(m_engine), p.c_str(), MA_SOUND_FLAG_DECODE, nullptr,
                                nullptr, s) != MA_SUCCESS)
        return;
    m_voiceLive[idx] = true;

    const glm::vec3 rel(worldPos - m_listener);
    ma_sound_set_position(s, rel.x, rel.y, rel.z);
    ma_sound_set_volume(s, std::clamp(volume, 0.0f, 1.0f));
    ma_sound_set_pitch(s, std::clamp(pitch, 0.25f, 4.0f));
    ma_sound_start(s);
}

bool AudioEngine::playIfPresent(const std::string& name, const glm::dvec3& worldPos,
                                float volume, float pitch) {
    if (!m_ready || !find(name))
        return false;
    playAt(name, worldPos, volume, pitch);
    return true;
}

void AudioEngine::play(const std::string& name, float volume, float pitch) {
    if (!m_ready)
        return;
    const Clip* clip = find(name);
    if (!clip)
        return;

    m_rng = m_rng * 1664525u + 1013904223u;
    const auto& path = clip->variants[m_rng % clip->variants.size()];

    const int idx = acquireVoice();
    ma_sound* s = &voices(m_voices)[idx];
    const std::string p = path.string();
    if (ma_sound_init_from_file(eng(m_engine), p.c_str(),
                                MA_SOUND_FLAG_DECODE | MA_SOUND_FLAG_NO_SPATIALIZATION,
                                nullptr, nullptr, s) != MA_SUCCESS)
        return;
    m_voiceLive[idx] = true;
    ma_sound_set_volume(s, std::clamp(volume, 0.0f, 1.0f));
    ma_sound_set_pitch(s, std::clamp(pitch, 0.25f, 4.0f));
    ma_sound_start(s);
}

} // namespace vx
