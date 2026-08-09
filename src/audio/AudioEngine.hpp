#pragma once

#include "audio/SoundBank.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace vx {

class JobSystem;

// Sound playback over miniaudio.
//
// NOTHING ships with the engine and nothing is downloaded at build time. Clips are whatever
// the user drops into sounds/ — the same rule the block textures follow, and the reason no
// Minecraft audio is bundled here. A missing clip is not an error: it simply doesn't play,
// so a fresh clone runs silently and correctly.
//
// The PCM lives in SoundBank, decoded once at init. A play here is O(1) with no file I/O and
// no decode: the voice gets its own ma_audio_buffer_ref over the bank's shared samples, so
// the same clip can run on several voices at once without copying a frame.
//
// Positions are world-space. The listener is kept at the origin internally and sounds are
// placed relative to it, so the world's large doubles never reach miniaudio's floats — and
// because that offset changes as you walk, live voices are re-placed every update(), which
// is what stops a sound following the player around.
class AudioEngine {
public:
    // `jobs` may be null; it only parallelises the initial decode. The bank loads even when
    // there is no output device, so a headless run still has something to assert about.
    //
    // `offline` builds the engine with no device and hands the mixing to readFrames(). That
    // exists because the smoke group written alongside the sound bank asserted that clips
    // were cached and that voices were acquired, and NEITHER of those can tell whether a
    // single sample ever reaches an output — which is how the engine shipped completely
    // silent with a green test. A test that cannot hear is not a test of audio.
    bool init(const std::filesystem::path& soundsDir, JobSystem* jobs, bool offline = false);

    // Offline mixing: pulls `frames` interleaved stereo frames from the engine. Only valid
    // when init() was given offline = true. Returns frames actually produced.
    int readFrames(float* out, int frames);
    // Peak absolute sample over `frames` of mixing. The one number that answers "is this
    // thing making any sound at all".
    float renderPeak(int frames);
    void shutdown();

    // Per-frame upkeep: re-place live voices against the current listener, retire faded ones,
    // and settle the underwater filter. Must be called once a frame.
    void update(float dt);

    void setListener(const glm::dvec3& pos, const glm::vec3& forward);

    // Mixer buses. World is spatialised, ui and ambient are not.
    enum class Bus { World, Ui, Ambient };
    void setMasterVolume(float v);
    void setBusVolume(Bus bus, float v);
    // Camera eye submerged: muffles the world bus. Crossfaded, so surfacing does not click.
    void setUnderwater(bool on);

    // One-shot at a world position, attenuated and panned relative to the listener.
    void playAt(const std::string& name, const glm::dvec3& worldPos, float volume = 1.0f,
                float pitch = 1.0f);
    // One-shot with no position — UI clicks, stings.
    void play(const std::string& name, float volume = 1.0f, float pitch = 1.0f);

    // Plays only if the user actually supplied that clip, and says whether it did. Lets
    // callers try a specific name first and fall back to a generic one.
    bool playIfPresent(const std::string& name, const glm::dvec3& worldPos,
                       float volume = 1.0f, float pitch = 1.0f);
    bool has(const std::string& name) const { return m_bank.has(name); }

    // How occluded a source is, 0 = clear line of sight, 1 = fully blocked. Supplied by the
    // caller because only the world knows: AudioEngine must not depend on ChunkManager.
    void playOccluded(const std::string& name, const glm::dvec3& worldPos, float occlusion,
                      float volume = 1.0f, float pitch = 1.0f);

    bool available() const { return m_ready; }
    const SoundBank& bank() const { return m_bank; }
    int clipCount() const { return m_bank.clipCount(); }
    int variantCount() const { return m_bank.variantCount(); }
    int liveVoices() const;
    // Voices mid-fade after being stolen. Retirement is driven by the dt handed to update(),
    // not by the audio device, so unlike liveVoices() this is deterministic and testable
    // without waiting on real playback.
    int retiringVoices() const;
    // Plays rejected by the rate limiter. --smoke asserts this against a synthetic burst.
    std::uint64_t throttledCount() const { return m_throttled; }

private:
    // 32 rather than 24: voices are far cheaper now that acquiring one neither opens a file
    // nor decodes, and a fluid tick plus footsteps plus a break can overlap.
    static constexpr int kVoices = 32;
    // A stolen voice is faded out over this long instead of being uninit'd mid-waveform,
    // which is what the old code did and why stealing clicked.
    static constexpr float kStealFadeMs = 15.0f;
    // Two plays of the same clip closer together than this are the same event as far as the
    // ear is concerned; the second is dropped rather than eating a voice.
    static constexpr double kMinRepeatSec = 0.04;

    struct Voice {
        bool live = false;
        bool spatial = false;
        bool retiring = false;   // fading out after being stolen; uninit when it expires
        float retireLeft = 0.0f; // seconds
        glm::dvec3 worldPos{0.0};
    };

    int acquireVoice();
    void startVoice(int idx, const SoundBank::Sample& s, float volume, float pitch,
                    bool spatialised, float lowpass);
    bool throttled(const std::string& name);
    const SoundBank::Sample* pick(const std::string& name);

    bool m_ready = false;
    bool m_offline = false;
    glm::dvec3 m_listener{0.0};
    SoundBank m_bank;
    std::uint32_t m_rng = 0x9E3779B9u;

    Voice m_voice[kVoices] = {};
    int m_nextVoice = 0;
    std::uint64_t m_throttled = 0;

    // name -> last play time, for the repeat throttle. Small and linear on purpose: it only
    // ever holds the handful of names actually being triggered.
    struct Recent {
        std::string name;
        double when = 0.0;
    };
    std::vector<Recent> m_recent;
    double m_clock = 0.0;

    bool m_underwater = false;
    float m_wet = 0.0f; // 0 dry .. 1 fully submerged, crossfaded in update()

    // miniaudio types are opaque here so its 90k-line header stays out of every TU that
    // wants to make a noise. Constructed in init().
    void* m_engine = nullptr;
    void* m_sounds = nullptr;  // ma_sound[kVoices]
    void* m_refs = nullptr;    // ma_audio_buffer_ref[kVoices], one cursor per voice
    void* m_groups = nullptr;  // ma_sound_group[3], indexed by Bus
    void* m_lowpass = nullptr; // ma_lpf_node on the world bus
};

} // namespace vx
