#pragma once

#include <glm/glm.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace vx {

// Sound playback over miniaudio.
//
// NOTHING ships with the engine and nothing is downloaded at build time. Clips are
// whatever the user drops into sounds/ — the same rule the block textures follow, and
// the reason no Minecraft audio is bundled here. A missing clip is not an error: it
// simply doesn't play, so a fresh clone runs silently and correctly.
//
// Clips are addressed by filename stem, so sounds/break_stone.wav is "break_stone".
// A trailing digit marks a variant — step_grass1.wav, step_grass2.wav, step_grass3.wav
// all register as "step_grass" and one is chosen at random per play, which is what keeps
// footsteps from sounding like a machine.
//
// Positions are world-space; the listener is kept at the origin internally and sounds are
// placed relative to it, so the world's large doubles never reach miniaudio's floats.
class AudioEngine {
public:
    bool init(const std::filesystem::path& soundsDir);
    void shutdown();

    void setListener(const glm::dvec3& pos, const glm::vec3& forward);
    void setMasterVolume(float v);

    // One-shot at a world position: attenuated and panned relative to the listener.
    void playAt(const std::string& name, const glm::dvec3& worldPos, float volume = 1.0f,
                float pitch = 1.0f);
    // One-shot with no position — UI clicks, ambience.
    void play(const std::string& name, float volume = 1.0f, float pitch = 1.0f);

    // Plays only if the user actually supplied that clip, and says whether it did. Lets
    // callers try a specific name first and fall back to a generic one.
    bool playIfPresent(const std::string& name, const glm::dvec3& worldPos,
                       float volume = 1.0f, float pitch = 1.0f);
    bool has(const std::string& name) const { return find(name) != nullptr; }

    bool available() const { return m_ready; }
    int clipCount() const { return int(m_clips.size()); }
    int variantCount() const;

private:
    struct Clip {
        std::string name;                     // stem with any trailing digit removed
        std::vector<std::filesystem::path> variants;
    };

    const Clip* find(const std::string& name) const;
    // Grabs a voice that has finished, or steals the oldest if every one is busy.
    int acquireVoice();

    static constexpr int kVoices = 24;

    bool m_ready = false;
    glm::dvec3 m_listener{0.0};
    std::vector<Clip> m_clips;
    std::uint32_t m_rng = 0x9E3779B9u;
    // miniaudio types are opaque here so its 90k-line header stays out of every TU that
    // wants to make a noise. sized/aligned storage, constructed in init().
    void* m_engine = nullptr;
    void* m_voices = nullptr;
    bool m_voiceLive[kVoices] = {};
    int m_nextVoice = 0;
};

} // namespace vx
