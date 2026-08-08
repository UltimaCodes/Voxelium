#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace vx {

class JobSystem;

// Every clip in sounds/, decoded once, held as PCM for the process lifetime.
//
// This exists because AudioEngine used to call ma_sound_init_from_file with
// MA_SOUND_FLAG_DECODE on EVERY play — open the file, decode the whole thing, synchronously,
// on the calling thread. The bundled clips are FLAC and the footstep call site is inside the
// fixed-step physics loop, so walking meant a full FLAC decode off disk on the main thread
// every 2.2 blocks. That is the audio jank; nothing else in the mixer mattered next to it.
//
// Splitting the bank out from the engine is not tidiness. It is what lets the decode path be
// tested at all: AudioEngine::init used to bail before scanning the directory when there was
// no output device, so a headless box — which is every CI box, and --smoke — could never
// reach the code. A bank with no device still loads, so the cache is assertable.
class SoundBank {
public:
    // Scans `dir` and decodes everything in it. Decodes are submitted to `jobs` so a hundred
    // clips land across every worker rather than serially; pass nullptr to decode inline.
    // Missing directory is not an error — the engine simply has nothing to play.
    void load(const std::filesystem::path& dir, JobSystem* jobs);

    struct Sample {
        std::vector<float> pcm; // interleaved, engine sample rate
        std::uint32_t frames = 0;
        std::uint32_t channels = 0;
    };

    // A name plus its variants. A trailing digit marks a variant — step_grass1/2/3 all
    // register as "step_grass" and one is chosen per play, which is what stops footsteps
    // sounding like a machine gun.
    struct Clip {
        std::string name;
        std::vector<Sample> variants;
    };

    const Clip* find(const std::string& name) const;
    bool has(const std::string& name) const { return find(name) != nullptr; }

    int clipCount() const { return int(m_clips.size()); }
    int variantCount() const;
    std::uint64_t totalFrames() const;
    // Monotonic count of file decodes performed by any bank, for the life of the process.
    // Never reset — a caller brackets the region it cares about and compares. That is what
    // makes it falsifiable: the first shape of this counter re-baselined itself inside
    // load(), so a second load cancelled out and the smoke group could not go red at all.
    static std::uint64_t decodeCount();

    // "step_grass3" -> "step_grass". Exposed for the smoke group that covers the three cases
    // this rule has to distinguish; see the comment on the definition.
    static std::string baseName(std::string stem);

private:
    std::vector<Clip> m_clips; // sorted by name; find() binary-searches
};

} // namespace vx
