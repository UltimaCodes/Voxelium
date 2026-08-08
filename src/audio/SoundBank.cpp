#include "audio/SoundBank.hpp"

#include "core/Log.hpp"
#include "jobs/JobSystem.hpp"

// miniaudio's implementation is compiled in AudioEngine.cpp; this TU only needs the decoder
// declarations. Keeping MINIAUDIO_IMPLEMENTATION to one TU is what stops its 90k-line body
// being compiled twice. The two feature macros must match the implementation TU exactly, or
// the declarations here describe a library that was not built.
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#include <miniaudio.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <mutex>

namespace vx {

namespace {

// The engine's mixing format. Decoding to it up front means a voice never converts at play
// time — the whole point of the bank.
constexpr ma_uint32 kSampleRate = 48000;
constexpr ma_uint32 kChannels = 1; // mono: spatialisation needs a point source anyway

// Monotonic for the process lifetime; see the header on why it is never reset.
std::atomic<std::uint64_t> g_decodes{0};

// miniaudio decodes these itself. ogg would need stb_vorbis wired in, so it is deliberately
// not accepted rather than silently failing at load.
bool decodable(const std::filesystem::path& ext) {
    return ext == ".wav" || ext == ".WAV" || ext == ".mp3" || ext == ".MP3" ||
           ext == ".flac" || ext == ".FLAC";
}

} // namespace

// "step_grass3" -> "step_grass". Variants of one sound differ only by a trailing number.
//
// The catch is that block sounds are looked up by numeric id — "break_3" for grass — so
// stripping every trailing digit turned that into "break_", which nothing ever asked for.
// Per-block sounds therefore never resolved at all; only the generic fallback did. Three
// cases, distinguished by what sits in front of the digits:
//   step_grass3  -> step_grass   an ordinary variant
//   break_3      -> break_3      the digits ARE the name, so keep them
//   step_1-2     -> step_1       '-' marks a variant of a numbered name explicitly
std::string SoundBank::baseName(std::string stem) {
    std::size_t end = stem.size();
    while (end > 0 && std::isdigit(static_cast<unsigned char>(stem[end - 1])))
        --end;
    if (end == stem.size())
        return stem; // no trailing digits at all
    if (end > 0 && stem[end - 1] == '-')
        return stem.substr(0, end - 1);
    if (end > 0 && stem[end - 1] == '_')
        return stem; // "_3" is a block id, not a variant index
    return stem.substr(0, end);
}

void SoundBank::load(const std::filesystem::path& dir, JobSystem* jobs) {
    m_clips.clear();

    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) {
        VX_INFO("audio: no %s directory — drop your own clips there to hear anything",
                dir.string().c_str());
        return;
    }

    // Gather first, decode second. The grouping is order-dependent (variants attach to the
    // clip that already exists) and doing it on the main thread keeps that deterministic
    // regardless of how the decode jobs interleave.
    struct Pending {
        std::filesystem::path path;
        std::size_t clip;
        std::size_t variant;
    };
    std::vector<Pending> pending;
    int skipped = 0;

    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
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
        auto it = std::find_if(m_clips.begin(), m_clips.end(),
                               [&](const Clip& c) { return c.name == base; });
        if (it == m_clips.end()) {
            m_clips.push_back({base, {}});
            it = m_clips.end() - 1;
        }
        const std::size_t ci = std::size_t(it - m_clips.begin());
        it->variants.emplace_back();
        pending.push_back({entry.path(), ci, it->variants.size() - 1});
    }

    // Decode. Each job owns one variant slot, so no two jobs touch the same memory and the
    // vector is never resized under them — it was sized above.
    const auto decodeOne = [this](const Pending& p) {
        ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, kChannels, kSampleRate);
        ma_uint64 frames = 0;
        void* raw = nullptr;
        const std::string path = p.path.string();
        if (ma_decode_file(path.c_str(), &cfg, &frames, &raw) != MA_SUCCESS || raw == nullptr) {
            VX_ERROR("audio: could not decode %s", path.c_str());
            return;
        }
        Sample& s = m_clips[p.clip].variants[p.variant];
        const float* f = static_cast<const float*>(raw);
        s.pcm.assign(f, f + std::size_t(frames) * kChannels);
        s.frames = std::uint32_t(frames);
        s.channels = kChannels;
        ma_free(raw, nullptr);
        g_decodes.fetch_add(1, std::memory_order_relaxed);
    };

    if (jobs != nullptr && pending.size() > 1) {
        std::mutex m;
        std::condition_variable cv;
        std::size_t done = 0;
        for (const Pending& p : pending) {
            jobs->submit([&, p] {
                decodeOne(p);
                {
                    std::lock_guard<std::mutex> lk(m);
                    ++done;
                }
                cv.notify_one();
            });
        }
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return done == pending.size(); });
    } else {
        for (const Pending& p : pending)
            decodeOne(p);
    }

    // Drop anything that failed to decode, so a clip that exists always has playable PCM and
    // callers never have to check.
    for (Clip& c : m_clips)
        c.variants.erase(std::remove_if(c.variants.begin(), c.variants.end(),
                                        [](const Sample& s) { return s.frames == 0; }),
                         c.variants.end());
    m_clips.erase(std::remove_if(m_clips.begin(), m_clips.end(),
                                 [](const Clip& c) { return c.variants.empty(); }),
                  m_clips.end());
    std::sort(m_clips.begin(), m_clips.end(),
              [](const Clip& a, const Clip& b) { return a.name < b.name; });

    const double mb = double(totalFrames() * kChannels * sizeof(float)) / (1024.0 * 1024.0);
    VX_INFO("audio: %d clip(s), %d variant file(s), %.2f MB of PCM from %s%s", clipCount(),
            variantCount(), mb, dir.string().c_str(),
            skipped > 0 ? " (some files skipped: wav/mp3/flac only)" : "");
}

std::uint64_t SoundBank::decodeCount() {
    return g_decodes.load(std::memory_order_relaxed);
}

const SoundBank::Clip* SoundBank::find(const std::string& name) const {
    // sorted by name, so this is a binary search rather than the linear scan the old engine
    // ran twice per playIfPresent
    const auto it = std::lower_bound(m_clips.begin(), m_clips.end(), name,
                                     [](const Clip& c, const std::string& n) {
                                         return c.name < n;
                                     });
    return (it != m_clips.end() && it->name == name) ? &*it : nullptr;
}

int SoundBank::variantCount() const {
    int n = 0;
    for (const Clip& c : m_clips)
        n += int(c.variants.size());
    return n;
}

std::uint64_t SoundBank::totalFrames() const {
    std::uint64_t n = 0;
    for (const Clip& c : m_clips)
        for (const Sample& s : c.variants)
            n += s.frames;
    return n;
}

} // namespace vx
