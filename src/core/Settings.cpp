#include "core/Settings.hpp"

#include "core/Log.hpp"
#include "world/Chunk.hpp" // lamp swatch count, for clamping on load

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <string_view>

namespace vx::Settings {

namespace {

// ONE list, walked by load, save and the clamp pass alike.
//
// This used to be three separate lists — an if/else chain in load, a stream of `<<` in save,
// and a run of std::clamp calls between them — and they drifted, silently and twice. At the
// point this was written load read 37 keys and save wrote 29, so autoExposure, emissive,
// lensFlare, cloudShadow, contactShadow, leafSss, relief and lampSwatch were read on startup
// and never written: every one of them reset to its default on the next launch, and seven
// were Tier 3 features that had been built and tuned. It happened AGAIN while adding the
// audio buses, in the same sitting as the fix. (contactShadow itself was later removed
// entirely — screen-space contact shadows never earned their compounding-darkness cost.)
//
// A key cannot now exist in one place and not another, because there is only one place. The
// cost of a new setting is one row.
struct Bound {
    const char* key;
    enum class Type { Bool, Int, Float } type;
    void* ptr;
    float lo, hi; // clamp range; ignored for Bool
};

// Every setting the engine persists. Adding one is a row here and nothing else.
constexpr std::size_t kSettingCount = 46;

// Binds the schema to a live set of values. Non-const because load writes through it; save
// binds a local copy for the same reason, which is cheaper than a second const walk and
// keeps one definition of the layout.
std::array<Bound, kSettingCount> bind(FxToggles& fx, bool& vsync, bool& caveCull, int& radius,
                                      float& fovDeg) {
    using T = Bound::Type;
    return {{
        {"shadows", T::Bool, &fx.shadows, 0, 0},
        {"ssao", T::Bool, &fx.ssao, 0, 0},
        {"volumetric", T::Bool, &fx.volumetric, 0, 0},
        {"bloom", T::Bool, &fx.bloom, 0, 0},
        {"clouds", T::Bool, &fx.clouds, 0, 0},
        {"ssgi", T::Bool, &fx.ssgi, 0, 0},
        {"taa", T::Bool, &fx.taa, 0, 0},
        {"motionBlur", T::Bool, &fx.motionBlur, 0, 0},
        {"fog", T::Bool, &fx.fog, 0, 0},
        {"autoExposure", T::Bool, &fx.autoExposure, 0, 0},
        {"dayCycle", T::Bool, &fx.dayCycle, 0, 0},
        {"timeOfDay", T::Float, &fx.timeOfDay, 0.0f, 1.0f},
        {"dayLengthSec", T::Float, &fx.dayLengthSec, 20.0f, 1200.0f},
        {"exposure", T::Float, &fx.exposure, 0.4f, 2.5f},
        {"bloomStrength", T::Float, &fx.bloomStrength, 0.0f, 1.5f},
        {"emissive", T::Float, &fx.emissive, 0.0f, 5.0f},
        {"lensFlare", T::Float, &fx.lensFlare, 0.0f, 2.0f},
        {"scotopic", T::Float, &fx.scotopic, 0.0f, 1.0f},
        {"cloudShadow", T::Float, &fx.cloudShadow, 0.0f, 1.0f},
        {"leafSss", T::Float, &fx.leafSss, 0.0f, 2.0f},
        {"relief", T::Float, &fx.relief, 0.0f, 2.0f},
        {"saturation", T::Float, &fx.saturation, 0.0f, 2.0f},
        {"contrast", T::Float, &fx.contrast, 0.5f, 1.8f},
        {"temperature", T::Float, &fx.temperature, -1.0f, 1.0f},
        {"vignette", T::Float, &fx.vignette, 0.0f, 1.0f},
        {"grain", T::Float, &fx.grain, 0.0f, 1.0f},
        {"crosshairR", T::Float, &fx.crosshair[0], 0.0f, 1.0f},
        {"crosshairG", T::Float, &fx.crosshair[1], 0.0f, 1.0f},
        {"crosshairB", T::Float, &fx.crosshair[2], 0.0f, 1.0f},
        {"crosshairA", T::Float, &fx.crosshair[3], 0.0f, 1.0f},
        {"lampSwatch", T::Int, &fx.lampSwatch, 0.0f, float(blocks::kLampCount - 1)},
        {"volume", T::Float, &fx.volume, 0.0f, 1.0f},
        {"volWorld", T::Float, &fx.volWorld, 0.0f, 1.0f},
        {"volUi", T::Float, &fx.volUi, 0.0f, 1.0f},
        {"volAmbient", T::Float, &fx.volAmbient, 0.0f, 1.0f},
        {"audioOcclusion", T::Float, &fx.audioOcclusion, 0.0f, 1.0f},
        {"dither", T::Float, &fx.dither, 0.0f, 2.0f},
        {"precip", T::Float, &fx.precip, 0.0f, 1.0f},
        {"snow", T::Float, &fx.snow, 0.0f, 1.0f},
        {"quality", T::Int, &fx.quality, 0.0f, 3.0f},
        {"shadowRes", T::Int, &fx.shadowRes, 1024.0f, 2560.0f},
        {"renderScale", T::Float, &fx.renderScale, 0.5f, 1.0f},
        {"vsync", T::Bool, &vsync, 0, 0},
        {"caveCull", T::Bool, &caveCull, 0, 0},
        {"radius", T::Int, &radius, 4.0f, 32.0f},
        {"fov", T::Float, &fovDeg, 50.0f, 110.0f},
    }};
}

std::string_view trim(std::string_view s) {
    const auto notSpace = [](char c) { return c != ' ' && c != '\t' && c != '\r'; };
    while (!s.empty() && !notSpace(s.front()))
        s.remove_prefix(1);
    while (!s.empty() && !notSpace(s.back()))
        s.remove_suffix(1);
    return s;
}

// "true"/"1" on, anything else off — keeps hand-edited files forgiving
void readInto(const Bound& b, std::string_view v) {
    switch (b.type) {
    case Bound::Type::Bool:
        *static_cast<bool*>(b.ptr) = (v == "1" || v == "true");
        return;
    case Bound::Type::Int: {
        int parsed = 0;
        if (std::from_chars(v.data(), v.data() + v.size(), parsed).ec == std::errc{})
            *static_cast<int*>(b.ptr) = parsed;
        return;
    }
    case Bound::Type::Float: {
        // strtof, not from_chars: libstdc++'s floating-point from_chars isn't available on
        // every target we build, and engine policy is no exceptions so stof is out. A
        // malformed value leaves end == start, which keeps the default.
        const std::string s(v);
        char* end = nullptr;
        const float parsed = std::strtof(s.c_str(), &end);
        if (end != s.c_str())
            *static_cast<float*>(b.ptr) = parsed;
        return;
    }
    }
}

// A corrupt or truncated file could leave nonsense behind, so every numeric value is pinned
// to the same range the ESC menu enforces — a bad value must not be able to wedge the app on
// the next launch.
void clampTo(const Bound& b) {
    if (b.type == Bound::Type::Int) {
        int& v = *static_cast<int*>(b.ptr);
        v = std::clamp(v, int(b.lo), int(b.hi));
    } else if (b.type == Bound::Type::Float) {
        float& v = *static_cast<float*>(b.ptr);
        v = std::clamp(v, b.lo, b.hi);
    }
}

void writeFrom(std::ofstream& out, const Bound& b) {
    out << b.key << '=';
    switch (b.type) {
    case Bound::Type::Bool:
        out << (*static_cast<const bool*>(b.ptr) ? "true" : "false");
        break;
    case Bound::Type::Int:
        out << *static_cast<const int*>(b.ptr);
        break;
    case Bound::Type::Float:
        out << *static_cast<const float*>(b.ptr);
        break;
    }
    out << '\n';
}

} // namespace

bool load(const std::filesystem::path& file, FxToggles& fx, bool& vsync, bool& caveCull,
          int& radius, float& fovDeg) {
    const auto schema = bind(fx, vsync, caveCull, radius, fovDeg);

    std::ifstream in(file);
    if (!in)
        return false;

    std::string line;
    int applied = 0;
    while (std::getline(in, line)) {
        const std::string_view raw = trim(line);
        if (raw.empty() || raw.front() == '#')
            continue;
        const auto eq = raw.find('=');
        if (eq == std::string_view::npos)
            continue;
        const std::string_view key = trim(raw.substr(0, eq));
        const std::string_view val = trim(raw.substr(eq + 1));

        const auto it = std::find_if(schema.begin(), schema.end(),
                                     [&](const Bound& b) { return key == b.key; });
        if (it == schema.end())
            continue; // unknown key from an older/newer build
        readInto(*it, val);
        ++applied;
    }

    for (const Bound& b : schema)
        clampTo(b);
    return applied > 0;
}

bool save(const std::filesystem::path& file, const FxToggles& fx, bool vsync, bool caveCull,
          int radius, float fovDeg) {
    std::error_code ec;
    if (file.has_parent_path())
        std::filesystem::create_directories(file.parent_path(), ec);

    std::ofstream out(file, std::ios::trunc);
    if (!out) {
        VX_ERROR("settings: could not write %s", file.string().c_str());
        return false;
    }

    // Local copy purely so the one schema definition can be reused: bind() needs somewhere
    // writable to point at, and FxToggles is small enough that copying it beats maintaining
    // a second const walk that could drift from this one.
    FxToggles tmp = fx;
    bool v = vsync, cc = caveCull;
    int r = radius;
    float f = fovDeg;
    const auto schema = bind(tmp, v, cc, r, f);

    out << "# Voxelium settings. Delete this file to go back to defaults.\n";
    for (const Bound& b : schema)
        writeFrom(out, b);
    return bool(out);
}

int keyCount() { return int(kSettingCount); }

int perturbAll(FxToggles& fx, bool& vsync, bool& caveCull, int& radius, float& fovDeg) {
    const auto schema = bind(fx, vsync, caveCull, radius, fovDeg);
    int changed = 0;
    for (const Bound& b : schema) {
        switch (b.type) {
        case Bound::Type::Bool: {
            bool& v = *static_cast<bool*>(b.ptr);
            v = !v;
            ++changed;
            break;
        }
        case Bound::Type::Int: {
            int& v = *static_cast<int*>(b.ptr);
            // Two candidates so the perturbed value is never accidentally the original —
            // a field that did not actually change would let a broken load pass unnoticed.
            const int a = int(b.lo) + int((b.hi - b.lo) * 0.37f);
            const int c = int(b.lo) + int((b.hi - b.lo) * 0.61f);
            const int want = (v == a) ? c : a;
            if (want != v) {
                v = want;
                ++changed;
            }
            break;
        }
        case Bound::Type::Float: {
            float& v = *static_cast<float*>(b.ptr);
            const float a = b.lo + (b.hi - b.lo) * 0.37f;
            const float c = b.lo + (b.hi - b.lo) * 0.61f;
            const float want = (std::abs(v - a) < 1e-5f) ? c : a;
            if (std::abs(want - v) > 1e-5f) {
                v = want;
                ++changed;
            }
            break;
        }
        }
    }
    return changed;
}

int differences(const FxToggles& a, bool vsyncA, bool caveCullA, int radiusA, float fovA,
                const FxToggles& b, bool vsyncB, bool caveCullB, int radiusB, float fovB) {
    FxToggles ta = a, tb = b;
    bool va = vsyncA, ca = caveCullA, vb = vsyncB, cb = caveCullB;
    int ra = radiusA, rb = radiusB;
    float fa = fovA, fb = fovB;
    const auto sa = bind(ta, va, ca, ra, fa);
    const auto sb = bind(tb, vb, cb, rb, fb);

    int diff = 0;
    for (std::size_t i = 0; i < sa.size(); ++i) {
        switch (sa[i].type) {
        case Bound::Type::Bool:
            if (*static_cast<const bool*>(sa[i].ptr) != *static_cast<const bool*>(sb[i].ptr))
                ++diff;
            break;
        case Bound::Type::Int:
            if (*static_cast<const int*>(sa[i].ptr) != *static_cast<const int*>(sb[i].ptr))
                ++diff;
            break;
        case Bound::Type::Float:
            // Tolerance, not equality: the value makes a round trip through decimal text.
            if (std::abs(*static_cast<const float*>(sa[i].ptr) -
                         *static_cast<const float*>(sb[i].ptr)) > 1e-4f)
                ++diff;
            break;
        }
    }
    return diff;
}

} // namespace vx::Settings
