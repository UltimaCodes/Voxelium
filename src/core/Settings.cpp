#include "core/Settings.hpp"

#include "core/Log.hpp"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <string>
#include <string_view>

namespace vx::Settings {

namespace {

std::string_view trim(std::string_view s) {
    const auto notSpace = [](char c) { return c != ' ' && c != '\t' && c != '\r'; };
    while (!s.empty() && !notSpace(s.front()))
        s.remove_prefix(1);
    while (!s.empty() && !notSpace(s.back()))
        s.remove_suffix(1);
    return s;
}

// "true"/"1" on, anything else off — keeps hand-edited files forgiving
void readBool(std::string_view v, bool& out) { out = v == "1" || v == "true"; }

void readInt(std::string_view v, int& out) {
    int parsed = 0;
    if (std::from_chars(v.data(), v.data() + v.size(), parsed).ec == std::errc{})
        out = parsed;
}

// strtof, not from_chars: libstdc++'s floating-point from_chars isn't available on every
// target we build, and engine policy is no exceptions so stof is out. A malformed value
// leaves end == start, which keeps the default.
void readFloat(std::string_view v, float& out) {
    const std::string s(v);
    char* end = nullptr;
    const float parsed = std::strtof(s.c_str(), &end);
    if (end != s.c_str())
        out = parsed;
}

} // namespace

bool load(const std::filesystem::path& file, FxToggles& fx, bool& vsync, bool& caveCull,
          int& radius, float& fovDeg) {
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
        ++applied;

        if (key == "shadows") readBool(val, fx.shadows);
        else if (key == "ssao") readBool(val, fx.ssao);
        else if (key == "volumetric") readBool(val, fx.volumetric);
        else if (key == "bloom") readBool(val, fx.bloom);
        else if (key == "clouds") readBool(val, fx.clouds);
        else if (key == "ssgi") readBool(val, fx.ssgi);
        else if (key == "taa") readBool(val, fx.taa);
        else if (key == "motionBlur") readBool(val, fx.motionBlur);
        else if (key == "fog") readBool(val, fx.fog);
        else if (key == "dayCycle") readBool(val, fx.dayCycle);
        else if (key == "timeOfDay") readFloat(val, fx.timeOfDay);
        else if (key == "dayLengthSec") readFloat(val, fx.dayLengthSec);
        else if (key == "exposure") readFloat(val, fx.exposure);
        else if (key == "bloomStrength") readFloat(val, fx.bloomStrength);
        else if (key == "crosshairR") readFloat(val, fx.crosshair[0]);
        else if (key == "crosshairG") readFloat(val, fx.crosshair[1]);
        else if (key == "crosshairB") readFloat(val, fx.crosshair[2]);
        else if (key == "crosshairA") readFloat(val, fx.crosshair[3]);
        else if (key == "lampR") readFloat(val, fx.lampColour[0]);
        else if (key == "lampG") readFloat(val, fx.lampColour[1]);
        else if (key == "lampB") readFloat(val, fx.lampColour[2]);
        else if (key == "volume") readFloat(val, fx.volume);
        else if (key == "renderScale") readFloat(val, fx.renderScale);
        else if (key == "vsync") readBool(val, vsync);
        else if (key == "caveCull") readBool(val, caveCull);
        else if (key == "radius") readInt(val, radius);
        else if (key == "fov") readFloat(val, fovDeg);
        else --applied; // unknown key from an older/newer build
    }

    // a corrupt or truncated file could leave nonsense here — clamp to the same ranges
    // the ESC menu enforces so a bad value can't wedge the app on the next launch
    radius = std::clamp(radius, 4, 32);
    fovDeg = std::clamp(fovDeg, 50.0f, 110.0f);
    fx.timeOfDay = std::clamp(fx.timeOfDay, 0.0f, 1.0f);
    fx.dayLengthSec = std::clamp(fx.dayLengthSec, 20.0f, 1200.0f);
    fx.exposure = std::clamp(fx.exposure, 0.4f, 2.5f);
    fx.bloomStrength = std::clamp(fx.bloomStrength, 0.0f, 1.5f);
    for (float& c : fx.lampColour)
        c = c < 0.0f ? 0.0f : (c > 1.0f ? 1.0f : c);
    for (float& c : fx.crosshair)
        c = std::clamp(c, 0.0f, 1.0f);
    fx.volume = std::clamp(fx.volume, 0.0f, 1.0f);
    fx.renderScale = std::clamp(fx.renderScale, 0.5f, 1.0f);
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

    const auto b = [](bool v) { return v ? "true" : "false"; };
    out << "# Voxelium settings. Delete this file to go back to defaults.\n"
        << "shadows=" << b(fx.shadows) << "\n"
        << "ssao=" << b(fx.ssao) << "\n"
        << "volumetric=" << b(fx.volumetric) << "\n"
        << "bloom=" << b(fx.bloom) << "\n"
        << "clouds=" << b(fx.clouds) << "\n"
        << "ssgi=" << b(fx.ssgi) << "\n"
        << "taa=" << b(fx.taa) << "\n"
        << "motionBlur=" << b(fx.motionBlur) << "\n"
        << "fog=" << b(fx.fog) << "\n"
        << "dayCycle=" << b(fx.dayCycle) << "\n"
        << "timeOfDay=" << fx.timeOfDay << "\n"
        << "dayLengthSec=" << fx.dayLengthSec << "\n"
        << "exposure=" << fx.exposure << "\n"
        << "bloomStrength=" << fx.bloomStrength << "\n"
        << "crosshairR=" << fx.crosshair[0] << "\n"
        << "crosshairG=" << fx.crosshair[1] << "\n"
        << "crosshairB=" << fx.crosshair[2] << "\n"
        << "crosshairA=" << fx.crosshair[3] << "\n"
        << "volume=" << fx.volume << "\n"
        << "renderScale=" << fx.renderScale << "\n"
        << "vsync=" << b(vsync) << "\n"
        << "caveCull=" << b(caveCull) << "\n"
        << "radius=" << radius << "\n"
        << "fov=" << fovDeg << "\n";
    return bool(out);
}

} // namespace vx::Settings
