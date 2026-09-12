#pragma once

#include "render/PostFx.hpp"

#include <string>

namespace vx {

// Quality presets (#42).
//
// A preset that only flips booleans is a toggle list wearing a hat — the cost in this
// renderer lives in loop bounds that are compile-time constants in GLSL, and in the shadow
// map's resolution and cascade count. So a preset drives three different KINDS of knob:
//
//   runtime   every toggle, render scale, render distance, the float strengths. Free to
//             change on any frame.
//   shader    sample and step counts. Changed by recompiling with a different VX_QUALITY,
//             which the program cache keys on automatically. Costs a rebuild.
//   resource  shadow map resolution and cascade count. Costs a GL reallocation.
//
// High is deliberately the build that existed before presets did. That makes the whole
// table an A/B against a known reference instead of four untested configurations, and it is
// why the shader tier defaults to 2 when nothing is injected.
enum class Preset { Low = 0, Medium = 1, High = 2, Ultra = 3, Custom = 4 };

struct PresetValues {
    int radius;
    float renderScale;
    int quality;    // VX_QUALITY tier handed to the shaders
    int shadowRes;  // shadow cascade texture size
    bool ssao;
    bool ssgi;
    bool volumetric;
    bool clouds;
    bool taa;
    bool motionBlur;
    float cloudShadow;
    float leafSss;
    float relief;
};

const PresetValues& presetValues(Preset p);
const char* presetName(Preset p);

// Writes a preset's values into `fx` and `radius`. Everything a preset does NOT own — the
// grade, exposure, audio, crosshair, time of day — is left exactly as it was, because those
// are taste rather than performance and resetting them on a preset change would be a
// surprise.
void applyPreset(Preset p, FxToggles& fx, int& radius);

// Which preset (if any) the current values correspond to. Returns Custom as soon as one
// preset-owned knob has been touched — that is the whole state machine behind the menu's
// "Custom" entry, and it means a preset never has to be remembered, only recognised.
Preset detectPreset(const FxToggles& fx, int radius);

// The define block injected into every shader for this tier. Shader.cpp splices it after
// #version; the program cache hashes the result, so each tier gets its own binary with no
// extra bookkeeping.
std::string qualityDefines(int quality, int shadowRes);

} // namespace vx
