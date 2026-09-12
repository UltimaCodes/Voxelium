#include "render/QualityPreset.hpp"

#include <cmath>

namespace vx {

namespace {

// Starting points, to be replaced by measured ones: a preset ships only once --bench and
// --tour have both been run at it. High is today's build and is therefore the reference the
// other three are judged against.
constexpr PresetValues kPresets[4] = {
    // Low: the goal is a playable frame on weak hardware, so the two things that actually
    // dominate cost go first — render distance and resolution. SSGI is off entirely rather
    // than reduced; it is a subtle effect whose cheap version reads as noise rather than as
    // a softer version of itself.
    {/*radius*/ 8, /*renderScale*/ 0.65f, /*quality*/ 0, /*shadowRes*/ 1024,
     /*ssao*/ true, /*ssgi*/ false, /*volumetric*/ false, /*clouds*/ false,
     /*taa*/ false, /*motionBlur*/ false, /*cloudShadow*/ 0.0f,
     /*leafSss*/ 0.0f, /*relief*/ 0.0f},
    // Medium: render scale 0.85 is only sensible because the upscale is sharpened now — as a
    // plain bilinear stretch it was soft enough that no preset should have chosen it.
    {16, 0.85f, 1, 1536, true, false, true, true, true, false, 0.0f, 0.8f, 0.6f},
    // High: the pre-preset build. Do not "improve" anything in this row without
    // re-measuring, it is still the reference the bench numbers below were measured against.
    {24, 1.0f, 2, 1536, true, true, true, true, true, true, 0.6f, 0.8f, 0.6f},
    // Ultra: spends on the things that resolve detail rather than on more effects.
    {32, 1.0f, 3, 2560, true, true, true, true, true, true, 0.7f, 1.0f, 0.8f},
};

bool near(float a, float b) { return std::fabs(a - b) < 1e-4f; }

} // namespace

const PresetValues& presetValues(Preset p) {
    const int i = int(p);
    return kPresets[(i >= 0 && i < 4) ? i : int(Preset::High)];
}

const char* presetName(Preset p) {
    switch (p) {
    case Preset::Low: return "Low";
    case Preset::Medium: return "Medium";
    case Preset::High: return "High";
    case Preset::Ultra: return "Ultra";
    default: return "Custom";
    }
}

void applyPreset(Preset p, FxToggles& fx, int& radius) {
    if (p == Preset::Custom)
        return;
    const PresetValues& v = presetValues(p);
    radius = v.radius;
    fx.renderScale = v.renderScale;
    fx.quality = v.quality;
    fx.shadowRes = v.shadowRes;
    fx.ssao = v.ssao;
    fx.ssgi = v.ssgi;
    fx.volumetric = v.volumetric;
    fx.clouds = v.clouds;
    fx.taa = v.taa;
    fx.motionBlur = v.motionBlur;
    fx.cloudShadow = v.cloudShadow;
    fx.leafSss = v.leafSss;
    fx.relief = v.relief;
}

Preset detectPreset(const FxToggles& fx, int radius) {
    for (int i = 0; i < 4; ++i) {
        const PresetValues& v = kPresets[i];
        if (radius == v.radius && near(fx.renderScale, v.renderScale) &&
            fx.quality == v.quality && fx.shadowRes == v.shadowRes && fx.ssao == v.ssao &&
            fx.ssgi == v.ssgi && fx.volumetric == v.volumetric && fx.clouds == v.clouds &&
            fx.taa == v.taa && fx.motionBlur == v.motionBlur &&
            near(fx.cloudShadow, v.cloudShadow) &&
            near(fx.leafSss, v.leafSss) && near(fx.relief, v.relief))
            return Preset(i);
    }
    return Preset::Custom;
}

std::string qualityDefines(int quality, int shadowRes) {
    // Shadow resolution is a define rather than a uniform because the shaders divide by it
    // to size their tap offsets, and a compile-time divisor folds away. It also has to match
    // the texture the cascades were actually allocated at, so passing it through the same
    // channel as the tier keeps the two from disagreeing.
    return "#define VX_QUALITY " + std::to_string(quality) + "\n" +
           "#define VX_SHADOW_RES " + std::to_string(shadowRes) + ".0\n";
}

} // namespace vx
