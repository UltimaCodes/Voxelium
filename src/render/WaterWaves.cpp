#include "render/WaterWaves.hpp"

#include <glad/gl.h>

#include <algorithm>

namespace vx {

namespace {

constexpr double kTwoPi = 6.283185307179586;
constexpr float kSharp = 1.2f;

// The swell runs with the same wind the sand and the leaves blow in (Particles.cpp kWindDir).
// Four waves within a few tens of degrees of it, none sharing a period, so the sum never settles
// into a visible repeat. Kept close to the wind on purpose: spread wider (the first try had one at
// 72 degrees) they cross-hatch into a field of round bumps, where a sea reads as long crests
// marching one way and only broken up along their length.
struct WaveDef {
    double angleDeg;  // from the wind
    double length;    // blocks
    double weight;
};
constexpr WaveDef kWaves[WaterWaves::kCount] = {
    {0.0, 16.0, 1.00}, {17.0, 10.5, 0.45}, {-21.0, 23.0, 0.60}, {38.0, 7.0, 0.20}};

// I0, the modified Bessel function: exp(s(sin - 1)) averages to exp(-s) I0(s) over a period, which
// is what is taken off so a sharpened wave still averages to the rest level.
double besselI0(double x) {
    double sum = 1.0, term = 1.0;
    for (int k = 1; k < 20; ++k) {
        term *= (x * 0.5) * (x * 0.5) / (double(k) * double(k));
        sum += term;
    }
    return sum;
}

} // namespace

WaterWaves::Frame WaterWaves::frame(double time, const glm::dvec3& cam, float storm, bool on) {
    Frame f;
    const double wind = std::atan2(0.55, 0.83);
    double total = 0.0;
    for (const WaveDef& w : kWaves)
        total += w.weight;
    for (int i = 0; i < kCount; ++i) {
        const WaveDef& w = kWaves[i];
        const double a = wind + w.angleDeg * kTwoPi / 360.0;
        const double dx = std::cos(a), dz = std::sin(a);
        const double k = kTwoPi / w.length;
        // deep-water dispersion, a touch slow: long waves travel faster than short ones
        const double omega = std::sqrt(9.81 * k) * 0.85;
        f.wave[i] = {float(dx), float(dz), float(k), float(w.weight / total)};
        f.phase[i] = float(std::fmod(k * (dx * cam.x + dz * cam.z) - omega * time, kTwoPi));
    }
    const double s = kSharp;
    const double e0 = std::exp(-2.0 * s);
    f.shape = {float(s), float(e0), float(std::exp(-s) * besselI0(s)), float(2.0 / (1.0 - e0))};
    f.scale = on ? kCalm * (1.0f + std::clamp(storm, 0.0f, 1.0f)) : 0.0f;
    f.camWrap = {float(cam.x - std::floor(cam.x / kWrap) * kWrap),
                 float(cam.z - std::floor(cam.z / kWrap) * kWrap)};
    return f;
}

float WaterWaves::unitHeight(const Frame& f, float relX, float relZ) {
    float h = 0.0f;
    for (int i = 0; i < kCount; ++i) {
        const glm::vec4& w = f.wave[i];
        const float th = w.z * (w.x * relX + w.y * relZ) + f.phase[i];
        const float e = std::exp(f.shape.x * (std::sin(th) - 1.0f));
        h += w.w * f.shape.w * (e - f.shape.z);
    }
    return h;
}

float WaterWaves::fade(float d) {
    const float t = std::clamp((d - kFadeStart) / (kFadeEnd - kFadeStart), 0.0f, 1.0f);
    return 1.0f - t * t * (3.0f - 2.0f * t);
}

void WaterWaves::apply(unsigned int program, const Frame& f, float tessDensity) {
    glProgramUniform4fv(program, 26, kCount, &f.wave[0].x);
    glProgramUniform4fv(program, 30, 1, &f.phase.x);
    glProgramUniform1f(program, 31, f.scale);
    glProgramUniform2f(program, 32, kFadeStart, kFadeEnd);
    glProgramUniform4fv(program, 33, 1, &f.shape.x);
    glProgramUniform2fv(program, 34, 1, &f.camWrap.x);
    glProgramUniform1f(program, 35, tessDensity);
}

} // namespace vx
