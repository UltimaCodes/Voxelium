#include "render/WeatherDirector.hpp"

#include <algorithm>

namespace vx {

namespace {
float smoothstep(float a, float b, float x) {
    const float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
// Time constants, in seconds: how slowly each quantity follows its target. Rain arrives over
// half a minute and clouds over a minute, which is what makes it read as weather rolling in
// rather than a switch being thrown.
constexpr float kPrecipRate = 0.05f;
constexpr float kCloudRate = 0.025f;
constexpr float kOvercast = 0.22f;
} // namespace

float WeatherDirector::rnd() {
    m_rng ^= m_rng << 13;
    m_rng ^= m_rng >> 17;
    m_rng ^= m_rng << 5;
    return float(m_rng & 0xFFFFFFu) / float(0x1000000u);
}

void WeatherDirector::reset() {
    m_phase = Phase::Clear;
    m_timer = 120.0f + rnd() * 180.0f;
    m_targetPrecip = 0.0f;
    m_now = {};
}

WeatherDirector::Out WeatherDirector::update(float dt, float camY, float desert, float coast,
                                             float timeOfDay) {
    dt = std::min(dt, 0.25f);
    m_timer -= dt;
    float cloudTarget = 0.0f;
    float precipTarget = 0.0f;
    switch (m_phase) {
        case Phase::Clear:
            if (m_timer <= 0.0f) {
                m_phase = Phase::Building;
                m_timer = 35.0f + rnd() * 25.0f;
                // some fronts are a shower, some a proper storm; a third never break at all
                m_targetPrecip = rnd() < 0.33f ? 0.0f : 0.3f + rnd() * 0.65f;
            }
            break;
        case Phase::Building:
            cloudTarget = kOvercast * (0.6f + 0.4f * m_targetPrecip);
            if (m_timer <= 0.0f) {
                m_phase = m_targetPrecip > 0.0f ? Phase::Falling : Phase::Clearing;
                m_timer = m_targetPrecip > 0.0f ? 60.0f + rnd() * 90.0f : 45.0f;
            }
            break;
        case Phase::Falling:
            cloudTarget = kOvercast;
            precipTarget = m_targetPrecip;
            if (m_timer <= 0.0f) {
                m_phase = Phase::Clearing;
                m_timer = 45.0f;
            }
            break;
        case Phase::Clearing:
            if (m_timer <= 0.0f) {
                m_phase = Phase::Clear;
                m_timer = 150.0f + rnd() * 270.0f;
            }
            break;
    }
    // no rain on the sand; the sky still fills with cloud
    precipTarget *= 1.0f - smoothstep(0.4f, 0.7f, desert);

    m_now.precip += (precipTarget - m_now.precip) * std::min(1.0f, kPrecipRate * dt * 4.0f);
    m_now.cloudBias += (cloudTarget - m_now.cloudBias) * std::min(1.0f, kCloudRate * dt * 4.0f);
    // cold enough up a mountain to snow
    m_now.snow = smoothstep(150.0f, 190.0f, camY);
    if (m_now.precip < 0.002f)
        m_now.precip = 0.0f;
    // sunrise is at 0.25; the mist is up before it and burns off over the following hour
    const float dawn = smoothstep(0.19f, 0.25f, timeOfDay) * (1.0f - smoothstep(0.29f, 0.38f, timeOfDay));
    const float wet = (0.35f + 0.65f * coast) * dawn + 0.3f * m_now.precip;
    // only the hard rain (a front that came in above 0.62) storms, and only once it is well under way
    const float stormTarget = (m_phase == Phase::Falling && m_targetPrecip > 0.62f)
                                  ? smoothstep(0.5f, 0.9f, m_now.precip)
                                  : 0.0f;
    m_now.storm += (stormTarget - m_now.storm) * std::min(1.0f, 0.4f * dt);
    m_now.mist = std::clamp(wet, 0.0f, 1.0f) * (1.0f - smoothstep(0.3f, 0.6f, desert));
    return m_now;
}

} // namespace vx
