#pragma once

#include <cstdint>

namespace vx {

// Lets the weather change on its own, so a long session is not one unbroken sky.
//
// A slow state machine over real seconds: clear for a few minutes, then the cloud cover builds
// for a while (the sky gets heavier before anything falls), then it rains or snows for a minute
// or two, then it clears again. Deserts stay dry, the clouds still come and go, nothing falls,
// and precipitation turns to snow up on the peaks.
//
// Mist is the odd one out: it is not a state, just a function of where and when you are. It
// gathers around sunrise, thickest over water, and thickens a little under falling rain.
//
// It only produces numbers. Turning them into a picture is the renderer's job: precip and snow
// go to the same fields the settings menu edits, and cloudBias thickens the sky and the cloud
// shadows together (both read the same coverage field, so they cannot disagree).
class WeatherDirector {
public:
    struct Out {
        float precip = 0.0f;    // 0 dry .. 1 downpour
        float snow = 0.0f;      // 0 rain .. 1 snow
        float cloudBias = 0.0f; // 0 as normal .. ~0.22 fully overcast
        float mist = 0.0f;      // 0 clear .. 1 thick; the renderer pulls the fog in with it
        float storm = 0.0f;     // 0 .. 1, only in the worst downpours: lightning and thunder
    };

    // desert and coast are 0..1 biome weights at the camera; timeOfDay is the sky's 0..1 clock.
    Out update(float dt, float camY, float desert, float coast, float timeOfDay);
    // Puts it back to a clear sky, e.g. on switching worlds.
    void reset();

private:
    enum class Phase : std::uint8_t { Clear, Building, Falling, Clearing };
    float rnd();

    Phase m_phase = Phase::Clear;
    float m_timer = 200.0f; // seconds left in this phase
    float m_targetPrecip = 0.0f;
    Out m_now;
    std::uint32_t m_rng = 0xA341316Cu;
};

} // namespace vx
