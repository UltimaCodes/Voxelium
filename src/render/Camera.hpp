#pragma once

#include <glm/glm.hpp>

namespace vx {

class Window;

// Free-fly debug camera. Hold RMB to fly: mouse look, WASD + Space/Ctrl, Shift = fast,
// scroll = base speed. Public fields on purpose — this is a tool, not an API surface.
class FlyCamera {
public:
    // Double so world coordinates stay exact far from the origin; everything the GPU
    // sees is camera-relative (view sits at the origin, chunks get float offsets).
    glm::dvec3 position{0.0, 0.0, 3.0};
    float yawDeg = 0.0f;   // 0 looks down -Z, positive turns right
    float pitchDeg = 0.0f; // clamped to +/-89
    float fovYDeg = 70.0f;
    float nearPlane = 0.1f;
    float baseSpeed = 8.0f;      // units/sec
    float sensitivity = 0.12f;   // degrees per pixel of mouse travel

    void update(Window& window, float dt, bool uiWantsMouse, bool uiWantsKeyboard);

    // Walk mode: cursor always captured, mouse look only — position comes from the
    // player. Reuses the capture state so mode switches hand off cleanly.
    void lookOnly(Window& window);

    bool flying() const { return m_flying; }
    glm::vec3 forward() const;
    glm::mat4 viewAtOrigin() const; // camera-relative rendering: translation lives elsewhere

    // Reversed-Z, infinite far plane, 0..1 clip depth (pairs with glClipControl +
    // GL_GREATER + clear depth 0). Float depth precision is concentrated near the eye
    // with reversed-Z spreading it out — at 32 chunks of draw distance a classic 0..1
    // near/far depth buffer z-fights badly, this doesn't. No far plane to tune either.
    glm::mat4 projReversedZ(float aspect) const;

private:
    bool m_flying = false;
    double m_lastX = 0.0;
    double m_lastY = 0.0;
};

} // namespace vx
