#include "render/Camera.hpp"

#include "platform/Window.hpp"

#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace vx {

static const glm::vec3 kUp{0.0f, 1.0f, 0.0f};

void FlyCamera::update(Window& window, float dt, bool uiWantsMouse, bool uiWantsKeyboard) {
    // Drain scroll every frame so it can't pile up while not flying.
    const double scroll = window.consumeScrollY();

    const bool rmb = window.mouseDown(GLFW_MOUSE_BUTTON_RIGHT);
    const bool wantFly = rmb && (m_flying || !uiWantsMouse);
    if (wantFly && !m_flying) {
        m_flying = true;
        window.setCursorCaptured(true);
        window.cursorPos(m_lastX, m_lastY);
    } else if (!wantFly && m_flying) {
        m_flying = false;
        window.setCursorCaptured(false);
    }
    if (!m_flying)
        return;

    double x = 0.0;
    double y = 0.0;
    window.cursorPos(x, y);
    yawDeg += float(x - m_lastX) * sensitivity;
    pitchDeg -= float(y - m_lastY) * sensitivity;
    pitchDeg = std::clamp(pitchDeg, -89.0f, 89.0f);
    m_lastX = x;
    m_lastY = y;

    if (scroll != 0.0)
        baseSpeed = std::clamp(baseSpeed * float(std::pow(1.15, scroll)), 0.5f, 256.0f);

    if (uiWantsKeyboard)
        return;

    const float speed = baseSpeed * (window.keyDown(GLFW_KEY_LEFT_SHIFT) ? 8.0f : 1.0f);
    const glm::vec3 fwd = forward();
    const glm::vec3 right = glm::normalize(glm::cross(fwd, kUp));

    glm::vec3 move{0.0f};
    if (window.keyDown(GLFW_KEY_W)) move += fwd;
    if (window.keyDown(GLFW_KEY_S)) move -= fwd;
    if (window.keyDown(GLFW_KEY_D)) move += right;
    if (window.keyDown(GLFW_KEY_A)) move -= right;
    if (window.keyDown(GLFW_KEY_SPACE)) move += kUp;
    if (window.keyDown(GLFW_KEY_LEFT_CONTROL)) move -= kUp;

    if (glm::dot(move, move) > 0.0f)
        position += glm::dvec3(glm::normalize(move) * speed * dt);
}

glm::vec3 FlyCamera::forward() const {
    const float yaw = glm::radians(yawDeg);
    const float pitch = glm::radians(pitchDeg);
    return {std::cos(pitch) * std::sin(yaw), std::sin(pitch), -std::cos(pitch) * std::cos(yaw)};
}

glm::mat4 FlyCamera::viewAtOrigin() const {
    return glm::lookAt(glm::vec3(0.0f), forward(), kUp);
}

glm::mat4 FlyCamera::projReversedZ(float aspect) const {
    const float f = 1.0f / std::tan(glm::radians(fovYDeg) * 0.5f);
    // Maps view z = -near to depth 1 and z = -inf to depth 0 (glm is column-major:
    // m[col][row]). Derivation: clip.z = near, clip.w = -view.z.
    glm::mat4 p(0.0f);
    p[0][0] = f / aspect;
    p[1][1] = f;
    p[2][3] = -1.0f;
    p[3][2] = nearPlane;
    return p;
}

} // namespace vx
