#pragma once

#include <glm/glm.hpp>

#include <filesystem>

namespace vx {

// Rain and snow (#29).
//
// No CPU simulation and no buffer: every drop's position is a pure function of its instance
// id and the clock, worked out in the vertex shader (see weather.vert). Drawing 30k drops
// therefore costs one draw call and no upload — where routing them through the existing
// particle pool would have meant simulating and re-uploading them every frame, for a pool
// that is 4096 and carries per-drop physics precipitation does not need.
//
// Whether it is raining at all is the caller's business; this only draws.
class Weather {
public:
    bool init(const std::filesystem::path& shaderDir);
    void shutdown();

    struct Params {
        glm::mat4 viewProj;
        glm::vec3 right, up;   // camera basis for the billboards
        glm::vec3 camFrac;     // camera position within its own cell; anchors the field
        float time = 0.0f;
        float strength = 0.0f; // 0 = clear, nothing drawn
        float snow = 0.0f;     // 0 rain .. 1 snow
        glm::vec3 tint{0.7f};  // sky-lit, so it darkens at night with everything else
    };

    // Call inside the HDR scene pass after the world and water: alpha blended, depth tested,
    // no depth write. Depth testing is what hides drops behind terrain for free.
    void draw(const Params& p);

    // Drops in the field. 30k streaks over a 26-unit box reads as steady rain; more is not
    // visibly denser, it just costs fill.
    static constexpr int kDrops = 30000;
    static constexpr float kExtent = 26.0f;

private:
    unsigned int m_prog = 0;
    unsigned int m_vao = 0;
};

} // namespace vx
