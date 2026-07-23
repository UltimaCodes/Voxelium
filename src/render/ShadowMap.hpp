#pragma once

#include <glm/glm.hpp>

namespace vx {

class Renderer;

// 3 stabilized shadow cascades in a depth array. Each cascade fits a bounding sphere of
// its view-frustum slice (fixed radius per cascade) and snaps its center to shadow
// texels in light space using double-precision world coordinates — rotating or moving
// the camera never makes shadows swim.
class ShadowMap {
public:
    static constexpr int kCascades = 3;
    static constexpr int kResolution = 1536;
    // splits reach ~9-10 chunks so shadows cover the whole radius-10 view instead of
    // vanishing a few chunks out (that was the "shadows disappear when I look far / fly
    // up"). the far cascade is the cost (vertex-bound — every quad drawn per cascade),
    // which is the price of consistent shadows without terrain LOD; per-cascade culling
    // keeps it bounded and each cascade is texel-snapped so nothing swims.
    static constexpr float kSplits[kCascades] = {28.0f, 82.0f, 220.0f};

    bool init(unsigned int shadowProgram);
    void shutdown();

    // Rebuilds cascade matrices for this frame (camera-relative space).
    void update(const glm::dvec3& cameraPos, const glm::vec3& forward, const glm::vec3& sunL,
                float fovYRad, float aspect);

    // Renders all cascades. Culls per-cascade against each cascade's ortho box so far
    // chunks aren't drawn into the tight near cascades.
    void render(Renderer& renderer, const glm::ivec3& cameraChunk, const glm::vec3& cameraFrac);

    const glm::mat4* cascadeVp() const { return m_vp; }
    glm::vec3 cascadeTexelWorld() const { return m_texelWorld; }
    unsigned int texture() const { return m_depthArray; }
    // a sampler with comparison OFF, so the same depth array can be read as raw depth for
    // the PCSS blocker search (bound to a second unit alongside the comparison sampler)
    unsigned int rawSampler() const { return m_rawSampler; }

private:
    unsigned int m_shadowProgram = 0;
    unsigned int m_depthArray = 0;
    unsigned int m_rawSampler = 0;
    unsigned int m_fbo = 0;
    glm::mat4 m_vp[kCascades] = {};
    glm::vec3 m_texelWorld{0.0f};
};

} // namespace vx
