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
    // Cascade far distances. The last one is where shadows stop entirely, and it used to
    // sit at 220 while full-detail terrain now runs to 12 chunks (384 blocks) — so the
    // ground ahead of you was simply unshadowed until you walked into range, which reads
    // as the shadows following you around. 420 puts the edge past the detailed band, and
    // the existing fade over the last 10% hides the cut instead of stepping into it.
    static constexpr float kSplits[kCascades] = {32.0f, 110.0f, 420.0f};

    bool init(unsigned int shadowProgram);
    void shutdown();

    // Rebuilds cascade matrices for this frame (camera-relative space).
    void update(const glm::dvec3& cameraPos, const glm::vec3& forward, const glm::vec3& sunL,
                float fovYRad, float aspect);

    // Renders all cascades. Culls per-cascade against each cascade's ortho box so far
    // chunks aren't drawn into the tight near cascades.
    void render(Renderer& renderer, const glm::ivec3& cameraChunk, const glm::vec3& cameraFrac);

    // The world changed (block edit, chunks meshed/unloaded): cached cascades hold stale
    // geometry, so force every one to redraw next frame.
    void invalidate() { m_forceAll = true; }

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

    // --- cascade caching -------------------------------------------------------------
    // A cascade's depth map only depends on its WORLD-space light frustum. Anchoring that
    // frustum to a fixed world point means it stays valid while the camera moves, so the
    // cascade doesn't have to be redrawn every frame — and cascades 1+2 are 96% of all
    // shadow draws. m_vp is still rebuilt every frame from (anchor - cameraPos) so the
    // matrix handed to the lighting shader is always correct camera-relative space; only
    // the *redraw* is skipped. The anchor is kept in doubles so precision doesn't degrade
    // far from the origin, same reason the rest of the renderer is camera-relative.
    //
    // Cascades are inflated by kSlack so the sphere they must cover is still inside the
    // frustum after the camera has drifted by that much — the cost is slightly coarser
    // texels, the payoff is redrawing a few times a second instead of 60.
    static constexpr float kSlack = 0.18f;
    glm::dvec3 m_anchorWorld[kCascades] = {};
    glm::vec3 m_anchorSun[kCascades] = {};
    float m_radius[kCascades] = {};
    bool m_dirty[kCascades] = {true, true, true};
    bool m_forceAll = true;
};

} // namespace vx
