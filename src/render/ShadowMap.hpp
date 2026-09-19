#pragma once

#include <glm/glm.hpp>

#include <chrono>
#include <cstdint>

namespace vx {

class Renderer;

// 3 stabilized shadow cascades in a depth array. Each cascade fits a bounding sphere of
// its view-frustum slice (fixed radius per cascade) and snaps its center to shadow
// texels in light space using double-precision world coordinates — rotating or moving
// the camera never makes shadows swim.
class ShadowMap {
public:
    static constexpr int kCascades = 3;
    // Default only. The live size is whatever init() was given, because the quality presets
    // own it — and the same number is injected into the shaders as VX_SHADOW_RES so their
    // tap offsets cannot disagree with the texture that was actually allocated.
    static constexpr int kDefaultResolution = 1536;
    // splits reach ~9-10 chunks so shadows cover the whole radius-10 view instead of
    // vanishing a few chunks out (that was the "shadows disappear when I look far / fly
    // up"). the far cascade is the cost (vertex-bound — every quad drawn per cascade),
    // which is the price of consistent shadows without terrain LOD; per-cascade culling
    // keeps it bounded and each cascade is texel-snapped so nothing swims.
    // Cascade far distances. The last one is where shadows stop entirely. It sat at a fixed
    // 420 (past the 384-block detailed band), which left everything beyond it flat-lit no
    // matter how far you could see - a visible ring of shadowed ground around the player.
    // It now follows the render distance up to kFarMax; past that the texels are too coarse
    // to be worth it and the fog has taken over anyway. The last 10% fades out rather than
    // cutting.
    static constexpr float kFarMin = 420.0f;
    static constexpr float kFarMax = 760.0f;
    glm::vec3 splits() const { return m_splits; }
    // How far the far cascade should reach. Changing it invalidates every cascade, since the
    // old contents were rendered at a different texel scale.
    void setViewDistance(float blocks);
    // Share of a leaf canopy that lets sun straight through (see shadow.frag). Baked into the
    // cascades, so a change redraws them.
    void setLeafGap(float gap);

    bool init(unsigned int shadowProgram, int resolution = kDefaultResolution);
    int resolution() const { return m_resolution; }
    // Rebuild the depth array at a new size and/or with a rebuilt program. Every cascade is
    // invalidated, since the old contents were rendered at the old texel scale.
    bool resize(unsigned int shadowProgram, int resolution);
    void shutdown();

    // Rebuilds cascade matrices for this frame (camera-relative space).
    void update(const glm::dvec3& cameraPos, const glm::vec3& forward, const glm::vec3& sunL,
                float fovYRad, float aspect);

    // Renders all cascades. Culls per-cascade against each cascade's ortho box so far
    // chunks aren't drawn into the tight near cascades.
    void render(Renderer& renderer, const glm::ivec3& cameraChunk, const glm::vec3& cameraFrac);

    // The world changed (block edit, chunks meshed/unloaded): cached cascades may hold stale
    // geometry. Rate-limited internally to a floor of kMinForceInterval between forced
    // redraws — see the .cpp for why a blanket "redraw everything, right now" here is exactly
    // the wrong default while the world is streaming hard.
    void invalidate();

    // How many cascade redraws render() has actually issued, ever. Not per-frame — callers
    // that want a rate sample it at two points and diff, the same pattern SoundBank's
    // decodeCount() uses and for the same reason: a counter that resets itself is much
    // easier to accidentally make unfalsifiable than one that only ever climbs.
    std::uint64_t redrawCount() const { return m_redraws; }
    // How many invalidate() calls actually set m_forceAll, as opposed to being throttled
    // away. Exists so the throttle itself is directly testable without a GL context —
    // invalidate() touches no GL state, only this and a clock.
    std::uint64_t forcedCount() const { return m_forcedCount; }

    const glm::mat4* cascadeVp() const { return m_vp; }
    glm::vec3 cascadeTexelWorld() const { return m_texelWorld; }
    unsigned int texture() const { return m_depthArray; }
    // a sampler with comparison OFF, so the same depth array can be read as raw depth for
    // the PCSS blocker search (bound to a second unit alongside the comparison sampler)
    unsigned int rawSampler() const { return m_rawSampler; }

private:
    unsigned int m_shadowProgram = 0;
    int m_resolution = kDefaultResolution;
    unsigned int m_depthArray = 0;
    unsigned int m_rawSampler = 0;
    unsigned int m_fbo = 0;
    glm::mat4 m_vp[kCascades] = {};
    glm::vec3 m_texelWorld{0.0f};
    glm::vec3 m_splits{32.0f, 110.0f, kFarMin};
    glm::vec3 m_lightX{1.0f, 0.0f, 0.0f}, m_lightY{0.0f, 1.0f, 0.0f};
    glm::vec2 m_camLightXY{0.0f};
    float m_leafGap = 0.22f;

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

    // invalidate()'s throttle. A point in the past so the very first call always fires.
    static constexpr std::chrono::milliseconds kMinForceInterval{100};
    std::chrono::steady_clock::time_point m_lastForcedRedraw{};
    std::uint64_t m_redraws = 0;
    std::uint64_t m_forcedCount = 0;
};

} // namespace vx
