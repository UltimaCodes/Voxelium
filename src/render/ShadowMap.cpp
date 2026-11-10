#include "render/ShadowMap.hpp"

#include "core/Log.hpp"
#include "render/Frustum.hpp"
#include "render/Renderer.hpp"

#include <glad/gl.h>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace vx {

bool ShadowMap::init(unsigned int shadowProgram, int resolution) {
    m_resolution = resolution;
    m_shadowProgram = shadowProgram;

    // one layer per cascade, plus the spares the mid and far ones are redrawn into a slice at a time
    glCreateTextures(GL_TEXTURE_2D_ARRAY, 1, &m_depthArray);
    glTextureStorage3D(m_depthArray, 1, GL_DEPTH_COMPONENT32F, m_resolution, m_resolution,
                       spareLayer(kCascades - 1) + 1);
    // new storage holds nothing yet, so everything is drawn whole on the next frame
    m_forceAll = true;
    for (Spare& s : m_spare)
        s.slice = -1;
    glTextureParameteri(m_depthArray, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(m_depthArray, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(m_depthArray, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTextureParameteri(m_depthArray, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    const float border[4] = {1.0f, 1.0f, 1.0f, 1.0f}; // outside the map = fully lit
    glTextureParameterfv(m_depthArray, GL_TEXTURE_BORDER_COLOR, border);
    glTextureParameteri(m_depthArray, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTextureParameteri(m_depthArray, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);

    // raw-depth sampler (no compare) for the PCSS blocker search
    glCreateSamplers(1, &m_rawSampler);
    glSamplerParameteri(m_rawSampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glSamplerParameteri(m_rawSampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glSamplerParameteri(m_rawSampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glSamplerParameteri(m_rawSampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    const float rawBorder[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    glSamplerParameterfv(m_rawSampler, GL_TEXTURE_BORDER_COLOR, rawBorder);

    glCreateFramebuffers(1, &m_fbo);
    glNamedFramebufferTextureLayer(m_fbo, GL_DEPTH_ATTACHMENT, m_depthArray, 0, 0);
    if (glCheckNamedFramebufferStatus(m_fbo, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        VX_ERROR("shadow framebuffer incomplete");
        return false;
    }
    return true;
}

bool ShadowMap::resize(unsigned int shadowProgram, int resolution) {
    if (resolution == m_resolution && shadowProgram == m_shadowProgram)
        return true;
    shutdown();
    return init(shadowProgram, resolution);
}

void ShadowMap::shutdown() {
    glDeleteFramebuffers(1, &m_fbo);
    glDeleteTextures(1, &m_depthArray);
    glDeleteSamplers(1, &m_rawSampler);
}

void ShadowMap::invalidate() {
    // Cascades 1 and 2 only ever check their OWN anchor drift; nothing about "new geometry
    // appeared under an anchor that hasn't moved" reaches them any other way, so something
    // has to force a redraw when a chunk streams in or an edit lands, or a chunk that pops
    // into an already-cached cascade would just never cast a shadow. That part is real and
    // this still does it.
    //
    // What was wrong is firing it unconditionally on every call. Streaming submits a mesh
    // upload most frames it's active, so a heavy burst called this every single frame for as
    // long as it lasted, and cascade caching exists specifically to avoid redrawing
    // cascades 1 and 2 every frame, which is 96% of all shadow draws (see the cache comment
    // above). During exactly the frames where the main thread is already paying the
    // heaviest cost this engine has (light seeding, see INFO.md #63), the GPU was ALSO
    // paying full uncached shadow cost on top of it, because the one thing that should have
    // stayed cheap kept invalidating itself.
    //
    // A shadow that lags the geometry it lights by up to one interval during active
    // streaming is not something a player can see, the mesh itself is still arriving
    // across those same frames, and the drift-slack mechanism above already spends exactly
    // this trade (staleness for redraw count) for camera movement. This is the same trade,
    // triggered by new geometry instead. Applies to edits too, not just streaming: a placed
    // block's own colour and shape still update same-frame (that path does not go through
    // this), only its cast shadow can lag by up to the interval, which reads as nothing at
    // normal edit rates.
    const auto now = std::chrono::steady_clock::now();
    if (now - m_lastForcedRedraw < kMinForceInterval)
        return;
    m_lastForcedRedraw = now;
    m_forceAll = true;
    ++m_forcedCount;
}

void ShadowMap::invalidateNear(float nearestBlocks) {
    // A cascade serves a band of distances; a changed block only matters to the pixels whose
    // shadows it can fall on, which is the band it sits in plus about a tree's height either
    // side. An edit beside the player therefore never touches the far cascade at all, and a
    // chunk streaming in at the horizon never touches the near one.
    constexpr float kReach = 64.0f;
    if (nearestBlocks <= m_splits.y + kReach)
        m_pending[1] = true;
    if (nearestBlocks + kReach >= m_splits.y)
        m_pending[2] = true;
}

void ShadowMap::setViewDistance(float blocks) {
    const float far = std::clamp(blocks, kFarMin, kFarMax);
    if (std::abs(far - m_splits.z) < 8.0f)
        return; // not worth redrawing the far cascade over a few blocks
    m_splits.z = far;
    m_forceAll = true;
}

void ShadowMap::setLeafGap(float gap) {
    if (std::abs(gap - m_leafGap) < 1e-3f)
        return;
    m_leafGap = gap;
    m_forceAll = true;
}

void ShadowMap::update(const glm::dvec3& cameraPos, const glm::vec3& sunL) {
    // light basis: looking along -sunL
    const glm::dvec3 lz = glm::normalize(-glm::dvec3(sunL));
    const glm::dvec3 upRef =
        std::abs(lz.y) > 0.95 ? glm::dvec3(1.0, 0.0, 0.0) : glm::dvec3(0.0, 1.0, 0.0);
    const glm::dvec3 lx = glm::normalize(glm::cross(upRef, lz));
    const glm::dvec3 ly = glm::cross(lz, lx);
    m_lightX = glm::vec3(lx);
    m_lightY = glm::vec3(ly);
    m_camLightXY = glm::vec2(float(glm::dot(lx, cameraPos)), float(glm::dot(ly, cameraPos)));

    const auto now = std::chrono::steady_clock::now();
    // the far cascade holds the most geometry by a distance, so it is the one made to wait
    constexpr std::chrono::milliseconds kInterval[kCascades] = {std::chrono::milliseconds(0),
                                                                std::chrono::milliseconds(60),
                                                                std::chrono::milliseconds(700)};
    // How far the sun may move before a cascade follows it (cosine of the angle): 0.57 degrees
    // for the mid one, 1.5 for the far one, whose texels are nine times the size and whose
    // shadows are seen from 130 blocks off. Following the sun at 0.57 there too restarted its
    // redraw, an eighth of ~30 ms a frame, every 0.4 s of a default day.
    constexpr float kSunStep[kCascades] = {1.0f, 0.99995f, 0.99966f};

    for (int c = 0; c < kCascades; ++c) {
        // The shaders pick a cascade by distance from the camera (length(pos) against the
        // splits), so cascade c only ever has to cover the ball of radius splits[c] around the
        // camera. It used to cover a bounding sphere of its slice of the view frustum instead,
        // and at any field of view the settings allow (50..110) that sphere is the bigger of
        // the two: at 85 degrees and 16:9 the far one had a radius of 1719 to shade points no
        // more than 760 away, so every texel was about twice the size it needed to be. Being
        // pushed out ahead of the camera, it also had to move, and the far one redrew the
        // whole visible world, every time the camera turned. A ball about the camera only
        // moves when the camera does. Inflated by the drift allowed before re-anchoring, so a
        // cached cascade still covers what it needs to.
        const float radius = m_splits[c] * (1.0f + kSlack);
        const float texel = radius * 2.0f / float(m_resolution);
        m_texelWorld[c] = texel;

        const glm::dvec3 idealWorld = cameraPos;

        // Cascade 0 always re-anchors: it's only ~130 chunks and contact shadows are the
        // ones you'd notice lagging. The far two re-anchor only once the ideal centre has
        // drifted past the slack, or the sun has actually moved.
        // They start early: a redraw lands up to kRedrawSec later, and until it does the old
        // contents must still cover the camera's ball at flying speed (40 blocks a second).
        const double drift = glm::length(idealWorld - m_anchorWorld[c]);
        const bool sunMoved = glm::dot(m_anchorSun[c], sunL) < kSunStep[c];
        const double reach = double(radius * kSlack) * (c == 1 ? 0.5 : 0.75);
        const bool reanchor = m_forceAll || c == 0 || m_radius[c] == 0.0f || drift > reach || sunMoved;

        // snap the centre to the texel grid in light space, double precision so the snap is
        // exact anywhere in the world and shadows never swim
        const auto snapped = [&] {
            const double cx = std::floor(glm::dot(lx, idealWorld) / texel) * texel;
            const double cy = std::floor(glm::dot(ly, idealWorld) / texel) * texel;
            return lx * cx + ly * cy + lz * glm::dot(lz, idealWorld);
        };
        // The mid and far cascades' routine redraws are drawn a slice a frame (see the header).
        // One already under way is what answers any drift or sun change against the old
        // contents, and a pending redraw waits for it rather than being spent on it. A jump the
        // old contents cannot cover at all (a teleport, a respawn) is drawn whole, as before.
        Spare& spare = m_spare[c];
        const bool sliced = c > 0 && !m_forceAll && m_radius[c] != 0.0f &&
                            drift <= 2.0 * double(radius * kSlack);
        if (!sliced)
            spare.slice = -1; // drawn whole this frame, so a half-drawn spare is stale
        bool redraw = reanchor;
        if (m_pending[c] && now - m_lastRedraw[c] >= kInterval[c] && !(sliced && spare.slice >= 0)) {
            m_pending[c] = false;
            redraw = true;
        }
        if (sliced) {
            if (redraw && spare.slice < 0) {
                m_lastRedraw[c] = now;
                spare.anchor = snapped();
                spare.sun = sunL;
                spare.up = glm::vec3(ly);
                spare.radius = radius;
                spare.slice = 0;
            }
        } else if (reanchor) {
            m_lastRedraw[c] = now;
            m_anchorWorld[c] = snapped();
            m_anchorSun[c] = sunL;
            m_anchorUp[c] = glm::vec3(ly);
            m_radius[c] = radius;
            m_dirty[c] = true;
        } else if (redraw) {
            m_lastRedraw[c] = now;
            m_dirty[c] = true;
        }

        // rebuilt every frame regardless: the depth map is anchored in WORLD space, but
        // the matrix the lighting shader samples with has to be camera-relative to THIS
        // frame's camera. The up vector is the one it was drawn with, so a sun that has moved
        // a little since does not turn the lookup against the map.
        m_vp[c] = cascadeMatrix(m_anchorWorld[c], m_anchorSun[c], m_anchorUp[c], m_radius[c],
                                cameraPos);
        if (spare.slice >= 0)
            spare.vp = cascadeMatrix(spare.anchor, spare.sun, spare.up, spare.radius, cameraPos);
    }
    m_forceAll = false;
}

glm::mat4 ShadowMap::cascadeMatrix(const glm::dvec3& anchor, const glm::vec3& sun,
                                   const glm::vec3& up, float r, const glm::dvec3& cameraPos) {
    const glm::vec3 centerRel(anchor - cameraPos);
    const glm::mat4 view = glm::lookAt(centerRel + sun * (r + 120.0f), centerRel, up);
    const glm::mat4 proj = glm::ortho(-r, r, -r, r, 0.1f, 2.0f * r + 160.0f);
    return proj * view;
}

void ShadowMap::render(Renderer& renderer, const glm::ivec3& cameraChunk,
                       const glm::vec3& cameraFrac) {
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    glViewport(0, 0, m_resolution, m_resolution);
    // shadow maps use conventional depth; the main pass runs reversed-z
    glDepthFunc(GL_LESS);
    glClearDepth(1.0);
    glCullFace(GL_FRONT); // closed voxel shells: front-culling trades acne for nothing
    glProgramUniform3fv(m_shadowProgram, 4, 1, &m_lightX.x);
    glProgramUniform3fv(m_shadowProgram, 5, 1, &m_lightY.x);
    glProgramUniform2fv(m_shadowProgram, 6, 1, &m_camLightXY.x);
    glProgramUniform1f(m_shadowProgram, 7, m_leafGap);

    for (int c = 0; c < kCascades; ++c) {
        if (!m_dirty[c])
            continue; // still anchored where it was drawn: the depth map is still valid
        m_dirty[c] = false;
        ++m_redraws;
        // cull to this cascade's ortho box: the near cascade draws a handful of chunks
        // instead of the whole world. cave culling stays off here, a chunk hidden from
        // the camera can still cast a shadow the camera sees
        // Only the far cascade skips sealed chunks. This pass draws faces turned from the light, so
        // the ground over a cave casts nothing and a cave's own ceiling is all that shades it. Far off
        // nobody sees into a cave, and the sky level keeps the sun out of it anyway.
        renderer.cullFrustum(Frustum::fromOrtho(m_vp[c]), cameraChunk, cameraFrac, false, c == kCascades - 1);
        glNamedFramebufferTextureLayer(m_fbo, GL_DEPTH_ATTACHMENT, m_depthArray, 0, c);
        glClear(GL_DEPTH_BUFFER_BIT);
        renderer.drawIndirect(m_shadowProgram, m_vp[c]);
    }

    // A slice of each routine redraw under way, into that cascade's spare layer: more than one
    // at a low frame rate, so the redraw is done before the camera can outrun the old contents.
    const auto now = std::chrono::steady_clock::now();
    const float dt = std::clamp(std::chrono::duration<float>(now - m_lastRender).count(), 0.0f, 0.25f);
    m_lastRender = now;
    for (int c = 1; c < kCascades; ++c) {
        Spare& spare = m_spare[c];
        if (spare.slice < 0)
            continue;
        const int due = int(std::ceil(float(kSlices[c]) * dt / kRedrawSec[c]));
        const int last = std::min(spare.slice + std::max(due, 1), kSlices[c]);
        glNamedFramebufferTextureLayer(m_fbo, GL_DEPTH_ATTACHMENT, m_depthArray, 0, spareLayer(c));
        for (; spare.slice < last; ++spare.slice) {
            renderer.cullFrustum(Frustum::fromOrtho(spare.vp), cameraChunk, cameraFrac, false,
                                 c == kCascades - 1, unsigned(spare.slice), unsigned(kSlices[c]));
            if (spare.slice == 0)
                glClear(GL_DEPTH_BUFFER_BIT);
            renderer.drawIndirect(m_shadowProgram, spare.vp);
        }
        if (spare.slice == kSlices[c]) {
            // complete: it replaces the cascade from this frame's lighting on
            glCopyImageSubData(m_depthArray, GL_TEXTURE_2D_ARRAY, 0, 0, 0, spareLayer(c), m_depthArray,
                               GL_TEXTURE_2D_ARRAY, 0, 0, 0, c, m_resolution, m_resolution, 1);
            m_anchorWorld[c] = spare.anchor;
            m_anchorSun[c] = spare.sun;
            m_anchorUp[c] = spare.up;
            m_radius[c] = spare.radius;
            m_vp[c] = spare.vp;
            spare.slice = -1;
            ++m_redraws;
        }
    }

    glCullFace(GL_BACK);
    glDepthFunc(GL_GREATER);
    glClearDepth(0.0);
}

} // namespace vx
