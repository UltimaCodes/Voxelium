#include "render/ShadowMap.hpp"

#include "core/Log.hpp"
#include "render/Frustum.hpp"
#include "render/Renderer.hpp"

#include <glad/gl.h>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

namespace vx {

bool ShadowMap::init(unsigned int shadowProgram, int resolution) {
    m_resolution = resolution;
    m_shadowProgram = shadowProgram;

    glCreateTextures(GL_TEXTURE_2D_ARRAY, 1, &m_depthArray);
    glTextureStorage3D(m_depthArray, 1, GL_DEPTH_COMPONENT32F, m_resolution, m_resolution,
                       kCascades);
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

void ShadowMap::update(const glm::dvec3& cameraPos, const glm::vec3& forward,
                       const glm::vec3& sunL, float fovYRad, float aspect) {
    // light basis: looking along -sunL
    const glm::dvec3 lz = glm::normalize(-glm::dvec3(sunL));
    const glm::dvec3 upRef =
        std::abs(lz.y) > 0.95 ? glm::dvec3(1.0, 0.0, 0.0) : glm::dvec3(0.0, 1.0, 0.0);
    const glm::dvec3 lx = glm::normalize(glm::cross(upRef, lz));
    const glm::dvec3 ly = glm::cross(lz, lx);

    const float tanH = std::tan(fovYRad * 0.5f);
    const float tanW = tanH * aspect;

    for (int c = 0; c < kCascades; ++c) {
        const float nearD = c == 0 ? 0.1f : kSplits[c - 1];
        const float farD = kSplits[c];
        // bounding sphere of the frustum slice, inflated by the drift we allow before
        // re-anchoring, so a cached cascade still covers what it needs to
        const float tc = (nearD + farD) * 0.5f;
        const float radius =
            glm::length(glm::vec3(tanW * farD, tanH * farD, farD - tc)) * (1.0f + kSlack);
        const float texel = radius * 2.0f / float(m_resolution);
        m_texelWorld[c] = texel;

        const glm::dvec3 idealWorld = cameraPos + glm::dvec3(forward) * double(tc);

        // Cascade 0 always re-anchors: it's only ~130 chunks and contact shadows are the
        // ones you'd notice lagging. The far two re-anchor only once the ideal centre has
        // drifted past the slack, or the sun has actually moved.
        const double drift = glm::length(idealWorld - m_anchorWorld[c]);
        const bool sunMoved = glm::dot(m_anchorSun[c], sunL) < 0.99995f;
        const bool reanchor = m_forceAll || c == 0 || m_radius[c] == 0.0f ||
                              drift > double(radius * kSlack) || sunMoved;

        if (reanchor) {
            // snap the centre to the texel grid in light space — double precision so the
            // snap is exact anywhere in the world and shadows never swim
            double cx = glm::dot(lx, idealWorld);
            double cy = glm::dot(ly, idealWorld);
            const double cz = glm::dot(lz, idealWorld);
            cx = std::floor(cx / texel) * texel;
            cy = std::floor(cy / texel) * texel;
            m_anchorWorld[c] = lx * cx + ly * cy + lz * cz;
            m_anchorSun[c] = sunL;
            m_radius[c] = radius;
            m_dirty[c] = true;
        }

        // rebuilt every frame regardless: the depth map is anchored in WORLD space, but
        // the matrix the lighting shader samples with has to be camera-relative to THIS
        // frame's camera
        const float r = m_radius[c];
        const glm::vec3 centerRel(m_anchorWorld[c] - cameraPos);
        const glm::mat4 view = glm::lookAt(centerRel - m_anchorSun[c] * (r + 120.0f),
                                           centerRel, glm::vec3(ly));
        const glm::mat4 proj = glm::ortho(-r, r, -r, r, 0.1f, 2.0f * r + 160.0f);
        m_vp[c] = proj * view;
    }
    m_forceAll = false;
}

void ShadowMap::render(Renderer& renderer, const glm::ivec3& cameraChunk,
                       const glm::vec3& cameraFrac) {
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    glViewport(0, 0, m_resolution, m_resolution);
    // shadow maps use conventional depth; the main pass runs reversed-z
    glDepthFunc(GL_LESS);
    glClearDepth(1.0);
    glCullFace(GL_FRONT); // closed voxel shells: front-culling trades acne for nothing

    for (int c = 0; c < kCascades; ++c) {
        if (!m_dirty[c])
            continue; // still anchored where it was drawn: the depth map is still valid
        m_dirty[c] = false;
        // cull to this cascade's ortho box: the near cascade draws a handful of chunks
        // instead of the whole world. cave culling stays off here — a chunk hidden from
        // the camera can still cast a shadow the camera sees
        renderer.cullFrustum(Frustum::fromOrtho(m_vp[c]), cameraChunk, cameraFrac, false, true);
        glNamedFramebufferTextureLayer(m_fbo, GL_DEPTH_ATTACHMENT, m_depthArray, 0, c);
        glClear(GL_DEPTH_BUFFER_BIT);
        renderer.drawIndirect(m_shadowProgram, m_vp[c]);
    }

    glCullFace(GL_BACK);
    glDepthFunc(GL_GREATER);
    glClearDepth(0.0);
}

} // namespace vx
