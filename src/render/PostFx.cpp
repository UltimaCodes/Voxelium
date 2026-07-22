#include "render/PostFx.hpp"

#include "core/Log.hpp"
#include "gfx/Shader.hpp"
#include "render/ShadowMap.hpp"

#include <glad/gl.h>

#include <algorithm>

namespace vx {

namespace {

unsigned int makeTex2D(GLenum format, int w, int h, GLenum filter = GL_LINEAR) {
    GLuint tex = 0;
    glCreateTextures(GL_TEXTURE_2D, 1, &tex);
    glTextureStorage2D(tex, 1, format, w, h);
    glTextureParameteri(tex, GL_TEXTURE_MIN_FILTER, GLint(filter));
    glTextureParameteri(tex, GL_TEXTURE_MAG_FILTER, GLint(filter));
    glTextureParameteri(tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return tex;
}

unsigned int makeFbo(unsigned int color, unsigned int depth = 0) {
    GLuint fbo = 0;
    glCreateFramebuffers(1, &fbo);
    glNamedFramebufferTexture(fbo, GL_COLOR_ATTACHMENT0, color, 0);
    if (depth)
        glNamedFramebufferTexture(fbo, GL_DEPTH_ATTACHMENT, depth, 0);
    if (glCheckNamedFramebufferStatus(fbo, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        VX_ERROR("postfx framebuffer incomplete");
    return fbo;
}

} // namespace

bool PostFx::init(const std::filesystem::path& shaderDir) {
    const auto fs = shaderDir / "fullscreen.vert";
    m_progSsao = loadProgram(fs, shaderDir / "ssao.frag");
    m_progBlur = loadProgram(fs, shaderDir / "blur4.frag");
    m_progComposite = loadProgram(fs, shaderDir / "composite.frag");
    m_progBloomDown = loadProgram(fs, shaderDir / "bloom_down.frag");
    m_progBloomUp = loadProgram(fs, shaderDir / "bloom_up.frag");
    m_progFinal = loadProgram(fs, shaderDir / "final.frag");
    if (!m_progSsao || !m_progBlur || !m_progComposite || !m_progBloomDown ||
        !m_progBloomUp || !m_progFinal)
        return false;
    glCreateVertexArrays(1, &m_vao);
    return true;
}

void PostFx::shutdown() {
    destroyTargets();
    const GLuint progs[] = {m_progSsao, m_progBlur, m_progComposite,
                            m_progBloomDown, m_progBloomUp, m_progFinal};
    for (const GLuint p : progs)
        glDeleteProgram(p);
    glDeleteVertexArrays(1, &m_vao);
}

void PostFx::destroyTargets() {
    if (m_width == 0)
        return;
    const GLuint fbos[] = {m_hdrFbo, m_hdr2Fbo, m_ssaoFbo, m_ssaoBlurFbo};
    glDeleteFramebuffers(4, fbos);
    glDeleteFramebuffers(kBloomMips, m_bloomFbo);
    const GLuint texes[] = {m_hdrColor, m_depth,   m_hdr2,
                            m_ssaoTex,  m_ssaoBlurTex, m_sceneCopy};
    glDeleteTextures(6, texes);
    glDeleteTextures(kBloomMips, m_bloomTex);
    m_width = 0;
}

void PostFx::createTargets(int width, int height) {
    m_width = width;
    m_height = height;

    m_hdrColor = makeTex2D(GL_RGBA16F, width, height);
    m_depth = makeTex2D(GL_DEPTH_COMPONENT32F, width, height, GL_NEAREST);
    m_hdrFbo = makeFbo(m_hdrColor, m_depth);
    m_sceneCopy = makeTex2D(GL_RGBA16F, width, height);

    m_hdr2 = makeTex2D(GL_RGBA16F, width, height);
    m_hdr2Fbo = makeFbo(m_hdr2);

    const int hw = std::max(width / 2, 1);
    const int hh = std::max(height / 2, 1);
    // RG16F: .r = AO, .g = linear view depth for the bilateral blur + composite
    m_ssaoTex = makeTex2D(GL_RG16F, hw, hh);
    m_ssaoFbo = makeFbo(m_ssaoTex);
    m_ssaoBlurTex = makeTex2D(GL_RG16F, hw, hh);
    m_ssaoBlurFbo = makeFbo(m_ssaoBlurTex);

    int bw = width;
    int bh = height;
    for (int i = 0; i < kBloomMips; ++i) {
        bw = std::max(bw / 2, 1);
        bh = std::max(bh / 2, 1);
        m_bloomSize[i] = {bw, bh};
        m_bloomTex[i] = makeTex2D(GL_RGBA16F, bw, bh);
        m_bloomFbo[i] = makeFbo(m_bloomTex[i]);
    }
}

void PostFx::resize(int width, int height) {
    if (width == m_width && height == m_height)
        return;
    destroyTargets();
    createTargets(width, height);
}

void PostFx::beginScene() {
    glBindFramebuffer(GL_FRAMEBUFFER, m_hdrFbo);
    glViewport(0, 0, m_width, m_height);
}

void PostFx::captureOpaqueForWater() {
    // GPU-side copy of the opaque color; the water shader reads it for SSR while it
    // blends new pixels into m_hdrColor (can't read+write the same texture).
    glCopyImageSubData(m_hdrColor, GL_TEXTURE_2D, 0, 0, 0, 0, m_sceneCopy, GL_TEXTURE_2D, 0,
                       0, 0, 0, m_width, m_height, 1);
    glBindFramebuffer(GL_FRAMEBUFFER, m_hdrFbo);
    glViewport(0, 0, m_width, m_height);
}

void PostFx::run(const RunParams& p) {
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glBindVertexArray(m_vao);

    // ssao at half res, then a box blur over the rotation noise
    if (p.toggles.ssao) {
        glBindFramebuffer(GL_FRAMEBUFFER, m_ssaoFbo);
        glViewport(0, 0, std::max(m_width / 2, 1), std::max(m_height / 2, 1));
        glUseProgram(m_progSsao);
        glProgramUniform2fv(m_progSsao, 0, 1, &p.projScale.x);
        glProgramUniform1f(m_progSsao, 1, p.nearPlane);
        glBindTextureUnit(0, m_depth);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glBindFramebuffer(GL_FRAMEBUFFER, m_ssaoBlurFbo);
        glUseProgram(m_progBlur);
        glBindTextureUnit(0, m_ssaoTex);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }

    // composite: AO + volumetric scattering, still HDR
    glBindFramebuffer(GL_FRAMEBUFFER, m_hdr2Fbo);
    glViewport(0, 0, m_width, m_height);
    glUseProgram(m_progComposite);
    glProgramUniform2fv(m_progComposite, 0, 1, &p.projScale.x);
    glProgramUniform1f(m_progComposite, 1, p.nearPlane);
    glProgramUniformMatrix3fv(m_progComposite, 2, 1, GL_FALSE, &p.invViewRot[0][0]);
    glProgramUniform1f(m_progComposite, 3, p.toggles.ssao ? 1.0f : 0.0f);
    glProgramUniform1f(m_progComposite, 4,
                       p.toggles.volumetric && p.toggles.shadows ? 1.0f : 0.0f);
    glProgramUniform3fv(m_progComposite, 5, 1, &p.sunL.x);
    glProgramUniformMatrix4fv(m_progComposite, 6, 3, GL_FALSE,
                              &p.shadows->cascadeVp()[0][0][0]);
    const glm::vec3 splits{ShadowMap::kSplits[0], ShadowMap::kSplits[1],
                           ShadowMap::kSplits[2]};
    glProgramUniform3fv(m_progComposite, 18, 1, &splits.x);
    glBindTextureUnit(0, m_hdrColor);
    glBindTextureUnit(1, p.shadows->texture());
    glBindTextureUnit(2, m_ssaoBlurTex);
    glBindTextureUnit(3, m_depth);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // bloom: 13-tap down chain, tent up chain with additive blending
    if (p.toggles.bloom) {
        glUseProgram(m_progBloomDown);
        GLuint src = m_hdr2;
        for (int i = 0; i < kBloomMips; ++i) {
            // threshold only the first tap so bloom picks up bright highlights, not haze
            glProgramUniform1f(m_progBloomDown, 0, i == 0 ? 1.15f : 0.0f);
            glProgramUniform1f(m_progBloomDown, 1, 0.6f);
            glBindFramebuffer(GL_FRAMEBUFFER, m_bloomFbo[i]);
            glViewport(0, 0, m_bloomSize[i].x, m_bloomSize[i].y);
            glBindTextureUnit(0, src);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            src = m_bloomTex[i];
        }
        glUseProgram(m_progBloomUp);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);
        for (int i = kBloomMips - 1; i > 0; --i) {
            glBindFramebuffer(GL_FRAMEBUFFER, m_bloomFbo[i - 1]);
            glViewport(0, 0, m_bloomSize[i - 1].x, m_bloomSize[i - 1].y);
            glBindTextureUnit(0, m_bloomTex[i]);
            glDrawArrays(GL_TRIANGLES, 0, 3);
        }
        glDisable(GL_BLEND);
    }

    // tonemap + sRGB into the caller's framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, p.targetFbo);
    glViewport(0, 0, p.targetWidth, p.targetHeight);
    glUseProgram(m_progFinal);
    glProgramUniform1f(m_progFinal, 0, p.toggles.bloom ? 0.55f : 0.0f);
    glProgramUniform1f(m_progFinal, 1, p.toggles.exposure);
    glBindTextureUnit(0, m_hdr2);
    glBindTextureUnit(2, m_bloomTex[0]);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
}

} // namespace vx
