#include "render/PostFx.hpp"

#include "core/Log.hpp"
#include "gfx/Shader.hpp"
#include "render/ShadowMap.hpp"

#include <glad/gl.h>

#include <algorithm>
#include <cmath>

namespace vx {

namespace {

unsigned int makeTex2D(GLenum format, int w, int h, GLenum filter = GL_LINEAR, int mips = 1) {
    GLuint tex = 0;
    glCreateTextures(GL_TEXTURE_2D, 1, &tex);
    glTextureStorage2D(tex, mips, format, w, h);
    glTextureParameteri(tex, GL_TEXTURE_MIN_FILTER,
                        mips > 1 ? GL_LINEAR_MIPMAP_LINEAR : GLint(filter));
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
    // all nine submitted before any of them is waited on, so the driver's compiler threads
    // have the whole set to chew through at once instead of one program at a time
    const auto fs = shaderDir / "fullscreen.vert";
    ProgramJob jobs[] = {
        submitProgram(fs, shaderDir / "ssao.frag"),
        submitProgram(fs, shaderDir / "blur4.frag"),
        submitProgram(fs, shaderDir / "composite.frag"),
        submitProgram(fs, shaderDir / "bloom_down.frag"),
        submitProgram(fs, shaderDir / "bloom_up.frag"),
        submitProgram(fs, shaderDir / "final.frag"),
        submitProgram(fs, shaderDir / "taa.frag"),
        submitProgram(fs, shaderDir / "ssgi.frag"),
        submitProgram(fs, shaderDir / "motionblur.frag"),
        submitProgram(fs, shaderDir / "lumdown.frag"),
        submitProgram(fs, shaderDir / "expose.frag"),
    };
    m_progSsao = resolveProgram(jobs[0]);
    m_progBlur = resolveProgram(jobs[1]);
    m_progComposite = resolveProgram(jobs[2]);
    m_progBloomDown = resolveProgram(jobs[3]);
    m_progBloomUp = resolveProgram(jobs[4]);
    m_progFinal = resolveProgram(jobs[5]);
    m_progTaa = resolveProgram(jobs[6]);
    m_progSsgi = resolveProgram(jobs[7]);
    m_progMotion = resolveProgram(jobs[8]);
    m_progLumDown = resolveProgram(jobs[9]);
    m_progExpose = resolveProgram(jobs[10]);
    if (!m_progSsao || !m_progBlur || !m_progComposite || !m_progBloomDown ||
        !m_progBloomUp || !m_progFinal || !m_progTaa || !m_progSsgi || !m_progMotion ||
        !m_progLumDown || !m_progExpose)
        return false;
    glCreateVertexArrays(1, &m_vao);
    return true;
}

void PostFx::shutdown() {
    destroyTargets();
    const GLuint progs[] = {m_progSsao,  m_progBlur,    m_progComposite, m_progBloomDown,
                            m_progBloomUp, m_progFinal, m_progTaa,       m_progSsgi,
                            m_progMotion, m_progLumDown, m_progExpose};
    for (const GLuint p : progs)
        glDeleteProgram(p);
    glDeleteVertexArrays(1, &m_vao);
}

void PostFx::destroyTargets() {
    if (m_width == 0)
        return;
    const GLuint fbos[] = {m_hdrFbo, m_hdr2Fbo, m_ssaoFbo, m_ssaoBlurFbo, m_taaFbo[0],
                           m_taaFbo[1]};
    glDeleteFramebuffers(6, fbos);
    glDeleteFramebuffers(kBloomMips, m_bloomFbo);
    const GLuint texes[] = {m_hdrColor,  m_depth,  m_hdr2,   m_ssaoTex,  m_ssaoBlurTex,
                            m_sceneCopy, m_taa[0], m_taa[1], m_ssgiTex, m_motionTex};
    glDeleteTextures(10, texes);
    glDeleteFramebuffers(1, &m_ssgiFbo);
    glDeleteFramebuffers(1, &m_motionFbo);
    glDeleteFramebuffers(1, &m_lumFbo);
    glDeleteFramebuffers(2, m_adaptFbo);
    glDeleteTextures(1, &m_lumTex);
    glDeleteTextures(2, m_adapt);
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

    for (int i = 0; i < 2; ++i) {
        m_taa[i] = makeTex2D(GL_RGBA16F, width, height);
        m_taaFbo[i] = makeFbo(m_taa[i]);
    }
    m_taaValid = false; // new targets: no history to reproject from yet

    m_motionTex = makeTex2D(GL_RGBA16F, width, height);
    m_motionFbo = makeFbo(m_motionTex);

    const int hw = std::max(width / 2, 1);
    const int hh = std::max(height / 2, 1);
    // RG16F: .r = AO, .g = linear view depth for the bilateral blur + composite
    m_ssaoTex = makeTex2D(GL_RG16F, hw, hh);
    m_ssaoFbo = makeFbo(m_ssaoTex);
    m_ssaoBlurTex = makeTex2D(GL_RG16F, hw, hh);
    m_ssaoBlurFbo = makeFbo(m_ssaoBlurTex);

    // half-res coloured bounce; LINEAR so the composite upsample is smooth
    m_ssgiTex = makeTex2D(GL_RGBA16F, hw, hh);
    m_ssgiFbo = makeFbo(m_ssgiTex);

    // Auto-exposure: a mipped log-luminance square whose last mip is the scene average, and
    // two 1x1 textures the adaptation ping-pongs between. Fixed size, so this costs the
    // same at any resolution and does not need rebuilding on resize (it is here only
    // because everything else target-shaped is).
    m_lumTex = makeTex2D(GL_R16F, kLumSize, kLumSize, GL_LINEAR, kLumMips);
    m_lumFbo = makeFbo(m_lumTex);
    for (int i = 0; i < 2; ++i) {
        m_adapt[i] = makeTex2D(GL_R16F, 1, 1);
        m_adaptFbo[i] = makeFbo(m_adapt[i]);
    }
    m_adaptValid = false; // nothing to adapt from until the first luminance pass has run

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

    // screen-space GI: gather one bounce of coloured light at half res from the lit scene
    if (p.toggles.ssgi) {
        glBindFramebuffer(GL_FRAMEBUFFER, m_ssgiFbo);
        glViewport(0, 0, std::max(m_width / 2, 1), std::max(m_height / 2, 1));
        glUseProgram(m_progSsgi);
        glProgramUniform2fv(m_progSsgi, 0, 1, &p.projScale.x);
        glProgramUniform1f(m_progSsgi, 1, p.nearPlane);
        glProgramUniform1f(m_progSsgi, 2, p.time);
        glBindTextureUnit(0, m_depth);
        glBindTextureUnit(1, m_hdrColor);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }

    // composite: AO + volumetric scattering, still HDR
    glBindFramebuffer(GL_FRAMEBUFFER, m_hdr2Fbo);
    glViewport(0, 0, m_width, m_height);
    glUseProgram(m_progComposite);
    glProgramUniform2fv(m_progComposite, 0, 1, &p.projScale.x);
    glProgramUniform1f(m_progComposite, 1, p.nearPlane);
    glProgramUniformMatrix3fv(m_progComposite, 2, 1, GL_FALSE, &p.invViewRot[0][0]);
    glProgramUniform1f(m_progComposite, 23, p.toggles.ssao ? 1.0f : 0.0f);
    glProgramUniform1f(m_progComposite, 24,
                       p.toggles.volumetric && p.toggles.shadows ? 1.0f : 0.0f);
    glProgramUniform3fv(m_progComposite, 5, 1, &p.sunL.x);
    glProgramUniform3fv(m_progComposite, 25, 1, &p.sunColor.x);
    glProgramUniform2fv(m_progComposite, 26, 1, &p.fog.x);
    glProgramUniform1f(m_progComposite, 27, p.toggles.lensFlare);
    glProgramUniform3fv(m_progComposite, 28, 1, &p.camPos.x);
    glProgramUniformMatrix4fv(m_progComposite, 6, 3, GL_FALSE,
                              &p.shadows->cascadeVp()[0][0][0]);
    const glm::vec3 splits{ShadowMap::kSplits[0], ShadowMap::kSplits[1],
                           ShadowMap::kSplits[2]};
    glProgramUniform3fv(m_progComposite, 18, 1, &splits.x);
    glProgramUniform1f(m_progComposite, 19, p.underwater ? 1.0f : 0.0f);
    glProgramUniform1f(m_progComposite, 20, p.toggles.clouds ? 1.0f : 0.0f);
    glProgramUniform1f(m_progComposite, 21, p.time);
    glProgramUniform1f(m_progComposite, 22, p.toggles.ssgi ? 1.0f : 0.0f);
    glBindTextureUnit(0, m_hdrColor);
    glBindTextureUnit(1, p.shadows->texture());
    glBindTextureUnit(2, m_ssaoBlurTex);
    glBindTextureUnit(3, m_depth);
    glBindTextureUnit(4, m_ssgiTex);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // temporal AA: reproject + blend last frame's resolve into this one. everything
    // downstream (bloom, tonemap) reads the resolved image instead of the raw composite.
    GLuint resolved = m_hdr2;
    if (p.toggles.taa) {
        const int cur = m_taaIndex, prev = cur ^ 1;
        glBindFramebuffer(GL_FRAMEBUFFER, m_taaFbo[cur]);
        glViewport(0, 0, m_width, m_height);
        glUseProgram(m_progTaa);
        glProgramUniform2fv(m_progTaa, 0, 1, &p.projScale.x);
        glProgramUniform1f(m_progTaa, 1, p.nearPlane);
        glProgramUniformMatrix3fv(m_progTaa, 2, 1, GL_FALSE, &p.invViewRot[0][0]);
        glProgramUniformMatrix4fv(m_progTaa, 5, 1, GL_FALSE, &p.prevViewProj[0][0]);
        glProgramUniform3fv(m_progTaa, 9, 1, &p.camDelta.x);
        glProgramUniform1f(m_progTaa, 10,
                           (m_taaValid && p.taaHistoryValid) ? 1.0f : 0.0f);
        const glm::vec2 texel(1.0f / float(m_width), 1.0f / float(m_height));
        glProgramUniform2fv(m_progTaa, 11, 1, &texel.x);
        glBindTextureUnit(0, m_hdr2);
        glBindTextureUnit(1, m_taa[prev]);
        glBindTextureUnit(2, m_depth);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        resolved = m_taa[cur];
        m_taaIndex = prev; // next frame reads what we just wrote
        m_taaValid = true;
    } else {
        m_taaValid = false; // stale while off; the next on-frame reprojects from scratch
    }

    // camera motion blur: reproject velocity from depth + prev view-proj, blur along it
    if (p.toggles.motionBlur) {
        glBindFramebuffer(GL_FRAMEBUFFER, m_motionFbo);
        glViewport(0, 0, m_width, m_height);
        glUseProgram(m_progMotion);
        glProgramUniform2fv(m_progMotion, 0, 1, &p.projScale.x);
        glProgramUniform1f(m_progMotion, 1, p.nearPlane);
        glProgramUniformMatrix3fv(m_progMotion, 2, 1, GL_FALSE, &p.invViewRot[0][0]);
        glProgramUniformMatrix4fv(m_progMotion, 5, 1, GL_FALSE, &p.prevViewProj[0][0]);
        glProgramUniform3fv(m_progMotion, 9, 1, &p.camDelta.x);
        glProgramUniform1f(m_progMotion, 10, p.taaHistoryValid ? 0.5f : 0.0f); // 0 = no prev yet
        glBindTextureUnit(0, resolved);
        glBindTextureUnit(1, m_depth);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        resolved = m_motionTex;
    }

    // bloom: 13-tap down chain, tent up chain with additive blending
    if (p.toggles.bloom) {
        glUseProgram(m_progBloomDown);
        GLuint src = resolved;
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

    // Eye adaptation, from the resolved HDR image and before the tonemap that consumes it.
    // Three tiny passes (64x64, a mip chain, and one texel), so the cost is noise next to
    // any of the full-res passes above.
    if (p.toggles.autoExposure) {
        glBindFramebuffer(GL_FRAMEBUFFER, m_lumFbo);
        glViewport(0, 0, kLumSize, kLumSize);
        glUseProgram(m_progLumDown);
        const glm::vec2 srcTexel(1.0f / float(m_width), 1.0f / float(m_height));
        glProgramUniform2fv(m_progLumDown, 0, 1, &srcTexel.x);
        glBindTextureUnit(0, resolved);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glGenerateTextureMipmap(m_lumTex); // last mip is the geometric mean of the frame

        const int next = 1 - m_adaptIndex;
        glBindFramebuffer(GL_FRAMEBUFFER, m_adaptFbo[next]);
        glViewport(0, 0, 1, 1);
        glUseProgram(m_progExpose);
        glProgramUniform1f(m_progExpose, 0, float(kLumMips - 1));
        // Time constants, not per-frame lerps: a fixed blend factor would adapt at a rate
        // that depends on framerate, so the same walk into a cave would settle differently
        // at 60 and 144 fps.
        const float dt = std::clamp(p.dt, 0.0f, 0.25f);
        glProgramUniform1f(m_progExpose, 1, 1.0f - std::exp(-dt / 0.6f)); // into light
        glProgramUniform1f(m_progExpose, 2, 1.0f - std::exp(-dt / 1.8f)); // into darkness
        glProgramUniform1f(m_progExpose, 3, m_adaptValid ? 1.0f : 0.0f);
        glBindTextureUnit(0, m_lumTex);
        glBindTextureUnit(1, m_adapt[m_adaptIndex]);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        m_adaptIndex = next;
        m_adaptValid = true;
    }

    // tonemap + sRGB into the caller's framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, p.targetFbo);
    glViewport(0, 0, p.targetWidth, p.targetHeight);
    glUseProgram(m_progFinal);
    glProgramUniform1f(m_progFinal, 0, p.toggles.bloom ? p.toggles.bloomStrength : 0.0f);
    glProgramUniform1f(m_progFinal, 1, p.toggles.exposure);
    // sharpen only when TAA is on, to undo its softening (off = already crisp/aliased)
    glProgramUniform1f(m_progFinal, 3, p.toggles.taa ? 0.22f : 0.0f);
    const glm::vec2 texel(1.0f / float(m_width), 1.0f / float(m_height));
    glProgramUniform2fv(m_progFinal, 4, 1, &texel.x);
    glProgramUniform1f(m_progFinal, 5, p.toggles.vignette);
    glProgramUniform1f(m_progFinal, 6, p.toggles.grain);
    glProgramUniform1f(m_progFinal, 7, p.time);
    glProgramUniform1f(m_progFinal, 8, p.toggles.saturation);
    glProgramUniform1f(m_progFinal, 9, p.toggles.contrast);
    glProgramUniform1f(m_progFinal, 10, p.toggles.temperature);
    // 0.18 is middle grey. Only meaningful when the adaptation actually ran this frame.
    glProgramUniform1f(m_progFinal, 11,
                       (p.toggles.autoExposure && m_adaptValid) ? 0.18f : 0.0f);
    glBindTextureUnit(0, resolved);
    glBindTextureUnit(2, m_bloomTex[0]);
    glBindTextureUnit(3, m_adapt[m_adaptIndex]);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
}

} // namespace vx
