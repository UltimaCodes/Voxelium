#include "render/PostFx.hpp"

#include "core/Log.hpp"
#include "gfx/Shader.hpp"
#include "render/ShadowMap.hpp"

#include <glad/gl.h>

#include <algorithm>
#include <cmath>
#include <iterator>

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

bool PostFx::init(const std::filesystem::path& shaderDir, std::string_view defines) {
    m_shaderDir = shaderDir;
    return reloadPrograms(defines);
}

bool PostFx::reloadPrograms(std::string_view defines) {
    const std::filesystem::path& shaderDir = m_shaderDir;
    // all submitted before any of them is waited on, so the driver's compiler threads
    // have the whole set to chew through at once instead of one program at a time
    const auto fs = shaderDir / "fullscreen.vert";
    ProgramJob jobs[] = {
        submitProgram(fs, shaderDir / "ssao.frag", defines),
        submitProgram(fs, shaderDir / "blur4.frag", defines),
        submitProgram(fs, shaderDir / "composite.frag", defines),
        submitProgram(fs, shaderDir / "bloom_down.frag", defines),
        submitProgram(fs, shaderDir / "bloom_up.frag", defines),
        submitProgram(fs, shaderDir / "final.frag", defines),
        submitProgram(fs, shaderDir / "taa.frag", defines),
        submitProgram(fs, shaderDir / "motionblur.frag", defines),
        submitProgram(fs, shaderDir / "apply_ao.frag", defines),
        submitProgram(fs, shaderDir / "cloudmap.frag", defines),
        submitProgram(fs, shaderDir / "scatter.frag", defines),
        submitProgram(fs, shaderDir / "clouds3d.frag", defines),
    };
    unsigned int built[std::size(jobs)];
    for (std::size_t i = 0; i < std::size(jobs); ++i)
        built[i] = resolveProgram(jobs[i]);
    for (const unsigned int p : built) {
        if (p == 0) { // leave the working set alone rather than half-swapping it
            for (const unsigned int q : built)
                if (q)
                    glDeleteProgram(q);
            return false;
        }
    }
    // Only now is it safe to drop the old set.
    const unsigned int old[] = {m_progSsao,    m_progBlur,  m_progComposite, m_progBloomDown,
                                m_progBloomUp, m_progFinal, m_progTaa,       m_progMotion,
                                m_progApplyAo, m_progCloudMap, m_progScatter, m_progClouds3d};
    for (const unsigned int p : old)
        if (p)
            glDeleteProgram(p);

    m_progSsao = built[0];
    m_progBlur = built[1];
    m_progComposite = built[2];
    m_progBloomDown = built[3];
    m_progBloomUp = built[4];
    m_progFinal = built[5];
    m_progTaa = built[6];
    m_progMotion = built[7];
    m_progApplyAo = built[8];
    m_progCloudMap = built[9];
    m_progScatter = built[10];
    m_progClouds3d = built[11];
    if (m_vao == 0)
        glCreateVertexArrays(1, &m_vao);
    if (m_cloudTex == 0) {
        m_cloudTex = makeTex2D(GL_R8, kCloudMapRes, kCloudMapRes, GL_LINEAR, 6);
        m_cloudFbo = makeFbo(m_cloudTex);
    }
    if (m_cloudShape == 0)
        bakeCloudNoise();
    if (m_causticsTex == 0) {
        m_progCaustics = loadComputeProgram(m_shaderDir / "caustics_tex.comp");
        glCreateTextures(GL_TEXTURE_2D, 1, &m_causticsTex);
        glTextureStorage2D(m_causticsTex, 1, GL_R8, 512, 512);
        glTextureParameteri(m_causticsTex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTextureParameteri(m_causticsTex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTextureParameteri(m_causticsTex, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTextureParameteri(m_causticsTex, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glClearTexImage(m_causticsTex, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
    }
    return true;
}

void PostFx::updateCaustics(float time) {
    if (m_progCaustics == 0)
        return;
    glUseProgram(m_progCaustics);
    glProgramUniform1f(m_progCaustics, 0, time);
    glBindImageTexture(0, m_causticsTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R8);
    glDispatchCompute(512 / 8, 512 / 8, 1);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
}

void PostFx::bakeCloudNoise() {
    const auto make3d = [](int size, int mips) {
        GLuint t = 0;
        glCreateTextures(GL_TEXTURE_3D, 1, &t);
        glTextureStorage3D(t, mips, GL_R8, size, size, size);
        glTextureParameteri(t, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTextureParameteri(t, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTextureParameteri(t, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTextureParameteri(t, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTextureParameteri(t, GL_TEXTURE_WRAP_R, GL_REPEAT);
        return t;
    };
    constexpr int kShape = 64, kDetail = 32, kWeather = 512;
    m_cloudShape = make3d(kShape, 5);
    m_cloudDetail = make3d(kDetail, 4);
    glCreateTextures(GL_TEXTURE_2D, 1, &m_cloudWeather);
    glTextureStorage2D(m_cloudWeather, 1, GL_RG8, kWeather, kWeather);
    glTextureParameteri(m_cloudWeather, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(m_cloudWeather, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(m_cloudWeather, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTextureParameteri(m_cloudWeather, GL_TEXTURE_WRAP_T, GL_REPEAT);

    const GLuint prog = loadComputeProgram(m_shaderDir / "cloudnoise.comp");
    if (prog == 0) {
        VX_ERROR("clouds: no noise bake, the sky will be empty");
        return;
    }
    glUseProgram(prog);
    glProgramUniform1i(prog, 2, 20260721);
    const auto run3d = [&](GLuint tex, int mode, int size) {
        glBindImageTexture(0, tex, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R8);
        glProgramUniform1i(prog, 0, mode);
        glProgramUniform1i(prog, 1, size);
        glDispatchCompute(GLuint(size + 3) / 4, GLuint(size + 3) / 4, GLuint(size + 3) / 4);
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        glGenerateTextureMipmap(tex);
    };
    run3d(m_cloudShape, 0, kShape);
    run3d(m_cloudDetail, 1, kDetail);
    glBindImageTexture(1, m_cloudWeather, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG8);
    glProgramUniform1i(prog, 0, 2);
    glProgramUniform1i(prog, 1, kWeather);
    glDispatchCompute(kWeather / 4, kWeather / 4, 1);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    glDeleteProgram(prog);
}

void PostFx::shutdown() {
    destroyTargets();
    const GLuint progs[] = {m_progSsao,  m_progBlur,    m_progComposite, m_progBloomDown,
                            m_progBloomUp, m_progFinal, m_progTaa,       m_progMotion,
                            m_progApplyAo, m_progCloudMap, m_progScatter, m_progClouds3d};
    for (const GLuint p : progs)
        glDeleteProgram(p);
    const GLuint noise[] = {m_cloudShape, m_cloudDetail, m_cloudWeather, m_causticsTex};
    glDeleteTextures(4, noise);
    glDeleteProgram(m_progCaustics);
    glDeleteFramebuffers(1, &m_cloudFbo);
    glDeleteTextures(1, &m_cloudTex);
    glDeleteVertexArrays(1, &m_vao);
}

void PostFx::updateCloudMap(const glm::dvec3& camPos, float time, float cloudBias) {
    constexpr float kTexel = kCloudMapSpan / float(kCloudMapRes);
    m_cloudOrigin = {std::floor(float(camPos.x) / kTexel) * kTexel - kCloudMapSpan * 0.5f,
                     std::floor(float(camPos.z) / kTexel) * kTexel - kCloudMapSpan * 0.5f};
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glBindFramebuffer(GL_FRAMEBUFFER, m_cloudFbo);
    glViewport(0, 0, kCloudMapRes, kCloudMapRes);
    glUseProgram(m_progCloudMap);
    glBindTextureUnit(11, m_cloudWeather);
    glProgramUniform2fv(m_progCloudMap, 0, 1, &m_cloudOrigin.x);
    glProgramUniform1f(m_progCloudMap, 1, kCloudMapSpan);
    glProgramUniform1f(m_progCloudMap, 2, time);
    glProgramUniform1f(m_progCloudMap, 33, cloudBias);
    glBindVertexArray(m_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glGenerateTextureMipmap(m_cloudTex); // the shader blurs by picking a coarser level
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
}

void PostFx::destroyTargets() {
    if (m_width == 0)
        return;
    const GLuint fbos[] = {m_hdrFbo, m_hdr2Fbo, m_ssaoFbo, m_ssaoBlurFbo, m_taaFbo[0],
                           m_taaFbo[1]};
    glDeleteFramebuffers(6, fbos);
    glDeleteFramebuffers(kBloomMips, m_bloomFbo);
    const GLuint texes[] = {m_hdrColor, m_depth,  m_hdr2,    m_ssaoTex,   m_ssaoBlurTex, m_sceneCopy,
                            m_taa[0],   m_taa[1], m_motionTex, m_terrainDepth};
    glDeleteTextures(GLsizei(std::size(texes)), texes);
    glDeleteFramebuffers(1, &m_scatterFbo);
    glDeleteTextures(1, &m_scatterTex);
    glDeleteFramebuffers(2, m_cloudsFbo);
    glDeleteTextures(2, m_clouds);
    glDeleteFramebuffers(1, &m_motionFbo);
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
    m_terrainDepth = makeTex2D(GL_DEPTH_COMPONENT32F, width, height, GL_NEAREST);

    m_hdr2 =makeTex2D(GL_RGBA16F, width, height);
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

    // half-res light in the air; read back with texelFetch, so NEAREST
    m_scatterTex = makeTex2D(GL_RGBA16F, hw, hh, GL_NEAREST);
    m_scatterFbo = makeFbo(m_scatterTex);
    for (int i = 0; i < 2; ++i) {
        m_clouds[i] = makeTex2D(GL_RGBA16F, hw, hh);
        m_cloudsFbo[i] = makeFbo(m_clouds[i]);
    }
    m_cloudsValid = false;

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

void PostFx::captureTerrainDepth() {
    glCopyImageSubData(m_depth, GL_TEXTURE_2D, 0, 0, 0, 0, m_terrainDepth, GL_TEXTURE_2D, 0, 0, 0, 0,
                       m_width, m_height, 1);
}

void PostFx::applyOpaqueSsao(bool enabled, const glm::vec2& projScale, float nearPlane) {
    if (!enabled)
        return;
    // Unlike run() (called at the very end of the frame, nothing after it needs depth
    // test/culling), this runs BETWEEN the opaque pass and drawWater(), water's own draw
    // never re-enables depth test itself (it only toggles the depth MASK, expecting the test
    // to already be on from the opaque pass), so it must be restored before returning here.
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glBindVertexArray(m_vao);

    // ssao at half res, then a box blur over the rotation noise. m_depth at this point holds
    // only the opaque scene (this runs before drawWater), which is exactly the geometry SSAO
    // should be reading.
    glBindFramebuffer(GL_FRAMEBUFFER, m_ssaoFbo);
    glViewport(0, 0, std::max(m_width / 2, 1), std::max(m_height / 2, 1));
    glUseProgram(m_progSsao);
    glProgramUniform2fv(m_progSsao, 0, 1, &projScale.x);
    glProgramUniform1f(m_progSsao, 1, nearPlane);
    glBindTextureUnit(0, m_depth);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glBindFramebuffer(GL_FRAMEBUFFER, m_ssaoBlurFbo);
    glUseProgram(m_progBlur);
    glBindTextureUnit(0, m_ssaoTex);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // Multiply the blurred AO onto the opaque scene now, before water gets drawn on top of
    // it. Blended (GL_DST_COLOR, GL_ZERO) rather than sampled-and-multiplied in the shader,
    // so apply_ao.frag never has to read m_hdrColor while it is also this draw's target,
    // it only needs m_sceneCopy (just captured above, by the caller) for the emitter check.
    glBindFramebuffer(GL_FRAMEBUFFER, m_hdrFbo);
    glViewport(0, 0, m_width, m_height);
    glEnable(GL_BLEND);
    glBlendFunc(GL_DST_COLOR, GL_ZERO);
    glUseProgram(m_progApplyAo);
    glBindTextureUnit(0, m_sceneCopy);
    glBindTextureUnit(1, m_ssaoBlurTex);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisable(GL_BLEND);

    // Restore state for whatever draws next (drawWater(), which assumes both are already on
    // the way the opaque pass left them).
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
}

void PostFx::run(const RunParams& p) {
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glBindVertexArray(m_vao);

    // SSAO itself now runs in applyOpaqueSsao(), BEFORE water is drawn, see that method and
    // apply_ao.frag for why. By this point in the frame water is already in m_hdrColor.

    // A shaft needs a sky to come from. In a cave whatever the cascades let through is not one:
    // with no sky over the camera there are none, coming back on the way to the surface.
    const float skyGate = std::clamp((p.toggles.eyeSky - 0.08f) / 0.32f, 0.0f, 1.0f);
    // Light in the air, sun shafts and lamp glow, marched at half res; the composite reads it
    // back up depth-aware. It was marched per full-res pixel inside the composite.
    const bool sunShafts = p.toggles.volumetric && p.toggles.shadows;
    const bool scatter = sunShafts || p.lightVolume != 0;
    const glm::vec3 splits = p.shadows->splits();
    if (scatter) {
        glBindFramebuffer(GL_FRAMEBUFFER, m_scatterFbo);
        glViewport(0, 0, std::max(m_width / 2, 1), std::max(m_height / 2, 1));
        glUseProgram(m_progScatter);
        glProgramUniform2fv(m_progScatter, 0, 1, &p.projScale.x);
        glProgramUniform1f(m_progScatter, 1, p.nearPlane);
        glProgramUniformMatrix3fv(m_progScatter, 2, 1, GL_FALSE, &p.invViewRot[0][0]);
        glProgramUniformMatrix4fv(m_progScatter, 6, 3, GL_FALSE, &p.shadows->cascadeVp()[0][0][0]);
        glProgramUniform3fv(m_progScatter, 18, 1, &splits.x);
        glProgramUniform1f(m_progScatter, 24, sunShafts && skyGate > 0.0f ? 1.0f : 0.0f);
        glProgramUniform1f(m_progScatter, 40, skyGate);
        glProgramUniform1f(m_progScatter, 41, p.underwater ? 1.0f : 0.0f);
        glProgramUniform3fv(m_progScatter, 42, 1, &p.camPos.x);
        glProgramUniform1f(m_progScatter, 43, std::fmod(p.time, 4096.0f));
        glProgramUniform3fv(m_progScatter, 30, 1, &p.lightL.x);
        glProgramUniform3fv(m_progScatter, 31, 1, &p.lightColor.x);
        glProgramUniform3fv(m_progScatter, 37, 1, &p.lightVolumeCam.x);
        glProgramUniform3fv(m_progScatter, 38, 1, &p.lightVolumeMin.x);
        glProgramUniform1f(m_progScatter, 39, p.lightVolume != 0 ? 1.0f : 0.0f);
        glBindTextureUnit(1, p.shadows->texture());
        glBindTextureUnit(3, m_depth);
        glBindTextureUnit(7, p.lightVolume);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }

    // The clouds, half res, a quarter of the pixels a frame, the rest carried over.
    if (p.toggles.clouds) {
        const int next = 1 - m_cloudsIndex;
        glBindFramebuffer(GL_FRAMEBUFFER, m_cloudsFbo[next]);
        glViewport(0, 0, std::max(m_width / 2, 1), std::max(m_height / 2, 1));
        glUseProgram(m_progClouds3d);
        const glm::vec3 camDelta = p.camPos - m_cloudsPrevCam;
        // a teleport has nothing to carry over
        const bool history = m_cloudsValid && glm::dot(camDelta, camDelta) < 64.0f * 64.0f;
        glProgramUniform2fv(m_progClouds3d, 0, 1, &p.projScale.x);
        glProgramUniform1f(m_progClouds3d, 1, p.nearPlane);
        glProgramUniformMatrix3fv(m_progClouds3d, 2, 1, GL_FALSE, &p.invViewRot[0][0]);
        glProgramUniformMatrix3fv(m_progClouds3d, 5, 1, GL_FALSE, &m_cloudsPrevRot[0][0]);
        glProgramUniform3fv(m_progClouds3d, 8, 1, &camDelta.x);
        glProgramUniform3fv(m_progClouds3d, 9, 1, &p.camPos.x);
        glProgramUniform1f(m_progClouds3d, 10, p.time);
        glProgramUniform3fv(m_progClouds3d, 11, 1, &p.lightL.x);
        glProgramUniform3fv(m_progClouds3d, 12, 1, &p.lightColor.x);
        glProgramUniform3fv(m_progClouds3d, 13, 1, &p.skyAmbient.x);
        glProgramUniform1f(m_progClouds3d, 14, p.daylight);
        glProgramUniform1f(m_progClouds3d, 15, p.toggles.flash);
        glProgramUniform1i(m_progClouds3d, 16, m_cloudsFrame++);
        glProgramUniform1f(m_progClouds3d, 17, history ? 1.0f : 0.0f);
        glProgramUniform3fv(m_progClouds3d, 18, 1, &p.sunL.x);
        glProgramUniform1f(m_progClouds3d, 19, p.toggles.precip * (1.0f - 0.4f * p.toggles.snow));
        glProgramUniform1f(m_progClouds3d, 33, p.toggles.cloudCover());
        glBindTextureUnit(3, m_depth);
        glBindTextureUnit(4, m_clouds[m_cloudsIndex]);
        glBindTextureUnit(11, m_cloudWeather);
        glBindTextureUnit(12, m_cloudShape);
        glBindTextureUnit(13, m_cloudDetail);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        m_cloudsIndex = next;
        m_cloudsValid = true;
        m_cloudsPrevRot = p.invViewRot;
        m_cloudsPrevCam = p.camPos;
    } else {
        m_cloudsValid = false;
    }

    // composite: sky, the air light, fog, still HDR
    glBindFramebuffer(GL_FRAMEBUFFER, m_hdr2Fbo);
    glViewport(0, 0, m_width, m_height);
    glUseProgram(m_progComposite);
    glProgramUniform2fv(m_progComposite, 0, 1, &p.projScale.x);
    glProgramUniform1f(m_progComposite, 1, p.nearPlane);
    glProgramUniformMatrix3fv(m_progComposite, 2, 1, GL_FALSE, &p.invViewRot[0][0]);
    // location 23 (uAoOn) retired, AO now applied in applyOpaqueSsao(), before water draws
    glProgramUniform1f(m_progComposite, 24, sunShafts ? 1.0f : 0.0f);
    glProgramUniform3fv(m_progComposite, 5, 1, &p.sunL.x);
    glProgramUniform3fv(m_progComposite, 25, 1, &p.sunColor.x);
    glProgramUniform2fv(m_progComposite, 26, 1, &p.fog.x);
    glProgramUniform1f(m_progComposite, 27, p.toggles.lensFlare);
    glProgramUniform3fv(m_progComposite, 28, 1, &p.camPos.x);
    // location 29 (uContact) retired, contact shadows removed, see composite.frag
    glProgramUniform3fv(m_progComposite, 30, 1, &p.lightL.x);
    glProgramUniform3fv(m_progComposite, 31, 1, &p.lightColor.x);
    glProgramUniform1f(m_progComposite, 32, p.daylight);
    glProgramUniform1f(m_progComposite, 33, p.toggles.cloudCover());
    glProgramUniform1f(m_progComposite, 34, p.toggles.flash);
    glProgramUniform3fv(m_progComposite, 35, 1, p.toggles.biomeTint);
    glProgramUniform1f(m_progComposite, 36, p.toggles.biomeHaze);
    glProgramUniform1f(m_progComposite, 43, p.toggles.aurora);
    glProgramUniformMatrix4fv(m_progComposite, 6, 3, GL_FALSE,
                              &p.shadows->cascadeVp()[0][0][0]);
    glProgramUniform3fv(m_progComposite, 18, 1, &splits.x);
    glProgramUniform1f(m_progComposite, 19, p.underwater ? 1.0f : 0.0f);
    glProgramUniform1f(m_progComposite, 20, p.toggles.clouds ? 1.0f : 0.0f);
    glProgramUniform1f(m_progComposite, 21, p.time);
    glProgramUniform1f(m_progComposite, 40, scatter ? 1.0f : 0.0f);
    glProgramUniform1f(m_progComposite, 41, skyGate);
    glProgramUniform1f(m_progComposite, 42, p.toggles.precip * (1.0f - 0.4f * p.toggles.snow));
    glBindTextureUnit(6, m_clouds[m_cloudsIndex]);
    glBindTextureUnit(5, m_scatterTex);
    glBindTextureUnit(0, m_hdrColor);
    glBindTextureUnit(1, p.shadows->texture());
    glBindTextureUnit(3, m_depth);
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
        glProgramUniform1f(m_progTaa, 12, p.camPos.y);
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

    // tonemap + sRGB into the caller's framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, p.targetFbo);
    glViewport(0, 0, p.targetWidth, p.targetHeight);
    glUseProgram(m_progFinal);
    glProgramUniform1f(m_progFinal, 0, p.toggles.bloom ? p.toggles.bloomStrength : 0.0f);
    glProgramUniform1f(m_progFinal, 1, p.toggles.exposure);
    // Two things soften the image and both want the same pass. TAA's history blend is one.
    // The other is render scale: below 1.0 the whole HDR chain runs smaller and only this
    // tonemap runs at native, so the upscale is a plain bilinear stretch, which is exactly
    // why 0.70x was never a sensible default despite being ~21% cheaper (Tier 1 #7).
    // Scaling the sharpen with the amount of upscaling is what makes the lower render scales
    // usable, and the quality presets below High depend on that.
    const float taaSharp = p.toggles.taa ? 0.22f : 0.0f;
    const float upscaleSharp =
        std::clamp((1.0f - p.toggles.renderScale) * 0.9f, 0.0f, 0.35f);
    glProgramUniform1f(m_progFinal, 3, std::min(taaSharp + upscaleSharp, 0.45f));
    const glm::vec2 texel(1.0f / float(m_width), 1.0f / float(m_height));
    glProgramUniform2fv(m_progFinal, 4, 1, &texel.x);
    glProgramUniform1f(m_progFinal, 5, p.toggles.vignette);
    glProgramUniform1f(m_progFinal, 6, p.toggles.grain);
    glProgramUniform1f(m_progFinal, 7, p.time);
    glProgramUniform1f(m_progFinal, 8, p.toggles.saturation);
    glProgramUniform1f(m_progFinal, 9, p.toggles.contrast);
    glProgramUniform1f(m_progFinal, 10, p.toggles.temperature);
    // Faded in as daylight goes, and down a cave at any hour: an eye in the dark sees it the same way.
    // An open day frame is untouched.
    const float enclosed = 1.0f - glm::smoothstep(0.1f, 0.6f, p.toggles.eyeSky);
    glProgramUniform1f(m_progFinal, 12, p.toggles.scotopic * std::max(1.0f - p.daylight, enclosed));
    glProgramUniform1f(m_progFinal, 14, p.toggles.dither);
    glBindTextureUnit(0, resolved);
    glBindTextureUnit(2, m_bloomTex[0]);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
}

} // namespace vx
