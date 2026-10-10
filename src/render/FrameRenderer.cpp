#include "render/FrameRenderer.hpp"

#include "render/Overlay.hpp"
#include "render/Particles.hpp"
#include "render/Renderer.hpp"
#include "render/Weather.hpp"
#include "render/FarTerrain.hpp"
#include "render/LightVolume.hpp"
#include "render/Wildlife.hpp"
#include "render/ShadowMap.hpp"
#include "render/WaterWaves.hpp"
#include "world/Chunk.hpp"

#include <glad/gl.h>
#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>

namespace vx {

CameraFrame buildCameraFrame(const FlyCamera& camera, int fbWidth, int fbHeight,
                             glm::vec2 jitterNdc) {
    CameraFrame f;
    glm::mat4 proj = camera.projReversedZ(float(fbWidth) / float(fbHeight));
    proj[2][0] += jitterNdc.x; // sub-pixel jitter for TAA (clip-space shift per unit w)
    proj[2][1] += jitterNdc.y;
    f.viewProj = proj * camera.viewAtOrigin();
    f.frustum = Frustum::fromViewProj(f.viewProj);
    f.cameraChunk = glm::ivec3(glm::floor(camera.position / double(kChunkSize)));
    f.cameraFrac = glm::vec3(camera.position - glm::dvec3(f.cameraChunk) * double(kChunkSize));
    return f;
}

SkyLight computeSky(float timeOfDay) {
    const float theta = (timeOfDay - 0.25f) * 6.2831853f;
    const glm::vec3 dir = glm::normalize(glm::vec3(std::cos(theta), std::sin(theta), 0.30f));
    const float alt = dir.y; // -1..1
    const float day = glm::clamp(alt * 3.0f + 0.12f, 0.0f, 1.0f); // 0 below the horizon
    const float high = glm::clamp(alt * 1.8f, 0.0f, 1.0f);
    const glm::vec3 sunColor = glm::mix(glm::vec3(2.7f, 1.25f, 0.5f),   // horizon warm
                                        glm::vec3(2.05f, 1.85f, 1.5f),  // overhead white
                                        high) *
                               day;

    // Moonlight. Night used to have NO directional light at all, `day` is 0 below the
    // horizon, so the direct term was exactly zero and the sky's beautifully cratered moon
    // lit nothing. Ambient was therefore the entire night image, which is why it kept being
    // raised: with nothing casting, a hollow has no shape to lose and every fix had to come
    // out of the fill. That is the actual cause of "night is too bright", and no amount of
    // turning the fill back down addresses it.
    //
    // Ramp starts where `day` has already hit 0 (alt = -0.04), so at the instant the light
    // direction flips from sun to moon BOTH terms are zero and the swap is invisible. That
    // is the whole reason for the offset; without it the handover is a visible step.
    const float moon = glm::clamp((-alt - 0.04f) * 3.0f, 0.0f, 1.0f);
    // Dim and cool, and deliberately not physical, real moonlight is ~1/400000 of sunlight
    // and would be black here. It only has to be strong enough that a hill has a lit side.
    const glm::vec3 moonColor = glm::vec3(0.115f, 0.145f, 0.235f) * moon;

    const bool moonLit = day <= 0.0f;
    // Night ambient can come back down now that the moon carries the shape. These are the
    // pre-lift values from before the fill was doing all the work on its own.
    const glm::vec3 ambient =
        glm::mix(glm::vec3(0.030f, 0.040f, 0.072f), glm::vec3(0.31f, 0.39f, 0.52f), day);

    SkyLight s;
    s.sunDir = dir;
    s.lightDir = moonLit ? -dir : dir;
    s.color = moonLit ? moonColor : sunColor;
    s.sunColor = sunColor;
    s.ambient = ambient;
    s.daylight = day;
    return s;
}

float halton(int i, int base) {
    float f = 1.0f, r = 0.0f;
    while (i > 0) {
        f /= float(base);
        r += f * float(i % base);
        i /= base;
    }
    return r;
}

void renderFrame(RenderContext& rc, const FlyCamera& camera, const FxToggles& fx,
                 bool caveCull, int fbW, int fbH, unsigned int targetFbo, float time,
                 FrameTimers* timers, bool hasSelection, const glm::vec3& selMinCamRel,
                 bool underwater, TaaState* taa, float fogEnd, int crackLayer,
                 unsigned int blockTex, float farEnd) {
    const float aspect = float(fbW) / float(fbH);
    // internal render size; the final tonemap still runs at fbW x fbH and upscales. the
    // aspect is unchanged because both axes scale together.
    const int rw = std::max(1, int(float(fbW) * fx.renderScale + 0.5f));
    const int rh = std::max(1, int(float(fbH) * fx.renderScale + 0.5f));
    const bool useTaa = fx.taa && taa != nullptr;
    const bool useMotion = fx.motionBlur && taa != nullptr;
    const bool useReproj = useTaa || useMotion; // both need last frame's view-proj + cam pos
    glm::vec2 jitter(0.0f);
    if (useTaa) {
        const int idx = taa->frame % 8 + 1; // halton is 1-indexed
        // jitter is a sub-pixel shift of the RENDER target, not the window
        jitter = {(halton(idx, 2) - 0.5f) * 2.0f / float(rw),
                  (halton(idx, 3) - 0.5f) * 2.0f / float(rh)};
    }
    const CameraFrame cam = buildCameraFrame(camera, fbW, fbH, jitter);
    SkyLight sky = computeSky(fx.timeOfDay);
    // A flash of lightning lights the ground from the whole sky, not from the sun: it lifts the
    // ambient (which the chunks, the far terrain and the water all read), not the direct light.
    sky.ambient += glm::vec3(0.60f, 0.68f, 0.95f) * (0.4f * fx.flash);
    // and the light itself takes on the colour of the air it came through: warm and dusty over
    // sand, a touch cool over the sea
    const glm::vec3 biome(fx.biomeTint[0], fx.biomeTint[1], fx.biomeTint[2]);
    sky.color *= glm::mix(glm::vec3(1.0f), biome, 0.7f);
    sky.sunColor *= glm::mix(glm::vec3(1.0f), biome, 0.7f);
    // Everything that SHADES takes the celestial direction, so the cascades, NdotL and every
    // specular lobe follow the moon after dusk. The sky pass is handed the true sun
    // separately, see SkyLight.
    const glm::vec3 sunL = sky.lightDir;

    rc.postfx.resize(rw, rh);

    if (timers) timers->shadow.begin();
    // shadows follow the render distance (fog end), not a fixed ring around the player
    rc.shadows.setViewDistance(fogEnd < 1e8f ? fogEnd : ShadowMap::kFarMin);
    rc.shadows.setLeafGap(fx.canopyLight);
    if (fx.shadows) {
        // The shadow pass alpha-tests cutout blocks now, so it needs the block texture bound
        // BEFORE it runs, it used to be bound further down for the scene pass only, and a
        // depth-only pass had no use for it. Without this the leaf test samples whatever
        // happened to be on unit 0.
        glBindTextureUnit(0, rc.blockTex);
        rc.shadows.update(camera.position, sunL);
        rc.shadows.render(rc.renderer, cam.cameraChunk, cam.cameraFrac);
    }
    if (timers) timers->shadow.end();

    if (timers) timers->scene.begin();
    const bool cloudShadows = fx.clouds && fx.cloudShadow > 0.0f;
    if (cloudShadows)
        rc.postfx.updateCloudMap(camera.position, time, fx.cloudBias);
    rc.postfx.beginScene();
    glClearColor(0.42f, 0.68f, 1.05f, 1.0f); // HDR sky
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glProgramUniform3fv(rc.drawProgram, 4, 1, &sunL.x);
    glProgramUniform3fv(rc.drawProgram, 21, 1, &sky.color.x);
    glProgramUniform3fv(rc.drawProgram, 22, 1, &sky.ambient.x);
    glProgramUniform1f(rc.drawProgram, 23, fx.emissive);
    // Camera world position as float. World coords are kept off the GPU for GEOMETRY, where
    // sub-block precision matters; the cloud field is low frequency and a float is exact
    // enough for it well past any distance the terrain generator covers.
    const glm::vec3 camPosF(camera.position);
    glProgramUniform3fv(rc.drawProgram, 24, 1, &camPosF.x);
    glProgramUniform1f(rc.drawProgram, 25, cloudShadows ? fx.cloudShadow : 0.0f);
    const glm::vec3 cloudXf = rc.postfx.cloudMapXf();
    glProgramUniform3fv(rc.drawProgram, 31, 1, &cloudXf.x);
    glBindTextureUnit(6, rc.postfx.cloudMapTex());
    glProgramUniform1f(rc.drawProgram, 26, time);
    glProgramUniform1f(rc.drawProgram, 27, fx.leafSss);
    glProgramUniform1f(rc.drawProgram, 28, fx.relief);
    // Was derived in-shader from uSunL.y, which stops meaning "how light is it" the moment
    // uSunL can point at the moon.
    glProgramUniform1f(rc.drawProgram, 29, sky.daylight);
    // Wetness follows the rain, and snow does not wet anything, it settles.
    glProgramUniform1f(rc.drawProgram, 30, fx.precip * (1.0f - fx.snow));
    glProgramUniform1f(rc.drawProgram, 34, fx.snowCover);
    glProgramUniform1f(rc.drawProgram, 5, fx.shadows ? 1.0f : 0.0f);
    glProgramUniformMatrix4fv(rc.drawProgram, 6, 3, GL_FALSE,
                              &rc.shadows.cascadeVp()[0][0][0]);
    const glm::vec3 splits = rc.shadows.splits();
    glProgramUniform3fv(rc.drawProgram, 18, 1, &splits.x);
    const glm::vec3 texel = rc.shadows.cascadeTexelWorld();
    glProgramUniform3fv(rc.drawProgram, 19, 1, &texel.x);
    glProgramUniform1f(rc.drawProgram, 20, time); // leaf sway
    glBindTextureUnit(0, rc.blockTex);
    glBindTextureUnit(1, rc.shadows.texture());
    glBindTextureUnit(5, rc.shadows.texture()); // same array, read raw for the PCSS blocker search
    glBindSampler(5, rc.shadows.rawSampler());

    rc.renderer.drawWorld(cam.viewProj, cam.frustum, cam.cameraChunk, cam.cameraFrac,
                          caveCull);
    glBindSampler(5, 0);

    // The horizon: the coarse world past the streamed chunks. Opaque, drawn after the chunks so
    // depth keeps the real terrain in front of it.
    if (rc.farTerrain && fx.farTerrain && farEnd > fogEnd && fogEnd < 1e8f)
        rc.farTerrain->draw(cam.viewProj, camera.position, sunL, sky.color, sky.ambient,
                            fx.farInner >= 0.0f ? fx.farInner : fogEnd);

    // Creatures go in with the opaque scene, not after the water: a fish under a lake has to be
    // seen THROUGH it, refracted and absorbed like the seabed, and the snapshot the water pass
    // takes next is what makes that happen. The depth of the world without them is kept first,
    // for the water's foam (see PostFx::captureTerrainDepth).
    rc.postfx.captureTerrainDepth();
    if (rc.wildlife && rc.wildlife->liveCount() > 0) {
        const glm::vec3 fwd = camera.forward();
        const glm::vec3 right = glm::normalize(glm::cross(fwd, glm::vec3(0.0f, 1.0f, 0.0f)));
        const glm::vec3 upv = glm::cross(right, fwd);
        rc.wildlife->draw(cam.viewProj, camera.position, right, upv, underwater);
    }

    // transparent water over the opaque scene: alpha blend, depth test but no write.
    // snapshot the opaque colour first so the water shader can reflect it (SSR).
    rc.postfx.captureOpaqueForWater();
    const float wf = 1.0f / std::tan(glm::radians(camera.fovYDeg) * 0.5f);
    const glm::vec2 projScale{aspect / wf, 1.0f / wf};
    // AO applied HERE, to the opaque scene only, before water exists in the colour buffer,
    // see PostFx::applyOpaqueSsao and apply_ao.frag. Water never writes depth (see the
    // drawWater call below), so computing/applying AO any later would read the LAKE BED's
    // geometry under any water on screen and darken the water surface by however occluded
    // the bed is, the owner-reported dark contour tracery on wide lake views.
    rc.postfx.applyOpaqueSsao(fx.ssao, projScale, camera.nearPlane);
    glProgramUniform1f(rc.waterProgram, 4, time);
    glProgramUniform3fv(rc.waterProgram, 5, 1, &sunL.x);
    glProgramUniform3fv(rc.waterProgram, 8, 1, &sky.color.x);
    glProgramUniform2fv(rc.waterProgram, 6, 1, &projScale.x);
    glProgramUniform1f(rc.waterProgram, 7, camera.nearPlane);
    glProgramUniform3fv(rc.waterProgram, 9, 1, &sky.ambient.x);
    glProgramUniform1f(rc.waterProgram, 10, fx.shadows ? 1.0f : 0.0f);
    // Water samples the same three cascades the terrain does; it had no shadow at all
    // before, which is why its sun lighting came from blocky voxel sky light.
    glProgramUniformMatrix4fv(rc.waterProgram, 11, 3, GL_FALSE,
                              &rc.shadows.cascadeVp()[0][0][0]);
    glProgramUniform3fv(rc.waterProgram, 23, 1, &splits.x);
    glProgramUniform3fv(rc.waterProgram, 24, 1, &texel.x);
    glProgramUniform1f(rc.waterProgram, 25, sky.daylight);
    // The swell: its phase is worked out at the camera in double so it stays fixed to the world.
    // Half-block tessellation on High and Ultra, a block on Low and Medium.
    WaterWaves::apply(rc.waterProgram,
                      WaterWaves::frame(double(time), camera.position, fx.waveStorm, fx.waves),
                      fx.quality >= 2 ? 2.0f : 1.0f);
    glBindTextureUnit(2, rc.postfx.sceneCopyTex());
    glBindTextureUnit(3, rc.postfx.depthTex());
    glBindTextureUnit(4, rc.postfx.terrainDepthTex());
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    rc.renderer.drawWater(rc.waterProgram, cam.viewProj, cam.frustum, cam.cameraChunk,
                          cam.cameraFrac);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);

    // Particles after the water, still inside the HDR pass so they are tonemapped and
    // bloomed with everything else rather than pasted on afterwards.
    if (rc.particles && rc.particles->liveCount() > 0) {
        const glm::vec3 fwd = camera.forward();
        const glm::vec3 right = glm::normalize(glm::cross(fwd, glm::vec3(0.0f, 1.0f, 0.0f)));
        const glm::vec3 up = glm::cross(right, fwd);
        rc.particles->draw(cam.viewProj, camera.position, right, up, rc.blockTex);
    }

    // Precipitation last in the scene pass: alpha blended over the lit world and the water,
    // still inside HDR so it tonemaps and fogs with everything else rather than being pasted
    // on afterwards.
    if (rc.weather && fx.precip > 0.001f) {
        Weather::Params wp;
        wp.viewProj = cam.viewProj;
        const glm::vec3 fwd = camera.forward();
        wp.right = glm::normalize(glm::cross(fwd, glm::vec3(0.0f, 1.0f, 0.0f)));
        wp.up = glm::cross(wp.right, fwd);
        // Fractional part of the camera within its own cell. Anchoring the field to the cell
        // rather than to the camera is what stops the whole thing sliding along with you.
        wp.camFrac = glm::vec3(camera.position - glm::floor(camera.position));
        wp.time = time;
        wp.strength = fx.precip;
        wp.snow = fx.snow;
        // Lit by the sky rather than a constant, so precipitation goes dark after dusk
        // instead of glowing, the same mistake the water body tint used to make.
        wp.tint = sky.ambient * 2.2f + sky.color * 0.10f;
        rc.weather->draw(wp);
    }

    // block-selection wireframe, in the HDR scene so it depth-tests against the world
    if (hasSelection) {
        rc.overlay.drawSelection(cam.viewProj, selMinCamRel);
        if (crackLayer >= 0)
            rc.overlay.drawBreak(cam.viewProj, selMinCamRel, crackLayer, blockTex);
    }
    if (timers) timers->scene.end();

    if (timers) timers->post.begin();
    const float f = 1.0f / std::tan(glm::radians(camera.fovYDeg) * 0.5f);
    PostFx::RunParams rp;
    rp.projScale = {aspect / f, 1.0f / f};
    rp.nearPlane = camera.nearPlane;
    rp.invViewRot = glm::transpose(glm::mat3(camera.viewAtOrigin()));
    // The post stack needs BOTH: the true sun to draw the sky and place the flare, the
    // celestial light to march volumetrics and contact shadows against the cascades.
    rp.sunL = sky.sunDir;
    rp.lightL = sunL;
    rp.lightColor = sky.color;
    rp.sunColor = sky.sunColor;
    rp.daylight = sky.daylight;
    // mist pulls the fog in; the shadow reach above uses fogEnd untouched so the cascades do
    // not redraw every time the mist moves
    const float mistScale = 1.0f - 0.5f * fx.mist;
    // With the far horizon on, the haze no longer hides the edge of the world, it reaches out
    // across the heightfield instead, closing in where that ends. It still starts where it
    // always did, so the near view looks the same.
    const float hazeEnd = (farEnd > fogEnd && fogEnd < 1e8f) ? farEnd : fogEnd;
    rp.fog = {fogEnd * 0.55f * mistScale, hazeEnd * mistScale};
    rp.shadows = &rc.shadows;
    rp.toggles = fx;
    rp.time = time;
    rp.camPos = glm::vec3(camera.position);
    // Eye adaptation runs off wall-clock delta so its speed does not depend on framerate.
    // Derived here rather than threaded through every caller, because the offscreen
    // harnesses render as fast as they can and would otherwise adapt at a different rate
    // than the game does.
    {
        static float sPrevTime = time;
        rp.dt = std::clamp(time - sPrevTime, 0.0f, 0.25f);
        sPrevTime = time;
    }
    // only let PostFx run these when we actually have the reprojection history for them
    rp.toggles.taa = useTaa;
    rp.toggles.motionBlur = useMotion;
    rp.underwater = underwater;
    if (rc.lightVolume && fx.volumetric && rc.lightVolume->anyLight()) {
        rp.lightVolume = rc.lightVolume->texture();
        rp.lightVolumeCam = rc.lightVolume->cameraTexel();
        rp.lightVolumeMin = rc.lightVolume->regionMinRel();
    }
    if (useReproj) {
        rp.prevViewProj = taa->prevViewProj;
        rp.camDelta = glm::vec3(camera.position - taa->prevCamPos);
        rp.taaHistoryValid = taa->valid;
    }
    rp.targetFbo = targetFbo;
    rp.targetWidth = fbW;
    rp.targetHeight = fbH;
    rc.postfx.run(rp);
    if (timers) timers->post.end();

    if (useReproj) { // remember this frame (unjittered) for next frame's reprojection
        taa->prevViewProj = camera.projReversedZ(aspect) * camera.viewAtOrigin();
        taa->prevCamPos = camera.position;
        ++taa->frame;
        taa->valid = true;
    }

    // gameplay HUD on top of the final tonemapped image
    glBindFramebuffer(GL_FRAMEBUFFER, targetFbo);
    glViewport(0, 0, fbW, fbH);
    rc.overlay.drawCrosshair(fbW, fbH, glm::vec4(fx.crosshair[0], fx.crosshair[1],
                                                 fx.crosshair[2], fx.crosshair[3]));
}

} // namespace vx
