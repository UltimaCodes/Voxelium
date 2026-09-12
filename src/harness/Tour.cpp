#include "harness/Harness.hpp"

#include "core/Bmp.hpp"
#include "core/Log.hpp"
#include "core/Paths.hpp"
#include "core/Settings.hpp"
#include "gfx/BlockTextures.hpp"
#include "gfx/GpuTimer.hpp"
#include "io/ChunkSerializer.hpp"
#include "io/RegionStore.hpp"
#include "io/WorldStore.hpp"
#include "jobs/JobSystem.hpp"
#include "light/LightEngine.hpp"
#include "mesh/GreedyMesher.hpp"
#include "mesh/NaiveMesher.hpp"
#include "platform/Window.hpp"
#include "render/Camera.hpp"
#include "render/FrameRenderer.hpp"
#include "render/Overlay.hpp"
#include "render/Particles.hpp"
#include "render/PostFx.hpp"
#include "render/Renderer.hpp"
#include "render/ShadowMap.hpp"
#include "ui/DebugHud.hpp"
#include "world/Chunk.hpp"
#include "world/ChunkManager.hpp"
#include "world/Player.hpp"
#include "world/Raycast.hpp"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace vx {

// Visual check while the world STREAMS (--tour). --smoke renders from a settled world at
// spawn, and --bench never calls world.update() at all, so neither can show what happens
// while chunks are loading, re-LODing and swapping under a moving camera. This flies a
// path, screenshots along the way, and reports what the world looks like at each stop.
int runTour(RenderContext& rc, vx::ChunkManager& world, vx::JobSystem& jobs,
            vx::LightEngine& light, vx::FlyCamera& camera, vx::Window& window, int radius,
            bool settle) {
    world.radius = radius;
    int w = 0, h = 0;
    window.framebufferSize(w, h);

    GLuint color = 0, depth = 0, fbo = 0;
    glCreateRenderbuffers(1, &color);
    glNamedRenderbufferStorage(color, GL_RGB8, w, h);
    glCreateRenderbuffers(1, &depth);
    glNamedRenderbufferStorage(depth, GL_DEPTH_COMPONENT32F, w, h);
    glCreateFramebuffers(1, &fbo);
    glNamedFramebufferRenderbuffer(fbo, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color);
    glNamedFramebufferRenderbuffer(fbo, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);

    vx::FxToggles fx;
    fx.taa = false;
    fx.dayCycle = false;
    fx.autoExposure = false; // stateful across frames; keeps the five stops comparable
    fx.timeOfDay = 0.32f;
    // Promoted instrumentation, not a leftover: the tour is the only way to look at the
    // renderer offscreen, and without this there is no way to inspect NIGHT at all — which
    // is where the water-glows-after-dark and shadows-too-harsh bugs both lived.
    // VX_TOUR_TIME=0.85 is a good night, 0.25 sunrise, 0.5 noon.
    if (const char* t = std::getenv("VX_TOUR_TIME"))
        fx.timeOfDay = std::clamp(float(std::atof(t)), 0.0f, 1.0f);
    // Same argument as the two above: render scale below 1.0 changes the IMAGE, not just the
    // frame time, and --bench only reports the time. Judging whether a lower scale is
    // acceptable — which is the entire question behind the quality presets — needs a picture
    // of it, and this is the only way to get one offscreen.
    if (const char* s = std::getenv("VX_TOUR_SCALE"))
        fx.renderScale = std::clamp(float(std::atof(s)), 0.5f, 1.0f);
    // Cloud shadows are the one lighting term with NO other offscreen A/B. The owner
    // reported hard diagonal bands sweeping the terrain (#58b) and the only way to separate
    // "that is the cloud shadow" from "that is the terrain" is the same frame with the term
    // off. Rendering both and diffing answers it in one run.
    if (const char* c = std::getenv("VX_TOUR_CLOUDSHADOW"))
        fx.cloudShadow = std::clamp(float(std::atof(c)), 0.0f, 1.0f);

    std::vector<std::uint8_t> pixels(std::size_t(w) * h * 3);
    // walk outward from spawn: the report is that things degrade with distance travelled.
    // height comes from the generator, not a guess — a fixed y buries the camera inside a
    // hill and every screenshot then shows the inside of terrain instead of the bug.
    const glm::dvec2 where[] = {
        {16.0, 16.0}, {320.0, 96.0}, {700.0, 320.0}, {1200.0, 700.0}, {1800.0, 1200.0},
    };
    glm::dvec3 stops[std::size(where)];
    for (int i = 0; i < int(std::size(where)); ++i)
        stops[i] = {where[i].x,
                    double(world.generator().surfaceHeight(int(where[i].x), int(where[i].y))) +
                        14.0,
                    where[i].y};
    // Promoted for the same reason as VX_TOUR_TIME: the underwater path (depth fog,
    // caustics, the shafts through the surface) has no other offscreen view, and "looks
    // right" is not something to take on trust.
    if (std::getenv("VX_TOUR_UNDERWATER")) {
        // sit under the sea surface out in open water and look up-ish toward the sun
        camera.position = {320.0, double(vx::kSeaLevelForSpawn) - 6.0, 96.0};
        camera.yawDeg = 108.0f;
        camera.pitchDeg = 22.0f;
        for (int f = 0; f < 40000; ++f) {
            window.pollEvents();
            rc.renderer.beginFrame();
            world.update(jobs, rc.renderer, light, camera.position);
            rc.renderer.endFrame();
            if (f > 200 && !world.busy() && light.idle())
                break;
        }
        std::vector<vx::ChunkDrawInfo> di, wi;
        world.collectDrawInfo(di, wi);
        rc.renderer.updateChunkMeta(di);
        rc.renderer.updateWaterMeta(wi);
        world.updateVisibilityIfNeeded(rc.renderer, camera.position, true);
        rc.renderer.beginFrame();
        renderFrame(rc, camera, fx, true, w, h, fbo, 0.6f, nullptr, false, glm::vec3(0.0f),
                    /*underwater=*/true);
        rc.renderer.endFrame();
        glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, pixels.data());
        vx::writeBmp24(settle ? "tour_underwater.bmp" : "tour_underwater_moving.bmp", w, h,
                       pixels);
        VX_INFO("tour underwater: (448,50,0), %d blocks below sea level -> tour_underwater.bmp",
                int(62.0 - 50.0));
    }
    // VX_TOUR_CUSTOM="x,y,z" - every fixed stop above uses the SAME yaw/pitch (-35/-12), so
    // an artifact that only shows at a steeper look-down angle or a grazing look-across angle
    // (owner report: water contour streaks, worse at some angles than others) can go
    // unreproduced indefinitely no matter how many of the fixed stops get checked. This
    // settles once at an exact owner-chosen position, then sweeps 8 yaws x 3 pitches (24
    // shots) from that one settle - "look around in every direction" as an actual repeatable
    // test rather than one more guessed angle.
    if (const char* cp = std::getenv("VX_TOUR_CUSTOM")) {
        double cx = 0.0, cy = 0.0, cz = 0.0;
        std::sscanf(cp, "%lf,%lf,%lf", &cx, &cy, &cz);
        camera.position = {cx, cy, cz};
        camera.yawDeg = 0.0f;
        camera.pitchDeg = -20.0f;
        for (int f = 0; f < 40000; ++f) {
            window.pollEvents();
            rc.renderer.beginFrame();
            world.update(jobs, rc.renderer, light, camera.position);
            rc.renderer.endFrame();
            if (f > 200 && !world.busy() && light.idle())
                break;
        }
        std::vector<vx::ChunkDrawInfo> di, wi;
        world.collectDrawInfo(di, wi);
        rc.renderer.updateChunkMeta(di);
        rc.renderer.updateWaterMeta(wi);
        const float pitches[] = {-55.0f, -20.0f, 10.0f};
        int shot = 0;
        for (float pitch : pitches) {
            camera.pitchDeg = pitch;
            for (int yaw = 0; yaw < 360; yaw += 45) {
                camera.yawDeg = float(yaw);
                world.updateVisibilityIfNeeded(rc.renderer, camera.position, true);
                rc.renderer.beginFrame();
                renderFrame(rc, camera, fx, true, w, h, fbo, 0.6f, nullptr);
                rc.renderer.endFrame();
                glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, pixels.data());
                char name[64];
                std::snprintf(name, sizeof(name), "tour_custom_%d.bmp", shot++);
                vx::writeBmp24(name, w, h, pixels);
            }
        }
        VX_INFO("tour custom: %d shots (8 yaw x 3 pitch) from (%.0f,%.0f,%.0f)", shot, cx, cy,
                cz);
    }
    for (int i = 0; i < int(std::size(stops)); ++i) {
        camera.position = stops[i];
        camera.yawDeg = -35.0f;
        camera.pitchDeg = -12.0f;
        // fly in rather than teleport when not settling, so streaming is mid-flight
        const int budget = settle ? 40000 : 240;
        for (int f = 0; f < budget; ++f) {
            window.pollEvents();
            rc.renderer.beginFrame();
            world.update(jobs, rc.renderer, light, camera.position);
            rc.renderer.endFrame();
            if (settle && f > 200 && !world.busy() && light.idle())
                break;
        }
        std::vector<vx::ChunkDrawInfo> di, wi;
        world.collectDrawInfo(di, wi);
        rc.renderer.updateChunkMeta(di);
        rc.renderer.updateWaterMeta(wi);
        world.updateVisibilityIfNeeded(rc.renderer, camera.position, true);
        rc.renderer.beginFrame();
        renderFrame(rc, camera, fx, true, w, h, fbo, 0.6f, nullptr);
        rc.renderer.endFrame();
        glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, pixels.data());
        char name[64];
        std::snprintf(name, sizeof(name), "tour_%d%s.bmp", i, settle ? "" : "_moving");
        vx::writeBmp24(name, w, h, pixels);
        // same view with the sun shadow off. if a frame is mostly dark in one and lit in
        // the other, the darkness is the shadow map, not the geometry or the voxel light.
        vx::FxToggles noShadow = fx;
        noShadow.shadows = false;
        rc.renderer.beginFrame();
        renderFrame(rc, camera, noShadow, true, w, h, fbo, 0.6f, nullptr);
        rc.renderer.endFrame();
        std::vector<std::uint8_t> lit(pixels.size());
        glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, lit.data());
        char litName[72];
        std::snprintf(litName, sizeof(litName), "tour_%d%s_noshadow.bmp", i,
                      settle ? "" : "_moving");
        vx::writeBmp24(litName, w, h, lit);
        // Sky-light invariant: every air voxel above the highest opaque block in a column
        // is open to the sky and must hold sky 15. This is what seeding is FOR, and it
        // does not depend on the sun, the time of day or the shadow map — so if it fails,
        // the darkness people see at night is here and not in the shadow pass.
        {
            int checked = 0, bad = 0, worst = 15;
            glm::ivec3 firstBad{0, 0, 0};
            const int cx = int(std::floor(camera.position.x));
            const int cz = int(std::floor(camera.position.z));
            for (int dz = -96; dz <= 96; dz += 3)
                for (int dx = -96; dx <= 96; dx += 3) {
                    const int wx = cx + dx, wz = cz + dz;
                    // Start at the top of the RESIDENT band, not the old world ceiling.
                    // #27 removed the ceiling, so y=383 is usually not loaded, and an absent
                    // chunk reads as STONE — which ended the downward scan on its first step
                    // and left this invariant reporting "0/0 ... -> ok" at three of the five
                    // stops. A check that silently tests nothing is worse than no check.
                    int by0 = 0, by1 = 0;
                    bool bsky = false;
                    world.columnRange(wx >> 5, wz >> 5, by0, by1, bsky);
                    const int ceiling = by1 * vx::kChunkSize + vx::kChunkSize - 1;
                    int top = ceiling;
                    while (top >= by0 * vx::kChunkSize &&
                           !vx::blocks::opaque(world.blockAtWorld({wx, top, wz})))
                        --top;
                    // sample a few air voxels above the surface, not all of them
                    for (int y = top + 1; y < std::min(top + 12, ceiling); ++y) {
                        const int sky = vx::light::sky(light.lightAt({wx, y, wz}));
                        ++checked;
                        if (sky != 15) {
                            if (bad == 0) firstBad = {wx, y, wz};
                            ++bad;
                            worst = std::min(worst, sky);
                        }
                    }
                }
            VX_INFO("tour %d sky-light: %d/%d open-air voxels BELOW full sky (worst %d)%s",
                    i, bad, checked, bad ? worst : 15,
                    bad ? "" : "  -> ok");
            if (bad)
                VX_INFO("   first at (%d,%d,%d)", firstBad.x, firstBad.y, firstBad.z);
        }
        // Both of these read stale by default and this harness only takes one sample per
        // stop, so ask for fresh numbers rather than printing whatever was last cached:
        // stats() re-scans every 20th call, and gpuVisibleChunks() is a ~3-frame-latency
        // readback that a few frames per stop never catches up with. Reported 0 drawn at
        // stop 0 and then the identical count at two different stops, which cost real time.
        world.refreshStats();
        const vx::ChunkManager::Stats st = world.stats();
        VX_INFO("tour %d @ (%.0f,%.0f,%.0f): %d chunks %d meshed %d UNLIT, %d drawn, "
                "busy=%d -> %s",
                i, stops[i].x, stops[i].y, stops[i].z, st.loaded, st.meshed, st.unlit,
                rc.renderer.debugReadDrawCount(), world.busy() ? 1 : 0, name);
    }
    // One frame from under the water. This is the only view that exercises the underwater
    // branch — caustics, the absorption tint, and the surface-as-occluder god rays — and it
    // immediately caught a NaN there (pow() with a negative base) that painted solid black
    // blocks across the seabed. Nothing above the waterline would ever have shown it.
    {
        // (448, 0) has a surface height of 34 against sea level 62, so this is 20 blocks down.
        camera.position = {448.0, 50.0, 0.0};
        camera.yawDeg = 20.0f;
        camera.pitchDeg = 12.0f;
        for (int f = 0; f < 40000; ++f) {
            window.pollEvents();
            rc.renderer.beginFrame();
            world.update(jobs, rc.renderer, light, camera.position);
            rc.renderer.endFrame();
            if (f > 200 && !world.busy() && light.idle())
                break;
        }
        std::vector<vx::ChunkDrawInfo> di2, wi2;
        world.collectDrawInfo(di2, wi2);
        rc.renderer.updateChunkMeta(di2);
        rc.renderer.updateWaterMeta(wi2);
        world.updateVisibilityIfNeeded(rc.renderer, camera.position, true);
        rc.renderer.beginFrame();
        renderFrame(rc, camera, fx, true, w, h, fbo, 0.6f, nullptr, false, glm::vec3(0.0f),
                    true /*underwater*/);
        rc.renderer.endFrame();
        glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, pixels.data());
        vx::writeBmp24(settle ? "tour_underwater.bmp" : "tour_underwater_moving.bmp", w, h,
                       pixels);
        VX_INFO("tour underwater: (448,50,0), %d blocks below sea level -> tour_underwater.bmp",
                int(62.0 - 50.0));
    }

    // One rain frame from stop 2. Same argument as the night and underwater views: #29
    // changes both the precipitation field and the SHADING of every surface the sky can
    // reach, and neither is visible in any of the five clear-weather stops.
    {
        vx::FxToggles wet = fx;
        wet.precip = 0.85f;
        wet.snow = 0.0f;
        if (const char* sn = std::getenv("VX_TOUR_SNOW"))
            wet.snow = std::clamp(float(std::atof(sn)), 0.0f, 1.0f);
        camera.position = stops[2];
        camera.yawDeg = -35.0f;
        camera.pitchDeg = -12.0f;
        rc.renderer.beginFrame();
        renderFrame(rc, camera, wet, true, w, h, fbo, 0.6f, nullptr);
        rc.renderer.endFrame();
        glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, pixels.data());
        vx::writeBmp24(settle ? "tour_rain.bmp" : "tour_rain_moving.bmp", w, h, pixels);
        VX_INFO("tour rain: stop 2 at precip %.2f snow %.2f -> tour_rain.bmp",
                double(wet.precip), double(wet.snow));
    }

    // One night frame from stop 0. Night has hidden two separate bugs now — water that
    // stayed at full daylight brightness after dark, and contact shadows darkening the
    // ambient that is the only light left once the sun is down — and neither was visible in
    // any of the five daylight stops. A shot that costs one frame is worth having.
    {
        vx::FxToggles night = fx;
        night.timeOfDay = 0.87f; // sun well below the horizon
        camera.position = stops[0];
        camera.yawDeg = -35.0f;
        camera.pitchDeg = -12.0f;
        rc.renderer.beginFrame();
        renderFrame(rc, camera, night, true, w, h, fbo, 0.6f, nullptr);
        rc.renderer.endFrame();
        glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, pixels.data());
        vx::writeBmp24(settle ? "tour_night.bmp" : "tour_night_moving.bmp", w, h, pixels);
        VX_INFO("tour night: stop 0 at timeOfDay %.2f -> tour_night.bmp", double(night.timeOfDay));
    }

    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &color);
    glDeleteRenderbuffers(1, &depth);
    return 0;
}

int runFlythrough(RenderContext& rc, vx::ChunkManager& world, vx::JobSystem& jobs,
                  vx::LightEngine& light, vx::FlyCamera& camera, vx::Window& window,
                  int frames) {
    world.radius = 16;
    int w = 0, h = 0;
    window.framebufferSize(w, h);

    GLuint color = 0, depth = 0, fbo = 0;
    glCreateRenderbuffers(1, &color);
    glNamedRenderbufferStorage(color, GL_RGB8, w, h);
    glCreateRenderbuffers(1, &depth);
    glNamedRenderbufferStorage(depth, GL_DEPTH_COMPONENT32F, w, h);
    glCreateFramebuffers(1, &fbo);
    glNamedFramebufferRenderbuffer(fbo, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color);
    glNamedFramebufferRenderbuffer(fbo, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);

    vx::FxToggles fx;
    fx.timeOfDay = 0.4f;
    fx.dayCycle = false;

    // Three waypoints close enough together that a modest, pre-settled radius covers the
    // whole path — a promotional clip has no business showing the pop-in the rest of this
    // harness exists to catch. Heights come from the generator, never hardcoded, for the
    // same reason runTour's do: a fixed y buries the camera in a hill wherever the terrain
    // happens to rise above it.
    struct Waypoint {
        glm::dvec2 xz;
        float aboveSurface;
        float yaw, pitch;
    };
    const Waypoint wps[] = {
        {{16.0, 16.0}, 14.0f, -35.0f, -12.0f},
        {{16.0, 16.0}, 14.0f, -5.0f, -10.0f},
        {{40.0, 40.0}, 14.0f, 25.0f, -12.0f},
    };
    glm::dvec3 pos[std::size(wps)];
    for (std::size_t i = 0; i < std::size(wps); ++i)
        pos[i] = {wps[i].xz.x,
                  double(world.generator().surfaceHeight(int(wps[i].xz.x), int(wps[i].xz.y))) +
                      double(wps[i].aboveSurface),
                  wps[i].xz.y};

    // Settle centred on the path's midpoint so the whole route is loaded before frame 0 —
    // otherwise the first waypoint decides what's resident and the far end streams in on
    // camera, which is exactly the artifact this mode exists to avoid.
    camera.position = pos[std::size(wps) / 2];
    for (int f = 0; f < 40000; ++f) {
        window.pollEvents();
        rc.renderer.beginFrame();
        world.update(jobs, rc.renderer, light, camera.position);
        rc.renderer.endFrame();
        if (f > 200 && !world.busy() && light.idle())
            break;
    }
    std::vector<vx::ChunkDrawInfo> di, wi;
    world.collectDrawInfo(di, wi);
    rc.renderer.updateChunkMeta(di);
    rc.renderer.updateWaterMeta(wi);

    std::vector<std::uint8_t> pixels(std::size_t(w) * h * 3);
    const int legs = int(std::size(wps)) - 1;
    for (int f = 0; f < frames; ++f) {
        const float t = float(f) / float(std::max(frames - 1, 1)) * float(legs);
        const int leg = std::min(int(t), legs - 1);
        const float lt = t - float(leg);
        camera.position = glm::mix(pos[leg], pos[leg + 1], double(lt));
        camera.yawDeg = wps[leg].yaw + (wps[leg + 1].yaw - wps[leg].yaw) * lt;
        camera.pitchDeg = wps[leg].pitch + (wps[leg + 1].pitch - wps[leg].pitch) * lt;

        window.pollEvents();
        rc.renderer.beginFrame();
        world.update(jobs, rc.renderer, light, camera.position);
        rc.renderer.endFrame();
        world.collectDrawInfo(di, wi);
        rc.renderer.updateChunkMeta(di);
        rc.renderer.updateWaterMeta(wi);
        world.updateVisibilityIfNeeded(rc.renderer, camera.position, true);
        rc.renderer.beginFrame();
        renderFrame(rc, camera, fx, true, w, h, fbo, 0.6f, nullptr);
        rc.renderer.endFrame();
        glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, pixels.data());
        char name[64];
        std::snprintf(name, sizeof(name), "flyframe_%04d.bmp", f);
        vx::writeBmp24(name, w, h, pixels);
    }
    VX_INFO("flythrough: wrote %d frames (flyframe_0000.bmp..flyframe_%04d.bmp)", frames,
            frames - 1);

    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &color);
    glDeleteRenderbuffers(1, &depth);
    return 0;
}

} // namespace vx
