#include "core/AllocTracker.hpp"
#include "core/Bmp.hpp"
#include "core/Log.hpp"
#include "core/Paths.hpp"
#include "gfx/BlockTextures.hpp"
#include "gfx/GlDebug.hpp"
#include "gfx/GpuTimer.hpp"
#include "gfx/Shader.hpp"
#include "audio/AudioEngine.hpp"
#include "core/Settings.hpp"
#include "io/ChunkSerializer.hpp"
#include "io/RegionStore.hpp"
#include "jobs/JobSystem.hpp"
#include "light/LightEngine.hpp"
#include "mesh/GreedyMesher.hpp"
#include "mesh/NaiveMesher.hpp"
#include "platform/Window.hpp"
#include "render/Camera.hpp"
#include "render/Frustum.hpp"
#include "render/Overlay.hpp"
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
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

namespace {

template <typename F>
double benchBestMs(int runs, F&& fn) {
    double best = 1e9;
    for (int i = 0; i < runs; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return best;
}

// Runs at every startup: naive vs greedy on the fixed test chunk. Merged quad area per
// direction must equal the naive face count exactly — catches mesher AND palette
// storage regressions in one go, since both meshers read through Chunk::get().
bool runMesherOracle() {
    auto chunk = std::make_unique<vx::Chunk>();
    vx::fillTestTerrain(*chunk);

    std::vector<vx::ChunkVertex> naiveVerts;
    vx::MeshStats naive;
    const double naiveBest =
        benchBestMs(50, [&] { naive = vx::meshNaive(*chunk, naiveVerts); });

    auto mesher = std::make_unique<vx::GreedyMesher>();
    std::vector<std::uint64_t> quads;
    std::vector<std::uint64_t> water;
    vx::GreedyStats greedy;
    const double greedyBest =
        benchBestMs(50, [&] { greedy = mesher->mesh(*chunk, nullptr, quads, water); });

    bool ok = true;
    for (int d = 0; d < 6; ++d)
        ok = ok && greedy.areaPerDir[d] == std::int64_t(naive.facesPerDir[d]);
    if (!ok) {
        for (int d = 0; d < 6; ++d)
            VX_ERROR("oracle dir %d: greedy area %lld vs naive faces %d", d,
                     static_cast<long long>(greedy.areaPerDir[d]), naive.facesPerDir[d]);
        return false;
    }
    VX_INFO("oracle ok | naive %d tris %.3f ms | greedy %d quads %.3f ms", naive.triangles,
            naiveBest, greedy.quads, greedyBest);
    return true;
}

struct CameraFrame {
    glm::mat4 viewProj;
    vx::Frustum frustum;
    glm::ivec3 cameraChunk;
    glm::vec3 cameraFrac;
};

CameraFrame buildCameraFrame(const vx::FlyCamera& camera, int fbWidth, int fbHeight,
                             glm::vec2 jitterNdc = glm::vec2(0.0f)) {
    CameraFrame f;
    glm::mat4 proj = camera.projReversedZ(float(fbWidth) / float(fbHeight));
    proj[2][0] += jitterNdc.x; // sub-pixel jitter for TAA (clip-space shift per unit w)
    proj[2][1] += jitterNdc.y;
    f.viewProj = proj * camera.viewAtOrigin();
    f.frustum = vx::Frustum::fromViewProj(f.viewProj);
    f.cameraChunk = glm::ivec3(glm::floor(camera.position / double(vx::kChunkSize)));
    f.cameraFrac = glm::vec3(camera.position - glm::dvec3(f.cameraChunk) * double(vx::kChunkSize));
    return f;
}

struct SkyLight {
    glm::vec3 dir;     // toward the sun/moon
    glm::vec3 color;   // direct radiance: warm+dim near the horizon, ~0 at night
    glm::vec3 ambient; // sky fill: blue by day, dark cool at night
};
// sun along a day arc from time-of-day (0.25 sunrise east, 0.5 noon, 0.75 sunset west)
SkyLight computeSky(float timeOfDay) {
    const float theta = (timeOfDay - 0.25f) * 6.2831853f;
    const glm::vec3 dir = glm::normalize(glm::vec3(std::cos(theta), std::sin(theta), 0.30f));
    const float alt = dir.y; // -1..1
    const float day = glm::clamp(alt * 3.0f + 0.12f, 0.0f, 1.0f); // 0 below the horizon
    const float high = glm::clamp(alt * 1.8f, 0.0f, 1.0f);
    const glm::vec3 color = glm::mix(glm::vec3(2.7f, 1.25f, 0.5f),   // horizon warm
                                     glm::vec3(2.05f, 1.85f, 1.5f),  // overhead white
                                     high) *
                            day;
    const glm::vec3 ambient =
        glm::mix(glm::vec3(0.03f, 0.045f, 0.08f), glm::vec3(0.26f, 0.36f, 0.55f), day);
    return {dir, color, ambient};
}

// low-discrepancy sub-pixel offset for the TAA jitter
float halton(int i, int base) {
    float f = 1.0f, r = 0.0f;
    while (i > 0) {
        f /= float(base);
        r += f * float(i % base);
        i /= base;
    }
    return r;
}

// carried across frames so the TAA resolve can reproject into the previous frame
struct TaaState {
    glm::mat4 prevViewProj{1.0f};
    glm::dvec3 prevCamPos{0.0};
    int frame = 0;
    bool valid = false;
};

struct RenderContext {
    vx::Renderer& renderer;
    vx::ShadowMap& shadows;
    vx::PostFx& postfx;
    vx::Overlay& overlay;
    unsigned int drawProgram;
    unsigned int waterProgram;
    unsigned int blockTex;
};

struct FrameTimers {
    vx::GpuTimer shadow, scene, post;
};

// One full frame: cascades -> HDR scene (opaque + transparent water) -> post into target.
void renderFrame(RenderContext& rc, const vx::FlyCamera& camera, const vx::FxToggles& fx,
                 bool caveCull, int fbW, int fbH, unsigned int targetFbo, float time,
                 FrameTimers* timers, bool hasSelection = false,
                 const glm::vec3& selMinCamRel = glm::vec3(0.0f), bool underwater = false,
                 TaaState* taa = nullptr, float fogEnd = 1e9f) {
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
    const SkyLight sky = computeSky(fx.timeOfDay);
    const glm::vec3 sunL = sky.dir;

    rc.postfx.resize(rw, rh);

    if (timers) timers->shadow.begin();
    if (fx.shadows) {
        rc.shadows.update(camera.position, camera.forward(), sunL,
                          glm::radians(camera.fovYDeg), aspect);
        rc.shadows.render(rc.renderer, cam.cameraChunk, cam.cameraFrac);
    }
    if (timers) timers->shadow.end();

    if (timers) timers->scene.begin();
    rc.postfx.beginScene();
    glClearColor(0.42f, 0.68f, 1.05f, 1.0f); // HDR sky
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glProgramUniform3fv(rc.drawProgram, 4, 1, &sunL.x);
    glProgramUniform3fv(rc.drawProgram, 21, 1, &sky.color.x);
    glProgramUniform3fv(rc.drawProgram, 22, 1, &sky.ambient.x);
    glProgramUniform1f(rc.drawProgram, 5, fx.shadows ? 1.0f : 0.0f);
    glProgramUniformMatrix4fv(rc.drawProgram, 6, 3, GL_FALSE,
                              &rc.shadows.cascadeVp()[0][0][0]);
    const glm::vec3 splits{vx::ShadowMap::kSplits[0], vx::ShadowMap::kSplits[1],
                           vx::ShadowMap::kSplits[2]};
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

    // transparent water over the opaque scene: alpha blend, depth test but no write.
    // snapshot the opaque colour first so the water shader can reflect it (SSR).
    rc.postfx.captureOpaqueForWater();
    const float wf = 1.0f / std::tan(glm::radians(camera.fovYDeg) * 0.5f);
    const glm::vec2 projScale{aspect / wf, 1.0f / wf};
    glProgramUniform1f(rc.waterProgram, 4, time);
    glProgramUniform3fv(rc.waterProgram, 5, 1, &sunL.x);
    glProgramUniform3fv(rc.waterProgram, 8, 1, &sky.color.x);
    glProgramUniform2fv(rc.waterProgram, 6, 1, &projScale.x);
    glProgramUniform1f(rc.waterProgram, 7, camera.nearPlane);
    glBindTextureUnit(2, rc.postfx.sceneCopyTex());
    glBindTextureUnit(3, rc.postfx.depthTex());
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    rc.renderer.drawWater(rc.waterProgram, cam.viewProj, cam.frustum, cam.cameraChunk,
                          cam.cameraFrac);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);

    // block-selection wireframe, in the HDR scene so it depth-tests against the world
    if (hasSelection)
        rc.overlay.drawSelection(cam.viewProj, selMinCamRel);
    if (timers) timers->scene.end();

    if (timers) timers->post.begin();
    const float f = 1.0f / std::tan(glm::radians(camera.fovYDeg) * 0.5f);
    vx::PostFx::RunParams rp;
    rp.projScale = {aspect / f, 1.0f / f};
    rp.nearPlane = camera.nearPlane;
    rp.invViewRot = glm::transpose(glm::mat3(camera.viewAtOrigin()));
    rp.sunL = sunL;
    rp.sunColor = sky.color;
    rp.fog = {fogEnd * 0.55f, fogEnd};
    rp.shadows = &rc.shadows;
    rp.toggles = fx;
    rp.time = time;
    // only let PostFx run these when we actually have the reprojection history for them
    rp.toggles.taa = useTaa;
    rp.toggles.motionBlur = useMotion;
    rp.underwater = underwater;
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

int runSmokeTest(RenderContext& rc, vx::ChunkManager& world, vx::JobSystem& jobs,
                 vx::LightEngine& light, vx::FlyCamera& camera, vx::DebugHud& hud,
                 vx::Window& window, const std::filesystem::path& outPath) {
    world.radius = 10; // keep the smoke quick; the regen test bumps to 16 on its own
    const auto drain = [&] {
        for (int i = 0; i < 20000 && (world.busy() || !light.idle() || i == 0); ++i) {
            rc.renderer.beginFrame();
            world.update(jobs, rc.renderer, light, camera.position);
            rc.renderer.endFrame();
        }
    };

    drain();
    world.logSummary();

    // far chunks generate coarse (no caves); approaching one must re-generate it detailed.
    // needs a radius past the coarse threshold, so bump briefly then drop back.
    {
        const auto caveAir = [&](int cx, int cz) {
            int n = 0;
            for (int y = 0; y < vx::kWorldChunksY; ++y) {
                const vx::Chunk* c = world.chunkData({cx, y, cz});
                if (!c)
                    continue;
                for (int lz = 0; lz < 32; ++lz)
                    for (int lx = 0; lx < 32; ++lx) {
                        const int surf =
                            world.generator().surfaceHeight(cx * 32 + lx, cz * 32 + lz);
                        for (int ly = 0; ly < 32; ++ly)
                            if (const int wy = y * 32 + ly;
                                wy > 4 && wy < surf - 4 && c->get(lx, ly, lz) == vx::blocks::Air)
                                ++n;
                    }
            }
            return n;
        };
        const int cx = 15; // 15 chunks east -> coarse (cave-less) at radius 16
        world.radius = 16;
        drain();
        const int farAir = caveAir(cx, 0);
        camera.position = {double(cx * 32 + 16), 110.0, 16.0}; // move next to it -> re-gen
        // a big move triggers heavy re-streaming; the detailed re-gens land over several
        // settle cycles (the coarse mesh covers meanwhile), so wait for them to finish
        int nearAir = 0;
        for (int i = 0; i < 80 && nearAir <= 100; ++i) {
            drain();
            nearAir = caveAir(cx, 0);
        }
        camera.position = {16.0, 110.0, 16.0};
        world.radius = 10;
        drain();
        const bool ok = farAir < 30 && nearAir > 100;
        VX_INFO("regen: far column cave-air %d (want ~0), detailed after approach %d (want "
                ">0) -> %s",
                farAir, nearAir, ok ? "PASS" : "FAIL");
        if (!ok)
            return 1;
    }

    // lighting checks against the enclosed glow room in the terrain
    {
        const int skyInRoom = vx::light::sky(light.lightAt({16, 48, 16}));
        const int glowAbove = vx::light::block(light.lightAt({16, 46, 16}));
        const int glowEdge = vx::light::block(light.lightAt({16, 48, 19}));
        // open air near the top of the world — always full sky (trees/terrain can't reach)
        const int skySurface = vx::light::sky(light.lightAt({16, vx::kWorldHeight - 4, 16}));
        bool ok = skyInRoom == 0 && glowAbove == 13 && glowEdge > 0 && glowEdge < 13 &&
                  skySurface == 15;
        VX_INFO("light: room sky %d (want 0), above-glow %d (want 13), edge %d, surface sky "
                "%d (want 15) -> %s",
                skyInRoom, glowAbove, glowEdge, skySurface, ok ? "PASS" : "FAIL");
        if (!ok)
            return 1;

        // M18: the three block-light channels propagate independently, so a warm emitter
        // must still be warm out at its falloff - red reaching further than blue. A
        // monochrome engine would report all three equal here.
        {
            const std::uint16_t v = light.lightAt({16, 46, 16});
            const int r = vx::light::red(v), g = vx::light::green(v), b = vx::light::blue(v);
            const std::uint16_t far = light.lightAt({16, 48, 19});
            const int fr = vx::light::red(far), fb = vx::light::blue(far);
            const bool tinted = r > g && g > b && fr > fb;
            VX_INFO("light: glow colour rgb %d/%d/%d, at falloff r%d b%d -> %s", r, g, b, fr,
                    fb, tinted ? "PASS" : "FAIL");
            if (!tinted)
                return 1;
        }


        vx::Chunk* roomChunk = world.chunkDataMutable({0, 1, 0});
        roomChunk->set(16, 13, 16, vx::blocks::Air);
        light.onBlockChanged({16, 45, 16}, vx::blocks::Glow, vx::blocks::Air);
        while (!light.idle())
            light.process(10.0);
        const int afterRemove = vx::light::block(light.lightAt({16, 46, 16}));
        roomChunk->set(16, 13, 16, vx::blocks::Glow);
        light.onBlockChanged({16, 45, 16}, vx::blocks::Air, vx::blocks::Glow);
        while (!light.idle())
            light.process(10.0);
        const int afterRestore = vx::light::block(light.lightAt({16, 46, 16}));
        ok = afterRemove == 0 && afterRestore == 13;
        VX_INFO("light: removal %d (want 0), re-add %d (want 13) -> %s", afterRemove,
                afterRestore, ok ? "PASS" : "FAIL");
        if (!ok)
            return 1;
        drain();
    }

    // hitch check: teleport 16 chunks east and re-stream the world
    camera.position.x += 512.0;
    world.resetPhasePeak();
    const auto restreamStart = std::chrono::steady_clock::now();
    double worstMs = 0.0;
    int frames = 0;
    for (; frames < 20000; ++frames) {
        rc.renderer.beginFrame();
        const auto t0 = std::chrono::steady_clock::now();
        world.update(jobs, rc.renderer, light, camera.position);
        worstMs = std::max(
            worstMs,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count());
        rc.renderer.endFrame();
        if (frames > 0 && !world.busy())
            break;
    }
    // Wall clock, not update count: the loop spins as fast as it can, so a cheaper main
    // thread just means MORE iterations. Only elapsed time says whether streaming actually
    // finished sooner.
    const double restreamMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                  restreamStart)
            .count();
    VX_INFO("stress: re-streamed after teleport in %.0f ms / %d updates, worst main-thread "
            "cost %.2f ms",
            restreamMs, frames, worstMs);
    {
        const auto pk = world.phasePeak();
        VX_INFO("stress: worst phase ms — rescan %.2f, evict %.2f, seed %.2f, propagate "
                "%.2f, scan %.2f, upload %.2f",
                pk.rescanMs, pk.evictMs, pk.seedMs, pk.propagateMs, pk.scanMs, pk.uploadMs);
        // The peaks above are one bad frame each. These are the whole re-stream, which is
        // what tells you which phase to actually go after.
        const auto tot = world.phaseTotal();
        const double n = std::max(1, world.updateCount());
        const double wall = tot.rescanMs + tot.evictMs + tot.seedMs + tot.propagateMs +
                            tot.scanMs + tot.uploadMs;
        VX_INFO("stress: total phase ms over %d updates — rescan %.1f, evict %.1f, seed "
                "%.1f, propagate %.1f, scan %.1f, upload %.1f (sum %.1f)",
                world.updateCount(), tot.rescanMs, tot.evictMs, tot.seedMs, tot.propagateMs,
                tot.scanMs, tot.uploadMs, wall);
        VX_INFO("stress: light = %.1f ms of %.1f ms main-thread (%.0f%%), %.2f ms/update avg",
                tot.seedMs + tot.propagateMs, wall,
                100.0 * (tot.seedMs + tot.propagateMs) / std::max(wall, 1e-6),
                (tot.seedMs + tot.propagateMs) / n);
        // Worker-side cost and arena health after the worst churn the engine ever sees.
        // The arena line is the reason there is no defrag scheduler: if the biggest free
        // block still holds nearly all the free space after this, there is nothing to
        // compact and a defrag pass would be pure code.
        const vx::ChunkManager::Stats ws = world.stats();
        const vx::Renderer::ArenaStats as = rc.renderer.arenaStats();
        VX_INFO("stress: worker gen %.3f ms/chunk, mesh %.3f ms/chunk", ws.genMsAvg,
                ws.meshMsAvg);
        VX_INFO("stress: arena %u/%u quads, %d free blocks, largest holds %.0f%% of free space",
                as.usedQuads, as.capacityQuads, as.freeBlocks,
                as.capacityQuads > as.usedQuads
                    ? 100.0 * as.largestFreeBlock / double(as.capacityQuads - as.usedQuads)
                    : 100.0);
    }
    // Return leg. The outbound teleport lands on ground this run has never seen, so it can
    // only pay the cost of caching evicted chunks; coming back is where a cache can be hit.
    // Timing only the outbound half would report the bill and hide the refund.
    camera.position.x -= 512.0;
    {
        const auto backStart = std::chrono::steady_clock::now();
        int backFrames = 0;
        for (; backFrames < 20000; ++backFrames) {
            rc.renderer.beginFrame();
            world.update(jobs, rc.renderer, light, camera.position);
            rc.renderer.endFrame();
            if (backFrames > 0 && !world.busy())
                break;
        }
        VX_INFO("stress: returned to origin in %.0f ms / %d updates",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                          backStart)
                    .count(),
                backFrames);
    }
    drain();

    int w = 0;
    int h = 0;
    window.framebufferSize(w, h);

    GLuint color = 0;
    glCreateRenderbuffers(1, &color);
    glNamedRenderbufferStorage(color, GL_RGB8, w, h);
    GLuint fbo = 0;
    glCreateFramebuffers(1, &fbo);
    glNamedFramebufferRenderbuffer(fbo, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color);
    if (glCheckNamedFramebufferStatus(fbo, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        VX_ERROR("smoke: framebuffer incomplete");
        return 1;
    }

    std::vector<vx::ChunkDrawInfo> drawInfo;
    std::vector<vx::ChunkDrawInfo> waterInfo;
    world.collectDrawInfo(drawInfo, waterInfo);
    rc.renderer.updateChunkMeta(drawInfo);
    rc.renderer.updateWaterMeta(waterInfo);
    world.updateVisibilityIfNeeded(rc.renderer, camera.position, true);

    vx::FxToggles fx;
    const vx::ChunkManager::Stats ws = world.stats();
    const vx::Renderer::ArenaStats as = rc.renderer.arenaStats();
    for (int i = 0; i < 5; ++i) {
        window.pollEvents();
        rc.renderer.beginFrame();
        hud.beginFrame();
        renderFrame(rc, camera, fx, true, w, h, fbo, 0.6f, nullptr);
        const vx::FrameStats stats{.frameMs = 1.0f,
                                   .gpuMs = 0.5f,
                                   .heapAllocs = 0,
                                   .drawCalls = 1,
                                   .quads = int(ws.totalQuads),
                                   .triangles = int(ws.totalQuads) * 2,
                                   .chunksLoaded = ws.loaded,
                                   .chunksMeshed = ws.meshed,
                                   .genQueue = ws.genQueue,
                                   .jobsInFlight = ws.jobsInFlight,
                                   .uploadsPending = ws.pendingUploads,
                                   .uploadMs = float(ws.uploadMsLast),
                                   .voxelBytes = ws.voxelBytes,
                                   .quadBytes = ws.quadBytes,
                                   .gpuChunksDrawn = rc.renderer.gpuVisibleChunks(),
                                   .arenaUsedQuads = as.usedQuads,
                                   .arenaCapQuads = as.capacityQuads,
                                   .arenaFreeBlocks = as.freeBlocks};
        hud.diagnostics(stats, camera);
        hud.render();
        rc.renderer.endFrame();
    }
    VX_INFO("smoke: gpu culled %d of %d chunks (1 draw call, cave culling on)",
            rc.renderer.debugReadDrawCount(), int(drawInfo.size()));

    std::vector<std::uint8_t> pixels(std::size_t(w) * std::size_t(h) * 3u);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glNamedFramebufferReadBuffer(fbo, GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, pixels.data());

    // post-processing owns the sky color now, so compare against an actual sky pixel
    // (top-left corner; the camera pitches down so the top rows are always sky)
    const std::size_t skyIdx = (std::size_t(h - 3) * std::size_t(w) + 4) * 3;
    std::size_t drawn = 0;
    for (std::size_t i = 0; i < pixels.size(); i += 3) {
        if (std::abs(int(pixels[i + 0]) - int(pixels[skyIdx + 0])) > 10 ||
            std::abs(int(pixels[i + 1]) - int(pixels[skyIdx + 1])) > 10 ||
            std::abs(int(pixels[i + 2]) - int(pixels[skyIdx + 2])) > 10)
            ++drawn;
    }

    if (!vx::writeBmp24(outPath, w, h, pixels))
        return 1;

    VX_INFO("smoke: %llu of %d pixels differ from the sky (%.1f%%), wrote %s",
            static_cast<unsigned long long>(drawn), w * h,
            100.0 * double(drawn) / (double(w) * double(h)), outPath.string().c_str());
    if (drawn < 1000) {
        VX_ERROR("smoke: nothing rendered");
        return 1;
    }

    // physics: drop a player from the sky onto the surface, fixed deterministic ticks
    {
        vx::Player testPlayer;
        testPlayer.position = {16.0, 125.0, 16.0};
        for (int i = 0; i < 1200; ++i)
            testPlayer.step(world, {});
        // tree-agnostic: it must come to rest on the ground (or a tree/leaf) — on solid
        // footing, not still falling
        const glm::ivec3 below{16, int(std::floor(testPlayer.position.y)) - 1, 16};
        const bool ok = testPlayer.onGround && vx::blocks::solid(world.blockAtWorld(below));
        VX_INFO("physics: landed at y=%.4f resting on block %d -> %s", testPlayer.position.y,
                int(world.blockAtWorld(below)), ok ? "PASS" : "FAIL");
        if (!ok)
            return 1;

        // ores: depth-banded, so coal must outnumber iron and both must actually exist.
        // counts a column of stone under spawn rather than trusting that it compiled.
        {
            int coal = 0, iron = 0, stone = 0;
            for (int x = 0; x < 48; ++x)
                for (int z = 0; z < 48; ++z) {
                    const int surf = world.generator().surfaceHeight(x, z);
                    for (int y = 4; y < surf - 4; ++y) {
                        const vx::BlockId b = world.blockAtWorld({x, y, z});
                        if (b == vx::blocks::CoalOre) ++coal;
                        else if (b == vx::blocks::IronOre) ++iron;
                        else if (b == vx::blocks::Stone) ++stone;
                    }
                }
            const bool oreOk = coal > 0 && iron > 0 && coal > iron && stone > coal * 4;
            VX_INFO("ores: %d coal, %d iron in %d stone -> %s", coal, iron, stone,
                    oreOk ? "PASS" : "FAIL");
            if (!oreOk)
                return 1;
        }

        const glm::ivec3 ep{18, world.generator().surfaceHeight(18, 18) + 6, 18};
        world.setBlock(rc.renderer, light, ep, vx::blocks::Stone);
        const bool placed = world.blockAtWorld(ep) == vx::blocks::Stone;
        world.setBlock(rc.renderer, light, ep, vx::blocks::Air);
        const bool broken = world.blockAtWorld(ep) == vx::blocks::Air;
        VX_INFO("edits: place %s, break %s", placed ? "PASS" : "FAIL",
                broken ? "PASS" : "FAIL");
        if (!placed || !broken)
            return 1;

        // Batch edits, the path a fluid/falling-block sim will drive. Deliberately spans
        // a chunk border and repeats a voxel, because those are the two things the batch
        // path has to get right that the single-edit path never exercises: one remesh per
        // chunk however many edits hit it, and last-wins ordering within the batch.
        const int surf = world.generator().surfaceHeight(30, 18) + 6;
        std::vector<vx::ChunkManager::BlockEdit> batch;
        for (int i = 0; i < 8; ++i) // 28..35 crosses the x=32 chunk boundary
            batch.push_back({{28 + i, surf, 18}, vx::blocks::Stone});
        batch.push_back({{30, surf, 18}, vx::blocks::Glow}); // overwrite one, last wins
        const int applied = world.setBlocks(rc.renderer, light, batch);
        bool batchOk = applied == 9;
        for (int i = 0; i < 8; ++i) {
            const vx::BlockId want =
                (28 + i) == 30 ? vx::blocks::Glow : vx::blocks::Stone;
            batchOk = batchOk && world.blockAtWorld({28 + i, surf, 18}) == want;
        }
        // the glow block must have lit its neighbourhood, i.e. light settled after the
        // whole batch rather than being skipped
        const bool batchLit = vx::light::block(light.lightAt({30, surf + 1, 18})) > 0;
        VX_INFO("edits: batch %d/9 applied, values %s, light %s", applied,
                batchOk ? "PASS" : "FAIL", batchLit ? "PASS" : "FAIL");
        if (!batchOk || !batchLit)
            return 1;

        std::vector<vx::ChunkManager::BlockEdit> undo;
        for (int i = 0; i < 8; ++i)
            undo.push_back({{28 + i, surf, 18}, vx::blocks::Air});
        world.setBlocks(rc.renderer, light, undo);

        // What the batch path is actually for: the same 64 edits one at a time versus in
        // one call. A sim tick is the second shape, and the first is what it used to cost.
        // per-column surface, so every one of these is genuinely in air — placing into
        // blocks that already hold the target id makes both paths no-ops and the
        // comparison meaningless
        std::vector<vx::ChunkManager::BlockEdit> wall, clear;
        for (int i = 0; i < 64; ++i) {
            const int wx = 40 + (i & 7);
            const glm::ivec3 wp{wx, world.generator().surfaceHeight(wx, 40) + 2 + (i >> 3),
                                40};
            wall.push_back({wp, vx::blocks::Stone});
            clear.push_back({wp, vx::blocks::Air});
        }
        const auto tOne = std::chrono::steady_clock::now();
        for (const auto& e : wall)
            world.setBlock(rc.renderer, light, e.pos, e.id);
        const double oneMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tOne)
                .count();
        world.setBlocks(rc.renderer, light, clear);

        const auto tBatch = std::chrono::steady_clock::now();
        world.setBlocks(rc.renderer, light, wall);
        const double batchMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                      tBatch)
                .count();
        world.setBlocks(rc.renderer, light, clear);
        VX_INFO("edits: 64 blocks one-at-a-time %.2f ms vs batched %.2f ms (%.1fx)", oneMs,
                batchMs, batchMs > 0.0 ? oneMs / batchMs : 0.0);
    }

    // persistence: serialize->disk->deserialize round-trip and an edit that survives a
    // full reload from the region file
    {
        // (a) serializer round-trip on a generated cave chunk: every voxel must match
        auto genChunk = std::make_unique<vx::Chunk>();
        world.generator().generate(*genChunk, {2, 1, 2});
        std::vector<std::uint8_t> bytes;
        const std::size_t n = vx::ChunkSerializer::serialize(*genChunk, bytes);
        auto roundTrip = std::make_unique<vx::Chunk>();
        bool same = vx::ChunkSerializer::deserialize(bytes, *roundTrip);
        for (int y = 0; y < vx::kChunkSize && same; ++y)
            for (int z = 0; z < vx::kChunkSize && same; ++z)
                for (int x = 0; x < vx::kChunkSize && same; ++x)
                    same = genChunk->get(x, y, z) == roundTrip->get(x, y, z);
        VX_INFO("persistence: chunk serialize round-trip %s (%llu bytes vs %d B raw, %.0fx)",
                same ? "PASS" : "FAIL", static_cast<unsigned long long>(n),
                vx::kChunkSize3 * 2, double(vx::kChunkSize3 * 2) / double(n));
        if (!same)
            return 1;

        // (a2) settings round-trip: write non-default values, read them into a fresh set
        // of defaults, and check every field came back. Uses its own file so it can never
        // touch the real saves/settings.cfg, and nothing here feeds the render.
        {
            const std::filesystem::path sf = outPath.parent_path() / "vxsettings_smoke.cfg";
            vx::FxToggles wrote;
            wrote.shadows = false;
            wrote.ssgi = false;
            wrote.taa = true;
            wrote.motionBlur = true;
            wrote.timeOfDay = 0.77f;
            wrote.exposure = 1.85f;
            wrote.crosshair[0] = 0.20f;
            wrote.crosshair[3] = 0.60f;
            vx::Settings::save(sf, wrote, true, false, 24, 95.0f);

            vx::FxToggles read; // all defaults; load must overwrite each one
            bool vsyncR = false;
            bool caveCullR = true;
            int radiusR = 16;
            float fovR = 70.0f;
            const bool got = vx::Settings::load(sf, read, vsyncR, caveCullR, radiusR, fovR);
            const bool ok = got && !read.shadows && !read.ssgi && read.taa &&
                            read.motionBlur && std::abs(read.timeOfDay - 0.77f) < 0.001f &&
                            std::abs(read.exposure - 1.85f) < 0.001f && vsyncR &&
                            !caveCullR && radiusR == 24 && std::abs(fovR - 95.0f) < 0.001f &&
                            std::abs(read.crosshair[0] - 0.20f) < 0.001f &&
                            std::abs(read.crosshair[3] - 0.60f) < 0.001f;
            std::error_code sec;
            std::filesystem::remove(sf, sec);
            VX_INFO("persistence: settings round-trip %s", ok ? "PASS" : "FAIL");
            if (!ok)
                return 1;
        }

        // (b) edit a surface block, save to disk, then load it back into a FRESH chunk
        // straight from the region file — the edit must survive
        const int surf = world.generator().surfaceHeight(70, 70);
        const glm::ivec3 wp{70, surf + 1, 70};
        const vx::ChunkPos cp{wp.x >> 5, wp.y >> 5, wp.z >> 5};
        world.setBlock(rc.renderer, light, wp, vx::blocks::Glow);
        world.saveDirty(); // serialize dirty chunks + flush region files

        auto reloaded = std::make_unique<vx::Chunk>();
        const bool present = world.regions().load(cp, *reloaded);
        const bool kept = present && reloaded->get(wp.x & 31, wp.y & 31, wp.z & 31) ==
                                         vx::blocks::Glow;
        const auto rs = world.regions().stats();
        VX_INFO("persistence: edit survived reload from disk %s (%d chunk(s), %.1f KB on disk)",
                kept ? "PASS" : "FAIL", rs.chunksStored,
                double(rs.bytesStored) / 1024.0);
        if (!kept)
            return 1;

        // (c) end-to-end: fly far enough to evict the chunk, then back — it must
        // regenerate FROM DISK (not the seed) with the edit intact
        camera.position = {4096.0, 110.0, 4096.0};
        drain();
        camera.position = {16.0, 110.0, 16.0};
        drain();
        const bool live = world.blockAtWorld(wp) == vx::blocks::Glow;
        VX_INFO("persistence: edit present after evict + reload from disk %s",
                live ? "PASS" : "FAIL");
        if (!live)
            return 1;

        // A region file records how many chunk slots a column has, which encodes
        // kWorldChunksY. Raising the world height is a legitimate change (M22 wants it and
        // it does NOT require cubic chunks), so what matters is that an old save is
        // rejected rather than read with the wrong stride. Forge a header from a
        // differently-tall world and check load() refuses it.
        const std::filesystem::path badDir = outPath.parent_path() / "vxsave_height";
        std::error_code bec;
        std::filesystem::remove_all(badDir, bec);
        std::filesystem::create_directories(badDir, bec);
        {
            constexpr int kOtherHeight = vx::kWorldChunksY * 2;
            const int slots = 16 * 16 * kOtherHeight; // kRegionSize is 16
            std::vector<std::uint8_t> f{'V', 'X', 'R', 'G'};
            const auto put32 = [&](std::uint32_t v) {
                for (int i = 0; i < 4; ++i)
                    f.push_back(std::uint8_t(v >> (8 * i)));
            };
            put32(1); // version
            put32(std::uint32_t(slots));
            f.resize(f.size() + std::size_t(slots) * 8, 0); // empty slot directory
            std::ofstream out(badDir / "r.0.0.vxr", std::ios::binary);
            out.write(reinterpret_cast<const char*>(f.data()), std::streamsize(f.size()));
        }
        vx::RegionStore probe(badDir);
        vx::Chunk scratch;
        const bool rejected = !probe.load({0, 0, 0}, scratch);
        VX_INFO("persistence: region from a %d-tall world rejected %s", vx::kWorldChunksY * 2,
                rejected ? "PASS" : "FAIL");
        std::filesystem::remove_all(badDir, bec);
        if (!rejected)
            return 1;
    }

    // screenshot from inside the glow room
    camera.position = {16.5, 48.0, 16.5};
    camera.yawDeg = 40.0f;
    camera.pitchDeg = -20.0f;
    world.updateVisibilityIfNeeded(rc.renderer, camera.position, true);
    rc.renderer.beginFrame();
    renderFrame(rc, camera, fx, true, w, h, fbo, 0.6f, nullptr);
    rc.renderer.endFrame();
    glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, pixels.data());
    std::filesystem::path roomPath = outPath;
    roomPath.replace_filename(outPath.stem().string() + "_room.bmp");
    vx::writeBmp24(roomPath, w, h, pixels);
    VX_INFO("smoke: glow room screenshot -> %s", roomPath.string().c_str());

    // per-effect GPU cost: back on the surface, time the shadow/scene/post stages with
    // GPU queries, then toggle each effect and measure the delta on the post stage.
    camera.position = {16.0, 110.0, 16.0};
    camera.yawDeg = -35.0f;
    camera.pitchDeg = -25.0f;
    for (int i = 0; i < 6000 && (world.busy() || !light.idle()); ++i) {
        rc.renderer.beginFrame();
        world.update(jobs, rc.renderer, light, camera.position); // settle LOD before timing
        rc.renderer.endFrame();
    }
    world.collectDrawInfo(drawInfo, waterInfo);
    rc.renderer.updateChunkMeta(drawInfo);
    rc.renderer.updateWaterMeta(waterInfo);
    world.updateVisibilityIfNeeded(rc.renderer, camera.position, true);
    // note: this GPU-query benchmark is unreliable while re-LOD workers churn in the
    // background (negative deltas) — trust the in-app HUD gpu timer for real numbers.
    FrameTimers ft;
    ft.shadow.init();
    ft.scene.init();
    ft.post.init();
    const auto timeConfig = [&](vx::FxToggles cfg) {
        double s = 0, sc = 0, p = 0;
        for (int i = 0; i < 40; ++i) {
            rc.renderer.beginFrame();
            renderFrame(rc, camera, cfg, true, w, h, fbo, 0.6f, &ft);
            rc.renderer.endFrame();
            glFinish();
            if (i >= 8) { // let the query ring fill
                s += ft.shadow.milliseconds();
                sc += ft.scene.milliseconds();
                p += ft.post.milliseconds();
            }
        }
        return glm::dvec3(s, sc, p) / 32.0;
    };
    vx::FxToggles all;
    const glm::dvec3 tAll = timeConfig(all);
    vx::FxToggles noShadow = all; noShadow.shadows = false;
    vx::FxToggles noSsao = all; noSsao.ssao = false;
    vx::FxToggles noVol = all; noVol.volumetric = false;
    vx::FxToggles noBloom = all; noBloom.bloom = false;
    vx::FxToggles none{}; none.shadows = none.ssao = none.volumetric = none.bloom = false;
    const glm::dvec3 tNoShadow = timeConfig(noShadow);
    const glm::dvec3 tNoSsao = timeConfig(noSsao);
    const glm::dvec3 tNoVol = timeConfig(noVol);
    const glm::dvec3 tNoBloom = timeConfig(noBloom);
    const glm::dvec3 tNone = timeConfig(none);
    VX_INFO("gpu cost @ %dx%d, radius %d: base scene %.2f ms, full pipeline %.2f ms", w, h,
            world.radius, tNone.y, tAll.x + tAll.y + tAll.z);
    VX_INFO("  shadows +%.2f | ssao +%.2f | volumetric +%.2f | bloom +%.2f ms",
            (tAll.x + tAll.y + tAll.z) - (tNoShadow.x + tNoShadow.y + tNoShadow.z),
            (tAll.z - tNoSsao.z), (tAll.z - tNoVol.z), (tAll.z - tNoBloom.z));
    return 0;
}

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
    fx.timeOfDay = 0.32f;

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
                    int top = vx::kWorldHeight - 1;
                    while (top >= 0 && !vx::blocks::opaque(world.blockAtWorld({wx, top, wz})))
                        --top;
                    // sample a few air voxels above the surface, not all of them
                    for (int y = top + 1; y < std::min(top + 12, vx::kWorldHeight); ++y) {
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
        const vx::ChunkManager::Stats st = world.stats();
        VX_INFO("tour %d @ (%.0f,%.0f,%.0f): %d chunks %d meshed %d UNLIT, %d drawn, "
                "busy=%d -> %s",
                i, stops[i].x, stops[i].y, stops[i].z, st.loaded, st.meshed, st.unlit,
                rc.renderer.gpuVisibleChunks(), world.busy() ? 1 : 0, name);
    }
    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &color);
    glDeleteRenderbuffers(1, &depth);
    return 0;
}

// Deterministic performance harness (--bench). The whole of Tier 1 is "make it faster",
// and the older in-smoke benchmark averages 32 frames while the world is still churning,
// which is why it could report negative deltas. This one settles the world completely
// first, holds a fixed camera and a fixed time of day, and reports the MEDIAN over many
// frames — medians shrug off the odd scheduling spike that wrecks a mean.
//
// TAA is off on purpose: its history makes consecutive frames cost different amounts,
// which is exactly the variance we're trying to measure through.
int runBenchmark(RenderContext& rc, vx::ChunkManager& world, vx::JobSystem& jobs,
                 vx::LightEngine& light, vx::FlyCamera& camera, vx::Window& window,
                 int radius) {
    struct View {
        const char* name;
        glm::dvec3 pos;
        float yaw, pitch;
    };
    // one looking across the surface (shadow + terrain heavy), one high looking down
    // (maximum chunk count in frame), one at ground level in trees (overdraw)
    const View views[] = {
        {"surface", {16.0, 110.0, 16.0}, -35.0f, -15.0f},
        {"aerial", {16.0, 190.0, 16.0}, -35.0f, -55.0f},
        {"ground", {96.0, 96.0, 96.0}, 25.0f, -2.0f},
    };

    world.radius = radius;
    int w = 0;
    int h = 0;
    window.framebufferSize(w, h);

    GLuint color = 0;
    glCreateRenderbuffers(1, &color);
    glNamedRenderbufferStorage(color, GL_RGB8, w, h);
    GLuint depth = 0;
    glCreateRenderbuffers(1, &depth);
    glNamedRenderbufferStorage(depth, GL_DEPTH_COMPONENT32F, w, h);
    GLuint fbo = 0;
    glCreateFramebuffers(1, &fbo);
    glNamedFramebufferRenderbuffer(fbo, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color);
    glNamedFramebufferRenderbuffer(fbo, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);

    vx::FxToggles fx;
    fx.taa = false;
    fx.dayCycle = false;
    fx.timeOfDay = 0.32f;

    FrameTimers ft;
    ft.shadow.init();
    ft.scene.init();
    ft.post.init();

    const auto median = [](std::vector<double>& v) {
        if (v.empty())
            return 0.0;
        const std::size_t mid = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + std::ptrdiff_t(mid), v.end());
        return v[mid];
    };

    VX_INFO("bench: %dx%d, radius %d, shadows+ssao+vol+bloom+ssgi on, taa off", w, h, radius);
    double totShadow = 0, totScene = 0, totPost = 0, totFrame = 0;
    double totScaledFrame = 0;
    double totMoveShadow = 0, totMoveFrame = 0, totMovePeak = 0;

    for (const View& v : views) {
        camera.position = v.pos;
        camera.yawDeg = v.yaw;
        camera.pitchDeg = v.pitch;

        // settle completely: every claim below is about steady-state cost, not streaming
        for (int i = 0; i < 40000; ++i) {
            window.pollEvents();
            rc.renderer.beginFrame();
            world.update(jobs, rc.renderer, light, camera.position);
            rc.renderer.endFrame();
            if (i > 200 && !world.busy() && light.idle())
                break;
        }
        std::vector<vx::ChunkDrawInfo> di;
        std::vector<vx::ChunkDrawInfo> wi;
        world.collectDrawInfo(di, wi);
        rc.renderer.updateChunkMeta(di);
        rc.renderer.updateWaterMeta(wi);
        world.updateVisibilityIfNeeded(rc.renderer, camera.position, true);

        constexpr int kWarm = 20;
        constexpr int kFrames = 140;
        std::vector<double> sh, sc, po, fr;
        sh.reserve(kFrames);
        for (int i = 0; i < kWarm + kFrames; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            rc.renderer.beginFrame();
            renderFrame(rc, camera, fx, true, w, h, fbo, 0.6f, &ft);
            rc.renderer.endFrame();
            glFinish(); // benchmark only: we want the whole frame's GPU work accounted for
            const double cpuMs =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
            if (i >= kWarm) {
                sh.push_back(ft.shadow.milliseconds());
                sc.push_back(ft.scene.milliseconds());
                po.push_back(ft.post.milliseconds());
                fr.push_back(cpuMs);
            }
        }
        const double mSh = median(sh), mSc = median(sc), mPo = median(po), mFr = median(fr);
        totShadow += mSh;
        totScene += mSc;
        totPost += mPo;
        totFrame += mFr;
        // drawn, not loaded: what survives frustum + cave culling is what an occlusion
        // pass would have left to remove, so this is the ceiling on #6
        VX_INFO("bench %-8s | shadow %5.2f | scene %5.2f | post %5.2f | frame %6.2f ms | "
                "%d of %d chunks drawn, %lld quads",
                v.name, mSh, mSc, mPo, mFr, rc.renderer.gpuVisibleChunks(), int(di.size()),
                static_cast<long long>(world.stats().totalQuads));

        // same view again at 70% internal resolution — shows what render scale buys
        vx::FxToggles scaled = fx;
        scaled.renderScale = 0.7f;
        std::vector<double> sh2, sc2, po2, fr2;
        for (int i = 0; i < kWarm + kFrames; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            rc.renderer.beginFrame();
            renderFrame(rc, camera, scaled, true, w, h, fbo, 0.6f, &ft);
            rc.renderer.endFrame();
            glFinish();
            const double cpuMs =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
            if (i >= kWarm) {
                sh2.push_back(ft.shadow.milliseconds());
                sc2.push_back(ft.scene.milliseconds());
                po2.push_back(ft.post.milliseconds());
                fr2.push_back(cpuMs);
            }
        }
        VX_INFO("   @0.70x  | shadow %5.2f | scene %5.2f | post %5.2f | frame %6.2f ms",
                median(sh2), median(sc2), median(po2), median(fr2));
        totScaledFrame += median(fr2);

        // Same view with the camera walking forward at ~8 units/s. A stationary camera is
        // the best possible case for cascade caching, so measuring only that would be
        // dishonest — this is what the cost actually looks like while you play.
        const glm::dvec3 startPos = camera.position;
        // fast flight, not a stroll: cascade 2 only re-anchors after ~68 units of travel,
        // so a slow walk over 140 frames would never trigger one and the test would prove
        // nothing.
        const glm::dvec3 step = glm::dvec3(camera.forward()) * (40.0 / 60.0);
        std::vector<double> sh3, fr3;
        for (int i = 0; i < kWarm + kFrames; ++i) {
            camera.position = startPos + step * double(i);
            const auto t0 = std::chrono::steady_clock::now();
            rc.renderer.beginFrame();
            renderFrame(rc, camera, fx, true, w, h, fbo, 0.6f, &ft);
            rc.renderer.endFrame();
            glFinish();
            const double cpuMs =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
            if (i >= kWarm) {
                sh3.push_back(ft.shadow.milliseconds());
                fr3.push_back(cpuMs);
            }
        }
        camera.position = startPos;
        // MEAN and MAX here, not median: caching trades a steady per-frame cost for a
        // periodic redraw, and a median would simply hide the spike that costs.
        double meanSh = 0.0, maxSh = 0.0, maxFr = 0.0;
        for (double v : sh3) {
            meanSh += v;
            maxSh = std::max(maxSh, v);
        }
        meanSh /= double(sh3.size());
        for (double v : fr3)
            maxFr = std::max(maxFr, v);
        VX_INFO("   moving   | shadow mean %5.2f peak %5.2f | frame med %6.2f peak %6.2f ms",
                meanSh, maxSh, median(fr3), maxFr);
        totMoveShadow += meanSh;
        totMoveFrame += median(fr3);
        totMovePeak = std::max(totMovePeak, maxFr);
    }

    const double n = double(std::size(views));
    VX_INFO("bench TOTAL   | shadow %5.2f | scene %5.2f | post %5.2f | frame %6.2f ms (mean "
            "of %d views)",
            totShadow / n, totScene / n, totPost / n, totFrame / n, int(n));
    VX_INFO("bench MOVING  | shadow mean %5.2f | frame med %6.2f | worst frame %6.2f ms",
            totMoveShadow / n, totMoveFrame / n, totMovePeak);
    VX_INFO("bench TOTAL   | render scale 0.70x -> frame %6.2f ms (%.0f%% of native)",
            totScaledFrame / n, 100.0 * (totScaledFrame / n) / std::max(totFrame / n, 1e-6));

    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &color);
    glDeleteRenderbuffers(1, &depth);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    bool smoke = false;
    bool bench = false;
    int benchRadius = 16;
    bool tour = false;
    int tourRadius = 16;
    bool tourSettle = true;
    std::filesystem::path smokeOut = "smoke.bmp";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--smoke") == 0) {
            smoke = true;
            if (i + 1 < argc)
                smokeOut = argv[++i];
        } else if (std::strcmp(argv[i], "--tour") == 0) {
            tour = true;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                tourRadius = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--tourmoving") == 0) {
            tour = true;
            tourSettle = false;
        } else if (std::strcmp(argv[i], "--bench") == 0) {
            bench = true;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                benchRadius = std::atoi(argv[++i]);
        }
    }

#ifndef NDEBUG
    const bool debugContext = true;
#else
    const bool debugContext = smoke; // smoke runs want loud GL errors even in Release
#endif

    auto window = vx::Window::create({
        .width = 1280,
        .height = 720,
        .title = "Voxelium",
        .visible = !smoke && !bench && !tour,
        .debugContext = debugContext,
    });
    if (!window)
        return 1;

    vx::installGlDebugCallback();
    VX_INFO("GL %s | %s | %s",
            reinterpret_cast<const char*>(glGetString(GL_VERSION)),
            reinterpret_cast<const char*>(glGetString(GL_RENDERER)),
            reinterpret_cast<const char*>(glGetString(GL_VENDOR)));

    // Reversed-Z depth, set up once (rationale in render/Camera.hpp).
    glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);
    glClearDepth(0.0);
    glEnable(GL_CULL_FACE);

    const std::filesystem::path assets = vx::findAssetsDir(argv[0]);
    if (assets.empty()) {
        VX_ERROR("could not locate assets/ directory");
        return 1;
    }
    const std::filesystem::path shaders = assets / "shaders";
    // shared across runs including --smoke: this caches driver output keyed by source and
    // GL identity, not world state, so there is nothing here for a test run to poison
    vx::initShaderCache("saves/shadercache");
    vx::ProgramJob worldJobs[] = {
        vx::submitProgram(shaders / "chunk.vert", shaders / "chunk.frag"),
        vx::submitComputeProgram(shaders / "cull.comp"),
        vx::submitProgram(shaders / "shadow.vert", shaders / "shadow.frag"),
        vx::submitProgram(shaders / "water.vert", shaders / "water.frag"),
    };
    const GLuint drawProgram = vx::resolveProgram(worldJobs[0]);
    const GLuint cullProgram = vx::resolveProgram(worldJobs[1]);
    const GLuint shadowProgram = vx::resolveProgram(worldJobs[2]);
    const GLuint waterProgram = vx::resolveProgram(worldJobs[3]);
    if (!drawProgram || !cullProgram || !shadowProgram || !waterProgram)
        return 1;

    if (!runMesherOracle())
        return 1;

    vx::Renderer renderer;
    if (!renderer.init(drawProgram, cullProgram))
        return 1;
    vx::ShadowMap shadowMap;
    if (!shadowMap.init(shadowProgram))
        return 1;
    vx::PostFx postfx;
    if (!postfx.init(shaders))
        return 1;
    vx::Overlay overlay;
    if (!overlay.init(shaders))
        return 1;
    vx::shutdownShaderCache(); // every program is built by here
    const GLuint blockTex = vx::createBlockTextureArray(assets / "textures");
    RenderContext rc{renderer, shadowMap, postfx, overlay, drawProgram, waterProgram, blockTex};

    // isolate the smoke test's save data so repeated runs start clean and can't touch a
    // real world save
    std::filesystem::path saveDir = "saves/world";
    if (smoke) {
        saveDir = smokeOut.parent_path() / "vxsave_smoke";
        std::error_code ec;
        std::filesystem::remove_all(saveDir, ec);
    }

    // declaration order matters: jobs is destroyed first, joining all workers, so no
    // job can touch world's chunks or queues during teardown
    auto world = std::make_unique<vx::ChunkManager>(saveDir);
    auto light = std::make_unique<vx::LightEngine>(*world);
    auto jobs = std::make_unique<vx::JobSystem>();
    VX_INFO("job system: %d workers", jobs->workerCount());

    vx::FlyCamera camera;
    camera.position = {16.0, 110.0, 16.0};
    camera.yawDeg = -35.0f;
    camera.pitchDeg = -25.0f;

    vx::DebugHud hud;
    if (!hud.init(window->handle()))
        return 1;
    window->hookScroll();

    if (tour) {
        const int rcT = runTour(rc, *world, *jobs, *light, camera, *window, tourRadius,
                                tourSettle);
        world->saveDirty();
        return rcT;
    }
    if (bench) {
        const int rcB = runBenchmark(rc, *world, *jobs, *light, camera, *window, benchRadius);
        hud.shutdown();
        overlay.shutdown();
        postfx.shutdown();
        shadowMap.shutdown();
        renderer.shutdown();
        return rcB;
    }

    if (smoke) {
        const int rc2 = runSmokeTest(rc, *world, *jobs, *light, camera, hud, *window, smokeOut);
        hud.shutdown();
        overlay.shutdown();
        postfx.shutdown();
        shadowMap.shutdown();
        renderer.shutdown();
        return rc2;
    }

    FrameTimers timers;
    timers.shadow.init();
    timers.scene.init();
    timers.post.init();

    std::vector<vx::ChunkDrawInfo> drawInfo;
    std::vector<vx::ChunkDrawInfo> waterInfo;
    vx::Player player;
    vx::FxToggles fx;
    fx.taa = true; // temporal AA on by default in the live app
    TaaState taaState;
    double prevTime = glfwGetTime();
    double physAccum = 0.0;
    bool f3Held = false;
    bool f2Held = false;
    bool f5Held = false;
    bool lmbHeld = false;
    bool mmbHeld = false;
    bool walkMode = false;
    bool vsync = false;
    bool caveCull = true;
    // sounds/ next to the assets dir; nothing ships in it, the user fills it themselves
    vx::AudioEngine audio;
    audio.init(assets.parent_path() / "sounds");
    double stepAccum = 0.0; // distance walked since the last footstep
    // live app only — a smoke run renders from fixed defaults, never a machine's saved
    // settings, or the reference image would move under it
    const std::filesystem::path settingsFile = "saves/settings.cfg";
    if (vx::Settings::load(settingsFile, fx, vsync, caveCull, world->radius, camera.fovYDeg))
        VX_INFO("settings: loaded %s", settingsFile.string().c_str());
    window->setVsync(vsync);
    bool loadLogged = false;
    bool everBusy = false; // don't log the "loaded" summary before streaming even starts
    bool f4Held = false;
    bool hotbarOpen = false;
    bool escHeld = false;
    bool settingsOpen = false;
    constexpr int kHotbar[] = {vx::blocks::Stone,  vx::blocks::Dirt,   vx::blocks::Grass,
                               vx::blocks::Sand,   vx::blocks::Snow,   vx::blocks::Wood,
                               vx::blocks::Planks, vx::blocks::Bricks, vx::blocks::Leaves,
                               vx::blocks::Glow,   vx::blocks::Water};
    constexpr int kHotbarCount = int(sizeof(kHotbar) / sizeof(kHotbar[0]));
    int hotbarSel = 0;
    int placeBlock = kHotbar[hotbarSel];
    std::uint64_t allocsPrevFrame = vx::heapAllocCount();

    while (!window->shouldClose()) {
        window->pollEvents();
        const bool esc = window->keyDown(GLFW_KEY_ESCAPE);
        if (esc && !escHeld) {
            settingsOpen = !settingsOpen; // ESC pauses into the settings menu, not quit
            if (settingsOpen)
                window->setCursorCaptured(false);
            else // closing the menu commits whatever was just changed
                vx::Settings::save(settingsFile, fx, vsync, caveCull, world->radius,
                                   camera.fovYDeg);
        }
        escHeld = esc;
        const bool f3 = window->keyDown(GLFW_KEY_F3);
        if (f3 && !f3Held)
            hud.toggle();
        f3Held = f3;
        const bool f2 = window->keyDown(GLFW_KEY_F2);
        if (f2 && !f2Held)
            world->saveDirty(); // save edited chunks to disk
        f2Held = f2;
        const bool f5 = window->keyDown(GLFW_KEY_F5);
        if (f5 && !f5Held) {
            walkMode = !walkMode;
            if (walkMode) {
                player.position = camera.position - glm::dvec3(0.0, vx::Player::kEyeHeight, 0.0);
                player.velocity = {0.0, 0.0, 0.0};
                physAccum = 0.0;
            }
        }
        f5Held = f5;
        const bool f4 = window->keyDown(GLFW_KEY_F4);
        if (f4 && !f4Held)
            hotbarOpen = !hotbarOpen;
        f4Held = f4;
        if (hotbarOpen && !settingsOpen) {
            for (int k = 0; k < kHotbarCount; ++k)
                if (window->keyDown(GLFW_KEY_1 + k))
                    hotbarSel = k;
            // scroll cycles slots while the hotbar is up (drained here so it doesn't also
            // change fly speed); when it's hidden the camera gets the scroll for speed
            const double sc = window->consumeScrollY();
            if (sc != 0.0)
                hotbarSel = (hotbarSel - (sc > 0.0 ? 1 : -1) + kHotbarCount) % kHotbarCount;
        }
        placeBlock = kHotbar[hotbarSel];

        const double now = glfwGetTime();
        const float frameMs = float((now - prevTime) * 1000.0);
        const float dt = std::min(float(now - prevTime), 0.1f);
        prevTime = now;
        if (fx.dayCycle && !settingsOpen) { // advance the day, wrap to [0,1)
            fx.timeOfDay += dt / std::max(fx.dayLengthSec, 1.0f);
            fx.timeOfDay -= std::floor(fx.timeOfDay);
        }

        renderer.beginFrame();
        hud.beginFrame();
        if (settingsOpen) {
            // paused in the settings menu: freeze look/move/edit, keep streaming + drawing
        } else if (walkMode) {
            camera.lookOnly(*window);
            physAccum += double(dt);
            while (physAccum >= vx::Player::kTickDt) {
                vx::PlayerInput in;
                in.yawDeg = camera.yawDeg;
                if (window->keyDown(GLFW_KEY_W)) in.forward += 1.0f;
                if (window->keyDown(GLFW_KEY_S)) in.forward -= 1.0f;
                if (window->keyDown(GLFW_KEY_D)) in.strafe += 1.0f;
                if (window->keyDown(GLFW_KEY_A)) in.strafe -= 1.0f;
                in.jump = window->keyDown(GLFW_KEY_SPACE);
                const glm::dvec3 before = player.position;
                player.step(*world, in);
                physAccum -= vx::Player::kTickDt;
                // footsteps are paced by distance, not time, so they don't machine-gun
                // when you sprint or stall when you walk into a wall
                if (player.onGround) {
                    const glm::dvec3 d = player.position - before;
                    stepAccum += std::sqrt(d.x * d.x + d.z * d.z);
                    if (stepAccum > 2.2) {
                        stepAccum = 0.0;
                        const glm::ivec3 under(glm::floor(player.position.x),
                                               glm::floor(player.position.y) - 1,
                                               glm::floor(player.position.z));
                        const int id = world->blockAtWorld(under);
                        if (!audio.playIfPresent("step_" + std::to_string(id),
                                                 player.position, 0.5f))
                            audio.playAt("step", player.position, 0.5f);
                    }
                } else {
                    stepAccum = 0.0;
                }
            }
            camera.position = player.eye();
        } else {
            camera.update(*window, dt, hud.wantsMouse(), hud.wantsKeyboard());
        }

        if (!settingsOpen && camera.flying()) {
            const bool lmb = window->mouseDown(GLFW_MOUSE_BUTTON_LEFT);
            const bool mmb = window->mouseDown(GLFW_MOUSE_BUTTON_MIDDLE);
            if ((lmb && !lmbHeld) || (mmb && !mmbHeld)) {
                const auto hit = vx::raycastVoxels(
                    camera.position, camera.forward(), 6.0, [&](const glm::ivec3& p) {
                        return vx::blocks::opaque(world->blockAtWorld(p));
                    });
                if (hit.hit) {
                    // sound names follow the block id so a user can be as specific as they
                    // like: break_3 for grass, or just break for everything
                    const glm::dvec3 at(hit.voxel);
                    if (lmb && !lmbHeld) {
                        const int id = world->blockAtWorld(hit.voxel);
                        world->setBlock(renderer, *light, hit.voxel, vx::blocks::Air);
                        if (!audio.playIfPresent("break_" + std::to_string(id), at))
                            audio.playAt("break", at);
                    } else {
                        world->setBlock(renderer, *light, hit.voxel + hit.normal,
                                        vx::BlockId(placeBlock));
                        if (!audio.playIfPresent("place_" + std::to_string(placeBlock), at))
                            audio.playAt("place", at);
                    }
                }
            }
            lmbHeld = lmb;
            mmbHeld = mmb;
        }

        audio.setListener(camera.position, camera.forward());
        audio.setMasterVolume(fx.volume);

        world->update(*jobs, renderer, *light, camera.position);
        if (world->consumeDirty()) {
            world->collectDrawInfo(drawInfo, waterInfo);
            renderer.updateChunkMeta(drawInfo);
            renderer.updateWaterMeta(waterInfo);
            // geometry changed, so cached cascades are holding stale casters
            shadowMap.invalidate();
        }
        world->updateVisibilityIfNeeded(renderer, camera.position, caveCull);
        everBusy = everBusy || world->busy();
        if (!loadLogged && everBusy && !world->busy()) {
            world->logSummary();
            loadLogged = true;
        }

        // block the crosshair is pointing at -> selection wireframe (camera-relative)
        bool hasSel = false;
        glm::vec3 selMin(0.0f);
        if (!settingsOpen) {
            const auto sel = vx::raycastVoxels(
                camera.position, camera.forward(), 6.0, [&](const glm::ivec3& p) {
                    return vx::blocks::opaque(world->blockAtWorld(p));
                });
            if (sel.hit) {
                hasSel = true;
                selMin = glm::vec3(glm::dvec3(sel.voxel) - camera.position);
            }
        }

        // eye submerged -> underwater fog/tint in the composite pass
        const bool eyeUnderwater =
            vx::blocks::isWater(world->blockAtWorld(glm::ivec3(glm::floor(camera.position))));

        int fbW = 0;
        int fbH = 0;
        window->framebufferSize(fbW, fbH);
        if (fbW > 0 && fbH > 0) {
            // fog the outer ~8% of the loaded radius so the world edge reads as haze
            const float fogEnd =
                fx.fog ? float(world->radius) * float(vx::kChunkSize) * 0.92f : 1e9f;
            renderFrame(rc, camera, fx, caveCull, fbW, fbH, 0, float(now), &timers, hasSel,
                        selMin, eyeUnderwater, &taaState, fogEnd);
            if (hotbarOpen)
                overlay.drawHotbar(fbW, fbH, kHotbar, kHotbarCount, hotbarSel, blockTex);
        }

        const vx::ChunkManager::Stats ws = world->stats();
        const vx::Renderer::ArenaStats as = renderer.arenaStats();
        const std::uint64_t allocsNow = vx::heapAllocCount();
        const vx::FrameStats stats{
            .frameMs = frameMs,
            .gpuMs = timers.scene.milliseconds(),
            .heapAllocs = allocsNow - allocsPrevFrame,
            .drawCalls = 1,
            .quads = int(ws.totalQuads),
            .triangles = int(ws.totalQuads) * 2,
            .chunksLoaded = ws.loaded,
            .chunksMeshed = ws.meshed,
            .genQueue = ws.genQueue,
            .jobsInFlight = ws.jobsInFlight,
            .uploadsPending = ws.pendingUploads,
            .uploadMs = float(ws.uploadMsLast),
            .voxelBytes = ws.voxelBytes,
            .quadBytes = ws.quadBytes,
            .gpuChunksDrawn = renderer.gpuVisibleChunks(),
            .arenaUsedQuads = as.usedQuads,
            .arenaCapQuads = as.capacityQuads,
            .arenaFreeBlocks = as.freeBlocks,
            .walkMode = walkMode,
            .placeBlock = placeBlock,
            .gpuShadowMs = timers.shadow.milliseconds(),
            .gpuSceneMs = timers.scene.milliseconds(),
            .gpuPostMs = timers.post.milliseconds(),
        };
        allocsPrevFrame = allocsNow;
        const bool vsyncBefore = vsync;
        bool quitRequested = false;
        hud.diagnostics(stats, camera);
        hud.settings(settingsOpen, vsync, caveCull, world->radius, camera.fovYDeg, fx,
                     quitRequested);
        if (vsync != vsyncBefore)
            window->setVsync(vsync);
        hud.render();
        if (quitRequested)
            window->requestClose();
        renderer.endFrame();

        window->swapBuffers();
    }

    VX_INFO("window closed, shutting down");
    vx::Settings::save(settingsFile, fx, vsync, caveCull, world->radius, camera.fovYDeg);
    audio.shutdown();
    world->saveDirty(); // persist edits before teardown (workers join after this)
    hud.shutdown();
    overlay.shutdown();
    postfx.shutdown();
    shadowMap.shutdown();
    renderer.shutdown();
    return 0;
}
