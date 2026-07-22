#include "core/AllocTracker.hpp"
#include "core/Bmp.hpp"
#include "core/Log.hpp"
#include "core/Paths.hpp"
#include "gfx/BlockTextures.hpp"
#include "gfx/GlDebug.hpp"
#include "gfx/GpuTimer.hpp"
#include "gfx/Shader.hpp"
#include "io/ChunkSerializer.hpp"
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

CameraFrame buildCameraFrame(const vx::FlyCamera& camera, int fbWidth, int fbHeight) {
    CameraFrame f;
    f.viewProj = camera.projReversedZ(float(fbWidth) / float(fbHeight)) * camera.viewAtOrigin();
    f.frustum = vx::Frustum::fromViewProj(f.viewProj);
    f.cameraChunk = glm::ivec3(glm::floor(camera.position / double(vx::kChunkSize)));
    f.cameraFrac = glm::vec3(camera.position - glm::dvec3(f.cameraChunk) * double(vx::kChunkSize));
    return f;
}

glm::vec3 sunFromAltitude(float deg) {
    const float a = glm::radians(deg);
    return glm::normalize(glm::vec3(0.55f * std::cos(a), std::sin(a), 0.35f * std::cos(a)));
}

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
                 const glm::vec3& selMinCamRel = glm::vec3(0.0f)) {
    const CameraFrame cam = buildCameraFrame(camera, fbW, fbH);
    const glm::vec3 sunL = sunFromAltitude(fx.sunAltitudeDeg);
    const float aspect = float(fbW) / float(fbH);

    rc.postfx.resize(fbW, fbH);

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

    rc.renderer.drawWorld(cam.viewProj, cam.frustum, cam.cameraChunk, cam.cameraFrac,
                          caveCull);

    // transparent water over the opaque scene: alpha blend, depth test but no write.
    // snapshot the opaque colour first so the water shader can reflect it (SSR).
    rc.postfx.captureOpaqueForWater();
    const float wf = 1.0f / std::tan(glm::radians(camera.fovYDeg) * 0.5f);
    const glm::vec2 projScale{aspect / wf, 1.0f / wf};
    glProgramUniform1f(rc.waterProgram, 4, time);
    glProgramUniform3fv(rc.waterProgram, 5, 1, &sunL.x);
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
    rp.shadows = &rc.shadows;
    rp.toggles = fx;
    rp.targetFbo = targetFbo;
    rp.targetWidth = fbW;
    rp.targetHeight = fbH;
    rc.postfx.run(rp);
    if (timers) timers->post.end();

    // gameplay HUD on top of the final tonemapped image
    glBindFramebuffer(GL_FRAMEBUFFER, targetFbo);
    glViewport(0, 0, fbW, fbH);
    rc.overlay.drawCrosshair(fbW, fbH);
}

int runSmokeTest(RenderContext& rc, vx::ChunkManager& world, vx::JobSystem& jobs,
                 vx::LightEngine& light, vx::FlyCamera& camera, vx::DebugHud& hud,
                 vx::Window& window, const std::filesystem::path& outPath) {
    const auto drain = [&] {
        for (int i = 0; i < 20000 && (world.busy() || !light.idle() || i == 0); ++i) {
            rc.renderer.beginFrame();
            world.update(jobs, rc.renderer, light, camera.position);
            rc.renderer.endFrame();
        }
    };

    drain();
    world.logSummary();

    // lighting checks against the enclosed glow room in the terrain
    {
        const int skyInRoom = light.lightAt({16, 48, 16}) >> 4;
        const int glowAbove = light.lightAt({16, 46, 16}) & 15;
        const int glowEdge = light.lightAt({16, 48, 19}) & 15;
        // open air near the top of the world — always full sky (trees/terrain can't reach)
        const int skySurface = light.lightAt({16, vx::kWorldHeight - 4, 16}) >> 4;
        bool ok = skyInRoom == 0 && glowAbove == 13 && glowEdge > 0 && glowEdge < 13 &&
                  skySurface == 15;
        VX_INFO("light: room sky %d (want 0), above-glow %d (want 13), edge %d, surface sky "
                "%d (want 15) -> %s",
                skyInRoom, glowAbove, glowEdge, skySurface, ok ? "PASS" : "FAIL");
        if (!ok)
            return 1;

        vx::Chunk* roomChunk = world.chunkDataMutable({0, 1, 0});
        roomChunk->set(16, 13, 16, vx::blocks::Air);
        light.onBlockChanged({16, 45, 16}, vx::blocks::Glow, vx::blocks::Air);
        while (!light.idle())
            light.process(10.0);
        const int afterRemove = light.lightAt({16, 46, 16}) & 15;
        roomChunk->set(16, 13, 16, vx::blocks::Glow);
        light.onBlockChanged({16, 45, 16}, vx::blocks::Air, vx::blocks::Glow);
        while (!light.idle())
            light.process(10.0);
        const int afterRestore = light.lightAt({16, 46, 16}) & 15;
        ok = afterRemove == 0 && afterRestore == 13;
        VX_INFO("light: removal %d (want 0), re-add %d (want 13) -> %s", afterRemove,
                afterRestore, ok ? "PASS" : "FAIL");
        if (!ok)
            return 1;
        drain();
    }

    // hitch check: teleport 16 chunks east and re-stream the world
    camera.position.x += 512.0;
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
    VX_INFO("stress: re-streamed after teleport in %d updates, worst main-thread cost %.2f ms",
            frames, worstMs);
    camera.position.x -= 512.0;
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

        const glm::ivec3 ep{18, world.generator().surfaceHeight(18, 18) + 6, 18};
        world.setBlock(rc.renderer, light, ep, vx::blocks::Stone);
        const bool placed = world.blockAtWorld(ep) == vx::blocks::Stone;
        world.setBlock(rc.renderer, light, ep, vx::blocks::Air);
        const bool broken = world.blockAtWorld(ep) == vx::blocks::Air;
        VX_INFO("edits: place %s, break %s", placed ? "PASS" : "FAIL",
                broken ? "PASS" : "FAIL");
        if (!placed || !broken)
            return 1;
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

} // namespace

int main(int argc, char** argv) {
    bool smoke = false;
    std::filesystem::path smokeOut = "smoke.bmp";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--smoke") == 0) {
            smoke = true;
            if (i + 1 < argc)
                smokeOut = argv[++i];
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
        .visible = !smoke,
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
    const GLuint drawProgram = vx::loadProgram(shaders / "chunk.vert", shaders / "chunk.frag");
    const GLuint cullProgram = vx::loadComputeProgram(shaders / "cull.comp");
    const GLuint shadowProgram =
        vx::loadProgram(shaders / "shadow.vert", shaders / "shadow.frag");
    const GLuint waterProgram = vx::loadProgram(shaders / "water.vert", shaders / "water.frag");
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
    bool loadLogged = false;
    bool everBusy = false; // don't log the "loaded" summary before streaming even starts
    bool e_Held = false;
    bool inventoryOpen = false;
    bool escHeld = false;
    bool settingsOpen = false;
    int placeBlock = vx::blocks::Stone;
    std::uint64_t allocsPrevFrame = vx::heapAllocCount();

    while (!window->shouldClose()) {
        window->pollEvents();
        const bool esc = window->keyDown(GLFW_KEY_ESCAPE);
        if (esc && !escHeld) {
            settingsOpen = !settingsOpen; // ESC pauses into the settings menu, not quit
            if (settingsOpen)
                window->setCursorCaptured(false);
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
        const bool eKey = window->keyDown(GLFW_KEY_E);
        if (eKey && !e_Held)
            inventoryOpen = !inventoryOpen;
        e_Held = eKey;
        if (window->keyDown(GLFW_KEY_1)) placeBlock = vx::blocks::Stone;
        if (window->keyDown(GLFW_KEY_2)) placeBlock = vx::blocks::Dirt;
        if (window->keyDown(GLFW_KEY_3)) placeBlock = vx::blocks::Glow;

        const double now = glfwGetTime();
        const float frameMs = float((now - prevTime) * 1000.0);
        const float dt = std::min(float(now - prevTime), 0.1f);
        prevTime = now;

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
                player.step(*world, in);
                physAccum -= vx::Player::kTickDt;
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
                    if (lmb && !lmbHeld)
                        world->setBlock(renderer, *light, hit.voxel, vx::blocks::Air);
                    else
                        world->setBlock(renderer, *light, hit.voxel + hit.normal,
                                        vx::BlockId(placeBlock));
                }
            }
            lmbHeld = lmb;
            mmbHeld = mmb;
        }

        world->update(*jobs, renderer, *light, camera.position);
        if (world->consumeDirty()) {
            world->collectDrawInfo(drawInfo, waterInfo);
            renderer.updateChunkMeta(drawInfo);
            renderer.updateWaterMeta(waterInfo);
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

        int fbW = 0;
        int fbH = 0;
        window->framebufferSize(fbW, fbH);
        if (fbW > 0 && fbH > 0)
            renderFrame(rc, camera, fx, caveCull, fbW, fbH, 0, float(now), &timers, hasSel,
                        selMin);

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
        hud.blockPicker(placeBlock, inventoryOpen);
        if (vsync != vsyncBefore)
            window->setVsync(vsync);
        hud.render();
        if (quitRequested)
            window->requestClose();
        renderer.endFrame();

        window->swapBuffers();
    }

    VX_INFO("window closed, shutting down");
    world->saveDirty(); // persist edits before teardown (workers join after this)
    hud.shutdown();
    overlay.shutdown();
    postfx.shutdown();
    shadowMap.shutdown();
    renderer.shutdown();
    return 0;
}
