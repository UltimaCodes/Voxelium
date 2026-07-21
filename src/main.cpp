#include "core/AllocTracker.hpp"
#include "core/Bmp.hpp"
#include "core/Log.hpp"
#include "core/Paths.hpp"
#include "gfx/GlDebug.hpp"
#include "gfx/GpuTimer.hpp"
#include "gfx/Shader.hpp"
#include "jobs/JobSystem.hpp"
#include "mesh/GreedyMesher.hpp"
#include "mesh/NaiveMesher.hpp"
#include "platform/Window.hpp"
#include "render/Camera.hpp"
#include "render/Frustum.hpp"
#include "render/Renderer.hpp"
#include "ui/DebugHud.hpp"
#include "world/Chunk.hpp"
#include "world/ChunkManager.hpp"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <vector>

namespace {

constexpr float kClearColor[3] = {0.35f, 0.58f, 0.85f};

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
    vx::GreedyStats greedy;
    const double greedyBest =
        benchBestMs(50, [&] { greedy = mesher->mesh(*chunk, nullptr, quads); });

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

void beginFrameGl(int fbWidth, int fbHeight) {
    glViewport(0, 0, fbWidth, fbHeight);
    glClearColor(kClearColor[0], kClearColor[1], kClearColor[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
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

// Renders the streamed world offscreen after draining the load queues, verifies pixels
// were produced, checks the GPU cull count, writes a BMP.
int runSmokeTest(vx::Renderer& renderer, vx::ChunkManager& world, vx::JobSystem& jobs,
                 vx::FlyCamera& camera, vx::DebugHud& hud, vx::Window& window,
                 const std::filesystem::path& outPath) {
    const auto drain = [&] {
        for (int i = 0; i < 20000 && (world.busy() || i == 0); ++i) {
            renderer.beginFrame();
            world.update(jobs, renderer, camera.position);
            renderer.endFrame();
        }
    };

    drain();
    world.logSummary();

    // hitch check: teleport 16 chunks east and re-stream the world, recording the worst
    // single main-thread update() cost. workers absorb gen+mesh; the budget caps uploads.
    camera.position.x += 512.0;
    double worstMs = 0.0;
    int frames = 0;
    for (; frames < 20000; ++frames) {
        renderer.beginFrame();
        const auto t0 = std::chrono::steady_clock::now();
        world.update(jobs, renderer, camera.position);
        worstMs = std::max(
            worstMs,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count());
        renderer.endFrame();
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
    GLuint depth = 0;
    glCreateRenderbuffers(1, &color);
    glNamedRenderbufferStorage(color, GL_RGB8, w, h);
    glCreateRenderbuffers(1, &depth);
    glNamedRenderbufferStorage(depth, GL_DEPTH_COMPONENT32F, w, h);
    GLuint fbo = 0;
    glCreateFramebuffers(1, &fbo);
    glNamedFramebufferRenderbuffer(fbo, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color);
    glNamedFramebufferRenderbuffer(fbo, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
    if (glCheckNamedFramebufferStatus(fbo, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        VX_ERROR("smoke: framebuffer incomplete");
        return 1;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    std::vector<vx::ChunkDrawInfo> drawInfo;
    world.collectDrawInfo(drawInfo);
    renderer.updateChunkMeta(drawInfo);
    const CameraFrame cam = buildCameraFrame(camera, w, h);

    const vx::ChunkManager::Stats ws = world.stats();
    const vx::Renderer::ArenaStats as = renderer.arenaStats();
    bool vsyncDummy = false;
    int radiusDummy = world.radius;
    for (int i = 0; i < 5; ++i) {
        window.pollEvents();
        renderer.beginFrame();
        hud.beginFrame();
        beginFrameGl(w, h);
        renderer.drawWorld(cam.viewProj, cam.frustum, cam.cameraChunk, cam.cameraFrac);
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
                                   .gpuChunksDrawn = renderer.gpuVisibleChunks(),
                                   .arenaUsedQuads = as.usedQuads,
                                   .arenaCapQuads = as.capacityQuads,
                                   .arenaFreeBlocks = as.freeBlocks};
        hud.buildUi(stats, camera, vsyncDummy, radiusDummy);
        hud.render();
        renderer.endFrame();
    }
    VX_INFO("smoke: gpu culled %d of %d chunks (1 draw call)", renderer.debugReadDrawCount(),
            int(drawInfo.size()));

    std::vector<std::uint8_t> pixels(std::size_t(w) * std::size_t(h) * 3u);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glNamedFramebufferReadBuffer(fbo, GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, pixels.data());

    const std::uint8_t clearBgr[3] = {
        std::uint8_t(kClearColor[2] * 255.0f + 0.5f),
        std::uint8_t(kClearColor[1] * 255.0f + 0.5f),
        std::uint8_t(kClearColor[0] * 255.0f + 0.5f),
    };
    std::size_t drawn = 0;
    for (std::size_t i = 0; i < pixels.size(); i += 3) {
        if (std::abs(int(pixels[i + 0]) - int(clearBgr[0])) > 8 ||
            std::abs(int(pixels[i + 1]) - int(clearBgr[1])) > 8 ||
            std::abs(int(pixels[i + 2]) - int(clearBgr[2])) > 8)
            ++drawn;
    }

    if (!vx::writeBmp24(outPath, w, h, pixels))
        return 1;

    VX_INFO("smoke: %llu of %d pixels differ from clear color (%.1f%%), wrote %s",
            static_cast<unsigned long long>(drawn), w * h,
            100.0 * double(drawn) / (double(w) * double(h)), outPath.string().c_str());
    if (drawn < 1000) {
        VX_ERROR("smoke: nothing rendered");
        return 1;
    }
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
    const GLuint drawProgram =
        vx::loadProgram(assets / "shaders/chunk.vert", assets / "shaders/chunk.frag");
    const GLuint cullProgram = vx::loadComputeProgram(assets / "shaders/cull.comp");
    if (!drawProgram || !cullProgram)
        return 1;

    if (!runMesherOracle())
        return 1;

    vx::Renderer renderer;
    if (!renderer.init(drawProgram, cullProgram))
        return 1;

    // declaration order matters: jobs is destroyed first, joining all workers, so no
    // job can touch world's chunks or queues during teardown
    auto world = std::make_unique<vx::ChunkManager>();
    auto jobs = std::make_unique<vx::JobSystem>();
    VX_INFO("job system: %d workers", jobs->workerCount());

    vx::FlyCamera camera;
    camera.position = {16.0, 104.0, 16.0};
    camera.yawDeg = -35.0f;
    camera.pitchDeg = -25.0f;

    vx::DebugHud hud;
    if (!hud.init(window->handle()))
        return 1;
    window->hookScroll();

    if (smoke) {
        const int rc =
            runSmokeTest(renderer, *world, *jobs, camera, hud, *window, smokeOut);
        hud.shutdown();
        renderer.shutdown();
        return rc;
    }

    vx::GpuTimer gpuTimer;
    gpuTimer.init();

    std::vector<vx::ChunkDrawInfo> drawInfo;
    double prevTime = glfwGetTime();
    bool f3Held = false;
    bool vsync = false;
    bool loadLogged = false;
    std::uint64_t allocsPrevFrame = vx::heapAllocCount();

    while (!window->shouldClose()) {
        window->pollEvents();
        if (window->keyDown(GLFW_KEY_ESCAPE)) {
            VX_INFO("esc pressed — closing");
            window->requestClose();
        }
        const bool f3 = window->keyDown(GLFW_KEY_F3);
        if (f3 && !f3Held)
            hud.toggle();
        f3Held = f3;

        const double now = glfwGetTime();
        const float frameMs = float((now - prevTime) * 1000.0);
        const float dt = std::min(float(now - prevTime), 0.1f);
        prevTime = now;

        renderer.beginFrame();
        hud.beginFrame();
        camera.update(*window, dt, hud.wantsMouse(), hud.wantsKeyboard());

        world->update(*jobs, renderer, camera.position);
        if (world->consumeDirty()) {
            world->collectDrawInfo(drawInfo);
            renderer.updateChunkMeta(drawInfo);
        }
        if (!loadLogged && !world->busy()) {
            world->logSummary();
            loadLogged = true;
        }

        int fbW = 0;
        int fbH = 0;
        window->framebufferSize(fbW, fbH);
        gpuTimer.begin();
        if (fbW > 0 && fbH > 0) {
            const CameraFrame cam = buildCameraFrame(camera, fbW, fbH);
            beginFrameGl(fbW, fbH);
            renderer.drawWorld(cam.viewProj, cam.frustum, cam.cameraChunk, cam.cameraFrac);
        }

        const vx::ChunkManager::Stats ws = world->stats();
        const vx::Renderer::ArenaStats as = renderer.arenaStats();
        const std::uint64_t allocsNow = vx::heapAllocCount();
        const vx::FrameStats stats{
            .frameMs = frameMs,
            .gpuMs = gpuTimer.milliseconds(),
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
        };
        allocsPrevFrame = allocsNow;
        const bool vsyncBefore = vsync;
        hud.buildUi(stats, camera, vsync, world->radius);
        if (vsync != vsyncBefore)
            window->setVsync(vsync);
        hud.render();
        gpuTimer.end();
        renderer.endFrame();

        window->swapBuffers();
    }

    VX_INFO("window closed, shutting down");
    hud.shutdown();
    renderer.shutdown();
    return 0;
}
