#include "core/AllocTracker.hpp"
#include "core/Bmp.hpp"
#include "core/Log.hpp"
#include "core/Paths.hpp"
#include "gfx/GlDebug.hpp"
#include "gfx/GpuTimer.hpp"
#include "gfx/Shader.hpp"
#include "mesh/GreedyMesher.hpp"
#include "mesh/NaiveMesher.hpp"
#include "platform/Window.hpp"
#include "render/Camera.hpp"
#include "ui/DebugHud.hpp"
#include "world/Chunk.hpp"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace {

constexpr float kClearColor[3] = {0.35f, 0.58f, 0.85f}; // sky-ish, terrain reads better

// A chunk's renderable mesh: packed quads in an SSBO, an empty VAO (core profile wants
// one bound even with zero attributes), and nothing else. No vertex buffer exists.
struct QuadMesh {
    GLuint vao = 0;
    GLuint ssbo = 0;
    int quadCount = 0;
};

QuadMesh uploadQuads(std::span<const std::uint64_t> quads) {
    QuadMesh mesh;
    mesh.quadCount = int(quads.size());
    glCreateBuffers(1, &mesh.ssbo);
    glNamedBufferStorage(mesh.ssbo, GLsizeiptr(quads.size_bytes()), quads.data(), 0);
    glCreateVertexArrays(1, &mesh.vao);
    return mesh;
}

void drawScene(GLuint program, const QuadMesh& mesh, const glm::mat4& viewProj, int fbWidth,
               int fbHeight) {
    glViewport(0, 0, fbWidth, fbHeight);
    glClearColor(kClearColor[0], kClearColor[1], kClearColor[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(program);
    glProgramUniformMatrix4fv(program, 0, 1, GL_FALSE, glm::value_ptr(viewProj));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mesh.ssbo);
    glBindVertexArray(mesh.vao);
    glDrawArrays(GL_TRIANGLES, 0, mesh.quadCount * 6);
}

// Renders a few frames (scene + HUD) into an offscreen FBO, verifies pixels were
// produced, writes a BMP, exits 0/1. Offscreen because reading back a hidden window's
// default framebuffer is undefined (pixel-ownership test); an FBO always owns its pixels.
int runSmokeTest(GLuint program, const QuadMesh& mesh, const vx::FlyCamera& camera,
                 vx::DebugHud& hud, vx::Window& window, const std::filesystem::path& outPath) {
    int w = 0;
    int h = 0;
    window.framebufferSize(w, h); // match the window so ImGui's layout maps 1:1

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
    const glm::mat4 vp = camera.projReversedZ(float(w) / float(h)) * camera.view();
    bool vsyncDummy = false;
    for (int i = 0; i < 5; ++i) {
        window.pollEvents();
        hud.beginFrame();
        drawScene(program, mesh, vp, w, h);
        const vx::FrameStats stats{.frameMs = 1.0f,
                                   .gpuMs = 0.5f,
                                   .heapAllocs = 0,
                                   .drawCalls = 1,
                                   .quads = mesh.quadCount,
                                   .triangles = mesh.quadCount * 2};
        hud.buildUi(stats, camera, vsyncDummy);
        hud.render();
    }

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

    // Reversed-Z depth, set up once: 0..1 clip range (instead of GL's default -1..1,
    // which throws away half the depth precision), clear to 0, pass what's greater.
    // Rationale in render/Camera.hpp next to projReversedZ().
    glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);
    glClearDepth(0.0);
    glEnable(GL_CULL_FACE); // quad corners wind CCW from outside; backfaces never rasterize

    const std::filesystem::path assets = vx::findAssetsDir(argv[0]);
    if (assets.empty()) {
        VX_ERROR("could not locate assets/ directory");
        return 1;
    }
    const GLuint program =
        vx::loadProgram(assets / "shaders/chunk.vert", assets / "shaders/chunk.frag");
    if (!program)
        return 1;

    // --- the one chunk, meshed both ways ----------------------------------------------
    auto chunk = std::make_unique<vx::Chunk>();
    vx::fillTestTerrain(*chunk);

    // Naive stays as the oracle: every merged quad covers exactly the unit faces the
    // naive mesher emits, so summed quad area per direction must match its face counts.
    std::vector<vx::ChunkVertex> naiveVerts;
    vx::MeshStats naive;
    const double naiveBest =
        benchBestMs(50, [&] { naive = vx::meshNaive(*chunk, naiveVerts); });

    auto mesher = std::make_unique<vx::GreedyMesher>();
    std::vector<std::uint64_t> quads;
    vx::GreedyStats greedy;
    const double greedyBest =
        benchBestMs(50, [&] { greedy = mesher->mesh(*chunk, quads); });

    bool oracleOk = true;
    for (int d = 0; d < 6; ++d)
        oracleOk = oracleOk && greedy.areaPerDir[d] == std::int64_t(naive.facesPerDir[d]);
    if (oracleOk) {
        VX_INFO("oracle: merged quad area == naive face count in all 6 directions");
    } else {
        for (int d = 0; d < 6; ++d)
            VX_ERROR("oracle dir %d: greedy area %lld vs naive faces %d", d,
                     static_cast<long long>(greedy.areaPerDir[d]), naive.facesPerDir[d]);
        return 1;
    }

    VX_INFO("naive : %d tris, %.0f KB vbo, best %.3f ms", naive.triangles,
            double(naive.vertices * sizeof(vx::ChunkVertex)) / 1024.0, naiveBest);
    VX_INFO("greedy: %d quads -> %d tris, %.1f KB ssbo, best %.3f ms", greedy.quads,
            greedy.triangles, double(quads.size() * sizeof(std::uint64_t)) / 1024.0,
            greedyBest);
    VX_INFO("       %.1fx fewer tris, %.0fx less geometry memory", double(naive.triangles) / double(greedy.triangles),
            double(naive.vertices * sizeof(vx::ChunkVertex)) /
                double(quads.size() * sizeof(std::uint64_t)));

    const QuadMesh mesh = uploadQuads(quads);

    vx::FlyCamera camera;
    camera.position = {48.0f, 44.0f, 48.0f};
    camera.yawDeg = -45.0f; // look back at the chunk center
    camera.pitchDeg = -32.0f;

    vx::DebugHud hud;
    if (!hud.init(window->handle()))
        return 1;
    window->hookScroll(); // wrap ImGui's scroll callback so both UI and camera see it

    if (smoke) {
        const int rc = runSmokeTest(program, mesh, camera, hud, *window, smokeOut);
        hud.shutdown();
        return rc;
    }

    vx::GpuTimer gpuTimer;
    gpuTimer.init();

    double prevTime = glfwGetTime();
    bool f3Held = false;
    bool vsync = false;
    std::uint64_t allocsPrevFrame = vx::heapAllocCount();

    while (!window->shouldClose()) {
        window->pollEvents();
        if (window->keyDown(GLFW_KEY_ESCAPE))
            window->requestClose();
        const bool f3 = window->keyDown(GLFW_KEY_F3);
        if (f3 && !f3Held)
            hud.toggle();
        f3Held = f3;

        const double now = glfwGetTime();
        const float frameMs = float((now - prevTime) * 1000.0);
        const float dt = std::min(float(now - prevTime), 0.1f); // clamp debugger pauses
        prevTime = now;

        hud.beginFrame();
        camera.update(*window, dt, hud.wantsMouse(), hud.wantsKeyboard());

        int fbW = 0;
        int fbH = 0;
        window->framebufferSize(fbW, fbH);
        gpuTimer.begin();
        if (fbW > 0 && fbH > 0) { // minimized window has a 0x0 framebuffer
            const glm::mat4 vp = camera.projReversedZ(float(fbW) / float(fbH)) * camera.view();
            drawScene(program, mesh, vp, fbW, fbH);
        }

        const std::uint64_t allocsNow = vx::heapAllocCount();
        const vx::FrameStats stats{
            .frameMs = frameMs,
            .gpuMs = gpuTimer.milliseconds(),
            .heapAllocs = allocsNow - allocsPrevFrame,
            .drawCalls = 1,
            .quads = mesh.quadCount,
            .triangles = mesh.quadCount * 2,
        };
        allocsPrevFrame = allocsNow;
        const bool vsyncBefore = vsync;
        hud.buildUi(stats, camera, vsync);
        if (vsync != vsyncBefore)
            window->setVsync(vsync);
        hud.render();
        gpuTimer.end();

        window->swapBuffers();
    }

    hud.shutdown();
    return 0;
}
