#include "core/AllocTracker.hpp"
#include "core/Bmp.hpp"
#include "core/Log.hpp"
#include "core/Paths.hpp"
#include "gfx/GlDebug.hpp"
#include "gfx/GpuTimer.hpp"
#include "gfx/Shader.hpp"
#include "platform/Window.hpp"
#include "render/Camera.hpp"
#include "ui/DebugHud.hpp"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <vector>

namespace {

constexpr float kClearColor[3] = {0.09f, 0.10f, 0.12f};

struct TriangleMesh {
    GLuint vao = 0;
    GLuint vbo = 0;
};

// One interleaved vertex: xy position, rgb color. Throwaway path — from M4 on there are
// no vertex attributes at all (quads get pulled from an SSBO by gl_VertexID) — but it
// exercises the DSA setup we keep using.
TriangleMesh createTriangle() {
    const float verts[] = {
        // x       y       r      g      b
         0.00f,  0.62f,  0.98f, 0.29f, 0.24f,
        -0.55f, -0.44f,  0.24f, 0.83f, 0.43f,
         0.55f, -0.44f,  0.28f, 0.53f, 0.96f,
    };

    TriangleMesh mesh;
    glCreateBuffers(1, &mesh.vbo);
    // Immutable storage: fixed size for the buffer's lifetime, so the driver never has
    // to guess about reallocation. Also the only storage kind that allows persistent
    // mapping, which the M7 quad arena needs.
    glNamedBufferStorage(mesh.vbo, sizeof(verts), verts, 0);

    glCreateVertexArrays(1, &mesh.vao);
    const GLuint binding = 0;
    glVertexArrayVertexBuffer(mesh.vao, binding, mesh.vbo, 0, 5 * sizeof(float));
    glEnableVertexArrayAttrib(mesh.vao, 0);
    glVertexArrayAttribFormat(mesh.vao, 0, 2, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(mesh.vao, 0, binding);
    glEnableVertexArrayAttrib(mesh.vao, 1);
    glVertexArrayAttribFormat(mesh.vao, 1, 3, GL_FLOAT, GL_FALSE, 2 * sizeof(float));
    glVertexArrayAttribBinding(mesh.vao, 1, binding);
    return mesh;
}

void drawScene(GLuint program, GLuint vao, float time, const glm::mat4& viewProj, int fbWidth,
               int fbHeight) {
    glViewport(0, 0, fbWidth, fbHeight);
    glClearColor(kClearColor[0], kClearColor[1], kClearColor[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(program);
    glProgramUniform1f(program, 0, time);
    glProgramUniformMatrix4fv(program, 1, 1, GL_FALSE, glm::value_ptr(viewProj));
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

// Renders a few frames (scene + HUD) into an offscreen FBO, verifies pixels were
// produced, writes a BMP, exits 0/1. Offscreen because reading back a hidden window's
// default framebuffer is undefined (pixel-ownership test); an FBO always owns its pixels.
int runSmokeTest(GLuint program, GLuint vao, const vx::FlyCamera& camera, vx::DebugHud& hud,
                 vx::Window& window, const std::filesystem::path& outPath) {
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
    for (int i = 0; i < 5; ++i) {
        window.pollEvents();
        hud.beginFrame();
        drawScene(program, vao, 0.6f, vp, w, h); // fixed time → deterministic image
        const vx::FrameStats stats{
            .frameMs = 1.0f, .gpuMs = 0.5f, .heapAllocs = 0, .drawCalls = 1, .triangles = 1};
        hud.buildUi(stats, camera);
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

    const std::filesystem::path assets = vx::findAssetsDir(argv[0]);
    if (assets.empty()) {
        VX_ERROR("could not locate assets/ directory");
        return 1;
    }
    const GLuint program = vx::loadProgram(assets / "shaders/triangle.vert",
                                           assets / "shaders/triangle.frag");
    if (!program)
        return 1;

    const TriangleMesh mesh = createTriangle();
    vx::FlyCamera camera;

    vx::DebugHud hud;
    if (!hud.init(window->handle()))
        return 1;
    window->hookScroll(); // wrap ImGui's scroll callback so both UI and camera see it

    if (smoke) {
        const int rc = runSmokeTest(program, mesh.vao, camera, hud, *window, smokeOut);
        hud.shutdown();
        return rc;
    }

    vx::GpuTimer gpuTimer;
    gpuTimer.init();

    double prevTime = glfwGetTime();
    float frameMs = 0.0f;
    bool f3Held = false;
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
        frameMs = float((now - prevTime) * 1000.0);
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
            drawScene(program, mesh.vao, float(now), vp, fbW, fbH);
        }

        const std::uint64_t allocsNow = vx::heapAllocCount();
        const vx::FrameStats stats{
            .frameMs = frameMs,
            .gpuMs = gpuTimer.milliseconds(),
            .heapAllocs = allocsNow - allocsPrevFrame,
            .drawCalls = 1,
            .triangles = 1,
        };
        allocsPrevFrame = allocsNow;
        hud.buildUi(stats, camera);
        hud.render();
        gpuTimer.end();

        window->swapBuffers();
    }

    hud.shutdown();
    return 0;
}
