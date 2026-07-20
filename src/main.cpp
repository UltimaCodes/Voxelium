#include "core/Bmp.hpp"
#include "core/Log.hpp"
#include "core/Paths.hpp"
#include "gfx/GlDebug.hpp"
#include "gfx/Shader.hpp"
#include "platform/Window.hpp"

#include <glad/gl.h>
#include <GLFW/glfw3.h>

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

// One interleaved vertex: xy position, rgb color. This whole path is throwaway — the
// real engine binds no vertex attributes at all (quads are pulled from an SSBO by
// gl_VertexID, milestone 4) — but it exercises every DSA pattern we keep using.
TriangleMesh createTriangle() {
    const float verts[] = {
        // x       y       r      g      b
         0.00f,  0.62f,  0.98f, 0.29f, 0.24f,
        -0.55f, -0.44f,  0.24f, 0.83f, 0.43f,
         0.55f, -0.44f,  0.28f, 0.53f, 0.96f,
    };

    TriangleMesh mesh;
    glCreateBuffers(1, &mesh.vbo);
    // Immutable storage (glNamedBufferStorage, not glNamedBufferData): size is fixed for
    // the buffer's lifetime, so the driver never guesses about reallocation — and it is
    // the only storage kind that later supports persistent mapping (milestone 7).
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

void drawFrame(GLuint program, GLuint vao, float time, int fbWidth, int fbHeight) {
    glViewport(0, 0, fbWidth, fbHeight);
    glClearColor(kClearColor[0], kClearColor[1], kClearColor[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(program);
    glProgramUniform1f(program, 0, time);                                   // uTime
    glProgramUniform1f(program, 1, float(fbWidth) / float(fbHeight));       // uAspect
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

// Renders into an offscreen FBO, verifies the triangle actually produced pixels, writes
// a BMP screenshot, exits 0/1. Offscreen because reading back a hidden window's default
// framebuffer is undefined (the pixel-ownership test); an FBO always owns its pixels.
int runSmokeTest(GLuint program, GLuint vao, const std::filesystem::path& outPath) {
    const int w = 800;
    const int h = 600;

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

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    for (int i = 0; i < 5; ++i)
        drawFrame(program, vao, 0.6f, w, h); // fixed time → deterministic image

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
        VX_ERROR("smoke: triangle did not render");
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

    if (smoke)
        return runSmokeTest(program, mesh.vao, smokeOut);

    double lastTitleUpdate = glfwGetTime();
    int framesSinceTitle = 0;
    while (!window->shouldClose()) {
        window->pollEvents();
        if (window->keyDown(GLFW_KEY_ESCAPE))
            window->requestClose();

        int fbW = 0;
        int fbH = 0;
        window->framebufferSize(fbW, fbH);
        if (fbW > 0 && fbH > 0) // minimized window has a 0x0 framebuffer
            drawFrame(program, mesh.vao, float(glfwGetTime()), fbW, fbH);
        window->swapBuffers();

        // Coarse frame stats in the title until the M2 HUD exists.
        ++framesSinceTitle;
        const double now = glfwGetTime();
        if (now - lastTitleUpdate >= 0.5) {
            char title[128];
            std::snprintf(title, sizeof(title), "Voxelium | %.2f ms/frame (%d fps, vsync on)",
                          1000.0 * (now - lastTitleUpdate) / framesSinceTitle,
                          int(framesSinceTitle / (now - lastTitleUpdate) + 0.5));
            window->setTitle(title);
            lastTitleUpdate = now;
            framesSinceTitle = 0;
        }
    }
    return 0;
}
