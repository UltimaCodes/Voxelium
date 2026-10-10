#include "platform/Window.hpp"

#include "core/Log.hpp"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <stb_image.h> // implementation compiled once in BlockTextures.cpp

#include <array>

namespace vx {

static void glfwErrorCb(int code, const char* description) {
    VX_ERROR("GLFW error %d: %s", code, description);
}

// Previous scroll callback in the chain (ImGui's, once the HUD is up).
static GLFWscrollfun s_prevScrollCb = nullptr;

std::unique_ptr<Window> Window::create(const WindowDesc& desc) {
    glfwSetErrorCallback(glfwErrorCb);
    if (!glfwInit()) {
        VX_ERROR("glfwInit failed");
        return nullptr;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, desc.debugContext ? GLFW_TRUE : GLFW_FALSE);
    glfwWindowHint(GLFW_VISIBLE, desc.visible ? GLFW_TRUE : GLFW_FALSE);

    GLFWwindow* window = glfwCreateWindow(desc.width, desc.height, desc.title, nullptr, nullptr);
    if (!window) {
        VX_ERROR("glfwCreateWindow failed, is the driver OpenGL 4.6 capable?");
        glfwTerminate();
        return nullptr;
    }

    glfwMakeContextCurrent(window);
    // vsync off by default: everything time-dependent runs off dt / absolute time, so
    // uncapped fps never changes game speed. The HUD has a vsync toggle.
    glfwSwapInterval(0);

    if (gladLoadGL(glfwGetProcAddress) == 0) {
        VX_ERROR("gladLoadGL failed");
        glfwDestroyWindow(window);
        glfwTerminate();
        return nullptr;
    }

    auto self = std::unique_ptr<Window>(new Window(window));
    glfwSetWindowUserPointer(window, self.get());
    return self;
}

Window::~Window() {
    if (m_window)
        glfwDestroyWindow(m_window);
    glfwTerminate();
}

bool Window::shouldClose() const { return glfwWindowShouldClose(m_window); }
void Window::requestClose() { glfwSetWindowShouldClose(m_window, GLFW_TRUE); }
void Window::pollEvents() { glfwPollEvents(); }
void Window::swapBuffers() { glfwSwapBuffers(m_window); }
bool Window::keyDown(int glfwKey) const { return glfwGetKey(m_window, glfwKey) == GLFW_PRESS; }

bool Window::mouseDown(int glfwButton) const {
    return glfwGetMouseButton(m_window, glfwButton) == GLFW_PRESS;
}

void Window::cursorPos(double& x, double& y) const { glfwGetCursorPos(m_window, &x, &y); }

void Window::framebufferSize(int& width, int& height) const {
    glfwGetFramebufferSize(m_window, &width, &height);
}

void Window::setTitle(const char* title) { glfwSetWindowTitle(m_window, title); }

void Window::setVsync(bool on) { glfwSwapInterval(on ? 1 : 0); }

void Window::setIcon(const std::filesystem::path& assetsDir) {
    static constexpr int kSizes[] = {16, 32, 48, 256};
    std::array<GLFWimage, std::size(kSizes)> images{};
    std::array<stbi_uc*, std::size(kSizes)> pixels{};
    int loaded = 0;
    for (std::size_t i = 0; i < std::size(kSizes); ++i) {
        const auto path = assetsDir / ("icon_" + std::to_string(kSizes[i]) + ".png");
        int w = 0, h = 0, comp = 0;
        pixels[i] = stbi_load(path.string().c_str(), &w, &h, &comp, 4);
        if (pixels[i]) {
            images[loaded] = {w, h, pixels[i]};
            ++loaded;
        }
    }
    if (loaded > 0)
        glfwSetWindowIcon(m_window, loaded, images.data());
    else
        VX_ERROR("window icon: no icon_*.png found under %s (cosmetic only, continuing)",
                 assetsDir.string().c_str());
    for (stbi_uc* p : pixels)
        if (p)
            stbi_image_free(p);
}

bool Window::focused() const {
    return m_window && glfwGetWindowAttrib(m_window, GLFW_FOCUSED) != 0;
}

void Window::setCursorCaptured(bool captured) {
    glfwSetInputMode(m_window, GLFW_CURSOR, captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    if (captured && glfwRawMouseMotionSupported())
        glfwSetInputMode(m_window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
}

void Window::hookScroll() {
    s_prevScrollCb = glfwSetScrollCallback(m_window, &Window::scrollCallback);
}

double Window::consumeScrollY() {
    const double v = m_scrollY;
    m_scrollY = 0.0;
    return v;
}

void Window::scrollCallback(GLFWwindow* window, double dx, double dy) {
    if (s_prevScrollCb)
        s_prevScrollCb(window, dx, dy);
    if (auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window)))
        self->m_scrollY += dy;
}

} // namespace vx
