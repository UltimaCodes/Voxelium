#include "platform/Window.hpp"

#include "core/Log.hpp"

#include <glad/gl.h>
#include <GLFW/glfw3.h>

namespace vx {

static void glfwErrorCb(int code, const char* description) {
    VX_ERROR("GLFW error %d: %s", code, description);
}

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
        VX_ERROR("glfwCreateWindow failed — is the driver OpenGL 4.6 capable?");
        glfwTerminate();
        return nullptr;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // vsync on by default; frame pacing gets revisited with the M2 HUD

    if (gladLoadGL(glfwGetProcAddress) == 0) {
        VX_ERROR("gladLoadGL failed");
        glfwDestroyWindow(window);
        glfwTerminate();
        return nullptr;
    }

    return std::unique_ptr<Window>(new Window(window));
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

void Window::framebufferSize(int& width, int& height) const {
    glfwGetFramebufferSize(m_window, &width, &height);
}

void Window::setTitle(const char* title) { glfwSetWindowTitle(m_window, title); }

} // namespace vx
