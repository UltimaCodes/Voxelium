#pragma once

#include <memory>

struct GLFWwindow;

namespace vx {

struct WindowDesc {
    int width = 1280;
    int height = 720;
    const char* title = "Voxelium";
    bool visible = true;
    bool debugContext = false;
};

// Owns GLFW init/terminate and one window with a GL 4.6 core context (made current, GL
// functions loaded). One window per process is an engine assumption, not a limitation to
// engineer around.
class Window {
public:
    // Returns nullptr on failure (errors are logged). No exceptions.
    static std::unique_ptr<Window> create(const WindowDesc& desc);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool shouldClose() const;
    void requestClose();
    void pollEvents();
    void swapBuffers();
    bool keyDown(int glfwKey) const;
    void framebufferSize(int& width, int& height) const;
    void setTitle(const char* title);
    GLFWwindow* handle() const { return m_window; }

private:
    explicit Window(GLFWwindow* window) : m_window(window) {}
    GLFWwindow* m_window = nullptr;
};

} // namespace vx
