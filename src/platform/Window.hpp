#pragma once

#include <filesystem>
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
    bool mouseDown(int glfwButton) const;
    void cursorPos(double& x, double& y) const;
    void framebufferSize(int& width, int& height) const;
    void setTitle(const char* title);
    void setVsync(bool on);
    // Taskbar/title-bar/alt-tab icon. `assetsDir` should hold icon_16.png/32/48/256 (see
    // assets/) — handing GLFW one image per real size it asks for looks sharp everywhere
    // instead of one image stretched to fit. Logs and no-ops on any missing/bad file: a
    // missing icon is cosmetic, never worth failing startup over.
    void setIcon(const std::filesystem::path& assetsDir);

    // Captured = cursor hidden + locked, raw motion if the platform has it (fly mode).
    void setCursorCaptured(bool captured);
    // Whether the OS focus is on this window. Mouse capture follows it: alt-tabbing away
    // must hand the cursor back, and coming back must take it again without a click.
    bool focused() const;

    // Call after ImGui installs its GLFW callbacks: wraps the scroll callback so both
    // ImGui and the engine see scroll events. Engine side accumulates into a delta that
    // consumeScrollY() drains once per frame.
    void hookScroll();
    double consumeScrollY();

    GLFWwindow* handle() const { return m_window; }

private:
    explicit Window(GLFWwindow* window) : m_window(window) {}
    static void scrollCallback(GLFWwindow* window, double dx, double dy);

    GLFWwindow* m_window = nullptr;
    double m_scrollY = 0.0;
};

} // namespace vx
