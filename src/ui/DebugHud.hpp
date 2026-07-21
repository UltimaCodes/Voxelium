#pragma once

#include <cstdint>

struct GLFWwindow;

namespace vx {

class FlyCamera;

struct FrameStats {
    float frameMs = 0.0f; // whole frame including vsync wait
    float gpuMs = 0.0f;
    std::uint64_t heapAllocs = 0; // this frame, engine code only (operator new)
    int drawCalls = 0;
    int quads = 0;
    int triangles = 0;
};

// ImGui debug overlay: frame graph, timings, alloc counter, camera state, log view.
// Debug tooling only — gameplay UI will never be ImGui.
class DebugHud {
public:
    bool init(GLFWwindow* window);
    void shutdown();

    void beginFrame();
    void buildUi(const FrameStats& stats, const FlyCamera& camera, bool& vsync);
    void render();

    void toggle() { m_visible = !m_visible; }
    bool visible() const { return m_visible; }
    bool wantsMouse() const;
    bool wantsKeyboard() const;

private:
    static constexpr int kSamples = 240;
    float m_frameMs[kSamples] = {};
    int m_cursor = 0;
    bool m_visible = true;
    bool m_initialized = false;
};

} // namespace vx
