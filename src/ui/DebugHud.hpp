#pragma once

#include "render/PostFx.hpp"

#include <cstdint>

struct GLFWwindow;

namespace vx {

class FlyCamera;

struct FrameStats {
    float frameMs = 0.0f; // whole frame including vsync wait
    float gpuMs = 0.0f;
    std::uint64_t heapAllocs = 0; // this frame, engine code only (operator new)
    int drawCalls = 0;
    int quads = 0;     // drawn this frame
    int triangles = 0;
    int chunksLoaded = 0;
    int chunksMeshed = 0;
    int genQueue = 0;
    int jobsInFlight = 0;
    int uploadsPending = 0;
    float uploadMs = 0.0f;
    std::uint64_t voxelBytes = 0;
    std::uint64_t quadBytes = 0;
    int gpuChunksDrawn = 0; // from the cull shader's counter, ~3 frames of latency
    std::uint32_t arenaUsedQuads = 0;
    std::uint32_t arenaCapQuads = 0;
    int arenaFreeBlocks = 0;
    bool walkMode = false;
    int placeBlock = 1;
    float gpuShadowMs = 0.0f;
    float gpuSceneMs = 0.0f;
    float gpuPostMs = 0.0f;
};

// ImGui debug/settings overlay: F3 diagnostics (frame graph, timings, alloc counter,
// camera state, log) and the ESC settings menu (every toggle + slider). Debug/settings
// tooling only — the in-world gameplay HUD (crosshair, hotbar, block outline) is real GL,
// never ImGui.
class DebugHud {
public:
    bool init(GLFWwindow* window);
    void shutdown();

    void beginFrame();
    // F3 window: read-only diagnostics, no controls.
    void diagnostics(const FrameStats& stats, const FlyCamera& camera);
    // ESC window: all the knobs. Sets open=false on Resume, quit=true on Quit.
    // Swatch picker for lamps. Opens where you middle-clicked one; picking sets `swatch`
    // and returns true so the caller can repaint that lamp. Deliberately a grid of the
    // real block colours rather than a free colour control: these are the only lamp
    // colours that exist, so offering more would mean quietly snapping the answer.
    bool lampPicker(bool& open, int& swatch);

    void settings(bool& open, bool& vsync, bool& caveCull, int& radius, float& fovDeg,
                  FxToggles& fx, bool& quit);
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

    // decimated readout: the raw numbers change every frame and are unreadable, so the
    // displayed frame/gpu/fps values only refresh a few times a second (averaged over the
    // window in between).
    float m_accumMs = 0.0f;
    float m_accumGpuMs = 0.0f;
    int m_accumFrames = 0;
    float m_dispFrameMs = 0.0f;
    float m_dispGpuMs = 0.0f;
    float m_dispFps = 0.0f;
};

} // namespace vx
