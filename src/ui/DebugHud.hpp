#pragma once

#include "io/WorldStore.hpp"
#include "render/PostFx.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct GLFWwindow;
struct ImFont;

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
// tooling only, the in-world gameplay HUD (crosshair, hotbar, block outline) is real GL,
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

    // Full-screen loading panel. Shown while a world settles, at startup and on every
    // world change, so the first frame the player sees has its LOD, shadows and chunks
    // already in place instead of assembling itself around them.
    void loadingScreen(float progress, const char* title, const char* phase);

    // What the ESC menu wants to say about worlds. The menu only ever sets a request; the
    // caller does the work, because loading a world means draining jobs and putting a
    // loading screen up, none of which belongs in an ImGui callback.
    struct WorldUi {
        int seed = 0;
        bool resetRequested = false;
        bool saveRequested = false;
        bool loadRequested = false;
        bool deleteRequested = false;
        char saveName[64] = "";
        std::string selected;                    // which saved world load/delete refers to
        std::vector<WorldStore::Entry> saved;    // filled by the caller each frame
    };

    // titleMode is the same window opened from the title screen: no world tab (there is no world
    // yet) and a Back button in place of resume/quit, laid out in the title's right-hand column.
    void settings(bool& open, bool& vsync, bool& caveCull, int& radius, float& fovDeg,
                  FxToggles& fx, bool& quit, WorldUi& worlds, bool titleMode = false);

    // Shown before any world exists, once at launch. Separate from WorldUi/settings() above:
    // that panel always assumes a live world already exists (it can save/replace it), where
    // this one is choosing what to create or open in the first place, and it is the only
    // place a SPECIFIC seed can be entered at all, the in-game "new world" button has never
    // taken one, it only ever rolls a random one.
    struct TitleUi {
        char seedText[32] = ""; // pre-filled with a rolled seed; the terrain behind the menu is this
        int page = 1;           // 1 = new world, 2 = saved worlds, 3 = settings (drawn by DebugHud::settings)
        float fade = 1.0f;      // 1 hides the backdrop (while its terrain is being built), eased by the caller
        bool newWorldRequested = false;
        bool loadRequested = false;
        bool quitRequested = false;
        bool rerollRequested = false;
        bool deleteRequested = false; // deletes `selected`; the caller does it and refreshes `saved`
        std::string selected;
        std::string deleteArmed; // the world whose delete button has been pressed once
        std::vector<WorldStore::Entry> saved; // filled by the caller each frame
    };
    // Caller drives its own frame loop (poll/clear/swap) until one of the UI's *Requested
    // flags comes back true; this only draws the ImGui content for one frame.
    void titleScreen(TitleUi& ui);

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
    // Faces for the title and loading screens, loaded from the system fonts if they exist; null
    // means "keep whatever font is current", which is the built-in one.
    ImFont* m_fontDisplay = nullptr;
    ImFont* m_fontUi = nullptr;
    double m_titleStart = -1.0;

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
