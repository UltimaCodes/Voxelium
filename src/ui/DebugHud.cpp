#include "ui/DebugHud.hpp"

#include "core/Log.hpp"
#include "render/Camera.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <algorithm>

namespace vx {

bool DebugHud::init(GLFWwindow* window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr; // don't scatter imgui.ini files around
    ImGui::StyleColorsDark();

    if (!ImGui_ImplGlfw_InitForOpenGL(window, true)) {
        VX_ERROR("ImGui GLFW backend init failed");
        return false;
    }
    if (!ImGui_ImplOpenGL3_Init("#version 460")) {
        VX_ERROR("ImGui OpenGL backend init failed");
        return false;
    }
    m_initialized = true;
    return true;
}

void DebugHud::shutdown() {
    if (!m_initialized)
        return;
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    m_initialized = false;
}

void DebugHud::beginFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

bool DebugHud::wantsMouse() const { return ImGui::GetIO().WantCaptureMouse; }
bool DebugHud::wantsKeyboard() const { return ImGui::GetIO().WantCaptureKeyboard; }

void DebugHud::diagnostics(const FrameStats& stats, const FlyCamera& camera) {
    m_frameMs[m_cursor] = stats.frameMs;
    m_cursor = (m_cursor + 1) % kSamples;

    // roll the frame/gpu numbers up over ~250 ms so the readout is legible instead of a blur
    m_accumMs += stats.frameMs;
    m_accumGpuMs += stats.gpuMs;
    ++m_accumFrames;
    if (m_accumMs >= 250.0f && m_accumFrames > 0) {
        m_dispFrameMs = m_accumMs / float(m_accumFrames);
        m_dispGpuMs = m_accumGpuMs / float(m_accumFrames);
        m_dispFps = m_dispFrameMs > 0.0f ? 1000.0f / m_dispFrameMs : 0.0f;
        m_accumMs = m_accumGpuMs = 0.0f;
        m_accumFrames = 0;
    }

    if (!m_visible)
        return;

    ImGui::SetNextWindowPos(ImVec2(12.0f, 12.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(400.0f, 0.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("debug (F3)");

    ImGui::Text("frame %6.2f ms   gpu %6.2f ms   %5.0f fps", m_dispFrameMs, m_dispGpuMs,
                m_dispFps);

    const float peak = *std::max_element(m_frameMs, m_frameMs + kSamples);
    ImGui::PlotLines("##frametimes", m_frameMs, kSamples, m_cursor, nullptr, 0.0f,
                     std::max(peak * 1.25f, 4.0f), ImVec2(-1.0f, 64.0f));

    if (stats.heapAllocs == 0)
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f), "heap allocs/frame: 0");
    else
        ImGui::TextColored(ImVec4(0.95f, 0.4f, 0.3f, 1.0f), "heap allocs/frame: %llu",
                           static_cast<unsigned long long>(stats.heapAllocs));
    ImGui::Text("draw calls %d   quads %d   tris %d", stats.drawCalls, stats.quads,
                stats.triangles);
    ImGui::Text("chunks %d loaded   %d meshed   gen queue %d", stats.chunksLoaded,
                stats.chunksMeshed, stats.genQueue);
    ImGui::Text("jobs %d in flight   uploads %d pending (%.2f ms)", stats.jobsInFlight,
                stats.uploadsPending, double(stats.uploadMs));
    ImGui::Text("%.1f MB voxels   %.2f MB quads",
                double(stats.voxelBytes) / (1024.0 * 1024.0),
                double(stats.quadBytes) / (1024.0 * 1024.0));
    ImGui::Text("gpu drew %d chunks   arena %.1f / %.0f MB (%d free blocks)",
                stats.gpuChunksDrawn,
                double(stats.arenaUsedQuads) * 8.0 / (1024.0 * 1024.0),
                double(stats.arenaCapQuads) * 8.0 / (1024.0 * 1024.0),
                stats.arenaFreeBlocks);
    ImGui::Text("gpu: shadow %.2f   scene %.2f   post %.2f ms", double(stats.gpuShadowMs),
                double(stats.gpuSceneMs), double(stats.gpuPostMs));

    ImGui::Separator();
    ImGui::Text("cam (%.1f, %.1f, %.1f)  yaw %.0f  pitch %.0f  speed %.1f%s",
                double(camera.position.x), double(camera.position.y), double(camera.position.z),
                double(camera.yawDeg), double(camera.pitchDeg), double(camera.baseSpeed),
                camera.flying() ? "  [flying]" : "");
    ImGui::TextDisabled("%s | ESC settings | F4 hotbar | LMB break, MMB place | F5 mode | F2 "
                        "save",
                        stats.walkMode ? "walk" : "fly");

    ImGui::Separator();
    if (ImGui::CollapsingHeader("log", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::BeginChild("##log", ImVec2(0.0f, 150.0f), ImGuiChildFlags_Borders);
        log::forEach(
            [](const log::Line& line, void*) {
                if (line.level == log::Level::Error)
                    ImGui::TextColored(ImVec4(0.95f, 0.4f, 0.3f, 1.0f), "%s", line.text);
                else
                    ImGui::TextUnformatted(line.text);
            },
            nullptr);
        if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 4.0f)
            ImGui::SetScrollHereY(1.0f);
        ImGui::EndChild();
    }

    ImGui::End();
}

void DebugHud::settings(bool& open, bool& vsync, bool& caveCull, int& radius, float& fovDeg,
                        FxToggles& fx, bool& quit) {
    if (!open)
        return;

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(vp->GetCenter().x, vp->GetCenter().y), ImGuiCond_Always,
                            ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(360.0f, 0.0f), ImGuiCond_Always);
    ImGui::Begin("settings", nullptr,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoCollapse);

    ImGui::SeparatorText("display");
    ImGui::Checkbox("vsync", &vsync);
    ImGui::SameLine();
    ImGui::Checkbox("cave cull", &caveCull);
    ImGui::SetNextItemWidth(-100.0f);
    ImGui::SliderInt("render dist", &radius, 4, 16, "%d chunks");
    ImGui::SetNextItemWidth(-100.0f);
    ImGui::SliderFloat("fov", &fovDeg, 50.0f, 100.0f, "%.0f deg");

    ImGui::SeparatorText("lighting");
    ImGui::SetNextItemWidth(-100.0f);
    ImGui::SliderFloat("sun angle", &fx.sunAltitudeDeg, 8.0f, 85.0f, "%.0f deg");
    ImGui::SetNextItemWidth(-100.0f);
    ImGui::SliderFloat("exposure", &fx.exposure, 0.4f, 2.5f, "%.2f");
    ImGui::Checkbox("shadows", &fx.shadows);

    ImGui::SeparatorText("effects");
    ImGui::Checkbox("ssao", &fx.ssao);
    ImGui::SameLine();
    ImGui::Checkbox("volumetric", &fx.volumetric);
    ImGui::SameLine();
    ImGui::Checkbox("bloom", &fx.bloom);
    ImGui::SameLine();
    ImGui::Checkbox("clouds", &fx.clouds);
    ImGui::Checkbox("ssgi (bounce light)", &fx.ssgi);
    ImGui::SameLine();
    ImGui::Checkbox("taa (anti-aliasing)", &fx.taa);

    ImGui::Separator();
    if (ImGui::Button("resume", ImVec2(160.0f, 32.0f)))
        open = false;
    ImGui::SameLine();
    if (ImGui::Button("quit", ImVec2(160.0f, 32.0f)))
        quit = true;

    ImGui::End();
}

void DebugHud::render() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

} // namespace vx
