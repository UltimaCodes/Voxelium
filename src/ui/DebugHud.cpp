#include "ui/DebugHud.hpp"

#include "render/QualityPreset.hpp"

#include "core/Log.hpp"
#include "render/Camera.hpp"
#include "world/Chunk.hpp" // lamp hue count, for the colour picker's tooltip

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

bool DebugHud::lampPicker(bool& open, int& swatch) {
    if (!open)
        return false;
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(vp->GetCenter().x, vp->GetCenter().y), ImGuiCond_Always,
                            ImVec2(0.5f, 0.5f));
    ImGui::Begin("lamp colour", &open,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize);

    bool picked = false;
    constexpr int kPerRow = 12; // hues across, so saturated sits directly above its pale
    for (int i = 0; i < blocks::kLampCount; ++i) {
        const blocks::LampRgb& c = blocks::kLampRGB[std::size_t(i)];
        const ImVec4 col(float(c.r) / 15.0f, float(c.g) / 15.0f, float(c.b) / 15.0f, 1.0f);
        ImGui::PushID(i);
        // the current swatch keeps a border so you can see what a lamp already is
        const ImGuiColorEditFlags f =
            ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoDragDrop;
        if (i == swatch)
            ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, i == swatch ? 2.0f : 0.0f);
        if (ImGui::ColorButton("##sw", col, f, ImVec2(26.0f, 26.0f))) {
            swatch = i;
            picked = true;
        }
        ImGui::PopStyleVar();
        if (i == swatch)
            ImGui::PopStyleColor();
        ImGui::PopID();
        if (i % kPerRow != kPerRow - 1 && i != blocks::kLampCount - 1)
            ImGui::SameLine();
    }
    ImGui::TextDisabled("everything you place next uses this colour");
    ImGui::End();
    if (picked)
        open = false;
    return picked;
}

void DebugHud::loadingScreen(float progress, const char* title, const char* phase) {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->Pos);
    ImGui::SetNextWindowSize(vp->Size);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0)); // the gradient below is the bg
    ImGui::Begin("##loading", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);

    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    const ImVec2 p0 = vp->Pos;
    const ImVec2 p1 = ImVec2(vp->Pos.x + vp->Size.x, vp->Pos.y + vp->Size.y);
    // Vertical gradient in the engine's own sky colours rather than a flat charcoal panel:
    // it is the first thing anyone sees, and a dead grey rectangle reads as a hang.
    const ImU32 top = IM_COL32(18, 24, 38, 255);
    const ImU32 bot = IM_COL32(38, 46, 58, 255);
    dl->AddRectFilledMultiColor(p0, p1, top, top, bot, bot);

    const float cx = vp->Pos.x + vp->Size.x * 0.5f;
    const float cy = vp->Pos.y + vp->Size.y * 0.5f;
    const float barW = std::min(520.0f, vp->Size.x * 0.55f);
    const float barH = 6.0f;
    const float barX = cx - barW * 0.5f;
    const float barY = cy + 18.0f;

    // title, tracked out a little by drawing it once per character
    {
        const float scale = 2.4f;
        ImFont* font = ImGui::GetFont();
        const float fs = ImGui::GetFontSize() * scale;
        float w = 0.0f;
        for (const char* c = title; *c; ++c)
            w += font->CalcTextSizeA(fs, FLT_MAX, 0.0f, c, c + 1).x + 3.0f;
        float x = cx - w * 0.5f;
        for (const char* c = title; *c; ++c) {
            dl->AddText(nullptr, fs, ImVec2(x, cy - 46.0f), IM_COL32(226, 232, 240, 255), c,
                        c + 1);
            x += font->CalcTextSizeA(fs, FLT_MAX, 0.0f, c, c + 1).x + 3.0f;
        }
    }

    // track, then fill, then a soft leading edge so it reads as moving even when a long
    // phase (a big light settle) holds the number still for a second
    const float t = std::clamp(progress, 0.0f, 1.0f);
    dl->AddRectFilled(ImVec2(barX, barY), ImVec2(barX + barW, barY + barH),
                      IM_COL32(255, 255, 255, 28), barH * 0.5f);
    if (t > 0.0f) {
        const float fillW = std::max(barH, barW * t);
        dl->AddRectFilled(ImVec2(barX, barY), ImVec2(barX + fillW, barY + barH),
                          IM_COL32(126, 196, 255, 255), barH * 0.5f);
        const float pulse = 0.55f + 0.45f * std::sin(float(ImGui::GetTime()) * 3.2f);
        dl->AddCircleFilled(ImVec2(barX + fillW, barY + barH * 0.5f), barH * 1.5f,
                            IM_COL32(190, 226, 255, int(90.0f * pulse)));
    }

    char pct[16];
    std::snprintf(pct, sizeof(pct), "%d%%", int(t * 100.0f + 0.5f));
    const ImVec2 ps = ImGui::CalcTextSize(pct);
    dl->AddText(ImVec2(barX + barW - ps.x, barY - ps.y - 8.0f), IM_COL32(226, 232, 240, 220),
                pct);
    const ImVec2 fs2 = ImGui::CalcTextSize(phase);
    dl->AddText(ImVec2(barX, barY - fs2.y - 8.0f), IM_COL32(160, 174, 192, 220), phase);

    ImGui::End();
    ImGui::PopStyleColor();
}

void DebugHud::settings(bool& open, bool& vsync, bool& caveCull, int& radius, float& fovDeg,
                        FxToggles& fx, bool& quit, WorldUi& worlds) {
    if (!open)
        return;

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(vp->GetCenter().x, vp->GetCenter().y), ImGuiCond_Always,
                            ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(430.0f, 480.0f), ImGuiCond_Always);
    ImGui::Begin("settings", nullptr,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoCollapse);

    // Tabs rather than one long column (#51). The old menu was a flat run of seven sections
    // with world save/load buried between "lighting" and "post", which is not where anyone
    // looks for it.
    if (ImGui::BeginTabBar("##tabs")) {
        if (ImGui::BeginTabItem("quality")) {
            // The preset is RECOGNISED from the values rather than stored, so it can never
            // claim to be High while sitting on hand-edited knobs. Touch anything a preset
            // owns and this reads Custom on the next frame; that is the whole state machine.
            const Preset current = detectPreset(fx, radius);
            for (int i = 0; i < 4; ++i) {
                const Preset p = Preset(i);
                if (i > 0)
                    ImGui::SameLine();
                const bool active = current == p;
                if (active)
                    ImGui::PushStyleColor(ImGuiCol_Button,
                                          ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
                if (ImGui::Button(presetName(p), ImVec2(95.0f, 0.0f)))
                    applyPreset(p, fx, radius);
                if (active)
                    ImGui::PopStyleColor();
            }
            if (current == Preset::Custom)
                ImGui::TextDisabled("custom - a preset knob has been changed");
            else
                ImGui::TextDisabled("preset: %s", presetName(current));

            ImGui::SeparatorText("cost");
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderInt("render dist", &radius, 4, 32, "%d chunks");
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderFloat("render scale", &fx.renderScale, 0.5f, 1.0f, "%.2fx");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("renders the whole HDR chain smaller and upscales at the\n"
                                  "end. the upscale is sharpened, so 0.85 stays crisp.");

            // Changing either of these recompiles the shaders, so the tab says so. The
            // rebuild is cached per tier, but the first visit to a tier is a real pause and
            // a control that stutters without explaining itself reads as a bug.
            ImGui::SeparatorText("shader tier (recompiles)");
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderInt("quality", &fx.quality, 0, 3, "%d");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("sample and step counts for ssao, ssgi, god rays, clouds,\n"
                                  "contact shadows and water reflections. baked into the\n"
                                  "programs, so changing it rebuilds them.");
            int shadowIdx = fx.shadowRes <= 1024   ? 0
                            : fx.shadowRes <= 1536 ? 1
                            : fx.shadowRes <= 2048 ? 2
                                                   : 3;
            static const char* const kShadowNames[] = {"1024", "1536", "2048", "2560"};
            static const int kShadowSizes[] = {1024, 1536, 2048, 2560};
            ImGui::SetNextItemWidth(-125.0f);
            if (ImGui::Combo("shadow map", &shadowIdx, kShadowNames, 4))
                fx.shadowRes = kShadowSizes[shadowIdx];

            ImGui::SeparatorText("effects");
            ImGui::Checkbox("shadows", &fx.shadows);
            ImGui::SameLine();
            ImGui::Checkbox("ssao", &fx.ssao);
            ImGui::SameLine();
            ImGui::Checkbox("ssgi", &fx.ssgi);
            ImGui::SameLine();
            ImGui::Checkbox("god rays", &fx.volumetric);
            ImGui::Checkbox("clouds", &fx.clouds);
            ImGui::SameLine();
            ImGui::Checkbox("fog", &fx.fog);
            ImGui::SameLine();
            ImGui::Checkbox("taa", &fx.taa);
            ImGui::SameLine();
            ImGui::Checkbox("motion blur", &fx.motionBlur);
            ImGui::BeginDisabled(!fx.shadows);
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderFloat("contact shadow", &fx.contactShadow, 0.0f, 1.0f, "%.2f");
            ImGui::EndDisabled();
            ImGui::BeginDisabled(!fx.clouds);
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderFloat("cloud shadow", &fx.cloudShadow, 0.0f, 1.0f, "%.2f");
            ImGui::EndDisabled();
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderFloat("leaf glow", &fx.leafSss, 0.0f, 2.0f, "%.2f");
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderFloat("surface relief", &fx.relief, 0.0f, 2.0f, "%.2f");

            ImGui::SeparatorText("system");
            ImGui::Checkbox("vsync", &vsync);
            ImGui::SameLine();
            ImGui::Checkbox("cave cull", &caveCull);
            ImGui::EndTabItem();
        }

        // Everything a preset does NOT own: taste, not performance. Kept apart so switching
        // preset never silently rewrites someone's look.
        if (ImGui::BeginTabItem("visuals")) {
            ImGui::SeparatorText("camera");
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderFloat("fov", &fovDeg, 50.0f, 110.0f, "%.0f deg");
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::ColorEdit4("crosshair", fx.crosshair,
                              ImGuiColorEditFlags_AlphaBar |
                                  ImGuiColorEditFlags_AlphaPreviewHalf);

            ImGui::SeparatorText("sky & time");
            ImGui::Checkbox("day/night cycle", &fx.dayCycle);
            ImGui::BeginDisabled(fx.dayCycle);
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderFloat("time of day", &fx.timeOfDay, 0.0f, 1.0f, "%.2f");
            ImGui::EndDisabled();
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderFloat("day length", &fx.dayLengthSec, 20.0f, 1200.0f, "%.0f s");
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderFloat("night vision", &fx.scotopic, 0.0f, 1.0f, "%.2f");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("rod-vision shift: desaturates and cools the darkest parts\n"
                                  "of the frame after dusk. faded in with darkness, so\n"
                                  "daylight is untouched.");

            ImGui::SeparatorText("weather");
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderFloat("precipitation", &fx.precip, 0.0f, 1.0f, "%.2f");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("rain/snow, and how wet surfaces get. wetness follows the\n"
                                  "rain rather than being its own slider, so the two cannot\n"
                                  "disagree. dry under cover: it reads the voxel sky access.");
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderFloat("snow", &fx.snow, 0.0f, 1.0f, "%.2f");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("0 rain, 1 snow. snow drifts and does not wet anything.");

            ImGui::SeparatorText("exposure & bloom");
            ImGui::Checkbox("auto exposure", &fx.autoExposure);
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderFloat("exposure", &fx.exposure, 0.4f, 2.5f, "%.2f");
            ImGui::Checkbox("bloom", &fx.bloom);
            ImGui::BeginDisabled(!fx.bloom);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderFloat("strength", &fx.bloomStrength, 0.0f, 1.5f, "%.2f");
            ImGui::EndDisabled();
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderFloat("emissive", &fx.emissive, 0.0f, 5.0f, "%.2f");
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderFloat("lens flare", &fx.lensFlare, 0.0f, 2.0f, "%.2f");

            ImGui::SeparatorText("grade");
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderFloat("saturation", &fx.saturation, 0.0f, 2.0f, "%.2f");
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderFloat("contrast", &fx.contrast, 0.5f, 1.8f, "%.2f");
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderFloat("temperature", &fx.temperature, -1.0f, 1.0f, "%.2f");
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderFloat("vignette", &fx.vignette, 0.0f, 1.0f, "%.2f");
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderFloat("grain", &fx.grain, 0.0f, 1.0f, "%.2f");
            if (ImGui::SmallButton("neutral")) {
                fx.saturation = 1.0f;
                fx.contrast = 1.0f;
                fx.temperature = 0.0f;
                fx.vignette = 0.0f;
                fx.grain = 0.0f;
            }
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("audio")) {
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderFloat("master", &fx.volume, 0.0f, 1.0f, "%.2f");
            ImGui::SeparatorText("buses");
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderFloat("world", &fx.volWorld, 0.0f, 1.0f, "%.2f");
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderFloat("interface", &fx.volUi, 0.0f, 1.0f, "%.2f");
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderFloat("ambience", &fx.volAmbient, 0.0f, 1.0f, "%.2f");
            ImGui::SeparatorText("world");
            ImGui::SetNextItemWidth(-125.0f);
            ImGui::SliderFloat("occlusion", &fx.audioOcclusion, 0.0f, 1.0f, "%.2f");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("muffle a sound whose path to your ear is blocked by solid\n"
                                  "blocks. one raycast per sound event, not per frame.");
            ImGui::Spacing();
            ImGui::TextDisabled("nothing ships in sounds/ but CC0 footsteps -\n"
                                "drop your own clips there to hear more.");
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("world")) {
            ImGui::Text("seed %d", worlds.seed);
            ImGui::TextDisabled("(this world is not kept unless you save it)");
            ImGui::Spacing();

            ImGui::SetNextItemWidth(-84.0f);
            ImGui::InputTextWithHint("##wname", "name this world", worlds.saveName,
                                     sizeof(worlds.saveName));
            ImGui::SameLine();
            ImGui::BeginDisabled(worlds.saveName[0] == 0);
            if (ImGui::Button("save", ImVec2(76.0f, 0.0f)))
                worlds.saveRequested = true;
            ImGui::EndDisabled();

            if (!worlds.saved.empty()) {
                ImGui::Spacing();
                ImGui::TextDisabled("saved worlds");
                if (ImGui::BeginListBox("##worlds", ImVec2(-1.0f, 130.0f))) {
                    for (const WorldStore::Entry& w : worlds.saved) {
                        char label[128];
                        std::snprintf(label, sizeof(label), "%s   seed %d   %zu KB",
                                      w.name.c_str(), w.seed, w.bytes / 1024);
                        if (ImGui::Selectable(label, worlds.selected == w.name))
                            worlds.selected = w.name;
                    }
                    ImGui::EndListBox();
                }
                ImGui::BeginDisabled(worlds.selected.empty());
                if (ImGui::Button("load"))
                    worlds.loadRequested = true;
                ImGui::SameLine();
                if (ImGui::Button("delete"))
                    ImGui::OpenPopup("delete world?");
                ImGui::EndDisabled();
                if (ImGui::BeginPopupModal("delete world?", nullptr,
                                           ImGuiWindowFlags_AlwaysAutoResize)) {
                    ImGui::Text("Delete \"%s\" from disk?", worlds.selected.c_str());
                    if (ImGui::Button("delete")) {
                        worlds.deleteRequested = true;
                        ImGui::CloseCurrentPopup();
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("cancel"))
                        ImGui::CloseCurrentPopup();
                    ImGui::EndPopup();
                }
                ImGui::SameLine();
            }

            // Two-step, because it throws away every edit in the live world. Held behind a
            // modal rather than a bare button for the same reason you would not put "delete
            // save" next to "vsync" and trust a steady hand.
            if (ImGui::Button("new world..."))
                ImGui::OpenPopup("new world?");
            if (ImGui::BeginPopupModal("new world?", nullptr,
                                       ImGuiWindowFlags_AlwaysAutoResize)) {
                ImGui::TextUnformatted("Starts a fresh world on a new seed.");
                ImGui::TextDisabled(
                    "Everything you have placed or broken here goes with it, unless\n"
                    "you have saved this world under a name.");
                if (ImGui::Button("new world")) {
                    worlds.resetRequested = true;
                    ImGui::CloseCurrentPopup();
                }
                ImGui::SameLine();
                if (ImGui::Button("cancel"))
                    ImGui::CloseCurrentPopup();
                ImGui::EndPopup();
            }
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("controls")) {
            // Read-only. There is no rebinding, and a list that looks editable but is not
            // would be worse than none.
            static constexpr const char* kKeys[][2] = {
                {"W A S D", "move"},
                {"space / ctrl", "up / down flying, jump when walking"},
                {"shift", "sprint"},
                {"LMB (hold)", "break - blocks have hardness"},
                {"MMB", "place"},
                {"MMB on a lamp", "recolour it"},
                {"scroll", "fly speed, or hotbar when F4 is open"},
                {"1-8", "pick a block (F4)"},
                {"F2", "save edited chunks"},
                {"F3", "diagnostics overlay"},
                {"F4", "hotbar"},
                {"F5", "walk / fly"},
                {"ESC", "this menu (pauses)"},
            };
            if (ImGui::BeginTable("##keys", 2, ImGuiTableFlags_RowBg)) {
                for (const auto& row : kKeys) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(row[0]);
                    ImGui::TableNextColumn();
                    ImGui::TextDisabled("%s", row[1]);
                }
                ImGui::EndTable();
            }
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::Separator();
    if (ImGui::Button("resume", ImVec2(195.0f, 30.0f)))
        open = false;
    ImGui::SameLine();
    if (ImGui::Button("quit", ImVec2(195.0f, 30.0f)))
        quit = true;

    ImGui::End();
}

void DebugHud::render() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

} // namespace vx
