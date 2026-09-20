// The title screen and the loading screen. Both are drawn over the live backdrop
// (ui/MenuBackdrop): the game's own sky and far terrain, so the first thing anyone sees is the
// world rather than a grey rectangle. Split from DebugHud.cpp, which is all debug and settings
// tooling, because these are the game's face and want their own look.

#include "ui/DebugHud.hpp"

#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace vx {

namespace {

constexpr ImU32 kAccent = IM_COL32(126, 196, 255, 255);

ImU32 rgba(int r, int g, int b, float a) {
    return IM_COL32(std::clamp(r, 0, 255), std::clamp(g, 0, 255), std::clamp(b, 0, 255),
                    int(std::clamp(a, 0.0f, 1.0f) * 255.0f + 0.5f));
}

float smooth01(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// A value that eases toward 1 while `on` and back toward 0 otherwise, remembered per widget.
float easeTo(ImGuiID id, bool on, float speed = 14.0f) {
    ImGuiStorage* st = ImGui::GetStateStorage();
    float v = st->GetFloat(id, 0.0f);
    const float target = on ? 1.0f : 0.0f;
    v += (target - v) * std::min(1.0f, ImGui::GetIO().DeltaTime * speed);
    st->SetFloat(id, v);
    return v;
}

ImFont* orCurrent(ImFont* f) { return f ? f : ImGui::GetFont(); }

// Text with the letters spaced out, drawn one glyph at a time, with a soft shadow under it.
void trackedText(ImDrawList* dl, ImFont* font, float size, ImVec2 pos, ImU32 col, const char* text,
                 float track, bool shadow = true) {
    font = orCurrent(font);
    float x = pos.x;
    for (const char* c = text; *c; ++c) {
        if (shadow)
            dl->AddText(font, size, ImVec2(x + 2.0f, pos.y + 3.0f), IM_COL32(0, 0, 0, 90), c, c + 1);
        dl->AddText(font, size, ImVec2(x, pos.y), col, c, c + 1);
        x += font->CalcTextSizeA(size, FLT_MAX, 0.0f, c, c + 1).x + track;
    }
}

void glass(ImDrawList* dl, ImVec2 a, ImVec2 b, float rounding, float alpha) {
    dl->AddRectFilled(a, b, rgba(8, 12, 20, 0.66f * alpha), rounding);
    dl->AddRect(a, b, rgba(255, 255, 255, 0.10f * alpha), rounding, 0, 1.0f);
    // a faint bright edge along the top, like light catching the glass
    dl->AddLine(ImVec2(a.x + rounding, a.y + 0.5f), ImVec2(b.x - rounding, a.y + 0.5f),
                rgba(255, 255, 255, 0.16f * alpha), 1.0f);
}

// One entry of the big menu on the left. Returns true when clicked.
bool menuItem(const char* label, ImVec2 pos, bool active, ImFont* font, float size, float alpha) {
    ImGui::PushFont(font, size);
    const ImVec2 ts = ImGui::CalcTextSize(label);
    ImGui::PopFont();

    ImGui::SetCursorScreenPos(ImVec2(pos.x - 24.0f, pos.y - 8.0f));
    ImGui::PushID(label);
    const bool clicked = ImGui::InvisibleButton("##item", ImVec2(ts.x + 84.0f, ts.y + 16.0f));
    const ImGuiID id = ImGui::GetItemID();
    const bool hovered = ImGui::IsItemHovered();
    ImGui::PopID();

    const float h = easeTo(id, hovered);
    const float a = easeTo(id + 1, active, 10.0f);
    const float lit = std::max(h, a);
    ImDrawList* dl = ImGui::GetWindowDrawList();

    const float x = pos.x + 10.0f * lit;
    // the accent bar grows out of the middle of the row
    const float bh = ts.y * 0.78f * lit;
    dl->AddRectFilled(ImVec2(pos.x - 16.0f, pos.y + ts.y * 0.5f - bh * 0.5f),
                      ImVec2(pos.x - 12.0f, pos.y + ts.y * 0.5f + bh * 0.5f),
                      rgba(126, 196, 255, alpha * lit), 2.0f);
    const int c = int(186.0f + 69.0f * lit);
    dl->AddText(font, size, ImVec2(x + 1.5f, pos.y + 2.5f), IM_COL32(0, 0, 0, int(100.0f * alpha)),
                label);
    dl->AddText(font, size, ImVec2(x, pos.y), rgba(c, c + 2, 255, alpha), label);
    return clicked;
}

const char* const kTips[] = {
    "Hold the left mouse button to break a block. The crack shows how far along you are.",
    "The middle mouse button places the selected block; F4 opens the hotbar.",
    "F5 switches between walking and flying.",
    "Sand and snow puff up when you walk on them, and more when you land.",
    "Esc opens settings, where you can save this world or start another.",
    "It rains and clears on its own. Turn auto weather off in settings to set the sky by hand.",
    "F3 shows diagnostics: frame times, chunks, memory.",
    "Middle-click a lamp to pick its colour.",
    "Look past the loaded chunks: the far horizon shows the terrain out to about five kilometres.",
};

} // namespace

void DebugHud::loadingScreen(float progress, const char* title, const char* phase) {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float W = vp->Size.x, H = vp->Size.y;
    const ImVec2 o = vp->Pos;
    const float scale = std::clamp(H / 900.0f, 0.75f, 1.6f);
    const float M = std::max(48.0f, W * 0.07f);

    ImGui::SetNextWindowPos(vp->Pos);
    ImGui::SetNextWindowSize(vp->Size);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
    ImGui::Begin("##loading", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);
    ImDrawList* dl = ImGui::GetBackgroundDrawList();

    // darken the left and the bottom so the text reads over whatever the sky is doing
    dl->AddRectFilledMultiColor(o, ImVec2(o.x + W * 0.62f, o.y + H), rgba(5, 8, 15, 0.74f),
                                rgba(5, 8, 15, 0.0f), rgba(5, 8, 15, 0.0f), rgba(5, 8, 15, 0.74f));
    dl->AddRectFilledMultiColor(ImVec2(o.x, o.y + H * 0.7f), ImVec2(o.x + W, o.y + H),
                                rgba(5, 8, 15, 0.0f), rgba(5, 8, 15, 0.0f), rgba(5, 8, 15, 0.6f),
                                rgba(5, 8, 15, 0.6f));

    const float t = std::clamp(progress, 0.0f, 1.0f);
    const float y0 = o.y + H * 0.50f;

    // the name of what is being made, big
    {
        char up[64];
        std::snprintf(up, sizeof(up), "%s", title);
        for (char* c = up; *c; ++c)
            *c = char(std::toupper(static_cast<unsigned char>(*c)));
        trackedText(dl, m_fontDisplay, 62.0f * scale, ImVec2(o.x + M, y0), rgba(244, 247, 252, 1.0f),
                    up, 5.0f * scale);
    }
    dl->AddRectFilled(ImVec2(o.x + M, y0 + 78.0f * scale), ImVec2(o.x + M + 96.0f * scale, y0 + 81.0f * scale),
                      kAccent, 1.5f);

    // the bar: a track, a fill with a bright head, and the phase and percentage above it
    const float bw = std::min(540.0f * scale, W * 0.44f);
    const float bh = 8.0f * scale;
    const float bx = o.x + M;
    const float by = y0 + 132.0f * scale;
    ImGui::PushFont(m_fontUi, 19.0f * scale);
    const ImVec2 phaseSize = ImGui::CalcTextSize(phase);
    dl->AddText(ImVec2(bx, by - phaseSize.y - 12.0f * scale), rgba(190, 202, 220, 0.95f), phase);
    char pct[16];
    std::snprintf(pct, sizeof(pct), "%d%%", int(t * 100.0f + 0.5f));
    const ImVec2 ps = ImGui::CalcTextSize(pct);
    dl->AddText(ImVec2(bx + bw - ps.x, by - ps.y - 12.0f * scale), rgba(244, 247, 252, 0.95f), pct);
    ImGui::PopFont();
    dl->AddRectFilled(ImVec2(bx, by), ImVec2(bx + bw, by + bh), rgba(255, 255, 255, 0.12f), bh * 0.5f);
    if (t > 0.0f) {
        const float fw = std::max(bh, bw * t);
        dl->AddRectFilledMultiColor(ImVec2(bx, by), ImVec2(bx + fw, by + bh), rgba(84, 160, 235, 1.0f),
                                    rgba(170, 222, 255, 1.0f), rgba(170, 222, 255, 1.0f),
                                    rgba(84, 160, 235, 1.0f));
        const float pulse = 0.55f + 0.45f * std::sin(float(ImGui::GetTime()) * 3.2f);
        dl->AddCircleFilled(ImVec2(bx + fw, by + bh * 0.5f), bh * 1.9f, rgba(190, 226, 255, 0.35f * pulse));
    }

    // a hint that changes now and then, so a long load has something to read
    const int tip = int(ImGui::GetTime() / 7.0) % int(sizeof(kTips) / sizeof(kTips[0]));
    ImGui::PushFont(m_fontUi, 17.0f * scale);
    const float ty = o.y + H - 64.0f * scale;
    dl->AddText(ImVec2(o.x + M, ty), rgba(126, 196, 255, 0.95f), "TIP");
    dl->AddText(ImVec2(o.x + M + 46.0f * scale, ty), rgba(200, 210, 226, 0.92f), kTips[tip]);
    ImGui::PopFont();

    ImGui::End();
    ImGui::PopStyleColor();
}

void DebugHud::titleScreen(TitleUi& ui) {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float W = vp->Size.x, H = vp->Size.y;
    const ImVec2 o = vp->Pos;
    const float scale = std::clamp(H / 900.0f, 0.75f, 1.6f);
    const float M = std::max(48.0f, W * 0.07f);
    if (m_titleStart < 0.0)
        m_titleStart = ImGui::GetTime();
    const float t = float(ImGui::GetTime() - m_titleStart);
    const float intro = smooth01(t / 0.9f);

    // ---- backdrop layer: legibility gradients, the wordmark, small print -------------------
    ImGui::SetNextWindowPos(vp->Pos);
    ImGui::SetNextWindowSize(vp->Size);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
    ImGui::Begin("##titlebg", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);
    ImDrawList* bg = ImGui::GetBackgroundDrawList();
    // opening fade, and the dip while a new seed's terrain is built
    bg->AddRectFilled(o, ImVec2(o.x + W, o.y + H),
                      rgba(4, 6, 12, std::max(1.0f - intro, ui.fade * 0.92f)));
    bg->AddRectFilledMultiColor(o, ImVec2(o.x + W * 0.60f, o.y + H), rgba(5, 8, 15, 0.70f * intro),
                                rgba(5, 8, 15, 0.0f), rgba(5, 8, 15, 0.0f), rgba(5, 8, 15, 0.70f * intro));
    bg->AddRectFilledMultiColor(ImVec2(o.x, o.y + H * 0.72f), ImVec2(o.x + W, o.y + H),
                                rgba(5, 8, 15, 0.0f), rgba(5, 8, 15, 0.0f), rgba(5, 8, 15, 0.55f * intro),
                                rgba(5, 8, 15, 0.55f * intro));

    const float wmSize = 98.0f * scale;
    const float wmY = o.y + H * 0.13f - (1.0f - intro) * 16.0f;
    trackedText(bg, m_fontDisplay, wmSize, ImVec2(o.x + M, wmY), rgba(246, 249, 253, intro), "VOXELIUM",
                8.0f * scale);
    const float lineY = wmY + wmSize * 1.10f;
    bg->AddRectFilled(ImVec2(o.x + M, lineY), ImVec2(o.x + M + 130.0f * scale * intro, lineY + 3.0f),
                      rgba(126, 196, 255, intro), 1.5f);
    ImGui::PushFont(m_fontUi, 20.0f * scale);
    bg->AddText(ImVec2(o.x + M, lineY + 16.0f * scale), rgba(196, 208, 226, 0.9f * intro),
                "a voxel engine, written from scratch");
    ImGui::PopFont();

    ImGui::PushFont(m_fontUi, 15.0f * scale);
    bg->AddText(ImVec2(o.x + M, o.y + H - 40.0f * scale), rgba(170, 184, 206, 0.8f * intro),
                "Esc pauses in game    F3 diagnostics    F5 walk / fly");
    char build[64];
    std::snprintf(build, sizeof(build), "C++20  OpenGL 4.6  %s", __DATE__);
    const ImVec2 bs = ImGui::CalcTextSize(build);
    bg->AddText(ImVec2(o.x + W - M - bs.x, o.y + H - 40.0f * scale), rgba(170, 184, 206, 0.6f * intro),
                build);
    ImGui::PopFont();
    ImGui::End();
    ImGui::PopStyleColor();

    // ---- the menu on the left --------------------------------------------------------------
    ImGui::SetNextWindowPos(vp->Pos);
    ImGui::SetNextWindowSize(vp->Size);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
    ImGui::Begin("##titlemenu", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoBackground);
    {
        const float itemSize = 36.0f * scale;
        const float step = 58.0f * scale;
        float y = o.y + H * 0.46f;
        const float ia = smooth01((t - 0.25f) / 0.6f);
        if (menuItem("New world", ImVec2(o.x + M, y), ui.page == 1, m_fontDisplay, itemSize, ia))
            ui.page = 1;
        y += step;
        if (!ui.saved.empty()) {
            if (menuItem("Continue", ImVec2(o.x + M, y), ui.page == 2, m_fontDisplay, itemSize, ia)) {
                ui.page = 2;
                if (ui.selected.empty())
                    ui.selected = ui.saved.front().name;
            }
            y += step;
        }
        if (menuItem("Settings", ImVec2(o.x + M, y), ui.page == 3, m_fontDisplay, itemSize, ia))
            ui.page = 3;
        y += step;
        if (menuItem("Quit", ImVec2(o.x + M, y), false, m_fontDisplay, itemSize, ia))
            ui.quitRequested = true;
    }
    ImGui::End();
    ImGui::PopStyleColor();

    // the settings page is DebugHud::settings itself, drawn by the caller in this column
    if (ui.page == 3)
        return;

    // ---- the panel on the right ------------------------------------------------------------
    const float px = o.x + std::max(W * 0.50f, M + 420.0f * scale);
    const float pw = std::min(560.0f * scale, o.x + W - M - px);
    const float py = o.y + H * 0.20f;
    const float pad = 28.0f * scale;
    const float pageT = smooth01((t - 0.35f) / 0.6f);
    // the panel slides in a little when a page changes
    ImGuiStorage* st = ImGui::GetStateStorage();
    const ImGuiID pageId = ImGui::GetID("##pagefade");
    const int lastPage = int(st->GetInt(pageId + 7, ui.page));
    if (lastPage != ui.page) {
        st->SetFloat(pageId, 0.0f);
        st->SetInt(pageId + 7, ui.page);
    }
    const float pf = st->GetFloat(pageId, 0.0f);
    st->SetFloat(pageId, std::min(1.0f, pf + ImGui::GetIO().DeltaTime * 5.0f));
    const float pa = smooth01(pf) * pageT;
    const float slide = (1.0f - smooth01(pf)) * 18.0f;

    const float ph = ui.page == 1 ? 402.0f * scale : std::min(H * 0.60f, 520.0f * scale);
    ImGui::SetNextWindowPos(ImVec2(px, py + slide));
    ImGui::SetNextWindowSize(ImVec2(pw, ph));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, pa);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::Begin("##titlepanel", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoScrollbar);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    glass(dl, ImVec2(px, py + slide), ImVec2(px + pw, py + slide + ph), 14.0f * scale, pa);
    const float innerW = pw - 2.0f * pad;
    ImGui::SetCursorPos(ImVec2(pad, pad));

    const auto heading = [&](const char* text) {
        ImGui::PushFont(m_fontDisplay, 36.0f * scale);
        ImGui::TextUnformatted(text);
        ImGui::PopFont();
        const ImVec2 p = ImGui::GetCursorScreenPos();
        dl->AddRectFilled(ImVec2(p.x, p.y + 2.0f), ImVec2(p.x + 56.0f * scale, p.y + 4.5f), kAccent, 1.0f);
        ImGui::Dummy(ImVec2(0, 14.0f * scale));
    };
    const auto accentButton = [&](const char* label, ImVec2 size) {
        ImGui::PushStyleColor(ImGuiCol_Button, rgba(126, 196, 255, 0.92f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, rgba(170, 222, 255, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, rgba(210, 238, 255, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(9, 16, 28, 255));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f * scale);
        ImGui::PushFont(m_fontDisplay, 26.0f * scale);
        const bool r = ImGui::Button(label, size);
        ImGui::PopFont();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(4);
        return r;
    };

    if (ui.page == 1) {
        heading("New world");
        ImGui::PushFont(m_fontUi, 18.0f * scale);
        ImGui::PushStyleColor(ImGuiCol_Text, rgba(184, 198, 218, 1.0f));
        ImGui::PushTextWrapPos(pad + innerW);
        ImGui::SetCursorPosX(pad);
        ImGui::TextWrapped("Every seed is its own world. What is behind this menu is the terrain of "
                           "the seed below: type one, or roll until you like the look of it.");
        ImGui::PopTextWrapPos();
        ImGui::PopStyleColor();
        ImGui::PopFont();
        ImGui::Dummy(ImVec2(0, 16.0f * scale));

        ImGui::SetCursorPosX(pad);
        ImGui::PushFont(m_fontUi, 14.0f * scale);
        ImGui::TextDisabled("SEED");
        ImGui::PopFont();
        ImGui::SetCursorPosX(pad);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(16.0f * scale, 12.0f * scale));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f * scale);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, rgba(3, 6, 12, 0.55f));
        ImGui::PushStyleColor(ImGuiCol_Border, rgba(255, 255, 255, 0.14f));
        const float rollW = 96.0f * scale;
        ImGui::PushFont(m_fontDisplay, 28.0f * scale);
        ImGui::SetNextItemWidth(innerW - rollW - 10.0f * scale);
        const bool enter = ImGui::InputText("##seed", ui.seedText, sizeof(ui.seedText),
                                            ImGuiInputTextFlags_CharsDecimal |
                                                ImGuiInputTextFlags_EnterReturnsTrue);
        const float rowH = ImGui::GetItemRectSize().y;
        ImGui::PopFont();
        ImGui::SameLine(0.0f, 10.0f * scale);
        ImGui::PushFont(m_fontUi, 19.0f * scale);
        if (ImGui::Button("Roll", ImVec2(rollW, rowH)))
            ui.rerollRequested = true;
        ImGui::PopFont();
        ImGui::PopStyleColor(2);
        ImGui::PopStyleVar(3);

        ImGui::SetCursorPosX(pad);
        ImGui::Dummy(ImVec2(0, 4.0f * scale));
        ImGui::SetCursorPosX(pad);
        ImGui::PushFont(m_fontUi, 16.0f * scale);
        ImGui::TextDisabled("The same seed always gives the same world, so you can share it.");
        ImGui::PopFont();

        ImGui::SetCursorPos(ImVec2(pad, ph - pad - 64.0f * scale));
        if (accentButton("Create world", ImVec2(innerW, 64.0f * scale)) || enter)
            ui.newWorldRequested = true;
    } else {
        heading("Saved worlds");
        ImGui::SetCursorPosX(pad);
        const float listH = ph - 2.0f * pad - 66.0f * scale - 60.0f * scale;
        ImGui::PushStyleColor(ImGuiCol_Header, rgba(126, 196, 255, 0.0f));
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, rgba(255, 255, 255, 0.05f));
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, rgba(255, 255, 255, 0.08f));
        ImGui::BeginChild("##worlds", ImVec2(innerW, listH), false, ImGuiWindowFlags_NoBackground);
        ImDrawList* cl = ImGui::GetWindowDrawList();
        const float rowH = 66.0f * scale;
        for (const WorldStore::Entry& w : ui.saved) {
            const ImVec2 p = ImGui::GetCursorScreenPos();
            const float rowW = ImGui::GetContentRegionAvail().x;
            const bool sel = ui.selected == w.name;
            ImGui::PushID(w.name.c_str());
            if (ImGui::Selectable("##row", sel, ImGuiSelectableFlags_AllowDoubleClick, ImVec2(rowW, rowH))) {
                ui.selected = w.name;
                ui.deleteArmed.clear();
                if (ImGui::IsMouseDoubleClicked(0))
                    ui.loadRequested = true;
            }
            ImGui::PopID();
            const float hv = easeTo(ImGui::GetItemID(), ImGui::IsItemHovered());
            const float lit = std::max(hv * 0.5f, sel ? 1.0f : 0.0f);
            cl->AddRectFilled(p, ImVec2(p.x + rowW, p.y + rowH - 6.0f * scale),
                              rgba(126, 196, 255, 0.05f + 0.13f * lit), 10.0f * scale);
            if (sel)
                cl->AddRectFilled(ImVec2(p.x, p.y + 10.0f * scale), ImVec2(p.x + 4.0f, p.y + rowH - 16.0f * scale),
                                  kAccent, 2.0f);
            ImGui::PushFont(m_fontDisplay, 26.0f * scale);
            cl->AddText(ImVec2(p.x + 20.0f * scale, p.y + 8.0f * scale), rgba(244, 247, 252, 1.0f),
                        w.name.c_str());
            ImGui::PopFont();
            char sub[96];
            std::snprintf(sub, sizeof(sub), "seed %d    %zu KB", w.seed, w.bytes / 1024);
            ImGui::PushFont(m_fontUi, 16.0f * scale);
            cl->AddText(ImVec2(p.x + 20.0f * scale, p.y + 8.0f * scale + 32.0f * scale),
                        rgba(170, 184, 206, 1.0f), sub);
            ImGui::PopFont();
        }
        ImGui::EndChild();
        ImGui::PopStyleColor(3);

        ImGui::SetCursorPos(ImVec2(pad, ph - pad - 56.0f * scale));
        ImGui::BeginDisabled(ui.selected.empty());
        const float dw = 150.0f * scale;
        if (accentButton("Load", ImVec2(innerW - dw - 10.0f * scale, 56.0f * scale)))
            ui.loadRequested = true;
        ImGui::SameLine(0.0f, 10.0f * scale);
        const bool armed = !ui.deleteArmed.empty() && ui.deleteArmed == ui.selected;
        ImGui::PushStyleColor(ImGuiCol_Button, armed ? rgba(224, 84, 84, 0.95f) : rgba(255, 255, 255, 0.07f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, armed ? rgba(240, 110, 110, 1.0f) : rgba(224, 84, 84, 0.45f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, rgba(240, 110, 110, 1.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f * scale);
        ImGui::PushFont(m_fontUi, 19.0f * scale);
        if (ImGui::Button(armed ? "Sure?" : "Delete", ImVec2(dw, 56.0f * scale))) {
            if (armed)
                ui.deleteRequested = true;
            else
                ui.deleteArmed = ui.selected;
        }
        ImGui::PopFont();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(3);
        ImGui::EndDisabled();
    }

    ImGui::End();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
}

} // namespace vx
