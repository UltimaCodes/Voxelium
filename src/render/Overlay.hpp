#pragma once

#include <glm/glm.hpp>

#include <filesystem>

namespace vx {

// The in-world gameplay HUD, drawn with real GL (never ImGui): the block-selection
// wireframe and the centre crosshair. Tiny, two vertex-pulled programs sharing one
// empty VAO.
class Overlay {
public:
    bool init(const std::filesystem::path& shaderDir);
    void shutdown();

    // Wireframe cube around a unit voxel (camera-relative min corner). Draws into the
    // currently bound framebuffer and depth-tests against the scene it's rendered over,
    // so it occludes correctly. Call inside the HDR scene pass.
    void drawSelection(const glm::mat4& viewProj, const glm::vec3& camRelMin) const;

    // Crack overlay on the block being broken: the six faces of the voxel, drawn a hair
    // proud of it with the crack stage from the block texture array. Same pass as the
    // selection wireframe.
    void drawBreak(const glm::mat4& viewProj, const glm::vec3& camRelMin, int crackLayer,
                   unsigned int blockTexArray) const;

    // Screen-space crosshair at the centre of the current framebuffer, in `color`
    // (straight alpha). Call after post, on the final image.
    void drawCrosshair(int fbWidth, int fbHeight, const glm::vec4& color) const;

    // Minecraft-style hotbar along the bottom: a panel, block-texture icons for each of
    // `blockIds`, and a frame on the selected slot. Samples `blockTexArray` for icons.
    void drawHotbar(int fbWidth, int fbHeight, const int* blockIds, int count, int selected,
                    unsigned int blockTexArray) const;

private:
    unsigned int m_boxProg = 0;
    unsigned int m_crackProg = 0;
    unsigned int m_crossProg = 0;
    unsigned int m_hudProg = 0;
    unsigned int m_vao = 0;
};

} // namespace vx
