#pragma once

#include <glm/glm.hpp>

#include <filesystem>

namespace vx {

// The in-world gameplay HUD, drawn with real GL (never ImGui): the block-selection
// wireframe and the centre crosshair. Tiny — two vertex-pulled programs sharing one
// empty VAO.
class Overlay {
public:
    bool init(const std::filesystem::path& shaderDir);
    void shutdown();

    // Wireframe cube around a unit voxel (camera-relative min corner). Draws into the
    // currently bound framebuffer and depth-tests against the scene it's rendered over,
    // so it occludes correctly. Call inside the HDR scene pass.
    void drawSelection(const glm::mat4& viewProj, const glm::vec3& camRelMin) const;

    // Inverted screen-space crosshair at the centre of the current framebuffer. Call
    // after post, on the final image.
    void drawCrosshair(int fbWidth, int fbHeight) const;

private:
    unsigned int m_boxProg = 0;
    unsigned int m_crossProg = 0;
    unsigned int m_vao = 0;
};

} // namespace vx
