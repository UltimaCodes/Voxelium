#include "render/Overlay.hpp"

#include "gfx/Shader.hpp"

#include <glad/gl.h>

namespace vx {

namespace {
// block id -> texture-array layer for the hotbar icon (a representative face). -1 means
// no texture (water) -> drawn as a solid tint. mirrors chunk.vert texLayer / BlockTextures.
int iconLayerFor(int block) {
    switch (block) {
        case 1: return 0; // stone
        case 2: return 1; // dirt
        case 3: return 3; // grass -> grassy side reads better than the flat top
        case 4: return 4; // glow
        case 6: return 5; // wood side
        case 7: return 7; // leaves
        case 8: return 8; // sand
        case 9: return 9; // snow
        case 10: return 10; // planks
        case 11: return 11; // bricks
        default: return -1; // water and anything untextured
    }
}
} // namespace

bool Overlay::init(const std::filesystem::path& shaderDir) {
    m_boxProg = loadProgram(shaderDir / "overlay_box.vert", shaderDir / "overlay_box.frag");
    m_crossProg =
        loadProgram(shaderDir / "overlay_cross.vert", shaderDir / "overlay_cross.frag");
    m_hudProg = loadProgram(shaderDir / "hud_quad.vert", shaderDir / "hud_quad.frag");
    if (!m_boxProg || !m_crossProg || !m_hudProg)
        return false;
    glCreateVertexArrays(1, &m_vao); // attribute-less; the shaders pull from gl_VertexID
    return true;
}

void Overlay::shutdown() {
    if (m_vao)
        glDeleteVertexArrays(1, &m_vao);
    if (m_boxProg)
        glDeleteProgram(m_boxProg);
    if (m_crossProg)
        glDeleteProgram(m_crossProg);
    if (m_hudProg)
        glDeleteProgram(m_hudProg);
    m_vao = m_boxProg = m_crossProg = m_hudProg = 0;
}

void Overlay::drawSelection(const glm::mat4& viewProj, const glm::vec3& camRelMin) const {
    glProgramUniformMatrix4fv(m_boxProg, 0, 1, GL_FALSE, &viewProj[0][0]);
    glProgramUniform3fv(m_boxProg, 4, 1, &camRelMin.x);
    glUseProgram(m_boxProg);
    glBindVertexArray(m_vao);
    glDepthMask(GL_FALSE); // occlude against the scene without perturbing its depth
    glLineWidth(2.0f);
    glDrawArrays(GL_LINES, 0, 24);
    glLineWidth(1.0f);
    glDepthMask(GL_TRUE);
}

void Overlay::drawCrosshair(int fbWidth, int fbHeight) const {
    const glm::vec2 invRes(1.0f / float(fbWidth), 1.0f / float(fbHeight));
    glProgramUniform2fv(m_crossProg, 0, 1, &invRes.x);
    glUseProgram(m_crossProg);
    glBindVertexArray(m_vao);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert the background
    glDrawArrays(GL_TRIANGLES, 0, 12);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
}

void Overlay::drawHotbar(int fbWidth, int fbHeight, const int* blockIds, int count,
                         int selected, unsigned int blockTexArray) const {
    const float slot = 46.0f, pad = 5.0f, margin = 18.0f;
    const float totalW = float(count) * slot + float(count + 1) * pad;
    const float left = (float(fbWidth) - totalW) * 0.5f;
    const glm::vec2 invRes(1.0f / float(fbWidth), 1.0f / float(fbHeight));

    glUseProgram(m_hudProg);
    glBindVertexArray(m_vao);
    glProgramUniform2fv(m_hudProg, 1, 1, &invRes.x);
    glBindTextureUnit(0, blockTexArray);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    const auto quad = [&](float x, float y, float w, float h, glm::vec4 col, int layer) {
        const glm::vec4 rect(x, y, w, h);
        glProgramUniform4fv(m_hudProg, 0, 1, &rect.x);
        glProgramUniform4fv(m_hudProg, 2, 1, &col.x);
        glProgramUniform1i(m_hudProg, 3, layer);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    };

    quad(left, margin, totalW, slot + 2.0f * pad, glm::vec4(0.05f, 0.05f, 0.06f, 0.55f), -1);
    if (selected >= 0 && selected < count) {
        const float sx = left + pad + float(selected) * (slot + pad);
        quad(sx - 2.0f, margin + pad - 2.0f, slot + 4.0f, slot + 4.0f,
             glm::vec4(0.95f, 0.96f, 1.0f, 1.0f), -1);
    }
    for (int i = 0; i < count; ++i) {
        const float sx = left + pad + float(i) * (slot + pad);
        const int layer = iconLayerFor(blockIds[i]);
        const glm::vec4 tint =
            layer < 0 ? glm::vec4(0.20f, 0.45f, 0.75f, 1.0f) : glm::vec4(1.0f);
        quad(sx, margin + pad, slot, slot, tint, layer);
    }

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
}

} // namespace vx
