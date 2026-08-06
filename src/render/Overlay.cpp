#include "render/Overlay.hpp"

#include "gfx/BlockTextures.hpp"
#include "world/Chunk.hpp"

#include "gfx/Shader.hpp"

#include <glad/gl.h>

namespace vx {

namespace {
// Block id -> texture-array layer for the hotbar icon. Defers to blockTopLayer so that
// every block added since gets an icon automatically; this used to be a hand-written
// switch that had silently fallen behind, leaving gravel, lamps and the ores all drawn as
// the same flat blue "no texture" tint. Water genuinely has no tile, so it keeps the tint.
int iconLayerFor(int block) {
    if (block == blocks::Water || blocks::isWater(BlockId(block)))
        return -1;
    if (block == blocks::Grass)
        return 3; // grassy side reads better in a slot than the flat top
    return blockTopLayer(block);
}
} // namespace

bool Overlay::init(const std::filesystem::path& shaderDir) {
    m_boxProg = loadProgram(shaderDir / "overlay_box.vert", shaderDir / "overlay_box.frag");
    m_crossProg =
        loadProgram(shaderDir / "overlay_cross.vert", shaderDir / "overlay_cross.frag");
    m_hudProg = loadProgram(shaderDir / "hud_quad.vert", shaderDir / "hud_quad.frag");
    m_crackProg =
        loadProgram(shaderDir / "overlay_crack.vert", shaderDir / "overlay_crack.frag");
    if (!m_boxProg || !m_crossProg || !m_hudProg || !m_crackProg)
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

void Overlay::drawBreak(const glm::mat4& viewProj, const glm::vec3& camRelMin,
                        int crackLayer, unsigned int blockTexArray) const {
    glProgramUniformMatrix4fv(m_crackProg, 0, 1, GL_FALSE, &viewProj[0][0]);
    glProgramUniform3fv(m_crackProg, 4, 1, &camRelMin.x);
    glProgramUniform1i(m_crackProg, 5, 0);
    glProgramUniform1f(m_crackProg, 6, float(crackLayer));
    glUseProgram(m_crackProg);
    glBindTextureUnit(0, blockTexArray);
    glBindVertexArray(m_vao);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE); // sits on the block without perturbing the scene depth
    // The six faces are built from hand-written basis tables, and the scene pass culls
    // backfaces — so any face I wound the wrong way silently drew nothing at all, which is
    // a crack overlay that never appears. Twelve triangles do not need culling; turning it
    // off makes the overlay correct regardless of which way each face happens to face.
    glDisable(GL_CULL_FACE);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glEnable(GL_CULL_FACE);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

void Overlay::drawCrosshair(int fbWidth, int fbHeight, const glm::vec4& color) const {
    const glm::vec2 invRes(1.0f / float(fbWidth), 1.0f / float(fbHeight));
    glProgramUniform2fv(m_crossProg, 0, 1, &invRes.x);
    glProgramUniform4fv(m_crossProg, 1, 1, &color.x);
    glUseProgram(m_crossProg);
    glBindVertexArray(m_vao);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    // Straight alpha, not the old GL_ONE_MINUS_DST_COLOR invert. Destination-colour
    // blending against an sRGB target makes NVIDIA patch and recompile the fragment
    // shader on first use (the "program is being recompiled based on GL state" warning),
    // and a fixed colour is what the user actually wants to be able to pick.
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
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
