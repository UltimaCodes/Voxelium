#include "render/Overlay.hpp"

#include "gfx/Shader.hpp"

#include <glad/gl.h>

namespace vx {

bool Overlay::init(const std::filesystem::path& shaderDir) {
    m_boxProg = loadProgram(shaderDir / "overlay_box.vert", shaderDir / "overlay_box.frag");
    m_crossProg =
        loadProgram(shaderDir / "overlay_cross.vert", shaderDir / "overlay_cross.frag");
    if (!m_boxProg || !m_crossProg)
        return false;
    glCreateVertexArrays(1, &m_vao); // attribute-less; both shaders pull from gl_VertexID
    return true;
}

void Overlay::shutdown() {
    if (m_vao)
        glDeleteVertexArrays(1, &m_vao);
    if (m_boxProg)
        glDeleteProgram(m_boxProg);
    if (m_crossProg)
        glDeleteProgram(m_crossProg);
    m_vao = m_boxProg = m_crossProg = 0;
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

} // namespace vx
