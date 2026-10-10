#include "render/Weather.hpp"

#include "core/Log.hpp"
#include "gfx/Shader.hpp"

#include <glad/gl.h>

namespace vx {

bool Weather::init(const std::filesystem::path& shaderDir) {
    m_prog = loadProgram(shaderDir / "weather.vert", shaderDir / "weather.frag");
    if (!m_prog)
        return false;
    // Empty VAO: there are no attributes, the vertex stage derives everything from
    // gl_VertexID. Same no-VBO approach as the chunk mesh and the particles.
    glCreateVertexArrays(1, &m_vao);
    return true;
}

void Weather::shutdown() {
    glDeleteProgram(m_prog);
    glDeleteVertexArrays(1, &m_vao);
    m_prog = 0;
    m_vao = 0;
}

void Weather::draw(const Params& p) {
    if (p.strength <= 0.001f)
        return;

    glUseProgram(m_prog);
    glBindVertexArray(m_vao);
    glProgramUniformMatrix4fv(m_prog, 0, 1, GL_FALSE, &p.viewProj[0][0]);
    glProgramUniform3fv(m_prog, 4, 1, &p.right.x);
    glProgramUniform3fv(m_prog, 5, 1, &p.up.x);
    glProgramUniform3fv(m_prog, 6, 1, &p.camFrac.x);
    glProgramUniform1f(m_prog, 7, p.time);
    glProgramUniform1f(m_prog, 8, p.snow);
    glProgramUniform1f(m_prog, 9, kExtent);
    glProgramUniform1f(m_prog, 10, p.strength);
    glProgramUniform3fv(m_prog, 11, 1, &p.tint.x);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // Depth test but no write: drops occlude against the world for free, and never against
    // each other, which is what stops a dense field turning into a stack of visible planes.
    glDepthMask(GL_FALSE);
    glDisable(GL_CULL_FACE); // billboards face the camera; winding is not meaningful
    glDrawArrays(GL_TRIANGLES, 0, kDrops * 6);
    glEnable(GL_CULL_FACE);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

} // namespace vx
