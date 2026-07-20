#include "gfx/GpuTimer.hpp"

#include <glad/gl.h>

namespace vx {

void GpuTimer::init() { glCreateQueries(GL_TIME_ELAPSED, kFrames, m_queries); }

void GpuTimer::begin() {
    const GLuint query = m_queries[m_slot];
    if (m_used[m_slot]) {
        GLuint available = 0;
        glGetQueryObjectuiv(query, GL_QUERY_RESULT_AVAILABLE, &available);
        if (available) {
            GLuint64 ns = 0;
            glGetQueryObjectui64v(query, GL_QUERY_RESULT, &ns);
            m_ms = float(double(ns) * 1e-6);
        }
    }
    glBeginQuery(GL_TIME_ELAPSED, query);
    m_used[m_slot] = true;
}

void GpuTimer::end() {
    glEndQuery(GL_TIME_ELAPSED);
    m_slot = (m_slot + 1) % kFrames;
}

} // namespace vx
