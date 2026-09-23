#include "gfx/MappedRing.hpp"

#include "core/Log.hpp"

#include <glad/gl.h>

namespace vx {

namespace {
constexpr GLbitfield kFlags = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
} // namespace

bool MappedRing::init(std::size_t regionBytes) {
    GLint align = 256;
    glGetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &align);
    const std::size_t a = std::size_t(align > 0 ? align : 256);
    m_bytes = regionBytes;
    m_stride = (regionBytes + a - 1) / a * a;
    glCreateBuffers(1, &m_buf);
    glNamedBufferStorage(m_buf, GLsizeiptr(m_stride * kRegions), nullptr, kFlags);
    m_map = static_cast<unsigned char*>(
        glMapNamedBufferRange(m_buf, 0, GLsizeiptr(m_stride * kRegions), kFlags));
    if (!m_map) {
        VX_ERROR("mapped ring: persistent mapping failed");
        return false;
    }
    return true;
}

void MappedRing::shutdown() {
    for (void*& f : m_fence)
        if (f) {
            glDeleteSync(static_cast<GLsync>(f));
            f = nullptr;
        }
    if (m_buf)
        glDeleteBuffers(1, &m_buf);
    m_buf = 0;
    m_map = nullptr;
}

void* MappedRing::begin() {
    m_cur = (m_cur + 1) % kRegions;
    if (void*& f = m_fence[m_cur]) {
        // one second, then give up waiting rather than hang: a GPU that far behind is already
        // a frozen frame, and a torn draw is the lesser failure than a dead app
        glClientWaitSync(static_cast<GLsync>(f), GL_SYNC_FLUSH_COMMANDS_BIT, 1'000'000'000);
        glDeleteSync(static_cast<GLsync>(f));
        f = nullptr;
    }
    return m_map + m_stride * std::size_t(m_cur);
}

void MappedRing::bind(unsigned int binding) const {
    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, binding, m_buf,
                      GLintptr(m_stride * std::size_t(m_cur)), GLsizeiptr(m_bytes));
}

void MappedRing::fence() {
    if (m_fence[m_cur])
        glDeleteSync(static_cast<GLsync>(m_fence[m_cur]));
    m_fence[m_cur] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}

} // namespace vx
