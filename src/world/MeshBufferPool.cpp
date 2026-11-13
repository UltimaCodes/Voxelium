#include "world/MeshBufferPool.hpp"

namespace vx {

std::vector<PackedQuad>* MeshBufferPool::checkout() {
    if (!m_freeList.empty()) {
        std::vector<PackedQuad>* buf = m_freeList.back();
        m_freeList.pop_back();
        return buf;
    }
    m_slots.emplace_back();
    return &m_slots.back();
}

void MeshBufferPool::checkin(std::vector<PackedQuad>* buf) {
    // A small buffer keeps its capacity, which is the reason this exists. A big one gives it back:
    // the next chunk to draw it is usually small, and it stays attached to that chunk for as long
    // as it is loaded, so over a long flight every slot held the largest mesh it ever carried.
    constexpr std::size_t kKeepQuads = 1024;
    if (buf->capacity() > kKeepQuads)
        std::vector<PackedQuad>().swap(*buf);
    else
        buf->clear();
    m_freeList.push_back(buf);
}

} // namespace vx
