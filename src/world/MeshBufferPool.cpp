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
    buf->clear(); // keeps capacity — the reason this exists
    m_freeList.push_back(buf);
}

} // namespace vx
