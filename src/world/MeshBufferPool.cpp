#include "world/MeshBufferPool.hpp"

namespace vx {

std::vector<std::uint64_t>* MeshBufferPool::checkout() {
    if (!m_freeList.empty()) {
        std::vector<std::uint64_t>* buf = m_freeList.back();
        m_freeList.pop_back();
        return buf;
    }
    m_slots.emplace_back();
    return &m_slots.back();
}

void MeshBufferPool::checkin(std::vector<std::uint64_t>* buf) {
    buf->clear(); // keeps capacity — the reason this exists
    m_freeList.push_back(buf);
}

} // namespace vx
