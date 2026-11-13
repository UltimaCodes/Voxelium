#pragma once

#include <memory>
#include <vector>

namespace vx {

// Slab pool: acquire pops a free object, release pushes it back. Objects are never destroyed
// while the pool lives. A type with a shed() gives up its big buffers on the way back in: the
// pool holds as many objects as were ever out at once, and each one keeping the largest buffers
// it ever had is the pool's high-water mark times the worst case.
template <typename T>
class Pool {
public:
    T* acquire() {
        if (m_free.empty()) {
            auto& slab = m_slabs.emplace_back(std::make_unique<T[]>(kSlabSize));
            for (std::size_t i = 0; i < kSlabSize; ++i)
                m_free.push_back(&slab[i]);
        }
        T* obj = m_free.back();
        m_free.pop_back();
        return obj;
    }

    void release(T* obj) {
        if constexpr (requires { obj->shed(); })
            obj->shed();
        m_free.push_back(obj);
    }

    std::size_t slabCount() const { return m_slabs.size(); }

private:
    static constexpr std::size_t kSlabSize = 128;
    std::vector<std::unique_ptr<T[]>> m_slabs;
    std::vector<T*> m_free;
};

} // namespace vx
