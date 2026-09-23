#pragma once

#include <cstddef>

namespace vx {

// A persistently mapped buffer holding kRegions copies of what it carries, written round-robin
// and fenced. With one copy, rewriting it each frame raced the GPU still drawing the frame
// before: under vsync the CPU runs a frame or two ahead, and at radius 32 the previous frame
// was still in flight when the next write began in half of all frames. A region is waited on
// before it is written again; three frames back, that wait is almost always already over.
class MappedRing {
public:
    static constexpr int kRegions = 3;

    bool init(std::size_t regionBytes);
    void shutdown();

    // Moves to the next region, waits until the GPU is done with it, and returns it to write.
    void* begin();
    // Binds the current region as a shader storage buffer.
    void bind(unsigned int binding) const;
    // Call after the last draw that reads the current region.
    void fence();

private:
    unsigned int m_buf = 0;
    unsigned char* m_map = nullptr;
    std::size_t m_stride = 0;
    std::size_t m_bytes = 0;
    int m_cur = 0;
    void* m_fence[kRegions] = {}; // GLsync
};

} // namespace vx
