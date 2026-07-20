#pragma once

namespace vx {

// GPU frame timer using GL_TIME_ELAPSED queries in a small ring. A slot's result is read
// only when the driver says it's available (kFrames of latency by then) — never a sync
// point. If a result isn't ready when its slot comes around again, that sample is
// dropped rather than stalled on.
class GpuTimer {
public:
    void init();
    void begin();
    void end();
    float milliseconds() const { return m_ms; }

private:
    static constexpr int kFrames = 4;
    unsigned int m_queries[kFrames] = {};
    bool m_used[kFrames] = {};
    int m_slot = 0;
    float m_ms = 0.0f;
};

} // namespace vx
