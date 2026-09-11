#include "jobs/JobSystem.hpp"

#include <algorithm>

namespace vx {

JobSystem::JobSystem() {
    const unsigned hw = std::thread::hardware_concurrency();
    const unsigned count = std::max(1u, hw > 0 ? hw - 1 : 3u); // one core stays with the main thread
    m_workers.reserve(count);
    for (unsigned i = 0; i < count; ++i)
        m_workers.emplace_back([this](std::stop_token stop) { workerLoop(stop); });
}

JobSystem::~JobSystem() {
    for (auto& w : m_workers)
        w.request_stop();
    m_cv.notify_all();
    // jthread joins in its destructor
}

void JobSystem::submit(Job job) {
    {
        std::lock_guard lock(m_mutex);
        m_queue.push_back(std::move(job));
    }
    m_cv.notify_one();
}

int JobSystem::queued() const {
    std::lock_guard lock(m_mutex);
    return int(m_queue.size() - m_head);
}

void JobSystem::workerLoop(std::stop_token stop) {
    while (true) {
        std::unique_lock lock(m_mutex);
        if (!m_cv.wait(lock, stop, [this] { return m_head < m_queue.size(); }))
            return; // stop requested
        Job job = std::move(m_queue[m_head]);
        ++m_head;
        // Compact once consumed entries are past a floor AND the majority of the buffer —
        // the floor keeps a small, bursty queue from compacting on nearly every pop; the
        // ratio bounds how far m_head (and so m_queue) can grow over a long session. Erasing
        // a trivially-copyable prefix is a memmove of whatever is left, not a reallocation,
        // so this never costs the allocation it exists to avoid.
        if (m_head > 64 && m_head * 2 > m_queue.size()) {
            m_queue.erase(m_queue.begin(), m_queue.begin() + std::ptrdiff_t(m_head));
            m_head = 0;
        }
        lock.unlock(); // never hold the mutex while a worker actually runs the job
        job();
    }
}

} // namespace vx
