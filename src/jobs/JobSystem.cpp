#include "jobs/JobSystem.hpp"

#include <algorithm>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace vx {

JobSystem::JobSystem() {
    const unsigned hw = std::thread::hardware_concurrency();
    // Two cores stay free: the main thread, and the GL driver's own submission thread. With
    // every core running a worker the render thread gets time-sliced against streaming, which
    // is felt as frame-rate loss exactly while chunks load.
    const unsigned count = std::max(1u, hw > 2 ? hw - 2 : 1u);
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
#ifdef _WIN32
    // Generation and meshing are background work: when a core is contended, the frame wins.
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
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
