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

void JobSystem::submit(std::function<void()> job) {
    {
        std::lock_guard lock(m_mutex);
        m_queue.push_back(std::move(job));
    }
    m_cv.notify_one();
}

int JobSystem::queued() const {
    std::lock_guard lock(m_mutex);
    return int(m_queue.size());
}

void JobSystem::workerLoop(std::stop_token stop) {
    while (true) {
        std::function<void()> job;
        {
            std::unique_lock lock(m_mutex);
            if (!m_cv.wait(lock, stop, [this] { return !m_queue.empty(); }))
                return; // stop requested
            job = std::move(m_queue.front());
            m_queue.pop_front();
        }
        job();
    }
}

} // namespace vx
