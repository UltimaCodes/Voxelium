#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace vx {

// Fixed pool of jthreads over a mutex+condvar deque. Jobs are std::function, which
// allocates on submit — that happens on streaming bursts, never in the steady render
// loop, so the zero-alloc rule holds where it matters. Workers never touch GL; results
// travel back through caller-owned queues. Shutdown is jthread stop_token + a condvar
// wake: workers finish their current job and exit, queued jobs are dropped.
class JobSystem {
public:
    JobSystem();
    ~JobSystem();

    void submit(std::function<void()> job);
    int workerCount() const { return int(m_workers.size()); }
    int queued() const;

private:
    void workerLoop(std::stop_token stop);

    mutable std::mutex m_mutex;
    std::condition_variable_any m_cv;
    std::deque<std::function<void()>> m_queue;
    std::vector<std::jthread> m_workers; // last member: joins before the queue dies
};

} // namespace vx
