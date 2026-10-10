#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <new>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace vx {

// The closure JobSystem queues.
//
// Jobs used to be std::function<void()>, which heap-allocates its closure the moment it
// doesn't fit the library's small-object buffer, a couple of pointers, at most. Every real
// job here carries `this` plus a Chunk* plus a ChunkPos at minimum (32+ bytes), so submit()
// was allocating on EVERY call. The comment that used to sit here called this fine because
// it "happens on streaming bursts, never in the steady render loop", true right up until a
// burst is long-lived, which is exactly what heavy streaming under load is: dozens of jobs
// submitted per frame for as long as the burst runs. That is the "heap allocs/frame" the
// debug HUD flags in red, not a one-off.
//
// Job stores its closure inline instead, no allocation, ever, at the cost of a fixed
// capture budget enforced at compile time. It also refuses (statically) any closure that
// isn't trivially copyable and trivially destructible: Job never calls a destructor and
// moves a queued job by copying raw bytes, so a captured std::string / std::vector /
// std::filesystem::path would be silently leaked or double-freed rather than caught. Capture
// pointers and indices into storage the caller keeps alive for the job's lifetime instead,
// every job already does this for the Chunk it operates on, via retainJob()/releaseJob().
class Job {
public:
    template <typename F>
        requires(!std::is_same_v<std::remove_cvref_t<F>, Job>)
    Job(F f) {
        static_assert(sizeof(F) <= kInlineBytes,
                     "job closure too large for Job's inline buffer, capture pointers or "
                     "indices instead of owned data, or raise kInlineBytes");
        static_assert(alignof(F) <= kInlineAlign, "job closure over-aligned for Job");
        static_assert(std::is_trivially_copyable_v<F> && std::is_trivially_destructible_v<F>,
                     "Job never runs a destructor and relocates by copying raw bytes, "
                     "capture raw pointers/indices, never an owning type");
        ::new (static_cast<void*>(m_storage)) F(std::move(f));
        m_invoke = [](const std::byte* s) {
            (*std::launder(reinterpret_cast<const F*>(s)))();
        };
    }

    void operator()() const { m_invoke(m_storage); }

private:
    // 136 bytes covers ChunkManager's largest job today (the mesh job, which carries ten
    // neighbour pointers, six faces and the four diagonals the water's wave corners read,
    // alongside the usual this/Chunk*/ChunkPos and its two output buffers); 160 leaves a little
    // headroom without costing anything a queue of pointer-sized jobs would not already pay.
    static constexpr std::size_t kInlineBytes = 160;
    static constexpr std::size_t kInlineAlign = alignof(std::max_align_t);
    alignas(kInlineAlign) std::byte m_storage[kInlineBytes];
    void (*m_invoke)(const std::byte*) = nullptr;
};

// Fixed pool of jthreads over a mutex+condvar queue. Workers never touch GL; results travel
// back through caller-owned queues. Shutdown is jthread stop_token + a condvar wake: workers
// finish their current job and exit, queued jobs are dropped.
class JobSystem {
public:
    JobSystem();
    ~JobSystem();

    void submit(Job job);
    int workerCount() const { return int(m_workers.size()); }
    int queued() const;

private:
    void workerLoop(std::stop_token stop);

    mutable std::mutex m_mutex;
    std::condition_variable_any m_cv;
    // A vector consumed from the front by index, not std::deque<Job>: deque allocates a
    // new fixed-size internal block every few elements as it grows, because its block size
    // targets a roughly fixed BYTE budget regardless of what the element is, fine for the
    // std::function<void()> this queue used to hold (32 bytes), not for Job at its actual
    // size. Measured: 250 allocations per 1000 push+pop cycles at 128 bytes/element, scaling
    // linearly with element size, so no inline-buffer size fixes this, it is the container.
    // A vector reuses the SAME buffer across an entire streaming burst once it has grown to
    // the deepest queue this session has seen; only a new high-water mark allocates, so
    // ordinary steady-state play (repeated bursts of similar size) is allocation-free after
    // warm-up, which is what "heap allocs/frame" being flagged in the debug HUD assumes.
    std::vector<Job> m_queue;
    std::size_t m_head = 0; // logical front; entries before this are already consumed
    std::vector<std::jthread> m_workers; // last member: joins before the queue dies
};

} // namespace vx
