#pragma once

#include <cstdint>
#include <deque>
#include <vector>

namespace vx {

// A pool of reusable CPU-side quad buffers, so a mesh job's output doesn't cost a fresh
// heap allocation every time it lands.
//
// Every resident chunk keeps its own quad data around for its whole residency — arena
// rebuilds re-upload from it (ChunkManager::rebuildArena) — so unlike the thread_local
// scratch GreedyMesher reuses DURING one mesh call, this buffer has to survive far past the
// job that filled it. The old design copied the scratch into a freshly allocated vector at
// the end of every job (`{scratch.begin(), scratch.end()}`), which is once-per-meshed-chunk,
// not once-per-frame, but under heavy streaming that is still dozens of allocations a
// second — measured as the ~11 heap-allocs/update left over after JobSystem stopped
// allocating on submit (SPEC.md #67).
//
// checkout()/checkin() are main-thread-only, exactly like every other piece of per-chunk
// bookkeeping in ChunkManager. The POINTER checkout() returns is what gets captured into a
// job and handed to a worker — not an index — which is the whole safety argument: a worker
// dereferences it directly and never touches the pool's own structure (the free list, or
// growing the slot storage) at all, so there is nothing for it to race with the main thread
// over. This matters because std::deque's growth guarantee is specifically about
// REFERENCES, not about concurrent structural access — "no references are invalidated" by
// push_back/emplace_back is a standard-mandated guarantee, but that says nothing about two
// threads touching the deque's own bookkeeping at once, which naive index-based access
// (`m_slots[i]` from a worker while the main thread grows the deque) would still race on.
// A pointer obtained on the main thread and only ever dereferenced afterward has no such
// hazard: the vector it points to never moves (never erased, never relocated) for the life
// of the pool, so the deque's own concurrent growth is simply irrelevant to it.
class MeshBufferPool {
public:
    // Returns a pointer to an empty, ready-to-write vector. Valid until checkin() is called
    // on it — capture it into a job and dereference from a worker thread freely; nothing
    // about growing or reusing OTHER slots can move or invalidate this one.
    std::vector<std::uint64_t>* checkout();

    // Clears the vector (capacity survives — that is the entire point) and returns it to
    // the free list. Main thread only, and only once nothing will dereference `buf` again.
    void checkin(std::vector<std::uint64_t>* buf);

    // For --smoke: proves growth stops once the pool has warmed up to a session's actual
    // working set, the same claim JobSystem::m_queue makes about itself.
    std::size_t slotCount() const { return m_slots.size(); }

private:
    std::deque<std::vector<std::uint64_t>> m_slots; // element addresses never move; see above
    std::vector<std::vector<std::uint64_t>*> m_freeList;
};

} // namespace vx
