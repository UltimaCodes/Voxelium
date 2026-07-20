#include "core/AllocTracker.hpp"

#include <atomic>
#include <cstdlib>
#include <new>

namespace {

std::atomic<std::uint64_t> g_count{0};
std::atomic<std::uint64_t> g_bytes{0};

void* countedAlloc(std::size_t size) {
    g_count.fetch_add(1, std::memory_order_relaxed);
    g_bytes.fetch_add(size, std::memory_order_relaxed);
    return std::malloc(size == 0 ? 1 : size);
}

} // namespace

namespace vx {
std::uint64_t heapAllocCount() { return g_count.load(std::memory_order_relaxed); }
std::uint64_t heapAllocBytes() { return g_bytes.load(std::memory_order_relaxed); }
} // namespace vx

// Global replacements. Engine policy is no exceptions, so OOM aborts instead of throwing
// bad_alloc. Aligned-new overloads are deliberately not replaced — they pair with the
// library defaults and nothing engine-side is over-aligned; revisit if that changes.

void* operator new(std::size_t size) {
    void* p = countedAlloc(size);
    if (!p)
        std::abort();
    return p;
}

void* operator new[](std::size_t size) {
    void* p = countedAlloc(size);
    if (!p)
        std::abort();
    return p;
}

void* operator new(std::size_t size, const std::nothrow_t&) noexcept { return countedAlloc(size); }
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept { return countedAlloc(size); }

void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
