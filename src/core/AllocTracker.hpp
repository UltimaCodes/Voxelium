#pragma once

#include <cstdint>

namespace vx {

// Process-wide counters fed by the global operator new replacements in AllocTracker.cpp.
// The HUD shows the per-frame delta — the render loop target is exactly 0. ImGui goes
// through malloc, not operator new, so the count is engine code only, which is the point.
std::uint64_t heapAllocCount();
std::uint64_t heapAllocBytes();

} // namespace vx
