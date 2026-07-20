#pragma once

// Logging with two sinks: stderr (immediate) and a fixed in-memory ring the debug HUD
// reads. Ring entries are fixed-size char arrays, so logging never allocates.
// Format string must be a literal.

namespace vx::log {

enum class Level : unsigned char { Info, Error };

struct Line {
    Level level;
    char text[236];
};

#if defined(__GNUC__)
#define VX_FMT_CHECK __attribute__((format(printf, 2, 3)))
#else
#define VX_FMT_CHECK
#endif

void write(Level level, const char* fmt, ...) VX_FMT_CHECK;

// Calls fn for each retained line, oldest first. Locks internally; keep fn cheap.
void forEach(void (*fn)(const Line& line, void* user), void* user);

} // namespace vx::log

#define VX_INFO(...)  vx::log::write(vx::log::Level::Info, __VA_ARGS__)
#define VX_ERROR(...) vx::log::write(vx::log::Level::Error, __VA_ARGS__)
