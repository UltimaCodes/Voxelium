#include "core/Log.hpp"

#include <array>
#include <cstdarg>
#include <cstdio>
#include <mutex>

namespace vx::log {

namespace {
constexpr std::size_t kCapacity = 256;
std::array<Line, kCapacity> g_ring;
std::size_t g_written = 0; // total lines ever written; ring index is modulo
std::mutex g_mutex;
} // namespace

void write(Level level, const char* fmt, ...) {
    Line line{};
    line.level = level;

    va_list args;
    va_start(args, fmt);
    std::vsnprintf(line.text, sizeof(line.text), fmt, args);
    va_end(args);

    std::fprintf(stderr, "%s %s\n", level == Level::Error ? "[error]" : "[info ]", line.text);

    std::lock_guard lock(g_mutex);
    g_ring[g_written % kCapacity] = line;
    ++g_written;
}

void forEach(void (*fn)(const Line&, void*), void* user) {
    std::lock_guard lock(g_mutex);
    const std::size_t count = g_written < kCapacity ? g_written : kCapacity;
    for (std::size_t i = g_written - count; i < g_written; ++i)
        fn(g_ring[i % kCapacity], user);
}

} // namespace vx::log
