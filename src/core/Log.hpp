#pragma once

#include <cstdio>

// Minimal logging for the early milestones: fprintf needs nothing, works from any
// thread, and can't allocate. The M2 debug HUD adds an in-app log sink on top; these
// macros stay the front end so call sites never change.
//
// The format string must be a literal (it is concatenated with the prefix).
#define VX_INFO(...)                                  \
    do {                                              \
        std::fprintf(stderr, "[info ] " __VA_ARGS__); \
        std::fputc('\n', stderr);                     \
    } while (0)

#define VX_ERROR(...)                                 \
    do {                                              \
        std::fprintf(stderr, "[error] " __VA_ARGS__); \
        std::fputc('\n', stderr);                     \
    } while (0)
