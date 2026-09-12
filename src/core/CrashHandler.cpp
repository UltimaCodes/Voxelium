#include "core/CrashHandler.hpp"

#ifdef _WIN32
#include <windows.h>

#include <dbghelp.h>

#include <cstdio>

namespace {

LONG WINAPI writeMinidump(EXCEPTION_POINTERS* info) {
    // Deliberately minimal and direct-WinAPI rather than the engine's own logging/filesystem
    // helpers: the process is already in a faulted state, and every extra piece of machinery
    // called here is one more thing that can itself be broken by whatever just corrupted.
    CreateDirectoryA("crashes", nullptr); // fine if it already exists; ignore the error either way

    char path[MAX_PATH];
    std::snprintf(path, sizeof(path), "crashes\\crash_%lu.dmp",
                  static_cast<unsigned long>(GetTickCount64() / 1000));

    const HANDLE file = CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                                    FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file != INVALID_HANDLE_VALUE) {
        MINIDUMP_EXCEPTION_INFORMATION mei{};
        mei.ThreadId = GetCurrentThreadId();
        mei.ExceptionPointers = info;
        mei.ClientPointers = FALSE;
        // Normal, not full: enough to see the crashing stack and locals in a debugger
        // without every dump being a multi-GB copy of the whole address space.
        MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), file, MiniDumpNormal,
                          &mei, nullptr, nullptr);
        CloseHandle(file);
        std::fprintf(stderr, "crash: wrote minidump to %s\n", path);
    } else {
        std::fprintf(stderr, "crash: could not open %s for the minidump\n", path);
    }
    return EXCEPTION_EXECUTE_HANDLER; // terminate cleanly rather than hand it to the OS dialog
}

} // namespace

namespace vx {
void installCrashHandler() { SetUnhandledExceptionFilter(writeMinidump); }
} // namespace vx

#else

namespace vx {
void installCrashHandler() {}
} // namespace vx

#endif
