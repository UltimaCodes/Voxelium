#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace vx {

// Compiles a vertex+fragment shader pair from files into a linked GL program.
// Returns 0 on failure (errors logged with file names and info logs).
//
// Shaders declare explicit layout(location = N) for attributes and uniforms, so there
// are no glGetUniformLocation string lookups anywhere in the engine.
unsigned int loadProgram(const std::filesystem::path& vertPath,
                         const std::filesystem::path& fragPath);

unsigned int loadComputeProgram(const std::filesystem::path& compPath);

// Points the program-binary cache at a directory and tells the driver it may use all of
// its compiler threads. Call once, before anything loads a shader. Without it everything
// still works, just always from source.
//
// The cache key covers the shader sources AND the GL vendor/renderer/version string, so a
// driver update or a different GPU misses cleanly rather than feeding glProgramBinary a
// blob it will reject. The driver is allowed to reject one anyway (the spec makes no
// promises), and that just falls back to compiling.
void initShaderCache(const std::filesystem::path& dir);
void shutdownShaderCache(); // reports hits/misses

// Two-phase load. submit() starts a program compiling+linking and returns immediately;
// resolve() waits for it. Calling submit on everything first leaves the whole set in
// flight at once, which is the only way the driver's compiler threads have anything to
// overlap — the one-call-at-a-time form blocks on each program's link status before the
// next one is even created.
struct ProgramJob {
    unsigned int program = 0;
    std::uint64_t key = 0; // 0 = don't write a cache entry (it came from one, or failed)
    std::string label;
};

ProgramJob submitProgram(const std::filesystem::path& vertPath,
                         const std::filesystem::path& fragPath);
ProgramJob submitComputeProgram(const std::filesystem::path& compPath);
unsigned int resolveProgram(ProgramJob& job);

} // namespace vx
