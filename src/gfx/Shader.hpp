#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

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
// overlap, the one-call-at-a-time form blocks on each program's link status before the
// next one is even created.
struct ProgramJob {
    unsigned int program = 0;
    std::uint64_t key = 0; // 0 = don't write a cache entry (it came from one, or failed)
    std::string label;
};

// `defines` is injected verbatim after the `#version` line, e.g. "#define VX_QUALITY 3\n".
//
// This is how the quality presets reach the shaders. The alternative was uniform loop
// bounds, but a dynamic bound does not unroll, so it would have made the TOP preset slower
// than the build had no presets at all, shipping a quality system that regresses Ultra is
// the wrong trade.
//
// It needs no new bookkeeping because the cache key already hashes the FULLY EXPANDED source
// (see readShaderSource: `#include` is resolved before hashing, deliberately). Injecting a
// define changes that text, so every preset gets its own cache entry and its own binary for
// free, and switching preset can never hand the driver a blob built for another one.
ProgramJob submitProgram(const std::filesystem::path& vertPath,
                         const std::filesystem::path& fragPath,
                         std::string_view defines = {});
// Vertex, tessellation control, tessellation evaluation and fragment, for the water surface.
ProgramJob submitTessProgram(const std::filesystem::path& vertPath,
                             const std::filesystem::path& tescPath,
                             const std::filesystem::path& tesePath,
                             const std::filesystem::path& fragPath, std::string_view defines = {});
ProgramJob submitComputeProgram(const std::filesystem::path& compPath,
                                std::string_view defines = {});
unsigned int resolveProgram(ProgramJob& job);

} // namespace vx
