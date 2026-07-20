#pragma once

#include <filesystem>

namespace vx {

// Compiles a vertex+fragment shader pair from files into a linked GL program.
// Returns 0 on failure (errors logged with file names and info logs).
//
// Shaders declare explicit layout(location = N) for attributes and uniforms, so there
// are no glGetUniformLocation string lookups anywhere in the engine.
unsigned int loadProgram(const std::filesystem::path& vertPath,
                         const std::filesystem::path& fragPath);

} // namespace vx
