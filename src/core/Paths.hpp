#pragma once

#include <filesystem>

namespace vx {

// Locates the assets/ directory by walking up from the executable's directory, then from
// the working directory. Dev-time convenience so the binary runs from any build tree or
// IDE without a "working directory" footgun. Returns an empty path if not found.
std::filesystem::path findAssetsDir(const char* argv0);

} // namespace vx
