#pragma once

#include "render/PostFx.hpp"

#include <filesystem>

namespace vx {

// The ESC menu's knobs, persisted as a flat key=value text file so it stays diffable and
// hand-editable. Unknown keys are ignored and missing ones keep their defaults, so an old
// file still loads after settings are added or dropped — no versioning needed.
//
// Deliberately NOT used by --smoke: those runs must render from fixed defaults, or a
// stray settings file on the machine would change the reference image.
namespace Settings {

// Fills the referenced values from `file`. Leaves them untouched if it's missing or
// unreadable, so callers just pass their defaults in. Returns false if nothing was read.
bool load(const std::filesystem::path& file, FxToggles& fx, bool& vsync, bool& caveCull,
          int& radius, float& fovDeg);

// Writes the values to `file`, creating parent directories as needed.
bool save(const std::filesystem::path& file, const FxToggles& fx, bool vsync, bool caveCull,
          int radius, float fovDeg);

} // namespace Settings

} // namespace vx
