#pragma once

#include "render/PostFx.hpp"

#include <filesystem>

namespace vx {

// The ESC menu's knobs, persisted as a flat key=value text file so it stays diffable and
// hand-editable. Unknown keys are ignored and missing ones keep their defaults, so an old
// file still loads after settings are added or dropped — no versioning needed.
//
// load, save and the clamp pass all walk ONE schema table (see the .cpp). They used to be
// three independent lists and drifted twice, most recently leaving eight tuned values that
// were read on startup and never written — so they silently reset every launch.
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

// How many settings the schema covers.
int keyCount();

// --- test support -------------------------------------------------------------------
// These exist so the smoke round-trip covers EVERY key by construction rather than the
// handful whoever wrote it remembered to assert. The old version checked 10 of 37 and
// passed for as long as the drift existed, with a comment claiming it checked every field.

// Writes a deterministic non-default value into every setting, guaranteed different from
// what was there. Returns how many it changed — compare against keyCount().
int perturbAll(FxToggles& fx, bool& vsync, bool& caveCull, int& radius, float& fovDeg);

// Counts settings that differ between two sets. Floats compare with a small tolerance,
// since the values make a round trip through decimal text.
int differences(const FxToggles& a, bool vsyncA, bool caveCullA, int radiusA, float fovA,
                const FxToggles& b, bool vsyncB, bool caveCullB, int radiusB, float fovB);

} // namespace Settings

} // namespace vx
