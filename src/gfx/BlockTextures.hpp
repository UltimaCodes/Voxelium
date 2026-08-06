#pragma once

#include <filesystem>

namespace vx {

// Builds a GL_TEXTURE_2D_ARRAY of 16x16 block textures — no atlas (per the spec: arrays
// don't bleed at mip edges), NEAREST magnification for the crisp pixel look, mipmapped +
// anisotropic minification.
//
// Each layer loads assets/textures/<name>.png if present (dropped in by the user — e.g.
// their own resource pack), otherwise falls back to a procedurally-generated pixel-art
// pattern. No copyrighted assets are bundled; the engine only loads what's on disk.
// Layer/name order matches texLayer() in chunk.vert; keep the two in sync.
unsigned int createBlockTextureArray(const std::filesystem::path& textureDir);

// Break-progress crack stages live at the end of the same array, so the block-break
// overlay samples the texture the world already has bound. crackLayer(0) is the first
// hairline, crackStages() - 1 is about to shatter.
int crackStages();
int crackLayer(int stage);

} // namespace vx
