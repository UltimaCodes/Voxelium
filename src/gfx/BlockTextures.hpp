#pragma once

#include <filesystem>

namespace vx {

// Builds a GL_TEXTURE_2D_ARRAY of 16x16 block textures, no atlas (per the spec: arrays
// don't bleed at mip edges), NEAREST magnification for the crisp pixel look, mipmapped +
// anisotropic minification.
//
// Each layer loads assets/textures/<name>.png if present (dropped in by the user, e.g.
// their own resource pack), otherwise falls back to a procedurally-generated pixel-art
// pattern. No copyrighted assets are bundled; the engine only loads what's on disk.
// Layer/name order matches texLayer() in chunk.vert; keep the two in sync.
unsigned int createBlockTextureArray(const std::filesystem::path& textureDir);

// Break-progress crack stages live at the end of the same array, so the block-break
// overlay samples the texture the world already has bound. crackLayer(0) is the first
// hairline, crackStages() - 1 is about to shatter.
// Representative texture layer for a block, for things outside the chunk shader that need
// to look like that block (particles, icons). Mirrors texLayer() in chunk.vert for the top
// face; keep the two in step.
int blockTopLayer(int blockId);

// The layer that best shows a block on its own (a log's bark, a grass block's side), for icons.
// -1 for water, which has no tile.
int blockIconLayer(int blockId);
// Every tile copied into one 2D texture, for the block picker's icons, and where a block's icon
// sits in it.
unsigned int createIconAtlas(unsigned int blockTexArray);
void iconUv(int blockId, float& u0, float& v0, float& u1, float& v1);

int crackStages();
int crackLayer(int stage);

} // namespace vx
