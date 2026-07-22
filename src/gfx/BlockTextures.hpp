#pragma once

namespace vx {

// Builds a GL_TEXTURE_2D_ARRAY of procedurally-generated pixel-art block textures
// (16x16 per layer) — no atlas (per the spec: arrays don't bleed at mip edges), NEAREST
// magnification for the crisp pixel look, mipmapped + anisotropic minification. Textures
// are generated on the CPU so there are no image files or copyrighted assets involved.
//
// Layer order matches texLayer() in chunk.vert; keep the two in sync.
unsigned int createBlockTextureArray();

} // namespace vx
