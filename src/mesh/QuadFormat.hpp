#pragma once

#include "world/Chunk.hpp"

#include <cassert>
#include <cstdint>

namespace vx {

// One quad = one uint64 in the chunk quad SSBO. chunk.vert unpacks this exact layout —
// keep the two in sync or enjoy the glitch art.
//
//   lo 32: x:5 | y:5 | z:5 | (w-1):5 | (h-1):5 | dir:3 | shade high nibble:4
//   hi 32: block:8 | shade low byte:8 | light:16 (r,g,b,sky nibbles)
//
// The light field is the packed uint16 from namespace light in Chunk.hpp, unchanged.
// Widening it from 8 to 16 bits cost nothing: block:12 addressed 4096 ids for the 12 that
// exist, so 8 bits (256) pays for the other byte and the hi word is now exactly full.
//
// SHADE is per-corner vertex brightness: 4 corners x 3 bits, 7 = full. It is the product
// of ambient occlusion and the corner's own light relative to the quad's, which is how a
// merged quad stops being one flat shade with a hard step to its neighbour. One combined
// term rather than two because there is nowhere near enough room for two, and because
// multiplying AO into vertex light is what the technique does anyway.
//
// It is the one field that spans both words: 12 bits do not fit the 8 the old ao field
// had, and the lo word's 4 spare bits were the only ones going. That costs the shader a
// two-instruction stitch and breaks the "no field crosses the 32-bit line" rule this
// header used to keep — deliberately, and asserted below in the exact shifts the shaders
// use, because a silent drift here is invisible until the lighting looks wrong.
//
// x/y/z is the quad's origin voxel; dir is +X,-X,+Y,-Y,+Z,-Z (0..5). w runs along z for X
// faces and x otherwise; h runs along y for X/Z faces and z for Y faces. Extents store
// value-1 so 1..32 fits 5 bits.

inline constexpr int kQuadYShift = 5;
inline constexpr int kQuadZShift = 10;
inline constexpr int kQuadWShift = 15;
inline constexpr int kQuadHShift = 20;
inline constexpr int kQuadDirShift = 25;
inline constexpr int kQuadShadeHiShift = 28; // lo word: shade bits 8..11
inline constexpr int kQuadBlockShift = 32;
inline constexpr int kQuadShadeLoShift = 40; // hi word: shade bits 0..7
inline constexpr int kQuadLightShift = 48;
inline constexpr int kQuadMaxBlockId = 256;

inline constexpr int kQuadShadeBits = 3;              // per corner
inline constexpr std::uint32_t kQuadShadeMax = 7;     // fully lit, unoccluded corner
// every corner at full brightness — the value a greedy-merged quad always carries, since
// a face only merges when it has no corner variation to lose
inline constexpr std::uint32_t kQuadShadeFull = 0xFFFu;

static_assert(kQuadDirShift + 3 <= kQuadShadeHiShift, "dir must not run into the shade nibble");
static_assert(kQuadBlockShift >= 32 && kQuadLightShift + 16 <= 64, "hi-word overflow");
static_assert(blocks::kBlockCount <= kQuadMaxBlockId, "block id no longer fits 8 bits");
static_assert(kQuadShadeBits * 4 == 12, "shade is 4 corners x 3 bits");

// Split a 12-bit shade word into the two places it lives.
inline constexpr std::uint64_t packShadeBits(std::uint32_t shade) {
    return (std::uint64_t(shade & 0xFFu) << kQuadShadeLoShift) |
           (std::uint64_t((shade >> 8) & 0xFu) << kQuadShadeHiShift);
}

// Pure bit layout, no validation, so it stays usable in a constant expression (MinGW's
// assert is not constexpr in a debug build). packQuad is this plus the range checks.
inline constexpr std::uint64_t packQuadBits(int x, int y, int z, int w, int h, int dir,
                                            std::uint16_t block, std::uint32_t shade,
                                            std::uint32_t light) {
    return std::uint64_t(x) | (std::uint64_t(y) << kQuadYShift) |
           (std::uint64_t(z) << kQuadZShift) | (std::uint64_t(w - 1) << kQuadWShift) |
           (std::uint64_t(h - 1) << kQuadHShift) | (std::uint64_t(dir) << kQuadDirShift) |
           (std::uint64_t(block) << kQuadBlockShift) | packShadeBits(shade) |
           (std::uint64_t(light) << kQuadLightShift);
}

inline std::uint64_t packQuad(int x, int y, int z, int w, int h, int dir,
                              std::uint16_t block, std::uint32_t shade = kQuadShadeFull,
                              std::uint32_t light = 0) {
    assert(x >= 0 && x < 32 && y >= 0 && y < 32 && z >= 0 && z < 32);
    assert(w >= 1 && w <= 32 && h >= 1 && h <= 32);
    assert(dir >= 0 && dir < 6);
    assert(block < kQuadMaxBlockId);
    assert(shade <= kQuadShadeFull);
    return packQuadBits(x, y, z, w, h, dir, block, shade, light);
}

// Field placement is duplicated in chunk.vert (block, AO, light), water.vert (sky) and
// nowhere else — the hi word is exactly full, so a shift that drifts silently corrupts a
// neighbouring field instead of overflowing. Pin every boundary.
namespace detail {
// corners 7, 2, 5, 1 — deliberately all different, and spanning the byte/nibble split at
// corner 2 (bits 6..8) so a mistake in the stitch cannot pass
inline constexpr std::uint32_t kProbeShade = 7u | (2u << 3) | (5u << 6) | (1u << 9);
static_assert(kProbeShade == 0x357u, "probe shade word");
inline constexpr std::uint64_t kProbe =
    packQuadBits(1, 2, 3, 4, 5, 5, 200, kProbeShade, light::pack(1, 2, 3, 4));
inline constexpr std::uint32_t kProbeLo = std::uint32_t(kProbe);
inline constexpr std::uint32_t kProbeHi = std::uint32_t(kProbe >> 32);

static_assert(((kProbe >> kQuadBlockShift) & 255u) == 200u, "block occupies hi bits 0..7");
static_assert(((kProbe >> kQuadLightShift) & 0xFFFFu) == light::pack(1, 2, 3, 4),
              "light occupies hi bits 16..31");
static_assert(((kProbeLo >> kQuadDirShift) & 7u) == 5u, "dir survives beside the shade nibble");

// exactly the stitch chunk.vert and water.vert perform, in the shifts they use
inline constexpr std::uint32_t kProbeShade12 =
    ((kProbeHi >> 8) & 0xFFu) | (((kProbeLo >> 28) & 0xFu) << 8);
static_assert(kProbeShade12 == kProbeShade, "shader stitch must rebuild the shade word");
static_assert(((kProbeShade12 >> 0) & 7u) == 7u, "shader reads corner 0 at shade>>0");
static_assert(((kProbeShade12 >> 3) & 7u) == 2u, "shader reads corner 1 at shade>>3");
static_assert(((kProbeShade12 >> 6) & 7u) == 5u, "shader reads corner 2 at shade>>6");
static_assert(((kProbeShade12 >> 9) & 7u) == 1u, "shader reads corner 3 at shade>>9");

// what chunk.vert actually does for light, spelled out in the same shifts it uses
static_assert((kProbeHi >> 16 & 15u) == 1u, "shader reads red at hi>>16");
static_assert((kProbeHi >> 20 & 15u) == 2u, "shader reads green at hi>>20");
static_assert((kProbeHi >> 24 & 15u) == 3u, "shader reads blue at hi>>24");
static_assert((kProbeHi >> 28 & 15u) == 4u, "shader reads sky at hi>>28");

// a merged quad is always fully bright at every corner
static_assert(((packQuadBits(0, 0, 0, 1, 1, 0, 1, kQuadShadeFull, 0) >> 32) >> 8 & 0xFFu) ==
                  0xFFu,
              "full shade fills the hi byte");
} // namespace detail

} // namespace vx
