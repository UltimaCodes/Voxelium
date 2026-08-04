#pragma once

#include "world/Chunk.hpp"

#include <cassert>
#include <cstdint>

namespace vx {

// One quad = one uint64 in the chunk quad SSBO. chunk.vert unpacks this exact layout —
// keep the two in sync or enjoy the glitch art.
//
//   lo 32: x:5 | y:5 | z:5 | (w-1):5 | (h-1):5 | dir:3 | spare:4
//   hi 32: block:8 | ao:8 (4 corners x 2 bit, M9) | light:16 (r,g,b,sky nibbles)
//
// The light field is the packed uint16 from namespace light in Chunk.hpp, unchanged.
// Widening it from 8 to 16 bits cost nothing: block:12 addressed 4096 ids for the 12 that
// exist, so 8 bits (256) pays for the other byte and the hi word is now exactly full.
//
// No field crosses the 32-bit boundary, so the shader reads a uvec2 and never has to
// stitch bits across words. x/y/z is the quad's origin voxel; dir is +X,-X,+Y,-Y,+Z,-Z
// (0..5). w runs along z for X faces and x otherwise; h runs along y for X/Z faces and
// z for Y faces. Extents store value-1 so 1..32 fits 5 bits.

inline constexpr int kQuadYShift = 5;
inline constexpr int kQuadZShift = 10;
inline constexpr int kQuadWShift = 15;
inline constexpr int kQuadHShift = 20;
inline constexpr int kQuadDirShift = 25;
inline constexpr int kQuadBlockShift = 32;
inline constexpr int kQuadAoShift = 40;
inline constexpr int kQuadLightShift = 48;
inline constexpr int kQuadMaxBlockId = 256;

static_assert(kQuadDirShift + 3 <= 32, "lo-word fields must not cross the 32-bit line");
static_assert(kQuadBlockShift >= 32 && kQuadLightShift + 16 <= 64, "hi-word overflow");
static_assert(blocks::kBlockCount <= kQuadMaxBlockId, "block id no longer fits 8 bits");

// Pure bit layout, no validation, so it stays usable in a constant expression (MinGW's
// assert is not constexpr in a debug build). packQuad is this plus the range checks.
inline constexpr std::uint64_t packQuadBits(int x, int y, int z, int w, int h, int dir,
                                            std::uint16_t block, std::uint32_t ao,
                                            std::uint32_t light) {
    return std::uint64_t(x) | (std::uint64_t(y) << kQuadYShift) |
           (std::uint64_t(z) << kQuadZShift) | (std::uint64_t(w - 1) << kQuadWShift) |
           (std::uint64_t(h - 1) << kQuadHShift) | (std::uint64_t(dir) << kQuadDirShift) |
           (std::uint64_t(block) << kQuadBlockShift) | (std::uint64_t(ao) << kQuadAoShift) |
           (std::uint64_t(light) << kQuadLightShift);
}

inline std::uint64_t packQuad(int x, int y, int z, int w, int h, int dir,
                              std::uint16_t block, std::uint32_t ao = 0,
                              std::uint32_t light = 0) {
    assert(x >= 0 && x < 32 && y >= 0 && y < 32 && z >= 0 && z < 32);
    assert(w >= 1 && w <= 32 && h >= 1 && h <= 32);
    assert(dir >= 0 && dir < 6);
    assert(block < kQuadMaxBlockId);
    return packQuadBits(x, y, z, w, h, dir, block, ao, light);
}

// Field placement is duplicated in chunk.vert (block, AO, light), water.vert (sky) and
// nowhere else — the hi word is exactly full, so a shift that drifts silently corrupts a
// neighbouring field instead of overflowing. Pin every boundary.
namespace detail {
inline constexpr std::uint64_t kProbe =
    packQuadBits(1, 2, 3, 4, 5, 6, 200, 0xA5u, light::pack(1, 2, 3, 4));
static_assert(((kProbe >> kQuadBlockShift) & 255u) == 200u, "block occupies hi bits 0..7");
static_assert(((kProbe >> kQuadAoShift) & 255u) == 0xA5u, "ao occupies hi bits 8..15");
static_assert(((kProbe >> kQuadLightShift) & 0xFFFFu) == light::pack(1, 2, 3, 4),
              "light occupies hi bits 16..31");
// what chunk.vert actually does, spelled out in the same shifts it uses
static_assert(((kProbe >> 32) >> 16 & 15u) == 1u, "shader reads red at hi>>16");
static_assert(((kProbe >> 32) >> 20 & 15u) == 2u, "shader reads green at hi>>20");
static_assert(((kProbe >> 32) >> 24 & 15u) == 3u, "shader reads blue at hi>>24");
static_assert(((kProbe >> 32) >> 28 & 15u) == 4u, "shader reads sky at hi>>28");
} // namespace detail

} // namespace vx
