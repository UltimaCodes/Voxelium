#pragma once

#include <cassert>
#include <cstdint>

namespace vx {

// One quad = one uint64 in the chunk quad SSBO. chunk.vert unpacks this exact layout —
// keep the two in sync or enjoy the glitch art.
//
//   lo 32: x:5 | y:5 | z:5 | (w-1):5 | (h-1):5 | dir:3 | spare:4
//   hi 32: block:12 | ao:8 (4 corners x 2 bit, M9) | light:8 (block+sky nibbles, M9) | spare:4
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
inline constexpr int kQuadAoShift = 44;
inline constexpr int kQuadLightShift = 52;

static_assert(kQuadDirShift + 3 <= 32, "lo-word fields must not cross the 32-bit line");
static_assert(kQuadBlockShift >= 32 && kQuadLightShift + 8 <= 64, "hi-word overflow");

inline std::uint64_t packQuad(int x, int y, int z, int w, int h, int dir,
                              std::uint16_t block, std::uint32_t ao = 0,
                              std::uint32_t light = 0xFF) {
    assert(x >= 0 && x < 32 && y >= 0 && y < 32 && z >= 0 && z < 32);
    assert(w >= 1 && w <= 32 && h >= 1 && h <= 32);
    assert(dir >= 0 && dir < 6);
    assert(block < 4096);
    return std::uint64_t(x) | (std::uint64_t(y) << kQuadYShift) |
           (std::uint64_t(z) << kQuadZShift) | (std::uint64_t(w - 1) << kQuadWShift) |
           (std::uint64_t(h - 1) << kQuadHShift) | (std::uint64_t(dir) << kQuadDirShift) |
           (std::uint64_t(block) << kQuadBlockShift) | (std::uint64_t(ao) << kQuadAoShift) |
           (std::uint64_t(light) << kQuadLightShift);
}

} // namespace vx
