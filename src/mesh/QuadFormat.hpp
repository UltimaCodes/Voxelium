#pragma once

#include "world/Chunk.hpp"

#include <cassert>
#include <cstdint>

namespace vx {

// One quad = one uvec4 (128 bits, 4 uint32 words) in the chunk quad SSBO. chunk.vert /
// water.vert unpack this exact layout, keep the three in sync or enjoy the glitch art.
//
//   w0: x:5 | y:5 | z:5 | (w-1):5 | (h-1):5 | dir:3 | 4 spare bits
//   w1: block:8 | shade:12 | 12 spare bits
//   w2: light, the raw uint32 from namespace light in Chunk.hpp (r:8, g:8, b:8, sky:8),
//       stored verbatim, no repacking
//   w3: spare, reserved, always 0 for now
//
// This used to be a single uint64 with both words "exactly full", no bit anywhere to
// grow into. Chunk's light word widening from 4 to 8 bits/channel (16 -> 256 levels, the
// real fix for visible light-falloff banding, see Chunk.hpp's light namespace) needed the
// light field to grow from 16 to 32 bits, which nothing here had room for, so the quad
// doubled from 64 to 128 bits instead of stealing bits from something else. Roughly
// doubles per-quad GPU memory (the arena is now ~768 MB fully packed with the default
// kArenaQuads budget); the alternative was moving the quad's own light copy to something
// coarser than what BFS actually computes, which was rejected as fighting the point of the
// widening rather than paying for it.
//
// The extra room also means SHADE no longer spans a word boundary the way it used to (the
// old layout's 4 spare bits weren't enough for 12, so it split 8/4 across the two 32-bit
// halves and every shader had a two-instruction stitch to reassemble it). It fits whole in
// w1 now, one clean unpack, no stitch, and the "no field crosses a word line" rule this
// header always wanted holds again without an exception.
//
// x/y/z is the quad's origin voxel; dir is +X,-X,+Y,-Y,+Z,-Z (0..5); 6 and 7 are the two
// diagonal planes of cross-plane foliage (grass tufts, flowers), which is what the 3-bit
// dir field always had spare for. w runs along z for X faces and x otherwise; h runs along
// y for X/Z faces and z for Y faces. Extents store value-1 so 1..32 fits 5 bits.
//
// SHADE is per-corner vertex brightness: 4 corners x 3 bits, 7 = full. It is the product of
// ambient occlusion and the corner's own light relative to the quad's, which is how a
// merged quad stops being one flat shade with a hard step to its neighbour.
//
// Water quads repurpose the light word's r/g/b bytes (24 bits) to carry per-corner surface
// depth instead of colour, water only ever reads sky back out of it (see
// GreedyMesher::waterCorners and water.vert), so those bytes are free for it. That
// repurposing lives in GreedyMesher, not here; this header just carries whatever 32-bit
// value it's given.

inline constexpr int kQuadYShift = 5;
inline constexpr int kQuadZShift = 10;
inline constexpr int kQuadWShift = 15;
inline constexpr int kQuadHShift = 20;
inline constexpr int kQuadDirShift = 25;
inline constexpr int kQuadBlockShift = 0;  // within w1
inline constexpr int kQuadShadeShift = 8;  // within w1
inline constexpr int kQuadMaxBlockId = 256;

inline constexpr int kQuadShadeBits = 3;              // per corner
inline constexpr std::uint32_t kQuadShadeMax = 7;     // fully lit, unoccluded corner
// every corner at full brightness, the value a greedy-merged quad always carries, since
// a face only merges when it has no corner variation to lose
inline constexpr std::uint32_t kQuadShadeFull = 0xFFFu;

static_assert(kQuadDirShift + 3 <= 32, "dir must not run into w1");
static_assert(blocks::kBlockCount <= kQuadMaxBlockId, "block id no longer fits 8 bits");
static_assert(kQuadShadeBits * 4 == 12, "shade is 4 corners x 3 bits");
static_assert(kQuadShadeShift + 12 <= 32, "shade must fit entirely inside w1");

// 128-bit quad: plain 4x uint32, zero padding, matching uvec4's std430 layout exactly so a
// vector of these can be memcpy'd straight into the SSBO.
struct PackedQuad {
    std::uint32_t w0 = 0, w1 = 0, w2 = 0, w3 = 0;

    friend bool operator==(const PackedQuad&, const PackedQuad&) = default;
};
static_assert(sizeof(PackedQuad) == 16, "PackedQuad must match uvec4 with no padding");

// Pure bit layout, no validation, so it stays usable in a constant expression (MinGW's
// assert is not constexpr in a debug build). packQuad is this plus the range checks.
inline constexpr PackedQuad packQuadBits(int x, int y, int z, int w, int h, int dir,
                                         std::uint16_t block, std::uint32_t shade,
                                         std::uint32_t light) {
    PackedQuad q{};
    q.w0 = std::uint32_t(x) | (std::uint32_t(y) << kQuadYShift) |
          (std::uint32_t(z) << kQuadZShift) | (std::uint32_t(w - 1) << kQuadWShift) |
          (std::uint32_t(h - 1) << kQuadHShift) | (std::uint32_t(dir) << kQuadDirShift);
    q.w1 = (std::uint32_t(block) << kQuadBlockShift) | (shade << kQuadShadeShift);
    q.w2 = light;
    return q;
}

inline PackedQuad packQuad(int x, int y, int z, int w, int h, int dir, std::uint16_t block,
                           std::uint32_t shade = kQuadShadeFull, std::uint32_t light = 0) {
    assert(x >= 0 && x < 32 && y >= 0 && y < 32 && z >= 0 && z < 32);
    assert(w >= 1 && w <= 32 && h >= 1 && h <= 32);
    // 0..5 are the six face directions; 6 and 7 are the cross-plane foliage quads, which
    // is what the third bit was always there for.
    assert(dir >= 0 && dir < 8);
    assert(block < kQuadMaxBlockId);
    assert(shade <= kQuadShadeFull);
    return packQuadBits(x, y, z, w, h, dir, block, shade, light);
}

// Field placement is duplicated in chunk.vert (block, shade, light) and water.vert (shade,
// sky) and nowhere else. Pin every boundary here so a shift that drifts breaks the build
// instead of silently corrupting a neighbouring field.
namespace detail {
// corners 7, 2, 5, 1, deliberately all different, so a mistake in the per-corner unpack
// cannot pass by accident
inline constexpr std::uint32_t kProbeShade = 7u | (2u << 3) | (5u << 6) | (1u << 9);
static_assert(kProbeShade == 0x357u, "probe shade word");
inline constexpr PackedQuad kProbe =
    packQuadBits(1, 2, 3, 4, 5, 5, 200, kProbeShade, light::pack(11, 22, 33, 44));

static_assert((kProbe.w1 >> kQuadBlockShift & 0xFFu) == 200u, "block occupies w1 bits 0..7");
static_assert((kProbe.w1 >> kQuadShadeShift & 0xFFFu) == kProbeShade,
              "shade occupies w1 bits 8..19, whole, no stitch");
static_assert((kProbe.w0 >> kQuadDirShift & 7u) == 5u, "dir survives beside the block/shade word");
static_assert(kProbe.w2 == light::pack(11, 22, 33, 44), "light word rides in w2 verbatim");

// exactly the per-corner unpack chunk.vert and water.vert perform
static_assert(((kProbeShade >> 0) & 7u) == 7u, "shader reads corner 0 at shade>>0");
static_assert(((kProbeShade >> 3) & 7u) == 2u, "shader reads corner 1 at shade>>3");
static_assert(((kProbeShade >> 6) & 7u) == 5u, "shader reads corner 2 at shade>>6");
static_assert(((kProbeShade >> 9) & 7u) == 1u, "shader reads corner 3 at shade>>9");

// what chunk.vert actually does for light, spelled out in the same shifts it uses
static_assert((kProbe.w2 >> 0 & 0xFFu) == 11u, "shader reads red at w2 bits 0..7");
static_assert((kProbe.w2 >> 8 & 0xFFu) == 22u, "shader reads green at w2 bits 8..15");
static_assert((kProbe.w2 >> 16 & 0xFFu) == 33u, "shader reads blue at w2 bits 16..23");
static_assert((kProbe.w2 >> 24 & 0xFFu) == 44u, "shader reads sky at w2 bits 24..31");

// a merged quad is always fully bright at every corner
static_assert(((packQuadBits(0, 0, 0, 1, 1, 0, 1, kQuadShadeFull, 0).w1 >> kQuadShadeShift) &
              0xFFFu) == 0xFFFu,
              "full shade fills all 12 bits");
} // namespace detail

} // namespace vx
