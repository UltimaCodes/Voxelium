#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace vx {

using BlockId = std::uint16_t;

inline constexpr int kChunkSize = 32;
inline constexpr int kChunkSize3 = kChunkSize * kChunkSize * kChunkSize;

// THERE IS NO WORLD CEILING. ChunkPos.y is a plain int and chunks exist at any Y; the
// resident set is a vertical band around the camera, exactly the way the horizontal radius
// already worked. Nothing below is a bound on where you may build or dig.
//
// kVerticalRadius is how many chunks above and below the camera stay loaded. It is a
// streaming budget, not a limit: walk up and the band comes with you.
inline constexpr int kVerticalRadius = 7; // 7 => 480 blocks of resident column

// Where the GENERATOR puts terrain. This is a property of the noise, not of the world, and
// exists so light and generation can bound their work: the generator never writes a solid
// voxel above kTerrainTop, so everything above it is open sky by construction — which is
// what lets sky light be seeded without a world top to descend from.
inline constexpr int kTerrainTop = 384;
inline constexpr int kTerrainBottom = 0; // bedrock floor; nothing generates below

// Legacy alias. A few places still want "how tall is generated terrain" in chunks.
inline constexpr int kWorldChunksY = kTerrainTop / kChunkSize;
inline constexpr int kWorldHeight = kTerrainTop;

namespace blocks {
inline constexpr BlockId Air = 0;
inline constexpr BlockId Stone = 1;
inline constexpr BlockId Dirt = 2;
inline constexpr BlockId Grass = 3;
inline constexpr BlockId Glow = 4;
inline constexpr BlockId Water = 5;
inline constexpr BlockId Wood = 6;
inline constexpr BlockId Leaves = 7;
inline constexpr BlockId Sand = 8;
inline constexpr BlockId Snow = 9;
inline constexpr BlockId Planks = 10;
inline constexpr BlockId Bricks = 11;
inline constexpr BlockId CoalOre = 12;
inline constexpr BlockId IronOre = 13;
// Flowing water. There is no per-voxel metadata in this engine, so a fluid's depth is
// carried by its block id: Water is a full source, WaterL1..L7 are the shallower steps
// (L1 is the thinnest film). Seven ids is cheap - the palette stores whatever a chunk
// actually uses - and it keeps fluid depth inside the existing voxel storage, the
// serializer and the region format with no format change anywhere.
inline constexpr BlockId WaterL1 = 14;
inline constexpr BlockId WaterL7 = 20;
inline constexpr BlockId Gravel = 21;
// Coloured lamps. One lamp BLOCK as far as the player is concerned — you recolour it in
// place — but the colour has to live somewhere, and this engine has no per-voxel metadata
// at all. So the hue IS the id, the same block-id-as-data trick the water levels use: no
// storage, serializer or region-format change, and a hue costs exactly one id.
//
// Twelve hues at two saturations, plus white. Block light is 4 bits per channel, so the
// emitted colour is coarse before it ever leaves the lamp and finer steps would be
// quantised away by the light itself — 25 is near the point where more ids buy nothing.
// The player picks from these swatches directly rather than from a free colour control, so
// what is offered is exactly what is storable and nothing has to be silently snapped.
inline constexpr int kLampHues = 12;
inline constexpr int kLampShades = 2; // saturated, then washed halfway to white
// +1 for plain white. A wheel of saturated hues alone cannot express "a white lamp", and a
// nearest-colour search over saturated entries answers a pale pick with a vivid one — which
// is exactly what made picking white do nothing sensible.
inline constexpr int kLampCount = kLampHues * kLampShades + 1;
inline constexpr BlockId LampFirst = 22;
inline constexpr BlockId LampLast = LampFirst + kLampCount - 1;
inline constexpr int kBlockCount = 49; // through Flower
inline constexpr bool isLamp(BlockId id) { return id >= LampFirst && id <= LampLast; }

struct LampRgb {
    int r, g, b;
};

// Built rather than written out: the pale row is the saturated row mixed halfway to white,
// so the two can never drift apart, and adding a hue means adding one line above.
inline constexpr std::array<LampRgb, kLampCount> makeLampTable() {
    constexpr int hue[kLampHues][3] = {
        {15, 0, 0},  {15, 7, 0}, {15, 14, 0}, {8, 15, 0},  {0, 15, 0},  {0, 15, 8},
        {0, 14, 14}, {0, 7, 15}, {0, 0, 15},  {8, 0, 15},  {15, 0, 14}, {15, 0, 7},
    };
    std::array<LampRgb, kLampCount> t{};
    for (int i = 0; i < kLampHues; ++i) {
        t[std::size_t(i)] = {hue[i][0], hue[i][1], hue[i][2]};
        t[std::size_t(kLampHues + i)] = {(hue[i][0] + 15) / 2, (hue[i][1] + 15) / 2,
                                         (hue[i][2] + 15) / 2};
    }
    t[std::size_t(kLampHues * kLampShades)] = {15, 15, 15};
    return t;
}
inline constexpr std::array<LampRgb, kLampCount> kLampRGB = makeLampTable();

// Cross-plane foliage: grass tufts and flowers, drawn as two diagonal quads through the
// voxel rather than as a cube. They ride the ordinary opaque quad list using dir 6 and 7,
// which the quad format's 3-bit dir field always had spare — so no third mesh layer, no
// second meta buffer and no extra draw call. Non-opaque, so they neither block light nor
// hide the faces behind them, and non-solid, so you walk through them.
inline constexpr BlockId GrassTuft = 47;
inline constexpr BlockId Flower = 48;
inline constexpr bool isCross(BlockId id) { return id == GrassTuft || id == Flower; }

// Blocks with nothing under them fall. Sand and gravel only — dirt and stone staying put
// is what keeps a cliff a cliff.
inline constexpr bool falls(BlockId id) { return id == Sand || id == Gravel; }

inline constexpr int luminance(BlockId id) { return (id == Glow || isLamp(id)) ? 14 : 0; }

// Seconds of held click to break a block. Ordered by what it feels like to dig, not by
// any real material property: loose ground gives way at once, worked blocks resist, stone
// and ore are the reason you go looking for a better tool later.
inline constexpr float hardness(BlockId id) {
    if (isCross(id))
        return 0.02f; // foliage comes away the instant you touch it
    if (id == Leaves)
        return 0.12f;
    if (id == Sand || id == Gravel || id == Snow)
        return 0.30f;
    if (id == Dirt || id == Grass)
        return 0.42f;
    if (id == Wood || id == Planks || id == Glow || isLamp(id))
        return 0.70f;
    if (id == CoalOre || id == IronOre)
        return 1.60f;
    if (id == Bricks)
        return 1.30f;
    return 1.10f; // stone and anything new: slow enough to notice, short enough to bear
}

// Emitted colour of a light-emitting block, 0..15 per channel. The three channels
// propagate independently, so an emitter's hue survives all the way out to its falloff
// rather than being a white blob tinted at the end. Glow is a warm ember: red carries
// furthest, blue dies first, so its pool goes orange at the edges the way firelight does.
// This table is the entire interface for adding an emitter.
inline constexpr int emitR(BlockId id) {
    return id == Glow ? 14 : (isLamp(id) ? kLampRGB[std::size_t(id - LampFirst)].r : 0);
}
inline constexpr int emitG(BlockId id) {
    return id == Glow ? 10 : (isLamp(id) ? kLampRGB[std::size_t(id - LampFirst)].g : 0);
}
inline constexpr int emitB(BlockId id) {
    return id == Glow ? 5 : (isLamp(id) ? kLampRGB[std::size_t(id - LampFirst)].b : 0);
}
inline constexpr bool emissive(BlockId id) {
    return emitR(id) > 0 || emitG(id) > 0 || emitB(id) > 0;
}

// opaque = blocks light and hides faces behind it. water is see-through (renders in the
// transparent pass, lets light through); leaves are opaque for cheap "fast" foliage.
inline constexpr bool isWater(BlockId id) {
    return id == Water || (id >= WaterL1 && id <= WaterL7);
}
inline constexpr bool opaque(BlockId id) {
    return id != Air && !isWater(id) && !isCross(id);
}

// CUTOUT blocks. The material is opaque where it exists, but the block is full of holes —
// leaves. Three things follow, and they used to be one flag:
//   - it does NOT hide what is behind it, so faces against it are still drawn and you can
//     see into and through a canopy (this is Minecraft's "fancy" leaves)
//   - it does NOT stop light, it attenuates it, so a canopy dapples instead of casting a
//     solid black interior
//   - it DOES still collide; a tree is not walk-through
inline constexpr bool isCutout(BlockId id) { return id == Leaves; }

// Hides the faces behind it, and blocks the cave-cull visibility flood. A cutout does not.
inline constexpr bool occludes(BlockId id) { return opaque(id) && !isCutout(id); }

// Hard stop for light propagation. Cutouts are not — see lightCost.
inline constexpr bool blocksLight(BlockId id) { return occludes(id); }

// What one step THROUGH this voxel costs the light BFS. Ordinary air is 1. Leaves take a
// couple of levels out of whatever crosses them, which is what turns a canopy from a black
// hole into a soft gradient with bright gaps.
inline constexpr int lightCost(BlockId id) { return isCutout(id) ? 3 : 1; }

// 8 = full source, 1 = thinnest film, 0 = not water. The source level never decays; every
// other level is recomputed from its neighbours each fluid tick.
inline constexpr int waterLevel(BlockId id) {
    if (id == Water)
        return 8;
    if (id >= WaterL1 && id <= WaterL7)
        return 1 + int(WaterL7 - id);
    return 0;
}
inline constexpr BlockId waterForLevel(int level) {
    if (level >= 8)
        return Water;
    if (level <= 0)
        return Air;
    return BlockId(WaterL7 - (level - 1));
}
// solid = collides with the player. water is swim-through (no collision).
inline constexpr bool solid(BlockId id) { return opaque(id); }

// Which family of sounds a block belongs to. Minecraft's model, and it is here for the same
// reason it is there: this engine carries data in the block id, so water is seven ids and a
// lamp is twenty-five. Keying audio on the id alone therefore asked for twenty-five copies
// of one clip and left almost every block falling through to the generic sound.
//
// Leaves and foliage answer "grass" on purpose. A canopy and a lawn are the same rustle in
// every game that has bothered to distinguish them, and it is one clip rather than three.
inline constexpr const char* soundMaterial(BlockId id) {
    if (isWater(id))
        return "water";
    if (isLamp(id) || id == Glow)
        return "glass";
    if (isCross(id) || id == Leaves || id == Grass)
        return "grass";
    switch (id) {
        case Wood:
        case Planks: return "wood";
        case Sand: return "sand";
        case Snow: return "snow";
        case Dirt: return "dirt";
        case Gravel: return "gravel";
        default: return "stone"; // stone, bricks, the ores, and anything added later
    }
}
} // namespace blocks

// Packed per-voxel light: 8 bits each of red, green, blue and sky.
//
// Block light gets three channels so a coloured emitter is representable; sky light keeps
// its own because it attenuates differently (full strength straight down) and the sky
// byte alone drives the sealed-chunk shadow skip.
//
// Widened from 4 bits/channel (16 levels) to 8 (256 levels) to fix visible light-falloff
// banding: BFS still drops exactly 1 of 16 discrete STEPS per block (LightEngine's queue
// logic, comparisons and propagation distance are completely unchanged), but
// LightEngine::getLevel/setLevel now translate that 0..15 step count to/from this wider
// 0..255 storage with a linear stored=level*17 (exact: 15*17=255, lossless round trip).
// The banding improvement comes from GreedyMesher's existing per-corner light averaging
// finally landing on genuine intermediate values instead of truncating straight back to
// one of only 16 integers the instant the average is computed — nothing here changes BFS
// itself. This is a wire format shared with chunk.vert / water.vert AND with
// mesh/QuadFormat.hpp's packed quad, which doubled from 64 to 128 bits to fit the wider
// light field with no spare bits to reclaim.
//
// Today the light engine writes r == g == b, so output is bit-identical to the old
// monochrome path.
namespace light {

inline constexpr std::uint32_t pack(int r, int g, int b, int sky) {
    return std::uint32_t((r & 255) | ((g & 255) << 8) | ((b & 255) << 16) |
                          ((sky & 255) << 24));
}
inline constexpr int red(std::uint32_t v) { return v & 255; }
inline constexpr int green(std::uint32_t v) { return (v >> 8) & 255; }
inline constexpr int blue(std::uint32_t v) { return (v >> 16) & 255; }
inline constexpr int sky(std::uint32_t v) { return (v >> 24) & 255; }
// Perceived block-light level, which while r == g == b is just the level back.
inline constexpr int block(std::uint32_t v) {
    const int r = red(v), g = green(v), b = blue(v);
    return r > g ? (r > b ? r : b) : (g > b ? g : b);
}
inline constexpr std::uint32_t withBlock(std::uint32_t v, int level) {
    return pack(level, level, level, sky(v));
}
inline constexpr std::uint32_t withSky(std::uint32_t v, int level) {
    return pack(red(v), green(v), blue(v), level);
}
inline constexpr std::uint32_t kFullSky = pack(0, 0, 0, 255);

// The layout is a wire format shared with chunk.vert / water.vert, so pin it here rather
// than trusting four call sites to agree about which byte is which.
static_assert(pack(1, 2, 3, 4) == 0x04030201, "light byte order is r,g,b,sky low to high");
static_assert(red(pack(1, 2, 3, 4)) == 1 && green(pack(1, 2, 3, 4)) == 2 &&
                  blue(pack(1, 2, 3, 4)) == 3 && sky(pack(1, 2, 3, 4)) == 4,
              "light accessors must invert pack");
static_assert(block(pack(2, 9, 5, 0)) == 9, "block level is the brightest colour channel");
static_assert(withSky(pack(1, 2, 3, 4), 255) == pack(1, 2, 3, 255), "withSky keeps colour");
static_assert(withBlock(pack(1, 2, 3, 4), 7) == pack(7, 7, 7, 4), "withBlock keeps sky");
static_assert(kFullSky == 0xFF000000, "full sky is the top byte");

} // namespace light

// Palette-compressed voxel storage. A uniform chunk (bits == 0) stores one id and no
// index array — most of a world is air or deep stone and must cost nothing. Otherwise
// indices into the palette are bit-packed at ceil(log2(palette size)) bits per voxel,
// growing one bit at a time (repack) as the palette grows.
//
// Packing is straddled: index i lives at bit i*bits and may span two words. The read
// is a multiply and two shifts — no division, no per-width magic numbers. A value's
// second word is only touched when it actually straddles, which the branch guards.
//
// Palette entries are never garbage collected when blocks are overwritten; stale ids
// cost palette bits until a future repack-on-save (M12) compacts them. Fine for now.
class Chunk {
public:
    Chunk() { makeUniform(blocks::Air); }

    // x/y/z in [0,32). Power-of-two size: indexing is shifts, never mul/div.
    static int index(int x, int y, int z) { return x | (z << 5) | (y << 10); }

    void makeUniform(BlockId id) {
        m_palette.clear();
        m_palette.push_back(id);
        m_indices.clear();
        m_bits = 0;
    }

    bool uniform() const { return m_bits == 0; }
    BlockId uniformId() const { return m_palette[0]; }

    BlockId get(int x, int y, int z) const {
        if (m_bits == 0)
            return m_palette[0];
        return m_palette[readIndex(index(x, y, z))];
    }

    void set(int x, int y, int z, BlockId id);

    // In-flight job references (this chunk's own gen/mesh job, or a neighbor's mesh
    // job reading across the border). Workers release on completion; the main thread
    // must not pool-release a chunk while any are held. Voxel data is immutable after
    // generation (until M10 edits), so refs guard lifetime only, not content.
    void retainJob() const { m_jobRefs.fetch_add(1, std::memory_order_relaxed); }
    void releaseJob() const { m_jobRefs.fetch_sub(1, std::memory_order_release); }
    bool hasJobRefs() const { return m_jobRefs.load(std::memory_order_acquire) != 0; }

    // Atomic 32-bit words (see namespace light above) so mesh jobs can
    // read while the main-thread BFS writes — relaxed loads/stores are plain MOVs on
    // x86, so this costs nothing and is formally race-free. A stale read only means the
    // chunk gets remeshed again when the light engine marks it touched.
    //
    // The array is allocated lazily (64 KB a chunk is far too much to hand every chunk
    // up front), so the POINTER is raced on too, not just the bytes: a worker can be in
    // lightAt() while the main thread's first setLight() allocates. It is published with
    // release / read with acquire so a worker that sees the pointer also sees the filled
    // array — otherwise it could read uninitialized heap. Same reason m_uniformLight is
    // atomic: it is the value lightAt() returns before the array exists.
    std::uint32_t lightAt(int x, int y, int z) const {
        const auto* l = m_light.load(std::memory_order_acquire);
        return l ? l[std::size_t(index(x, y, z))].load(std::memory_order_relaxed)
                 : m_uniformLight.load(std::memory_order_relaxed);
    }
    void setLight(int x, int y, int z, std::uint32_t v) {
        ensureLightArray();
        m_light.load(std::memory_order_relaxed)[std::size_t(index(x, y, z))].store(
            v, std::memory_order_relaxed);
    }
    void fillLight(std::uint32_t v) {
        m_uniformLight.store(v, std::memory_order_relaxed);
        if (auto* l = m_light.load(std::memory_order_acquire))
            for (int i = 0; i < kChunkSize3; ++i)
                l[std::size_t(i)].store(v, std::memory_order_relaxed);
    }

    bool paletteHas(BlockId id) const {
        for (const BlockId b : m_palette)
            if (b == id)
                return true;
        return false;
    }

    // Does this chunk contain ANY emitter? The light seeder uses it to skip the per-voxel
    // scan for the overwhelming majority of chunks. Asking the predicate rather than a
    // specific id matters: this was `paletteHas(Glow)`, so the moment a second kind of
    // emitter existed every chunk holding only the new one was skipped and its lamps
    // silently never lit.
    bool paletteHasEmissive() const {
        for (const BlockId b : m_palette)
            if (blocks::emissive(b))
                return true;
        return false;
    }

    int paletteBits() const { return m_bits; }
    std::size_t paletteSize() const { return m_palette.size(); }
    std::size_t storageBytes() const {
        return m_indices.size() * sizeof(std::uint64_t) + m_palette.size() * sizeof(BlockId);
    }

private:
    std::uint32_t readIndex(int voxel) const {
        const std::size_t bitPos = std::size_t(voxel) * std::size_t(m_bits);
        const std::size_t word = bitPos >> 6;
        const unsigned shift = unsigned(bitPos & 63);
        std::uint64_t v = m_indices[word] >> shift;
        if (shift + unsigned(m_bits) > 64)
            v |= m_indices[word + 1] << (64u - shift);
        return std::uint32_t(v) & ((1u << m_bits) - 1u);
    }

    void writeIndex(int voxel, std::uint32_t value);
    std::uint32_t paletteIndexOf(BlockId id); // adds the id, growing bits if needed
    void repackTo(int newBits);

    // Only ever called from the main thread (setLight); readers may be workers, so the
    // array is filled BEFORE the pointer is published.
    void ensureLightArray() {
        if (m_light.load(std::memory_order_relaxed))
            return;
        auto arr = std::make_unique<std::atomic<std::uint32_t>[]>(kChunkSize3);
        const std::uint32_t fill = m_uniformLight.load(std::memory_order_relaxed);
        for (int i = 0; i < kChunkSize3; ++i)
            arr[std::size_t(i)].store(fill, std::memory_order_relaxed);
        m_lightOwn = std::move(arr);
        m_light.store(m_lightOwn.get(), std::memory_order_release);
    }

    std::vector<BlockId> m_palette; // [0] doubles as the uniform id when bits == 0
    std::vector<std::uint64_t> m_indices;
    int m_bits = 0;
    // light array survives pool reuse (fillLight resets values, never frees). m_lightOwn
    // holds it; m_light is the published view workers read.
    std::unique_ptr<std::atomic<std::uint32_t>[]> m_lightOwn;
    std::atomic<std::atomic<std::uint32_t>*> m_light{nullptr};
    std::atomic<std::uint32_t> m_uniformLight{0};
    mutable std::atomic<std::uint32_t> m_jobRefs{0};
};

// Deterministic single-chunk test fixture for the mesher oracle (see docs/M3, M4).
void fillTestTerrain(Chunk& chunk);

} // namespace vx
