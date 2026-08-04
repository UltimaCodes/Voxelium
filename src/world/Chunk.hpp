#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace vx {

using BlockId = std::uint16_t;

inline constexpr int kChunkSize = 32;
inline constexpr int kChunkSize3 = kChunkSize * kChunkSize * kChunkSize;

// World height in chunks. 8 => 256 blocks of vertical range so terrain isn't capped so
// low. Most vertical chunks are uniform air (above) or uniform stone (below), which cost
// nothing, so this barely moves memory or quad counts — only the surface band meshes.
inline constexpr int kWorldChunksY = 8;
inline constexpr int kWorldHeight = kWorldChunksY * kChunkSize;

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
inline constexpr int kBlockCount = 12;

inline constexpr int luminance(BlockId id) { return id == Glow ? 14 : 0; }

// opaque = blocks light and hides faces behind it. water is see-through (renders in the
// transparent pass, lets light through); leaves are opaque for cheap "fast" foliage.
inline constexpr bool opaque(BlockId id) { return id != Air && id != Water; }
inline constexpr bool isWater(BlockId id) { return id == Water; }
// solid = collides with the player. water is swim-through (no collision).
inline constexpr bool solid(BlockId id) { return opaque(id); }
} // namespace blocks

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

    // Light nibbles: low = block light, high = sky light. Atomic bytes so mesh jobs can
    // read while the main-thread BFS writes — relaxed loads/stores are plain MOVs on
    // x86, so this costs nothing and is formally race-free. A stale read only means the
    // chunk gets remeshed again when the light engine marks it touched.
    //
    // The array is allocated lazily (32 KB a chunk is far too much to hand every chunk
    // up front), so the POINTER is raced on too, not just the bytes: a worker can be in
    // lightAt() while the main thread's first setLight() allocates. It is published with
    // release / read with acquire so a worker that sees the pointer also sees the filled
    // array — otherwise it could read uninitialized heap. Same reason m_uniformLight is
    // atomic: it is the value lightAt() returns before the array exists.
    std::uint8_t lightAt(int x, int y, int z) const {
        const auto* l = m_light.load(std::memory_order_acquire);
        return l ? l[std::size_t(index(x, y, z))].load(std::memory_order_relaxed)
                 : m_uniformLight.load(std::memory_order_relaxed);
    }
    void setLight(int x, int y, int z, std::uint8_t v) {
        ensureLightArray();
        m_light.load(std::memory_order_relaxed)[std::size_t(index(x, y, z))].store(
            v, std::memory_order_relaxed);
    }
    void fillLight(std::uint8_t v) {
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
        auto arr = std::make_unique<std::atomic<std::uint8_t>[]>(kChunkSize3);
        const std::uint8_t fill = m_uniformLight.load(std::memory_order_relaxed);
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
    std::unique_ptr<std::atomic<std::uint8_t>[]> m_lightOwn;
    std::atomic<std::atomic<std::uint8_t>*> m_light{nullptr};
    std::atomic<std::uint8_t> m_uniformLight{0};
    mutable std::atomic<std::uint32_t> m_jobRefs{0};
};

// Deterministic single-chunk test fixture for the mesher oracle (see docs/M3, M4).
void fillTestTerrain(Chunk& chunk);

} // namespace vx
