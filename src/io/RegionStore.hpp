#pragma once

#include "world/Chunk.hpp"
#include "world/ChunkPos.hpp"

#include <ankerl/unordered_dense.h>

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <vector>

namespace vx {

// Disk-backed chunk storage in region files. A region groups kRegionSize x kRegionSize
// columns (unbounded in Y) into one file so the world isn't a swarm of tiny
// files. Regions are held fully in memory once touched and written back on flush().
//
// Thread-safe: worker gen jobs read through load() while the main thread writes through
// save() and flush(), all under one mutex. Region reads are rare (once per region) and
// fast, so the contention is negligible.
class RegionStore {
public:
    static constexpr int kRegionSize = 16; // 16x16 columns per region file

    explicit RegionStore(std::filesystem::path dir);

    // Fills `chunk` from disk if this chunk was saved; returns false if absent (caller
    // then generates it). Safe to call from worker threads.
    bool load(ChunkPos pos, Chunk& chunk);

    // Stores a chunk's serialized bytes for `pos`, marking its region dirty. Main thread.
    void save(ChunkPos pos, std::vector<std::uint8_t> bytes);

    // Writes every dirty region to disk. Main thread; call on save-all and shutdown.
    void flush();

    // Drop every cached region and delete the region files. The save only ever holds the
    // player's diffs from procedural gen, so this is exactly "give me the world the seed
    // describes again" rather than a destructive wipe of anything generated.
    void discardAll();

    // Point at a different world's directory, dropping every cached region first. Nothing
    // on disk is touched — switching worlds must not delete the one being left.
    void setDir(std::filesystem::path dir);
    const std::filesystem::path& dir() const { return m_dir; }

    struct Stats {
        int regionsResident = 0;
        int chunksStored = 0;
        std::size_t bytesStored = 0;
    };
    Stats stats() const;

private:
    struct Region {
        // Sparse, keyed by (chunkY, localX, localZ). It used to be a dense vector sized
        // kRegionSize^2 * kWorldChunksY, which cannot address a world with no ceiling —
        // and would be almost entirely empty even if it could, since only edited chunks
        // are ever stored.
        ankerl::unordered_dense::map<std::int64_t, std::vector<std::uint8_t>> slots;
        bool dirty = false;
    };

    struct RegionKey {
        std::int32_t x, z;
        bool operator==(const RegionKey&) const = default;
    };
    struct RegionKeyHash {
        using is_avalanching = void;
        std::uint64_t operator()(const RegionKey& k) const {
            return ankerl::unordered_dense::detail::wyhash::hash(&k, sizeof(k));
        }
    };

    static RegionKey regionOf(ChunkPos pos);
    // (chunkY, localX, localZ) packed into one key. chunkY is signed and unbounded, so it
    // takes the high bits and the two 0..15 locals sit in the low byte.
    static std::int64_t localKey(ChunkPos pos);
    std::filesystem::path regionPath(RegionKey key) const;
    Region& ensureLoaded(RegionKey key); // caller holds the lock
    void writeRegion(RegionKey key, const Region& region) const;

    std::filesystem::path m_dir;
    mutable std::mutex m_mutex;
    ankerl::unordered_dense::map<RegionKey, Region, RegionKeyHash> m_regions;
};

} // namespace vx
