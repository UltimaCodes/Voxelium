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
// columns (all kWorldChunksY tall) into one file so the world isn't a swarm of tiny
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

    struct Stats {
        int regionsResident = 0;
        int chunksStored = 0;
        std::size_t bytesStored = 0;
    };
    Stats stats() const;

private:
    struct Region {
        // one blob per local chunk slot; empty = absent
        std::vector<std::vector<std::uint8_t>> slots;
        bool dirty = false;
        Region() : slots(std::size_t(kRegionSize) * kRegionSize * kWorldChunksY) {}
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
    static int localIndex(ChunkPos pos);
    std::filesystem::path regionPath(RegionKey key) const;
    Region& ensureLoaded(RegionKey key); // caller holds the lock
    void writeRegion(RegionKey key, const Region& region) const;

    std::filesystem::path m_dir;
    mutable std::mutex m_mutex;
    ankerl::unordered_dense::map<RegionKey, Region, RegionKeyHash> m_regions;
};

} // namespace vx
