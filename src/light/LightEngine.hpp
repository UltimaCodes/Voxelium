#pragma once

#include "world/Chunk.hpp"
#include "world/ChunkPos.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

namespace vx {

class ChunkManager;

// Block + sky light over the loaded world. All propagation runs on the main thread
// through explicit BFS queues under a per-frame time budget — light is memory-bound
// pointer chasing that's miserable to parallelize safely, and budgeted-incremental
// means it can never stall a frame. Chunks whose light changes land in touched() and
// get remeshed through the normal pipeline.
//
// Sky light: full strength straight down (seeding walks each column from the top of
// the world), attenuates by 1 per step in every other direction. Removal is the pass
// everyone gets wrong: its own BFS that zeroes everything the removed source fed,
// collecting still-lit boundary voxels to re-propagate from.
class LightEngine {
public:
    explicit LightEngine(ChunkManager& world) : m_world(world) {}

    // Seeds sky columns + emissive blocks for one 32x32 column of chunks (all
    // kWorldChunksY of them at once — sky descent doesn't care about chunk seams).
    // Runs to completion; fine for tests and one-off re-seeds.
    void seedColumn(std::int32_t chunkX, std::int32_t chunkZ);

    // Incremental form. A column is 32x32x256 voxels and costs 5-10 ms as one lump,
    // which no per-frame budget can slice — it was the largest remaining streaming
    // hitch. begin/step spreads it over frames, splitting at (dx,dz) cell granularity.
    // The caller must not treat the column as lit until step returns true.
    void beginSeedColumn(std::int32_t chunkX, std::int32_t chunkZ);
    bool stepSeedColumn(double budgetMs); // true = column finished
    bool seedInProgress() const { return m_seed.phase != SeedPhase::Idle; }
    void abortSeedColumn() { m_seed.phase = SeedPhase::Idle; }
    std::int32_t seedColumnX() const { return m_seed.chunkX; }
    std::int32_t seedColumnZ() const { return m_seed.chunkZ; }

    // Call on any voxel edit, after the chunk itself was updated.
    void onBlockChanged(const glm::ivec3& wp, BlockId oldId, BlockId newId);

    void process(double budgetMs);
    bool idle() const {
        return m_queue[0].empty() && m_queue[1].empty() && m_removal[0].empty() &&
               m_removal[1].empty();
    }

    // Drained by the manager. Clearing goes through clearTouched() rather than handing
    // out a mutable vector, because the dedup set has to be dropped with it.
    const std::vector<ChunkPos>& touched() const { return m_touched; }
    void clearTouched() {
        m_touched.clear();
        m_touchedSet.clear();
    }

    // debug/verification
    std::uint8_t lightAt(const glm::ivec3& wp) const;
    int queueDepth() const {
        return int(m_queue[0].size() + m_queue[1].size() + m_removal[0].size() +
                   m_removal[1].size());
    }

private:
    enum Channel : int { Block = 0, Sky = 1 };

    struct Removal {
        glm::ivec3 pos;
        std::uint8_t oldLevel;
    };

    // Direct-mapped chunk lookup cache. The BFS probes six neighbours per voxel and
    // nearly all of them land in the chunk it just read, yet every probe went through a
    // wyhash of the chunk coords plus a map lookup — up to nineteen of them per voxel
    // once blockAt, getLevel and setLevel are counted. Slotting on the low coord bits
    // makes that an index and a compare instead. A 4x4 horizontal by 8 vertical block of
    // chunks maps to distinct slots, which is wider than a BFS front ever spans, so the
    // hot working set never collides with itself.
    //
    // Caching a null is the point as much as caching a hit: probes off the loaded edge
    // are common and used to pay a full lookup each time to be told "nothing there".
    //
    // Only valid within one main-thread entry point. Between calls the chunk set moves —
    // eviction, and the coarse->detailed re-gen swapping a chunk's voxels out — so a
    // pointer held across one would be the same use-after-free seedStackValid refetches
    // to avoid. Every public entry point drops the cache on the way in.
    static constexpr int kChunkCacheSize = 128;
    struct ChunkCacheEntry {
        ChunkPos key{};
        Chunk* value = nullptr;
        bool valid = false;
    };
    static int cacheSlot(ChunkPos p) {
        return (p.x & 3) | ((p.z & 3) << 2) | ((p.y & 7) << 4);
    }
    mutable ChunkCacheEntry m_chunkCache[kChunkCacheSize];
    Chunk* lookupChunk(ChunkPos p) const;
    void resetChunkCache();

    BlockId blockAt(const glm::ivec3& wp) const;
    int getLevel(const glm::ivec3& wp, int channel) const;
    void setLevel(const glm::ivec3& wp, int channel, int level);
    void enqueue(int channel, const glm::ivec3& wp) { m_queue[channel].push_back(wp); }
    // by value on purpose: both steps push into the queue they were popped from, and a
    // reallocation would invalidate a reference mid-step
    void propagateStep(int channel, glm::ivec3 wp);
    void removeStep(int channel, Removal r);
    void markTouched(const glm::ivec3& wp);

    // Resumable seed cursor. heights[] is produced by the descent phase and consumed by
    // the spread phase, so it has to survive across the yield between them.
    enum class SeedPhase : std::uint8_t { Idle, Descent, Spread, Emissive };
    struct SeedState {
        SeedPhase phase = SeedPhase::Idle;
        std::int32_t chunkX = 0;
        std::int32_t chunkZ = 0;
        int cursor = 0; // cell index within Descent/Spread, chunk index within Emissive
        int heights[kChunkSize][kChunkSize] = {};
    };
    SeedState m_seed;
    bool seedStackValid(Chunk** stack) const; // refetch + validate the column's chunks

    ChunkManager& m_world;
    std::vector<glm::ivec3> m_queue[2];
    std::vector<Removal> m_removal[2];
    std::size_t m_queueHead[2] = {0, 0};
    std::size_t m_removalHead[2] = {0, 0};
    // Every light write that changes a value marks its chunk, plus a neighbour per border
    // face — up to seven membership tests per write. That was a linear scan of m_touched,
    // which grows to hundreds of chunks over a streaming frame, so the dedup cost rose
    // with the number of chunks already touched. The set makes it flat; the vector stays
    // because the manager wants a stable order to walk.
    std::vector<ChunkPos> m_touched;
    ankerl::unordered_dense::set<ChunkPos, ChunkPosHash> m_touchedSet;
};

} // namespace vx
