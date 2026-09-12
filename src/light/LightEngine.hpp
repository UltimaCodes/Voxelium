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
    bool stepSeedColumn(double budgetMs); // true = column finished (check seedAborted!)
    // True when the last seed gave up because the column's chunks were not all resident.
    // stepSeedColumn still returns "finished" in that case — there is nothing left to do —
    // but the column was NOT lit, so the caller must re-queue it instead of marking it
    // done. Getting this wrong marks a column seeded forever with sky light 0.
    bool seedAborted() const { return m_seedAborted; }
    bool seedInProgress() const { return m_seed.phase != SeedPhase::Idle; }
    void abortSeedColumn() { m_seed.phase = SeedPhase::Idle; }
    std::int32_t seedColumnX() const { return m_seed.chunkX; }
    std::int32_t seedColumnZ() const { return m_seed.chunkZ; }

    // One chunk that appeared inside a column already marked seeded. The band can grow
    // without anything being evicted — the load range is center.y +- kVerticalRadius while
    // eviction only fires past kVerticalRadius + 1 — so descend one chunk and every column
    // you are standing in gains a chunk while staying in m_seededColumns. The column seed
    // never runs again, and since meshing waits on light, that chunk is never meshed.
    //
    // Deliberately NOT a column re-seed: that relights every chunk in the column and
    // remeshes the lot, and a one-chunk descent would do it to every resident column at
    // once. `skyAbove` says this chunk is the new TOP of a column that reaches daylight,
    // which is the only case with nothing overhead to take light from.
    void seedNewChunk(ChunkPos pos, bool skyAbove);

    // Call on any voxel edit, after the chunk itself was updated.
    void onBlockChanged(const glm::ivec3& wp, BlockId oldId, BlockId newId);

    void process(double budgetMs);
    bool idle() const {
        for (int c = 0; c < kChannels; ++c)
            if (!m_queue[c].empty() || !m_removal[c].empty())
                return false;
        return true;
    }

    // Drained by the manager. Clearing goes through clearTouched() rather than handing
    // out a mutable vector, because the dedup set has to be dropped with it.
    const std::vector<ChunkPos>& touched() const { return m_touched; }
    void clearTouched() {
        m_touched.clear();
        m_touchedSet.clear();
    }

    // debug/verification. Packs getLevel()'s 0..15 step-count values (the BFS's own domain,
    // unrelated to Chunk's wider stored representation - see Chunk.hpp's light namespace),
    // so callers comparing against 0..15 (as every existing caller does) are unaffected by
    // the storage widening. Returns uint32_t (Chunk's own light word type) purely because
    // that's what light::pack now produces - the values packed are still 0..15 per channel.
    std::uint32_t lightAt(const glm::ivec3& wp) const;
    int queueDepth() const {
        std::size_t n = 0;
        for (int c = 0; c < kChannels; ++c)
            n += m_queue[c].size() + m_removal[c].size();
        return int(n);
    }

    // For SPEC #3's "where does column-seeding time actually go" question: descent bulk-skips
    // uniform-air chunks and stops at the first blocker, so it was suspected cheap even for a
    // radius-32 settle's thousands of coarse (no-cave) columns already, before spending any
    // engineering on an analytical fast path for them — this is what actually measures that
    // suspicion instead of assuming it. Spread is where a cave column's cost lives: every cave
    // mouth/overhang reachable from the surface is a boundary case pure heightmap terrain never
    // hits (see stepSeedColumn's own comments). Nanoseconds, cumulative since the last reset.
    struct SeedPhaseNanos {
        std::uint64_t descent = 0;
        std::uint64_t spread = 0;
        std::uint64_t emissive = 0;
    };
    SeedPhaseNanos seedPhaseNanos() const { return m_seedPhaseNanos; }
    void resetSeedPhaseNanos() { m_seedPhaseNanos = {}; }

private:
    // Four independent BFS channels. Block light is three of them so a coloured emitter
    // keeps its hue across its whole falloff; sky is the fourth because it seeds and
    // attenuates differently (full strength straight down).
    enum Channel : int { R = 0, G = 1, B = 2, Sky = 3, kChannels = 4 };

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
        // Resident chunk-Y range this seed was started against, and whether its top is
        // open sky. Captured at begin and re-checked every step: if the band moved under
        // us the partial work is invalid, because the descent origin changed.
        int y0 = 0;
        int y1 = 0;
        bool skyAtTop = false;
        SeedPhase phase = SeedPhase::Idle;
        std::int32_t chunkX = 0;
        std::int32_t chunkZ = 0;
        int cursor = 0; // cell index within Descent/Spread, chunk index within Emissive
        int heights[kChunkSize][kChunkSize] = {};
    };
    SeedState m_seed;
    bool m_seedAborted = false;
    SeedPhaseNanos m_seedPhaseNanos;
    // Refetch + validate the column's resident chunks. The range is no longer 0..world
    // top — there is no world top — so it comes back through y0/y1, and skyAtTop says
    // whether the descent starts in open sky or buried.
    static constexpr int kMaxStack = 2 * kVerticalRadius + 4; // band, plus the surface reach
    bool seedStackValid(Chunk** stack, int& y0, int& y1, bool& skyAtTop) const;

    ChunkManager& m_world;
    std::vector<glm::ivec3> m_queue[kChannels];
    std::vector<Removal> m_removal[kChannels];
    std::size_t m_queueHead[kChannels] = {};
    std::size_t m_removalHead[kChannels] = {};
    // Every light write that changes a value marks its chunk, plus a neighbour per border
    // face — up to seven membership tests per write. That was a linear scan of m_touched,
    // which grows to hundreds of chunks over a streaming frame, so the dedup cost rose
    // with the number of chunks already touched. The set makes it flat; the vector stays
    // because the manager wants a stable order to walk.
    std::vector<ChunkPos> m_touched;
    ankerl::unordered_dense::set<ChunkPos, ChunkPosHash> m_touchedSet;
};

} // namespace vx
