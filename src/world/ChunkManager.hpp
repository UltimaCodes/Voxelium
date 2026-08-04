#pragma once

#include "core/Pool.hpp"
#include "gen/TerrainGenerator.hpp"
#include "io/RegionStore.hpp"
#include "mesh/GreedyMesher.hpp"
#include "render/Renderer.hpp"
#include "world/Chunk.hpp"
#include "world/ChunkPos.hpp"

#include <glm/glm.hpp>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <vector>

namespace vx {

struct Frustum;
class JobSystem;
class LightEngine;

// Per-chunk pipeline state. Only the main thread reads or writes it — workers are pure
// functions that hand results back through queues, so no state needs to be atomic. The
// one cross-thread liveness concern (a mesh job reading a neighbor while it unloads)
// is handled by the job refcount on Chunk itself.
enum class ChunkState : std::uint8_t {
    Generating, // gen job in flight, voxels being written on a worker
    Ready,      // voxel data present and immutable, not meshed yet
    Meshing,    // mesh job in flight
    Meshed,     // quads uploaded (possibly zero of them)
};

// Owns the loaded world. Generation and meshing run on JobSystem workers; the main
// thread schedules jobs, drains results, and does the only GL work (arena uploads)
// under a per-frame time budget so a streaming burst never stalls a frame. Quads live
// in the Renderer's arena; a CPU copy stays here so the arena can be rebuilt after a
// hard reset without remeshing.
class ChunkManager {
public:
    explicit ChunkManager(std::filesystem::path saveDir);

    struct Stats {
        int loaded = 0;
        int meshed = 0;
        int uniformChunks = 0;
        int genQueue = 0;
        int jobsInFlight = 0;
        int pendingUploads = 0;
        std::int64_t totalQuads = 0;
        std::size_t voxelBytes = 0;
        std::size_t quadBytes = 0;
        double genMsAvg = 0.0;
        double meshMsAvg = 0.0;
        double uploadMsLast = 0.0;
    };

    void update(JobSystem& jobs, Renderer& renderer, LightEngine& light,
                const glm::dvec3& cameraPos);

    const Chunk* chunkData(ChunkPos pos) const { return voxelsAt(pos); }
    Chunk* chunkDataMutable(ChunkPos pos) {
        const auto it = m_chunks.find(pos);
        return it != m_chunks.end() ? it->second.voxels : nullptr;
    }

    BlockId blockAtWorld(const glm::ivec3& wp) const;

    // The block-edit fast path: sets the voxel, settles light, and synchronously
    // remeshes the affected chunk (plus border neighbors and anything light touched)
    // on the main thread — edit to visible in the same frame, per the spec target.
    bool setBlock(Renderer& renderer, LightEngine& light, const glm::ivec3& wp, BlockId id);

    const TerrainGenerator& generator() const { return m_gen; }
    RegionStore& regions() { return m_regions; }

    // Serializes every dirty (player-edited) chunk to its region and flushes to disk.
    // Unedited chunks regenerate from the seed, so only diffs from procedural gen are
    // stored. Main thread.
    void saveDirty();

    // For the cull compute pass; call when consumeDirty() says the chunk set changed.
    // Assigns each drawable chunk its meta index (the visibility bitset uses the same
    // indexing), so the two must always be rebuilt together. Also fills the separate
    // water draw list for the transparent pass.
    void collectDrawInfo(std::vector<ChunkDrawInfo>& out, std::vector<ChunkDrawInfo>& water);
    bool consumeDirty() {
        const bool was = m_dirty;
        m_dirty = false;
        return was;
    }

    // Recomputes the camera-outward visibility flood (cave culling) only when needed:
    // the chunk set changed, the camera crossed a chunk boundary, or the toggle
    // flipped. Steady-state cost is zero.
    void updateVisibilityIfNeeded(Renderer& renderer, const glm::dvec3& cameraPos,
                                  bool enabled);

    // worst single-frame cost of each main-thread phase since the last reset. only used
    // to find where a streaming hitch actually is, rather than guessing at it.
    struct PhasePeak {
        double rescanMs = 0.0;
        double evictMs = 0.0;
        double scanMs = 0.0;
        double seedMs = 0.0;
        double propagateMs = 0.0;
        double uploadMs = 0.0;
    };
    PhasePeak phasePeak() const { return m_phase; }
    void resetPhasePeak() { m_phase = {}; }

    bool busy() const {
        // the seed queue counts: a column that hasn't been light-seeded yet is unlit and
        // therefore unmeshed, so the world is not done streaming no matter how empty the
        // mesh side looks. leaving it out let "re-streamed" report a finish while chunks
        // were still dark.
        return !m_genQueue.empty() || m_jobsInFlight > 0 || !m_uploads.empty() ||
               m_pendingMesh > 0 || !m_seedQueue.empty() || m_seedActive;
    }
    Stats stats() const;
    void logSummary() const;
    // force the next stats() to do the full-chunk scan (used before logSummary)
    void refreshStats() const { m_scanAge = 999; }

    int radius = 16; // render radius in chunks, horizontal

private:
    struct Entry {
        Chunk* voxels = nullptr;
        Renderer::Range range;      // opaque quads in the arena
        Renderer::Range waterRange; // transparent quads in the arena
        int quads = 0;
        int waterQuads = 0;
        ChunkState state = ChunkState::Generating;
        bool lit = false;      // sky/emissive seeding done; meshing waits for it
        bool dirty = false;    // player-edited since load; needs saving before eviction
        bool detailed = true;  // false = coarse gen (no caves/trees), for far chunks
        // a detailed re-gen in flight into a second chunk (double buffer); the coarse mesh
        // stays drawn until this lands, so approaching a coarse chunk never flashes a hole
        Chunk* pendingDetail = nullptr;
        std::int8_t lod = 0;         // detail level of the current drawn mesh
        std::int8_t meshingLod = -1; // LOD of an in-flight (re)mesh job, -1 = none
        // all faces connected until meshed: never cave-cull what isn't known yet
        std::uint16_t connectivity = 0x7FFF;
        std::int32_t metaIndex = -1;   // slot in the cull meta buffer, -1 = not drawn
        std::uint32_t floodStamp = 0;  // visited marker for the visibility flood
        std::vector<std::uint64_t> quadsCpu;      // opaque, kept for arena rebuilds
        std::vector<std::uint64_t> waterQuadsCpu; // transparent
    };

    struct MeshResult {
        ChunkPos pos;
        std::vector<std::uint64_t> quads;
        std::vector<std::uint64_t> water;
        GreedyStats stats;
        std::uint16_t connectivity = 0;
        std::int8_t lod = 0;
    };

    struct FloodNode {
        ChunkPos pos;
        std::int8_t entryFace; // face we came in through; -1 for the camera chunk
        std::uint8_t dirMask;  // directions traveled so far; no backtracking
        std::uint16_t connectivity;
    };

    void rescan(ChunkPos center);
    void processEvictions(Renderer& renderer);
    void rebuildArena(Renderer& renderer);
    void rebuildVisibility(ChunkPos cameraChunk);
    void remeshNow(Renderer& renderer, ChunkPos pos);
    const Chunk* voxelsAt(ChunkPos pos) const;

    static constexpr double kUploadBudgetMs = 2.0;
    // sky/emissive column seeding, then the whole light phase (seed + propagate). one
    // deadline for both so their overshoots can't stack.
    static constexpr double kSeedBudgetMs = 2.0;
    static constexpr double kLightBudgetMs = 3.5;

    ankerl::unordered_dense::map<ChunkPos, Entry, ChunkPosHash> m_chunks;
    ankerl::unordered_dense::set<ChunkPos, ChunkPosHash> m_seededColumns; // y always 0
    std::vector<ChunkPos> m_seedQueue; // columns with full data, waiting for light seed
    std::vector<ChunkPos> m_seedDeferred; // this frame's not-yet-settled columns
    Pool<Chunk> m_pool;
    TerrainGenerator m_gen{20260721};
    RegionStore m_regions;
    GreedyMesher m_mesher; // main-thread instance for the synchronous edit remesh path
    std::vector<std::uint64_t> m_quadScratch;
    std::vector<std::uint64_t> m_waterScratch;
    std::vector<std::uint8_t> m_saveScratch;

    std::vector<ChunkPos> m_genQueue; // farthest-first, pop_back serves nearest
    std::vector<ChunkPos> m_meshScratch;
    std::vector<ChunkPos> m_remeshScratch;
    std::vector<ChunkPos> m_evictList; // outside the ring; may wait on in-flight jobs

    // a detailed re-gen finished on a worker; carries its own chunk so completion doesn't
    // depend on the entry still existing (it's released if the entry went away)
    struct RegenResult {
        ChunkPos pos;
        Chunk* voxels;
    };

    // worker -> main queues
    std::mutex m_genDoneMutex;
    std::vector<ChunkPos> m_genDone;
    std::mutex m_meshDoneMutex;
    std::vector<MeshResult> m_meshDone;
    std::mutex m_regenDoneMutex;
    std::vector<RegenResult> m_regenDone;

    std::vector<ChunkPos> m_genDoneScratch;
    std::vector<RegenResult> m_regenDoneScratch;
    std::vector<ChunkPos> m_regenScratch; // coarse chunks that want detail this frame
    std::vector<MeshResult> m_uploads; // drained results awaiting budgeted GL upload

    ChunkPos m_center{INT32_MIN, 0, INT32_MIN};
    int m_lastRadius = -1;
    bool m_dirty = false;

    // cached full-chunk scan for the HUD, refreshed a few times a second
    struct Scan {
        int meshed = 0;
        int uniformChunks = 0;
        std::int64_t totalQuads = 0;
        std::size_t voxelBytes = 0;
        std::size_t quadBytes = 0;
    };
    mutable Scan m_scanCache;
    mutable int m_scanAge = 999;

    // visibility flood state
    std::vector<std::uint32_t> m_visBits;
    std::vector<FloodNode> m_floodQueue;
    ChunkPos m_lastFloodCenter{INT32_MIN, 0, INT32_MIN};
    std::uint32_t m_floodStamp = 0;
    int m_metaCount = 0;
    bool m_visDirty = true;
    bool m_lastCaveCull = false;
    int m_jobsInFlight = 0; // main-thread view: submit increments, result-drain decrements
    int m_pendingMesh = 0;
    // the per-frame mesh scan is O(loaded chunks); only run it when something could have
    // created mesh work (gen/light/LOD change) instead of every frame. set by those events,
    // cleared once scanned. when the world is idle there is provably nothing to mesh.
    bool m_meshScanDirty = true;
    double m_uploadMsLast = 0.0;
    bool m_seedActive = false; // a column is part-seeded and will resume next frame
    PhasePeak m_phase;

    std::atomic<std::uint64_t> m_genNanos{0};
    std::atomic<std::uint64_t> m_meshNanos{0};
    std::atomic<int> m_genCount{0};
    std::atomic<int> m_meshCount{0};
};

} // namespace vx
