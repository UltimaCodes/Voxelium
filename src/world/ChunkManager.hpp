#pragma once

#include "core/Pool.hpp"
#include "gen/TerrainGenerator.hpp"
#include "mesh/GreedyMesher.hpp"
#include "render/Renderer.hpp"
#include "world/Chunk.hpp"
#include "world/ChunkPos.hpp"

#include <glm/glm.hpp>

#include <atomic>
#include <cstdint>
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

    // For the cull compute pass; call when consumeDirty() says the chunk set changed.
    // Assigns each drawable chunk its meta index (the visibility bitset uses the same
    // indexing), so the two must always be rebuilt together.
    void collectDrawInfo(std::vector<ChunkDrawInfo>& out);
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

    bool busy() const {
        return !m_genQueue.empty() || m_jobsInFlight > 0 || !m_uploads.empty() ||
               m_pendingMesh > 0;
    }
    Stats stats() const;
    void logSummary() const;

    int radius = 10; // render radius in chunks, horizontal

private:
    struct Entry {
        Chunk* voxels = nullptr;
        Renderer::Range range; // quads in the arena
        int quads = 0;
        ChunkState state = ChunkState::Generating;
        bool lit = false; // sky/emissive seeding done; meshing waits for it
        // all faces connected until meshed: never cave-cull what isn't known yet
        std::uint16_t connectivity = 0x7FFF;
        std::int32_t metaIndex = -1;   // slot in the cull meta buffer, -1 = not drawn
        std::uint32_t floodStamp = 0;  // visited marker for the visibility flood
        std::vector<std::uint64_t> quadsCpu; // kept for arena rebuilds
    };

    struct MeshResult {
        ChunkPos pos;
        std::vector<std::uint64_t> quads;
        GreedyStats stats;
        std::uint16_t connectivity = 0;
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

    ankerl::unordered_dense::map<ChunkPos, Entry, ChunkPosHash> m_chunks;
    ankerl::unordered_dense::set<ChunkPos, ChunkPosHash> m_seededColumns; // y always 0
    std::vector<ChunkPos> m_seedQueue; // columns with full data, waiting for light seed
    Pool<Chunk> m_pool;
    TerrainGenerator m_gen{20260721};
    GreedyMesher m_mesher; // main-thread instance for the synchronous edit remesh path
    std::vector<std::uint64_t> m_quadScratch;

    std::vector<ChunkPos> m_genQueue; // farthest-first, pop_back serves nearest
    std::vector<ChunkPos> m_meshScratch;
    std::vector<ChunkPos> m_remeshScratch;
    std::vector<ChunkPos> m_evictList; // outside the ring; may wait on in-flight jobs

    // worker -> main queues
    std::mutex m_genDoneMutex;
    std::vector<ChunkPos> m_genDone;
    std::mutex m_meshDoneMutex;
    std::vector<MeshResult> m_meshDone;

    std::vector<ChunkPos> m_genDoneScratch;
    std::vector<MeshResult> m_uploads; // drained results awaiting budgeted GL upload

    ChunkPos m_center{INT32_MIN, 0, INT32_MIN};
    int m_lastRadius = -1;
    bool m_dirty = false;

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
    double m_uploadMsLast = 0.0;

    std::atomic<std::uint64_t> m_genNanos{0};
    std::atomic<std::uint64_t> m_meshNanos{0};
    std::atomic<int> m_genCount{0};
    std::atomic<int> m_meshCount{0};
};

} // namespace vx
