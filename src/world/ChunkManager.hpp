#pragma once

#include "core/Pool.hpp"
#include "mesh/GreedyMesher.hpp"
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

// Fixed world height for now: 4 chunks (128 blocks). Sky light (M9) needs a defined
// top of the world anyway; this grows when terrain does.
inline constexpr int kWorldChunksY = 4;

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
// thread schedules jobs, drains results, and does the only GL work (uploads) under a
// per-frame time budget so a streaming burst never stalls a frame.
class ChunkManager {
public:
    ~ChunkManager();

    struct DrawItem {
        glm::vec3 offset; // chunk origin relative to the camera
        unsigned int ssbo;
        int quads;
    };

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

    void update(JobSystem& jobs, const glm::dvec3& cameraPos);
    void collectVisible(const Frustum& frustum, const glm::dvec3& cameraPos,
                        std::vector<DrawItem>& out) const;

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
        unsigned int ssbo = 0;
        int quads = 0;
        ChunkState state = ChunkState::Generating;
    };

    struct MeshResult {
        ChunkPos pos;
        std::vector<std::uint64_t> quads;
        GreedyStats stats;
    };

    void rescan(ChunkPos center);
    void processEvictions();
    const Chunk* voxelsAt(ChunkPos pos) const;

    static constexpr double kUploadBudgetMs = 2.0;

    ankerl::unordered_dense::map<ChunkPos, Entry, ChunkPosHash> m_chunks;
    Pool<Chunk> m_pool;

    std::vector<ChunkPos> m_genQueue; // farthest-first, pop_back serves nearest
    std::vector<ChunkPos> m_meshScratch;
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
    int m_jobsInFlight = 0; // main-thread view: submit increments, result-drain decrements
    int m_pendingMesh = 0;
    double m_uploadMsLast = 0.0;

    std::atomic<std::uint64_t> m_genNanos{0};
    std::atomic<std::uint64_t> m_meshNanos{0};
    std::atomic<int> m_genCount{0};
    std::atomic<int> m_meshCount{0};
};

} // namespace vx
