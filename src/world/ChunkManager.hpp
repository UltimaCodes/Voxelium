#pragma once

#include "core/Pool.hpp"
#include "mesh/GreedyMesher.hpp"
#include "world/Chunk.hpp"
#include "world/ChunkPos.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

namespace vx {

struct Frustum;

// Fixed world height for now: 4 chunks (128 blocks). Sky light (M9) needs a defined
// top of the world anyway; this grows when terrain does.
inline constexpr int kWorldChunksY = 4;

// Owns the loaded world: voxel data from a pool, one quad SSBO per meshed chunk,
// load/unload by distance around the camera. Chunks are meshed only once all six
// neighbors have voxel data, so seams cull correctly the first time and nothing gets
// remeshed when a neighbor appears. Data is generated one ring beyond the render
// radius to guarantee that.
//
// Everything runs on the main thread under per-frame budgets until M6 moves gen and
// mesh to workers; the budgets exist so a load burst degrades into a few slightly
// longer frames instead of one giant stall.
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
        std::int64_t totalQuads = 0;
        std::size_t voxelBytes = 0;
        std::size_t quadBytes = 0;
        double genMsTotal = 0.0;
        double meshMsTotal = 0.0;
        int genCount = 0;
        int meshCount = 0;
    };

    void update(const glm::dvec3& cameraPos);
    void collectVisible(const Frustum& frustum, const glm::dvec3& cameraPos,
                        std::vector<DrawItem>& out) const;

    bool busy() const { return !m_genQueue.empty() || m_pendingMesh > 0; }
    Stats stats() const;
    void logSummary() const;

    int radius = 10; // render radius in chunks, horizontal

private:
    struct Entry {
        Chunk* voxels = nullptr;
        unsigned int ssbo = 0;
        int quads = 0;
        bool meshed = false;
    };

    void rescan(ChunkPos center);
    void meshOne(ChunkPos pos, Entry& entry);
    const Chunk* voxelsAt(ChunkPos pos) const;
    bool neighborsHaveData(ChunkPos pos) const;

    static constexpr int kGenBudget = 24;  // chunks generated per frame
    static constexpr int kMeshBudget = 16; // chunks meshed per frame

    ankerl::unordered_dense::map<ChunkPos, Entry, ChunkPosHash> m_chunks;
    Pool<Chunk> m_pool;
    GreedyMesher m_mesher;
    std::vector<std::uint64_t> m_quadScratch;
    std::vector<ChunkPos> m_genQueue; // farthest-first, pop_back serves nearest
    std::vector<ChunkPos> m_meshScratch;
    std::vector<ChunkPos> m_evictScratch;
    ChunkPos m_center{INT32_MIN, 0, INT32_MIN};
    int m_lastRadius = -1;
    int m_pendingMesh = 0;
    double m_genMsTotal = 0.0;
    double m_meshMsTotal = 0.0;
    int m_genCount = 0;
    int m_meshCount = 0;
};

} // namespace vx
