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
    void seedColumn(std::int32_t chunkX, std::int32_t chunkZ);

    // Call on any voxel edit, after the chunk itself was updated.
    void onBlockChanged(const glm::ivec3& wp, BlockId oldId, BlockId newId);

    void process(double budgetMs);
    bool idle() const {
        return m_queue[0].empty() && m_queue[1].empty() && m_removal[0].empty() &&
               m_removal[1].empty();
    }

    std::vector<ChunkPos>& touched() { return m_touched; } // drained by the manager

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

    BlockId blockAt(const glm::ivec3& wp) const;
    int getLevel(const glm::ivec3& wp, int channel) const;
    void setLevel(const glm::ivec3& wp, int channel, int level);
    void enqueue(int channel, const glm::ivec3& wp) { m_queue[channel].push_back(wp); }
    // by value on purpose: both steps push into the queue they were popped from, and a
    // reallocation would invalidate a reference mid-step
    void propagateStep(int channel, glm::ivec3 wp);
    void removeStep(int channel, Removal r);
    void markTouched(const glm::ivec3& wp);

    ChunkManager& m_world;
    std::vector<glm::ivec3> m_queue[2];
    std::vector<Removal> m_removal[2];
    std::size_t m_queueHead[2] = {0, 0};
    std::size_t m_removalHead[2] = {0, 0};
    std::vector<ChunkPos> m_touched;
};

} // namespace vx
