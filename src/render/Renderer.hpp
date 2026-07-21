#pragma once

#include "render/Frustum.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace vx {

// What the cull compute pass needs to know about one chunk.
struct ChunkDrawInfo {
    glm::ivec3 chunkPos; // in chunk units
    std::uint32_t firstQuad;
    std::uint32_t quadCount;
};

// The whole-world renderer: one persistent-mapped quad arena with a free-list
// sub-allocator, per-chunk metadata for the cull compute shader, and a single
// glMultiDrawArraysIndirectCount for all opaque geometry. CPU cost per frame is a
// handful of uniforms and one dispatch — zero per-chunk work.
//
// Freed arena ranges are recycled through a 3-frame fence ring: a range becomes
// reusable only after the GPU provably finished every frame that might read it.
class Renderer {
public:
    struct Range {
        std::uint32_t first = 0;
        std::uint32_t count = 0; // in quads; 0 = empty/invalid
    };

    struct ArenaStats {
        std::uint32_t capacityQuads = 0;
        std::uint32_t usedQuads = 0;
        int freeBlocks = 0;
        std::uint32_t largestFreeBlock = 0;
    };

    bool init(unsigned int drawProgram, unsigned int cullProgram);
    void shutdown();

    // Uploads into the arena (memcpy into the persistent map). count == 0 on failure —
    // the caller rebuilds the arena from its CPU copies (hardReset + realloc all).
    Range alloc(std::span<const std::uint64_t> quads);
    void free(Range range);

    // Emergency defrag: waits for the GPU, then resets the allocator to one free block.
    // Rare by construction (arena is ~6x the working set); logged when it happens.
    void hardReset();

    void beginFrame(); // recycle fence-cleared frees, poll the visible-count readback
    void updateChunkMeta(std::span<const ChunkDrawInfo> chunks);
    void updateVisibility(std::span<const std::uint32_t> bits); // indexed by meta order
    void drawWorld(const glm::mat4& viewProj, const Frustum& frustum,
                   const glm::ivec3& cameraChunk, const glm::vec3& cameraFrac,
                   bool caveCull);
    void endFrame(); // fence this frame's GPU work

    int gpuVisibleChunks() const { return m_visibleChunks; } // ~3 frames of latency
    int debugReadDrawCount() const;                          // syncs; smoke test only
    ArenaStats arenaStats() const;

private:
    void recycle(Range range);

    static constexpr std::uint32_t kArenaQuads = 8u << 20; // 8M quads = 64 MB
    static constexpr std::uint32_t kMaxChunks = 8192;
    static constexpr int kFrameRing = 3;

    unsigned int m_drawProgram = 0;
    unsigned int m_cullProgram = 0;
    unsigned int m_vao = 0;
    unsigned int m_arenaBuf = 0;
    unsigned int m_metaBuf = 0;
    unsigned int m_cmdBuf = 0;
    unsigned int m_countBuf = 0;
    unsigned int m_offsetsBuf = 0;
    unsigned int m_readbackBuf = 0;
    unsigned int m_visBuf = 0;

    std::uint64_t* m_arenaMap = nullptr;
    void* m_metaMap = nullptr;
    const std::uint32_t* m_readbackMap = nullptr;
    std::uint32_t* m_visMap = nullptr;

    std::vector<Range> m_freeList; // sorted by first; adjacent blocks merged
    std::vector<Range> m_pendingFree[kFrameRing];
    void* m_fence[kFrameRing] = {}; // GLsync
    int m_slot = 0;

    std::uint32_t m_usedQuads = 0;
    std::uint32_t m_chunkCount = 0;
    int m_visibleChunks = 0;
};

} // namespace vx
