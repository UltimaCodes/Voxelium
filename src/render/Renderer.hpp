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
    std::uint32_t sunlit; // 0 = sealed from the sky, skipped by the shadow pass
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
    // Before reporting failure it reclaims the deferred frees still sitting in the fence
    // ring, which is usually all it takes: a streaming burst frees plenty, it's just not
    // recyclable for 3 frames yet.
    Range alloc(std::span<const std::uint64_t> quads);
    void free(Range range);

    // Emergency defrag: an allocation could not be satisfied. Same work as resetArena, but
    // it shouts, because it means the arena is too small or too fragmented and the caller
    // is about to re-upload the world. Rare by construction.
    void hardReset();

    // The same reset, deliberately and silently: every range that referenced the arena has
    // just been dropped (a world switch). Separate from hardReset because logging "alloc
    // failed" on a clean teardown is a lie, and one that reads as a real defect in the logs
    // of a perfectly healthy "new world".
    void resetArena();

    void beginFrame(); // recycle fence-cleared frees, poll the visible-count readback
    void updateChunkMeta(std::span<const ChunkDrawInfo> chunks);
    void updateWaterMeta(std::span<const ChunkDrawInfo> chunks); // transparent pass
    void updateVisibility(std::span<const std::uint32_t> bits);  // indexed by meta order
    void drawWorld(const glm::mat4& viewProj, const Frustum& frustum,
                   const glm::ivec3& cameraChunk, const glm::vec3& cameraFrac,
                   bool caveCull);
    // Transparent pass: frustum-cull the water meta and draw with `program` (the caller
    // sets blending + depth-write-off). No cave culling — water shares its chunk's
    // visibility but uses a separate meta list.
    void drawWater(unsigned int program, const glm::mat4& viewProj, const Frustum& frustum,
                   const glm::ivec3& cameraChunk, const glm::vec3& cameraFrac);
    void endFrame(); // fence this frame's GPU work

    // For the shadow pass: cull chunks against an arbitrary frustum (a cascade's ortho
    // box) into the command buffers, then draw with any depth program. Called once per
    // cascade so far casters aren't drawn into tight near cascades.
    void cullFrustum(const Frustum& frustum, const glm::ivec3& cameraChunk,
                     const glm::vec3& cameraFrac, bool caveCull, bool sunlitOnly = false);
    void drawIndirect(unsigned int program, const glm::mat4& viewProj);
    void cullInto(unsigned int metaBuf, unsigned int chunkCount, const Frustum& frustum,
                  const glm::ivec3& cameraChunk, const glm::vec3& cameraFrac, bool caveCull,
                  bool sunlitOnly = false);

    int gpuVisibleChunks() const { return m_visibleChunks; } // ~3 frames of latency
    int debugReadDrawCount() const;                          // syncs; smoke test only
    ArenaStats arenaStats() const;

private:
    void recycle(Range range);
    Range tryAlloc(std::span<const std::uint64_t> quads, std::uint32_t need);
    bool reclaimPendingFrees(); // wait out the fence ring, recycle everything in it

    // 48M quads = 384 MB. Sized for radius 32, which the render-distance slider offers and
    // the settings clamp allows, so it is a supported setting and not an exotic one.
    // Measured settled (2026-08-07): radius 16 holds 5.29M quads, radius 32 holds 21.31M —
    // #27's vertical band roughly quadrupled the far case. At the old 24M that is 85% full
    // with the largest free block down to 1.38M, and a moving `--tour 32` failed to alloc
    // and hard-reset the arena at every one of its five stops. A hardReset is a glFinish
    // plus a re-upload of every live chunk: precisely the stall Tier 0 #4 removed.
    // 2.25x the settled radius-32 set is the headroom fragmentation actually needs.
    static constexpr std::uint32_t kArenaQuads = 48u << 20;
    // Drawable chunks the GPU meta/command/visibility buffers can hold. This is a HARD
    // cap: anything past it gets no meta entry, so it is not drawn and does not appear in
    // the shadow cull either — invisible geometry that also casts no shadow. 8192 was
    // already under water at the DEFAULT radius 16 (~8700 meshed chunks), so several
    // hundred chunks were being dropped silently.
    //
    // 32768 "covered radius 32 with room to spare" only before #27. Measured 2026-08-07:
    // 7778 drawable at radius 16, 30861 settled at radius 32, and a moving `--tour 32`
    // logged 36318 — over the cap, so it was silently dropping chunks again. At ~32 B per
    // chunk across meta + water meta + commands + offsets, 65536 costs ~2 MB.
    static constexpr std::uint32_t kMaxChunks = 65536;
    static constexpr int kFrameRing = 3;

    unsigned int m_drawProgram = 0;
    unsigned int m_cullProgram = 0;
    unsigned int m_vao = 0;
    unsigned int m_arenaBuf = 0;
    unsigned int m_metaBuf = 0;
    unsigned int m_waterMetaBuf = 0;
    unsigned int m_cmdBuf = 0;
    unsigned int m_countBuf = 0;
    unsigned int m_offsetsBuf = 0;
    unsigned int m_readbackBuf = 0;
    unsigned int m_visBuf = 0;

    std::uint64_t* m_arenaMap = nullptr;
    void* m_metaMap = nullptr;
    void* m_waterMetaMap = nullptr;
    const std::uint32_t* m_readbackMap = nullptr;
    std::uint32_t* m_visMap = nullptr;

    bool m_reclaimedThisFrame = false; // reclaim blocks on the GPU: once a frame, no more
    bool m_loggedChunkCap = false;     // kMaxChunks overflow is reported once, not per frame
    std::vector<Range> m_freeList; // sorted by first; adjacent blocks merged
    std::vector<Range> m_pendingFree[kFrameRing];
    void* m_fence[kFrameRing] = {}; // GLsync
    int m_slot = 0;

    std::uint32_t m_usedQuads = 0;
    std::uint32_t m_chunkCount = 0;
    std::uint32_t m_waterChunkCount = 0;
    int m_visibleChunks = 0;
};

} // namespace vx
