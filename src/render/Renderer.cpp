#include "render/Renderer.hpp"

#include "core/Log.hpp"

#include <glad/gl.h>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cstring>

namespace vx {

namespace {

// CPU mirror of the std430 ChunkMeta in cull.comp.
struct GpuChunkMeta {
    std::int32_t x, y, z, first;
    std::uint32_t count, pad0, pad1, pad2;
};
static_assert(sizeof(GpuChunkMeta) == 32);

constexpr GLbitfield kWriteMapFlags =
    GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
constexpr GLbitfield kReadMapFlags =
    GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;

} // namespace

bool Renderer::init(unsigned int drawProgram, unsigned int cullProgram) {
    m_drawProgram = drawProgram;
    m_cullProgram = cullProgram;

    glCreateVertexArrays(1, &m_vao);

    glCreateBuffers(1, &m_arenaBuf);
    glNamedBufferStorage(m_arenaBuf, GLsizeiptr(kArenaQuads) * 8, nullptr, kWriteMapFlags);
    m_arenaMap = static_cast<std::uint64_t*>(
        glMapNamedBufferRange(m_arenaBuf, 0, GLsizeiptr(kArenaQuads) * 8, kWriteMapFlags));

    glCreateBuffers(1, &m_metaBuf);
    glNamedBufferStorage(m_metaBuf, GLsizeiptr(kMaxChunks) * sizeof(GpuChunkMeta), nullptr,
                         kWriteMapFlags);
    m_metaMap = glMapNamedBufferRange(m_metaBuf, 0,
                                      GLsizeiptr(kMaxChunks) * sizeof(GpuChunkMeta),
                                      kWriteMapFlags);

    glCreateBuffers(1, &m_waterMetaBuf);
    glNamedBufferStorage(m_waterMetaBuf, GLsizeiptr(kMaxChunks) * sizeof(GpuChunkMeta),
                         nullptr, kWriteMapFlags);
    m_waterMetaMap = glMapNamedBufferRange(
        m_waterMetaBuf, 0, GLsizeiptr(kMaxChunks) * sizeof(GpuChunkMeta), kWriteMapFlags);

    glCreateBuffers(1, &m_cmdBuf);
    glNamedBufferStorage(m_cmdBuf, GLsizeiptr(kMaxChunks) * 16, nullptr, 0);
    glCreateBuffers(1, &m_countBuf);
    glNamedBufferStorage(m_countBuf, sizeof(GLuint), nullptr, 0);
    glCreateBuffers(1, &m_offsetsBuf);
    glNamedBufferStorage(m_offsetsBuf, GLsizeiptr(kMaxChunks) * 16, nullptr, 0);

    glCreateBuffers(1, &m_readbackBuf);
    glNamedBufferStorage(m_readbackBuf, kFrameRing * sizeof(GLuint), nullptr, kReadMapFlags);
    m_readbackMap = static_cast<const std::uint32_t*>(
        glMapNamedBufferRange(m_readbackBuf, 0, kFrameRing * sizeof(GLuint), kReadMapFlags));

    glCreateBuffers(1, &m_visBuf);
    glNamedBufferStorage(m_visBuf, kMaxChunks / 8, nullptr, kWriteMapFlags);
    m_visMap = static_cast<std::uint32_t*>(
        glMapNamedBufferRange(m_visBuf, 0, kMaxChunks / 8, kWriteMapFlags));
    if (m_visMap)
        std::memset(m_visMap, 0xFF, kMaxChunks / 8); // everything visible until a flood runs

    if (!m_arenaMap || !m_metaMap || !m_waterMetaMap || !m_readbackMap || !m_visMap) {
        VX_ERROR("renderer: persistent mapping failed");
        return false;
    }

    m_freeList.push_back({0, kArenaQuads});
    return true;
}

void Renderer::shutdown() {
    for (int i = 0; i < kFrameRing; ++i)
        if (m_fence[i]) {
            glDeleteSync(static_cast<GLsync>(m_fence[i]));
            m_fence[i] = nullptr;
        }
    const GLuint buffers[] = {m_arenaBuf,   m_metaBuf,     m_cmdBuf, m_countBuf,
                              m_offsetsBuf, m_readbackBuf, m_visBuf, m_waterMetaBuf};
    glDeleteBuffers(8, buffers);
    glDeleteVertexArrays(1, &m_vao);
}

Renderer::Range Renderer::alloc(std::span<const std::uint64_t> quads) {
    const std::uint32_t need = std::uint32_t(quads.size());
    if (need == 0)
        return {};
    if (const Range r = tryAlloc(quads, need); r.count != 0)
        return r;
    // Nothing fit. Before making the caller rebuild the whole arena, take back the frees
    // that are only waiting on the fence ring — during a teleport the world re-meshes far
    // faster than 3 frames, so the space is almost always already free in practice, just
    // not recyclable yet. This stalls on the GPU, but a hard reset does that AND
    // re-uploads every chunk, which measured as a ~1 second freeze.
    // At most one reclaim a frame. It blocks on the GPU, and once the arena is genuinely
    // full (not merely fence-delayed) every subsequent alloc would pay that wait again —
    // hundreds of allocs in a streaming frame turned into seconds of stall, which is
    // exactly the freeze this was meant to remove.
    if (m_reclaimedThisFrame)
        return {};
    m_reclaimedThisFrame = true;
    if (reclaimPendingFrees())
        return tryAlloc(quads, need);
    return {};
}

bool Renderer::reclaimPendingFrees() {
    bool any = false;
    for (int i = 0; i < kFrameRing; ++i) {
        if (m_pendingFree[i].empty())
            continue;
        // NO fence means we cannot prove the GPU is done with these ranges, so they are
        // untouchable — never "no fence, therefore free". Two slots look like that: the
        // current one (beginFrame dropped its fence and this frame's frees are still
        // piling into it while the GPU renders off them) and any slot whose fence came
        // back unsignaled (beginFrame drops the fence but deliberately keeps the frees).
        // Recycling either hands live memory back to the allocator, the next upload
        // memcpys over quads the GPU is still reading, and the world renders as garbage.
        if (!m_fence[i])
            continue;
        // Short cap per slot. Waiting is only worth it while it stays cheaper than the
        // hard reset it's avoiding — an unbounded wait just trades one freeze for another
        // (a 1s cap measured as a 170 ms stall in a bad frame).
        const GLenum state =
            glClientWaitSync(static_cast<GLsync>(m_fence[i]), GL_SYNC_FLUSH_COMMANDS_BIT,
                             GLuint64(20'000'000)); // 20 ms
        if (state != GL_ALREADY_SIGNALED && state != GL_CONDITION_SATISFIED)
            continue; // still in use: leave it fenced, the caller can hard reset
        glDeleteSync(static_cast<GLsync>(m_fence[i]));
        m_fence[i] = nullptr;
        for (const Range r : m_pendingFree[i]) {
            recycle(r);
            any = true;
        }
        m_pendingFree[i].clear();
    }
    return any;
}

Renderer::Range Renderer::tryAlloc(std::span<const std::uint64_t> quads, std::uint32_t need) {
    for (std::size_t i = 0; i < m_freeList.size(); ++i) {
        Range& block = m_freeList[i];
        if (block.count < need)
            continue;
        const Range r{block.first, need};
        block.first += need;
        block.count -= need;
        if (block.count == 0)
            m_freeList.erase(m_freeList.begin() + std::ptrdiff_t(i));
        std::memcpy(m_arenaMap + r.first, quads.data(), std::size_t(need) * 8);
        m_usedQuads += need;
        return r;
    }
    return {}; // no block fits — caller rebuilds
}

void Renderer::free(Range range) {
    if (range.count == 0)
        return;
    m_usedQuads -= range.count;
    m_pendingFree[m_slot].push_back(range);
}

void Renderer::recycle(Range range) {
    // insert sorted by first, then merge with both neighbors where adjacent
    const auto it = std::lower_bound(
        m_freeList.begin(), m_freeList.end(), range,
        [](const Range& a, const Range& b) { return a.first < b.first; });
    const auto pos = m_freeList.insert(it, range);
    const std::size_t i = std::size_t(pos - m_freeList.begin());
    if (i + 1 < m_freeList.size() &&
        m_freeList[i].first + m_freeList[i].count == m_freeList[i + 1].first) {
        m_freeList[i].count += m_freeList[i + 1].count;
        m_freeList.erase(m_freeList.begin() + std::ptrdiff_t(i + 1));
    }
    if (i > 0 && m_freeList[i - 1].first + m_freeList[i - 1].count == m_freeList[i].first) {
        m_freeList[i - 1].count += m_freeList[i].count;
        m_freeList.erase(m_freeList.begin() + std::ptrdiff_t(i));
    }
}

void Renderer::hardReset() {
    VX_ERROR("renderer: arena alloc failed — hard reset (glFinish + full rebuild)");
    glFinish();
    m_freeList.clear();
    m_freeList.push_back({0, kArenaQuads});
    for (auto& pending : m_pendingFree)
        pending.clear();
    m_usedQuads = 0;
}

void Renderer::beginFrame() {
    m_reclaimedThisFrame = false;
    m_slot = (m_slot + 1) % kFrameRing;
    if (m_fence[m_slot]) {
        const GLenum state =
            glClientWaitSync(static_cast<GLsync>(m_fence[m_slot]), 0, 0);
        glDeleteSync(static_cast<GLsync>(m_fence[m_slot]));
        m_fence[m_slot] = nullptr;
        if (state == GL_ALREADY_SIGNALED || state == GL_CONDITION_SATISFIED) {
            for (const Range r : m_pendingFree[m_slot])
                recycle(r);
            m_pendingFree[m_slot].clear();
            m_visibleChunks = int(m_readbackMap[m_slot]);
        }
        // unsignaled: leave the frees in this slot — the fence created at the end of
        // this frame signals after the old work anyway (GL commands are ordered)
    }
}

void Renderer::updateChunkMeta(std::span<const ChunkDrawInfo> chunks) {
    m_chunkCount = std::min(std::uint32_t(chunks.size()), kMaxChunks);
    auto* meta = static_cast<GpuChunkMeta*>(m_metaMap);
    for (std::uint32_t i = 0; i < m_chunkCount; ++i) {
        const ChunkDrawInfo& c = chunks[i];
        meta[i] = {c.chunkPos.x, c.chunkPos.y, c.chunkPos.z, std::int32_t(c.firstQuad),
                   c.quadCount, 0, 0, 0};
    }
}

void Renderer::updateVisibility(std::span<const std::uint32_t> bits) {
    const std::size_t words = std::min(bits.size(), std::size_t(kMaxChunks / 32));
    std::memcpy(m_visMap, bits.data(), words * sizeof(std::uint32_t));
}

void Renderer::updateWaterMeta(std::span<const ChunkDrawInfo> chunks) {
    m_waterChunkCount = std::min(std::uint32_t(chunks.size()), kMaxChunks);
    auto* meta = static_cast<GpuChunkMeta*>(m_waterMetaMap);
    for (std::uint32_t i = 0; i < m_waterChunkCount; ++i) {
        const ChunkDrawInfo& c = chunks[i];
        meta[i] = {c.chunkPos.x, c.chunkPos.y, c.chunkPos.z, std::int32_t(c.firstQuad),
                   c.quadCount, 0, 0, 0};
    }
}

void Renderer::cullInto(unsigned int metaBuf, unsigned int chunkCount, const Frustum& frustum,
                        const glm::ivec3& cameraChunk, const glm::vec3& cameraFrac,
                        bool caveCull) {
    const GLuint zero = 0;
    glClearNamedBufferSubData(m_countBuf, GL_R32UI, 0, sizeof(GLuint), GL_RED_INTEGER,
                              GL_UNSIGNED_INT, &zero);

    glUseProgram(m_cullProgram);
    glProgramUniform4fv(m_cullProgram, 0, frustum.count, &frustum.planes[0].x);
    glProgramUniform3iv(m_cullProgram, 6, 1, &cameraChunk.x);
    glProgramUniform3fv(m_cullProgram, 7, 1, &cameraFrac.x);
    glProgramUniform1ui(m_cullProgram, 8, chunkCount);
    glProgramUniform1ui(m_cullProgram, 9, caveCull ? 1u : 0u);
    glProgramUniform1ui(m_cullProgram, 10, GLuint(frustum.count));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, metaBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_cmdBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_countBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_offsetsBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_visBuf);
    glDispatchCompute((chunkCount + 63) / 64, 1, 1);
    glMemoryBarrier(GL_COMMAND_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
}

void Renderer::cullFrustum(const Frustum& frustum, const glm::ivec3& cameraChunk,
                           const glm::vec3& cameraFrac, bool caveCull) {
    cullInto(m_metaBuf, m_chunkCount, frustum, cameraChunk, cameraFrac, caveCull);
}

void Renderer::drawIndirect(unsigned int program, const glm::mat4& viewProj) {
    if (m_chunkCount == 0)
        return;
    glUseProgram(program);
    glProgramUniformMatrix4fv(program, 0, 1, GL_FALSE, glm::value_ptr(viewProj));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_arenaBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_offsetsBuf);
    glBindVertexArray(m_vao);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, m_cmdBuf);
    glBindBuffer(GL_PARAMETER_BUFFER, m_countBuf);
    glMultiDrawArraysIndirectCount(GL_TRIANGLES, nullptr, 0, GLsizei(kMaxChunks), 0);
}

void Renderer::drawWorld(const glm::mat4& viewProj, const Frustum& frustum,
                         const glm::ivec3& cameraChunk, const glm::vec3& cameraFrac,
                         bool caveCull) {
    if (m_chunkCount == 0)
        return;
    cullInto(m_metaBuf, m_chunkCount, frustum, cameraChunk, cameraFrac, caveCull);
    drawIndirect(m_drawProgram, viewProj);

    // HUD visible-count: copied now, read kFrameRing frames later off the fence ring
    glCopyNamedBufferSubData(m_countBuf, m_readbackBuf, 0,
                             GLintptr(m_slot) * GLintptr(sizeof(GLuint)), sizeof(GLuint));
}

void Renderer::drawWater(unsigned int program, const glm::mat4& viewProj,
                         const Frustum& frustum, const glm::ivec3& cameraChunk,
                         const glm::vec3& cameraFrac) {
    if (m_waterChunkCount == 0)
        return;
    cullInto(m_waterMetaBuf, m_waterChunkCount, frustum, cameraChunk, cameraFrac, false);
    glUseProgram(program);
    glProgramUniformMatrix4fv(program, 0, 1, GL_FALSE, glm::value_ptr(viewProj));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_arenaBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_offsetsBuf);
    glBindVertexArray(m_vao);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, m_cmdBuf);
    glBindBuffer(GL_PARAMETER_BUFFER, m_countBuf);
    glMultiDrawArraysIndirectCount(GL_TRIANGLES, nullptr, 0, GLsizei(kMaxChunks), 0);
}

void Renderer::endFrame() {
    m_fence[m_slot] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}

int Renderer::debugReadDrawCount() const {
    GLuint count = 0;
    glGetNamedBufferSubData(m_countBuf, 0, sizeof(GLuint), &count); // implicit sync
    return int(count);
}

Renderer::ArenaStats Renderer::arenaStats() const {
    ArenaStats s;
    s.capacityQuads = kArenaQuads;
    s.usedQuads = m_usedQuads;
    s.freeBlocks = int(m_freeList.size());
    for (const Range& r : m_freeList)
        s.largestFreeBlock = std::max(s.largestFreeBlock, r.count);
    return s;
}

} // namespace vx
