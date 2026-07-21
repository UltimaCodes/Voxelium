#include "world/ChunkManager.hpp"

#include "core/Log.hpp"
#include "gen/TestTerrain.hpp"
#include "render/Frustum.hpp"

#include <glad/gl.h>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace vx {

namespace {

double msSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
        .count();
}

std::int64_t horizontalDist2(ChunkPos p, ChunkPos center) {
    const std::int64_t dx = p.x - center.x;
    const std::int64_t dz = p.z - center.z;
    return dx * dx + dz * dz;
}

} // namespace

ChunkManager::~ChunkManager() {
    for (const auto& item : m_chunks) {
        if (item.second.ssbo != 0) {
            const GLuint ssbo = item.second.ssbo;
            glDeleteBuffers(1, &ssbo);
        }
    }
}

const Chunk* ChunkManager::voxelsAt(ChunkPos pos) const {
    const auto it = m_chunks.find(pos);
    return it != m_chunks.end() ? it->second.voxels : nullptr;
}

bool ChunkManager::neighborsHaveData(ChunkPos pos) const {
    static constexpr int kOffsets[6][3] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                           {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
    for (const auto& o : kOffsets) {
        const ChunkPos n{pos.x + o[0], pos.y + o[1], pos.z + o[2]};
        if (n.y < 0 || n.y >= kWorldChunksY)
            continue; // outside the world counts as present (air)
        if (!m_chunks.contains(n))
            return false;
    }
    return true;
}

void ChunkManager::rescan(ChunkPos center) {
    const int dataR = radius + 1;

    // evict outside the data ring, +1 of hysteresis so border chunks don't thrash
    m_evictScratch.clear();
    for (const auto& item : m_chunks)
        if (std::abs(item.first.x - center.x) > dataR + 1 ||
            std::abs(item.first.z - center.z) > dataR + 1)
            m_evictScratch.push_back(item.first);
    for (const ChunkPos pos : m_evictScratch) {
        auto it = m_chunks.find(pos);
        if (it->second.ssbo != 0) {
            const GLuint ssbo = it->second.ssbo;
            glDeleteBuffers(1, &ssbo);
        }
        m_pool.release(it->second.voxels);
        m_chunks.erase(it);
    }

    // queue every missing data chunk, farthest first so pop_back serves nearest
    m_genQueue.clear();
    for (int dz = -dataR; dz <= dataR; ++dz)
        for (int dx = -dataR; dx <= dataR; ++dx)
            for (int y = 0; y < kWorldChunksY; ++y) {
                const ChunkPos pos{center.x + dx, y, center.z + dz};
                if (!m_chunks.contains(pos))
                    m_genQueue.push_back(pos);
            }
    std::sort(m_genQueue.begin(), m_genQueue.end(), [center](ChunkPos a, ChunkPos b) {
        return horizontalDist2(a, center) > horizontalDist2(b, center);
    });
}

void ChunkManager::update(const glm::dvec3& cameraPos) {
    const ChunkPos center{int(std::floor(cameraPos.x / kChunkSize)), 0,
                          int(std::floor(cameraPos.z / kChunkSize))};
    if (center != m_center || radius != m_lastRadius) {
        m_center = center;
        m_lastRadius = radius;
        rescan(center);
    }

    for (int budget = kGenBudget; budget > 0 && !m_genQueue.empty();) {
        const ChunkPos pos = m_genQueue.back();
        m_genQueue.pop_back();
        if (m_chunks.contains(pos))
            continue;
        const auto t0 = std::chrono::steady_clock::now();
        Chunk* voxels = m_pool.acquire();
        generateTerrain(*voxels, pos);
        m_genMsTotal += msSince(t0);
        ++m_genCount;
        m_chunks.try_emplace(pos, Entry{voxels, 0, 0, false});
        --budget;
    }

    // mesh whatever is ready, nearest first
    m_meshScratch.clear();
    for (const auto& item : m_chunks) {
        if (item.second.meshed)
            continue;
        if (std::abs(item.first.x - center.x) > radius ||
            std::abs(item.first.z - center.z) > radius)
            continue;
        if (!neighborsHaveData(item.first))
            continue;
        m_meshScratch.push_back(item.first);
    }
    std::sort(m_meshScratch.begin(), m_meshScratch.end(), [center](ChunkPos a, ChunkPos b) {
        return horizontalDist2(a, center) < horizontalDist2(b, center);
    });
    m_pendingMesh = int(m_meshScratch.size());

    const int count = std::min(kMeshBudget, int(m_meshScratch.size()));
    for (int i = 0; i < count; ++i) {
        auto it = m_chunks.find(m_meshScratch[std::size_t(i)]);
        meshOne(it->first, it->second);
        --m_pendingMesh;
    }
}

void ChunkManager::meshOne(ChunkPos pos, Entry& entry) {
    const auto t0 = std::chrono::steady_clock::now();

    const Chunk* neighbors[6] = {
        voxelsAt({pos.x + 1, pos.y, pos.z}), voxelsAt({pos.x - 1, pos.y, pos.z}),
        voxelsAt({pos.x, pos.y + 1, pos.z}), voxelsAt({pos.x, pos.y - 1, pos.z}),
        voxelsAt({pos.x, pos.y, pos.z + 1}), voxelsAt({pos.x, pos.y, pos.z - 1}),
    };
    const GreedyStats stats = m_mesher.mesh(*entry.voxels, neighbors, m_quadScratch);

    entry.meshed = true;
    entry.quads = stats.quads;
    if (stats.quads > 0) {
        GLuint ssbo = 0;
        glCreateBuffers(1, &ssbo);
        glNamedBufferStorage(ssbo, GLsizeiptr(m_quadScratch.size() * sizeof(std::uint64_t)),
                             m_quadScratch.data(), 0);
        entry.ssbo = ssbo;
    }

    m_meshMsTotal += msSince(t0);
    ++m_meshCount;
}

void ChunkManager::collectVisible(const Frustum& frustum, const glm::dvec3& cameraPos,
                                  std::vector<DrawItem>& out) const {
    out.clear();
    for (const auto& item : m_chunks) {
        const Entry& e = item.second;
        if (e.quads == 0)
            continue;
        const ChunkPos pos = item.first;
        const glm::vec3 rel(glm::dvec3(pos.x * kChunkSize, pos.y * kChunkSize,
                                       pos.z * kChunkSize) -
                            cameraPos);
        const float half = kChunkSize * 0.5f;
        if (!frustum.intersects(rel + half, glm::vec3(half)))
            continue;
        out.push_back({rel, e.ssbo, e.quads});
    }
}

ChunkManager::Stats ChunkManager::stats() const {
    Stats s;
    s.loaded = int(m_chunks.size());
    s.genQueue = int(m_genQueue.size());
    s.genMsTotal = m_genMsTotal;
    s.meshMsTotal = m_meshMsTotal;
    s.genCount = m_genCount;
    s.meshCount = m_meshCount;
    for (const auto& item : m_chunks) {
        const Entry& e = item.second;
        if (e.meshed)
            ++s.meshed;
        if (e.voxels->uniform())
            ++s.uniformChunks;
        s.totalQuads += e.quads;
        s.voxelBytes += e.voxels->storageBytes();
        s.quadBytes += std::size_t(e.quads) * sizeof(std::uint64_t);
    }
    return s;
}

void ChunkManager::logSummary() const {
    const Stats s = stats();
    VX_INFO("world: %d chunks (%d uniform), %d meshed, %lld quads", s.loaded,
            s.uniformChunks, s.meshed, static_cast<long long>(s.totalQuads));
    VX_INFO("world: %.1f MB voxels, %.1f MB quads | gen avg %.3f ms, mesh avg %.3f ms",
            double(s.voxelBytes) / (1024.0 * 1024.0), double(s.quadBytes) / (1024.0 * 1024.0),
            s.genCount > 0 ? s.genMsTotal / s.genCount : 0.0,
            s.meshCount > 0 ? s.meshMsTotal / s.meshCount : 0.0);
}

} // namespace vx
