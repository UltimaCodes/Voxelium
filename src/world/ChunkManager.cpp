#include "world/ChunkManager.hpp"

#include "core/Log.hpp"
#include "gen/TestTerrain.hpp"
#include "jobs/JobSystem.hpp"
#include "render/Frustum.hpp"

#include <glad/gl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>

namespace vx {

namespace {

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

std::uint64_t nsSince(Clock::time_point t0) {
    return std::uint64_t(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count());
}

std::int64_t horizontalDist2(ChunkPos p, ChunkPos center) {
    const std::int64_t dx = p.x - center.x;
    const std::int64_t dz = p.z - center.z;
    return dx * dx + dz * dz;
}

} // namespace

ChunkManager::~ChunkManager() {
    // JobSystem is destroyed before the manager (declaration order in main), so no
    // worker can be touching chunks by the time this runs.
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

void ChunkManager::rescan(ChunkPos center) {
    const int dataR = radius + 1;

    // rebuilt from scratch so chunks that came back into range drop off the evict list
    m_evictList.clear();
    for (const auto& item : m_chunks)
        if (std::abs(item.first.x - center.x) > dataR + 1 ||
            std::abs(item.first.z - center.z) > dataR + 1)
            m_evictList.push_back(item.first);

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

void ChunkManager::processEvictions() {
    for (std::size_t i = 0; i < m_evictList.size();) {
        const auto it = m_chunks.find(m_evictList[i]);
        if (it == m_chunks.end()) {
            m_evictList[i] = m_evictList.back();
            m_evictList.pop_back();
            continue;
        }
        Entry& e = it->second;
        const bool inFlight = e.state == ChunkState::Generating ||
                              e.state == ChunkState::Meshing || e.voxels->hasJobRefs();
        if (inFlight) {
            ++i; // a worker still holds it — retry next frame
            continue;
        }
        if (e.ssbo != 0) {
            const GLuint ssbo = e.ssbo;
            glDeleteBuffers(1, &ssbo);
        }
        m_pool.release(e.voxels);
        m_chunks.erase(it);
        m_evictList[i] = m_evictList.back();
        m_evictList.pop_back();
    }
}

void ChunkManager::update(JobSystem& jobs, const glm::dvec3& cameraPos) {
    const ChunkPos center{int(std::floor(cameraPos.x / kChunkSize)), 0,
                          int(std::floor(cameraPos.z / kChunkSize))};
    if (center != m_center || radius != m_lastRadius) {
        m_center = center;
        m_lastRadius = radius;
        rescan(center);
    }

    processEvictions();

    // drain finished generations (cheap: a state flip per chunk)
    {
        std::lock_guard lock(m_genDoneMutex);
        m_genDoneScratch.swap(m_genDone);
    }
    for (const ChunkPos pos : m_genDoneScratch) {
        const auto it = m_chunks.find(pos);
        if (it != m_chunks.end() && it->second.state == ChunkState::Generating)
            it->second.state = ChunkState::Ready;
        --m_jobsInFlight;
    }
    m_genDoneScratch.clear();

    // keep enough jobs in flight to feed the workers without racing miles ahead of
    // the camera — stale far-away work is wasted work when it moves
    const int inflightCap = std::max(8, jobs.workerCount() * 6);

    while (m_jobsInFlight < inflightCap && !m_genQueue.empty()) {
        const ChunkPos pos = m_genQueue.back();
        m_genQueue.pop_back();
        if (m_chunks.contains(pos))
            continue;
        Chunk* voxels = m_pool.acquire();
        voxels->retainJob();
        m_chunks.try_emplace(pos, Entry{voxels, 0, 0, ChunkState::Generating});
        ++m_jobsInFlight;
        jobs.submit([this, voxels, pos] {
            const auto t0 = Clock::now();
            generateTerrain(*voxels, pos);
            m_genNanos.fetch_add(nsSince(t0), std::memory_order_relaxed);
            m_genCount.fetch_add(1, std::memory_order_relaxed);
            voxels->releaseJob();
            std::lock_guard lock(m_genDoneMutex);
            m_genDone.push_back(pos);
        });
    }

    // schedule meshing for chunks whose whole neighborhood has data, nearest first
    m_meshScratch.clear();
    for (const auto& item : m_chunks) {
        if (item.second.state != ChunkState::Ready)
            continue;
        if (std::abs(item.first.x - center.x) > radius ||
            std::abs(item.first.z - center.z) > radius)
            continue;
        bool ready = true;
        static constexpr int kOffsets[6][3] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                               {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
        for (const auto& o : kOffsets) {
            const ChunkPos n{item.first.x + o[0], item.first.y + o[1], item.first.z + o[2]};
            if (n.y < 0 || n.y >= kWorldChunksY)
                continue; // outside the world counts as present (air)
            const auto nit = m_chunks.find(n);
            if (nit == m_chunks.end() || nit->second.state == ChunkState::Generating) {
                ready = false;
                break;
            }
        }
        if (ready)
            m_meshScratch.push_back(item.first);
    }
    std::sort(m_meshScratch.begin(), m_meshScratch.end(), [center](ChunkPos a, ChunkPos b) {
        return horizontalDist2(a, center) < horizontalDist2(b, center);
    });
    m_pendingMesh = int(m_meshScratch.size());

    for (const ChunkPos pos : m_meshScratch) {
        if (m_jobsInFlight >= inflightCap)
            break;
        const auto it = m_chunks.find(pos);
        Entry& e = it->second;
        e.state = ChunkState::Meshing;

        std::array<const Chunk*, 6> nb{
            voxelsAt({pos.x + 1, pos.y, pos.z}), voxelsAt({pos.x - 1, pos.y, pos.z}),
            voxelsAt({pos.x, pos.y + 1, pos.z}), voxelsAt({pos.x, pos.y - 1, pos.z}),
            voxelsAt({pos.x, pos.y, pos.z + 1}), voxelsAt({pos.x, pos.y, pos.z - 1}),
        };
        e.voxels->retainJob();
        for (const Chunk* n : nb)
            if (n)
                n->retainJob();

        ++m_jobsInFlight;
        --m_pendingMesh;
        Chunk* voxels = e.voxels;
        jobs.submit([this, voxels, nb, pos] {
            // one mesher + scratch per worker thread, reused across jobs
            thread_local GreedyMesher mesher;
            thread_local std::vector<std::uint64_t> quadScratch;
            const auto t0 = Clock::now();
            const GreedyStats stats = mesher.mesh(*voxels, nb.data(), quadScratch);
            m_meshNanos.fetch_add(nsSince(t0), std::memory_order_relaxed);
            m_meshCount.fetch_add(1, std::memory_order_relaxed);

            MeshResult result{pos, {quadScratch.begin(), quadScratch.end()}, stats};
            voxels->releaseJob();
            for (const Chunk* n : nb)
                if (n)
                    n->releaseJob();
            std::lock_guard lock(m_meshDoneMutex);
            m_meshDone.push_back(std::move(result));
        });
    }

    // drain mesh results; the GL upload is the only part with a time budget
    {
        std::lock_guard lock(m_meshDoneMutex);
        for (auto& r : m_meshDone)
            m_uploads.push_back(std::move(r));
        m_meshDone.clear();
    }
    const auto uploadStart = Clock::now();
    std::size_t uploaded = 0;
    for (auto& r : m_uploads) {
        if (uploaded > 0 && msSince(uploadStart) > kUploadBudgetMs)
            break; // rest carries over to next frame; always at least one per frame
        const auto it = m_chunks.find(r.pos);
        if (it != m_chunks.end() && it->second.state == ChunkState::Meshing) {
            Entry& e = it->second;
            e.quads = int(r.quads.size());
            if (!r.quads.empty()) {
                GLuint ssbo = 0;
                glCreateBuffers(1, &ssbo);
                glNamedBufferStorage(ssbo, GLsizeiptr(r.quads.size() * sizeof(std::uint64_t)),
                                     r.quads.data(), 0);
                e.ssbo = ssbo;
            }
            e.state = ChunkState::Meshed;
        }
        --m_jobsInFlight;
        ++uploaded;
    }
    m_uploads.erase(m_uploads.begin(), m_uploads.begin() + std::ptrdiff_t(uploaded));
    m_uploadMsLast = msSince(uploadStart);
}

void ChunkManager::collectVisible(const Frustum& frustum, const glm::dvec3& cameraPos,
                                  std::vector<DrawItem>& out) const {
    out.clear();
    for (const auto& item : m_chunks) {
        const Entry& e = item.second;
        if (e.state != ChunkState::Meshed || e.quads == 0)
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
    s.jobsInFlight = m_jobsInFlight;
    s.pendingUploads = int(m_uploads.size());
    s.uploadMsLast = m_uploadMsLast;
    const int gc = m_genCount.load(std::memory_order_relaxed);
    const int mc = m_meshCount.load(std::memory_order_relaxed);
    s.genMsAvg = gc > 0 ? double(m_genNanos.load(std::memory_order_relaxed)) * 1e-6 / gc : 0.0;
    s.meshMsAvg = mc > 0 ? double(m_meshNanos.load(std::memory_order_relaxed)) * 1e-6 / mc : 0.0;
    for (const auto& item : m_chunks) {
        const Entry& e = item.second;
        if (e.state == ChunkState::Meshed)
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
            s.genMsAvg, s.meshMsAvg);
}

} // namespace vx
