#include "world/ChunkManager.hpp"

#include "core/Log.hpp"
#include "io/ChunkSerializer.hpp"
#include "jobs/JobSystem.hpp"
#include "light/LightEngine.hpp"
#include "mesh/Connectivity.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <thread>
#include <utility>

namespace vx {

ChunkManager::ChunkManager(std::filesystem::path saveDir, int seed)
    : m_seed(seed), m_gen(seed), m_regions(std::move(saveDir)) {}


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

// detail level from squared chunk distance. near chunks full res; far chunks downsample
// so their meshes collapse to a fraction of the triangles (main pass + shadows). the far
// bands (L4 16x, L5 32x = one dominant block per chunk) make big render distances almost
// free — a very-distant chunk is a handful of quads. near bands are unchanged, so raising
// the radius only adds cheap coarse geometry out on the horizon.
// Bands scale with the render distance instead of sitting at fixed chunk counts, by `s`
// (ChunkManager::lodScale). They were tuned at radius 16 and stayed there, so raising the
// radius did not push detail outward with it — it just added more of the coarsest band, and
// full detail still ran out at 8 chunks no matter how far you could see. The ratios below
// reproduce the old 8/14/22/30/42 exactly at radius 16, so that case is unchanged.
int lodForDist2(std::int64_t d2, double s) {
    const auto band = [&](double chunks) {
        const double d = chunks * s;
        return std::int64_t(d * d);
    };
    // Full detail runs to 12 chunks, up from 8: the first downsample was well inside the
    // range you can still resolve individual blocks at, so it read as the ground coarsening
    // in front of you.
    //
    // ONLY that band moved. The 1 -> 2 transition is where kCoarseLod kicks in and chunks
    // start generating without caves, and at the default radius 16 the loaded ring
    // is only 17 chunks — push that threshold out to 24 and nothing is ever coarse, which
    // silently turns the whole coarse-gen optimisation off and costs +78% quads. It stays
    // at 14. The smoke regen group samples a column at 15 chunks precisely to hold this.
    if (d2 <= band(12)) return 0;
    if (d2 <= band(14)) return 1;
    if (d2 <= band(22)) return 2;
    if (d2 <= band(30)) return 3;
    if (d2 <= band(42)) return 4;
    return 5;
}

// chunks meshed at this LOD or coarser generate without caves (they'd be averaged away by
// the downsample; trees still generate, see TerrainGenerator::generate). approaching one —
// desired LOD drops below this — re-generates it detailed. caves cost ~4x gen/mesh/quads,
// so skipping them far out is the big win.
constexpr int kCoarseLod = 2;

constexpr int kFaceOffsets[6][3] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                    {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};

// Faces whose neighbour meshes coarser than we do. The mesher skirts those borders
// instead of culling against voxels the neighbour isn't actually drawing — see
// GreedyMesher::mesh. LOD comes from horizontal distance alone, so the two vertical
// faces always match and never end up in the mask.
//
// The neighbour is judged from a distance nudged OUTWARD by kSkirtMargin, which makes the
// answer deliberately pessimistic: we start skirting a border a couple of chunks before
// that neighbour actually coarsens. That is what lets the mask go stale harmlessly. A
// mesh is built against one camera centre and keeps that mask until the chunk re-meshes
// for its own reasons, so an exact mask would need every LOD change to invalidate its
// finer neighbours — and after a teleport (where half the world re-LODs) that cost 4.5x
// the frames to re-stream. Over-skirting only leaves a hidden wall behind the neighbour's
// geometry; under-skirting leaves a hole, so this errs to the side that can't be seen.
// The margin is safe because neighbours are one chunk apart: by the time the camera has
// receded far enough for a neighbour to coarsen, this chunk has crossed its own band too.
constexpr double kSkirtMargin = 2.0;

std::uint8_t coarseNeighborMask(ChunkPos pos, int lod, ChunkPos center, double scale) {
    std::uint8_t mask = 0;
    for (int f = 0; f < 6; ++f) {
        const ChunkPos n{pos.x + kFaceOffsets[f][0], pos.y + kFaceOffsets[f][1],
                         pos.z + kFaceOffsets[f][2]};
        const double d = std::sqrt(double(horizontalDist2(n, center))) + kSkirtMargin;
        if (lodForDist2(std::int64_t(d * d), scale) > lod)
            mask |= std::uint8_t(1u << f);
    }
    return mask;
}

// The visibility flood (Tommo's cave culling) only ever steps away from the camera: on each axis a
// path moves one way. So a step in direction d from pos is allowed unless pos already lies on the
// other side of the camera along d's axis.
bool floodMayStep(ChunkPos pos, ChunkPos camera, int d) {
    const int off = d < 2 ? pos.x - camera.x : d < 4 ? pos.y - camera.y : pos.z - camera.z;
    return (d & 1) ? off <= 0 : off >= 0;
}

// Air in the chunk joins at least one face it was entered by to the face d leaves through.
bool floodPasses(std::uint16_t connectivity, std::uint8_t entered, int d) {
    for (int e = 0; e < 6; ++e)
        if (e != d && (entered >> e & 1u) && facesConnected(connectivity, e, d))
            return true;
    return false;
}

constexpr std::uint8_t kFloodQueued = 1u << 6;

} // namespace

// What landing a job's empty result would have done, without the job: the chunk is drawn as
// nothing at this LOD, and cave culling may flood through it.
void ChunkManager::landEmpty(ChunkPos pos, int lod, Renderer& renderer) {
    Entry& e = m_chunks.find(pos)->second;
    if (e.range.count != 0) {
        renderer.free(e.range);
        e.range = {};
    }
    if (e.waterRange.count != 0) {
        renderer.free(e.waterRange);
        e.waterRange = {};
    }
    if (e.quadBuf) {
        m_quadPool.checkin(e.quadBuf);
        e.quadBuf = nullptr;
    }
    if (e.waterBuf) {
        m_waterPool.checkin(e.waterBuf);
        e.waterBuf = nullptr;
    }
    e.quads = 0;
    e.waterQuads = 0;
    e.connectivity = computeConnectivity(*e.voxels);
    e.sunlit = false;
    e.lod = std::int8_t(lod);
    e.remesh = false;
    e.lightDirty = false;
    e.state = ChunkState::Meshed;
    noteDirty(pos);
    m_visDirty = true;
    m_meshScanDirty = true;
}

double ChunkManager::lodScale() const {
    return std::min(std::max(radius, 16) / 16.0, double(std::max(detailScale, 1.0f)));
}

void ChunkManager::noteDirty(ChunkPos p) {
    m_dirty = true;
    // before the first update there is no centre, and everything counts as near
    if (m_center.x == INT32_MIN) {
        m_dirtyNearest = 0.0f;
        return;
    }
    const double dx = double(p.x - m_center.x), dy = double(p.y - m_center.y),
                 dz = double(p.z - m_center.z);
    const float blocks = float(std::sqrt(dx * dx + dy * dy + dz * dz) * kChunkSize) -
                         0.5f * float(kChunkSize);
    m_dirtyNearest = std::min(m_dirtyNearest, std::max(blocks, 0.0f));
}

ChunkManager::ColumnHeights ChunkManager::columnHeights(int cx, int cz) const {
    const ChunkPos key{cx, 0, cz};
    if (const auto it = m_columnHeights.find(key); it != m_columnHeights.end())
        return it->second;
    // The top is the highest of nine samples across the column, not the one at its centre: a
    // steep peak stands tens of blocks higher at a corner, and the band stopped short of it.
    // The chunk holding those blocks then had no neighbour above it and never meshed, and
    // anything over it was never even generated (reproduced: a peak at radius 32 seen from a
    // low camera). The margin covers what grows on the ground - a spruce reaches ~17 above it,
    // a lighthouse ~21 - and terrain rising between the samples.
    constexpr int kMargin = kChunkSize;
    constexpr int kAt[3] = {0, kChunkSize / 2, kChunkSize - 1};
    ColumnHeights h;
    h.centre = m_gen.surfaceHeight(cx * kChunkSize + kChunkSize / 2, cz * kChunkSize + kChunkSize / 2);
    h.top = h.centre;
    for (const int dz : kAt)
        for (const int dx : kAt)
            h.top = std::max(h.top, m_gen.surfaceHeight(cx * kChunkSize + dx, cz * kChunkSize + dz));
    h.top += kMargin;
    if (m_columnHeights.size() > 200000)
        m_columnHeights.clear(); // a long flight's worth; it only ever costs a recompute
    m_columnHeights.emplace(key, h);
    return h;
}

void ChunkManager::columnRange(int cx, int cz, int& y0, int& y1, bool& skyAtTop) const {
    const ColumnHeights heights = columnHeights(cx, cz);
    const int surf = heights.centre;
    // chunk holding the first open-air voxel above the surface
    const int surfChunk = (surf + 1) >= 0 ? (surf + 1) / kChunkSize
                                          : -((-(surf + 1) + kChunkSize - 1) / kChunkSize);
    y0 = m_center.y - kVerticalRadius;
    const int bandTop = m_center.y + kVerticalRadius;
    if (bandTop + 1 >= surfChunk) {
        // over everything the column can hold, so the top chunk is open sky everywhere; the
        // buried test above stays on the centre sample, as it always was. Never past
        // bandTop + 1, where rescan() evicts: a chunk loaded above that is thrown out again at
        // the next rescan, its column never completes, and the whole column stays unlit
        // (a --tour run left 208 chunks UNLIT before this clamp).
        const int topChunk = heights.top >= 0 ? heights.top / kChunkSize
                                              : -((-heights.top + kChunkSize - 1) / kChunkSize);
        y1 = std::max({bandTop, surfChunk, std::min(topChunk, bandTop + 1)});
        skyAtTop = true;
    } else {
        y1 = bandTop; // buried: a chunk of separation is 32 blocks, sky reaches 15
        skyAtTop = false;
    }
}

bool ChunkManager::columnPresent(ChunkPos col) const {
    // "Complete" now means the resident range is present, because that is all there ever
    // will be — an unbounded column can never be fully loaded.
    int y0 = 0, y1 = 0;
    bool sky = false;
    columnRange(col.x, col.z, y0, y1, sky);
    for (int cy = y0; cy <= y1; ++cy)
        if (!m_chunks.contains({col.x, cy, col.z}))
            return false;
    return true;
}

const Chunk* ChunkManager::voxelsAt(ChunkPos pos) const {
    const auto it = m_chunks.find(pos);
    return it != m_chunks.end() ? it->second.voxels : nullptr;
}

void ChunkManager::rescan(ChunkPos center) {
    m_meshScanDirty = true; // moved: LOD bands shifted, new chunks incoming
    // +2, not +1: owner report, live — chunks visibly popping in/unrendering when moving
    // fast (fly speed is user-scrollable well past what generation can keep pace with, by
    // design — there's no fix for that end of the range, only more runway before you reach
    // it). One extra ring of pre-generated buffer ahead of the camera is real runway: it
    // does not make generation faster, but it means outrunning it takes one more ring of
    // travel at any given speed, which is where "how far can I go before I see it" actually
    // lives for a streaming world.
    const int dataR = radius + 2;

    // rebuilt from scratch so chunks that came back into range drop off the evict list
    m_evictList.clear();
    for (const auto& item : m_chunks)
        if (std::abs(item.first.x - center.x) > dataR + 1 ||
            std::abs(item.first.z - center.z) > dataR + 1 ||
            std::abs(item.first.y - center.y) > kVerticalRadius + 1)
            m_evictList.push_back(item.first);

    m_genQueue.clear();
    // Vertical band around the camera, not a fixed column: there is no world ceiling, so
    // "the whole column" is not a finite thing to load. Walk up and the band follows.
    for (int dz = -dataR; dz <= dataR; ++dz)
        for (int dx = -dataR; dx <= dataR; ++dx) {
            int y0 = 0, y1 = 0;
            bool sky = false;
            columnRange(center.x + dx, center.z + dz, y0, y1, sky);
            for (int y = y0; y <= y1; ++y) {
                const ChunkPos pos{center.x + dx, y, center.z + dz};
                if (!m_chunks.contains(pos))
                    m_genQueue.push_back(pos);
            }
        }
    std::sort(m_genQueue.begin(), m_genQueue.end(), [center](ChunkPos a, ChunkPos b) {
        return horizontalDist2(a, center) > horizontalDist2(b, center);
    });
}

ChunkManager::LodAudit ChunkManager::auditLod() const {
    LodAudit a;
    for (const auto& item : m_chunks) {
        const ChunkPos pos = item.first;
        const Entry& e = item.second;
        // Matches rescan()'s own eviction threshold (dataR + 1 where dataR = radius + 2)
        // exactly — anything past this should have been on the evict list and gone by now.
        if (std::abs(pos.x - m_center.x) > radius + 3 || std::abs(pos.z - m_center.z) > radius + 3 ||
            std::abs(pos.y - m_center.y) > kVerticalRadius + 1) {
            ++a.stranded;
            continue;
        }
        // Beyond `radius` itself is the buffer band the scan loop deliberately skips too
        // (rescan()'s evict threshold is radius+3, the scan's is radius) — nothing to check
        // for a chunk sitting in that gap on purpose.
        if (std::abs(pos.x - m_center.x) > radius || std::abs(pos.z - m_center.z) > radius)
            continue;
        if (e.state != ChunkState::Meshed)
            continue; // still in flight isn't a bug on its own; busy() should have caught it
        const int desired = lodForDist2(horizontalDist2(pos, m_center), lodScale());
        if (e.lod != desired)
            ++a.lodMismatch;
        // Exactly the self-heal condition in update() that queues a coarse->detailed regen
        // (dirty is excluded there on purpose too — see #64 — an unsaved edit keeps a chunk
        // coarse until it's saved, which is correct, not stuck).
        if (desired < kCoarseLod && !e.detailed && !e.voxels->uniform() && !e.dirty)
            ++a.coarseInDetailBand;
    }
    return a;
}

void ChunkManager::processEvictions(Renderer& renderer, JobSystem& jobs) {
    for (std::size_t i = 0; i < m_evictList.size();) {
        const auto it = m_chunks.find(m_evictList[i]);
        if (it == m_chunks.end()) {
            m_evictList[i] = m_evictList.back();
            m_evictList.pop_back();
            continue;
        }
        Entry& e = it->second;
        const bool inFlight = e.state == ChunkState::Generating ||
                              e.state == ChunkState::Meshing || e.voxels->hasJobRefs() ||
                              (e.pendingDetail && e.pendingDetail->hasJobRefs());
        if (inFlight) {
            ++i; // a worker still holds it — retry next frame
            continue;
        }
        // an in-flight re-gen whose job already finished: the entry vanishes here, and the
        // regen-done drain releases that orphaned chunk when it can't find the entry
        // Persist player edits; everything else regenerates from the seed (caching generated
        // chunks measured as a no-op, SPEC #16).
        // Serialize on a worker, never here. A teleport evicts thousands of chunks at
        // once and doing this inline measured as a 128 ms freeze — and blocking the main
        // thread on disk work is exactly what the streaming design forbids. The job holds
        // a ref, so the chunk cannot return to the pool underneath it; this entry simply
        // waits out another pass or two of the evict list, which it is already on. It is
        // not one of the counted jobs, so it has its own count, which busy(), loadWorld and
        // saveDirty wait on (SPEC #142).
        if (e.dirty && !e.saving) {
            e.saving = true;
            Chunk* voxels = e.voxels;
            const ChunkPos at = m_evictList[i];
            voxels->retainJob();
            m_savesInFlight.fetch_add(1, std::memory_order_relaxed);
            jobs.submit([this, voxels, at] {
                if (const int delay = m_saveDelayMs.load(std::memory_order_relaxed); delay > 0)
                    std::this_thread::sleep_for(std::chrono::milliseconds(delay));
                std::vector<std::uint8_t> bytes;
                ChunkSerializer::serialize(*voxels, bytes);
                m_regions.save(at, std::move(bytes));
                voxels->releaseJob();
                m_savesInFlight.fetch_sub(1, std::memory_order_release);
            });
            ++i; // comes back once the job drops its ref
            continue;
        }
        if (e.range.count != 0) {
            renderer.free(e.range);
            noteDirty(m_evictList[i]);
        }
        if (e.waterRange.count != 0) {
            renderer.free(e.waterRange);
            noteDirty(m_evictList[i]);
        }
        // Return the mesh buffers to the pool before the Entry (and its pointers) are gone
        // for good — miss this and the slot is still sitting in m_slots, just unreachable
        // from the free list, so the pool grows a little every eviction for the rest of the
        // session instead of steady-stating.
        if (e.quadBuf)
            m_quadPool.checkin(e.quadBuf);
        if (e.waterBuf)
            m_waterPool.checkin(e.waterBuf);
        m_pool.release(e.voxels);
        // Only a column leaving the ring loses its seed. The band trimming a chunk off the
        // bottom or top when the camera changes height leaves the rest of the column lit
        // correctly, and if the chunk comes back it is lit on arrival like any other band
        // growth. Dropping the seed there made the self-heal relight every column in view on
        // each vertical chunk step - ~4000 at radius 32, half a minute of seeding during which
        // nothing new anywhere could light or mesh: walk over a hill and the world stopped
        // rendering a few chunks out (owner report).
        const ChunkPos gone = m_evictList[i];
        if (std::abs(gone.x - m_center.x) > radius + 3 || std::abs(gone.z - m_center.z) > radius + 3) {
            m_seededColumns.erase(ChunkPos{gone.x, 0, gone.z});
            m_provisionalColumns.erase(ChunkPos{gone.x, 0, gone.z});
            m_reseedColumns.erase(ChunkPos{gone.x, 0, gone.z});
        }
        m_chunks.erase(it);
        // a neighbour waiting on this chunk may count it as out of its band now
        for (const auto& o : kFaceOffsets)
            touchScan({gone.x + o[0], gone.y + o[1], gone.z + o[2]});
        m_visDirty = true; // reshuffles meta order; see the note at try_emplace
        m_evictList[i] = m_evictList.back();
        m_evictList.pop_back();
    }
}

void ChunkManager::rebuildArena(Renderer& renderer) {
    renderer.hardReset();
    for (auto& item : m_chunks) {
        Entry& e = item.second;
        if (e.state != ChunkState::Meshed)
            continue;
        if (e.quadBuf && !e.quadBuf->empty()) {
            e.range = renderer.alloc(*e.quadBuf);
            if (e.range.count == 0)
                VX_ERROR("arena rebuild: chunk still does not fit — arena too small");
        }
        if (e.waterBuf && !e.waterBuf->empty())
            e.waterRange = renderer.alloc(*e.waterBuf);
    }
    m_dirty = true;
    m_dirtyNearest = 0.0f; // every range moved
}

void ChunkManager::update(JobSystem& jobs, Renderer& renderer, LightEngine& light,
                          const glm::dvec3& cameraPos) {
    // Y is part of the centre now. It was hard-coded to 0 because the world was a fixed
    // stack of columns and the vertical extent never moved — with the ceiling gone, that
    // pinned the resident band around y=0 no matter where the camera actually was, so
    // flying up left you outside your own loaded world.
    const ChunkPos center{int(std::floor(cameraPos.x / kChunkSize)),
                          int(std::floor(cameraPos.y / kChunkSize)),
                          int(std::floor(cameraPos.z / kChunkSize))};
    const auto phase0 = Clock::now();
    if (center != m_center || radius != m_lastRadius) {
        m_center = center;
        m_lastRadius = radius;
        rescan(center);
    }
    ++m_updates;
    const double rescanMs = msSince(phase0);
    m_phase.rescanMs = std::max(m_phase.rescanMs, rescanMs);
    m_phaseSum.rescanMs += rescanMs;

    const auto phase1 = Clock::now();
    processEvictions(renderer, jobs);
    const double evictMs = msSince(phase1);
    m_phase.evictMs = std::max(m_phase.evictMs, evictMs);
    m_phaseSum.evictMs += evictMs;

    // drain finished generations (cheap: a state flip per chunk)
    {
        std::lock_guard lock(m_genDoneMutex);
        m_genDoneScratch.swap(m_genDone);
    }
    if (!m_genDoneScratch.empty())
        m_meshScanDirty = true; // new data may be meshable now
    for (const ChunkPos pos : m_genDoneScratch) {
        const auto it = m_chunks.find(pos);
        if (it != m_chunks.end() && it->second.state == ChunkState::Generating)
            it->second.state = ChunkState::Ready;
        touchScan(pos);
        --m_jobsInFlight;
        --m_genInFlight;

        // once the whole vertical column has data, it can be light-seeded (sky descent
        // spans all of it at once)
        const ChunkPos col{pos.x, 0, pos.z};
        if (!m_seededColumns.contains(col)) {
            bool complete = true;
            int cy0 = 0, cy1 = 0;
            bool colSky = false;
            columnRange(col.x, col.z, cy0, cy1, colSky);
            for (int cy = cy0; cy <= cy1 && complete; ++cy) {
                const auto cit = m_chunks.find({pos.x, cy, pos.z});
                complete = cit != m_chunks.end() && cit->second.state != ChunkState::Generating;
            }
            if (complete) {
                m_seededColumns.insert(col);
                m_seedQueue.push_back(col);
                // this column just appeared next to columns that were lit as if it were
                // solid rock; they need their light redone against the real terrain
                static constexpr int kSide[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
                for (const auto& d : kSide) {
                    const ChunkPos n{col.x + d[0], 0, col.z + d[1]};
                    if (m_provisionalColumns.erase(n) != 0) {
                        m_seededColumns.erase(n);
                        m_seedQueue.push_back(n);
                    }
                }
            }
        } else if (it != m_chunks.end() && !it->second.lit) {
            // The column was already seeded when this chunk turned up, so the column seed
            // will never run for it — and meshing waits on lit, so it would never be
            // meshed either. The band grows without evicting anything: the load range is
            // center.y +- kVerticalRadius while eviction only fires past
            // kVerticalRadius + 1, so descending one chunk hands every column you are
            // standing in a new bottom chunk and drops none. Measured at 550 chunks in one
            // column each, at a single --tour 32 stop.
            //
            // Light just this chunk. Re-seeding the column would relight and remesh all
            // fifteen of them, times every resident column, for one chunk of descent.
            int cy0 = 0, cy1 = 0;
            bool colSky = false;
            columnRange(col.x, col.z, cy0, cy1, colSky);
            ++m_lateSeeds;
            light.seedNewChunk(pos, colSky && pos.y >= cy1);
            it->second.lit = true;
            touchScan(pos);
            m_meshScanDirty = true;
        }
    }
    m_genDoneScratch.clear();

    // Budgeted light seeding and propagation, then remesh whatever light touched. In the game
    // the middle step runs on the light thread while the frame draws (startLightSlice /
    // finishLightSlice) and the last frame's slice is already applied by now; anything else
    // that calls update() gets all three steps here.
    if (m_sliceRunning)
        finishLightSlice(light);
    if (!std::exchange(m_sliceDriven, false)) {
        prepareLightSlice(light);
        runLightSlice(light);
        applyLightSlice(light);
    }

    // keep enough jobs in flight to feed the workers without racing miles ahead of
    // the camera — stale far-away work is wasted work when it moves
    const auto scanStart = Clock::now();
    const int inflightCap = std::max(8, jobs.workerCount() * 6);
    // Generation used to go first and take every slot, and while the camera moves its queue
    // never empties - so meshing only got what generation left over, which flying at radius 32
    // was nothing most frames, and ~10k chunks sat ready to mesh. With meshes waiting it now
    // keeps to half the slots, counted on its own jobs so meshing cannot starve it either.
    const int genCap = m_pendingMesh > 0 ? inflightCap / 2 : inflightCap;

    while (m_genInFlight < genCap && m_jobsInFlight < inflightCap && !m_genQueue.empty()) {
        const ChunkPos pos = m_genQueue.back();
        m_genQueue.pop_back();
        if (m_chunks.contains(pos))
            continue;
        Chunk* voxels = m_pool.acquire();
        // Pooled chunks keep their light array across reuse (fillLight only rewrites
        // values, it never frees), so a recycled chunk arrives carrying the PREVIOUS
        // occupant's light. Seeding does not fully overwrite it: the sky descent stops at
        // the first opaque block, and nothing clears the block-light nibbles at all, so
        // whatever was under the old terrain's surface survives under the new one. That
        // made lighting depend on which chunk happened to be recycled into this slot.
        voxels->fillLight(0);
        voxels->retainJob();
        // far chunks generate coarse (no caves/trees); re-gen'd detailed when approached
        const bool detailed = lodForDist2(horizontalDist2(pos, center), lodScale()) < kCoarseLod;
        Entry entry; // state defaults to Generating
        entry.voxels = voxels;
        entry.detailed = detailed;
        m_chunks.try_emplace(pos, std::move(entry));
        // The cave-cull visibility bits are indexed by META ORDER, and collectDrawInfo
        // rebuilds that order by walking m_chunks — an open-addressed map, so ANY insert
        // or erase can reshuffle every index. Leaving the bits alone across that reshuffle
        // points each one at a different chunk, which culls whatever it lands on: chunk
        // tops blinking out and the caves under them showing through while you fly.
        m_visDirty = true;
        ++m_jobsInFlight;
        ++m_genInFlight;
        jobs.submit([this, voxels, pos, detailed] {
            const auto t0 = Clock::now();
            // disk first: a saved (player-edited) chunk loads instead of regenerating.
            // load() is thread-safe; the read happens here on the worker, never the
            // main thread. absent chunks fall back to deterministic gen.
            if (!m_regions.load(pos, *voxels))
                m_gen.generate(*voxels, pos, detailed);
            m_genNanos.fetch_add(nsSince(t0), std::memory_order_relaxed);
            m_genCount.fetch_add(1, std::memory_order_relaxed);
            voxels->releaseJob();
            std::lock_guard lock(m_genDoneMutex);
            m_genDone.push_back(pos);
        });
    }

    // schedule meshing for chunks whose whole neighborhood has data AND light, nearest
    // first — meshes bake light values into quads, so lighting has to land first. also
    // re-mesh already-meshed chunks whose LOD should change as the camera moves (they
    // stay drawn at the old LOD until the new mesh swaps in, so no flicker).
    // this loop is O(loaded chunks); skip it whenever nothing could have changed the mesh
    // set (no streaming in flight and the camera hasn't moved). a loaded, still world then
    // costs nothing here instead of re-scanning ~36k chunks every single frame.
    const bool doMeshScan = m_meshScanDirty || m_scanConfirm || m_jobsInFlight > 0 ||
                            !m_genQueue.empty() || !m_uploads.empty();
    // Walking every resident chunk was 5.4 ms of a 12 ms frame flying at radius 32, nearly all
    // of it finding again the same chunks still waiting for a slot. The whole world is only
    // walked when the answer can have changed everywhere at once - the camera changed chunk
    // (LOD bands, band and ring all move), the radius or detail changed - plus every
    // kFullScanEvery as a safety net, and once after the last partial scan before busy() can
    // clear. In between: what something touched, their neighbours (a chunk's readiness is its
    // neighbours'), and the nearest of the candidates still waiting.
    const auto scanNow = Clock::now();
    const double scanLod = lodScale();
    const bool fullScan =
        doMeshScan && (!(center == m_scanCenter) || radius != m_scanRadius ||
                       scanLod != m_scanLodScale || scanNow - m_lastFullScan >= kFullScanEvery ||
                       (m_scanConfirm && m_scanTouched.empty() && m_scanCarry.empty()));
    m_meshScanDirty = false;
    m_meshScratch.clear();
    m_regenScratch.clear();
    m_emptyScratch.clear();
    // The self-heal below is a safety net for a rare path, so it rides every other full scan.
    const bool heal = fullScan && (m_healTick++ & 1u) == 0;
    const bool seedingIdle = m_seedQueue.empty() && !light.seedInProgress();
    const auto consider = [&](const ChunkPos pos, Entry& e) {
        // Catch columns that became complete without anyone noticing. A column is only
        // ever put up for seeding from the gen-done drain, keyed on one of ITS chunks
        // finishing generation — so a column that completes by any other route (an entry
        // erased while its job was in flight, a re-gen swap, a rescan reshuffling what was
        // queued) is never offered at all, and stays unseeded forever holding sky light 0.
        // That is open sky rendering pitch black regardless of time of day.
        //
        // The scan already walks every chunk and only runs when something changed, so
        // checking here costs a set lookup on work that was happening anyway.
        // NB: gated on m_seededColumns, never on e.lit. a column leaving the ring drops from
        // m_seededColumns with its first evicted chunk while the rest wait their turn still
        // flagged lit, so e.lit says "done" for a column that no longer has a seed to its name.
        if (heal && e.state != ChunkState::Generating && !e.pendingDetail) {
            const ChunkPos col{pos.x, 0, pos.z};
            if (!m_seededColumns.contains(col) && columnPresent(col)) {
                bool ready = true;
                // The resident band, not 0..kWorldChunksY. #27 left this walking the old
                // fixed column: stand above y=384 or below 0 and none of those chunks is
                // resident, so `ready` never came true and the self-heal never healed.
                int y0 = 0, y1 = 0;
                bool sky = false;
                columnRange(col.x, col.z, y0, y1, sky);
                for (int cy = y0; cy <= y1 && ready; ++cy)
                    if (const auto cit = m_chunks.find({col.x, cy, col.z});
                        cit == m_chunks.end() || cit->second.state == ChunkState::Generating)
                        ready = false;
                if (ready) {
                    m_seededColumns.insert(col);
                    m_seedQueue.push_back(col);
                }
            } else if (!e.lit && seedingIdle && m_seededColumns.contains(col) &&
                       !m_reseedColumns.contains(col)) {
                // An unlit chunk in a column whose seed is done. A regen landing clears the
                // column's lit flags over the band and queues a re-seed, and if the camera
                // changes chunk height before that runs, the re-seed covers a different band:
                // the chunk at the old edge is never marked again and never meshes. Light it
                // the way a chunk arriving in a seeded column is lit.
                int cy0 = 0, cy1 = 0;
                bool colSky = false;
                columnRange(col.x, col.z, cy0, cy1, colSky);
                light.seedNewChunk(pos, colSky && pos.y >= cy1);
                e.lit = true;
                if (e.meshingLod >= 0)
                    e.lightDirty = true;
                else if (e.state == ChunkState::Meshed)
                    e.remesh = true;
                m_meshScanDirty = true;
            }
        }
        if (!e.lit || e.pendingDetail) // pendingDetail: a detailed re-gen is landing
            return;
        if (std::abs(pos.x - center.x) > radius ||
            std::abs(pos.z - center.z) > radius)
            return;
        const int desired = lodForDist2(horizontalDist2(pos, center), lodScale());
        const bool firstMesh = e.state == ChunkState::Ready;
        const bool reLod = e.state == ChunkState::Meshed && (e.lod != desired || e.remesh);
        // A coarse chunk close enough to want detail re-generates (with caves). The regen is
        // deliberately NOT keyed off a LOD
        // change: it used to require reLod, so a chunk that got its FIRST mesh at a fine LOD
        // while still coarse was stuck cave-less forever — e.lod already equalled desired,
        // so reLod could never fire for it again. That happens whenever the camera moves
        // between a chunk being queued for generation (which picks coarse/detailed) and
        // being meshed (which picks the LOD), which vertical streaming made common.
        //
        // `!e.dirty` is load-bearing, not an optimisation. The regen job below tries
        // `m_regions.load` first and only calls the generator if nothing is saved yet — so a
        // chunk edited while coarse but not yet WRITTEN to disk (saveDirty hasn't run, it
        // hasn't been evicted) would regenerate from the seed and silently discard the edit
        // the moment it crossed into the detail band, which for a coarse chunk means gaining
        // caves it never had. That reads as "I dig a hole and the world corrupts, stone
        // reappears" — found by reading the swap below, which replaces e.voxels outright
        // with no check for unsaved edits. The correct fix is not to serialize it inline
        // here to unblock the regen (INFO.md: never block the main thread on disk I/O, and
        // this scan runs every frame) — it is to leave a dirty coarse chunk coarse until it
        // is saved by the normal path (F2, or eviction), at which point this condition opens
        // and it regens detailed like any other chunk, loading back exactly what was saved.
        //
        // While that regen is on its way, the cave-less data is re-meshed at the LOD the chunk
        // should be at now. Everything above ground is already in it (the generator only skips
        // caves), and holding the far mesh until the regen landed put LOD 2-5 chunks right at
        // the camera: 410 of the 1183 chunks within 6 of it, on average, flying at 45 b/s.
        if (e.state == ChunkState::Meshed && !e.detailed && desired < kCoarseLod &&
            !e.voxels->uniform() && !e.dirty)
            m_regenScratch.push_back(pos);
        if (!(firstMesh || reLod) || e.meshingLod == desired)
            return;
        // Solid air has nothing to draw at any LOD and bakes no light, so it needs neither a job
        // nor its neighbours: at radius 32 it was 25k of the ~35k chunks waiting to mesh, each
        // holding a job slot like a real mesh.
        if (e.meshingLod < 0 && e.voxels->uniform() && e.voxels->uniformId() == blocks::Air) {
            m_emptyScratch.push_back(pos);
            return;
        }
        bool ready = true;
        for (const auto& o : kFaceOffsets) {
            const ChunkPos n{pos.x + o[0], pos.y + o[1], pos.z + o[2]};
            const auto nit = m_chunks.find(n);
            if (nit == m_chunks.end()) {
                // Outside its column's resident band it will never be loaded, so waiting for it
                // stranded this chunk unmeshed for good: open air above the band (it reaches
                // over the column's terrain) or the dark rock left below it. The old test was
                // the fixed world's floor and ceiling, from before the world had none (INFO #11).
                int ny0 = 0, ny1 = 0;
                bool nsky = false;
                columnRange(n.x, n.z, ny0, ny1, nsky);
                if (n.y < ny0 || n.y > ny1)
                    continue;
                ready = false;
                break;
            }
            if (nit->second.state == ChunkState::Generating || !nit->second.lit) {
                ready = false;
                break;
            }
        }
        if (ready)
            m_meshScratch.push_back(pos);
    };
    std::vector<ChunkPos> carryOn; // waiting candidates not looked at this frame
    if (fullScan) {
        for (auto& item : m_chunks)
            consider(item.first, item.second);
        m_scanTouched.clear();
        m_scanCarry.clear();
        m_lastFullScan = scanNow;
        m_scanCenter = center;
        m_scanRadius = radius;
        m_scanLodScale = scanLod;
        m_scanConfirm = false;
    } else if (doMeshScan) {
        m_scanVisit.clear();
        for (const ChunkPos p : m_scanTouched) {
            m_scanVisit.insert(p);
            for (const auto& o : kFaceOffsets)
                m_scanVisit.insert({p.x + o[0], p.y + o[1], p.z + o[2]});
        }
        m_scanTouched.clear();
        // Only the nearest waiting candidates are looked at again; the rest keep their place
        // unchecked. The centre has not moved since they were found (that would have been a
        // full scan), so nearest stays nearest, and nothing unchecked is ever scheduled.
        const std::size_t look = std::size_t(4 * inflightCap);
        if (m_scanCarry.size() > look) {
            std::nth_element(m_scanCarry.begin(), m_scanCarry.begin() + std::ptrdiff_t(look),
                             m_scanCarry.end(), [center](ChunkPos a, ChunkPos b) {
                                 return horizontalDist2(a, center) < horizontalDist2(b, center);
                             });
            carryOn.assign(m_scanCarry.begin() + std::ptrdiff_t(look), m_scanCarry.end());
            m_scanCarry.resize(look);
        }
        for (const ChunkPos p : m_scanCarry)
            m_scanVisit.insert(p);
        m_scanCarry.clear();
        for (const ChunkPos p : m_scanVisit)
            if (const auto it = m_chunks.find(p); it != m_chunks.end())
                consider(p, it->second);
        m_scanConfirm = true;
    }
    // schedule detailed re-gens for approached coarse chunks. double-buffered into a fresh
    // chunk so the coarse mesh keeps drawing until the detailed one lands (no hole), and
    // the worker only touches its own fresh chunk (no race with neighbour mesh jobs).
    // nearest first so the detail you're walking into fills in before the far stuff.
    // Ahead of the meshes, with a quarter of the slots of their own: they used to get whatever
    // meshing left, which flying was nothing, so chunks came right up to the camera cave-less.
    {
        // every coarse chunk in the detail band lands here each frame, thousands while flying,
        // and at most a quarter of the slots can start - so only those are put in order
        const auto nearer = [center](ChunkPos a, ChunkPos b) {
            return horizontalDist2(a, center) < horizontalDist2(b, center);
        };
        const std::size_t room =
            std::size_t(std::max(0, std::min(inflightCap - m_jobsInFlight, inflightCap / 4 - m_regenInFlight)));
        if (m_regenScratch.size() > room) {
            std::nth_element(m_regenScratch.begin(), m_regenScratch.begin() + std::ptrdiff_t(room),
                             m_regenScratch.end(), nearer);
            // the rest wait for a slot, carried: a partial scan does not walk the world to
            // find them again
            m_scanCarry.insert(m_scanCarry.end(), m_regenScratch.begin() + std::ptrdiff_t(room),
                               m_regenScratch.end());
            m_regenScratch.resize(room);
        }
        std::sort(m_regenScratch.begin(), m_regenScratch.end(), nearer);
    }
    for (std::size_t ri = 0; ri < m_regenScratch.size(); ++ri) {
        const ChunkPos pos = m_regenScratch[ri];
        if (m_jobsInFlight >= inflightCap || m_regenInFlight >= inflightCap / 4) {
            m_scanCarry.insert(m_scanCarry.end(), m_regenScratch.begin() + std::ptrdiff_t(ri),
                               m_regenScratch.end());
            break;
        }
        const auto it = m_chunks.find(pos);
        if (it == m_chunks.end() || it->second.pendingDetail)
            continue;
        Chunk* c2 = m_pool.acquire();
        c2->fillLight(0); // same reuse hazard as the initial gen path
        c2->retainJob();
        it->second.pendingDetail = c2;
        ++m_jobsInFlight;
        ++m_regenInFlight;
        jobs.submit([this, c2, pos] {
            if (!m_regions.load(pos, *c2))
                m_gen.generate(*c2, pos, /*detailed=*/true);
            c2->releaseJob();
            std::lock_guard lock(m_regenDoneMutex);
            m_regenDone.push_back({pos, c2});
        });
    }

    for (const ChunkPos pos : m_emptyScratch)
        landEmpty(pos, lodForDist2(horizontalDist2(pos, center), lodScale()), renderer);
    // Only as many as there are free slots get scheduled, so only those need to be in order.
    const auto nearer = [center](ChunkPos a, ChunkPos b) {
        return horizontalDist2(a, center) < horizontalDist2(b, center);
    };
    const std::size_t slots = std::size_t(std::max(0, inflightCap - m_jobsInFlight));
    if (m_meshScratch.size() > slots) {
        std::nth_element(m_meshScratch.begin(), m_meshScratch.begin() + std::ptrdiff_t(slots),
                         m_meshScratch.end(), nearer);
        std::sort(m_meshScratch.begin(), m_meshScratch.begin() + std::ptrdiff_t(slots), nearer);
    } else {
        std::sort(m_meshScratch.begin(), m_meshScratch.end(), nearer);
    }
    m_pendingMesh = int(m_meshScratch.size() + carryOn.size());

    for (std::size_t mi = 0; mi < m_meshScratch.size(); ++mi) {
        const ChunkPos pos = m_meshScratch[mi];
        if (m_jobsInFlight >= inflightCap) {
            m_scanCarry.insert(m_scanCarry.end(), m_meshScratch.begin() + std::ptrdiff_t(mi),
                               m_meshScratch.end());
            break;
        }
        const auto it = m_chunks.find(pos);
        Entry& e = it->second;
        const int lod = lodForDist2(horizontalDist2(pos, center), lodScale());
        if (e.state == ChunkState::Ready)
            e.state = ChunkState::Meshing; // first mesh: not drawn until it lands
        // (a re-LOD or relight keeps state Meshed so the old mesh stays drawn meanwhile)
        e.remesh = false;
        e.meshingLod = std::int8_t(lod);

        const std::uint8_t skirt = coarseNeighborMask(pos, lod, center, lodScale());

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
        // Checked out here, on the main thread, and captured as raw pointers — the worker
        // dereferences them directly and never touches the pool's own bookkeeping (the free
        // list, or growing its storage), so there is nothing for it to race the main thread
        // over; see MeshBufferPool.hpp for why that matters. It also gets exclusive access
        // to the pointee for the job's duration, the same ownership discipline
        // retainJob()/releaseJob() already gives it over the voxels. The mesher writes
        // STRAIGHT into pool storage now: no thread_local scratch, no copy at the end. That
        // copy — `{scratch.begin(), scratch.end()}`, a fresh heap allocation once per
        // meshed chunk — was the ~11 heap-allocs/update left over after JobSystem stopped
        // allocating on submit (SPEC.md #67).
        std::vector<PackedQuad>* quadBuf = m_quadPool.checkout();
        std::vector<PackedQuad>* waterBuf = m_waterPool.checkout();
        jobs.submit([this, voxels, nb, pos, lod, skirt, quadBuf, waterBuf] {
            // one per worker thread, reused across jobs, and never destroyed on purpose: see
            // computeConnectivity (a thread_local with a destructor corrupts the heap when a
            // worker exits, on this toolchain)
            thread_local GreedyMesher& mesher = *new GreedyMesher();
            const auto t0 = Clock::now();
            const GreedyStats stats =
                mesher.mesh(*voxels, nb.data(), *quadBuf, *waterBuf, lod, skirt);
            const std::uint16_t connectivity = computeConnectivity(*voxels);
            m_meshNanos.fetch_add(nsSince(t0), std::memory_order_relaxed);
            m_meshCount.fetch_add(1, std::memory_order_relaxed);

            MeshResult result{pos, quadBuf, waterBuf, stats, connectivity, std::int8_t(lod)};
            voxels->releaseJob();
            for (const Chunk* n : nb)
                if (n)
                    n->releaseJob();
            std::lock_guard lock(m_meshDoneMutex);
            m_meshDone.push_back(std::move(result));
        });
    }

    m_scanCarry.insert(m_scanCarry.end(), carryOn.begin(), carryOn.end());
    if (!m_scanCarry.empty())
        m_meshScanDirty = true; // backlog: keep going until the rest get scheduled

    // land finished re-gens: swap the detailed chunk in once no worker is still reading the
    // old (coarse) voxels, then re-seed the column so the new caves/glow light correctly
    // (which re-meshes it; the coarse mesh stays up until that lands).
    {
        std::lock_guard lock(m_regenDoneMutex);
        m_regenDoneScratch.swap(m_regenDone);
    }
    for (const RegenResult& r : m_regenDoneScratch) {
        touchScan(r.pos);
        const auto it = m_chunks.find(r.pos);
        if (it == m_chunks.end() || it->second.pendingDetail != r.voxels) {
            m_pool.release(r.voxels); // entry evicted or superseded: drop this re-gen
            --m_jobsInFlight;
            --m_regenInFlight;
            if (it != m_chunks.end())
                reseedWhenSettled({r.pos.x, 0, r.pos.z});
            continue;
        }
        Entry& e = it->second;
        if (e.voxels->hasJobRefs()) { // a neighbour mesh job still reads the old voxels
            std::lock_guard lock(m_regenDoneMutex);
            m_regenDone.push_back(r); // retry next frame; keep the job slot counted
            continue;
        }
        // The chunk was edited AFTER this regen was submitted (selection only checks dirty
        // at the moment it queues the job) — the edit landed on e.voxels, the coarse chunk,
        // while r.voxels was being built from the pre-edit seed on a worker. Swapping in
        // r.voxels now would discard that edit exactly like the selection guard above is
        // meant to prevent. Drop the stale regen instead: the chunk stays coarse, and once
        // it is saved it will be picked up and regenerated (loading the edit back) normally.
        if (e.dirty) {
            m_pool.release(r.voxels);
            e.pendingDetail = nullptr;
            --m_jobsInFlight;
            --m_regenInFlight;
            reseedWhenSettled({r.pos.x, 0, r.pos.z}); // one fewer regen pending in the column
            continue;
        }
        Chunk* old = e.voxels;
        e.voxels = r.voxels;
        e.pendingDetail = nullptr;
        e.detailed = true;
        m_pool.release(old);
        --m_jobsInFlight;
        --m_regenInFlight;
        // The new voxels carry no light, so this chunk waits unlit (its coarse mesh stays drawn)
        // and its column is re-seeded once the last regen in it has landed. A column entering
        // the detail band regenerates every chunk in it, and re-seeding on each landing relit
        // and re-meshed the whole column once per chunk.
        e.lit = false;
        m_reseedColumns.insert({r.pos.x, 0, r.pos.z});
        reseedWhenSettled({r.pos.x, 0, r.pos.z});
        m_meshScanDirty = true;
    }
    m_regenDoneScratch.clear();

    // drain mesh results; the GL upload is the only part with a time budget
    {
        std::lock_guard lock(m_meshDoneMutex);
        for (auto& r : m_meshDone)
            m_uploads.push_back(std::move(r));
        m_meshDone.clear();
    }
    const double scanMs = msSince(scanStart);
    m_phase.scanMs = std::max(m_phase.scanMs, scanMs);
    m_phaseSum.scanMs += scanMs;

    const auto uploadStart = Clock::now();
    std::size_t uploaded = 0;
    for (auto& r : m_uploads) {
        if (uploaded > 0 && msSince(uploadStart) > kUploadBudgetMs)
            break; // rest carries over to next frame; always at least one per frame
        touchScan(r.pos); // landed, or dropped stale and wanting a mesh at the right LOD
        const auto it = m_chunks.find(r.pos);
        // accept a first mesh (Meshing) or a re-LOD whose result still matches what we
        // asked for (Meshed, meshingLod == r.lod); a stale re-LOD result is dropped and
        // the next scan re-submits at the current desired LOD
        const bool firstMesh = it != m_chunks.end() && it->second.state == ChunkState::Meshing;
        const bool accept =
            it != m_chunks.end() &&
            (firstMesh ||
             (it->second.state == ChunkState::Meshed && it->second.meshingLod == r.lod));
        if (it != m_chunks.end())
            it->second.meshingLod = -1;
        if (accept) {
            Entry& e = it->second;
            if (e.range.count != 0) { // remesh: old ranges go back to the arena
                renderer.free(e.range);
                e.range = {};
            }
            if (e.waterRange.count != 0) {
                renderer.free(e.waterRange);
                e.waterRange = {};
            }
            e.quads = int(r.quadBuf->size());
            e.waterQuads = int(r.waterBuf->size());
            if (!r.quadBuf->empty()) {
                e.range = renderer.alloc(*r.quadBuf);
                if (e.range.count == 0) { // arena full — defrag and retry once
                    // e.quadBuf is still the OLD buffer here, untouched — rebuildArena walks
                    // every Meshed entry including this one, and this entry's e.range was
                    // just freed above, so it needs e.quadBuf to still hold valid data to
                    // re-upload. Only swap in the new buffer once nothing can read the old
                    // one out from under it.
                    rebuildArena(renderer);
                    e.range = renderer.alloc(*r.quadBuf);
                }
                if (e.quadBuf)
                    m_quadPool.checkin(e.quadBuf);
                e.quadBuf = r.quadBuf;
            } else {
                if (e.quadBuf)
                    m_quadPool.checkin(e.quadBuf);
                m_quadPool.checkin(r.quadBuf); // fresh but empty — nothing worth keeping
                e.quadBuf = nullptr;
            }
            if (!r.waterBuf->empty()) {
                e.waterRange = renderer.alloc(*r.waterBuf);
                if (e.waterRange.count == 0) {
                    rebuildArena(renderer);
                    e.waterRange = renderer.alloc(*r.waterBuf);
                }
                if (e.waterBuf)
                    m_waterPool.checkin(e.waterBuf);
                e.waterBuf = r.waterBuf;
            } else {
                if (e.waterBuf)
                    m_waterPool.checkin(e.waterBuf);
                m_waterPool.checkin(r.waterBuf);
                e.waterBuf = nullptr;
            }
            e.connectivity = r.connectivity;
            e.sunlit = r.stats.sunlit;
            e.lod = r.lod;
            e.state = ChunkState::Meshed;
            noteDirty(r.pos);
            m_visDirty = true;
            // A landed mesh is a state change the LOD scan has to see. reLod only fires for
            // chunks in state Meshed, so one that was still Meshing when the camera moved
            // was skipped — and nothing re-flagged the scan afterwards, so it kept its old
            // LOD forever and a coarse chunk never regenerated detailed. busy() reported
            // settled the whole time, which is what made it look like a test fixture bug
            // rather than a lost re-LOD.
            m_meshScanDirty = true;
        } else {
            // Rejected: a stale re-LOD result (dropped so the scan can re-submit at the
            // current LOD), or the chunk was evicted entirely while the job was in flight.
            // Either way nothing will ever read these buffers again — hand them back now,
            // or the pool leaks a slot every time this happens.
            m_quadPool.checkin(r.quadBuf);
            m_waterPool.checkin(r.waterBuf);
        }
        // Light moved while this job was out. The geometry it brought back is still the
        // newest there is, so keep it and send the chunk round once more rather than
        // leaving quads that baked light the world has since moved past. Handled here
        // rather than inside accept so a dropped stale result clears the flag too.
        if (it != m_chunks.end() && it->second.lightDirty) {
            it->second.lightDirty = false;
            if (it->second.state == ChunkState::Meshed)
                it->second.remesh = true;
            m_meshScanDirty = true;
        }
        --m_jobsInFlight;
        ++uploaded;
    }
    m_uploads.erase(m_uploads.begin(), m_uploads.begin() + std::ptrdiff_t(uploaded));
    m_uploadMsLast = msSince(uploadStart);
    m_phase.uploadMs = std::max(m_phase.uploadMs, m_uploadMsLast);
    m_phaseSum.uploadMs += m_uploadMsLast;
}

void ChunkManager::loadWorld(Renderer& renderer, int seed, std::filesystem::path saveDir) {
    // A light slice still running reads the chunks about to go back to the pool. Its results are
    // for the old world and go with it.
    if (m_sliceRunning) {
        m_lightDone.acquire();
        m_sliceRunning = false;
    }
    m_sliceDriven = false;
    // An eviction save still running would write the old world's chunk into whichever world
    // the store points at when it finishes, and keep reading a chunk that is about to go back
    // to the pool and be regenerated into. Let every one land first.
    waitForSaves();
    // Point at the new world BEFORE tearing down, so nothing that runs during teardown can
    // write the old world's chunks into the new world's directory.
    m_regions.flush();
    m_regions.setDir(std::move(saveDir));
    m_seed = seed;
    m_gen = TerrainGenerator(seed);
    // Teardown, minus any discard: switching worlds must never delete the one being left,
    // and the directory we were just pointed at is the one we are about to fill.
    for (auto& item : m_chunks) {
        Entry& e = item.second;
        if (e.range.count != 0)
            renderer.free(e.range);
        if (e.waterRange.count != 0)
            renderer.free(e.waterRange);
        if (e.quadBuf)
            m_quadPool.checkin(e.quadBuf);
        if (e.waterBuf)
            m_waterPool.checkin(e.waterBuf);
        if (e.pendingDetail)
            m_pool.release(e.pendingDetail);
        if (e.voxels)
            m_pool.release(e.voxels);
    }
    m_chunks.clear();
    m_seededColumns.clear();
    m_provisionalColumns.clear();
    m_reseedColumns.clear();
    m_seedQueue.clear();
    m_seedDeferred.clear();
    m_genQueue.clear();
    m_evictList.clear();
    m_uploads.clear();
    m_fallQueue.clear();
    m_fluidQueue.clear();
    m_pendingMesh = 0;
    m_lateSeeds = 0;
    renderer.resetArena(); // deliberate, not a failure: every range was just dropped
    m_center = {INT32_MIN, 0, INT32_MIN};
    m_scanCenter = {INT32_MIN, 0, INT32_MIN}; // the next scan is a full one
    m_scanTouched.clear();
    m_scanCarry.clear();
    m_scanConfirm = false;
    m_lastRadius = -1;
    m_meshScanDirty = true;
    m_dirty = true;
    m_dirtyNearest = 0.0f;
    m_visDirty = true;
}

int ChunkManager::expectedMeshedChunks() const {
    // Meshing covers |dx|,|dz| <= radius (the +1 data ring is loaded but never meshed), and
    // each column's height is whatever columnRange says — which varies with the terrain, so
    // this is summed rather than assumed. m_center is the streaming centre the caller has
    // already settled around.
    int total = 0;
    for (int dz = -radius; dz <= radius; ++dz)
        for (int dx = -radius; dx <= radius; ++dx) {
            int y0 = 0, y1 = 0;
            bool sky = false;
            columnRange(m_center.x + dx, m_center.z + dz, y0, y1, sky);
            total += y1 - y0 + 1;
        }
    return std::max(total, 1);
}

void ChunkManager::collectDrawInfo(std::vector<ChunkDrawInfo>& out,
                                   std::vector<ChunkDrawInfo>& water) {
    out.clear();
    water.clear();
    for (auto& item : m_chunks) {
        Entry& e = item.second;
        if (e.state != ChunkState::Meshed) {
            e.metaIndex = -1;
            continue;
        }
        const glm::ivec3 cp{item.first.x, item.first.y, item.first.z};
        if (e.quads > 0 && e.range.count != 0) {
            e.metaIndex = std::int32_t(out.size());
            out.push_back({cp, e.range.first, e.range.count, e.sunlit ? 1u : 0u});
        } else {
            e.metaIndex = -1;
        }
        if (e.waterQuads > 0 && e.waterRange.count != 0)
            water.push_back({cp, e.waterRange.first, e.waterRange.count, 1u});
    }
    m_metaCount = int(out.size());
    m_visDirty = true;
}

// The flood over a dense grid of the loaded chunks. It used to look every neighbour up in the
// hash map (about six lookups per visible chunk, ~1 ms at radius 24) on every frame the chunk set
// changed, which during streaming is every frame. One pass over the loaded chunks fills the grid;
// the flood then only reads it.
//
// A chunk goes on through every face it was entered by, not just the first. It used to keep only
// the path that reached it first and drop the rest, so whatever lay behind a chunk whose first way
// in had no air path onward was culled even when another way in did, and which path came first was
// only queue order: the same flood walking its directions in reverse disagreed from 27 of 27
// vantage points (--smoke). Now a chunk is drawn exactly when a path that never turns back toward
// the camera reaches it through air. A straight line of sight is such a path, so nothing in plain
// view is culled. Every step lands one step further from the camera, so the queue runs in rings
// and a chunk has been entered by every face it ever will be before it is taken off it, which is
// what makes collecting them first exact.
void ChunkManager::rebuildVisibility(ChunkPos cameraChunk) {
    m_visBits.assign(std::size_t(std::max(1, (m_metaCount + 31) / 32)), 0u);
    const auto markVisible = [&](const Entry& e) {
        if (e.metaIndex >= 0)
            m_visBits[std::size_t(e.metaIndex) >> 5] |= 1u << (e.metaIndex & 31);
    };
    const auto camIt = m_chunks.find(cameraChunk);
    if (camIt == m_chunks.end()) {
        // camera outside the loaded world — draw everything rather than nothing
        std::fill(m_visBits.begin(), m_visBits.end(), 0xFFFFFFFFu);
        return;
    }

    const int span = 2 * radius + 1;
    const std::size_t cells = std::size_t(span) * std::size_t(span) * std::size_t(kWorldChunksY);
    m_floodGrid.assign(cells, nullptr);
    m_floodSeen.assign(cells, 0);
    const int x0 = m_center.x - radius, z0 = m_center.z - radius;
    const auto cellOf = [&](const ChunkPos& p) -> std::size_t {
        const int gx = p.x - x0, gz = p.z - z0;
        if (gx < 0 || gx >= span || gz < 0 || gz >= span || p.y < 0 || p.y >= kWorldChunksY)
            return std::size_t(-1);
        return (std::size_t(p.y) * std::size_t(span) + std::size_t(gx)) * std::size_t(span) + std::size_t(gz);
    };
    for (const auto& item : m_chunks)
        if (const std::size_t c = cellOf(item.first); c != std::size_t(-1))
            m_floodGrid[c] = &item.second;

    m_floodQueue.clear();
    markVisible(camIt->second);
    if (const std::size_t c = cellOf(cameraChunk); c != std::size_t(-1))
        m_floodSeen[c] = kFloodQueued;
    m_floodQueue.push_back(cameraChunk);

    for (std::size_t head = 0; head < m_floodQueue.size(); ++head) {
        const ChunkPos pos = m_floodQueue[head];
        const std::size_t here = cellOf(pos);
        // the camera's own chunk is open every way: it is seen from inside
        const bool fromCamera = pos == cameraChunk;
        const std::uint16_t connectivity =
            fromCamera || here == std::size_t(-1) ? kAllFacesConnected : m_floodGrid[here]->connectivity;
        const std::uint8_t entered = here == std::size_t(-1) ? 0 : m_floodSeen[here] & 0x3Fu;
        for (int d = 0; d < 6; ++d) {
            if (!floodMayStep(pos, cameraChunk, d))
                continue;
            if (!fromCamera && !floodPasses(connectivity, entered, d))
                continue;
            const ChunkPos next{pos.x + kFaceOffsets[d][0], pos.y + kFaceOffsets[d][1],
                                pos.z + kFaceOffsets[d][2]};
            const std::size_t c = cellOf(next);
            if (c == std::size_t(-1) || m_floodGrid[c] == nullptr)
                continue;
            if (!(m_floodSeen[c] & kFloodQueued)) {
                m_floodSeen[c] |= kFloodQueued;
                markVisible(*m_floodGrid[c]);
                m_floodQueue.push_back(next);
            }
            m_floodSeen[c] |= std::uint8_t(1u << (d ^ 1));
        }
    }
}

bool ChunkManager::floodMatchesReference(ChunkPos cameraChunk, double* gridMs, double* hashMs) {
    const auto t0 = Clock::now();
    rebuildVisibility(cameraChunk);
    const double g = msSince(t0);
    const std::vector<std::uint32_t> mine = m_visBits;
    std::vector<std::uint32_t> ref;
    const auto t1 = Clock::now();
    floodByHash(cameraChunk, ref);
    const double h = msSince(t1);
    if (gridMs != nullptr)
        *gridMs += g;
    if (hashMs != nullptr)
        *hashMs += h;
    return mine == ref;
}

void ChunkManager::floodByHash(ChunkPos cameraChunk, std::vector<std::uint32_t>& out) {
    out.assign(std::size_t(std::max(1, (m_metaCount + 31) / 32)), 0u);
    const auto markVisible = [&](const Entry& e) {
        if (e.metaIndex >= 0)
            out[std::size_t(e.metaIndex) >> 5] |= 1u << (e.metaIndex & 31);
    };

    const auto camIt = m_chunks.find(cameraChunk);
    if (camIt == m_chunks.end()) {
        // camera outside the loaded world — draw everything rather than nothing
        std::fill(out.begin(), out.end(), 0xFFFFFFFFu);
        return;
    }

    ankerl::unordered_dense::map<ChunkPos, std::uint8_t, ChunkPosHash> seen;
    m_floodQueue.clear();
    markVisible(camIt->second);
    seen[cameraChunk] = kFloodQueued;
    m_floodQueue.push_back(cameraChunk);

    for (std::size_t head = 0; head < m_floodQueue.size(); ++head) {
        const ChunkPos pos = m_floodQueue[head];
        const bool fromCamera = pos == cameraChunk;
        const std::uint16_t connectivity =
            fromCamera ? kAllFacesConnected : m_chunks.find(pos)->second.connectivity;
        const std::uint8_t entered = seen[pos] & 0x3Fu;
        for (int d = 5; d >= 0; --d) {
            if (!floodMayStep(pos, cameraChunk, d))
                continue;
            if (!fromCamera && !floodPasses(connectivity, entered, d))
                continue;
            const ChunkPos next{pos.x + kFaceOffsets[d][0], pos.y + kFaceOffsets[d][1],
                                pos.z + kFaceOffsets[d][2]};
            if (next.y < 0 || next.y >= kWorldChunksY)
                continue;
            if (std::abs(next.x - m_center.x) > radius || std::abs(next.z - m_center.z) > radius)
                continue;
            const auto it = m_chunks.find(next);
            if (it == m_chunks.end())
                continue;
            std::uint8_t& s = seen[next];
            if (!(s & kFloodQueued)) {
                s |= kFloodQueued;
                markVisible(it->second);
                m_floodQueue.push_back(next);
            }
            s |= std::uint8_t(1u << (d ^ 1));
        }
    }
}

void ChunkManager::updateVisibilityIfNeeded(Renderer& renderer, const glm::dvec3& cameraPos,
                                            bool enabled) {
    if (!enabled) {
        m_lastCaveCull = false;
        return; // shader bypasses the bitset; stale bits are fine
    }
    const ChunkPos cameraChunk{
        int(std::floor(cameraPos.x / kChunkSize)),
        std::clamp(int(std::floor(cameraPos.y / kChunkSize)), 0, kWorldChunksY - 1),
        int(std::floor(cameraPos.z / kChunkSize))};
    if (!m_visDirty && cameraChunk == m_lastFloodCenter && m_lastCaveCull)
        return;
    m_lastFloodCenter = cameraChunk;
    m_lastCaveCull = true;
    m_visDirty = false;
    rebuildVisibility(cameraChunk);
    renderer.updateVisibility(m_visBits);
}

// seedColumn reads the column's 4 horizontal neighbors too (cave-mouth spread), so none of
// them may be mid-generation on a worker — a Generating chunk has a live writer racing
// makeUniform's clear()/push_back() and would read a torn palette. Absent neighbors are fine
// (blockAt treats them as stone); only Generating blocks.
//
// The whole resident band, with room for chunks still waiting to be evicted after a move - not
// 0..kWorldChunksY, a #27 leftover that never looked below y=0 or above the old world top. The
// band reaches seven chunks under the camera, so standing anywhere below y=224 made deep-stone
// chunks resident at negative y, and a seed could read one while a worker was clearing its
// palette to regenerate it.
bool ChunkManager::neighborhoodSettled(ChunkPos col) const {
    static constexpr int kCols[5][2] = {{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (const auto& c : kCols)
        for (int cy = m_center.y - kVerticalRadius - 3; cy <= m_center.y + kVerticalRadius + 3; ++cy) {
            const auto it = m_chunks.find({col.x + c[0], cy, col.z + c[1]});
            if (it != m_chunks.end() && it->second.state == ChunkState::Generating)
                return false;
        }
    return true;
}

// A column only counts as lit once it is FULLY seeded — a half-seeded one would otherwise get
// meshed dark.
void ChunkManager::markColumnLit(ChunkPos col, LightEngine& light) {
    int my0 = 0, my1 = 0;
    bool mSky = false;
    columnRange(col.x, col.z, my0, my1, mSky);
    for (int cy = my0; cy <= my1; ++cy)
        if (const auto cit = m_chunks.find({col.x, cy, col.z}); cit != m_chunks.end()) {
            cit->second.lit = true;
            touchScan(cit->first);
            // A RE-seed changes light in a chunk that is already meshed, and the seeder writes
            // through Chunk::setLight directly (the sky descent and fillLight) rather than
            // LightEngine::setLevel, which is the only thing that marks a chunk touched. So
            // nothing asked for a remesh and the chunk kept its stale baked light until some
            // unrelated edit forced one - that is the lighting that looks frozen until you break
            // a block next to it. Meshes bake light, so a re-seed has to invalidate them.
            // ...and a mesh job already in flight can't be invalidated by a state flip, because
            // its result lands afterwards and puts it straight back.
            if (cit->second.meshingLod >= 0)
                cit->second.lightDirty = true;
            else if (cit->second.state == ChunkState::Meshed)
                cit->second.remesh = true;
        }
    m_meshScanDirty = true; // newly lit -> meshable
    light.checkColumnEdges(col.x, col.z, my0, my1);
    // Sky light spreads sideways out of this column, and an absent neighbour reads as stone — so
    // a column lit before its neighbours exist gets a dark edge that nothing would ever correct.
    // Remember it and redo it when they turn up.
    static constexpr int kSide[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    bool missing = false;
    for (const auto& d : kSide)
        missing = missing || !columnPresent({col.x + d[0], 0, col.z + d[1]});
    if (missing)
        m_provisionalColumns.insert(col);
    else
        m_provisionalColumns.erase(col);
}

// A column whose chunks were not all resident when its turn came was never lit. It must go back
// on the queue AND out of m_seededColumns, or it stays marked seeded forever holding sky light
// 0 - open sky reading as pitch black, at any time of day, which is exactly what it looked like.
//
// Unless it is past the ring rescan() generates (radius + 2): nothing will ever fill its missing
// chunks there, so it failed again every frame, forever, and busy() never cleared (a --tour stop
// ran out its 40000 frames on one such column). It gets queued again the normal way if the
// camera comes back and its chunks are generated.
void ChunkManager::requeueSeed(ChunkPos col) {
    m_seededColumns.erase(col);
    m_provisionalColumns.erase(col);
    if (std::abs(col.x - m_center.x) <= radius + 2 && std::abs(col.z - m_center.z) <= radius + 2)
        m_seedDeferred.push_back(col);
}

// Main thread: decide what the slice seeds. Everything here reads the chunk map, which the
// light thread may only read while the main thread is not changing it.
void ChunkManager::prepareLightSlice(LightEngine& light) {
    LightSlice& s = m_slice;
    s.toSeed.clear();
    s.finished.clear();
    s.aborted.clear();
    s.unstarted.clear();
    s.continueSeed = false;
    // Finish a column left mid-seed by an earlier frame before starting a new one.
    if (light.seedInProgress()) {
        const ChunkPos col{light.seedColumnX(), 0, light.seedColumnZ()};
        if (!neighborhoodSettled(col)) {
            // a neighbour went back to Generating under it (coarse->detailed re-gen): drop the
            // partial work and retry later. re-seeding is idempotent.
            light.abortSeedColumn();
            m_seedDeferred.push_back(col);
        } else {
            s.continueSeed = true;
            s.continued = col;
        }
    }

    // Nearest column first, same reason m_genQueue is sorted the same way: columns finish
    // generating in WORKER COMPLETION order, not distance order, and a coarse (no-cave) far
    // column generates roughly 4x faster than a detailed near one — so under real load, far
    // columns push_back continuously while a near column (quite possibly the one the camera is
    // standing in) is still generating. The queue used to be consumed pure LIFO, which meant
    // every fresh far-column completion cut in line ahead of anything already waiting, and the
    // one column that mattered most could starve: "nothing generates in the area I spawn in".
    // Sorting by distance every frame, right before consuming, guarantees the camera's own
    // column always wins that race regardless of completion order.
    const ChunkPos center = m_center;
    std::sort(m_seedQueue.begin(), m_seedQueue.end(), [center](ChunkPos a, ChunkPos b) {
        return horizontalDist2(a, center) > horizontalDist2(b, center);
    });
    // Bounded by count and by time even when nothing is ready - right after a teleport every
    // column's neighbourhood is still mid-generation, and checking them all is real work.
    const auto t0 = Clock::now();
    const std::size_t queued = m_seedQueue.size();
    for (std::size_t examined = 0; examined < queued && s.toSeed.size() < kSliceColumns; ++examined) {
        if (msSince(t0) > kSlicePrepareMs)
            break;
        const ChunkPos col = m_seedQueue.back();
        m_seedQueue.pop_back();
        // Wait for every chunk of the band to be there before starting: a column queued whole
        // loses that the moment the camera changes chunk height and its band grows a chunk still
        // to be generated, and starting it anyway only aborted and queued it again, every frame
        // - 87,655 aborted starts against 3,504 finished seeds in one ten-chunk walk at radius 32.
        if (!neighborhoodSettled(col) || !columnPresent(col)) {
            // Past the ring rescan() generates (radius + 2) its missing chunks will never come,
            // so waiting for them kept it here, and busy() true, for good: a --tour run sat out
            // the 40000-frame cap at four stops in a row on one such column, and the in-game
            // world switch drains on busy(). Dropped like requeueSeed drops one; the gen-done
            // drain queues it again if the camera comes back and it completes.
            if (std::abs(col.x - m_center.x) > radius + 2 || std::abs(col.z - m_center.z) > radius + 2) {
                m_seededColumns.erase(col);
                m_provisionalColumns.erase(col);
                continue;
            }
            m_seedDeferred.push_back(col); // retry once the column and neighbours settle
            continue;
        }
        s.toSeed.push_back(col);
    }
}

// The work itself: on the light thread in the game, inline otherwise. Touches only the light
// engine (and through it chunk light, which readers take as atomics) and m_slice.
void ChunkManager::runLightSlice(LightEngine& light) {
    LightSlice& s = m_slice;
    const auto t0 = Clock::now();
    const auto remaining = [&] { return std::max(0.1, kSeedBudgetMs - msSince(t0)); };
    if (s.continueSeed && light.stepSeedColumn(remaining()))
        (light.seedAborted() ? s.aborted : s.finished).push_back(s.continued);
    std::size_t i = 0;
    for (; i < s.toSeed.size() && !light.seedInProgress(); ++i) {
        if (msSince(t0) > kSeedBudgetMs)
            break;
        const ChunkPos col = s.toSeed[i];
        light.beginSeedColumn(col.x, col.z);
        if (light.stepSeedColumn(remaining()))
            (light.seedAborted() ? s.aborted : s.finished).push_back(col);
    }
    s.unstarted.assign(s.toSeed.begin() + std::ptrdiff_t(i), s.toSeed.end());
    s.seedMs = msSince(t0);
    // one shared deadline for the whole light phase: if seeding ran long, propagation gives way
    // rather than stacking a second overshoot on top of the first
    const auto t1 = Clock::now();
    light.process(std::max(0.25, kLightBudgetMs - msSince(t0)));
    s.propagateMs = msSince(t1);
}

// Main thread again: take in what the slice did.
void ChunkManager::applyLightSlice(LightEngine& light) {
    LightSlice& s = m_slice;
    for (const ChunkPos col : s.finished)
        markColumnLit(col, light);
    for (const ChunkPos col : s.aborted)
        requeueSeed(col);
    // Deferred and unstarted columns go back in at their normal distance priority (the next
    // sort places them) rather than a fixed slot.
    m_seedQueue.insert(m_seedQueue.end(), s.unstarted.begin(), s.unstarted.end());
    m_seedQueue.insert(m_seedQueue.end(), m_seedDeferred.begin(), m_seedDeferred.end());
    m_seedDeferred.clear();
    m_seedActive = light.seedInProgress(); // a part-seeded column still counts as busy()
    m_phase.seedMs = std::max(m_phase.seedMs, s.seedMs);
    m_phaseSum.seedMs += s.seedMs;
    m_phase.propagateMs = std::max(m_phase.propagateMs, s.propagateMs);
    m_phaseSum.propagateMs += s.propagateMs;

    for (const ChunkPos pos : light.touched()) {
        const auto it = m_chunks.find(pos);
        if (it == m_chunks.end())
            continue;
        touchScan(pos);
        Entry& e = it->second;
        // A mesh job reading this chunk right now sampled the light as it was when it started,
        // and its result lands after this loop — so flipping the state here would just be
        // overwritten and the change lost until an unrelated edit forced a remesh. That race
        // is why the same seed settled to a different quad count run to run.
        if (e.meshingLod >= 0)
            e.lightDirty = true;
        else if (e.state == ChunkState::Meshed) {
            e.remesh = true; // back through the mesh pipeline, still drawn meanwhile
            m_meshScanDirty = true;
        }
    }
    light.clearTouched();
}

void ChunkManager::startLightSlice(LightEngine& light) {
    if (m_sliceRunning)
        return;
    if (!m_lightThread.joinable())
        m_lightThread = std::jthread([this](std::stop_token stop) {
            for (;;) {
                m_lightGo.acquire();
                if (stop.stop_requested())
                    return;
                runLightSlice(*m_sliceLight);
                m_lightDone.release();
            }
        });
    prepareLightSlice(light);
    m_sliceLight = &light;
    m_sliceRunning = true;
    m_lightGo.release();
}

void ChunkManager::finishLightSlice(LightEngine& light) {
    if (!m_sliceRunning)
        return;
    const auto t0 = Clock::now();
    m_lightDone.acquire();
    const double waitMs = msSince(t0);
    m_phase.lightWaitMs = std::max(m_phase.lightWaitMs, waitMs);
    m_phaseSum.lightWaitMs += waitMs;
    m_sliceRunning = false;
    applyLightSlice(light);
    m_sliceDriven = true;
}

ChunkManager::~ChunkManager() {
    if (m_sliceRunning)
        m_lightDone.acquire();
    if (m_lightThread.joinable()) {
        m_lightThread.request_stop();
        m_lightGo.release();
        m_lightThread.join();
    }
}

void ChunkManager::reseedWhenSettled(ChunkPos col) {
    if (!m_reseedColumns.contains(col))
        return;
    for (int cy = m_center.y - kVerticalRadius - 3; cy <= m_center.y + kVerticalRadius + 3; ++cy)
        if (const auto it = m_chunks.find({col.x, cy, col.z}); it != m_chunks.end() && it->second.pendingDetail)
            return; // another regen in this column is still out
    m_reseedColumns.erase(col);
    // Lit stays false over the band until the seed runs (meshing trusts lit), and the column
    // stays in m_seededColumns as "scheduled", the same as a column the gen-done drain queues:
    // dropping it there let the self-heal queue the same column again behind this seed.
    int y0 = 0, y1 = 0;
    bool sky = false;
    columnRange(col.x, col.z, y0, y1, sky);
    for (int cy = y0; cy <= y1; ++cy)
        if (const auto it = m_chunks.find({col.x, cy, col.z}); it != m_chunks.end())
            it->second.lit = false;
    m_seededColumns.insert(col);
    m_provisionalColumns.erase(col);
    m_seedQueue.push_back(col);
}

void ChunkManager::waitForSaves() const {
    // they are a serialize and a map insert each, so this is microseconds unless a test is
    // holding them open
    while (m_savesInFlight.load(std::memory_order_acquire) > 0)
        std::this_thread::yield();
}

void ChunkManager::saveDirty() {
    // an edited chunk evicted a moment ago is still being saved on a worker; flushing
    // without it would leave that edit out of the files
    waitForSaves();
    int saved = 0;
    for (auto& [pos, e] : m_chunks)
        if (e.dirty && e.voxels) {
            m_saveScratch.clear();
            ChunkSerializer::serialize(*e.voxels, m_saveScratch);
            m_regions.save(pos, m_saveScratch);
            e.dirty = false;
            touchScan(pos); // unsaved edits held it coarse; it may regenerate now
            ++saved;
        }
    m_regions.flush();
    if (saved > 0)
        VX_INFO("persistence: saved %d edited chunk(s)", saved);
}

BlockId ChunkManager::blockAtWorld(const glm::ivec3& wp) const {
    if (wp.y < 0)
        return blocks::Stone;
    if (wp.y >= kWorldHeight)
        return blocks::Air;
    const Chunk* c = voxelsAt({wp.x >> 5, wp.y >> 5, wp.z >> 5});
    return c ? c->get(wp.x & 31, wp.y & 31, wp.z & 31) : blocks::Stone;
}

int ChunkManager::skyLightAtWorld(const glm::ivec3& wp) const {
    const Chunk* c = voxelsAt({wp.x >> 5, wp.y >> 5, wp.z >> 5});
    return c ? light::sky(c->lightAt(wp.x & 31, wp.y & 31, wp.z & 31)) : 0;
}

void ChunkManager::remeshNow(Renderer& renderer, ChunkPos pos) {
    const auto it = m_chunks.find(pos);
    if (it == m_chunks.end() || it->second.state == ChunkState::Generating)
        return;
    Entry& e = it->second;

    const Chunk* neighbors[6] = {
        voxelsAt({pos.x + 1, pos.y, pos.z}), voxelsAt({pos.x - 1, pos.y, pos.z}),
        voxelsAt({pos.x, pos.y + 1, pos.z}), voxelsAt({pos.x, pos.y - 1, pos.z}),
        voxelsAt({pos.x, pos.y, pos.z + 1}), voxelsAt({pos.x, pos.y, pos.z - 1}),
    };
    // Synchronous (this runs on an edit, right on the main thread — no worker involved), but
    // still pool-backed rather than a thread_local scratch: this entry's OLD buffers need to
    // stay valid through a possible rebuildArena() retry below, same as the async path in
    // update(), so the same checkout-fresh/defer-checkin-old pattern applies even without a
    // thread hop.
    std::vector<PackedQuad>* quadBuf = m_quadPool.checkout();
    std::vector<PackedQuad>* waterBuf = m_waterPool.checkout();
    // edits are always on full-detail chunks, but one can still border a coarser ring
    const GreedyStats stats = m_mesher.mesh(*e.voxels, neighbors, *quadBuf, *waterBuf, 0,
                                            coarseNeighborMask(pos, 0, m_center, lodScale()));
    e.connectivity = computeConnectivity(*e.voxels);
    e.sunlit = stats.sunlit; // an edit can open a sealed chunk to the sky, or seal it

    if (e.range.count != 0) {
        renderer.free(e.range);
        e.range = {};
    }
    if (e.waterRange.count != 0) {
        renderer.free(e.waterRange);
        e.waterRange = {};
    }
    e.quads = stats.quads;
    e.waterQuads = stats.waterQuads;
    if (stats.quads > 0) {
        e.range = renderer.alloc(*quadBuf);
        if (e.range.count == 0) {
            rebuildArena(renderer); // e.quadBuf is still the OLD buffer here — still valid
            e.range = renderer.alloc(*quadBuf);
        }
        if (e.quadBuf)
            m_quadPool.checkin(e.quadBuf);
        e.quadBuf = quadBuf;
    } else {
        if (e.quadBuf)
            m_quadPool.checkin(e.quadBuf);
        m_quadPool.checkin(quadBuf); // fresh but empty — nothing worth keeping
        e.quadBuf = nullptr;
    }
    if (stats.waterQuads > 0) {
        e.waterRange = renderer.alloc(*waterBuf);
        if (e.waterRange.count == 0) {
            rebuildArena(renderer);
            e.waterRange = renderer.alloc(*waterBuf);
        }
        if (e.waterBuf)
            m_waterPool.checkin(e.waterBuf);
        e.waterBuf = waterBuf;
    } else {
        if (e.waterBuf)
            m_waterPool.checkin(e.waterBuf);
        m_waterPool.checkin(waterBuf);
        e.waterBuf = nullptr;
    }
    e.lod = 0;         // edits are always on near (full-detail) chunks
    e.meshingLod = -1; // drop any in-flight async re-LOD result for this chunk
    e.remesh = false;  // just meshed against current light
    e.state = ChunkState::Meshed; // a stale worker result for this chunk now gets dropped
    noteDirty(pos);
    m_visDirty = true;
}

bool ChunkManager::setBlock(Renderer& renderer, LightEngine& light, const glm::ivec3& wp,
                            BlockId id) {
    const BlockEdit one{wp, id};
    return setBlocks(renderer, light, std::span<const BlockEdit>(&one, 1)) > 0;
}

int ChunkManager::setBlocks(Renderer& renderer, LightEngine& light,
                            std::span<const BlockEdit> edits) {
    m_remeshScratch.clear();
    m_remeshSeen.clear();
    const auto want = [&](ChunkPos p) {
        if (m_remeshSeen.insert(p).second)
            m_remeshScratch.push_back(p);
    };

    // Pass 1: apply every voxel write and tell the light engine about each one. Nothing
    // is meshed yet — a chunk hit by several edits in this batch must not be rebuilt once
    // per edit, and light must not settle against a half-applied batch. The streaming backlog
    // is parked meanwhile, so what Pass 2 settles is this batch's light and nothing else.
    light.beginEdit();
    int applied = 0;
    for (const BlockEdit& e : edits) {
        if (e.pos.y < 0 || e.pos.y >= kWorldHeight)
            continue;
        const ChunkPos pos{e.pos.x >> 5, e.pos.y >> 5, e.pos.z >> 5};
        const auto it = m_chunks.find(pos);
        if (it == m_chunks.end() || it->second.state == ChunkState::Generating)
            continue;
        Chunk* c = it->second.voxels;
        const int lx = e.pos.x & 31;
        const int ly = e.pos.y & 31;
        const int lz = e.pos.z & 31;
        const BlockId old = c->get(lx, ly, lz);
        if (old == e.id)
            continue;

        // palette writes are not safe under a concurrent worker read; in-flight mesh jobs
        // finish in well under a millisecond WHEN THE JOB SYSTEM IS IDLE, so a bounded spin
        // beats copy-on-write in the common case. Owner report, live, reproduced by
        // reasoning through this exact loop rather than by testing it: a hot spin with no
        // yield can starve the very worker thread it is waiting on once the job system is
        // saturated - right after launch, while the initial world-settle backlog (still
        // draining for the ~30s the owner measured) keeps every worker busy, and
        // hasJobRefs() can be held not just by this chunk's own mesh job but by any
        // NEIGHBOUR's job still reading across a shared border (see the identical check at
        // processEvictions above). A spin that never yields does not just wait for that
        // ref to clear - on an oversubscribed machine it can actively delay the worker
        // that would clear it, turning an assumed sub-millisecond wait into the owner's
        // reported multi-second freeze. It also explains the OTHER half of the report
        // ("random chunks disappear until updated"): nothing else runs on the main thread
        // while this spins - no upload drain, no eviction, no visibility update - so
        // whatever the world was mid-transition on when the freeze started stays visibly
        // wrong until the spin finally releases it. Yielding fixes the CPU-starvation half
        // directly; the elapsed-time log below is new visibility into the other half (how
        // saturated the job system actually was), so this cannot silently regress again
        // unnoticed the way it evidently already did once.
        const auto waitStart = Clock::now();
        while (c->hasJobRefs())
            std::this_thread::yield();
        if (const double waitMs =
                std::chrono::duration<double, std::milli>(Clock::now() - waitStart).count();
            waitMs > 5.0)
            VX_ERROR("setBlocks: waited %.1f ms for an in-flight job on an edited chunk "
                     "(want <1 ms) - the job system was saturated",
                     waitMs);
        c->set(lx, ly, lz, e.id);
        it->second.dirty = true; // content changed: this chunk now needs persisting
        // if a save job already ran for this chunk it captured the pre-edit voxels, so
        // clear the guard or the new content would never be written
        it->second.saving = false;
        light.onBlockChanged(e.pos, old, e.id);
        ++applied;

        // Anything disturbed here is worth re-testing for gravity: this voxel (it may now
        // BE a falling block with nothing under it) and the one above it (whatever was
        // resting on what just changed). This is the only thing that ever starts a fall,
        // which is what keeps untouched terrain still.
        queueFallCheck(e.pos);
        queueFallCheck({e.pos.x, e.pos.y + 1, e.pos.z});
        queueFluidCheck(e.pos);

        // this chunk, plus any neighbour whose mesh samples across the touched border
        want(pos);
        if (lx == 0) want({pos.x - 1, pos.y, pos.z});
        if (lx == 31) want({pos.x + 1, pos.y, pos.z});
        if (ly == 0) want({pos.x, pos.y - 1, pos.z});
        if (ly == 31) want({pos.x, pos.y + 1, pos.z});
        if (lz == 0) want({pos.x, pos.y, pos.z - 1});
        if (lz == 31) want({pos.x, pos.y, pos.z + 1});
    }
    if (applied == 0) {
        light.endEdit();
        light.clearTouched();
        return 0;
    }

    // Pass 2: settle the whole batch's light in one BFS. Removals and re-propagation from
    // different edits overlap heavily, so settling once is strictly less work than
    // settling per edit — and it means no chunk is ever meshed against half-updated light.
    while (!light.idle())
        light.process(4.0);
    light.endEdit();
    for (const ChunkPos p : light.touched())
        want(p);
    light.clearTouched();

    // Pass 3: one remesh per affected chunk, however many edits landed in it.
    for (const ChunkPos p : m_remeshScratch)
        remeshNow(renderer, p);
    return applied;
}

void ChunkManager::queueFallCheck(const glm::ivec3& p) {
    if (p.y < 1 || p.y >= kWorldHeight)
        return;
    if (m_fallQueue.size() < 200000) // a runaway cascade must not eat memory
        m_fallQueue.push_back(p);
}

int ChunkManager::stepFalling(Renderer& renderer, LightEngine& light, int* landed,
                              glm::ivec3* landedAt) {
    if (landed != nullptr)
        *landed = 0;
    if (m_fallQueue.empty()) {
        m_fellLast.clear();
        return 0;
    }
    // Swap out first: setBlocks below feeds new candidates back into m_fallQueue, and that
    // is exactly how a column keeps coming down over successive ticks.
    m_fallScratch.clear();
    m_fallScratch.swap(m_fallQueue);

    m_fluidEdits.clear();
    m_fellNow.clear();
    for (const glm::ivec3& p : m_fallScratch) {
        const BlockId here = blockAtWorld(p);
        // Cross-plane foliage cannot stand on nothing. Break the dirt under a flower and the
        // flower has to go with it, or it is left hovering — which is the one thing that
        // makes decoration read as pasted on rather than grown.
        //
        // This rides the falling queue rather than adding a scan of its own because
        // setBlocks already queues the changed voxel AND the one above it, and the one above
        // a broken block is exactly where the foliage is. Nothing new has to be searched for.
        // Foliage does not FALL, it is removed: a tuft that slid down a cliff one block per
        // tick would be worse than one that vanishes.
        if (blocks::isCross(here)) {
            const BlockId under = blockAtWorld({p.x, p.y - 1, p.z});
            // Stands on anything solid; kelp may also stand on more kelp. Waterlogged plants
            // are replaced by water, not air, or a popped stalk would leave a hole in the sea.
            const bool wet = blocks::isWaterlogged(here);
            if (!blocks::opaque(under) && !(wet && blocks::isWaterlogged(under)))
                m_fluidEdits.push_back({p, wet ? blocks::Water : blocks::Air});
            continue;
        }
        if (!blocks::falls(here))
            continue;
        const BlockId below = blockAtWorld({p.x, p.y - 1, p.z});
        // falls through air and through water — sinking is what you want when you drop
        // gravel into a lake, and water is not a support
        if (below != blocks::Air && !blocks::isWater(below)) {
            // Come to rest. Only counts if it actually moved last tick; a block placed onto
            // solid ground reaches here too, and it already made a placement sound.
            if (landed != nullptr &&
                std::find(m_fellLast.begin(), m_fellLast.end(), p) != m_fellLast.end()) {
                if (*landed == 0 && landedAt != nullptr)
                    *landedAt = p;
                ++*landed;
            }
            continue;
        }
        m_fluidEdits.push_back({p, blocks::Air});
        m_fluidEdits.push_back({{p.x, p.y - 1, p.z}, here});
        m_fellNow.push_back({p.x, p.y - 1, p.z});
    }
    m_fellLast.swap(m_fellNow);

    if (m_fluidEdits.empty())
        return 0;
    return setBlocks(renderer, light, m_fluidEdits);
}

// Depth lost per cell of horizontal travel. At 1 a source reached seven cells in every
// direction — a puddle the size of a small room from a single block, which is what made
// placed water sprawl across the landscape. At 2 it thins out in four, which reads as a
// spill rather than a flood, and it halves the cells a tick has to touch.
static constexpr int kFluidDecay = 2;

void ChunkManager::queueFluidCheck(const glm::ivec3& p) {
    static constexpr int kNeighbours[7][3] = {{0, 0, 0},  {1, 0, 0}, {-1, 0, 0}, {0, 1, 0},
                                              {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    if (m_fluidQueue.size() > 400000)
        return; // a runaway cascade must not eat memory
    for (const auto& d : kNeighbours)
        m_fluidQueue.push_back({p.x + d[0], p.y + d[1], p.z + d[2]});
}

int ChunkManager::stepFluid(Renderer& renderer, LightEngine& light,
                            const glm::dvec3& around) {
    if (m_fluidQueue.empty())
        return 0;
    // Only cells next to an edit can change, so only those are looked at. This used to scan
    // an 81x81x41 box around the camera every tick (~270k cells, several hash lookups each)
    // to find the nothing that had happened - a hitch six times a second, forever.
    constexpr int kFarBlocks = 128;
    constexpr std::size_t kPerTick = 60000;
    const glm::ivec3 c{int(std::floor(around.x)), int(std::floor(around.y)),
                       int(std::floor(around.z))};
    m_fluidScratch.clear();
    m_fluidScratch.swap(m_fluidQueue);
    std::sort(m_fluidScratch.begin(), m_fluidScratch.end(),
              [](const glm::ivec3& a, const glm::ivec3& b) {
                  return a.y != b.y ? a.y > b.y : (a.z != b.z ? a.z < b.z : a.x < b.x);
              });
    m_fluidScratch.erase(std::unique(m_fluidScratch.begin(), m_fluidScratch.end()),
                         m_fluidScratch.end());

    m_fluidEdits.clear();
    std::size_t looked = 0;
    for (const glm::ivec3& p : m_fluidScratch) {
        if (looked >= kPerTick || std::abs(p.x - c.x) > kFarBlocks ||
            std::abs(p.z - c.z) > kFarBlocks) {
            m_fluidQueue.push_back(p); // not this tick: far from the camera, or over budget
            continue;
        }
        ++looked;
        const int x = p.x, y = p.y, z = p.z;
        const BlockId here = blockAtWorld(p);
        const int level = blocks::waterLevel(here);
        if (level == 8)
            continue; // source: never decays, never recomputed
        if (here != blocks::Air && level == 0)
            continue; // solid

        // Fed from directly above at near-full depth (a falling column stays a column),
        // otherwise from the deepest horizontal neighbour minus one.
        int want = 0;
        if (blocks::isWater(blockAtWorld({x, y + 1, z}))) {
            want = 7;
        } else {
            static constexpr int kSide[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (const auto& d : kSide)
                want = std::max(want, blocks::waterLevel(blockAtWorld(
                                          {x + d[0], y, z + d[1]})) - kFluidDecay);
            // water only spreads sideways once it has something to sit on
            if (want > 0 && !blocks::opaque(blockAtWorld({x, y - 1, z})) &&
                !blocks::isWater(blockAtWorld({x, y - 1, z})))
                want = std::min(want, 7);
        }
        if (want == level)
            continue;
        const BlockId next = blocks::waterForLevel(want);
        if (next != here)
            m_fluidEdits.push_back({p, next});
    }

    if (m_fluidEdits.empty())
        return 0;
    // setBlocks queues every voxel it changes (and its neighbours), which is what carries a
    // spreading sheet outward one ring per tick until nothing changes any more
    return setBlocks(renderer, light, m_fluidEdits);
}

ChunkManager::Stats ChunkManager::stats() const {
    Stats s;
    s.loaded = int(m_chunks.size());
    s.genQueue = int(m_genQueue.size());
    s.jobsInFlight = m_jobsInFlight;
    s.pendingUploads = int(m_uploads.size());
    s.lateSeeds = m_lateSeeds;
    s.uploadMsLast = m_uploadMsLast;
    const int gc = m_genCount.load(std::memory_order_relaxed);
    const int mc = m_meshCount.load(std::memory_order_relaxed);
    s.genMsAvg = gc > 0 ? double(m_genNanos.load(std::memory_order_relaxed)) * 1e-6 / gc : 0.0;
    s.meshMsAvg = mc > 0 ? double(m_meshNanos.load(std::memory_order_relaxed)) * 1e-6 / mc : 0.0;

    // The full-chunk scan (meshed/uniform/byte counts) is O(loaded) — at radius 16 that
    // is ~5000 chunks. It only feeds HUD numbers that don't need per-frame precision, so
    // recompute it a few times a second and cache the rest.
    if (m_scanAge++ >= 20) {
        m_scanAge = 0;
        Scan sc;
        for (const auto& item : m_chunks) {
            const Entry& e = item.second;
            if (e.state == ChunkState::Meshed)
                ++sc.meshed;
            if (!e.lit)
                ++sc.unlit;
            if (e.voxels->uniform())
                ++sc.uniformChunks;
            sc.totalQuads += e.quads;
            sc.voxelBytes += e.voxels->storageBytes();
            sc.quadBytes += std::size_t(e.quads) * sizeof(PackedQuad);
        }
        m_scanCache = sc;
    }
    s.meshed = m_scanCache.meshed;
    s.unlit = m_scanCache.unlit;
    s.uniformChunks = m_scanCache.uniformChunks;
    s.totalQuads = m_scanCache.totalQuads;
    s.voxelBytes = m_scanCache.voxelBytes;
    s.quadBytes = m_scanCache.quadBytes;
    return s;
}

void ChunkManager::logSummary() const {
    refreshStats();
    const Stats s = stats();
    VX_INFO("world: %d chunks (%d uniform), %d meshed, %lld quads", s.loaded,
            s.uniformChunks, s.meshed, static_cast<long long>(s.totalQuads));
    VX_INFO("world: %.1f MB voxels, %.1f MB quads | gen avg %.3f ms, mesh avg %.3f ms",
            double(s.voxelBytes) / (1024.0 * 1024.0), double(s.quadBytes) / (1024.0 * 1024.0),
            s.genMsAvg, s.meshMsAvg);
}

} // namespace vx
