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
        // Persist player edits, and also cache DETAILED generated chunks so coming back
        // reloads them instead of paying full generation again. Coarse chunks are left to
        // regenerate: they are the cheap half (no caves, which are ~4x the cost) and
        // caching them would spend main-thread serialize time to save very little.
        // Serialize on a worker, never here. A teleport evicts thousands of chunks at
        // once and doing this inline measured as a 128 ms freeze — and blocking the main
        // thread on disk work is exactly what the streaming design forbids. The job holds
        // a ref, so the chunk cannot return to the pool underneath it; this entry simply
        // waits out another pass or two of the evict list, which it is already on.
        if (e.dirty && !e.saving) {
            e.saving = true;
            Chunk* voxels = e.voxels;
            const ChunkPos at = m_evictList[i];
            voxels->retainJob();
            jobs.submit([this, voxels, at] {
                std::vector<std::uint8_t> bytes;
                ChunkSerializer::serialize(*voxels, bytes);
                m_regions.save(at, std::move(bytes));
                voxels->releaseJob();
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
        m_seededColumns.erase(ChunkPos{m_evictList[i].x, 0, m_evictList[i].z});
        m_provisionalColumns.erase(ChunkPos{m_evictList[i].x, 0, m_evictList[i].z});
        m_chunks.erase(it);
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
            m_meshScanDirty = true;
        }
    }
    m_genDoneScratch.clear();

    // budgeted light seeding + propagation, then remesh whatever light touched.
    // seeding costs ~0.5-1 ms per column, so keep the per-frame count low.
    //
    // seedColumn reads the column's 4 horizontal neighbors too (cave-mouth spread), so
    // none of them may be mid-generation on a worker — a Generating chunk has a live
    // writer racing makeUniform's clear()/push_back() and would read a torn palette.
    // absent neighbors are fine (blockAt treats them as stone); only Generating blocks.
    const auto neighborhoodSettled = [this](ChunkPos col) {
        static constexpr int kCols[5][2] = {{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (const auto& c : kCols)
            for (int cy = 0; cy < kWorldChunksY; ++cy) {
                const auto it = m_chunks.find({col.x + c[0], cy, col.z + c[1]});
                if (it != m_chunks.end() && it->second.state == ChunkState::Generating)
                    return false;
            }
        return true;
    };
    // seed as many columns as fit a small time budget (always at least one). a fixed
    // count/frame was far too slow once the default radius grew — columns backed up and
    // their chunks sat dark/unmeshed for seconds.
    const auto seedStart = Clock::now();
    {
        // A column only counts as lit once it is FULLY seeded — a half-seeded one would
        // otherwise get meshed dark.
        const auto markLit = [&](ChunkPos col) {
            int my0 = 0, my1 = 0;
            bool mSky = false;
            columnRange(col.x, col.z, my0, my1, mSky);
            for (int cy = my0; cy <= my1; ++cy)
                if (const auto cit = m_chunks.find({col.x, cy, col.z}); cit != m_chunks.end()) {
                    cit->second.lit = true;
                    // A RE-seed changes light in a chunk that is already meshed, and the
                    // seeder writes through Chunk::setLight directly (the sky descent and
                    // fillLight) rather than LightEngine::setLevel, which is the only thing
                    // that marks a chunk touched. So nothing asked for a remesh and the
                    // chunk kept its stale baked light until some unrelated edit forced
                    // one - that is the lighting that looks frozen until you break a block
                    // next to it. Meshes bake light, so a re-seed has to invalidate them.
                    // ...and a mesh job already in flight can't be invalidated by a state
                    // flip, because its result lands afterwards and puts it straight back.
                    if (cit->second.meshingLod >= 0)
                        cit->second.lightDirty = true;
                    else if (cit->second.state == ChunkState::Meshed)
                        cit->second.remesh = true;
                }
            m_meshScanDirty = true; // newly lit -> meshable
            light.checkColumnEdges(col.x, col.z, my0, my1);
            // Sky light spreads sideways out of this column, and an absent neighbour reads
            // as stone — so a column lit before its neighbours exist gets a dark edge that
            // nothing would ever correct. Remember it and redo it when they turn up.
            static constexpr int kSide[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            bool missing = false;
            for (const auto& d : kSide)
                missing = missing || !columnPresent({col.x + d[0], 0, col.z + d[1]});
            if (missing)
                m_provisionalColumns.insert(col);
            else
                m_provisionalColumns.erase(col);
        };
        // A column whose chunks were not all resident when its turn came was never lit.
        // It must go back on the queue AND out of m_seededColumns, or it stays marked
        // seeded forever holding sky light 0 - open sky reading as pitch black, at any
        // time of day, which is exactly what it looked like.
        //
        // Unless it is past the ring rescan() generates (radius + 2): nothing will ever fill
        // its missing chunks there, so it failed again every frame, forever, and busy() never
        // cleared (a --tour stop ran out its 40000 frames on one such column). It gets queued
        // again the normal way if the camera comes back and its chunks are generated.
        const auto requeueSeed = [&](ChunkPos col) {
            m_seededColumns.erase(col);
            m_provisionalColumns.erase(col);
            if (std::abs(col.x - center.x) <= radius + 2 && std::abs(col.z - center.z) <= radius + 2)
                m_seedDeferred.push_back(col);
        };
        const auto remaining = [&] {
            return std::max(0.1, kSeedBudgetMs - msSince(seedStart));
        };

        // Finish a column left mid-seed by an earlier frame before starting a new one.
        if (light.seedInProgress()) {
            const ChunkPos col{light.seedColumnX(), 0, light.seedColumnZ()};
            if (!neighborhoodSettled(col)) {
                // a neighbour went back to Generating under it (coarse->detailed re-gen):
                // drop the partial work and retry later. re-seeding is idempotent.
                light.abortSeedColumn();
                m_seedDeferred.push_back(col);
            } else if (light.stepSeedColumn(remaining())) {
                if (light.seedAborted())
                    requeueSeed(col);
                else
                    markLit(col);
            }
        }

        // Nearest column first, same reason m_genQueue is sorted the same way: columns
        // finish generating in WORKER COMPLETION order, not distance order, and a coarse
        // (no-cave) far column generates roughly 4x faster than a detailed near one — so
        // under real load, far columns push_back continuously while a near column (quite
        // possibly the one the camera is standing in) is still generating. The queue used
        // to be consumed pure LIFO (pop_back, whatever pushed most recently), which meant
        // every fresh far-column completion cut in line ahead of anything already waiting —
        // an old entry can be starved indefinitely for as long as new ones keep arriving
        // faster than the 2ms/frame budget drains them. That is "nothing generates in the
        // area I spawn in and I have to fly to the nearest chunk": voxels exist (gen queue
        // is fine), but light-seeding — which meshing waits on — never reached the one
        // column that mattered most, while whichever far columns won the completion race
        // got lit and meshed instead. Sorting by distance every frame, right before
        // consuming, guarantees the camera's own column always wins that race regardless of
        // completion order — same fix shape as m_genQueue, just applied per-frame since this
        // queue receives new entries continuously instead of all at once from rescan().
        std::sort(m_seedQueue.begin(), m_seedQueue.end(), [center](ChunkPos a, ChunkPos b) {
            return horizontalDist2(a, center) > horizontalDist2(b, center);
        });
        // The budget has to apply even when nothing seeds — right after a teleport every
        // column's neighbourhood is still mid-generation, and the old "only check the
        // clock once something seeded" walked the whole queue every frame for no work.
        const std::size_t queued = m_seedQueue.size();
        for (std::size_t examined = 0; examined < queued && !light.seedInProgress();
             ++examined) {
            if (msSince(seedStart) > kSeedBudgetMs)
                break;
            const ChunkPos col = m_seedQueue.back();
            m_seedQueue.pop_back();
            if (!neighborhoodSettled(col)) {
                m_seedDeferred.push_back(col); // retry once neighbours settle
                continue;
            }
            light.beginSeedColumn(col.x, col.z);
            if (light.stepSeedColumn(remaining())) {
                if (light.seedAborted())
                    requeueSeed(col);
                else
                    markLit(col);
            }
        }
        // Deferred columns go back in at their normal distance priority (next frame's sort
        // places them correctly) rather than a fixed front/back slot — a near column that
        // was momentarily deferred still deserves to be seeded before a far one once ready.
        if (!m_seedDeferred.empty()) {
            m_seedQueue.insert(m_seedQueue.end(), m_seedDeferred.begin(), m_seedDeferred.end());
            m_seedDeferred.clear();
        }
    }
    m_seedActive = light.seedInProgress(); // a part-seeded column still counts as busy()
    const double seedMs = msSince(seedStart);
    m_phase.seedMs = std::max(m_phase.seedMs, seedMs);
    m_phaseSum.seedMs += seedMs;
    // one shared deadline for the whole light phase: if seeding ran long, propagation
    // gives way rather than stacking a second overshoot on top of the first
    const auto propStart = Clock::now();
    light.process(std::max(0.25, kLightBudgetMs - msSince(seedStart)));
    for (const ChunkPos pos : light.touched()) {
        const auto it = m_chunks.find(pos);
        if (it == m_chunks.end())
            continue;
        Entry& e = it->second;
        // A mesh job reading this chunk right now sampled the light as it was when it
        // started, and its result lands after this loop — so flipping the state here would
        // just be overwritten and the change lost until an unrelated edit forced a remesh.
        // That race is why the same seed settled to a different quad count run to run.
        if (e.meshingLod >= 0)
            e.lightDirty = true;
        else if (e.state == ChunkState::Meshed) {
            e.remesh = true; // back through the mesh pipeline, still drawn meanwhile
            m_meshScanDirty = true;
        }
    }
    light.clearTouched();
    const double propagateMs = msSince(propStart);
    m_phase.propagateMs = std::max(m_phase.propagateMs, propagateMs);
    m_phaseSum.propagateMs += propagateMs;

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
    const bool doMeshScan =
        m_meshScanDirty || m_jobsInFlight > 0 || !m_genQueue.empty() || !m_uploads.empty();
    m_meshScanDirty = false;
    m_meshScratch.clear();
    m_regenScratch.clear();
    m_emptyScratch.clear();
    // The self-heal below is a safety net for a rare path, and it was a hash lookup for every
    // resident chunk on every scanned frame (1.3 ms a frame at radius 32). An eighth of the
    // scans is still well under a second behind.
    const bool heal = doMeshScan && (m_healTick++ & 7u) == 0;
    if (doMeshScan)
    for (const auto& item : m_chunks) {
        const Entry& e = item.second;
        // Catch columns that became complete without anyone noticing. A column is only
        // ever put up for seeding from the gen-done drain, keyed on one of ITS chunks
        // finishing generation — so a column that completes by any other route (an entry
        // erased while its job was in flight, a re-gen swap, a rescan reshuffling what was
        // queued) is never offered at all, and stays unseeded forever holding sky light 0.
        // That is open sky rendering pitch black regardless of time of day.
        //
        // The scan already walks every chunk and only runs when something changed, so
        // checking here costs a set lookup on work that was happening anyway.
        // NB: gated on m_seededColumns, never on e.lit. evicting one chunk of a column
        // drops the column from m_seededColumns but leaves the surviving chunks flagged
        // lit, so e.lit says "done" for a column that no longer has a seed to its name.
        if (heal && e.state != ChunkState::Generating && !e.pendingDetail) {
            const ChunkPos col{item.first.x, 0, item.first.z};
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
            }
        }
        if (!e.lit || e.pendingDetail) // pendingDetail: a detailed re-gen is landing
            continue;
        if (std::abs(item.first.x - center.x) > radius ||
            std::abs(item.first.z - center.z) > radius)
            continue;
        const int desired = lodForDist2(horizontalDist2(item.first, center), lodScale());
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
            m_regenScratch.push_back(item.first);
        if (!(firstMesh || reLod) || e.meshingLod == desired)
            continue;
        // Solid air has nothing to draw at any LOD and bakes no light, so it needs neither a job
        // nor its neighbours: at radius 32 it was 25k of the ~35k chunks waiting to mesh, each
        // holding a job slot like a real mesh.
        if (e.meshingLod < 0 && e.voxels->uniform() && e.voxels->uniformId() == blocks::Air) {
            m_emptyScratch.push_back(item.first);
            continue;
        }
        bool ready = true;
        for (const auto& o : kFaceOffsets) {
            const ChunkPos n{item.first.x + o[0], item.first.y + o[1], item.first.z + o[2]};
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
            m_meshScratch.push_back(item.first);
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
            m_regenScratch.resize(room + 1); // one past, so the loop below still flags the backlog
        }
        std::sort(m_regenScratch.begin(), m_regenScratch.end(), nearer);
    }
    for (const ChunkPos pos : m_regenScratch) {
        if (m_jobsInFlight >= inflightCap || m_regenInFlight >= inflightCap / 4) {
            m_meshScanDirty = true; // backlog: re-scan next frame to schedule the rest
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
    m_pendingMesh = int(m_meshScratch.size());

    for (const ChunkPos pos : m_meshScratch) {
        if (m_jobsInFlight >= inflightCap) {
            m_meshScanDirty = true; // backlog: keep scanning until the rest get scheduled
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
            thread_local GreedyMesher mesher; // one per worker thread, reused across jobs
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

    // land finished re-gens: swap the detailed chunk in once no worker is still reading the
    // old (coarse) voxels, then re-seed the column so the new caves/glow light correctly
    // (which re-meshes it; the coarse mesh stays up until that lands).
    {
        std::lock_guard lock(m_regenDoneMutex);
        m_regenDoneScratch.swap(m_regenDone);
    }
    for (const RegenResult& r : m_regenDoneScratch) {
        const auto it = m_chunks.find(r.pos);
        if (it == m_chunks.end() || it->second.pendingDetail != r.voxels) {
            m_pool.release(r.voxels); // entry evicted or superseded: drop this re-gen
            --m_jobsInFlight;
            --m_regenInFlight;
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
            continue;
        }
        Chunk* old = e.voxels;
        e.voxels = r.voxels;
        e.pendingDetail = nullptr;
        e.detailed = true;
        m_pool.release(old);
        --m_jobsInFlight;
        --m_regenInFlight;
        const ChunkPos col{r.pos.x, 0, r.pos.z};
        if (m_seededColumns.erase(col)) { // voxels came back with light wiped -> re-seed
            m_seedQueue.push_back(col);
            // Same #27 leftover as the seed sweep: clearing lit over 0..kWorldChunksY misses
            // every resident chunk outside that range, which would then keep a lit flag for
            // light that was just wiped — and meshing trusts lit.
            int y0 = 0, y1 = 0;
            bool sky = false;
            columnRange(col.x, col.z, y0, y1, sky);
            for (int cy = y0; cy <= y1; ++cy)
                if (const auto cit = m_chunks.find({r.pos.x, cy, r.pos.z}); cit != m_chunks.end())
                    cit->second.lit = false;
        }
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
        m_floodSeen[c] = 1;
    m_floodQueue.push_back({cameraChunk, -1, 0, camIt->second.connectivity});

    static constexpr int kStep[6][3] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                        {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
    for (std::size_t head = 0; head < m_floodQueue.size(); ++head) {
        const FloodNode node = m_floodQueue[head]; // copy: pushes may reallocate
        for (int d = 0; d < 6; ++d) {
            // never step back toward the camera: each axis flows one way per path.
            // (this is what keeps the flood bounded; it can over-cull wrap-around
            // paths, which is the algorithm's documented trade)
            if (node.dirMask & (1u << (d ^ 1)))
                continue;
            // continue only through a face pair this chunk's air actually connects
            if (node.entryFace >= 0 && !facesConnected(node.connectivity, node.entryFace, d))
                continue;
            const ChunkPos next{node.pos.x + kStep[d][0], node.pos.y + kStep[d][1],
                                node.pos.z + kStep[d][2]};
            const std::size_t c = cellOf(next);
            if (c == std::size_t(-1) || m_floodSeen[c] || m_floodGrid[c] == nullptr)
                continue;
            m_floodSeen[c] = 1;
            const Entry& e = *m_floodGrid[c];
            markVisible(e);
            m_floodQueue.push_back({next, std::int8_t(d ^ 1),
                                    std::uint8_t(node.dirMask | (1u << d)), e.connectivity});
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
    ++m_floodStamp;
    out.assign(std::size_t(std::max(1, (m_metaCount + 31) / 32)), 0u);
    const auto markVisible = [&](Entry& e) {
        e.floodStamp = m_floodStamp;
        if (e.metaIndex >= 0)
            out[std::size_t(e.metaIndex) >> 5] |= 1u << (e.metaIndex & 31);
    };

    const auto camIt = m_chunks.find(cameraChunk);
    if (camIt == m_chunks.end()) {
        // camera outside the loaded world — draw everything rather than nothing
        std::fill(out.begin(), out.end(), 0xFFFFFFFFu);
        return;
    }

    m_floodQueue.clear();
    markVisible(camIt->second);
    m_floodQueue.push_back({cameraChunk, -1, 0, camIt->second.connectivity});

    static constexpr int kStep[6][3] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                        {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
    for (std::size_t head = 0; head < m_floodQueue.size(); ++head) {
        const FloodNode node = m_floodQueue[head]; // copy: pushes may reallocate
        for (int d = 0; d < 6; ++d) {
            if (node.dirMask & (1u << (d ^ 1)))
                continue;
            if (node.entryFace >= 0 && !facesConnected(node.connectivity, node.entryFace, d))
                continue;
            const ChunkPos next{node.pos.x + kStep[d][0], node.pos.y + kStep[d][1],
                                node.pos.z + kStep[d][2]};
            if (next.y < 0 || next.y >= kWorldChunksY)
                continue;
            if (std::abs(next.x - m_center.x) > radius || std::abs(next.z - m_center.z) > radius)
                continue;
            const auto it = m_chunks.find(next);
            if (it == m_chunks.end() || it->second.floodStamp == m_floodStamp)
                continue;
            markVisible(it->second);
            m_floodQueue.push_back({next, std::int8_t(d ^ 1),
                                    std::uint8_t(node.dirMask | (1u << d)),
                                    it->second.connectivity});
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

void ChunkManager::saveDirty() {
    int saved = 0;
    for (auto& [pos, e] : m_chunks)
        if (e.dirty && e.voxels) {
            m_saveScratch.clear();
            ChunkSerializer::serialize(*e.voxels, m_saveScratch);
            m_regions.save(pos, m_saveScratch);
            e.dirty = false;
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
    // per edit, and light must not settle against a half-applied batch.
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
        light.clearTouched();
        return 0;
    }

    // Pass 2: settle the whole batch's light in one BFS. Removals and re-propagation from
    // different edits overlap heavily, so settling once is strictly less work than
    // settling per edit — and it means no chunk is ever meshed against half-updated light.
    while (!light.idle())
        light.process(4.0);
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
