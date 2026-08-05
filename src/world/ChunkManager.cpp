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
#include <utility>

namespace vx {

ChunkManager::ChunkManager(std::filesystem::path saveDir) : m_regions(std::move(saveDir)) {}


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
// Bands scale with the render distance instead of sitting at fixed chunk counts. They
// were tuned at radius 16 and stayed there, so raising the radius did not push detail
// outward with it — it just added more of the coarsest band, and full detail still ran
// out at 8 chunks no matter how far you could see. The ratios below reproduce the old
// 8/14/22/30/42 exactly at radius 16, so that case is unchanged.
int lodForDist2(std::int64_t d2, int radius) {
    const double s = std::max(radius, 16) / 16.0; // only ever widens; 16 is the floor
    const auto band = [&](double chunks) {
        const double d = chunks * s;
        return std::int64_t(d * d);
    };
    if (d2 <= band(8)) return 0;
    if (d2 <= band(14)) return 1;
    if (d2 <= band(22)) return 2;
    if (d2 <= band(30)) return 3;
    if (d2 <= band(42)) return 4;
    return 5;
}

// chunks meshed at this LOD or coarser generate without caves/trees (they'd be averaged
// away by the downsample). approaching one — desired LOD drops below this — re-generates
// it detailed. caves cost ~4x gen/mesh/quads, so skipping them far out is the big win.
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

std::uint8_t coarseNeighborMask(ChunkPos pos, int lod, ChunkPos center, int radius) {
    std::uint8_t mask = 0;
    for (int f = 0; f < 6; ++f) {
        const ChunkPos n{pos.x + kFaceOffsets[f][0], pos.y + kFaceOffsets[f][1],
                         pos.z + kFaceOffsets[f][2]};
        const double d = std::sqrt(double(horizontalDist2(n, center))) + kSkirtMargin;
        if (lodForDist2(std::int64_t(d * d), radius) > lod)
            mask |= std::uint8_t(1u << f);
    }
    return mask;
}

} // namespace

bool ChunkManager::columnPresent(ChunkPos col) const {
    for (int cy = 0; cy < kWorldChunksY; ++cy)
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
            m_dirty = true;
        }
        if (e.waterRange.count != 0) {
            renderer.free(e.waterRange);
            m_dirty = true;
        }
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
        if (!e.quadsCpu.empty()) {
            e.range = renderer.alloc(e.quadsCpu);
            if (e.range.count == 0)
                VX_ERROR("arena rebuild: chunk still does not fit — arena too small");
        }
        if (!e.waterQuadsCpu.empty())
            e.waterRange = renderer.alloc(e.waterQuadsCpu);
    }
    m_dirty = true;
}

void ChunkManager::update(JobSystem& jobs, Renderer& renderer, LightEngine& light,
                          const glm::dvec3& cameraPos) {
    const ChunkPos center{int(std::floor(cameraPos.x / kChunkSize)), 0,
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

        // once the whole vertical column has data, it can be light-seeded (sky descent
        // spans all of it at once)
        const ChunkPos col{pos.x, 0, pos.z};
        if (!m_seededColumns.contains(col)) {
            bool complete = true;
            for (int cy = 0; cy < kWorldChunksY && complete; ++cy) {
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
            for (int cy = 0; cy < kWorldChunksY; ++cy)
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
                        cit->second.state = ChunkState::Ready;
                }
            m_meshScanDirty = true; // newly lit -> meshable
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
        const auto requeueSeed = [&](ChunkPos col) {
            m_seededColumns.erase(col);
            m_provisionalColumns.erase(col);
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
        // Put deferred columns back at the front so they're retried only after everything
        // else (pop_back serves the tail). One splice a frame, not one memmove of the
        // whole vector per deferral, which is what made this quadratic.
        if (!m_seedDeferred.empty()) {
            m_seedQueue.insert(m_seedQueue.begin(), m_seedDeferred.begin(),
                               m_seedDeferred.end());
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
            e.state = ChunkState::Ready; // back through the mesh pipeline
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

    while (m_jobsInFlight < inflightCap && !m_genQueue.empty()) {
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
        const bool detailed = lodForDist2(horizontalDist2(pos, center), radius) < kCoarseLod;
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
        if (e.state != ChunkState::Generating && !e.pendingDetail) {
            const ChunkPos col{item.first.x, 0, item.first.z};
            if (!m_seededColumns.contains(col) && columnPresent(col)) {
                bool ready = true;
                for (int cy = 0; cy < kWorldChunksY && ready; ++cy)
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
        const int desired = lodForDist2(horizontalDist2(item.first, center), radius);
        const bool firstMesh = e.state == ChunkState::Ready;
        const bool reLod = e.state == ChunkState::Meshed && e.lod != desired;
        if (!(firstMesh || reLod) || e.meshingLod == desired)
            continue;
        // a coarse chunk that's now close enough to want detail: re-generate it (with
        // caves) rather than re-meshing the cave-less data at a detailed LOD
        if (reLod && !e.detailed && desired < kCoarseLod && !e.voxels->uniform()) {
            m_regenScratch.push_back(item.first);
            continue;
        }
        bool ready = true;
        for (const auto& o : kFaceOffsets) {
            const ChunkPos n{item.first.x + o[0], item.first.y + o[1], item.first.z + o[2]};
            if (n.y < 0 || n.y >= kWorldChunksY)
                continue; // outside the world counts as present (air)
            const auto nit = m_chunks.find(n);
            if (nit == m_chunks.end() || nit->second.state == ChunkState::Generating ||
                !nit->second.lit) {
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
        if (m_jobsInFlight >= inflightCap) {
            m_meshScanDirty = true; // backlog: keep scanning until the rest get scheduled
            break;
        }
        const auto it = m_chunks.find(pos);
        Entry& e = it->second;
        const int lod = lodForDist2(horizontalDist2(pos, center), radius);
        if (e.state == ChunkState::Ready)
            e.state = ChunkState::Meshing; // first mesh: not drawn until it lands
        // (a re-LOD keeps state Meshed so the old mesh stays drawn meanwhile)
        e.meshingLod = std::int8_t(lod);

        const std::uint8_t skirt = coarseNeighborMask(pos, lod, center, radius);

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
        jobs.submit([this, voxels, nb, pos, lod, skirt] {
            // one mesher + scratch per worker thread, reused across jobs
            thread_local GreedyMesher mesher;
            thread_local std::vector<std::uint64_t> quadScratch;
            thread_local std::vector<std::uint64_t> waterScratch;
            const auto t0 = Clock::now();
            const GreedyStats stats =
                mesher.mesh(*voxels, nb.data(), quadScratch, waterScratch, lod, skirt);
            const std::uint16_t connectivity = computeConnectivity(*voxels);
            m_meshNanos.fetch_add(nsSince(t0), std::memory_order_relaxed);
            m_meshCount.fetch_add(1, std::memory_order_relaxed);

            MeshResult result{pos,
                              {quadScratch.begin(), quadScratch.end()},
                              {waterScratch.begin(), waterScratch.end()},
                              stats,
                              connectivity,
                              std::int8_t(lod)};
            voxels->releaseJob();
            for (const Chunk* n : nb)
                if (n)
                    n->releaseJob();
            std::lock_guard lock(m_meshDoneMutex);
            m_meshDone.push_back(std::move(result));
        });
    }

    // schedule detailed re-gens for approached coarse chunks. double-buffered into a fresh
    // chunk so the coarse mesh keeps drawing until the detailed one lands (no hole), and
    // the worker only touches its own fresh chunk (no race with neighbour mesh jobs).
    // nearest first so the detail you're walking into fills in before the far stuff.
    std::sort(m_regenScratch.begin(), m_regenScratch.end(), [center](ChunkPos a, ChunkPos b) {
        return horizontalDist2(a, center) < horizontalDist2(b, center);
    });
    for (const ChunkPos pos : m_regenScratch) {
        if (m_jobsInFlight >= inflightCap) {
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
        jobs.submit([this, c2, pos] {
            if (!m_regions.load(pos, *c2))
                m_gen.generate(*c2, pos, /*detailed=*/true);
            c2->releaseJob();
            std::lock_guard lock(m_regenDoneMutex);
            m_regenDone.push_back({pos, c2});
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
            continue;
        }
        Entry& e = it->second;
        if (e.voxels->hasJobRefs()) { // a neighbour mesh job still reads the old voxels
            std::lock_guard lock(m_regenDoneMutex);
            m_regenDone.push_back(r); // retry next frame; keep the job slot counted
            continue;
        }
        Chunk* old = e.voxels;
        e.voxels = r.voxels;
        e.pendingDetail = nullptr;
        e.detailed = true;
        m_pool.release(old);
        --m_jobsInFlight;
        const ChunkPos col{r.pos.x, 0, r.pos.z};
        if (m_seededColumns.erase(col)) { // voxels came back with light wiped -> re-seed
            m_seedQueue.push_back(col);
            for (int cy = 0; cy < kWorldChunksY; ++cy)
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
            e.quads = int(r.quads.size());
            e.waterQuads = int(r.water.size());
            if (!r.quads.empty()) {
                e.range = renderer.alloc(r.quads);
                if (e.range.count == 0) { // arena full — defrag and retry once
                    rebuildArena(renderer);
                    e.range = renderer.alloc(r.quads);
                }
                e.quadsCpu = std::move(r.quads);
            } else {
                e.quadsCpu.clear();
            }
            if (!r.water.empty()) {
                e.waterRange = renderer.alloc(r.water);
                if (e.waterRange.count == 0) {
                    rebuildArena(renderer);
                    e.waterRange = renderer.alloc(r.water);
                }
                e.waterQuadsCpu = std::move(r.water);
            } else {
                e.waterQuadsCpu.clear();
            }
            e.connectivity = r.connectivity;
            e.sunlit = r.stats.sunlit;
            e.lod = r.lod;
            e.state = ChunkState::Meshed;
            m_dirty = true;
            m_visDirty = true;
        }
        // Light moved while this job was out. The geometry it brought back is still the
        // newest there is, so keep it and send the chunk round once more rather than
        // leaving quads that baked light the world has since moved past. Handled here
        // rather than inside accept so a dropped stale result clears the flag too.
        if (it != m_chunks.end() && it->second.lightDirty) {
            it->second.lightDirty = false;
            if (it->second.state == ChunkState::Meshed)
                it->second.state = ChunkState::Ready;
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

void ChunkManager::rebuildVisibility(ChunkPos cameraChunk) {
    ++m_floodStamp;
    m_visBits.assign(std::size_t(std::max(1, (m_metaCount + 31) / 32)), 0u);
    const auto markVisible = [&](Entry& e) {
        e.floodStamp = m_floodStamp;
        if (e.metaIndex >= 0)
            m_visBits[std::size_t(e.metaIndex) >> 5] |= 1u << (e.metaIndex & 31);
    };

    const auto camIt = m_chunks.find(cameraChunk);
    if (camIt == m_chunks.end()) {
        // camera outside the loaded world — draw everything rather than nothing
        std::fill(m_visBits.begin(), m_visBits.end(), 0xFFFFFFFFu);
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
    // edits are always on full-detail chunks, but one can still border a coarser ring
    const GreedyStats stats = m_mesher.mesh(*e.voxels, neighbors, m_quadScratch,
                                            m_waterScratch, 0,
                                            coarseNeighborMask(pos, 0, m_center, radius));
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
        e.range = renderer.alloc(m_quadScratch);
        if (e.range.count == 0) {
            rebuildArena(renderer);
            e.range = renderer.alloc(m_quadScratch);
        }
        e.quadsCpu.assign(m_quadScratch.begin(), m_quadScratch.end());
    } else {
        e.quadsCpu.clear();
    }
    if (stats.waterQuads > 0) {
        e.waterRange = renderer.alloc(m_waterScratch);
        if (e.waterRange.count == 0) {
            rebuildArena(renderer);
            e.waterRange = renderer.alloc(m_waterScratch);
        }
        e.waterQuadsCpu.assign(m_waterScratch.begin(), m_waterScratch.end());
    } else {
        e.waterQuadsCpu.clear();
    }
    e.lod = 0;         // edits are always on near (full-detail) chunks
    e.meshingLod = -1; // drop any in-flight async re-LOD result for this chunk
    e.state = ChunkState::Meshed; // a stale worker result for this chunk now gets dropped
    m_dirty = true;
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
        // finish in well under a millisecond, so a bounded spin here beats copy-on-write
        while (c->hasJobRefs()) {
        }
        c->set(lx, ly, lz, e.id);
        it->second.dirty = true; // content changed: this chunk now needs persisting
        // if a save job already ran for this chunk it captured the pre-edit voxels, so
        // clear the guard or the new content would never be written
        it->second.saving = false;
        light.onBlockChanged(e.pos, old, e.id);
        ++applied;

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

int ChunkManager::stepFluid(Renderer& renderer, LightEngine& light,
                            const glm::dvec3& around) {
    // Radius is small on purpose: fluid only has to be right where it can be seen moving,
    // and the cost is a scan of every cell in the box. Water far away is already settled.
    constexpr int kR = 40;
    const glm::ivec3 c{int(std::floor(around.x)), int(std::floor(around.y)),
                       int(std::floor(around.z))};
    const int y0 = std::max(c.y - kR / 2, 0);
    const int y1 = std::min(c.y + kR / 2, kWorldHeight - 1);

    m_fluidEdits.clear();
    for (int y = y1; y >= y0; --y) // top-down: water falls this tick rather than next
        for (int z = c.z - kR; z <= c.z + kR; ++z)
            for (int x = c.x - kR; x <= c.x + kR; ++x) {
                const glm::ivec3 p{x, y, z};
                const BlockId here = blockAtWorld(p);
                const int level = blocks::waterLevel(here);
                if (level == 8)
                    continue; // source: never decays, never recomputed
                if (here != blocks::Air && level == 0)
                    continue; // solid

                // Fed from directly above at near-full depth (a falling column stays a
                // column), otherwise from the deepest horizontal neighbour minus one.
                int want = 0;
                if (blocks::isWater(blockAtWorld({x, y + 1, z}))) {
                    want = 7;
                } else {
                    static constexpr int kSide[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
                    for (const auto& d : kSide)
                        want = std::max(want,
                                        blocks::waterLevel(blockAtWorld({x + d[0], y, z + d[1]})) - 1);
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
    return setBlocks(renderer, light, m_fluidEdits);
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
            sc.quadBytes += std::size_t(e.quads) * sizeof(std::uint64_t);
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
