#pragma once

#include "core/Pool.hpp"
#include "gen/TerrainGenerator.hpp"
#include "io/RegionStore.hpp"
#include "mesh/GreedyMesher.hpp"
#include "render/Renderer.hpp"
#include "world/Chunk.hpp"
#include "world/ChunkPos.hpp"
#include "world/MeshBufferPool.hpp"

#include <glm/glm.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <semaphore>
#include <span>
#include <thread>
#include <vector>

namespace vx {

struct Frustum;
class JobSystem;
class LightEngine;

// Per-chunk pipeline state. Only the main thread reads or writes it, workers are pure
// functions that hand results back through queues, so no state needs to be atomic. The
// one cross-thread liveness concern (a mesh job reading a neighbor while it unloads)
// is handled by the job refcount on Chunk itself.
enum class ChunkState : std::uint8_t {
    Generating, // gen job in flight, voxels being written on a worker
    Ready,      // voxel data present and immutable, not meshed yet
    Meshing,    // mesh job in flight
    Meshed,     // quads uploaded (possibly zero of them)
};

// Owns the loaded world. Generation and meshing run on JobSystem workers; the main
// thread schedules jobs, drains results, and does the only GL work (arena uploads)
// under a per-frame time budget so a streaming burst never stalls a frame. Quads live
// in the Renderer's arena; a CPU copy stays here so the arena can be rebuilt after a
// hard reset without remeshing.
class ChunkManager {
public:
    // The seed is a runtime value, not a constant: every launch generates a fresh world, so
    // a world you want to keep has to be saved deliberately (see Worlds in main).
    // genVersion: which TerrainGenerator made this world (a saved world keeps its own).
    ChunkManager(std::filesystem::path saveDir, int seed,
                 int genVersion = TerrainGenerator::kLatestVersion);
    ~ChunkManager();
    ChunkManager(const ChunkManager&) = delete;
    ChunkManager& operator=(const ChunkManager&) = delete;

    struct Stats {
        int loaded = 0;
        int meshed = 0;
        int unlit = 0;
        int uniformChunks = 0;
        int genQueue = 0;
        int jobsInFlight = 0;
        int pendingUploads = 0;
        std::int64_t totalQuads = 0;
        std::size_t voxelBytes = 0;
        std::size_t quadBytes = 0;
        // Chunks that arrived into a column already marked seeded and were lit on the
        // spot. Nonzero is normal (it is what vertical movement does); it is here because
        // the smoke asserts it is nonzero, without that, a change that stops chunks ever
        // landing this way would leave the vertical group passing while testing nothing.
        int lateSeeds = 0;
        double genMsAvg = 0.0;
        double meshMsAvg = 0.0;
        double uploadMsLast = 0.0;
    };

    void update(JobSystem& jobs, Renderer& renderer, LightEngine& light,
                const glm::dvec3& cameraPos);

    // The light phase off the main thread, for the game loop: start hands this frame's seeding
    // and propagation to the light thread, finish waits for it and takes in what it did, and
    // the next update() skips doing it again. Between the two the main thread may READ chunks
    // and their light (draw, sample) but change nothing in the world or the light engine.
    // Without the pair, update() runs the same steps inline, as the harnesses rely on.
    void startLightSlice(LightEngine& light);
    void finishLightSlice(LightEngine& light);

    // A chunk's voxels once generated, null while a worker is still writing them. The light
    // engine reads and writes through these, and light spreading sideways used to walk straight
    // into chunks being generated at the ring's edge: reading a palette mid-rebuild, and writing
    // light the chunk then did or did not keep depending on timing.
    const Chunk* chunkData(ChunkPos pos) const {
        const auto it = m_chunks.find(pos);
        return it != m_chunks.end() && it->second.state != ChunkState::Generating ? it->second.voxels : nullptr;
    }
    // Its column has been through the light seed (see LightEngine::checkColumnEdges).
    bool chunkLit(ChunkPos pos) const {
        const auto it = m_chunks.find(pos);
        return it != m_chunks.end() && it->second.lit;
    }
    Chunk* chunkDataMutable(ChunkPos pos) {
        const auto it = m_chunks.find(pos);
        return it != m_chunks.end() && it->second.state != ChunkState::Generating ? it->second.voxels : nullptr;
    }

    BlockId blockAtWorld(const glm::ivec3& wp) const;
    // Sky light 0..15 at a world voxel, 0 where nothing is loaded. The ambience director
    // wants "how enclosed is the ear", and this already answers it exactly, the light BFS
    // has done the work, so asking costs one chunk lookup instead of a cover raycast every
    // frame. An absent chunk reads 0 (sealed) rather than 15, so streaming in never makes a
    // cave briefly sound like open sky.
    int skyLightAtWorld(const glm::ivec3& wp) const;
    // debug: has this column been through a completed light seed?
    bool columnSeeded(ChunkPos col) const { return m_seededColumns.contains(col); }
    bool columnProvisional(ChunkPos col) const { return m_provisionalColumns.contains(col); }

    // The block-edit fast path: sets the voxel, settles light, and synchronously
    // remeshes the affected chunk (plus border neighbors and anything light touched)
    // on the main thread, edit to visible in the same frame, per the spec target.
    bool setBlock(Renderer& renderer, LightEngine& light, const glm::ivec3& wp, BlockId id);

    // Batch form. A simulation (flowing water, falling sand, break progress) changes many
    // voxels per tick, and setBlock is built for one: it settles the whole light BFS and
    // then synchronously remeshes every affected chunk, so N edits cost N settles and
    // re-remesh the same chunks N times. This applies every edit first, settles light
    // ONCE, then remeshes each affected chunk once. Returns how many edits took effect.
    //
    // Edits are applied in order, so two edits to the same voxel resolve last-wins, and
    // light sees the finished state of every other voxel rather than an intermediate one.
    struct BlockEdit {
        glm::ivec3 pos;
        BlockId id;
    };
    int setBlocks(Renderer& renderer, LightEngine& light, std::span<const BlockEdit> edits);

    // One fluid step over the cells queued by edits (near `around` first). Cellular, not
    // particle-based: every non-source water cell recomputes its depth from its neighbours,
    // so the field is stateless and self-correcting, no per-cell velocity to keep
    // consistent, and an interrupted or budget-truncated tick just resolves on the next one.
    // Returns the number of voxels that changed. Drives the batched edit path, because a
    // tick changes many voxels and one-at-a-time would settle light and remesh per voxel.
    int stepFluid(Renderer& renderer, LightEngine& light, const glm::dvec3& around);

    // One gravity step for sand and gravel, over the DISTURBED set only. Same batched path
    // as the fluid: a tick moves many voxels and doing them one at a time would settle
    // light and remesh per voxel.
    //
    // Event-driven, not a scan. A box scan looked simpler and was unusable: the box is
    // ~118k cells of hash lookups per tick, and worse, terrain generates gravel inside
    // stone that caves then undercut, so a scan finds thousands of unsupported blocks the
    // moment a world loads and never stops moving them, remeshing continuously. Gravel
    // that nobody has touched staying put is also the behaviour you want.
    // `landed` counts blocks that came to rest THIS tick having actually fallen, and
    // `landedAt` is one of them, enough for a sound, since a collapsing pile is one event to
    // the ear rather than thirty. The distinction matters: a block that cannot fall was not
    // necessarily falling, it may have just been placed on solid ground, and that already
    // made its own noise.
    int stepFalling(Renderer& renderer, LightEngine& light, int* landed = nullptr,
                    glm::ivec3* landedAt = nullptr);
    // A voxel worth re-testing for falling: the block itself, and whatever sits on it.
    void queueFallCheck(const glm::ivec3& p);
    // A voxel whose water depth may have changed: itself and its six neighbours. Fluid is
    // event-driven for the same reason falling is, a scan finds nothing almost every tick.
    void queueFluidCheck(const glm::ivec3& p);

    const TerrainGenerator& generator() const { return m_gen; }
    RegionStore& regions() { return m_regions; }

    // Serializes every dirty (player-edited) chunk to its region and flushes to disk.
    // Unedited chunks regenerate from the seed, so only diffs from procedural gen are
    // stored. Main thread.
    void saveDirty();

    // Release everything resident and come up on a different world: new seed, new save
    // directory. Used by "new world" and by loading a saved one. Safe only when no jobs are
    // in flight, which the caller guarantees by draining first, and the caller must put a
    // loading screen in front of the re-stream, otherwise the whole world generates around
    // the player, which is both ugly and a long stall.
    void loadWorld(Renderer& renderer, int seed, std::filesystem::path saveDir,
                   int genVersion = TerrainGenerator::kLatestVersion);
    int seed() const { return m_seed; }
    int genVersion() const { return m_gen.version(); }

    // How many chunks a settled world at the current radius will have meshed. The loading
    // screen divides by this, so it has to be the real figure: radius*radius*kWorldChunksY
    // was out by 5x at radius 16 (3072 against ~15200), which pinned the bar at 100% for
    // almost the entire load. Summed from columnRange, which is what streaming actually
    // uses, so the two cannot drift apart.
    int expectedMeshedChunks() const;


    // For the cull compute pass; call when consumeDirty() says the chunk set changed.
    // Assigns each drawable chunk its meta index (the visibility bitset uses the same
    // indexing), so the two must always be rebuilt together. Also fills the separate
    // water draw list for the transparent pass.
    void collectDrawInfo(std::vector<ChunkDrawInfo>& out, std::vector<ChunkDrawInfo>& water);
    bool consumeDirty() {
        const bool was = m_dirty;
        m_dirty = false;
        return was;
    }
    // Runs the dense-grid flood and the hash-map reference from `cameraChunk` and says whether
    // they agree bit for bit. Test only: the game only ever runs the grid one.
    bool floodMatchesReference(ChunkPos cameraChunk, double* gridMs = nullptr,
                               double* hashMs = nullptr);

    // How close to the camera, in blocks, the nearest chunk changed since the last call (the
    // shadow cascades only need redrawing where the change is, not everywhere). Infinite if
    // nothing did.
    float takeDirtyNearest() {
        const float d = m_dirtyNearest;
        m_dirtyNearest = 1e30f;
        return d;
    }

    // Recomputes the camera-outward visibility flood (cave culling) only when needed:
    // the chunk set changed, the camera crossed a chunk boundary, or the toggle
    // flipped. Steady-state cost is zero.
    void updateVisibilityIfNeeded(Renderer& renderer, const glm::dvec3& cameraPos,
                                  bool enabled);

    // worst single-frame cost of each main-thread phase since the last reset. only used
    // to find where a streaming hitch actually is, rather than guessing at it.
    struct PhasePeak {
        double rescanMs = 0.0;
        double evictMs = 0.0;
        double scanMs = 0.0;
        double seedMs = 0.0;
        double propagateMs = 0.0;
        double uploadMs = 0.0;
        // How long the main thread stood waiting for the light thread at finishLightSlice. Seed
        // and propagate are the light thread's own time when it runs them; this is what of it
        // the frame still paid.
        double lightWaitMs = 0.0;
    };
    PhasePeak phasePeak() const { return m_phase; }
    // Same phases summed over every update instead of maxed. A peak tells you how bad one
    // frame got; only the total tells you which phase actually owns the streaming budget,
    // and the two answers differ a lot for a phase that is time-budgeted per frame.
    PhasePeak phaseTotal() const { return m_phaseSum; }
    int updateCount() const { return m_updates; }
    void resetPhasePeak() {
        m_phase = {};
        m_phaseSum = {};
        m_updates = 0;
    }

    bool busy() const {
        // the seed queue counts: a column that hasn't been light-seeded yet is unlit and
        // therefore unmeshed, so the world is not done streaming no matter how empty the
        // mesh side looks. leaving it out let "re-streamed" report a finish while chunks
        // were still dark.
        // m_meshScanDirty counts too: a chunk just sent back for a relight has no job and
        // no queue entry until the next scan finds it, so leaving it out reports "settled"
        // one frame early, with stale light still baked into the quads on screen.
        // An edited chunk's eviction save counts as well: it runs on a worker but is not one of
        // the counted jobs, and a caller that drains on this before switching worlds would
        // otherwise pull the world out from under it (SPEC #142).
        return !m_genQueue.empty() || m_jobsInFlight > 0 || !m_uploads.empty() ||
               m_pendingMesh > 0 || !m_seedQueue.empty() || m_seedActive || m_meshScanDirty ||
               m_scanConfirm || m_sliceRunning ||
               m_savesInFlight.load(std::memory_order_acquire) > 0;
    }

    // Columns waiting for their light seed. Cheap; --smoke watches it frame by frame.
    int seedBacklog() const { return int(m_seedQueue.size()); }

    // --smoke only: hold every eviction save this long before it serializes, so the window in
    // which one is still running can be made wide enough to test.
    void setSaveDelayForTest(int ms) { m_saveDelayMs.store(ms, std::memory_order_relaxed); }
    Stats stats() const;
    void logSummary() const;
    // force the next stats() to do the full-chunk scan (used before logSummary)
    void refreshStats() const { m_scanAge = 999; }

    // For --smoke, specifically a live radius change (SPEC #57): once busy() is false
    // (settled, every queue drained, mesh scan clean), every resident chunk's LOD/detail
    // state should agree with what its distance and the CURRENT radius call for. Any of
    // these being nonzero after a full settle is a real bug, not a mid-stream snapshot,
    // see auditLod()'s own comment for why each one specifically means that.
    struct LodAudit {
        int stranded = 0;          // resident past the eviction threshold, should be gone
        int lodMismatch = 0;       // Meshed at an LOD other than what desired-now calls for
        int coarseInDetailBand = 0; // wants detail (self-heal condition true) but never got it
    };
    LodAudit auditLod() const;

    // For --smoke: is this specific chunk actually meshed and lit, as opposed to "voxels
    // exist but nothing drew it yet", the distinction a starved light-seed queue can leave
    // indefinitely on exactly the column that matters most (SPEC: "nothing generates in the
    // area I spawn in").
    bool chunkMeshedAndLit(ChunkPos pos) const {
        const auto it = m_chunks.find(pos);
        return it != m_chunks.end() && it->second.state == ChunkState::Meshed &&
               it->second.lit && !it->second.remesh;
    }

    int radius = 16; // render radius in chunks, horizontal
    // How far out full detail reaches, as a cap on how much the LOD bands may widen with the
    // radius: 1 keeps them at their radius-16 size at any radius, 2 lets them double at 32.
    // Full detail is most of the frame at a big radius, at radius 32, 2 against 1 was 22.4M
    // quads against 5.6M and about twice the frame time, so this is the player's trade. Each
    // quality preset sets it (Ultra 2); the default is High's.
    static constexpr float kDefaultDetailScale = 1.5f;
    float detailScale = kDefaultDetailScale;
    double lodScale() const; // what the bands are actually scaled by right now

private:
    struct Entry {
        Chunk* voxels = nullptr;
        Renderer::Range range;      // opaque quads in the arena
        Renderer::Range waterRange; // transparent quads in the arena
        int quads = 0;
        int waterQuads = 0;
        ChunkState state = ChunkState::Generating;
        bool lit = false;      // sky/emissive seeding done; meshing waits for it
        // Light moved after a mesh job for this chunk was already in flight, so the quads
        // it is producing bake values that are stale before they land. The state flip that
        // normally sends a chunk back round can't be used here, the landing result would
        // overwrite it, so the intent is parked until the drain can act on it.
        bool lightDirty = false;
        // Baked light is stale on an already-drawn chunk. The chunk stays Meshed and keeps
        // drawing its old quads until the fresh mesh lands; flipping it back to Ready made it
        // vanish in between, and one relit column takes its neighbours with it.
        bool remesh = false;
        // any face reached by sky light. false = sealed underground, so the shadow pass
        // skips it entirely (it can never be the surface nearest the sun)
        bool sunlit = true;
        bool dirty = false;    // player-edited since load; needs saving before eviction
        bool detailed = true;  // false = coarse gen (no caves/trees), for far chunks
        // a serialize+save job is running on a worker; the chunk cannot go back to the
        // pool until it finishes, and the job must not be submitted twice
        bool saving = false;
        // a detailed re-gen in flight into a second chunk (double buffer); the coarse mesh
        // stays drawn until this lands, so approaching a coarse chunk never flashes a hole
        Chunk* pendingDetail = nullptr;
        std::int8_t lod = 0;         // detail level of the current drawn mesh
        std::int8_t meshingLod = -1; // LOD of an in-flight (re)mesh job, -1 = none
        // all faces connected until meshed: never cave-cull what isn't known yet
        std::uint16_t connectivity = 0x7FFF;
        std::int32_t metaIndex = -1;   // slot in the cull meta buffer, -1 = not drawn
        // Opaque/transparent quads, kept for the chunk's whole residency (arena rebuilds
        // re-upload from them). Borrowed from m_quadPool/m_waterPool, not owned directly,
        // see MeshBufferPool.hpp. null = no geometry of that kind, or nothing meshed yet.
        std::vector<PackedQuad>* quadBuf = nullptr;
        std::vector<PackedQuad>* waterBuf = nullptr;
    };

    struct MeshResult {
        ChunkPos pos;
        std::vector<PackedQuad>* quadBuf = nullptr;
        std::vector<PackedQuad>* waterBuf = nullptr;
        GreedyStats stats;
        std::uint16_t connectivity = 0;
        std::int8_t lod = 0;
    };

    void rescan(ChunkPos center);
    void processEvictions(Renderer& renderer, JobSystem& jobs);
    void rebuildArena(Renderer& renderer);
    void rebuildVisibility(ChunkPos cameraChunk);
    // The same flood written against the hash map, kept as the reference the dense-grid one is
    // checked against (--smoke). It walks the six directions in the opposite order, so the check
    // also proves the result does not depend on the order. Fills `out` instead of m_visBits.
    void floodByHash(ChunkPos cameraChunk, std::vector<std::uint32_t>& out);
    void remeshNow(Renderer& renderer, ChunkPos pos);
    const Chunk* voxelsAt(ChunkPos pos) const;

    static constexpr double kUploadBudgetMs = 2.0;
    // sky/emissive column seeding, then the whole light phase (seed + propagate). one
    // deadline for both so their overshoots can't stack.
    static constexpr double kSeedBudgetMs = 2.0;
    static constexpr double kLightBudgetMs = 3.5;
    // Columns one slice may take on, and how long picking them may take.
    static constexpr std::size_t kSliceColumns = 12;
    static constexpr double kSlicePrepareMs = 0.5;

    struct LightSlice {
        bool continueSeed = false;
        ChunkPos continued{0, 0, 0};
        std::vector<ChunkPos> toSeed, finished, aborted, unstarted;
        double seedMs = 0.0, propagateMs = 0.0;
    };
    LightSlice m_slice;
    bool m_sliceRunning = false; // the light thread holds the slice right now
    bool m_sliceDriven = false;  // the last frame's slice ran on the light thread
    LightEngine* m_sliceLight = nullptr;
    std::binary_semaphore m_lightGo{0}, m_lightDone{0};
    std::jthread m_lightThread;
    bool neighborhoodSettled(ChunkPos col) const;
    void markColumnLit(ChunkPos col, LightEngine& light);
    void requeueSeed(ChunkPos col);
    void prepareLightSlice(LightEngine& light);
    void runLightSlice(LightEngine& light);
    void applyLightSlice(LightEngine& light);

    ankerl::unordered_dense::map<ChunkPos, Entry, ChunkPosHash> m_chunks;
    ankerl::unordered_dense::set<ChunkPos, ChunkPosHash> m_seededColumns; // y always 0
    std::vector<ChunkPos> m_seedQueue; // columns with full data, waiting for light seed
    std::vector<ChunkPos> m_seedDeferred; // this frame's not-yet-settled columns
    Pool<Chunk> m_pool;
    int m_seed;
    TerrainGenerator m_gen;
    RegionStore m_regions;
    GreedyMesher m_mesher; // main-thread instance for the synchronous edit remesh path
    // Every chunk's quad data, kept for its whole residency, lives in a slot here rather
    // than in a vector it owns directly, see MeshBufferPool.hpp. The async mesh-job path
    // (workers) and the synchronous edit-remesh path (remeshNow, main-thread-only) share
    // both pools; only checkout()/checkin() need to run on the main thread, which both
    // paths already do (the async path via the upload-drain loop, this one directly).
    MeshBufferPool m_quadPool;
    MeshBufferPool m_waterPool;
    std::vector<std::uint8_t> m_saveScratch;

    std::vector<ChunkPos> m_genQueue; // farthest-first, pop_back serves nearest
    std::vector<ChunkPos> m_meshScratch;
    std::vector<BlockEdit> m_fluidEdits;
    std::vector<ChunkPos> m_remeshScratch;
    // dedup for m_remeshScratch. a linear scan is fine for one edit's handful of chunks
    // but goes quadratic on a batch, which is exactly the case setBlocks exists for.
    ankerl::unordered_dense::set<ChunkPos, ChunkPosHash> m_remeshSeen;
    // Columns that were light-seeded while at least one horizontal neighbour column had
    // not loaded yet. The light engine treats an absent chunk as SOLID STONE, so those
    // columns were lit against a wall that does not exist, and m_seededColumns would
    // otherwise mark them finished forever. They get re-seeded once the missing neighbour
    // actually arrives.
    ankerl::unordered_dense::set<ChunkPos, ChunkPosHash> m_provisionalColumns;
    // Columns where a coarse->detailed regen has landed and that need one re-seed, held until
    // the last regen still pending in the column has landed or dropped (see reseedWhenSettled).
    ankerl::unordered_dense::set<ChunkPos, ChunkPosHash> m_reseedColumns;
    // The mesh scan in update() only walks every resident chunk on a full scan: the camera
    // changed chunk, the radius or detail changed, kFullScanEvery passed, or a partial scan has
    // run since the last full one and there is nothing left to look at (busy() waits for that
    // one). In between it looks at what was touched, their neighbours, and the nearest of the
    // candidates still waiting for a slot (m_scanCarry).
    static constexpr std::chrono::milliseconds kFullScanEvery{250};
    ankerl::unordered_dense::set<ChunkPos, ChunkPosHash> m_scanTouched;
    ankerl::unordered_dense::set<ChunkPos, ChunkPosHash> m_scanVisit;
    std::vector<ChunkPos> m_scanCarry;
    std::chrono::steady_clock::time_point m_lastFullScan{};
    ChunkPos m_scanCenter{INT32_MIN, 0, INT32_MIN};
    int m_scanRadius = -1;
    double m_scanLodScale = -1.0;
    bool m_scanConfirm = false;
    // Something about this chunk changed that can make it a mesh or regen candidate.
    void touchScan(ChunkPos p) { m_scanTouched.insert(p); }
    void reseedWhenSettled(ChunkPos col);
    bool columnPresent(ChunkPos col) const;

public:
    // Resident chunk-Y range for a column, inclusive. There is no world ceiling, so this
    // is the vertical band around the camera, EXCEPT that it is extended upward to just
    // above the terrain surface whenever the band comes near it. That extension is what
    // makes sky light exact without a world top to descend from: the generator never puts
    // a solid voxel above kTerrainTop, so a descent starting above the surface starts in
    // open sky by construction.
    //
    // skyAtTop says whether the top of that range is open sky (15) or buried (0). When the
    // band sits well below the surface the answer is 0 and no descent is needed at all,
    // sky attenuates 1 per step, so it cannot reach more than 15 blocks below the lowest
    // lit voxel, and a whole chunk of separation is 32. That bound is what keeps digging
    // deep from having to load every chunk back up to daylight.
    void columnRange(int cx, int cz, int& y0, int& y1, bool& skyAtTop) const;

private:
    // The surface at the column's centre, and the highest block the generator can put anywhere
    // in it plus room for what grows on top. Cached: it is ten noise samples and columnRange is
    // asked constantly.
    struct ColumnHeights {
        int centre = 0;
        int top = 0;
    };
    ColumnHeights columnHeights(int cx, int cz) const;
    // Filled on demand, including from the light thread (seeding calls columnRange), so while a
    // light slice runs nothing on the main thread may call columnRange/columnHeights.
    mutable ankerl::unordered_dense::map<ChunkPos, ColumnHeights, ChunkPosHash> m_columnHeights;
    void landEmpty(ChunkPos pos, int lod, Renderer& renderer);
    std::vector<ChunkPos> m_emptyScratch; // solid air ready to land without a mesh job
    std::uint32_t m_healTick = 0;
    std::vector<ChunkPos> m_evictList; // outside the ring; may wait on in-flight jobs
    // Voxels disturbed since the last falling tick. Duplicates are allowed: the tick
    // re-checks each one against the world anyway, so a repeat is a no-op and that is far
    // cheaper than hashing every push.
    std::vector<glm::ivec3> m_fallQueue;
    std::vector<glm::ivec3> m_fallScratch;
    // Where blocks moved TO on the previous tick. A block that cannot fall only counts as
    // having landed if it is one of these; otherwise it was never in motion.
    std::vector<glm::ivec3> m_fellLast;
    std::vector<glm::ivec3> m_fellNow;
    std::vector<glm::ivec3> m_fluidQueue;
    std::vector<glm::ivec3> m_fluidScratch;

    // a detailed re-gen finished on a worker; carries its own chunk so completion doesn't
    // depend on the entry still existing (it's released if the entry went away)
    struct RegenResult {
        ChunkPos pos;
        Chunk* voxels;
    };

    // worker -> main queues
    std::mutex m_genDoneMutex;
    std::vector<ChunkPos> m_genDone;
    std::mutex m_meshDoneMutex;
    std::vector<MeshResult> m_meshDone;
    std::mutex m_regenDoneMutex;
    std::vector<RegenResult> m_regenDone;

    std::vector<ChunkPos> m_genDoneScratch;
    std::vector<RegenResult> m_regenDoneScratch;
    std::vector<ChunkPos> m_regenScratch; // coarse chunks that want detail this frame
    std::vector<MeshResult> m_uploads; // drained results awaiting budgeted GL upload

    ChunkPos m_center{INT32_MIN, 0, INT32_MIN};
    int m_lastRadius = -1;
    bool m_dirty = false;
    float m_dirtyNearest = 1e30f; // blocks; see takeDirtyNearest()
    void noteDirty(ChunkPos p);

    // cached full-chunk scan for the HUD, refreshed a few times a second
    struct Scan {
        int meshed = 0;
        int unlit = 0;
        int uniformChunks = 0;
        std::int64_t totalQuads = 0;
        std::size_t voxelBytes = 0;
        std::size_t quadBytes = 0;
    };
    mutable Scan m_scanCache;
    mutable int m_scanAge = 999;

    // visibility flood state
    std::vector<std::uint32_t> m_visBits;
    std::vector<ChunkPos> m_floodQueue;
    // The flood's lookup table: every loaded chunk within the radius, by position, so a step is an
    // array read instead of a hash lookup. Rebuilt per flood (one pass over the loaded chunks) and
    // reused, so it costs nothing after the first.
    std::vector<const Entry*> m_floodGrid;
    std::vector<std::uint8_t> m_floodSeen; // faces each chunk was entered by (bits 0-5), queued (bit 6)
    ChunkPos m_lastFloodCenter{INT32_MIN, 0, INT32_MIN};
    int m_metaCount = 0;
    bool m_visDirty = true;
    bool m_lastCaveCull = false;
    int m_jobsInFlight = 0; // main-thread view: submit increments, result-drain decrements
    int m_genInFlight = 0;  // the generation jobs among them
    int m_regenInFlight = 0; // and the detailed re-gens
    int m_pendingMesh = 0;
    int m_lateSeeds = 0; // see Stats::lateSeeds
    // the per-frame mesh scan is O(loaded chunks); only run it when something could have
    // created mesh work (gen/light/LOD change) instead of every frame. set by those events,
    // cleared once scanned. when the world is idle there is provably nothing to mesh.
    bool m_meshScanDirty = true;
    double m_uploadMsLast = 0.0;
    bool m_seedActive = false; // a column is part-seeded and will resume next frame
    PhasePeak m_phase;
    PhasePeak m_phaseSum;
    int m_updates = 0;

    std::atomic<std::uint64_t> m_genNanos{0};
    std::atomic<std::uint64_t> m_meshNanos{0};
    std::atomic<int> m_genCount{0};
    std::atomic<int> m_meshCount{0};
    std::atomic<int> m_savesInFlight{0}; // eviction saves still on a worker (see busy())
    std::atomic<int> m_saveDelayMs{0};
    void waitForSaves() const;
};

} // namespace vx
