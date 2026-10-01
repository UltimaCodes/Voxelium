#include "harness/Harness.hpp"

#include "audio/Ambience.hpp"
#include "audio/AudioEngine.hpp"
#include "audio/PlayerSounds.hpp"
#include "core/AllocTracker.hpp"
#include "core/Bmp.hpp"
#include "core/Log.hpp"
#include "core/Paths.hpp"
#include "core/Settings.hpp"
#include "gfx/BlockTextures.hpp"
#include "gfx/GpuTimer.hpp"
#include "io/ChunkSerializer.hpp"
#include "io/RegionStore.hpp"
#include "io/WorldStore.hpp"
#include "jobs/JobSystem.hpp"
#include "light/LightEngine.hpp"
#include "mesh/GreedyMesher.hpp"
#include "mesh/NaiveMesher.hpp"
#include "platform/Window.hpp"
#include "render/Camera.hpp"
#include "render/FrameRenderer.hpp"
#include "render/Lightning.hpp"
#include "render/Overlay.hpp"
#include "render/Particles.hpp"
#include "render/PostFx.hpp"
#include "render/QualityPreset.hpp"
#include "render/Renderer.hpp"
#include "render/ShadowMap.hpp"
#include "ui/DebugHud.hpp"
#include "world/Chunk.hpp"
#include "world/ChunkManager.hpp"
#include "world/Player.hpp"
#include "world/Raycast.hpp"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace vx {

// Runs at every startup: naive vs greedy on the fixed test chunk. Merged quad area per
// direction must equal the naive face count exactly — catches mesher AND palette
// storage regressions in one go, since both meshers read through Chunk::get().
bool runMesherOracle() {
    auto chunk = std::make_unique<vx::Chunk>();
    vx::fillTestTerrain(*chunk);

    std::vector<vx::ChunkVertex> naiveVerts;
    vx::MeshStats naive;
    const double naiveBest =
        benchBestMs(50, [&] { naive = vx::meshNaive(*chunk, naiveVerts); });

    auto mesher = std::make_unique<vx::GreedyMesher>();
    std::vector<vx::PackedQuad> quads;
    std::vector<vx::PackedQuad> water;
    vx::GreedyStats greedy;
    const double greedyBest =
        benchBestMs(50, [&] { greedy = mesher->mesh(*chunk, nullptr, quads, water); });

    bool ok = true;
    for (int d = 0; d < 6; ++d)
        ok = ok && greedy.areaPerDir[d] == std::int64_t(naive.facesPerDir[d]);
    if (!ok) {
        for (int d = 0; d < 6; ++d)
            VX_ERROR("oracle dir %d: greedy area %lld vs naive faces %d", d,
                     static_cast<long long>(greedy.areaPerDir[d]), naive.facesPerDir[d]);
        return false;
    }
    VX_INFO("oracle ok | naive %d tris %.3f ms | greedy %d quads %.3f ms", naive.triangles,
            naiveBest, greedy.quads, greedyBest);
    return true;
}

int runSmokeTest(RenderContext& rc, vx::ChunkManager& world, vx::JobSystem& jobs,
                 vx::LightEngine& light, vx::FlyCamera& camera, vx::DebugHud& hud,
                 vx::Window& window, const std::filesystem::path& outPath,
                 const std::filesystem::path& soundsDir) {
    world.radius = 10; // keep the smoke quick; the regen test bumps to 16 on its own
    const auto drain = [&] {
        for (int i = 0; i < 20000 && (world.busy() || !light.idle() || i == 0); ++i) {
            rc.renderer.beginFrame();
            world.update(jobs, rc.renderer, light, camera.position);
            rc.renderer.endFrame();
        }
    };

    drain();
    world.logSummary();

    // far chunks generate coarse (no caves); approaching one must re-generate it detailed.
    // needs a radius past the coarse threshold, so bump briefly then drop back.
    {
        // Counts over the RESIDENT band, not a fixed 0..world-top column: with the ceiling
        // gone (#27) only a band around the camera is loaded, so a fixed range silently
        // counts unloaded chunks as empty.
        const auto caveAir = [&](int cx, int cz) {
            int n = 0;
            int by0 = 0, by1 = 0;
            bool bsky = false;
            world.columnRange(cx, cz, by0, by1, bsky);
            for (int y = by0; y <= by1; ++y) {
                const vx::Chunk* c = world.chunkData({cx, y, cz});
                if (!c)
                    continue;
                for (int lz = 0; lz < 32; ++lz)
                    for (int lx = 0; lx < 32; ++lx) {
                        const int surf =
                            world.generator().surfaceHeight(cx * 32 + lx, cz * 32 + lz);
                        for (int ly = 0; ly < 32; ++ly)
                            if (const int wy = y * 32 + ly;
                                wy > 4 && wy < surf - 4 && c->get(lx, ly, lz) == vx::blocks::Air)
                                ++n;
                    }
            }
            return n;
        };
        const int cx = 15; // 15 chunks east -> coarse (cave-less) at radius 16
        world.radius = 16;
        drain();
        const int farAir = caveAir(cx, 0);
        // Stand ON that column's surface. y was hard-coded to 110, which with vertical
        // streaming puts the band nowhere near the ground once terrain rises above it.
        camera.position = {double(cx * 32 + 16),
                           double(world.generator().surfaceHeight(cx * 32 + 16, 16) + 3),
                           16.0}; // move next to it -> re-gen
        // a big move triggers heavy re-streaming; the detailed re-gens land over several
        // settle cycles (the coarse mesh covers meanwhile), so wait for them to finish
        int nearAir = 0;
        for (int i = 0; i < 80 && nearAir <= 100; ++i) {
            drain();
            nearAir = caveAir(cx, 0);
        }
        camera.position = {16.0, 110.0, 16.0};
        world.radius = 10;
        drain();
        const bool ok = farAir < 30 && nearAir > 100;
        VX_INFO("regen: far column cave-air %d (want ~0), detailed after approach %d (want "
                ">0) -> %s",
                farAir, nearAir, ok ? "PASS" : "FAIL");
        if (!ok)
            return 1;
    }

    // An edit made to a chunk while it is still COARSE must survive that chunk's later
    // regeneration to detailed. The regen path tries a disk load before it ever calls the
    // generator, but an edit that has not been SAVED yet (no F2, no eviction) has nothing on
    // disk to load — so the naive version of this regen just re-ran the generator over the
    // dirty chunk and threw the edit away the instant the player got close enough to see it
    // happen, which reads as "I dig a hole and the world corrupts, stone reappears". Minimal
    // repro: place a block nobody's generator will ever put there (Bricks) in a chunk while
    // it is far enough to be coarse, then walk up to it exactly as the regen group above
    // does, and check the block is still Bricks once the approach settles — not whatever the
    // generator would have put there instead.
    {
        const int cx = 15, cz = 8; // same coarse distance as the group above, different
                                   // column so the two tests cannot leave stale state for
                                   // each other
        world.radius = 16;
        drain();
        const glm::ivec3 editAt{cx * 32 + 16,
                                world.generator().surfaceHeight(cx * 32 + 16, cz * 32 + 16) + 1,
                                cz * 32 + 16};
        const bool placed = world.setBlock(rc.renderer, light, editAt, vx::blocks::Bricks);
        camera.position = {double(editAt.x), double(editAt.y + 3), double(editAt.z)};
        for (int i = 0; i < 80; ++i)
            drain();
        // Check HERE, while the chunk this edit lives in is still resident. Moving the
        // camera back to spawn afterwards (as the group above does) evicts it — correctly,
        // since it is now far outside a radius-10 world — and an absent chunk legitimately
        // answers blockAtWorld with Stone by the same convention meshing uses for unloaded
        // neighbours. Checking after that eviction would be testing THAT convention, not
        // this fix, and very nearly was: it read back as Stone and looked exactly like the
        // bug this test exists to catch until a per-iteration trace showed the edit was
        // still Bricks all the way through the approach and only vanished once the chunk
        // itself did.
        const vx::BlockId survived = world.blockAtWorld(editAt);
        camera.position = {16.0, 110.0, 16.0};
        world.radius = 10;
        drain();
        const bool editOk = placed && survived == vx::blocks::Bricks;
        VX_INFO("regen: edit to a coarse chunk placed=%d, block after approach %d (want %d, "
                "Bricks) -> %s",
                placed ? 1 : 0, int(survived), int(vx::blocks::Bricks),
                editOk ? "PASS" : "FAIL");
        if (!editOk)
            return 1;
    }

    // Moving vertically must not strand the chunk that enters the bottom of the band.
    // The load range is center.y +- kVerticalRadius while eviction only fires past
    // kVerticalRadius + 1, so descending one chunk grows every resident column WITHOUT
    // evicting anything from it. The column therefore stays in m_seededColumns, its new
    // chunk is never offered for light, and meshing waits on light — so it is never meshed
    // and never drawn. Measured at 550 stranded chunks, one per column, at --tour 32
    // stop 4, and invisible before the stats cache was made to report fresh numbers.
    {
        // Two things have to hold at once or this proves nothing, and both were learned the
        // hard way from versions of this test that passed with the fix reverted:
        //   1. EXACTLY ONE CHUNK per step. Eviction fires at |y - center.y| >
        //      kVerticalRadius + 1, so a two-chunk move evicts from the top of the column,
        //      which drops it out of m_seededColumns and re-seeds the whole thing.
        //   2. Into territory never loaded. The band has to actually GROW; stepping back
        //      through chunks that were resident earlier creates nothing.
        // Hence a fresh column far from anywhere the smoke has been, and single steps.
        const double vxz = 2000.0;
        const double vsurf = world.generator().surfaceHeight(int(vxz), int(vxz));
        camera.position = {vxz, vsurf + 160.0, vxz};
        drain();
        world.refreshStats();
        const int before = world.stats().unlit;
        int worst = before;
        for (int step = 1; step <= 3; ++step) {
            camera.position.y -= 32.0; // one chunk, straight down, no horizontal movement
            drain();
            world.refreshStats();
            worst = std::max(worst, world.stats().unlit);
        }
        // lateSeeds > 0 is part of the assertion, not colour: it is what proves the descent
        // actually put chunks into already-seeded columns. Two earlier versions of this
        // test moved too far, or through chunks that were already resident, and passed
        // with the fix reverted because that path was never reached.
        const int late = world.stats().lateSeeds;
        const bool vertOk = before == 0 && worst == 0 && late > 0;
        VX_INFO("vertical: unlit settled %d, worst over 3 one-chunk descents %d, %d chunks "
                "landed in already-seeded columns (want >0) -> %s",
                before, worst, late, vertOk ? "PASS" : "FAIL");
        if (!vertOk)
            return 1;

        // Climbing back must not relight the world. Two chunks up from here the band trims a
        // chunk off the bottom of every column at once, and that used to cost every column its
        // seed: the self-heal then relit all of them (about 4000 at radius 32), and until that
        // was through nothing new could light or mesh - walk over a hill and the world stopped
        // rendering a few chunks out (owner report). A trim is not a column leaving the ring.
        int worstBacklog = 0;
        for (int step = 1; step <= 2; ++step) {
            camera.position.y += 32.0;
            // long enough for several self-heal passes (every 8th scan), however busy() reads
            for (int f = 0; f < 4000 && (f < 64 || world.busy() || !light.idle()); ++f) {
                rc.renderer.beginFrame();
                world.update(jobs, rc.renderer, light, camera.position);
                rc.renderer.endFrame();
                worstBacklog = std::max(worstBacklog, world.seedBacklog());
            }
        }
        world.refreshStats();
        const int climbUnlit = world.stats().unlit;
        const int columns = (2 * world.radius + 5) * (2 * world.radius + 5);
        const bool climbOk = worstBacklog * 10 < columns && climbUnlit == 0;
        VX_INFO("vertical: two one-chunk climbs back, worst seed backlog %d of %d columns "
                "(want under a tenth), unlit after %d -> %s",
                worstBacklog, columns, climbUnlit, climbOk ? "PASS" : "FAIL");
        if (!climbOk)
            return 1;
        camera.position = {16.0, 110.0, 16.0};
        drain();
    }

    // Live radius change (owner report #57: "streaming breaks when render distance
    // changes"). Not camera movement — a change to `radius` itself. lodForDist2()'s bands
    // SCALE WITH radius, so every resident chunk's desired LOD shifts, not just the
    // streaming ring at the edge; rescan() is supposed to catch this (its own comment:
    // "moved: LOD bands shifted, new chunks incoming") by marking the mesh scan dirty, which
    // re-evaluates every resident chunk fresh against the new radius next pass. Reading the
    // code never turned up a bug in that path — this checks it directly instead of trusting
    // the reasoning, per SPEC #57's own "not directly reproduced".
    {
        const double rxz = -3000.0;
        const double rsurf = world.generator().surfaceHeight(int(rxz), int(rxz));
        camera.position = {rxz, rsurf + 3.0, rxz};
        world.radius = 10;
        drain();
        world.radius = 20; // grow: e.g. a Low -> Ultra preset switch mid-game
        drain();
        const auto grown = world.auditLod();
        const bool growOk =
            grown.stranded == 0 && grown.lodMismatch == 0 && grown.coarseInDetailBand == 0;
        VX_INFO("radius change: grew 10 -> 20, stranded %d, LOD mismatch %d, coarse-in-detail "
                "%d (want 0/0/0) -> %s",
                grown.stranded, grown.lodMismatch, grown.coarseInDetailBand,
                growOk ? "PASS" : "FAIL");
        if (!growOk)
            return 1;
        world.radius = 8; // shrink past the original radius: stresses eviction hardest
        drain();
        const auto shrunk = world.auditLod();
        const bool shrinkOk = shrunk.stranded == 0 && shrunk.lodMismatch == 0 &&
                              shrunk.coarseInDetailBand == 0;
        VX_INFO("radius change: shrank 20 -> 8, stranded %d, LOD mismatch %d, coarse-in-detail "
                "%d (want 0/0/0) -> %s",
                shrunk.stranded, shrunk.lodMismatch, shrunk.coarseInDetailBand,
                shrinkOk ? "PASS" : "FAIL");
        if (!shrinkOk)
            return 1;
        camera.position = {16.0, 110.0, 16.0};
        world.radius = 10;
        drain();
    }

    // Owner report, live: "nothing generates in the area i spawn in and i have to fly to
    // the nearest chunk." m_seedQueue used to be consumed pure LIFO (pop_back, whatever
    // pushed most recently) — under real load a coarse (no-cave) far column generates
    // roughly 4x faster than a detailed near one, so far columns can keep completing and
    // cutting in line ahead of a slower near column already waiting its turn, including the
    // one the CAMERA is actually standing in. A real loading screen has a finite budget
    // (main.cpp's settleBehindLoadingScreen caps at 4000 frames) and drops the player in
    // once it runs out regardless of what is still queued — so the question is not "does the
    // center eventually get lit" (given enough frames, everything does, LIFO or not, which
    // is why an earlier version of this test that waited for the WHOLE gen queue to drain
    // passed even with the fix deliberately disabled: by then the backlog had long since
    // washed out). It is "how much OTHER work happens before the center's turn comes,"
    // since that is what determines whether a bounded settle reaches it in time. Measured
    // as a fraction of total expected meshing: the center is distance 0, dispatched
    // essentially first by m_genQueue's own nearest-first order, so a fair queue lights it
    // within a small fraction of total progress — not after most of a wide radius already
    // finished.
    {
        // Radius set to 24 BEFORE the first update() at this spot, and the spot itself is
        // fresh (never settled at any radius before) — center must start genuinely
        // unresident, or its first update() here finds it already lit from an earlier phase
        // and the whole measurement is trivially "instant" regardless of queue fairness.
        // That is exactly what the first version of this test got wrong: it grew radius
        // in place at an already-settled position, so the center was already lit before the
        // interesting work even began, and passed even with the fix deliberately disabled.
        world.radius = 24; // a wide coarse ring around a much smaller detailed core
        const double sxz = 5000.0, szz = -9000.0;
        const double ssurf = world.generator().surfaceHeight(int(sxz), int(szz));
        camera.position = {sxz, ssurf + 3.0, szz};
        const ChunkPos center{int(std::floor(sxz / double(kChunkSize))),
                              int(std::floor((ssurf + 3.0) / double(kChunkSize))),
                              int(std::floor(szz / double(kChunkSize)))};
        const int target = std::max(1, world.expectedMeshedChunks());
        int frames = 0;
        double progressAtCenterLit = -1.0;
        // 20000: drain()'s own established ceiling elsewhere in this file, not a tuned
        // number — this loop breaks the moment the center lights up regardless (under 1000
        // frames on Release/harden), so raising it costs fast builds nothing. It matters for
        // Debug specifically: at this radius, Debug's own unoptimized cave-generation cost
        // (measured elsewhere this session at up to ~70ms/chunk, vs ~2-15ms release-side)
        // means the center's OWN voxels can still be mid-generation well past 5000 main-
        // thread iterations, which is a real speed difference, not a queue-ordering one —
        // the assertion below is about progress FRACTION at the moment it lights, not about
        // how many iterations that took to reach.
        for (; frames < 20000; ++frames) {
            rc.renderer.beginFrame();
            world.update(jobs, rc.renderer, light, camera.position);
            rc.renderer.endFrame();
            if (world.chunkMeshedAndLit(center)) {
                world.refreshStats();
                progressAtCenterLit = double(world.stats().meshed) / double(target);
                break;
            }
        }
        const bool centerLit = progressAtCenterLit >= 0.0;
        // 15%: generous slack over "essentially first" — this is about catching gross
        // starvation (the center landing at 50-90%+ of total progress), not chasing an exact
        // number.
        const bool prompt = centerLit && progressAtCenterLit < 0.15;
        VX_INFO("spawn priority: center column lit+meshed after frame %d, at %.0f%% of total "
                "expected progress (want lit, and < 15%%) -> %s",
                frames, 100.0 * std::max(progressAtCenterLit, 0.0),
                (centerLit && prompt) ? "PASS" : "FAIL");
        if (!(centerLit && prompt))
            return 1;
        camera.position = {16.0, 110.0, 16.0};
        world.radius = 10;
        drain();
    }

    // Switching worlds is what "new world" and "load world" both do, and it swaps the seed,
    // the save directory and every resident chunk at once. Assert it actually lands on
    // different terrain and comes back fully lit, because a half-swapped world would look
    // like a streaming bug somewhere else entirely.
    {
        const int beforeSeed = world.seed();
        const int beforeSurf = world.generator().surfaceHeight(16, 16);
        const auto altDir = outPath.parent_path() / "vxsave_altworld";
        std::error_code ec;
        std::filesystem::remove_all(altDir, ec);
        world.loadWorld(rc.renderer, beforeSeed + 12345, altDir);
        camera.position = {16.0, double(world.generator().surfaceHeight(16, 16) + 8), 16.0};
        drain();
        world.refreshStats();
        const vx::ChunkManager::Stats ws = world.stats();
        const int afterSurf = world.generator().surfaceHeight(16, 16);
        // Same column, different seed: the ground height moving is the cheapest proof the
        // generator really was re-seeded rather than just re-run.
        const bool swapped = world.seed() != beforeSeed && afterSurf != beforeSurf;
        const bool healthy = ws.meshed > 0 && ws.unlit == 0;
        VX_INFO("world switch: seed %d -> %d, surface %d -> %d, %d meshed, %d unlit -> %s",
                beforeSeed, world.seed(), beforeSurf, afterSurf, ws.meshed, ws.unlit,
                (swapped && healthy) ? "PASS" : "FAIL");
        if (!swapped || !healthy)
            return 1;
        // back to the original world; everything after this point expects that terrain
        world.loadWorld(rc.renderer, beforeSeed, outPath.parent_path() / "vxsave_smoke");
        camera.position = {16.0, 110.0, 16.0};
        drain();
    }

    // An edited chunk is saved on a worker when it is evicted, and that save was invisible to
    // busy(): switching worlds right after it could flush and repoint the store before it
    // landed, so the edit never reached its own world and arrived in the next one instead
    // (SPEC #142). Shrinking the radius evicts the edit with nothing else left to stream, and
    // the save is held open long enough that the window cannot be missed.
    {
        const int homeSeed = world.seed();
        const auto homeDir = outPath.parent_path() / "vxsave_smoke";
        const auto otherDir = outPath.parent_path() / "vxsave_raceworld";
        std::error_code ec;
        std::filesystem::remove_all(otherDir, ec);
        const int ex = 16 + 8 * 32, ez = 16;
        const glm::ivec3 at{ex, world.generator().surfaceHeight(ex, ez) + 1, ez};
        const vx::ChunkPos cp{at.x >> 5, at.y >> 5, at.z >> 5};
        world.setBlock(rc.renderer, light, at, vx::blocks::Bricks);
        drain();
        world.setSaveDelayForTest(400);
        world.radius = 3; // the edit sits 8 chunks out, past 3 + the eviction margin
        drain();
        world.loadWorld(rc.renderer, homeSeed + 777, otherDir);
        world.setSaveDelayForTest(0);
        std::this_thread::sleep_for(std::chrono::milliseconds(600)); // a stray save lands now
        auto probe = std::make_unique<vx::Chunk>();
        const bool inOther = world.regions().load(cp, *probe);
        world.loadWorld(rc.renderer, homeSeed, homeDir);
        world.radius = 10;
        camera.position = {16.0, 110.0, 16.0};
        drain();
        const bool inHome = world.regions().load(cp, *probe) &&
                            probe->get(at.x & 31, at.y & 31, at.z & 31) == vx::blocks::Bricks;
        std::filesystem::remove_all(otherDir, ec);
        const bool ok = inHome && !inOther;
        VX_INFO("persistence: edit evicted just before a world switch - in its own world %d "
                "(want 1), in the next world %d (want 0) -> %s",
                int(inHome), int(inOther), ok ? "PASS" : "FAIL");
        if (!ok)
            return 1;
    }

    // lighting checks against the enclosed glow room in the terrain
    {
        const int skyInRoom = vx::light::sky(light.lightAt({16, 48, 16}));
        const int glowAbove = vx::light::block(light.lightAt({16, 46, 16}));
        const int glowEdge = vx::light::block(light.lightAt({16, 48, 19}));
        // Open air just above the ground — always full sky. This used to probe near the
        // top of the world; there is no top any more (#27), and the resident vertical band
        // does not reach y=380, so that sampled an unloaded chunk and read 0.
        // Open air above the ground — always full sky. Two things this has to clear:
        // there is no world top any more (#27) so it must sit inside the resident vertical
        // band, and it must be above the TREES, which now reach ~20 blocks (a pine trunk is
        // up to 14 plus its tiers). +6 landed inside a canopy and read 0, correctly.
        const int skyProbeY = world.generator().surfaceHeight(16, 16) + 34;
        const int skySurface = vx::light::sky(light.lightAt({16, skyProbeY, 16}));
        bool ok = skyInRoom == 0 && glowAbove == 13 && glowEdge > 0 && glowEdge < 13 &&
                  skySurface == 15;
        VX_INFO("light: room sky %d (want 0), above-glow %d (want 13), edge %d, surface sky "
                "%d (want 15) -> %s",
                skyInRoom, glowAbove, glowEdge, skySurface, ok ? "PASS" : "FAIL");
        if (!ok)
            return 1;

        // M18: the three block-light channels propagate independently, so a warm emitter
        // must still be warm out at its falloff - red reaching further than blue. A
        // monochrome engine would report all three equal here.
        {
            const std::uint32_t v = light.lightAt({16, 46, 16});
            const int r = vx::light::red(v), g = vx::light::green(v), b = vx::light::blue(v);
            const std::uint32_t far = light.lightAt({16, 48, 19});
            const int fr = vx::light::red(far), fb = vx::light::blue(far);
            const bool tinted = r > g && g > b && fr > fb;
            VX_INFO("light: glow colour rgb %d/%d/%d, at falloff r%d b%d -> %s", r, g, b, fr,
                    fb, tinted ? "PASS" : "FAIL");
            if (!tinted)
                return 1;
        }

        // M19b: a coloured lamp must carry ITS hue, not the warm one baked in for Glow.
        // Blue and green are checked together because the failure this catches is a lamp
        // whose emit row was filled in for the wrong channel, which a single colour hides.
        {
            const int lx = 40, lz = 40;
            const int ly = world.generator().surfaceHeight(lx, lz) + 3;
            world.setBlock(rc.renderer, light, {lx, ly, lz}, vx::BlockId(vx::blocks::LampFirst + 8));
            const std::uint32_t bv = light.lightAt({lx, ly + 1, lz});
            const int br = vx::light::red(bv), bb = vx::light::blue(bv);
            world.setBlock(rc.renderer, light, {lx, ly, lz}, vx::BlockId(vx::blocks::LampFirst + 4));
            const std::uint32_t gv = light.lightAt({lx, ly + 1, lz});
            const int gg = vx::light::green(gv), gbl = vx::light::blue(gv);
            // White is its own swatch, and it is the one that used to be unreachable: a
            // wheel of saturated hues cannot express it, so picking white gave you whatever
            // vivid colour happened to be nearest. All three channels must come out equal.
            world.setBlock(rc.renderer, light, {lx, ly, lz},
                           vx::BlockId(vx::blocks::LampFirst + vx::blocks::kLampCount - 1));
            const std::uint32_t wv = light.lightAt({lx, ly + 1, lz});
            const int wr = vx::light::red(wv), wg = vx::light::green(wv),
                      wb = vx::light::blue(wv);
            const bool whiteOk = wr >= 13 && wr == wg && wg == wb;
            // and removing it has to take the colour with it, through the removal BFS
            world.setBlock(rc.renderer, light, {lx, ly, lz}, vx::blocks::Air);
            const std::uint32_t off = light.lightAt({lx, ly + 1, lz});
            const int rest = vx::light::red(off) + vx::light::green(off) + vx::light::blue(off);
            const bool lampOk =
                bb > br && bb >= 13 && gg > gbl && gg >= 13 && whiteOk && rest == 0;
            VX_INFO("light: lamp blue r%d b%d, green g%d b%d, white %d/%d/%d, after removal "
                    "%d (want 0) -> %s",
                    br, bb, gg, gbl, wr, wg, wb, rest, lampOk ? "PASS" : "FAIL");
            if (!lampOk)
                return 1;
        }

        vx::Chunk* roomChunk = world.chunkDataMutable({0, 1, 0});
        roomChunk->set(16, 13, 16, vx::blocks::Air);
        light.onBlockChanged({16, 45, 16}, vx::blocks::Glow, vx::blocks::Air);
        while (!light.idle())
            light.process(10.0);
        const int afterRemove = vx::light::block(light.lightAt({16, 46, 16}));
        roomChunk->set(16, 13, 16, vx::blocks::Glow);
        light.onBlockChanged({16, 45, 16}, vx::blocks::Air, vx::blocks::Glow);
        while (!light.idle())
            light.process(10.0);
        const int afterRestore = vx::light::block(light.lightAt({16, 46, 16}));
        ok = afterRemove == 0 && afterRestore == 13;
        VX_INFO("light: removal %d (want 0), re-add %d (want 13) -> %s", afterRemove,
                afterRestore, ok ? "PASS" : "FAIL");
        if (!ok)
            return 1;
        drain();
    }

    // hitch check: teleport 16 chunks east and re-stream the world
    camera.position.x += 512.0;
    world.resetPhasePeak();
    const auto restreamStart = std::chrono::steady_clock::now();
    const std::uint64_t allocsBefore = vx::heapAllocCount();
    const std::uint64_t forcedBefore = rc.shadows.forcedCount();
    light.resetSeedPhaseNanos();
    double worstMs = 0.0;
    int frames = 0;
    for (; frames < 20000; ++frames) {
        rc.renderer.beginFrame();
        const auto t0 = std::chrono::steady_clock::now();
        world.update(jobs, rc.renderer, light, camera.position);
        worstMs = std::max(
            worstMs,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count());
        // Mirrors main.cpp's real per-frame call exactly (consumeDirty-gated invalidate),
        // which the rest of this loop otherwise never exercises — world.update() alone
        // never touches ShadowMap. Without this line the throttle below would be measuring
        // nothing: consumeDirty() is what actually FIRES during a burst; forcedCount() is
        // what proves the throttle catches it.
        if (world.consumeDirty())
            rc.shadows.invalidate();
        rc.renderer.endFrame();
        if (frames > 0 && !world.busy())
            break;
    }
    // Wall clock, not update count: the loop spins as fast as it can, so a cheaper main
    // thread just means MORE iterations. Only elapsed time says whether streaming actually
    // finished sooner.
    const double restreamMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                  restreamStart)
            .count();
    // Every job used to cost one malloc on submit (JobSystem::Job now stores the closure
    // inline instead of relying on std::function's own heap-allocated storage), AND the job
    // queue itself used to cost roughly one more every few submits regardless of what it
    // held (std::deque grows in fixed-size blocks sized for a small element — Job at its
    // real size meant a new block every third push; measured 250 allocs/1000 push+pop
    // cycles at 128 bytes vs 62 at std::function's own 32, confirmed via a queue holding
    // plain byte blobs so element size was the only variable). Both are fixed and can regress
    // back silently, which is what this number is for.
    //
    // What is left is NOT a queue problem: the return leg below re-streams the SAME radius
    // through already-explored, already-cached ground and costs almost exactly the same
    // per update as this fresh leg (see its own log line) — if the remainder were queue or
    // cache-miss churn, revisiting known ground would be far cheaper. It is not, because a
    // mesh is never disk-cached, only the voxels are: MeshResult copies each job's quad and
    // water buffers into fresh, long-lived storage (ChunkManager.cpp, the upload-drain loop)
    // because that CPU copy has to outlive the job and survive for the chunk's whole
    // residency (arena rebuilds read it back). That is once-per-meshed-chunk, not per-frame,
    // and it is genuine work, not waste — fixing it for real means a pooled/reusable buffer
    // allocator for mesh output, which is a bigger and riskier change than this pass, so it
    // is not attempted here. No hard gate on this number for that reason: it is diagnostic,
    // not a pass/fail, the same as the phase and arena lines around it.
    const std::uint64_t allocsDuring = vx::heapAllocCount() - allocsBefore;
    VX_INFO("stress: re-streamed after teleport in %.0f ms / %d updates, worst main-thread "
            "cost %.2f ms, %llu heap alloc(s) (%.2f/update)",
            restreamMs, frames, worstMs, (unsigned long long)allocsDuring,
            double(allocsDuring) / double(std::max(frames, 1)));
    // ShadowMap::invalidate() used to force a full 3-cascade redraw on every one of those
    // `frames` updates — cascade caching (shadow -89%, Tier 1's biggest single win) provided
    // ~zero benefit for as long as a burst like this one lasted, because "geometry changed
    // somewhere" fired on nearly every update. The throttle inside invalidate() means the
    // forced count should track wall-clock time (one per ~100ms), not update count — this is
    // the one number in this whole group that is actually asserted, because "far fewer
    // forced redraws than updates, during a burst that mirrors main.cpp's real call site" is
    // exactly the claim #66 makes and it is cheap to keep honest.
    const std::uint64_t forcedDuring = rc.shadows.forcedCount() - forcedBefore;
    const bool throttleOk = frames < 50 || forcedDuring < std::uint64_t(frames) / 4;
    VX_INFO("stress: shadow invalidate throttle — %llu forced redraw(s) over %d updates "
            "(want << updates) -> %s",
            (unsigned long long)forcedDuring, frames, throttleOk ? "PASS" : "FAIL");
    if (!throttleOk)
        return 1;
    {
        const auto pk = world.phasePeak();
        VX_INFO("stress: worst phase ms — rescan %.2f, evict %.2f, seed %.2f, propagate "
                "%.2f, scan %.2f, upload %.2f",
                pk.rescanMs, pk.evictMs, pk.seedMs, pk.propagateMs, pk.scanMs, pk.uploadMs);
        // The peaks above are one bad frame each. These are the whole re-stream, which is
        // what tells you which phase to actually go after.
        const auto tot = world.phaseTotal();
        const double n = std::max(1, world.updateCount());
        const double wall = tot.rescanMs + tot.evictMs + tot.seedMs + tot.propagateMs +
                            tot.scanMs + tot.uploadMs;
        VX_INFO("stress: total phase ms over %d updates — rescan %.1f, evict %.1f, seed "
                "%.1f, propagate %.1f, scan %.1f, upload %.1f (sum %.1f)",
                world.updateCount(), tot.rescanMs, tot.evictMs, tot.seedMs, tot.propagateMs,
                tot.scanMs, tot.uploadMs, wall);
        VX_INFO("stress: light = %.1f ms of %.1f ms main-thread (%.0f%%), %.2f ms/update avg",
                tot.seedMs + tot.propagateMs, wall,
                100.0 * (tot.seedMs + tot.propagateMs) / std::max(wall, 1e-6),
                (tot.seedMs + tot.propagateMs) / n);
        // SPEC #3: is column-seeding's cost actually in descent (the docs' "5-10ms/column,
        // needs an analytical fast path for coarse columns" claim), or somewhere the existing
        // uniform-air bulk-skip + stop-at-first-blocker optimizations already made cheap?
        // Spread is where a CAVE column's cost lives (every mouth/overhang is a boundary case
        // pure heightmap terrain never hits) — if that dominates, an analytical sky-light
        // shortcut for coarse columns targets the wrong phase, since detailed/cave columns
        // can't use it at all.
        const auto sp = light.seedPhaseNanos();
        const double descentMs = double(sp.descent) / 1e6, spreadMs = double(sp.spread) / 1e6,
                     emissiveMs = double(sp.emissive) / 1e6;
        VX_INFO("stress: seed phase ms — descent %.1f, spread %.1f, emissive %.1f (%.0f%% / "
                "%.0f%% / %.0f%% of seed total)",
                descentMs, spreadMs, emissiveMs,
                100.0 * descentMs / std::max(descentMs + spreadMs + emissiveMs, 1e-6),
                100.0 * spreadMs / std::max(descentMs + spreadMs + emissiveMs, 1e-6),
                100.0 * emissiveMs / std::max(descentMs + spreadMs + emissiveMs, 1e-6));
        // Worker-side cost and arena health after the worst churn the engine ever sees.
        // The arena line is the reason there is no defrag scheduler: if the biggest free
        // block still holds nearly all the free space after this, there is nothing to
        // compact and a defrag pass would be pure code.
        const vx::ChunkManager::Stats ws = world.stats();
        const vx::Renderer::ArenaStats as = rc.renderer.arenaStats();
        VX_INFO("stress: worker gen %.3f ms/chunk, mesh %.3f ms/chunk", ws.genMsAvg,
                ws.meshMsAvg);
        VX_INFO("stress: arena %u/%u quads, %d free blocks, largest holds %.0f%% of free space",
                as.usedQuads, as.capacityQuads, as.freeBlocks,
                as.capacityQuads > as.usedQuads
                    ? 100.0 * as.largestFreeBlock / double(as.capacityQuads - as.usedQuads)
                    : 100.0);
    }
    // Return leg. The outbound teleport lands on ground this run has never seen, so it can
    // only pay the cost of caching evicted chunks; coming back is where a cache can be hit.
    // Timing only the outbound half would report the bill and hide the refund.
    camera.position.x -= 512.0;
    {
        const auto backStart = std::chrono::steady_clock::now();
        const std::uint64_t backAllocsBefore = vx::heapAllocCount();
        int backFrames = 0;
        for (; backFrames < 20000; ++backFrames) {
            rc.renderer.beginFrame();
            world.update(jobs, rc.renderer, light, camera.position);
            rc.renderer.endFrame();
            if (backFrames > 0 && !world.busy())
                break;
        }
        // If the outbound leg's allocation rate is genuinely proportional to NEW content
        // (fresh per-chunk mesh/gen output, which has to be allocated somewhere no matter
        // how the job queue itself is built) rather than a queue-container leak, revisiting
        // already-cached ground should cost far less per update — most of what streams back
        // in here was resident a moment ago and takes the disk-cache path, not fresh
        // generation. Not a hard PASS/FAIL: this is diagnostic, telling the outbound number
        // apart from "still broken" vs "that's just what new content costs."
        const std::uint64_t backAllocs = vx::heapAllocCount() - backAllocsBefore;
        VX_INFO("stress: returned to origin in %.0f ms / %d updates, %llu heap alloc(s) "
                "(%.2f/update vs %.2f/update outbound)",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                          backStart)
                    .count(),
                backFrames, (unsigned long long)backAllocs,
                double(backAllocs) / double(std::max(backFrames, 1)),
                double(allocsDuring) / double(std::max(frames, 1)));
    }
    drain();

    int w = 0;
    int h = 0;
    window.framebufferSize(w, h);

    GLuint color = 0;
    glCreateRenderbuffers(1, &color);
    glNamedRenderbufferStorage(color, GL_RGB8, w, h);
    GLuint fbo = 0;
    glCreateFramebuffers(1, &fbo);
    glNamedFramebufferRenderbuffer(fbo, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color);
    if (glCheckNamedFramebufferStatus(fbo, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        VX_ERROR("smoke: framebuffer incomplete");
        return 1;
    }

    std::vector<vx::ChunkDrawInfo> drawInfo;
    std::vector<vx::ChunkDrawInfo> waterInfo;
    world.collectDrawInfo(drawInfo, waterInfo);
    rc.renderer.updateChunkMeta(drawInfo);
    rc.renderer.updateWaterMeta(waterInfo);
    world.updateVisibilityIfNeeded(rc.renderer, camera.position, true);

    vx::FxToggles fx;
    // Eye adaptation carries state between frames, so the image would depend on how many
    // frames ran before the readback. Off here for the same reason TAA is.
    fx.autoExposure = false;
    const vx::ChunkManager::Stats ws = world.stats();
    const vx::Renderer::ArenaStats as = rc.renderer.arenaStats();
    for (int i = 0; i < 5; ++i) {
        window.pollEvents();
        rc.renderer.beginFrame();
        hud.beginFrame();
        renderFrame(rc, camera, fx, true, w, h, fbo, 0.6f, nullptr);
        const vx::FrameStats stats{.frameMs = 1.0f,
                                   .gpuMs = 0.5f,
                                   .heapAllocs = 0,
                                   .drawCalls = 1,
                                   .quads = int(ws.totalQuads),
                                   .triangles = int(ws.totalQuads) * 2,
                                   .chunksLoaded = ws.loaded,
                                   .chunksMeshed = ws.meshed,
                                   .genQueue = ws.genQueue,
                                   .jobsInFlight = ws.jobsInFlight,
                                   .uploadsPending = ws.pendingUploads,
                                   .uploadMs = float(ws.uploadMsLast),
                                   .voxelBytes = ws.voxelBytes,
                                   .quadBytes = ws.quadBytes,
                                   .gpuChunksDrawn = rc.renderer.gpuVisibleChunks(),
                                   .arenaUsedQuads = as.usedQuads,
                                   .arenaCapQuads = as.capacityQuads,
                                   .arenaFreeBlocks = as.freeBlocks};
        hud.diagnostics(stats, camera);
        hud.render();
        rc.renderer.endFrame();
    }
    VX_INFO("smoke: gpu culled %d of %d chunks (1 draw call, cave culling on)",
            rc.renderer.debugReadDrawCount(), int(drawInfo.size()));

    std::vector<std::uint8_t> pixels(std::size_t(w) * std::size_t(h) * 3u);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glNamedFramebufferReadBuffer(fbo, GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, pixels.data());

    // post-processing owns the sky color now, so compare against an actual sky pixel
    // (top-left corner; the camera pitches down so the top rows are always sky)
    const std::size_t skyIdx = (std::size_t(h - 3) * std::size_t(w) + 4) * 3;
    std::size_t drawn = 0;
    for (std::size_t i = 0; i < pixels.size(); i += 3) {
        if (std::abs(int(pixels[i + 0]) - int(pixels[skyIdx + 0])) > 10 ||
            std::abs(int(pixels[i + 1]) - int(pixels[skyIdx + 1])) > 10 ||
            std::abs(int(pixels[i + 2]) - int(pixels[skyIdx + 2])) > 10)
            ++drawn;
    }

    if (!vx::writeBmp24(outPath, w, h, pixels))
        return 1;

    VX_INFO("smoke: %llu of %d pixels differ from the sky (%.1f%%), wrote %s",
            static_cast<unsigned long long>(drawn), w * h,
            100.0 * double(drawn) / (double(w) * double(h)), outPath.string().c_str());
    if (drawn < 1000) {
        VX_ERROR("smoke: nothing rendered");
        return 1;
    }

    // physics: drop a player from the sky onto the surface, fixed deterministic ticks
    {
        vx::Player testPlayer;
        testPlayer.position = {16.0, 125.0, 16.0};
        for (int i = 0; i < 1200; ++i)
            testPlayer.step(world, {});
        // tree-agnostic: it must come to rest on the ground (or a tree/leaf) — on solid
        // footing, not still falling
        const glm::ivec3 below{16, int(std::floor(testPlayer.position.y)) - 1, 16};
        const bool ok = testPlayer.onGround && vx::blocks::solid(world.blockAtWorld(below));
        VX_INFO("physics: landed at y=%.4f resting on block %d -> %s", testPlayer.position.y,
                int(world.blockAtWorld(below)), ok ? "PASS" : "FAIL");
        if (!ok)
            return 1;

        // ores: depth-banded, so coal must outnumber iron and both must actually exist.
        // counts a column of stone under spawn rather than trusting that it compiled.
        {
            int coal = 0, iron = 0, stone = 0;
            for (int x = 0; x < 48; ++x)
                for (int z = 0; z < 48; ++z) {
                    const int surf = world.generator().surfaceHeight(x, z);
                    for (int y = 4; y < surf - 4; ++y) {
                        const vx::BlockId b = world.blockAtWorld({x, y, z});
                        if (b == vx::blocks::CoalOre) ++coal;
                        else if (b == vx::blocks::IronOre) ++iron;
                        else if (b == vx::blocks::Stone) ++stone;
                    }
                }
            const bool oreOk = coal > 0 && iron > 0 && coal > iron && stone > coal * 4;
            VX_INFO("ores: %d coal, %d iron in %d stone -> %s", coal, iron, stone,
                    oreOk ? "PASS" : "FAIL");
            if (!oreOk)
                return 1;
        }

        // M17: a source on flat ground must spread outward with decaying depth and stop,
        // not flood forever. cellular, so ticking it repeatedly has to converge.
        {
            // Build a sealed stone basin first. On real terrain a source just runs
            // downhill forever, which is correct behaviour but tests nothing about
            // convergence — a flat floor with a rim is what has a settled state to reach.
            const int sx = 60, sz = 60;
            const int sy = world.generator().surfaceHeight(sx, sz) + 6;
            // #27 removed the world ceiling, so a column is no longer loaded end to end —
            // only a band around the camera is. An edit outside that band lands nowhere,
            // which is exactly what a player would see, so stand at the basin first.
            camera.position = {double(sx), double(sy) + 2.0, double(sz)};
            drain();
            std::vector<vx::ChunkManager::BlockEdit> basin;
            for (int dx = -6; dx <= 6; ++dx)
                for (int dz = -6; dz <= 6; ++dz) {
                    basin.push_back({{sx + dx, sy - 1, sz + dz}, vx::blocks::Stone});
                    const bool rim = std::abs(dx) == 6 || std::abs(dz) == 6;
                    for (int dy = 0; dy < 3; ++dy)
                        basin.push_back({{sx + dx, sy + dy, sz + dz},
                                         rim ? vx::blocks::Stone : vx::blocks::Air});
                }
            world.setBlocks(rc.renderer, light, basin);
            world.setBlock(rc.renderer, light, {sx, sy, sz}, vx::blocks::Water);
            int changed = 0;
            for (int t = 0; t < 12; ++t)
                changed = world.stepFluid(rc.renderer, light, {double(sx), double(sy), double(sz)});
            int spread = 0, deepest = 0;
            for (int dx = -5; dx <= 5; ++dx)
                for (int dz = -5; dz <= 5; ++dz) {
                    const int l = vx::blocks::waterLevel(world.blockAtWorld({sx + dx, sy, sz + dz}));
                    if (l > 0 && (dx || dz)) {
                        ++spread;
                        deepest = std::max(deepest, l);
                    }
                }
            // spread happened, decayed below source depth, settled (a further tick is a no-op)
            const bool flowOk = spread > 4 && deepest < 8 && changed == 0;
            VX_INFO("fluid: source spread to %d cells, deepest %d, settled=%d -> %s", spread,
                    deepest, changed == 0 ? 1 : 0, flowOk ? "PASS" : "FAIL");
            if (!flowOk)
                return 1;
        }

        // M20: a floating column of sand must come down and stack, one layer per tick, and
        // then stop. The per-tick part matters as much as the resting place — reading the
        // state mid-tick instead of at the start of it would teleport the whole column.
        {
            const int sx = 90, sz = 90;
            const int floorY = world.generator().surfaceHeight(sx, sz) + 4;
            std::vector<vx::ChunkManager::BlockEdit> setup;
            setup.push_back({{sx, floorY, sz}, vx::blocks::Stone});
            for (int i = 1; i <= 3; ++i) // clear the shaft so nothing else is in the way
                for (int dy = 0; dy < 12; ++dy)
                    setup.push_back({{sx, floorY + dy + 1, sz}, vx::blocks::Air});
            for (int i = 0; i < 3; ++i) // three sand blocks hanging with a gap under them
                setup.push_back({{sx, floorY + 5 + i, sz}, vx::blocks::Sand});
            world.setBlocks(rc.renderer, light, setup);

            world.stepFalling(rc.renderer, light);
            // One tick moves the LOWEST block of the column only. Checked on the column
            // itself rather than on the returned edit count: the scan box is 49 blocks
            // across and catches natural sand and gravel elsewhere in the world, so the
            // count is not ours to predict.
            const bool stepOk = world.blockAtWorld({sx, floorY + 4, sz}) == vx::blocks::Sand &&
                                world.blockAtWorld({sx, floorY + 5, sz}) == vx::blocks::Air &&
                                world.blockAtWorld({sx, floorY + 6, sz}) == vx::blocks::Sand;
            int settleTick = 1;
            int landings = 0;
            for (; settleTick < 40; ++settleTick) {
                int landed = 0;
                glm::ivec3 landedAt{0};
                const int moved = world.stepFalling(rc.renderer, light, &landed, &landedAt);
                landings += landed;
                if (moved == 0)
                    break;
            }
            int stack = 0;
            for (int dy = 1; dy <= 4; ++dy)
                if (world.blockAtWorld({sx, floorY + dy, sz}) == vx::blocks::Sand)
                    ++stack;
            const bool airAbove = world.blockAtWorld({sx, floorY + 5, sz}) == vx::blocks::Air;

            // A block set down on solid ground goes through this same queue and reaches the
            // same "cannot fall" branch, but it was never in motion and must not report a
            // landing — it already made its own placement sound.
            int placedLanded = 0;
            world.setBlock(rc.renderer, light, {sx, floorY + 4, sz}, vx::blocks::Air);
            world.setBlock(rc.renderer, light, {sx + 2, floorY, sz + 2}, vx::blocks::Stone);
            world.setBlock(rc.renderer, light, {sx + 2, floorY + 1, sz + 2}, vx::blocks::Sand);
            for (int t = 0; t < 4; ++t) {
                int landed = 0;
                world.stepFalling(rc.renderer, light, &landed, nullptr);
                placedLanded += landed;
            }

            const bool fallOk = stepOk && stack == 3 && airAbove && landings > 0 &&
                                placedLanded == 0;
            VX_INFO("falling: one block/tick=%d, stacked %d of 3 on the floor, source clear=%d,"
                    " settled after %d ticks, %d landing(s) reported, %d from a block merely"
                    " placed (want 0) -> %s",
                    stepOk ? 1 : 0, stack, airAbove ? 1 : 0, settleTick, landings,
                    placedLanded, fallOk ? "PASS" : "FAIL");
            if (!fallOk)
                return 1;
        }

        // M19: particles must live, then die, and the pool must compact back to empty.
        // The swap-remove in update() is the part worth guarding: get it wrong and either
        // particles leak forever or the draw call submits stale slots.
        if (rc.particles) {
            const int before = rc.particles->liveCount();
            for (int i = 0; i < 40; ++i)
                rc.particles->spawnBlockBreak({double(i), 70.0, 0.0}, 0);
            const int spawned = rc.particles->liveCount();
            for (int i = 0; i < 200; ++i) // 200 * 16ms = 3.2s, past the longest life
                rc.particles->update(0.016f);
            const int after = rc.particles->liveCount();
            // capacity is a hard cap, never exceeded, and everything expires
            const bool partOk = before == 0 && spawned > 0 &&
                                spawned <= vx::Particles::kCapacity && after == 0;
            VX_INFO("particles: %d spawned (cap %d), %d alive after 3.2 s -> %s", spawned,
                    vx::Particles::kCapacity, after, partOk ? "PASS" : "FAIL");
            if (!partOk)
                return 1;

            // A full pool must still take a whole burst. Every spawn into a full pool used to
            // overwrite slot 0, so a block broken while lightning had filled it left one shard.
            for (int i = 0; rc.particles->liveCount() < vx::Particles::kCapacity && i < 64; ++i)
                rc.particles->spawnBolt({0.0, 440.0, 0.0}, {0.0, 70.0, 0.0}, 77u + unsigned(i), 1.0f);
            const bool full = rc.particles->liveCount() == vx::Particles::kCapacity;
            rc.particles->spawnBlockBreak({0.0, 70.0, 0.0}, 7); // layer 7 marks the burst, no puff
            const int shards = rc.particles->countLayer(7);
            rc.particles->update(10.0f); // empty again for the groups after
            const bool fullOk = full && shards == 14 && rc.particles->liveCount() == 0;
            VX_INFO("particles: full pool (%s) took %d of 14 shards of a burst -> %s",
                    full ? "full" : "NOT full", shards, fullOk ? "PASS" : "FAIL");
            if (!fullOk)
                return 1;
        }

        // Audio: the clips must be decoded ONCE, at init, and never again.
        //
        // This is the direct regression test for the jank. Every play used to call
        // ma_sound_init_from_file with MA_SOUND_FLAG_DECODE — a file open plus a full FLAC
        // decode, synchronously, on the calling thread, from inside the physics loop. The
        // counter is the only way to see that from a headless run, because a CI box has no
        // output device and hears nothing either way.
        //
        // It runs WITHOUT a device on purpose: the bank loads independently of the mixer, so
        // this group exercises the real decode path wherever it runs. To watch it fail, put
        // the decode back on the play path and this goes nonzero immediately.
        {
            // OFFLINE, so the mixer can be pulled and listened to. The previous version of
            // this group used a real device and asserted only that clips were cached and
            // voices acquired — neither of which can tell whether a single sample reaches an
            // output, which is exactly how the engine shipped completely silent with this
            // group green. "31 voices live" only means miniaudio believes it is playing.
            vx::AudioEngine au;
            au.init(soundsDir, &jobs, /*offline=*/true);

            // The one assertion that matters: play something and listen.
            if (au.clipCount() > 0) {
                au.setMasterVolume(1.0f);
                au.setBusVolume(vx::AudioEngine::Bus::World, 1.0f);
                au.setListener({0.0, 0.0, 0.0}, {0.0f, 0.0f, -1.0f});
                au.play("step"); // 2D, so no attenuation can explain silence away
                if (!au.has("step"))
                    au.play(au.bank().anyName());
                const float peak2d = au.renderPeak(4096);

                // And again through the SPATIALISED path, which is the one the game
                // actually uses for footsteps, breaking and placing. Testing only the 2D
                // path would leave the whole world bus — attenuation, positioning, the
                // listener — uncovered, and that is where a silence would hide.
                au.setListener({10.0, 70.0, 10.0}, {0.0f, 0.0f, -1.0f});
                au.playAt("step", {10.0, 70.0, 9.0}); // one block in front of the ear
                const float peak3d = au.renderPeak(4096);

                // Now the GAME's exact per-frame sequence, with the game's defaults. The
                // two probes above pass with hand-set volumes and a listener at the origin;
                // this is the one that has to match what the player actually gets.
                vx::FxToggles def;
                float peakGame = 0.0f;
                {
                    const glm::dvec3 feet{100.0, 70.0, 100.0};
                    const glm::dvec3 eye{100.0, 71.6, 100.0};
                    for (int frame = 0; frame < 8; ++frame) {
                        if (frame == 2) { // a footstep, exactly as the physics loop plays it
                            if (!au.playIfPresent("step_3", feet, 0.5f))
                                au.playAt("step", feet, 0.5f);
                        }
                        au.setListener(eye, {0.0f, 0.0f, -1.0f});
                        au.setMasterVolume(def.volume);
                        au.setBusVolume(vx::AudioEngine::Bus::World, def.volWorld);
                        au.setBusVolume(vx::AudioEngine::Bus::Ui, def.volUi);
                        au.setBusVolume(vx::AudioEngine::Bus::Ambient, def.volAmbient);
                        au.update(1.0f / 60.0f);
                        peakGame = std::max(peakGame, au.renderPeak(800));
                    }
                }

                const bool audible = peak2d > 0.001f && peak3d > 0.001f && peakGame > 0.001f;
                VX_INFO("audio: peak 2d %.4f, spatialised %.4f, GAME sequence %.4f "
                        "(want > 0.001) -> %s",
                        double(peak2d), double(peak3d), double(peakGame),
                        audible ? "PASS" : "FAIL");
                if (!audible)
                    return 1;
            }
            const int clips = au.clipCount();
            const std::uint64_t framesLoaded = au.bank().totalFrames();
            // Bracket the play loop. Anything that decodes a file to satisfy a play shows up
            // as a nonzero delta, which is the entire regression this group exists for.
            const std::uint64_t decodesBefore = vx::SoundBank::decodeCount();

            // Fire far more plays than there are voices, from several names, and drive the
            // per-frame upkeep so voice retirement is exercised too.
            for (int i = 0; i < 400; ++i) {
                au.playAt("step", {double(i % 32), 70.0, 0.0}, 0.5f);
                au.playAt("break", {0.0, 70.0, double(i % 32)});
                au.play("place");
                if (i % 8 == 0)
                    au.update(0.05f); // past the repeat throttle, so plays are not all dropped
            }
            for (int i = 0; i < 80; ++i)
                au.update(0.05f); // long past the steal fade, so every retirement expires

            const std::uint64_t decodes = vx::SoundBank::decodeCount() - decodesBefore;
            const int live = au.liveVoices();
            const int retiring = au.retiringVoices();
            // A bank that decoded nothing would satisfy a decode counter trivially, so the
            // frame total has to be nonzero too whenever clips were actually found.
            const bool bankOk = clips == 0 || framesLoaded > 0;
            // NOT asserting live == 0. A voice ends when the audio DEVICE finishes playing
            // it, which needs real wall-clock time, and this loop runs in microseconds — an
            // assertion on it would be a sleep-dependent flake. What is deterministic is
            // that the pool never overflows and that no stolen voice is left stuck mid-fade,
            // since retirement is driven by the dt passed to update().
            const bool poolOk = live <= 32 && retiring == 0;
            const bool audioOk = decodes == 0 && poolOk && bankOk;
            VX_INFO("audio: %d clip(s) %llu frames cached, %d plays, %llu decode(s) after "
                    "init (want 0), %d live / %d stuck retiring (want 0), %llu throttled -> %s",
                    clips, static_cast<unsigned long long>(framesLoaded), 1200,
                    static_cast<unsigned long long>(decodes), live, retiring,
                    static_cast<unsigned long long>(au.throttledCount()),
                    audioOk ? "PASS" : "FAIL");
            au.shutdown();
            if (!audioOk)
                return 1;
        }

        // Submerging must take the TOP off the world, not just turn it down.
        //
        // Ducking the volume and low-pass filtering both make the meter drop, and only one of
        // them sounds like being underwater — so a peak comparison would pass either way and
        // prove nothing. This measures brightness instead: mean absolute sample-to-sample
        // difference over mean absolute level, which falls when high frequencies go and does
        // not move when the gain does. Nothing is added to AudioEngine for it; readFrames is
        // already public because the mixer has to be pullable at all.
        {
            const auto brightness = [](vx::AudioEngine& au) {
                std::vector<float> buf(4096 * 2, 0.0f);
                const int got = au.readFrames(buf.data(), 4096);
                double level = 0.0, delta = 0.0;
                for (int i = 2; i < got * 2; i += 2) { // left channel only
                    level += std::abs(double(buf[std::size_t(i)]));
                    delta += std::abs(double(buf[std::size_t(i)]) -
                                      double(buf[std::size_t(i - 2)]));
                }
                return level > 1e-9 ? delta / level : 0.0;
            };
            vx::AudioEngine au;
            au.init(soundsDir, &jobs, /*offline=*/true);
            bool wetOk = true;
            double dry = 0.0, wet = 0.0;
            if (au.clipCount() > 0) {
                au.setMasterVolume(1.0f);
                au.setBusVolume(vx::AudioEngine::Bus::World, 1.0f);
                au.setListener({0.0, 0.0, 0.0}, {0.0f, 0.0f, -1.0f});
                au.setUnderwater(false);
                for (int f = 0; f < 30; ++f) // settle the filter at the dry end
                    au.update(1.0f / 60.0f);
                au.playAt("step", {0.0, 0.0, -1.0});
                dry = brightness(au);

                au.setUnderwater(true);
                for (int f = 0; f < 60; ++f) // ~1 s: well past the 150 ms crossover
                    au.update(1.0f / 60.0f);
                au.playAt("break_stone", {0.0, 0.0, -1.0});
                if (!au.has("break_stone"))
                    au.playAt("step", {0.0, 0.0, -1.0});
                wet = brightness(au);
                // Half the brightness is a wide margin; the filter takes far more than that
                // out. A wide margin is deliberate — this must not flake on clip choice.
                wetOk = dry > 0.0 && wet < dry * 0.5;
            }
            VX_INFO("underwater: brightness dry %.4f -> submerged %.4f (want < half) -> %s",
                    dry, wet, wetOk ? "PASS" : "FAIL");
            au.shutdown();
            if (!wetOk)
                return 1;
        }

        // Every block must resolve to a real clip for every action it can be on the
        // receiving end of.
        //
        // This is the group that catches a content bug, and content bugs here are silent by
        // construction: a clip whose name does not match what the lookup asks for simply
        // never plays, and nothing anywhere reports it. Renaming one file is enough to make
        // this go red, which is the point — the id/material/generic chain has three tiers and
        // the middle one exists precisely because ids alone left almost everything unmatched.
        {
            vx::AudioEngine au;
            au.init(soundsDir, &jobs, /*offline=*/true);
            int unmatched = 0;
            int viaId = 0, viaMaterial = 0, viaGeneric = 0;
            std::string firstMiss;
            if (au.clipCount() > 0) {
                for (int id = 1; id < vx::blocks::kBlockCount; ++id) {
                    const char* mat = vx::blocks::soundMaterial(vx::BlockId(id));
                    for (const char* kind : {"break", "place", "step"}) {
                        const std::string k = kind;
                        if (au.has(k + "_" + std::to_string(id)))
                            ++viaId;
                        else if (au.has(k + "_" + mat))
                            ++viaMaterial;
                        else if (au.has(k))
                            ++viaGeneric;
                        else if (au.has("step_" + std::string(mat)) || au.has("step")) {
                            ++viaGeneric; // the pitched-footstep stand-in, still audible
                        } else {
                            ++unmatched;
                            if (firstMiss.empty())
                                firstMiss = k + " id " + std::to_string(id) + " (" + mat + ")";
                        }
                    }
                }
            }
            // Nothing silent is the floor, but on its own it is nearly unfalsifiable: one
            // generic break.wav satisfies every block in the world, so deleting a whole
            // material's content leaves this green. The second assertion is the real one —
            // if the bundled pack is installed (break_stone is the marker), then EVERY
            // material a block can name must resolve at its own tier rather than sliding
            // down to the generic. Removing sounds/break_glass*.wav turns this red, which is
            // exactly the failure the first assertion cannot see.
            std::string thinMaterial;
            if (au.has("break_stone")) {
                for (int id = 1; id < vx::blocks::kBlockCount; ++id) {
                    const std::string mat = vx::blocks::soundMaterial(vx::BlockId(id));
                    if (!au.has("break_" + mat) && thinMaterial.empty())
                        thinMaterial = "break_" + mat;
                }
            }
            const bool covOk =
                (au.clipCount() == 0 || unmatched == 0) && thinMaterial.empty();
            VX_INFO("audio coverage: %d block(s) x 3 actions -> %d by id, %d by material, "
                    "%d generic, %d SILENT%s%s -> %s",
                    vx::blocks::kBlockCount - 1, viaId, viaMaterial, viaGeneric, unmatched,
                    firstMiss.empty() ? "" : (" first: " + firstMiss).c_str(),
                    thinMaterial.empty() ? ""
                                         : (", MISSING " + thinMaterial).c_str(),
                    covOk ? "PASS" : "FAIL");
            au.shutdown();
            if (!covOk)
                return 1;
        }

        // Ambience: the decision, then the sound.
        //
        // mixAmbience is deliberately a pure function of world state so the first half can be
        // asserted with no device, no clips and no clock. The second half is the part that
        // has burned this project before — the Ambient bus existed, was wired, and had never
        // once been proven to carry a sample.
        {
            vx::AmbienceState open; // open field, noon
            open.daylight = 1.0f;
            open.skyAccess = 1.0f;
            open.depth = -2.0f;
            const vx::BedMix dayMix = vx::mixAmbience(open);

            vx::AmbienceState cave; // sealed, forty blocks down
            cave.daylight = 1.0f;
            cave.skyAccess = 0.0f;
            cave.depth = 40.0f;
            const vx::BedMix caveMix = vx::mixAmbience(cave);

            vx::AmbienceState night = open;
            night.daylight = 0.0f;
            const vx::BedMix nightMix = vx::mixAmbience(night);

            vx::AmbienceState sub = open;
            sub.underwater = true;
            const vx::BedMix subMix = vx::mixAmbience(sub);

            // Each of these is a thing a player would immediately notice being wrong.
            // Standing in a roofed house at ground level must NOT sound like a cave. It is
            // the case the enclosure test gets wrong if it forgets that being under cover
            // and being underground are two different things.
            vx::AmbienceState house;
            house.daylight = 1.0f;
            house.skyAccess = 0.0f;
            house.depth = -1.6f; // eye height above the surface, i.e. indoors at grade
            const vx::BedMix houseMix = vx::mixAmbience(house);

            // And wind has to actually respond to height, rather than sitting on its floor
            // for every value a player will ever stand at.
            vx::AmbienceState tower = open;
            tower.depth = -60.0f; // sixty blocks up
            const float windLow = vx::mixAmbience(open).wind;
            const float windHigh = vx::mixAmbience(tower).wind;

            const bool mixOk = dayMix.day > 0.5f && dayMix.cave == 0.0f &&
                               dayMix.night == 0.0f && caveMix.cave > 0.9f &&
                               caveMix.day < 0.05f && caveMix.deep > 0.0f &&
                               nightMix.night > 0.5f && nightMix.day == 0.0f &&
                               subMix.underwater > 0.9f && subMix.day == 0.0f &&
                               subMix.night == 0.0f && subMix.rain == 0.0f &&
                               houseMix.cave == 0.0f && windHigh > windLow * 1.4f;
            VX_INFO("ambience mix: field day/night %.2f/%.2f, cave cave/deep/day "
                    "%.2f/%.2f/%.2f, night night %.2f, submerged under/day %.2f/%.2f, "
                    "roofed house cave %.2f (want 0), wind ground %.2f -> 60 up %.2f -> %s",
                    double(dayMix.day), double(dayMix.night), double(caveMix.cave),
                    double(caveMix.deep), double(caveMix.day), double(nightMix.night),
                    double(subMix.underwater), double(subMix.day), double(houseMix.cave),
                    double(windLow), double(windHigh), mixOk ? "PASS" : "FAIL");
            if (!mixOk)
                return 1;

            vx::AudioEngine au;
            au.init(soundsDir, &jobs, /*offline=*/true);
            bool bedOk = true;
            if (au.has("amb_cave")) {
                au.setMasterVolume(1.0f);
                au.setBusVolume(vx::AudioEngine::Bus::Ambient, 1.0f);
                float peakBed = 0.0f;
                // Two seconds of frames: long enough for the crossfade to climb out of the
                // noise, which is itself the assertion that beds fade rather than snap.
                for (int f = 0; f < 120; ++f) {
                    vx::applyAmbience(au, caveMix, 1.0f);
                    au.update(1.0f / 60.0f);
                    peakBed = std::max(peakBed, au.renderPeak(800));
                }
                const float gainIn = au.bedGain("amb_cave");
                const int beds = au.liveBeds();
                // Now walk out of the cave and make sure it lets go.
                for (int f = 0; f < 600; ++f) {
                    vx::applyAmbience(au, dayMix, 1.0f);
                    au.update(1.0f / 60.0f);
                    au.renderPeak(800);
                }
                const float gainOut = au.bedGain("amb_cave");
                bedOk = peakBed > 0.001f && gainIn > 0.5f && beds > 0 && gainOut == 0.0f;
                VX_INFO("ambience beds: cave peak %.4f (want > 0.001), gain in %.2f -> out "
                        "%.2f, %d bed(s) live -> %s",
                        double(peakBed), double(gainIn), double(gainOut), beds,
                        bedOk ? "PASS" : "FAIL");
            } else {
                VX_INFO("ambience beds: no amb_cave clip supplied, nothing to assert -> PASS");
            }
            // Thunder has to be rare, has to happen at all, and must not fire in fair
            // weather. A storm that rumbles on a fixed short interval is the failure mode,
            // so the count over a fixed stretch of simulated time is the assertion.
            vx::StormSounds storm;
            int strikes = 0;
            for (int f = 0; f < 60 * 300; ++f) // five simulated minutes of heavy rain
                strikes += storm.update(au, 1.0f, 1.0f, 1.0f / 60.0f) ? 1 : 0;
            vx::StormSounds calm;
            int dryStrikes = 0;
            for (int f = 0; f < 60 * 300; ++f)
                dryStrikes += calm.update(au, 0.0f, 1.0f, 1.0f / 60.0f) ? 1 : 0;
            // Thunder must ride the AMBIENCE slider, not the interface one. It went out on
            // the UI bus first time round — `play()` defaults there, which is right for a
            // menu click and wrong for weather — and nothing would have caught it except a
            // routing assertion, because it is perfectly audible either way.
            // Each case gets a FRESH engine. Reusing the one above measured ~0 for the second
            // case whichever bus thunder was on — voices and bed state carried over — so the
            // check passed without testing anything. The control below is there so that can
            // never go unnoticed again: it plays a clip explicitly on the interface bus, and
            // if THAT is inaudible the measurement is broken rather than the routing correct.
            float onAmbient = 0.0f, onUi = 0.0f, control = 0.0f;
            const bool haveThunder = au.has("thunder");
            if (haveThunder) {
                const auto fireAndMeasure = [&](float ambientVol, float uiVol) {
                    vx::AudioEngine one;
                    one.init(soundsDir, &jobs, /*offline=*/true);
                    one.setMasterVolume(1.0f);
                    one.setBusVolume(vx::AudioEngine::Bus::Ambient, ambientVol);
                    one.setBusVolume(vx::AudioEngine::Bus::Ui, uiVol);
                    vx::StormSounds s;
                    for (int f = 0; f < 60 * 200; ++f)
                        if (s.update(one, 1.0f, 1.0f, 1.0f / 60.0f))
                            break;
                    const float p = one.renderPeak(16384);
                    one.shutdown();
                    return p;
                };
                onAmbient = fireAndMeasure(1.0f, 0.0f);
                onUi = fireAndMeasure(0.0f, 1.0f);

                vx::AudioEngine ctl;
                ctl.init(soundsDir, &jobs, /*offline=*/true);
                ctl.setMasterVolume(1.0f);
                ctl.setBusVolume(vx::AudioEngine::Bus::Ambient, 0.0f);
                ctl.setBusVolume(vx::AudioEngine::Bus::Ui, 1.0f);
                ctl.play("thunder", 1.0f, 1.0f, vx::AudioEngine::Bus::Ui);
                control = ctl.renderPeak(16384);
                ctl.shutdown();
            }
            const bool routeOk =
                !haveThunder ||
                (onAmbient > 0.001f && control > 0.001f && onUi < onAmbient * 0.05f);
            const bool stormOk = strikes >= 4 && strikes <= 40 && dryStrikes == 0 && routeOk;
            VX_INFO("storm: %d thunder in 5 min of downpour (want 4..40), %d in clear sky "
                    "(want 0); peak on ambience %.4f, on interface %.4f (want ~0), "
                    "interface control %.4f (want > 0) -> %s",
                    strikes, dryStrikes, double(onAmbient), double(onUi), double(control),
                    stormOk ? "PASS" : "FAIL");

            au.shutdown();
            if (!bedOk || !stormOk)
                return 1;
        }

        // Visibility flood: the dense-grid version has to agree, bit for bit, with the hash-map one it
        // replaced, from the camera's own chunk and from a spread of others around it. The hash one
        // walks the six directions in the opposite order, so this also fails if the result depends
        // on which path reaches a chunk first - the flood that made distant chunks blink did.
        {
            const int cy0 = std::clamp(int(std::floor(camera.position.y / vx::kChunkSize)), 0, vx::kWorldChunksY - 1);
            const int cx0 = int(std::floor(camera.position.x / vx::kChunkSize));
            const int cz0 = int(std::floor(camera.position.z / vx::kChunkSize));
            int agree = 0, total = 0;
            double gridMs = 0.0, hashMs = 0.0;
            for (int dz = -4; dz <= 4; dz += 4)
                for (int dx = -4; dx <= 4; dx += 4)
                    for (int dy = -1; dy <= 1; ++dy) {
                        const int cy = cy0 + dy;
                        if (cy < 0 || cy >= vx::kWorldChunksY)
                            continue;
                        ++total;
                        agree += world.floodMatchesReference({cx0 + dx, cy, cz0 + dz}, &gridMs, &hashMs) ? 1 : 0;
                    }
            const bool floodOk = total > 0 && agree == total;
            VX_INFO("visibility flood: grid version matches the reverse-order hash reference from %d of %d "
                    "vantage points (radius %d: %.3f ms a flood against %.3f ms for the hash version) -> %s",
                    agree, total, world.radius, total > 0 ? gridMs / total : 0.0,
                    total > 0 ? hashMs / total : 0.0, floodOk ? "PASS" : "FAIL");
            if (!floodOk)
                return 1;
        }

        // Lightning: a storm strikes now and then and the thunder follows the flash; a clear sky
        // never does either. Run on the real particle pool, so a bolt's few hundred particles
        // per strike are also proven to fit and to expire.
        if (rc.particles) {
            int thunderStorm = 0, thunderClear = 0, liveMax = 0;
            float maxFlash = 0.0f;
            {
                vx::Lightning storm;
                for (int f = 0; f < 6000; ++f) { // five minutes at 20 Hz
                    const auto c = storm.update(0.05f, 1.0f, {0.0, 80.0, 0.0}, {0.0f, -1.0f},
                                                world.generator(), *rc.particles);
                    thunderStorm += c.thunder ? 1 : 0;
                    maxFlash = std::max(maxFlash, storm.flash());
                    rc.particles->update(0.05f);
                    liveMax = std::max(liveMax, rc.particles->liveCount());
                }
            }
            {
                vx::Lightning calm;
                for (int f = 0; f < 6000; ++f)
                    thunderClear += calm.update(0.05f, 0.0f, {0.0, 80.0, 0.0}, {0.0f, -1.0f},
                                                world.generator(), *rc.particles).thunder ? 1 : 0;
            }
            rc.particles->update(10.0f); // let the last bolt expire
            const int leftover = rc.particles->liveCount();
            const bool lightningOk = thunderStorm >= 8 && thunderStorm <= 70 && thunderClear == 0 &&
                                     maxFlash > 0.8f && liveMax < vx::Particles::kCapacity && leftover == 0;
            VX_INFO("lightning: %d thunder cues in 5 min of storm (want 8..70), %d in clear sky "
                    "(want 0), peak flash %.2f (want > 0.8), at most %d particles live, %d left "
                    "after expiry (want 0) -> %s",
                    thunderStorm, thunderClear, double(maxFlash), liveMax, leftover,
                    lightningOk ? "PASS" : "FAIL");
            if (!lightningOk)
                return 1;
        }

        // Jumping and landing. Both are EDGES, and the edge is the whole feature: "a sound
        // when you go down a block" is not a state anything can read off the player.
        {
            vx::AudioEngine au;
            au.init(soundsDir, &jobs, /*offline=*/true);
            vx::PlayerSounds ps;
            vx::Player p;
            const int gy = world.generator().surfaceHeight(24, 24);

            // Drop from well above the surface: exactly one landing, and it must register as
            // a hard one rather than as an ordinary stride.
            p.position = {24.5, double(gy) + 12.0, 24.5};
            int landings = 0, steps = 0;
            for (int t = 0; t < 240 && landings == 0; ++t) {
                const glm::dvec3 before = p.position;
                p.step(world, {});
                const vx::PlayerSounds::Event e = ps.tick(au, world, p, before);
                landings += e.landed ? 1 : 0;
                steps += e.stepped ? 1 : 0;
            }
            // Keep ticking on the ground: a landing must not repeat itself.
            int extra = 0;
            for (int t = 0; t < 120; ++t) {
                const glm::dvec3 before = p.position;
                p.step(world, {});
                extra += ps.tick(au, world, p, before).landed ? 1 : 0;
            }
            // Walking on the flat must produce strides but never a landing. It needs a floor
            // that is actually flat: the first version of this walked over open terrain and
            // counted six landings, which was not a bug but a hillside — every step DOWN a
            // slope is a real drop, and reading that as a failure would have hidden the
            // opposite mistake.
            const int fy = world.generator().surfaceHeight(200, 200) + 6;
            std::vector<vx::ChunkManager::BlockEdit> deck;
            for (int dz = -3; dz <= 3; ++dz)
                for (int dx = -3; dx <= 34; ++dx)
                    deck.push_back({{200 + dx, fy, 200 + dz}, vx::blocks::Planks});
            world.setBlocks(rc.renderer, light, deck);

            vx::PlayerSounds walkPs;
            vx::Player w;
            w.position = {200.5, double(fy) + 1.0, 200.5};
            vx::PlayerInput in;
            in.forward = 1.0f;
            // yaw 90 is +x: fwd is (sin y, 0, -cos y). Getting this backwards walked him
            // straight off the edge, which read as a landing and looked like a real failure.
            in.yawDeg = 90.0f;
            int walkSteps = 0, walkLandings = 0;
            for (int t = 0; t < 300; ++t) {
                const glm::dvec3 before = w.position;
                w.step(world, in);
                const vx::PlayerSounds::Event e = walkPs.tick(au, world, w, before);
                walkSteps += e.stepped ? 1 : 0;
                walkLandings += e.landed ? 1 : 0;
            }
            // Being MOVED is not falling. Fly up, switch to walk mode, and the first step on
            // the ground must not arrive as a landing from wherever the camera had got to —
            // the fall counter only ever climbs while airborne, so without reset() it carries
            // the whole flight over. Simulated here the way F5 does it: place the player and
            // reset, then walk.
            vx::PlayerSounds tpPs;
            vx::Player tp;
            tp.position = {200.5, double(fy) + 90.0, 200.5}; // "flew up"
            for (int t = 0; t < 30; ++t) {
                const glm::dvec3 before = tp.position;
                tp.step(world, {});
                tpPs.tick(au, world, tp, before);
            }
            tp.position = {200.5, double(fy) + 1.0, 200.5}; // F5: placed on the deck
            tp.velocity = {0.0, 0.0, 0.0};
            tpPs.reset(tp);
            int ghostLandings = 0;
            for (int t = 0; t < 120; ++t) {
                const glm::dvec3 before = tp.position;
                tp.step(world, in);
                ghostLandings += tpPs.tick(au, world, tp, before).landed ? 1 : 0;
            }

            const bool jumpOk = landings == 1 && extra == 0 && walkSteps >= 2 &&
                                walkLandings == 0 && ghostLandings == 0;
            VX_INFO("player sounds: 12-block drop -> %d landing (%d step), %d repeat while "
                    "resting (want 0); walking -> %d stride, %d landing (want 0); after a "
                    "90-block teleport down -> %d landing (want 0) -> %s",
                    landings, steps, extra, walkSteps, walkLandings, ghostLandings,
                    jumpOk ? "PASS" : "FAIL");
            au.shutdown();
            if (!jumpOk)
                return 1;
        }

        // Foliage cannot stand on nothing: break the block under a tuft and the tuft goes
        // with it. Rides the falling queue, so this also proves that queue reaches the voxel
        // ABOVE an edit — which is the only reason it needs no scan of its own.
        {
            const int fx0 = 40, fz0 = 40;
            const int gy = world.generator().surfaceHeight(fx0, fz0);
            const glm::ivec3 ground{fx0, gy, fz0};
            const glm::ivec3 above{fx0, gy + 1, fz0};
            world.setBlock(rc.renderer, light, ground, vx::blocks::Grass);
            world.setBlock(rc.renderer, light, above, vx::blocks::GrassTuft);
            const bool placed = world.blockAtWorld(above) == vx::blocks::GrassTuft;
            // knock the ground out from under it
            world.setBlock(rc.renderer, light, ground, vx::blocks::Air);
            int ticks = 0;
            for (; ticks < 8 && world.blockAtWorld(above) == vx::blocks::GrassTuft; ++ticks)
                world.stepFalling(rc.renderer, light);
            const bool gone = world.blockAtWorld(above) == vx::blocks::Air;
            VX_INFO("foliage: tuft placed %d, cleared after %d tick(s) -> %s", placed ? 1 : 0,
                    ticks, (placed && gone) ? "PASS" : "FAIL");
            if (!(placed && gone))
                return 1;
        }

        const glm::ivec3 ep{18, world.generator().surfaceHeight(18, 18) + 6, 18};
        world.setBlock(rc.renderer, light, ep, vx::blocks::Stone);
        const bool placed = world.blockAtWorld(ep) == vx::blocks::Stone;
        world.setBlock(rc.renderer, light, ep, vx::blocks::Air);
        const bool broken = world.blockAtWorld(ep) == vx::blocks::Air;
        VX_INFO("edits: place %s, break %s", placed ? "PASS" : "FAIL",
                broken ? "PASS" : "FAIL");
        if (!placed || !broken)
            return 1;

        // Batch edits, the path a fluid/falling-block sim will drive. Deliberately spans
        // a chunk border and repeats a voxel, because those are the two things the batch
        // path has to get right that the single-edit path never exercises: one remesh per
        // chunk however many edits hit it, and last-wins ordering within the batch.
        const int surf = world.generator().surfaceHeight(30, 18) + 6;
        std::vector<vx::ChunkManager::BlockEdit> batch;
        for (int i = 0; i < 8; ++i) // 28..35 crosses the x=32 chunk boundary
            batch.push_back({{28 + i, surf, 18}, vx::blocks::Stone});
        batch.push_back({{30, surf, 18}, vx::blocks::Glow}); // overwrite one, last wins
        const int applied = world.setBlocks(rc.renderer, light, batch);
        bool batchOk = applied == 9;
        for (int i = 0; i < 8; ++i) {
            const vx::BlockId want =
                (28 + i) == 30 ? vx::blocks::Glow : vx::blocks::Stone;
            batchOk = batchOk && world.blockAtWorld({28 + i, surf, 18}) == want;
        }
        // the glow block must have lit its neighbourhood, i.e. light settled after the
        // whole batch rather than being skipped
        const bool batchLit = vx::light::block(light.lightAt({30, surf + 1, 18})) > 0;
        VX_INFO("edits: batch %d/9 applied, values %s, light %s", applied,
                batchOk ? "PASS" : "FAIL", batchLit ? "PASS" : "FAIL");
        if (!batchOk || !batchLit)
            return 1;

        std::vector<vx::ChunkManager::BlockEdit> undo;
        for (int i = 0; i < 8; ++i)
            undo.push_back({{28 + i, surf, 18}, vx::blocks::Air});
        world.setBlocks(rc.renderer, light, undo);

        // What the batch path is actually for: the same 64 edits one at a time versus in
        // one call. A sim tick is the second shape, and the first is what it used to cost.
        // per-column surface, so every one of these is genuinely in air — placing into
        // blocks that already hold the target id makes both paths no-ops and the
        // comparison meaningless
        std::vector<vx::ChunkManager::BlockEdit> wall, clear;
        for (int i = 0; i < 64; ++i) {
            const int wx = 40 + (i & 7);
            const glm::ivec3 wp{wx, world.generator().surfaceHeight(wx, 40) + 2 + (i >> 3),
                                40};
            wall.push_back({wp, vx::blocks::Stone});
            clear.push_back({wp, vx::blocks::Air});
        }
        const auto tOne = std::chrono::steady_clock::now();
        for (const auto& e : wall)
            world.setBlock(rc.renderer, light, e.pos, e.id);
        const double oneMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tOne)
                .count();
        world.setBlocks(rc.renderer, light, clear);

        const auto tBatch = std::chrono::steady_clock::now();
        world.setBlocks(rc.renderer, light, wall);
        const double batchMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                      tBatch)
                .count();
        world.setBlocks(rc.renderer, light, clear);
        VX_INFO("edits: 64 blocks one-at-a-time %.2f ms vs batched %.2f ms (%.1fx)", oneMs,
                batchMs, batchMs > 0.0 ? oneMs / batchMs : 0.0);
    }

    // persistence: serialize->disk->deserialize round-trip and an edit that survives a
    // full reload from the region file
    {
        // (a) serializer round-trip on a generated cave chunk: every voxel must match
        auto genChunk = std::make_unique<vx::Chunk>();
        world.generator().generate(*genChunk, {2, 1, 2});
        std::vector<std::uint8_t> bytes;
        const std::size_t n = vx::ChunkSerializer::serialize(*genChunk, bytes);
        auto roundTrip = std::make_unique<vx::Chunk>();
        bool same = vx::ChunkSerializer::deserialize(bytes, *roundTrip);
        for (int y = 0; y < vx::kChunkSize && same; ++y)
            for (int z = 0; z < vx::kChunkSize && same; ++z)
                for (int x = 0; x < vx::kChunkSize && same; ++x)
                    same = genChunk->get(x, y, z) == roundTrip->get(x, y, z);
        VX_INFO("persistence: chunk serialize round-trip %s (%llu bytes vs %d B raw, %.0fx)",
                same ? "PASS" : "FAIL", static_cast<unsigned long long>(n),
                vx::kChunkSize3 * 2, double(vx::kChunkSize3 * 2) / double(n));
        if (!same)
            return 1;

        // (a2) settings round-trip, over EVERY key rather than a chosen few.
        //
        // The previous version of this group set 8 fields and asserted 10 of the 37 the
        // loader knew about — and its comment claimed it checked "every field". It passed
        // for as long as save and load had drifted apart, which is exactly how eight tuned
        // settings came to be read on startup and never written. A test that inspects a
        // subset reports success about the subset.
        //
        // perturbAll walks the schema, so this covers whatever the schema covers, forever.
        // Uses its own file so it can never touch the real saves/settings.cfg.
        {
            const std::filesystem::path sf = outPath.parent_path() / "vxsettings_smoke.cfg";
            vx::FxToggles wrote;
            bool vsyncW = false;
            bool caveCullW = true;
            int radiusW = 16;
            float fovW = 70.0f;
            const int touched =
                vx::Settings::perturbAll(wrote, vsyncW, caveCullW, radiusW, fovW);
            vx::Settings::save(sf, wrote, vsyncW, caveCullW, radiusW, fovW);

            vx::FxToggles read; // all defaults; load must overwrite every one of them
            bool vsyncR = false;
            bool caveCullR = true;
            int radiusR = 16;
            float fovR = 70.0f;
            const bool got = vx::Settings::load(sf, read, vsyncR, caveCullR, radiusR, fovR);
            const int diff = vx::Settings::differences(wrote, vsyncW, caveCullW, radiusW,
                                                       fovW, read, vsyncR, caveCullR,
                                                       radiusR, fovR);
            // touched == keyCount() is the half that matters: it proves the perturbation
            // actually moved every field, so a value that failed to save cannot coincide
            // with the default it would be compared against.
            const bool ok = got && touched == vx::Settings::keyCount() && diff == 0;
            std::error_code sec;
            std::filesystem::remove(sf, sec);
            VX_INFO("persistence: settings round-trip %d/%d keys perturbed, %d differ after "
                    "reload (want 0) -> %s",
                    touched, vx::Settings::keyCount(), diff, ok ? "PASS" : "FAIL");
            if (!ok)
                return 1;
        }

        // (a3) every preset reads back as itself, and a file saved before the detail distance
        // was a preset knob (it has the old "lodDetail=1.25" and no "detailDist") loads with
        // the reach of the preset the rest of it matches. That second part is what brings an
        // Ultra player's mountains back; without it their file stays capped at 1.25 forever.
        {
            int recognised = 0;
            for (int i = 0; i < 4; ++i) {
                vx::FxToggles fx;
                int radius = 16;
                vx::applyPreset(vx::Preset(i), fx, radius);
                recognised += vx::detectPreset(fx, radius) == vx::Preset(i);
            }
            const std::filesystem::path sf = outPath.parent_path() / "vxsettings_old_smoke.cfg";
            const auto oldFile = [&](vx::Preset p) {
                vx::FxToggles fx;
                int radius = 16;
                vx::applyPreset(p, fx, radius);
                fx.lodDetail = 1.0f; // must not survive: the file will not carry it
                vx::Settings::save(sf, fx, false, true, radius, 70.0f);
                std::ifstream in(sf);
                std::string text, line;
                while (std::getline(in, line))
                    if (line.rfind("detailDist=", 0) != 0)
                        text += line + "\n";
                in.close();
                std::ofstream(sf, std::ios::trunc) << text << "lodDetail=1.25\n";
                vx::FxToggles got;
                bool vs = false, cc = true;
                int r = 16;
                float fov = 70.0f;
                vx::Settings::load(sf, got, vs, cc, r, fov);
                return got.lodDetail;
            };
            const float ultra = oldFile(vx::Preset::Ultra);
            const float high = oldFile(vx::Preset::High);
            std::error_code sec;
            std::filesystem::remove(sf, sec);
            const bool ok = recognised == 4 && std::fabs(ultra - 2.0f) < 1e-4f &&
                            std::fabs(high - 1.5f) < 1e-4f;
            VX_INFO("persistence: %d/4 presets recognised; a pre-preset file loads detail %.2f "
                    "as Ultra (want 2.00), %.2f as High (want 1.50) -> %s",
                    recognised, double(ultra), double(high), ok ? "PASS" : "FAIL");
            if (!ok)
                return 1;
        }

        // (b) edit a surface block, save to disk, then load it back into a FRESH chunk
        // straight from the region file — the edit must survive
        const int surf = world.generator().surfaceHeight(70, 70);
        const glm::ivec3 wp{70, surf + 1, 70};
        const vx::ChunkPos cp{wp.x >> 5, wp.y >> 5, wp.z >> 5};
        world.setBlock(rc.renderer, light, wp, vx::blocks::Glow);
        world.saveDirty(); // serialize dirty chunks + flush region files

        auto reloaded = std::make_unique<vx::Chunk>();
        const bool present = world.regions().load(cp, *reloaded);
        const bool kept = present && reloaded->get(wp.x & 31, wp.y & 31, wp.z & 31) ==
                                         vx::blocks::Glow;
        const auto rs = world.regions().stats();
        VX_INFO("persistence: edit survived reload from disk %s (%d chunk(s), %.1f KB on disk)",
                kept ? "PASS" : "FAIL", rs.chunksStored,
                double(rs.bytesStored) / 1024.0);
        if (!kept)
            return 1;

        // (c) end-to-end: fly far enough to evict the chunk, then back — it must
        // regenerate FROM DISK (not the seed) with the edit intact
        camera.position = {4096.0, 110.0, 4096.0};
        drain();
        camera.position = {16.0, 110.0, 16.0};
        drain();
        const bool live = world.blockAtWorld(wp) == vx::blocks::Glow;
        VX_INFO("persistence: edit present after evict + reload from disk %s",
                live ? "PASS" : "FAIL");
        if (!live)
            return 1;

        // The save format used to encode the world height in its slot count, and the test
        // here checked that a differently-tall world was rejected. There is no world height
        // any more (#27), so what has to hold is the version gate: an old v1 region must be
        // refused rather than read with the wrong stride and silently mined for garbage.
        const std::filesystem::path badDir = outPath.parent_path() / "vxsave_oldfmt";
        std::error_code bec;
        std::filesystem::remove_all(badDir, bec);
        std::filesystem::create_directories(badDir, bec);
        {
            std::vector<std::uint8_t> f{'V', 'X', 'R', 'G'};
            const auto put32 = [&](std::uint32_t v) {
                for (int i = 0; i < 4; ++i)
                    f.push_back(std::uint8_t(v >> (8 * i)));
            };
            put32(1);      // version 1: the dense-slot-grid format
            put32(16 * 16 * 12); // its slot count, which used to encode the world height
            f.resize(f.size() + std::size_t(16 * 16 * 12) * 8, 0);
            std::ofstream out(badDir / "r.0.0.vxr", std::ios::binary);
            out.write(reinterpret_cast<const char*>(f.data()), std::streamsize(f.size()));
        }
        vx::RegionStore probe(badDir);
        vx::Chunk scratch;
        const bool rejected = !probe.load({0, 0, 0}, scratch);
        VX_INFO("persistence: v1 region refused by the version gate %s",
                rejected ? "PASS" : "FAIL");
        std::filesystem::remove_all(badDir, bec);
        if (!rejected)
            return 1;

        // #27: chunk Y is unbounded, so a chunk far outside the old 0..11 range has to
        // store and reload. Negative Y is the case the old dense slot index could not even
        // address — it multiplied Y into an offset and would have run off the front.
        {
            const std::filesystem::path deepDir = outPath.parent_path() / "vxsave_deep";
            std::filesystem::remove_all(deepDir, bec);
            vx::Chunk deep;
            deep.makeUniform(vx::blocks::Stone);
            deep.set(3, 4, 5, vx::blocks::Bricks);
            std::vector<std::uint8_t> bytes;
            vx::ChunkSerializer::serialize(deep, bytes);
            const vx::ChunkPos far{2, -40, -3}; // 1280 blocks below the old world floor
            {
                vx::RegionStore rs(deepDir);
                rs.save(far, bytes);
                rs.flush();
            }
            vx::Chunk back;
            vx::RegionStore rs2(deepDir);
            const bool got = rs2.load(far, back);
            const bool ok = got && back.get(3, 4, 5) == vx::blocks::Bricks &&
                            back.get(0, 0, 0) == vx::blocks::Stone;
            VX_INFO("persistence: chunk at y=%d round-tripped %s", far.y,
                    ok ? "PASS" : "FAIL");
            std::filesystem::remove_all(deepDir, bec);
            if (!ok)
                return 1;
        }
    }

    // screenshot from inside the glow room
    camera.position = {16.5, 48.0, 16.5};
    camera.yawDeg = 40.0f;
    camera.pitchDeg = -20.0f;
    // Rebuild the draw metadata first. Everything between the main screenshot and here has
    // moved the chunk set — the fluid and falling tests edit it, and the persistence test
    // teleports 4096 blocks away and back — so the meta buffer this used to draw from
    // described chunks that no longer existed. The result was a featureless grey frame that
    // has been passing for a picture of the glow room.
    world.collectDrawInfo(drawInfo, waterInfo);
    rc.renderer.updateChunkMeta(drawInfo);
    rc.renderer.updateWaterMeta(waterInfo);
    world.updateVisibilityIfNeeded(rc.renderer, camera.position, true);
    rc.renderer.beginFrame();
    renderFrame(rc, camera, fx, true, w, h, fbo, 0.6f, nullptr);
    rc.renderer.endFrame();
    glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, pixels.data());
    std::filesystem::path roomPath = outPath;
    roomPath.replace_filename(outPath.stem().string() + "_room.bmp");
    vx::writeBmp24(roomPath, w, h, pixels);
    VX_INFO("smoke: glow room screenshot -> %s", roomPath.string().c_str());

    // per-effect GPU cost: back on the surface, time the shadow/scene/post stages with
    // GPU queries, then toggle each effect and measure the delta on the post stage.
    camera.position = {16.0, 110.0, 16.0};
    camera.yawDeg = -35.0f;
    camera.pitchDeg = -25.0f;
    for (int i = 0; i < 6000 && (world.busy() || !light.idle()); ++i) {
        rc.renderer.beginFrame();
        world.update(jobs, rc.renderer, light, camera.position); // settle LOD before timing
        rc.renderer.endFrame();
    }
    world.collectDrawInfo(drawInfo, waterInfo);
    rc.renderer.updateChunkMeta(drawInfo);
    rc.renderer.updateWaterMeta(waterInfo);
    world.updateVisibilityIfNeeded(rc.renderer, camera.position, true);
    // note: this GPU-query benchmark is unreliable while re-LOD workers churn in the
    // background (negative deltas) — trust the in-app HUD gpu timer for real numbers.
    FrameTimers ft;
    ft.shadow.init();
    ft.scene.init();
    ft.post.init();
    const auto timeConfig = [&](vx::FxToggles cfg) {
        double s = 0, sc = 0, p = 0;
        for (int i = 0; i < 40; ++i) {
            rc.renderer.beginFrame();
            renderFrame(rc, camera, cfg, true, w, h, fbo, 0.6f, &ft);
            rc.renderer.endFrame();
            glFinish();
            if (i >= 8) { // let the query ring fill
                s += ft.shadow.milliseconds();
                sc += ft.scene.milliseconds();
                p += ft.post.milliseconds();
            }
        }
        return glm::dvec3(s, sc, p) / 32.0;
    };
    vx::FxToggles all;
    const glm::dvec3 tAll = timeConfig(all);
    vx::FxToggles noShadow = all; noShadow.shadows = false;
    vx::FxToggles noSsao = all; noSsao.ssao = false;
    vx::FxToggles noVol = all; noVol.volumetric = false;
    vx::FxToggles noBloom = all; noBloom.bloom = false;
    vx::FxToggles none{}; none.shadows = none.ssao = none.volumetric = none.bloom = false;
    const glm::dvec3 tNoShadow = timeConfig(noShadow);
    const glm::dvec3 tNoSsao = timeConfig(noSsao);
    const glm::dvec3 tNoVol = timeConfig(noVol);
    const glm::dvec3 tNoBloom = timeConfig(noBloom);
    const glm::dvec3 tNone = timeConfig(none);
    VX_INFO("gpu cost @ %dx%d, radius %d: base scene %.2f ms, full pipeline %.2f ms", w, h,
            world.radius, tNone.y, tAll.x + tAll.y + tAll.z);
    VX_INFO("  shadows +%.2f | ssao +%.2f | volumetric +%.2f | bloom +%.2f ms",
            (tAll.x + tAll.y + tAll.z) - (tNoShadow.x + tNoShadow.y + tNoShadow.z),
            (tAll.z - tNoSsao.z), (tAll.z - tNoVol.z), (tAll.z - tNoBloom.z));
    return 0;
}

} // namespace vx
