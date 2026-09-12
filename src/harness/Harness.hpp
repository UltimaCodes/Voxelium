#pragma once

#include <chrono>
#include <filesystem>

namespace vx {

class ChunkManager;
class DebugHud;
class FlyCamera;
class JobSystem;
class LightEngine;
class Window;
struct RenderContext;

// The three offscreen entry points, plus the startup oracle. They live outside main.cpp
// because together they were 1259 of its 2326 lines and none of them is the game — they are
// the test suite, the profiler and the screenshot rig, and each is read on its own.
//
// All four render through render/FrameRenderer, deliberately: a harness that grew its own
// copy of the frame would eventually be measuring a frame nobody plays.

// Best of `runs` wall-clock timings, in milliseconds. Best rather than mean because this is
// used for single-threaded CPU work where the floor is the signal and everything above it is
// scheduler noise. NOT for GPU frame cost — see INFO.md §0 on mean-and-peak.
template <typename F>
double benchBestMs(int runs, F&& fn) {
    double best = 1e9;
    for (int i = 0; i < runs; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return best;
}

// Naive vs greedy on the fixed test chunk: merged quad area per direction must equal the
// naive face count exactly. Catches mesher AND palette-storage regressions in one go, since
// both meshers read through Chunk::get(). If this fails, stop.
bool runMesherOracle();

// The test suite. Twenty-three groups, all must pass. Returns a process exit code.
int runSmokeTest(RenderContext& rc, ChunkManager& world, JobSystem& jobs, LightEngine& light,
                 FlyCamera& camera, DebugHud& hud, Window& window,
                 const std::filesystem::path& outPath,
                 const std::filesystem::path& soundsDir);

// Screenshots at five stops walking outward from spawn, plus night and underwater frames and
// a sky-light invariant check. `settle` false does not wait, so shots land mid-stream.
int runTour(RenderContext& rc, ChunkManager& world, JobSystem& jobs, LightEngine& light,
            FlyCamera& camera, Window& window, int radius, bool settle);

// GPU frame timing only: settled world, fixed cameras, TAA off. `saveBaseline` writes this
// run's numbers to bench_baseline.txt for future runs to compare against; otherwise, if that
// file already exists, this run compares against it and flags regressions in the log.
int runBenchmark(RenderContext& rc, ChunkManager& world, JobSystem& jobs, LightEngine& light,
                 FlyCamera& camera, Window& window, int radius, bool saveBaseline = false);

// README GIF material: settles the world FIRST (unlike --tour, which deliberately captures
// mid-stream) at a modest radius around spawn, then smoothly interpolates the camera through
// a short path over `frames` frames, writing flyframe_NNNN.bmp for every one of them — a
// promotional clip has no business showing the streaming pop-in the rest of the harness
// exists to catch. Assembling the frames into an actual .gif is a Python/PIL job, not
// something worth a new C++ dependency for.
int runFlythrough(RenderContext& rc, ChunkManager& world, JobSystem& jobs, LightEngine& light,
                  FlyCamera& camera, Window& window, int frames);

} // namespace vx
