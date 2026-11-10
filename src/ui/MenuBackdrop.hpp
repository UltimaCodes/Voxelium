#pragma once

#include <atomic>

namespace vx {

class JobSystem;
class TerrainGenerator;
struct FxToggles;
struct RenderContext;

// The scene behind the title and loading screens: the real sky, clouds, sea and far-terrain
// heightfield of a seed, seen from a camera turning slowly high over it. No chunks exist yet
// (or need to), so it is the far horizon and nothing else, which is also why it is cheap enough
// to run at the menu's frame rate.
//
// It draws with the game's own renderer, so what the menu shows is what the world looks like,
// and the seed field on the title screen previews the terrain of whatever is typed in it.
class MenuBackdrop {
public:
    ~MenuBackdrop() { quiesce(); }

    // Renders one frame into the default framebuffer. `gen` is the terrain to show; the caller
    // owns it and must not replace it while busy() (see also FarTerrain::reset). Until the camera's
    // spot for a new seed is found, on a worker, it draws black. Without `look` it is the title's
    // morning; a loading screen passes the game's settings, so it opens at the world's own hour.
    void draw(RenderContext& rc, JobSystem& jobs, const TerrainGenerator& gen, int fbW, int fbH,
              double time, const FxToggles* look = nullptr);
    // The spot for a seed is being looked for, reading the generator.
    bool busy() const { return m_jobRunning.load(std::memory_order_acquire); }
    void quiesce() const;

private:
    // Where the camera hangs for a given seed: found once by looking at what surrounds a grid of
    // candidate spots (some sea, some height to look up at, nothing too steep underneath).
    struct Vantage {
        double x = 0.0, z = 0.0, y = 100.0;
        float yaw = 0.0f; // the direction with the most to look at
    };
    static Vantage findVantage(const TerrainGenerator& gen, JobSystem* jobs);

    bool m_metaCleared = false;
    bool m_haveVantage = false;
    int m_vantageSeed = 0, m_vantageGen = 0;
    Vantage m_vantage;
    // the search on a worker: it writes these, then clears m_jobRunning
    std::atomic<bool> m_jobRunning{false};
    int m_jobSeed = 0, m_jobGen = 0;
    bool m_jobDone = false;
    Vantage m_jobResult;
};

} // namespace vx
