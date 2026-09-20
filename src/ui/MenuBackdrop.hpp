#pragma once

namespace vx {

class JobSystem;
class TerrainGenerator;
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
    // Renders one frame into the default framebuffer. `gen` is the terrain to show; the caller
    // owns it and must reset the far terrain before replacing it (see FarTerrain::reset).
    void draw(RenderContext& rc, JobSystem& jobs, const TerrainGenerator& gen, int fbW, int fbH,
              double time);

private:
    // Where the camera hangs for a given seed: found once by looking at what surrounds a grid of
    // candidate spots (some sea, some height to look up at, nothing too steep underneath).
    struct Vantage {
        double x = 0.0, z = 0.0, y = 100.0;
        float yaw = 0.0f; // the direction with the most to look at
    };
    static Vantage findVantage(const TerrainGenerator& gen);

    bool m_metaCleared = false;
    bool m_haveVantage = false;
    int m_vantageSeed = 0;
    Vantage m_vantage;
};

} // namespace vx
