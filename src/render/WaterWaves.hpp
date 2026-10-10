#pragma once

#include "mesh/WaveRule.hpp"
#include "world/Chunk.hpp"

#include <glm/glm.hpp>

#include <cmath>

namespace vx {

// The swell on open water: four long waves running with the wind, summed, with crests sharpened
// and troughs broadened (exp of a sine rather than a plain sine). They move the water surface for
// real within kFadeEnd of the camera (water.tesc / water.tese) and shade it beyond that.
//
// Everything about them lives here and reaches the shaders as uniforms, so the surface the GPU
// draws and the one the camera's underwater test checks against are the same function.
class WaterWaves {
public:
    static constexpr int kCount = 4;
    // Height of the swell's biggest waves on deep water on a calm day, in blocks. A storm doubles
    // it. Shallow water gets less (WaveRule.hpp), and the four waves seldom line up, so what a
    // surface usually shows is well under this.
    static constexpr float kCalm = 0.30f;
    // The waves fade out over this band of distance; beyond it the water is flat geometry again
    // and only its normal carries a swell.
    static constexpr float kFadeStart = 72.0f;
    static constexpr float kFadeEnd = 112.0f;

    // One frame's worth of shader input. Each wave's phase is worked out at the camera in double,
    // so the pattern stays fixed to the world at any distance from the origin; the shaders only
    // add the camera-relative part, which is small.
    struct Frame {
        glm::vec4 wave[kCount]{}; // direction xz, wavenumber (radians a block), weight
        glm::vec4 phase{0.0f};    // each wave's phase at the camera, time included
        glm::vec4 shape{0.0f};    // sharpness s, exp(-2s), mean of exp(s(sin - 1)), 2 / (1 - exp(-2s))
        float scale = 0.0f;       // blocks at full strength, weather included; 0 = flat water
        glm::vec2 camWrap{0.0f};  // camera xz modulo kWrap, for the chop and the foam
    };
    // The chop and foam noise repeat every kWrap blocks, and the camera's position modulo it is
    // what they are offset by. That keeps them still when the camera moves (they used to slide
    // along with it) without ever feeding a large coordinate to a float.
    static constexpr double kWrap = 512.0;

    // storm: 0 calm .. 1 the worst a storm gets. on = false gives flat water.
    static Frame frame(double time, const glm::dvec3& cam, float storm, bool on);
    // The swell's height at a camera-relative point, before strength and fade (-0.8 .. 1.2).
    static float unitHeight(const Frame& f, float relX, float relZ);
    // The swell's strength for distance d from the camera, 1 near it, 0 past kFadeEnd.
    static float fade(float d);
    // Sets the uniforms water.tesc / water.tese / water.frag read (locations 26..35).
    // tessDensity: water vertices per block along an edge, 1 or 2.
    static void apply(unsigned int program, const Frame& f, float tessDensity);
};

// The world height of the water surface in the top water cell (x, cellY, z) at the camera's own
// xz: its rest level (a flowing cell sits lower, as water.vert draws it) plus the swell, scaled by
// the wave strength the mesher gives the corners around it. This is the line the camera has to be
// under to be underwater, so the fog switches where the surface is drawn, not at the block edge.
template <class BlockAt>
double waterSurfaceY(const WaterWaves::Frame& f, const BlockAt& blockAt, const glm::dvec3& cam,
                     int cellY) {
    const int x = int(std::floor(cam.x)), z = int(std::floor(cam.z));
    const int level = blocks::waterLevel(blockAt(x, cellY, z));
    double y = double(cellY) + 1.0 - (0.10 + double(8 - std::max(level, 1)) * 0.105);
    if (f.scale <= 0.0f)
        return y;
    // the chunk row the cell is in bounds the depth count, exactly as the mesher's does
    const int floorY = cellY & ~(kChunkSize - 1);
    const int ceilY = floorY + kChunkSize - 1;
    const double fx = cam.x - double(x), fz = cam.z - double(z);
    const auto amp = [&](int lx, int lz) {
        return double(waveAtLattice(blockAt, lx, cellY, lz, floorY, ceilY)) / double(kWaveFull);
    };
    const double a = (amp(x, z) * (1.0 - fx) + amp(x + 1, z) * fx) * (1.0 - fz) +
                     (amp(x, z + 1) * (1.0 - fx) + amp(x + 1, z + 1) * fx) * fz;
    return y + a * double(f.scale) * double(WaterWaves::unitHeight(f, 0.0f, 0.0f));
}

} // namespace vx
