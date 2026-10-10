#pragma once

#include "world/Chunk.hpp"

#include <algorithm>

namespace vx {

// How strongly the water surface at one lattice point (a block corner) may move with the waves:
// 0 = not at all, kWaveFull = fully. Shared by the mesher, which bakes it into every water top
// face, and by the camera's underwater test, which has to agree with the surface it sees.
//
// All four cells around the point must be full source water with open air directly above, so a
// wave never lifts water into a block, a beach, an overhang or a flowing sheet, and never pulls
// a surface away from a wall. The strength is the shallowest of the four water depths: waves
// shoal, so the surface flattens toward the waterline over the last few blocks instead of
// stopping dead at it.
//
// Depth is only counted down to floorY and the cell above is skipped when it would be in the
// next chunk row up (y == ceilY): the mesher sees those cells for its own column but not for the
// neighbouring ones in its pad, and every chunk meeting at a corner has to reach the same answer
// or the surface tears there. Running off the bottom still in water counts as deep.
inline constexpr int kWaveFull = 7;

template <class BlockAt>
int waveAtLattice(const BlockAt& blockAt, int lx, int y, int lz, int floorY, int ceilY) {
    int amp = kWaveFull;
    for (int dz = -1; dz <= 0; ++dz)
        for (int dx = -1; dx <= 0; ++dx) {
            const int cx = lx + dx, cz = lz + dz;
            if (blockAt(cx, y, cz) != blocks::Water)
                return 0;
            if (y < ceilY && blockAt(cx, y + 1, cz) != blocks::Air)
                return 0;
            int depth = 1;
            while (depth < kWaveFull && y - depth >= floorY &&
                   blocks::isWater(blockAt(cx, y - depth, cz)))
                ++depth;
            if (y - depth < floorY)
                depth = kWaveFull;
            amp = std::min(amp, depth);
        }
    return amp;
}

} // namespace vx
