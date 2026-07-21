#pragma once

#include "world/Chunk.hpp"

#include <cstdint>

namespace vx {

// Which of a chunk's 6 faces can see each other through its non-solid voxels — 15
// face-pair bits (Tommo's "advanced cave culling"). Computed at mesh time on a worker;
// the per-frame visibility flood in ChunkManager walks these to decide which chunks
// can possibly be seen from the camera.
//
// Faces use the engine's dir order: 0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z.

constexpr int facePairIndex(int a, int b) {
    // a < b required; pairs packed triangularly: (0,1)..(0,5), (1,2)..(1,5), ...
    return a * (11 - a) / 2 + (b - a - 1);
}
static_assert(facePairIndex(0, 1) == 0 && facePairIndex(4, 5) == 14);

constexpr bool facesConnected(std::uint16_t bits, int faceA, int faceB) {
    const int a = faceA < faceB ? faceA : faceB;
    const int b = faceA < faceB ? faceB : faceA;
    return ((bits >> facePairIndex(a, b)) & 1u) != 0;
}

inline constexpr std::uint16_t kAllFacesConnected = 0x7FFF;

std::uint16_t computeConnectivity(const Chunk& chunk);

} // namespace vx
