#pragma once

#include "world/Chunk.hpp"

#include <cstdint>
#include <vector>

namespace vx {

// 16 bytes: position + face-shaded color. Expanded-vertex format is itself part of the
// baseline — M4 replaces the whole thing with one packed uint64 per *quad* in an SSBO.
struct ChunkVertex {
    float x, y, z;
    std::uint8_t r, g, b, a;
};

struct MeshStats {
    int faces = 0;
    int triangles = 0;
    int vertices = 0;
    int facesPerDir[6] = {}; // +X,-X,+Y,-Y,+Z,-Z — the greedy mesher's oracle numbers
};

// The M3 baseline mesher: visits every voxel, emits 6 verts per face that borders air.
// Slow and correct on purpose — M4's binary greedy mesher gets measured against this.
// `out` is cleared, not freed, so repeat calls don't reallocate once warm.
MeshStats meshNaive(const Chunk& chunk, std::vector<ChunkVertex>& out);

} // namespace vx
