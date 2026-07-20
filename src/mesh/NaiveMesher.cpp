#include "mesh/NaiveMesher.hpp"

namespace vx {

namespace {

// Corner order is counter-clockwise seen from outside the block (GL front face), so
// backface culling works. Triangles are (0,1,2) and (0,2,3).
struct FaceDef {
    std::int8_t nx, ny, nz;         // neighbor that must be air for the face to exist
    std::uint8_t corner[4][3];      // offsets from the voxel min corner
    float shade;                    // baked directional shade, classic minecraft look
};

constexpr FaceDef kFaces[6] = {
    {+1, 0, 0, {{1, 0, 0}, {1, 1, 0}, {1, 1, 1}, {1, 0, 1}}, 0.80f},
    {-1, 0, 0, {{0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {0, 0, 0}}, 0.80f},
    {0, +1, 0, {{0, 1, 0}, {0, 1, 1}, {1, 1, 1}, {1, 1, 0}}, 1.00f},
    {0, -1, 0, {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}}, 0.50f},
    {0, 0, +1, {{0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}}, 0.65f},
    {0, 0, -1, {{1, 0, 0}, {0, 0, 0}, {0, 1, 0}, {1, 1, 0}}, 0.65f},
};

constexpr std::uint8_t kBlockColor[][3] = {
    {0, 0, 0},       // air (never emitted)
    {125, 125, 125}, // stone
    {134, 96, 67},   // dirt
    {106, 170, 64},  // grass
};

bool solidAt(const Chunk& chunk, int x, int y, int z) {
    // Outside the chunk counts as air until M5 wires real neighbors, so border faces
    // are emitted and a lone chunk looks right.
    if (x < 0 || x >= kChunkSize || y < 0 || y >= kChunkSize || z < 0 || z >= kChunkSize)
        return false;
    return chunk.get(x, y, z) != blocks::Air;
}

} // namespace

MeshStats meshNaive(const Chunk& chunk, std::vector<ChunkVertex>& out) {
    out.clear();
    MeshStats stats;

    for (int y = 0; y < kChunkSize; ++y) {
        for (int z = 0; z < kChunkSize; ++z) {
            for (int x = 0; x < kChunkSize; ++x) {
                const BlockId id = chunk.get(x, y, z);
                if (id == blocks::Air)
                    continue;

                for (const FaceDef& face : kFaces) {
                    if (solidAt(chunk, x + face.nx, y + face.ny, z + face.nz))
                        continue;

                    const std::uint8_t r = std::uint8_t(float(kBlockColor[id][0]) * face.shade);
                    const std::uint8_t g = std::uint8_t(float(kBlockColor[id][1]) * face.shade);
                    const std::uint8_t b = std::uint8_t(float(kBlockColor[id][2]) * face.shade);

                    const auto corner = [&](int i) {
                        return ChunkVertex{float(x + face.corner[i][0]),
                                           float(y + face.corner[i][1]),
                                           float(z + face.corner[i][2]), r, g, b, 255};
                    };
                    // two triangles, no index buffer — that would be an optimization,
                    // and this mesher isn't allowed any
                    out.push_back(corner(0));
                    out.push_back(corner(1));
                    out.push_back(corner(2));
                    out.push_back(corner(0));
                    out.push_back(corner(2));
                    out.push_back(corner(3));
                    ++stats.faces;
                }
            }
        }
    }

    stats.triangles = stats.faces * 2;
    stats.vertices = stats.faces * 6;
    return stats;
}

} // namespace vx
