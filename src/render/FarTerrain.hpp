#pragma once

#include <glm/glm.hpp>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace vx {

class JobSystem;
class TerrainGenerator;

// The world past the streamed chunks: one coarse heightfield, thousands of blocks across.
//
// The chunk ring can only reach a few hundred blocks before generation and memory stop being
// worth it, and past it the world simply ended in fog. But terrain is a pure function of the
// seed and x/z, so what lies beyond can be sampled directly: a grid of columns (height, how
// desert, how wooded) turned into a lit, coloured mesh, with no voxels and no chunks. It is
// what makes a mountain ten kilometres off visible at all.
//
// Rebuilt on a worker whenever the camera has moved far enough from the last build's centre,
// on a world-anchored grid so it never swims. Drawn under the chunks: it skips everything
// within the streamed radius and dips a few blocks at the seam so the real terrain always wins.
// It has no shadows and no detail and is meant to be seen through haze.
class FarTerrain {
public:
    static constexpr int kCells = 160;        // cells per side
    static constexpr int kSpacing = 64;       // blocks per cell
    static constexpr float kHalfExtent = float(kCells * kSpacing) * 0.5f; // 5120
    // Where the haze fully closes in, so the grid's edge is never seen.
    static constexpr float kFogEnd = kHalfExtent - 640.0f;

    bool init(const std::filesystem::path& shaderDir);
    void shutdown();

    // Once a frame from the main thread: notices when a rebuild is due, hands it to a worker,
    // and uploads a finished one. Cheap when nothing is happening.
    void update(const TerrainGenerator& gen, JobSystem& jobs, const glm::dvec3& cam);
    // Forgets the current mesh (a new world has different terrain).
    void reset();

    // Inside the HDR scene pass, after the chunks. `innerRadius` is how far the real world
    // reaches, in blocks: nothing is drawn inside it.
    void draw(const glm::mat4& viewProj, const glm::dvec3& cam, const glm::vec3& sunL,
              const glm::vec3& sunColor, const glm::vec3& skyAmbient, float innerRadius);

    bool ready() const { return m_haveMesh; }

private:
    struct Vertex {
        float h;
        std::uint8_t r, g, b, water;
        std::int8_t nx, ny, nz, pad;
    };
    void build(const TerrainGenerator& gen, std::int64_t originX, std::int64_t originZ,
               std::vector<Vertex>& out) const;

    unsigned int m_prog = 0;
    unsigned int m_vao = 0;
    unsigned int m_ssbo = 0;
    bool m_haveMesh = false;
    glm::dvec2 m_origin{0.0};      // world XZ of the grid's first vertex, as uploaded
    glm::dvec2 m_centre{1e18};     // where the uploaded mesh was centred

    // hand-off from the worker
    std::vector<Vertex> m_staging;
    glm::dvec2 m_stagingOrigin{0.0};
    std::atomic<bool> m_building{false};
    std::atomic<bool> m_stagingReady{false};
    std::uint32_t m_seedEpoch = 0; // bumped by reset(), so a stale job's result is dropped
    std::atomic<std::uint32_t> m_jobEpoch{0};
};

} // namespace vx
