#pragma once

#include <glm/glm.hpp>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace vx {

class JobSystem;
class TerrainGenerator;

// The world past the streamed chunks: nested heightfields out to about 8 km.
//
// The chunk ring can only reach so far before generation and memory stop being worth it, but
// terrain is a pure function of the seed and x/z, so what lies beyond can be sampled directly: grids
// of columns turned into a lit, coloured mesh, with no voxels and no chunks. The inner ring is fine
// enough to show ridgelines, the outer one reaches the horizon.
//
// Each ring is rebuilt on a worker once the camera has moved far enough from its centre, on a
// world-anchored grid so it never swims. It skips what is nearer in and dips a few blocks at the seam
// so the real terrain always wins. It is lit by the same model as the chunks, with its shadows marched
// across its own height texture.
class FarTerrain {
public:
    static constexpr int kRings = 2;
    static constexpr int kCells[kRings] = {320, 256};  // cells per side
    static constexpr int kSpacing[kRings] = {16, 64};  // blocks per cell
    // Where the haze fully closes in, so the grid's edge is never seen.
    static constexpr float kFogEnd = float(kCells[1] * kSpacing[1]) * 0.5f - 640.0f;

    bool init(const std::filesystem::path& shaderDir);
    void shutdown();

    // Once a frame from the main thread: notices when a rebuild is due, hands it to a worker,
    // and uploads a finished one. Cheap when nothing is happening.
    void update(const TerrainGenerator& gen, JobSystem& jobs, const glm::dvec3& cam);
    // Forgets the current mesh (a new world has different terrain).
    void reset();

    // Inside the HDR scene pass, after the chunks, with the clouds' weather map still on unit 6.
    // `innerRadius` is how far the real world reaches, in blocks: nothing is drawn inside it.
    void draw(const glm::mat4& viewProj, const glm::dvec3& cam, const glm::vec3& sunL,
              const glm::vec3& sunColor, const glm::vec3& skyAmbient, const glm::vec3& landSun,
              const glm::vec3& landAmbient, const glm::vec3& waterLevel, float cloudShadow,
              float cloudTime, float cloudCover, float innerRadius);

    bool ready() const { return m_rings[0].haveMesh && m_rings[1].haveMesh; }
    // Moisture and temperature around the camera, red and green, for tinting the near world.
    unsigned int climateTexture() const { return m_climateTex; }
    // x, z of its corner, then 1 / span. z = 0 until there is one.
    glm::vec3 climateXf() const { return m_climateXf; }
    // Blocks inland, around the camera: what steers the breakers and the wash.
    unsigned int coastTexture() const { return m_rings[0].coastTex; }
    // Its corner relative to the camera, then 1 / span. z = 0 until there is one.
    glm::vec3 coastXf(const glm::dvec3& cam) const {
        const Ring& r = m_rings[0];
        return r.haveMesh ? glm::vec3(float(r.origin.x - cam.x), float(r.origin.y - cam.z),
                                      1.0f / float(kCells[0] * kSpacing[0]))
                          : glm::vec3(0.0f);
    }
    // Waits for a build in flight, so the generator it reads can be destroyed.
    void quiesce() { waitIdle(); }
    // A build is reading the generator right now: replacing it would have to wait.
    bool busy() const {
        for (const Ring& ring : m_rings)
            if (ring.jobRunning.load(std::memory_order_acquire))
                return true;
        return false;
    }

private:
    struct Vertex {
        float h;
        std::uint8_t r, g, b, water;
        std::int8_t nx, ny, nz;
        std::uint8_t ao;
    };
    struct Ring {
        unsigned int ssbo = 0;
        unsigned int heightTex = 0;
        unsigned int coastTex = 0;
        bool haveMesh = false;
        glm::dvec2 origin{0.0};   // world XZ of the grid's first vertex, as uploaded
        glm::dvec2 centre{1e18};  // where the uploaded mesh was centred

        // hand-off from the worker
        std::vector<Vertex> staging;
        std::vector<float> stagingHeights;
        std::vector<float> stagingCoast;
        std::vector<std::uint8_t> stagingClimate;
        glm::dvec2 stagingOrigin{0.0};
        std::atomic<bool> building{false};   // a build is owed or in flight, until update() consumes it
        std::atomic<bool> jobRunning{false}; // the worker is (or is about to be) inside build()
        std::atomic<bool> stagingReady{false};
        std::atomic<std::uint32_t> jobEpoch{0};
    };
    void build(const TerrainGenerator& gen, int ring, std::int64_t originX, std::int64_t originZ, JobSystem* jobs);
    void waitIdle();

    unsigned int m_prog = 0;
    unsigned int m_vao = 0;
    unsigned int m_climateTex = 0;
    glm::vec3 m_climateXf{0.0f};
    Ring m_rings[kRings];
    std::uint32_t m_seedEpoch = 0; // bumped by reset(), so a stale job's result is dropped
};

} // namespace vx
