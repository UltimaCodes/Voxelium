#include "ui/MenuBackdrop.hpp"

#include "gen/TerrainGenerator.hpp"
#include "render/Camera.hpp"
#include "render/FarTerrain.hpp"
#include "render/FrameRenderer.hpp"
#include "render/PostFx.hpp"
#include "render/Renderer.hpp"
#include "world/ChunkManager.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace vx {

MenuBackdrop::Vantage MenuBackdrop::findVantage(const TerrainGenerator& gen) {
    constexpr int kSea = kSeaLevelForSpawn;
    constexpr int kDirs = 12;
    constexpr int kRadii[3] = {450, 1000, 1800};
    Vantage best;
    float bestScore = -1e9f;
    for (int gz = -1400; gz <= 1400; gz += 200)
        for (int gx = -1400; gx <= 1400; gx += 200) {
            const int hc = gen.farSample(gx, gz).h;
            if (hc > 170)
                continue; // a summit has nothing to look up at
            // the highest ground close by: the camera has to clear it
            int nearMax = hc;
            for (int d = 0; d < kDirs; ++d) {
                const float a = float(d) * 6.2831853f / float(kDirs);
                for (const int r : {90, 220})
                    nearMax = std::max(nearMax, gen.farSample(gx + int(std::cos(a) * float(r)),
                                                               gz + int(std::sin(a) * float(r))).h);
            }
            int water = 0, samples = 0;
            float bestRise = 0.0f;
            int bestDir = 0;
            for (int d = 0; d < kDirs; ++d) {
                const float a = float(d) * 6.2831853f / float(kDirs);
                float rise = 0.0f;
                for (const int r : kRadii) {
                    const int h = gen.farSample(gx + int(std::cos(a) * float(r)),
                                                gz + int(std::sin(a) * float(r))).h;
                    ++samples;
                    if (h < kSea)
                        ++water;
                    rise = std::max(rise, float(h - std::max(hc, kSea)));
                }
                if (rise > bestRise) {
                    bestRise = rise;
                    bestDir = d;
                }
            }
            const float w = float(water) / float(samples);
            // some sea (about a third of what is around), and something tall to look at
            float score = 2.0f * std::max(0.0f, 1.0f - std::abs(w - 0.30f) / 0.30f) +
                          1.5f * std::min(bestRise / 220.0f, 1.0f);
            score -= 0.4f * std::max(0.0f, float(hc - 110) / 60.0f);
            if (score > bestScore) {
                bestScore = score;
                best.x = double(gx) + 0.5;
                best.z = double(gz) + 0.5;
                best.y = double(std::max(nearMax, kSea)) + 110.0;
                const float a = float(bestDir) * 6.2831853f / float(kDirs);
                // yaw 0 looks down -Z and turns right (+X)
                best.yaw = std::atan2(std::cos(a), -std::sin(a)) * 57.29578f;
            }
        }
    return best;
}

void MenuBackdrop::draw(RenderContext& rc, JobSystem& jobs, const TerrainGenerator& gen, int fbW,
                        int fbH, double time) {
    if (rc.farTerrain == nullptr || fbW <= 0 || fbH <= 0)
        return;
    if (!m_metaCleared) {
        // the cull pass reads a chunk list that has never been written; an empty one says so
        std::vector<ChunkDrawInfo> none;
        rc.renderer.updateChunkMeta(none);
        rc.renderer.updateWaterMeta(none);
        m_metaCleared = true;
    }

    // High enough over the ground to see out across it, and never under the sea; turning slowly
    // enough to read as a drift rather than a spin.
    if (!m_haveVantage || m_vantageSeed != gen.seed()) {
        m_vantage = findVantage(gen);
        m_vantageSeed = gen.seed();
        m_haveVantage = true;
    }
    // Hung at its vantage, swaying slowly across what is worth looking at instead of turning all
    // the way round (most of the way round is not).
    FlyCamera cam;
    cam.position = {m_vantage.x, m_vantage.y + 5.0 * std::sin(time * 0.11), m_vantage.z};
    cam.yawDeg = m_vantage.yaw + 32.0f * float(std::sin(time * 0.05));
    cam.pitchDeg = -6.0f + 1.5f * float(std::sin(time * 0.13));
    cam.fovYDeg = 64.0f;

    // Golden hour, drifting a little so the light is never quite the same twice. Only what shows
    // up in this view is left on: the sky and haze do the work, and everything that needs
    // chunks (shadows, AO, god rays) has nothing to act on.
    FxToggles fx;
    fx.taa = false;
    fx.motionBlur = false;
    fx.autoExposure = false;
    fx.dayCycle = false;
    fx.shadows = false;
    fx.ssao = false;
    fx.ssgi = false;
    fx.volumetric = false;
    fx.cloudShadow = 0.0f;
    fx.clouds = true;
    fx.fog = true;
    fx.bloom = true;
    fx.farTerrain = true;
    fx.autoWeather = false;
    fx.precip = 0.0f;
    fx.vignette = 0.32f;
    fx.crosshair[3] = 0.0f;
    fx.farInner = 0.0f; // the far terrain is the whole scene here, all the way in
    fx.exposure = 1.12f;
    fx.timeOfDay = 0.305f + 0.012f * float(std::sin(time * 0.05));

    rc.farTerrain->update(gen, jobs, cam.position);

    // The real world's streamed radius is where the far terrain takes over; here there is no
    // streamed world, so it takes over almost at once.
    constexpr float kInner = 1000.0f; // where the haze gets going (it starts at 55% of this)
    rc.renderer.beginFrame();
    renderFrame(rc, cam, fx, false, fbW, fbH, 0, float(time), nullptr, false, glm::vec3(0.0f),
                false, nullptr, kInner, -1, 0, FarTerrain::kFogEnd);
    rc.renderer.endFrame();
}

} // namespace vx
