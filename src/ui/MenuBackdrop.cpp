#include "ui/MenuBackdrop.hpp"

#include "gen/TerrainGenerator.hpp"
#include "render/Camera.hpp"
#include "render/FarTerrain.hpp"
#include "render/FrameRenderer.hpp"
#include "render/PostFx.hpp"
#include "render/Renderer.hpp"
#include "world/ChunkManager.hpp"
#include "jobs/JobSystem.hpp"

#include <glad/gl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

namespace vx {

MenuBackdrop::Vantage MenuBackdrop::findVantage(const TerrainGenerator& gen, JobSystem* jobs) {
    constexpr int kSea = kSeaLevelForSpawn;
    constexpr int kDirs = 12;
    constexpr int kRadii[3] = {450, 1000, 1800};
    // Every spot looked at comes off one coarse grid, read between its points. Asking the generator
    // column by column built a whole detailed chunk column per question, 14 000 of them: seconds.
    constexpr int kStep = 32, kReach = 1400 + 1800 + kStep, kN = 2 * kReach / kStep + 1;
    std::vector<TerrainGenerator::FarSample> grid(std::size_t(kN) * kN);
    gen.farGrid(-kReach, -kReach, kN, kStep, grid.data(), jobs);
    const auto heightAt = [&](int x, int z) {
        const float fx = float(x + kReach) / float(kStep), fz = float(z + kReach) / float(kStep);
        const int ix = std::clamp(int(fx), 0, kN - 2), iz = std::clamp(int(fz), 0, kN - 2);
        const float tx = fx - float(ix), tz = fz - float(iz);
        const auto at = [&](int i, int j) { return float(grid[std::size_t(j) * kN + std::size_t(i)].h); };
        const float a = at(ix, iz) + (at(ix + 1, iz) - at(ix, iz)) * tx;
        const float b = at(ix, iz + 1) + (at(ix + 1, iz + 1) - at(ix, iz + 1)) * tx;
        return int(std::lround(a + (b - a) * tz));
    };
    Vantage best;
    float bestScore = -1e9f;
    for (int gz = -1400; gz <= 1400; gz += 200)
        for (int gx = -1400; gx <= 1400; gx += 200) {
            const int hc = heightAt(gx, gz);
            if (hc > 170)
                continue; // a summit has nothing to look up at
            // the highest ground close by: the camera has to clear it
            int nearMax = hc;
            for (int d = 0; d < kDirs; ++d) {
                const float a = float(d) * 6.2831853f / float(kDirs);
                for (const int r : {90, 220})
                    nearMax = std::max(nearMax, heightAt(gx + int(std::cos(a) * float(r)), gz + int(std::sin(a) * float(r))));
            }
            int water = 0, samples = 0;
            float bestRise = 0.0f;
            int bestDir = 0;
            for (int d = 0; d < kDirs; ++d) {
                const float a = float(d) * 6.2831853f / float(kDirs);
                float rise = 0.0f;
                for (const int r : kRadii) {
                    const int h = heightAt(gx + int(std::cos(a) * float(r)), gz + int(std::sin(a) * float(r)));
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
                        int fbH, double time, const FxToggles* look) {
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
    // enough to read as a drift rather than a spin. Looked for on a worker, so the menu keeps
    // running while it is; black until then (the title fades in once the terrain is there).
    if (!m_haveVantage || m_vantageSeed != gen.seed() || m_vantageGen != gen.version()) {
        if (!busy()) {
            if (m_jobDone && m_jobSeed == gen.seed() && m_jobGen == gen.version()) {
                m_vantage = m_jobResult;
                m_vantageSeed = m_jobSeed;
                m_vantageGen = m_jobGen;
                m_haveVantage = true;
            } else {
                m_jobSeed = gen.seed();
                m_jobGen = gen.version();
                m_jobDone = false;
                m_jobRunning.store(true, std::memory_order_release);
                MenuBackdrop* self = this;
                const TerrainGenerator* g = &gen;
                JobSystem* js = &jobs;
                jobs.submit([self, g, js] {
                    self->m_jobResult = findVantage(*g, js);
                    self->m_jobDone = true;
                    self->m_jobRunning.store(false, std::memory_order_release);
                });
            }
        }
        if (!m_haveVantage || m_vantageSeed != gen.seed() || m_vantageGen != gen.version()) {
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            return;
        }
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
    if (look != nullptr) {
        // a loading screen shows the world as the game is about to: at its hour and its exposure,
        // not a bright morning that cuts to night
        fx.timeOfDay = look->timeOfDay;
        fx.exposure = look->exposure;
        fx.autoExposure = true;
        fx.eyeGain = 1.0f;
        fx.scotopic = look->scotopic;
        fx.saturation = look->saturation;
        fx.contrast = look->contrast;
        fx.temperature = look->temperature;
    }

    rc.farTerrain->update(gen, jobs, cam.position);

    // The real world's streamed radius is where the far terrain takes over; here there is no
    // streamed world, so it takes over almost at once.
    constexpr float kInner = 1000.0f; // where the haze gets going (it starts at 55% of this)
    LightVolume* const vol = rc.lightVolume;
    rc.lightVolume = nullptr; // no chunks here for it to light
    rc.renderer.beginFrame();
    renderFrame(rc, cam, fx, false, fbW, fbH, 0, float(time), nullptr, false, glm::vec3(0.0f),
                false, nullptr, kInner, -1, 0, FarTerrain::kFogEnd);
    rc.renderer.endFrame();
    rc.lightVolume = vol;
}

void MenuBackdrop::quiesce() const {
    while (busy())
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
}

} // namespace vx
