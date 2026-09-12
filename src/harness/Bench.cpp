#include "harness/Harness.hpp"

#include "core/Bmp.hpp"
#include "core/Log.hpp"
#include "core/Paths.hpp"
#include "core/Settings.hpp"
#include "gfx/BlockTextures.hpp"
#include "gfx/GpuTimer.hpp"
#include "io/ChunkSerializer.hpp"
#include "io/RegionStore.hpp"
#include "io/WorldStore.hpp"
#include "jobs/JobSystem.hpp"
#include "light/LightEngine.hpp"
#include "mesh/GreedyMesher.hpp"
#include "mesh/NaiveMesher.hpp"
#include "platform/Window.hpp"
#include "render/Camera.hpp"
#include "render/FrameRenderer.hpp"
#include "render/Overlay.hpp"
#include "render/Particles.hpp"
#include "render/PostFx.hpp"
#include "render/Renderer.hpp"
#include "render/ShadowMap.hpp"
#include "ui/DebugHud.hpp"
#include "world/Chunk.hpp"
#include "world/ChunkManager.hpp"
#include "world/Player.hpp"
#include "world/Raycast.hpp"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace vx {

// Deterministic performance harness (--bench). The whole of Tier 1 is "make it faster",
// and the older in-smoke benchmark averages 32 frames while the world is still churning,
// which is why it could report negative deltas. This one settles the world completely
// first, holds a fixed camera and a fixed time of day, and reports the MEDIAN over many
// frames — medians shrug off the odd scheduling spike that wrecks a mean.
//
// TAA is off on purpose: its history makes consecutive frames cost different amounts,
// which is exactly the variance we're trying to measure through.
namespace {

// key=value, one per line: simple enough to eyeball in a diff, which a JSON regression
// baseline checked into (or attached to) a PR arguably wants to be.
constexpr const char* kBaselinePath = "bench_baseline.txt";

void loadBaseline(std::vector<std::pair<std::string, double>>& out) {
    std::ifstream in(kBaselinePath);
    std::string key;
    double val = 0.0;
    while (in >> key >> val)
        out.push_back({key, val});
}

double findBaseline(const std::vector<std::pair<std::string, double>>& base, const char* key) {
    for (const auto& [k, v] : base)
        if (k == key)
            return v;
    return -1.0;
}

// >8%: real GPU/scheduling noise on this rig is a couple percent run-to-run (the same
// reason the rest of the harness uses medians, not means); a regression worth flagging
// should clear that noise by a wide margin, not just tip over a hair-trigger threshold.
constexpr double kRegressionRatio = 1.08;

void reportAgainstBaseline(const std::vector<std::pair<std::string, double>>& base,
                           const char* key, double value) {
    const double baseline = findBaseline(base, key);
    if (baseline < 0.0)
        return; // no prior baseline for this metric — nothing to compare
    const double ratio = value / std::max(baseline, 1e-6);
    if (ratio > kRegressionRatio)
        VX_ERROR("bench REGRESSION: %s %.2f ms vs baseline %.2f ms (%.0f%% slower)", key, value,
                 baseline, 100.0 * (ratio - 1.0));
    else
        VX_INFO("bench baseline: %s %.2f ms vs %.2f ms (%.0f%% change)", key, value, baseline,
                100.0 * (ratio - 1.0));
}

} // namespace

int runBenchmark(RenderContext& rc, vx::ChunkManager& world, vx::JobSystem& jobs,
                 vx::LightEngine& light, vx::FlyCamera& camera, vx::Window& window, int radius,
                 bool saveBaseline) {
    struct View {
        const char* name;
        glm::dvec3 pos;
        float yaw, pitch;
        // Blocks above the terrain surface at pos.xz. Nonzero means pos.y is IGNORED and
        // the camera comes from the generator instead. The ground view was pinned at y=96;
        // #22 raised the terrain over it and it spent everything since sealed inside stone,
        // drawing zero chunks into a number that still counted toward the mean.
        int aboveSurface = 0;
    };
    // one looking across the surface (shadow + terrain heavy), one high looking down
    // (maximum chunk count in frame), one at ground level in trees (overdraw)
    const View views[] = {
        {"surface", {16.0, 110.0, 16.0}, -35.0f, -15.0f, 0},
        {"aerial", {16.0, 190.0, 16.0}, -35.0f, -55.0f, 0},
        // (96,96) was the old spot; #22's relief turned it into a 289-high peak, above the
        // snow line and treeless. This column measures 111 with 2515 leaves and 1162 cross
        // quads within 32 blocks — actual canopy overdraw, which is what the view is for.
        {"ground", {128.0, 0.0, 384.0}, 25.0f, -2.0f, 2},
    };

    world.radius = radius;
    int w = 0;
    int h = 0;
    window.framebufferSize(w, h);

    GLuint color = 0;
    glCreateRenderbuffers(1, &color);
    glNamedRenderbufferStorage(color, GL_RGB8, w, h);
    GLuint depth = 0;
    glCreateRenderbuffers(1, &depth);
    glNamedRenderbufferStorage(depth, GL_DEPTH_COMPONENT32F, w, h);
    GLuint fbo = 0;
    glCreateFramebuffers(1, &fbo);
    glNamedFramebufferRenderbuffer(fbo, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color);
    glNamedFramebufferRenderbuffer(fbo, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);

    vx::FxToggles fx;
    fx.taa = false;
    fx.dayCycle = false;
    fx.timeOfDay = 0.32f;

    FrameTimers ft;
    ft.shadow.init();
    ft.scene.init();
    ft.post.init();

    const auto median = [](std::vector<double>& v) {
        if (v.empty())
            return 0.0;
        const std::size_t mid = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + std::ptrdiff_t(mid), v.end());
        return v[mid];
    };

    VX_INFO("bench: %dx%d, radius %d, shadows+ssao+vol+bloom+ssgi+clouds on, taa off", w, h,
            radius);
    double totShadow = 0, totScene = 0, totPost = 0, totFrame = 0;
    double totScaledFrame = 0;
    double totMoveShadow = 0, totMoveFrame = 0, totMovePeak = 0;

    for (const View& v : views) {
        camera.position = v.pos;
        if (v.aboveSurface)
            camera.position.y =
                world.generator().surfaceHeight(int(v.pos.x), int(v.pos.z)) + v.aboveSurface;
        camera.yawDeg = v.yaw;
        camera.pitchDeg = v.pitch;

        // settle completely: every claim below is about steady-state cost, not streaming
        for (int i = 0; i < 40000; ++i) {
            window.pollEvents();
            rc.renderer.beginFrame();
            world.update(jobs, rc.renderer, light, camera.position);
            rc.renderer.endFrame();
            if (i > 200 && !world.busy() && light.idle())
                break;
        }
        std::vector<vx::ChunkDrawInfo> di;
        std::vector<vx::ChunkDrawInfo> wi;
        world.collectDrawInfo(di, wi);
        rc.renderer.updateChunkMeta(di);
        rc.renderer.updateWaterMeta(wi);
        world.updateVisibilityIfNeeded(rc.renderer, camera.position, true);

        constexpr int kWarm = 20;
        constexpr int kFrames = 140;
        std::vector<double> sh, sc, po, fr;
        sh.reserve(kFrames);
        for (int i = 0; i < kWarm + kFrames; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            rc.renderer.beginFrame();
            renderFrame(rc, camera, fx, true, w, h, fbo, 0.6f, &ft);
            rc.renderer.endFrame();
            glFinish(); // benchmark only: we want the whole frame's GPU work accounted for
            const double cpuMs =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
            if (i >= kWarm) {
                sh.push_back(ft.shadow.milliseconds());
                sc.push_back(ft.scene.milliseconds());
                po.push_back(ft.post.milliseconds());
                fr.push_back(cpuMs);
            }
        }
        const double mSh = median(sh), mSc = median(sc), mPo = median(po), mFr = median(fr);
        totShadow += mSh;
        totScene += mSc;
        totPost += mPo;
        totFrame += mFr;
        // drawn, not loaded: what survives frustum + cave culling is what an occlusion
        // pass would have left to remove, so this is the ceiling on #6
        VX_INFO("bench %-8s | shadow %5.2f | scene %5.2f | post %5.2f | frame %6.2f ms | "
                "%d of %d chunks drawn, %lld quads, cam y=%.0f",
                v.name, mSh, mSc, mPo, mFr, rc.renderer.gpuVisibleChunks(), int(di.size()),
                static_cast<long long>(world.stats().totalQuads), camera.position.y);
        // A view that draws nothing still contributes its ~1 ms to the mean, which reads as
        // a faster engine. That is how the ground view stayed dead unnoticed, so say it.
        if (rc.renderer.gpuVisibleChunks() == 0)
            VX_ERROR("bench %s drew NO chunks — the camera is sealed inside terrain and this "
                     "view measures nothing", v.name);

        // same view again at 70% internal resolution — shows what render scale buys
        vx::FxToggles scaled = fx;
        scaled.renderScale = 0.7f;
        std::vector<double> sh2, sc2, po2, fr2;
        for (int i = 0; i < kWarm + kFrames; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            rc.renderer.beginFrame();
            renderFrame(rc, camera, scaled, true, w, h, fbo, 0.6f, &ft);
            rc.renderer.endFrame();
            glFinish();
            const double cpuMs =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
            if (i >= kWarm) {
                sh2.push_back(ft.shadow.milliseconds());
                sc2.push_back(ft.scene.milliseconds());
                po2.push_back(ft.post.milliseconds());
                fr2.push_back(cpuMs);
            }
        }
        VX_INFO("   @0.70x  | shadow %5.2f | scene %5.2f | post %5.2f | frame %6.2f ms",
                median(sh2), median(sc2), median(po2), median(fr2));
        totScaledFrame += median(fr2);

        // Same view with the camera walking forward at ~8 units/s. A stationary camera is
        // the best possible case for cascade caching, so measuring only that would be
        // dishonest — this is what the cost actually looks like while you play.
        const glm::dvec3 startPos = camera.position;
        // fast flight, not a stroll: cascade 2 only re-anchors after ~68 units of travel,
        // so a slow walk over 140 frames would never trigger one and the test would prove
        // nothing.
        const glm::dvec3 step = glm::dvec3(camera.forward()) * (40.0 / 60.0);
        std::vector<double> sh3, fr3;
        for (int i = 0; i < kWarm + kFrames; ++i) {
            camera.position = startPos + step * double(i);
            const auto t0 = std::chrono::steady_clock::now();
            rc.renderer.beginFrame();
            renderFrame(rc, camera, fx, true, w, h, fbo, 0.6f, &ft);
            rc.renderer.endFrame();
            glFinish();
            const double cpuMs =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
            if (i >= kWarm) {
                sh3.push_back(ft.shadow.milliseconds());
                fr3.push_back(cpuMs);
            }
        }
        camera.position = startPos;
        // MEAN and MAX here, not median: caching trades a steady per-frame cost for a
        // periodic redraw, and a median would simply hide the spike that costs.
        double meanSh = 0.0, maxSh = 0.0, maxFr = 0.0;
        for (double v : sh3) {
            meanSh += v;
            maxSh = std::max(maxSh, v);
        }
        meanSh /= double(sh3.size());
        for (double v : fr3)
            maxFr = std::max(maxFr, v);
        VX_INFO("   moving   | shadow mean %5.2f peak %5.2f | frame med %6.2f peak %6.2f ms",
                meanSh, maxSh, median(fr3), maxFr);
        totMoveShadow += meanSh;
        totMoveFrame += median(fr3);
        totMovePeak = std::max(totMovePeak, maxFr);
    }

    const double n = double(std::size(views));
    VX_INFO("bench TOTAL   | shadow %5.2f | scene %5.2f | post %5.2f | frame %6.2f ms (mean "
            "of %d views)",
            totShadow / n, totScene / n, totPost / n, totFrame / n, int(n));
    VX_INFO("bench MOVING  | shadow mean %5.2f | frame med %6.2f | worst frame %6.2f ms",
            totMoveShadow / n, totMoveFrame / n, totMovePeak);
    VX_INFO("bench TOTAL   | render scale 0.70x -> frame %6.2f ms (%.0f%% of native)",
            totScaledFrame / n, 100.0 * (totScaledFrame / n) / std::max(totFrame / n, 1e-6));

    const std::vector<std::pair<std::string, double>> current = {
        {"frame_mean_ms", totFrame / n},
        {"moving_frame_median_ms", totMoveFrame / n},
        {"moving_frame_worst_ms", totMovePeak},
        {"scaled_0_7x_frame_ms", totScaledFrame / n},
    };
    if (saveBaseline) {
        std::ofstream out(kBaselinePath, std::ios::trunc);
        for (const auto& [key, value] : current)
            out << key << ' ' << value << '\n';
        VX_INFO("bench: saved baseline to %s", kBaselinePath);
    } else {
        std::vector<std::pair<std::string, double>> base;
        loadBaseline(base);
        if (base.empty())
            VX_INFO("bench: no %s yet — run with --savebaseline to create one", kBaselinePath);
        else
            for (const auto& [key, value] : current)
                reportAgainstBaseline(base, key.c_str(), value);
    }

    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &color);
    glDeleteRenderbuffers(1, &depth);
    return 0;
}

} // namespace vx
