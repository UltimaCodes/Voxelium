#include "audio/Ambience.hpp"
#include "audio/AudioEngine.hpp"
#include "audio/PlayerSounds.hpp"
#include "core/AllocTracker.hpp"
#include "core/Bmp.hpp"
#include "core/CrashHandler.hpp"
#include "core/Log.hpp"
#include "core/Paths.hpp"
#include "core/Settings.hpp"
#include "gfx/BlockTextures.hpp"
#include "gfx/GlDebug.hpp"
#include "gfx/GpuTimer.hpp"
#include "gfx/Shader.hpp"
#include "harness/Harness.hpp"
#include "io/RegionStore.hpp"
#include "io/WorldStore.hpp"
#include "jobs/JobSystem.hpp"
#include "light/LightEngine.hpp"
#include "platform/Window.hpp"
#include "render/Camera.hpp"
#include "render/FrameRenderer.hpp"
#include "render/Overlay.hpp"
#include "render/Particles.hpp"
#include "render/PostFx.hpp"
#include "render/QualityPreset.hpp"
#include "render/Renderer.hpp"
#include "render/Weather.hpp"
#include "render/FarTerrain.hpp"
#include "render/Lightning.hpp"
#include "ui/MenuBackdrop.hpp"
#include "render/WeatherDirector.hpp"
#include "render/Wildlife.hpp"
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
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using vx::FrameTimers;
using vx::RenderContext;
using vx::TaaState;
using vx::renderFrame;
using vx::runBenchmark;
using vx::runFlythrough;
using vx::runMesherOracle;
using vx::runSmokeTest;
using vx::runTour;

int main(int argc, char** argv) {
    vx::installCrashHandler();
    bool smoke = false;
    bool bench = false;
    int benchRadius = 16;
    bool benchSaveBaseline = false;
    bool tour = false;
    int tourRadius = 16;
    bool tourSettle = true;
    bool flythrough = false;
    int flythroughFrames = 90;
    std::filesystem::path smokeOut = "smoke.bmp";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--smoke") == 0) {
            smoke = true;
            if (i + 1 < argc)
                smokeOut = argv[++i];
        } else if (std::strcmp(argv[i], "--tour") == 0) {
            tour = true;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                tourRadius = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--tourmoving") == 0) {
            tour = true;
            tourSettle = false;
        } else if (std::strcmp(argv[i], "--flythrough") == 0) {
            flythrough = true;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                flythroughFrames = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--bench") == 0) {
            bench = true;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                benchRadius = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--savebaseline") == 0) {
            benchSaveBaseline = true;
        } else if (std::strcmp(argv[i], "--crashtest") == 0) {
            // Verifies the crash handler actually writes a dump, rather than trusting that a
            // clean compile means a correct SetUnhandledExceptionFilter/MiniDumpWriteDump
            // call — the whole point of this file existing is #0's own rule against exactly
            // that kind of unverified claim.
            volatile int* p = nullptr;
            *p = 1;
        }
    }

#ifndef NDEBUG
    const bool debugContext = true;
#else
    const bool debugContext = smoke; // smoke runs want loud GL errors even in Release
#endif

    auto window = vx::Window::create({
        .width = 1280,
        .height = 720,
        .title = "Voxelium",
        .visible = !smoke && !bench && !tour && !flythrough,
        .debugContext = debugContext,
    });
    if (!window)
        return 1;

    vx::installGlDebugCallback();
    VX_INFO("GL %s | %s | %s",
            reinterpret_cast<const char*>(glGetString(GL_VERSION)),
            reinterpret_cast<const char*>(glGetString(GL_RENDERER)),
            reinterpret_cast<const char*>(glGetString(GL_VENDOR)));

    // Reversed-Z depth, set up once (rationale in render/Camera.hpp).
    glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);
    glClearDepth(0.0);
    glEnable(GL_CULL_FACE);

    const std::filesystem::path assets = vx::findAssetsDir(argv[0]);
    if (assets.empty()) {
        VX_ERROR("could not locate assets/ directory");
        return 1;
    }
    window->setIcon(assets);
    // Settings are read BEFORE any shader is built. The quality tier is injected into the
    // shader source, so the compile depends on it — reading settings afterwards would mean
    // building the whole set at the default tier and then throwing it away.
    //
    // Only fx is applied here; radius and fov land on `world` and `camera`, which do not
    // exist yet, so they are held and applied at their declarations below.
    vx::FxToggles fx;
    fx.taa = true; // temporal AA on by default in the live app
    bool vsync = false;
    bool caveCull = true;
    int savedRadius = 16;
    float savedFov = 70.0f;
    const std::filesystem::path settingsFile = "saves/settings.cfg";
    // A smoke/bench/tour/flythrough run renders from fixed defaults, never a machine's saved
    // settings, or the reference images would move under whoever happened to run it.
    const bool liveApp = !smoke && !bench && !tour && !flythrough;
    if (liveApp && vx::Settings::load(settingsFile, fx, vsync, caveCull, savedRadius, savedFov))
        VX_INFO("settings: loaded %s", settingsFile.string().c_str());

    // VX_PRESET=0..3 forces a quality preset. The harnesses otherwise render from fixed
    // defaults on purpose, but a preset table nobody has measured is a table of guesses —
    // this is what lets --bench and --tour be run at each tier without editing the source.
    // It applies to the live app too, which makes it the quickest way to reproduce a report
    // that only happens on Low.
    if (const char* pv = std::getenv("VX_PRESET")) {
        const int idx = std::atoi(pv);
        if (idx >= 0 && idx <= 3) {
            vx::applyPreset(vx::Preset(idx), fx, savedRadius);
            benchRadius = savedRadius;
            tourRadius = savedRadius;
            VX_INFO("preset: %s (radius %d, scale %.2f, tier %d, shadow %d)",
                    vx::presetName(vx::Preset(idx)), savedRadius, double(fx.renderScale),
                    fx.quality, fx.shadowRes);
        }
    }

    const std::filesystem::path shaders = assets / "shaders";
    // shared across runs including --smoke: this caches driver output keyed by source and
    // GL identity, not world state, so there is nothing here for a test run to poison
    vx::initShaderCache("saves/shadercache");
    // The four world programs, rebuilt whenever the quality tier changes. Submitted all at
    // once so the driver's compiler threads have the whole set to overlap (#15).
    struct WorldPrograms {
        GLuint draw = 0, cull = 0, shadow = 0, water = 0;
        bool ok() const { return draw && cull && shadow && water; }
    };
    const auto buildWorldPrograms = [&](std::string_view defines) {
        vx::ProgramJob jobs[] = {
            vx::submitProgram(shaders / "chunk.vert", shaders / "chunk.frag", defines),
            vx::submitComputeProgram(shaders / "cull.comp", defines),
            vx::submitProgram(shaders / "shadow.vert", shaders / "shadow.frag", defines),
            vx::submitProgram(shaders / "water.vert", shaders / "water.frag", defines),
        };
        WorldPrograms wp;
        wp.draw = vx::resolveProgram(jobs[0]);
        wp.cull = vx::resolveProgram(jobs[1]);
        wp.shadow = vx::resolveProgram(jobs[2]);
        wp.water = vx::resolveProgram(jobs[3]);
        return wp;
    };
    WorldPrograms world_p = buildWorldPrograms(vx::qualityDefines(fx.quality, fx.shadowRes));
    if (!world_p.ok())
        return 1;
    const GLuint cullProgram0 = world_p.cull;
    const GLuint drawProgram = world_p.draw;
    const GLuint shadowProgram = world_p.shadow;
    const GLuint waterProgram = world_p.water;

    if (!runMesherOracle())
        return 1;

    vx::Renderer renderer;
    if (!renderer.init(drawProgram, cullProgram0))
        return 1;
    vx::ShadowMap shadowMap;
    if (!shadowMap.init(shadowProgram, fx.shadowRes))
        return 1;
    vx::PostFx postfx;
    if (!postfx.init(shaders, vx::qualityDefines(fx.quality, fx.shadowRes)))
        return 1;
    vx::Overlay overlay;
    if (!overlay.init(shaders))
        return 1;
    vx::Particles particles;
    if (!particles.init(shaders))
        return 1;
    vx::Weather weather;
    if (!weather.init(shaders))
        return 1;
    vx::Wildlife wildlife;
    vx::WeatherDirector weatherDirector;
    vx::Lightning lightning;
    float stormLevel = 0.0f;
    vx::FarTerrain farTerrain;
    if (!farTerrain.init(shaders))
        return 1;
    if (!wildlife.init(shaders))
        return 1;
    vx::shutdownShaderCache(); // every program is built by here
    const GLuint blockTex = vx::createBlockTextureArray(assets / "textures");
    RenderContext rc{renderer,   shadowMap,    postfx,   overlay,  &particles,
                     &weather,   &wildlife,    &farTerrain, drawProgram, waterProgram, blockTex};

    // A quality change is the one setting that is not free: the sample counts are baked into the
    // programs, so the tier has to be recompiled rather than branched on. The binary cache makes
    // the second visit to a preset cheap (#15: 349 ms cold for the whole set, 56 ms warm), and
    // nothing about the world is touched - the arena, the meta buffers and every uploaded chunk
    // stay exactly where they are, so switching preset does not re-stream anything. Shared by the
    // settings window in the game and the one on the title screen.
    const auto applyQualityChange = [&](int qualityBefore, int shadowResBefore) {
        if (fx.quality == qualityBefore && fx.shadowRes == shadowResBefore)
            return;
        vx::initShaderCache("saves/shadercache");
        const std::string defines = vx::qualityDefines(fx.quality, fx.shadowRes);
        WorldPrograms next = buildWorldPrograms(defines);
        if (next.ok() && postfx.reloadPrograms(defines)) {
            glDeleteProgram(rc.drawProgram);
            glDeleteProgram(rc.waterProgram);
            renderer.setPrograms(next.draw, next.cull);
            rc.drawProgram = next.draw;
            rc.waterProgram = next.water;
            shadowMap.resize(next.shadow, fx.shadowRes);
            shadowMap.invalidate();
            postfx.invalidateTaaHistory();
            VX_INFO("quality: tier %d, shadow map %d", fx.quality, fx.shadowRes);
        } else {
            // Keep running on what already works rather than half-swapping the set.
            VX_ERROR("quality: rebuild failed, staying on tier %d", qualityBefore);
            fx.quality = qualityBefore;
            fx.shadowRes = shadowResBefore;
        }
        vx::shutdownShaderCache();
    };

    vx::DebugHud hud;
    if (!hud.init(window->handle()))
        return 1;
    window->hookScroll();

    // isolate the smoke test's save data so repeated runs start clean and can't touch a
    // real world save
    vx::WorldStore worldStore("saves");
    std::filesystem::path saveDir = worldStore.liveDir();
    // Made before the title screen: the terrain previewed behind it is built on these workers.
    auto jobs = std::make_unique<vx::JobSystem>();
    VX_INFO("job system: %d workers", jobs->workerCount());
    vx::MenuBackdrop menuBackdrop;

    // Every launch is a new world unless the title screen below loads one. The live one is
    // scratch either way — keeping it means saving it under a name from the ESC menu, which
    // is what makes "new seed each time" safe.
    int worldSeed = 20260721;
    if (smoke || bench || tour || flythrough) {
        // The harnesses assert against specific terrain (cave counts, ore counts, the glow
        // room, bench camera heights), so they pin the seed and never see the title screen —
        // a human has to be there to click it, which none of these runs have.
        saveDir = smoke ? smokeOut.parent_path() / "vxsave_smoke" : saveDir;
        if (smoke) {
            std::error_code ec;
            std::filesystem::remove_all(saveDir, ec);
        }
    } else {
        // Title screen: the only place a specific seed can be typed at all (the in-game
        // "new world" button has never taken one, only ever rolled a random one) and the
        // first thing shown, rather than always dropping straight into a fresh random world
        // the way every earlier launch did.
        vx::DebugHud::TitleUi title;
        title.saved = worldStore.list();
        const auto rollSeed = [] {
            return int(std::chrono::steady_clock::now().time_since_epoch().count() ^
                       std::chrono::system_clock::now().time_since_epoch().count()) &
                   0x7FFFFFFF;
        };
        // 0 for an empty or unreadable field; the caller falls back to the rolled seed
        const auto parseSeed = [](const char* text, int fallback) {
            if (!text[0])
                return fallback;
            const long long v = std::strtoll(text, nullptr, 10);
            return int((v < 0 ? -v : v) & 0x7FFFFFFF);
        };
        std::snprintf(title.seedText, sizeof(title.seedText), "%d", rollSeed());
        if (const char* fixed = std::getenv("VX_MENU_SEED")) // harness: a seed to look at, not a random one
            std::snprintf(title.seedText, sizeof(title.seedText), "%s", fixed);
        int rolledSeed = std::atoi(title.seedText);
        // The menu is a real scene, so it runs at the display's rate rather than flat out.
        window->setVsync(true);
        // The terrain behind the menu is the seed being shown: the one typed, or the world
        // selected on the saved page. Changing it waits for a pause in typing, then rebuilds.
        std::unique_ptr<vx::TerrainGenerator> previewGen;
        bool settingsPage = false;     // the settings window is showing
        vx::DebugHud::WorldUi noWorld; // the window wants one; there is no world to save or load yet
        int previewSeed = 0, pendingSeed = 0;
        double pendingSince = 0.0;
        double lastFrame = glfwGetTime();
        bool loaded = false;
        while (!window->shouldClose()) {
            window->pollEvents();
            const double nowT = glfwGetTime();
            const float dtT = float(std::min(nowT - lastFrame, 0.1));
            lastFrame = nowT;
            hud.beginFrame();
            hud.titleScreen(title);
            if (title.page == 3) {
                bool open = true, unusedQuit = false;
                const int qualityBefore = fx.quality;
                const int shadowResBefore = fx.shadowRes;
                hud.settings(open, vsync, caveCull, savedRadius, savedFov, fx, unusedQuit, noWorld,
                             /*titleMode=*/true);
                applyQualityChange(qualityBefore, shadowResBefore);
                if (!open)
                    title.page = 1;
                settingsPage = true;
            } else if (settingsPage) {
                // leaving the page commits what was changed, the same as closing it in the game
                vx::Settings::save(settingsFile, fx, vsync, caveCull, savedRadius, savedFov);
                settingsPage = false;
            }
            if (title.rerollRequested) {
                title.rerollRequested = false;
                rolledSeed = rollSeed();
                std::snprintf(title.seedText, sizeof(title.seedText), "%d", rolledSeed);
            }
            if (title.deleteRequested) {
                title.deleteRequested = false;
                if (!title.selected.empty())
                    worldStore.remove(title.selected);
                title.saved = worldStore.list();
                title.selected.clear();
                title.deleteArmed.clear();
                if (title.saved.empty())
                    title.page = 1;
                else
                    title.selected = title.saved.front().name;
            }

            int wantSeed = parseSeed(title.seedText, rolledSeed);
            if (title.page == 2)
                for (const vx::WorldStore::Entry& e : title.saved)
                    if (e.name == title.selected)
                        wantSeed = e.seed;
            if (!previewGen || wantSeed != previewSeed) {
                if (wantSeed != pendingSeed || !previewGen) {
                    pendingSeed = wantSeed;
                    pendingSince = nowT;
                }
                if (!previewGen || nowT - pendingSince > 0.35) {
                    farTerrain.reset(); // waits for a build that is still reading the old one
                    previewGen = std::make_unique<vx::TerrainGenerator>(wantSeed);
                    previewSeed = wantSeed;
                }
            }
            // black while the terrain is being built, easing out once it is there
            title.fade = farTerrain.ready() ? std::max(0.0f, title.fade - dtT * 2.2f) : 1.0f;

            int fbw = 0, fbh = 0;
            window->framebufferSize(fbw, fbh);
            if (fbw > 0 && fbh > 0) {
                glViewport(0, 0, fbw, fbh);
                if (previewGen)
                    menuBackdrop.draw(rc, *jobs, *previewGen, fbw, fbh, nowT);
                else {
                    glClearColor(0.07f, 0.09f, 0.14f, 1.0f);
                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                }
            }
            hud.render();
            // VX_MENU_SHOT=<frame>[,<page>] writes menu_shot.bmp on that frame and exits: the only
            // offscreen look at the title screen (VX_MENU_PAGE=2 opens the saved-worlds page).
            if (const char* ms = std::getenv("VX_MENU_SHOT")) {
                static int menuFrame = 0;
                if (++menuFrame == std::atoi(ms) && std::atoi(ms) > 0 && fbw > 0 && fbh > 0) {
                    std::vector<std::uint8_t> px(std::size_t(fbw) * fbh * 3);
                    glReadPixels(0, 0, fbw, fbh, GL_BGR, GL_UNSIGNED_BYTE, px.data());
                    vx::writeBmp24("menu_shot.bmp", fbw, fbh, px);
                    farTerrain.quiesce();
                    hud.shutdown();
                    return 0;
                }
                if (menuFrame == 5 && std::getenv("VX_MENU_PLAY"))
                    title.newWorldRequested = true; // harness: carry on into the loading screen
                if (menuFrame == 1 && std::getenv("VX_MENU_PAGE")) {
                    title.page = std::atoi(std::getenv("VX_MENU_PAGE"));
                    if (title.page == 2 && !title.saved.empty())
                        title.selected = title.saved.front().name;
                }
            }
            window->swapBuffers();
            if (title.quitRequested) {
                farTerrain.quiesce();
                hud.shutdown();
                return 0;
            }
            if (title.newWorldRequested) {
                if (settingsPage)
                    vx::Settings::save(settingsFile, fx, vsync, caveCull, savedRadius, savedFov);
                worldSeed = parseSeed(title.seedText, rolledSeed);
                worldStore.clearLive();
                break;
            }
            if (title.loadRequested && worldStore.load(title.selected, worldSeed)) {
                loaded = true;
                break;
            }
            title.loadRequested = false; // load() failed (deleted mid-screen?) — stay put
        }
        // the preview generator is about to go; a build still reading it must finish first. The
        // mesh it made is kept: the world about to start has the same seed, so it is the same
        // terrain and the loading screen and the game carry on from it.
        farTerrain.quiesce();
        previewGen.reset();
        if (window->shouldClose())
            return 0;
        VX_INFO(loaded ? "world loaded, seed %d" : "world seed %d (save it from the ESC "
                                                    "menu to keep it)",
                worldSeed);
    }

    // declaration order matters: jobs is destroyed first, joining all workers, so no
    // job can touch world's chunks or queues during teardown
    auto world = std::make_unique<vx::ChunkManager>(saveDir, worldSeed);
    auto light = std::make_unique<vx::LightEngine>(*world);

    vx::FlyCamera camera;
    // Spawn ON the ground, not at a fixed altitude. y was hardcoded to 110, which is inside
    // the rock anywhere the terrain rises above it — and with exponential relief putting
    // peaks past 300, that is a lot of the world. surfaceHeight is the generator's own
    // answer for this column, so it lands on whatever is actually there.
    {
        constexpr int kSpawnX = 16, kSpawnZ = 16;
        const int ground = world->generator().surfaceHeight(kSpawnX, kSpawnZ);
        camera.position = {double(kSpawnX) + 0.5,
                           double(std::max(ground, vx::kSeaLevelForSpawn) + 3),
                           double(kSpawnZ) + 0.5};
    }
    camera.yawDeg = -35.0f;
    camera.pitchDeg = -25.0f;
    // Applied HERE, before the initial settle below: world and camera both already exist
    // (constructed above), and the saved radius/fov must be live BEFORE the loading screen
    // decides how much world to stream and light. Applying it after the settle (as this
    // used to) meant the first settle always ran at ChunkManager's built-in default of 16
    // regardless of what was saved or what preset was picked — so choosing Ultra (radius 32)
    // loaded and lit a radius-16 world behind the loading screen, then silently doubled the
    // streaming target the instant gameplay started. That is the exact "chunks don't render
    // past spawn, and it's laggy" report: the game never finished loading, it just stopped
    // showing a loading screen.
    if (liveApp) {
        world->radius = savedRadius;
        camera.fovYDeg = savedFov;
    }

    if (tour) {
        const int rcT = runTour(rc, *world, *jobs, *light, camera, *window, tourRadius,
                                tourSettle);
        world->saveDirty();
        return rcT;
    }
    if (flythrough) {
        const int rcF = runFlythrough(rc, *world, *jobs, *light, camera, *window,
                                      flythroughFrames);
        world->saveDirty();
        return rcF;
    }
    if (bench) {
        const int rcB = runBenchmark(rc, *world, *jobs, *light, camera, *window, benchRadius,
                                     benchSaveBaseline);
        hud.shutdown();
        particles.shutdown();
        wildlife.shutdown();
        farTerrain.shutdown();
        overlay.shutdown();
        postfx.shutdown();
        shadowMap.shutdown();
        renderer.shutdown();
        return rcB;
    }

    if (smoke) {
        const int rc2 = runSmokeTest(rc, *world, *jobs, *light, camera, hud, *window, smokeOut,
                                    assets.parent_path() / "sounds");
        hud.shutdown();
        particles.shutdown();
        wildlife.shutdown();
        farTerrain.shutdown();
        overlay.shutdown();
        postfx.shutdown();
        shadowMap.shutdown();
        renderer.shutdown();
        return rc2;
    }

    FrameTimers timers;
    timers.shadow.init();
    timers.scene.init();
    timers.post.init();

    std::vector<vx::ChunkDrawInfo> drawInfo;
    std::vector<vx::ChunkDrawInfo> waterInfo;
    vx::Player player;
    TaaState taaState;
    // Settle the world before the player ever sees it. Streaming in front of someone is
    // what makes LOD swaps, shadow cascades and chunk pop-in obvious; doing it behind a
    // panel means the first real frame already has all three in place. Bounded, because a
    // world that never settles must not be an infinite loading screen — if the budget runs
    // out we start anyway and it streams as it always did.
    // Settle the world behind the panel. Used at startup AND for every world change: before,
    // only startup did this, so "reset world" dropped you into an empty world that then
    // generated itself around you — every chunk, mesh and light settle competing with the
    // frames you were trying to play, which is exactly the stall the loading screen exists
    // to hide.
    const auto settleBehindLoadingScreen = [&](const char* title) {
        constexpr int kMaxFrames = 4000;
        // The real figure, summed from columnRange. It used to be radius*radius*
        // kWorldChunksY, which at radius 16 is 3072 against ~15200 actually meshed — so the
        // bar hit 100% about a fifth of the way in and sat there.
        const int target = world->expectedMeshedChunks();
        // Monotonic, and capped below 1 until the world really is settled. The estimate is
        // a few percent high (it counts chunks in radius that end up meshing to nothing),
        // so left alone the bar would stall near the end; the completed frame after the
        // loop is what fills it. A bar that goes backwards or sits at 100% while you wait
        // is worse than a slightly conservative one.
        float shown = 0.0f;
        for (int f = 0; f < kMaxFrames; ++f) {
            window->pollEvents();
            if (window->shouldClose())
                break;
            renderer.beginFrame();
            world->update(*jobs, renderer, *light, camera.position);
            renderer.endFrame();
            if (f > 30 && !world->busy() && light->idle())
                break;

            int fbw = 0, fbh = 0;
            window->framebufferSize(fbw, fbh);
            if (fbw > 0 && fbh > 0) {
                // refreshStats, not stats: the cached scan only refreshes every 20th call,
                // so the bar would step in visible jumps and lag what is really happening.
                world->refreshStats();
                const vx::ChunkManager::Stats ls = world->stats();
                shown = std::max(shown, std::min(float(ls.meshed) / float(target), 0.97f));
                const float p = shown;
                char phase[96];
                if (ls.genQueue > 0)
                    std::snprintf(phase, sizeof(phase), "generating terrain - %d chunks left",
                                  ls.genQueue);
                else if (ls.unlit > 0)
                    std::snprintf(phase, sizeof(phase), "settling light - %d columns left",
                                  ls.unlit);
                else
                    std::snprintf(phase, sizeof(phase), "building meshes - %d of %d",
                                  ls.meshed, target);
                hud.beginFrame();
                hud.loadingScreen(p, title, phase);
                glViewport(0, 0, fbw, fbh);
                menuBackdrop.draw(rc, *jobs, world->generator(), fbw, fbh, glfwGetTime());
                hud.render();
                // VX_LOAD_SHOT=<n> writes load_shot.bmp on the n-th loading frame and exits
                if (const char* ls = std::getenv("VX_LOAD_SHOT"); ls != nullptr && f == std::atoi(ls)) {
                    std::vector<std::uint8_t> px(std::size_t(fbw) * fbh * 3);
                    glReadPixels(0, 0, fbw, fbh, GL_BGR, GL_UNSIGNED_BYTE, px.data());
                    vx::writeBmp24("load_shot.bmp", fbw, fbh, px);
                    std::exit(0);
                }
                window->swapBuffers();
            }
        }
        world->collectDrawInfo(drawInfo, waterInfo);
        renderer.updateChunkMeta(drawInfo);
        renderer.updateWaterMeta(waterInfo);
        shadowMap.invalidate();
        // One completed frame, so the bar lands on 100% instead of cutting away at 97 —
        // and the first real frame still has to build its shadow cascades, so this is not
        // dead time either.
        {
            int fbw = 0, fbh = 0;
            window->framebufferSize(fbw, fbh);
            if (fbw > 0 && fbh > 0) {
                hud.beginFrame();
                hud.loadingScreen(1.0f, title, "ready");
                glViewport(0, 0, fbw, fbh);
                menuBackdrop.draw(rc, *jobs, world->generator(), fbw, fbh, glfwGetTime());
                hud.render();
                window->swapBuffers();
            }
        }
        world->refreshStats();
        VX_INFO("world ready: %d chunks meshed of %d expected (seed %d)",
                world->stats().meshed, target, world->seed());
    };
    settleBehindLoadingScreen("voxelium");

    double prevTime = glfwGetTime();
    double physAccum = 0.0;
    bool f3Held = false;
    bool f2Held = false;
    bool f5Held = false;
    bool mmbHeld = false;
    bool walkMode = false;
    // sounds/ next to the assets dir; nothing ships in it, the user fills it themselves
    vx::AudioEngine audio;
    // Decoded up front on the workers. Every clip is held as PCM for the process lifetime,
    // which is the point: a play used to open the file and decode it whole, on this thread.
    audio.init(assets.parent_path() / "sounds", jobs.get());
    vx::PlayerSounds playerSounds;
    vx::AmbienceState ambience;
    vx::StormSounds storm;
    storm.selfSchedule = false; // thunder now follows the lightning, after its own delay
    // How blocked the path from a sound to the ear is, 0 clear .. 1 solid. One DDA per sound
    // EVENT, which is far cheaper than it sounds — the crosshair already runs this same
    // raycast every single frame. A break heard through a wall should not arrive at full
    // volume, and this is the only thing that distinguishes the two.
    //
    // Four tiers, narrowest first: this exact block, its material family, the generic clip,
    // and finally the material's own FOOTSTEP at a shifted pitch. The material tier is the
    // one that matters — block sounds are keyed by numeric id, and this engine stores data in
    // the id, so water is seven ids and a lamp is twenty-five. Without it, per-block audio
    // meant twenty-five copies of one file and nearly everything fell through to generic.
    const auto audioHit = [](vx::AudioEngine& au, const char* kind, int blockId,
                             const glm::dvec3& at, float occ, float pitch, float vol) {
        const std::string k = kind;
        const char* mat = vx::blocks::soundMaterial(vx::BlockId(blockId));
        for (const std::string& name :
             {k + "_" + std::to_string(blockId), k + "_" + mat, k}) {
            if (au.has(name)) {
                au.playOccluded(name, at, occ, vol, 1.0f);
                return;
            }
        }
        // Nothing supplied for this action at all. A footstep on stone and a stone being
        // struck are the same material making the same noise, so a pitch shift stands in
        // rather than leaving the most common interaction in the engine silent.
        for (const std::string& name :
             {"step_" + std::to_string(blockId), "step_" + std::string(mat), std::string("step")}) {
            if (au.has(name)) {
                au.playOccluded(name, at, occ, vol, pitch);
                return;
            }
        }
    };

    const auto occlusionTo = [&](const glm::dvec3& src) {
        const glm::dvec3 toEar = camera.position - src;
        const double dist = glm::length(toEar);
        if (dist < 1.5 || dist > 48.0)
            return 0.0f; // close enough to be direct, or already past hearing range
        const auto hit = vx::raycastVoxels(src, glm::vec3(toEar / dist), float(dist),
                                           [&](const glm::ivec3& p) {
                                               return vx::blocks::occludes(
                                                   world->blockAtWorld(p));
                                           });
        return hit.hit ? fx.audioOcclusion : 0.0f;
    };
    window->setVsync(vsync);
    bool loadLogged = false;
    bool everBusy = false; // don't log the "loaded" summary before streaming even starts
    bool f4Held = false;
    bool f6Held = false;
    bool takeScreenshot = false;
    bool hotbarOpen = false;
    bool escHeld = false;
    bool settingsOpen = false;
    vx::DebugHud::WorldUi worldUi;
    // Lamp picker: opened by middle-clicking a lamp, and it remembers which one so the
    // choice can be applied to that block as well as to everything placed afterwards.
    bool lampPickerOpen = false;
    bool cursorCaptured = false; // mirrors what we last told the window, so we only tell it on change
    glm::ivec3 lampPickerTarget{0};
    // Hold-to-break. Progress is per TARGET, not per click: looking away resets it, so you
    // cannot chip at one block, switch, and come back to a half-broken one.
    glm::ivec3 breakTarget{INT32_MIN, 0, 0};
    float breakProgress = 0.0f;
    bool breaking = false;
    double digClock = 0.0; // paces the chipping sound while a break is held
    constexpr int kHotbar[] = {vx::blocks::Stone,  vx::blocks::Dirt,   vx::blocks::Grass,
                               vx::blocks::Sand,   vx::blocks::Snow,   vx::blocks::Wood,
                               vx::blocks::Planks, vx::blocks::Bricks, vx::blocks::Leaves,
                               vx::blocks::Gravel, vx::blocks::Glow,   vx::blocks::Water,
                               // one Lamp entry; the hue comes from the settings colour and
                               // is resolved at placement time, so the bar stays a bar
                               vx::blocks::LampFirst};
    constexpr int kHotbarCount = int(sizeof(kHotbar) / sizeof(kHotbar[0]));
    int hotbarSel = 0;
    int placeBlock = kHotbar[hotbarSel];
    std::uint64_t allocsPrevFrame = vx::heapAllocCount();

    while (!window->shouldClose()) {
        window->pollEvents();
        // BEFORE anything can play. These used to sit near the end of the frame, after the
        // physics loop had already fired footsteps — so every sound was positioned against
        // the PREVIOUS frame's listener, and on the very first frame against a listener
        // still at the world origin. A footstep a hundred blocks from the ear is attenuated
        // to nothing, which is a silent first step and a subtly wrong position after that.
        audio.setListener(camera.position, camera.forward());
        audio.setMasterVolume(fx.volume);
        audio.setBusVolume(vx::AudioEngine::Bus::World, fx.volWorld);
        audio.setBusVolume(vx::AudioEngine::Bus::Ui, fx.volUi);
        audio.setBusVolume(vx::AudioEngine::Bus::Ambient, fx.volAmbient);
        const bool esc = window->keyDown(GLFW_KEY_ESCAPE);
        if (esc && !escHeld) {
            settingsOpen = !settingsOpen; // ESC pauses into the settings menu, not quit
            if (!settingsOpen) // closing the menu commits whatever was just changed
                vx::Settings::save(settingsFile, fx, vsync, caveCull, world->radius,
                                   camera.fovYDeg);
        }
        escHeld = esc;
        const bool f3 = window->keyDown(GLFW_KEY_F3);
        if (f3 && !f3Held)
            hud.toggle();
        f3Held = f3;
        const bool f2 = window->keyDown(GLFW_KEY_F2);
        if (f2 && !f2Held)
            world->saveDirty(); // save edited chunks to disk
        f2Held = f2;
        const bool f5 = window->keyDown(GLFW_KEY_F5);
        if (f5 && !f5Held) {
            walkMode = !walkMode;
            if (walkMode) {
                player.position = camera.position - glm::dvec3(0.0, vx::Player::kEyeHeight, 0.0);
                player.velocity = {0.0, 0.0, 0.0};
                physAccum = 0.0;
                // The player was placed here, not dropped here. Without this the first
                // landing is measured from however high the camera had flown.
                playerSounds.reset(player);
            }
        }
        f5Held = f5;
        const bool f6 = window->keyDown(GLFW_KEY_F6);
        if (f6 && !f6Held)
            takeScreenshot = true; // consumed after this frame renders, see below endFrame()
        f6Held = f6;
        const bool f4 = window->keyDown(GLFW_KEY_F4);
        if (f4 && !f4Held)
            hotbarOpen = !hotbarOpen;
        f4Held = f4;
        if (hotbarOpen && !settingsOpen) {
            for (int k = 0; k < kHotbarCount; ++k)
                if (window->keyDown(GLFW_KEY_1 + k))
                    hotbarSel = k;
            // scroll cycles slots while the hotbar is up (drained here so it doesn't also
            // change fly speed); when it's hidden the camera gets the scroll for speed
            const double sc = window->consumeScrollY();
            if (sc != 0.0)
                hotbarSel = (hotbarSel - (sc > 0.0 ? 1 : -1) + kHotbarCount) % kHotbarCount;
        }
        placeBlock = kHotbar[hotbarSel];

        const double now = glfwGetTime();
        const float frameMs = float((now - prevTime) * 1000.0);
        const float dt = std::min(float(now - prevTime), 0.1f);
        prevTime = now;
        if (fx.dayCycle && !settingsOpen) { // advance the day, wrap to [0,1)
            fx.timeOfDay += dt / std::max(fx.dayLengthSec, 1.0f);
            fx.timeOfDay -= std::floor(fx.timeOfDay);
        }

        renderer.beginFrame();
        hud.beginFrame();
        // One authority on who owns the mouse: the window is focused and no menu is up.
        // Menus are the ONLY thing that hands the cursor back — alt-tabbing counts because
        // focus is gone. Everything below reads this rather than testing buttons itself.
        const bool menuOpen = settingsOpen || lampPickerOpen;
        const bool mouseOurs = window->focused() && !menuOpen && !hud.wantsMouse();
        if (mouseOurs != cursorCaptured) {
            cursorCaptured = mouseOurs;
            window->setCursorCaptured(mouseOurs);
        }

        if (settingsOpen || lampPickerOpen) {
            // paused in a menu: freeze look/move/edit, keep streaming + drawing
        } else if (walkMode) {
            if (mouseOurs)
                camera.lookOnly(*window);
            physAccum += double(dt);
            while (physAccum >= vx::Player::kTickDt) {
                vx::PlayerInput in;
                in.yawDeg = camera.yawDeg;
                if (window->keyDown(GLFW_KEY_W)) in.forward += 1.0f;
                if (window->keyDown(GLFW_KEY_S)) in.forward -= 1.0f;
                if (window->keyDown(GLFW_KEY_D)) in.strafe += 1.0f;
                if (window->keyDown(GLFW_KEY_A)) in.strafe -= 1.0f;
                in.jump = window->keyDown(GLFW_KEY_SPACE);
                const glm::dvec3 before = player.position;
                player.step(*world, in);
                physAccum -= vx::Player::kTickDt;
                // Footsteps, jumps, landings, splashes and swim strokes all live in
                // PlayerSounds because every one of them is an edge that has to be
                // remembered across ticks. It hands back what it fired so the matching
                // particle still spawns here, where the particle system is.
                const vx::PlayerSounds::Event se =
                    playerSounds.tick(audio, *world, player, before);
                if (se.stepped) {
                    const double top = std::floor(player.position.y);
                    if (vx::blocks::opaque(vx::BlockId(se.blockId))) {
                        if (se.landed)
                            particles.spawnLanding({se.at.x, top, se.at.z},
                                                   vx::blockTopLayer(se.blockId), se.blockId);
                        else
                            particles.spawnFootstep({se.at.x, top, se.at.z},
                                                    vx::blockTopLayer(se.blockId), se.blockId);
                    } else if (vx::blocks::isWater(vx::BlockId(se.blockId)))
                        particles.spawnSplash({se.at.x, top, se.at.z});
                }
            }
            camera.position = player.eye();
        } else {
            camera.update(*window, dt, hud.wantsMouse(), hud.wantsKeyboard(), mouseOurs);
        }

        if (!settingsOpen && !lampPickerOpen && camera.flying()) {
            const bool lmb = window->mouseDown(GLFW_MOUSE_BUTTON_LEFT);
            const bool mmb = window->mouseDown(GLFW_MOUSE_BUTTON_MIDDLE);
            const auto hit = vx::raycastVoxels(
                camera.position, camera.forward(), 6.0, [&](const glm::ivec3& p) {
                    const vx::BlockId b = world->blockAtWorld(p);
                    return vx::blocks::opaque(b) || vx::blocks::isCross(b);
                });

            // Breaking is held, not clicked. Anything that changes what you are aiming at
            // -- or letting go -- drops the progress, which is what makes the crack read as
            // damage to THAT block rather than to a global timer.
            breaking = false;
            if (lmb && hit.hit) {
                if (hit.voxel != breakTarget) {
                    breakTarget = hit.voxel;
                    breakProgress = 0.0f;
                }
                const vx::BlockId id = world->blockAtWorld(hit.voxel);
                breakProgress += float(dt) / std::max(0.05f, vx::blocks::hardness(id));
                breaking = true;
                // Chipping away is audible, not just the moment it gives. Without this,
                // holding the button on stone is a second and a half of silence followed by
                // one crack, which reads as a delay rather than as work.
                //
                // Nothing supplies a "dig" clip, and that is fine: audioHit falls through to
                // the material's own footstep, pitched, which is exactly what this should be.
                // Dropping dig_stone.wav into sounds/ later takes over with no code change.
                digClock += dt;
                if (digClock > 0.28) {
                    digClock = 0.0;
                    audioHit(audio, "dig", int(id), glm::dvec3(hit.voxel),
                             occlusionTo(glm::dvec3(hit.voxel)), 1.15f, 0.22f);
                }
                if (breakProgress >= 1.0f) {
                    const glm::dvec3 at(hit.voxel);
                    particles.spawnBlockBreak(at, vx::blockTopLayer(int(id)), int(id));
                    world->setBlock(renderer, *light, hit.voxel, vx::blocks::Air);
                    const float occ = occlusionTo(at);
                    audioHit(audio, "break", int(id), at, occ, 0.72f, 1.0f);
                    breakProgress = 0.0f;
                    breakTarget = {INT32_MIN, 0, 0};
                    breaking = false;
                }
            } else {
                breakProgress = 0.0f;
                breakTarget = {INT32_MIN, 0, 0};
                // Let go and the next press chips immediately rather than waiting out
                // whatever was left of the last interval.
                digClock = 1.0;
            }

            if (mmb && !mmbHeld && hit.hit) {
                const glm::dvec3 at(hit.voxel);
                const vx::BlockId aimed = world->blockAtWorld(hit.voxel);
                if (vx::blocks::isLamp(aimed)) {
                    // Aiming at a lamp means you want to recolour that lamp, not stack
                    // another onto its face. The repaint happens when a swatch is chosen.
                    lampPickerOpen = true;
                    lampPickerTarget = hit.voxel;
                } else {
                    const int put = placeBlock == vx::blocks::LampFirst
                                        ? int(vx::blocks::LampFirst + fx.lampSwatch)
                                        : placeBlock;
                    world->setBlock(renderer, *light, hit.voxel + hit.normal,
                                    vx::BlockId(put));
                    const float occ = occlusionTo(at);
                    // Same chain, pitched UP: setting a block down is a lighter sound than
                    // breaking one.
                    audioHit(audio, "place", put, at, occ, 1.25f, 0.85f);
                }
            }
            mmbHeld = mmb;
        }

        // eye submerged -> underwater fog/tint in the composite pass, and the same flag
        // muffles the world bus and swaps the ambience bed. Computed here rather than next
        // to the render call because everything below wants it in the SAME frame the camera
        // moved, not one behind.
        const bool eyeUnderwater =
            vx::blocks::isWater(world->blockAtWorld(glm::ivec3(glm::floor(camera.position))));
        audio.setUnderwater(eyeUnderwater);
        playerSounds.frame(audio, camera.position, eyeUnderwater, float(dt));

        // What the world sounds like from here. Sky light answers "how enclosed am I" for
        // free — the light BFS has already done that work, so this is one chunk lookup
        // rather than a cover raycast every frame.
        {
            const glm::ivec3 ear(glm::floor(camera.position));
            ambience.skyAccess = float(world->skyLightAtWorld(ear)) / 15.0f;
            ambience.depth =
                float(world->generator().surfaceHeight(ear.x, ear.z) - ear.y);
            // Same function the frame shades from, so dusk sounds like dusk looks rather
            // than crossing over at its own private time of day.
            ambience.daylight = vx::computeSky(fx.timeOfDay).daylight;
            // Auto weather runs BEFORE anything reads precip below (ambience, storm audio,
            // wildlife and the next render), so sound, creatures and picture agree in-frame.
            // desert/coast are last frame's, which is plenty for a quantity that moves over
            // minutes.
            if (fx.autoWeather) {
                const auto w = weatherDirector.update(float(dt), float(camera.position.y),
                                                      ambience.desert, ambience.coast,
                                                      fx.timeOfDay);
                fx.precip = w.precip;
                fx.snow = w.snow;
                stormLevel = w.storm;
                fx.cloudBias = w.cloudBias;
                fx.mist = w.mist;
            } else {
                fx.cloudBias = 0.0f;
                fx.mist = 0.0f;
                // set by hand, a hard enough downpour still storms
                stormLevel = std::clamp((fx.precip - 0.7f) / 0.3f, 0.0f, 1.0f) * (1.0f - fx.snow);
            }
            ambience.precip = fx.precip;
            ambience.underwater = eyeUnderwater;
            // A coarse count of the water around the ear, so standing by a lake is audibly
            // different from standing in a field. Sampled on a lattice rather than swept:
            // this runs every frame and the answer only has to be roughly right.
            int wet = 0;
            for (int dy = -4; dy <= 4; dy += 4)
                for (int dz = -6; dz <= 6; dz += 6)
                    for (int dx = -6; dx <= 6; dx += 6)
                        wet += vx::blocks::isWater(
                                   world->blockAtWorld(ear + glm::ivec3(dx, dy, dz)))
                                   ? 1
                                   : 0;
            ambience.nearWater = float(wet) / 27.0f;
            // Where the ear is standing: the generator's biome function at the ear and four
            // points around it (so a shoreline is heard before you are on it), plus a lattice
            // count of what is physically here - leaves for the canopy, coral for the reef.
            {
                constexpr int kSpread = 24;
                const int offs[5][2] = {{0, 0}, {kSpread, 0}, {-kSpread, 0}, {0, kSpread},
                                        {0, -kSpread}};
                float coast = 0.0f, desert = 0.0f;
                for (const auto& o : offs) {
                    const auto b = world->generator().biomeAt(ear.x + o[0], ear.z + o[1]);
                    coast += std::max(b.ocean, b.beach);
                    desert += b.desert;
                }
                ambience.coast = coast / 5.0f;
                ambience.desert = desert / 5.0f;
                int leaves = 0, reef = 0;
                for (int dy = -3; dy <= 9; dy += 6)
                    for (int dz = -8; dz <= 8; dz += 8)
                        for (int dx = -8; dx <= 8; dx += 8) {
                            const vx::BlockId b =
                                world->blockAtWorld(ear + glm::ivec3(dx, dy, dz));
                            leaves += vx::blocks::isLeafBlock(b) ? 1 : 0;
                            reef += (vx::blocks::isCoral(b) || b == vx::blocks::GlowCoral ||
                                     b == vx::blocks::Kelp || b == vx::blocks::Seagrass)
                                        ? 1
                                        : 0;
                        }
                ambience.forest = std::min(float(leaves) / 5.0f, 1.0f);
                ambience.reef = std::min(float(reef) / 4.0f, 1.0f);
            }
            vx::applyAmbience(audio, vx::mixAmbience(ambience), fx.volAmbient);
            storm.update(audio, fx.precip, ambience.skyAccess, float(dt));
        }

        // Re-places live voices against the moved listener, retires faded steals, crossfades
        // the beds, and settles the submerged filter. None of that can happen at play time.
        audio.update(float(dt));

        world->update(*jobs, renderer, *light, camera.position);
        if (fx.farTerrain)
            farTerrain.update(world->generator(), *jobs, camera.position);
        {
            vx::Particles::Ambient pa;
            pa.desert = ambience.desert;
            pa.coast = ambience.coast;
            pa.precip = fx.precip;
            pa.snow = fx.snow;
            pa.time = float(now);
            pa.underwater = eyeUnderwater;
            particles.emitAmbient(float(dt), *world, camera.position, pa);
        }
        {
            const glm::vec3 fwd = camera.forward();
            const auto cue = lightning.update(float(dt), stormLevel, camera.position,
                                              glm::vec2(fwd.x, fwd.z), world->generator(), particles);
            fx.flash = lightning.flash();
            if (cue.thunder)
                audio.play("thunder", cue.volume * (0.25f + 0.75f * std::clamp(ambience.skyAccess, 0.0f, 1.0f)),
                           cue.pitch, vx::AudioEngine::Bus::Ambient);
        }
        particles.update(float(dt), &*world);
        {
            vx::Wildlife::Env we;
            we.daylight = ambience.daylight;
            we.precip = ambience.precip;
            we.skyAccess = ambience.skyAccess;
            we.coast = ambience.coast;
            we.forest = ambience.forest;
            we.desert = ambience.desert;
            we.reef = ambience.reef;
            we.nearWater = ambience.nearWater;
            we.underwater = eyeUnderwater;
            wildlife.update(float(dt), *world, camera.position, we);
        }

        // Fluid runs on its own slow clock, not per frame: water moving at 60 Hz looks
        // like a nervous twitch rather than a flow, and each tick is a batched edit that
        // remeshes whatever it touched. 6 Hz reads as liquid and keeps the cost off most
        // frames entirely.
        static double fluidClock = 0.0;
        fluidClock += dt;
        if (fluidClock >= 1.0 / 6.0) {
            fluidClock = 0.0;
            world->stepFluid(renderer, *light, camera.position);
        }
        // Falling runs faster than water: 6 Hz makes a collapsing pile look like it is
        // being dragged down a step at a time, where gravity should read as quick. Costs
        // nothing when nothing has been disturbed — the queue is empty and it returns.
        static double fallClock = 0.0;
        fallClock += dt;
        if (fallClock >= 1.0 / 12.0) {
            fallClock = 0.0;
            int landed = 0;
            glm::ivec3 landedAt{0};
            world->stepFalling(renderer, *light, &landed, &landedAt);
            if (landed > 0) {
                // One sound for the whole collapse, not one per block — thirty grains of
                // sand arriving in the same tick is a single event to the ear. The count
                // only sets how big it sounds.
                const glm::dvec3 at(landedAt);
                const vx::BlockId what = world->blockAtWorld(landedAt);
                audioHit(audio, "break", int(what), at, occlusionTo(at), 0.85f,
                         std::min(0.30f + 0.06f * float(landed), 0.75f));
            }
        }
        if (world->consumeDirty()) {
            world->collectDrawInfo(drawInfo, waterInfo);
            renderer.updateChunkMeta(drawInfo);
            renderer.updateWaterMeta(waterInfo);
            // geometry changed, so cached cascades are holding stale casters - but only the
            // ones that can see where it changed
            shadowMap.invalidateNear(world->takeDirtyNearest());
        }
        world->updateVisibilityIfNeeded(renderer, camera.position, caveCull);
        everBusy = everBusy || world->busy();
        if (!loadLogged && everBusy && !world->busy()) {
            world->logSummary();
            loadLogged = true;
        }

        // block the crosshair is pointing at -> selection wireframe (camera-relative)
        bool hasSel = false;
        glm::vec3 selMin(0.0f);
        if (!settingsOpen) {
            const auto sel = vx::raycastVoxels(
                camera.position, camera.forward(), 6.0, [&](const glm::ivec3& p) {
                    const vx::BlockId b = world->blockAtWorld(p);
                    return vx::blocks::opaque(b) || vx::blocks::isCross(b);
                });
            if (sel.hit) {
                hasSel = true;
                selMin = glm::vec3(glm::dvec3(sel.voxel) - camera.position);
            }
        }

        int fbW = 0;
        int fbH = 0;
        window->framebufferSize(fbW, fbH);
        if (fbW > 0 && fbH > 0) {
            // fog the outer ~8% of the loaded radius so the world edge reads as haze
            const float fogEnd =
                fx.fog ? float(world->radius) * float(vx::kChunkSize) * 0.92f : 1e9f;
            // crack stage from break progress; -1 while nothing is being broken
            const int crack =
                breaking ? vx::crackLayer(int(breakProgress * float(vx::crackStages())))
                         : -1;
            renderFrame(rc, camera, fx, caveCull, fbW, fbH, 0, float(now), &timers, hasSel,
                        selMin, eyeUnderwater, &taaState, fogEnd, crack, blockTex,
                        fx.farTerrain ? vx::FarTerrain::kFogEnd : 0.0f);
            if (hotbarOpen)
                overlay.drawHotbar(fbW, fbH, kHotbar, kHotbarCount, hotbarSel, blockTex);
        }

        const vx::ChunkManager::Stats ws = world->stats();
        const vx::Renderer::ArenaStats as = renderer.arenaStats();
        const std::uint64_t allocsNow = vx::heapAllocCount();
        const vx::FrameStats stats{
            .frameMs = frameMs,
            .gpuMs = timers.scene.milliseconds(),
            .heapAllocs = allocsNow - allocsPrevFrame,
            .drawCalls = 1,
            .quads = int(ws.totalQuads),
            .triangles = int(ws.totalQuads) * 2,
            .chunksLoaded = ws.loaded,
            .chunksMeshed = ws.meshed,
            .genQueue = ws.genQueue,
            .jobsInFlight = ws.jobsInFlight,
            .uploadsPending = ws.pendingUploads,
            .uploadMs = float(ws.uploadMsLast),
            .voxelBytes = ws.voxelBytes,
            .quadBytes = ws.quadBytes,
            .gpuChunksDrawn = renderer.gpuVisibleChunks(),
            .arenaUsedQuads = as.usedQuads,
            .arenaCapQuads = as.capacityQuads,
            .arenaFreeBlocks = as.freeBlocks,
            .walkMode = walkMode,
            .placeBlock = placeBlock,
            .gpuShadowMs = timers.shadow.milliseconds(),
            .gpuSceneMs = timers.scene.milliseconds(),
            .gpuPostMs = timers.post.milliseconds(),
        };
        allocsPrevFrame = allocsNow;
        const bool vsyncBefore = vsync;
        bool quitRequested = false;
        hud.diagnostics(stats, camera);
        if (hud.lampPicker(lampPickerOpen, fx.lampSwatch)) {
            // repaint the lamp that was middle-clicked; an emitter-to-emitter swap, which
            // is the path that has to retract the old hue's light as well as add the new
            if (vx::blocks::isLamp(world->blockAtWorld(lampPickerTarget)))
                world->setBlock(renderer, *light, lampPickerTarget,
                                vx::BlockId(vx::blocks::LampFirst + fx.lampSwatch));
            vx::Settings::save(settingsFile, fx, vsync, caveCull, world->radius,
                               camera.fovYDeg);
        }
        worldUi.seed = world->seed();
        if (settingsOpen && worldUi.saved.empty())
            worldUi.saved = worldStore.list(); // only touch the disk while the menu is up
        const int qualityBefore = fx.quality;
        const int shadowResBefore = fx.shadowRes;
        hud.settings(settingsOpen, vsync, caveCull, world->radius, camera.fovYDeg, fx,
                     quitRequested, worldUi);
        if (!settingsOpen)
            worldUi.saved.clear();

        applyQualityChange(qualityBefore, shadowResBefore);

        if (worldUi.saveRequested) {
            worldUi.saveRequested = false;
            world->saveDirty(); // the live folder has to hold the edits before we copy it
            if (worldStore.save(worldUi.saveName, world->seed()))
                worldUi.saved = worldStore.list();
            else
                VX_ERROR("world save failed: name must have at least one letter or digit");
        }
        if (worldUi.deleteRequested) {
            worldUi.deleteRequested = false;
            worldStore.remove(worldUi.selected);
            worldUi.selected.clear();
            worldUi.saved = worldStore.list();
        }

        // Both of these swap the world out from under the renderer, so they share one path:
        // drain the workers, swap, then settle behind the loading screen. That last part is
        // what was missing — a reset used to hand you an empty world and generate it around
        // you while you tried to play.
        const bool wantNew = worldUi.resetRequested;
        const bool wantLoad = worldUi.loadRequested && !worldUi.selected.empty();
        if (wantNew || wantLoad) {
            worldUi.resetRequested = false;
            worldUi.loadRequested = false;
            settingsOpen = false;
            // resetWorld/loadWorld hand chunks back to the pool, so no worker may still be
            // reading one. update() drains results and schedules nothing new once streamed,
            // so this converges rather than spinning.
            for (int guard = 0; guard < 20000 && world->busy(); ++guard) {
                renderer.beginFrame();
                world->update(*jobs, renderer, *light, camera.position);
                renderer.endFrame();
            }
            weatherDirector.reset();
            lightning.reset();
            farTerrain.reset();
            int newSeed = 0;
            const char* title = "voxelium";
            if (wantLoad && worldStore.load(worldUi.selected, newSeed)) {
                world->loadWorld(renderer, newSeed, worldStore.liveDir());
                title = "loading world";
            } else {
                newSeed = int(std::chrono::steady_clock::now().time_since_epoch().count() ^
                              std::chrono::system_clock::now().time_since_epoch().count()) &
                          0x7FFFFFFF;
                worldStore.clearLive();
                world->loadWorld(renderer, newSeed, worldStore.liveDir());
                title = "new world";
                VX_INFO("new world, seed %d", newSeed);
            }
            // Spawn on the new terrain, not wherever the old world left the camera — the
            // seed changed, so the ground under that point did too.
            {
                constexpr int kSpawnX = 16, kSpawnZ = 16;
                const int ground = world->generator().surfaceHeight(kSpawnX, kSpawnZ);
                camera.position = {double(kSpawnX) + 0.5,
                                   double(std::max(ground, vx::kSeaLevelForSpawn) + 3),
                                   double(kSpawnZ) + 0.5};
                player.position = camera.position -
                                  glm::dvec3(0.0, vx::Player::kEyeHeight, 0.0);
                player.velocity = {0.0, 0.0, 0.0};
            }
            settleBehindLoadingScreen(title);
            worldUi.saved = worldStore.list();
            prevTime = glfwGetTime(); // the settle ate wall-clock time; don't hand it to dt
        }
        if (vsync != vsyncBefore)
            window->setVsync(vsync);
        hud.render();
        if (quitRequested)
            window->requestClose();
        renderer.endFrame();

        // Read back before the swap: the just-rendered frame is still the front-buffer
        // target's contents at this point (single-buffered read of what endFrame() drew),
        // and reading after the swap would capture whichever frame the driver happens to
        // present next instead of this one.
        if (takeScreenshot) {
            takeScreenshot = false;
            int fbw = 0, fbh = 0;
            window->framebufferSize(fbw, fbh);
            if (fbw > 0 && fbh > 0) {
                std::error_code ec;
                std::filesystem::create_directories("screenshots", ec);
                std::vector<std::uint8_t> pixels(std::size_t(fbw) * std::size_t(fbh) * 3);
                glReadPixels(0, 0, fbw, fbh, GL_BGR, GL_UNSIGNED_BYTE, pixels.data());
                // Seconds-since-epoch, not a per-session counter: a counter restarting at 1
                // every launch collides with an earlier session's screenshots the moment you
                // play twice in one day. One screenshot per second is not a real limit for a
                // manual keypress.
                const auto epochSec =
                    std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
                char name[64];
                std::snprintf(name, sizeof(name), "screenshots/screenshot_%lld.bmp",
                              (long long)epochSec);
                if (vx::writeBmp24(name, fbw, fbh, pixels))
                    VX_INFO("screenshot saved: %s", name);
                else
                    VX_ERROR("screenshot failed to write: %s", name);
            }
        }

        window->swapBuffers();
    }

    VX_INFO("window closed, shutting down");
    vx::Settings::save(settingsFile, fx, vsync, caveCull, world->radius, camera.fovYDeg);
    audio.shutdown();
    world->saveDirty(); // persist edits before teardown (workers join after this)
    hud.shutdown();
    particles.shutdown();
    wildlife.shutdown();
        farTerrain.shutdown();
    overlay.shutdown();
    postfx.shutdown();
    shadowMap.shutdown();
    renderer.shutdown();
    return 0;
}
