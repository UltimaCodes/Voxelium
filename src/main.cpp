#include "audio/AudioEngine.hpp"
#include "core/AllocTracker.hpp"
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
using vx::runMesherOracle;
using vx::runSmokeTest;
using vx::runTour;

int main(int argc, char** argv) {
    bool smoke = false;
    bool bench = false;
    int benchRadius = 16;
    bool tour = false;
    int tourRadius = 16;
    bool tourSettle = true;
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
        } else if (std::strcmp(argv[i], "--bench") == 0) {
            bench = true;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                benchRadius = std::atoi(argv[++i]);
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
        .visible = !smoke && !bench && !tour,
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
    // A smoke/bench/tour run renders from fixed defaults, never a machine's saved settings,
    // or the reference images would move under whoever happened to run it.
    const bool liveApp = !smoke && !bench && !tour;
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
    vx::shutdownShaderCache(); // every program is built by here
    const GLuint blockTex = vx::createBlockTextureArray(assets / "textures");
    RenderContext rc{renderer,   shadowMap,    postfx,   overlay,  &particles,
                     &weather,   drawProgram,  waterProgram, blockTex};

    // isolate the smoke test's save data so repeated runs start clean and can't touch a
    // real world save
    vx::WorldStore worldStore("saves");
    std::filesystem::path saveDir = worldStore.liveDir();
    // Every launch is a new world. The live one is scratch — keeping it means saving it
    // under a name from the ESC menu, which is what makes "new seed each time" safe.
    int worldSeed = 20260721;
    if (smoke || bench || tour) {
        // The harnesses assert against specific terrain (cave counts, ore counts, the glow
        // room, bench camera heights), so they pin the seed. Only the live game rolls one.
        saveDir = smoke ? smokeOut.parent_path() / "vxsave_smoke" : saveDir;
        if (smoke) {
            std::error_code ec;
            std::filesystem::remove_all(saveDir, ec);
        }
    } else {
        worldSeed = int(std::chrono::steady_clock::now().time_since_epoch().count() ^
                        std::chrono::system_clock::now().time_since_epoch().count()) &
                    0x7FFFFFFF;
        worldStore.clearLive();
        VX_INFO("world seed %d (save it from the ESC menu to keep it)", worldSeed);
    }

    // declaration order matters: jobs is destroyed first, joining all workers, so no
    // job can touch world's chunks or queues during teardown
    auto world = std::make_unique<vx::ChunkManager>(saveDir, worldSeed);
    auto light = std::make_unique<vx::LightEngine>(*world);
    auto jobs = std::make_unique<vx::JobSystem>();
    VX_INFO("job system: %d workers", jobs->workerCount());

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

    vx::DebugHud hud;
    if (!hud.init(window->handle()))
        return 1;
    window->hookScroll();

    if (tour) {
        const int rcT = runTour(rc, *world, *jobs, *light, camera, *window, tourRadius,
                                tourSettle);
        world->saveDirty();
        return rcT;
    }
    if (bench) {
        const int rcB = runBenchmark(rc, *world, *jobs, *light, camera, *window, benchRadius);
        hud.shutdown();
        particles.shutdown();
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
                glClearColor(0.07f, 0.09f, 0.14f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                hud.render();
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
                glClearColor(0.07f, 0.09f, 0.14f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
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
    double stepAccum = 0.0; // distance walked since the last footstep
    // How blocked the path from a sound to the ear is, 0 clear .. 1 solid. One DDA per sound
    // EVENT, which is far cheaper than it sounds — the crosshair already runs this same
    // raycast every single frame. A break heard through a wall should not arrive at full
    // volume, and this is the only thing that distinguishes the two.
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
    // The two values the early load could not place, because their owners did not exist yet.
    if (liveApp) {
        world->radius = savedRadius;
        camera.fovYDeg = savedFov;
    }
    window->setVsync(vsync);
    bool loadLogged = false;
    bool everBusy = false; // don't log the "loaded" summary before streaming even starts
    bool f4Held = false;
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
            }
        }
        f5Held = f5;
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
                // footsteps are paced by distance, not time, so they don't machine-gun
                // when you sprint or stall when you walk into a wall
                if (player.onGround) {
                    const glm::dvec3 d = player.position - before;
                    stepAccum += std::sqrt(d.x * d.x + d.z * d.z);
                    if (stepAccum > 2.2) {
                        stepAccum = 0.0;
                        const glm::ivec3 under(glm::floor(player.position.x),
                                               glm::floor(player.position.y) - 1,
                                               glm::floor(player.position.z));
                        const int id = world->blockAtWorld(under);
                        // a puff off the ground you are actually standing on
                        if (vx::blocks::opaque(vx::BlockId(id)))
                            particles.spawnFootstep(
                                {player.position.x, double(under.y) + 1.0, player.position.z},
                                vx::blockTopLayer(id));
                        else if (vx::blocks::isWater(vx::BlockId(id)))
                            particles.spawnSplash(
                                {player.position.x, double(under.y) + 1.0, player.position.z});
                        if (!audio.playIfPresent("step_" + std::to_string(id),
                                                 player.position, 0.5f))
                            audio.playAt("step", player.position, 0.5f);
                    }
                } else {
                    stepAccum = 0.0;
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
                    return vx::blocks::opaque(world->blockAtWorld(p));
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
                if (breakProgress >= 1.0f) {
                    const glm::dvec3 at(hit.voxel);
                    particles.spawnBlockBreak(at, vx::blockTopLayer(int(id)));
                    world->setBlock(renderer, *light, hit.voxel, vx::blocks::Air);
                    const float occ = occlusionTo(at);
                    if (!audio.has("break_" + std::to_string(int(id))))
                        audio.playOccluded("break", at, occ);
                    else
                        audio.playOccluded("break_" + std::to_string(int(id)), at, occ);
                    breakProgress = 0.0f;
                    breakTarget = {INT32_MIN, 0, 0};
                    breaking = false;
                }
            } else {
                breakProgress = 0.0f;
                breakTarget = {INT32_MIN, 0, 0};
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
                    if (!audio.has("place_" + std::to_string(put)))
                        audio.playOccluded("place", at, occ);
                    else
                        audio.playOccluded("place_" + std::to_string(put), at, occ);
                }
            }
            mmbHeld = mmb;
        }

        audio.setListener(camera.position, camera.forward());
        audio.setMasterVolume(fx.volume);
        audio.setBusVolume(vx::AudioEngine::Bus::World, fx.volWorld);
        audio.setBusVolume(vx::AudioEngine::Bus::Ui, fx.volUi);
        audio.setBusVolume(vx::AudioEngine::Bus::Ambient, fx.volAmbient);
        // Re-places live voices against the moved listener, retires faded steals, and
        // settles the submerged filter. None of that can happen at play time.
        audio.update(float(dt));

        world->update(*jobs, renderer, *light, camera.position);
        particles.update(float(dt));

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
            world->stepFalling(renderer, *light);
        }
        if (world->consumeDirty()) {
            world->collectDrawInfo(drawInfo, waterInfo);
            renderer.updateChunkMeta(drawInfo);
            renderer.updateWaterMeta(waterInfo);
            // geometry changed, so cached cascades are holding stale casters
            shadowMap.invalidate();
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
                    return vx::blocks::opaque(world->blockAtWorld(p));
                });
            if (sel.hit) {
                hasSel = true;
                selMin = glm::vec3(glm::dvec3(sel.voxel) - camera.position);
            }
        }

        // eye submerged -> underwater fog/tint in the composite pass, and the same flag
        // muffles the world bus. The renderer already knew this; audio never asked.
        const bool eyeUnderwater =
            vx::blocks::isWater(world->blockAtWorld(glm::ivec3(glm::floor(camera.position))));
        audio.setUnderwater(eyeUnderwater);

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
                        selMin, eyeUnderwater, &taaState, fogEnd, crack, blockTex);
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

        // A quality change is the one setting that is not free: the sample counts are baked
        // into the programs, so the tier has to be recompiled rather than branched on. The
        // binary cache makes the second visit to a preset cheap (#15: 349 ms cold for the
        // whole set, 56 ms warm), and nothing about the world is touched — the arena, the
        // meta buffers and every uploaded chunk stay exactly where they are, so switching
        // preset does not re-stream anything.
        if (fx.quality != qualityBefore || fx.shadowRes != shadowResBefore) {
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
        }

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

        window->swapBuffers();
    }

    VX_INFO("window closed, shutting down");
    vx::Settings::save(settingsFile, fx, vsync, caveCull, world->radius, camera.fovYDeg);
    audio.shutdown();
    world->saveDirty(); // persist edits before teardown (workers join after this)
    hud.shutdown();
    particles.shutdown();
    overlay.shutdown();
    postfx.shutdown();
    shadowMap.shutdown();
    renderer.shutdown();
    return 0;
}
