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
#include "render/LightVolume.hpp"
#include "ui/MenuBackdrop.hpp"
#include "render/WeatherDirector.hpp"
#include "render/Wildlife.hpp"
#include "render/ShadowMap.hpp"
#include "render/WaterWaves.hpp"
#include "ui/DebugHud.hpp"
#include "world/BlockInfo.hpp"
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

static_assert(vx::FxToggles{}.lodDetail == vx::ChunkManager::kDefaultDetailScale,
              "a world nobody has set a reach on must match the setting's default");

int main(int, char** argv) {
    vx::installCrashHandler();

#ifndef NDEBUG
    const bool debugContext = true;
#else
    const bool debugContext = false;
#endif

    auto window = vx::Window::create({
        .width = 1280,
        .height = 720,
        .title = "Voxelium",
        .visible = true,
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
    // shader source, so the compile depends on it, reading settings afterwards would mean
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
    if (vx::Settings::load(settingsFile, fx, vsync, caveCull, savedRadius, savedFov))
        VX_INFO("settings: loaded %s", settingsFile.string().c_str());

    const std::filesystem::path shaders = assets / "shaders";
    // caches driver output keyed by source and GL identity, not world state
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
            vx::submitTessProgram(shaders / "water.vert", shaders / "water.tesc",
                                  shaders / "water.tese", shaders / "water.frag", defines),
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
    float snowCover = 0.0f; // fills while it snows, melts when it stops (faster below the snow line)
    glm::vec3 airTint(1.0f); // eased toward the biome's colour, so crossing a border is not a pop
    float airHaze = 0.0f;
    vx::FarTerrain farTerrain;
    if (!farTerrain.init(shaders))
        return 1;
    if (!wildlife.init(shaders))
        return 1;
    // Lamp light in the air, and the sunlight the ground throws back.
    vx::LightVolume lightVolume;
    if (!lightVolume.initBounce(shaders))
        VX_ERROR("bounce light shaders did not build; the shade goes without it");
    vx::shutdownShaderCache(); // every program is built by here
    const GLuint blockTex = vx::createBlockTextureArray(assets / "textures");
    RenderContext rc{renderer,   shadowMap,    postfx,   overlay,  &particles,
                     &weather,   &wildlife,    &farTerrain, drawProgram, waterProgram, blockTex};
    rc.lightVolume = &lightVolume;

    // A quality change is the one setting that is not free: the sample counts are baked into the
    // programs, so the tier has to be recompiled rather than branched on. The binary cache makes
    // the second visit to a preset cheap (#15: 349 ms cold for the whole set, 56 ms warm), and
    // nothing about the world is touched, the arena, the meta buffers and every uploaded chunk
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

    vx::WorldStore worldStore("saves");
    const std::filesystem::path saveDir = worldStore.liveDir();
    // Made before the title screen: the terrain previewed behind it is built on these workers.
    auto jobs = std::make_unique<vx::JobSystem>();
    VX_INFO("job system: %d workers", jobs->workerCount());
    vx::MenuBackdrop menuBackdrop;
    vx::AudioEngine audio;
    // Decoded up front on the workers. Every clip is held as PCM for the process lifetime,
    // which is the point: a play used to open the file and decode it whole, on this thread.
    audio.init(assets.parent_path() / "sounds", jobs.get());

    // The saved world being played, once there is one: loaded from it or saved as it. F2 saves into it.
    std::string worldName;
    // Every launch is a new world unless the title screen below loads one. The live one is
    // scratch either way, keeping it means saving it under a name from the ESC menu, which
    // is what makes "new seed each time" safe.
    int worldSeed = 20260721;
    int worldGen = vx::TerrainGenerator::kLatestVersion; // a loaded world keeps the one it was made with
    {
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
            audio.setMasterVolume(fx.volume);
            audio.setBusVolume(vx::AudioEngine::Bus::Ui, fx.volUi);
            audio.setBusVolume(vx::AudioEngine::Bus::Ambient, fx.volAmbient);
            audio.setBusVolume(vx::AudioEngine::Bus::Music, fx.volMusic);
            // The theme, with the sea the menu looks out on faintly under it.
            audio.setBed("menu_theme", 0.9f, vx::AudioEngine::Bus::Music);
            audio.setBed("amb_surf", 0.22f);
            audio.setBed("amb_wind", 0.10f);
            audio.endUnsetBeds();
            {
                const auto ui = hud.uiSounds();
                if (ui.hover)
                    audio.play("ui_hover", 0.5f);
                if (ui.click)
                    audio.play("ui_click", 0.8f);
            }
            audio.update(dtT);
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
            int wantGen = vx::TerrainGenerator::kLatestVersion;
            if (title.page == 2)
                for (const vx::WorldStore::Entry& e : title.saved)
                    if (e.name == title.selected) {
                        wantSeed = e.seed;
                        wantGen = e.gen; // a saved world is shown as the terrain it was made with
                    }
            if (!previewGen || wantSeed != previewSeed || wantGen != previewGen->version()) {
                if (wantSeed != pendingSeed || !previewGen) {
                    pendingSeed = wantSeed;
                    pendingSince = nowT;
                }
                // Only once nothing is reading the old one: waiting for a build of it here froze the menu.
                if (!previewGen || (nowT - pendingSince > 0.35 && !farTerrain.busy() && !menuBackdrop.busy())) {
                    farTerrain.reset();
                    previewGen = std::make_unique<vx::TerrainGenerator>(wantSeed, wantGen);
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
            window->swapBuffers();
            if (title.quitRequested) {
                menuBackdrop.quiesce();
                farTerrain.quiesce();
                hud.shutdown();
                return 0;
            }
            if (title.newWorldRequested || title.loadRequested)
                audio.play("ui_start", 0.8f);
            if (title.newWorldRequested) {
                if (settingsPage)
                    vx::Settings::save(settingsFile, fx, vsync, caveCull, savedRadius, savedFov);
                worldSeed = parseSeed(title.seedText, rolledSeed);
                worldStore.clearLive();
                break;
            }
            if (title.loadRequested && worldStore.load(title.selected, worldSeed, worldGen)) {
                loaded = true;
                worldName = title.selected;
                break;
            }
            title.loadRequested = false; // load() failed (deleted mid-screen?), stay put
        }
        // the preview generator is about to go; a build still reading it must finish first. The
        // mesh it made is kept: the world about to start has the same seed, so it is the same
        // terrain and the loading screen and the game carry on from it, unless the preview had not
        // caught up with a seed rolled a moment before.
        menuBackdrop.quiesce();
        farTerrain.quiesce();
        if (!previewGen || previewSeed != worldSeed || previewGen->version() != worldGen)
            farTerrain.reset();
        previewGen.reset();
        if (window->shouldClose())
            return 0;
        VX_INFO(loaded ? "world loaded, seed %d" : "world seed %d (save it from the ESC "
                                                    "menu to keep it)",
                worldSeed);
    }

    auto world = std::make_unique<vx::ChunkManager>(saveDir, worldSeed, worldGen);
    auto light = std::make_unique<vx::LightEngine>(*world);
    // The workers are joined before world goes, on every way out of main: a mesh, generation
    // or far-terrain job still running, or still queued, since a stopping worker empties the
    // queue first, writes into world's chunks, pools and result queues, or reads its
    // generator. Declaring jobs after world used to see to that, until the title screen needed
    // workers first (9a93428) and moved it above; from then on, quitting while chunks were
    // streaming let jobs finish into a freed ChunkManager. A probe that quit mid-flight caught
    // it as heap corruption on exit (0xC0000374).
    struct JoinWorkers {
        std::unique_ptr<vx::JobSystem>& jobs;
        ~JoinWorkers() { jobs.reset(); }
    } joinWorkers{jobs};

    vx::FlyCamera camera;
    // Spawn ON the ground, not at a fixed altitude. y was hardcoded to 110, which is inside
    // the rock anywhere the terrain rises above it, and with exponential relief putting
    // peaks past 300, that is a lot of the world. surfaceHeight is the generator's own
    // answer for this column, so it lands on whatever is actually there, and spawnColumn
    // picks dry land, which matters once a world has real seas.
    {
        const glm::ivec2 sp = world->generator().spawnColumn();
        const int ground = world->generator().surfaceHeight(sp.x, sp.y);
        camera.position = {double(sp.x) + 0.5, double(std::max(ground, vx::kSeaLevelForSpawn) + 3),
                           double(sp.y) + 0.5};
    }
    camera.yawDeg = -35.0f;
    camera.pitchDeg = -25.0f;
    // Applied HERE, before the initial settle below: world and camera both already exist
    // (constructed above), and the saved radius/fov must be live BEFORE the loading screen
    // decides how much world to stream and light. Applying it after the settle (as this
    // used to) meant the first settle always ran at ChunkManager's built-in default of 16
    // regardless of what was saved or what preset was picked, so choosing Ultra (radius 32)
    // loaded and lit a radius-16 world behind the loading screen, then silently doubled the
    // streaming target the instant gameplay started. That is the exact "chunks don't render
    // past spawn, and it's laggy" report: the game never finished loading, it just stopped
    // showing a loading screen.
    world->detailScale = fx.lodDetail;
    world->radius = savedRadius;
    camera.fovYDeg = savedFov;

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
    // world that never settles must not be an infinite loading screen, if the budget runs
    // out we start anyway and it streams as it always did.
    // Settle the world behind the panel. Used at startup AND for every world change: before,
    // only startup did this, so "reset world" dropped you into an empty world that then
    // generated itself around you, every chunk, mesh and light settle competing with the
    // frames you were trying to play, which is exactly the stall the loading screen exists
    // to hide.
    const auto settleBehindLoadingScreen = [&](const char* title) {
        constexpr int kMaxFrames = 4000;
        // The real figure, summed from columnRange. It used to be radius*radius*
        // kWorldChunksY, which at radius 16 is 3072 against ~15200 actually meshed, so the
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
                    std::snprintf(phase, sizeof(phase), "generating terrain (%d chunks left)",
                                  ls.genQueue);
                else if (ls.unlit > 0)
                    std::snprintf(phase, sizeof(phase), "settling light (%d columns left)",
                                  ls.unlit);
                else
                    std::snprintf(phase, sizeof(phase), "building meshes (%d of %d)",
                                  ls.meshed, target);
                hud.beginFrame();
                hud.loadingScreen(p, title, phase);
                glViewport(0, 0, fbw, fbh);
                menuBackdrop.draw(rc, *jobs, world->generator(), fbw, fbh, glfwGetTime(), &fx);
                hud.render();
                window->swapBuffers();
            }
        }
        // The spawn column can grow a tree, and the world then opened inside a trunk or against
        // one. The chunks are here now: move to the nearest column with open air about the head.
        {
            const glm::ivec3 at(glm::floor(camera.position));
            const auto clear = [&](int x, int eye, int z) {
                for (int dz = -1; dz <= 1; ++dz)
                    for (int dx = -1; dx <= 1; ++dx)
                        for (int y = eye - 2; y <= eye + 1; ++y) {
                            const vx::BlockId b = world->blockAtWorld({x + dx, y, z + dz});
                            if (b != vx::blocks::Air && !vx::blocks::isCross(b))
                                return false;
                        }
                return true;
            };
            bool placed = false;
            for (int r = 0; r <= 8 && !placed; ++r)
                for (int dz = -r; dz <= r && !placed; ++dz)
                    for (int dx = -r; dx <= r && !placed; ++dx) {
                        if (std::max(std::abs(dx), std::abs(dz)) != r)
                            continue;
                        const int x = at.x + dx, z = at.z + dz;
                        const int eye = std::max(world->generator().surfaceHeight(x, z), vx::kSeaLevelForSpawn) + 3;
                        if (clear(x, eye, z)) {
                            camera.position = {double(x) + 0.5, double(eye), double(z) + 0.5};
                            player.position = camera.position - glm::dvec3(0.0, vx::Player::kEyeHeight, 0.0);
                            placed = true;
                        }
                    }
        }
        world->collectDrawInfo(drawInfo, waterInfo);
        renderer.updateChunkMeta(drawInfo);
        renderer.updateWaterMeta(waterInfo);
        shadowMap.invalidate();
        // One completed frame, so the bar lands on 100% instead of cutting away at 97,
        // and the first real frame still has to build its shadow cascades, so this is not
        // dead time either.
        {
            int fbw = 0, fbh = 0;
            window->framebufferSize(fbw, fbh);
            if (fbw > 0 && fbh > 0) {
                hud.beginFrame();
                hud.loadingScreen(1.0f, title, "ready");
                glViewport(0, 0, fbw, fbh);
                menuBackdrop.draw(rc, *jobs, world->generator(), fbw, fbh, glfwGetTime(), &fx);
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
    bool lmbHeld = false;            // a punch lands on the press, not every frame it is held
    glm::dvec3 flyPrev{0.0};          // last frame's camera in fly mode, for its velocity
    bool walkMode = false;
    vx::PlayerSounds playerSounds;
    vx::AmbienceState ambience;
    vx::StormSounds storm;
    vx::AmbientCalls calls;
    vx::MusicDirector music;
    glm::dvec3 earPrev{0.0};
    storm.selfSchedule = false; // thunder now follows the lightning, after its own delay
    // How blocked the path from a sound to the ear is, 0 clear .. 1 solid. One DDA per sound
    // EVENT, which is far cheaper than it sounds, the crosshair already runs this same
    // raycast every single frame. A break heard through a wall should not arrive at full
    // volume, and this is the only thing that distinguishes the two.
    //
    // Four tiers, narrowest first: this exact block, its material family, the generic clip,
    // and finally the material's own FOOTSTEP at a shifted pitch. The material tier is the
    // one that matters, block sounds are keyed by numeric id, and this engine stores data in
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
    bool hotbarOpen = true;
    bool pickerOpen = false;
    bool eHeld = false;
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
    constexpr int kHotbarCount = 9;
    int hotbarSel = 0;
    int placeBlock = fx.hotbar[hotbarSel];
    const GLuint iconAtlas = vx::createIconAtlas(blockTex);
    bool rmbHeld = false;
    // the first-person hand
    float bobPhase = 0.0f, bobAmount = 0.0f;
    float swing = 1.0f;  // 0..1 through an arm swing, 1 at rest
    float equip = 1.0f;  // 0 just switched, 1 held up
    int heldPrev = -1;
    std::uint64_t allocsPrevFrame = vx::heapAllocCount();

    while (!window->shouldClose()) {
        window->pollEvents();
        // BEFORE anything can play. These used to sit near the end of the frame, after the
        // physics loop had already fired footsteps, so every sound was positioned against
        // the PREVIOUS frame's listener, and on the very first frame against a listener
        // still at the world origin. A footstep a hundred blocks from the ear is attenuated
        // to nothing, which is a silent first step and a subtly wrong position after that.
        audio.setListener(camera.position, camera.forward());
        audio.setMasterVolume(fx.volume);
        audio.setBusVolume(vx::AudioEngine::Bus::World, fx.volWorld);
        audio.setBusVolume(vx::AudioEngine::Bus::Ui, fx.volUi);
        audio.setBusVolume(vx::AudioEngine::Bus::Ambient, fx.volAmbient);
        audio.setBusVolume(vx::AudioEngine::Bus::Music, fx.volMusic);
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
        if (f2 && !f2Held && !hud.typing()) {
            // Into the world this one was loaded from or last saved as. The live folder alone is
            // scratch, the next world clears it. One with no name yet gets the ESC menu, where it
            // is named.
            if (worldName.empty()) {
                settingsOpen = true;
            } else {
                world->saveDirty();
                if (!worldStore.save(worldName, world->seed(), world->genVersion()))
                    VX_ERROR("could not save \"%s\"", worldName.c_str());
            }
        }
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
        const bool eKey = window->keyDown(GLFW_KEY_E);
        if (eKey && !eHeld && !settingsOpen && !lampPickerOpen && !hud.typing())
            pickerOpen = !pickerOpen;
        eHeld = eKey;
        if (pickerOpen && esc)
            settingsOpen = false; // escape closes the picker, not into the pause menu
        if (pickerOpen && esc)
            pickerOpen = false;
        if (hotbarOpen && !settingsOpen && !pickerOpen) {
            for (int k = 0; k < kHotbarCount; ++k)
                if (window->keyDown(GLFW_KEY_1 + k))
                    hotbarSel = k;
            // scroll cycles slots while the hotbar is up (drained here so it doesn't also
            // change fly speed); when it's hidden the camera gets the scroll for speed
            const double sc = window->consumeScrollY();
            if (sc != 0.0)
                hotbarSel = (hotbarSel - (sc > 0.0 ? 1 : -1) + kHotbarCount) % kHotbarCount;
        }
        placeBlock = fx.hotbar[hotbarSel];

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
        // Menus are the ONLY thing that hands the cursor back, alt-tabbing counts because
        // focus is gone. Everything below reads this rather than testing buttons itself.
        const bool menuOpen = settingsOpen || lampPickerOpen || pickerOpen;
        const bool mouseOurs = window->focused() && !menuOpen && !hud.wantsMouse();
        if (mouseOurs != cursorCaptured) {
            cursorCaptured = mouseOurs;
            window->setCursorCaptured(mouseOurs);
            if (mouseOurs)
                camera.reanchor(*window);
        }

        if (settingsOpen || lampPickerOpen || pickerOpen) {
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
                in.sprint = window->keyDown(GLFW_KEY_LEFT_CONTROL);
                const glm::dvec3 before = player.position;
                player.step(*world, in);
                // The animals are solid: shoved aside, and a turtle pushes back a little.
                const glm::dvec2 give = wildlife.interact(
                    {player.position, vx::Player::kHalfWidth, vx::Player::kHeight,
                     glm::vec3(player.velocity), true},
                    *world);
                if (give.x != 0.0 || give.y != 0.0)
                    player.nudge(*world, give);
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
            // the head bobs with the stride, and a sprint widens the view a little
            {
                const float hs = float(std::sqrt(player.velocity.x * player.velocity.x + player.velocity.z * player.velocity.z));
                const float amt = player.onGround && !player.inWater ? std::min(hs / 4.4f, 1.6f) : 0.0f;
                bobAmount += (amt - bobAmount) * std::min(1.0f, float(dt) * 8.0f);
                bobPhase += float(dt) * hs * 1.9f;
                const glm::vec3 right = glm::normalize(glm::cross(camera.forward(), glm::vec3(0.0f, 1.0f, 0.0f)));
                camera.position += glm::dvec3(right * (std::sin(bobPhase) * 0.035f * bobAmount)) +
                                   glm::dvec3(0.0, (std::abs(std::cos(bobPhase)) * 0.055 - 0.03) * double(bobAmount), 0.0);
                const float want = hs > 5.2f ? 7.0f : 0.0f;
                camera.fovKick += (want - camera.fovKick) * std::min(1.0f, float(dt) * 6.0f);
            }
        } else {
            camera.update(*window, dt, hud.wantsMouse(), hud.wantsKeyboard(), mouseOurs);
            {
                const float want = window->keyDown(GLFW_KEY_LEFT_SHIFT) && mouseOurs ? 9.0f : 0.0f;
                camera.fovKick += (want - camera.fovKick) * std::min(1.0f, float(dt) * 5.0f);
                bobAmount += (0.0f - bobAmount) * std::min(1.0f, float(dt) * 8.0f);
            }
            // Flying through them still scatters them; nothing is shoved, a fly camera has no body.
            const glm::vec3 flyVel = dt > 0.0f ? glm::vec3(camera.position - flyPrev) / dt : glm::vec3(0.0f);
            wildlife.interact({camera.position - glm::dvec3(0.0, vx::Player::kEyeHeight, 0.0),
                               vx::Player::kHalfWidth, vx::Player::kHeight, flyVel, false},
                              *world);
        }
        flyPrev = camera.position;

        if (!menuOpen && camera.flying()) {
            const bool lmb = window->mouseDown(GLFW_MOUSE_BUTTON_LEFT);
            const bool mmb = window->mouseDown(GLFW_MOUSE_BUTTON_MIDDLE);
            const bool rmb = window->mouseDown(GLFW_MOUSE_BUTTON_RIGHT);
            const auto hit = vx::raycastVoxels(
                camera.position, camera.forward(), 6.0, [&](const glm::ivec3& p) {
                    const vx::BlockId b = world->blockAtWorld(p);
                    return vx::blocks::opaque(b) || vx::blocks::isCross(b);
                });

            // An animal in front of the block being aimed at takes the click instead: a punch
            // on the press (knocked back, runs off, nothing hurt), and no digging past it.
            float mobDist = 0.0f;
            const int mob = lmb ? wildlife.pick(camera.position, camera.forward(), 4.5f, mobDist) : -1;
            const bool mobFirst = mob >= 0 && (!hit.hit || double(mobDist) < hit.distance);
            if (mobFirst && !lmbHeld) {
                wildlife.punch(mob, camera.forward());
                const glm::dvec3 at = wildlife.positionOf(mob);
                const bool wet = wildlife.swimsAt(mob);
                audio.playOccluded(wet ? "swim" : "land_hard", at, occlusionTo(at), wet ? 0.7f : 0.55f,
                                   wet ? 1.35f : 1.8f);
            }
            lmbHeld = lmb;

            // Breaking is held, not clicked. Anything that changes what you are aiming at,
            // or letting go, drops the progress, which is what makes the crack read as
            // damage to THAT block rather than to a global timer.
            breaking = false;
            if (lmb && hit.hit && !mobFirst) {
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

            // Middle click picks the block you are looking at into the slot; with shift held on a
            // lamp it opens the colour picker for that lamp instead.
            if (mmb && !mmbHeld && hit.hit) {
                const vx::BlockId aimed = world->blockAtWorld(hit.voxel);
                const bool shift = window->keyDown(GLFW_KEY_LEFT_SHIFT) || window->keyDown(GLFW_KEY_RIGHT_SHIFT);
                if (shift && vx::blocks::isLamp(aimed)) {
                    lampPickerOpen = true;
                    lampPickerTarget = hit.voxel;
                } else if (vx::blocks::inPicker(aimed)) {
                    const vx::BlockId z2x = aimed == vx::blocks::WoodZ          ? vx::blocks::WoodX
                                            : aimed == vx::blocks::SpruceWoodZ ? vx::blocks::SpruceWoodX
                                            : aimed == vx::blocks::BirchWoodZ  ? vx::blocks::BirchWoodX
                                                                                : aimed;
                    int slot = -1;
                    for (int k = 0; k < kHotbarCount; ++k)
                        if (fx.hotbar[k] == int(z2x))
                            slot = k;
                    if (slot >= 0)
                        hotbarSel = slot;
                    else
                        fx.hotbar[hotbarSel] = int(z2x);
                    placeBlock = fx.hotbar[hotbarSel];
                }
            }
            mmbHeld = mmb;
            if (rmb && !rmbHeld && hit.hit) {
                const glm::dvec3 at(hit.voxel);
                const vx::BlockId aimed = world->blockAtWorld(hit.voxel);
                // a plant in the way is replaced, the way you would plant over a weed
                const glm::ivec3 cell = vx::blocks::isCross(aimed) && !vx::blocks::isWaterlogged(aimed)
                                            ? hit.voxel
                                            : hit.voxel + hit.normal;
                int put = placeBlock;
                // a lying log turns to lie the way you are looking
                const glm::vec3 f = camera.forward();
                if (std::abs(f.z) > std::abs(f.x) &&
                    (put == vx::blocks::WoodX || put == vx::blocks::SpruceWoodX || put == vx::blocks::BirchWoodX))
                    put += 1;
                world->setBlock(renderer, *light, cell, vx::BlockId(put));
                const float occ = occlusionTo(at);
                // Same chain, pitched UP: setting a block down is a lighter sound than
                // breaking one.
                audioHit(audio, "place", put, at, occ, 1.25f, 0.85f);
                swing = 0.0f;
            }
            rmbHeld = rmb;
        }

        // eye submerged -> underwater fog/tint in the composite pass, and the same flag
        // muffles the world bus and swaps the ambience bed. Computed here rather than next
        // to the render call because everything below wants it in the SAME frame the camera
        // moved, not one behind.
        //
        // The top water cell holds the surface, and the swell lifts it and drops it: the eye is
        // under it only below the surface as drawn, which can be a little past either edge of
        // the cell. Testing the cell alone switched the fog on a fraction of a block early.
        const bool eyeUnderwater = [&] {
            const glm::ivec3 eye(glm::floor(camera.position));
            const auto blockAt = [&](int x, int y, int z) { return world->blockAtWorld({x, y, z}); };
            const auto wf = vx::WaterWaves::frame(now, camera.position, fx.waveStorm, fx.waves);
            if (vx::blocks::isWater(blockAt(eye.x, eye.y, eye.z))) {
                if (vx::blocks::isWater(blockAt(eye.x, eye.y + 1, eye.z)))
                    return true;
                return camera.position.y < vx::waterSurfaceY(wf, blockAt, camera.position, eye.y);
            }
            // a crest can reach past the top of its cell into the air above
            if (vx::blocks::isWater(blockAt(eye.x, eye.y - 1, eye.z)))
                return camera.position.y < vx::waterSurfaceY(wf, blockAt, camera.position, eye.y - 1);
            return false;
        }();
        audio.setUnderwater(eyeUnderwater);
        playerSounds.frame(audio, camera.position, eyeUnderwater, float(dt));

        // What the world sounds like from here. Sky light answers "how enclosed am I" for
        // free, the light BFS has already done that work, so this is one chunk lookup
        // rather than a cover raycast every frame.
        {
            const glm::ivec3 ear(glm::floor(camera.position));
            ambience.skyAccess = world->skyAccessAt(ear);
            // Slower going into the dark than coming back out, the way an eye does it. A lamp by you
            // or in your hand is light to adjust to as well, or it would glare.
            const float lamp = std::max(world->blockLightAt(ear),
                                        std::max(fx.handLight.r, std::max(fx.handLight.g, fx.handLight.b)) / 15.0f);
            const float eyeTarget = glm::mix(2.4f, 1.0f, std::max(glm::smoothstep(0.1f, 0.7f, ambience.skyAccess),
                                                                  glm::smoothstep(0.15f, 0.65f, lamp)));
            fx.eyeGain += (eyeTarget - fx.eyeGain) *
                          (1.0f - std::exp(-float(dt) / (eyeTarget > fx.eyeGain ? 1.8f : 0.6f)));
            // Sky light dies with depth in water, but the light above it is what counts for shafts:
            // go by the first cell over the water the eye is in, so the sea has them at any depth
            // and a lake in a cave has none.
            glm::ivec3 over = ear;
            if (eyeUnderwater)
                for (int k = 0; k < 128 && vx::blocks::isWater(world->blockAtWorld(over)); ++k)
                    ++over.y;
            fx.eyeSky += (world->skyAccessAt(over) - fx.eyeSky) * (1.0f - std::exp(-float(dt) / 0.8f));
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
                                                      fx.timeOfDay, ambience.snow);
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
            // The sea builds up under heavy rain and a storm and lies down again after, more
            // slowly than the weather itself turns: about 20 s to rise, 40 to settle.
            {
                const float want = std::clamp(std::max(stormLevel, (fx.precip - 0.3f) / 0.7f), 0.0f, 1.0f);
                const float rate = (want > fx.waveStorm ? 0.05f : 0.025f) * float(dt);
                fx.waveStorm += std::clamp(want - fx.waveStorm, -rate, rate);
            }
            {
                const float snowing = (fx.snow > 0.5f && fx.precip > 0.05f) ? fx.precip : 0.0f;
                const float melt = camera.position.y < 140.0 ? 0.03f : 0.008f;
                snowCover += float(dt) * (snowing > 0.0f ? snowing * 0.02f : -melt);
                snowCover = std::clamp(snowCover, 0.0f, 1.0f);
                fx.snowCover = snowCover;
            }
            // The air follows the biome around the camera. Each biome pulls the colour its way and
            // adds a little haze; all of it eases in over a few seconds because the biome weights
            // it reads are a five-point sample that steps as the sample points cross a border.
            {
                const glm::vec3 desertAir(1.10f, 0.97f, 0.80f), seaAir(0.90f, 1.00f, 1.08f),
                    forestAir(0.95f, 1.04f, 0.96f);
                const float coastOnly = ambience.coast * (1.0f - ambience.desert);
                glm::vec3 want(1.0f);
                want *= glm::mix(glm::vec3(1.0f), desertAir, ambience.desert);
                want *= glm::mix(glm::vec3(1.0f), seaAir, coastOnly);
                want *= glm::mix(glm::vec3(1.0f), forestAir, ambience.forest * (1.0f - ambience.desert));
                const float wantHaze = 0.17f * ambience.desert + 0.07f * coastOnly + 0.04f * ambience.forest;
                const float k = std::min(1.0f, float(dt) * 0.35f);
                airTint += (want - airTint) * k;
                airHaze += (wantHaze - airHaze) * k;
                fx.biomeTint[0] = airTint.x;
                fx.biomeTint[1] = airTint.y;
                fx.biomeTint[2] = airTint.z;
                fx.biomeHaze = airHaze;
                // northern lights over the cold biomes, eased in as you travel into them
                const float wantAurora = std::clamp(ambience.snow * 1.4f, 0.0f, 1.0f) * (1.0f - fx.precip);
                fx.aurora += (wantAurora - fx.aurora) * std::min(1.0f, float(dt) * 0.2f);
            }
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
            // count of what is physically here, leaves for the canopy, coral for the reef.
            {
                constexpr int kSpread = 24;
                const int offs[5][2] = {{0, 0}, {kSpread, 0}, {-kSpread, 0}, {0, kSpread},
                                        {0, -kSpread}};
                float coast = 0.0f, desert = 0.0f;
                float w[int(vx::Biome::Count)] = {};
                const auto& gen = world->generator();
                for (const auto& o : offs) {
                    const auto b = gen.biomeAt(ear.x + o[0], ear.z + o[1]);
                    coast += std::max(b.ocean, b.beach);
                    desert += b.desert;
                    if (gen.version() >= 3)
                        w[int(gen.biomeOf(ear.x + o[0], ear.z + o[1]))] += 0.2f;
                }
                ambience.coast = coast / 5.0f;
                ambience.desert = desert / 5.0f;
                using B = vx::Biome;
                if (gen.version() >= 3) {
                    ambience.desert = std::max(ambience.desert, w[int(B::Desert)] + w[int(B::RedDesert)] + w[int(B::Badlands)]);
                    ambience.open = w[int(B::Plains)] + w[int(B::Meadow)] + w[int(B::Savanna)] + 0.5f * w[int(B::Beach)];
                    ambience.conifer = w[int(B::Taiga)] + 0.6f * w[int(B::SnowyTaiga)];
                    ambience.jungle = w[int(B::Jungle)];
                    ambience.swamp = w[int(B::Swamp)];
                    ambience.snow = w[int(B::Tundra)] + w[int(B::SnowyTaiga)] + w[int(B::Snowcap)];
                    ambience.alpine = w[int(B::Alpine)] + w[int(B::Snowcap)];
                    ambience.forest = w[int(B::Forest)] + w[int(B::BirchForest)];
                } else {
                    ambience.open = 1.0f - ambience.desert - ambience.coast;
                }
                const glm::dvec3 earNow = camera.position;
                const float moved = dt > 0.0f ? float(glm::length(earNow - earPrev) / double(dt)) : 0.0f;
                earPrev = earNow;
                ambience.speed += (std::min(moved, 120.0f) - ambience.speed) * std::min(1.0f, float(dt) * 3.0f);
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
                ambience.canopy = std::min(float(leaves) / 5.0f, 1.0f);
                if (world->generator().version() < 3)
                    ambience.forest = ambience.canopy;
                ambience.reef = std::min(float(reef) / 4.0f, 1.0f);
            }
            vx::applyAmbience(audio, vx::mixAmbience(ambience), fx.volAmbient);
            calls.update(audio, ambience, camera.position, float(dt));
            music.update(audio, float(dt), ambience.daylight, ambience.depth > 12.0f && ambience.skyAccess < 0.3f,
                         eyeUnderwater);
            storm.update(audio, fx.precip, ambience.skyAccess, float(dt));
        }

        // Re-places live voices against the moved listener, retires faded steals, crossfades
        // the beds, and settles the submerged filter. None of that can happen at play time.
        audio.update(float(dt));

        // Before the world update: these change blocks and light, and from the end of the update
        // until the frame is drawn the light thread owns the light (see startLightSlice).
        //
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
        // nothing when nothing has been disturbed, the queue is empty and it returns.
        static double fallClock = 0.0;
        fallClock += dt;
        if (fallClock >= 1.0 / 12.0) {
            fallClock = 0.0;
            int landed = 0;
            glm::ivec3 landedAt{0};
            world->stepFalling(renderer, *light, &landed, &landedAt);
            if (landed > 0) {
                // One sound for the whole collapse, not one per block, thirty grains of
                // sand arriving in the same tick is a single event to the ear. The count
                // only sets how big it sounds.
                const glm::dvec3 at(landedAt);
                const vx::BlockId what = world->blockAtWorld(landedAt);
                audioHit(audio, "break", int(what), at, occlusionTo(at), 0.85f,
                         std::min(0.30f + 0.06f * float(landed), 0.75f));
            }
        }
        world->detailScale = fx.lodDetail;
        world->update(*jobs, renderer, *light, camera.position);
        // Light seeding and propagation run on the light thread from here until the frame is
        // drawn: everything in between only reads the world.
        world->startLightSlice(*light);
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
            we.swamp = ambience.swamp;
            we.jungle = ambience.jungle;
            we.snow = ambience.snow;
            we.underwater = eyeUnderwater;
            wildlife.update(float(dt), *world, camera.position, we);
            vx::Wildlife::Call heard[16];
            for (int k = 0, n = wildlife.takeCalls(heard, 16); k < n; ++k)
                audio.playCall(heard[k].clip, heard[k].pos, 0.8f, 0.9f + 0.2f * float(k % 5) / 4.0f);
        }

        if (world->consumeDirty()) {
            world->collectDrawInfo(drawInfo, waterInfo);
            renderer.updateChunkMeta(drawInfo);
            renderer.updateWaterMeta(waterInfo);
            // geometry changed, so cached cascades are holding stale casters, but only the
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
            if (rc.lightVolume)
                rc.lightVolume->update(*world, camera.position);
            // a lamp in the hand lights the way, coming up as it is drawn
            // Lit half in its own colour and half in white: a green lamp in the hand used to turn
            // everything about you neon green.
            const vx::BlockId held = vx::BlockId(placeBlock);
            const glm::vec3 emit(float(vx::blocks::emitR(held)), float(vx::blocks::emitG(held)),
                                 float(vx::blocks::emitB(held)));
            const float emitLevel = std::max(emit.r, std::max(emit.g, emit.b));
            fx.handLight = settingsOpen || emitLevel <= 0.0f
                               ? glm::vec3(0.0f)
                               : glm::mix(emit, glm::vec3(emitLevel), 0.5f) * equip;
            renderFrame(rc, camera, fx, caveCull, fbW, fbH, 0, float(now), &timers, hasSel,
                        selMin, eyeUnderwater, &taaState, fogEnd, crack, blockTex,
                        fx.farTerrain ? vx::FarTerrain::kFogEnd : 0.0f);
            if (!settingsOpen) {
                if (placeBlock != heldPrev) {
                    heldPrev = placeBlock;
                    equip = 0.0f;
                }
                equip = std::min(1.0f, equip + float(dt) * 4.5f);
                if (breaking && swing >= 1.0f)
                    swing = 0.0f;
                swing = std::min(1.0f, swing + float(dt) * 3.6f);
                const float lightHere = vx::blocks::emissive(held)
                                            ? 1.0f
                                            : 0.18f + 0.82f * std::clamp(ambience.daylight * fx.eyeSky, 0.0f, 1.0f);
                const glm::vec2 bob(std::sin(bobPhase) * 0.018f * bobAmount, -std::abs(std::cos(bobPhase)) * 0.022f * bobAmount);
                overlay.drawHeld(fbW, fbH, placeBlock, blockTex, swing, equip, bob, glm::vec3(lightHere));
            }
            if (hotbarOpen && camera.flying())
                overlay.drawHotbar(fbW, fbH, fx.hotbar, kHotbarCount, hotbarSel, blockTex);
        }

        world->finishLightSlice(*light); // before anything below can change the world
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
        {
            const bool wasOpen = pickerOpen;
            hud.blockPicker(pickerOpen, fx.hotbar, hotbarSel, iconAtlas);
            if (wasOpen && !pickerOpen)
                vx::Settings::save(settingsFile, fx, vsync, caveCull, world->radius, camera.fovYDeg);
        }
        if (hud.lampPicker(lampPickerOpen, fx.lampSwatch)) {
            // repaint the lamp that was middle-clicked; an emitter-to-emitter swap, which
            // is the path that has to retract the old hue's light as well as add the new
            if (vx::blocks::isLamp(world->blockAtWorld(lampPickerTarget)))
                world->setBlock(renderer, *light, lampPickerTarget,
                                vx::BlockId(vx::blocks::LampFirst + fx.lampSwatch));
            if (vx::blocks::isLamp(vx::BlockId(fx.hotbar[hotbarSel])))
                fx.hotbar[hotbarSel] = vx::blocks::LampFirst + fx.lampSwatch;
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
            if (worldStore.save(worldUi.saveName, world->seed(), world->genVersion())) {
                worldName = vx::WorldStore::sanitize(worldUi.saveName);
                worldUi.saved = worldStore.list();
            } else
                VX_ERROR("world save failed (the name needs a letter or digit, or the disk said no)");
        }
        if (worldUi.deleteRequested) {
            worldUi.deleteRequested = false;
            worldStore.remove(worldUi.selected);
            if (worldUi.selected == worldName)
                worldName.clear(); // F2 must not bring a deleted world back
            worldUi.selected.clear();
            worldUi.saved = worldStore.list();
        }

        // Both of these swap the world out from under the renderer, so they share one path:
        // drain the workers, swap, then settle behind the loading screen. That last part is
        // what was missing, a reset used to hand you an empty world and generate it around
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
            // Both worlds live in the same folder. Whatever the old one still had waiting to be
            // written goes out now, before that folder is cleared or refilled; loadWorld would
            // otherwise write it on top of the next world's regions.
            world->saveDirty();
            weatherDirector.reset();
            lightning.reset();
            snowCover = 0.0f;
            menuBackdrop.quiesce(); // its search may still be reading the old world's generator
            farTerrain.reset();
            lightVolume.invalidate();
            int newSeed = 0, newGen = vx::TerrainGenerator::kLatestVersion;
            const char* title = "voxelium";
            if (wantLoad && worldStore.load(worldUi.selected, newSeed, newGen)) {
                world->loadWorld(renderer, newSeed, worldStore.liveDir(), newGen);
                worldName = worldUi.selected;
                title = "loading world";
            } else {
                worldName.clear();
                newSeed = int(std::chrono::steady_clock::now().time_since_epoch().count() ^
                              std::chrono::system_clock::now().time_since_epoch().count()) &
                          0x7FFFFFFF;
                worldStore.clearLive();
                world->loadWorld(renderer, newSeed, worldStore.liveDir());
                title = "new world";
                VX_INFO("new world, seed %d", newSeed);
            }
            // Spawn on the new terrain, not wherever the old world left the camera, the
            // seed changed, so the ground under that point did too.
            {
                const glm::ivec2 sp = world->generator().spawnColumn();
                const int ground = world->generator().surfaceHeight(sp.x, sp.y);
                camera.position = {double(sp.x) + 0.5,
                                   double(std::max(ground, vx::kSeaLevelForSpawn) + 3),
                                   double(sp.y) + 0.5};
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
        if (settingsOpen) {
            const auto ui = hud.uiSounds();
            if (ui.hover)
                audio.play("ui_hover", 0.5f);
            if (ui.click)
                audio.play("ui_click", 0.8f);
        }
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
