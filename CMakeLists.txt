cmake_minimum_required(VERSION 3.25)

project(voxelium LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(cmake/Dependencies.cmake)

add_subdirectory(external/glad)

add_executable(voxelium
    src/main.cpp
    src/core/AllocTracker.cpp
    src/core/Bmp.cpp
    src/core/CrashHandler.cpp
    src/core/Log.cpp
    src/core/Paths.cpp
    src/core/Settings.cpp
    src/audio/Ambience.cpp
    src/audio/AudioEngine.cpp
    src/audio/PlayerSounds.cpp
    src/audio/SoundBank.cpp
    src/gfx/BlockTextures.cpp
    src/gfx/GlDebug.cpp
    src/gfx/GpuTimer.cpp
    src/gfx/MappedRing.cpp
    src/gen/TerrainGenerator.cpp
    src/gfx/Shader.cpp
    src/harness/Bench.cpp
    src/harness/Smoke.cpp
    src/harness/Tour.cpp
    src/io/ChunkSerializer.cpp
    src/io/RegionStore.cpp
    src/io/WorldStore.cpp
    src/jobs/JobSystem.cpp
    src/light/LightEngine.cpp
    src/mesh/Connectivity.cpp
    src/mesh/GreedyMesher.cpp
    src/mesh/NaiveMesher.cpp
    src/platform/Window.cpp
    src/render/Particles.cpp
    src/render/Camera.cpp
    src/render/FrameRenderer.cpp
    src/render/Overlay.cpp
    src/render/PostFx.cpp
    src/render/QualityPreset.cpp
    src/render/Renderer.cpp
    src/render/ShadowMap.cpp
    src/render/Weather.cpp
    src/render/Wildlife.cpp
    src/render/FarTerrain.cpp
    src/render/Lightning.cpp
    src/render/WeatherDirector.cpp
    src/ui/DebugHud.cpp
    src/ui/MenuScreens.cpp
    src/ui/MenuBackdrop.cpp
    src/world/Chunk.cpp
    src/world/ChunkManager.cpp
    src/world/MeshBufferPool.cpp
    src/world/Player.cpp
)

target_include_directories(voxelium PRIVATE src)
target_link_libraries(voxelium PRIVATE glad glfw glm::glm imgui unordered_dense::unordered_dense FastNoise stb miniaudio)
if(WIN32)
    # CrashHandler.cpp's MiniDumpWriteDump — part of the platform SDK on MSVC, a separate
    # import lib to name explicitly on MinGW.
    target_link_libraries(voxelium PRIVATE dbghelp)
endif()

# GLFW must not drag in the system GL headers; glad owns all GL declarations.
# GLM depth range 0..1 to match glClipControl(GL_ZERO_TO_ONE) — we run reversed-Z.
target_compile_definitions(voxelium PRIVATE GLFW_INCLUDE_NONE GLM_FORCE_DEPTH_ZERO_TO_ONE)

# Our code builds warning-clean as errors on both compilers. Dependencies are SYSTEM
# includes (see Dependencies.cmake / external/glad) so their headers stay out of the gate.
if(MSVC)
    target_compile_options(voxelium PRIVATE /W4 /WX /permissive- /utf-8)
else()
    target_compile_options(voxelium PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
