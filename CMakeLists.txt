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
    src/core/Log.cpp
    src/core/Paths.cpp
    src/gfx/GlDebug.cpp
    src/gfx/GpuTimer.cpp
    src/gfx/Shader.cpp
    src/mesh/NaiveMesher.cpp
    src/platform/Window.cpp
    src/render/Camera.cpp
    src/ui/DebugHud.cpp
    src/world/Chunk.cpp
)

target_include_directories(voxelium PRIVATE src)
target_link_libraries(voxelium PRIVATE glad glfw glm::glm imgui)

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
