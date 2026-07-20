cmake_minimum_required(VERSION 3.25)

project(voxelium LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(cmake/Dependencies.cmake)

add_subdirectory(external/glad)

add_executable(voxelium
    src/main.cpp
    src/core/Bmp.cpp
    src/core/Paths.cpp
    src/gfx/GlDebug.cpp
    src/gfx/Shader.cpp
    src/platform/Window.cpp
)

target_include_directories(voxelium PRIVATE src)
target_link_libraries(voxelium PRIVATE glad glfw)

# GLFW must not drag in the system GL headers; glad owns all GL declarations.
target_compile_definitions(voxelium PRIVATE GLFW_INCLUDE_NONE)

# Our code builds warning-clean as errors on both compilers. Dependencies are SYSTEM
# includes (see Dependencies.cmake / external/glad) so their headers stay out of the gate.
if(MSVC)
    target_compile_options(voxelium PRIVATE /W4 /WX /permissive- /utf-8)
else()
    target_compile_options(voxelium PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
