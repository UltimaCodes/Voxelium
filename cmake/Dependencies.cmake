# All third-party code is either vendored (external/) or fetched here, pinned to an exact
# tag. FetchContent over git submodules so a fresh clone configures with zero manual steps;
# SYSTEM so dependency headers are exempt from our -Werror/-WX warning gate.
#
# Dependencies arrive at the milestone that needs them, nothing is fetched speculatively.

include(FetchContent)

# --- GLFW 3.4 (M1: windowing + input) -------------------------------------------------
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)

FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
    GIT_SHALLOW    ON
    SYSTEM
)

# --- GLM 1.0.1 (M2: camera math) ------------------------------------------------------
FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
    GIT_SHALLOW    ON
    SYSTEM
)

# --- Dear ImGui 1.92.8 (M2: debug HUD only, never gameplay UI) ------------------------
FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.92.8
    GIT_SHALLOW    ON
    SYSTEM
)

# --- FastNoise2 1.1.1 (M10: SIMD terrain noise) ---------------------------------------
set(FASTNOISE2_NOISETOOL OFF CACHE BOOL "" FORCE)
set(FASTNOISE2_TESTS     OFF CACHE BOOL "" FORCE)

FetchContent_Declare(fastnoise2
    GIT_REPOSITORY https://github.com/Auburn/FastNoise2.git
    GIT_TAG        v1.1.1
    GIT_SHALLOW    ON
    SYSTEM
)

# --- ankerl::unordered_dense 4.8.1 (M5: world chunk map) ------------------------------
FetchContent_Declare(unordered_dense
    GIT_REPOSITORY https://github.com/martinus/unordered_dense.git
    GIT_TAG        v4.8.1
    GIT_SHALLOW    ON
    SYSTEM
)

# --- stb_image (block texture loading; header-only, no stable tags so pin a commit) ----
FetchContent_Declare(stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        f0569113c93ad095470c54bf34a17b36646bbbb5
    GIT_SHALLOW    OFF
    SYSTEM
)

# --- miniaudio 0.11.21 (audio playback; single header, public domain) -----------------
# Chosen because it is one header with no external deps and decodes wav/flac/mp3 itself,
# so adding sound costs nothing at configure time on a fresh clone.
FetchContent_Declare(miniaudio
    GIT_REPOSITORY https://github.com/mackron/miniaudio.git
    GIT_TAG        0.11.21
    GIT_SHALLOW    ON
    SYSTEM
)

FetchContent_MakeAvailable(glfw glm imgui unordered_dense fastnoise2 stb miniaudio)

# stb ships no CMake, expose it as a header-only interface target.
add_library(stb INTERFACE)
target_include_directories(stb SYSTEM INTERFACE ${stb_SOURCE_DIR})

# miniaudio's own CMakeLists builds tests/examples we don't want, take the header only.
add_library(miniaudio INTERFACE)
target_include_directories(miniaudio SYSTEM INTERFACE ${miniaudio_SOURCE_DIR})

# imgui ships no CMakeLists, build the core + the two backends we use as a static lib.
add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui SYSTEM PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PRIVATE glfw)
