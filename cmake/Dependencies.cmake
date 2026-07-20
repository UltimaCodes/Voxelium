# All third-party code is either vendored (external/) or fetched here, pinned to an exact
# tag. FetchContent over git submodules so a fresh clone configures with zero manual steps;
# SYSTEM so dependency headers are exempt from our -Werror/-WX warning gate.
#
# Dependencies arrive at the milestone that needs them — nothing is fetched speculatively.

include(FetchContent)

# --- GLFW 3.4 (milestone 1: windowing + input) ---------------------------------------
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

FetchContent_MakeAvailable(glfw)
