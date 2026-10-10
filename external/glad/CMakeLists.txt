# glad2-generated GL 4.6 core loader, generated output is vendored, not built from a
# generator at configure time, so builds are deterministic and offline-capable.
# glad2 v2.0.8, OpenGL 4.6 core, no extensions (everything the engine uses is core in 4.6).
# Never hand-edit src/ or include/; regenerate instead:
#   python -m pip install glad2==2.0.8
#   python -m glad --api "gl:core=4.6" "--extensions=" --reproducible --out-path external/glad c
add_library(glad STATIC src/gl.c)
target_include_directories(glad SYSTEM PUBLIC include)
