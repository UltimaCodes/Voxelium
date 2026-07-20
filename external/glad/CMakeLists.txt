# glad2-generated GL 4.6 core loader — generated output is vendored, not built from a
# generator at configure time, so builds are deterministic and offline-capable.
# Regeneration command: see README.md in this directory. Never hand-edit src/ or include/.
add_library(glad STATIC src/gl.c)
target_include_directories(glad SYSTEM PUBLIC include)
