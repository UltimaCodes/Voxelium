# Voxelium

voxel engine from scratch in C++20 + OpenGL 4.6 (core profile, DSA everywhere). no engine,
no bloat. the whole design is written up in [SPEC.md](SPEC.md), and every milestone gets its
numbers recorded in `docs/` so optimizations are proven, not vibes.

## building

needs cmake 3.25+, a C++20 compiler, and git + network on the first configure (deps are
fetched and pinned in `cmake/Dependencies.cmake`, glad is vendored in `external/`).

windows with gcc/mingw (what I use, no visual studio needed):

    cmake --preset mingw
    cmake --build --preset mingw-release

linux:

    cmake --preset gcc
    cmake --build --preset gcc-release

there's also an `msvc` preset if you have VS lying around. I don't.

works out of the box in JetBrains Rider/CLion — open the repo folder, pick the `mingw`
preset (or the IDE's bundled toolchain), hit run.

## running

run `voxelium` from anywhere — it finds `assets/` by walking up from the exe, so build
trees and IDE working dirs just work.

F3 toggles the debug HUD. hold RMB to fly (wasd + space/ctrl, shift = fast, scroll =
speed). ESC quits.

`voxelium --smoke [out.bmp]` renders offscreen, checks that pixels actually got drawn,
writes a screenshot and exits 0/1. headless self-check, useful for CI later.
