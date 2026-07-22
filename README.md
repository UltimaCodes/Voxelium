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

controls:

- F3 debug HUD, F5 walk/fly, F2 save edited chunks, ESC quit
- fly: hold RMB, wasd + space/ctrl, shift = fast, scroll = speed
- walk: wasd, space to jump, mouse look
- LMB break block, MMB place, keys 1-3 pick stone/dirt/glow

the HUD has live toggles for every shader effect (shadows, ssao, volumetric, bloom) plus
sun angle, exposure, vsync and render distance.

## what's in it

all 12 milestones from [SPEC.md](SPEC.md), each written up with real numbers in `docs/`:
binary greedy meshing + vertex pulling (8-byte quads, no VBO), palette-compressed chunks,
a jthread job system, GPU-driven culling with the whole world in one indirect draw call,
cave culling, flood-fill block+sky lighting, FastNoise2 terrain with caves, player physics
+ block editing, a forward HDR pipeline (cascaded shadows, SSAO, volumetrics, bloom, ACES),
and region-file persistence.

`voxelium --smoke [out.bmp]` renders offscreen, checks that pixels actually got drawn,
writes a screenshot and exits 0/1. headless self-check, useful for CI later.