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

there's also an `msvc` preset if you have VS lying around. I don't, so it's unverified.

`harden` is the same build with libstdc++ container assertions turned on (`cmake --preset
harden`, `cmake --build --preset harden-release`). worth running when you suspect a data
race — the assertions are the reason a real one got caught, and unlike a debug build this
one is fast enough to actually stream a world.

works out of the box in JetBrains Rider/CLion — open the repo folder, pick the `mingw`
preset (or the IDE's bundled toolchain), hit run.

## running

run `voxelium` from anywhere — it finds `assets/` by walking up from the exe, so build
trees and IDE working dirs just work.

controls:

- F3 debug HUD, ESC settings/pause, F4 hotbar, F5 walk/fly, F2 save edited chunks
- fly: hold RMB, wasd + space/ctrl, shift = fast, scroll = speed
- walk: wasd, space to jump, mouse look
- LMB **hold** to break (blocks have hardness, and the crack shows on the block), MMB place
- MMB while looking at a lamp opens its colour picker; the colour sticks for lamps you place
- F4 opens the hotbar (keys 1-8 or scroll to pick a block)

ESC opens the settings menu (it pauses too): every shader toggle — shadows, ssao, ssgi,
god rays, bloom (with strength and emissive), clouds, fog, taa, motion blur — a grade
section (saturation, contrast, temperature, vignette, grain), auto exposure, plus a
day/night cycle (time of day + day length), exposure, fov, render scale, volume, vsync,
cave culling and render distance. F3 is the read-only diagnostics overlay — frame graph,
timings, chunk/arena stats, log.

## worlds

every launch rolls a **new seed**, so the world you are in is scratch. if you want to keep
one, open ESC, give it a name and hit save — it lands in `saves/worlds/<name>/` with the
seed that made it. the same panel lists what you have saved (seed and size) and will load
or delete any of them. loading and starting a new world both go behind the loading screen,
so the world is built before you are standing in it.

only your edits are stored, never the terrain — that regenerates from the seed — so a saved
world is usually a few tens of KB.

## what's in it

all 12 milestones from [SPEC.md](SPEC.md), each written up with real numbers in `docs/`:
binary greedy meshing + vertex pulling (8-byte quads, no VBO), palette-compressed chunks,
a jthread job system, GPU-driven culling with the whole world in one indirect draw call,
cave culling, flood-fill RGB block + sky lighting, FastNoise2 terrain with caves and ores,
player physics + block editing, a forward HDR pipeline (cascaded shadows, SSAO,
volumetrics, bloom, ACES), flowing water, falling sand and gravel, cross-plane grass and
flowers, particles, and region-file persistence.

there's no world height limit either — chunks exist at any Y, so you can dig or build as
far as you like and the loaded band follows you.

`voxelium --smoke [out.bmp]` renders offscreen, checks that pixels actually got drawn,
writes a screenshot and exits 0/1. headless self-check, useful for CI later.
